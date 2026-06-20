"""
E5 weather-informed features for AeroTwin v3.

No direct METAR/GRIB in the dataset. Features are derived from:
- ISA atmosphere at cruise altitude (temperature, pressure, density altitude)
- TAS / groundspeed / track decomposition (headwind, crosswind proxies)
- ISA deviation proxy from kinematic mismatch where air data exists
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from physics.openap_baseline import _infer_tas

# ISA troposphere constants (below 11 km)
T0_K = 288.15
P0_PA = 101_325.0
LAPSE_K_PER_M = 0.0065
G = 9.80665
R_AIR = 287.05


def isa_temperature_k(alt_m: float) -> float:
    h = max(0.0, min(float(alt_m), 11_000.0))
    return T0_K - LAPSE_K_PER_M * h


def isa_pressure_pa(alt_m: float) -> float:
    t = isa_temperature_k(alt_m)
    return P0_PA * (t / T0_K) ** (G / (R_AIR * LAPSE_K_PER_M))


def isa_density_kg_m3(alt_m: float) -> float:
    t = isa_temperature_k(alt_m)
    p = isa_pressure_pa(alt_m)
    return p / (R_AIR * t)


def density_altitude_m(pressure_alt_m: float, oat_k: float) -> float:
    """DA = PA + 120 * (OAT_C - ISA_C)."""
    isa_t = isa_temperature_k(pressure_alt_m)
    oat_c = oat_k - 273.15
    isa_c = isa_t - 273.15
    return pressure_alt_m + 120.0 * (oat_c - isa_c)


def _track_rad(track_deg: float) -> float:
    return math.radians(float(track_deg) % 360.0)


def wind_components(tas: float, gs: float, track_deg: float) -> tuple[float, float]:
    """Estimate headwind (+ into wind) and crosswind (+ from left) along track."""
    if tas <= 0 or gs <= 0:
        return 0.0, 0.0
    tr = _track_rad(track_deg)
    # Ground & air velocity along / across track (heading ≈ track).
    tas_along, tas_across = tas * math.cos(tr), tas * math.sin(tr)
    gs_along, gs_across = gs * math.cos(tr), gs * math.sin(tr)
    w_along = gs_along - tas_along
    w_across = gs_across - tas_across
    headwind = -w_along  # positive headwind opposes motion
    crosswind = w_across
    return headwind, crosswind


def compute_weather_features(
    win: pl.DataFrame,
    mean_altitude: float | None = None,
) -> dict[str, float | None]:
    """Weather proxies from trajectory window."""
    empty: dict[str, float | None] = {
        "headwind_mps": None,
        "crosswind_mps": None,
        "temperature_k": None,
        "pressure_pa": None,
        "isa_deviation_k": None,
        "density_altitude_m": None,
    }
    alt = mean_altitude
    if alt is None and not win.is_empty() and "altitude" in win.columns:
        alt = float(win["altitude"].mean() or 0.0)
    if alt is None:
        return empty

    isa_t = isa_temperature_k(alt)
    isa_p = isa_pressure_pa(alt)
    isa_d = isa_density_kg_m3(alt)

    headwinds: list[float] = []
    crosswinds: list[float] = []
    isa_devs: list[float] = []

    for row in win.iter_rows(named=True):
        h = float(row.get("altitude") or alt)
        gs = row.get("groundspeed")
        track = row.get("track")
        if gs is None or track is None:
            continue
        gs = float(gs)
        tas = _infer_tas(row, h)
        if tas is None or tas <= 0:
            continue
        hw, cw = wind_components(tas, gs, float(track))
        headwinds.append(hw)
        crosswinds.append(cw)
        # Kinematic ISA-deviation proxy: excess TAS over GS vs ISA density ratio.
        rho = isa_density_kg_m3(h)
        expected_tas = gs * math.sqrt(isa_d / max(rho, 1e-6))
        isa_devs.append(tas - expected_tas)

    oat_proxy = isa_t + (float(np.mean(isa_devs)) if isa_devs else 0.0)
    da = density_altitude_m(alt, oat_proxy)

    return {
        "headwind_mps": float(np.mean(headwinds)) if headwinds else None,
        "crosswind_mps": float(np.mean(np.abs(crosswinds))) if crosswinds else None,
        "temperature_k": isa_t,
        "pressure_pa": isa_p,
        "isa_deviation_k": float(np.mean(isa_devs)) if isa_devs else 0.0,
        "density_altitude_m": da,
    }


def enrich_weather_from_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add E5 weather columns from existing parquet fields (no trajectory reload)."""
    alt = pl.col("mean_altitude").fill_null(pl.col("alt_used")).fill_null(10_000.0)
    tas = pl.col("tas_used").fill_null(pl.col("mean_groundspeed")).fill_null(200.0)
    gs = pl.col("mean_groundspeed").fill_null(pl.col("tas_used")).fill_null(200.0)

    # ISA scalars from mean altitude (vectorized via map on unique alt bins is overkill; row-wise expr)
    df = df.with_columns(
        alt.alias("_wx_alt"),
        tas.alias("_wx_tas"),
        gs.alias("_wx_gs"),
    )

    alts = df["_wx_alt"].to_numpy()
    isa_t = np.array([isa_temperature_k(a) for a in alts])
    isa_p = np.array([isa_pressure_pa(a) for a in alts])
    isa_d = np.array([isa_density_kg_m3(a) for a in alts])

    tas_v = df["_wx_tas"].to_numpy()
    gs_v = df["_wx_gs"].to_numpy()
    headwind = tas_v - gs_v
    crosswind = np.sqrt(np.maximum(0.0, tas_v**2 - gs_v**2))
    # Kinematic proxy: TAS>GS often indicates headwind / non-ISA atmosphere.
    isa_dev = tas_v - gs_v
    da = alts + 120.0 * (isa_dev * 0.15)

    return df.with_columns(
        pl.Series("headwind_mps", headwind),
        pl.Series("crosswind_mps", crosswind),
        pl.Series("temperature_k", isa_t),
        pl.Series("pressure_pa", isa_p),
        pl.Series("isa_deviation_k", isa_dev),
        pl.Series("density_altitude_m", da),
    ).drop("_wx_alt", "_wx_tas", "_wx_gs")


WEATHER_FEATURES = [
    "headwind_mps",
    "crosswind_mps",
    "temperature_k",
    "pressure_pa",
    "isa_deviation_k",
    "density_altitude_m",
]