from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from physics.openap_baseline import DEFAULT_REF_MASS_FRAC, _infer_tas, _ref_mass

GRAVITY = 9.80665
CLIMB_VR_THRESHOLD = 1.5
DESCENT_VR_THRESHOLD = -1.5
HOLDING_GS_THRESHOLD = 80.0  # m/s
HOLDING_VR_THRESHOLD = 0.5  # m/s


def _specific_energy(alt_m: float, tas_mps: float) -> float:
    """SE = g*h + 0.5*TAS^2  [J/kg]."""
    return GRAVITY * alt_m + 0.5 * tas_mps * tas_mps


def _tas_series(win: pl.DataFrame) -> np.ndarray:
    """Per-point TAS array from trajectory window."""
    n = len(win)
    out = np.empty(n, dtype=np.float64)
    for i, row in enumerate(win.iter_rows(named=True)):
        alt = float(row.get("altitude") or 10000.0)
        tas = _infer_tas(row, alt)
        out[i] = float(tas if tas and tas > 0 else row.get("groundspeed") or 200.0)
    return out


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def compute_energy_features(
    win: pl.DataFrame,
    ac_type: str,
    duration_s: float,
    physics_fuel_kg: float | None = None,
) -> dict[str, float | None]:
    """Energy-state features from a trajectory window."""
    mass = _ref_mass(ac_type)
    empty: dict[str, float | None] = {
        "ref_mass_kg": mass,
        "mean_potential_energy_j": None,
        "mean_kinetic_energy_j": None,
        "mean_specific_energy_jpkg": None,
        "specific_energy_start": None,
        "specific_energy_end": None,
        "energy_change_jpkg": None,
        "energy_rate_jpkg_s": None,
        "climb_efficiency": None,
        "energy_efficiency": None,
    }
    if win.is_empty():
        return empty

    alt = win["altitude"].to_numpy().astype(np.float64)
    tas = _tas_series(win)
    pe = mass * GRAVITY * alt
    ke = 0.5 * mass * tas * tas
    se = pe / mass + ke / mass

    se_start = float(se[0])
    se_end = float(se[-1])
    energy_change = se_end - se_start
    dur = max(duration_s, 1.0)
    energy_rate = energy_change / dur

    alt_gain = float(alt[-1] - alt[0])
    climb_eff = energy_change / alt_gain if abs(alt_gain) > 1.0 else None

    phys = physics_fuel_kg if physics_fuel_kg and physics_fuel_kg > 0 else None
    energy_eff = energy_change / phys if phys else None

    return {
        "ref_mass_kg": mass,
        "mean_potential_energy_j": float(pe.mean()),
        "mean_kinetic_energy_j": float(ke.mean()),
        "mean_specific_energy_jpkg": float(se.mean()),
        "specific_energy_start": se_start,
        "specific_energy_end": se_end,
        "energy_change_jpkg": energy_change,
        "energy_rate_jpkg_s": energy_rate,
        "climb_efficiency": climb_eff,
        "energy_efficiency": energy_eff,
    }


def compute_operational_features(
    win: pl.DataFrame,
    duration_s: float,
) -> dict[str, float | None]:
    """Operational behaviour features from a trajectory window."""
    defaults: dict[str, float | None] = {
        "time_to_cruise_s": None,
        "climb_duration_s": None,
        "descent_duration_s": None,
        "cruise_speed_std": None,
        "tas_std": None,
        "vertical_rate_std": None,
        "number_of_level_segments": None,
        "holding_indicator": 0.0,
        "path_efficiency": None,
        "distance_ratio": None,
        "altitude_stability": None,
        "segment_acceleration_mean": None,
    }
    if win.is_empty():
        return defaults

    n = len(win)
    dur = max(duration_s, 1.0)
    dt = dur / max(n - 1, 1)

    vr = win["vertical_rate"].to_numpy().astype(np.float64)
    gs = win["groundspeed"].to_numpy().astype(np.float64)
    tas = _tas_series(win)
    alt = win["altitude"].to_numpy().astype(np.float64)

    climb_mask = vr > CLIMB_VR_THRESHOLD
    descent_mask = vr < DESCENT_VR_THRESHOLD
    cruise_mask = ~(climb_mask | descent_mask)

    climb_duration = float(climb_mask.sum() * dt)
    descent_duration = float(descent_mask.sum() * dt)

    time_to_cruise = None
    if climb_mask.any():
        first_cruise = np.flatnonzero(cruise_mask)
        if len(first_cruise) > 0:
            time_to_cruise = float(first_cruise[0] * dt)

    cruise_speed_std = float(gs[cruise_mask].std()) if cruise_mask.sum() > 1 else float(gs.std())
    tas_std = float(tas.std()) if n > 1 else 0.0
    vr_std = float(vr.std()) if n > 1 else 0.0

    level_segments = 0
    if n > 1:
        is_level = cruise_mask.astype(int)
        level_segments = int(np.sum(np.diff(is_level) == 1))

    hold_frac = float(((gs < HOLDING_GS_THRESHOLD) & (np.abs(vr) < HOLDING_VR_THRESHOLD)).mean())

    path_eff = None
    dist_ratio = None
    if {"latitude", "longitude"}.issubset(win.columns):
        lat = win["latitude"].to_numpy().astype(np.float64)
        lon = win["longitude"].to_numpy().astype(np.float64)
        seg_dists = [
            _haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i]) for i in range(1, n)
        ]
        path_dist = float(sum(seg_dists)) if seg_dists else 0.0
        straight = _haversine_m(lat[0], lon[0], lat[-1], lon[-1]) if n >= 2 else 0.0
        if path_dist > 1.0:
            path_eff = straight / path_dist
            dist_ratio = path_dist / max(straight, 1.0)

    alt_stab = 1.0 / (1.0 + float(alt.std()))

    if n > 1:
        accel = np.abs(np.diff(gs)) / max(dt, 1e-3)
        seg_accel = float(accel.mean())
    else:
        seg_accel = 0.0

    return {
        "time_to_cruise_s": time_to_cruise,
        "climb_duration_s": climb_duration,
        "descent_duration_s": descent_duration,
        "cruise_speed_std": cruise_speed_std,
        "tas_std": tas_std,
        "vertical_rate_std": vr_std,
        "number_of_level_segments": float(level_segments),
        "holding_indicator": hold_frac,
        "path_efficiency": path_eff,
        "distance_ratio": dist_ratio,
        "altitude_stability": alt_stab,
        "segment_acceleration_mean": seg_accel,
    }


def _mass_lookup_df(df: pl.DataFrame) -> pl.DataFrame:
    """Build aircraft_type → ref_mass_kg lookup (small, fast join)."""
    types = df.select("aircraft_type").unique().drop_nulls()
    return types.with_columns(
        pl.col("aircraft_type")
        .cast(pl.Utf8)
        .map_batches(lambda s: pl.Series([_ref_mass(str(x)) for x in s.to_list()]))
        .alias("ref_mass_kg")
    )


def enrich_from_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add energy + operational features from existing parquet columns (no trajectory reload)."""
    g = GRAVITY
    mass_lut = _mass_lookup_df(df)
    df = df.join(mass_lut, on="aircraft_type", how="left")

    alt = pl.col("mean_altitude").fill_null(pl.col("alt_used"))
    tas = pl.col("tas_used").fill_null(pl.col("mean_groundspeed"))
    med_alt = pl.col("median_altitude").fill_null(alt)
    dur = pl.col("duration_s").fill_null(0.0).clip(lower_bound=1.0)

    df = df.with_columns(
        (pl.col("ref_mass_kg") * g * alt).alias("mean_potential_energy_j"),
        (0.5 * pl.col("ref_mass_kg") * tas.pow(2)).alias("mean_kinetic_energy_j"),
        (g * alt + 0.5 * tas.pow(2)).alias("mean_specific_energy_jpkg"),
        (g * pl.col("alt_used").fill_null(alt) + 0.5 * tas.pow(2)).alias("specific_energy_start"),
        (g * med_alt + 0.5 * pl.col("max_groundspeed").fill_null(tas).pow(2) * 0.5).alias("specific_energy_end"),
    )
    df = df.with_columns(
        (pl.col("specific_energy_end") - pl.col("specific_energy_start")).alias("energy_change_jpkg"),
    )
    df = df.with_columns(
        (pl.col("energy_change_jpkg") / dur).alias("energy_rate_jpkg_s"),
        (
            pl.col("energy_change_jpkg")
            / (pl.col("mean_vertical_rate").abs() * dur + 1.0)
        ).alias("climb_efficiency"),
        (
            pl.col("energy_change_jpkg") / pl.col("physics_fuel_kg").clip(lower_bound=1.0)
        ).alias("energy_efficiency"),
    )

    df = df.with_columns(
        (pl.col("climb_fraction") * pl.col("duration_s")).alias("climb_duration_s"),
        (pl.col("descent_fraction") * pl.col("duration_s")).alias("descent_duration_s"),
        (pl.col("climb_fraction") * pl.col("duration_s")).alias("time_to_cruise_s"),
        pl.col("std_groundspeed").alias("cruise_speed_std"),
        pl.col("std_groundspeed").alias("tas_std"),
        pl.col("std_vertical_rate").alias("vertical_rate_std"),
        pl.max_horizontal(pl.col("cruise_fraction") * pl.col("n_traj_pts"), pl.lit(1.0))
        .alias("number_of_level_segments"),
        (
            (pl.col("mean_groundspeed") < HOLDING_GS_THRESHOLD)
            & (pl.col("mean_vertical_rate").abs() < HOLDING_VR_THRESHOLD)
        )
        .cast(pl.Float64)
        .alias("holding_indicator"),
        pl.lit(1.0).alias("path_efficiency"),
        pl.lit(1.0).alias("distance_ratio"),
        (1.0 / (1.0 + pl.col("std_altitude").fill_null(0.0))).alias("altitude_stability"),
        (pl.col("std_groundspeed") / dur).alias("segment_acceleration_mean"),
    )

    if "flight_id" in df.columns:
        df = df.sort("flight_id", "start_fraction_of_flight")
        df = df.with_columns(
            pl.col("energy_change_jpkg").cum_sum().over("flight_id").alias("cumulative_energy_change_jpkg")
        )
    else:
        df = df.with_columns(pl.col("energy_change_jpkg").alias("cumulative_energy_change_jpkg"))

    return df


ENERGY_FEATURES = [
    "ref_mass_kg",
    "mean_potential_energy_j",
    "mean_kinetic_energy_j",
    "mean_specific_energy_jpkg",
    "specific_energy_start",
    "specific_energy_end",
    "energy_change_jpkg",
    "energy_rate_jpkg_s",
    "climb_efficiency",
    "energy_efficiency",
    "cumulative_energy_change_jpkg",
]

OPERATIONAL_FEATURES = [
    "time_to_cruise_s",
    "climb_duration_s",
    "descent_duration_s",
    "cruise_speed_std",
    "tas_std",
    "vertical_rate_std",
    "number_of_level_segments",
    "holding_indicator",
    "path_efficiency",
    "distance_ratio",
    "altitude_stability",
    "segment_acceleration_mean",
]