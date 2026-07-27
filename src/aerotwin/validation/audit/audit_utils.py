"""Dataset-agnostic helpers for external audit pipelines.

Phase thresholds, energy-rate definitions, and sparsity signals mirror the
PRC2025 / AeroTwin conventions used in ``physics/openap_baseline.py`` and
``physics/feature_engineering.py`` so external results stay comparable.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

LOGGER = logging.getLogger(__name__)

# Match openap_baseline / feature_engineering phase thresholds (m/s)
CLIMB_VR_THRESHOLD = 1.5
DESCENT_VR_THRESHOLD = -1.5

GRAVITY = 9.80665  # m/s^2
DEFAULT_INTERVAL_S = 600.0  # 10-minute windows (PRC-like ACARS cadence)
MIN_INTERVAL_S = 60.0
EARTH_RADIUS_M = 6_371_000.0

# Sparsity bins used elsewhere in AeroTwin diagnostics
SPARSITY_BINS: list[tuple[str, int, int]] = [
    ("very_sparse", 0, 5),
    ("sparse", 5, 50),
    ("medium", 50, 500),
    ("dense", 500, 10**9),
]


# --------------------------------------------------------------------------- #
# Phase detection
# --------------------------------------------------------------------------- #


def classify_phase(
    vertical_rate: float | None,
    climb_thr: float = CLIMB_VR_THRESHOLD,
    descent_thr: float = DESCENT_VR_THRESHOLD,
) -> str:
    """Classify a single vertical-rate sample into climb / cruise / descent."""
    if vertical_rate is None or (
        isinstance(vertical_rate, float) and math.isnan(vertical_rate)
    ):
        return "unknown"
    if vertical_rate > climb_thr:
        return "climb"
    if vertical_rate < descent_thr:
        return "descent"
    return "cruise"


def classify_interval_phase(
    traj_win: pl.DataFrame,
    vr_col: str = "vertical_rate",
    climb_thr: float = CLIMB_VR_THRESHOLD,
    descent_thr: float = DESCENT_VR_THRESHOLD,
) -> str:
    """Classify an interval by median vertical rate (OpenAP baseline convention)."""
    if traj_win.is_empty() or vr_col not in traj_win.columns:
        return "unknown"
    med = traj_win[vr_col].median()
    if med is None or (isinstance(med, float) and math.isnan(med)):
        return "unknown"
    return classify_phase(float(med), climb_thr=climb_thr, descent_thr=descent_thr)


def compute_phase_fractions(
    vertical_rates: np.ndarray | Sequence[float],
    climb_thr: float = CLIMB_VR_THRESHOLD,
    descent_thr: float = DESCENT_VR_THRESHOLD,
) -> dict[str, float]:
    """Return climb / cruise / descent fractions for a vertical-rate series."""
    vr = np.asarray(vertical_rates, dtype=np.float64)
    vr = vr[np.isfinite(vr)]
    n = len(vr)
    if n == 0:
        return {"climb_fraction": 0.0, "cruise_fraction": 0.0, "descent_fraction": 0.0}
    climb = int(np.sum(vr > climb_thr))
    descent = int(np.sum(vr < descent_thr))
    cruise = n - climb - descent
    return {
        "climb_fraction": climb / n,
        "cruise_fraction": cruise / n,
        "descent_fraction": descent / n,
    }


def phase_mask(
    vertical_rates: np.ndarray,
    climb_thr: float = CLIMB_VR_THRESHOLD,
    descent_thr: float = DESCENT_VR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks for climb, cruise, descent."""
    vr = np.asarray(vertical_rates, dtype=np.float64)
    climb = vr > climb_thr
    descent = vr < descent_thr
    cruise = ~(climb | descent) & np.isfinite(vr)
    return climb, cruise, descent


# --------------------------------------------------------------------------- #
# Energy features (kinetic + potential rates)
# --------------------------------------------------------------------------- #


def specific_energy_jpkg(alt_m: float, tas_mps: float) -> float:
    """Specific energy SE = g·h + ½·TAS²  [J/kg]."""
    return GRAVITY * float(alt_m) + 0.5 * float(tas_mps) ** 2


def compute_energy_rates(
    altitude_m: np.ndarray | Sequence[float],
    tas_mps: np.ndarray | Sequence[float],
    duration_s: float,
    mass_kg: float | None = None,
    physics_fuel_kg: float | None = None,
) -> dict[str, float | None]:
    """Compute interval-level energy-state features.

    Parameters
    ----------
    altitude_m, tas_mps:
        Per-point altitude [m] and true airspeed [m/s].
    duration_s:
        Interval duration in seconds (used for energy rate).
    mass_kg:
        Optional reference mass; if given, absolute PE/KE means are returned.
    physics_fuel_kg:
        Optional OpenAP fuel estimate for energy-efficiency ratio.

    Returns
    -------
    dict with keys aligned to ``ENERGY_FEATURES`` in feature_engineering.
    """
    alt = np.asarray(altitude_m, dtype=np.float64)
    tas = np.asarray(tas_mps, dtype=np.float64)
    n = min(len(alt), len(tas))
    empty: dict[str, float | None] = {
        "ref_mass_kg": mass_kg,
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
    if n == 0:
        return empty

    alt = alt[:n]
    tas = tas[:n]
    # Replace non-finite with nearest finite or safe defaults
    if not np.isfinite(alt).any():
        return empty
    alt = np.where(np.isfinite(alt), alt, np.nanmedian(alt))
    tas = np.where(np.isfinite(tas), tas, 200.0)

    se = GRAVITY * alt + 0.5 * tas * tas
    se_start = float(se[0])
    se_end = float(se[-1])
    energy_change = se_end - se_start
    dur = max(float(duration_s), 1.0)
    energy_rate = energy_change / dur

    mass = float(mass_kg) if mass_kg and mass_kg > 0 else None
    pe_mean = float(np.mean(mass * GRAVITY * alt)) if mass else None
    ke_mean = float(np.mean(0.5 * mass * tas * tas)) if mass else None

    alt_gain = float(alt[-1] - alt[0])
    climb_eff = energy_change / alt_gain if abs(alt_gain) > 1.0 else None

    phys = physics_fuel_kg if physics_fuel_kg and physics_fuel_kg > 0 else None
    energy_eff = energy_change / phys if phys else None

    return {
        "ref_mass_kg": mass,
        "mean_potential_energy_j": pe_mean,
        "mean_kinetic_energy_j": ke_mean,
        "mean_specific_energy_jpkg": float(np.mean(se)),
        "specific_energy_start": se_start,
        "specific_energy_end": se_end,
        "energy_change_jpkg": energy_change,
        "energy_rate_jpkg_s": energy_rate,
        "climb_efficiency": climb_eff,
        "energy_efficiency": energy_eff,
    }


# --------------------------------------------------------------------------- #
# Sparsity / data-quality signals
# --------------------------------------------------------------------------- #


def compute_sparsity_signals(
    n_traj_pts: int,
    duration_s: float,
    timestamps: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, float | str | None]:
    """Telemetry density and gap diagnostics for one interval.

    Assumptions
    -----------
    * ``pts_per_min`` treats duration as wall-clock coverage, not active sampling.
    * Gap stats require sorted timestamps; irregular ADS-B sampling is expected
      for OpenSky and will produce larger ``max_gap_s`` than fused PRC data.
    """
    dur = max(float(duration_s), 1e-6)
    pts = int(n_traj_pts)
    pts_per_min = pts / (dur / 60.0)
    pts_per_s = pts / dur

    sparsity_bin = "very_sparse"
    for label, lo, hi in SPARSITY_BINS:
        if lo <= pts < hi:
            sparsity_bin = label
            break

    mean_gap_s: float | None = None
    max_gap_s: float | None = None
    gap_cv: float | None = None
    if timestamps is not None and len(timestamps) >= 2:
        try:
            ts = pl.Series("t", list(timestamps)).cast(pl.Datetime("us"))
            # seconds between consecutive points
            diffs = ts.diff().drop_nulls().dt.total_seconds().to_numpy().astype(np.float64)
            diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
            if len(diffs) > 0:
                mean_gap_s = float(np.mean(diffs))
                max_gap_s = float(np.max(diffs))
                std = float(np.std(diffs))
                gap_cv = std / mean_gap_s if mean_gap_s > 1e-9 else None
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("gap stats failed: %s", exc)

    return {
        "n_traj_pts": float(pts),
        "pts_per_min": float(pts_per_min),
        "pts_per_s": float(pts_per_s),
        "sparsity_bin": sparsity_bin,
        "mean_gap_s": mean_gap_s,
        "max_gap_s": max_gap_s,
        "gap_cv": gap_cv,
        "coverage_ratio": min(1.0, pts_per_s) if pts > 0 else 0.0,
    }


def window_trajectory_stats(
    win: pl.DataFrame,
    alt_col: str = "altitude",
    gs_col: str = "groundspeed",
    vr_col: str = "vertical_rate",
) -> dict[str, float | None]:
    """Core altitude / speed / vertical-rate stats matching featured_dataset."""
    out: dict[str, float | None] = {
        "mean_altitude": None,
        "median_altitude": None,
        "max_altitude": None,
        "std_altitude": None,
        "mean_groundspeed": None,
        "std_groundspeed": None,
        "max_groundspeed": None,
        "mean_vertical_rate": None,
        "std_vertical_rate": None,
        "climb_fraction": 0.0,
        "cruise_fraction": 0.0,
        "descent_fraction": 0.0,
    }
    if win.is_empty():
        return out

    def _stat(col: str, fn: str) -> float | None:
        if col not in win.columns:
            return None
        s = win[col]
        val = getattr(s, fn)()
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 0.0 if fn == "std" else None
        return float(val)

    out["mean_altitude"] = _stat(alt_col, "mean")
    out["median_altitude"] = _stat(alt_col, "median")
    out["max_altitude"] = _stat(alt_col, "max")
    out["std_altitude"] = _stat(alt_col, "std") or 0.0
    out["mean_groundspeed"] = _stat(gs_col, "mean")
    out["std_groundspeed"] = _stat(gs_col, "std") or 0.0
    out["max_groundspeed"] = _stat(gs_col, "max")
    out["mean_vertical_rate"] = _stat(vr_col, "mean")
    out["std_vertical_rate"] = _stat(vr_col, "std") or 0.0

    if vr_col in win.columns:
        vr = win[vr_col].to_numpy().astype(np.float64)
        out.update(compute_phase_fractions(vr))
    return out


# --------------------------------------------------------------------------- #
# Interval construction helpers
# --------------------------------------------------------------------------- #


def _to_datetime(ts: Any) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        # assume unix seconds if small-ish epoch-like
        if ts > 1e12:  # ms
            return datetime.utcfromtimestamp(ts / 1000.0)
        return datetime.utcfromtimestamp(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def construct_fixed_intervals(
    t_start: datetime | Any,
    t_end: datetime | Any,
    interval_s: float = DEFAULT_INTERVAL_S,
    min_interval_s: float = MIN_INTERVAL_S,
) -> pl.DataFrame:
    """Build non-overlapping fixed-width intervals covering [t_start, t_end].

    Mirrors a regular ACARS / reporting cadence when native fuel intervals are
    absent (common for OpenSky; sometimes needed for DASHlink flow integration).
    """
    start = _to_datetime(t_start)
    end = _to_datetime(t_end)
    if start is None or end is None or end <= start:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "duration_s": pl.Float64,
            }
        )

    step = max(float(interval_s), min_interval_s)
    rows: list[dict[str, Any]] = []
    cur = start
    idx = 0
    while cur < end:
        nxt = min(cur + timedelta(seconds=step), end)
        dur = (nxt - cur).total_seconds()
        if dur >= min_interval_s:
            rows.append(
                {
                    "idx": idx,
                    "start": cur,
                    "end": nxt,
                    "duration_s": dur,
                }
            )
            idx += 1
        cur = nxt

    if not rows:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "duration_s": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("start").cast(pl.Datetime("us")),
        pl.col("end").cast(pl.Datetime("us")),
    )


def construct_intervals_from_events(
    event_times: Sequence[Any],
    min_interval_s: float = MIN_INTERVAL_S,
    max_interval_s: float = 3600.0,
) -> pl.DataFrame:
    """Build intervals between consecutive event timestamps (e.g. fuel reports)."""
    times: list[datetime] = []
    for t in event_times:
        dt = _to_datetime(t)
        if dt is not None:
            times.append(dt)
    times = sorted(set(times))
    if len(times) < 2:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "duration_s": pl.Float64,
            }
        )

    rows: list[dict[str, Any]] = []
    idx = 0
    for i in range(len(times) - 1):
        s, e = times[i], times[i + 1]
        dur = (e - s).total_seconds()
        if dur < min_interval_s or dur > max_interval_s:
            continue
        rows.append({"idx": idx, "start": s, "end": e, "duration_s": dur})
        idx += 1

    if not rows:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "duration_s": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("start").cast(pl.Datetime("us")),
        pl.col("end").cast(pl.Datetime("us")),
    )


def flight_fraction(
    t: datetime | Any,
    takeoff: datetime | Any,
    landed: datetime | Any,
) -> float:
    """Fraction of flight elapsed at time ``t`` (clamped to [0, 1])."""
    t0 = _to_datetime(takeoff)
    t1 = _to_datetime(landed)
    tt = _to_datetime(t)
    if t0 is None or t1 is None or tt is None:
        return 0.0
    total = (t1 - t0).total_seconds()
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, (tt - t0).total_seconds() / total))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def integrate_rate(
    times: Sequence[Any],
    rates: Sequence[float],
    unit: str = "kg_s",
) -> float:
    """Trapezoidal integration of a rate series over time → cumulative quantity.

    Parameters
    ----------
    times:
        Timestamps (datetime or unix seconds).
    rates:
        Rate values. ``unit`` of ``kg_s`` integrates to kg; ``kg_h`` is
        converted to kg/s first; ``lb_h`` is converted via 0.453592 kg/lb.
    """
    if len(times) < 2 or len(rates) < 2:
        return 0.0

    t_sec: list[float] = []
    for t in times:
        if isinstance(t, datetime):
            t_sec.append(t.timestamp())
        elif isinstance(t, (int, float)):
            t_sec.append(float(t) / 1000.0 if t > 1e12 else float(t))
        else:
            dt = _to_datetime(t)
            t_sec.append(dt.timestamp() if dt else float("nan"))

    r = np.asarray(rates, dtype=np.float64)
    t = np.asarray(t_sec, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(r)
    t, r = t[mask], r[mask]
    if len(t) < 2:
        return 0.0

    order = np.argsort(t)
    t, r = t[order], r[order]

    if unit == "kg_h":
        r = r / 3600.0
    elif unit == "lb_h":
        r = r * 0.45359237 / 3600.0
    elif unit == "lb_s":
        r = r * 0.45359237
    elif unit != "kg_s":
        LOGGER.warning("Unknown rate unit %r; treating as kg/s", unit)

    # np.trapezoid (NumPy ≥2.0) falls back to deprecated np.trapz
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(r, t))


def ensure_standard_traj_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename common external-source aliases to AeroTwin trajectory schema.

    Target columns: timestamp, altitude, groundspeed, vertical_rate,
    latitude, longitude, (optional) mach, CAS, typecode, source.
    """
    rename_map: dict[str, str] = {}
    aliases = {
        "timestamp": [
            "timestamp", "time", "datetime", "t", "Time", "TIME",
            "utc", "UTC", "ts",
        ],
        "altitude": [
            "altitude", "alt", "baro_altitude", "geo_altitude", "ALT",
            "Altitude", "pressure_altitude", "h", "alt_m",
        ],
        "groundspeed": [
            "groundspeed", "gs", "ground_speed", "GS", "GroundSpeed",
            "velocity", "speed", "gs_mps",
        ],
        "vertical_rate": [
            "vertical_rate", "vr", "vs", "vertical_speed", "VerticalSpeed",
            "baro_rate", "roc", "ROCD", "vs_mps",
        ],
        "latitude": ["latitude", "lat", "Latitude", "LAT"],
        "longitude": ["longitude", "lon", "lng", "Longitude", "LON", "LONG"],
        "typecode": ["typecode", "aircraft_type", "type", "ac_type", "icao_type"],
        "mach": ["mach", "Mach", "M"],
        "CAS": ["CAS", "cas", "calibrated_airspeed", "airspeed"],
    }
    lower_cols = {c.lower(): c for c in df.columns}
    for target, cands in aliases.items():
        if target in df.columns:
            continue
        for cand in cands:
            if cand in df.columns:
                rename_map[cand] = target
                break
            if cand.lower() in lower_cols and lower_cols[cand.lower()] not in rename_map:
                rename_map[lower_cols[cand.lower()]] = target
                break

    if rename_map:
        LOGGER.info("Renaming trajectory columns: %s", rename_map)
        df = df.rename(rename_map)

    # Deduplicate if multiple aliases mapped to same target (keep first)
    if len(df.columns) != len(set(df.columns)):
        seen: set[str] = set()
        keep: list[str] = []
        for c in df.columns:
            if c not in seen:
                keep.append(c)
                seen.add(c)
        df = df.select(keep)

    if "timestamp" in df.columns:
        dtype = df.schema["timestamp"]
        if dtype in (pl.Int64, pl.Int32, pl.Float64, pl.Float32, pl.UInt64):
            # Heuristic: ms vs s epoch
            sample = df["timestamp"].drop_nulls().head(1)
            unit = "ms" if (len(sample) and float(sample[0]) > 1e12) else "s"
            df = df.with_columns(
                pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit=unit).alias(
                    "timestamp"
                )
            )
        elif not str(dtype).startswith("Datetime"):
            try:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
            except Exception:
                LOGGER.warning("Could not cast timestamp to Datetime")

    return df


def synthesize_demo_trajectory(
    flight_id: str = "demo_flight_001",
    n_points: int = 120,
    ac_type: str = "B737",
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Build a synthetic climb-cruise-descent trajectory + fuel intervals.

    Used when no external data is available so the audit pipeline stays
    runnable for smoke tests and CI.
    """
    rng = np.random.default_rng(seed)
    t0 = datetime(2024, 6, 1, 10, 0, 0)
    # 2-hour flight, 1 sample / minute
    times = [t0 + timedelta(seconds=i * 60) for i in range(n_points)]
    phase_cut = n_points // 5
    alt = np.zeros(n_points)
    vr = np.zeros(n_points)
    gs = np.zeros(n_points)
    for i in range(n_points):
        if i < phase_cut:
            alt[i] = 500 + i * (10000 / phase_cut)
            vr[i] = 8.0 + rng.normal(0, 0.5)
            gs[i] = 120 + i * 2
        elif i > n_points - phase_cut:
            j = i - (n_points - phase_cut)
            alt[i] = 10500 - j * (10000 / phase_cut)
            vr[i] = -7.0 + rng.normal(0, 0.5)
            gs[i] = 200 - j * 1.5
        else:
            alt[i] = 10500 + rng.normal(0, 30)
            vr[i] = rng.normal(0, 0.3)
            gs[i] = 220 + rng.normal(0, 3)

    traj = pl.DataFrame(
        {
            "timestamp": times,
            "altitude": alt.astype(np.float64),
            "groundspeed": gs.astype(np.float64),
            "vertical_rate": vr.astype(np.float64),
            "latitude": 40.0 + np.linspace(0, 5, n_points),
            "longitude": -74.0 + np.linspace(0, 8, n_points),
            "typecode": [ac_type] * n_points,
            "source": ["synthetic"] * n_points,
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))

    # 10-minute fuel intervals with crude physics-like burn
    intervals = construct_fixed_intervals(times[0], times[-1], interval_s=600.0)
    fuel_rows = []
    for row in intervals.iter_rows(named=True):
        # ~0.6 kg/s cruise-ish burn scaled by phase
        dur = row["duration_s"]
        fuel_rows.append(
            {
                "idx": row["idx"],
                "start": row["start"],
                "end": row["end"],
                "fuel_kg": 0.55 * dur + float(rng.normal(0, 20)),
            }
        )
    fuel = pl.DataFrame(fuel_rows).with_columns(
        pl.col("start").cast(pl.Datetime("us")),
        pl.col("end").cast(pl.Datetime("us")),
    )

    meta = {
        "flight_id": flight_id,
        "aircraft_type": ac_type,
        "origin_icao": "KJFK",
        "destination_icao": "KORD",
        "takeoff": times[0],
        "landed": times[-1],
        "label_source": "synthetic_demo",
    }
    return traj, fuel, meta
