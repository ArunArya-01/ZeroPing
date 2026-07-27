"""Physics-informed dynamic mass model for the PRC Fuel Prediction project.

Replaces the crude MTOW*0.75 reference mass with:
  - Phase-aware mass estimation (climb/cruise/descent)
  - Fuel-burn-consistent linear mass decay from takeoff to landing
  - Per-interval mass statistics (mean, min, max, consumed)

Design principles:
  - Train-only -- no Rank/Final leakage
  - Uses only deployable information (aircraft_type, flight fractions, altitude, duration)
  - Physically constrained: mass decreases monotonically, landing mass >= OEW, fuel burn >= 0
  - All features are computed per-interval from the interval's position in flight
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

GRAVITY = 9.80665

# Phase mass fractions (fraction of MTOW-OEW range used above OEW baseline)
# Climb: near takeoff weight, Cruise: mid-weight after climb burn, Descent: near landing
PHASE_MASS_FRAC = {
    "climb": 0.85,    # 85% of MTOW range above OEW at climb
    "cruise": 0.50,   # 50% through fuel burn at cruise midpoint
    "descent": 0.15,  # 15% remaining above OEW at descent
}

# Typical fuel fractions of total flight fuel burn by phase
FUEL_FRAC_BY_PHASE = {
    "climb": 0.18,
    "cruise": 0.76,
    "descent": 0.06,
}


def _load_aircraft_specs() -> pl.DataFrame:
    """Load MTOW/MLW/OEW from OpenAP descriptors CSV."""
    from aerotwin.engine.gap_closing import OPENAP_DESCRIPTOR_PATH
    return pl.read_csv(OPENAP_DESCRIPTOR_PATH).select([
        "aircraft_type", "mtow_kg", "mlw_kg", "oew_kg", "mfc_kg", "max_thrust_n", "wing_area_m2",
    ])


def estimate_tow(ac_type: str) -> float:
    """Estimate takeoff weight from aircraft type.

    Uses a physically-grounded approximation:
      TOW = OEW + payload + fuel_load
    Without payload/fuel data, approximate as:
      TOW = OEW + 0.65 * (MTOW - OEW)

    This represents a typical mid-range mission, not max-payload takeoff.
    """
    specs = _load_aircraft_specs()
    row = specs.filter(pl.col("aircraft_type") == ac_type)
    if row.is_empty():
        # Fallback: use ref_mass logic
        from aerotwin.engine.openap_baseline import _ref_mass
        return _ref_mass(ac_type)
    mtow = row["mtow_kg"][0]
    oew = row["oew_kg"][0]
    return float(oew + 0.65 * (mtow - oew))


def estimate_landing_mass(ac_type: str, flight_hours: float) -> float:
    """Estimate landing mass based on type and flight duration.

    Longer flights consume more fuel and end closer to OEW.
    Uses a duration-dependent fuel-burn estimate.
    """
    specs = _load_aircraft_specs()
    row = specs.filter(pl.col("aircraft_type") == ac_type)
    if row.is_empty():
        return 100_000.0

    mtow = float(row["mtow_kg"][0])
    oew = float(row["oew_kg"][0])
    mlw = float(row["mlw_kg"][0])
    mfc = float(row["mfc_kg"][0])

    # Typical cruise fuel flow ~ MTOW-dependent: ~0.002-0.004 * MTOW depending on type
    # Heavier aircraft: higher absolute flow; relative to MTOW it's somewhat type-constant
    # Conservative: 2.5% MTOW per hour for typical cruise
    fuel_burn_rate_kgph = 0.025 * mtow  # kg per flight-hour (rough for widebodies)
    estimated_total_fuel_burn = fuel_burn_rate_kgph * flight_hours

    # Landing mass: takeoff mass minus total fuel (clamped)
    tow = estimate_tow(ac_type)
    landing_mass = tow - min(estimated_total_fuel_burn, tow - oew)

    # Enforce physical constraints
    landing_mass = max(landing_mass, oew)
    landing_mass = min(landing_mass, mlw)
    return float(landing_mass)


def compute_mass_at_fraction(
    ac_type: str,
    fraction_of_flight: float,
    flight_hours: float,
) -> float:
    """Interpolate mass at a given fraction of flight.

    Uses a linear fuel-burn model: mass decreases linearly from TOW (frac=0) to landing_mass (frac=1).
    This is the simplest physically-consistent model: constant fuel flow approximation.
    """
    tow = estimate_tow(ac_type)
    landing = estimate_landing_mass(ac_type, flight_hours)
    frac = np.clip(float(fraction_of_flight), 0.0, 1.0)
    return float(tow + frac * (landing - tow))


def compute_phase_mass(
    ac_type: str,
    flight_hours: float,
    phase: str | None = None,
    cruise_fraction: float | None = None,
    start_fraction: float | None = None,
) -> float:
    """Estimate mass at a given flight phase or position.

    Prioritizes phase-specific estimation; falls back to fraction-of-flight interpolation.
    """
    tow = estimate_tow(ac_type)
    oew = _load_aircraft_specs().filter(pl.col("aircraft_type") == ac_type)
    if oew.is_empty():
        oew_val = 100_000.0
    else:
        oew_val = float(oew["oew_kg"][0])

    if phase is not None and phase in PHASE_MASS_FRAC:
        frac = PHASE_MASS_FRAC[phase]
        return float(oew_val + frac * (tow - oew_val))

    # Fallback: use fraction-of-flight position
    frac = np.clip(start_fraction or 0.5, 0.0, 1.0)
    return compute_mass_at_fraction(ac_type, frac, flight_hours)


def compute_mass_features(
    ac_type: str,
    flight_hours: float,
    start_fraction: float,
    end_fraction: float,
    duration_s: float,
    mean_altitude: float | None = None,
    climb_fraction: float = 0.0,
    cruise_fraction: float = 0.0,
    descent_fraction: float = 0.0,
    physics_fuel_kg: float | None = None,
) -> dict[str, float | None]:
    """Compute rich mass features for a single fuel interval.

    Returns a dict of mass features safe for ML training.
    No information leakage: uses only aircraft_type and flight-position parameters.
    """
    tow = estimate_tow(ac_type)
    oew_val = 100_000.0
    specs = _load_aircraft_specs().filter(pl.col("aircraft_type") == ac_type)
    if not specs.is_empty():
        oew_val = float(specs["oew_kg"][0])

    mid_frac = (start_fraction + end_fraction) / 2.0
    mass_start = compute_mass_at_fraction(ac_type, start_fraction, flight_hours)
    mass_end = compute_mass_at_fraction(ac_type, end_fraction, flight_hours)
    mean_mass_val = (mass_start + mass_end) / 2.0
    mass_consumed_est = mass_start - mass_end

    # Phase-aware mass based on dominant phase
    dom_phase = "cruise"
    if cruise_fraction >= max(climb_fraction, descent_fraction):
        dom_phase = "cruise"
    elif climb_fraction >= descent_fraction:
        dom_phase = "climb"
    else:
        dom_phase = "descent"

    phase_mass = compute_phase_mass(ac_type, flight_hours, phase=dom_phase)
    cruise_mass = compute_phase_mass(ac_type, flight_hours, phase="cruise")

    # Fuel fraction: what fraction of MTOW-OEW payload is consumed
    fuel_capacity = (tow - oew_val)
    fuel_fraction = mass_consumed_est / fuel_capacity if fuel_capacity > 1 else 0.0

    # Remaining fuel (percent of initial) at interval midpoint
    remaining_fuel_frac = (mean_mass_val - oew_val) / fuel_capacity if fuel_capacity > 1 else 1.0

    # Mass rate (kg/s): negative means burning
    mass_rate = mass_consumed_est / max(duration_s, 1.0) if duration_s > 0 else 0.0

    # Wing loading at current mass (for energy features)
    wing_area = 300.0  # fallback
    if not specs.is_empty():
        wing_area = float(specs["wing_area_m2"][0])
    wing_loading_cur = mean_mass_val / max(wing_area, 1.0)

    features: dict[str, float | None] = {
        "r3_tow_kg": tow,
        "r3_landing_mass_kg": estimate_landing_mass(ac_type, flight_hours),
        "r3_mass_start_kg": mass_start,
        "r3_mass_end_kg": mass_end,
        "r3_mean_mass_kg": mean_mass_val,
        "r3_min_mass_kg": min(mass_start, mass_end),
        "r3_max_mass_kg": max(mass_start, mass_end),
        "r3_mass_std_kg": abs(mass_start - mass_end) / np.sqrt(12.0) if mass_start != mass_end else 0.0,
        "r3_mass_consumed_kg": mass_consumed_est,
        "r3_mass_rate_kgps": mass_rate,
        "r3_fuel_fraction": max(0.0, min(1.0, fuel_fraction)),
        "r3_remaining_fuel_frac": max(0.0, min(1.0, remaining_fuel_frac)),
        "r3_phase_mass_kg": phase_mass,
        "r3_cruise_mass_kg": cruise_mass,
        "r3_wing_loading_cur": wing_loading_cur,
        "r3_oew_base_kg": oew_val,
    }

    # Energy features with dynamic mass
    if mean_altitude is not None and mean_altitude > 0:
        features["r3_mean_pe_j"] = mean_mass_val * GRAVITY * mean_altitude
        features["r3_mean_ke_j"] = 0.5 * mean_mass_val * 225.0 * 225.0  # ~225 m/s TAS typical

    # Fuel-to-mass efficiency ratio
    if physics_fuel_kg is not None and physics_fuel_kg > 0 and mass_consumed_est > 0:
        features["r3_fuel_mass_efficiency"] = mass_consumed_est / physics_fuel_kg

    return features


R3_MASS_FEATURES = [
    "r3_tow_kg",
    "r3_landing_mass_kg",
    "r3_mass_start_kg",
    "r3_mass_end_kg",
    "r3_mean_mass_kg",
    "r3_min_mass_kg",
    "r3_max_mass_kg",
    "r3_mass_std_kg",
    "r3_mass_consumed_kg",
    "r3_mass_rate_kgps",
    "r3_fuel_fraction",
    "r3_remaining_fuel_frac",
    "r3_phase_mass_kg",
    "r3_cruise_mass_kg",
    "r3_wing_loading_cur",
    "r3_oew_base_kg",
    "r3_mean_pe_j",
    "r3_mean_ke_j",
    "r3_fuel_mass_efficiency",
    "r3_tow_mtow_ratio",
    "r3_cruise_mass_fuel_ratio",
]

# Additional derived features computed in bulk
R3_DERIVED_FEATURES = [
    "r3_tow_mtow_ratio",
    "r3_cruise_mass_fuel_ratio",
]


def enrich_mass_from_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Bulk mass feature enrichment for a featured dataframe.

    Computes all R3 mass features for every row using only deployable information.
    Safe for Rank/Final evaluation -- no target leakage.
    """
    specs = _load_aircraft_specs()

    # Join aircraft specs
    df = df.join(specs, on="aircraft_type", how="left")

    has_mtow = "mtow_kg" in df.columns
    has_oew = "oew_kg" in df.columns
    has_dur = "duration_s" in df.columns
    has_sf = "start_fraction_of_flight" in df.columns
    has_ef = "end_fraction_of_flight" in df.columns
    has_alt = "mean_altitude" in df.columns
    has_cf = "cruise_fraction" in df.columns
    has_clf = "climb_fraction" in df.columns
    has_phf = "physics_fuel_kg" in df.columns

    if not (has_mtow and has_oew and has_dur and has_sf):
        return df

    mtow_col = pl.col("mtow_kg").fill_null(200_000.0)
    oew_col = pl.col("oew_kg").fill_null(100_000.0)
    dur_col = pl.col("duration_s").clip(lower_bound=1.0)
    sf_col = pl.col("start_fraction_of_flight").clip(0.0, 1.0)
    ef_col = pl.col("end_fraction_of_flight").clip(0.0, 1.0)

    # **Design Note**: The following uses mean_altitude (deployable for all splits).
    # duration_s and flight fractions are available in all splits without leakage.

    # Build flight-level estimates: hours per flight
    # We estimate total flight hours from the max end_fraction and total duration
    # This avoids needing flight-level grouping for efficiency
    flight_hours_est = dur_col / (ef_col - sf_col).clip(lower_bound=0.01) / 3600.0

    # TOW estimate: OEW + 0.65*(MTOW - OEW)
    tow_est = oew_col + 0.65 * (mtow_col - oew_col)

    # Landing mass: fuel-burn estimate = 2.5% MTOW * hours, clamped to [OEW, MLW]
    fuel_burn_est = 0.025 * mtow_col * flight_hours_est
    landing_est = (tow_est - fuel_burn_est).clip(
        lower_bound=oew_col,
        upper_bound=pl.col("mlw_kg").fill_null(mtow_col * 0.82)
    )

    # Mass at start/end via linear interpolation
    mass_start = tow_est + sf_col * (landing_est - tow_est)
    mass_end = tow_est + ef_col * (landing_est - tow_est)

    mean_mass = (mass_start + mass_end) / 2.0
    mass_consumed = (mass_start - mass_end).clip(lower_bound=0.0)
    mass_range = (mass_start - mass_end).abs()
    mass_std = mass_range / pl.lit(12.0).sqrt()
    mass_rate = mass_consumed / dur_col

    fuel_capacity = (tow_est - oew_col).clip(lower_bound=1.0)
    fuel_frac = mass_consumed / fuel_capacity
    remaining_frac = (mean_mass - oew_col) / fuel_capacity

    # Dominant phase
    has_phase_info = has_cf and has_clf
    if has_phase_info:
        phase_mass = (
            pl.when(pl.col("cruise_fraction") >= pl.col("climb_fraction"))
            .then(oew_col + 0.50 * (mtow_col - oew_col))
            .otherwise(oew_col + 0.85 * (mtow_col - oew_col))
        )
        cruise_mass = oew_col + 0.50 * (mtow_col - oew_col)
    else:
        phase_mass = oew_col + 0.50 * (mtow_col - oew_col)
        cruise_mass = phase_mass

    wing_area_col = pl.col("wing_area_m2").fill_null(300.0)
    wing_loading_cur = mean_mass / wing_area_col.clip(lower_bound=1.0)

    exprs = [
        tow_est.alias("r3_tow_kg"),
        landing_est.alias("r3_landing_mass_kg"),
        mass_start.alias("r3_mass_start_kg"),
        mass_end.alias("r3_mass_end_kg"),
        mean_mass.alias("r3_mean_mass_kg"),
        pl.min_horizontal(mass_start, mass_end).alias("r3_min_mass_kg"),
        pl.max_horizontal(mass_start, mass_end).alias("r3_max_mass_kg"),
        mass_std.alias("r3_mass_std_kg"),
        mass_consumed.alias("r3_mass_consumed_kg"),
        mass_rate.alias("r3_mass_rate_kgps"),
        fuel_frac.clip(0.0, 1.0).alias("r3_fuel_fraction"),
        remaining_frac.clip(0.0, 1.0).alias("r3_remaining_fuel_frac"),
        phase_mass.alias("r3_phase_mass_kg"),
        cruise_mass.alias("r3_cruise_mass_kg"),
        wing_loading_cur.alias("r3_wing_loading_cur"),
        oew_col.alias("r3_oew_base_kg"),
    ]

    # Energy with dynamic mass (requires altitude)
    if has_alt:
        alt_col = pl.col("mean_altitude").fill_null(0.0)
        exprs.append((mean_mass * GRAVITY * alt_col).alias("r3_mean_pe_j"))
        exprs.append((0.5 * mean_mass * 225.0 * 225.0).alias("r3_mean_ke_j"))

    # Efficiency ratio
    if has_phf:
        phys = pl.col("physics_fuel_kg").clip(lower_bound=1.0)
        exprs.append((mass_consumed / phys).alias("r3_fuel_mass_efficiency"))

    # Derived ratios
    exprs.append((tow_est / mtow_col.clip(1.0)).alias("r3_tow_mtow_ratio"))
    if has_phf:
        exprs.append(
            ((cruise_mass * dur_col) / pl.col("physics_fuel_kg").clip(lower_bound=1.0)).alias("r3_cruise_mass_fuel_ratio")
        )

    return df.with_columns(exprs)


def validate_mass_features(df: pl.DataFrame) -> dict[str, Any]:
    """Validate mass feature physical plausibility.

    Returns dict with validation results.
    """
    issues = []
    present = [c for c in R3_MASS_FEATURES + R3_DERIVED_FEATURES if c in df.columns]
    if not present:
        return {"valid": False, "reason": "no R3 mass features found", "issues": []}

    n = len(df)

    # Check mass non-negativity
    for col in ["r3_tow_kg", "r3_mass_start_kg", "r3_mean_mass_kg", "r3_cruise_mass_kg"]:
        if col in df.columns:
            neg = df.filter(pl.col(col) < 0).height
            if neg > 0:
                issues.append(f"{col}: {neg}/{n} negative values")

    # Check mass decreases (start >= end)
    if "r3_mass_start_kg" in df.columns and "r3_mass_end_kg" in df.columns:
        bad = df.filter(pl.col("r3_mass_start_kg") < pl.col("r3_mass_end_kg")).height
        if bad > 0:
            issues.append(f"mass_start < mass_end: {bad}/{n} intervals")

    # Check landing mass >= OEW
    if "r3_landing_mass_kg" in df.columns and "r3_oew_base_kg" in df.columns:
        bad = df.filter(pl.col("r3_landing_mass_kg") < pl.col("r3_oew_base_kg")).height
        if bad > 0:
            issues.append(f"landing_mass < OEW: {bad}/{n} intervals")

    # Check mass consumed >= 0
    if "r3_mass_consumed_kg" in df.columns:
        neg = df.filter(pl.col("r3_mass_consumed_kg") < 0).height
        if neg > 0:
            issues.append(f"negative mass consumed: {neg}/{n} intervals")

    # Summary stats
    stats = {}
    for col in present[:8]:
        vals = df[col].drop_nulls()
        if vals.is_empty():
            continue
        stats[col] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
        }

    return {
        "valid": len(issues) == 0,
        "n_rows": n,
        "n_features": len(present),
        "issues": issues,
        "stats": stats,
    }
