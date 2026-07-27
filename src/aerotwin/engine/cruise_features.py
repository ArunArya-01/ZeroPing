"""Physics-informed cruise feature engineering for the PRC Fuel Prediction project.

Extracts cruise-specific features from existing parquet columns without trajectory reload.
Uses: cruise_fraction, alt_used, tas_used, mean_groundspeed, physics_fuel_kg, duration_s,
      headwind_mps, crosswind_mps, max_altitude, ref_mass_kg, r3_mean_mass_kg, r3_tow_kg.

Design principles:
  - Train-only features from deployable columns (no trajectory reload on Rank/Final)
  - Physics-justified: cruise Mach, cruise load, cruise efficiency
  - Interaction features only with clear aviation rationale
  - All features definable from existing parquet columns without leaking target
"""

from __future__ import annotations

import numpy as np
import polars as pl

GRAVITY = 9.80665
MACH_A0 = 340.3  # Speed of sound at sea level (m/s), typical cruise Mach reference


def enrich_cruise_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cruise-level features to a featured parquet dataframe.

    Features:
      Core:
        r4_cruise_duration_s       - cruise time (duration_s * cruise_fraction)
        r4_cruise_altitude_m        - mean altitude during cruise (cap at alt_used)
        r4_cruise_mach_est          - estimated Mach number (tas_used ~ TAS / MACH_A0)
        r4_cruise_tas_mps           - representative TAS (tas_used)
        r4_cruise_fuel_flow_kgps    - fuel flow during cruise (kg/s)
        r4_cruise_efficiency        - cruise fuel per unit mass per unit duration
        r4_cruise_load_factor        - mass / MTOW (how heavy is the aircraft in cruise)
        r4_cruise_altitude_band      - cruise altitude / service ceiling proxy
        r4_cruise_pct_max_alt        - altitude / max_altitude (cruise altitude utilization)
        r4_cruise_spd_stability      - 1 / (1 + std_groundspeed) in cruise
        r4_cruise_tailwind_mps       - tailwind component (positive = tailwind)
        r4_cruise_headwind_mps       - headwind component (positive = headwind)

      Interactions:
        r4_cruise_alt_x_dur          - altitude * duration
        r4_cruise_mach_x_dur         - Mach * duration
        r4_cruise_ff_x_mass          - fuel flow * mass (absolute power)
        r4_cruise_mass_x_mach        - mass * Mach (momentum proxy)
        r4_cruise_tas_x_dur          - TAS * duration (distance flown in cruise)
        r4_cruise_tailwind_x_dur     - tailwind * duration
        r4_cruise_headwind_x_dur     - headwind * duration
        r4_cruise_alt_x_ff            - altitude * fuel flow (power at altitude)
    """
    has_cf = "cruise_fraction" in df.columns
    has_dur = "duration_s" in df.columns
    has_tas = "tas_used" in df.columns
    has_alt = "alt_used" in df.columns
    has_gs = "mean_groundspeed" in df.columns
    has_phf = "physics_fuel_kg" in df.columns
    has_hw = "headwind_mps" in df.columns
    has_cw = "crosswind_mps" in df.columns
    has_maxalt = "max_altitude" in df.columns
    has_mass = "r3_mean_mass_kg" in df.columns
    has_tow = "r3_tow_kg" in df.columns
    has_ref = "ref_mass_kg" in df.columns

    if not (has_cf and has_dur):
        return df

    cf = pl.col("cruise_fraction").clip(0.0, 1.0)
    dur = pl.col("duration_s").clip(lower_bound=1.0)

    exprs = []

    # === Core cruise features ===

    # Cruise duration (seconds)
    exprs.append((cf * dur).alias("r4_cruise_duration_s"))

    # Cruise altitude: use alt_used if available, else mean_altitude
    cruise_alt = pl.col("alt_used").fill_null(pl.col("mean_altitude").fill_null(0.0)) if has_alt else \
                 pl.col("mean_altitude").fill_null(0.0) if "mean_altitude" in df.columns else pl.lit(0.0)
    exprs.append(cruise_alt.alias("r4_cruise_altitude_m"))

    # Cruise Mach (estimated: TAS / speed of sound, adjusted for altitude)
    if has_tas:
        # Mach at altitude ~ TAS / sqrt(1.4 * 287 * T), simplified as TAS / 300 at cruise alt
        tas_col = pl.col("tas_used").fill_null(200.0)
        cruise_mach = tas_col / MACH_A0
        # Adjust for altitude: lower speed of sound at altitude reduces Mach
        alt_adj = cruise_alt / 11000.0
        cruise_mach = cruise_mach / (1.0 - 0.15 * alt_adj.clip(0.0, 1.0))
        exprs.append(cruise_mach.clip(0.5, 1.0).alias("r4_cruise_mach_est"))
        exprs.append(tas_col.alias("r4_cruise_tas_mps"))

    # Cruise fuel flow rate (approximate: physics_fuel_kg * cruise_frac / cruise_dur)
    if has_phf:
        cruise_dur_safe = (cf * dur).clip(lower_bound=1.0)
        cruise_ff = pl.col("physics_fuel_kg") * cf / cruise_dur_safe
        exprs.append(cruise_ff.clip(0.0, 20.0).alias("r4_cruise_fuel_flow_kgps"))

    # Cruise efficiency: physics_fuel_kg / (mass * cruise_duration)
    if has_phf and (has_mass or has_ref):
        mass_col = pl.col("r3_mean_mass_kg") if has_mass else pl.col("ref_mass_kg")
        cruise_dur_safe = (cf * dur).clip(lower_bound=1.0)
        cruise_eff = (pl.col("physics_fuel_kg") * cf) / (mass_col.clip(lower_bound=1.0) * cruise_dur_safe)
        exprs.append(cruise_eff.clip(0.0, 0.01).alias("r4_cruise_efficiency"))

    # Cruise load factor: current mass / TOW (or MTOW)
    if has_mass and has_tow:
        load = pl.col("r3_mean_mass_kg") / pl.col("r3_tow_kg").clip(lower_bound=1.0)
        exprs.append(load.clip(0.3, 1.0).alias("r4_cruise_load_factor"))
    elif has_ref:
        load2 = pl.col("ref_mass_kg") / pl.col("ref_mass_kg")  # degenerates to 1.0
        exprs.append(load2.alias("r4_cruise_load_factor"))

    # Cruise altitude band (normalized to 0-1 range, 0 = sea level, 1 = 40k ft)
    if has_alt or "mean_altitude" in df.columns or "max_altitude" in df.columns:
        alt_ref = cruise_alt
        exprs.append((alt_ref / 12500.0).clip(0.0, 1.0).alias("r4_cruise_altitude_band"))

    # Cruise altitude utilization (how close to max altitude is cruise?)
    if has_maxalt and (has_alt or "mean_altitude" in df.columns):
        maxalt = pl.col("max_altitude").fill_null(cruise_alt).clip(lower_bound=1.0)
        exprs.append((cruise_alt / maxalt).clip(0.0, 1.0).alias("r4_cruise_pct_max_alt"))

    # Cruise speed stability
    if "std_groundspeed" in df.columns:
        gs_std = pl.col("std_groundspeed").fill_null(10.0)
        exprs.append((1.0 / (1.0 + gs_std)).alias("r4_cruise_spd_stability"))

    # Wind components (already in WEATHER_FEATURES, but explicitly cruise-weighted)
    if has_hw:
        hw_col = pl.col("headwind_mps").fill_null(0.0)
        # Tailwind is negative headwind
        exprs.append((-hw_col.clip(upper_bound=0.0)).alias("r4_cruise_tailwind_mps"))
        exprs.append(hw_col.clip(lower_bound=0.0).alias("r4_cruise_headwind_mps"))

    # === Cruise interaction features ===

    # Cruise altitude × duration (distance-time proxy for cruise energy)
    exprs.append(
        (cruise_alt * dur * cf).alias("r4_cruise_alt_x_dur")
    )

    # Cruise Mach × duration
    if has_tas:
        cruise_mach_val = tas_col / MACH_A0
        alt_adj = cruise_alt / 11000.0
        cruise_mach_val = cruise_mach_val / (1.0 - 0.15 * alt_adj.clip(0.0, 1.0))
        exprs.append(
            (cruise_mach_val.clip(0.5, 1.0) * dur * cf).alias("r4_cruise_mach_x_dur")
        )

    # Cruise fuel flow × mass (absolute power during cruise)
    if has_phf and (has_mass or has_ref):
        mass_col = pl.col("r3_mean_mass_kg") if has_mass else pl.col("ref_mass_kg")
        cruise_dur_safe = (cf * dur).clip(lower_bound=1.0)
        cruise_ff = pl.col("physics_fuel_kg") * cf / cruise_dur_safe
        exprs.append(
            (cruise_ff.clip(0.0, 20.0) * mass_col).alias("r4_cruise_ff_x_mass")
        )

    # Mass × Mach during cruise (momentum-type proxy)
    if has_tas and (has_mass or has_ref):
        mass_col = pl.col("r3_mean_mass_kg") if has_mass else pl.col("ref_mass_kg")
        cruise_mach_val = tas_col / MACH_A0
        alt_adj = cruise_alt / 11000.0
        cruise_mach_val = cruise_mach_val / (1.0 - 0.15 * alt_adj.clip(0.0, 1.0))
        exprs.append(
            (mass_col * cruise_mach_val.clip(0.5, 1.0)).alias("r4_cruise_mass_x_mach")
        )

    # TAS × duration (distance flown in cruise)
    if has_tas:
        exprs.append(
            (tas_col * dur * cf).alias("r4_cruise_tas_x_dur")
        )

    # Cruise wind × duration
    if has_hw:
        hw_col = pl.col("headwind_mps").fill_null(0.0)
        cruise_dur = (cf * dur)
        exprs.append((-hw_col.clip(upper_bound=0.0) * cruise_dur).alias("r4_cruise_tailwind_x_dur"))
        exprs.append((hw_col.clip(lower_bound=0.0) * cruise_dur).alias("r4_cruise_headwind_x_dur"))

    # Altitude × fuel flow
    if has_phf and has_alt:
        cruise_dur_safe = (cf * dur).clip(lower_bound=1.0)
        cruise_ff = pl.col("physics_fuel_kg") * cf / cruise_dur_safe
        exprs.append(
            (cruise_alt * cruise_ff.clip(0.0, 20.0)).alias("r4_cruise_alt_x_ff")
        )

    return df.with_columns(exprs)


R4_CRUISE_CORE = [
    "r4_cruise_duration_s",
    "r4_cruise_altitude_m",
    "r4_cruise_mach_est",
    "r4_cruise_tas_mps",
    "r4_cruise_fuel_flow_kgps",
    "r4_cruise_efficiency",
    "r4_cruise_load_factor",
    "r4_cruise_altitude_band",
    "r4_cruise_pct_max_alt",
    "r4_cruise_spd_stability",
    "r4_cruise_tailwind_mps",
    "r4_cruise_headwind_mps",
]

R4_CRUISE_INTERACTIONS = [
    "r4_cruise_alt_x_dur",
    "r4_cruise_mach_x_dur",
    "r4_cruise_ff_x_mass",
    "r4_cruise_mass_x_mach",
    "r4_cruise_tas_x_dur",
    "r4_cruise_tailwind_x_dur",
    "r4_cruise_headwind_x_dur",
    "r4_cruise_alt_x_ff",
]

R4_ALL = R4_CRUISE_CORE + R4_CRUISE_INTERACTIONS
