"""Physics-derived feature sets for mechanism ablation (Phase 3).

Physics features = quantities generated from OpenAP / mass model / energy physics
priors. Trajectory kinematics, weather, operational fractions, and categoricals
are retained under the ``nophysics`` feature set.
"""

from __future__ import annotations

# Explicit physics / mass / energy columns in the distillation feature list.
PHYSICS_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "ref_mass_kg",
        "physics_fuel_kg",
        # energy-state physics
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
        # R3 dynamic mass / physics mass model
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
    }
)

# Prefix rules for robustness if new columns appear
PHYSICS_PREFIXES: tuple[str, ...] = ("r3_",)


def is_physics_feature(name: str) -> bool:
    if name in PHYSICS_FEATURE_NAMES:
        return True
    return any(name.startswith(p) for p in PHYSICS_PREFIXES)


def split_features(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Return (physics_cols, non_physics_cols) preserving order."""
    phys, keep = [], []
    for c in feature_cols:
        (phys if is_physics_feature(c) else keep).append(c)
    return phys, keep


def nophysics_feature_cols(feature_cols: list[str]) -> list[str]:
    return split_features(feature_cols)[1]


def classify_numeric(name: str) -> str:
    """Bucket for attribution shift: physics | trajectory | weather | operational | other."""
    if is_physics_feature(name):
        return "physics"
    weather = {
        "headwind_mps",
        "crosswind_mps",
        "temperature_k",
        "pressure_pa",
        "isa_deviation_k",
        "density_altitude_m",
    }
    if name in weather:
        return "weather"
    traj = {
        "duration_s",
        "mean_altitude",
        "median_altitude",
        "max_altitude",
        "std_altitude",
        "mean_groundspeed",
        "std_groundspeed",
        "max_groundspeed",
        "mean_vertical_rate",
        "std_vertical_rate",
        "n_traj_pts",
    }
    if name in traj:
        return "trajectory"
    ops = {
        "start_fraction_of_flight",
        "end_fraction_of_flight",
        "climb_fraction",
        "cruise_fraction",
        "descent_fraction",
        "has_acars_in_window",
    }
    if name in ops:
        return "operational"
    return "other"
