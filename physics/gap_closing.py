"""Hypothesis-driven gap closing for official PRC Rank/Final RMSE.

Train-only: calibrators, specialists, and ensemble weights fit exclusively on
train OOF / train rows. Rank and Final are evaluation-only.

Baseline: official ensemble Combined RMSE ≈ 228.3 kg.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import polars as pl

from physics.eval_framework import evaluate, project_root
from physics.official_benchmark import (
    apply_bases,
    build_oof_matrix,
    choose_meta_on_train_folds,
    ew_feature_cols,
    featured_path,
    fit_meta,
    predict_fuel_kg,
    prepare_xy,
    train_model,
)

LOGGER = logging.getLogger(__name__)

RANDOM_STATE = 42
CACHE_DIR = project_root() / "cache"
OOF_CACHE = CACHE_DIR / "official_ensemble_cache.pkl"

ENSEMBLE_BASES = [
    ("xgb", "direct"),
    ("lgbm", "direct"),
    ("cat", "direct"),
    ("xgb", "fuel_flow"),
    ("lgbm", "fuel_flow"),
    ("cat", "fuel_flow"),
]

# Error-dominant widebodies (from official error analysis)
HEAVY_TYPES = frozenset(
    {
        "A359",
        "B77W",
        "B744",
        "A332",
        "A333",
        "B789",
        "B788",
        "B772",
        "B77L",
        "A306",
    }
)
NARROW_TYPES = frozenset(
    {"A320", "A20N", "A319", "A321", "A21N", "B738", "B737", "B739", "B38M", "B734"}
)

BASELINE_OFFICIAL = {
    "combined_rmse": 228.2505398783207,
    "rank_rmse": 239.18031519148374,
    "final_rmse": 220.8570993417157,
    "combined_mae": 88.74505740577311,
}


def clean_featured(df: pl.DataFrame, *, require_physics: bool = False) -> pl.DataFrame:
    """Match official notebook 17: keep rows with valid labels; physics may be imputed.

    Official eval trained on 119_032 train rows including null physics_fuel_kg
    (handled by median imputer). Dropping those ~3k rows degrades the ensemble.
    """
    out = df.filter(
        pl.col("actual_fuel_kg").is_not_null()
        & pl.col("flight_id").is_not_null()
        & pl.col("actual_fuel_kg").is_finite()
        & pl.col("duration_s").is_finite()
        & (pl.col("duration_s") > 0)
    )
    if require_physics:
        out = out.filter(
            pl.col("physics_fuel_kg").is_not_null() & pl.col("physics_fuel_kg").is_finite()
        )
    return out


def load_splits() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = clean_featured(pl.read_parquet(featured_path("train")))
    rank = clean_featured(pl.read_parquet(featured_path("rank")))
    final = clean_featured(pl.read_parquet(featured_path("final")))
    return train, rank, final


def ensure_features(df: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(None).alias(c) for c in missing])
    return df


def aircraft_class(ac: str) -> str:
    ac = str(ac)
    if ac in HEAVY_TYPES:
        return "heavy"
    if ac in NARROW_TYPES:
        return "narrow"
    return "other"


def dominant_phase_row(row: dict) -> str:
    ph = row.get("phase")
    if ph in ("climb", "cruise", "descent"):
        return str(ph)
    c = float(row.get("climb_fraction") or 0)
    cr = float(row.get("cruise_fraction") or 0)
    d = float(row.get("descent_fraction") or 0)
    m = max(c, cr, d)
    if m < 1e-9:
        return "unknown"
    if m == c:
        return "climb"
    if m == d:
        return "descent"
    return "cruise"


def est_flight_hours(df: pl.DataFrame) -> np.ndarray:
    frac = (
        df["end_fraction_of_flight"].to_numpy() - df["start_fraction_of_flight"].to_numpy()
    )
    frac = np.clip(frac.astype(np.float64), 1e-3, None)
    dur = df["duration_s"].to_numpy().astype(np.float64)
    return (dur / frac) / 3600.0


def haul_bucket(hours: float) -> str:
    if hours != hours:
        return "unknown"
    if hours < 2:
        return "short_<2h"
    if hours < 5:
        return "medium_2-5h"
    if hours < 8:
        return "long_5-8h"
    return "ultralong_>=8h"


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(p - y)))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(p - y))


def subgroup_rmse(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 5:
        return float("nan")
    return rmse(y[mask], p[mask])


def full_scorecard(
    name: str,
    rank_df: pl.DataFrame,
    final_df: pl.DataFrame,
    pred_rank: np.ndarray,
    pred_final: np.ndarray,
    *,
    hypothesis: str = "",
    expected_gain: str = "",
) -> dict[str, Any]:
    """Official metrics + key subgroup RMSEs on Rank/Final."""
    y_r = rank_df["actual_fuel_kg"].to_numpy()
    y_f = final_df["actual_fuel_kg"].to_numpy()
    y_c = np.concatenate([y_r, y_f])
    p_c = np.concatenate([pred_rank, pred_final])

    ac_r = rank_df["aircraft_type"].to_numpy().astype(str)
    ac_f = final_df["aircraft_type"].to_numpy().astype(str)
    ac_c = np.concatenate([ac_r, ac_f])

    hours_c = np.concatenate([est_flight_hours(rank_df), est_flight_hours(final_df)])
    haul_c = np.array([haul_bucket(h) for h in hours_c])

    # phase on combined
    phases_r = np.array([dominant_phase_row(r) for r in rank_df.iter_rows(named=True)])
    phases_f = np.array([dominant_phase_row(r) for r in final_df.iter_rows(named=True)])
    phase_c = np.concatenate([phases_r, phases_f])

    heavy_m = np.isin(ac_c, list(HEAVY_TYPES))
    narrow_m = np.isin(ac_c, list(NARROW_TYPES))
    a20n = ac_c == "A20N"
    a320 = ac_c == "A320"
    a359 = ac_c == "A359"
    b77w = ac_c == "B77W"
    b744 = ac_c == "B744"

    row = {
        "variant": name,
        "hypothesis": hypothesis,
        "expected_gain": expected_gain,
        "rank_rmse": rmse(y_r, pred_rank),
        "final_rmse": rmse(y_f, pred_final),
        "combined_rmse": rmse(y_c, p_c),
        "rank_mae": mae(y_r, pred_rank),
        "final_mae": mae(y_f, pred_final),
        "combined_mae": mae(y_c, p_c),
        "combined_bias": bias(y_c, p_c),
        "pct_overpredict": float(np.mean(p_c > y_c) * 100),
        "narrow_rmse": subgroup_rmse(y_c, p_c, narrow_m),
        "heavy_rmse": subgroup_rmse(y_c, p_c, heavy_m),
        "a20n_rmse": subgroup_rmse(y_c, p_c, a20n),
        "a320_rmse": subgroup_rmse(y_c, p_c, a320),
        "a359_rmse": subgroup_rmse(y_c, p_c, a359),
        "b77w_rmse": subgroup_rmse(y_c, p_c, b77w),
        "b744_rmse": subgroup_rmse(y_c, p_c, b744),
        "cruise_rmse": subgroup_rmse(y_c, p_c, phase_c == "cruise"),
        "climb_rmse": subgroup_rmse(y_c, p_c, phase_c == "climb"),
        "descent_rmse": subgroup_rmse(y_c, p_c, phase_c == "descent"),
        "ultralong_rmse": subgroup_rmse(y_c, p_c, haul_c == "ultralong_>=8h"),
        "medium_rmse": subgroup_rmse(y_c, p_c, haul_c == "medium_2-5h"),
        "delta_combined_vs_baseline": rmse(y_c, p_c) - BASELINE_OFFICIAL["combined_rmse"],
    }
    return row


def accept_gate(
    cand: dict[str, Any],
    baseline: dict[str, Any],
    *,
    narrow_tol: float = 3.0,
    official_floor: float | None = None,
) -> tuple[bool, str]:
    """Accept if Combined RMSE improves vs comparison baseline.

    ``official_floor`` (default BASELINE_OFFICIAL combined RMSE) is the
    published official ensemble score. A KEEP requires beating *both* the
    session comparison baseline and that floor when provided — avoids
    accepting improvements that only beat a degraded re-run.
    """
    floor = official_floor if official_floor is not None else BASELINE_OFFICIAL["combined_rmse"]
    if cand["combined_rmse"] >= baseline["combined_rmse"] - 0.05:
        return False, "combined_rmse not improved vs comparison baseline"
    if cand["combined_rmse"] >= floor - 0.05:
        return False, f"combined_rmse not below official floor {floor:.2f}"
    # allow tiny rank/final noise but not large regression vs official
    if cand["rank_rmse"] > BASELINE_OFFICIAL["rank_rmse"] + 5:
        return False, "rank_rmse regression >5 kg vs official"
    if cand["final_rmse"] > BASELINE_OFFICIAL["final_rmse"] + 5:
        return False, "final_rmse regression >5 kg vs official"
    if (
        np.isfinite(cand["a20n_rmse"])
        and np.isfinite(baseline.get("a20n_rmse", np.nan))
        and cand["a20n_rmse"] > baseline["a20n_rmse"] + narrow_tol
    ):
        return False, "a20n_rmse regression"
    if (
        np.isfinite(cand["a320_rmse"])
        and np.isfinite(baseline.get("a320_rmse", np.nan))
        and cand["a320_rmse"] > baseline["a320_rmse"] + narrow_tol
    ):
        return False, "a320_rmse regression"
    return True, "accepted"


# ---------------------------------------------------------------------------
# Base ensemble cache
# ---------------------------------------------------------------------------


@dataclass
class EnsembleBundle:
    feat_cols: list[str]
    full_models: list[Any]
    meta: Any
    meta_kind: str
    oof_pred: np.ndarray
    y_train: np.ndarray
    train_flight_ids: np.ndarray
    train_aircraft: np.ndarray
    P_oof: np.ndarray


def build_or_load_ensemble(
    train: pl.DataFrame,
    *,
    force: bool = False,
) -> EnsembleBundle:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feat_cols = ew_feature_cols(train)
    if OOF_CACHE.exists() and not force:
        LOGGER.info("Loading ensemble cache %s", OOF_CACHE)
        with open(OOF_CACHE, "rb") as f:
            bundle: EnsembleBundle = pickle.load(f)
        # quick consistency check
        if len(bundle.oof_pred) == len(train) and bundle.feat_cols == feat_cols:
            return bundle
        LOGGER.warning("Cache mismatch; rebuilding ensemble")

    LOGGER.info("Building official OOF ensemble (train only) — slow once...")
    P_oof, y_kg, full_models = build_oof_matrix(train, feat_cols, ENSEMBLE_BASES, n_splits=5)
    groups = train["flight_id"].to_numpy()
    meta_kind, meta = choose_meta_on_train_folds(P_oof, y_kg, groups, n_splits=5)
    oof_pred = np.asarray(meta.predict(P_oof), dtype=np.float64)
    # re-fit meta on full OOF for deployment (already done in choose)
    bundle = EnsembleBundle(
        feat_cols=feat_cols,
        full_models=full_models,
        meta=meta,
        meta_kind=meta_kind,
        oof_pred=oof_pred,
        y_train=y_kg,
        train_flight_ids=groups,
        train_aircraft=train["aircraft_type"].to_numpy().astype(str),
        P_oof=P_oof,
    )
    with open(OOF_CACHE, "wb") as f:
        pickle.dump(bundle, f)
    LOGGER.info("Cached ensemble to %s (meta=%s)", OOF_CACHE, meta_kind)
    return bundle


def predict_ensemble(
    bundle: EnsembleBundle,
    df: pl.DataFrame,
) -> np.ndarray:
    df = ensure_features(df, bundle.feat_cols)
    P = apply_bases(bundle.full_models, df, bundle.feat_cols)
    return np.asarray(bundle.meta.predict(P), dtype=np.float64)


# ---------------------------------------------------------------------------
# P1 calibrators (fit on train OOF only)
# ---------------------------------------------------------------------------


class AffineCalibrator:
    """pred' = a * pred + b, fit by least squares on OOF."""

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, y: np.ndarray, p: np.ndarray) -> AffineCalibrator:
        # design [p, 1]
        A = np.column_stack([p, np.ones_like(p)])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.a, self.b = float(coef[0]), float(coef[1])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.a * p + self.b


class IsotonicCalibrator:
    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)

    def fit(self, y: np.ndarray, p: np.ndarray) -> IsotonicCalibrator:
        self.iso.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(self.iso.predict(p), dtype=np.float64)


class ConditionalAffineCalibrator:
    """Separate affine map per group key (aircraft class / haul / phase)."""

    def __init__(self, group_fn: Callable[[pl.DataFrame], np.ndarray]) -> None:
        self.group_fn = group_fn
        self.models: dict[str, AffineCalibrator] = {}
        self.global_ = AffineCalibrator()

    def fit(self, df: pl.DataFrame, y: np.ndarray, p: np.ndarray) -> ConditionalAffineCalibrator:
        self.global_.fit(y, p)
        keys = self.group_fn(df)
        for k in np.unique(keys):
            m = keys == k
            if m.sum() < 50:
                continue
            cal = AffineCalibrator().fit(y[m], p[m])
            self.models[str(k)] = cal
        return self

    def transform(self, df: pl.DataFrame, p: np.ndarray) -> np.ndarray:
        keys = self.group_fn(df)
        out = np.empty_like(p)
        for i, k in enumerate(keys):
            cal = self.models.get(str(k), self.global_)
            out[i] = cal.a * p[i] + cal.b
        return out


def group_aircraft_class(df: pl.DataFrame) -> np.ndarray:
    return np.array([aircraft_class(a) for a in df["aircraft_type"].to_list()])


def group_haul(df: pl.DataFrame) -> np.ndarray:
    return np.array([haul_bucket(h) for h in est_flight_hours(df)])


def group_phase(df: pl.DataFrame) -> np.ndarray:
    return np.array([dominant_phase_row(r) for r in df.iter_rows(named=True)])


def apply_calibrator(cal: Any, df: pl.DataFrame, p: np.ndarray) -> np.ndarray:
    if isinstance(cal, ConditionalAffineCalibrator):
        return cal.transform(df, p)
    return cal.transform(p)


# ---------------------------------------------------------------------------
# P2 heavy specialist
# ---------------------------------------------------------------------------


def train_heavy_specialist(
    train: pl.DataFrame,
    feat_cols: list[str],
    model_key: str = "lgbm",
) -> Any:
    """Fuel-flow model on heavy aircraft rows only."""
    heavy = train.filter(pl.col("aircraft_type").is_in(list(HEAVY_TYPES)))
    LOGGER.info("Heavy specialist train rows: %d / %d", len(heavy), len(train))
    if len(heavy) < 500:
        raise RuntimeError("Too few heavy rows for specialist")
    X, y_space, _, dur = prepare_xy(heavy, feat_cols, "fuel_flow")
    return train_model(model_key, X, y_space, feat_cols)


def predict_heavy_routed(
    specialist: Any,
    feat_cols: list[str],
    df: pl.DataFrame,
    base_pred: np.ndarray,
) -> np.ndarray:
    """Hard route: heavy types use specialist FuelFlow; else base_pred."""
    df = ensure_features(df, feat_cols)
    ac = df["aircraft_type"].to_numpy().astype(str)
    heavy_m = np.isin(ac, list(HEAVY_TYPES))
    out = base_pred.copy()
    if heavy_m.any():
        sub = df.filter(pl.Series(heavy_m))
        X, _, _, dur = prepare_xy(sub, feat_cols, "direct")
        pred_h = predict_fuel_kg(specialist, X, dur, "fuel_flow")
        out[np.flatnonzero(heavy_m)] = pred_h
    return out


# ---------------------------------------------------------------------------
# R1 — Heavy specialist with OpenAP descriptors + interactions
# ---------------------------------------------------------------------------

OPENAP_DESCRIPTOR_PATH = project_root() / "figures" / "table_aircraft_openap_descriptors.csv"

OPENAP_DESCRIPTOR_COLS = [
    "mtow_kg",
    "mlw_kg",
    "oew_kg",
    "mfc_kg",
    "cruise_mach",
    "cruise_range_km",
    "wing_area_m2",
    "wing_span_m",
    "mmo",
    "max_thrust_n",
]

R1_INTERACTION_COLS = [
    "r1_cruise_alt_x_dur",
    "r1_mean_alt_x_dur",
    "r1_cruise_ratio_x_dur",
    "r1_wing_loading_mtow_wingarea",
    "r1_thrust_loading_mtow_thrust",
    "r1_aspect_ratio",
    "r1_oew_mtow_ratio",
    "r1_fuel_capacity_ratio",
]


def _load_openap_descriptors() -> pl.DataFrame:
    return pl.read_csv(OPENAP_DESCRIPTOR_PATH).select(
        ["aircraft_type"] + OPENAP_DESCRIPTOR_COLS
    )


def _make_r1_interactions(df: pl.DataFrame) -> pl.DataFrame:
    has_cruise_alt = "max_altitude" in df.columns
    has_dur = "duration_s" in df.columns
    has_cruise_frac = "cruise_fraction" in df.columns

    exprs = []
    if has_cruise_alt and has_dur:
        exprs.append(
            (pl.col("max_altitude") * pl.col("duration_s")).alias("r1_cruise_alt_x_dur")
        )
    if has_dur:
        alt_col = "mean_altitude" if "mean_altitude" in df.columns else "max_altitude"
        if alt_col in df.columns:
            exprs.append(
                (pl.col(alt_col) * pl.col("duration_s")).alias("r1_mean_alt_x_dur")
            )
    if has_cruise_frac and has_dur:
        exprs.append(
            (pl.col("cruise_fraction") * pl.col("duration_s")).alias("r1_cruise_ratio_x_dur")
        )

    if "mtow_kg" in df.columns and "wing_area_m2" in df.columns:
        exprs.append(
            (pl.col("mtow_kg") / pl.col("wing_area_m2").clip(1.0)).alias("r1_wing_loading_mtow_wingarea")
        )
    if "mtow_kg" in df.columns and "max_thrust_n" in df.columns:
        exprs.append(
            (pl.col("mtow_kg") / pl.col("max_thrust_n").clip(1.0)).alias("r1_thrust_loading_mtow_thrust")
        )
    if "wing_span_m" in df.columns and "wing_area_m2" in df.columns:
        exprs.append(
            ((pl.col("wing_span_m") ** 2) / pl.col("wing_area_m2").clip(1.0)).alias("r1_aspect_ratio")
        )
    if "oew_kg" in df.columns and "mtow_kg" in df.columns:
        exprs.append(
            (pl.col("oew_kg") / pl.col("mtow_kg").clip(1.0)).alias("r1_oew_mtow_ratio")
        )
    if "mfc_kg" in df.columns and "mtow_kg" in df.columns:
        exprs.append(
            (pl.col("mfc_kg") / pl.col("mtow_kg").clip(1.0)).alias("r1_fuel_capacity_ratio")
        )

    if not exprs:
        return df
    return df.with_columns(exprs)


def _augment_heavy_with_descriptors(
    df: pl.DataFrame,
    *,
    include_interactions: bool = True,
) -> pl.DataFrame:
    desc = _load_openap_descriptors()
    df = df.join(desc, on="aircraft_type", how="left")
    if include_interactions:
        df = _make_r1_interactions(df)
    return df


def r1_feature_cols(base_feat_cols: list[str]) -> list[str]:
    extra = OPENAP_DESCRIPTOR_COLS + R1_INTERACTION_COLS
    return list(dict.fromkeys(base_feat_cols + extra))


def train_heavy_specialist_r1(
    train: pl.DataFrame,
    feat_cols: list[str],
    model_key: str = "lgbm",
) -> tuple[Any, list[str]]:
    heavy = train.filter(pl.col("aircraft_type").is_in(list(HEAVY_TYPES)))
    LOGGER.info("R1 heavy specialist train rows: %d / %d", len(heavy), len(train))
    if len(heavy) < 500:
        raise RuntimeError("Too few heavy rows for R1 specialist")
    heavy = _augment_heavy_with_descriptors(heavy)
    r1_cols = r1_feature_cols(feat_cols)
    present = [c for c in r1_cols if c in heavy.columns]
    X, y_space, _, dur = prepare_xy(heavy, present, "fuel_flow")
    model = train_model(model_key, X, y_space, present)
    return model, present


def predict_heavy_routed_r1(
    specialist: Any,
    feat_cols: list[str],
    df: pl.DataFrame,
    base_pred: np.ndarray,
) -> np.ndarray:
    df = ensure_features(df, feat_cols)
    ac = df["aircraft_type"].to_numpy().astype(str)
    heavy_m = np.isin(ac, list(HEAVY_TYPES))
    out = base_pred.copy()
    if heavy_m.any():
        sub = df.filter(pl.Series(heavy_m))
        sub = _augment_heavy_with_descriptors(sub)
        r1_cols = r1_feature_cols(feat_cols)
        present = [c for c in r1_cols if c in sub.columns]
        X, _, _, dur = prepare_xy(sub, present, "direct")
        pred_h = predict_fuel_kg(specialist, X, dur, "fuel_flow")
        out[np.flatnonzero(heavy_m)] = pred_h
    return out


# ---------------------------------------------------------------------------
# R2 — Heavy specialist with expanded physics features
# (aircraft chars, mass proxies, cruise features, physics interactions)
# ---------------------------------------------------------------------------

R2_AIRCRAFT_FEATURES = [
    "r2_engine_count",
    "r2_thrust_to_weight",
    "r2_wing_loading",
    "r2_payload_capacity",
]

R2_MASS_FEATURES = [
    "r2_oew_as_mass",             # OEW = minimum realistic mass
    "r2_tofr_mass_est",           # takeoff frac-based mass estimate
    "r2_phase_aware_mass",        # phase-varying mass proxy
]

R2_CRUISE_FEATURES = [
    "r2_cruise_duration_s",       # duration_s * cruise_fraction
    "r2_cruise_altitude_band",    # altitude / design cruise altitude proxy
    "r2_cruise_fuel_flow",        # physics_fuel_kg * cruise_fraction / cruise_duration
]

R2_PHYSICS_INTERACTIONS = [
    "r2_mtow_x_cruise_duration",
    "r2_mtow_x_cruise_mach",
    "r2_cruise_mass_x_mach",
    "r2_wl_altitude",
    "r2_twr_climb_rate",
    "r2_cruise_ratio_x_dur_sq",
]


def _compute_r2_features(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []

    has_mtow = "mtow_kg" in df.columns
    has_oew = "oew_kg" in df.columns
    has_thr = "max_thrust_n" in df.columns
    has_wa = "wing_area_m2" in df.columns
    has_cm = "cruise_mach" in df.columns
    has_df = "duration_s" in df.columns
    has_cf = "cruise_fraction" in df.columns
    has_ma = "max_altitude" in df.columns
    has_al = "mean_altitude" in df.columns
    has_pf = "physics_fuel_kg" in df.columns
    has_ref = "ref_mass_kg" in df.columns

    dur = pl.col("duration_s").clip(lower_bound=1.0) if has_df else pl.lit(1.0)

    # --- Aircraft characteristics ---
    if has_thr:
        exprs.append(
            (pl.col("max_thrust_n") / 100000.0).round(0).cast(pl.Int64).alias("r2_engine_count")
        )
    if has_mtow and has_thr:
        exprs.append(
            (pl.col("max_thrust_n") / pl.col("mtow_kg").clip(1.0)).alias("r2_thrust_to_weight")
        )
    if has_mtow and has_wa:
        exprs.append(
            (pl.col("mtow_kg") / pl.col("wing_area_m2").clip(1.0)).alias("r2_wing_loading")
        )
    if has_mtow and has_oew:
        exprs.append(
            ((pl.col("mtow_kg") - pl.col("oew_kg")) / pl.col("mtow_kg").clip(1.0)).alias("r2_payload_capacity")
        )

    # --- Mass proxies ---
    if has_oew:
        exprs.append(pl.col("oew_kg").alias("r2_oew_as_mass"))
    if has_ref and "start_fraction_of_flight" in df.columns:
        exprs.append(
            (pl.col("ref_mass_kg") * (1.0 - 0.15 * pl.col("start_fraction_of_flight"))).alias("r2_tofr_mass_est")
        )
    if has_oew and has_mtow and has_df and has_cf:
        mass_mid = pl.col("oew_kg") + 0.5 * (pl.col("mtow_kg") - pl.col("oew_kg"))
        exprs.append(
            pl.when(pl.col("cruise_fraction") > 0.7)
            .then(pl.col("oew_kg") + 0.25 * (pl.col("mtow_kg") - pl.col("oew_kg")))
            .otherwise(mass_mid)
            .alias("r2_phase_aware_mass")
        )

    # --- Cruise features ---
    if has_df and has_cf:
        exprs.append(
            (pl.col("duration_s") * pl.col("cruise_fraction")).alias("r2_cruise_duration_s")
        )
    if has_ma and has_df and has_cf:
        cruise_alt = pl.col("mean_altitude").fill_null(pl.col("max_altitude").fill_null(0.0))
        exprs.append(
            (cruise_alt / 11000.0).clip(0.0, 1.5).alias("r2_cruise_altitude_band")
        )
    if has_pf and has_df and has_cf:
        cruise_dur = pl.col("duration_s") * pl.col("cruise_fraction").clip(lower_bound=1.0)
        cruise_ff = pl.col("physics_fuel_kg") * pl.col("cruise_fraction") / cruise_dur
        exprs.append(cruise_ff.alias("r2_cruise_fuel_flow"))

    # --- Physics interactions ---
    if has_mtow and has_df and has_cf:
        exprs.append(
            (pl.col("mtow_kg") * pl.col("duration_s") * pl.col("cruise_fraction")).alias("r2_mtow_x_cruise_duration")
        )
    if has_mtow and has_cm:
        cm = pl.col("cruise_mach").clip(0.6, 1.0)
        exprs.append(
            (pl.col("mtow_kg") * cm).alias("r2_mtow_x_cruise_mach")
        )
    if has_oew and has_mtow and has_cm and has_df and has_cf:
        cruise_mass = pl.col("oew_kg") + 0.3 * (pl.col("mtow_kg") - pl.col("oew_kg"))
        cm = pl.col("cruise_mach").clip(0.6, 1.0)
        exprs.append(
            (cruise_mass * cm).alias("r2_cruise_mass_x_mach")
        )
    if has_mtow and has_wa and has_al:
        wl = pl.col("mtow_kg") / pl.col("wing_area_m2").clip(1.0)
        exprs.append(
            (wl * pl.col("mean_altitude").fill_null(0.0) / 1000.0).alias("r2_wl_altitude")
        )
    if has_mtow and has_thr:
        twr = pl.col("max_thrust_n") / pl.col("mtow_kg").clip(1.0)
        if "mean_vertical_rate" in df.columns:
            exprs.append(
                (twr * pl.col("mean_vertical_rate").fill_null(0.0)).alias("r2_twr_climb_rate")
            )
        else:
            exprs.append((twr * pl.lit(0.0)).alias("r2_twr_climb_rate"))
    if has_df and has_cf:
        exprs.append(
            (pl.col("cruise_fraction").pow(2) * pl.col("duration_s")).alias("r2_cruise_ratio_x_dur_sq")
        )

    if not exprs:
        return df
    return df.with_columns(exprs)


def _augment_heavy_r2(
    df: pl.DataFrame,
    *,
    include_interactions: bool = True,
) -> pl.DataFrame:
    """Augment heavy subframe with OpenAP descriptors and R2 physics features."""
    desc = _load_openap_descriptors()
    df = df.join(desc, on="aircraft_type", how="left")
    df = _make_r1_interactions(df)
    if include_interactions:
        df = _compute_r2_features(df)
    return df


R2_EXTRA_COLS = (
    OPENAP_DESCRIPTOR_COLS
    + R1_INTERACTION_COLS
    + R2_AIRCRAFT_FEATURES
    + R2_MASS_FEATURES
    + R2_CRUISE_FEATURES
    + R2_PHYSICS_INTERACTIONS
)


def r2_feature_cols(base_feat_cols: list[str]) -> list[str]:
    return list(dict.fromkeys(base_feat_cols + R2_EXTRA_COLS))


def train_heavy_specialist_r2(
    train: pl.DataFrame,
    feat_cols: list[str],
    model_key: str = "cat",
) -> tuple[Any, list[str]]:
    heavy = train.filter(pl.col("aircraft_type").is_in(list(HEAVY_TYPES)))
    LOGGER.info("R2 heavy specialist train rows: %d / %d", len(heavy), len(train))
    if len(heavy) < 500:
        raise RuntimeError("Too few heavy rows for R2 specialist")
    heavy = _augment_heavy_r2(heavy)
    r2_cols = r2_feature_cols(feat_cols)
    present = [c for c in r2_cols if c in heavy.columns]
    X, y_space, _, dur = prepare_xy(heavy, present, "fuel_flow")
    model = train_model(model_key, X, y_space, present)
    return model, present


def predict_heavy_routed_r2(
    specialist: Any,
    feat_cols: list[str],
    df: pl.DataFrame,
    base_pred: np.ndarray,
) -> np.ndarray:
    df = ensure_features(df, feat_cols)
    ac = df["aircraft_type"].to_numpy().astype(str)
    heavy_m = np.isin(ac, list(HEAVY_TYPES))
    out = base_pred.copy()
    if heavy_m.any():
        sub = df.filter(pl.Series(heavy_m))
        sub = _augment_heavy_r2(sub)
        r2_cols = r2_feature_cols(feat_cols)
        present = [c for c in r2_cols if c in sub.columns]
        X, _, _, dur = prepare_xy(sub, present, "direct")
        pred_h = predict_fuel_kg(specialist, X, dur, "fuel_flow")
        out[np.flatnonzero(heavy_m)] = pred_h
    return out


# ---------------------------------------------------------------------------
# P5 simple ensemble reweight on OOF
# ---------------------------------------------------------------------------


def nonnegative_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to 1 minimizing RMSE (SLSQP)."""
    from scipy.optimize import minimize

    m = P.shape[1]

    def loss(w):
        pred = P @ w
        return np.sqrt(np.mean((pred - y) ** 2))

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * m
    w0 = np.ones(m) / m
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return w


def flow_only_indices() -> list[int]:
    # ENSEMBLE_BASES order: 0 xgb d, 1 lgbm d, 2 cat d, 3 xgb f, 4 lgbm f, 5 cat f
    return [3, 4, 5]
