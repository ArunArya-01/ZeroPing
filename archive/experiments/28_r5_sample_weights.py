"""R5 — Sample Weighting Strategies for PRC Fuel Prediction.

Implements aviation-justified sample weighting during LGBM/CatBoost training.
All weights derived exclusively from training-data attributes (aircraft_type,
flight_hours, cruise_fraction) — never from targets or validation labels.

Strategies:
  R5a: Uniform (baseline, weight=1.0)
  R5b: Aircraft-family weighting — heavy types weighted by SSE contribution
  R5c: Heavy-specific weighting — A359/B77W/B744 receive higher weights
  R5d: Duration weighting — longer flights receive more weight
  R5e: Cruise weighting — cruise-dominated intervals weighted higher
  R5f: Combined — heavy + duration
  R5g: Inverse-count weighting — rare aircraft types upweighted
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import project_root
from physics.gap_closing import (
    BASELINE_OFFICIAL, HEAVY_TYPES, NARROW_TYPES, ENSEMBLE_BASES,
    ConditionalAffineCalibrator, full_scorecard, group_phase,
    load_splits, est_flight_hours, ensure_features,
)
from physics.mass_model import enrich_mass_from_columns, R3_MASS_FEATURES
from physics.official_benchmark import (
    ew_feature_cols, build_oof_matrix, choose_meta_on_train_folds,
    apply_bases, prepare_xy, train_model,
    CAT_FEATURES, LGBM_PARAMS, XGB_PARAMS, CAT_PARAMS,
    fit_meta, predict_fuel_kg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r5_weights")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


# =========================================================================
# Modified train_model with sample_weight support
# =========================================================================
def train_model_weighted(model_key: str, X, y: np.ndarray, feature_cols: list[str],
                         sample_weight: np.ndarray | None = None):
    """Train a tree model with optional sample weights."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = [c for c in feature_cols if c not in CAT_FEATURES]
    cat = [c for c in feature_cols if c in CAT_FEATURES]
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="drop",
    )
    if model_key == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
    elif model_key == "cat":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**CAT_PARAMS)
    elif model_key == "xgb":
        import xgboost as xgb
        model = xgb.XGBRegressor(**XGB_PARAMS)
    else:
        raise ValueError(model_key)

    pipe = Pipeline([("prep", prep), ("model", model)])

    if sample_weight is not None and model_key in ("lgbm", "cat", "xgb"):
        # GBDT models accept sample_weight via fit()
        pipe.fit(X, y, model__sample_weight=sample_weight)
    else:
        pipe.fit(X, y)
    return pipe


# =========================================================================
# Weight generation functions (train-data only, no target leakage)
# =========================================================================

def make_weights_heavy(df: pl.DataFrame, multiplier: float = 3.0) -> np.ndarray:
    """Weight heavy aircraft higher (multiplier)."""
    ac = df["aircraft_type"].to_numpy().astype(str)
    w = np.ones(len(df), dtype=np.float64)
    heavy_mask = np.isin(ac, list(HEAVY_TYPES))
    w[heavy_mask] = multiplier
    return w / w.mean()  # normalize to mean=1


def make_weights_target_heavies(df: pl.DataFrame, mult_b744: float = 5.0,
                                 mult_b77w: float = 4.0, mult_a359: float = 3.0) -> np.ndarray:
    """Weight A359/B77W/B744 higher based on observed SSE concentration."""
    ac = df["aircraft_type"].to_numpy().astype(str)
    w = np.ones(len(df), dtype=np.float64)
    w[ac == "B744"] = mult_b744
    w[ac == "B77W"] = mult_b77w
    w[ac == "A359"] = mult_a359
    return w / w.mean()


def make_weights_duration(df: pl.DataFrame) -> np.ndarray:
    """Weight by flight duration — longer flights get more weight."""
    hours = est_flight_hours(df)
    # Longer flights: weight proportional to log(hours), clamped
    w = np.log1p(np.clip(hours, 0.5, 20.0))
    return w / w.mean()


def make_weights_cruise(df: pl.DataFrame, cruise_boost: float = 2.0) -> np.ndarray:
    """Weight cruise-dominated intervals higher."""
    cf = df["cruise_fraction"].to_numpy().astype(np.float64)
    w = np.ones(len(df), dtype=np.float64)
    cruise_mask = cf > 0.7
    w[cruise_mask] = cruise_boost
    return w / w.mean()


def make_weights_combined(df: pl.DataFrame) -> np.ndarray:
    """Combine heavy + duration weighting multiplicatively."""
    wh = make_weights_target_heavies(df)
    wd = make_weights_duration(df)
    w = wh * wd
    return w / w.mean()


def make_weights_inverse_count(df: pl.DataFrame) -> np.ndarray:
    """Inverse frequency weighting per aircraft type."""
    ac = df["aircraft_type"].to_numpy().astype(str)
    unique, counts = np.unique(ac, return_counts=True)
    type_weight = {u: 1.0 / max(c, 1) for u, c in zip(unique, counts)}
    # Normalize to mean=1
    w_raw = np.array([type_weight.get(a, 1.0) for a in ac], dtype=np.float64)
    return w_raw / w_raw.mean()


WEIGHT_STRATEGIES = {
    "R5a_uniform": (None, "Baseline: all samples weight=1.0"),
    "R5b_heavy_3x": (lambda df: make_weights_heavy(df, 3.0), "Heavy types weighted 3x"),
    "R5c_target_heavies": (lambda df: make_weights_target_heavies(df),
                           "B744=5x, B77W=4x, A359=3x (SSE-proportional)"),
    "R5d_duration": (lambda df: make_weights_duration(df), "Weight by log(flight_hours)"),
    "R5e_cruise": (lambda df: make_weights_cruise(df, 2.0), "Cruise-dominant intervals weighted 2x"),
    "R5f_combined": (lambda df: make_weights_combined(df), "Heavy × duration combined"),
    "R5g_inverse_count": (lambda df: make_weights_inverse_count(df), "Inverse aircraft frequency"),
}


# =========================================================================
# Modified build_oof_matrix with sample weights
# =========================================================================
def build_oof_matrix_weighted(
    df_train: pl.DataFrame,
    feature_cols: list[str],
    model_specs: list[tuple[str, str]],
    sample_weight: np.ndarray | None = None,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """GroupKFold OOF with per-sample weights."""
    from sklearn.model_selection import GroupKFold

    X, y_direct, y_kg, dur = prepare_xy(df_train, feature_cols, "direct")
    y_flow = y_kg / dur
    groups = df_train["flight_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    n = len(y_kg)
    m = len(model_specs)
    P = np.zeros((n, m), dtype=np.float64)

    for j, (mkey, target) in enumerate(model_specs):
        LOGGER.info("OOF base model %s / %s", mkey, target)
        y_space = y_flow if target == "fuel_flow" else y_direct
        for fold, (tr, va) in enumerate(gkf.split(X, y_space, groups)):
            w_tr = sample_weight[tr] if sample_weight is not None else None
            pipe = train_model_weighted(mkey, X.iloc[tr], y_space[tr], feature_cols, w_tr)
            pred = predict_fuel_kg(pipe, X.iloc[va], dur[va], target)
            P[va, j] = pred

    full_models = []
    for mkey, target in model_specs:
        y_space = y_flow if target == "fuel_flow" else y_direct
        pipe = train_model_weighted(mkey, X, y_space, feature_cols, sample_weight)
        full_models.append((mkey, target, pipe))
    return P, y_kg, full_models


# =========================================================================
# Main evaluation
# =========================================================================
def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    # Enrich with mass features
    train = enrich_mass_from_columns(train)
    rank = enrich_mass_from_columns(rank)
    final = enrich_mass_from_columns(final)

    base_feat_cols = ew_feature_cols(train)
    mass_cols = [c for c in R3_MASS_FEATURES if c in train.columns]
    feat_cols = list(dict.fromkeys(base_feat_cols + mass_cols))
    LOGGER.info("Features: %d (base=%d + mass=%d)", len(feat_cols), len(base_feat_cols), len(mass_cols))

    leaderboard: list[dict] = []

    # =========================================================================
    # Step 1: Error distribution audit (train OOF with uniform weights)
    # =========================================================================
    LOGGER.info("=== STEP 1: Train OOF baseline (uniform) ===")
    P_base, y_base, models_base = build_oof_matrix_weighted(train, feat_cols, ENSEMBLE_BASES, None, n_splits=5)
    oof_pred = np.asarray(fit_meta(P_base, y_base, "ridge").predict(P_base), dtype=np.float64)
    groups = train["flight_id"].to_numpy()
    ac_train = train["aircraft_type"].to_numpy().astype(str)
    hours_train = est_flight_hours(train)

    # Per-group error audit on OOF
    errors = oof_pred - y_base
    abs_err = np.abs(errors)

    audit_rows = []
    # By aircraft type
    for ac in ["A359", "B77W", "B744", "A320", "A20N", "B738"]:
        m = ac_train == ac
        if m.sum() < 20: continue
        sse = float((errors[m] ** 2).sum())
        total_sse = float((errors ** 2).sum())
        audit_rows.append({
            "group": f"aircraft={ac}",
            "count": int(m.sum()),
            "train_oof_rmse": float(np.sqrt((errors[m] ** 2).mean())),
            "train_oof_mae": float(abs_err[m].mean()),
            "bias": float(errors[m].mean()),
            "sse_pct": 100 * sse / total_sse,
        })

    # By duration
    for label, lo, hi in [("<2h", 0, 2), ("2-4h", 2, 4), ("4-8h", 4, 8), ("≥8h", 8, 99)]:
        h = hours_train
        m = (h >= lo) & (h < hi)
        if m.sum() < 20: continue
        sse = float((errors[m] ** 2).sum())
        total_sse = float((errors ** 2).sum())
        audit_rows.append({
            "group": f"duration={label}",
            "count": int(m.sum()),
            "train_oof_rmse": float(np.sqrt((errors[m] ** 2).mean())),
            "train_oof_mae": float(abs_err[m].mean()),
            "bias": float(errors[m].mean()),
            "sse_pct": 100 * sse / total_sse,
        })

    # By cruise fraction
    cf_train = train["cruise_fraction"].to_numpy().astype(np.float64)
    for label, lo, hi in [("low_cruise<0.5", 0, 0.5), ("med_cruise0.5-0.8", 0.5, 0.8), ("high_cruise>0.8", 0.8, 1.01)]:
        m = (cf_train >= lo) & (cf_train < hi)
        if m.sum() < 20: continue
        sse = float((errors[m] ** 2).sum())
        total_sse = float((errors ** 2).sum())
        audit_rows.append({
            "group": f"cruise={label}",
            "count": int(m.sum()),
            "train_oof_rmse": float(np.sqrt((errors[m] ** 2).mean())),
            "train_oof_mae": float(abs_err[m].mean()),
            "bias": float(errors[m].mean()),
            "sse_pct": 100 * sse / total_sse,
        })

    LOGGER.info("Error audit: %d groups", len(audit_rows))
    for r in sorted(audit_rows, key=lambda x: x["sse_pct"], reverse=True)[:8]:
        LOGGER.info("  %s: n=%d rmse=%.0f bias=%+.0f sse=%.1f%%",
                    r["group"], r["count"], r["train_oof_rmse"], r["bias"], r["sse_pct"])

    audit_df = pl.DataFrame(audit_rows)
    audit_df.write_csv(OUT / "table_r5_error_audit.csv")

    # =========================================================================
    # Step 2-5: Evaluate each weighting strategy
    # =========================================================================
    rank_b = ensure_features(rank, feat_cols)
    final_b = ensure_features(final, feat_cols)

    for strategy_name, (weight_fn, description) in WEIGHT_STRATEGIES.items():
        LOGGER.info("=== %s: %s ===", strategy_name, description)

        # Compute weights from train data
        w_train = weight_fn(train) if weight_fn is not None else None

        # Quick single-model check first
        X_tr, y_flow, y_kg_tr, dur_tr = prepare_xy(train, feat_cols, "fuel_flow")
        model = train_model_weighted("lgbm", X_tr, y_flow, feat_cols, w_train)

        X_r, _, _, dur_r = prepare_xy(rank_b, feat_cols, "direct")
        X_f, _, _, dur_f = prepare_xy(final_b, feat_cols, "direct")
        pred_r = predict_fuel_kg(model, X_r, dur_r, "fuel_flow")
        pred_f = predict_fuel_kg(model, X_f, dur_f, "fuel_flow")

        card = full_scorecard(strategy_name, rank_b, final_b, pred_r, pred_f,
                              hypothesis=description, expected_gain="−1 to −5 kg")
        card["weight_strategy"] = strategy_name
        card["delta_vs_221_33"] = card["combined_rmse"] - 221.33
        leaderboard.append(card)

        LOGGER.info("%s: combined=%.2f heavy=%.1f narrow=%.1f bias=%.2f delta=%.2f",
                    strategy_name, card["combined_rmse"], card["heavy_rmse"],
                    card["narrow_rmse"], card["combined_bias"],
                    card["combined_rmse"] - 221.33)

    # =========================================================================
    # Full ensemble OOF with best weighting strategy
    # =========================================================================
    for strategy_name, (weight_fn, description) in [
        ("R5b_heavy_3x", WEIGHT_STRATEGIES["R5b_heavy_3x"]),
        ("R5c_target_heavies", WEIGHT_STRATEGIES["R5c_target_heavies"]),
        ("R5f_combined", WEIGHT_STRATEGIES["R5f_combined"]),
    ]:
        LOGGER.info("=== Ensemble OOF + %s ===", strategy_name)
        w_train = weight_fn(train)
        P, y, models = build_oof_matrix_weighted(train, feat_cols, ENSEMBLE_BASES, w_train, n_splits=5)
        meta_kind, meta = choose_meta_on_train_folds(P, y, groups, n_splits=5)
        oof_pred = np.asarray(meta.predict(P), dtype=np.float64)
        LOGGER.info("Meta: %s, train OOF RMSE: %.2f", meta_kind,
                    float(np.sqrt(np.mean((oof_pred - y) ** 2))))

        rank_a = ensure_features(rank, feat_cols)
        final_a = ensure_features(final, feat_cols)
        P_r = apply_bases(models, rank_a, feat_cols)
        P_f = apply_bases(models, final_a, feat_cols)
        pred_r0 = np.asarray(meta.predict(P_r), dtype=np.float64)
        pred_f0 = np.asarray(meta.predict(P_f), dtype=np.float64)

        # P1E
        cal_phase = ConditionalAffineCalibrator(group_phase).fit(train, y, oof_pred)
        pr = cal_phase.transform(rank_a, pred_r0)
        pf = cal_phase.transform(final_a, pred_f0)
        card = full_scorecard(f"{strategy_name}_ensemble_P1E", rank_a, final_a, pr, pf,
                              hypothesis=f"Ensemble + P1E with {description}")
        card["weight_strategy"] = strategy_name
        card["delta_vs_221_33"] = card["combined_rmse"] - 221.33
        leaderboard.append(card)
        LOGGER.info("Ensemble %s: combined=%.2f heavy=%.1f narrow=%.1f delta=%.2f",
                    strategy_name, card["combined_rmse"], card.get("heavy_rmse", 0),
                    card.get("narrow_rmse", 0), card["combined_rmse"] - 221.33)

    # =========================================================================
    # Save results
    # =========================================================================
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    r5_rows = [r for r in leaderboard if "R5" in r["variant"]]
    if r5_rows:
        pl.DataFrame(r5_rows).write_csv(OUT / "table_rmse_R5_weights.csv")
    lb.write_csv(OUT / "table_rmse_R5_full_leaderboard.csv")

    summary = {
        "task": "R5",
        "baseline_221_33": 221.33,
        "best_variant": min(r5_rows, key=lambda x: x["combined_rmse"])["variant"] if r5_rows else "none",
        "best_combined_rmse": min(r5_rows, key=lambda x: x["combined_rmse"])["combined_rmse"] if r5_rows else None,
        "n_variants": len(r5_rows),
    }
    (OUT / "r5_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== R5 SAMPLE WEIGHTS SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\nLeaderboard:")
    for row in lb.head(12).iter_rows(named=True):
        d = f"  Δ221={row['combined_rmse'] - 221.33:+.2f}" if "combined_rmse" in row else ""
        print(f"  {row['variant']:<50s} Combined={row['combined_rmse']:.2f}{d}")


if __name__ == "__main__":
    main()
