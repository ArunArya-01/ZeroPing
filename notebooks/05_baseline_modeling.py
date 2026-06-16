"""
Phase 4: Baseline Modeling — residual learning vs OpenAP alone.

Uses a strict flight-level train/test split (no flight in both sets).

Run:
    python notebooks/05_baseline_modeling.py

Input:  featured_dataset.parquet (project root, must include flight_id)
Output: figures/table_*.csv, figures/fig_*.png
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "featured_dataset.parquet"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "duration_s",
    "start_fraction_of_flight",
    "end_fraction_of_flight",
    "n_traj_pts",
    "has_acars_in_window",
    "mean_altitude",
    "median_altitude",
    "max_altitude",
    "std_altitude",
    "mean_groundspeed",
    "std_groundspeed",
    "max_groundspeed",
    "mean_vertical_rate",
    "std_vertical_rate",
    "climb_fraction",
    "cruise_fraction",
    "descent_fraction",
]

CATEGORICAL_FEATURES = [
    "aircraft_type",
    "method",
    "origin_icao",
    "destination_icao",
]

CORR_FEATURES = NUMERIC_FEATURES + ["physics_fuel_kg"]

KEY_FEATURES = [
    "aircraft_type",
    "n_traj_pts",
    "duration_s",
    "method",
    "cruise_fraction",
]


def load_and_clean() -> tuple[pl.DataFrame, int, int]:
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} not found. Run physics/build_featured_dataset.py first.")

    df = pl.read_parquet(PARQUET)
    if "flight_id" not in df.columns:
        raise ValueError(
            "featured_dataset.parquet is missing flight_id. "
            "Run: python physics/build_featured_dataset.py --patch-flight-id"
        )

    n_before = len(df)
    df = (
        df.drop_nulls(subset=["physics_fuel_kg", "residual_kg", "flight_id"])
        .filter(
            pl.col("physics_fuel_kg").is_finite()
            & pl.col("residual_kg").is_finite()
            & pl.col("actual_fuel_kg").is_finite()
        )
    )
    n_after = len(df)
    return df, n_before, n_after


def flight_level_split(
    flight_ids: np.ndarray, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_flights = np.unique(flight_ids)
    train_fids, test_fids = train_test_split(
        unique_flights, test_size=test_size, random_state=random_state
    )
    train_fid_set = set(train_fids.tolist())
    test_fid_set = set(test_fids.tolist())
    overlap = train_fid_set & test_fid_set
    if overlap:
        raise RuntimeError(f"Flight leakage detected: {len(overlap)} flights in both splits.")

    train_mask = np.isin(flight_ids, train_fids)
    test_mask = np.isin(flight_ids, test_fids)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    return train_idx, test_idx, train_fids, test_fids


def compute_residual_correlations(df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    residual = df["residual_kg"].to_numpy()

    for col in CORR_FEATURES:
        if col not in df.columns:
            continue
        x = df[col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(residual)
        if mask.sum() < 3:
            continue
        pearson = float(np.corrcoef(x[mask], residual[mask])[0, 1])
        spearman, _ = spearmanr(x[mask], residual[mask])
        rows.append(
            {
                "feature": col,
                "pearson": pearson,
                "spearman": float(spearman),
                "abs_pearson": abs(pearson),
            }
        )

    return (
        pl.DataFrame(rows)
        .sort("abs_pearson", descending=True)
        .drop("abs_pearson")
    )


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_models() -> dict[str, object]:
    return {
        "Linear Regression": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    lgb.LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    xgb.XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        verbosity=0,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            ohe: OneHotEncoder = transformer
            names.extend(ohe.get_feature_names_out(cols).tolist())
    return names


def aggregate_importance(
    feature_names: list[str], importances: np.ndarray
) -> dict[str, float]:
    agg: dict[str, float] = {k: 0.0 for k in KEY_FEATURES}
    for fname, imp in zip(feature_names, importances):
        for key in KEY_FEATURES:
            if fname == key or fname.startswith(f"{key}_"):
                agg[key] += float(imp)
    return agg


def main() -> None:
    print("=" * 70)
    print("PHASE 4: BASELINE MODELING (FLIGHT-LEVEL SPLIT)")
    print("=" * 70)

    df, n_before, n_after = load_and_clean()
    print(f"\n1. DATA CLEANING")
    print(f"   Rows before: {n_before:,}")
    print(f"   Rows after:  {n_after:,}")
    print(f"   Removed:     {n_before - n_after:,} ({100 * (n_before - n_after) / n_before:.2f}%)")

    print(f"\n2. RESIDUAL CORRELATIONS (cleaned data, n={n_after:,})")
    corr_df = compute_residual_correlations(df)
    corr_path = OUT / "table_top_residual_predictors.csv"
    corr_df.write_csv(corr_path)
    print(corr_df.head(20))
    print(f"   Saved: {corr_path}")

    pdf = df.to_pandas()

    feature_sets = {
        "actual_fuel_kg": NUMERIC_FEATURES + ["physics_fuel_kg"] + CATEGORICAL_FEATURES,
        "residual_kg": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    flight_ids = pdf["flight_id"].to_numpy()
    train_idx, test_idx, train_fids, test_fids = flight_level_split(flight_ids)

    y_actual = pdf["actual_fuel_kg"].to_numpy()
    y_residual = pdf["residual_kg"].to_numpy()
    physics = pdf["physics_fuel_kg"].to_numpy()

    y_actual_test = y_actual[test_idx]
    physics_test = physics[test_idx]

    print(f"\n3. FLIGHT-LEVEL TRAIN/TEST SPLIT")
    print(f"   Total flights: {len(np.unique(flight_ids)):,}")
    print(f"   Train flights: {len(train_fids):,}  ({len(train_idx):,} intervals)")
    print(f"   Test flights:  {len(test_fids):,}  ({len(test_idx):,} intervals)")
    print(f"   Flight overlap: 0 (strict split)")

    openap_metrics = evaluate_predictions(y_actual_test, physics_test)
    print(f"\n4. OPENAP BASELINE (held-out flights, n={len(test_idx):,} intervals)")
    print(
        f"   MAE={openap_metrics['mae']:.2f} kg  "
        f"RMSE={openap_metrics['rmse']:.2f} kg  "
        f"R²={openap_metrics['r2']:.4f}"
    )

    models = get_models()
    results: list[dict] = []
    fitted_residual_models: dict[str, Pipeline] = {}
    fitted_actual_models: dict[str, Pipeline] = {}

    print(f"\n5. TRAINING BASELINE MODELS")
    for target_name, feature_cols in feature_sets.items():
        X = pdf[feature_cols]
        y = pdf[target_name].to_numpy()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, model in models.items():
            print(f"   Training {model_name} -> {target_name} ...", flush=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if target_name == "actual_fuel_kg":
                metrics = evaluate_predictions(y_actual_test, y_pred)
                fitted_actual_models[model_name] = model
            else:
                metrics_residual = evaluate_predictions(y_test, y_pred)
                y_pred_actual = physics_test + y_pred
                metrics = evaluate_predictions(y_actual_test, y_pred_actual)
                fitted_residual_models[model_name] = model

            results.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "mae_kg": metrics["mae"],
                    "rmse_kg": metrics["rmse"],
                    "r2": metrics["r2"],
                    "evaluation_scale": "actual_fuel_kg",
                }
            )

            if target_name == "residual_kg":
                results[-1]["residual_mae_kg"] = metrics_residual["mae"]
                results[-1]["residual_rmse_kg"] = metrics_residual["rmse"]
                results[-1]["residual_r2"] = metrics_residual["r2"]

    results.append(
        {
            "target": "openap_only",
            "model": "OpenAP",
            "mae_kg": openap_metrics["mae"],
            "rmse_kg": openap_metrics["rmse"],
            "r2": openap_metrics["r2"],
            "evaluation_scale": "actual_fuel_kg",
        }
    )

    results_df = pl.DataFrame(results).sort(["evaluation_scale", "mae_kg"])
    comp_path = OUT / "table_model_comparison_flight_split.csv"
    results_df = results_df.with_columns(pl.lit("flight_level").alias("split_strategy"))
    results_df.write_csv(comp_path)

    print(f"\n6. MODEL COMPARISON (held-out flights, metrics on actual_fuel_kg scale)")
    print(results_df)

    residual_rows = (
        results_df.filter(pl.col("target") == "residual_kg")
        .sort("mae_kg")
    )
    best_residual_model = residual_rows["model"][0]
    best_residual_metrics = residual_rows.row(0, named=True)

    print(f"\n7. RESIDUAL-LEARNING EVALUATION (held-out flights)")
    print(f"   OpenAP alone:  MAE={openap_metrics['mae']:.2f}  RMSE={openap_metrics['rmse']:.2f}  R²={openap_metrics['r2']:.4f}")
    print(
        f"   Best residual ({best_residual_model}): "
        f"MAE={best_residual_metrics['mae_kg']:.2f}  "
        f"RMSE={best_residual_metrics['rmse_kg']:.2f}  "
        f"R²={best_residual_metrics['r2']:.4f}"
    )

    mae_improvement = openap_metrics["mae"] - best_residual_metrics["mae_kg"]
    rmse_improvement = openap_metrics["rmse"] - best_residual_metrics["rmse_kg"]
    r2_improvement = best_residual_metrics["r2"] - openap_metrics["r2"]
    pct_mae = 100 * mae_improvement / openap_metrics["mae"]

    print(f"   MAE improvement:  {mae_improvement:.2f} kg ({pct_mae:.1f}%)")
    print(f"   RMSE improvement: {rmse_improvement:.2f} kg")
    print(f"   R² improvement: {r2_improvement:.4f}")

    materially_better = (
        mae_improvement > 0
        and rmse_improvement > 0
        and r2_improvement > 0
        and pct_mae >= 5.0
    )
    verdict = (
        "YES — residual learning materially outperforms OpenAP alone."
        if materially_better
        else "PARTIAL — residual learning improves some metrics but gains are modest."
        if mae_improvement > 0
        else "NO — residual learning does not beat OpenAP on this split."
    )
    print(f"\n   SUCCESS CRITERION: {verdict}")

    print(f"\n8. LIGHTGBM FEATURE IMPORTANCE")
    lgbm_residual = fitted_residual_models["LightGBM"]
    prep: ColumnTransformer = lgbm_residual.named_steps["prep"]
    lgbm_model: lgb.LGBMRegressor = lgbm_residual.named_steps["model"]
    feature_names = get_feature_names(prep)

    gain_importance = lgbm_model.feature_importances_
    gain_rows = sorted(
        zip(feature_names, gain_importance), key=lambda x: x[1], reverse=True
    )

    print("   Top 15 features (gain):")
    for fname, imp in gain_rows[:15]:
        print(f"     {fname:40s} {imp:.1f}")

    key_gain = aggregate_importance(feature_names, gain_importance)
    print("\n   Key feature groups (gain, aggregated):")
    for k, v in sorted(key_gain.items(), key=lambda x: x[1], reverse=True):
        print(f"     {k:20s} {v:.1f}")

    X_test_residual = pdf[feature_sets["residual_kg"]].iloc[test_idx]
    subsample_n = min(5000, len(X_test_residual))
    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = rng.choice(len(X_test_residual), size=subsample_n, replace=False)
    X_perm = X_test_residual.iloc[sub_idx]
    y_perm = y_residual[test_idx][sub_idx]

    print(f"   Computing permutation importance (n={subsample_n}) ...", flush=True)
    perm_result = permutation_importance(
        lgbm_residual,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_importance = perm_result.importances_mean

    perm_rows = sorted(
        zip(feature_names, perm_importance), key=lambda x: x[1], reverse=True
    )
    print("\n   Top 15 features (permutation):")
    for fname, imp in perm_rows[:15]:
        print(f"     {fname:40s} {imp:.4f}")

    key_perm = aggregate_importance(feature_names, perm_importance)
    print("\n   Key feature groups (permutation, aggregated):")
    for k, v in sorted(key_perm.items(), key=lambda x: x[1], reverse=True):
        print(f"     {k:20s} {v:.4f}")

    importance_records = []
    for fname, gain, perm in zip(feature_names, gain_importance, perm_importance):
        importance_records.append(
            {
                "feature": fname,
                "gain_importance": float(gain),
                "permutation_importance": float(perm),
            }
        )
    importance_df = (
        pl.DataFrame(importance_records)
        .sort("gain_importance", descending=True)
    )
    imp_path = OUT / "table_feature_importance_lgbm.csv"
    importance_df.write_csv(imp_path)
    print(f"   Saved: {imp_path}")

    print(f"\n9. GENERATING PLOTS")
    best_lgbm = fitted_residual_models["LightGBM"]
    y_lgbm_residual = best_lgbm.predict(X_test_residual)
    y_lgbm_actual = physics_test + y_lgbm_residual

    best_direct = fitted_actual_models["LightGBM"]
    X_test_actual = pdf[feature_sets["actual_fuel_kg"]].iloc[test_idx]
    y_direct_actual = best_direct.predict(X_test_actual)

    plot_sample = min(8000, len(y_actual_test))
    plot_idx = rng.choice(len(y_actual_test), size=plot_sample, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panel_metrics = [
        ("OpenAP alone", physics_test, evaluate_predictions(y_actual_test, physics_test)),
        (
            "OpenAP + LightGBM residual",
            y_lgbm_actual,
            evaluate_predictions(y_actual_test, y_lgbm_actual),
        ),
        (
            "LightGBM direct (actual)",
            y_direct_actual,
            evaluate_predictions(y_actual_test, y_direct_actual),
        ),
    ]
    for ax, (title, preds, m) in zip(axes, panel_metrics):
        ax.scatter(
            y_actual_test[plot_idx], preds[plot_idx], alpha=0.15, s=8, edgecolors="none"
        )
        lim = [
            min(y_actual_test[plot_idx].min(), preds[plot_idx].min()),
            max(y_actual_test[plot_idx].max(), preds[plot_idx].max()),
        ]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(f"{title}\nMAE={m['mae']:.0f} kg, R²={m['r2']:.3f}")
        ax.set_xlabel("Actual fuel (kg)")
        ax.set_ylabel("Predicted fuel (kg)")
    fig.suptitle("Actual vs Predicted (held-out flights)", y=1.02)
    fig.tight_layout()
    avp_path = OUT / "fig_actual_vs_predicted.png"
    fig.savefig(avp_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    error_panels = [
        ("OpenAP error", y_actual_test - physics_test),
        ("Residual model error", y_actual_test - y_lgbm_actual),
        ("True residual distribution", y_residual[test_idx]),
    ]
    for ax, (title, values) in zip(axes, error_panels):
        ax.hist(values, bins=60, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", lw=1)
        ax.set_title(f"{title}\nmean={values.mean():.1f}, std={values.std():.1f}")
        ax.set_xlabel("kg")
    fig.suptitle("Residual / Error Distributions (held-out flights)", y=1.02)
    fig.tight_layout()
    res_path = OUT / "fig_residual_distributions.png"
    fig.savefig(res_path, bbox_inches="tight")
    plt.close(fig)

    top_n = 20
    top_gain = importance_df.head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].barh(
        top_gain["feature"].to_list()[::-1],
        top_gain["gain_importance"].to_list()[::-1],
        color="seagreen",
    )
    axes[0].set_title("LightGBM Gain Importance (top 20)")
    axes[0].set_xlabel("Gain")

    top_perm = (
        importance_df.sort("permutation_importance", descending=True)
        .head(top_n)
    )
    axes[1].barh(
        top_perm["feature"].to_list()[::-1],
        top_perm["permutation_importance"].to_list()[::-1],
        color="darkorange",
    )
    axes[1].set_title("LightGBM Permutation Importance (top 20)")
    axes[1].set_xlabel("Mean MAE increase")
    fig.suptitle("Feature Importance — LightGBM residual model", y=1.01)
    fig.tight_layout()
    fi_path = OUT / "fig_feature_importance.png"
    fig.savefig(fi_path, bbox_inches="tight")
    plt.close(fig)

    print(f"   Saved: {avp_path}")
    print(f"   Saved: {res_path}")
    print(f"   Saved: {fi_path}")
    print(f"   Saved: {comp_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()