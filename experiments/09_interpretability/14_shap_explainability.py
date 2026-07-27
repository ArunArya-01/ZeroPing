from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    evaluate,
    flight_level_split,
    load_and_clean,
    project_root,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES  # noqa: E402
from aerotwin.engine.weather_features import WEATHER_FEATURES  # noqa: E402

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

PARQUET = project_root() / "featured_dataset.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
SHAP_SAMPLE_N = 5_000
TOP_N = 20
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]


def get_feature_set(df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    weather = [c for c in WEATHER_FEATURES if c in df.columns]
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    return list(dict.fromkeys(c for c in cols if c in df.columns))


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    feat_cols: list[str],
    cat_names: list[str],
) -> CatBoostRegressor:
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx, feature_names=feat_cols)
    eval_pool = Pool(X_eval, y_eval, cat_features=cat_idx, feature_names=feat_cols)
    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return model


def feature_group(name: str) -> str:
    if name in CAT_FEATURES:
        return "categorical"
    if name == "physics_fuel_kg":
        return "physics"
    if name in ENERGY_FEATURES:
        return "energy"
    if name in WEATHER_FEATURES:
        return "weather"
    if name in BASE_NUMERIC:
        return "trajectory"
    return "other"


def plot_top_shap(shap_table: pd.DataFrame, path: Path, top_n: int = TOP_N) -> None:
    top = shap_table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#3b82f6", alpha=0.88)
    ax.set_xlabel("Mean absolute SHAP value (kg)")
    ax.set_ylabel("")
    ax.set_title("CatBoost SHAP Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_shap_summary(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feat_cols: list[str],
    top_features: list[str],
    path: Path,
) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(10, 7))

    for row, feature in enumerate(reversed(top_features)):
        col_idx = feat_cols.index(feature)
        vals = shap_values[:, col_idx]
        y = row + rng.normal(0, 0.08, size=len(vals))
        if pd.api.types.is_numeric_dtype(X_sample[feature]):
            colors = pd.to_numeric(X_sample[feature], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(colors).any():
                lo, hi = np.nanpercentile(colors, [1, 99])
                colors = np.clip(colors, lo, hi)
                colors = np.nan_to_num(colors, nan=float(np.nanmedian(colors)))
            else:
                colors = np.zeros(len(vals), dtype=float)
            ax.scatter(
                vals,
                y,
                c=colors,
                cmap="viridis",
                s=8,
                alpha=0.45,
                edgecolors="none",
            )
        else:
            codes = X_sample[feature].astype("category").cat.codes.to_numpy()
            ax.scatter(
                vals,
                y,
                c=codes,
                cmap="tab20",
                s=8,
                alpha=0.45,
                edgecolors="none",
            )

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel("SHAP value contribution to predicted fuel (kg)")
    ax.set_title("CatBoost SHAP Summary")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("SHAP EXPLAINABILITY: CatBoost Energy+Weather+Physics")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    flight_ids = df["flight_id"].to_numpy()
    train_idx, test_idx, train_fids, test_fids = flight_level_split(flight_ids)

    feat_cols = get_feature_set(df)
    cat_names = [c for c in CAT_FEATURES if c in feat_cols]
    y = df["actual_fuel_kg"].to_numpy()

    print(f"Loaded: {len(df):,} intervals | {df['flight_id'].n_unique():,} flights")
    print(f"Split: {len(train_fids):,} train flights / {len(test_fids):,} held-out flights")
    print(f"Features: {len(feat_cols)} total ({len(cat_names)} categorical)")

    X_all = pdf[feat_cols].copy()
    X_train = X_all.iloc[train_idx]
    X_test = X_all.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("\nTraining CatBoost explainer model ...", flush=True)
    t0 = time.perf_counter()
    model = train_catboost(X_train, y_train, X_test, y_test, feat_cols, cat_names)
    test_pool = Pool(
        X_test,
        cat_features=[feat_cols.index(c) for c in cat_names],
        feature_names=feat_cols,
    )
    preds = model.predict(test_pool)
    metrics = evaluate(y_test, preds)
    print(
        f"Model ready in {time.perf_counter() - t0:.1f}s | "
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.4f}"
    )

    sample_n = min(SHAP_SAMPLE_N, len(X_test))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_pos = rng.choice(len(X_test), size=sample_n, replace=False)
    X_sample = X_test.iloc[sample_pos].reset_index(drop=True)
    y_sample = y_test[sample_pos]

    cat_idx = [feat_cols.index(c) for c in cat_names]
    sample_pool = Pool(X_sample, y_sample, cat_features=cat_idx, feature_names=feat_cols)

    print(f"\nComputing CatBoost native SHAP values on {sample_n:,} held-out intervals ...", flush=True)
    shap_raw = model.get_feature_importance(sample_pool, type="ShapValues")
    shap_values = np.asarray(shap_raw)[:, :-1]
    expected_value = float(np.asarray(shap_raw)[0, -1])

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    shap_table = (
        pd.DataFrame(
            {
                "feature": feat_cols,
                "feature_group": [feature_group(c) for c in feat_cols],
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_table["rank"] = np.arange(1, len(shap_table) + 1)

    group_table = (
        shap_table.groupby("feature_group", as_index=False)
        .agg(
            n_features=("feature", "count"),
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .sort_values("total_mean_abs_shap", ascending=False)
    )

    table_path = OUT / "table_shap_catboost.csv"
    group_path = OUT / "table_shap_catboost_groups.csv"
    meta_path = OUT / "table_shap_catboost_model.csv"
    shap_table.to_csv(table_path, index=False)
    group_table.to_csv(group_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "target": "actual_fuel_kg",
                "feature_set": "Energy+Weather+Physics",
                "n_train_flights": len(train_fids),
                "n_test_flights": len(test_fids),
                "n_shap_intervals": sample_n,
                "expected_value": expected_value,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "best_iteration": model.get_best_iteration(),
            }
        ]
    ).to_csv(meta_path, index=False)

    top_plot = OUT / "fig_shap_catboost_top_features.png"
    summary_plot = OUT / "fig_shap_catboost_summary.png"
    ordered_features = shap_table["feature"].to_list()
    plot_top_shap(shap_table, top_plot)
    plot_shap_summary(X_sample, shap_values, feat_cols, ordered_features[:15], summary_plot)

    print(f"\nSaved {table_path}")
    print(f"Saved {group_path}")
    print(f"Saved {meta_path}")
    print(f"Saved {top_plot}")
    print(f"Saved {summary_plot}")

    print("\nTop SHAP drivers:")
    for row in shap_table.head(10).itertuples(index=False):
        print(
            f"  {row.rank:2d}. {row.feature:30s} "
            f"mean|SHAP|={row.mean_abs_shap:8.2f} kg "
            f"signed={row.mean_signed_shap:+8.2f} kg"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
