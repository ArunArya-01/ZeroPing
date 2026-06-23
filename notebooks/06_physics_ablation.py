
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def load_and_clean() -> tuple[pl.DataFrame, int, int]:
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} not found.")

    df = pl.read_parquet(PARQUET)
    if "flight_id" not in df.columns:
        raise ValueError("Missing flight_id. Run: python physics/build_featured_dataset.py --patch-flight-id")

    n_before = len(df)
    df = (
        df.drop_nulls(subset=["physics_fuel_kg", "residual_kg", "flight_id"])
        .filter(
            pl.col("physics_fuel_kg").is_finite()
            & pl.col("residual_kg").is_finite()
            & pl.col("actual_fuel_kg").is_finite()
        )
    )
    return df, n_before, len(df)


def flight_level_split(
    flight_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_flights = np.unique(flight_ids)
    train_fids, test_fids = train_test_split(
        unique_flights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if set(train_fids.tolist()) & set(test_fids.tolist()):
        raise RuntimeError("Flight leakage detected.")
    train_idx = np.flatnonzero(np.isin(flight_ids, train_fids))
    test_idx = np.flatnonzero(np.isin(flight_ids, test_fids))
    return train_idx, test_idx, train_fids, test_fids


def make_preprocessor(include_physics: bool) -> ColumnTransformer:
    numeric_cols = NUMERIC_FEATURES + (["physics_fuel_kg"] if include_physics else [])
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_model_pipelines(include_physics: bool) -> dict[str, Pipeline]:
    prep = make_preprocessor(include_physics)
    return {
        "LightGBM": Pipeline(
            [
                ("prep", prep),
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
                ("prep", prep),
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
        "Random Forest": Pipeline(
            [
                ("prep", prep),
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
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def interpret_results(results: pl.DataFrame) -> str:
    hybrid = results.filter(pl.col("feature_set") == "Full Hybrid")
    no_phys = results.filter(pl.col("feature_set") == "No Physics")
    physics_only = results.filter(pl.col("feature_set") == "Physics Only")

    best_hybrid = hybrid.sort("mae").row(0, named=True)
    best_no_phys = no_phys.sort("mae").row(0, named=True)
    openap = physics_only.row(0, named=True)

    mae_delta = best_no_phys["mae"] - best_hybrid["mae"]
    mae_pct = 100 * mae_delta / best_hybrid["mae"]
    r2_delta = best_hybrid["r2"] - best_no_phys["r2"]

    if mae_pct >= 15:
        verdict = (
            "Physics prior is valuable: models rely substantially on OpenAP; "
            "trajectory/metadata alone cannot fully replace physics_fuel_kg."
        )
    elif mae_pct >= 5:
        verdict = (
            "Physics prior is moderately valuable: data-driven features carry "
            "much of the signal, but physics_fuel_kg still provides a meaningful boost."
        )
    else:
        verdict = (
            "Model is mostly data-driven: removing physics causes little degradation, "
            "suggesting trajectory and metadata capture fuel patterns independently."
        )

    return "\n".join(
        [
            "INTERPRETATION",
            "=" * 60,
            f"Best Full Hybrid:  {best_hybrid['model']}  MAE={best_hybrid['mae']:.1f} kg  R²={best_hybrid['r2']:.3f}",
            f"Best No Physics:   {best_no_phys['model']}  MAE={best_no_phys['mae']:.1f} kg  R²={best_no_phys['r2']:.3f}",
            f"Physics Only:      MAE={openap['mae']:.1f} kg  R²={openap['r2']:.3f}",
            "",
            f"Removing physics_fuel_kg increases MAE by {mae_delta:.1f} kg ({mae_pct:.1f}%) "
            f"and lowers R² by {r2_delta:.3f}.",
            "",
            f"Conclusion: {verdict}",
        ]
    )


def plot_ablation(results: pl.DataFrame, path: Path) -> None:
    ml_results = results.filter(pl.col("feature_set") != "Physics Only").to_pandas()
    openap_mae = results.filter(pl.col("feature_set") == "Physics Only")["mae"][0]
    openap_rmse = results.filter(pl.col("feature_set") == "Physics Only")["rmse"][0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = {"Full Hybrid": "#27ae60", "No Physics": "#c0392b"}

    for ax, metric, ylabel in zip(axes, ["mae", "rmse", "r2"], ["MAE (kg)", "RMSE (kg)", "R²"]):
        sns.barplot(
            data=ml_results,
            x="model",
            y=metric,
            hue="feature_set",
            hue_order=["Full Hybrid", "No Physics"],
            palette=palette,
            ax=ax,
        )
        if metric in ("mae", "rmse"):
            ref = openap_mae if metric == "mae" else openap_rmse
            ax.axhline(ref, color="#2980b9", linestyle="--", linewidth=1.5)
        ax.set_title(ylabel)
        ax.set_xlabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, title="Feature set", fontsize=8)
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[1].text(
        0.98, 0.95, "dashed = Physics Only", transform=axes[1].transAxes,
        ha="right", va="top", fontsize=8, color="#2980b9",
    )

    fig.suptitle("Physics Ablation — Held-Out Flights", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("PHYSICS ABLATION STUDY (FLIGHT-LEVEL SPLIT)")
    print("=" * 70)

    df, n_before, n_after = load_and_clean()
    print(f"\nRows: {n_before:,} -> {n_after:,} after cleaning")

    pdf = df.to_pandas()
    train_idx, test_idx, train_fids, test_fids = flight_level_split(pdf["flight_id"].to_numpy())
    print(f"Train: {len(train_fids):,} flights  |  Test: {len(test_fids):,} flights  |  Overlap: 0")

    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]

    feature_sets = {
        "Full Hybrid": NUMERIC_FEATURES + ["physics_fuel_kg"] + CATEGORICAL_FEATURES,
        "No Physics": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    rows: list[dict] = []

    physics_metrics = evaluate(y_test, physics_test)
    rows.append(
        {
            "model": "Physics Only",
            "physics": "Yes",
            "feature_set": "Physics Only",
            **physics_metrics,
        }
    )
    print(
        f"\nPhysics Only: MAE={physics_metrics['mae']:.1f}  "
        f"RMSE={physics_metrics['rmse']:.1f}  R²={physics_metrics['r2']:.3f}"
    )

    for feature_set_name, feature_cols in feature_sets.items():
        include_physics = feature_set_name == "Full Hybrid"
        X_train = pdf[feature_cols].iloc[train_idx]
        X_test = pdf[feature_cols].iloc[test_idx]

        for model_name, model in get_model_pipelines(include_physics).items():
            print(f"Training {model_name} | {feature_set_name} ...", flush=True)
            model.fit(X_train, y_train)
            metrics = evaluate(y_test, model.predict(X_test))
            rows.append(
                {
                    "model": model_name,
                    "physics": "Yes" if include_physics else "No",
                    "feature_set": feature_set_name,
                    **metrics,
                }
            )
            print(
                f"  MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  R²={metrics['r2']:.3f}"
            )

    results = pl.DataFrame(rows).sort(["feature_set", "mae"])
    table_path = OUT / "table_physics_ablation.csv"
    results.write_csv(table_path)

    print("\n" + "-" * 70)
    print(results.select(["model", "physics", "feature_set", "mae", "rmse", "r2"]))
    print("-" * 70)
    print("\n" + interpret_results(results))

    fig_path = OUT / "fig_physics_ablation.png"
    plot_ablation(results, fig_path)
    print(f"\nSaved: {table_path}")
    print(f"Saved: {fig_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()