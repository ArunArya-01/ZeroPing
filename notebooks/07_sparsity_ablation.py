"""
Physics under sparse observations — sparsity-bucket ablation.

Uses the same flight-level train/test split as 05_baseline_modeling.py.
For each n_traj_pts bucket, trains Full Hybrid vs No Physics (LightGBM)
and evaluates OpenAP on held-out intervals in that bucket.

Run:
    python notebooks/07_sparsity_ablation.py

Output:
    figures/table_sparsity_ablation.csv
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

SPARSITY_BUCKETS = [
    ("Dense", pl.col("n_traj_pts") > 1000),
    ("Medium", (pl.col("n_traj_pts") >= 100) & (pl.col("n_traj_pts") <= 1000)),
    ("Sparse", (pl.col("n_traj_pts") >= 10) & (pl.col("n_traj_pts") < 100)),
    ("Very Sparse", pl.col("n_traj_pts") < 10),
]

BUCKET_ORDER = ["Dense", "Medium", "Sparse", "Very Sparse"]


def load_and_clean() -> pl.DataFrame:
    df = pl.read_parquet(PARQUET)
    if "flight_id" not in df.columns:
        raise ValueError("Missing flight_id. Run: python physics/build_featured_dataset.py --patch-flight-id")
    return (
        df.drop_nulls(subset=["physics_fuel_kg", "residual_kg", "flight_id"])
        .filter(
            pl.col("physics_fuel_kg").is_finite()
            & pl.col("residual_kg").is_finite()
            & pl.col("actual_fuel_kg").is_finite()
        )
    )


def flight_level_split(flight_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_flights = np.unique(flight_ids)
    train_fids, test_fids = train_test_split(
        unique_flights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx = np.flatnonzero(np.isin(flight_ids, train_fids))
    test_idx = np.flatnonzero(np.isin(flight_ids, test_fids))
    return train_idx, test_idx


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


def make_lgbm_pipeline(include_physics: bool) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(include_physics)),
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
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def interpret(results: pl.DataFrame) -> str:
    lines = ["INTERPRETATION", "=" * 60]
    deltas: list[tuple[str, float, float]] = []

    for bucket in BUCKET_ORDER:
        sub = results.filter(pl.col("sparsity_bucket") == bucket)
        if sub.is_empty():
            continue
        hybrid = sub.filter(pl.col("approach") == "Full Hybrid").row(0, named=True)
        no_phys = sub.filter(pl.col("approach") == "No Physics").row(0, named=True)
        openap = sub.filter(pl.col("approach") == "OpenAP").row(0, named=True)
        mae_gain = no_phys["mae"] - hybrid["mae"]
        mae_gain_pct = 100 * mae_gain / no_phys["mae"] if no_phys["mae"] else 0.0
        deltas.append((bucket, mae_gain, mae_gain_pct))
        lines.append(
            f"{bucket:12s}  physics saves {mae_gain:5.1f} kg ({mae_gain_pct:4.1f}%)  "
            f"| Hybrid MAE={hybrid['mae']:.1f}  NoPhys={no_phys['mae']:.1f}  OpenAP={openap['mae']:.1f}  "
            f"(n_test={hybrid['n_test']:,})"
        )

    dense_gain = deltas[0][1]
    very_sparse_gain = deltas[-1][1]
    if very_sparse_gain > dense_gain * 1.5 and very_sparse_gain >= 5:
        verdict = (
            "YES — physics becomes more valuable as observations become sparse. "
            f"MAE gain from physics rises from {dense_gain:.1f} kg (Dense) "
            f"to {very_sparse_gain:.1f} kg (Very Sparse)."
        )
    elif very_sparse_gain > dense_gain:
        verdict = (
            "PARTIALLY — physics helps more in sparser buckets, but the effect is modest. "
            f"Gain increases from {dense_gain:.1f} kg (Dense) to {very_sparse_gain:.1f} kg (Very Sparse)."
        )
    else:
        verdict = (
            "NO — physics does not consistently help more under sparsity in this setup. "
            "Trajectory/metadata features may already encode enough signal even with few points."
        )

    lines.extend(["", f"Conclusion: {verdict}"])
    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("PHYSICS UNDER SPARSE OBSERVATIONS (FLIGHT-LEVEL SPLIT)")
    print("=" * 70)

    df = load_and_clean()
    pdf = df.to_pandas()
    train_idx, test_idx = flight_level_split(pdf["flight_id"].to_numpy())

    train_df = df[train_idx.tolist()]
    test_df = df[test_idx.tolist()]

    print(f"\nTotal after cleaning: {len(df):,} intervals")
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    feature_sets = {
        "Full Hybrid": NUMERIC_FEATURES + ["physics_fuel_kg"] + CATEGORICAL_FEATURES,
        "No Physics": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    rows: list[dict] = []

    print("\nBucket counts (train / test):")
    for bucket_name, predicate in SPARSITY_BUCKETS:
        n_train = train_df.filter(predicate).height
        n_test = test_df.filter(predicate).height
        print(f"  {bucket_name:12s}  {n_train:6,} / {n_test:5,}")

    for bucket_name, predicate in SPARSITY_BUCKETS:
        train_bucket = train_df.filter(predicate)
        test_bucket = test_df.filter(predicate)
        n_train = train_bucket.height
        n_test = test_bucket.height

        if n_train < 50 or n_test < 10:
            print(f"\nSkipping {bucket_name}: insufficient data (train={n_train}, test={n_test})")
            continue

        train_pdf = train_bucket.to_pandas()
        test_pdf = test_bucket.to_pandas()
        y_train = train_pdf["actual_fuel_kg"].to_numpy()
        y_test = test_pdf["actual_fuel_kg"].to_numpy()
        physics_test = test_pdf["physics_fuel_kg"].to_numpy()

        print(f"\n--- {bucket_name} (train={n_train:,}, test={n_test:,}) ---")

        openap_metrics = evaluate(y_test, physics_test)
        rows.append(
            {
                "sparsity_bucket": bucket_name,
                "approach": "OpenAP",
                "physics": "Yes",
                "n_train": n_train,
                "n_test": n_test,
                **openap_metrics,
            }
        )
        print(f"  OpenAP:       MAE={openap_metrics['mae']:.1f}  R²={openap_metrics['r2']:.3f}")

        for approach, feature_cols in feature_sets.items():
            include_physics = approach == "Full Hybrid"
            model = make_lgbm_pipeline(include_physics)
            X_train = train_pdf[feature_cols]
            X_test = test_pdf[feature_cols]
            model.fit(X_train, y_train)
            metrics = evaluate(y_test, model.predict(X_test))
            rows.append(
                {
                    "sparsity_bucket": bucket_name,
                    "approach": approach,
                    "physics": "Yes" if include_physics else "No",
                    "n_train": n_train,
                    "n_test": n_test,
                    **metrics,
                }
            )
            print(
                f"  {approach:12s}  MAE={metrics['mae']:.1f}  R²={metrics['r2']:.3f}"
            )

    results = pl.DataFrame(rows)
    bucket_rank = {b: i for i, b in enumerate(BUCKET_ORDER)}
    results = results.with_columns(
        pl.col("sparsity_bucket")
        .map_elements(lambda x: bucket_rank.get(x, 99), return_dtype=pl.Int32)
        .alias("_rank")
    ).sort(["_rank", "approach"]).drop("_rank")

    # Physics benefit column: MAE(No Physics) - MAE(Full Hybrid) per bucket
    benefit_rows = []
    for bucket in BUCKET_ORDER:
        sub = results.filter(pl.col("sparsity_bucket") == bucket)
        if sub.is_empty():
            continue
        hybrid_mae = sub.filter(pl.col("approach") == "Full Hybrid")["mae"][0]
        no_phys_mae = sub.filter(pl.col("approach") == "No Physics")["mae"][0]
        benefit_rows.append({"sparsity_bucket": bucket, "physics_mae_gain_kg": no_phys_mae - hybrid_mae})

    benefit = pl.DataFrame(benefit_rows)
    results = results.join(benefit, on="sparsity_bucket", how="left")

    table_path = OUT / "table_sparsity_ablation.csv"
    results.write_csv(table_path)

    print("\n" + "-" * 70)
    print(results.select(["sparsity_bucket", "approach", "n_test", "mae", "rmse", "r2", "physics_mae_gain_kg"]))
    print("-" * 70)
    print("\n" + interpret(results))
    print(f"\nSaved: {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()