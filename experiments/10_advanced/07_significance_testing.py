
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

def _project_root() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parents[2])
    except NameError:
        pass
    candidates.extend([Path.cwd(), Path.cwd().parent])
    for root in candidates:
        if (root / "featured_dataset.parquet").exists():
            return root
    return candidates[0]


ROOT = _project_root()
PARQUET = ROOT / "featured_dataset.parquet"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 10_000  # flight-level bootstrap iterations

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

FEATURE_HYBRID = NUMERIC_FEATURES + ["physics_fuel_kg"] + CATEGORICAL_FEATURES
FEATURE_NO_PHYSICS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean() -> pl.DataFrame:
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} not found.")
    df = pl.read_parquet(PARQUET)
    if "flight_id" not in df.columns:
        raise ValueError("Missing flight_id.")
    return (
        df.drop_nulls(subset=["physics_fuel_kg", "residual_kg", "flight_id"])
        .filter(
            pl.col("physics_fuel_kg").is_finite()
            & pl.col("residual_kg").is_finite()
            & pl.col("actual_fuel_kg").is_finite()
        )
    )


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


def make_model_pipeline(model_key: str, include_physics: bool) -> Pipeline:
    prep = make_preprocessor(include_physics)
    if model_key == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_key == "xgb":
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
    elif model_key == "lgbm":
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")
    return Pipeline([("prep", prep), ("model", model)])


def cohens_d_paired(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return float(diff.mean() / std)


def effect_size_category(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "Negligible"
    if ad < 0.5:
        return "Small"
    if ad < 0.8:
        return "Medium"
    return "Large"


def flight_error_sums(
    err_a: np.ndarray, err_b: np.ndarray, flight_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-interval errors to per-flight sums for vectorized bootstrap."""
    _, flight_codes = np.unique(flight_ids, return_inverse=True)
    order = np.argsort(flight_codes, kind="stable")
    sorted_a = err_a[order]
    sorted_b = err_b[order]
    boundaries = np.flatnonzero(np.diff(flight_codes[order])) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(flight_ids)]))
    sums_a = np.add.reduceat(sorted_a, starts)
    sums_b = np.add.reduceat(sorted_b, starts)
    counts = ends - starts
    return sums_a, sums_b, counts


def bootstrap_mae_diff(
    err_a: np.ndarray,
    err_b: np.ndarray,
    flight_ids: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> np.ndarray:
    """Resample test flights with replacement; return ΔMAE = MAE_a - MAE_b."""
    sums_a, sums_b, counts = flight_error_sums(err_a, err_b, flight_ids)
    n_flights = len(counts)
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_flights, size=(n_iter, n_flights))
    boot_sum_a = sums_a[sampled_idx].sum(axis=1)
    boot_sum_b = sums_b[sampled_idx].sum(axis=1)
    boot_count = counts[sampled_idx].sum(axis=1)
    return boot_sum_a / boot_count - boot_sum_b / boot_count


def run_comparison(
    name: str,
    err_hybrid: np.ndarray,
    err_other: np.ndarray,
    flight_ids: np.ndarray,
    other_label: str,
) -> dict:
    """Compare hybrid vs other on paired absolute errors."""
    delta_err = err_hybrid - err_other
    mae_hybrid = float(err_hybrid.mean())
    mae_other = float(err_other.mean())
    delta_mae = mae_hybrid - mae_other

    boot = bootstrap_mae_diff(err_hybrid, err_other, flight_ids)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    boot_p = float((boot > 0).mean())

    try:
        w_stat, w_p = wilcoxon(err_hybrid, err_other, alternative="less", zero_method="wilcox")
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    d = cohens_d_paired(delta_err)
    category = effect_size_category(d)

    if ci_high < 0 and boot_p < 0.05:
        interpretation = "Physics significantly helps"
    elif ci_low > 0 and boot_p > 0.95:
        interpretation = "Other approach significantly better"
    else:
        interpretation = "No significant evidence"

    return {
        "comparison": name,
        "vs": other_label,
        "n_intervals": len(err_hybrid),
        "mae_hybrid": mae_hybrid,
        "mae_other": mae_other,
        "delta_mae": delta_mae,
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_median": float(np.median(boot)),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "bootstrap_p": boot_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "cohens_d": d,
        "effect_size": category,
        "interpretation": interpretation,
        "bootstrap_dist": boot,
        "delta_err": delta_err,
    }


def plot_bootstrap(dist: np.ndarray, title: str, path: Path) -> None:
    ci_low, ci_high = np.percentile(dist, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dist, bins=60, color="steelblue", alpha=0.85, edgecolor="white", density=True)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="ΔMAE = 0")
    ax.axvline(dist.mean(), color="darkorange", linestyle="--", linewidth=1.5, label=f"Mean={dist.mean():.2f}")
    ax.axvspan(ci_low, ci_high, alpha=0.2, color="green", label=f"95% CI [{ci_low:.2f}, {ci_high:.2f}]")
    ax.set_xlabel("ΔMAE = MAE(Hybrid) − MAE(Other)  [kg]")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train_and_predict(
    model_key: str,
    X_train_hybrid: object,
    X_test_hybrid: object,
    X_train_no: object,
    X_test_no: object,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hybrid = make_model_pipeline(model_key, include_physics=True)
    no_phys = make_model_pipeline(model_key, include_physics=False)
    print(f"    Training hybrid ({model_key})...", flush=True)
    hybrid.fit(X_train_hybrid, y_train)
    print(f"    Training no-physics ({model_key})...", flush=True)
    no_phys.fit(X_train_no, y_train)
    pred_hybrid = hybrid.predict(X_test_hybrid)
    pred_no = no_phys.predict(X_test_no)
    return pred_hybrid, pred_no


def main() -> None:
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTING — PHYSICS ABLATION")
    print("=" * 70)

    df = load_and_clean()
    pdf = df.to_pandas()
    flight_ids_all = pdf["flight_id"].to_numpy()
    train_idx, test_idx, train_fids, test_fids = flight_level_split(flight_ids_all)

    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]
    test_flight_ids = flight_ids_all[test_idx]

    X_train_hybrid = pdf[FEATURE_HYBRID].iloc[train_idx]
    X_test_hybrid = pdf[FEATURE_HYBRID].iloc[test_idx]
    X_train_no = pdf[FEATURE_NO_PHYSICS].iloc[train_idx]
    X_test_no = pdf[FEATURE_NO_PHYSICS].iloc[test_idx]

    print(f"\nData: {len(df):,} intervals after cleaning")
    print(f"Train flights: {len(train_fids):,}  |  Test flights: {len(test_fids):,}  |  Overlap: 0")

    model_configs = [
        ("rf", "Random Forest", "fig_bootstrap_rf.png", "table_significance_rf.csv"),
        ("xgb", "XGBoost", "fig_bootstrap_xgb.png", "table_significance_xgb.csv"),
        ("lgbm", "LightGBM", "fig_bootstrap_lgbm.png", "table_significance_lgbm.csv"),
    ]

    all_results: dict[str, list[dict]] = {}

    for model_key, model_name, fig_name, table_name in model_configs:
        print(f"\n--- {model_name} ---")
        pred_hybrid, pred_no = train_and_predict(
            model_key, X_train_hybrid, X_test_hybrid, X_train_no, X_test_no, y_train, y_test
        )
        err_hybrid = np.abs(y_test - pred_hybrid)
        err_no = np.abs(y_test - pred_no)
        err_physics = np.abs(y_test - physics_test)

        comparisons = [
            run_comparison(
                f"Hybrid {model_name} vs NoPhysics {model_name}",
                err_hybrid,
                err_no,
                test_flight_ids,
                "No Physics",
            ),
            run_comparison(
                f"Hybrid {model_name} vs Physics Only",
                err_hybrid,
                err_physics,
                test_flight_ids,
                "Physics Only",
            ),
        ]

        for c in comparisons:
            print(
                f"  {c['vs']:14s}  ΔMAE={c['delta_mae']:+.2f} kg  "
                f"95%CI=[{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}]  "
                f"boot_p={c['bootstrap_p']:.4f}  wilcoxon_p={c['wilcoxon_p']:.2e}  "
                f"d={c['cohens_d']:.3f} ({c['effect_size']})  → {c['interpretation']}"
            )

        plot_bootstrap(
            comparisons[0]["bootstrap_dist"],
            f"Bootstrap ΔMAE Distribution — Hybrid vs No Physics ({model_name})",
            OUT / fig_name,
        )

        table_rows = [{k: v for k, v in c.items() if k not in ("bootstrap_dist", "delta_err")} for c in comparisons]
        pl.DataFrame(table_rows).write_csv(OUT / table_name)
        all_results[model_key] = comparisons
        print(f"  Saved: {OUT / fig_name}")
        print(f"  Saved: {OUT / table_name}")

    # Sparsity bucket analysis — Hybrid RF vs NoPhysics RF
    print("\n--- Sparsity Bucket Analysis (Hybrid RF vs NoPhysics RF) ---")
    test_df = df[test_idx.tolist()]
    train_df = df[train_idx.tolist()]

    sparse_rows: list[dict] = []
    bucket_deltas: list[float] = []
    bucket_cis: list[tuple[float, float]] = []
    bucket_names_plot: list[str] = []

    for bucket_name, predicate in SPARSITY_BUCKETS:
        train_bucket = train_df.filter(predicate)
        test_bucket = test_df.filter(predicate)
        n_train = train_bucket.height
        n_test = test_bucket.height

        if n_train < 50 or n_test < 10:
            print(f"  Skipping {bucket_name}: insufficient data")
            continue

        train_pdf = train_bucket.to_pandas()
        test_pdf = test_bucket.to_pandas()
        y_tr = train_pdf["actual_fuel_kg"].to_numpy()
        y_te = test_pdf["actual_fuel_kg"].to_numpy()
        fids_te = test_pdf["flight_id"].to_numpy()

        pred_h, pred_n = train_and_predict(
            "rf",
            train_pdf[FEATURE_HYBRID],
            test_pdf[FEATURE_HYBRID],
            train_pdf[FEATURE_NO_PHYSICS],
            test_pdf[FEATURE_NO_PHYSICS],
            y_tr,
            y_te,
        )
        err_h = np.abs(y_te - pred_h)
        err_n = np.abs(y_te - pred_n)

        comp = run_comparison(
            f"Hybrid RF vs NoPhysics RF — {bucket_name}",
            err_h,
            err_n,
            fids_te,
            "No Physics",
        )
        sparse_rows.append(
            {
                "bucket": bucket_name,
                "n_intervals": comp["n_intervals"],
                "mae_hybrid": comp["mae_hybrid"],
                "mae_nophysics": comp["mae_other"],
                "delta_mae": comp["delta_mae"],
                "ci_lower": comp["ci_lower"],
                "ci_upper": comp["ci_upper"],
                "bootstrap_p": comp["bootstrap_p"],
                "wilcoxon_p": comp["wilcoxon_p"],
                "cohens_d": comp["cohens_d"],
                "effect_size": comp["effect_size"],
                "interpretation": comp["interpretation"],
            }
        )
        bucket_deltas.append(comp["delta_mae"])
        bucket_cis.append((comp["ci_lower"], comp["ci_upper"]))
        bucket_names_plot.append(bucket_name)
        print(
            f"  {bucket_name:12s}  n={comp['n_intervals']:,}  ΔMAE={comp['delta_mae']:+.2f}  "
            f"CI=[{comp['ci_lower']:+.2f}, {comp['ci_upper']:+.2f}]  "
            f"p={comp['wilcoxon_p']:.2e}  ({comp['effect_size']})"
        )

    sparse_df = pl.DataFrame(sparse_rows)
    sparse_path = OUT / "table_sparse_significance.csv"
    sparse_df.write_csv(sparse_path)
    print(f"  Saved: {sparse_path}")

    # Sparse bucket figure
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bucket_names_plot))
    deltas = np.array(bucket_deltas)
    ci_lows = np.array([c[0] for c in bucket_cis])
    ci_highs = np.array([c[1] for c in bucket_cis])
    yerr = np.vstack([deltas - ci_lows, ci_highs - deltas])

    colors = ["#27ae60" if d < 0 and hi < 0 else "#95a5a6" for d, (_, hi) in zip(deltas, bucket_cis)]
    ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor="white", yerr=yerr, capsize=6, error_kw={"elinewidth": 1.5})
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names_plot)
    ax.set_xlabel("Sparsity bucket")
    ax.set_ylabel("ΔMAE = MAE(Hybrid) − MAE(No Physics)  [kg]")
    ax.set_title("Physics Benefit by Trajectory Sparsity (Hybrid RF vs NoPhysics RF)\nError bars = 95% bootstrap CI")
    fig.tight_layout()
    sparse_fig_path = OUT / "fig_sparse_bucket_significance.png"
    fig.savefig(sparse_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {sparse_fig_path}")

    # Final interpretation
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    rf_comp = all_results["rf"][0]
    if rf_comp["ci_upper"] < 0 and rf_comp["bootstrap_p"] < 0.05:
        print(
            f"Overall (RF): Physics provides a statistically significant but practically "
            f"small improvement (ΔMAE={rf_comp['delta_mae']:.2f} kg, "
            f"95%CI=[{rf_comp['ci_lower']:.2f}, {rf_comp['ci_upper']:.2f}], "
            f"p={rf_comp['wilcoxon_p']:.4f})."
        )
    else:
        print(
            f"Overall (RF): No evidence that physics materially improves performance "
            f"when rich trajectory features exist (ΔMAE={rf_comp['delta_mae']:.2f} kg, "
            f"95%CI=[{rf_comp['ci_lower']:.2f}, {rf_comp['ci_upper']:.2f}], "
            f"p={rf_comp['wilcoxon_p']:.4f})."
        )

    for row in sparse_rows:
        if row["ci_upper"] < 0 and row["bootstrap_p"] < 0.05:
            print(f"  {row['bucket']}: Physics benefit is statistically significant (ΔMAE={row['delta_mae']:.2f} kg).")
        else:
            print(f"  {row['bucket']}: No significant physics benefit detected.")

    print("=" * 70)


if __name__ == "__main__":
    main()