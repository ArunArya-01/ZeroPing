"""
Shared evaluation + bootstrap significance framework for AeroTwin experiments.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BOOTSTRAP = 10_000

BASE_NUMERIC = [
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

CATEGORICAL = ["aircraft_type", "method", "origin_icao", "destination_icao"]


def project_root() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parents[1])
    except NameError:
        pass
    candidates.extend([Path.cwd(), Path.cwd().parent])
    for root in candidates:
        if (root / "featured_dataset.parquet").exists():
            return root
    return candidates[0]


def load_and_clean(parquet: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet)
    return (
        df.drop_nulls(subset=["physics_fuel_kg", "residual_kg", "flight_id"])
        .filter(
            pl.col("physics_fuel_kg").is_finite()
            & pl.col("residual_kg").is_finite()
            & pl.col("actual_fuel_kg").is_finite()
        )
    )


def flight_level_split(flight_ids: np.ndarray):
    unique = np.unique(flight_ids)
    train_fids, test_fids = train_test_split(
        unique, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx = np.flatnonzero(np.isin(flight_ids, train_fids))
    test_idx = np.flatnonzero(np.isin(flight_ids, test_fids))
    return train_idx, test_idx, train_fids, test_fids


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_pipeline(model_key: str, feature_cols: list[str]) -> Pipeline:
    numeric = [c for c in feature_cols if c not in CATEGORICAL]
    cat = [c for c in feature_cols if c in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )
    if model_key == "rf":
        model = RandomForestRegressor(
            n_estimators=100, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    elif model_key == "xgb":
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        )
    elif model_key == "lgbm":
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1,
        )
    elif model_key == "cat":
        if not HAS_CATBOOST:
            raise ImportError("catboost is required for 'cat' model. pip install catboost")
        # Use same prep (OHE cats); CatBoost can learn from it (native cats would be better but keeps API uniform)
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=8,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        )
    else:
        raise ValueError(model_key)
    return Pipeline([("prep", prep), ("model", model)])


def flight_error_sums(err_a, err_b, flight_ids):
    _, codes = np.unique(flight_ids, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    sa, sb = err_a[order], err_b[order]
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(flight_ids)]))
    return (
        np.add.reduceat(sa, starts),
        np.add.reduceat(sb, starts),
        ends - starts,
    )


def bootstrap_mae_diff(err_a, err_b, flight_ids, n_iter=N_BOOTSTRAP, seed=RANDOM_STATE):
    sums_a, sums_b, counts = flight_error_sums(err_a, err_b, flight_ids)
    n_flights = len(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_flights, size=(n_iter, n_flights))
    boot_a = sums_a[idx].sum(axis=1)
    boot_b = sums_b[idx].sum(axis=1)
    boot_n = counts[idx].sum(axis=1)
    return boot_a / boot_n - boot_b / boot_n


def cohens_d(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    s = diff.std(ddof=1)
    return 0.0 if s == 0 else float(diff.mean() / s)


def effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "Negligible"
    if a < 0.5:
        return "Small"
    if a < 0.8:
        return "Medium"
    return "Large"


def significance_test(
    err_new: np.ndarray,
    err_baseline: np.ndarray,
    flight_ids: np.ndarray,
    new_name: str,
    baseline_name: str,
) -> dict:
    delta = err_new - err_baseline
    boot = bootstrap_mae_diff(err_new, err_baseline, flight_ids)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    try:
        w_stat, w_p = wilcoxon(err_new, err_baseline, alternative="less")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    d = cohens_d(delta)
    mae_new, mae_base = float(err_new.mean()), float(err_baseline.mean())
    if ci_hi < 0 and float((boot > 0).mean()) < 0.05:
        interp = f"{new_name} significantly better than {baseline_name}"
    elif ci_lo > 0:
        interp = f"{baseline_name} significantly better than {new_name}"
    else:
        interp = "No significant evidence"
    return {
        "comparison": f"{new_name} vs {baseline_name}",
        "mae_new": mae_new,
        "mae_baseline": mae_base,
        "delta_mae": mae_new - mae_base,
        "bootstrap_mean": float(boot.mean()),
        "bootstrap_median": float(np.median(boot)),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "bootstrap_p": float((boot > 0).mean()),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohens_d": d,
        "effect_size": effect_label(d),
        "interpretation": interp,
        "bootstrap_dist": boot,
    }


def plot_bootstrap_hist(dist, title, path, color="steelblue"):
    ci_lo, ci_hi = np.percentile(dist, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dist, bins=60, color=color, alpha=0.85, edgecolor="white", density=True)
    ax.axvline(0, color="black", lw=1.5, label="ΔMAE = 0")
    ax.axvline(dist.mean(), color="darkorange", ls="--", lw=1.5, label=f"Mean={dist.mean():.2f}")
    ax.axvspan(ci_lo, ci_hi, alpha=0.2, color="green", label=f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    ax.set_xlabel("ΔMAE = MAE(New) − MAE(Baseline)  [kg]")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# V4 feature groups (populated by notebooks 09/10/11)
MASS_FEATURES: list[str] = [
    "mtow",
    "mlw",
    "oew",
    "takeoff_mass_est",
    "landing_mass_est",
    "mean_mass",
    "std_mass",
    "mass_slope",
    "mass_consumed_est",
]

VRATE_BIN_FEATURES: list[str] = [
    *(f"vr_mean_{i}" for i in range(1, 11)),
    *(f"vr_std_{i}" for i in range(1, 11)),
]


def train_predict(
    model_key: str,
    feature_cols: list[str],
    X_train,
    X_test,
    y_train: np.ndarray,
    residual_mode: bool = False,
    physics_train: np.ndarray | None = None,
    physics_test: np.ndarray | None = None,
) -> np.ndarray:
    pipe = make_pipeline(model_key, feature_cols)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    if residual_mode:
        pred = physics_test + pred
    return pred


def plot_comparison_bars(results_df: pl.DataFrame, title: str, path: Path):
    import seaborn as sns

    pdf = results_df.to_pandas()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        sns.barplot(data=pdf, x="model", y=metric, hue="approach", ax=ax)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)