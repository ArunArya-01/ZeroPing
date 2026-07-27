"""Official PRC2025 benchmark utilities (frozen AeroTwin methodology).

This module supports the one-shot official evaluation on Rank/Final after
labels were released. It intentionally:

* trains ONLY on the train split
* reuses the existing V4 feature pipeline (``predict_fuel_intervals``)
* does not introduce new features or hyperparameter search

See ``notebooks/16_dataset_audit.py`` and ``notebooks/17_official_prc_evaluation.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.data import AeroDataLoader
from aerotwin.engine.eval_framework import (
    BASE_NUMERIC,
    CATEGORICAL,
    evaluate,
    project_root,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES
from aerotwin.engine.openap_baseline import predict_fuel_intervals
from aerotwin.engine.weather_features import WEATHER_FEATURES

SplitName = Literal["train", "rank", "final"]

RANDOM_STATE = 42
# Frozen V4 tree hyper-parameters (from notebooks/10 + stacking L1 style)
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
)
CAT_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=8,
    random_seed=RANDOM_STATE,
    verbose=0,
    thread_count=-1,
    allow_writing_files=False,
)

# Official paper (Sun et al., JOAS 2026): winner RMSE ~201 kg on combined eval
OFFICIAL_WINNER_RMSE_COMBINED = 201.0
# Legacy internal reference often cited as 200.83
LEGACY_WINNER_RMSE = 200.83

CAT_FEATURES = list(CATEGORICAL)  # frozen: no phase for sklearn-path Direct/Flow singles


def featured_path(split: SplitName, root: Path | None = None) -> Path:
    root = root or project_root()
    if split == "train":
        # Prefer existing train artifact built from train-only flights
        p = root / "featured_dataset.parquet"
        if p.exists():
            return p
    return root / f"featured_dataset_{split}.parquet"


def build_featured_for_split(
    split: SplitName,
    *,
    max_flights: int | None = None,
    out_path: Path | None = None,
    resume: bool = True,
) -> pl.DataFrame:
    """Build interval-level featured dataset for one official split.

    Uses the same OpenAP + energy/operational/weather path as
    ``physics/build_featured_dataset.py`` (frozen methodology).
    """
    out = out_path or featured_path(split)
    cache_dir = project_root() / "cache" / f"featured_{split}_parts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    loader = AeroDataLoader()
    fl = loader.get_flightlist(split)
    fuel_all = loader.get_fuel_labels(split)
    usable = loader.get_usable_flight_ids(split)
    LOGGER.info("Split %s: %d usable flights with trajectories", split, len(usable))

    if max_flights is not None:
        usable = usable[:max_flights]
        LOGGER.info("Limiting to first %d flights", max_flights)

    # Index metadata / fuel once (avoid per-flight HF flightlist reloads)
    meta_by_id = {
        str(r["flight_id"]): r
        for r in fl.iter_rows(named=True)
    }
    # Prefetch fuel grouped by flight_id via partition
    fuel_map: dict[str, pl.DataFrame] = {}
    for fid, grp in fuel_all.group_by("flight_id", maintain_order=True):
        key = fid[0] if isinstance(fid, tuple) else fid
        fuel_map[str(key)] = grp

    done_ids: set[str] = set()
    if resume:
        for part in cache_dir.glob("*.parquet"):
            done_ids.add(part.stem)

    all_dfs: list[pl.DataFrame] = []
    # Load already-done parts
    for part in sorted(cache_dir.glob("*.parquet")):
        try:
            all_dfs.append(pl.read_parquet(part))
        except Exception as exc:
            LOGGER.warning("Could not read cache part %s: %s", part, exc)

    for i, fid in enumerate(usable):
        if fid in done_ids:
            continue
        if i % 50 == 0:
            LOGGER.info("  %s %d/%d %s (done_parts=%d)", split, i + 1, len(usable), fid, len(done_ids) + len(all_dfs))
        try:
            # Direct path — skip flightlist lookup on every flight (major HF bottleneck)
            # HF layout: flights_{split}/prc....parquet (train may nest flights_train/flights_train/)
            candidates = [
                f"flights_{split}/{fid}.parquet",
                f"flights_{split}/flights_{split}/{fid}.parquet",
            ]
            traj = None
            last_err: Exception | None = None
            for rel in candidates:
                try:
                    traj = loader.load_flight(rel)
                    break
                except Exception as exc:  # try next layout
                    last_err = exc
                    traj = None
            if traj is None or traj.is_empty():
                # Fallback to id resolver once
                try:
                    traj = loader.load_flight_by_id(fid, split=split)
                except Exception as exc:
                    raise last_err or exc

            fu = fuel_map.get(str(fid))
            if fu is None or fu.is_empty() or traj is None or traj.is_empty():
                continue
            meta_row = meta_by_id.get(str(fid))
            if not meta_row:
                continue
            interval_df = predict_fuel_intervals(traj, fu, flight_meta=meta_row)
            if interval_df.is_empty():
                continue
            interval_df = interval_df.with_columns(
                pl.lit(fid).alias("flight_id"),
                (pl.col("actual_fuel_kg") - pl.col("physics_fuel_kg")).alias("residual_kg"),
                pl.lit(split).alias("dataset_split"),
            )
            # cache per flight
            safe = fid.replace("/", "_")
            interval_df.write_parquet(cache_dir / f"{safe}.parquet")
            all_dfs.append(interval_df)
            done_ids.add(safe)
        except Exception as exc:
            LOGGER.warning("ERROR %s %s: %s", split, fid, exc)
            continue

    if not all_dfs:
        LOGGER.error("No intervals for split %s", split)
        return pl.DataFrame()

    dataset = pl.concat(all_dfs, how="diagonal_relaxed")
    if "energy_change_jpkg" in dataset.columns and "flight_id" in dataset.columns:
        sort_cols = ["flight_id"]
        if "start_fraction_of_flight" in dataset.columns:
            sort_cols.append("start_fraction_of_flight")
        dataset = dataset.sort(sort_cols).with_columns(
            pl.col("energy_change_jpkg").cum_sum().over("flight_id").alias("cumulative_energy_change_jpkg")
        )

    # Clean like load_and_clean
    need = ["actual_fuel_kg", "physics_fuel_kg", "flight_id", "duration_s"]
    present = [c for c in need if c in dataset.columns]
    dataset = dataset.drop_nulls(subset=present).filter(
        pl.col("actual_fuel_kg").is_finite()
        & pl.col("physics_fuel_kg").is_finite()
        & (pl.col("duration_s") > 0)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(out)
    LOGGER.info("Wrote %s: %d rows, %d flights", out, len(dataset), dataset["flight_id"].n_unique())
    return dataset


def ew_feature_cols(df: pl.DataFrame) -> list[str]:
    """Frozen Energy+Weather + base + physics + categoricals."""
    cols = list(BASE_NUMERIC)
    cols += [c for c in ENERGY_FEATURES if c in df.columns]
    cols += [c for c in WEATHER_FEATURES if c in df.columns]
    if "physics_fuel_kg" in df.columns:
        cols.append("physics_fuel_kg")
    cols += [c for c in CAT_FEATURES if c in df.columns]
    return list(dict.fromkeys([c for c in cols if c in df.columns]))


def prepare_xy(
    df: pl.DataFrame,
    feature_cols: list[str],
    target: Literal["direct", "fuel_flow"],
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_pandas, y_train_space, y_true_kg, duration)."""
    pdf = df.to_pandas()
    for c in feature_cols:
        if c in CAT_FEATURES and c in pdf.columns:
            pdf[c] = pdf[c].astype(str).fillna("missing")
    X = pdf[feature_cols]
    y_kg = pdf["actual_fuel_kg"].to_numpy(dtype=np.float64)
    dur = np.clip(pdf["duration_s"].to_numpy(dtype=np.float64), 1.0, None)
    if target == "fuel_flow":
        y = y_kg / dur
    else:
        y = y_kg
    return X, y, y_kg, dur


def train_model(model_key: str, X, y: np.ndarray, feature_cols: list[str]):
    """Train a frozen tree model (or pipeline with OHE for cats)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = [c for c in feature_cols if c not in CAT_FEATURES]
    cat = [c for c in feature_cols if c in CAT_FEATURES]
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )
    if model_key == "xgb":
        import xgboost as xgb

        model = xgb.XGBRegressor(**XGB_PARAMS)
    elif model_key == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
    elif model_key == "cat":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**CAT_PARAMS)
    else:
        raise ValueError(model_key)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X, y)
    return pipe


def predict_fuel_kg(pipe, X, duration: np.ndarray, target: Literal["direct", "fuel_flow"]) -> np.ndarray:
    raw = np.asarray(pipe.predict(X), dtype=np.float64)
    if target == "fuel_flow":
        return raw * np.clip(duration.astype(np.float64), 1.0, None)
    return raw


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flight_ids: np.ndarray,
    metric: str = "rmse",
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Flight-clustered bootstrap CI for MAE or RMSE."""
    unique, inverse = np.unique(flight_ids, return_inverse=True)
    n_f = len(unique)
    # precompute per-flight error sums / counts
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2
    order = np.argsort(inverse, kind="stable")
    inv = inverse[order]
    bounds = np.flatnonzero(np.diff(inv)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(inv)]))
    sum_abs = np.add.reduceat(abs_err[order], starts)
    sum_sq = np.add.reduceat(sq_err[order], starts)
    counts = ends - starts

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_f, size=(n_boot, n_f))
    if metric == "mae":
        boot = sum_abs[idx].sum(axis=1) / counts[idx].sum(axis=1)
        point = float(abs_err.mean())
    else:
        boot = np.sqrt(sum_sq[idx].sum(axis=1) / counts[idx].sum(axis=1))
        point = float(np.sqrt(np.mean(sq_err)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "point": point,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_flights": n_f,
        "n_boot": n_boot,
    }


def build_oof_matrix(
    df_train: pl.DataFrame,
    feature_cols: list[str],
    model_specs: list[tuple[str, str]],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """GroupKFold OOF predictions for ensemble base models.

    model_specs: list of (name, target) where target in {direct, fuel_flow}.
    Returns (P_oof [n, m], y_kg, fitted_full_models on all train).
    """
    from sklearn.model_selection import GroupKFold

    X, y_direct, y_kg, dur = prepare_xy(df_train, feature_cols, "direct")
    # y spaces per target
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
            pipe = train_model(mkey, X.iloc[tr], y_space[tr], feature_cols)
            pred = predict_fuel_kg(pipe, X.iloc[va], dur[va], target)  # type: ignore[arg-type]
            P[va, j] = pred

    # Fit full-train models for test application
    full_models = []
    for mkey, target in model_specs:
        y_space = y_flow if target == "fuel_flow" else y_direct
        pipe = train_model(mkey, X, y_space, feature_cols)
        full_models.append((mkey, target, pipe))
    return P, y_kg, full_models


def fit_meta(P_oof: np.ndarray, y_kg: np.ndarray, kind: str = "lgbm"):
    if kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0).fit(P_oof, y_kg)
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        random_state=RANDOM_STATE,
        verbose=-1,
    ).fit(P_oof, y_kg)


def choose_meta_on_train_folds(
    P_oof: np.ndarray,
    y_kg: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[str, Any]:
    """Pick Ridge vs LGBM meta using only training OOF folds (nested-ish CV on OOF)."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    scores = {"ridge": [], "lgbm": []}
    for tr, va in gkf.split(P_oof, y_kg, groups):
        for kind in scores:
            meta = fit_meta(P_oof[tr], y_kg[tr], kind)
            pred = meta.predict(P_oof[va])
            scores[kind].append(float(np.sqrt(np.mean((pred - y_kg[va]) ** 2))))
    mean_rmse = {k: float(np.mean(v)) for k, v in scores.items()}
    LOGGER.info("Meta CV RMSE on train OOF: %s", mean_rmse)
    best = min(mean_rmse, key=mean_rmse.get)  # type: ignore[arg-type]
    meta = fit_meta(P_oof, y_kg, best)
    return best, meta


def apply_bases(
    full_models: list[tuple[str, str, Any]],
    df: pl.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    X, _, y_kg, dur = prepare_xy(df, feature_cols, "direct")
    cols = []
    for mkey, target, pipe in full_models:
        cols.append(predict_fuel_kg(pipe, X, dur, target))  # type: ignore[arg-type]
    return np.column_stack(cols)


def protocol_manifest() -> dict[str, Any]:
    return {
        "methodology": "frozen_aerotwin_v4",
        "official_metric": "RMSE_kg",
        "paper": "Sun et al., Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge, JOAS 2026",
        "winner_rmse_combined_paper": OFFICIAL_WINNER_RMSE_COMBINED,
        "winner_rmse_legacy_internal_cite": LEGACY_WINNER_RMSE,
        "train_period": "Apr–Aug 2025",
        "rank_period": "Sep 2025",
        "final_period": "Oct 2025",
        "hyperparameters": {
            "lgbm": LGBM_PARAMS,
            "xgb": XGB_PARAMS,
            "cat": {k: v for k, v in CAT_PARAMS.items() if k != "allow_writing_files"},
        },
        "no_tuning_on_rank_final": True,
        "features": "BASE_NUMERIC + ENERGY_FEATURES + WEATHER_FEATURES + physics_fuel_kg + CATEGORICAL",
    }
