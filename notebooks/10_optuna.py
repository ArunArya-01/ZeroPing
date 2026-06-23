
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (
    BASE_NUMERIC,
    flight_level_split,
    load_and_clean,
    project_root,
)
import physics.eval_framework as ef
ef.CATEGORICAL = list(dict.fromkeys(list(ef.CATEGORICAL) + ["phase"]))
from physics.feature_engineering import ENERGY_FEATURES
from physics.weather_features import WEATHER_FEATURES

import lightgbm as lgb

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

PARQUET = project_root() / "featured_dataset.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]
N_TRIALS_CAT = 25
N_TRIALS_LGB = 25


def get_feature_set(df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    weather = [c for c in WEATHER_FEATURES if c in df.columns]
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    return list(dict.fromkeys(cols))


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def objective_cat(trial, X_tr, y_tr, X_va, y_va, feat_cols, cat_names):
    params = {
        "iterations": trial.suggest_int("iterations", 800, 4000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    Xtrp = X_tr.to_pandas() if hasattr(X_tr, "to_pandas") else X_tr
    Xvap = X_va.to_pandas() if hasattr(X_va, "to_pandas") else X_va
    pool_tr = Pool(Xtrp, y_tr, cat_features=cat_idx, feature_names=feat_cols)
    pool_va = Pool(Xvap, y_va, cat_features=cat_idx, feature_names=feat_cols)
    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)
    p = model.predict(pool_va)
    return rmse(y_va, p)


def objective_lgb(trial, X_tr, y_tr, X_va, y_va):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 600, 3000),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(120, verbose=False)],
    )
    p = model.predict(X_va, num_iteration=model.best_iteration_)
    return rmse(y_va, p)


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("=" * 72)
    print("OPTUNA TUNING (CatBoost + LGBM) — Energy+Weather+Physics | FLIGHT SPLIT")
    print(f"Trials: Cat={N_TRIALS_CAT} LGBM={N_TRIALS_LGB} (spec was 500; reduced for runtime)")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    train_idx, test_idx, _, _ = flight_level_split(fids)

    # For optuna we use inner val from train flights
    # Reuse a sub split
    from sklearn.model_selection import train_test_split as tts
    train_fids = np.unique(fids[train_idx])
    sub_tr_f, sub_va_f = tts(train_fids, test_size=0.2, random_state=RANDOM_STATE)
    sub_tr_mask = np.isin(fids[train_idx], sub_tr_f)
    sub_va_mask = np.isin(fids[train_idx], sub_va_f)
    gtr = np.flatnonzero(sub_tr_mask)
    gva = np.flatnonzero(sub_va_mask)
    # map
    train_sub_idx = train_idx[gtr]
    val_sub_idx = train_idx[gva]

    y_tr = df["actual_fuel_kg"].to_numpy()[train_sub_idx]
    y_va = df["actual_fuel_kg"].to_numpy()[val_sub_idx]

    feat_cols = get_feature_set(df)
    cat_names = [c for c in CAT_FEATURES if c in df.columns]
    X_tr = pdf[feat_cols].iloc[train_sub_idx].copy()
    X_va = pdf[feat_cols].iloc[val_sub_idx].copy()
    for c in cat_names:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype("category")
            X_va[c] = X_va[c].astype("category")
    print(f"Optuna inner: tr={len(train_sub_idx)} va={len(val_sub_idx)} | feats={len(feat_cols)}")

    # CatBoost study
    print("\n--- Optuna CatBoost ---")
    study_cat = optuna.create_study(direction="minimize", pruner=MedianPruner(n_warmup_steps=5), study_name="catboost_fuel")
    t0 = time.perf_counter()
    study_cat.optimize(
        lambda t: objective_cat(t, X_tr, y_tr, X_va, y_va, feat_cols, cat_names),
        n_trials=N_TRIALS_CAT,
        show_progress_bar=True,
    )
    print(f"Cat best RMSE (inner val): {study_cat.best_value:.4f} in {time.perf_counter()-t0:.0f}s")
    print("Cat best params:", study_cat.best_params)

    # LGBM study
    print("\n--- Optuna LightGBM ---")
    study_lgb = optuna.create_study(direction="minimize", pruner=MedianPruner(), study_name="lgbm_fuel")
    t0 = time.perf_counter()
    study_lgb.optimize(
        lambda t: objective_lgb(t, X_tr, y_tr, X_va, y_va),
        n_trials=N_TRIALS_LGB,
        show_progress_bar=True,
    )
    print(f"LGBM best RMSE (inner val): {study_lgb.best_value:.4f} in {time.perf_counter()-t0:.0f}s")
    print("LGBM best params:", study_lgb.best_params)

    # Now retrain best on full train, eval on held-out test
    print("\nRetraining best configs on full train + eval on held-out ...")
    full_tr_idx = train_idx
    y_full_tr = df["actual_fuel_kg"].to_numpy()[full_tr_idx]
    y_te = df["actual_fuel_kg"].to_numpy()[test_idx]
    X_full_tr = pdf[feat_cols].iloc[full_tr_idx].copy()
    X_te = pdf[feat_cols].iloc[test_idx].copy()
    for c in cat_names:
        if c in X_full_tr.columns:
            X_full_tr[c] = X_full_tr[c].astype("category")
            X_te[c] = X_te[c].astype("category")

    # Cat full
    cat_p = {**study_cat.best_params}
    cat_p.setdefault("loss_function", "RMSE")
    cat_p["random_seed"] = RANDOM_STATE
    cat_p["allow_writing_files"] = False
    cat_p["thread_count"] = -1
    cat_p["verbose"] = False
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    Xftrp = X_full_tr.to_pandas() if hasattr(X_full_tr, "to_pandas") else X_full_tr
    Xtep = X_te.to_pandas() if hasattr(X_te, "to_pandas") else X_te
    pool_full = Pool(Xftrp, y_full_tr, cat_features=cat_idx, feature_names=feat_cols)
    pool_te = Pool(Xtep, cat_features=cat_idx, feature_names=feat_cols)
    best_cat = CatBoostRegressor(**cat_p)
    best_cat.fit(pool_full)
    p_cat_te = best_cat.predict(pool_te)
    rmse_cat = rmse(y_te, p_cat_te)
    mae_cat = float(np.abs(y_te - p_cat_te).mean())
    r2_cat = float(1 - ((y_te - p_cat_te)**2).sum() / ((y_te - y_te.mean())**2).sum())

    # LGBM full
    lgb_p = study_lgb.best_params.copy()
    lgb_p["random_state"] = RANDOM_STATE
    lgb_p["verbose"] = -1
    lgb_p["n_jobs"] = -1
    best_lgb = lgb.LGBMRegressor(**lgb_p)
    best_lgb.fit(X_full_tr, y_full_tr)
    p_lgb_te = best_lgb.predict(X_te)
    rmse_lgb = rmse(y_te, p_lgb_te)
    mae_lgb = float(np.abs(y_te - p_lgb_te).mean())
    r2_lgb = float(1 - ((y_te - p_lgb_te)**2).sum() / ((y_te - y_te.mean())**2).sum())

    rows = [
        {"model": "Optuna-CatBoost", "mae": mae_cat, "rmse": rmse_cat, "r2": r2_cat, "trials": N_TRIALS_CAT},
        {"model": "Optuna-LightGBM", "mae": mae_lgb, "rmse": rmse_lgb, "r2": r2_lgb, "trials": N_TRIALS_LGB},
    ]
    pl.DataFrame(rows).write_csv(OUT / "table_optuna.csv")
    print(f"Saved {OUT / 'table_optuna.csv'}")

    # History plot (cat + lgb)
    fig, ax = plt.subplots(figsize=(9, 5))
    cat_vals = [t.value for t in study_cat.trials if t.value is not None]
    lgb_vals = [t.value for t in study_lgb.trials if t.value is not None]
    ax.plot(np.arange(len(cat_vals)), cat_vals, label="CatBoost", alpha=0.7)
    ax.plot(np.arange(len(lgb_vals)), lgb_vals, label="LightGBM", alpha=0.7)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Val RMSE (inner)")
    ax.set_title("Optuna Optimization History (inner val flights)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_optuna_history.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_optuna_history.png'}")

    # best params json
    best = {
        "catboost": study_cat.best_params,
        "lightgbm": study_lgb.best_params,
        "test_rmse_cat": rmse_cat,
        "test_rmse_lgb": rmse_lgb,
        "note": "tuned on inner flight split; final eval on held-out test flights",
    }
    with open(OUT / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved {OUT / 'best_params.json'}")

    print("\n" + "=" * 72)
    print("OPTUNA BEST (held-out test flights)")
    print(f"  CatBoost  MAE={mae_cat:.2f} RMSE={rmse_cat:.2f} R2={r2_cat:.4f}")
    print(f"  LightGBM  MAE={mae_lgb:.2f} RMSE={rmse_lgb:.2f} R2={r2_lgb:.4f}")
    winner = "CatBoost" if rmse_cat < rmse_lgb else "LightGBM"
    print(f"  Winner: {winner}")
    print("=" * 72)


if __name__ == "__main__":
    main()
