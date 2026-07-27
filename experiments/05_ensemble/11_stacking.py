
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (
    BASE_NUMERIC,
    flight_level_split,
    load_and_clean,
    project_root,
)
import physics.eval_framework as ef
ef.CATEGORICAL = list(dict.fromkeys(list(ef.CATEGORICAL) + ["phase"]))
from aerotwin.engine.feature_engineering import ENERGY_FEATURES
from aerotwin.engine.weather_features import WEATHER_FEATURES

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

PARQUET = project_root() / "featured_dataset.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]
N_FOLDS = 5


def get_feature_set(df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    weather = [c for c in WEATHER_FEATURES if c in df.columns]
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    return list(dict.fromkeys(cols))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_l1_one(name: str, X: "pd.DataFrame | pl.DataFrame", y: np.ndarray, feat_cols: list[str], cat_names: list[str]):
    if name == "lgbm":
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        m.fit(X, y)
        return m
    if name == "xgb":
        m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=7,
                             subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                             enable_categorical=True)
        m.fit(X, y)
        return m
    if name == "rf":
        # RF needs numeric; use label codes for cats
        Xr = X.copy()
        if hasattr(Xr, "to_pandas"):
            Xr = Xr.to_pandas()
        for c in cat_names:
            if c in Xr.columns:
                Xr[c] = Xr[c].astype("category").cat.codes.astype(float)
        m = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=5,
                                  n_jobs=-1, random_state=RANDOM_STATE)
        m.fit(Xr, y)
        m._rf_cat_codes = True
        return m
    if name == "cat":
        cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
        Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
        pool = Pool(Xp, y, cat_features=cat_idx, feature_names=feat_cols)
        m = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=7,
                              loss_function="RMSE", random_seed=RANDOM_STATE,
                              allow_writing_files=False, thread_count=-1, verbose=False)
        m.fit(pool)
        return m
    raise ValueError(name)


def predict_l1_one(name: str, model, X: "pd.DataFrame | pl.DataFrame", feat_cols: list[str], cat_names: list[str]):
    if name == "cat":
        cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
        Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
        pool = Pool(Xp, cat_features=cat_idx, feature_names=feat_cols)
        return model.predict(pool)
    if name == "rf" and getattr(model, "_rf_cat_codes", False):
        Xr = X.copy()
        if hasattr(Xr, "to_pandas"):
            Xr = Xr.to_pandas()
        for c in cat_names:
            if c in Xr.columns:
                Xr[c] = Xr[c].astype("category").cat.codes.astype(float)
        return model.predict(Xr)
    return model.predict(X)


def main() -> None:
    print("=" * 72)
    print("STACKING (L1: LGBM/XGB/RF/CAT  |  L2: Ridge/Elastic/LGBM) — strict flight K-fold OOF")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids_all = df["flight_id"].to_numpy()

    outer_train_idx, outer_test_idx, train_fids, _ = flight_level_split(fids_all)
    y_train = df["actual_fuel_kg"].to_numpy()[outer_train_idx]
    y_test = df["actual_fuel_kg"].to_numpy()[outer_test_idx]
    feat_cols = get_feature_set(df)
    cat_names = [c for c in CAT_FEATURES if c in df.columns]
    X = pdf[feat_cols].copy()
    for c in cat_names:
        if c in X.columns:
            X[c] = X[c].astype("category")
    print(f"Train rows={len(outer_train_idx)} Test rows={len(outer_test_idx)} | feats={len(feat_cols)} | K={N_FOLDS}")

    # Build flight-grouped folds on TRAIN flights only
    train_fids_uniq = np.unique(fids_all[outer_train_idx])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {m: np.zeros(len(outer_train_idx)) for m in ["lgbm", "xgb", "rf", "cat"]}
    fold = 0
    t0 = time.perf_counter()
    for tr_f_idx, va_f_idx in kf.split(train_fids_uniq):
        fold += 1
        tr_f = train_fids_uniq[tr_f_idx]
        va_f = train_fids_uniq[va_f_idx]
        tr_mask = np.isin(fids_all[outer_train_idx], tr_f)
        va_mask = np.isin(fids_all[outer_train_idx], va_f)
        tr_rows = outer_train_idx[tr_mask]
        va_rows = outer_train_idx[va_mask]
        print(f"  Fold {fold}: train_f={len(tr_f)} val_f={len(va_f)} | tr_rows={len(tr_rows)} va={len(va_rows)}")

        X_tr = X.iloc[tr_rows]
        y_tr = y_train[tr_mask]
        X_va = X.iloc[va_rows]

        for mname in ["lgbm", "xgb", "rf", "cat"]:
            model = train_l1_one(mname, X_tr, y_tr, feat_cols, cat_names)
            p = predict_l1_one(mname, model, X_va, feat_cols, cat_names)
            oof_preds[mname][va_mask] = p
    print(f"OOF generation done in {time.perf_counter()-t0:.1f}s")

    # OOF matrix (n_train x 4)
    order = ["lgbm", "xgb", "rf", "cat"]
    P_oof = np.column_stack([oof_preds[m] for m in order])
    print(f"OOF matrix shape: {P_oof.shape}")

    # Train level-2 metas on OOF
    metas = {}
    meta_names = ["Ridge", "ElasticNet", "LGBM"]
    metas["Ridge"] = Ridge(alpha=0.5).fit(P_oof, y_train)
    metas["ElasticNet"] = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=3000).fit(P_oof, y_train)
    metas["LGBM"] = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                      random_state=RANDOM_STATE, verbose=-1).fit(P_oof, y_train)

    # Retrain L1 on full train
    print("\nRetraining L1 bases on full train ...")
    t0 = time.perf_counter()
    l1_full = {}
    for mname in order:
        l1_full[mname] = train_l1_one(mname, X.iloc[outer_train_idx], y_train, feat_cols, cat_names)
    print(f"  L1 full done in {time.perf_counter()-t0:.1f}s")

    # L1 test preds
    P_test = np.column_stack([
        predict_l1_one(m, l1_full[m], X.iloc[outer_test_idx], feat_cols, cat_names) for m in order
    ])

    # Meta predictions on test
    results = []
    for mname, meta in metas.items():
        p = meta.predict(P_test)
        met = evaluate(y_test, p)
        results.append({"level2": mname, "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"]})

    # Also L1 individuals on test for ref
    for i, m in enumerate(order):
        met = evaluate(y_test, P_test[:, i])
        results.append({"level2": f"L1-{m.upper()}", "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"]})

    table = pl.DataFrame(results)
    table.write_csv(OUT / "table_stacking.csv")
    print(f"\nSaved {OUT / 'table_stacking.csv'}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    pdfp = table.to_pandas()
    sns.barplot(data=pdfp, x="level2", y="rmse", ax=ax, palette="coolwarm")
    ax.set_title("Stacking Level-2 (and L1 baselines) RMSE — Held-out Flights")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("RMSE (kg)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stacking.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_stacking.png'}")

    print("\n" + "=" * 72)
    print("STACKING RESULTS (held-out)")
    for r in results:
        print(f"  {r['level2']:12s} MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} R2={r['r2']:.4f}")
    best = min([r for r in results if not r["level2"].startswith("L1")], key=lambda x: x["rmse"])
    print(f"\nBest meta: {best['level2']} RMSE={best['rmse']:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
