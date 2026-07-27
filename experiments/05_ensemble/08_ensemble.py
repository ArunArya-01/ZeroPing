
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (
    BASE_NUMERIC,
    CATEGORICAL,
    RANDOM_STATE,
    flight_level_split,
    load_and_clean,
    make_pipeline,
    project_root,
    train_predict,
)
import physics.eval_framework as ef
ef.CATEGORICAL = list(dict.fromkeys(list(ef.CATEGORICAL) + ["phase"]))  # ensure phase treated as cat in OHE pipeline
from aerotwin.engine.feature_engineering import ENERGY_FEATURES
from aerotwin.engine.weather_features import WEATHER_FEATURES

import lightgbm as lgb
import xgboost as xgb

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

PARQUET = project_root() / "featured_dataset.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]


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


def flight_split_indices(flight_ids: np.ndarray, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Helper to get train/test idx from flight ids (no full fids return)."""
    from sklearn.model_selection import train_test_split as tts
    uniq = np.unique(flight_ids)
    tr_f, te_f = tts(uniq, test_size=test_size, random_state=seed)
    tr_mask = np.isin(flight_ids, tr_f)
    te_mask = np.isin(flight_ids, te_f)
    return np.flatnonzero(tr_mask), np.flatnonzero(te_mask)


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def train_light_rf(X_tr, y_tr, X_val, cat_names):
    """Light RF using label codes for cats (avoids high-dim OHE slowdown for RF in ensemble)."""
    Xr_tr = _to_pandas(X_tr).copy()
    Xr_va = _to_pandas(X_val).copy()
    for c in cat_names:
        if c in Xr_tr.columns:
            Xr_tr[c] = Xr_tr[c].astype("category").cat.codes.astype(float)
            Xr_va[c] = Xr_va[c].astype("category").cat.codes.astype(float)
    m = RandomForestRegressor(
        n_estimators=50, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    m.fit(Xr_tr, y_tr)
    return m.predict(Xr_va)


def predict_light_rf(model, X, cat_names):
    Xr = _to_pandas(X).copy()
    for c in cat_names:
        if c in Xr.columns:
            Xr[c] = Xr[c].astype("category").cat.codes.astype(float)
    return model.predict(Xr)


def train_base_models(
    X_tr: "pd.DataFrame | pl.DataFrame",
    y_tr: np.ndarray,
    X_val: "pd.DataFrame | pl.DataFrame",
    y_val: np.ndarray,
    cat_names: list[str],
    feat_cols: list[str],
) -> dict[str, np.ndarray]:
    """Train 4 bases on tr, return val predictions dict. Use shared pipeline for LGBM/XGB/RF; light RF and CAT with labels on eval for early stopping."""
    preds_val = {}
    # LGBM/XGB via shared (handles OHE cats inside); RF light with codes for speed
    for mk, short in [("lgbm", "lgbm"), ("xgb", "xgb")]:
        p = train_predict(mk, feat_cols, X_tr, X_val, y_tr)
        preds_val[short] = p
    preds_val["rf"] = train_light_rf(X_tr, y_tr, X_val, cat_names)

    # CAT native (cat features)
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    X_tr_p = X_tr.to_pandas() if hasattr(X_tr, "to_pandas") else X_tr
    X_val_p = X_val.to_pandas() if hasattr(X_val, "to_pandas") else X_val
    pool_tr = Pool(X_tr_p, y_tr, cat_features=cat_idx, feature_names=feat_cols)
    pool_val = Pool(X_val_p, y_val, cat_features=cat_idx, feature_names=feat_cols)
    model_cat = CatBoostRegressor(
        iterations=300, learning_rate=0.03, depth=8,
        loss_function="RMSE", eval_metric="RMSE",
        early_stopping_rounds=50, random_seed=RANDOM_STATE,
        allow_writing_files=False, thread_count=-1, verbose=False,
    )
    model_cat.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    preds_val["cat"] = model_cat.predict(pool_val)

    return preds_val, {"cat": model_cat}  # only cat kept if needed; others via pipeline not returned


def weighted_avg_objective(w, P, y):
    """P: (n,4) preds, w len=4, return rmse to min."""
    pred = (P * w).sum(axis=1)
    return np.sqrt(mean_squared_error(y, pred))


def find_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Optimize non-negative weights sum to 1."""
    n = P.shape[1]
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = minimize(
        weighted_avg_objective, w0, args=(P, y),
        bounds=bounds, constraints=cons,
        method="SLSQP", options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        w = np.ones(n) / n
    else:
        w = np.clip(res.x, 0, 1)
        w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    return w


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

    print("=" * 72)
    print("ENSEMBLES: Avg / Weighted / Ridge / ElasticNet (strict flight-level)")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()

    # Outer strict split (same as all)
    train_idx, test_idx, train_fids, test_fids = flight_level_split(fids)
    print(f"Outer split: train_flights={len(train_fids)} test_flights={len(test_fids)}")

    y_train = df["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = df["actual_fuel_kg"].to_numpy()[test_idx]

    feat_cols = get_feature_set(df)
    cat_names = [c for c in CAT_FEATURES if c in df.columns]
    print(f"Features: {len(feat_cols)} (cats: {cat_names})")

    # Inner split on TRAIN flights only for meta/weight selection (no test leakage)
    train_flight_ids_inner = pdf["flight_id"].iloc[train_idx].to_numpy()
    subtrain_idx, subval_idx = flight_split_indices(train_flight_ids_inner, test_size=0.2, seed=RANDOM_STATE)
    # Map back to global indices
    global_subtrain = train_idx[subtrain_idx]
    global_subval = train_idx[subval_idx]
    print(f"Inner val split: subtrain={len(global_subtrain)} subval={len(global_subval)} rows (train flights only)")

    X_all = pdf[feat_cols]
    X_subtr = X_all.iloc[global_subtrain]
    y_subtr = y_train[subtrain_idx]
    X_subva = X_all.iloc[global_subval]
    y_subva = y_train[subval_idx]
    X_test = X_all.iloc[test_idx]

    print("\nTraining base models on inner subtrain ...")
    t0 = time.perf_counter()
    val_preds, base_models = train_base_models(X_subtr, y_subtr, X_subva, y_subva, cat_names, feat_cols)
    print(f"  Bases trained in {time.perf_counter()-t0:.1f}s")

    # Stack matrix for inner val: order lgbm, xgb, rf, cat
    order = ["lgbm", "xgb", "rf", "cat"]
    P_val = np.column_stack([val_preds[m] for m in order])
    print(f"Val preds shape: {P_val.shape}")

    # Simple avg
    w_simple = np.ones(4) / 4
    pred_simple_val = (P_val * w_simple).sum(1)
    m_simple_val = evaluate(y_subva, pred_simple_val)

    # Weighted avg auto
    w_opt = find_weights(P_val, y_subva)
    pred_w_val = (P_val * w_opt).sum(1)
    m_w_val = evaluate(y_subva, pred_w_val)
    print(f"  Optimized weights: {dict(zip(order, [round(x,4) for x in w_opt]))}")

    # Ridge meta
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(P_val, y_subva)
    pred_r_val = ridge.predict(P_val)
    m_r_val = evaluate(y_subva, pred_r_val)

    # ElasticNet meta
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    en.fit(P_val, y_subva)
    pred_en_val = en.predict(P_val)
    m_en_val = evaluate(y_subva, pred_en_val)

    print("\nInner val ensemble metrics (for selection):")
    for nm, mm in [("simple", m_simple_val), ("weighted", m_w_val), ("ridge", m_r_val), ("elastic", m_en_val)]:
        print(f"  {nm:8s} RMSE={mm['rmse']:.2f} MAE={mm['mae']:.2f}")

    # Now retrain bases on FULL outer train, predict test
    print("\nRetraining bases on full train for test eval ...")
    t0 = time.perf_counter()
    X_tr_full = X_all.iloc[train_idx]
    X_te = X_all.iloc[test_idx]
    # LGBM/XGB via shared pipeline (OHE cats); RF light with codes for speed
    p_lgb_test = train_predict("lgbm", feat_cols, X_tr_full, X_te, y_train)
    p_xgb_test = train_predict("xgb", feat_cols, X_tr_full, X_te, y_train)
    # Train light RF for test preds
    rf_model = RandomForestRegressor(
        n_estimators=50, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    Xr_tr = _to_pandas(X_tr_full).copy()
    Xr_te = _to_pandas(X_te).copy()
    for c in cat_names:
        if c in Xr_tr.columns:
            Xr_tr[c] = Xr_tr[c].astype("category").cat.codes.astype(float)
            Xr_te[c] = Xr_te[c].astype("category").cat.codes.astype(float)
    rf_model.fit(Xr_tr, y_train)
    p_rf_test = rf_model.predict(Xr_te)

    # CAT full
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    pool_full_tr = Pool(X_tr_full.to_pandas() if hasattr(X_tr_full, "to_pandas") else X_tr_full, y_train, cat_features=cat_idx, feature_names=feat_cols)
    pool_test = Pool(X_te.to_pandas() if hasattr(X_te, "to_pandas") else X_te, cat_features=cat_idx, feature_names=feat_cols)
    m_cat = CatBoostRegressor(
        iterations=300, learning_rate=0.03, depth=8, loss_function="RMSE",
        early_stopping_rounds=50, random_seed=RANDOM_STATE,
        allow_writing_files=False, thread_count=-1, verbose=False,
    )
    m_cat.fit(pool_full_tr, verbose=False)
    p_cat_test = m_cat.predict(pool_test)
    print(f"  Full bases retrained in {time.perf_counter()-t0:.1f}s")

    P_test = np.column_stack([p_lgb_test, p_xgb_test, p_rf_test, p_cat_test])

    # Apply ensembles on test
    ens_results = []
    # 1. simple
    p_simple = (P_test * w_simple).sum(1)
    m_simple = evaluate(y_test, p_simple)
    ens_results.append({"method": "SimpleAvg", "mae": m_simple["mae"], "rmse": m_simple["rmse"], "r2": m_simple["r2"], "note": "equal 0.25"})

    # 2. weighted
    p_w = (P_test * w_opt).sum(1)
    m_w = evaluate(y_test, p_w)
    ens_results.append({"method": "WeightedAvg", "mae": m_w["mae"], "rmse": m_w["rmse"], "r2": m_w["r2"], "note": f"w={np.round(w_opt,3).tolist()}"})

    # 3. ridge
    p_r = ridge.predict(P_test)
    m_r = evaluate(y_test, p_r)
    ens_results.append({"method": "RidgeStack", "mae": m_r["mae"], "rmse": m_r["rmse"], "r2": m_r["r2"], "note": f"alpha=1.0 coefs~{np.round(ridge.coef_,3).tolist()}"})

    # 4. elastic
    p_en = en.predict(P_test)
    m_en = evaluate(y_test, p_en)
    ens_results.append({"method": "ElasticNetStack", "mae": m_en["mae"], "rmse": m_en["rmse"], "r2": m_en["r2"], "note": f"alpha=0.1 l1=0.5"})

    # Also report individual bases on test for ref
    base_test = {
        "lgbm": evaluate(y_test, p_lgb_test),
        "xgb": evaluate(y_test, p_xgb_test),
        "rf": evaluate(y_test, p_rf_test),
        "cat": evaluate(y_test, p_cat_test),
    }

    table = pl.DataFrame(ens_results)
    table_path = OUT / "table_ensemble.csv"
    table.write_csv(table_path)
    print(f"\nSaved {table_path}")

    # Also append bases to table? separate or include
    base_rows = [
        {"method": f"Base-{k.upper()}", "mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"], "note": "individual"}
        for k, v in base_test.items()
    ]
    full_table = pl.concat([table, pl.DataFrame(base_rows)])
    full_table.write_csv(OUT / "table_ensemble.csv")
    print("  (table includes bases too)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    pdf_tbl = table.to_pandas()
    for ax, met in zip(axes, ["mae", "rmse", "r2"]):
        sns.barplot(data=pdf_tbl, x="method", y=met, ax=ax, palette="viridis")
        ax.set_title(met.upper())
        ax.tick_params(axis="x", rotation=20)
        if met != "r2":
            ax.set_ylabel("kg")
    fig.suptitle("Ensemble Methods on Held-Out Flights (Energy+Weather+Physics)", y=1.02)
    fig.tight_layout()
    fig_path = OUT / "fig_ensemble.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")

    print("\n" + "=" * 72)
    print("ENSEMBLE RESULTS (held-out flights)")
    for r in ens_results:
        print(f"  {r['method']:16s} MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} R2={r['r2']:.4f}  {r['note']}")
    best = min(ens_results, key=lambda r: r["rmse"])
    print(f"\nBest ensemble: {best['method']} RMSE={best['rmse']:.2f}")
    print("Individual bases:")
    for k, v in base_test.items():
        print(f"  {k.upper():4s} RMSE={v['rmse']:.2f}")
    print("=" * 72)

    # Save best weights for ref
    with open(OUT / "ensemble_weights.txt", "w") as f:
        f.write(f"simple: {w_simple.tolist()}\n")
        f.write(f"opt: {w_opt.tolist()}\n")
        f.write(f"ridge_coefs: {ridge.coef_.tolist()}\n")
        f.write(f"en_coefs: {en.coef_.tolist()}\n")
    print(f"Saved {OUT / 'ensemble_weights.txt'}")


if __name__ == "__main__":
    main()
