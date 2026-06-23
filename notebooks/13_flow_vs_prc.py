
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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (
    BASE_NUMERIC,
    MASS_FEATURES,
    flight_level_split,
    load_and_clean,
    project_root,
)
import physics.eval_framework as ef
ef.CATEGORICAL = list(dict.fromkeys(list(ef.CATEGORICAL) + ["phase"]))
from physics.feature_engineering import ENERGY_FEATURES

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

MASS_PARQUET = project_root() / "featured_dataset_mass.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]
N_FOLDS = 5
WINNER_RMSE = 200.83
REFERENCE_2029 = {"mae": 84.3, "rmse": 202.9, "r2": 0.9481}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_variant_feats(vname: str, df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    mass = [c for c in MASS_FEATURES if c in df.columns]
    cats = [c for c in CAT_FEATURES if c in df.columns]
    base = list(BASE_NUMERIC) + ["physics_fuel_kg"] + cats
    if vname == "FuelFlow":
        extra: list[str] = []
    elif vname == "FuelFlow+Energy":
        extra = energy
    else:
        extra = energy + mass
    cols = base + extra
    return list(dict.fromkeys(cols))


def train_l1_one(name: str, X, y: np.ndarray, feat_cols: list[str], cat_names: list[str]):
    if name == "lgbm":
        m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
        m.fit(X, y)
        return m
    if name == "xgb":
        m = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            enable_categorical=True
        )
        m.fit(X, y)
        return m
    if name == "rf":
        Xr = X.copy()
        if hasattr(Xr, "to_pandas"):
            Xr = Xr.to_pandas()
        for c in cat_names:
            if c in Xr.columns:
                Xr[c] = Xr[c].astype("category").cat.codes.astype(float)
        m = RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        )
        m.fit(Xr, y)
        m._rf_cat_codes = True
        return m
    if name == "cat":
        cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
        Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
        pool = Pool(Xp, y, cat_features=cat_idx, feature_names=feat_cols)
        m = CatBoostRegressor(
            iterations=800, learning_rate=0.03, depth=7,
            loss_function="RMSE", random_seed=RANDOM_STATE,
            allow_writing_files=False, thread_count=-1, verbose=False
        )
        m.fit(pool)
        return m
    raise ValueError(name)


def predict_l1_one(name: str, model, X, feat_cols: list[str], cat_names: list[str]):
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


def bootstrap_flight_rmse(y_true: np.ndarray, y_pred: np.ndarray, fids: np.ndarray, n_iter: int = 10000, seed: int = 42):
    """Flight-clustered bootstrap CI for RMSE (resample whole flights)."""
    err = y_true - y_pred
    sse = err ** 2
    g = pd.DataFrame({"fid": fids, "sse": sse})
    per_f_sse = g.groupby("fid")["sse"].sum().to_numpy()
    per_f_n = g.groupby("fid").size().to_numpy().astype(float)
    n_f = len(per_f_sse)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_f, size=(n_iter, n_f))
    b_sse = per_f_sse[idx].sum(axis=1)
    b_n = per_f_n[idx].sum(axis=1)
    b_rmse = np.sqrt(b_sse / np.maximum(b_n, 1.0))
    point = float(np.sqrt(sse.mean()))
    lo, hi = float(np.percentile(b_rmse, 2.5)), float(np.percentile(b_rmse, 97.5))
    return point, lo, hi


def run_flow_variant(
    vname: str,
    df: pl.DataFrame,
    pdf: pd.DataFrame,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    dur_all: np.ndarray,
    fids_all: np.ndarray,
) -> dict:
    """Run the full 5f OOF + meta pipeline for one flow variant. Return LGBM meta metrics + test p + L1 test for ref."""
    print(f"\n=== Variant: {vname} ===")
    feat_cols = get_variant_feats(vname, df)
    cat_names = [c for c in CAT_FEATURES if c in feat_cols]
    X = pdf[feat_cols].copy()
    for c in cat_names:
        if c in X.columns:
            X[c] = X[c].astype("category")
    print(f"  feats={len(feat_cols)} (cats={len(cat_names)})")

    # Outer train y/dur
    dur_train = dur_all[outer_train_idx]
    y_flow_train = y_train / np.clip(dur_train, 1.0, None)

    # Grouped folds on *train flight ids*
    train_fids_uniq = np.unique(fids_all[outer_train_idx])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {m: np.zeros(len(outer_train_idx)) for m in ["lgbm", "xgb", "rf", "cat"]}
    t0 = time.perf_counter()
    fold = 0
    for tr_f_idx, va_f_idx in kf.split(train_fids_uniq):
        fold += 1
        tr_f = train_fids_uniq[tr_f_idx]
        va_f = train_fids_uniq[va_f_idx]
        tr_mask = np.isin(fids_all[outer_train_idx], tr_f)
        va_mask = np.isin(fids_all[outer_train_idx], va_f)
        tr_rows = outer_train_idx[tr_mask]
        va_rows = outer_train_idx[va_mask]

        X_tr = X.iloc[tr_rows]
        y_tr_f = y_train[tr_mask]
        dur_tr = dur_train[tr_mask]
        yf_tr = y_tr_f / np.clip(dur_tr, 1.0, None)

        X_va = X.iloc[va_rows]
        dur_va = dur_train[va_mask]

        for mname in ["lgbm", "xgb", "rf", "cat"]:
            model = train_l1_one(mname, X_tr, yf_tr, feat_cols, cat_names)
            p_flow = predict_l1_one(mname, model, X_va, feat_cols, cat_names)
            p_kg = p_flow * np.clip(dur_va, 1.0, None)
            oof_preds[mname][va_mask] = p_kg
    print(f"  OOF done in {time.perf_counter()-t0:.1f}s")

    order = ["lgbm", "xgb", "rf", "cat"]
    P_oof = np.column_stack([oof_preds[m] for m in order])

    # L2 metas on OOF (kg space)
    metas = {}
    metas["Ridge"] = Ridge(alpha=0.5).fit(P_oof, y_train)
    metas["ElasticNet"] = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=3000).fit(P_oof, y_train)
    metas["LGBM"] = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        random_state=RANDOM_STATE, verbose=-1
    ).fit(P_oof, y_train)

    # Retrain L1 on full outer train (flow y)
    print("  Retraining L1 on full train (flow)...")
    t0 = time.perf_counter()
    l1_full = {}
    for mname in order:
        l1_full[mname] = train_l1_one(mname, X.iloc[outer_train_idx], y_flow_train, feat_cols, cat_names)
    print(f"    L1 full in {time.perf_counter()-t0:.1f}s")

    # L1 test -> kg
    dur_test = dur_all[outer_test_idx]
    P_test_flows = [
        predict_l1_one(m, l1_full[m], X.iloc[outer_test_idx], feat_cols, cat_names) for m in order
    ]
    P_test = np.column_stack([pf * np.clip(dur_test, 1.0, None) for pf in P_test_flows])

    # Meta on test
    variant_results = []
    lgbm_p = None
    for mname, meta in metas.items():
        p = meta.predict(P_test)
        met = evaluate(y_test, p)
        variant_results.append({
            "variant": vname,
            "level2": mname,
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
        })
        if mname == "LGBM":
            lgbm_p = p
            lgbm_met = met

    # L1 refs (flow-recovered)
    for i, m in enumerate(order):
        met = evaluate(y_test, P_test[:, i])
        variant_results.append({
            "variant": vname,
            "level2": f"L1-{m.upper()}",
            "mae": met["mae"],
            "rmse": met["rmse"],
            "r2": met["r2"],
        })

    print(f"  LGBM_meta for {vname}: MAE={lgbm_met['mae']:.2f} RMSE={lgbm_met['rmse']:.2f} R2={lgbm_met['r2']:.4f}")
    return {
        "results": variant_results,
        "lgbm_p": lgbm_p,
        "lgbm_met": lgbm_met,
        "y_test": y_test,
        "fids_test": fids_all[outer_test_idx],
    }


def main() -> None:
    print("=" * 72)
    print("FuelFlow variants (5f OOF LGBM_meta pipeline) vs PRC2025 Winner 200.83")
    print("=" * 72)

    df = load_and_clean(MASS_PARQUET)
    pdf = df.to_pandas()
    fids_all = df["flight_id"].to_numpy()
    dur_all = df["duration_s"].to_numpy()

    outer_train_idx, outer_test_idx, _, _ = flight_level_split(fids_all)
    y_train = df["actual_fuel_kg"].to_numpy()[outer_train_idx]
    y_test = df["actual_fuel_kg"].to_numpy()[outer_test_idx]
    print(f"Outer: train_rows={len(outer_train_idx)} test_rows={len(outer_test_idx)} | flights_train≈{len(np.unique(fids_all[outer_train_idx]))}")

    variants = ["FuelFlow", "FuelFlow+Energy", "FuelFlow+Energy+Mass"]
    all_rows = []
    variant_ps = {}  # for bootstrap + table

    for v in variants:
        out = run_flow_variant(v, df, pdf, outer_train_idx, outer_test_idx, y_train, y_test, dur_all, fids_all)
        all_rows.extend(out["results"])
        variant_ps[v] = {
            "p": out["lgbm_p"],
            "met": out["lgbm_met"],
            "y": out["y_test"],
            "fids": out["fids_test"],
        }

    # Reference rows
    all_rows.append({
        "variant": "E+W+phys Ensemble (LGBM_meta 5f OOF)",
        "level2": "LGBM_meta",
        "mae": REFERENCE_2029["mae"],
        "rmse": REFERENCE_2029["rmse"],
        "r2": REFERENCE_2029["r2"],
    })
    all_rows.append({
        "variant": "PRC2025 Winner",
        "level2": "Reported",
        "mae": np.nan,
        "rmse": WINNER_RMSE,
        "r2": np.nan,
    })

    table = pl.DataFrame(all_rows)
    table.write_csv(OUT / "table_flow_vs_prc_raw.csv")
    print(f"\nSaved raw table -> {OUT / 'table_flow_vs_prc_raw.csv'}")

    # Now bootstrap CIs only for the 3 flow LGBM_metas
    print("\n=== Bootstrap flight-clustered 95% CI (RMSE) for the 3 variants ===")
    ci_rows = []
    for v in variants:
        info = variant_ps[v]
        pt, lo, hi = bootstrap_flight_rmse(info["y"], info["p"], info["fids"], n_iter=10000, seed=42)
        # point from met should match pt
        print(f"{v}: RMSE={pt:.2f}  95%CI=[{lo:.2f}, {hi:.2f}]")
        ci_rows.append({
            "variant": v,
            "mae": info["met"]["mae"],
            "rmse": pt,
            "r2": info["met"]["r2"],
            "rmse_ci_lo": lo,
            "rmse_ci_hi": hi,
        })

    # Build final comparison table
    final_rows = []
    for r in ci_rows:
        cls = classify_vs_winner(r["rmse"], r["rmse_ci_lo"], r["rmse_ci_hi"])
        final_rows.append({
            "variant": r["variant"],
            "mae": round(r["mae"], 2),
            "rmse": round(r["rmse"], 2),
            "r2": round(r["r2"], 4),
            "rmse_95ci_lo": round(r["rmse_ci_lo"], 2),
            "rmse_95ci_hi": round(r["rmse_ci_hi"], 2),
            "vs_prc_winner_200.83": cls,
        })
    # refs
    final_rows.append({
        "variant": "E+W+phys Ensemble (LGBM_meta 5f OOF)",
        "mae": 84.3,
        "rmse": 202.9,
        "r2": 0.9481,
        "rmse_95ci_lo": np.nan,
        "rmse_95ci_hi": np.nan,
        "vs_prc_winner_200.83": "Worse (historical)",
    })
    final_rows.append({
        "variant": "PRC2025 Winner (reported)",
        "mae": np.nan,
        "rmse": 200.83,
        "r2": np.nan,
        "rmse_95ci_lo": np.nan,
        "rmse_95ci_hi": np.nan,
        "vs_prc_winner_200.83": "Reference",
    })

    final_table = pl.DataFrame(final_rows)
    final_table.write_csv(OUT / "table_flow_vs_prc.csv")
    print(f"Saved {OUT / 'table_flow_vs_prc.csv'}")
    print(final_table)

    # Figure
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [r["variant"] for r in final_rows if "PRC2025" not in r["variant"] and "E+W+phys" not in r["variant"]]
    rmses = [r["rmse"] for r in final_rows if "PRC2025" not in r["variant"] and "E+W+phys" not in r["variant"]]
    los = [r["rmse_95ci_lo"] for r in final_rows if "PRC2025" not in r["variant"] and "E+W+phys" not in r["variant"]]
    his = [r["rmse_95ci_hi"] for r in final_rows if "PRC2025" not in r["variant"] and "E+W+phys" not in r["variant"]]

    # our 3
    y_pos = np.arange(len(labels))
    colors = ["#2ecc71", "#3498db", "#9b59b6"]
    for i, (lab, rm, l, h) in enumerate(zip(labels, rmses, los, his)):
        ax.barh(i, rm, color=colors[i], alpha=0.85, label=lab if i==0 else None)
        ax.errorbar(rm, i, xerr=[[rm-l], [h-rm]], fmt='none', color='black', capsize=4, lw=1.5)

    # refs as vlines
    ax.axvline(202.9, color="#e67e22", ls="--", lw=2, label="202.9 (E+W+phys 5f LGBM_meta)")
    ax.axvline(200.83, color="#c0392b", ls="-", lw=2.5, label="PRC2025 Winner 200.83")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RMSE (kg) on strict held-out flights (5f GroupKFold OOF + LGBM meta)")
    ax.set_title("FuelFlow variants vs PRC2025 Winner (bootstrap 95% CI on RMSE)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_prc_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_prc_comparison.png'}")

    print("\n" + "=" * 72)
    print("Done. See table_flow_vs_prc.csv and fig_prc_comparison.png")
    print("=" * 72)


def classify_vs_winner(rmse: float, lo: float, hi: float, winner: float = 200.83) -> str:
    if hi < winner:
        return "Better"
    elif lo > winner:
        return "Worse"
    else:
        return "Statistically indistinguishable"


if __name__ == "__main__":
    main()
