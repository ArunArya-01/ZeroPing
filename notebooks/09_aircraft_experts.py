
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (
    BASE_NUMERIC,
    flight_level_split,
    load_and_clean,
    project_root,
)
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


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def get_feature_set(df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    weather = [c for c in WEATHER_FEATURES if c in df.columns]
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    return list(dict.fromkeys(cols))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


GROUPS = {
    "A320": {"A320"},
    "A20N": {"A20N"},
    "A359": {"A359"},
    "B738": {"B738"},
    "B788": {"B788"},
    "Other": None,  # everything else
}


def assign_group(ac: str) -> str:
    for g, types in GROUPS.items():
        if types is None:
            continue
        if ac in types:
            return g
    return "Other"


def train_one_cat(X: "pd.DataFrame | pl.DataFrame", y: np.ndarray, feat_cols: list[str], cat_names: list[str]) -> CatBoostRegressor:
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
    pool = Pool(Xp, y, cat_features=cat_idx, feature_names=feat_cols)
    m = CatBoostRegressor(
        iterations=1200, learning_rate=0.03, depth=7,
        loss_function="RMSE", random_seed=RANDOM_STATE,
        allow_writing_files=False, thread_count=-1, verbose=False,
    )
    m.fit(pool)
    return m


def train_one_lgb(X: "pd.DataFrame | pl.DataFrame", y: np.ndarray) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
    )
    m.fit(X, y)
    return m


def train_gating_model(X, groups, feat_cols, cat_names):
    from catboost import CatBoostClassifier, Pool
    from sklearn.preprocessing import LabelEncoder

    y_group = np.asarray(groups)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_group)

    Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    pool = Pool(Xp, y_enc, cat_features=cat_idx, feature_names=feat_cols)

    m = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )
    m.fit(pool)
    return m, le


def predict_moe_soft(gating_model, le, X, specs_cat, specs_lgb, global_cat, global_lgb, use_cat=True):
    Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
    probs = gating_model.predict_proba(Xp)
    ordered_groups = le.inverse_transform(np.arange(probs.shape[1]))

    out = np.zeros(len(X), dtype=np.float64)
    for gi, g in enumerate(ordered_groups):
        if g in specs_cat and use_cat:
            preds = specs_cat[g].predict(Xp)
        elif g in specs_lgb:
            preds = specs_lgb[g].predict(Xp)
        else:
            preds = global_cat.predict(Xp) if use_cat else global_lgb.predict(Xp)
        out += probs[:, gi] * preds
    return out


def main() -> None:
    print("=" * 72)
    print("AIRCRAFT EXPERTS (specialist per airframe family) vs GLOBAL")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()

    train_idx, test_idx, _, _ = flight_level_split(fids)
    y_train = df["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = df["actual_fuel_kg"].to_numpy()[test_idx]

    feat_cols = get_feature_set(df)
    cat_names = [c for c in CAT_FEATURES if c in df.columns]
    print(f"Feature set size: {len(feat_cols)}")

    # Assign groups
    pdf = pdf.copy()
    pdf["ac_group"] = pdf["aircraft_type"].map(assign_group)
    train_groups = pdf["ac_group"].iloc[train_idx].values
    test_groups = pdf["ac_group"].iloc[test_idx].values

    print("Train group counts:")
    for g in GROUPS:
        n = (train_groups == g).sum()
        print(f"  {g}: {n} intervals")

    X_all = pdf[feat_cols].copy()
    for c in cat_names:
        if c in X_all.columns:
            X_all[c] = X_all[c].astype("category")

    # Train GLOBAL models (Cat + LGB for ref)
    print("\nTraining GLOBAL models on all train ...")
    t0 = time.perf_counter()
    global_cat = train_one_cat(X_all.iloc[train_idx], y_train, feat_cols, cat_names)
    global_lgb = train_one_lgb(X_all.iloc[train_idx], y_train)
    print(f"  Globals trained in {time.perf_counter() - t0:.1f}s")

    test_df = X_all.iloc[test_idx]
    p_global_cat_test = global_cat.predict(_to_pandas(test_df))
    p_global_lgb_test = global_lgb.predict(test_df)

    # Train SPECIALISTS per group
    specialists_cat: dict[str, CatBoostRegressor] = {}
    specialists_lgb: dict[str, lgb.LGBMRegressor] = {}

    print("\nTraining SPECIALIST models per group ...")
    t0 = time.perf_counter()
    for g in GROUPS:
        mask_tr = train_groups == g
        n = mask_tr.sum()
        if n < 50:
            print(f"  {g}: too few ({n}), skip specialist")
            continue
        Xg = X_all.iloc[train_idx][mask_tr]
        yg = y_train[mask_tr]
        print(f"  {g}: training on {n} ...", end=" ", flush=True)
        specialists_cat[g] = train_one_cat(Xg, yg, feat_cols, cat_names)
        specialists_lgb[g] = train_one_lgb(Xg, yg)
        print("ok")
    print(f"  Specialists done in {time.perf_counter() - t0:.1f}s")

    # Inference routing for experts
    def predict_expert(groups: np.ndarray, X: "pd.DataFrame | pl.DataFrame", specs_cat: dict, specs_lgb: dict, use_cat: bool = True):
        out = np.zeros(len(groups), dtype=np.float64)
        Xp = X.to_pandas() if hasattr(X, "to_pandas") else X
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            if len(idx) == 0:
                continue
            if g in specs_cat and use_cat:
                out[idx] = specs_cat[g].predict(Xp.iloc[idx])
            elif g in specs_lgb:
                # fallback lgb if no cat or not use_cat
                out[idx] = specs_lgb[g].predict(Xp.iloc[idx])
            else:
                # fallback global
                out[idx] = global_cat.predict(Xp.iloc[idx]) if use_cat else global_lgb.predict(Xp.iloc[idx])
        return out

    p_expert_cat = predict_expert(test_groups, X_all.iloc[test_idx], specialists_cat, specialists_lgb, use_cat=True)
    p_expert_lgb = predict_expert(test_groups, X_all.iloc[test_idx], specialists_cat, specialists_lgb, use_cat=False)

    # MoE soft gating
    print("\nTraining MoE soft gating model ...")
    t0 = time.perf_counter()
    gating_model, le = train_gating_model(X_all.iloc[train_idx], train_groups, feat_cols, cat_names)
    print(f"  Gating trained in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    p_moe_cat = predict_moe_soft(gating_model, le, X_all.iloc[test_idx], specialists_cat, specialists_lgb, global_cat, global_lgb, use_cat=True)
    p_moe_lgb = predict_moe_soft(gating_model, le, X_all.iloc[test_idx], specialists_cat, specialists_lgb, global_cat, global_lgb, use_cat=False)
    print(f"  MoE predicted in {time.perf_counter() - t0:.1f}s")

    m_moe_cat = evaluate(y_test, p_moe_cat)
    m_moe_lgb = evaluate(y_test, p_moe_lgb)

    # Metrics
    m_global_cat = evaluate(y_test, p_global_cat_test)
    m_global_lgb = evaluate(y_test, p_global_lgb_test)
    m_expert_cat = evaluate(y_test, p_expert_cat)
    m_expert_lgb = evaluate(y_test, p_expert_lgb)

    rows = [
        {"approach": "Global", "model": "CatBoost", "mae": m_global_cat["mae"], "rmse": m_global_cat["rmse"], "r2": m_global_cat["r2"]},
        {"approach": "Global", "model": "LightGBM", "mae": m_global_lgb["mae"], "rmse": m_global_lgb["rmse"], "r2": m_global_lgb["r2"]},
        {"approach": "Experts", "model": "CatBoost", "mae": m_expert_cat["mae"], "rmse": m_expert_cat["rmse"], "r2": m_expert_cat["r2"]},
        {"approach": "Experts", "model": "LightGBM", "mae": m_expert_lgb["mae"], "rmse": m_expert_lgb["rmse"], "r2": m_expert_lgb["r2"]},
        {"approach": "MoE", "model": "CatBoost", "mae": m_moe_cat["mae"], "rmse": m_moe_cat["rmse"], "r2": m_moe_cat["r2"]},
        {"approach": "MoE", "model": "LightGBM", "mae": m_moe_lgb["mae"], "rmse": m_moe_lgb["rmse"], "r2": m_moe_lgb["r2"]},
    ]
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "table_aircraft_experts.csv")
    print(f"\nSaved {OUT / 'table_aircraft_experts.csv'}")

    # Per group error breakdown for experts vs global (using cat versions)
    group_rows = []
    for g in GROUPS:
        gmask = test_groups == g
        if gmask.sum() < 5:
            continue
        yt = y_test[gmask]
        # global cat err
        eg = np.abs(yt - p_global_cat_test[gmask])
        ee = np.abs(yt - p_expert_cat[gmask])
        em = np.abs(yt - p_moe_cat[gmask])
        group_rows.append({
            "group": g,
            "n_intervals": int(gmask.sum()),
            "global_mae": float(eg.mean()),
            "expert_mae": float(ee.mean()),
            "moe_mae": float(em.mean()),
            "global_rmse": float(np.sqrt((eg**2).mean())),
            "expert_rmse": float(np.sqrt((ee**2).mean())),
            "moe_rmse": float(np.sqrt((em**2).mean())),
            "delta_rmse": float(np.sqrt((eg**2).mean()) - np.sqrt((ee**2).mean())),
        })
    gtable = pl.DataFrame(group_rows)
    gtable.write_csv(OUT / "table_aircraft_experts_pergroup.csv")
    print(f"Saved per-group breakdown")

    # Plot: errors by group expert vs global
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gp = gtable.to_pandas()
    sns.barplot(data=gp, x="group", y="global_rmse", ax=axes[0], color="#e74c3c", label="Global Cat")
    sns.barplot(data=gp, x="group", y="expert_rmse", ax=axes[0], color="#27ae60", label="Expert Cat", alpha=0.8)
    sns.barplot(data=gp, x="group", y="moe_rmse", ax=axes[0], color="#3498db", label="MoE Cat", alpha=0.8)
    axes[0].set_title("RMSE by Aircraft Group (held-out)")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=gp, x="group", y="global_mae", ax=axes[1], color="#e74c3c", label="Global Cat")
    sns.barplot(data=gp, x="group", y="expert_mae", ax=axes[1], color="#27ae60", label="Expert Cat", alpha=0.8)
    sns.barplot(data=gp, x="group", y="moe_mae", ax=axes[1], color="#3498db", label="MoE Cat", alpha=0.8)
    axes[1].set_title("MAE by Aircraft Group (held-out)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Aircraft Expert Models vs Global (Energy+Weather+Physics)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_aircraft_errors.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_aircraft_errors.png'}")

    print("\n" + "=" * 72)
    print("AIRCRAFT EXPERTS vs GLOBAL (held-out)")
    for r in rows:
        print(f"  {r['approach']:7s} {r['model']:8s} MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} R2={r['r2']:.4f}")
    # improvement?
    delta = m_global_cat["rmse"] - m_expert_cat["rmse"]
    delta_moe = m_global_cat["rmse"] - m_moe_cat["rmse"]
    print(f"\nExpert Cat vs Global Cat: ΔRMSE={delta:+.2f} kg")
    print(f"MoE Cat vs Global Cat:   ΔRMSE={delta_moe:+.2f} kg")
    print("=" * 72)


if __name__ == "__main__":
    main()
