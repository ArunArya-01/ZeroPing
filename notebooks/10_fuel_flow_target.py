"""
AeroTwin V4 — Fuel Flow Target (instead of fuel kg) + Fuel Flow Ablation.

Task 2: Predict fuel_flow_kgps = actual_fuel_kg / duration_s
         Recover fuel_pred = flow * duration_s
         Train LGBM/XGB/CatBoost
         Compare direct fuel target vs flow target (recovered)
         Outputs: fig_fuel_vs_flow.png table_fuel_flow.csv

Task 5: Fuel Flow Ablation variants:
         - Direct fuel (Energy+Weather)
         - Fuel flow (Energy+Weather)
         - Fuel flow + Mass
         - Fuel flow + Energy
         - Fuel flow + Energy + Weather
         With bootstrap sig + 95% CI vs direct baseline.

Run:
    python notebooks/10_fuel_flow_target.py

Uses featured_dataset_mass.parquet (has E/W/Mass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (
    BASE_NUMERIC,
    CATEGORICAL,
    MASS_FEATURES,
    N_BOOTSTRAP,
    evaluate,
    flight_level_split,
    load_and_clean,
    plot_bootstrap_hist,
    project_root,
    significance_test,
    train_predict,
)
from physics.feature_engineering import ENERGY_FEATURES
from physics.weather_features import WEATHER_FEATURES

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["xgb", "lgbm", "cat"]
MODEL_LABELS = {"xgb": "XGB", "lgbm": "LGBM", "cat": "CatBoost"}

MASS_PARQUET = project_root() / "featured_dataset_mass.parquet"


def avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feats(df: pl.DataFrame, extra: list[str], physics: bool = True) -> list[str]:
    cols = list(BASE_NUMERIC) + avail(extra, df)
    if physics:
        cols.append("physics_fuel_kg")
    cols += CATEGORICAL
    return list(dict.fromkeys(cols))


def run_direct_or_flow(
    model_key: str,
    feature_cols: list[str],
    pdf,
    train_idx,
    test_idx,
    y_train_actual: np.ndarray,
    y_test_actual: np.ndarray,
    dur_train: np.ndarray,
    dur_test: np.ndarray,
    physics_test: np.ndarray | None = None,
    use_flow_target: bool = False,
    residual: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train on actual or on flow=actual/dur; always return recovered fuel kg preds + metrics on fuel kg."""
    X_tr = pdf[feature_cols].iloc[train_idx]
    X_te = pdf[feature_cols].iloc[test_idx]

    if use_flow_target:
        y_tr = (y_train_actual / np.clip(dur_train, 1.0, None)).astype(np.float64)
    else:
        y_tr = y_train_actual.astype(np.float64) if not residual else (pdf["residual_kg"].to_numpy()[train_idx])

    pred = train_predict(model_key, feature_cols, X_tr, X_te, y_tr, residual_mode=residual, physics_test=physics_test)

    if use_flow_target:
        fuel_pred = pred * np.clip(dur_test, 1.0, None)
    else:
        fuel_pred = pred

    if residual and not use_flow_target:
        # if residual on fuel, already handled inside train_predict for non-flow
        pass

    mets = evaluate(y_test_actual, fuel_pred)
    return fuel_pred, mets


def metrics_rows_direct_flow(all_res: dict) -> list[dict]:
    rows = []
    for approach, by_m in all_res.items():
        for mk, m in by_m.items():
            rows.append({
                "approach": approach,
                "model": MODEL_LABELS.get(mk, mk),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
            })
    return rows


def main() -> None:
    print("=" * 70)
    print("AeroTwin V4 — Fuel Flow Target + Flow Ablation (Tasks 2+5)")
    print("=" * 70)

    df = load_and_clean(MASS_PARQUET)
    print(f"Using mass-enriched: {len(df):,} intervals / {df['flight_id'].n_unique():,} flights")

    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    massf = avail(MASS_FEATURES, df)
    ew = energy + weather
    print(f"  E:{len(energy)} W:{len(weather)} M:{len(massf)}")

    pdf = df.to_pandas()
    fids = pdf["flight_id"].to_numpy()
    train_idx, test_idx, _, _ = flight_level_split(fids)
    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    dur_train = pdf["duration_s"].to_numpy()[train_idx]
    dur_test = pdf["duration_s"].to_numpy()[test_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]
    test_fids = pdf["flight_id"].to_numpy()[test_idx]

    # Baseline direct Energy+Weather (fuel kg target)
    ew_feats = feats(df, ew, physics=True)
    direct_preds = {}
    direct_mets = {}
    for mk in MODELS:
        p, m = run_direct_or_flow(
            mk, ew_feats, pdf, train_idx, test_idx, y_train, y_test, dur_train, dur_test, physics_test, use_flow_target=False
        )
        direct_preds[mk] = p
        direct_mets[mk] = m
        direct_mets[mk]["approach"] = "Direct (E+W)"
        direct_mets[mk]["model"] = MODEL_LABELS[mk]
    print(f"Direct E+W XGB fuel-kg MAE: {direct_mets['xgb']['mae']:.2f}")

    # Flow target on same E+W features
    flow_preds = {}
    flow_mets = {}
    for mk in MODELS:
        p, m = run_direct_or_flow(
            mk, ew_feats, pdf, train_idx, test_idx, y_train, y_test, dur_train, dur_test, physics_test, use_flow_target=True
        )
        flow_preds[mk] = p
        flow_mets[mk] = m
        flow_mets[mk]["approach"] = "Flow (E+W)"
        flow_mets[mk]["model"] = MODEL_LABELS[mk]
    print(f"Flow target E+W XGB recovered-fuel MAE: {flow_mets['xgb']['mae']:.2f}")

    # === Task 2 table + fig: direct vs flow (focus XGB + LGBM + Cat) ===
    task2_res = {
        "Direct fuel (E+W)": direct_mets,
        "Fuel flow (E+W)": flow_mets,
    }
    t2_df = pl.DataFrame(metrics_rows_direct_flow(task2_res)).sort(["approach", "mae"])
    t2_df.write_csv(OUT / "table_fuel_flow.csv")
    print("Saved table_fuel_flow.csv")

    # fig: grouped bars for mae/rmse/r2 , two approaches
    pdf_t2 = t2_df.to_pandas()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, met in zip(axes, ["mae", "rmse", "r2"]):
        sns.barplot(data=pdf_t2, x="model", y=met, hue="approach", ax=ax)
        ax.set_title(met.upper())
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Task 2: Direct Fuel Target vs Fuel-Flow Target (recovered) — Energy+Weather features", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fuel_vs_flow.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_fuel_vs_flow.png")

    # Bootstrap compare flow vs direct on XGB (representative)
    direct_err = np.abs(y_test - direct_preds["xgb"])
    flow_err = np.abs(y_test - flow_preds["xgb"])
    sig_flow = significance_test(flow_err, direct_err, test_fids, "Fuel-Flow (E+W)", "Direct (E+W)")
    boot_flow = sig_flow.pop("bootstrap_dist")
    pl.DataFrame([sig_flow]).write_csv(OUT / "table_significance_fuel_flow.csv")
    plot_bootstrap_hist(boot_flow, "Task2: Fuel-Flow vs Direct (E+W) XGB — flight bootstrap", OUT / "fig_fuel_flow_bootstrap.png")
    print(
        f"Flow vs Direct: ΔMAE={sig_flow['delta_mae']:+.2f} CI=[{sig_flow['ci_lower']:+.2f},{sig_flow['ci_upper']:+.2f}] "
        f"p={sig_flow['bootstrap_p']:.4f} → {sig_flow['interpretation']}"
    )

    # === TASK 5: Fuel Flow Ablation ===
    print("\n" + "=" * 70)
    print("TASK 5 — Fuel Flow Ablation")
    print("=" * 70)

    # All using FLOW target + recover, different feature groups
    # 1. Direct fuel baseline (E+W)  -- keep for ref
    # Flow variants:
    # F1: Flow only (base numeric + cats?) 
    # F2: Flow + Mass
    # F3: Flow + Energy
    # F4: Flow + Energy + Weather
    # Also F5: Flow + E + W + Mass ? but focus listed

    base_no_phys = list(BASE_NUMERIC) + CATEGORICAL
    flow_approaches = {
        "Direct (E+W) ref": (ew_feats, False, False),  # feat, use_flow, residual
        "Flow + base": (base_no_phys, True, False),
        "Flow + Mass": (list(BASE_NUMERIC) + massf + CATEGORICAL, True, False),
        "Flow + Energy": (feats(df, energy, physics=False), True, False),
        "Flow + Energy+Weather": (ew_feats, True, False),
        "Flow + Energy+Weather+Mass": (feats(df, ew + massf, physics=True), True, False),
    }

    flow_all: dict = {}
    flow_sig_rows: list = []
    flow_boots: dict = {}
    flow_test_errs = {}  # for sig later

    # ref direct err
    flow_test_errs["Direct (E+W) ref"] = direct_err

    for name, (fcols, use_flow, res) in flow_approaches.items():
        print(f"  {name} (flow={use_flow}) ...", flush=True)
        mets_per = {}
        preds_per = {}
        for mk in MODELS:
            p, m = run_direct_or_flow(
                mk, fcols, pdf, train_idx, test_idx, y_train, y_test,
                dur_train, dur_test, physics_test if res else None,
                use_flow_target=use_flow, residual=res
            )
            m["approach"] = name
            m["model"] = MODEL_LABELS[mk]
            mets_per[mk] = m
            preds_per[mk] = p
        flow_all[name] = mets_per

        # XGB for sig compare to direct
        xgb_p = preds_per["xgb"]
        err = np.abs(y_test - xgb_p)
        flow_test_errs[name] = err

        if name != "Direct (E+W) ref":
            sig = significance_test(err, direct_err, test_fids, name, "Direct (E+W) ref")
            flow_boots[name] = sig.pop("bootstrap_dist")
            flow_sig_rows.append(sig)
            xgb_mae = mets_per["xgb"]["mae"]
            print(
                f"    XGB recov MAE={xgb_mae:.2f}  Δ={sig['delta_mae']:+.2f}  "
                f"CI=[{sig['ci_lower']:+.2f},{sig['ci_upper']:+.2f}] p={sig['bootstrap_p']:.4f} → {sig['interpretation']}"
            )

    # Save ablation table (flow variants)
    fa_df = pl.DataFrame(metrics_rows_direct_flow(flow_all)).sort(["approach", "mae"])
    fa_df.write_csv(OUT / "table_fuel_flow_ablation.csv")

    # fig for flow ablation
    pdf_fa = fa_df.to_pandas()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, met in zip(axes, ["mae", "rmse", "r2"]):
        sns.barplot(data=pdf_fa, x="model", y=met, hue="approach", ax=ax)
        ax.set_title(met.upper())
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Task 5: Fuel-Flow Target Ablation (various feature groups)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_fuel_flow_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved table_fuel_flow_ablation.csv + fig_fuel_flow_ablation.png")

    # combined sig for flow ablation
    if flow_sig_rows:
        pl.DataFrame(flow_sig_rows).write_csv(OUT / "table_significance_fuel_flow_ablation.csv")
        # one combined bootstrap plot? or per
        for lbl, dist in flow_boots.items():
            plot_bootstrap_hist(
                dist,
                f"Flow Ablation: {lbl} vs Direct (E+W)",
                OUT / f"fig_fuel_flow_ablation_{lbl[:20].replace(' ', '_').replace(':', '')}_bootstrap.png",
            )

    # Update partial v4 lb with best here (XGB flow E+W)
    best_flow_xgb = flow_all["Flow + Energy+Weather"]["xgb"]["mae"]
    print(f"\nBest flow (E+W) XGB recovered MAE: {best_flow_xgb:.2f}")

    # partial lb append
    lb_add = [
        {"experiment": "fuel_flow", "approach": "Direct (E+W)", "model": "XGB", "mae": direct_mets["xgb"]["mae"]},
        {"experiment": "fuel_flow", "approach": "Flow + Energy+Weather", "model": "XGB", "mae": best_flow_xgb},
    ]
    pl.DataFrame(lb_add).write_csv(OUT / "leaderboard_v4_partial_flow.csv")

    print("\n" + "=" * 70)
    print("V4 FUEL FLOW TASKS (2+5) COMPLETE")
    print("Deliverables: fig_fuel_vs_flow.png table_fuel_flow.csv + ablation variants")
    print("=" * 70)


if __name__ == "__main__":
    main()
