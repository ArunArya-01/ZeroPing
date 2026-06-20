"""
AeroTwin v3 — Weather, combined hybrids, MLP correction, conditional BADA path.

E5: Weather features
E6: Energy + Weather + OpenAP hybrid
E7: MLP residual correction
E8: Wind-adjusted physics (BADA-style) if E5–E7 lack substantial gains

Run:
    python physics/enrich_v3_features.py
    python notebooks/09_physics_features_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (
    BASE_NUMERIC,
    CATEGORICAL,
    N_BOOTSTRAP,
    evaluate,
    flight_level_split,
    load_and_clean,
    plot_bootstrap_hist,
    plot_comparison_bars,
    project_root,
    significance_test,
    train_predict,
)
from physics.feature_engineering import ENERGY_FEATURES, enrich_from_columns
from physics.mlp_residual import train_mlp_residual
from physics.weather_features import WEATHER_FEATURES, enrich_weather_from_columns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)
PARQUET = project_root() / "featured_dataset.parquet"

MODELS = ["rf", "xgb", "lgbm"]
MODEL_LABELS = {"rf": "RF", "xgb": "XGB", "lgbm": "LGBM"}
SUBSTANTIAL_MAE_KG = 1.5  # threshold to skip E8


def ensure_v3(df: pl.DataFrame) -> pl.DataFrame:
    changed = False
    if "mean_specific_energy_jpkg" not in df.columns:
        df = enrich_from_columns(df)
        changed = True
    if "headwind_mps" not in df.columns:
        df = enrich_weather_from_columns(df)
        changed = True
    if "physics_wind_adj_kg" not in df.columns:
        hw = pl.col("headwind_mps").fill_null(0.0)
        tas = pl.col("tas_used").fill_null(pl.col("mean_groundspeed")).fill_null(200.0)
        df = df.with_columns(
            (pl.col("physics_fuel_kg") * (1.0 + pl.max_horizontal(hw, pl.lit(0.0)) / tas * 0.12))
            .alias("physics_wind_adj_kg")
        )
        changed = True
    if changed:
        df.write_parquet(PARQUET)
    return df


def avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feats(df: pl.DataFrame, extra: list[str], physics: bool = True) -> list[str]:
    cols = list(BASE_NUMERIC) + avail(extra, df)
    if physics:
        cols.append("physics_fuel_kg")
    cols += CATEGORICAL
    return list(dict.fromkeys(cols))


def run_tree_models(
    approach: str,
    feature_cols: list[str],
    pdf,
    train_idx,
    test_idx,
    y_train,
    y_test,
    residual: bool = False,
    physics_test=None,
) -> tuple[dict, dict]:
    metrics, preds = {}, {}
    X_tr = pdf[feature_cols].iloc[train_idx]
    X_te = pdf[feature_cols].iloc[test_idx]
    y_tr = pdf["residual_kg"].to_numpy()[train_idx] if residual else y_train
    for mk in MODELS:
        pred = train_predict(mk, feature_cols, X_tr, X_te, y_tr, residual, None, physics_test)
        metrics[mk] = evaluate(y_test, pred)
        metrics[mk]["approach"] = approach
        metrics[mk]["model"] = MODEL_LABELS[mk]
        preds[mk] = pred
    return metrics, preds


def metrics_rows(all_metrics: dict) -> list[dict]:
    rows = []
    for approach, by_m in all_metrics.items():
        for mk, m in by_m.items():
            rows.append({"approach": approach, "model": m["model"], "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
    return rows


def experiment(
    title: str,
    approaches: dict[str, tuple[list[str], bool]],
    pdf,
    train_idx,
    test_idx,
    y_train,
    y_test,
    physics_test,
    baseline_err,
    baseline_label: str,
    table_path: Path,
    sig_path: Path,
    fig_path: Path,
    extra_train_fn=None,
) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    test_fids = pdf["flight_id"].to_numpy()[test_idx]
    all_metrics: dict = {}
    sig_rows: list[dict] = []
    boot_dists: dict[str, np.ndarray] = {}
    best_xgb_mae = float("inf")

    for name, (fcols, residual) in approaches.items():
        print(f"  {name} ...", flush=True)
        if extra_train_fn and name in extra_train_fn:
            preds = extra_train_fn[name](fcols)
            mets = {mk: evaluate(y_test, preds[mk]) for mk in MODELS}
            for mk in MODELS:
                mets[mk]["approach"] = name
                mets[mk]["model"] = MODEL_LABELS[mk]
        else:
            mets, preds = run_tree_models(
                name, fcols, pdf, train_idx, test_idx, y_train, y_test, residual, physics_test
            )
        all_metrics[name] = mets
        xgb_mae = mets["xgb"]["mae"]
        best_xgb_mae = min(best_xgb_mae, xgb_mae)

        if name != baseline_label:
            err = np.abs(y_test - preds["xgb"])
            sig = significance_test(err, baseline_err, test_fids, name, baseline_label)
            boot_dists[name] = sig.pop("bootstrap_dist")
            sig_rows.append(sig)
            print(
                f"    XGB MAE={xgb_mae:.2f}  Δ={sig['delta_mae']:+.2f}  "
                f"CI=[{sig['ci_lower']:+.2f},{sig['ci_upper']:+.2f}]  → {sig['interpretation']}"
            )
        else:
            print(f"    XGB MAE={xgb_mae:.2f}  (baseline)")

    res_df = pl.DataFrame(metrics_rows(all_metrics)).sort(["approach", "mae"])
    res_df.write_csv(table_path)
    sig_df = pl.DataFrame(sig_rows) if sig_rows else pl.DataFrame()
    if not sig_df.is_empty():
        sig_df.write_csv(sig_path)
    plot_comparison_bars(res_df, title, fig_path)

    if boot_dists:
        fig, axes = plt.subplots(1, len(boot_dists), figsize=(5 * len(boot_dists), 4))
        if len(boot_dists) == 1:
            axes = [axes]
        for ax, (lbl, dist) in zip(axes, boot_dists.items()):
            lo, hi = np.percentile(dist, [2.5, 97.5])
            ax.hist(dist, bins=50, color="#3498db", alpha=0.85, density=True)
            ax.axvline(0, color="k", lw=1.2)
            ax.axvspan(lo, hi, alpha=0.2, color="green")
            ax.set_title(lbl)
            ax.set_xlabel("ΔMAE [kg]")
        fig.suptitle(f"{title} — flight-clustered bootstrap ({N_BOOTSTRAP:,} resamples)", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_path.with_name(fig_path.stem + "_bootstrap.png"), bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {table_path}, {fig_path}")
    return res_df, sig_df, best_xgb_mae


def plot_leaderboard(leaderboard: pl.DataFrame, path: Path) -> None:
    top = leaderboard.sort("mae").head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#27ae60" if a != "OpenAP Hybrid" else "#2980b9" for a in top["approach"]]
    ax.barh(top["label"].to_list()[::-1], top["mae"].to_list()[::-1], color=colors[::-1], alpha=0.9)
    ax.set_xlabel("MAE (kg) — XGB on held-out flights")
    ax.set_title("AeroTwin v3 Leaderboard (lower is better)")
    ax.axvline(
        float(leaderboard.filter(pl.col("approach") == "OpenAP Hybrid")["mae"][0]),
        color="#c0392b", ls="--", lw=1.5, label="OpenAP Hybrid baseline",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("AeroTwin v3 — Weather, Combined Hybrid, MLP, Conditional E8")
    print("=" * 70)

    df = ensure_v3(load_and_clean(PARQUET))
    pdf = df.to_pandas()
    fids = pdf["flight_id"].to_numpy()
    train_idx, test_idx, _, _ = flight_level_split(fids)
    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]

    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    ew = energy + weather

    print(f"Intervals: {len(df):,}  |  Energy feats: {len(energy)}  |  Weather feats: {len(weather)}")

    # Baseline
    base_feats = feats(df, [], physics=True)
    _, base_preds = run_tree_models(
        "OpenAP Hybrid", base_feats, pdf, train_idx, test_idx, y_train, y_test
    )
    baseline_err = np.abs(y_test - base_preds["xgb"])
    baseline_mae = float(baseline_err.mean())
    print(f"OpenAP Hybrid (XGB) baseline MAE: {baseline_mae:.2f} kg")

    # E5
    e5_approaches = {
        "OpenAP Hybrid": (base_feats, False),
        "Weather Hybrid": (feats(df, weather, True), False),
        "No Physics": (feats(df, [], False), False),
    }
    e5_res, e5_sig, e5_best = experiment(
        "E5: Weather Features",
        e5_approaches, pdf, train_idx, test_idx, y_train, y_test, physics_test,
        baseline_err, "OpenAP Hybrid",
        OUT / "table_v3_e5_weather_results.csv",
        OUT / "table_significance_v3_e5.csv",
        OUT / "fig_v3_e5_weather.png",
    )

    # E6
    e6_approaches = {
        "OpenAP Hybrid": (base_feats, False),
        "Energy Hybrid": (feats(df, energy, True), False),
        "Energy+Weather Hybrid": (feats(df, ew, True), False),
    }
    e6_res, e6_sig, e6_best = experiment(
        "E6: Energy + Weather + OpenAP Hybrid",
        e6_approaches, pdf, train_idx, test_idx, y_train, y_test, physics_test,
        baseline_err, "OpenAP Hybrid",
        OUT / "table_v3_e6_combined_results.csv",
        OUT / "table_significance_v3_e6.csv",
        OUT / "fig_v3_e6_combined.png",
    )

    # E7 MLP
    mlp_feats = feats(df, ew, physics=True)
    X_tr = pdf[mlp_feats].iloc[train_idx]
    X_te = pdf[mlp_feats].iloc[test_idx]
    y_res_tr = pdf["residual_kg"].to_numpy()[train_idx]

    def train_mlp_all(fcols):
        out = {}
        for mk in MODELS:
            if mk == "xgb":
                pred = train_mlp_residual(fcols, X_tr, X_te, y_res_tr, physics_test)
            else:
                # RF/LGBM tree baselines on same feature set for table completeness
                pred = train_predict(mk, fcols, X_tr, X_te, y_res_tr, True, None, physics_test)
            out[mk] = pred
        return out

    print(f"\n{'=' * 70}\nE7: MLP Residual Correction\n{'=' * 70}")
    print("  Training MLP + tree residual models ...", flush=True)
    mlp_preds = train_mlp_all(mlp_feats)
    mlp_mets = {mk: evaluate(y_test, mlp_preds[mk]) for mk in MODELS}
    for mk in MODELS:
        mlp_mets[mk]["approach"] = "MLP Residual" if mk == "xgb" else f"Residual-{MODEL_LABELS[mk]}"
        mlp_mets[mk]["model"] = MODEL_LABELS[mk]

    mlp_err = np.abs(y_test - mlp_preds["xgb"])
    mlp_sig = significance_test(
        mlp_err, baseline_err, pdf["flight_id"].to_numpy()[test_idx],
        "MLP Residual", "OpenAP Hybrid",
    )
    mlp_boot = mlp_sig.pop("bootstrap_dist")
    print(
        f"  XGB MLP Residual MAE={mlp_mets['xgb']['mae']:.2f}  "
        f"Δ={mlp_sig['delta_mae']:+.2f}  CI=[{mlp_sig['ci_lower']:+.2f},{mlp_sig['ci_upper']:+.2f}]  "
        f"→ {mlp_sig['interpretation']}"
    )

    e7_rows = metrics_rows({"MLP Residual": mlp_mets})
    e7_res = pl.DataFrame(e7_rows)
    e7_res.write_csv(OUT / "table_v3_e7_mlp_results.csv")
    pl.DataFrame([mlp_sig]).write_csv(OUT / "table_significance_v3_e7.csv")
    plot_bootstrap_hist(mlp_boot, "E7 MLP Residual vs OpenAP Hybrid", OUT / "fig_v3_e7_mlp_bootstrap.png", "#9b59b6")
    e7_best = mlp_mets["xgb"]["mae"]

    # E8 conditional
    prior_best = min(e5_best, e6_best, e7_best)
    prior_gain = baseline_mae - prior_best
    run_e8 = prior_gain < SUBSTANTIAL_MAE_KG
    e8_res, e8_sig = pl.DataFrame(), pl.DataFrame()

    if run_e8:
        print(f"\nE5–E7 best gain ({prior_gain:.2f} kg) < {SUBSTANTIAL_MAE_KG} kg → running E8 wind-adjusted physics")
        bada_feats = list(dict.fromkeys(BASE_NUMERIC + weather + ["physics_wind_adj_kg"] + CATEGORICAL))
        e8_approaches = {
            "OpenAP Hybrid": (base_feats, False),
            "BADA-Style Wind Adj": (bada_feats, False),
        }
        e8_res, e8_sig, _ = experiment(
            "E8: BADA-Style Wind-Adjusted Physics",
            e8_approaches, pdf, train_idx, test_idx, y_train, y_test, physics_test,
            baseline_err, "OpenAP Hybrid",
            OUT / "table_v3_e8_bada_results.csv",
            OUT / "table_significance_v3_e8.csv",
            OUT / "fig_v3_e8_bada.png",
        )
    else:
        print(f"\nE8 skipped: E5–E7 achieved {prior_gain:.2f} kg gain (≥ {SUBSTANTIAL_MAE_KG} kg threshold)")

    # Leaderboard
    lb_rows = [
        {"experiment": "baseline", "approach": "OpenAP Hybrid", "model": "XGB", "mae": baseline_mae},
    ]
    for exp_name, res in [("E5", e5_res), ("E6", e6_res), ("E7", e7_res), ("E8", e8_res)]:
        if res.is_empty():
            continue
        for row in res.filter(pl.col("model") == "XGB").iter_rows(named=True):
            lb_rows.append({
                "experiment": exp_name,
                "approach": row["approach"],
                "model": "XGB",
                "mae": row["mae"],
            })
    leaderboard = pl.DataFrame(lb_rows).with_columns(
        (pl.col("experiment") + " / " + pl.col("approach")).alias("label")
    )
    leaderboard.write_csv(OUT / "table_v3_leaderboard.csv")
    plot_leaderboard(leaderboard, OUT / "fig_v3_leaderboard.png")

    # Combined significance table
    sig_parts = [s for s in [e5_sig, e6_sig, pl.DataFrame([mlp_sig]), e8_sig] if not s.is_empty()]
    if sig_parts:
        pl.concat(sig_parts).write_csv(OUT / "table_significance_v3_all.csv")

    print("\n" + "=" * 70)
    print("V3 CONCLUSIONS")
    print("=" * 70)
    winner = leaderboard.sort("mae").row(0, named=True)
    print(f"Best overall: {winner['label']}  MAE={winner['mae']:.2f} kg  (baseline {baseline_mae:.2f})")

    sig_winners = pl.concat(sig_parts).filter(pl.col("ci_upper") < 0) if sig_parts else pl.DataFrame()
    if sig_winners.is_empty():
        print("No approach beats OpenAP hybrid with flight-clustered bootstrap significance.")
    else:
        for r in sig_winners.iter_rows(named=True):
            print(f"  ✓ {r['comparison']}: ΔMAE={r['delta_mae']:.2f} kg, CI=[{r['ci_lower']:.2f},{r['ci_upper']:.2f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()