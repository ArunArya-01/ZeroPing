
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (
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
from aerotwin.engine.feature_engineering import ENERGY_FEATURES, OPERATIONAL_FEATURES, enrich_from_columns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)
PARQUET = project_root() / "featured_dataset.parquet"

MODELS = ["rf", "xgb", "lgbm"]
MODEL_LABELS = {"rf": "RF", "xgb": "XGB", "lgbm": "LGBM"}


def ensure_enriched(df: pl.DataFrame) -> pl.DataFrame:
    if "mean_specific_energy_jpkg" not in df.columns:
        print("Enriching dataset with E2/E3 features ...")
        df = enrich_from_columns(df)
        df.write_parquet(PARQUET)
    return df


def available(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_set(name: str, df: pl.DataFrame, extra: list[str], include_physics: bool = True) -> list[str]:
    cols = list(BASE_NUMERIC) + available(extra, df)
    if include_physics:
        cols.append("physics_fuel_kg")
    cols += CATEGORICAL
    return list(dict.fromkeys(cols))


def run_approach(
    approach: str,
    feature_cols: list[str],
    pdf,
    train_idx,
    test_idx,
    y_train,
    y_test,
    physics_train,
    physics_test,
    residual_mode: bool = False,
) -> dict[str, dict[str, float]]:
    """Train all model families for one approach; return test metrics per model."""
    metrics: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    X_train = pdf[feature_cols].iloc[train_idx]
    X_test = pdf[feature_cols].iloc[test_idx]
    y_tr = pdf["residual_kg"].to_numpy()[train_idx] if residual_mode else y_train

    for mk in MODELS:
        pred = train_predict(
            mk, feature_cols, X_train, X_test, y_tr,
            residual_mode=residual_mode,
            physics_test=physics_test if residual_mode else None,
        )
        preds[mk] = pred
        metrics[mk] = evaluate(y_test, pred)
        metrics[mk]["approach"] = approach
        metrics[mk]["model"] = MODEL_LABELS[mk]
    return metrics, preds


def metrics_to_rows(metrics_by_approach: dict) -> list[dict]:
    rows = []
    for approach, by_model in metrics_by_approach.items():
        for mk, m in by_model.items():
            rows.append({"approach": approach, "model": m["model"], **{k: m[k] for k in ("mae", "rmse", "r2")}})
    return rows


def best_xgb_baseline_preds(pdf, train_idx, test_idx, y_train, y_test, physics_test):
    """Current OpenAP hybrid (base + physics) with XGB — reference baseline."""
    feats = feature_set("openap_hybrid", pdf, [], include_physics=True)
    _, preds = run_approach(
        "OpenAP Hybrid", feats, pdf, train_idx, test_idx,
        y_train, y_test, None, physics_test, residual_mode=False,
    )
    return np.abs(y_test - preds["xgb"]), preds


def experiment_block(
    name: str,
    approaches: dict[str, tuple[list[str], bool]],
    pdf,
    train_idx,
    test_idx,
    y_train,
    y_test,
    physics_train,
    physics_test,
    baseline_err: np.ndarray,
    baseline_label: str,
    fig_path: Path,
    table_path: Path,
    sig_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    test_fids = pdf["flight_id"].to_numpy()[test_idx]

    all_metrics: dict = {}
    all_preds: dict[str, dict[str, np.ndarray]] = {}
    sig_rows: list[dict] = []
    boot_dists: dict[str, np.ndarray] = {}

    for approach, (feats, residual) in approaches.items():
        print(f"  Training: {approach} ({'residual' if residual else 'direct'}) ...", flush=True)
        mets, preds = run_approach(
            approach, feats, pdf, train_idx, test_idx,
            y_train, y_test, physics_train, physics_test, residual_mode=residual,
        )
        all_metrics[approach] = mets
        all_preds[approach] = preds

        if approach != baseline_label:
            err = np.abs(y_test - preds["xgb"])
            sig = significance_test(err, baseline_err, test_fids, approach, baseline_label)
            boot_dists[approach] = sig.pop("bootstrap_dist")
            sig_rows.append(sig)
            print(
                f"    XGB {approach}: MAE={mets['xgb']['mae']:.2f}  "
                f"ΔMAE={sig['delta_mae']:+.2f}  CI=[{sig['ci_lower']:+.2f},{sig['ci_upper']:+.2f}]  "
                f"p={sig['wilcoxon_p']:.2e}  → {sig['interpretation']}"
            )
        else:
            print(f"    XGB {approach}: MAE={mets['xgb']['mae']:.2f}  (baseline)")

    results_df = pl.DataFrame(metrics_to_rows(all_metrics)).sort(["approach", "mae"])
    results_df.write_csv(table_path)

    sig_df = pl.DataFrame(sig_rows) if sig_rows else pl.DataFrame()
    if not sig_df.is_empty():
        sig_df.write_csv(sig_path)

    plot_comparison_bars(results_df, f"{name} — Model Comparison (held-out flights)", fig_path)

    if boot_dists:
        fig, axes = plt.subplots(1, len(boot_dists), figsize=(5 * len(boot_dists), 4))
        if len(boot_dists) == 1:
            axes = [axes]
        for ax, (label, dist) in zip(axes, boot_dists.items()):
            ci_lo, ci_hi = np.percentile(dist, [2.5, 97.5])
            ax.hist(dist, bins=50, color="steelblue", alpha=0.85, density=True)
            ax.axvline(0, color="black", lw=1.2)
            ax.axvspan(ci_lo, ci_hi, alpha=0.2, color="green")
            ax.set_title(f"{label}\nΔMAE vs {baseline_label}")
            ax.set_xlabel("ΔMAE [kg]")
        fig.suptitle(f"{name} — Bootstrap ΔMAE (XGB, {N_BOOTSTRAP:,} flight resamples)", y=1.02)
        fig.tight_layout()
        boot_path = fig_path.with_name(fig_path.stem + "_bootstrap.png")
        fig.savefig(boot_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: {table_path}")
    print(f"  Saved: {fig_path}")
    if not sig_df.is_empty():
        print(f"  Saved: {sig_path}")

    return results_df, sig_df, all_metrics


def main() -> None:
    print("=" * 70)
    print("AeroTwin v2 — Physics-Informed Feature Experiments")
    print("=" * 70)

    df = load_and_clean(PARQUET)
    df = ensure_enriched(df)
    pdf = df.to_pandas()

    flight_ids = pdf["flight_id"].to_numpy()
    train_idx, test_idx, train_fids, test_fids = flight_level_split(flight_ids)

    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    physics_train = pdf["physics_fuel_kg"].to_numpy()[train_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]

    print(f"\nData: {len(df):,} intervals")
    print(f"Train flights: {len(train_fids):,}  |  Test flights: {len(test_fids):,}")

    energy_extra = available(ENERGY_FEATURES, df)
    ops_extra = available(OPERATIONAL_FEATURES, df)

    baseline_err, _ = best_xgb_baseline_preds(
        pdf, train_idx, test_idx, y_train, y_test, physics_test
    )
    baseline_mae = float(baseline_err.mean())
    print(f"\nReference baseline (XGB OpenAP Hybrid): MAE={baseline_mae:.2f} kg")

    # --- E2: Energy features ---
    e2_approaches = {
        "OpenAP Hybrid": (feature_set("base", pdf, [], True), False),
        "Energy Hybrid": (feature_set("energy", pdf, energy_extra, True), False),
        "No Physics": (feature_set("base", pdf, [], False), False),
    }
    e2_results, e2_sig, _ = experiment_block(
        "E2: Energy-State Features",
        e2_approaches, pdf, train_idx, test_idx,
        y_train, y_test, physics_train, physics_test,
        baseline_err, "OpenAP Hybrid",
        OUT / "fig_energy_features.png",
        OUT / "table_energy_results.csv",
        OUT / "table_significance_energy.csv",
    )

    # --- E3: Operational features ---
    e3_approaches = {
        "OpenAP Hybrid": (feature_set("base", pdf, [], True), False),
        "Operational Hybrid": (feature_set("ops", pdf, ops_extra, True), False),
        "No Physics": (feature_set("base", pdf, [], False), False),
    }
    e3_results, e3_sig, _ = experiment_block(
        "E3: Operational Features",
        e3_approaches, pdf, train_idx, test_idx,
        y_train, y_test, physics_train, physics_test,
        baseline_err, "OpenAP Hybrid",
        OUT / "fig_operational_features.png",
        OUT / "table_operational_results.csv",
        OUT / "table_significance_operational.csv",
    )

    # --- E4: Residual learning ---
    base_feats = feature_set("base", pdf, [], include_physics=False)
    energy_feats = feature_set("energy", pdf, energy_extra, include_physics=False)
    ops_feats = feature_set("ops", pdf, ops_extra, include_physics=False)

    e4_approaches = {
        "OpenAP Hybrid": (feature_set("base", pdf, [], True), False),
        "No Physics": (base_feats, False),
        "Energy Hybrid": (feature_set("energy", pdf, energy_extra, True), False),
        "Operational Hybrid": (feature_set("ops", pdf, ops_extra, True), False),
        "Residual-RF/XGB/LGBM": (base_feats, True),
        "Energy Residual": (energy_feats, True),
        "Operational Residual": (ops_feats, True),
    }
    e4_results, e4_sig, e4_metrics = experiment_block(
        "E4: Residual Learning",
        e4_approaches, pdf, train_idx, test_idx,
        y_train, y_test, physics_train, physics_test,
        baseline_err, "OpenAP Hybrid",
        OUT / "fig_residual_learning.png",
        OUT / "table_residual_results.csv",
        OUT / "table_significance_residual.csv",
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Best XGB per experiment vs OpenAP Hybrid baseline")
    print("=" * 70)

    summaries = [
        ("E2 Energy", e2_results, e2_sig),
        ("E3 Operational", e3_results, e3_sig),
        ("E4 Residual", e4_results, e4_sig),
    ]
    best_overall = ("OpenAP Hybrid", baseline_mae, "reference")

    for label, res, sig in summaries:
        xgb_rows = res.filter(pl.col("model") == "XGB").sort("mae")
        best = xgb_rows.row(0, named=True)
        print(f"\n{label}:")
        print(f"  Best: {best['approach']}  MAE={best['mae']:.2f}  RMSE={best['rmse']:.2f}  R²={best['r2']:.4f}")
        if not sig.is_empty():
            best_sig = sig.sort("delta_mae").row(0, named=True)
            print(
                f"  Strongest vs baseline: {best_sig['comparison']}  "
                f"ΔMAE={best_sig['delta_mae']:+.2f}  "
                f"CI=[{best_sig['ci_lower']:+.2f},{best_sig['ci_upper']:+.2f}]  "
                f"→ {best_sig['interpretation']}"
            )
            if best_sig["delta_mae"] < best_overall[1] - baseline_mae:
                best_overall = (best_sig["comparison"], best["mae"], best_sig["interpretation"])

    print("\n" + "-" * 70)
    print("HYPOTHESIS TEST:")
    print(
        "OpenAP alone is insufficient (confirmed in prior work). "
        "Among new physics-informed families, the statistically supported "
        "improvements over the OpenAP hybrid are summarized in table_significance_*.csv."
    )
    if not e2_sig.is_empty() or not e3_sig.is_empty() or not e4_sig.is_empty():
        all_sig = pl.concat([s for s in [e2_sig, e3_sig, e4_sig] if not s.is_empty()])
        winners = all_sig.filter(pl.col("ci_upper") < 0)
        if winners.is_empty():
            print("→ No new feature family shows bootstrap-significant improvement over OpenAP hybrid.")
        else:
            for row in winners.iter_rows(named=True):
                print(f"→ {row['comparison']}: ΔMAE={row['delta_mae']:.2f} kg ({row['interpretation']})")
    print("=" * 70)


if __name__ == "__main__":
    main()