"""R2 — Fuel-flow filtering audit.

Verify:
  - Target calculation (actual_fuel_kg / duration_s)
  - Filtering intervals with FuelFlow <0.05 kg/s and >6.5 kg/s
  - Report the percentage of removed samples and RMSE impact.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import project_root
from physics.gap_closing import (
    BASELINE_OFFICIAL,
    load_splits,
    full_scorecard,
    build_or_load_ensemble,
    ensure_features,
    predict_ensemble,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r2_fuel_flow")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    # Compute fuel flow (kg/s) for all splits
    for name, df in [("train", train), ("rank", rank), ("final", final)]:
        ff = df["actual_fuel_kg"].to_numpy() / np.clip(df["duration_s"].to_numpy(), 1.0, None)
        LOGGER.info(
            "%s fuel_flow (kg/s): mean=%.3f median=%.3f std=%.3f "
            "min=%.4f max=%.4f p1=%.4f p99=%.4f",
            name,
            float(np.mean(ff)),
            float(np.median(ff)),
            float(np.std(ff)),
            float(np.min(ff)),
            float(np.max(ff)),
            float(np.percentile(ff, 1)),
            float(np.percentile(ff, 99)),
        )

    # Detailed fuel flow distribution analysis
    ff_train = train["actual_fuel_kg"].to_numpy() / np.clip(train["duration_s"].to_numpy(), 1.0, None)

    thresholds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 10.0]
    for lo, hi in zip(thresholds, thresholds[1:] + [1e9]):
        mask = (ff_train >= lo) & (ff_train < hi)
        if mask.any():
            LOGGER.info("  Flow [%.4f, %.1f]: n=%d (%.2f%%)", lo, hi, mask.sum(), 100 * mask.sum() / len(ff_train))

    # Test filtering thresholds
    LOW_THRESHOLDS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1]
    HIGH_THRESHOLDS = [1e9, 10.0, 8.0, 7.0, 6.5, 6.0, 5.0]

    # Quick ablation: evaluate impact of filtering on a single LGBM FuelFlow model
    # (not full ensemble - fast proxy for direction)
    from physics.official_benchmark import prepare_xy, train_model, predict_fuel_kg, ew_feature_cols

    feat_cols = ew_feature_cols(train)
    rank_small = ensure_features(rank, feat_cols)
    final_small = ensure_features(final, feat_cols)

    # Track removed % for key thresholds
    results_filter = []

    def train_eval_filtered(name, train_df, feat_cols, flow_lo, flow_hi, out):
        nonlocal results_filter
        ff = train_df["actual_fuel_kg"].to_numpy() / np.clip(train_df["duration_s"].to_numpy(), 1.0, None)

        if flow_lo > 0 or flow_hi < 1e8:
            mask = (ff >= flow_lo) & (ff <= flow_hi)
            removed = (~mask).sum()
            removed_pct = 100 * removed / len(train_df)
            train_f = train_df.filter(mask)
        else:
            removed = 0
            removed_pct = 0.0
            train_f = train_df

        LOGGER.info("%s: n_train=%d removed=%d (%.2f%%)", name, len(train_f), removed, removed_pct)

        try:
            X_tr, y_flow, y_kg, dur_tr = prepare_xy(train_f, feat_cols, "fuel_flow")
            model = train_model("lgbm", X_tr, y_flow, feat_cols)

            # Evaluate on Rank and Final
            pred_r = None
            pred_f = None
            for split_name, df in [("rank", rank_small), ("final", final_small)]:
                X, _, _, dur = prepare_xy(df, feat_cols, "direct")
                pred = predict_fuel_kg(model, X, dur, "fuel_flow")
                if split_name == "rank":
                    pred_r = pred
                else:
                    pred_f = pred

            card = full_scorecard(
                name,
                rank_small,
                final_small,
                pred_r,
                pred_f,
                hypothesis=f"Fuel-flow filter: [{flow_lo}, {flow_hi}]",
                expected_gain="audit",
            )
            card["flow_lo"] = flow_lo
            card["flow_hi"] = flow_hi
            card["removed_pct"] = removed_pct
            results_filter.append(card)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)

    # Baseline (no filter)
    train_eval_filtered("R2_baseline_no_filter", train, feat_cols, 0.0, 1e9, OUT)

    # Low-end filters
    for lo in [0.05, 0.1]:
        train_eval_filtered(f"R2_filter_flow_geq_{lo}", train, feat_cols, lo, 1e9, OUT)

    # High-end filters
    for hi in [6.5, 6.0]:
        train_eval_filtered(f"R2_filter_flow_leq_{hi}", train, feat_cols, 0.0, hi, OUT)

    # Combination
    train_eval_filtered("R2_filter_flow_0.05_to_6.5", train, feat_cols, 0.05, 6.5, OUT)

    # Save results
    if results_filter:
        pl.DataFrame(results_filter).write_csv(OUT / "table_rmse_R2_fuel_flow_filter.csv")
        LOGGER.info("Wrote %s", OUT / "table_rmse_R2_fuel_flow_filter.csv")

    # Distribution plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, df) in zip(axes, [("Train", train), ("Rank", rank), ("Final", final)]):
        ff = df["actual_fuel_kg"].to_numpy() / np.clip(df["duration_s"].to_numpy(), 1.0, None)
        ax.hist(np.clip(ff, 0, 10), bins=100, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(0.05, color="red", ls="--", label="0.05 kg/s")
        ax.axvline(6.5, color="red", ls="--", label="6.5 kg/s")
        ax.set_title(f"{name} — Fuel Flow Distribution")
        ax.set_xlabel("Fuel Flow (kg/s)")
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_fuel_flow_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary
    ff_all = np.concatenate([
        train["actual_fuel_kg"].to_numpy() / np.clip(train["duration_s"].to_numpy(), 1.0, None),
        rank["actual_fuel_kg"].to_numpy() / np.clip(rank["duration_s"].to_numpy(), 1.0, None),
        final["actual_fuel_kg"].to_numpy() / np.clip(final["duration_s"].to_numpy(), 1.0, None),
    ])

    summary = {
        "task": "R2",
        "fuel_flow_stats": {
            "mean_kgps": float(np.mean(ff_all)),
            "median_kgps": float(np.median(ff_all)),
            "std_kgps": float(np.std(ff_all)),
            "min_kgps": float(np.min(ff_all)),
            "max_kgps": float(np.max(ff_all)),
            "pct_below_0.05": float(100 * np.mean(ff_all < 0.05)),
            "pct_above_6.5": float(100 * np.mean(ff_all > 6.5)),
            "pct_extreme": float(100 * np.mean((ff_all < 0.05) | (ff_all > 6.5))),
        },
        "n_results": len(results_filter),
    }
    (OUT / "r2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== R2 FUEL-FLOW AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    if results_filter:
        print("\nFilter results:")
        print(pl.DataFrame(results_filter)[["variant", "combined_rmse", "removed_pct"]])


if __name__ == "__main__":
    main()
