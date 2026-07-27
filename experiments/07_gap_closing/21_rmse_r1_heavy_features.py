"""R1 — Heavy specialist with OpenAP continuous descriptors + interaction features.

Protocol:
- Modify only the heavy specialist (HEAVY_TYPES).
- Add OpenAP descriptors (MTOW, OEW, wing area, max thrust, engine count,
  aspect ratio, etc.) from table_aircraft_openap_descriptors.csv.
- Create interaction features: cruise altitude x duration, mean alt x duration,
  cruise ratio x duration, FuelFlow x MTOW, wing loading proxies.
- Do not modify routing logic (hard route: heavy → specialist, else base).
- Train only on train data.
- Run official evaluation pipeline (Rank/Final).
- Report Rank RMSE, Final RMSE, Combined RMSE, bias, subgroup metrics.
- Keep only if Combined RMSE improves without materially hurting narrowbodies.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import project_root
from aerotwin.engine.gap_closing import (
    BASELINE_OFFICIAL,
    AffineCalibrator,
    accept_gate,
    apply_calibrator,
    build_or_load_ensemble,
    ensure_features,
    full_scorecard,
    group_phase,
    load_splits,
    predict_ensemble,
    predict_heavy_routed,
    predict_heavy_routed_r1,
    train_heavy_specialist,
    train_heavy_specialist_r1,
)
from aerotwin.engine.official_benchmark import prepare_xy, predict_fuel_kg, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r1_heavy_features")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    force_rebuild = True
    cache_path = project_root() / "cache" / "official_ensemble_cache.pkl"
    if cache_path.exists():
        import pickle

        with open(cache_path, "rb") as f:
            old = pickle.load(f)
        force_rebuild = len(old.oof_pred) != len(train)
    bundle = build_or_load_ensemble(train, force=force_rebuild)
    feat_cols = bundle.feat_cols

    rank = ensure_features(rank, feat_cols)
    final = ensure_features(final, feat_cols)

    pred_r0 = predict_ensemble(bundle, rank)
    pred_f0 = predict_ensemble(bundle, final)
    oof = bundle.oof_pred
    y_tr = bundle.y_train
    train_oof_df = train

    leaderboard: list[dict] = []

    # Session baseline
    session_card = full_scorecard(
        "session_rebuild_ensemble",
        rank,
        final,
        pred_r0,
        pred_f0,
        hypothesis="Rebuilt OOF ensemble this run",
        expected_gain="reference_session",
    )
    leaderboard.append(session_card)

    official_card = {
        **session_card,
        "variant": "baseline_official_v1_recorded",
        "hypothesis": "Recorded official ensemble from notebook 17",
        "expected_gain": "reference_official",
        "rank_rmse": BASELINE_OFFICIAL["rank_rmse"],
        "final_rmse": BASELINE_OFFICIAL["final_rmse"],
        "combined_rmse": BASELINE_OFFICIAL["combined_rmse"],
        "combined_mae": BASELINE_OFFICIAL["combined_mae"],
        "delta_combined_vs_baseline": 0.0,
        "gate": "REFERENCE",
        "gate_reason": "official_v1",
    }
    leaderboard.append(official_card)

    current_best = session_card
    best_pred_r, best_pred_f = pred_r0.copy(), pred_f0.copy()
    LOGGER.info(
        "SESSION combined_rmse=%.2f | OFFICIAL floor=%.2f",
        session_card["combined_rmse"],
        BASELINE_OFFICIAL["combined_rmse"],
    )

    # -----------------------------------------------------------------------
    # P1E — Phase-conditional affine (from gap_closing v1.1)
    # -----------------------------------------------------------------------
    cal_phase = (
        __import__("physics.gap_closing", fromlist=["ConditionalAffineCalibrator"])
        .ConditionalAffineCalibrator(group_phase)
        .fit(train_oof_df, y_tr, oof)
    )
    pr = apply_calibrator(cal_phase, rank, pred_r0)
    pf = apply_calibrator(cal_phase, final, pred_f0)
    p1e_card = full_scorecard(
        "P1E_phase_affine",
        rank,
        final,
        pr,
        pf,
        hypothesis="Phase-conditional affine bias calibration",
        expected_gain="-0.1 kg",
    )
    p1e_card["train_oof_rmse"] = float(
        np.sqrt(np.mean((apply_calibrator(cal_phase, train_oof_df, oof) - y_tr) ** 2))
    )
    leaderboard.append(p1e_card)
    LOGGER.info("P1E combined=%.2f bias=%.2f", p1e_card["combined_rmse"], p1e_card["combined_bias"])

    # Use P1E as base for specialists
    base_r, base_f = pr.copy(), pf.copy()

    # -----------------------------------------------------------------------
    # Baseline heavy specialist (no new features) — for comparison
    # -----------------------------------------------------------------------
    p2_results = []
    for mkey in ("lgbm", "cat", "xgb"):
        name = f"P2_heavy_{mkey}_flow_on_P1base"
        try:
            spec = train_heavy_specialist(train, feat_cols, model_key=mkey)
        except Exception as exc:
            LOGGER.warning("Baseline specialist %s failed: %s", mkey, exc)
            continue
        pr = predict_heavy_routed(spec, feat_cols, rank, base_r)
        pf = predict_heavy_routed(spec, feat_cols, final, base_f)
        card = full_scorecard(
            name,
            rank,
            final,
            pr,
            pf,
            hypothesis="Baseline heavy FuelFlow specialist (no R1 features)",
            expected_gain="reference",
        )
        ok, reason = accept_gate(card, current_best)
        if not ok and card["combined_rmse"] < BASELINE_OFFICIAL["combined_rmse"] - 0.05:
            ok, reason = True, "accepted_vs_official_floor"
        card["gate"] = "KEEP" if ok else "REJECT"
        card["gate_reason"] = reason
        p2_results.append(card)
        leaderboard.append(card)
        LOGGER.info(
            "%s combined=%.2f heavy=%.1f a359=%.1f gate=%s",
            name,
            card["combined_rmse"],
            card["heavy_rmse"],
            card["a359_rmse"],
            card["gate"],
        )
        if ok:
            current_best = card
            best_pred_r, best_pred_f = pr, pf

    # -----------------------------------------------------------------------
    # R1 — Heavy specialist with OpenAP descriptors + interactions
    # -----------------------------------------------------------------------
    r1_results = []
    for mkey in ("lgbm", "cat", "xgb"):
        name = f"R1_heavy_{mkey}_openap_descriptors"
        hyp = (
            "OpenAP continuous descriptors (MTOW, OEW, wing area, max thrust, "
            "aspect ratio, etc.) + interactions (cruise alt x dur, mean alt x dur, "
            "wing loading, thrust loading, OEW ratio) inside heavy specialist"
        )
        try:
            spec, r1_cols = train_heavy_specialist_r1(train, feat_cols, model_key=mkey)
        except Exception as exc:
            LOGGER.warning("R1 specialist %s failed: %s", mkey, exc)
            continue
        pr = predict_heavy_routed_r1(spec, feat_cols, rank, base_r)
        pf = predict_heavy_routed_r1(spec, feat_cols, final, base_f)
        card = full_scorecard(name, rank, final, pr, pf, hypothesis=hyp, expected_gain="-3 to -12 kg")
        card["r1_feature_count"] = len(r1_cols)
        card["r1_model_key"] = mkey
        ok, reason = accept_gate(card, current_best)
        if not ok and card["combined_rmse"] < BASELINE_OFFICIAL["combined_rmse"] - 0.05:
            ok, reason = True, "accepted_vs_official_floor"
        card["gate"] = "KEEP" if ok else "REJECT"
        card["gate_reason"] = reason
        r1_results.append(card)
        leaderboard.append(card)
        LOGGER.info(
            "%s combined=%.2f heavy=%.1f a359=%.1f b77w=%.1f b744=%.1f narrow=%.1f gate=%s",
            name,
            card["combined_rmse"],
            card["heavy_rmse"],
            card["a359_rmse"],
            card["b77w_rmse"],
            card["b744_rmse"],
            card["narrow_rmse"],
            card["gate"],
        )
        if ok:
            current_best = card
            best_pred_r, best_pred_f = pr, pf

    # -----------------------------------------------------------------------
    # Also R1 on raw ensemble (no P1E)
    # -----------------------------------------------------------------------
    for mkey in ("lgbm", "cat"):
        name = f"R1b_heavy_{mkey}_openap_on_raw_ensemble"
        try:
            spec, r1_cols = train_heavy_specialist_r1(train, feat_cols, model_key=mkey)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)
            continue
        pr = predict_heavy_routed_r1(spec, feat_cols, rank, pred_r0)
        pf = predict_heavy_routed_r1(spec, feat_cols, final, pred_f0)
        card = full_scorecard(
            name,
            rank,
            final,
            pr,
            pf,
            hypothesis="R1 heavy specialist on uncalibrated ensemble",
            expected_gain="-3 to -12 kg",
        )
        ok, reason = accept_gate(card, current_best)
        if not ok and card["combined_rmse"] < BASELINE_OFFICIAL["combined_rmse"] - 0.05:
            ok, reason = True, "accepted_vs_official_floor"
        card["gate"] = "KEEP" if ok else "REJECT"
        card["gate_reason"] = reason
        card["r1_feature_count"] = len(r1_cols)
        r1_results.append(card)
        leaderboard.append(card)

    # -----------------------------------------------------------------------
    # Save R1 results
    # -----------------------------------------------------------------------
    if r1_results:
        r1_df = pl.DataFrame(r1_results)
        r1_df.write_csv(OUT / "table_rmse_R1.csv")
        LOGGER.info("Wrote %s", OUT / "table_rmse_R1.csv")
    else:
        LOGGER.warning("No R1 results produced!")

    # Full leaderboard
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    lb.write_csv(OUT / "table_rmse_R1_full_leaderboard.csv")

    # Summary
    summary = {
        "task": "R1",
        "baseline": BASELINE_OFFICIAL,
        "best_variant": current_best["variant"],
        "best_combined_rmse": current_best["combined_rmse"],
        "best_rank_rmse": current_best["rank_rmse"],
        "best_final_rmse": current_best["final_rmse"],
        "best_bias": current_best["combined_bias"],
        "delta_vs_official": current_best["combined_rmse"] - BASELINE_OFFICIAL["combined_rmse"],
        "delta_vs_227_44": current_best["combined_rmse"] - 227.44,
        "best_heavy_rmse": current_best["heavy_rmse"],
        "best_narrow_rmse": current_best["narrow_rmse"],
        "best_a359_rmse": current_best["a359_rmse"],
        "best_b77w_rmse": current_best["b77w_rmse"],
        "best_b744_rmse": current_best["b744_rmse"],
        "n_r1_variants": len(r1_results),
    }
    (OUT / "r1_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Figure
    sns.set_theme(style="whitegrid")
    plot = lb.to_pandas()
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for _, row in plot.iterrows():
        v = row["variant"]
        if v == "baseline_official_v1_recorded":
            colors.append("#2E5A88")
        elif v.startswith("R1"):
            colors.append("#E86850")
        elif row.get("gate") == "KEEP":
            colors.append("#2F7D4F")
        else:
            colors.append("#D1D5DB")
    ax.barh(plot["variant"], plot["combined_rmse"], color=colors)
    ax.axvline(201.0, color="black", ls="--", label="Winner ~201")
    ax.axvline(BASELINE_OFFICIAL["combined_rmse"], color="#C45C26", ls=":", label="Baseline 228.3")
    ax.set_xlabel("Combined RMSE [kg]")
    ax.set_title("R1 — Heavy Specialist with OpenAP Descriptors + Interactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_r1_heavy_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\n=== R1 HEAVY FEATURES SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\nTop 10 by Combined RMSE:")
    print(lb.head(10))


if __name__ == "__main__":
    main()
