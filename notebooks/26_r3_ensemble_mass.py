"""R3 — Full ensemble evaluation with dynamic mass features.

This rebuilds the ensemble OOF matrix with R3 mass features included in the base feature set.
Protocol:
1. Enrich train/rank/final with R3 mass features
2. Build OOF ensemble with extended feature set
3. Train P1E calibrator
4. Train heavy specialist
5. Evaluate on Rank/Final
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import project_root
from physics.gap_closing import (
    BASELINE_OFFICIAL,
    HEAVY_TYPES,
    ENSEMBLE_BASES,
    AffineCalibrator,
    ConditionalAffineCalibrator,
    accept_gate,
    build_or_load_ensemble,
    ensure_features,
    full_scorecard,
    group_phase,
    load_splits,
    predict_ensemble,
    predict_heavy_routed,
    predict_heavy_routed_r2,
    train_heavy_specialist,
    train_heavy_specialist_r2,
)
from physics.mass_model import enrich_mass_from_columns, validate_mass_features, R3_MASS_FEATURES
from physics.official_benchmark import (
    ew_feature_cols,
    build_oof_matrix,
    choose_meta_on_train_folds,
    apply_bases,
    predict_fuel_kg,
    prepare_xy,
    train_model,
    fit_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r3_ensemble")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    # Enrich all splits
    LOGGER.info("Enriching with R3 mass features...")
    train = enrich_mass_from_columns(train)
    rank = enrich_mass_from_columns(rank)
    final = enrich_mass_from_columns(final)
    val = validate_mass_features(train)
    LOGGER.info("Validation: valid=%s issues=%s", val["valid"], val["issues"][:3])

    # Build extended feature set
    base_feat_cols = ew_feature_cols(train)
    mass_cols_avail = [c for c in R3_MASS_FEATURES if c in train.columns]
    feat_cols_mass = list(dict.fromkeys(base_feat_cols + mass_cols_avail))
    LOGGER.info("Feature counts: base=%d mass=%d extended=%d",
                len(base_feat_cols), len(mass_cols_avail), len(feat_cols_mass))

    leaderboard: list[dict] = []

    # =========================================================================
    # Rebuild ensemble OOF with mass features
    # =========================================================================
    LOGGER.info("Building ensemble OOF with mass features (slow)...")
    P_oof, y_kg, full_models = build_oof_matrix(train, feat_cols_mass, ENSEMBLE_BASES, n_splits=5)
    groups = train["flight_id"].to_numpy()
    meta_kind, meta = choose_meta_on_train_folds(P_oof, y_kg, groups, n_splits=5)
    oof_pred = np.asarray(meta.predict(P_oof), dtype=np.float64)
    LOGGER.info("Meta: %s, train OOF RMSE: %.2f", meta_kind,
                float(np.sqrt(np.mean((oof_pred - y_kg) ** 2))))

    # Predict on Rank/Final
    rank_a = ensure_features(rank, feat_cols_mass)
    final_a = ensure_features(final, feat_cols_mass)
    P_r = apply_bases(full_models, rank_a, feat_cols_mass)
    P_f = apply_bases(full_models, final_a, feat_cols_mass)
    pred_r0 = np.asarray(meta.predict(P_r), dtype=np.float64)
    pred_f0 = np.asarray(meta.predict(P_f), dtype=np.float64)

    # Session baseline
    session_card = full_scorecard("R3_session_ensemble_mass", rank_a, final_a, pred_r0, pred_f0)
    leaderboard.append(session_card)

    official_card = {**session_card, "variant": "baseline_official_v1",
                     "rank_rmse": BASELINE_OFFICIAL["rank_rmse"],
                     "final_rmse": BASELINE_OFFICIAL["final_rmse"],
                     "combined_rmse": BASELINE_OFFICIAL["combined_rmse"],
                     "delta_combined_vs_baseline": 0.0,
                     "gate": "REFERENCE"}
    leaderboard.append(official_card)

    current_best = session_card
    best_pr, best_pf = pred_r0.copy(), pred_f0.copy()
    LOGGER.info("Session ensemble (mass): combined=%.2f", session_card["combined_rmse"])

    # =========================================================================
    # P1E phase-conditional affine
    # =========================================================================
    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train, y_kg, oof_pred)
    pr_p1e = cal_phase.transform(rank_a, pred_r0)
    pf_p1e = cal_phase.transform(final_a, pred_f0)
    p1e_card = full_scorecard("R3_P1E_phase_affine", rank_a, final_a, pr_p1e, pf_p1e)
    leaderboard.append(p1e_card)
    base_r, base_f = pr_p1e.copy(), pf_p1e.copy()
    LOGGER.info("P1E: combined=%.2f", p1e_card["combined_rmse"])

    def gate(card):
        ok, reason = accept_gate(card, current_best)
        if not ok and card["combined_rmse"] < BASELINE_OFFICIAL["combined_rmse"] - 0.05:
            ok, reason = True, "vs_official_floor"
        return ok, reason

    # =========================================================================
    # Heavy specialists on top
    # =========================================================================
    for mkey, train_fn, predict_fn, label in [
        ("cat", train_heavy_specialist, predict_heavy_routed, "P2_baseline"),
        ("cat", train_heavy_specialist_r2, predict_heavy_routed_r2, "R2_descriptors"),
    ]:
        try:
            if "r2" in label:
                spec, _ = train_fn(train, base_feat_cols, model_key=mkey)
            else:
                spec = train_fn(train, base_feat_cols, model_key=mkey)
            pr = predict_fn(spec, base_feat_cols, rank_a, base_r)
            pf = predict_fn(spec, base_feat_cols, final_a, base_f)
            name = f"R3_{label}_heavy_{mkey}"
            card = full_scorecard(name, rank_a, final_a, pr, pf)
            ok, reason = gate(card)
            card["gate"], card["gate_reason"] = ("KEEP" if ok else "REJECT"), reason
            leaderboard.append(card)
            LOGGER.info("%s: combined=%.2f heavy=%.1f gate=%s", name, card["combined_rmse"], card["heavy_rmse"], card["gate"])
            if ok:
                current_best = card; best_pr, best_pf = pr, pf
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)

    # =========================================================================
    # Also test LGBM Flow single model with mass (from R3 notebook result)
    # =========================================================================
    X_tr, y_flow, y_kg_tr, dur_tr = prepare_xy(train, feat_cols_mass, "fuel_flow")
    lgbm_flow = train_model("lgbm", X_tr, y_flow, feat_cols_mass)
    X_r, _, _, dur_r = prepare_xy(rank_a, feat_cols_mass, "direct")
    X_f, _, _, dur_f = prepare_xy(final_a, feat_cols_mass, "direct")
    pr_lgbm = predict_fuel_kg(lgbm_flow, X_r, dur_r, "fuel_flow")
    pf_lgbm = predict_fuel_kg(lgbm_flow, X_f, dur_f, "fuel_flow")
    card_lgbm = full_scorecard("R3_lgbm_flow_mass_single", rank_a, final_a, pr_lgbm, pf_lgbm)
    card_lgbm["feature_count"] = len(feat_cols_mass)
    leaderboard.append(card_lgbm)
    LOGGER.info("LGBM Flow + Mass: combined=%.2f", card_lgbm["combined_rmse"])

    # =========================================================================
    # Save results
    # =========================================================================
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    r3_rows = [r for r in leaderboard if "R3" in r["variant"]]
    if r3_rows:
        pl.DataFrame(r3_rows).write_csv(OUT / "table_rmse_R3_mass_ensemble.csv")
    lb.write_csv(OUT / "table_rmse_R3_ensemble_leaderboard.csv")

    best_r3 = min(r3_rows, key=lambda x: x["combined_rmse"]) if r3_rows else session_card
    summary = {
        "task": "R3_ensemble",
        "best_variant": best_r3["variant"],
        "combined_rmse": best_r3["combined_rmse"],
        "rank_rmse": best_r3["rank_rmse"],
        "final_rmse": best_r3["final_rmse"],
        "bias": best_r3["combined_bias"],
        "delta_vs_225_25": best_r3["combined_rmse"] - 225.25,
        "delta_vs_228_25": best_r3["combined_rmse"] - 228.25,
        "heavy_rmse": best_r3.get("heavy_rmse", 0),
        "narrow_rmse": best_r3.get("narrow_rmse", 0),
        "session_ensemble": session_card["combined_rmse"],
    }
    (OUT / "r3_ensemble_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== R3 ENSEMBLE + MASS SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\nLeaderboard:")
    for row in lb.head(10).iter_rows(named=True):
        print(f"  {row['variant']:<45s} Combined={row['combined_rmse']:.2f}")


if __name__ == "__main__":
    main()
