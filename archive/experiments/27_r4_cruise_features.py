"""R4 — Cruise feature engineering with ablation.

Adds physics-informed cruise-specific features derived from existing parquet columns.
Tests each sub-family independently and full set on top of mass-enriched ensemble.

Ablation families:
  R4a: Core cruise features (duration, altitude, Mach, TAS, fuel flow, efficiency, load, wind)
  R4b: Interaction features (alt×dur, Mach×dur, ff×mass, mass×Mach, TAS×dur, wind×dur)
  R4c: Full cruise stack (core + interactions)
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
    BASELINE_OFFICIAL, HEAVY_TYPES, NARROW_TYPES, ENSEMBLE_BASES,
    ConditionalAffineCalibrator, accept_gate, ensure_features, full_scorecard,
    group_phase, load_splits, build_or_load_ensemble,
)
from physics.mass_model import enrich_mass_from_columns, R3_MASS_FEATURES
from physics.cruise_features import (
    enrich_cruise_features, R4_CRUISE_CORE, R4_CRUISE_INTERACTIONS, R4_ALL,
)
from physics.official_benchmark import (
    ew_feature_cols, build_oof_matrix, choose_meta_on_train_folds,
    apply_bases, predict_fuel_kg, prepare_xy, train_model, fit_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r4_cruise")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    # Enrich with mass first (prerequisite)
    LOGGER.info("Enriching with mass features...")
    train = enrich_mass_from_columns(train)
    rank = enrich_mass_from_columns(rank)
    final = enrich_mass_from_columns(final)

    # Enrich with cruise features
    LOGGER.info("Enriching with cruise features...")
    train = enrich_cruise_features(train)
    rank = enrich_cruise_features(rank)
    final = enrich_cruise_features(final)

    # Available feature sets
    base_feat_cols = ew_feature_cols(train)
    mass_cols = [c for c in R3_MASS_FEATURES if c in train.columns]
    cruise_core = [c for c in R4_CRUISE_CORE if c in train.columns]
    cruise_inter = [c for c in R4_CRUISE_INTERACTIONS if c in train.columns]

    LOGGER.info("Features: base=%d mass=%d cruise_core=%d cruise_inter=%d",
                len(base_feat_cols), len(mass_cols), len(cruise_core), len(cruise_inter))

    feat_mass = list(dict.fromkeys(base_feat_cols + mass_cols))
    feat_mass_core = list(dict.fromkeys(feat_mass + cruise_core))
    feat_mass_inter = list(dict.fromkeys(feat_mass + cruise_inter))
    feat_full = list(dict.fromkeys(feat_mass + cruise_core + cruise_inter))

    leaderboard: list[dict] = []

    def train_eval(name, train_df, rank_df, final_df, feat_cols, model_key="lgbm"):
        X_tr, y_flow, y_kg_tr, dur_tr = prepare_xy(train_df, feat_cols, "fuel_flow")
        model = train_model(model_key, X_tr, y_flow, feat_cols)
        X_r, _, y_kg_r, dur_r = prepare_xy(rank_df, feat_cols, "direct")
        X_f, _, y_kg_f, dur_f = prepare_xy(final_df, feat_cols, "direct")
        pred_r = predict_fuel_kg(model, X_r, dur_r, "fuel_flow")
        pred_f = predict_fuel_kg(model, X_f, dur_f, "fuel_flow")
        card = full_scorecard(name, rank_df, final_df, pred_r, pred_f)
        card["feature_count"] = len(feat_cols)
        return card, pred_r, pred_f, model

    # =========================================================================
    # Baseline: mass only (no cruise)
    # =========================================================================
    card_base, pr_base, pf_base, _ = train_eval("R4_baseline_mass_only", train, rank, final, feat_mass)
    card_base["gate"] = "BASELINE"
    leaderboard.append(card_base)
    baseline_rmse = card_base["combined_rmse"]
    LOGGER.info("Baseline (mass only): combined=%.2f bias=%.2f", baseline_rmse, card_base["combined_bias"])

    # =========================================================================
    # R4a: Core cruise features
    # =========================================================================
    card_core, pr_core, pf_core, _ = train_eval("R4a_core_cruise", train, rank, final, feat_mass_core)
    card_core["delta_vs_baseline"] = card_core["combined_rmse"] - baseline_rmse
    leaderboard.append(card_core)
    LOGGER.info("R4a core cruise: combined=%.2f delta=%.2f bias=%.2f",
                card_core["combined_rmse"], card_core["delta_vs_baseline"], card_core["combined_bias"])

    # =========================================================================
    # R4b: Cruise interaction features
    # =========================================================================
    card_inter, pr_inter, pf_inter, _ = train_eval("R4b_cruise_interactions", train, rank, final, feat_mass_inter)
    card_inter["delta_vs_baseline"] = card_inter["combined_rmse"] - baseline_rmse
    leaderboard.append(card_inter)
    LOGGER.info("R4b cruise inter: combined=%.2f delta=%.2f",
                card_inter["combined_rmse"], card_inter["delta_vs_baseline"])

    # =========================================================================
    # R4c: Full cruise stack
    # =========================================================================
    card_full, pr_full, pf_full, _ = train_eval("R4c_full_cruise", train, rank, final, feat_full)
    card_full["delta_vs_baseline"] = card_full["combined_rmse"] - baseline_rmse
    leaderboard.append(card_full)
    LOGGER.info("R4c full cruise: combined=%.2f delta=%.2f heavy=%.1f bias=%.2f",
                card_full["combined_rmse"], card_full["delta_vs_baseline"],
                card_full["heavy_rmse"], card_full["combined_bias"])

    # =========================================================================
    # Ablate cruise core sub-families
    # =========================================================================
    core_sub = {
        "cruise_kinematics": ["r4_cruise_duration_s", "r4_cruise_altitude_m",
                              "r4_cruise_mach_est", "r4_cruise_tas_mps"],
        "cruise_efficiency": ["r4_cruise_fuel_flow_kgps", "r4_cruise_efficiency",
                              "r4_cruise_load_factor"],
        "cruise_altitude": ["r4_cruise_altitude_band", "r4_cruise_pct_max_alt",
                           "r4_cruise_spd_stability"],
        "cruise_wind": ["r4_cruise_tailwind_mps", "r4_cruise_headwind_mps"],
    }

    for fam, fam_cols in core_sub.items():
        avail = [c for c in fam_cols if c in train.columns]
        if not avail:
            continue
        feats = list(dict.fromkeys(feat_mass + avail))
        card, _, _, _ = train_eval(f"R4_ablate_{fam}", train, rank, final, feats)
        card["delta_vs_baseline"] = card["combined_rmse"] - baseline_rmse
        leaderboard.append(card)
        LOGGER.info("R4 ablate %s: combined=%.2f delta=%.2f",
                    fam, card["combined_rmse"], card["delta_vs_baseline"])

    # =========================================================================
    # Ensemble OOF with full cruise features
    # =========================================================================
    LOGGER.info("=== Ensemble OOF with full cruise + mass ===")
    P_oof, y_kg, full_models = build_oof_matrix(train, feat_full, ENSEMBLE_BASES, n_splits=5)
    groups = train["flight_id"].to_numpy()
    meta_kind, meta = choose_meta_on_train_folds(P_oof, y_kg, groups, n_splits=5)
    oof_pred = np.asarray(meta.predict(P_oof), dtype=np.float64)
    LOGGER.info("Meta: %s, train OOF RMSE: %.2f", meta_kind,
                float(np.sqrt(np.mean((oof_pred - y_kg) ** 2))))

    rank_a = ensure_features(rank, feat_full)
    final_a = ensure_features(final, feat_full)
    P_r = apply_bases(full_models, rank_a, feat_full)
    P_f = apply_bases(full_models, final_a, feat_full)
    pred_r0 = np.asarray(meta.predict(P_r), dtype=np.float64)
    pred_f0 = np.asarray(meta.predict(P_f), dtype=np.float64)

    # P1E calibration
    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train, y_kg, oof_pred)
    pr_p1e = cal_phase.transform(rank_a, pred_r0)
    pf_p1e = cal_phase.transform(final_a, pred_f0)

    ensemble_card = full_scorecard("R4_ensemble_cruise_mass_P1E", rank_a, final_a, pr_p1e, pf_p1e,
                                   hypothesis="Full cruise features in ensemble (mass + cruise + P1E)")
    ensemble_card["delta_vs_baseline"] = ensemble_card["combined_rmse"] - baseline_rmse
    ensemble_card["feature_count"] = len(feat_full)
    leaderboard.append(ensemble_card)

    ok, reason = accept_gate(ensemble_card, card_base)
    ensemble_card["gate"] = "KEEP" if ok else "REJECT"
    ensemble_card["gate_reason"] = reason
    LOGGER.info("Ensemble: combined=%.2f heavy=%.1f narrow=%.1f bias=%.2f gate=%s",
                ensemble_card["combined_rmse"], ensemble_card.get("heavy_rmse", 0),
                ensemble_card.get("narrow_rmse", 0), ensemble_card["combined_bias"],
                ensemble_card["gate"])

    # =========================================================================
    # Save results
    # =========================================================================
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    r4_rows = [r for r in leaderboard if "R4" in r["variant"]]
    if r4_rows:
        pl.DataFrame(r4_rows).write_csv(OUT / "table_rmse_R4_cruise.csv")
    lb.write_csv(OUT / "table_rmse_R4_full_leaderboard.csv")

    best_r4 = min(r4_rows, key=lambda x: x["combined_rmse"]) if r4_rows else ensemble_card
    summary = {
        "task": "R4",
        "best_variant": best_r4["variant"],
        "combined_rmse": best_r4["combined_rmse"],
        "rank_rmse": best_r4["rank_rmse"],
        "final_rmse": best_r4["final_rmse"],
        "bias": best_r4["combined_bias"],
        "delta_vs_baseline_mass_only": best_r4["combined_rmse"] - baseline_rmse,
        "delta_vs_221_33": best_r4["combined_rmse"] - 221.33,
        "delta_vs_228_25": best_r4["combined_rmse"] - 228.25,
        "heavy_rmse": best_r4.get("heavy_rmse", float("nan")),
        "narrow_rmse": best_r4.get("narrow_rmse", float("nan")),
        "n_cruise_core": len(cruise_core),
        "n_cruise_inter": len(cruise_inter),
        "n_variants": len(r4_rows),
    }
    (OUT / "r4_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 70)
    print("R4 CRUISE FEATURES SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    print("\nLeaderboard (top 10):")
    for row in lb.head(10).iter_rows(named=True):
        d = ""
        if "delta_vs_baseline" in row and row["delta_vs_baseline"] is not None:
            d = f"  Δ={row['delta_vs_baseline']:+.2f}"
        print(f"  {row['variant']:<45s} Combined={row['combined_rmse']:.2f}{d}")


if __name__ == "__main__":
    main()
