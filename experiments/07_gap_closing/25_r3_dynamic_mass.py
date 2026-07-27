"""R3 — Dynamic mass model evaluation.

Adds physics-informed mass features to the BASE ENSEMBLE feature set.
The mass features are computed from R3 dynamic mass model per interval using
only deployable flight-position information (aircraft_type, flight fractions, altitude).

Ablation: test mass features added to base ensemble, then test mass + heavy specialist.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import project_root
from aerotwin.engine.gap_closing import (
    BASELINE_OFFICIAL,
    AffineCalibrator,
    ConditionalAffineCalibrator,
    apply_calibrator,
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
from aerotwin.engine.mass_model import (
    enrich_mass_from_columns,
    validate_mass_features,
    R3_MASS_FEATURES,
)
from aerotwin.engine.official_benchmark import (
    apply_bases,
    ew_feature_cols,
    prepare_xy,
    train_model,
    predict_fuel_kg,
    fit_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r3_mass")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    # Enrich all splits with dynamic mass features
    LOGGER.info("Enriching splits with R3 mass features...")
    train = enrich_mass_from_columns(train)
    rank = enrich_mass_from_columns(rank)
    final = enrich_mass_from_columns(final)

    # Validate mass features
    val = validate_mass_features(train)
    LOGGER.info("Mass validation: valid=%s n_rows=%d n_features=%d issues=%s",
                val["valid"], val["n_rows"], val["n_features"], val["issues"][:3])
    LOGGER.info("Mass stats (train): %s",
                {k: f"mean={v['mean']:.0f}" for k, v in sorted(val.get("stats", {}).items())[:5]})

    # =========================================================================
    # Rebuild ensemble cache with mass features in the feature set
    # =========================================================================
    cache_path = project_root() / "cache" / "official_ensemble_cache.pkl"
    force = True
    if cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            old = pickle.load(f)
        force = len(old.oof_pred) != len(train)

    # Note: build_or_load_ensemble uses ew_feature_cols() which includes ENERGY_FEATURES,
    # WEATHER_FEATURES, etc. We need to also include the new R3 mass features.
    # We extend feat_cols to include R3_MASS_FEATURES.
    base_feat_cols = ew_feature_cols(train)
    mass_feat_cols_avail = [c for c in R3_MASS_FEATURES if c in train.columns]
    LOGGER.info("R3 mass features available in train: %d/%d", len(mass_feat_cols_avail), len(R3_MASS_FEATURES))

    # Two feature sets to compare:
    # A: baseline (ew_feature_cols only)
    # B: baseline + R3 mass features
    feat_set_A = base_feat_cols
    feat_set_B = list(dict.fromkeys(base_feat_cols + mass_feat_cols_avail))

    leaderboard: list[dict] = []

    # =========================================================================
    # Train and evaluate: baseline (no mass) using single LGBM FuelFlow model
    # =========================================================================
    def train_eval(name, train_df, rank_df, final_df, feat_cols, model_key="lgbm"):
        X_tr, y_flow, y_kg_tr, dur_tr = prepare_xy(train_df, feat_cols, "fuel_flow")
        model = train_model(model_key, X_tr, y_flow, feat_cols)

        X_r, _, y_kg_r, dur_r = prepare_xy(rank_df, feat_cols, "direct")
        X_f, _, y_kg_f, dur_f = prepare_xy(final_df, feat_cols, "direct")

        pred_r = predict_fuel_kg(model, X_r, dur_r, "fuel_flow")
        pred_f = predict_fuel_kg(model, X_f, dur_f, "fuel_flow")

        card = full_scorecard(name, rank_df, final_df, pred_r, pred_f)
        card["feature_count"] = len(feat_cols)
        card["model_key"] = model_key
        return card, pred_r, pred_f, model

    # Baseline (no mass)
    LOGGER.info("=== Baseline: LGBM FuelFlow (no mass) ===")
    card_a, pr_a, pf_a, model_a = train_eval("R3_baseline_no_mass", train, rank, final, feat_set_A)
    card_a["gate"] = "BASELINE"
    leaderboard.append(card_a)
    LOGGER.info("Baseline: combined=%.2f bias=%.2f", card_a["combined_rmse"], card_a["combined_bias"])

    # With R3 mass features
    LOGGER.info("=== R3: LGBM FuelFlow + Dynamic Mass ===")
    card_b, pr_b, pf_b, model_b = train_eval("R3_lgbm_dynamic_mass", train, rank, final, feat_set_B)
    card_b["delta_vs_baseline"] = card_b["combined_rmse"] - card_a["combined_rmse"]
    leaderboard.append(card_b)
    LOGGER.info("R3 mass: combined=%.2f bias=%.2f (delta=%.2f)",
                card_b["combined_rmse"], card_b["combined_bias"], card_b["delta_vs_baseline"])

    # CatBoost with mass
    LOGGER.info("=== R3: CatBoost FuelFlow + Dynamic Mass ===")
    card_cat, pr_cat, pf_cat, model_cat = train_eval("R3_cat_dynamic_mass", train, rank, final, feat_set_B, "cat")
    card_cat["delta_vs_baseline"] = card_cat["combined_rmse"] - card_a["combined_rmse"]
    leaderboard.append(card_cat)
    LOGGER.info("R3 Cat: combined=%.2f bias=%.2f", card_cat["combined_rmse"], card_cat["combined_bias"])

    # XGBoost with mass
    LOGGER.info("=== R3: XGBoost FuelFlow + Dynamic Mass ===")
    card_xgb, pr_xgb, pf_xgb, model_xgb = train_eval("R3_xgb_dynamic_mass", train, rank, final, feat_set_B, "xgb")
    card_xgb["delta_vs_baseline"] = card_xgb["combined_rmse"] - card_a["combined_rmse"]
    leaderboard.append(card_xgb)
    LOGGER.info("R3 XGB: combined=%.2f bias=%.2f", card_xgb["combined_rmse"], card_xgb["combined_bias"])

    # =========================================================================
    # Feature ablation: add mass features one sub-family at a time
    # =========================================================================
    # Sub-families
    mass_sub_families = {
        "tow_landing": ["r3_tow_kg", "r3_landing_mass_kg", "r3_tow_mtow_ratio"],
        "interval_stats": ["r3_mass_start_kg", "r3_mass_end_kg", "r3_mean_mass_kg",
                          "r3_min_mass_kg", "r3_max_mass_kg", "r3_mass_std_kg"],
        "consumption": ["r3_mass_consumed_kg", "r3_mass_rate_kgps", "r3_fuel_fraction",
                       "r3_remaining_fuel_frac"],
        "phase_mass": ["r3_phase_mass_kg", "r3_cruise_mass_kg", "r3_oew_base_kg"],
        "energy_mass": ["r3_mean_pe_j", "r3_mean_ke_j", "r3_wing_loading_cur",
                       "r3_fuel_mass_efficiency", "r3_cruise_mass_fuel_ratio"],
    }

    for family_name, family_cols in mass_sub_families.items():
        avail = [c for c in family_cols if c in train.columns]
        if not avail:
            continue
        feat_cols = list(dict.fromkeys(base_feat_cols + avail))
        card, _, _, _ = train_eval(
            f"R3_ablate_{family_name}", train, rank, final, feat_cols
        )
        card["delta_vs_baseline"] = card["combined_rmse"] - card_a["combined_rmse"]
        leaderboard.append(card)
        LOGGER.info("R3 ablate %s: combined=%.2f (delta=%+.2f)",
                    family_name, card["combined_rmse"], card["delta_vs_baseline"])

    # =========================================================================
    # R3 mass + R2 heavy specialist
    # =========================================================================
    LOGGER.info("=== R3 mass + R2 heavy specialist ===")

    # Build ensemble OOF with mass features for calibration
    # For the heavy specialist, we use the R2 pipeline which already includes descriptors
    # Train P1E calibration
    bundle = build_or_load_ensemble(train, force=force)
    feat_cols_base = bundle.feat_cols
    oof = bundle.oof_pred
    y_tr = bundle.y_train
    train_oof_df = train

    rank_b = ensure_features(rank, feat_cols_base)
    final_b = ensure_features(final, feat_cols_base)

    pred_r0 = predict_ensemble(bundle, rank_b)
    pred_f0 = predict_ensemble(bundle, final_b)

    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train_oof_df, y_tr, oof)
    pr_p1e = apply_calibrator(cal_phase, rank_b, pred_r0)
    pf_p1e = apply_calibrator(cal_phase, final_b, pred_f0)

    # R2 heavy Cat specialist on top of P1E + mass-enriched base
    try:
        spec_r2, r2_cols = train_heavy_specialist_r2(train, feat_cols_base, model_key="cat")
        pr = predict_heavy_routed_r2(spec_r2, feat_cols_base, rank_b, pr_p1e)
        pf = predict_heavy_routed_r2(spec_r2, feat_cols_base, final_b, pf_p1e)
        card_r3_heavy = full_scorecard("R3_mass_R2_heavy_cat", rank_b, final_b, pr, pf,
                                       hypothesis="R3 dynamic mass + R2 heavy specialist")
        card_r3_heavy["delta_vs_baseline"] = card_r3_heavy["combined_rmse"] - card_a["combined_rmse"]
        leaderboard.append(card_r3_heavy)
        LOGGER.info("R3+R2 heavy: combined=%.2f heavy=%.1f",
                    card_r3_heavy["combined_rmse"], card_r3_heavy["heavy_rmse"])
    except Exception as exc:
        LOGGER.warning("R3+R2 heavy failed: %s", exc)

    # =========================================================================
    # Save results
    # =========================================================================
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    r3_rows = [r for r in leaderboard if "R3" in r["variant"]]
    if r3_rows:
        pl.DataFrame(r3_rows).write_csv(OUT / "table_rmse_R3_mass.csv")
    lb.write_csv(OUT / "table_rmse_R3_full_leaderboard.csv")

    # Best R3 variant
    best_r3 = min((r for r in leaderboard if "R3" in r["variant"]), key=lambda x: x["combined_rmse"], default=card_a)
    summary = {
        "task": "R3",
        "best_variant": best_r3["variant"],
        "combined_rmse": best_r3["combined_rmse"],
        "rank_rmse": best_r3["rank_rmse"],
        "final_rmse": best_r3["final_rmse"],
        "bias": best_r3["combined_bias"],
        "delta_vs_225_25": best_r3["combined_rmse"] - 225.25,
        "delta_vs_226_19": best_r3["combined_rmse"] - 226.19,
        "delta_vs_228_25": best_r3["combined_rmse"] - 228.25,
        "heavy_rmse": best_r3["heavy_rmse"],
        "narrow_rmse": best_r3["narrow_rmse"],
        "baseline_no_mass": card_a["combined_rmse"],
        "delta_vs_baseline": best_r3["combined_rmse"] - card_a["combined_rmse"],
        "n_mass_features": len(mass_feat_cols_avail),
        "n_r3_variants": len(r3_rows),
    }
    (OUT / "r3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # =========================================================================
    # Mass profile visualization (for a few sample flights)
    # =========================================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mass vs flight fraction (sample)
    ax = axes[0, 0]
    sample = train.sample(n=min(2000, len(train)), seed=42)
    sf = sample["start_fraction_of_flight"].to_numpy()
    if "r3_mean_mass_kg" in sample.columns:
        mass = sample["r3_mean_mass_kg"].to_numpy()
        ax.scatter(sf, mass / 1000, alpha=0.3, s=5, c="steelblue")
        ax.set_xlabel("Fraction of Flight")
        ax.set_ylabel("Mean Mass (tons)")
        ax.set_title("R3 Dynamic Mass vs Flight Position (Train)")

    # Plot 2: Mass by aircraft type
    ax = axes[0, 1]
    for ac in ["A359", "B77W", "B744", "A320", "A20N"]:
        sub = sample.filter(pl.col("aircraft_type") == ac)
        if sub.is_empty() or "r3_mean_mass_kg" not in sub.columns:
            continue
        ax.hist(sub["r3_mean_mass_kg"].to_numpy() / 1000, bins=30, alpha=0.5, label=ac)
    ax.set_xlabel("Mean Mass (tons)")
    ax.set_ylabel("Count")
    ax.set_title("Mass Distribution by Aircraft Type")
    ax.legend(fontsize=8)

    # Plot 3: Fuel fraction by phase
    ax = axes[1, 0]
    if "r3_fuel_fraction" in sample.columns:
        cf = sample["cruise_fraction"].to_numpy().astype(np.float64)
        ff = sample["r3_fuel_fraction"].to_numpy()
        valid = np.isfinite(cf) & np.isfinite(ff)
        ax.scatter(cf[valid], ff[valid], alpha=0.3, s=5, c="darkorange")
        ax.set_xlabel("Cruise Fraction")
        ax.set_ylabel("Fuel Fraction Consumed")
        ax.set_title("Fuel Fraction vs Cruise Fraction")

    # Plot 4: RMSE by metrics
    ax = axes[1, 1]
    variants = lb.head(8).to_pandas()
    ax.barh(variants["variant"].str[:40], variants["combined_rmse"], color="steelblue")
    ax.axvline(228.25, color="red", ls="--", label="Official 228.25")
    ax.axvline(225.25, color="orange", ls=":", label="R2 best 225.25")
    ax.set_xlabel("Combined RMSE (kg)")
    ax.set_title("R3 Mass: Top Variants")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_r3_dynamic_mass.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("R3 DYNAMIC MASS SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=str))
    print("\nTop 10 by Combined RMSE:")
    for row in lb.head(10).iter_rows(named=True):
        delta = ""
        if "delta_vs_baseline" in row and row["delta_vs_baseline"] is not None:
            delta = f"  Δ={row['delta_vs_baseline']:+.2f}"
        print(f"  {row['variant']:<45s} Combined={row['combined_rmse']:.2f} Heavy={row.get('heavy_rmse', 0):.1f}{delta}")


if __name__ == "__main__":
    main()
