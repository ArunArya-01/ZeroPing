"""R6 — Ultra-Long-Haul FuelFlow Specialist (>= 8h flights).

Hypothesis: A dedicated FuelFlow model trained exclusively on ultra-long-haul flights
(>= 8h) will reduce error in the 59.2% SSE regime without affecting shorter flights.

This is the SMALLEST possible implementation: one train function, one predict function,
no new features, no architecture changes. Identical pattern to the heavy specialist.
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
    BASELINE_OFFICIAL, HEAVY_TYPES, NARROW_TYPES,
    ConditionalAffineCalibrator, accept_gate, full_scorecard,
    group_phase, load_splits, est_flight_hours, ensure_features,
    train_haul_specialist, predict_haul_routed, ULTRA_LONG_HAUL_THRESHOLD,
)
from physics.mass_model import enrich_mass_from_columns, R3_MASS_FEATURES
from physics.official_benchmark import (
    ew_feature_cols, build_oof_matrix, choose_meta_on_train_folds,
    apply_bases, prepare_xy, train_model, predict_fuel_kg,
    CAT_FEATURES, LGBM_PARAMS, CAT_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r6_haul")
OUT = project_root() / "figures" / "r6_haul_specialist"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    train = enrich_mass_from_columns(train)
    rank = enrich_mass_from_columns(rank)
    final = enrich_mass_from_columns(final)

    base_feat_cols = ew_feature_cols(train)
    mass_cols = [c for c in R3_MASS_FEATURES if c in train.columns]
    feat_cols = list(dict.fromkeys(base_feat_cols + mass_cols))
    LOGGER.info("Features: %d", len(feat_cols))

    rank = ensure_features(rank, feat_cols)
    final = ensure_features(final, feat_cols)

    y_r = rank["actual_fuel_kg"].to_numpy()
    y_f = final["actual_fuel_kg"].to_numpy()
    ac_r = rank["aircraft_type"].to_numpy().astype(str)
    ac_f = final["aircraft_type"].to_numpy().astype(str)
    hours_r = est_flight_hours(rank)
    hours_f = est_flight_hours(final)

    # =========================================================================
    # Baseline: full ensemble OOF with mass features
    # =========================================================================
    cache_path = project_root() / "cache" / "official_ensemble_cache.pkl"
    force = False
    if cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            old = pickle.load(f)
        force = len(old.oof_pred) != len(train)
    if not cache_path.exists() or force:
        LOGGER.info("Building ensemble OOF (baseline)...")
        P, y, models = build_oof_matrix(train, feat_cols,
                                         [("xgb","direct"),("lgbm","direct"),("cat","direct"),
                                          ("xgb","fuel_flow"),("lgbm","fuel_flow"),("cat","fuel_flow")],
                                         n_splits=5)
        meta = fit_meta(P, y, "ridge")
        oof_base = np.asarray(meta.predict(P), dtype=np.float64)
    else:
        with open(cache_path, "rb") as f:
            bundle = pickle.load(f)
        models = [(m[0], m[1], m[2]) for m in bundle.full_models] if hasattr(bundle.full_models[0], '__iter__') else bundle.full_models
        P_r = apply_bases(models if isinstance(models, list) else bundle.full_models, rank, feat_cols)
        P_f = apply_bases(models if isinstance(models, list) else bundle.full_models, final, feat_cols)
        meta = bundle.meta
        oof_base = bundle.oof_pred
        pred_r0 = np.asarray(meta.predict(P_r), dtype=np.float64)
        pred_f0 = np.asarray(meta.predict(P_f), dtype=np.float64)

    # P1E calibration
    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train, bundle.y_train if hasattr(bundle, 'y_train') else y, oof_base)
    if 'pred_r0' not in dir():
        P_r = apply_bases(models if isinstance(models, list) else bundle.full_models, rank, feat_cols)
        P_f = apply_bases(models if isinstance(models, list) else bundle.full_models, final, feat_cols)
        pred_r0 = np.asarray(meta.predict(P_r), dtype=np.float64)
        pred_f0 = np.asarray(meta.predict(P_f), dtype=np.float64)

    pr_base = cal_phase.transform(rank, pred_r0)
    pf_base = cal_phase.transform(final, pred_f0)
    card_base = full_scorecard("R6_baseline_mass_P1E", rank, final, pr_base, pf_base)
    LOGGER.info("Baseline: combined=%.2f heavy=%.1f bias=%.2f", card_base["combined_rmse"], card_base["heavy_rmse"], card_base["combined_bias"])

    # =========================================================================
    # R6: Ultra-long-haul specialist
    # =========================================================================
    LOGGER.info("=== R6: Ultra-long-haul specialist ===")
    for mkey in ("lgbm", "cat"):
        name = f"R6_haul_specialist_{mkey}"
        try:
            spec = train_haul_specialist(train, feat_cols, model_key=mkey)
            pr = predict_haul_routed(spec, feat_cols, rank, pr_base)
            pf = predict_haul_routed(spec, feat_cols, final, pf_base)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)
            continue

        card = full_scorecard(name, rank, final, pr, pf,
                              hypothesis=f"FuelFlow specialist on >= {ULTRA_LONG_HAUL_THRESHOLD}h flights",
                              expected_gain="-2 to -8 kg")
        card["delta_vs_221_33"] = card["combined_rmse"] - 221.33
        LOGGER.info("%s: combined=%.2f heavy=%.1f a359=%.1f b77w=%.1f b744=%.1f bias=%.2f delta=%.2f",
                    name, card["combined_rmse"], card["heavy_rmse"],
                    card["a359_rmse"], card["b77w_rmse"], card["b744_rmse"],
                    card["combined_bias"], card["delta_vs_221_33"])

        # Keep metrics for best variant (CatBoost preferred)
        if mkey == "cat" or ("best_card" not in dir() and "pr_best" not in dir()):
            best_card = card
            pr_best = pr.copy()
            pf_best = pf.copy()
            best_name = name

    # =========================================================================
    # Use best variant for figures
    # =========================================================================
    p_c_base = np.concatenate([pr_base, pf_base])
    p_c_r6 = np.concatenate([pr_best, pf_best])
    y_c = np.concatenate([y_r, y_f])
    ac_c = np.concatenate([ac_r, ac_f])
    hours_c = np.concatenate([hours_r, hours_f])
    phase_c = np.concatenate([
        np.array([__import__("physics.gap_closing", fromlist=["dominant_phase_row"]).dominant_phase_row(r)
                  for r in rank.iter_rows(named=True)]),
        np.array([__import__("physics.gap_closing", fromlist=["dominant_phase_row"]).dominant_phase_row(r)
                  for r in final.iter_rows(named=True)])
    ])

    # =========================================================================
    # FIGURE 1: Overall RMSE Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Rank", "Final", "Combined"]
    base_vals = [card_base["rank_rmse"], card_base["final_rmse"], card_base["combined_rmse"]]
    r6_vals = [best_card["rank_rmse"], best_card["final_rmse"], best_card["combined_rmse"]]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, base_vals, w, label="Baseline (221.33)", color="#2E5A88", edgecolor="white")
    ax.bar(x + w/2, r6_vals, w, label=f"{best_name}", color="#E86850", edgecolor="white")
    ax.axhline(221.33, color="#2E5A88", ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("RMSE (kg)"); ax.set_title("R6: Overall RMSE Comparison")
    ax.legend(fontsize=9)
    for i, (b, r) in enumerate(zip(base_vals, r6_vals)):
        ax.text(i - w/2, b + 1, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, r + 1, f"{r:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "01_overall_rmse.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 2: Aircraft Performance
    # =========================================================================
    ac_groups = ["heavy", "narrow", "A359", "B77W", "B744"]
    base_ac = [card_base[f"{g}_rmse"] if f"{g}_rmse" in card_base else float("nan") for g in ac_groups]
    r6_ac = [best_card[f"{g}_rmse"] if f"{g}_rmse" in best_card else float("nan") for g in ac_groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ac_groups))
    ax.bar(x - w/2, base_ac, w, label="Baseline", color="#2E5A88", edgecolor="white")
    ax.bar(x + w/2, r6_ac, w, label="R6 Haul Specialist", color="#E86850", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(ac_groups)
    ax.set_ylabel("RMSE (kg)"); ax.set_title("R6: Aircraft Performance")
    ax.legend()
    for i, (b, r) in enumerate(zip(base_ac, r6_ac)):
        if not np.isnan(b): ax.text(i - w/2, b + 2, f"{b:.0f}", ha="center", fontsize=7)
        if not np.isnan(r): ax.text(i + w/2, r + 2, f"{r:.0f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "02_aircraft_rmse.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 3: Flight Duration Performance
    # =========================================================================
    dur_buckets = [("<2h", 0, 2), ("2-4h", 2, 4), ("4-8h", 4, 8), (">=8h", 8, 99)]
    dur_base = []; dur_r6 = []
    dur_labels = []
    for label, lo, hi in dur_buckets:
        m = (hours_c >= lo) & (hours_c < hi)
        if m.sum() < 20: continue
        dur_labels.append(label)
        dur_base.append(float(np.sqrt(np.mean((p_c_base[m] - y_c[m])**2))))
        dur_r6.append(float(np.sqrt(np.mean((p_c_r6[m] - y_c[m])**2))))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(dur_labels))
    ax.bar(x - w/2, dur_base, w, label="Baseline", color="#2E5A88", edgecolor="white")
    ax.bar(x + w/2, dur_r6, w, label="R6", color="#E86850", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(dur_labels)
    ax.set_ylabel("RMSE (kg)"); ax.set_title("R6: Flight Duration Performance")
    ax.legend()
    for i, (b, r) in enumerate(zip(dur_base, dur_r6)):
        ax.text(i - w/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, r + 1, f"{r:.0f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "03_duration_rmse.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 4: Bias Analysis by Aircraft
    # =========================================================================
    bias_groups = ["Heavy", "Narrow", "A359", "B77W", "B744"]
    fig, ax = plt.subplots(figsize=(8, 5))
    base_bias = []; r6_bias = []
    for label, mask_fn in [("Heavy", lambda a: np.isin(a, list(HEAVY_TYPES))),
                            ("Narrow", lambda a: np.isin(a, list(NARROW_TYPES))),
                            ("A359", lambda a: a == "A359"),
                            ("B77W", lambda a: a == "B77W"),
                            ("B744", lambda a: a == "B744")]:
        m = mask_fn(ac_c)
        if m.sum() < 10: continue
        base_bias.append(float(np.mean(p_c_base[m] - y_c[m])))
        r6_bias.append(float(np.mean(p_c_r6[m] - y_c[m])))
    x = np.arange(len(bias_groups))
    ax.barh(x, base_bias, w, label="Baseline", color="#2E5A88", edgecolor="white")
    ax.barh(x + w/2, r6_bias, w/2, label="R6", color="#E86850", edgecolor="white")
    ax.set_yticks(x + w/4); ax.set_yticklabels(bias_groups)
    ax.set_xlabel("Bias (kg)"); ax.set_title("R6: Bias by Aircraft Family")
    ax.axvline(0, color="black", ls="-", alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "04_bias.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 5: Error Distribution
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    err_base = p_c_base - y_c
    err_r6 = p_c_r6 - y_c
    axes[0].hist(err_base, bins=80, color="#2E5A88", alpha=0.7, edgecolor="white", density=True)
    axes[0].axvline(0, color="black", ls="--")
    axes[0].set_xlabel("Prediction - Actual (kg)"); axes[0].set_ylabel("Density")
    axes[0].set_title(f"Baseline (bias={np.mean(err_base):+.1f})")
    axes[1].hist(err_r6, bins=80, color="#E86850", alpha=0.7, edgecolor="white", density=True)
    axes[1].axvline(0, color="black", ls="--")
    axes[1].set_xlabel("Prediction - Actual (kg)"); axes[1].set_ylabel("Density")
    axes[1].set_title(f"R6 Haul Specialist (bias={np.mean(err_r6):+.1f})")
    fig.suptitle("R6: Prediction Error Distribution", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "05_error_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 6: Predicted vs Actual
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    max_val = max(y_c.max(), p_c_base.max()) * 1.05
    for ax, p, title, color in [(axes[0], p_c_base, "Baseline", "#2E5A88"),
                                   (axes[1], p_c_r6, "R6 Haul Specialist", "#E86850")]:
        ax.scatter(y_c, p, alpha=0.15, s=3, c=color, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5)
        ax.set_xlabel("Actual Fuel (kg)"); ax.set_ylabel("Predicted Fuel (kg)")
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / "06_predicted_vs_actual.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 7: Residual Plot
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    heavy_mask = np.isin(ac_c, list(HEAVY_TYPES))
    for ax, p, title, color in [(axes[0], p_c_base, "Baseline", "#2E5A88"),
                                   (axes[1], p_c_r6, "R6 Haul Specialist", "#E86850")]:
        ax.scatter(p[~heavy_mask], (p - y_c)[~heavy_mask], alpha=0.15, s=3, c="gray", label="Non-heavy")
        ax.scatter(p[heavy_mask], (p - y_c)[heavy_mask], alpha=0.3, s=5, c="red", label="Heavy")
        ax.axhline(0, color="black", ls="--", alpha=0.5)
        ax.set_xlabel("Predicted Fuel (kg)"); ax.set_ylabel("Residual (kg)")
        ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "07_residual_plot.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 8: Feature Importance (train a CatBoost on full train for SHAP)
    # =========================================================================
    try:
        X_tr, y_flow, y_kg, dur = prepare_xy(train, feat_cols, "fuel_flow")
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(**CAT_PARAMS)
        cb.fit(X_tr, y_flow, silent=True)
        importances = cb.get_feature_importance()
        fi = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)[:20]
        fig, ax = plt.subplots(figsize=(8, 7))
        names, vals = zip(*reversed(fi))
        ax.barh(names, vals, color="#2E5A88", edgecolor="white")
        ax.set_xlabel("Feature Importance"); ax.set_title("R6: Top 20 Feature Importances (CatBoost)")
        fig.tight_layout(); fig.savefig(OUT / "08_feature_importance.png", dpi=200, bbox_inches="tight"); plt.close(fig)

        # FIGURE 9: SHAP Summary (top 15)
        try:
            import shap
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder
            num_cols = [c for c in feat_cols if c not in CAT_FEATURES]
            X_samp = X_tr[num_cols].iloc[:2000]
            imp = SimpleImputer(strategy="median").fit(X_samp)
            X_imp = imp.transform(X_samp)
            explainer = shap.TreeExplainer(cb, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(X_imp)
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_imp, feature_names=num_cols[:15], show=False, max_display=15)
            fig.tight_layout(); fig.savefig(OUT / "09_shap_summary.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        except Exception as exc:
            LOGGER.warning("SHAP failed (non-fatal): %s", exc)
    except Exception as exc:
        LOGGER.warning("Feature importance failed: %s", exc)

    # =========================================================================
    # FIGURE 10: SSE Breakdown
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    sse_total = float(((p_c_base - y_c)**2).sum())
    sse_parts = {}
    for label, mask_fn in [("A359", lambda a: a == "A359"), ("B77W", lambda a: a == "B77W"),
                            ("B744", lambda a: a == "B744"),
                            ("Other Heavy", lambda a: np.isin(a, list(HEAVY_TYPES)) & ~np.isin(a, ["A359","B77W","B744"])),
                            ("Narrowbody", lambda a: np.isin(a, list(NARROW_TYPES)))]:
        m = mask_fn(ac_c)
        sse_parts[label] = float(((p_c_base[m] - y_c[m])**2).sum()) / sse_total * 100
    parts = sorted(sse_parts.items(), key=lambda x: x[1], reverse=True)
    ax.barh([p[0] for p in parts], [p[1] for p in parts], color="#E86850", edgecolor="white")
    ax.set_xlabel("SSE Contribution (%)"); ax.set_title("R6: Aircraft SSE Breakdown (Baseline)")
    for i, (_, v) in enumerate(parts):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "10_sse_breakdown.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # =========================================================================
    # FIGURE 11: Before/After Metrics Table (CSV)
    # =========================================================================
    metrics_table = [
        {"metric": "Rank RMSE", "baseline": card_base["rank_rmse"], "r6": best_card["rank_rmse"],
         "delta": best_card["rank_rmse"] - card_base["rank_rmse"]},
        {"metric": "Final RMSE", "baseline": card_base["final_rmse"], "r6": best_card["final_rmse"],
         "delta": best_card["final_rmse"] - card_base["final_rmse"]},
        {"metric": "Combined RMSE", "baseline": card_base["combined_rmse"], "r6": best_card["combined_rmse"],
         "delta": best_card["combined_rmse"] - card_base["combined_rmse"]},
        {"metric": "Bias", "baseline": card_base["combined_bias"], "r6": best_card["combined_bias"],
         "delta": best_card["combined_bias"] - card_base["combined_bias"]},
        {"metric": "Heavy RMSE", "baseline": card_base["heavy_rmse"], "r6": best_card["heavy_rmse"],
         "delta": best_card["heavy_rmse"] - card_base["heavy_rmse"]},
        {"metric": "Narrow RMSE", "baseline": card_base["narrow_rmse"], "r6": best_card["narrow_rmse"],
         "delta": best_card["narrow_rmse"] - card_base["narrow_rmse"]},
        {"metric": "A359 RMSE", "baseline": card_base["a359_rmse"], "r6": best_card["a359_rmse"],
         "delta": best_card["a359_rmse"] - card_base["a359_rmse"]},
        {"metric": "B77W RMSE", "baseline": card_base["b77w_rmse"], "r6": best_card["b77w_rmse"],
         "delta": best_card["b77w_rmse"] - card_base["b77w_rmse"]},
        {"metric": "B744 RMSE", "baseline": card_base["b744_rmse"], "r6": best_card["b744_rmse"],
         "delta": best_card["b744_rmse"] - card_base["b744_rmse"]},
    ]
    pl.DataFrame(metrics_table).write_csv(OUT / "12_before_after_metrics.csv")
    pl.DataFrame([card_base, best_card]).write_csv(OUT / "11_metrics_table.csv")

    # =========================================================================
    # experiment_summary.md and technical_report.md
    # =========================================================================
    delta_combined = best_card["combined_rmse"] - card_base["combined_rmse"]
    verdict = "KEEP" if delta_combined < -0.5 else "NO-GO" if abs(delta_combined) < 0.5 else "REJECT"

    md = f"""# R6 — Ultra-Long-Haul FuelFlow Specialist

## Problem Statement

Ultra-long-haul flights (>= {ULTRA_LONG_HAUL_THRESHOLD}h) contribute ~59% of train SSE
while representing only ~33% of intervals. Their per-interval RMSE (335 kg) is 2.3x higher
than 2-4h flights (143 kg). The baseline ensemble has no haul-aware routing.

## Hypothesis

A dedicated FuelFlow model trained exclusively on >= {ULTRA_LONG_HAUL_THRESHOLD}h flights
will reduce error in this dominant regime without affecting shorter flights.

## Implementation

- **One new function** in `gap_closing.py`: `train_haul_specialist()` (identical pattern to heavy specialist)
- **One new routing function**: `predict_haul_routed()`  
- **No new features**, no architecture changes
- **Train on train data only**

## Results

| Metric | Baseline | R6 | Delta |
|--------|----------|----|-------|
| Combined RMSE | {card_base["combined_rmse"]:.2f} | {best_card["combined_rmse"]:.2f} | {delta_combined:+.2f} |
| Heavy RMSE | {card_base["heavy_rmse"]:.1f} | {best_card["heavy_rmse"]:.1f} | {best_card["heavy_rmse"] - card_base["heavy_rmse"]:+.1f} |
| Narrow RMSE | {card_base["narrow_rmse"]:.1f} | {best_card["narrow_rmse"]:.1f} | {best_card["narrow_rmse"] - card_base["narrow_rmse"]:+.1f} |
| Bias | {card_base["combined_bias"]:+.1f} | {best_card["combined_bias"]:+.1f} | {best_card["combined_bias"] - card_base["combined_bias"]:+.1f} |

## Decision

**{verdict}** ({delta_combined:+.2f} kg delta vs baseline)
"""

    (OUT / "experiment_summary.md").write_text(md)

    tech_md = md + f"""
## Technical Details

### Specialist Architecture
- CatBoost FuelFlow regressor, {ULTRA_LONG_HAUL_THRESHOLD}h threshold
- Features: 60 (39 base + 21 R3 mass)
- Trained on flight-level OOF splits to prevent leakage
- Hard-routed: >= {ULTRA_LONG_HAUL_THRESHOLD}h → specialist, otherwise → baseline

### Error Analysis
- >= 8h flights targeted by specialist
- No change to < 8h flight predictions
- {len(feat_cols)} features per model

### Limitations
- Single threshold ({ULTRA_LONG_HAUL_THRESHOLD}h) — no gradual blending
- No interaction with heavy specialist (heavily overlapping populations)
- CatBoost FuelFlow only — not extended to ensemble base models

### Recommended Next Experiment
{'- Investigate why the specialist does not outperform the baseline' if delta_combined > -0.5 else '- Merge into production pipeline\n- Consider per-haul-bucket specialists or graduated routing'}
"""
    (OUT / "technical_report.md").write_text(tech_md)

    # Print final summary
    print("\n" + "=" * 70)
    print(f"R6 HAUL SPECIALIST — {verdict}")
    print("=" * 70)
    print(f"  Baseline Combined RMSE: {card_base['combined_rmse']:.2f}")
    print(f"  R6 Combined RMSE:       {best_card['combined_rmse']:.2f}")
    print(f"  Delta:                  {delta_combined:+.2f} kg")
    print(f"  Decision:               {verdict}")
    print(f"\n  Deliverables: {OUT}")


if __name__ == "__main__":
    main()
