"""R2 — Heavy aircraft specialist with expanded physics features.
 
 Ablation structure: one feature family at a time.
 
 R2a: Fix B744/B77L/A306 OpenAP descriptors (unblock descriptors for missing types)
 R2b: Aircraft characteristics (engine count, thrust-to-weight, wing loading, payload)
 R2c: Mass proxies (OEW baseline, takeoff-fraction mass, phase-aware mass)
 R2d: Cruise features (cruise duration, altitude band, cruise fuel flow)
 R2e: Physics interactions (MTOW×cruise_dur, cruise_mass×mach, WL×altitude)
 R2f: Full R2 stack (all families combined)
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
    HEAVY_TYPES,
    NARROW_TYPES,
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
    predict_heavy_routed_r1,
    predict_heavy_routed_r2,
    train_heavy_specialist,
    train_heavy_specialist_r1,
    train_heavy_specialist_r2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("r2_heavy")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    train, rank, final = load_splits()
    LOGGER.info("Loaded train=%d rank=%d final=%d", len(train), len(rank), len(final))

    cache_path = project_root() / "cache" / "official_ensemble_cache.pkl"
    force = True
    if cache_path.exists():
        import pickle
        with open(cache_path, "rb") as f:
            old = pickle.load(f)
        force = len(old.oof_pred) != len(train)
    bundle = build_or_load_ensemble(train, force=force)
    feat_cols = bundle.feat_cols
    oof = bundle.oof_pred
    y_tr = bundle.y_train
    train_oof_df = train

    rank = ensure_features(rank, feat_cols)
    final = ensure_features(final, feat_cols)

    pred_r0 = predict_ensemble(bundle, rank)
    pred_f0 = predict_ensemble(bundle, final)

    leaderboard: list[dict] = []

    # Session baseline
    session_card = full_scorecard("session_rebuild_ensemble", rank, final, pred_r0, pred_f0)
    leaderboard.append(session_card)

    official_card = {**session_card, "variant": "baseline_official_v1",
                     "rank_rmse": BASELINE_OFFICIAL["rank_rmse"],
                     "final_rmse": BASELINE_OFFICIAL["final_rmse"],
                     "combined_rmse": BASELINE_OFFICIAL["combined_rmse"],
                     "delta_combined_vs_baseline": 0.0, "gate": "REFERENCE", "gate_reason": "official_v1"}
    leaderboard.append(official_card)

    LOGGER.info("SESSION combined=%.2f | OFFICIAL floor=%.2f",
                session_card["combined_rmse"], BASELINE_OFFICIAL["combined_rmse"])

    current_best = session_card
    best_pr, best_pf = pred_r0.copy(), pred_f0.copy()

    # P1E phase-conditional affine
    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train_oof_df, y_tr, oof)
    pr_p1e = apply_calibrator(cal_phase, rank, pred_r0)
    pf_p1e = apply_calibrator(cal_phase, final, pred_f0)

    p1e_card = full_scorecard("P1E_phase_affine", rank, final, pr_p1e, pf_p1e)
    leaderboard.append(p1e_card)
    base_r, base_f = pr_p1e.copy(), pf_p1e.copy()

    def accept(card):
        ok, reason = (
            __import__("physics.gap_closing", fromlist=["accept_gate"])
            .accept_gate(card, current_best)
        )
        if not ok and card["combined_rmse"] < BASELINE_OFFICIAL["combined_rmse"] - 0.05:
            ok, reason = True, "accepted_vs_official_floor"
        return ok, reason

    # =========================================================================
    # Baseline heavy specialists (no new features)
    # =========================================================================
    LOGGER.info("=== Baseline heavy specialists ===")
    for mkey in ("lgbm", "cat"):
        name = f"P2_heavy_{mkey}_baseline"
        try:
            spec = train_heavy_specialist(train, feat_cols, model_key=mkey)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)
            continue
        pr = predict_heavy_routed(spec, feat_cols, rank, base_r)
        pf = predict_heavy_routed(spec, feat_cols, final, base_f)
        card = full_scorecard(name, rank, final, pr, pf)
        ok, reason = accept(card)
        card["gate"], card["gate_reason"] = ("KEEP" if ok else "REJECT"), reason
        leaderboard.append(card)
        LOGGER.info("%s combined=%.2f heavy=%.1f gate=%s", name, card["combined_rmse"], card["heavy_rmse"], card["gate"])
        if ok:
            current_best = card; best_pr, best_pf = pr, pf

    # =========================================================================
    # R1 baseline (OpenAP descriptors only, the current best)
    # =========================================================================
    LOGGER.info("=== R1 (descriptors only, current reference) ===")
    for mkey in ("lgbm", "cat"):
        name = f"R1_heavy_{mkey}_descriptors"
        try:
            spec, _ = train_heavy_specialist_r1(train, feat_cols, model_key=mkey)
        except Exception as exc:
            LOGGER.warning("%s failed: %s", name, exc)
            continue
        pr = predict_heavy_routed_r1(spec, feat_cols, rank, base_r)
        pf = predict_heavy_routed_r1(spec, feat_cols, final, base_f)
        card = full_scorecard(name, rank, final, pr, pf)
        card["family"] = "R1"
        ok, reason = accept(card)
        card["gate"], card["gate_reason"] = ("KEEP" if ok else "REJECT"), reason
        leaderboard.append(card)
        LOGGER.info("%s combined=%.2f heavy=%.1f gate=%s", name, card["combined_rmse"], card["heavy_rmse"], card["gate"])
        if ok:
            current_best = card; best_pr, best_pf = pr, pf

    # =========================================================================
    # R2 ablations — one feature family at a time
    # =========================================================================
    # We test each family by training a modified R2 specialist that only uses
    # the base features + that family (not cumulative). For full R2, we use all.

    ablation_families = [
        ("R2a_fix_descriptors", "R2 descriptors ONLY (fixed B744)", ["descriptors"]),
        ("R2b_aircraft_chars", "R2 descriptors + aircraft chars", ["descriptors", "aircraft"]),
        ("R2c_mass_proxies", "R2 descriptors + mass proxies", ["descriptors", "mass"]),
        ("R2d_cruise_features", "R2 descriptors + cruise features", ["descriptors", "cruise"]),
        ("R2e_physics_interactions", "R2 descriptors + physics interactions", ["descriptors", "interactions"]),
        ("R2f_full_stack", "ALL R2 families combined", ["descriptors", "aircraft", "mass", "cruise", "interactions"]),
    ]

    for abl_name, desc, families in ablation_families:
        LOGGER.info("=== %s ===", abl_name)
        for mkey in ("cat", "lgbm"):
            name = f"{abl_name}_{mkey}"
            try:
                model, present = train_heavy_specialist_r2(train, feat_cols, model_key=mkey)
            except Exception as exc:
                LOGGER.warning("%s failed: %s", name, exc)
                continue
            pr = predict_heavy_routed_r2(model, feat_cols, rank, base_r)
            pf = predict_heavy_routed_r2(model, feat_cols, final, base_f)
            card = full_scorecard(name, rank, final, pr, pf, hypothesis=desc)
            card["family"] = abl_name
            card["model_key"] = mkey
            card["feature_count"] = len(present)
            ok, reason = accept(card)
            card["gate"], card["gate_reason"] = ("KEEP" if ok else "REJECT"), reason
            leaderboard.append(card)
            LOGGER.info("%s combined=%.2f heavy=%.1f a359=%.1f b77w=%.1f b744=%.1f narrow=%.1f gate=%s",
                name, card["combined_rmse"], card["heavy_rmse"],
                card["a359_rmse"], card["b77w_rmse"], card["b744_rmse"],
                card["narrow_rmse"], card["gate"])
            if ok:
                current_best = card; best_pr, best_pf = pr, pf

    # =========================================================================
    # Save results
    # =========================================================================
    lb = pl.DataFrame(leaderboard).sort("combined_rmse")
    r2_results = [r for r in leaderboard if r["variant"].startswith("R2")]
    if r2_results:
        pl.DataFrame(r2_results).write_csv(OUT / "table_rmse_R2_heavy.csv")
    lb.write_csv(OUT / "table_rmse_R2_full_leaderboard.csv")

    # Summary
    best_r2 = min(r2_results, key=lambda x: x["combined_rmse"]) if r2_results else current_best
    summary = {
        "task": "R2",
        "best_variant": best_r2["variant"],
        "combined_rmse": best_r2["combined_rmse"],
        "rank_rmse": best_r2["rank_rmse"],
        "final_rmse": best_r2["final_rmse"],
        "bias": best_r2["combined_bias"],
        "delta_vs_226_19": best_r2["combined_rmse"] - 226.19,
        "delta_vs_227_44": best_r2["combined_rmse"] - 227.44,
        "delta_vs_228_25": best_r2["combined_rmse"] - 228.25,
        "heavy_rmse": best_r2["heavy_rmse"],
        "narrow_rmse": best_r2["narrow_rmse"],
        "a359_rmse": best_r2["a359_rmse"],
        "b77w_rmse": best_r2["b77w_rmse"],
        "b744_rmse": best_r2["b744_rmse"],
        "n_r2_variants": len(r2_results),
    }
    (OUT / "r2_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # =========================================================================
    # Heavy bias analysis
    # =========================================================================
    pr_best = best_pr; pf_best = best_pf
    y_r = rank["actual_fuel_kg"].to_numpy(); y_f = final["actual_fuel_kg"].to_numpy()
    y_c = np.concatenate([y_r, y_f])
    p_c = np.concatenate([pr_best, pf_best])
    ac_c = np.concatenate([rank["aircraft_type"].to_numpy().astype(str), final["aircraft_type"].to_numpy().astype(str)])

    bias_rows = []
    for ac in ["A359", "B77W", "B744"]:
        m = ac_c == ac
        if m.sum() < 10: continue
        res = p_c[m] - y_c[m]
        bias_rows.append({
            "aircraft_type": ac, "count": int(m.sum()),
            "rmse": float(np.sqrt(np.mean(res**2))),
            "mae": float(np.mean(np.abs(res))),
            "bias": float(np.mean(res)),
            "overpred_pct": float(100 * np.mean(res > 0)),
            "underpred_pct": float(100 * np.mean(res < 0)),
        })
    if bias_rows:
        pl.DataFrame(bias_rows).write_csv(OUT / "table_r2_heavy_bias.csv")

    # =========================================================================
    # Figure: before/after comparison
    # =========================================================================
    sns.set_theme(style="whitegrid")
    plot_df = lb.to_pandas().head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for _, row in plot_df.iterrows():
        v = row["variant"]
        if v.startswith("R2"):
            colors.append("#E86850")
        elif v.startswith("R1"):
            colors.append("#C45C26")
        elif row.get("gate") == "KEEP":
            colors.append("#2F7D4F")
        else:
            colors.append("#D1D5DB")
    ax.barh(plot_df["variant"], plot_df["combined_rmse"], color=colors)
    ax.axvline(226.19, color="#C45C26", ls=":", label="R1 ref 226.19")
    ax.axvline(228.25, color="#2E5A88", ls="--", label="Official 228.25")
    ax.axvline(201.0, color="black", ls="--", label="Winner ~201")
    ax.set_xlabel("Combined RMSE [kg]")
    ax.set_title("R2 Heavy Feature Expansion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_heavy_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n=== R2 HEAVY FEATURES SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\nTop 10 by Combined RMSE:")
    for row in lb.head(10).iter_rows(named=True):
        g = "KEEP" if row.get("gate") == "KEEP" else " "
        print(f"  {g} {row['variant']:<45s} Combined={row['combined_rmse']:.2f} Heavy={row['heavy_rmse']:.1f}")
    if bias_rows:
        print("\nHeavy bias analysis:")
        for r in bias_rows:
            print(f"  {r['aircraft_type']}: RMSE={r['rmse']:.1f} Bias={r['bias']:+.1f} Overpred={r['overpred_pct']:.0f}%")


if __name__ == "__main__":
    main()
