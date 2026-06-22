"""
AeroTwin V4 — Heuristic Mass Features (winner-inspired) + Mass Ablation.

Task 1: Implement per-aircraft OpenAP MTOW/MLW/OEW -> takeoff/landing/mass trajectory heuristics.
Adds 9 mass-derived features.

Task 4: Mass ablation (A: Energy+Weather, B: +Mass, C: Mass only, D: Mass+OpenAP)
with flight-level split + 10k flight-clustered bootstrap significance.

Run:
    python notebooks/09_mass_features.py

Outputs:
    featured_dataset_mass.parquet
    figures/table_mass_ablation.csv
    figures/fig_mass_ablation.png
    (plus bootstrap fig)
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
    MASS_FEATURES,
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
from physics.weather_features import WEATHER_FEATURES, enrich_weather_from_columns

try:
    from openap import prop
except ImportError as e:
    raise RuntimeError("openap required: pip install -r requirements.txt") from e

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)
MASS_PARQUET = project_root() / "featured_dataset_mass.parquet"

MODELS = ["xgb", "lgbm", "cat"]
MODEL_LABELS = {"rf": "RF", "xgb": "XGB", "lgbm": "LGBM", "cat": "CatBoost"}


def get_ac_masses(ac_type: str) -> dict[str, float]:
    """Extract MTOW/MLW/OEW from OpenAP; robust fallbacks."""
    ac_type = str(ac_type) if ac_type else "A320"
    try:
        ac = prop.aircraft(ac_type)
        mtow = float(
            ac.get("mtow")
            or ac.get("MTOW")
            or (ac.get("limits") or {}).get("MTOW")
            or 200_000.0
        )
        mlw = float(
            ac.get("mlw")
            or ac.get("MLW")
            or (ac.get("limits") or {}).get("MLW")
            or 0.82 * mtow
        )
        oew = float(
            ac.get("oew")
            or ac.get("OEW")
            or (ac.get("limits") or {}).get("OEW")
            or 0.52 * mtow
        )
    except Exception:
        mtow, mlw, oew = 200_000.0, 164_000.0, 104_000.0
    return {"mtow": mtow, "mlw": mlw, "oew": oew}


def add_heuristic_mass_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add MTOW/MLW/OEW + interval mass trajectory heuristics per aircraft_type."""
    print("  Computing per-aircraft OpenAP mass parameters (MTOW/MLW/OEW)...")
    types = df["aircraft_type"].drop_nulls().unique().to_list()
    mass_rows = []
    for t in types:
        m = get_ac_masses(str(t))
        mass_rows.append({"aircraft_type": str(t), **m})
    mass_lut = pl.DataFrame(mass_rows)

    df = df.join(mass_lut, on="aircraft_type", how="left")

    # robust defaults
    df = df.with_columns(
        pl.col("mtow").fill_null(200_000.0),
        pl.col("mlw").fill_null(pl.col("mtow") * 0.82),
        pl.col("oew").fill_null(pl.col("mtow") * 0.52),
    )

    sf = pl.col("start_fraction_of_flight").fill_null(0.5)
    ef = pl.col("end_fraction_of_flight").fill_null(0.5)
    dur = pl.col("duration_s").fill_null(300.0).clip(lower_bound=1.0)

    df = df.with_columns(
        (0.8 * pl.col("mtow")).alias("takeoff_mass_est"),
        (0.5 * (pl.col("mlw") + pl.col("oew"))).alias("landing_mass_est"),
    )

    df = df.with_columns(
        (pl.col("takeoff_mass_est") * (1.0 - sf) + pl.col("landing_mass_est") * sf).alias("mass_start"),
        (pl.col("takeoff_mass_est") * (1.0 - ef) + pl.col("landing_mass_est") * ef).alias("mass_end"),
    )

    df = df.with_columns(
        ((pl.col("mass_start") + pl.col("mass_end")) / 2.0).alias("mean_mass"),
        ((pl.col("mass_start") - pl.col("mass_end")).abs() / (12.0 ** 0.5)).alias("std_mass"),
        ((pl.col("mass_end") - pl.col("mass_start")) / dur).alias("mass_slope"),
        (pl.col("mass_start") - pl.col("mass_end")).alias("mass_consumed_est"),
    )

    # keep only the declared MASS_FEATURES (drop internal temps)
    drop_temps = [c for c in ["mass_start", "mass_end"] if c in df.columns]
    if drop_temps:
        df = df.drop(drop_temps)

    # ensure all MASS_FEATURES exist (even if some ac missing)
    for c in MASS_FEATURES:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(c))

    added = [c for c in MASS_FEATURES if c in df.columns]
    print(f"  Added {len(added)} mass features: {added}")
    return df


def ensure_mass_enriched() -> pl.DataFrame:
    """Load original, add mass (and ensure E/W for self-contained), write _mass.parquet."""
    print(f"Loading base {project_root() / 'featured_dataset.parquet'} ...")
    df = pl.read_parquet(project_root() / "featured_dataset.parquet")
    print(f"  {len(df):,} rows, {len(df.columns)} cols, {df['flight_id'].n_unique():,} flights")

    changed = False
    if "mean_mass" not in df.columns:
        df = add_heuristic_mass_features(df)
        changed = True

    if "mean_specific_energy_jpkg" not in df.columns:
        print("  Also enriching Energy features (fast path)...")
        df = enrich_from_columns(df)
        changed = True
    if "headwind_mps" not in df.columns:
        print("  Also enriching Weather features (fast path)...")
        df = enrich_weather_from_columns(df)
        changed = True

    if changed:
        df.write_parquet(MASS_PARQUET)
        print(f"Saved -> {MASS_PARQUET} ({len(df.columns)} columns)")
    else:
        print("  Mass features already present.")
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
        try:
            pred = train_predict(mk, feature_cols, X_tr, X_te, y_tr, residual, None, physics_test)
        except Exception as e:
            print(f"    WARN: {mk} failed on {approach}: {e}")
            pred = np.full(len(y_test), y_test.mean())
        metrics[mk] = evaluate(y_test, pred)
        metrics[mk]["approach"] = approach
        metrics[mk]["model"] = MODEL_LABELS.get(mk, mk.upper())
        preds[mk] = pred
    return metrics, preds


def metrics_rows(all_metrics: dict) -> list[dict]:
    rows = []
    for approach, by_m in all_metrics.items():
        for mk, m in by_m.items():
            rows.append(
                {"approach": approach, "model": m.get("model", mk), "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]}
            )
    return rows


def run_mass_ablation(pdf, train_idx, test_idx, y_train, y_test, physics_test, test_fids):
    """Task 4: Mass ablation with bootstrap significance."""
    print("\n" + "=" * 70)
    print("TASK 4 — Mass Ablation")
    print("=" * 70)

    energy = avail(ENERGY_FEATURES, pdf)  # wait pdf is pandas, use cols from df later
    # caller passes, recompute inside

    # We'll receive energy/weather/mass lists from caller

    # Recompute here? No, pass in
    # Use globals from closure? Better to compute before call.

    # This func will be defined inside main after lists ready. See main.
    pass


def main() -> None:
    print("=" * 70)
    print("AeroTwin V4 — Heuristic Mass Features + Mass Ablation")
    print("=" * 70)

    # 1. Build / ensure mass-augmented dataset (full, with E/W too for convenience)
    _ = ensure_mass_enriched()

    # 2. Load cleaned for experiments (drops rows w/o physics/residual etc)
    df = load_and_clean(MASS_PARQUET)
    print(f"Cleaned for modeling: {len(df):,} intervals, {df['flight_id'].n_unique():,} flights")

    # Verify mass features present
    mass_present = avail(MASS_FEATURES, df)
    print(f"  Mass features available: {len(mass_present)}/{len(MASS_FEATURES)} -> {mass_present}")

    pdf = df.to_pandas()
    fids = pdf["flight_id"].to_numpy()
    train_idx, test_idx, _, _ = flight_level_split(fids)
    y_train = pdf["actual_fuel_kg"].to_numpy()[train_idx]
    y_test = pdf["actual_fuel_kg"].to_numpy()[test_idx]
    physics_test = pdf["physics_fuel_kg"].to_numpy()[test_idx]
    test_fids = pdf["flight_id"].to_numpy()[test_idx]

    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    massf = avail(MASS_FEATURES, df)
    ew = energy + weather

    print(f"  Energy: {len(energy)} | Weather: {len(weather)} | Mass: {len(massf)}")

    # Baseline OpenAP Hybrid (for ref)
    base_feats = feats(df, [], physics=True)
    _, base_preds = run_tree_models(
        "OpenAP Hybrid", base_feats, pdf, train_idx, test_idx, y_train, y_test
    )
    baseline_err = np.abs(y_test - base_preds["xgb"])
    baseline_mae = float(baseline_err.mean())
    print(f"OpenAP Hybrid (XGB) baseline MAE: {baseline_mae:.2f} kg")

    # Current best approx: Energy+Weather (on mass parquet same)
    ew_feats = feats(df, ew, physics=True)
    _, ew_preds = run_tree_models(
        "Energy+Weather", ew_feats, pdf, train_idx, test_idx, y_train, y_test
    )
    ew_err = np.abs(y_test - ew_preds["xgb"])
    print(f"Energy+Weather (XGB) MAE: {float(ew_err.mean()):.2f} kg")

    # === TASK 4: Mass Ablation ===
    print("\n" + "=" * 70)
    print("TASK 4 — Mass Ablation (A/B/C/D)")
    print("=" * 70)

    # Define the 4 models + ref
    # A: Energy+Weather (current)
    # B: Energy+Weather + Mass
    # C: Mass only  (base + mass; no physics_fuel to isolate mass signal)
    # D: Mass + OpenAP  (physics_fuel + mass + base)
    approaches = {
        "OpenAP Hybrid (ref)": (base_feats, False),
        "A: Energy+Weather": (feats(df, ew, True), False),
        "B: Energy+Weather+Mass": (feats(df, ew + massf, True), False),
        "C: Mass only": (list(BASE_NUMERIC) + massf + CATEGORICAL, False),  # direct, no physics
        "D: Mass+OpenAP": (list(BASE_NUMERIC) + massf + ["physics_fuel_kg"] + CATEGORICAL, False),
    }

    all_metrics: dict = {}
    sig_rows: list[dict] = []
    boot_dists: dict[str, np.ndarray] = {}
    best_mae = float("inf")

    for name, (fcols, residual) in approaches.items():
        print(f"  {name} ...", flush=True)
        mets, preds = run_tree_models(
            name, fcols, pdf, train_idx, test_idx, y_train, y_test, residual, physics_test
        )
        all_metrics[name] = mets
        xgb_mae = mets["xgb"]["mae"]
        best_mae = min(best_mae, xgb_mae)

        if name != "A: Energy+Weather":
            err = np.abs(y_test - preds["xgb"])
            # compare all interesting to A (Energy+Weather) as the "current best"
            sig = significance_test(err, ew_err, test_fids, name, "A: Energy+Weather")
            boot_dists[name] = sig.pop("bootstrap_dist")
            sig_rows.append(sig)
            print(
                f"    XGB MAE={xgb_mae:.2f}  Δ={sig['delta_mae']:+.2f}  "
                f"CI=[{sig['ci_lower']:+.2f},{sig['ci_upper']:+.2f}]  p_boot={sig['bootstrap_p']:.4f}  → {sig['interpretation']}"
            )
        else:
            print(f"    XGB MAE={xgb_mae:.2f}  (current best ref)")

    res_df = pl.DataFrame(metrics_rows(all_metrics)).sort(["approach", "mae"])
    res_df.write_csv(OUT / "table_mass_ablation.csv")
    sig_df = pl.DataFrame(sig_rows) if sig_rows else pl.DataFrame()
    if not sig_df.is_empty():
        sig_df.write_csv(OUT / "table_significance_mass_ablation.csv")
    plot_comparison_bars(res_df, "V4 Mass Ablation (XGB/LGBM/CatBoost)", OUT / "fig_mass_ablation.png")

    if boot_dists:
        n = len(boot_dists)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (lbl, dist) in zip(axes, boot_dists.items()):
            lo, hi = np.percentile(dist, [2.5, 97.5])
            ax.hist(dist, bins=50, color="#e74c3c", alpha=0.85, density=True)
            ax.axvline(0, color="k", lw=1.2)
            ax.axvspan(lo, hi, alpha=0.2, color="green")
            ax.set_title(lbl[:30])
            ax.set_xlabel("ΔMAE [kg] vs Energy+Weather")
        fig.suptitle(f"Mass Ablation — flight-clustered bootstrap ({N_BOOTSTRAP:,} resamples)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / "fig_mass_ablation_bootstrap.png", bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved: table_mass_ablation.csv , fig_mass_ablation.png (and _bootstrap)")

    # Quick mass feature stats (scientific)
    print("\nMass feature summary (on cleaned):")
    print(df.select(mass_present).describe())

    # Also save a small leaderboard slice for v4
    lb_path = OUT / "leaderboard_v4_partial_mass.csv"
    lb = res_df.filter(pl.col("model") == "XGB").with_columns(
        pl.lit("mass_ablation").alias("experiment")
    )
    lb.write_csv(lb_path)
    print(f"Partial v4 leaderboard slice: {lb_path}")

    print("\n" + "=" * 70)
    print("V4 MASS TASKS COMPLETE (09_mass_features)")
    print("Next: run 10_fuel_flow_target.py and 11_vertical_embeddings.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
