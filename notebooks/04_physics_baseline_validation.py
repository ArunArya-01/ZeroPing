from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from data import AeroDataLoader, DEFAULT_RANDOM_SEED
from physics.openap_baseline import (
    predict_fuel_intervals, 
    compute_physics_errors,
    classify_interval_phase
)

logging.basicConfig(level=logging.WARNING)
for lib in ["httpx", "httpcore", "huggingface_hub"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
OUT = Path.cwd() / "figures"
OUT.mkdir(exist_ok=True)

N_SAMPLES = 50  # 50-100 practical; scale up for production paper run
SEED = DEFAULT_RANDOM_SEED

loader = AeroDataLoader()
fuel_all = loader.get_fuel_labels()

print("=" * 60)
print(f"PHYSICS BASELINE + RESIDUALS (sample of {N_SAMPLES} usable flights)")
print("=" * 60)

usable = loader.get_usable_flight_ids(split="train")
random.seed(SEED)
sampled_fids = random.sample(usable, min(N_SAMPLES, len(usable)))

all_preds_list = []
naive_rates = []  # for simple naive: fuel_kg / dur_min

for fid in sampled_fids:
    try:
        tr = loader.load_flight_by_id(fid)
        ac = tr["typecode"][0] if "typecode" in tr.columns and tr["typecode"][0] is not None else "A320"
        fu = fuel_all.filter(pl.col("flight_id") == fid).sort("start")
        if fu.is_empty() or tr.is_empty():
            continue

        pr = predict_fuel_intervals(tr, fu, ac_type=ac)
        if pr.is_empty():
            continue

        pr = pr.with_columns(
            pl.lit(fid).alias("flight_id"),
            pl.lit(ac).alias("aircraft_type"),
            ((pl.col("end") - pl.col("start")).dt.total_seconds() / 60.0).alias("dur_min")
        )

        # Add naive baseline: use average rate (kg per minute) from this flight's labels as simple proxy
        # (in full run we could use global mean from all train)
        if pr["dur_min"].sum() > 0:
            rate = pr["actual_fuel_kg"].sum() / pr["dur_min"].sum()
        else:
            rate = 200.0  # fallback median-ish
        naive_rates.append(rate)

        # naive per row
        pr = pr.with_columns(
            (pl.col("dur_min") * rate).alias("naive_fuel_kg")
        )

        all_preds_list.append(pr)

    except Exception as exc:
        print(f"  skip {fid}: {exc}")
        continue

if not all_preds_list:
    print("No preds collected.")
else:
    preds = pl.concat(all_preds_list, how="diagonal_relaxed")
    print(f"Total intervals analyzed: {len(preds)}")

    # Physics errors
    phys_errs = compute_physics_errors(preds)
    print("\n[Physics Errors]")
    print(phys_errs.get("overall"))

    # Add phase if not present (should be from enhanced baseline)
    if "phase" not in preds.columns:
        # fallback recompute (rare)
        pass

    # Naive errors (duration * avg_rate)
    naive_err = (preds["naive_fuel_kg"] - preds["actual_fuel_kg"]).abs()
    print(f"\n[Naive (dur * flight-specific rate) Errors] MAE={naive_err.mean():.1f} RMSE={(naive_err**2).mean()**0.5:.1f}")

    # By phase breakdown (physics)
    if "phase" in preds.columns:
        print("\n[Physics MAE by phase]")
        for ph in ["climb", "cruise", "descent", "unknown"]:
            sub = preds.filter(pl.col("phase") == ph)
            if len(sub) > 0:
                e = (sub["physics_fuel_kg"] - sub["actual_fuel_kg"]).abs()
                print(f"  {ph}: n={len(sub)}, MAE={e.mean():.1f}")

    # By sparsity bins
    print("\n[Physics MAE by traj density bin]")
    bins = [(0,5,"very_sparse"), (5,50,"sparse"), (50,500,"medium"), (500, 999999,"dense")]
    for lo, hi, name in bins:
        sub = preds.filter((pl.col("n_traj_pts") >= lo) & (pl.col("n_traj_pts") < hi))
        if len(sub) > 3:
            e = (sub["physics_fuel_kg"] - sub["actual_fuel_kg"]).abs()
            print(f"  {name} ({lo}-{hi}): n={len(sub)}, MAE={e.mean():.1f}")

    # Save summary table
    summary = pl.DataFrame({
        "metric": ["physics_mae", "physics_rmse", "naive_mae", "naive_rmse", "n_intervals"],
        "value": [
            phys_errs["overall"]["mae_kg"],
            phys_errs["overall"]["rmse_kg"],
            float(naive_err.mean()),
            float((naive_err**2).mean()**0.5),
            len(preds)
        ]
    })
    summary.write_csv(OUT / "table_physics_vs_naive_errors.csv")
    print("\nSaved table_physics_vs_naive_errors.csv")

    # Plots
    # 1. Physics vs actual (color by density)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(
        data=preds.to_pandas(),
        x="actual_fuel_kg", y="physics_fuel_kg",
        hue="n_traj_pts", palette="viridis", s=30, alpha=0.7, ax=ax
    )
    mx = max(preds["actual_fuel_kg"].max(), preds["physics_fuel_kg"].max())
    ax.plot([0, mx], [0, mx], "k--", lw=0.7)
    ax.set_title(f"Physics vs Actual (N={len(preds)} intervals, {N_SAMPLES} flights)\ncolor= n_traj_pts in window")
    ax.legend(title="pts", bbox_to_anchor=(1.05,1))
    fig.savefig(OUT / "fig_physics_vs_actual_sample.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved fig_physics_vs_actual_sample.png")

    # 2. Error by phase boxplot
    if "phase" in preds.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        preds_pd = preds.with_columns(
            ((pl.col("physics_fuel_kg") - pl.col("actual_fuel_kg")).abs()).alias("abs_err")
        ).to_pandas()
        sns.boxplot(data=preds_pd, x="phase", y="abs_err", ax=ax, order=["climb","cruise","descent","unknown"])
        ax.set_title("Physics absolute error by inferred flight phase")
        ax.set_ylabel("abs(physics - actual) kg")
        fig.savefig(OUT / "fig_error_by_phase.png", bbox_inches="tight", dpi=150)
        plt.close()
        print("Saved fig_error_by_phase.png")

    # 3. Error vs density (binned)
    fig, ax = plt.subplots(figsize=(7, 4))
    preds_pd = preds.with_columns(
        ((pl.col("physics_fuel_kg") - pl.col("actual_fuel_kg")).abs()).alias("abs_err")
    ).to_pandas()
    preds_pd["log_pts"] = preds_pd["n_traj_pts"].apply(lambda x: max(1, x))
    sns.scatterplot(data=preds_pd, x="n_traj_pts", y="abs_err", hue="has_acars_in_window", ax=ax, alpha=0.5, s=20)
    ax.set_xscale("log")
    ax.set_title("Physics error vs trajectory density per interval")
    ax.set_ylabel("abs error (kg)")
    fig.savefig(OUT / "fig_error_vs_density.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved fig_error_vs_density.png")

    print("\n04 complete. Compare physics vs naive, and note how errors are much higher on low-density (sparse ACARS) intervals.")
    print("This is the justification for the residual NN + physics-informed features.")
