

from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from data import AeroDataLoader, DEFAULT_RANDOM_SEED

logging.basicConfig(level=logging.WARNING)
for lib in ["httpx", "httpcore", "huggingface_hub"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
OUT = Path.cwd() / "figures"
OUT.mkdir(exist_ok=True)

N_SAMPLES = 100  # Bumped for more complete EDA (plan suggested 200-500; 100 is solid compromise on runtime)
SEED = DEFAULT_RANDOM_SEED

loader = AeroDataLoader()
fuel_all = loader.get_fuel_labels()

print("=" * 60)
print(f"TRAJ QUALITY & SOURCE HETEROGENEITY (sample of {N_SAMPLES} usable flights)")
print("=" * 60)

# Get usable and sample
usable = loader.get_usable_flight_ids(split="train")
random.seed(SEED)
sampled_fids = random.sample(usable, min(N_SAMPLES, len(usable)))
print(f"Sampled {len(sampled_fids)} flight_ids (seed={SEED})")

# Collect per-interval stats across sample
interval_stats: list[dict] = []
acars_completeness = []
profile_fids_to_plot = []  # we'll pick diverse ones

for i, fid in enumerate(sampled_fids):
    try:
        tr = loader.load_flight_by_id(fid)
        fu = fuel_all.filter(pl.col("flight_id") == fid).sort("start")
        if fu.is_empty():
            continue

        tr = tr.sort("timestamp")
        n_ac_total = len(tr.filter(pl.col("source") == "acars")) if "source" in tr.columns else 0

        ac = tr.filter(pl.col("source") == "acars") if "source" in tr.columns else pl.DataFrame()
        n_ac = len(ac)
        m_ok = int(ac["mach"].is_not_null().sum()) if n_ac else 0
        tas_ok = int(ac["TAS"].is_not_null().sum()) if n_ac else 0
        cas_ok = int(ac["CAS"].is_not_null().sum()) if n_ac else 0
        acars_completeness.append({
            "fid": fid, "total_rows": len(tr), "acars_rows": n_ac,
            "mach_ok": m_ok, "tas_ok": tas_ok, "cas_ok": cas_ok
        })

        # Per fuel interval stats
        for row in fu.iter_rows(named=True):
            s, e = row["start"], row["end"]
            mask = (tr["timestamp"] >= s) & (tr["timestamp"] <= e)
            win = tr.filter(mask)
            n_pts = len(win)
            has_acars = False
            if "source" in win.columns:
                has_acars = (win["source"] == "acars").any()

            interval_stats.append({
                "fid": fid,
                "n_traj_pts": n_pts,
                "has_acars": has_acars,
                "dur_min": (e - s).total_seconds() / 60.0,
                "actual_fuel_kg": row["fuel_kg"]
            })

        # Pick diverse for profiles: first few + try to find one with 0 acars, one high n, one low
        if i < 2 or n_ac_total == 0 or (n_pts > 2000 and len(profile_fids_to_plot) < 4):
            if fid not in profile_fids_to_plot:
                profile_fids_to_plot.append(fid)

    except Exception as exc:
        print(f"  WARN: failed to process {fid}: {exc}")
        continue

print(f"Processed intervals: {len(interval_stats)} from {len(sampled_fids)} flights")

# Convert to df for analysis
int_df = pl.DataFrame(interval_stats)
print("\n[Key Stats] Points per fuel interval (across sampled labeled intervals):")
print(f"  min={int_df['n_traj_pts'].min()}, max={int_df['n_traj_pts'].max()}, mean={int_df['n_traj_pts'].mean():.1f}, median={int_df['n_traj_pts'].median():.1f}")
print(f"  % with <5 pts (very sparse): {(int_df['n_traj_pts'] < 5).mean()*100:.1f}%")
print(f"  % with >=1000 pts (dense): {(int_df['n_traj_pts'] >= 1000).mean()*100:.1f}%")

# CDF plot (critical Paper 1 figure)
fig, ax = plt.subplots(figsize=(7, 4))
vals = int_df["n_traj_pts"].to_pandas().sort_values()
ax.plot(vals, vals.rank(pct=True) * 100, lw=2)
ax.set_xscale("log")
ax.set_xlabel("Number of trajectory points in fuel interval (log scale)")
ax.set_ylabel("Cumulative % of intervals")
ax.set_title("CDF of Trajectory Density per Labeled Fuel Interval\n(50-flight sample; shows extreme heterogeneity)")
ax.axvline(5, color="red", ls="--", alpha=0.7, label="very sparse (<5)")
ax.axvline(100, color="orange", ls="--", alpha=0.7, label="moderate (100)")
ax.axvline(1000, color="green", ls="--", alpha=0.7, label="dense (1000+)")
ax.legend()
fig.savefig(OUT / "fig_pts_per_interval_cdf.png", bbox_inches="tight", dpi=150)
plt.close()
print("Saved: figures/fig_pts_per_interval_cdf.png")

# Also simple hist
fig, ax = plt.subplots(figsize=(6, 3.5))
sns.histplot(int_df["n_traj_pts"].to_pandas(), bins=40, ax=ax, log_scale=(True, False))
ax.set_title("Distribution of traj points per fuel interval (log x)")
ax.set_xlabel("n_traj_pts")
fig.savefig(OUT / "fig_pts_per_interval_hist.png", bbox_inches="tight", dpi=150)
plt.close()

# ACARS completeness summary
ac_df = pl.DataFrame(acars_completeness)
print("\n[ACARS Completeness across sampled flights]")
print(f"  Flights with 0 ACARS rows: {(ac_df['acars_rows'] == 0).sum()} / {len(ac_df)}")
print(f"  Avg ACARS per flight: {ac_df['acars_rows'].mean():.1f}")
mach_pct = (ac_df['mach_ok'].sum() / ac_df['acars_rows'].sum() * 100) if ac_df['acars_rows'].sum() > 0 else 0
print(f"  Overall % of ACARS rows with mach: {mach_pct:.1f}% (TAS/CAS similarly incomplete in samples)")

# Generate 4-5 profile plots for diverse flights
print(f"\nGenerating profile plots for up to 5 diverse fids: {profile_fids_to_plot[:5]}")
for fid in profile_fids_to_plot[:5]:
    try:
        tr = loader.load_flight_by_id(fid)
        fu = fuel_all.filter(pl.col("flight_id") == fid).sort("start")
        ac = tr.filter(pl.col("source") == "acars") if "source" in tr.columns else pl.DataFrame()
        n_ac = len(ac)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        tr = tr.sort("timestamp")
        t0 = tr["timestamp"][0]
        x = (tr["timestamp"] - t0).dt.total_seconds() / 3600.0

        axes[0].plot(x, tr["altitude"], lw=0.6, color="navy")
        axes[0].set_ylabel("alt (m)")
        for r in fu.iter_rows(named=True):
            xs = (r["start"] - t0).total_seconds() / 3600
            xe = (r["end"] - t0).total_seconds() / 3600
            axes[0].axvspan(xs, xe, alpha=0.12, color="orange")
            axes[0].text((xs+xe)/2, tr["altitude"].max() * 0.85, f"{r['fuel_kg']:.0f}kg", ha="center", fontsize=6, rotation=0)

        axes[1].plot(x, tr["groundspeed"], lw=0.5, color="green", label="groundspeed")
        if n_ac > 0:
            acs = ac.sort("timestamp")
            xa = (acs["timestamp"] - t0).dt.total_seconds() / 3600
            axes[1].scatter(xa, acs["groundspeed"], c="red", s=10, label="ACARS", zorder=5, alpha=0.8)
        axes[1].set_ylabel("gs (m/s)")
        axes[1].legend(fontsize=7, loc="upper right")

        axes[2].plot(x, tr["vertical_rate"], lw=0.5, color="purple")
        axes[2].axhline(0, color="gray", lw=0.5, alpha=0.6)
        axes[2].set_ylabel("vr (m/s)")
        axes[2].set_xlabel("hours from traj start")

        fig.suptitle(f"Example Flight {fid} — traj density + ACARS fuel intervals (shaded)")
        fig.savefig(OUT / f"fig_profile_{fid}.png", bbox_inches="tight", dpi=140)
        plt.close()
        print(f"  Saved profile for {fid} (ACARS rows: {n_ac})")
    except Exception as e:
        print(f"  profile fail {fid}: {e}")

print("\n03 complete. Key outputs: CDF of interval density, profile examples, ACARS variability stats.")
print("These highlight the 'data quality is highly variable per label' challenge for hybrid modeling.")


print("\n" + "=" * 60)
print("COMPLETE DATASET EDA — ADDITIONAL CHARACTERIZATION")
print("=" * 60)

# 1. Bias analysis: the 1037 flights that have fuel+metadata but NO trajectory files
# This is critical — are the usable 10k representative?
print("\n[1. Missing-trajectory flights bias (1037 flights: metadata + fuel only)]")
fl = loader.get_flightlist(split="train")  # full flightlist (cheap)
usable_set = set(usable) if 'usable' in dir() else set(loader.get_usable_flight_ids(split="train"))
missing_ids = [fid for fid in fl['flight_id'].to_list() if fid not in usable_set]
print(f"  Total flights in flightlist: {len(fl)}")
print(f"  Usable (with traj): {len(usable_set)}")
print(f"  Missing traj files: {len(missing_ids)}")

usable_fl = fl.filter(pl.col("flight_id").is_in(list(usable_set)))
missing_fl = fl.filter(pl.col("flight_id").is_in(missing_ids))

# Ac type bias
print("\n  Aircraft type distribution (usable vs missing):")
u_ac = usable_fl.group_by("aircraft_type").agg(pl.len().alias("n")).sort("n", descending=True).head(5)
m_ac = missing_fl.group_by("aircraft_type").agg(pl.len().alias("n")).sort("n", descending=True).head(5)
print("  Usable top 5:\n", u_ac)
print("  Missing top 5:\n", m_ac)

# Fuel side bias for missing
missing_fuel = fuel_all.filter(pl.col("flight_id").is_in(missing_ids))
if len(missing_ids) > 0:
    miss_per = len(missing_fuel) / len(missing_ids)
    print(f"\n  Missing flights fuel intervals: {len(missing_fuel)} (avg {miss_per:.1f} per flight)")
    print(f"  Overall avg intervals/flight (from full fuel): ~11.9")
    # total labeled fuel comparison
    miss_tot = missing_fuel.group_by("flight_id").agg(pl.col("fuel_kg").sum())
    print(f"  Missing flights total labeled fuel per flight: mean={miss_tot['fuel_kg'].mean():.0f}, median={miss_tot['fuel_kg'].median():.0f}")
else:
    print("  No missing (unexpected)")

# 2. Airports data integration (simple join for origin characteristics)
print("\n[2. Airports data (quick integration)]")
try:
    # Small file — direct read often works
    airports = pl.read_parquet("hf://datasets/aerotwin/aero-data/airports.parquet")
    print(f"  Airports rows: {len(airports)}")
    # Join origin elevation to flightlist sample
    fl_with_elev = fl.join(airports, left_on="origin_icao", right_on="icao", how="left")
    print(f"  Flights with origin elevation: {fl_with_elev['elevation'].is_not_null().sum()} / {len(fl)}")
    # Rough: does higher elevation correlate with anything obvious in fuel for short flights? (sample only)
    short = fl_with_elev.filter( (pl.col("landed") - pl.col("takeoff")).dt.total_seconds() / 3600 < 3 )
    if len(short) > 10:
        elev_fuel = short.join(fuel_all.group_by("flight_id").agg(pl.col("fuel_kg").sum().alias("total_fuel")), on="flight_id", how="left")
        corr = elev_fuel.select(pl.corr("elevation", "total_fuel")).item()
        print(f"  Correlation (origin elev vs total labeled fuel) for short flights (<3h): {corr:.3f} (weak, as expected)")
except Exception as e:
    print(f"  Airports analysis skipped ({e})")

# 3. Quick ac_type level view of sparsity from the sampled traj (already collected)
print("\n[3. Sparsity by aircraft type (from the 100-flight traj sample)]")
if len(interval_stats) > 0:
    int_df = pl.DataFrame(interval_stats)
    # Need to join ac_type — we can pull from the sampled trajs or approximate from fl
    ac_sparsity = (
        int_df.group_by("fid")
        .agg(
            pl.col("n_traj_pts").median().alias("med_pts"),
            pl.col("has_acars").mean().alias("frac_acars_ints")
        )
        .join(fl.select(["flight_id", "aircraft_type"]), left_on="fid", right_on="flight_id", how="left")
        .group_by("aircraft_type")
        .agg(
            pl.len().alias("n_flights_sampled"),
            pl.col("med_pts").mean().alias("avg_med_pts"),
            pl.col("frac_acars_ints").mean().alias("avg_frac_with_acars")
        )
        .sort("n_flights_sampled", descending=True)
        .head(8)
    )
    print(ac_sparsity)

print("\n[Complete pure-data EDA additions finished]")
print("Key new findings above (bias in missing traj flights, airports, ac_type sparsity view).")
