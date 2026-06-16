from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from data import AeroDataLoader, DEFAULT_SPLIT

logging.basicConfig(level=logging.WARNING)
for lib in ["httpx", "httpcore", "huggingface_hub", "fsspec"]:
    logging.getLogger(lib).setLevel(logging.ERROR)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT = Path.cwd() / "figures"
OUT.mkdir(exist_ok=True)

loader = AeroDataLoader()

fl = loader.get_flightlist(split=DEFAULT_SPLIT)
fu = loader.get_fuel_labels(split=DEFAULT_SPLIT)

print("=" * 60)
print("FUEL LABELS CHARACTERIZATION (train)")
print("=" * 60)

# Intervals per flight (Table 3)
per_f = (
    fu.group_by("flight_id")
    .agg(pl.len().alias("n_intervals"))
    .select(pl.col("n_intervals"))
)
print("\n[Table] Intervals per flight quantiles")
qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print({q: per_f.quantile(q).item() for q in qs})
print(f"min={per_f.min().item()}, max={per_f.max().item()}, mean={per_f.mean().item():.1f}")

# Total labeled fuel per flight
tot_fuel = (
    fu.group_by("flight_id")
    .agg(pl.col("fuel_kg").sum().alias("total_kg"))
    .select("total_kg")
)
print(f"\nTotal labeled fuel/flight (kg): mean={tot_fuel.mean().item():.0f}, median={tot_fuel.median().item():.0f}, max={tot_fuel.max().item():.0f}")

# Interval duration
fu = fu.with_columns(
    ((pl.col("end") - pl.col("start")).dt.total_seconds() / 60.0).alias("dur_min")
)
print(f"Interval dur (min): mean={fu['dur_min'].mean():.1f}, median={fu['dur_min'].median():.1f}")

# Coverage vs full flight (requires join to takeoff/landed)
fl_fuel = fl.join(
    fu.group_by("flight_id").agg(
        pl.col("start").min().alias("fuel_first"),
        pl.col("end").max().alias("fuel_last"),
        pl.len().alias("n_int"),
    ),
    on="flight_id",
    how="left",
)
fl_fuel = fl_fuel.with_columns(
    ((pl.col("landed") - pl.col("takeoff")).dt.total_seconds() / 3600).alias("full_dur_h"),
    ((pl.col("fuel_last") - pl.col("fuel_first")).dt.total_seconds() / 3600).alias("labeled_dur_h"),
)
fl_fuel = fl_fuel.with_columns(
    (pl.col("labeled_dur_h") / pl.col("full_dur_h")).alias("frac_labeled")
)
print(f"\nFraction of flight time covered by labels: mean={fl_fuel['frac_labeled'].mean():.2f}, median={fl_fuel['frac_labeled'].median():.2f}")
print("Note: many flights have labels covering << full profile (ACARS report availability).")

# Figs
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
sns.histplot(per_f["n_intervals"].to_pandas(), bins=30, ax=axes[0], color="teal")
axes[0].set_title("Intervals per flight")
axes[0].axvline(per_f.median().item(), color="red", ls="--", label="median")
axes[0].legend()

sns.histplot(tot_fuel["total_kg"].to_pandas(), bins=40, ax=axes[1], color="coral")
axes[1].set_title("Total labeled fuel per flight (kg)")
axes[1].set_xlim(0, 30000)
plt.tight_layout()
fig.savefig(OUT / "fig_fuel_intervals_and_total.png", bbox_inches="tight")
plt.close()
print(f"Saved: {OUT / 'fig_fuel_intervals_and_total.png'}")


fig, ax = plt.subplots(figsize=(6, 3.5))
sns.histplot(
    fu["fuel_kg"].to_pandas(),
    bins=60,
    ax=ax,
    log_scale=(False, True),
    color="steelblue",
    edgecolor="white",
    linewidth=0.2,
)
ax.set_xlim(0, 4000)
ax.set_title("fuel_kg per interval (log y; x truncated to bulk of data)\n"
             "Full range to 32 t; see text stats for tail")
ax.set_xlabel("fuel_kg (kg)")
ax.set_ylabel("Count (log)")
fig.savefig(OUT / "fig_fuel_kg_per_interval.png", bbox_inches="tight")
plt.close()

print("\nSaved fuel figs + quantiles. Key for Paper 1: partial coverage + interval stats.")
