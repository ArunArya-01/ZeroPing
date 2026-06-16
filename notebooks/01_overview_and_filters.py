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
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 9

OUT = Path.cwd() / "figures"
OUT.mkdir(exist_ok=True)

loader = AeroDataLoader()

print("=" * 60)
print("AEROTWIN DATASET OVERVIEW (train split, remote HF)")
print("=" * 60)

# 1. Summary counts (Table 1 skeleton)
summary = loader.dataset_summary(split=DEFAULT_SPLIT)
print("\n[Table 1] Dataset summary (train)")
print(summary)
summary.write_csv(OUT / "table_dataset_summary.csv")

usable = loader.get_usable_flight_ids(split=DEFAULT_SPLIT)
print(f"\nUsable (flightlist + traj overlap): {len(usable)} / {summary.filter(pl.col('resource')=='flightlist')['rows'][0]}")

# Full metadata for stats (small)
fl = loader.get_flightlist()
fu = loader.get_fuel_labels()

print("\n[Table 2] Aircraft type distribution (top 8 + total)")
ac = (
    fl.group_by("aircraft_type")
    .agg(pl.len().alias("n_flights"))
    .sort("n_flights", descending=True)
)
print(ac.head(8))
print(f"... total types: {ac.height}, total flights: {ac['n_flights'].sum()}")
ac.write_csv(OUT / "table_aircraft_types.csv")

# Basic date / airport
print(f"\nDate range: {fl['flight_date'].min()} to {fl['flight_date'].max()}")
print(f"Unique origins: {fl['origin_icao'].n_unique()}, dests: {fl['destination_icao'].n_unique()}")

# Flight duration (takeoff-landed)
fl = fl.with_columns(
    ((pl.col("landed") - pl.col("takeoff")).dt.total_seconds() / 3600.0).alias("dur_h")
)
print(f"Flight duration (h): mean={fl['dur_h'].mean():.2f}, median={fl['dur_h'].median():.2f}, 95%={fl['dur_h'].quantile(0.95):.1f}")

# Simple fig: aircraft counts
fig, ax = plt.subplots(figsize=(8, 4))
top = ac.head(10)
sns.barplot(data=top.to_pandas(), x="aircraft_type", y="n_flights", ax=ax, color="steelblue")
ax.set_title("Flights per Aircraft Type (train, top 10)")
ax.set_ylabel("n flights")
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(OUT / "fig_ac_types.png", bbox_inches="tight")
plt.close()
print(f"Saved fig: {OUT / 'fig_ac_types.png'}")

# Usable filter impact (if we had more metadata join, but simple count)
print("\n[Note] 1037 train flights have fuel+metadata but no trajectory parquet (filter required for modeling).")

# Schema note (from loader.get_schema, already printed in loader main)
print("\nSchemas confirmed via loader.get_schema() and direct probes (see 02/03 for fuel/traj deep dive).")

print("\nDone 01. Next: run 02 for fuel label characterization.")
