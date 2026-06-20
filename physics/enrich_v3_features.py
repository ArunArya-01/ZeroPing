"""
Add AeroTwin v3 weather features to featured_dataset.parquet.

Run: python physics/enrich_v3_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from physics.feature_engineering import enrich_from_columns
from physics.weather_features import WEATHER_FEATURES, enrich_weather_from_columns


def main() -> None:
    path = root / "featured_dataset.parquet"
    df = pl.read_parquet(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

    if "mean_specific_energy_jpkg" not in df.columns:
        print("Adding v2 energy/operational features ...")
        df = enrich_from_columns(df)

    if "headwind_mps" not in df.columns:
        print("Adding v3 weather features ...")
        df = enrich_weather_from_columns(df)

    # Wind-adjusted physics proxy for E8 (BADA-style correction path)
    if "physics_wind_adj_kg" not in df.columns:
        hw = pl.col("headwind_mps").fill_null(0.0)
        tas = pl.col("tas_used").fill_null(pl.col("mean_groundspeed")).fill_null(200.0)
        df = df.with_columns(
            (pl.col("physics_fuel_kg") * (1.0 + pl.max_horizontal(hw, pl.lit(0.0)) / tas * 0.12))
            .alias("physics_wind_adj_kg")
        )

    df.write_parquet(path)
    wx = [c for c in WEATHER_FEATURES if c in df.columns]
    print(f"Saved {len(df.columns)} columns ({len(wx)} weather features)")


if __name__ == "__main__":
    main()