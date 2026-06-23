
from __future__ import annotations

from pathlib import Path

import polars as pl


def main():
    parquet_path = Path("featured_dataset.parquet")
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found in current directory.")
        print("Run the builder first: python physics/build_featured_dataset.py")
        return

    print("=" * 70)
    print("FEATURED DATASET - QUICK UNDERSTANDING REPORT")
    print("=" * 70)
    print(f"Source: {parquet_path.resolve()}")

    df = pl.read_parquet(parquet_path)
    n_rows, n_cols = df.shape
    print(f"\nShape: {n_rows:,} rows × {n_cols} columns")

    print("\n" + "-" * 70)
    print("1. SCHEMA & SAMPLE")
    print("-" * 70)
    print("Columns:")
    for i, (col, dtype) in enumerate(df.schema.items()):
        print(f"  {i+1:2d}. {col:30s} : {dtype}")
        if i >= 12:  # first 13 + note
            print(f"     ... ({n_cols - 13} more)")
            break

    print("\nFirst 3 rows (selected key columns):")
    key_cols = [
        "actual_fuel_kg", "physics_fuel_kg", "residual_kg",
        "aircraft_type", "n_traj_pts", "has_acars_in_window",
        "climb_fraction", "cruise_fraction", "descent_fraction",
        "method"
    ]
    available = [c for c in key_cols if c in df.columns]
    print(df.select(available).head(3))

    print("\n" + "-" * 70)
    print("2. TARGET & PHYSICS BASELINE")
    print("-" * 70)
    target_cols = ["actual_fuel_kg", "physics_fuel_kg", "residual_kg"]
    print(df.select(target_cols).describe())

    # Residual distribution
    print("\nResidual (actual - physics) quantiles:")
    res_q = df.select(
        pl.col("residual_kg").quantile(0.05).alias("q05"),
        pl.col("residual_kg").quantile(0.25).alias("q25"),
        pl.col("residual_kg").quantile(0.5).alias("median"),
        pl.col("residual_kg").quantile(0.75).alias("q75"),
        pl.col("residual_kg").quantile(0.95).alias("q95"),
    )
    print(res_q)

    print("\n" + "-" * 70)
    print("3. TRAJECTORY QUALITY & SPARSITY (core signal from EDA)")
    print("-" * 70)
    print("n_traj_pts distribution:")
    print(df.select("n_traj_pts").describe())

    sparsity = df.with_columns(
        pl.when(pl.col("n_traj_pts") < 5).then(pl.lit("very_sparse"))
        .when(pl.col("n_traj_pts") < 50).then(pl.lit("sparse"))
        .when(pl.col("n_traj_pts") < 500).then(pl.lit("medium"))
        .otherwise(pl.lit("dense"))
        .alias("sparsity_bin")
    ).group_by("sparsity_bin").agg(
        pl.len().alias("n_intervals"),
        pl.col("actual_fuel_kg").mean().alias("mean_actual"),
        pl.col("residual_kg").mean().alias("mean_residual"),
    ).sort("n_intervals", descending=True)
    print("\nSparsity bins (intervals):")
    print(sparsity)

    acars_break = df.group_by("has_acars_in_window").agg(
        pl.len().alias("n_intervals"),
        pl.col("residual_kg").mean().alias("mean_residual"),
    )
    print("\nBy has_acars_in_window:")
    print(acars_break)

    print("\n" + "-" * 70)
    print("4. PHASE BREAKDOWN (within each interval window)")
    print("-" * 70)
    phase_cols = ["climb_fraction", "cruise_fraction", "descent_fraction"]
    print(df.select(phase_cols).describe())

    # Dominant phase per interval
    dominant = df.with_columns(
        pl.when(pl.col("climb_fraction") > 0.5).then(pl.lit("climb"))
        .when(pl.col("descent_fraction") > 0.5).then(pl.lit("descent"))
        .otherwise(pl.lit("cruise"))
        .alias("dominant_phase")
    ).group_by("dominant_phase").agg(
        pl.len().alias("n_intervals"),
        pl.col("actual_fuel_kg").mean().alias("mean_actual"),
        pl.col("residual_kg").mean().alias("mean_residual"),
    ).sort("n_intervals", descending=True)
    print("\nDominant phase per interval:")
    print(dominant)

    print("\n" + "-" * 70)
    print("5. CORRELATIONS WITH TARGET & RESIDUAL")
    print("-" * 70)
    numeric_for_corr = [
        "n_traj_pts", "duration_s",
        "mean_altitude", "std_altitude",
        "mean_groundspeed", "std_groundspeed",
        "mean_vertical_rate", "std_vertical_rate",
        "climb_fraction", "cruise_fraction", "descent_fraction",
        "start_fraction_of_flight",
    ]
    present = [c for c in numeric_for_corr if c in df.columns]

    print("Correlation with actual_fuel_kg:")
    for col in present:
        c = df.select(pl.corr("actual_fuel_kg", col)).item()
        print(f"  {col:30s}: {c:7.4f}")

    print("\nCorrelation with residual_kg (physics error):")
    for col in present:
        c = df.select(pl.corr("residual_kg", col)).item()
        print(f"  {col:30s}: {c:7.4f}")

    print("\n" + "-" * 70)
    print("6. BREAKDOWNS BY AIRCRAFT & METHOD")
    print("-" * 70)
    ac_top = df.group_by("aircraft_type").agg(
        pl.len().alias("n_intervals"),
        pl.col("actual_fuel_kg").mean().alias("mean_actual"),
        pl.col("residual_kg").mean().alias("mean_residual"),
    ).sort("n_intervals", descending=True).head(8)
    print("Top aircraft types:")
    print(ac_top)

    method_break = df.group_by("method").agg(
        pl.len().alias("n_intervals"),
        pl.col("actual_fuel_kg").mean().alias("mean_actual"),
        pl.col("residual_kg").mean().alias("mean_residual"),
    ).sort("n_intervals", descending=True)
    print("\nBy physics method (how TAS was obtained):")
    print(method_break)

    print("\n" + "-" * 70)
    print("7. KEY INSIGHTS (from full EDA + this dataset)")
    print("-" * 70)
    print("""
- Residuals are systematically positive in this sample (physics over-predicts on average).
- Sparsity matters: very_sparse intervals (<5 pts) have different residual behavior.
- Phase fractions are meaningful: cruise dominates most intervals.
- n_traj_pts and has_acars_in_window are strong signals (use them as features!).
- Aircraft type has clear effect on both scale of fuel and typical residual.
- Duration + start_fraction are useful (early intervals often different).
- Many intervals come from long-haul flights (B789 etc.) with very different
  characteristics from narrow-body.

This dataset encodes exactly the challenges discovered in the EDA:
partial observability, source-dependent data quality, unknown mass,
and the need for the residual model to correct the physics baseline
in a data-dependent way.
""")

    print("\n" + "=" * 70)
    print("To explore interactively: load in a notebook or python REPL")
    print("  df = pl.read_parquet('featured_dataset.parquet')")
    print("=" * 70)


if __name__ == "__main__":
    main()
