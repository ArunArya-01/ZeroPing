from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# ensure project root on path when run directly
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from data import AeroDataLoader
from physics.openap_baseline import predict_fuel_intervals


def patch_flight_id(parquet_path: Path | None = None) -> pl.DataFrame:
    """Add flight_id to an existing parquet by joining fuel labels."""
    out_path = parquet_path or Path("featured_dataset.parquet")
    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} not found.")

    df = pl.read_parquet(out_path)
    if "flight_id" in df.columns and df["flight_id"].null_count() == 0:
        print(f"{out_path} already has flight_id ({df['flight_id'].n_unique()} flights).")
        return df

    loader = AeroDataLoader()
    fuel = loader.get_fuel_labels().rename({"fuel_kg": "actual_fuel_kg", "idx": "interval_idx"})
    fuel = fuel.with_columns(
        pl.col("start").cast(pl.Datetime("us")),
        pl.col("end").cast(pl.Datetime("us")),
    )

    if "flight_id" in df.columns:
        df = df.drop("flight_id")

    patched = df.join(
        fuel.select(["flight_id", "interval_idx", "start", "end", "actual_fuel_kg"]),
        on=["interval_idx", "start", "end", "actual_fuel_kg"],
        how="left",
    )
    unmatched = patched["flight_id"].null_count()
    if unmatched:
        raise RuntimeError(f"Could not assign flight_id to {unmatched} rows.")

    preferred = ["flight_id"] + [c for c in patched.columns if c != "flight_id"]
    if "actual_fuel_kg" in patched.columns:
        preferred = (
            ["actual_fuel_kg", "physics_fuel_kg", "residual_kg", "flight_id"]
            + [c for c in patched.columns if c not in {"actual_fuel_kg", "physics_fuel_kg", "residual_kg", "flight_id"}]
        )
    patched = patched.select(preferred)
    patched.write_parquet(out_path)
    print(f"Patched {len(patched):,} rows with flight_id ({patched['flight_id'].n_unique():,} flights) -> {out_path}")
    return patched


def main(n: int | None = None):
    loader = AeroDataLoader()

    print("Loading metadata tables (full flightlist + fuel labels)...")
    fl = loader.get_flightlist()
    fuel_all = loader.get_fuel_labels()

    usable = loader.get_usable_flight_ids()
    print(f"Total usable flights (with traj): {len(usable)}")

    if n is not None and n < len(usable):
        usable = usable[:n]
        print(f"Limiting to first {n} flights for this run.")

    all_interval_dfs: list[pl.DataFrame] = []

    for i, fid in enumerate(usable):
        if i % 20 == 0 or i == len(usable) - 1:
            print(f"  Processing {i+1}/{len(usable)}: {fid}")

        try:
            traj = loader.load_flight_by_id(fid)
            fu = fuel_all.filter(pl.col("flight_id") == fid)
            if fu.is_empty() or traj.is_empty():
                continue

            meta_row = fl.filter(pl.col("flight_id") == fid).row(0, named=True)

            # This now returns the full set of requested features (when flight_meta given)
            interval_df = predict_fuel_intervals(traj, fu, flight_meta=meta_row)

            if interval_df.is_empty():
                continue

            # Target + physics residual (as specified)
            interval_df = interval_df.with_columns(
                pl.lit(fid).alias("flight_id"),
                (pl.col("actual_fuel_kg") - pl.col("physics_fuel_kg")).alias("residual_kg"),
            )

            all_interval_dfs.append(interval_df)

        except Exception as exc:
            print(f"    ERROR on {fid}: {exc}")
            continue

    if not all_interval_dfs:
        print("No intervals collected. Nothing to save.")
        return

    print(f"Concatenating {len(all_interval_dfs)} per-flight dataframes...")
    dataset = pl.concat(all_interval_dfs, how="diagonal_relaxed")

    if "energy_change_jpkg" in dataset.columns and "flight_id" in dataset.columns:
        dataset = dataset.sort("flight_id", "start_fraction_of_flight").with_columns(
            pl.col("energy_change_jpkg").cum_sum().over("flight_id").alias("cumulative_energy_change_jpkg")
        )

    # Optional: nice column order (target first, then physics, metadata, features)
    preferred_order = [
        # target
        "actual_fuel_kg",
        # physics
        "physics_fuel_kg", "residual_kg",
        # flight meta
        "flight_id", "aircraft_type", "origin_icao", "destination_icao",
        # interval meta
        "duration_s", "start_fraction_of_flight", "end_fraction_of_flight",
        # quality
        "n_traj_pts", "has_acars_in_window",
        # alt
        "mean_altitude", "median_altitude", "max_altitude", "std_altitude",
        # speed
        "mean_groundspeed", "std_groundspeed", "max_groundspeed",
        # vertical
        "mean_vertical_rate", "std_vertical_rate",
        # phase
        "climb_fraction", "cruise_fraction", "descent_fraction",
        # energy-state (E2)
        "ref_mass_kg", "mean_potential_energy_j", "mean_kinetic_energy_j",
        "mean_specific_energy_jpkg", "specific_energy_start", "specific_energy_end",
        "energy_change_jpkg", "energy_rate_jpkg_s", "climb_efficiency",
        "energy_efficiency", "cumulative_energy_change_jpkg",
        # operational (E3)
        "time_to_cruise_s", "climb_duration_s", "descent_duration_s",
        "cruise_speed_std", "tas_std", "vertical_rate_std",
        "number_of_level_segments", "holding_indicator", "path_efficiency",
        "distance_ratio", "altitude_stability", "segment_acceleration_mean",
        # telemetry / other
        "method",
        # keep a few original for debug if present
        "interval_idx", "start", "end", "tas_used", "alt_used", "vs_used", "phase",
    ]
    cols = [c for c in preferred_order if c in dataset.columns] + [
        c for c in dataset.columns if c not in preferred_order
    ]
    dataset = dataset.select(cols)

    out_path = Path("featured_dataset.parquet")
    print(f"Writing {len(dataset)} rows / {len(dataset.columns)} cols to {out_path} ...")
    dataset.write_parquet(out_path)
    print("Done.")
    print("Schema:", dict(dataset.schema))
    print("Sample row:")
    print(dataset.head(1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--patch-flight-id":
        patch_flight_id()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else None  # None = full usable set (~10k flights, 100k+ rows). Pass e.g. 500 for a quick subset.
        main(n)
