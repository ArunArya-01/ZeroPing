"""Build ``featured_dataset_audit.parquet`` from external flight records.

Reuses ``physics.openap_baseline.predict_fuel_intervals`` for physics fuel +
trajectory features (base, energy-state, operational, weather when available).

Supports two target modes documented in the audit package:
* **Direct** – model ``actual_fuel_kg`` (interval burn).
* **Fuel-Flow** – model ``actual_fuel_kg / duration_s``; recover kg at eval time.

Target mode is recorded in the parquet metadata column ``target_mode`` and does
not alter stored ``actual_fuel_kg`` (always kg). Flow conversion happens in
``run_audit_pilot`` / ``external_vs_flow_eval``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

import polars as pl

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from physics.openap_baseline import predict_fuel_intervals

LOGGER = logging.getLogger(__name__)

TargetMode = Literal["direct", "fuel_flow", "both"]

PREFERRED_ORDER = [
    "actual_fuel_kg",
    "physics_fuel_kg",
    "residual_kg",
    "flight_id",
    "aircraft_type",
    "origin_icao",
    "destination_icao",
    "duration_s",
    "start_fraction_of_flight",
    "end_fraction_of_flight",
    "n_traj_pts",
    "has_acars_in_window",
    "mean_altitude",
    "median_altitude",
    "max_altitude",
    "std_altitude",
    "mean_groundspeed",
    "std_groundspeed",
    "max_groundspeed",
    "mean_vertical_rate",
    "std_vertical_rate",
    "climb_fraction",
    "cruise_fraction",
    "descent_fraction",
    "ref_mass_kg",
    "mean_potential_energy_j",
    "mean_kinetic_energy_j",
    "mean_specific_energy_jpkg",
    "specific_energy_start",
    "specific_energy_end",
    "energy_change_jpkg",
    "energy_rate_jpkg_s",
    "climb_efficiency",
    "energy_efficiency",
    "cumulative_energy_change_jpkg",
    "time_to_cruise_s",
    "climb_duration_s",
    "descent_duration_s",
    "cruise_speed_std",
    "tas_std",
    "vertical_rate_std",
    "number_of_level_segments",
    "holding_indicator",
    "path_efficiency",
    "distance_ratio",
    "altitude_stability",
    "segment_acceleration_mean",
    "method",
    "interval_idx",
    "start",
    "end",
    "tas_used",
    "alt_used",
    "vs_used",
    "phase",
    "label_source",
    "label_is_physics_derived",
    "dataset_source",
    "pts_per_min",
    "sparsity_bin",
]


def preferred_column_order(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder columns like ``build_featured_dataset.py``."""
    cols = [c for c in PREFERRED_ORDER if c in df.columns] + [
        c for c in df.columns if c not in PREFERRED_ORDER
    ]
    return df.select(cols)


def _fuel_frame(fuel: pl.DataFrame) -> pl.DataFrame:
    """Normalise fuel interval schema to what ``predict_fuel_intervals`` expects."""
    rename: dict[str, str] = {}
    if "interval_idx" in fuel.columns and "idx" not in fuel.columns:
        rename["interval_idx"] = "idx"
    if "actual_fuel_kg" in fuel.columns and "fuel_kg" not in fuel.columns:
        rename["actual_fuel_kg"] = "fuel_kg"
    if rename:
        fuel = fuel.rename(rename)
    need = ["start", "end"]
    for c in need:
        if c not in fuel.columns:
            raise ValueError(f"Fuel frame missing column {c}; have {fuel.columns}")
    if "fuel_kg" not in fuel.columns:
        fuel = fuel.with_columns(pl.lit(None).cast(pl.Float64).alias("fuel_kg"))
    if "idx" not in fuel.columns:
        fuel = fuel.with_columns(pl.arange(0, pl.len()).alias("idx"))
    return fuel.select(
        [c for c in ("idx", "start", "end", "fuel_kg", "label_method") if c in fuel.columns]
    )


def process_one_flight(
    traj: pl.DataFrame,
    fuel: pl.DataFrame,
    meta: dict[str, Any],
    *,
    dataset_source: str = "external",
    force_physics_as_actual: bool = False,
) -> pl.DataFrame:
    """Run OpenAP + feature extraction for one flight.

    Parameters
    ----------
    force_physics_as_actual:
        If True (OpenSky path), replace actual with physics after prediction.
        If False (DASHlink with reconstructed fuel), keep reconstructed labels.
    """
    flight_id = meta.get("flight_id") or "unknown"
    ac_type = meta.get("aircraft_type")
    if ac_type is None and "typecode" in traj.columns:
        ac_type = traj["typecode"].drop_nulls().first()
    ac_type = str(ac_type or "A320")

    fuel_in = _fuel_frame(fuel)
    # Drop intervals with null fuel when we have independent labels
    if not force_physics_as_actual and "fuel_kg" in fuel_in.columns:
        valid = fuel_in.filter(pl.col("fuel_kg").is_not_null() & pl.col("fuel_kg").is_finite())
        if valid.is_empty():
            LOGGER.warning(
                "%s: no valid fuel_kg intervals; falling back to physics-as-actual",
                flight_id,
            )
            force_physics_as_actual = True
            fuel_in = fuel_in.with_columns(pl.col("fuel_kg").fill_null(1.0))
        else:
            fuel_in = valid

    if force_physics_as_actual:
        fuel_in = fuel_in.with_columns(pl.col("fuel_kg").fill_null(1.0).clip(lower_bound=0.1))

    try:
        interval_df = predict_fuel_intervals(
            traj, fuel_in, ac_type=ac_type, flight_meta=meta
        )
    except Exception as exc:
        LOGGER.exception("predict_fuel_intervals failed for %s: %s", flight_id, exc)
        return pl.DataFrame()

    if interval_df.is_empty():
        return interval_df

    label_source = meta.get("label_source") or (
        "physics_openap" if force_physics_as_actual else "external_reconstructed"
    )
    is_physics = bool(force_physics_as_actual) or str(label_source).startswith("physics")

    if force_physics_as_actual:
        interval_df = interval_df.with_columns(
            pl.col("physics_fuel_kg").alias("actual_fuel_kg"),
        )

    interval_df = interval_df.with_columns(
        pl.lit(flight_id).alias("flight_id"),
        (pl.col("actual_fuel_kg") - pl.col("physics_fuel_kg")).alias("residual_kg"),
        pl.lit(label_source).alias("label_source"),
        pl.lit(is_physics).alias("label_is_physics_derived"),
        pl.lit(dataset_source).alias("dataset_source"),
    )

    # Sparsity convenience columns
    if "n_traj_pts" in interval_df.columns and "duration_s" in interval_df.columns:
        interval_df = interval_df.with_columns(
            (
                pl.col("n_traj_pts")
                / (pl.col("duration_s").clip(lower_bound=1.0) / 60.0)
            ).alias("pts_per_min"),
        )

    # Ensure aircraft_type filled
    if "aircraft_type" not in interval_df.columns or interval_df["aircraft_type"].null_count() == len(
        interval_df
    ):
        interval_df = interval_df.with_columns(pl.lit(ac_type).alias("aircraft_type"))

    return interval_df


def build_featured_from_trajectories(
    flights: Sequence[dict[str, Any]],
    *,
    dataset_source: str = "external",
    force_physics_as_actual: bool | None = None,
) -> pl.DataFrame:
    """Concatenate per-flight featured intervals.

    Each element of ``flights`` is a dict with keys ``traj``, ``fuel``, ``meta``
    (as returned by dashlink_loader / opensky_loader).

    Parameters
    ----------
    force_physics_as_actual:
        ``None`` → auto-detect from ``meta['label_source']`` / dataset_source.
    """
    all_dfs: list[pl.DataFrame] = []
    for i, rec in enumerate(flights):
        traj = rec["traj"]
        fuel = rec["fuel"]
        meta = rec.get("meta") or {"flight_id": rec.get("flight_id", f"flight_{i}")}
        fid = meta.get("flight_id", f"flight_{i}")

        if force_physics_as_actual is None:
            ls = str(meta.get("label_source") or "")
            src = str(meta.get("source") or dataset_source)
            use_physics = ls.startswith("physics") or "opensky" in src.lower()
        else:
            use_physics = force_physics_as_actual

        if i % 10 == 0:
            LOGGER.info(
                "Featured build %d/%d: %s (physics_as_actual=%s)",
                i + 1,
                len(flights),
                fid,
                use_physics,
            )

        try:
            part = process_one_flight(
                traj,
                fuel,
                meta,
                dataset_source=dataset_source,
                force_physics_as_actual=use_physics,
            )
        except Exception as exc:
            LOGGER.exception("Skipping %s: %s", fid, exc)
            continue
        if not part.is_empty():
            all_dfs.append(part)

    if not all_dfs:
        LOGGER.error("No intervals collected — empty featured audit dataset")
        return pl.DataFrame()

    dataset = pl.concat(all_dfs, how="diagonal_relaxed")

    if "energy_change_jpkg" in dataset.columns and "flight_id" in dataset.columns:
        sort_cols = ["flight_id"]
        if "start_fraction_of_flight" in dataset.columns:
            sort_cols.append("start_fraction_of_flight")
        elif "start" in dataset.columns:
            sort_cols.append("start")
        dataset = dataset.sort(sort_cols).with_columns(
            pl.col("energy_change_jpkg")
            .cum_sum()
            .over("flight_id")
            .alias("cumulative_energy_change_jpkg")
        )

    # Drop rows without usable physics / actual for ML
    if "physics_fuel_kg" in dataset.columns and "actual_fuel_kg" in dataset.columns:
        before = len(dataset)
        dataset = dataset.filter(
            pl.col("physics_fuel_kg").is_not_null()
            & pl.col("physics_fuel_kg").is_finite()
            & pl.col("actual_fuel_kg").is_not_null()
            & pl.col("actual_fuel_kg").is_finite()
            & (pl.col("duration_s") >= 60.0)
        )
        LOGGER.info("Filtered intervals %d → %d (finite fuel + duration≥60s)", before, len(dataset))

    dataset = preferred_column_order(dataset)
    LOGGER.info(
        "Featured audit dataset: %d rows, %d cols, %d flights, sources=%s",
        len(dataset),
        len(dataset.columns),
        dataset["flight_id"].n_unique() if "flight_id" in dataset.columns else 0,
        dataset["dataset_source"].unique().to_list() if "dataset_source" in dataset.columns else [],
    )
    return dataset


def write_featured_audit(
    dataset: pl.DataFrame,
    out_path: Path | str | None = None,
) -> Path:
    """Write parquet and return path."""
    out_path = Path(out_path or "featured_dataset_audit.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(out_path)
    LOGGER.info("Wrote %s (%d rows)", out_path, len(dataset))
    return out_path


def build_from_dashlink(
    directory: Path | str,
    *,
    max_flights: int | None = 10,
    out_path: Path | str | None = None,
    interval_s: float = 600.0,
) -> pl.DataFrame:
    """Convenience: load DASHlink dir → featured audit parquet."""
    from physics.external_audit.dashlink_loader import load_dashlink_directory

    flights = load_dashlink_directory(
        directory, max_flights=max_flights, interval_s=interval_s
    )
    ds = build_featured_from_trajectories(flights, dataset_source="dashlink")
    if out_path is not None and not ds.is_empty():
        write_featured_audit(ds, out_path)
    return ds


def build_from_opensky(
    start: str,
    stop: str,
    *,
    max_flights: int | None = 20,
    out_path: Path | str | None = None,
    interval_s: float = 600.0,
    icao24: str | None = None,
    synthetic_fallback: bool = True,
) -> pl.DataFrame:
    """Convenience: OpenSky query (or synthetic) → featured audit parquet."""
    from physics.external_audit.opensky_loader import (
        load_opensky_flights,
        make_synthetic_opensky_flights,
    )

    flights = load_opensky_flights(
        start, stop, icao24=icao24, max_flights=max_flights, interval_s=interval_s
    )
    if not flights and synthetic_fallback:
        LOGGER.warning(
            "OpenSky query empty — using synthetic OpenSky-like flights "
            "(labels will still be physics-derived after OpenAP)"
        )
        flights = make_synthetic_opensky_flights(n_flights=max_flights or 5)

    ds = build_featured_from_trajectories(
        flights,
        dataset_source="opensky",
        force_physics_as_actual=True,
    )
    if out_path is not None and not ds.is_empty():
        write_featured_audit(ds, out_path)
    return ds


def build_demo_featured(
    n_flights: int = 8,
    out_path: Path | str | None = None,
) -> pl.DataFrame:
    """Fully offline demo dataset for CI / smoke tests."""
    from physics.external_audit.audit_utils import synthesize_demo_trajectory

    flights = []
    types = ["B737", "A320", "CRJ9", "E190", "B738", "A321", "B77W", "A333"]
    for i in range(n_flights):
        fid = f"demo_audit_{i:03d}"
        traj, fuel, meta = synthesize_demo_trajectory(
            flight_id=fid, ac_type=types[i % len(types)], seed=100 + i, n_points=90 + i * 3
        )
        meta["label_source"] = "synthetic_demo"
        meta["source"] = "demo"
        flights.append({"flight_id": fid, "traj": traj, "fuel": fuel, "meta": meta})

    ds = build_featured_from_trajectories(
        flights, dataset_source="demo", force_physics_as_actual=False
    )
    if out_path is not None and not ds.is_empty():
        write_featured_audit(ds, out_path)
    return ds


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Build featured_dataset_audit.parquet")
    p.add_argument(
        "--source",
        choices=["demo", "dashlink", "opensky"],
        default="demo",
        help="Data source (default demo for offline runs)",
    )
    p.add_argument("--dashlink-dir", type=str, default=None)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--stop", default="2024-01-02")
    p.add_argument("--max-flights", type=int, default=8)
    p.add_argument(
        "--out",
        type=str,
        default="featured_dataset_audit.parquet",
        help="Output parquet path",
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    if args.source == "demo":
        ds = build_demo_featured(n_flights=args.max_flights, out_path=out)
    elif args.source == "dashlink":
        if not args.dashlink_dir:
            raise SystemExit("--dashlink-dir required for source=dashlink")
        ds = build_from_dashlink(
            args.dashlink_dir, max_flights=args.max_flights, out_path=out
        )
    else:
        ds = build_from_opensky(
            args.start, args.stop, max_flights=args.max_flights, out_path=out
        )

    if ds.is_empty():
        raise SystemExit("Empty featured dataset — check data source")
    print(f"OK: {len(ds)} rows → {out}")
    print("Schema sample:", list(ds.columns)[:20])


if __name__ == "__main__":
    main()
