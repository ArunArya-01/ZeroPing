"""OpenSky Network historical data loader for AeroTwin external audit.

Data access
-----------
Preferred path: ``pyopensky.trino.Trino`` (academic Trino historical DB).
Fallback: ``traffic`` library history helpers when configured.

**Critical label caveat**
    OpenSky state vectors have **no native fuel labels**. Interval fuel targets
    produced by this module are **physics-derived** via OpenAP
    (``physics.openap_baseline.predict_fuel_intervals`` / FuelFlow.enroute).
    Treat results as a *physics-label robustness* and *telemetry-shift* test,
    not independent fuel validation.

Trajectory reconstruction is event-driven (ADS-B); sparsity patterns differ
from fused PRC ACARS/ADS-B data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
import polars as pl

from physics.external_audit.audit_utils import (
    DEFAULT_INTERVAL_S,
    MIN_INTERVAL_S,
    construct_fixed_intervals,
    ensure_standard_traj_columns,
)

LOGGER = logging.getLogger(__name__)

# OpenSky baro altitude is metres; geo_altitude also metres; velocity m/s;
# vertical_rate m/s. Ground speed from traffic/pyopensky may be in kts depending
# on backend — we normalise below.


def _try_import_trino():
    try:
        from pyopensky.trino import Trino

        return Trino
    except ImportError:
        return None


def _try_import_traffic():
    try:
        import traffic  # noqa: F401
        from traffic.data import opensky

        return opensky
    except ImportError:
        return None


def load_opensky_history(
    start: str | datetime,
    stop: str | datetime,
    *,
    icao24: str | Sequence[str] | None = None,
    callsign: str | Sequence[str] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    airport: str | None = None,
    limit_rows: int | None = 50_000,
) -> pl.DataFrame:
    """Query OpenSky historical state vectors.

    Parameters
    ----------
    start, stop:
        ISO date/datetime strings or datetime objects.
    icao24, callsign:
        Optional filters (single or list).
    bounds:
        Optional (west, south, east, north) degrees.
    airport:
        Optional ICAO airport filter (backend-dependent).
    limit_rows:
        Soft cap after load for pilot scale (None = no cap).

    Returns
    -------
    Polars DataFrame with raw OpenSky columns (backend-dependent) plus
    ``source="opensky"``. Empty frame if no client is available.

    Notes
    -----
    Requires academic Trino credentials for ``pyopensky`` (see pyopensky docs).
    Without credentials the function returns an empty DataFrame and logs a
    clear message so the rest of the audit stack can fall back to demo data.
    """
    Trino = _try_import_trino()
    opensky = _try_import_traffic()

    raw = None
    backend = None

    if Trino is not None:
        try:
            LOGGER.info(
                "Querying OpenSky Trino history start=%s stop=%s icao24=%s "
                "(ASSUMPTION: academic access configured)",
                start,
                stop,
                icao24,
            )
            trino = Trino()
            kwargs: dict[str, Any] = {"start": start, "stop": stop}
            if icao24 is not None:
                kwargs["icao24"] = icao24
            if callsign is not None:
                kwargs["callsign"] = callsign
            if bounds is not None:
                kwargs["bounds"] = bounds
            # API surface varies slightly across pyopensky versions
            try:
                raw = trino.history(**kwargs)
            except TypeError:
                # older signature
                raw = trino.history(start, stop, icao24=icao24)
            backend = "pyopensky.trino"
        except Exception as exc:
            LOGGER.warning("pyopensky Trino query failed: %s", exc)
            raw = None

    if raw is None and opensky is not None:
        try:
            LOGGER.info("Falling back to traffic.data.opensky.history")
            raw = opensky.history(
                start=start,
                stop=stop,
                icao24=icao24,
                callsign=callsign,
                bounds=bounds,
                airport=airport,
            )
            backend = "traffic.opensky"
        except Exception as exc:
            LOGGER.warning("traffic opensky history failed: %s", exc)
            raw = None

    if raw is None:
        LOGGER.error(
            "No OpenSky data retrieved. Install/configure pyopensky (Trino) or "
            "traffic, and ensure academic credentials. Returning empty DataFrame."
        )
        return pl.DataFrame()

    df = _to_polars(raw)
    if limit_rows is not None and len(df) > limit_rows:
        LOGGER.info("Truncating OpenSky result %d → %d rows (pilot cap)", len(df), limit_rows)
        df = df.head(limit_rows)

    df = df.with_columns(pl.lit("opensky").alias("source"))
    LOGGER.info("OpenSky load via %s: %d rows, cols=%s", backend, len(df), df.columns[:15])
    return df


def _to_polars(raw: Any) -> pl.DataFrame:
    """Convert traffic Flight/Traffic, pandas, or list-of-dicts to Polars."""
    if isinstance(raw, pl.DataFrame):
        return raw
    # traffic Traffic / Flight
    if hasattr(raw, "data"):
        data = raw.data
        if isinstance(data, pl.DataFrame):
            return data
        try:
            return pl.from_pandas(data)
        except Exception:
            pass
    try:
        import pandas as pd

        if isinstance(raw, pd.DataFrame):
            return pl.from_pandas(raw)
    except ImportError:
        pass
    if isinstance(raw, list):
        return pl.DataFrame(raw)
    raise TypeError(f"Unsupported OpenSky result type: {type(raw)}")


def clean_opensky_trajectory(
    df: pl.DataFrame,
    *,
    min_altitude_m: float = 0.0,
    max_altitude_m: float = 20_000.0,
    max_gap_s: float = 600.0,
    min_points: int = 10,
) -> pl.DataFrame:
    """Basic cleaning + schema normalisation for OpenSky state vectors.

    Steps
    -----
    1. Rename to AeroTwin columns (baro_altitude → altitude, etc.).
    2. Drop null lat/lon/alt; filter altitude range.
    3. Sort by (icao24, timestamp); drop exact duplicate timestamps per aircraft.
    4. Optionally split on large time gaps (gap segments get segment_id).

    Groundspeed: if values look like knots (median 50–600), convert to m/s.
    """
    if df.is_empty():
        return df

    # Common OpenSky / traffic column names
    rename = {}
    for src, dst in [
        ("baroaltitude", "altitude"),
        ("baro_altitude", "altitude"),
        ("geoaltitude", "altitude"),
        ("geo_altitude", "altitude"),
        ("velocity", "groundspeed"),
        ("groundspeed", "groundspeed"),
        ("vertrate", "vertical_rate"),
        ("vertical_rate", "vertical_rate"),
        ("lat", "latitude"),
        ("lon", "longitude"),
        ("time", "timestamp"),
        ("timestamp", "timestamp"),
        ("typecode", "typecode"),
        ("icao24", "icao24"),
        ("callsign", "callsign"),
    ]:
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
        elif src in df.columns and src != dst and dst in df.columns:
            pass  # keep dst
    if rename:
        df = df.rename(rename)

    df = ensure_standard_traj_columns(df)

    required = ["timestamp", "altitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        LOGGER.warning("OpenSky traj missing %s after rename; columns=%s", missing, df.columns)
        return pl.DataFrame()

    df = df.filter(
        pl.col("altitude").is_not_null()
        & pl.col("altitude").is_between(min_altitude_m, max_altitude_m)
    )
    if "latitude" in df.columns:
        df = df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

    # Groundspeed unit heuristic
    if "groundspeed" in df.columns:
        med = df["groundspeed"].drop_nulls().median()
        if med is not None and 50 < float(med) < 600:
            LOGGER.info(
                "OpenSky groundspeed median=%.1f looks like knots → converting to m/s",
                med,
            )
            df = df.with_columns((pl.col("groundspeed") * 0.514444).alias("groundspeed"))

    if "vertical_rate" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("vertical_rate"))
    if "groundspeed" not in df.columns:
        df = df.with_columns(pl.lit(200.0).alias("groundspeed"))

    sort_keys = [c for c in ("icao24", "callsign", "timestamp") if c in df.columns]
    if not sort_keys:
        sort_keys = ["timestamp"]
    df = df.sort(sort_keys)

    # Deduplicate timestamps per aircraft
    subset = [c for c in ("icao24", "timestamp") if c in df.columns]
    if subset:
        df = df.unique(subset=subset, keep="first")

    # Segment on large gaps
    if "icao24" in df.columns and "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp").diff().over("icao24").dt.total_seconds().alias("_dt")
        )
        df = df.with_columns(
            (pl.col("_dt").fill_null(0.0) > max_gap_s).cast(pl.Int64).alias("_gap")
        )
        df = df.with_columns(
            pl.col("_gap").cum_sum().over("icao24").alias("segment_id")
        ).drop(["_dt", "_gap"])
    else:
        df = df.with_columns(pl.lit(0).alias("segment_id"))

    # Filter tiny segments
    if "icao24" in df.columns:
        counts = df.group_by(["icao24", "segment_id"]).len().rename({"len": "n"})
        keep = counts.filter(pl.col("n") >= min_points).select(["icao24", "segment_id"])
        df = df.join(keep, on=["icao24", "segment_id"], how="inner")
    elif len(df) < min_points:
        LOGGER.warning("Trajectory has only %d points (< %d); returning empty", len(df), min_points)
        return pl.DataFrame()

    if "typecode" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("typecode"))
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("opensky").alias("source"))

    LOGGER.info(
        "Cleaned OpenSky trajectory: %d points, %d aircraft",
        len(df),
        df["icao24"].n_unique() if "icao24" in df.columns else 1,
    )
    return df


def split_into_flights(
    df: pl.DataFrame,
    *,
    id_col: str = "icao24",
) -> list[dict[str, Any]]:
    """Split cleaned state vectors into per-flight (icao24 × segment) records."""
    if df.is_empty():
        return []

    if id_col not in df.columns:
        fid = "opensky_flight_0"
        t0, t1 = df["timestamp"].min(), df["timestamp"].max()
        ac = df["typecode"].drop_nulls().first() if "typecode" in df.columns else "A320"
        return [
            {
                "flight_id": fid,
                "traj": df,
                "meta": {
                    "flight_id": fid,
                    "aircraft_type": str(ac) if ac else "A320",
                    "origin_icao": None,
                    "destination_icao": None,
                    "takeoff": t0,
                    "landed": t1,
                    "source": "opensky",
                    "label_source": "physics_openap",
                },
            }
        ]

    out: list[dict[str, Any]] = []
    group_cols = [id_col] + (["segment_id"] if "segment_id" in df.columns else [])
    for keys, group in df.group_by(group_cols, maintain_order=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        icao = keys[0]
        seg = keys[1] if len(keys) > 1 else 0
        fid = f"opensky_{icao}_{seg}"
        group = group.sort("timestamp")
        ac = None
        if "typecode" in group.columns:
            ac = group["typecode"].drop_nulls().first()
        t0, t1 = group["timestamp"].min(), group["timestamp"].max()
        callsign = None
        if "callsign" in group.columns:
            callsign = group["callsign"].drop_nulls().first()
        out.append(
            {
                "flight_id": fid,
                "traj": group.with_columns(pl.lit(fid).alias("flight_id")),
                "meta": {
                    "flight_id": fid,
                    "aircraft_type": str(ac) if ac else "A320",
                    "origin_icao": None,
                    "destination_icao": None,
                    "takeoff": t0,
                    "landed": t1,
                    "icao24": icao,
                    "callsign": callsign,
                    "source": "opensky",
                    "label_source": "physics_openap",
                },
            }
        )
    LOGGER.info("Split OpenSky data into %d flight segments", len(out))
    return out


def trajectory_to_intervals(
    traj: pl.DataFrame,
    interval_s: float = DEFAULT_INTERVAL_S,
    min_interval_s: float = MIN_INTERVAL_S,
) -> pl.DataFrame:
    """Fixed-width reporting intervals for a single trajectory.

    OpenSky has no native ACARS fuel windows; we use regular intervals so
    feature construction matches the AeroTwin interval schema.
    """
    if traj.is_empty() or "timestamp" not in traj.columns:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "duration_s": pl.Float64,
                "fuel_kg": pl.Float64,
            }
        )
    t0 = traj["timestamp"].min()
    t1 = traj["timestamp"].max()
    intervals = construct_fixed_intervals(
        t0, t1, interval_s=interval_s, min_interval_s=min_interval_s
    )
    # Placeholder fuel_kg — filled by physics labels
    return intervals.with_columns(pl.lit(None).cast(pl.Float64).alias("fuel_kg"))


def generate_physics_fuel_labels(
    traj: pl.DataFrame,
    intervals: pl.DataFrame | None = None,
    ac_type: str | None = None,
    flight_meta: dict[str, Any] | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
) -> pl.DataFrame:
    """Create interval fuel labels + features via OpenAP baseline.

    **LABEL SOURCE: physics_openap — NOT independent ground truth.**

    Sets ``actual_fuel_kg = physics_fuel_kg`` so the featured-dataset schema
    remains compatible with the AeroTwin eval stack. Downstream audits MUST
    report ``label_source=physics_openap`` when interpreting metrics.

    Residual_kg will be ~0 by construction unless a separate actual is later
    joined; for OpenSky pilots, prefer relative comparisons (energy ablation,
    model ranking) over absolute MAE vs "truth".
    """
    LOGGER.warning(
        "OpenSky labels are PHYSICS-DERIVED (OpenAP FuelFlow.enroute). "
        "actual_fuel_kg will equal physics_fuel_kg. This is a robustness / "
        "telemetry-shift experiment, not independent fuel validation."
    )

    if intervals is None or intervals.is_empty():
        intervals = trajectory_to_intervals(traj, interval_s=interval_s)

    fuel = intervals.select(
        [
            pl.col("idx"),
            pl.col("start"),
            pl.col("end"),
            pl.col("fuel_kg") if "fuel_kg" in intervals.columns else pl.lit(0.0).alias("fuel_kg"),
        ]
    ).with_columns(
        # Temporary placeholder; overwritten after OpenAP
        pl.col("fuel_kg").fill_null(0.0)
    )

    if ac_type is None and "typecode" in traj.columns:
        ac_type = traj["typecode"].drop_nulls().first()
    ac_type = str(ac_type or "A320")

    from physics.openap_baseline import predict_fuel_intervals

    # predict_fuel_intervals uses fuel_kg as actual; we first get physics then
    # overwrite actual with physics for schema compatibility.
    # Use a dummy actual so the function still emits rows.
    fuel_in = fuel.rename({"idx": "idx"}).with_columns(
        pl.when(pl.col("fuel_kg").is_null() | (pl.col("fuel_kg") == 0.0))
        .then(pl.lit(1.0))  # non-null dummy so rows are kept
        .otherwise(pl.col("fuel_kg"))
        .alias("fuel_kg")
    )

    featured = predict_fuel_intervals(traj, fuel_in, ac_type=ac_type, flight_meta=flight_meta)
    if featured.is_empty():
        return featured

    featured = featured.with_columns(
        pl.col("physics_fuel_kg").alias("actual_fuel_kg"),
        (pl.col("physics_fuel_kg") - pl.col("physics_fuel_kg")).alias("residual_kg"),
        pl.lit("physics_openap").alias("label_source"),
        pl.lit(True).alias("label_is_physics_derived"),
    )
    LOGGER.info(
        "Physics-labelled %d OpenSky intervals (ac=%s, mean physics_fuel_kg=%.2f)",
        len(featured),
        ac_type,
        float(featured["physics_fuel_kg"].drop_nulls().mean() or 0.0),
    )
    return featured


def load_opensky_flights(
    start: str | datetime,
    stop: str | datetime,
    *,
    icao24: str | Sequence[str] | None = None,
    max_flights: int | None = 20,
    interval_s: float = DEFAULT_INTERVAL_S,
    **query_kwargs: Any,
) -> list[dict[str, Any]]:
    """End-to-end: query → clean → split flights → attach interval shells.

    Does **not** run OpenAP (that happens in ``build_featured_audit``). Each
    record has ``traj``, ``fuel`` (empty fuel_kg), ``meta``.
    """
    raw = load_opensky_history(start, stop, icao24=icao24, **query_kwargs)
    if raw.is_empty():
        return []
    clean = clean_opensky_trajectory(raw)
    flights = split_into_flights(clean)
    if max_flights is not None:
        flights = flights[:max_flights]

    for rec in flights:
        rec["fuel"] = trajectory_to_intervals(rec["traj"], interval_s=interval_s)
    return flights


def make_synthetic_opensky_flights(
    n_flights: int = 5,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Synthetic multi-flight OpenSky-like records for offline pilot runs."""
    from physics.external_audit.audit_utils import synthesize_demo_trajectory

    out = []
    types = ["A320", "B738", "A321", "B77W", "E190"]
    for i in range(n_flights):
        fid = f"opensky_synth_{i:03d}"
        traj, fuel, meta = synthesize_demo_trajectory(
            flight_id=fid,
            n_points=100 + i * 5,
            ac_type=types[i % len(types)],
            seed=seed + i,
        )
        traj = traj.with_columns(
            pl.lit(f"{i:06x}").alias("icao24"),
            pl.lit("opensky").alias("source"),
        )
        meta = {
            **meta,
            "source": "opensky_synthetic",
            "label_source": "physics_openap",
            "origin_icao": "EHAM",
            "destination_icao": "LFPG",
        }
        out.append({"flight_id": fid, "traj": traj, "fuel": fuel, "meta": meta})
    return out


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="OpenSky history probe")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--stop", default="2024-01-01 06:00")
    p.add_argument("--icao24", default=None)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    if args.synthetic:
        flights = make_synthetic_opensky_flights(3)
        print(f"Synthetic flights: {len(flights)}")
        print(flights[0]["traj"].head())
    else:
        df = load_opensky_history(args.start, args.stop, icao24=args.icao24, limit_rows=5000)
        print(df.head() if not df.is_empty() else "Empty — check credentials / use --synthetic")
