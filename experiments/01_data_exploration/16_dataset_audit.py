"""Official PRC2025 dataset audit (Train / Rank / Final).

Produces tables and figures under figures/ for split integrity, schema
comparison, and distributional checks. Does NOT train models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.data import AeroDataLoader
from aerotwin.engine.eval_framework import project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("dataset_audit")

OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)
SPLITS = ("train", "rank", "final")


def main() -> None:
    loader = AeroDataLoader()
    flightlists: dict[str, pl.DataFrame] = {}
    fuels: dict[str, pl.DataFrame] = {}
    traj_counts: dict[str, int] = {}
    traj_stems: dict[str, set[str]] = {}

    for s in SPLITS:
        LOGGER.info("Loading metadata for split=%s", s)
        flightlists[s] = loader.get_flightlist(s)
        fuels[s] = loader.get_fuel_labels(s)
        files = loader.list_flight_files(s)
        traj_counts[s] = len(files)
        stems = set()
        for p in files:
            stems.add(Path(p).stem)
        traj_stems[s] = stems

    # ----- table_dataset_audit.csv -----
    audit_rows = []
    for s in SPLITS:
        fl, fu = flightlists[s], fuels[s]
        audit_rows.append(
            {
                "split": s,
                "n_flightlist_rows": len(fl),
                "n_unique_flight_ids_flightlist": fl["flight_id"].n_unique(),
                "n_fuel_intervals": len(fu),
                "n_unique_flight_ids_fuel": fu["flight_id"].n_unique(),
                "n_trajectory_files": traj_counts[s],
                "n_aircraft_types": fl["aircraft_type"].n_unique(),
                "n_origin_icao": fl["origin_icao"].n_unique(),
                "n_destination_icao": fl["destination_icao"].n_unique(),
                "mean_fuel_kg": float(fu["fuel_kg"].mean()),
                "median_fuel_kg": float(fu["fuel_kg"].median()),
                "mean_interval_duration_s": float(
                    (fu["end"] - fu["start"]).dt.total_seconds().mean()
                ),
                "flight_date_min": str(fl["flight_date"].min()),
                "flight_date_max": str(fl["flight_date"].max()),
            }
        )
    audit_df = pl.DataFrame(audit_rows)
    audit_df.write_csv(OUT / "table_dataset_audit.csv")
    LOGGER.info("Wrote table_dataset_audit.csv\n%s", audit_df)

    # ----- table_schema_comparison.csv -----
    schema_rows = []
    for s in SPLITS:
        for table_name, df in (("flightlist", flightlists[s]), ("fuel", fuels[s])):
            for col, dtype in df.schema.items():
                schema_rows.append(
                    {
                        "split": s,
                        "table": table_name,
                        "column": col,
                        "dtype": str(dtype),
                    }
                )
        # one trajectory schema sample
        try:
            sample_id = flightlists[s]["flight_id"][0]
            traj = loader.load_flight_by_id(sample_id, split=s)
            for col, dtype in traj.schema.items():
                schema_rows.append(
                    {
                        "split": s,
                        "table": "trajectory_sample",
                        "column": col,
                        "dtype": str(dtype),
                    }
                )
        except Exception as exc:
            LOGGER.warning("traj schema sample failed for %s: %s", s, exc)
    schema_df = pl.DataFrame(schema_rows)
    schema_df.write_csv(OUT / "table_schema_comparison.csv")

    # Cross-split column set equality
    for table in ("flightlist", "fuel"):
        cols = {
            s: set(schema_df.filter((pl.col("table") == table) & (pl.col("split") == s))["column"].to_list())
            for s in SPLITS
        }
        LOGGER.info(
            "%s column equality train=rank=%s train=final=%s rank=final=%s",
            table,
            cols["train"] == cols["rank"],
            cols["train"] == cols["final"],
            cols["rank"] == cols["final"],
        )

    # ----- table_split_statistics.csv -----
    stat_rows = []
    for s in SPLITS:
        fl, fu = flightlists[s], fuels[s]
        dur = (fu["end"] - fu["start"]).dt.total_seconds()
        stat_rows.append(
            {
                "split": s,
                "fuel_kg_p05": float(fu["fuel_kg"].quantile(0.05)),
                "fuel_kg_p50": float(fu["fuel_kg"].quantile(0.50)),
                "fuel_kg_p95": float(fu["fuel_kg"].quantile(0.95)),
                "duration_s_p05": float(dur.quantile(0.05)),
                "duration_s_p50": float(dur.quantile(0.50)),
                "duration_s_p95": float(dur.quantile(0.95)),
                "intervals_per_flight_mean": float(len(fu) / max(fu["flight_id"].n_unique(), 1)),
            }
        )
    pl.DataFrame(stat_rows).write_csv(OUT / "table_split_statistics.csv")

    # ----- table_overlap_check.csv -----
    ids = {s: set(flightlists[s]["flight_id"].to_list()) for s in SPLITS}
    fuel_ids = {s: set(fuels[s]["flight_id"].to_list()) for s in SPLITS}
    pairs = [("train", "rank"), ("train", "final"), ("rank", "final")]
    overlap_rows = []
    for a, b in pairs:
        o_fl = ids[a] & ids[b]
        o_fu = fuel_ids[a] & fuel_ids[b]
        o_traj = traj_stems[a] & traj_stems[b]
        # interval keys
        ka = fuels[a].select(
            pl.concat_str([pl.col("flight_id"), pl.col("start").cast(pl.Utf8), pl.col("end").cast(pl.Utf8)]).alias("k")
        )["k"].to_list()
        kb = fuels[b].select(
            pl.concat_str([pl.col("flight_id"), pl.col("start").cast(pl.Utf8), pl.col("end").cast(pl.Utf8)]).alias("k")
        )["k"].to_list()
        o_iv = set(ka) & set(kb)
        overlap_rows.append(
            {
                "pair": f"{a}_vs_{b}",
                "flight_id_overlap": len(o_fl),
                "fuel_flight_id_overlap": len(o_fu),
                "trajectory_stem_overlap": len(o_traj),
                "interval_key_overlap": len(o_iv),
                "pass_zero_overlap": int(
                    len(o_fl) == 0 and len(o_fu) == 0 and len(o_traj) == 0 and len(o_iv) == 0
                ),
            }
        )
    overlap_df = pl.DataFrame(overlap_rows)
    overlap_df.write_csv(OUT / "table_overlap_check.csv")
    LOGGER.info("Overlap check:\n%s", overlap_df)

    # ----- aircraft / route distributions -----
    ac_rows = []
    for s in SPLITS:
        counts = (
            flightlists[s]
            .group_by("aircraft_type")
            .len()
            .rename({"len": "n_flights"})
            .with_columns(pl.lit(s).alias("split"))
            .sort("n_flights", descending=True)
        )
        ac_rows.append(counts)
    ac_df = pl.concat(ac_rows)
    ac_df.write_csv(OUT / "table_aircraft_distribution.csv")

    route_rows = []
    for s in SPLITS:
        r = (
            flightlists[s]
            .with_columns(
                (pl.col("origin_icao") + "-" + pl.col("destination_icao")).alias("route")
            )
            .group_by("route")
            .len()
            .rename({"len": "n_flights"})
            .with_columns(pl.lit(s).alias("split"))
            .sort("n_flights", descending=True)
            .head(50)
        )
        route_rows.append(r)
    pl.concat(route_rows).write_csv(OUT / "table_route_distribution.csv")

    # ----- training protocol verification (static code audit summary) -----
    protocol_rows = [
        {
            "check": "featured_dataset.parquet built from train only via AeroDataLoader default split",
            "status": "PASS_expected",
            "detail": "aerotwin.engine.build_featured_dataset uses AeroDataLoader() default train",
        },
        {
            "check": "internal notebooks use featured_dataset*.parquet (train-derived)",
            "status": "PASS_expected",
            "detail": "No notebook loads split=rank or split=final for training",
        },
        {
            "check": "Rank/Final never used for hyperparameter tuning in repo scripts",
            "status": "PASS_expected",
            "detail": "Rank/Final previously unavailable; eval framework uses train-only parquet",
        },
        {
            "check": "Zero flight_id overlap train/rank/final",
            "status": "PASS" if all(overlap_df["pass_zero_overlap"].to_list()) else "FAIL",
            "detail": str(overlap_df.to_dicts()),
        },
    ]
    pl.DataFrame(protocol_rows).write_csv(OUT / "table_training_protocol_verification.csv")

    # ----- figure -----
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # A: flights per split
    ax = axes[0, 0]
    xs = list(SPLITS)
    ys = [audit_df.filter(pl.col("split") == s)["n_unique_flight_ids_flightlist"][0] for s in xs]
    ax.bar(xs, ys, color=["#2E5A88", "#C45C26", "#2F7D4F"])
    ax.set_title("Flights per split")
    ax.set_ylabel("N flights")
    for i, v in enumerate(ys):
        ax.text(i, v, str(v), ha="center", va="bottom")

    # B: intervals
    ax = axes[0, 1]
    ys = [audit_df.filter(pl.col("split") == s)["n_fuel_intervals"][0] for s in xs]
    ax.bar(xs, ys, color=["#2E5A88", "#C45C26", "#2F7D4F"])
    ax.set_title("Fuel intervals per split")
    ax.set_ylabel("N intervals")

    # C: top aircraft types (train)
    ax = axes[1, 0]
    top = ac_df.filter(pl.col("split") == "train").head(10)
    ax.barh(top["aircraft_type"].to_list()[::-1], top["n_flights"].to_list()[::-1], color="#2E5A88")
    ax.set_title("Train: top aircraft types")
    ax.set_xlabel("N flights")

    # D: fuel kg histograms (sample)
    ax = axes[1, 1]
    for s, color in zip(SPLITS, ["#2E5A88", "#C45C26", "#2F7D4F"]):
        vals = fuels[s]["fuel_kg"].clip(0, 2000).to_numpy()
        ax.hist(vals, bins=40, alpha=0.45, label=s, color=color, density=True)
    ax.set_title("Fuel kg per interval (density, clipped 0–2000)")
    ax.set_xlabel("fuel_kg")
    ax.legend()

    fig.suptitle("PRC2025 Official Split Distribution Audit", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote fig_dataset_distribution.png")

    # Paper comparison note
    paper = pl.DataFrame(
        [
            {"source": "paper_Table1", "split": "train", "flights": 11037, "intervals": 131530},
            {"source": "paper_Table1", "split": "rank", "flights": 1888, "intervals": 24289},
            {"source": "paper_Table1", "split": "final", "flights": 2836, "intervals": 37456},
            {"source": "hf_aerotwin", "split": "train", "flights": audit_rows[0]["n_unique_flight_ids_flightlist"], "intervals": audit_rows[0]["n_fuel_intervals"]},
            {"source": "hf_aerotwin", "split": "rank", "flights": audit_rows[1]["n_unique_flight_ids_flightlist"], "intervals": audit_rows[1]["n_fuel_intervals"]},
            {"source": "hf_aerotwin", "split": "final", "flights": audit_rows[2]["n_unique_flight_ids_flightlist"], "intervals": audit_rows[2]["n_fuel_intervals"]},
        ]
    )
    paper.write_csv(OUT / "table_paper_vs_hf_counts.csv")
    LOGGER.info("Audit complete.")


if __name__ == "__main__":
    main()
