"""Tests for the cross-dataset replication analysis.

The replication analysis runs the equivalent Flow-vs-Direct protocol across
several datasets and decides, per dataset, whether the qualitative finding
(Flow+Energy generalizes better than Direct) replicates. It then aggregates a
meta-verdict across datasets (see PROJECT_STATUS_REPORT §20, step 7).

Pure aggregation helpers are unit tested directly; the multi-dataset protocol
run is marked ``slow`` and skips cleanly when ``catboost`` is unavailable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")
from aerotwin.validation import cross_dataset_replication as cdr

from aerotwin.engine.eval_framework import CATEGORICAL

def _make_synthetic_external(n_flights: int = 30, intervals_per_flight: int = 20, seed: int = 42) -> pl.DataFrame:
    """Mirror of the synthetic-data builder used by the external generalization tests."""
    rng = np.random.default_rng(seed)
    rows = n_flights * intervals_per_flight
    fids = np.repeat([f"F{i:03d}" for i in range(n_flights)], intervals_per_flight)

    duration = rng.uniform(30, 400, size=rows)
    mean_altitude = rng.uniform(5000, 12000, size=rows)
    mean_groundspeed = rng.uniform(150, 350, size=rows)
    climb = rng.uniform(0.0, 0.4, size=rows)
    cruise = rng.uniform(0.3, 0.7, size=rows)
    descent = np.clip(1.0 - climb - cruise, 0.0, 1.0)
    ref_mass = rng.uniform(60000, 120000, size=rows)
    spec_energy = rng.uniform(1e6, 5e6, size=rows)
    energy_rate = rng.uniform(1e3, 1e4, size=rows)
    headwind = rng.uniform(-20, 20, size=rows)
    temperature = rng.uniform(210, 230, size=rows)
    phase = rng.choice(["climb", "cruise", "descent"], size=rows)
    ac_type = rng.choice(["A320", "B738", "A350"], size=rows)
    method = rng.choice(["acars", "adsb"], size=rows)
    origin = rng.choice(["EGLL", "LFPG", "EDDF"], size=rows)
    dest = rng.choice(["EHAM", "LEMD", "LIRF"], size=rows)

    physics = (
        0.5 * duration
        + 0.00002 * mean_altitude * duration
        + rng.normal(0, 5, size=rows)
    )
    actual = (
        physics
        + 1.0 * headwind
        + 0.000002 * spec_energy
        + rng.normal(0, 4, size=rows)
    )

    return pl.DataFrame(
        {
            "flight_id": fids,
            "duration_s": duration,
            "actual_fuel_kg": actual,
            "physics_fuel_kg": physics,
            "mean_altitude": mean_altitude,
            "mean_groundspeed": mean_groundspeed,
            "climb_fraction": climb,
            "cruise_fraction": cruise,
            "descent_fraction": descent,
            "ref_mass_kg": ref_mass,
            "mean_specific_energy_jpkg": spec_energy,
            "energy_rate_jpkg_s": energy_rate,
            "headwind_mps": headwind,
            "temperature_k": temperature,
            "phase": phase,
            "aircraft_type": ac_type,
            "method": method,
            "origin_icao": origin,
            "destination_icao": dest,
        }
    )


def _result_dict() -> dict:
    """A representative result dict as produced by ``run_protocol``."""
    return {
        "n_intervals": 1000,
        "n_test_intervals": 200,
        "n_test_flights": 10,
        "metrics": {
            "direct": {
                "label": "Direct · E+W",
                "target": "direct_fuel",
                "feature_group": "ew",
                "mae": 84.0,
                "rmse": 120.0,
                "r2": 0.81,
            },
            "flow": {
                "label": "Flow+Energy",
                "target": "fuel_flow",
                "feature_group": "flow_energy",
                "mae": 80.0,
                "rmse": 115.0,
                "r2": 0.83,
            },
        },
        "significance": {
            "delta_mae": -4.0,
            "ci_lower": -7.5,
            "ci_upper": -0.5,
            "bootstrap_p": 0.02,
            "interpretation": "Flow+Energy significantly better than Direct",
        },
        "y_test": np.array([1.0, 2.0]),
        "flight_ids_test": np.array(["A", "B"]),
    }


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _results_table(delta_mae: float, p_better: float, replicated: bool) -> pl.DataFrame:
    """Build an external_results_table-shaped frame for one dataset."""
    base = _result_dict()
    base["significance"]["delta_mae"] = delta_mae
    base["significance"]["bootstrap_p"] = 1.0 - p_better
    tbl = pl.DataFrame(
        {
            "approach": ["direct", "flow"],
            "label": ["Direct · E+W", "Flow+Energy"],
            "target": ["direct_fuel", "fuel_flow"],
            "feature_group": ["ew", "flow_energy"],
            "mae_kg": [84.0, 84.0 + delta_mae],
            "rmse_kg": [120.0, 115.0],
            "r2": [0.81, 0.83],
            "flow_minus_direct_delta_mae": [delta_mae, delta_mae],
            "ci_lower": [-7.5, -7.5],
            "ci_upper": [-0.5, -0.5],
            "bootstrap_p_flow_better": [p_better, p_better],
            "interpretation": ["x", "Flow+Energy significantly better than Direct"],
            "n_test_flights": [10, 10],
            "n_test_intervals": [200, 200],
        }
    )
    return tbl


# --------------------------------------------------------------------------- #
# dataset_replicates_flow_better
# --------------------------------------------------------------------------- #
def test_replicates_when_flow_lower_and_significant():
    tbl = _results_table(delta_mae=-4.0, p_better=0.98, replicated=True)
    dec = cdr.dataset_replicates_flow_better(tbl)
    assert dec["replicated"] is True
    assert dec["flow_mae_kg"] < dec["direct_mae_kg"]
    assert dec["delta_mae_kg"] == pytest.approx(-4.0)


def test_not_replicated_when_p_below_threshold():
    # Flow lower in MAE, but bootstrap probability below the 0.95 threshold.
    tbl = _results_table(delta_mae=-3.0, p_better=0.90, replicated=False)
    dec = cdr.dataset_replicates_flow_better(tbl)
    assert dec["replicated"] is False


def test_not_replicated_when_flow_not_lower():
    # High P but Flow MAE is not actually lower -> not replicated.
    tbl = _results_table(delta_mae=+2.0, p_better=0.98, replicated=False)
    dec = cdr.dataset_replicates_flow_better(tbl)
    assert dec["replicated"] is False


def test_replicates_requires_both_rows():
    with pytest.raises(ValueError):
        cdr.dataset_replicates_flow_better(pl.DataFrame({"approach": ["direct"]}))


# --------------------------------------------------------------------------- #
# build_replication_table
# --------------------------------------------------------------------------- #
def test_build_replication_table_shape_and_columns():
    results = [
        {"name": "PRC2025", "result": _result_dict(), "table": _results_table(-4.0, 0.98, True)},
        {"name": "ExternalA", "result": _result_dict(), "table": _results_table(+2.0, 0.98, False)},
    ]
    tbl = cdr.build_replication_table(results)
    assert tbl.height == 2
    expected = {
        "dataset",
        "n_test_flights",
        "n_test_intervals",
        "direct_mae_kg",
        "flow_mae_kg",
        "delta_mae_kg",
        "p_flow_better",
        "replicated",
        "interpretation",
    }
    assert expected.issubset(set(tbl.columns))
    assert tbl.filter(pl.col("dataset") == "PRC2025")["replicated"][0] is True
    assert tbl.filter(pl.col("dataset") == "ExternalA")["replicated"][0] is False


# --------------------------------------------------------------------------- #
# aggregate_replication
# --------------------------------------------------------------------------- #
def test_aggregate_all_replicated():
    results = [
        {"name": "A", "result": _result_dict(), "table": _results_table(-4.0, 0.98, True)},
        {"name": "B", "result": _result_dict(), "table": _results_table(-6.0, 0.99, True)},
    ]
    agg = cdr.aggregate_replication(cdr.build_replication_table(results))
    assert agg["n_datasets"] == 2
    assert agg["n_replicated"] == 2
    assert agg["fraction_replicated"] == pytest.approx(1.0)
    assert "all datasets" in agg["verdict"]


def test_aggregate_partial_replication():
    results = [
        {"name": "A", "result": _result_dict(), "table": _results_table(-4.0, 0.98, True)},
        {"name": "B", "result": _result_dict(), "table": _results_table(+2.0, 0.98, False)},
    ]
    agg = cdr.aggregate_replication(cdr.build_replication_table(results))
    assert agg["n_replicated"] == 1
    assert "Partial" in agg["verdict"]


def test_aggregate_none_replicated():
    results = [
        {"name": "A", "result": _result_dict(), "table": _results_table(+3.0, 0.98, False)},
    ]
    agg = cdr.aggregate_replication(cdr.build_replication_table(results))
    assert agg["n_replicated"] == 0
    assert "failed to replicate" in agg["verdict"]


def test_aggregate_empty():
    agg = cdr.aggregate_replication(pl.DataFrame())
    assert agg["n_datasets"] == 0
    assert "No datasets" in agg["verdict"]


# --------------------------------------------------------------------------- #
# End-to-end multi-dataset protocol run (slow, requires catboost)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_run_protocol_on_datasets_end_to_end():
    pytest.importorskip("catboost")
    ds = [
        ("PRC2025", _make_synthetic_external(seed=1)),
        ("ExternalA", _make_synthetic_external(seed=2)),
    ]
    results = cdr.run_protocol_on_datasets(
        [(n, d) for n, d in ds], test_size=0.2, iterations=50
    )
    assert len(results) == 2
    for entry in results:
        assert set(entry["result"]["metrics"].keys()) == {"direct", "flow"}
        assert entry["table"].height == 2

    table = cdr.build_replication_table(results)
    assert table.height == 2
    agg = cdr.aggregate_replication(table)
    assert agg["n_datasets"] == 2
    assert 0.0 <= agg["fraction_replicated"] <= 1.0


@pytest.mark.slow
def test_run_protocol_on_datasets_handles_missing_columns():
    """External datasets may lack weather/energy columns; avail() subsets them."""
    pytest.importorskip("catboost")
    slim = _make_synthetic_external(seed=3).select(
        [
            "flight_id",
            "duration_s",
            "actual_fuel_kg",
            "physics_fuel_kg",
            "mean_altitude",
            "phase",
            "aircraft_type",
        ]
    )
    results = cdr.run_protocol_on_datasets(
        [("SlimExternal", slim)], test_size=0.2, iterations=50
    )
    assert len(results) == 1
    assert results[0]["result"]["n_test_intervals"] > 0
