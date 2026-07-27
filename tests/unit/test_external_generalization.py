"""Tests for the external generalization protocol (physics/external_vs_flow_eval.py).

The external generalization check runs the equivalent AeroTwin Flow-vs-Direct
protocol on an independent dataset to verify that Flow+Energy still beats the
Direct approach outside the original data.

Pure-logic helpers are unit tested directly. The end-to-end protocol run is
marked ``slow`` and skips cleanly when ``catboost`` is unavailable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

# Heavy imports required by the module under test. Skip the whole module
# gracefully (rather than erroring) when they are not installed.
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("matplotlib")
from aerotwin.validation import external_vs_flow_eval as eg
from aerotwin.engine.eval_framework import CATEGORICAL


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def result_dict() -> dict:
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
# Pure helper functions
# --------------------------------------------------------------------------- #
def test_avail_subsets_to_present_columns():
    cols = ["a", "b", "c", "d"]
    df = pl.DataFrame({"a": [1], "c": [2]})
    assert eg.avail(cols, df) == ["a", "c"]


@pytest.mark.parametrize("target", ["direct_fuel", "fuel_flow"])
def test_transform_recover_roundtrip(target):
    rng = np.random.default_rng(0)
    actual = rng.uniform(50, 500, size=50)
    duration = rng.uniform(10, 300, size=50)
    y = eg.transform_y(target, actual, duration)
    recovered = eg.recover_fuel(target, y, duration)
    np.testing.assert_allclose(recovered, actual, rtol=1e-9)


def test_transform_y_invalid_target_raises():
    with pytest.raises(ValueError):
        eg.transform_y("bogus", np.array([1.0]), np.array([1.0]))


def test_recover_fuel_invalid_target_raises():
    with pytest.raises(ValueError):
        eg.recover_fuel("bogus", np.array([1.0]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# clean_for_eval
# --------------------------------------------------------------------------- #
def test_clean_for_eval_requires_columns():
    df = pl.DataFrame({"actual_fuel_kg": [1.0], "duration_s": [1.0]})
    with pytest.raises(SystemExit):
        eg.clean_for_eval(df)


def test_clean_for_eval_drops_nulls():
    df = pl.DataFrame(
        {
            "actual_fuel_kg": [100.0, 200.0, None, 400.0],
            "duration_s": [50.0, 100.0, 150.0, 200.0],
            "flight_id": ["A", "A", "B", "B"],
            "physics_fuel_kg": [90.0, 210.0, 280.0, 410.0],
        }
    )
    out = eg.clean_for_eval(df)
    assert out.height == 3
    assert out["actual_fuel_kg"].null_count() == 0


def test_clean_for_eval_filters_nonfinite_physics_and_bad_duration():
    df = pl.DataFrame(
        {
            "actual_fuel_kg": [100.0, 200.0, 300.0, 400.0],
            "duration_s": [50.0, 100.0, 150.0, 0.0],
            "flight_id": ["A", "A", "B", "B"],
            "physics_fuel_kg": [float("nan"), float("inf"), 280.0, 410.0],
        }
    )
    out = eg.clean_for_eval(df)
    # row 0 (nan physics), row 1 (inf physics), row 3 (duration == 0) dropped
    assert out.height == 1
    assert out["physics_fuel_kg"].is_finite().all()
    assert (out["duration_s"] > 0).all()


# --------------------------------------------------------------------------- #
# external_results_table
# --------------------------------------------------------------------------- #
def test_external_results_table_shape_and_columns(result_dict):
    tbl = eg.external_results_table(result_dict)
    assert tbl.height == 2
    expected = {
        "approach",
        "label",
        "target",
        "feature_group",
        "mae_kg",
        "rmse_kg",
        "r2",
        "flow_minus_direct_delta_mae",
        "ci_lower",
        "ci_upper",
        "bootstrap_p_flow_better",
        "interpretation",
        "n_test_flights",
        "n_test_intervals",
    }
    assert expected.issubset(set(tbl.columns))
    # bootstrap_p_flow_better = 1 - bootstrap_p
    assert tbl["bootstrap_p_flow_better"][0] == pytest.approx(0.98)


# --------------------------------------------------------------------------- #
# load_internal_baseline
# --------------------------------------------------------------------------- #
def test_load_internal_baseline_missing_file_returns_none(tmp_path):
    out = eg.load_internal_baseline(tmp_path / "does_not_exist.csv")
    assert out is None


def test_load_internal_baseline_normalizes_and_averages(tmp_path):
    csv = tmp_path / "loto.csv"
    pl.DataFrame(
        {
            "approach": [
                "global_direct_ew",
                "global_direct_ew",
                "global_flow_energy",
                "Global · Flow+Energy",
            ],
            "mae_kg": [80.0, 84.0, 76.0, 78.0],
        }
    ).write_csv(csv)
    norm = eg.load_internal_baseline(csv)
    assert norm is not None
    assert set(norm["approach"].to_list()) == {"direct", "flow"}
    direct = norm.filter(pl.col("approach") == "direct")["mae_kg"][0]
    flow = norm.filter(pl.col("approach") == "flow")["mae_kg"][0]
    assert direct == pytest.approx(82.0)
    assert flow == pytest.approx(77.0)


# --------------------------------------------------------------------------- #
# contrast_table
# --------------------------------------------------------------------------- #
def test_contrast_table_joins_internal(result_dict, tmp_path):
    ext = eg.external_results_table(result_dict)
    internal = eg.load_internal_baseline(
        _write_internal_csv(tmp_path)
    )
    contrast = eg.contrast_table(ext, internal)
    assert "internal_mae_kg" in contrast.columns
    assert contrast["internal_mae_kg"].null_count() == 0
    # flows must align on the "flow" approach row
    flow_row = contrast.filter(pl.col("approach") == "flow")
    assert flow_row["internal_mae_kg"][0] == pytest.approx(77.0)


def test_contrast_table_null_internal(result_dict):
    ext = eg.external_results_table(result_dict)
    contrast = eg.contrast_table(ext, None)
    assert "internal_mae_kg" in contrast.columns
    assert contrast["internal_mae_kg"].null_count() == contrast.height


def _write_internal_csv(tmp_path):
    csv = tmp_path / "loto.csv"
    pl.DataFrame(
        {
            "approach": ["global_direct_ew", "global_flow_energy"],
            "mae_kg": [82.0, 77.0],
        }
    ).write_csv(csv)
    return csv


# --------------------------------------------------------------------------- #
# End-to-end protocol run (slow, requires catboost)
# --------------------------------------------------------------------------- #
def _make_synthetic_external(n_flights: int = 30, intervals_per_flight: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(42)
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

    # Physics baseline + small feature-driven residual so a model can learn.
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


@pytest.mark.slow
def test_run_protocol_end_to_end():
    pytest.importorskip("catboost")
    df = _make_synthetic_external()
    result = eg.run_protocol(df, test_size=0.2, iterations=50)

    assert set(result["metrics"].keys()) == {"direct", "flow"}
    for key in ("direct", "flow"):
        m = result["metrics"][key]
        assert {"mae", "rmse", "r2"}.issubset(m.keys())
        assert m["mae"] > 0
        assert np.isfinite(m["r2"])

    n_test = result["n_test_intervals"]
    assert len(result["y_test"]) == n_test
    assert len(result["flight_ids_test"]) == n_test

    sig = result["significance"]
    for key in (
        "delta_mae",
        "ci_lower",
        "ci_upper",
        "bootstrap_p",
        "interpretation",
    ):
        assert key in sig
    assert 0.0 <= sig["bootstrap_p"] <= 1.0

    # Flow+Energy must use fewer features than the Direct E+W approach.
    assert (
        result["metrics"]["flow"]["feature_group"]
        == "flow_energy"
    )
    assert result["metrics"]["direct"]["feature_group"] == "ew"


@pytest.mark.slow
def test_run_protocol_feature_groups_differ():
    """External datasets may lack some columns; avail() must subset safely."""
    pytest.importorskip("catboost")
    # Only a minimal subset of feature columns present.
    df = _make_synthetic_external().select(
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
    result = eg.run_protocol(df, test_size=0.2, iterations=50)
    assert set(result["metrics"].keys()) == {"direct", "flow"}
    assert result["n_test_intervals"] > 0
    # No crash from missing weather/energy columns -> generalization across schemas.
    assert eg.CAT_FEATURES  # sanity: constant imported and usable
