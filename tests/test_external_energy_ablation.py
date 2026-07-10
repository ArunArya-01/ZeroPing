"""Tests for the external Energy-feature ablation (physics/external_energy_ablation.py).

The ablation runs the equivalent AeroTwin Energy-feature protocol (V3 E6) on an
independent dataset to verify whether energy-state representations improve
prediction outside the original data.

Pure-logic helpers are unit tested directly. The end-to-end ablation run is
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
from physics import external_energy_ablation as ea  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
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

    # Physics baseline + feature-driven residual so a model can learn.
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


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_base_feature_cols_includes_physics_when_present():
    df = _make_synthetic_external()
    cols = ea.base_feature_cols(df)
    assert "physics_fuel_kg" in cols
    assert "mean_altitude" in cols
    assert "aircraft_type" in cols


def test_base_feature_cols_omits_physics_when_absent():
    df = _make_synthetic_external().drop("physics_fuel_kg")
    cols = ea.base_feature_cols(df)
    assert "physics_fuel_kg" not in cols


def test_enrich_energy_weather_idempotent():
    df = _make_synthetic_external()
    out = ea.enrich_energy_weather(df)
    # All energy features that the source columns can support are added.
    assert "mean_specific_energy_jpkg" in out.columns
    assert "headwind_mps" in out.columns


def test_load_internal_e6_none_when_missing(tmp_path):
    assert ea.load_internal_e6(tmp_path / "nope.csv") is None


def test_load_internal_e6_normalizes(tmp_path):
    csv = tmp_path / "e6.csv"
    pl.DataFrame(
        {
            "comparison": [
                "Energy Hybrid vs OpenAP Hybrid",
                "Energy+Weather Hybrid vs OpenAP Hybrid",
            ],
            "delta_mae": [-1.8, -2.5],
        }
    ).write_csv(csv)
    norm = ea.load_internal_e6(csv)
    assert norm is not None
    assert set(norm["approach"].to_list()) == {"energy", "energy_weather"}
    energy = norm.filter(pl.col("approach") == "energy")["internal_delta_mae"][0]
    assert energy == pytest.approx(-1.8)


def test_contrast_ablation_null_internal():
    df = pl.DataFrame({"approach": ["base"], "delta_mae_vs_base": [0.0]})
    contrast = ea.contrast_ablation(df, None)
    assert "internal_delta_mae" in contrast.columns
    assert contrast["internal_delta_mae"].null_count() == contrast.height


# --------------------------------------------------------------------------- #
# End-to-end ablation run (slow, requires catboost)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_run_energy_ablation_end_to_end():
    pytest.importorskip("catboost")
    df = _make_synthetic_external()
    result = ea.run_energy_ablation(df, test_size=0.2, iterations=50)

    assert set(result["metrics"].keys()) == {"base", "energy", "energy_weather"}
    for key in ("base", "energy", "energy_weather"):
        m = result["metrics"][key]
        assert {"mae", "rmse", "r2"}.issubset(m.keys())
        assert m["mae"] > 0
        assert np.isfinite(m["r2"])
        assert m["n_features"] > 0

    n_test = result["n_test_intervals"]
    assert len(result["y_test"]) == n_test
    assert len(result["flight_ids_test"]) == n_test

    # Two augmentations (energy, energy_weather) produce two significance rows.
    assert len(result["significance"]) == 2
    for s in result["significance"]:
        for key in ("delta_mae", "ci_lower", "ci_upper", "bootstrap_p", "interpretation"):
            assert key in s
        assert 0.0 <= s["bootstrap_p"] <= 1.0


@pytest.mark.slow
def test_run_energy_ablation_partial_schema():
    """External datasets may lack many columns; the ablation must still run."""
    pytest.importorskip("catboost")
    df = _make_synthetic_external().select(
        [
            "flight_id",
            "duration_s",
            "actual_fuel_kg",
            "physics_fuel_kg",
            "mean_altitude",
            "mean_groundspeed",
            "cruise_fraction",
            "aircraft_type",
            "method",
        ]
    )
    result = ea.run_energy_ablation(df, test_size=0.2, iterations=50)
    # Energy feature set should still be buildable from the present columns.
    assert "energy" in result["metrics"]
    assert result["n_test_intervals"] > 0


@pytest.mark.slow
def test_tables_roundtrip(tmp_path):
    pytest.importorskip("catboost")
    df = _make_synthetic_external()
    result = ea.run_energy_ablation(df, test_size=0.2, iterations=50)

    res = ea.ablation_results_table(result)
    assert res.height == 3
    assert "delta_mae_vs_base" in res.columns

    sig = ea.ablation_significance_table(result)
    assert sig.height == 2
    assert "comparison" in sig.columns

    # Missing internal path -> contrast carries null internal column.
    contrast = ea.contrast_ablation(res, ea.load_internal_e6(tmp_path / "x.csv"))
    assert "internal_delta_mae" in contrast.columns
