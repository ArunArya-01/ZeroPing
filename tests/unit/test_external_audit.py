"""Unit tests for physics/external_audit (offline, no network / .mat required)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aerotwin.validation.audit.audit_utils import (
    classify_phase,
    compute_energy_rates,
    compute_phase_fractions,
    compute_sparsity_signals,
    construct_fixed_intervals,
    ensure_standard_traj_columns,
    integrate_rate,
    synthesize_demo_trajectory,
)
from aerotwin.validation.audit.build_featured_audit import (
    build_demo_featured,
    build_featured_from_trajectories,
    preferred_column_order,
)
from aerotwin.validation.audit.dashlink_loader import extract_fuel_series
from aerotwin.validation.audit.opensky_loader import (
    clean_opensky_trajectory,
    make_synthetic_opensky_flights,
)


def test_classify_phase_thresholds():
    assert classify_phase(5.0) == "climb"
    assert classify_phase(-5.0) == "descent"
    assert classify_phase(0.1) == "cruise"
    assert classify_phase(None) == "unknown"


def test_phase_fractions_sum_to_one():
    vr = np.array([5.0, 5.0, 0.0, 0.0, -5.0, -5.0])
    fr = compute_phase_fractions(vr)
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert fr["climb_fraction"] == pytest.approx(1 / 3)
    assert fr["descent_fraction"] == pytest.approx(1 / 3)


def test_energy_rates_nonzero_climb():
    alt = np.linspace(1000, 5000, 20)
    tas = np.full(20, 200.0)
    out = compute_energy_rates(alt, tas, duration_s=600.0, mass_kg=50_000.0)
    assert out["energy_change_jpkg"] is not None
    assert out["energy_change_jpkg"] > 0
    assert out["energy_rate_jpkg_s"] is not None


def test_sparsity_bins():
    s = compute_sparsity_signals(3, duration_s=600.0)
    assert s["sparsity_bin"] == "very_sparse"
    s2 = compute_sparsity_signals(100, duration_s=600.0)
    assert s2["sparsity_bin"] == "medium"


def test_construct_fixed_intervals():
    from datetime import datetime

    iv = construct_fixed_intervals(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 1, 0, 0),
        interval_s=600.0,
    )
    assert len(iv) == 6
    assert "start" in iv.columns and "end" in iv.columns


def test_integrate_rate_kg_s():
    from datetime import datetime, timedelta

    t0 = datetime(2024, 1, 1, 0, 0, 0)
    times = [t0 + timedelta(seconds=i * 10) for i in range(7)]
    rates = [1.0] * 7  # 1 kg/s for 60 s → ~60 kg
    kg = integrate_rate(times, rates, unit="kg_s")
    assert kg == pytest.approx(60.0, rel=0.05)


def test_ensure_standard_traj_columns():
    df = pl.DataFrame(
        {
            "time": [1_700_000_000, 1_700_000_060],
            "baro_altitude": [10000.0, 10100.0],
            "velocity": [200.0, 205.0],
            "vertical_rate": [1.0, 0.5],
        }
    )
    out = ensure_standard_traj_columns(df)
    assert "timestamp" in out.columns
    assert "altitude" in out.columns
    assert "groundspeed" in out.columns


def test_synthesize_demo_trajectory():
    traj, fuel, meta = synthesize_demo_trajectory(n_points=60)
    assert len(traj) == 60
    assert "fuel_kg" in fuel.columns
    assert meta["flight_id"]


def test_extract_fuel_flow_channels():
    ch = {
        "FuelFlow_Eng1": np.ones(50) * 500.0,
        "FuelFlow_Eng2": np.ones(50) * 500.0,
        "ALT": np.linspace(0, 10000, 50),
    }
    info = extract_fuel_series(ch)
    assert info["mode"] == "flow"
    assert len(info["names"]) == 2


def test_build_demo_featured():
    pytest.importorskip("openap")
    ds = build_demo_featured(n_flights=3, out_path=None)
    assert not ds.is_empty()
    assert "actual_fuel_kg" in ds.columns
    assert "physics_fuel_kg" in ds.columns
    assert "flight_id" in ds.columns
    assert "energy_rate_jpkg_s" in ds.columns or "mean_altitude" in ds.columns
    ordered = preferred_column_order(ds)
    assert ordered.columns[0] == "actual_fuel_kg"


def test_opensky_synthetic_physics_labels():
    pytest.importorskip("openap")
    flights = make_synthetic_opensky_flights(n_flights=2)
    ds = build_featured_from_trajectories(
        flights, dataset_source="opensky", force_physics_as_actual=True
    )
    assert not ds.is_empty()
    assert ds["label_is_physics_derived"].all()
    # residual ~0 when actual = physics
    res = ds["residual_kg"].drop_nulls().to_numpy()
    assert np.allclose(res, 0.0, atol=1e-6)


def test_clean_opensky_empty():
    assert clean_opensky_trajectory(pl.DataFrame()).is_empty()
