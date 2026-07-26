# Production Cleanup Verification Report

**Date:** 2026-07-26
**Protected baseline:** Combined RMSE 221.33 kg

---

## Verification Results

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Combined RMSE | 221.33 kg | 221.33 kg | ✅ |
| Rank RMSE | 232.53 kg | 232.53 kg | ✅ |
| Final RMSE | 213.73 kg | 213.73 kg | ✅ |
| Bias | +3.85 kg | +3.85 kg | ✅ |
| Heavy RMSE | 416.1 kg | 416.1 kg | ✅ |
| Narrow RMSE | 75.0 kg | 75.0 kg | ✅ |
| Delta vs 228.25 | -6.92 kg | -6.92 kg | ✅ |

**All production metrics preserved.**

---

## Files Kept (Production)

| Path | Purpose |
|------|---------|
| `physics/mass_model.py` | R3 dynamic mass model (21 features) |
| `physics/gap_closing.py` | P1E calibration, R1 heavy specialist |
| `physics/official_benchmark.py` | 6-base ensemble, OOF matrix, meta-learner |
| `physics/feature_engineering.py` | Energy/operational features |
| `physics/weather_features.py` | ISA and wind proxies |
| `physics/eval_framework.py` | Evaluation, bootstrap, feature constants |
| `physics/openap_baseline.py` | OpenAP integration, TAS inference |
| `physics/cruise_features.py` | Documented experiment (not imported) |
| `notebooks/25_r3_dynamic_mass.py` | R3 single-model evaluation |
| `notebooks/26_r3_ensemble_mass.py` | R3 ensemble evaluation |
| `notebooks/23_rmse_audit_agent.py` | RMSE audit agent |
| `notebooks/21_rmse_r1_heavy_features.py` | R1 heavy specialist |
| `figures/table_rmse_R3_mass_ensemble.csv` | Production ensemble results |
| `figures/r3_ensemble_summary.json` | Production metrics |
| `figures/table_aircraft_openap_descriptors.csv` | With B744/B77L/A306 |

## Files Removed (from production path)

| Path | Reason |
|------|--------|
| `notebooks/24_r2_heavy_features.py` | R2 families rejected, moved to archive |
| `notebooks/27_r4_cruise_features.py` | R4 cruise rejected, moved to archive |
| `notebooks/28_r5_sample_weights.py` | R5 weights rejected, moved to archive |
| `notebooks/22_r2_fuel_flow_audit.py` | R2 audit superseeded, moved to archive |
| R2 feature functions in `gap_closing.py` | Dead code, removed in prior cleanup |

## Files Archived (for reproducibility)

| Path | Description |
|------|-------------|
| `archive/experiments/22_r2_fuel_flow_audit.py` | R2 fuel flow audit |
| `archive/experiments/24_r2_heavy_features.py` | R2 heavy expansion |
| `archive/experiments/27_r4_cruise_features.py` | R4 cruise engineering |
| `archive/experiments/28_r5_sample_weights.py` | R5 sample weighting |
| `archive/figures/table_rmse_R2_*.csv` | R2 results |
| `archive/figures/table_rmse_R4_*.csv` | R4 results |
| `archive/figures/table_rmse_R5_*.csv` | R5 results |
| `archive/figures/r2_summary.json` | R2 summary |
| `archive/figures/r4_summary.json` | R4 summary |
| `archive/figures/r5_summary.json` | R5 summary |

## Production Feature Inventory (60 features)

| Family | Count | Examples |
|--------|-------|----------|
| Base trajectory stats | 17 | duration_s, altitude stats, groundspeed stats, phase fractions |
| Energy-state | 11 | ref_mass_kg, pe, ke, specific_energy, energy_change, efficiency |
| Weather proxies | 6 | headwind_mps, temperature_k, pressure_pa, density_altitude_m |
| Physics baseline | 1 | physics_fuel_kg |
| Categorical | 4 | aircraft_type, method, origin_icao, destination_icao |
| **Dynamic mass (R3)** | **21** | tow_kg, landing_mass, interval_mass, consumption, phase_mass |

## Production Pipeline

```
featured_dataset.parquet
  → enrich_mass_from_columns() [mass_model.py]   +21 mass features
  → ew_feature_cols() [official_benchmark.py]     39 base features
  → build_oof_matrix() [official_benchmark.py]    6-base ensemble
  → P1E phase calibration [gap_closing.py]
  → predict_ensemble() [gap_closing.py]
  = 221.33 kg Combined RMSE
```

## Compliance Checklist

- [x] All R2 experimental code removed from `gap_closing.py`
- [x] Rejected experiment notebooks moved to `archive/experiments/`
- [x] Rejected experiment results moved to `archive/figures/`
- [x] Production pipeline modules all compile
- [x] Production metrics verified against `r3_ensemble_summary.json`
- [x] Production feature inventory documented
- [x] README updated with clean architecture
- [x] No dead code in production path
