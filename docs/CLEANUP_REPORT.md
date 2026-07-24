# Project Cleanup Report — R4 Post-Mortem

**Date:** July 2026  
**Current Best:** Combined RMSE 221.33 kg (R3 Dynamic Mass Model + P1E calibration)  
**R4 Cruise Features:** ALL REJECTED (every family degraded RMSE)

---

## 1. Feature Decision Matrix

### R4 Cruise Features — ALL REJECTED

| Feature | Purpose | RMSE Impact | Decision | Reason |
|---------|---------|-------------|----------|--------|
| `r4_cruise_duration_s` | cruise_time = dur × cruise_frac | +1.30 | REJECT | Redundant: dur and cruise_frac already in base |
| `r4_cruise_altitude_m` | alt_used | +1.30 | REJECT | Redundant: mean/max_altitude in BASE_NUMERIC |
| `r4_cruise_mach_est` | TAS / speed_of_sound | +1.30 | REJECT | Redundant: tas_used used in energy features |
| `r4_cruise_tas_mps` | tas_used | +1.30 | REJECT | Redundant: tas_used already in parquet |
| `r4_cruise_fuel_flow_kgps` | phys / cruise_dur | +2.67 | REJECT | Redundant: physics_fuel_kg and dur in base |
| `r4_cruise_efficiency` | ff / (mass × cruise_dur) | +2.67 | REJECT | Derived from existing features |
| `r4_cruise_load_factor` | mass / MTOW | +2.67 | REJECT | R3 mass features already cover |
| `r4_cruise_altitude_band` | alt / 12500 | +4.14 | REJECT | Linear scaling of existing altitude |
| `r4_cruise_pct_max_alt` | alt / max_alt | +4.14 | REJECT | Ratio of existing features |
| `r4_cruise_spd_stability` | 1/(1+std_gs) | +4.14 | REJECT | Inverse of existing std_groundspeed |
| `r4_cruise_tailwind_mps` | -clip(headwind,0) | +4.54 | REJECT | Sign flip of existing headwind_mps |
| `r4_cruise_headwind_mps` | +clip(headwind,0) | +4.54 | REJECT | Clip of existing headwind_mps |
| `r4_cruise_alt_x_dur` | alt × dur × cf | +5.71 | REJECT | Collinear product of existing features |
| `r4_cruise_mach_x_dur` | mach × dur × cf | +5.71 | REJECT | Collinear product |
| `r4_cruise_ff_x_mass` | ff × mass | +5.71 | REJECT | Collinear product |
| `r4_cruise_mass_x_mach` | mass × mach | +5.71 | REJECT | Collinear product |
| `r4_cruise_tas_x_dur` | TAS × dur × cf | +5.71 | REJECT | Collinear product |
| `r4_cruise_tailwind_x_dur` | tailwind × dur | +5.71 | REJECT | Collinear product |
| `r4_cruise_headwind_x_dur` | headwind × dur | +5.71 | REJECT | Collinear product |
| `r4_cruise_alt_x_ff` | alt × ff | +5.71 | REJECT | Collinear product |

### R2 Heavy Feature Families — REDUNDANT (superseded by R3 Mass)

| Family | Features | RMSE Impact | Decision | Reason |
|--------|----------|-------------|----------|--------|
| B744/B77L/A306 descriptors | 3 rows added to CSV | −3.00 | KEEP (in CSV) | Fills missing OpenAP data |
| R2 Aircraft chars | engine_count, TWR, WL, payload | 0.00 | REMOVE from code | Redundant with R1 descriptors |
| R2 Mass proxies | oew_as_mass, tofr_mass, phase_mass | 0.00 | REMOVE from code | R3 mass model supersedes |
| R2 Cruise features | cruise_dur, alt_band, cruise_ff | 0.00 | REMOVE from code | Redundant (same logic as R4) |
| R2 Physics interactions | mtow×cruise_dur, wl×alt, etc. | 0.00 | REMOVE from code | Collinear products |

### R3 Dynamic Mass Features — KEPT

All 21 R3 mass features are in production via `physics/mass_model.py`.
Validation passed: 0 physics violations across 119,032 train intervals.

### R1 OpenAP Descriptors — KEPT

10 descriptors (mtow_kg, oew_kg, wing_area, max_thrust, etc.) + 8 interactions remain in `gap_closing.py`.
Used by: `train_heavy_specialist_r1()` / `predict_heavy_routed_r1()`.

---

## 2. Production Feature Inventory

### Base Ensemble (ew_feature_cols + R3 mass)

| Family | Count | Features |
|--------|-------|----------|
| BASE_NUMERIC | 17 | duration_s, start/end fraction, n_traj_pts, has_acars, altitude stats, groundspeed stats, vertical_rate stats, phase fractions |
| ENERGY_FEATURES | 11 | ref_mass_kg, pe, ke, specific_energy*, energy_change, energy_rate, efficiencies, cumulative |
| WEATHER_FEATURES | 6 | headwind, crosswind, temperature, pressure, isa_deviation, density_altitude |
| Physics | 1 | physics_fuel_kg |
| Categorical | 4 | aircraft_type, method, origin_icao, destination_icao |
| **R3 Mass** | **21** | tow, landing, mass_start/end/mean/min/max/std, consumed, rate, fuel_fraction, remaining_fuel, phase_mass, cruise_mass, wing_loading, oew_base, pe_j, ke_j, fuel_mass_efficiency, tow_mtow_ratio, cruise_mass_fuel_ratio |
| **Total** | **60** | |

### Heavy Specialist (R1 — optional, adds)

| Family | Count | Features |
|--------|-------|----------|
| OpenAP descriptors | 10 | mtow_kg, mlw_kg, oew_kg, mfc_kg, cruise_mach, cruise_range_km, wing_area_m2, wing_span_m, mmo, max_thrust_n |
| R1 interactions | 8 | cruise_alt_x_dur, mean_alt_x_dur, cruise_ratio_x_dur, wing_loading, thrust_loading, aspect_ratio, oew_ratio, fuel_capacity_ratio |

---

## 3. KEEP / REMOVE / ARCHIVE

### KEEP (Production)

| File/Component | Status |
|----------------|--------|
| `physics/mass_model.py` | Production — R3 dynamic mass model |
| `physics/gap_closing.py` | Production — P1E calibrators, R1 heavy specialist, ensemble utils |
| `physics/official_benchmark.py` | Production — frozen training, OOF matrix, meta-learner |
| `physics/feature_engineering.py` | Production — energy/operational features |
| `physics/weather_features.py` | Production — ISA and wind proxies |
| `physics/eval_framework.py` | Production — evaluation, bootstrap |
| `physics/openap_baseline.py` | Production — OpenAP integration |
| `physics/cruise_features.py` | KEEP (documented experiment) — not imported by production |
| `figures/table_aircraft_openap_descriptors.csv` | Production — with B744/B77L/A306 added |
| `notebooks/25_r3_dynamic_mass.py` | Production — R3 evaluation |
| `notebooks/26_r3_ensemble_mass.py` | Production — R3 ensemble evaluation |
| `notebooks/23_rmse_audit_agent.py` | Production — audit agent |

### REMOVE (Dead Code)

| File/Component | Reason |
|----------------|--------|
| Functions in `gap_closing.py`: `_compute_r2_features`, `_augment_heavy_r2`, `R2_*` constants, `r2_feature_cols`, `train_heavy_specialist_r2`, `predict_heavy_routed_r2` | R2 families show no marginal improvement; removed from code |
| `notebooks/27_r4_cruise_features.py` | All variants rejected; results preserved in figures |
| `notebooks/24_r2_heavy_features.py` | Superseded by R3; results preserved in figures |

### ARCHIVE (Keep for Reproducibility)

| File/Component | Status |
|----------------|--------|
| `figures/table_rmse_R4_cruise.csv` | Rejected experiment results |
| `figures/table_rmse_R2_full_leaderboard.csv` | R2 experiment results |
| `figures/table_rmse_R3_mass.csv` | R3 single-model results |
| `figures/table_rmse_R3_mass_ensemble.csv` | R3 ensemble results |
| `figures/r3_summary.json` | R3 summary |
| `figures/r4_summary.json` | R4 summary |
| `notebooks/21_rmse_r1_heavy_features.py` | R1 experiment (reproducible) |
| `figures/table_rmse_R1.csv` | R1 results |

### Files Safe to Delete

- `notebooks/24_r2_heavy_features.py` — replaced by R3
- `notebooks/27_r4_cruise_features.py` — rejected, results in `figures/`

---

## 4. Updated Production Architecture

```text
featured_dataset.parquet
    │
    ├── enrich_mass_from_columns()  [physics/mass_model.py]      ← R3: +21 mass features
    │
    ├── ew_feature_cols()           [physics/official_benchmark.py] ← 39 base features
    │
    ├── build_oof_matrix()          [physics/official_benchmark.py] ← 6-base ensemble
    │
    ├── ConditionalAffineCalibrator [physics/gap_closing.py]     ← P1E phase calibration
    │
    └── predict_ensemble()          [physics/gap_closing.py]
        │
        └── Optional: train_heavy_specialist_r1()  ← R1 descriptors (heavy only)
```

---

## 5. Verified Production Pipeline

The current best pipeline (Combined RMSE 221.33) uses:

1. ✅ `physics/mass_model.py` — Dynamic mass features (21 features)
2. ✅ `physics/official_benchmark.py` — 6-base OOF ensemble (XGB/LGBM/CatBoost × Direct/FuelFlow)
3. ✅ `physics/gap_closing.py` — P1E phase-conditional affine calibration
4. ✅ Ridge meta-learner (chosen over LGBM by GroupKFold CV)
5. ✅ Frozen hyperparameters (lr=0.05, n_estimators=300)
6. ✅ Train-only fits; Rank/Final evaluation only

Removed from production:
- ❌ R2 aircraft chars / mass proxies / cruise / interaction features
- ❌ R4 cruise features (physics/cruise_features.py kept as documented experiment only)
- ❌ All rejected calibrators (global affine, isotonic, class/haul affine)

---

## 6. Next Recommended Experiment

The remaining gap to winner (~201 kg) is **20.3 kg**. The error is concentrated in:

- Ultra-long-haul flights (≥8h): ~85% SSE
- B744: systematic over-prediction (+272 kg bias)

Highest-confidence next steps:
1. **Haul-aware specialist** for ≥8h flights (FuelFlow model on ultra-long subset)
2. **B744-specific calibration** or asymmetric loss to reduce +272 kg bias
3. These are **model architecture / routing** changes, not feature engineering — the feature set at 60 columns is at saturation for GBDT models
