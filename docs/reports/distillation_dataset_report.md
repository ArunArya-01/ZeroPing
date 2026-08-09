# Teacher Distillation Dataset Report

**Stage:** AeroTwin Distillation - Step 1 (export only)

**Teacher (frozen):** R3 dynamic mass (21 features) + 6-base GBDT ensemble + Ridge/LGBM meta + P1E phase-conditional affine

Reference published metrics (Rank/Final): Combined RMSE **221.33 kg**, Final RMSE **213.73 kg**, bias **~+3.7 kg** (`R3_P1E_phase_affine`).

---

## Dataset summary

| Field | Value |
|-------|------:|
| Number of samples | **119,032** |
| Number of features (ensemble input) | **60** |
| Dataset size (parquet) | **40.83 MB** (42,818,403 bytes) |
| Output path | `<project_root>/distillation_dataset.parquet` |
| Columns with any missing (null/NaN) | 25 |
| Total missing cells | 236,691 |
| Meta learner | `ridge` |
| Train OOF RMSE (pre-P1E) | 252.32 kg |
| Train OOF RMSE (teacher / P1E) | 250.27 kg |
| Train OOF bias (teacher) | +0.00 kg |
| Exported train teacher RMSE (sanity) | 250.27 kg |
| Exported train teacher bias (sanity) | +0.00 kg |

### Split counts

| Split | Rows |
|-------|-----:|
| train | 119,032 |

### Missing values (top feature columns)

| Feature | Missing count | Missing fraction |
|---------|--------------:|-----------------:|
| `mean_groundspeed` | 42,696 | 0.3587 |
| `max_groundspeed` | 42,696 | 0.3587 |
| `mean_vertical_rate` | 42,696 | 0.3587 |
| `climb_efficiency` | 42,696 | 0.3587 |
| `mean_altitude` | 3,947 | 0.0332 |
| `median_altitude` | 3,947 | 0.0332 |
| `max_altitude` | 3,947 | 0.0332 |
| `energy_efficiency` | 3,037 | 0.0255 |
| `physics_fuel_kg` | 3,037 | 0.0255 |
| `r3_fuel_mass_efficiency` | 3,037 | 0.0255 |
| `r3_cruise_mass_fuel_ratio` | 3,037 | 0.0255 |
| `std_altitude` | 2,987 | 0.0251 |
| `std_groundspeed` | 2,987 | 0.0251 |
| `std_vertical_rate` | 2,987 | 0.0251 |
| `mean_potential_energy_j` | 2,987 | 0.0251 |

### Exported auxiliary targets / signals

These are teacher soft labels or intermediate physics quantities already present in the pipeline (not recomputed beyond the frozen teacher path):

- `teacher_prediction`
- `ridge_prediction`
- `xgb_direct_prediction`
- `lgbm_direct_prediction`
- `cat_direct_prediction`
- `xgb_flow_prediction`
- `lgbm_flow_prediction`
- `cat_flow_prediction`
- `xgb_prediction`
- `lgbm_prediction`
- `cat_prediction`
- `calibrated_prediction`
- `openap_prediction`
- `residual`
- `phase`
- `dynamic_mass`
- `ref_mass_kg`
- `r3_mass_start_kg`
- `r3_mass_end_kg`
- `r3_mass_consumed_kg`
- `r3_phase_mass_kg`
- `r3_fuel_fraction`
- `p1e_phase_group`

### Feature columns (ensemble input order)

```
duration_s, start_fraction_of_flight, end_fraction_of_flight, n_traj_pts, has_acars_in_window, mean_altitude, median_altitude, max_altitude, std_altitude, mean_groundspeed, std_groundspeed, max_groundspeed, mean_vertical_rate, std_vertical_rate, climb_fraction, cruise_fraction, descent_fraction, ref_mass_kg, mean_potential_energy_j, mean_kinetic_energy_j, mean_specific_energy_jpkg, specific_energy_start, specific_energy_end, energy_change_jpkg, energy_rate_jpkg_s, climb_efficiency, energy_efficiency, cumulative_energy_change_jpkg, headwind_mps, crosswind_mps, temperature_k, pressure_pa, isa_deviation_k, density_altitude_m, physics_fuel_kg, aircraft_type, method, origin_icao, destination_icao, r3_tow_kg, r3_landing_mass_kg, r3_mass_start_kg, r3_mass_end_kg, r3_mean_mass_kg, r3_min_mass_kg, r3_max_mass_kg, r3_mass_std_kg, r3_mass_consumed_kg, r3_mass_rate_kgps, r3_fuel_fraction, r3_remaining_fuel_frac, r3_phase_mass_kg, r3_cruise_mass_kg, r3_wing_loading_cur, r3_oew_base_kg, r3_mean_pe_j, r3_mean_ke_j, r3_fuel_mass_efficiency, r3_tow_mtow_ratio, r3_cruise_mass_fuel_ratio
```

### Schema notes

- `ground_truth` = `actual_fuel_kg` (interval fuel burn, kg)
- `teacher_prediction` = final R3 teacher after P1E calibration
- `ridge_prediction` = meta-ensemble output before P1E
- `xgb_*` / `lgbm_*` / `cat_*` = base model kg predictions (Direct and Fuel-Flow)
- `xgb_prediction` / `lgbm_prediction` / `cat_prediction` = Direct-target aliases
- `openap_prediction` = existing `physics_fuel_kg` (OpenAP baseline)
- `dynamic_mass` = existing `r3_mean_mass_kg`
- `residual` = existing `residual_kg` (actual − OpenAP)
- Train rows use GroupKFold OOF base predictions (leakage-safe soft labels)

### What this stage does *not* do

- Train any neural student
- Implement distillation loss
- Change AeroTwin feature engineering or hyperparameters

*Generated 2026-07-29 19:26:02*
