# AeroTwin Project Status Report

**Date:** June 2026  
**Repository:** ZeroPing (AeroTwin)  
**Dataset:** [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) (EUROCONTROL PRC 2025)

---

## 1. Project Overview

### What is AeroTwin?

AeroTwin is a **physics-informed machine learning system** for predicting aircraft fuel burn on trajectory intervals. Each prediction target is a labeled ACARS fuel interval: the kilograms of fuel consumed between two reported fuel-on-board (FOB) timestamps during a commercial flight.

The project addresses a core challenge in aviation analytics: fuel burn depends on aircraft type, mass, flight phase, and kinematic state, but operational data is **partially observable**—dense ADS-B trajectories coexist with sparse ACARS reports, missing air-data, and unknown aircraft mass.

### Why aircraft fuel estimation matters

Accurate fuel burn prediction supports:

- **Emissions accounting** and regulatory reporting
- **Operational efficiency** (route, speed, and altitude optimization)
- **Digital twin** and fleet analytics applications
- **Benchmarking** physics models against real-world telemetry

The EUROCONTROL PRC 2025 challenge formalizes this problem using fused ADS-B + ACARS data across thousands of commercial flights.

### Physics-informed ML concept

AeroTwin follows a **hybrid architecture**:

1. A **physics baseline** (OpenAP) predicts fuel from aircraft type, representative altitude, speed, and flight phase.
2. A **machine learning model** learns the **residual**—the structured deviation between physics predictions and ACARS ground truth—using trajectory-derived features.

Final prediction:

```
predicted_fuel_kg = physics_fuel_kg + predicted_residual_kg
```

Alternatively, models can predict `actual_fuel_kg` directly while including `physics_fuel_kg` as a strong input feature.

### OpenAP as baseline

[OpenAP](https://github.com/junzis/openap) provides an aircraft performance model. AeroTwin uses `FuelFlow.enroute` with:

- TAS inference priority: Mach → CAS → groundspeed (fallback)
- Reference mass: MTOW × 0.75 (documented limitation)
- Per-interval representative kinematic point from the trajectory window

OpenAP is **not** expected to be production-accurate out of the box. Its role is to supply a **structured physics prior** whose errors are learnable.

### ML correction approach

Machine learning models consume engineered features from each fuel interval window:

- Trajectory statistics (altitude, speed, vertical rate)
- Flight phase fractions (climb / cruise / descent)
- Data quality signals (`n_traj_pts`, `has_acars_in_window`, `method`)
- Aircraft and route metadata

The ML layer corrects systematic physics bias—especially under sparse telemetry, uncertain mass, and fallback TAS inference.

---

## 2. Dataset Description

### Raw data (HuggingFace)

| Resource | Description | Scale (train split) |
|---|---|---|
| **Flightlist** | Flight metadata (type, origin, destination, times) | 11,037 flights |
| **Fuel labels** | ACARS-derived interval fuel burns | 131,530 intervals |
| **Trajectory parquets** | Per-flight fused ADS-B + ACARS time series | 10,000 usable flights |
| **Airports** | Airport reference table | — |

**Usable flights:** 10,000 flights have both metadata and trajectory files. The remaining ~1,037 training flights lack trajectory parquets and are excluded from modeling.

Remote access is implemented via `data/AeroDataLoader` using `hf://` paths and Polars—no full dataset download required.

### Featured dataset (`featured_dataset.parquet`)

Built by `physics/build_featured_dataset.py`, which:

1. Iterates over usable flights
2. Runs `physics/openap_baseline.predict_fuel_intervals()` per flight
3. Extracts per-interval trajectory features
4. Computes `residual_kg = actual_fuel_kg - physics_fuel_kg`
5. Attaches `flight_id` for flight-level evaluation

| Property | Value |
|---|---|
| Total intervals | 119,032 |
| Columns | 32 (31 engineered + `flight_id`) |
| Flights | 10,000 |
| Cleaned intervals (modeling) | 115,995 (after removing null/NaN physics rows) |
| Cleaned flights | 9,976 |

### Feature groups

| Group | Columns | Purpose |
|---|---|---|
| **Target** | `actual_fuel_kg` | Ground-truth fuel burn from ACARS FOB differences |
| **Physics baseline** | `physics_fuel_kg` | OpenAP prediction for the interval |
| **Residual target** | `residual_kg` | `actual - physics`; primary target for residual learning |
| **Flight identity** | `flight_id` | Enables strict flight-level train/test splits |
| **Aircraft metadata** | `aircraft_type` | ICAO typecode (26 types in data; A320 family dominant) |
| **Route metadata** | `origin_icao`, `destination_icao` | Airport pair |
| **Interval timing** | `duration_s`, `start_fraction_of_flight`, `end_fraction_of_flight` | Interval length and position within flight |
| **Altitude statistics** | `mean_altitude`, `median_altitude`, `max_altitude`, `std_altitude` | Cruise/climb context |
| **Groundspeed statistics** | `mean_groundspeed`, `std_groundspeed`, `max_groundspeed` | Speed variability in window |
| **Vertical-rate statistics** | `mean_vertical_rate`, `std_vertical_rate` | Climb/descent activity |
| **Phase fractions** | `climb_fraction`, `cruise_fraction`, `descent_fraction` | Phase mix (vr thresholds ±1.5 m/s) |
| **Trajectory density** | `n_traj_pts`, `has_acars_in_window` | Observability / sparsity signals |
| **Physics method** | `method` | How TAS was obtained (`tas_from_mach`, `tas_from_cas`, `tas_from_gs`, fallbacks) |
| **Debug / traceability** | `interval_idx`, `start`, `end`, `tas_used`, `alt_used`, `vs_used`, `phase` | Reproducibility and diagnostics |

---

## 3. Exploratory Data Analysis

EDA is implemented in `notebooks/01_overview_and_filters.py` through `04_physics_baseline_validation.py`, with summary analysis in `notebooks/analyze_featured_dataset.py`.

### Aircraft distribution

The fleet is dominated by narrow-body Airbus types:

| Aircraft | Flights (usable set) |
|---|---|
| A20N | 3,220 |
| A320 | 3,023 |
| A359 | 1,571 |
| B788 | 583 |
| B738 | 551 |
| A332 | 532 |
| A21N | 393 |
| … | 19 additional types |

Wide-bodies (A359, B789, B77W) have distinct fuel scales and error patterns. Typecode is one of the strongest ML predictors.

### Fuel distribution

- Interval fuel (`fuel_kg`) is **highly right-skewed**: median ~200 kg/interval with a heavy tail of large burns on long cruise segments.
- Flights contain a **median of ~10 labeled intervals** (typical range 5–25).
- Labeled time covers a **median of ~32%** of takeoff-to-landed duration—partial observability is structural, not incidental.

### Trajectory density

Trajectory point counts per interval (`n_traj_pts`) vary enormously:

- **Very sparse:** <5 points (~35–46% of intervals; often 2-point ACARS boundaries only)
- **Sparse:** 5–99 points
- **Medium:** 100–1,000 points
- **Dense:** >1,000 points

ADS-B provides dense kinematics but rarely includes Mach/TAS/CAS. ACARS rows carry air-data and fuel reports but are sparse and sometimes incomplete within labeled windows.

### OpenAP error analysis

Physics baseline validation (`notebooks/04_physics_baseline_validation.py`) revealed:

- **Systematic overprediction** on real data (positive residuals on average: physics under-estimates actual burn direction depends on sign convention; residuals correlate negatively with `physics_fuel_kg` at ρ ≈ −0.95)
- **Phase dependence:** cruise-dominated intervals show structured errors
- **Sparsity dependence:** very sparse intervals (few trajectory points) exhibit larger relative errors
- **Method dependence:** fallback TAS paths (`tas_from_gs`, `fallback_tas_200`) correlate with worse residuals

**Implications:**

- A pure physics model cannot be deployed without correction
- Residuals are structured and learnable from observability features
- Feature engineering must encode sparsity, phase, and data-source quality explicitly

---

## 4. Physics Baseline

### Implementation (`physics/openap_baseline.py`)

For each fuel interval:

1. Select trajectory points in `[start, end]`
2. Classify dominant phase from median vertical rate
3. Infer TAS from best available air data (Mach → CAS → groundspeed)
4. Call `FuelFlow.enroute(ac_type, mass, tas, alt, vs)` at a representative point
5. Integrate fuel flow over interval duration
6. Extract window statistics as ML features

Reference mass uses MTOW × 0.75 when true mass is unknown—a major documented error source.

### OpenAP-only metrics (flight-level test set)

Evaluated on 1,996 held-out flights (23,031 intervals):

| Metric | Value |
|---|---|
| **MAE** | 668 kg |
| **RMSE** | 1,582 kg |
| **R²** | −2.16 |

### Interpretation

- **Strong systematic bias:** negative R² indicates OpenAP performs worse than predicting the mean fuel burn.
- **Not deployable alone:** raw physics predictions are unsuitable as final outputs.
- **Valuable as a physics prior:** provides a structured baseline for hybrid/residual learning and interpretable comparison points.

---

## 5. Feature Engineering Pipeline

Feature extraction is integrated into `predict_fuel_intervals()` and materialized by `physics/build_featured_dataset.py`.

### Trajectory features

Computed over all trajectory points in the fuel interval window:

- **Altitude stats** capture cruise level and variability
- **Groundspeed stats** proxy energy regime
- **Vertical-rate stats** indicate climb/descent intensity

These features are available even when air-data (Mach/CAS) is missing, making them critical for sparse intervals.

### Flight phase features

Point-wise vertical-rate classification:

- `vr > +1.5 m/s` → climb
- `vr < −1.5 m/s` → descent
- otherwise → cruise

Fractions (`climb_fraction`, `cruise_fraction`, `descent_fraction`) summarize phase mix per interval. Fuel burn regimes differ substantially across phases.

### Data quality features

- **`n_traj_pts`:** number of trajectory samples in the window; primary sparsity signal
- **`has_acars_in_window`:** whether any ACARS-sourced points exist in the interval
- **`method`:** encodes how TAS was obtained for OpenAP (proxy for air-data quality)

### Physics features

- **`physics_fuel_kg`:** OpenAP prediction (usable as input feature or baseline)
- **`residual_kg`:** `actual_fuel_kg - physics_fuel_kg` (target for residual learning)

### Why each group exists

| Group | Rationale |
|---|---|
| Trajectory stats | Encode kinematic regime when physics inputs are incomplete |
| Phase fractions | Capture climb/cruise/descent fuel differences |
| Data quality | Model must condition on observability; EDA shows errors vary sharply with sparsity |
| Physics | Provide structured prior and explicit correction target |
| Metadata | Aircraft type and route proxy mass, engine, and operational context |

---

## 6. Machine Learning Baselines

Implemented in `notebooks/05_baseline_modeling.py` with supporting ablation scripts.

### Models tested

| Model | Configuration |
|---|---|
| **Linear Regression** | StandardScaler + one-hot categoricals |
| **Random Forest** | 100 trees, max depth 15 |
| **LightGBM** | 300 estimators, lr=0.05 |
| **XGBoost** | 300 estimators, lr=0.05, max depth 8 |

Categorical features (`aircraft_type`, `method`, `origin_icao`, `destination_icao`) are one-hot encoded. Numeric features use median imputation for missing kinematic values.

### Direct prediction

- **Target:** `actual_fuel_kg`
- **Features:** trajectory + metadata + `physics_fuel_kg`
- The model learns fuel burn end-to-end, with physics as a strong input

### Residual learning

- **Target:** `residual_kg`
- **Features:** trajectory + metadata (excluding `physics_fuel_kg`)
- **Final prediction:** `physics_fuel_kg + predicted_residual_kg`
- Explicitly separates physics prior from data-driven correction

### Initial results (row-level split, superseded)

Row-level 80/20 split showed MAE ~85–100 kg and R² ~0.91–0.94, but intervals from the same flight could leak between train and test. Flight-level validation was required.

---

## 7. Flight-Level Validation

### Why row-level split is insufficient

Multiple labeled intervals come from the same flight. Row-level splitting allows near-duplicate intervals (adjacent windows, shared aircraft/route context) to appear in both train and test, **inflating metrics**. Proper generalization requires **flight-level separation**.

### Strict split (`notebooks/05_baseline_modeling.py`)

All subsequent experiments (ablation, significance testing, V2/V3 feature studies) reuse this exact split:

- Split by `flight_id` (80/20, `random_state=42`)
- **Train:** 7,980 flights → 92,964 intervals
- **Test:** 1,996 flights → 23,031 intervals
- **Overlap:** 0 flights

### Results (held-out flights)

| Approach | Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|---|
| OpenAP only | — | 668 | 1,582 | −2.16 |
| Direct hybrid | Random Forest | 87.1 | 232.8 | 0.93 |
| Direct hybrid | XGBoost | 89.5 | 230.6 | 0.93 |
| Direct hybrid | LightGBM | 91.8 | 219.6 | 0.94 |
| Residual | XGBoost | 107.1 | 307.6 | 0.88 |
| Residual | LightGBM | 108.7 | 293.3 | 0.89 |
| Residual | Random Forest | 107.5 | 312.5 | 0.88 |

### Comparison to row-level split

| Metric | Row-level | Flight-level | Change |
|---|---|---|---|
| OpenAP MAE | 655 kg | 668 kg | +2% |
| Best direct hybrid MAE | ~86 kg | ~87–90 kg | modest |
| Best residual MAE | 100 kg | 107 kg | +7% |

### Statistical inference methodology

All hypothesis tests from `notebooks/07_significance_testing.py` onward use **flight-clustered bootstrap**:

- **10,000 bootstrap iterations**
- Resample **test flights with replacement** (not individual intervals)
- Preserve within-flight dependence among intervals
- Report **95% bootstrap confidence intervals** on ΔMAE and one-sided bootstrap *p*-values

**Wilcoxon signed-rank tests** are computed on paired per-interval absolute errors as a **supplementary** check. They treat intervals as independent and are often more optimistic when within-flight correlation is present.

**Primary inference criterion:** flight-clustered bootstrap CI. A gain is considered statistically supported only when the 95% CI excludes zero and `bootstrap_p < 0.05`.

### Conclusion

**Performance generalizes to unseen flights.** Direct hybrid models (trajectory + metadata + `physics_fuel_kg`) achieve MAE ~87–90 kg on held-out flights. Residual learning reduces MAE relative to raw OpenAP (~668 kg → ~107 kg) but **does not outperform direct hybrid prediction** on this feature set.

**Artifacts:** `figures/table_model_comparison_flight_split.csv`, `figures/fig_actual_vs_predicted.png`

---

## 8. Physics Ablation Study

**Scripts:** `notebooks/06_physics_ablation.py`, `notebooks/07_significance_testing.py`  
**Question:** How much does `physics_fuel_kg` contribute as an ML feature, and is the gain statistically real?

### Experimental conditions

| Condition | Description |
|---|---|
| **Full Hybrid** | All features including `physics_fuel_kg` |
| **No Physics** | Remove `physics_fuel_kg`; predict from trajectory/metadata only |
| **Physics Only** | Use raw OpenAP prediction (no ML) |

Evaluated on the same flight-level test set. Tree models (RF, XGBoost, LightGBM) tested.

### Descriptive results (point estimates)

| Condition | Best Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|---|
| Full Hybrid | Random Forest | 86.3 | 228.8 | 0.93 |
| Full Hybrid | XGBoost | 86.3 | 224.1 | 0.94 |
| No Physics | XGBoost | 89.5 | 230.6 | 0.93 |
| Physics Only | OpenAP | 667.6 | 1,582.4 | −2.16 |

### Bootstrap significance (Hybrid vs No Physics)

| Model | ΔMAE (kg) | 95% Bootstrap CI | Significant? | Effect size |
|---|---|---|---|---|
| **XGBoost** | −3.15 | [−7.08, −0.42] | **Yes** | Negligible (Cohen's *d* ≈ −0.04) |
| **Random Forest** | −0.86 | [−2.27, +0.52] | **No** (CI crosses zero) | Negligible |
| **LightGBM** | −2.63 | [−6.12, +0.01] | Marginal (CI barely touches zero) | Negligible |

### Interpretation

- **OpenAP-derived features provide only modest improvements** when rich trajectory features are available (~0.9–3.2 kg MAE depending on model).
- **XGBoost shows a statistically significant but practically small gain** (ΔMAE ≈ −3.1 kg; effect size negligible).
- **RF gain is not bootstrap-significant** despite a descriptive ~0.8 kg improvement.
- **OpenAP alone performs poorly** (MAE 668 kg)—not deployable as a standalone predictor.
- **Main message:** OpenAP helps modestly as an ML input feature; it is not the primary source of predictive power for strong tree ensembles.

**Artifacts:** `figures/table_physics_ablation.csv`, `figures/fig_physics_ablation.png`, `figures/table_significance_{rf,xgb,lgbm}.csv`, `figures/fig_bootstrap_{rf,xgb,lgbm}.png`

---

## 9. Sparsity Study

**Scripts:** `notebooks/07_sparsity_ablation.py` (descriptive), `notebooks/07_significance_testing.py` (inference)  
**Question:** Does physics become more valuable when telemetry is limited?

### Sparsity buckets (by `n_traj_pts`)

| Bucket | Definition | Test intervals |
|---|---|---|
| **Dense** | > 1,000 points | 2,432 |
| **Medium** | 100–1,000 points | 11,278 |
| **Sparse** | 10–99 points | 1,141 |
| **Very Sparse** | < 10 points | 8,180 |

### Descriptive ablation (LightGBM per bucket)

Earlier descriptive results suggested a large Sparse-bucket gain:

| Bucket | Full Hybrid MAE | No Physics MAE | Descriptive gain |
|---|---|---|---|
| Dense | 151.5 kg | 156.7 kg | 5.2 kg |
| Medium | 48.3 kg | 50.1 kg | 1.8 kg |
| Sparse | 74.5 kg | 88.9 kg | **14.4 kg** |
| Very Sparse | 129.9 kg | 130.6 kg | 0.7 kg |

### Bootstrap significance (Hybrid RF vs NoPhysics RF, flight-clustered)

| Bucket | ΔMAE (kg) | 95% Bootstrap CI | Significant? |
|---|---|---|---|
| Dense | −4.65 | [−11.48, +2.15] | No |
| Medium | −0.30 | [−1.28, +0.63] | No |
| Sparse | −0.12 | [−7.24, +7.17] | No |
| Very Sparse | −1.07 | [−3.65, +1.54] | No |

Inference uses **10,000 flight-clustered bootstrap resamples** (test flights resampled with replacement). Wilcoxon tests on intervals are supplementary.

### Interpretation — **sparse hypothesis rejected**

- **Bootstrap confidence intervals overlap zero in all sparsity buckets.** No statistically supported evidence exists that physics gains concentrate in sparse trajectories.
- The descriptive Sparse-bucket gap (74.5 vs 88.9 kg under LightGBM) **does not survive flight-level significance testing** under RF with bootstrap inference.
- **Previous claim that physics helps most in sparse intervals is no longer supported** and is explicitly rejected.
- Medium-density intervals remain easiest to predict descriptively (MAE ~48 kg), but physics benefit is not significant in any bucket.

**Artifacts:** `figures/table_sparsity_ablation.csv`, `figures/table_sparse_significance.csv`, `figures/fig_sparse_bucket_significance.png`

---

## 10. Physics-Informed Inductive Bias Study (V2/V3)

**Scripts:** `notebooks/08_physics_features_v2.py`, `notebooks/09_physics_features_v3.py`  
**Modules:** `physics/feature_engineering.py`, `physics/weather_features.py`, `physics/mlp_residual.py`

**Goal:** Determine which physical priors remain useful for strong gradient-boosted ensembles, beyond raw OpenAP fuel estimates.

All experiments reuse the strict flight-level split (7,980 / 1,996 flights) and flight-clustered bootstrap significance testing. Baseline reference: **OpenAP Hybrid (XGB), MAE ≈ 86.3 kg**.

### E2 — Energy-state features

Features: potential energy, kinetic energy, specific energy, energy rate, energy change, climb/energy efficiency, cumulative energy change (`physics/feature_engineering.py`).

| Model | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| Energy Hybrid (XGB) | **84.48** | −1.82 kg | [−2.92, −0.67] | **Accepted** |

**Conclusion:** Energy-state representations provide **statistically robust gains**. CI excludes zero.

### E3 — Operational features

Features: climb/descent duration, cruise speed variability, holding indicators, path efficiency proxies, altitude stability, segment acceleration (`physics/feature_engineering.py`).

| Model | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| Operational Hybrid (XGB) | 86.76 | +0.46 kg | [−0.10, +1.01] | **Rejected** |

**Conclusion:** Operational descriptors alone provide **no significant improvement**.

### E4 — Residual learning (tree models)

Architecture: predict `residual_kg`, final fuel = `physics_fuel_kg + predicted_residual`.

| Model | MAE (kg) | ΔMAE vs direct hybrid | Verdict |
|---|---|---|---|
| Residual-XGB | **107.1** | +20.8 kg | **Rejected** |

**Conclusion:** Residual learning **fails** to beat direct hybrid prediction. MAE ≈ 107 kg—worse than OpenAP hybrid (~86 kg). Negative result preserved from `notebooks/05_baseline_modeling.py`.

### E5 — Weather features

Features derived from ISA atmosphere at altitude and TAS/GS/track wind proxies: headwind, crosswind, temperature, pressure, ISA deviation, density altitude (`physics/weather_features.py`). No direct METAR/GRIB in dataset.

| Model | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| Weather Hybrid (XGB) | 86.59 | +0.28 kg | [−0.40, +1.07] | **Rejected** |

**Conclusion:** Weather-only features **not significant**; CI overlaps zero.

### E6 — Energy + Weather + OpenAP hybrid

Combines E2 energy-state and E5 weather proxies with OpenAP hybrid features.

| Model | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| **Energy+Weather Hybrid (XGB)** | **83.76** | **−2.55 kg** | **[−3.58, −1.50]** | **Accepted — best overall** |

**Conclusion:** Most successful inductive bias discovered. Strong bootstrap significance; CI excludes zero.

### E7 — MLP residual correction

Architecture: OpenAP prediction → MLP predicts residual (`physics/mlp_residual.py`). Compared against direct hybrid baselines.

| Model | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| MLP Residual (XGB slot) | **103.7** | +17.4 kg | [+7.84, +34.99] | **Rejected** |

**Conclusion:** Learned MLP residual correction **fails**; significantly worse than direct hybrid trees.

### E8 — BADA-style wind-adjusted physics

Conditional experiment (skipped when E5–E7 exceed 1.5 kg improvement threshold). Not run—E6 achieved 2.55 kg gain.

**Artifacts:** `figures/table_energy_results.csv`, `figures/table_operational_results.csv`, `figures/table_residual_results.csv`, `figures/table_v3_e6_combined_results.csv`, `figures/table_v3_leaderboard.csv`, `figures/fig_v3_leaderboard.png`, `figures/table_significance_*.csv`

---

## 11. Statistical Significance Framework

**Script:** `notebooks/07_significance_testing.py` (ablation); extended in V2/V3 experiment runners.

### Methods

| Component | Specification |
|---|---|
| **Bootstrap** | 10,000 iterations |
| **Resampling unit** | Test **flights** with replacement (not intervals) |
| **Dependence** | Preserves within-flight correlation among intervals |
| **Primary statistic** | ΔMAE = MAE(model A) − MAE(model B) on bootstrap resample |
| **Primary inference** | 95% bootstrap CI; one-sided `bootstrap_p = P(ΔMAE > 0)` |
| **Supplementary** | Wilcoxon signed-rank on paired interval absolute errors (one-sided) |
| **Effect size** | Cohen's *d* on paired error differences (Negligible / Small / Medium / Large) |

### Interpretation policy

- **Bootstrap CI is primary.** Wilcoxon *p*-values are reported but can be optimistic when intervals within a flight are correlated.
- Claims of improvement require **CI excluding zero** and `bootstrap_p < 0.05`.
- **Negative results are scientifically valuable** and are reported explicitly (residual learning, operational features, weather-only, sparsity hypothesis).

**Artifacts:** `figures/fig_bootstrap_{rf,xgb,lgbm}.png`, `figures/fig_sparse_bucket_significance.png`, `figures/table_significance_v3_all.csv`

---

## 12. Updated Scientific Findings

1. **OpenAP alone performs poorly** on held-out flights (MAE ≈ 668 kg, R² ≈ −2.2).

2. **OpenAP helps only modestly as an ML input feature** (~0.9–3.2 kg MAE; statistically significant for XGBoost but negligible effect size).

3. **Sparse hypothesis rejected:** bootstrap CIs overlap zero in all sparsity buckets; no evidence physics gains concentrate in sparse trajectories.

4. **Operational descriptors rejected** (E3): no bootstrap-significant improvement over OpenAP hybrid.

5. **Residual learning rejected** (E4, E7): tree and MLP residual architectures (~107–104 kg MAE) underperform direct hybrid prediction (~86 kg).

6. **Weather-only features rejected** (E5): CI overlaps zero; no significant gain alone.

7. **Energy-state representations significantly improve prediction** (E2): ΔMAE ≈ −1.8 kg, CI excludes zero.

8. **Energy + Weather achieves best performance** (E6): **MAE ≈ 83.7 kg**, ΔMAE ≈ −2.55 kg, strong bootstrap significance—**primary scientific contribution**.

9. **Not all physics-informed priors are equally useful.** Explicit energy conservation relationships add value; OpenAP point estimates, operational summaries, and residual-correction architectures add little.

10. **Strong tree ensembles already recover much trajectory information** independently; energy-state features encode structure trees do not fully infer from kinematic summaries alone.

11. **Partial observability remains structural** (median ~32% labeled flight time; many 2-point intervals).

12. **Aircraft type and interval duration remain dominant predictors** in baseline models (LightGBM importance from `notebooks/05_baseline_modeling.py`).

---

## 13. Current Project Status

| Milestone | Status |
|---|---|
| Dataset ingestion (`AeroDataLoader`, HuggingFace remote access) | ✅ Complete |
| Exploratory data analysis (notebooks 01–04) | ✅ Complete |
| Physics baseline (OpenAP per-interval pipeline) | ✅ Complete |
| Feature engineering (`featured_dataset.parquet`, 119k intervals, 62 columns with V3 features) | ✅ Complete |
| `flight_id` integration for flight-level splits | ✅ Complete |
| ML baselines (LR, RF, XGBoost, LightGBM) | ✅ Complete |
| Flight-level validation | ✅ Complete |
| Physics ablation study | ✅ Complete |
| Sparsity-conditioned ablation | ✅ Complete |
| Bootstrap significance testing (`07_significance_testing`) | ✅ Complete |
| Energy-state study (E2) | ✅ Complete |
| Operational feature study (E3) | ✅ Complete |
| Residual learning study (E4) | ✅ Complete |
| Weather feature study (E5) | ✅ Complete |
| Energy + Weather hybrid study (E6) | ✅ Complete |
| MLP residual study (E7) | ✅ Complete |
| SHAP explainability analysis | ⬜ Not Completed |
| Aircraft-level error analysis | ⬜ Not Completed |
| Cross-aircraft generalization (leave-one-type-out) | ⬜ Not Completed |
| Transformer residual model | ⬜ Not Completed |
| Paper drafting (Paper 1: characterization; Paper 2: hybrid model) | ⬜ In progress |

### Key artifacts

| File | Description |
|---|---|
| `featured_dataset.parquet` | Training-ready featured dataset (62 columns) |
| `physics/feature_engineering.py` | E2 energy + E3 operational features |
| `physics/weather_features.py` | E5 weather proxies |
| `physics/mlp_residual.py` | E7 MLP residual corrector |
| `physics/eval_framework.py` | Shared evaluation + bootstrap framework |
| `notebooks/07_significance_testing.py` | Bootstrap significance for ablations |
| `notebooks/08_physics_features_v2.py` | V2 experiments (E2–E4) |
| `notebooks/09_physics_features_v3.py` | V3 experiments (E5–E7) |
| `figures/table_v3_leaderboard.csv` | V3 model leaderboard |
| `figures/table_significance_v3_all.csv` | Combined significance results |
| `Dataset_explanation.md` | Dataset schema documentation |

### Strongest results

| Result | Detail |
|---|---|
| **Best overall MAE** | **83.76 kg** (Energy+Weather Hybrid, XGBoost, flight-level) |
| **Previous best (OpenAP hybrid)** | 86.31 kg (XGBoost) |
| **Bootstrap-significant gain** | ΔMAE −2.55 kg, 95% CI [−3.58, −1.50] |
| **Rejected: residual learning** | ~107 kg MAE (trees), ~104 kg (MLP) |
| **Rejected: sparse physics hypothesis** | All bucket bootstrap CIs include zero |
| **Rejected: weather-only** | CI [−0.40, +1.07] |

---

## Final Executive Summary

AeroTwin evolved from evaluating OpenAP into investigating **which physics-informed inductive biases remain useful** for modern gradient-boosted ensembles trained on partially observable trajectory data.

The project now demonstrates that:

- **Aviation fuel burn can be predicted accurately** on unseen flights when trajectory, metadata, and appropriate physics-informed features are available.
- **OpenAP alone is insufficient** (MAE ≈ 668 kg) and provides **only modest incremental value** as an ML input (~1–3 kg).
- **Several hypothesized mechanisms do not hold:** physics gains in sparse trajectories, operational descriptors alone, weather-only features, and residual-correction architectures (tree and MLP) all **fail bootstrap significance** or underperform direct hybrid models.
- **Explicit energy-state representations provide statistically robust improvements** (E2), and **Energy + Weather hybrid achieves the best performance** (E6): **MAE ≈ 83.7 kg** with flight-clustered 95% CI excluding zero.

**Best model discovered:** Energy+Weather Hybrid (XGBoost), MAE ≈ 83.7 kg. This is the **primary scientific contribution** of AeroTwin to date.

Negative results—residual learning failure, sparsity hypothesis rejection, weather-only insignificance—are preserved as scientifically valuable outcomes. Future work (SHAP, aircraft-level analysis, leave-one-type-out, transformer residuals) is deferred.

---

*Report updated June 2026. Reproduce: `python notebooks/05_baseline_modeling.py`, `06_physics_ablation.py`, `07_significance_testing.py`, `08_physics_features_v2.py`, `09_physics_features_v3.py`.*