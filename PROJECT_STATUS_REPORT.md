# AeroTwin Project Status Report

**Date:** July 2026  
**Repository:** ZeroPing (AeroTwin)  
**Dataset:** [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) (EUROCONTROL PRC 2025)

---

## Executive Summary

### Established findings (standard flight-level split; unseen flights, known aircraft families)

- **Energy-state representations** (E2) and **Energy + Weather** (E6) yield bootstrap-supported MAE gains over the OpenAP hybrid baseline on strict flight-level holdout (ΔMAE ≈ −1.8 to −2.6 kg; 95% CIs exclude zero).
- **Direct hybrid tree ensembles** generalize to unseen flights at MAE ≈ 84–88 kg and RMSE ≈ 210–224 kg.
- **Ensemble stacking** reaches **RMSE = 202.90 kg** on the PRC evaluation protocol — within ~1% of the published winner (200.83 kg) on the **same** dataset and metric. This is **competition benchmarking**, not external scientific validation.
- **Rejected under standard-split inference:** raw OpenAP alone; sparse-physics hypothesis; operational descriptors; weather-only; residual trees; MLP residual; heuristic mass features; vertical-rate embeddings; simple body-class routing as a universal LOTO solution.

### Suggestive / unresolved findings (aircraft-family shift; LOTO)

- **LOTO reveals a large transfer gap:** macro-average MAE rises from ~88 kg (standard split) to ~283 kg (global direct) — standard flight-level metrics substantially overestimate robustness under **unseen-aircraft-family** shift.
- **Fuel-Flow + Energy** achieves a **lower LOTO macro-average MAE** than Direct E+W by approximately **17.4 kg**, but the improvement is **heterogeneous** (7 wins, 5 losses across 12 types), **not statistically robust** under paired type-level or hierarchical bootstrap inference (both 95% CIs cross zero), and **strongly influenced by the B77W fold** (excluding B77W shrinks ΔMAE to ~−4.0 kg). Interpret as **suggestive, not confirmed**.
- **Physical aircraft-specification distance** correlates with LOTO error in the full 12-type sample (Pearson *r* ≈ 0.76 for NN distance vs direct MAE) but **collapses after removing B77W** (*r* ≈ 0.15). Physical similarity alone is insufficient to explain transfer failure.
- **No single inductive bias dominates** across unseen aircraft families. The unresolved question is: *under what forms of distribution shift does fuel-flow normalization improve cross-aircraft transfer, and when does it fail?*

### Failed / rejected hypotheses (preserved)

- Residual learning (trees and MLP); sparsity-conditioned physics gains; MoE/experts as ensemble improvement; body-class hierarchical routing at LOTO macro level; Mahalanobis physical distance as primary transfer predictor.

### Two major remaining research tasks

1. **Explain cross-aircraft transfer heterogeneity** — operational distribution-shift analysis on PRC data to determine when Fuel-Flow vs Direct prediction transfers better (§20 Priority 1–2).
2. **External dataset validation** — test whether qualitative findings (energy features, flow targets, cross-type degradation, physics insufficiency) replicate on an independent aviation dataset, not only within EUROCONTROL PRC 2025 (§20 External Dataset Validation).

### Immediate next step

**Operational distribution-shift analysis** on the current PRC dataset (§20 Priority 1). Dataset compatibility audit for NASA DASHlink and alternatives is **step 2** in the recommended execution order (§20).

**Paper submission:** Not ready. The project currently has strong internal validation under held-out flights and held-out aircraft types, but **external dataset validation remains necessary** before making broad claims that the identified inductive biases generalize across aviation datasets. Statistical protocol freeze and figure consolidation also remain required.

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
2. A **machine learning model** learns structured deviation from ACARS ground truth using trajectory-derived features.

Final prediction (direct hybrid):

```
predicted_fuel_kg = ML_model(trajectory_features, metadata, physics_fuel_kg)
```

Alternatively, models can predict `actual_fuel_kg` directly while including `physics_fuel_kg` as a strong input feature, or predict **fuel flow** (`actual_fuel_kg / duration_s`) and recover kilograms via multiplication.

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

### Three levels of generalization

| Level | Split protocol | What is held out | Current status |
|---|---|---|---|
| **Level 1** | Flight-level 80/20 | Unseen flights; aircraft types seen in training | **Strong** (MAE ~84–88 kg; ensemble RMSE ~203 kg) |
| **Level 2** | Leave-one-type-out (LOTO) | Entire aircraft families | **Weak** (macro MAE ~266–283 kg; 3× inflation vs Level 1) |
| **Level 3** | Transfer mechanism | When Flow vs Direct wins under shift | **Unresolved** — operational shift analysis pending |

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
| **Residual target** | `residual_kg` | `actual - physics`; target for residual learning (rejected) |
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

**Evidence scope:** All primary ablations and significance tests use this single EUROCONTROL PRC dataset. External validation on a second independent dataset is a major remaining requirement (§20 External Dataset Validation).

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

Wide-bodies (A359, B789, B77W) have distinct fuel scales and error patterns. Typecode is one of the strongest ML predictors under **Level 1** generalization but is **unavailable at deployment** for unseen types under **Level 2** (LOTO).

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

- **Structured residuals** correlated with physics scale (ρ ≈ −0.95 with `physics_fuel_kg`)
- **Phase dependence:** cruise-dominated intervals show structured errors
- **Sparsity dependence:** very sparse intervals exhibit larger relative errors
- **Method dependence:** fallback TAS paths correlate with worse residuals
- **Under LOTO:** OpenAP physics alone is catastrophic on some unseen heavy wide-bodies (e.g. B77W physics MAE ≈ 4,343 kg)

**Implications:**

- A pure physics model cannot be deployed without correction
- Residuals are structured and learnable from observability features under Level 1
- Feature engineering must encode sparsity, phase, and data-source quality explicitly
- Reference-mass assumptions break under cross-type shift

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
- **Valuable as a physics prior:** provides a structured baseline for hybrid learning and interpretable comparison points.

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

Fractions (`climb_fraction`, `cruise_fraction`, `descent_fraction`) summarize phase mix per interval.

### Data quality features

- **`n_traj_pts`:** number of trajectory samples in the window; primary sparsity signal
- **`has_acars_in_window`:** whether any ACARS-sourced points exist in the interval
- **`method`:** encodes how TAS was obtained for OpenAP (proxy for air-data quality)

### Physics features

- **`physics_fuel_kg`:** OpenAP prediction (usable as input feature or baseline)
- **`residual_kg`:** `actual_fuel_kg - physics_fuel_kg` (target for residual learning — **rejected**)

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
- **Verdict:** **Rejected** — underperforms direct hybrid (~107 kg vs ~87 kg MAE)

### Initial results (row-level split, superseded)

Row-level 80/20 split showed MAE ~85–100 kg and R² ~0.91–0.94, but intervals from the same flight could leak between train and test. Flight-level validation was required.

---

## 7. Flight-Level Validation (Level 1 Generalization)

### Why row-level split is insufficient

Multiple labeled intervals come from the same flight. Row-level splitting allows near-duplicate intervals to appear in both train and test, **inflating metrics**. Proper generalization requires **flight-level separation**.

### Strict split (`notebooks/05_baseline_modeling.py`)

All subsequent standard-split experiments reuse this exact split:

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

### Statistical inference methodology (standard split)

From `notebooks/07_significance_testing.py` onward:

| Component | Specification |
|---|---|
| **Bootstrap** | 10,000 iterations |
| **Resampling unit** | Test **flights** with replacement (not intervals) |
| **Primary inference** | 95% bootstrap CI on ΔMAE; `bootstrap_p < 0.05` |
| **Supplementary** | Wilcoxon signed-rank on paired interval errors |

**Inference units must match the claim:**

| Claim type | Correct resampling unit |
|---|---|
| Unseen-flight generalization | Flight-level bootstrap |
| Unseen-aircraft-family generalization (LOTO) | Type-level and/or hierarchical type→flight bootstrap |
| Interval-level Wilcoxon | Supplementary only when flights are correlated |

### Conclusion (Level 1)

**Performance generalizes to unseen flights** when aircraft families are represented in both train and test. Direct hybrid models achieve MAE ~87–90 kg. Residual learning does not outperform direct hybrid prediction.

**Artifacts:** `figures/table_model_comparison_flight_split.csv`, `figures/fig_actual_vs_predicted.png`

---

## 8. Physics Ablation Study

**Scripts:** `notebooks/06_physics_ablation.py`, `notebooks/07_significance_testing.py`  
**Question:** How much does `physics_fuel_kg` contribute as an ML feature?

### Bootstrap significance (Hybrid vs No Physics)

| Model | ΔMAE (kg) | 95% Bootstrap CI | Significant? |
|---|---|---|---|
| **XGBoost** | −3.15 | [−7.08, −0.42] | **Yes** |
| **Random Forest** | −0.86 | [−2.27, +0.52] | **No** |
| **LightGBM** | −2.63 | [−6.12, +0.01] | Marginal |

### Interpretation

OpenAP-derived features provide **modest** improvements when rich trajectory features are available. OpenAP alone performs poorly (MAE 668 kg).

**Artifacts:** `figures/table_physics_ablation.csv`, `figures/fig_physics_ablation.png`, `figures/table_significance_{rf,xgb,lgbm}.csv`

---

## 9. Sparsity Study — **Rejected**

**Scripts:** `notebooks/07_sparsity_ablation.py`, `notebooks/07_significance_testing.py`

### Bootstrap significance (Hybrid RF vs NoPhysics RF, flight-clustered)

| Bucket | ΔMAE (kg) | 95% Bootstrap CI | Significant? |
|---|---|---|---|
| Dense | −4.65 | [−11.48, +2.15] | No |
| Medium | −0.30 | [−1.28, +0.63] | No |
| Sparse | −0.12 | [−7.24, +7.17] | No |
| Very Sparse | −1.07 | [−3.65, +1.54] | No |

**Verdict:** Sparse hypothesis **rejected**. Bootstrap CIs overlap zero in all buckets.

**Artifacts:** `figures/table_sparsity_ablation.csv`, `figures/table_sparse_significance.csv`

---

## 10. Physics-Informed Inductive Bias Study (V2/V3)

**Scripts:** `notebooks/08_physics_features_v2.py`, `notebooks/09_physics_features_v3.py`  
**Scope:** Standard flight-level split only unless noted.

All experiments reuse the strict flight-level split and flight-clustered bootstrap. Baseline: **OpenAP Hybrid (XGB), MAE ≈ 86.3 kg**.

| Experiment | MAE (kg) | ΔMAE vs baseline | 95% Bootstrap CI | Verdict |
|---|---|---|---|---|
| **E2 — Energy Hybrid** | **84.48** | −1.82 | [−2.92, −0.67] | **Established** |
| E3 — Operational | 86.76 | +0.46 | [−0.10, +1.01] | **Rejected** |
| E4 — Residual trees | 107.1 | +20.8 | — | **Rejected** |
| E5 — Weather only | 86.59 | +0.28 | [−0.40, +1.07] | **Rejected** |
| **E6 — Energy+Weather** | **83.76** | **−2.55** | **[−3.58, −1.50]** | **Established — best single-model MAE** |
| E7 — MLP residual | 103.7 | +17.4 | [+7.84, +34.99] | **Rejected** |

**Artifacts:** `figures/table_v3_leaderboard.csv`, `figures/table_significance_v3_all.csv`

---

## 11. Statistical Significance Framework

**Script:** `notebooks/07_significance_testing.py`; extended in V2/V3 and `notebooks/17_loto_significance_and_transfer_distance.py`.

### Interpretation policy

- **Bootstrap CI is primary.** Do not claim significance when CI crosses zero.
- **Match resampling unit to claim:** flight-level for Level 1; type-level / hierarchical for LOTO.
- **Distinguish confirmatory vs exploratory:** LOTO transfer-distance correlations and physical-distance analyses are **exploratory** (n = 12 types).
- **Negative results are scientifically valuable** and are reported explicitly.
- **Competition proximity ≠ external validation.** Matching a leaderboard on the same dataset is benchmarking, not independent replication.

---

## 12. Updated Scientific Findings

Findings are classified as **Established**, **Suggestive**, **Exploratory**, or **Rejected**.

### Established (Level 1 — unseen flights, known aircraft families)

1. **OpenAP alone performs poorly** (MAE ≈ 668 kg, R² ≈ −2.2).
2. **OpenAP helps only modestly as an ML input feature** (~1–3 kg MAE; XGB bootstrap-significant but negligible effect size).
3. **Energy-state representations improve prediction** (E2): ΔMAE ≈ −1.8 kg, CI excludes zero.
4. **Energy + Weather achieves best single-model MAE** (E6): **83.76 kg**, ΔMAE ≈ −2.55 kg, CI excludes zero.
5. **Ensemble stacking reaches RMSE 202.90 kg** on the PRC protocol (same dataset as winner).
6. **Fuel-flow target is promising under standard split:** significant MAE gains in targeted V4 ablations (~79.5 kg single-model MAE equivalents); competitive ensemble RMSE with flow variants.
7. **SHAP and importance analyses agree:** `duration_s`, `physics_fuel_kg`, `ref_mass_kg`, and energy features dominate; weather contributes little.

### Rejected

8. **Sparse hypothesis rejected:** all sparsity-bucket bootstrap CIs include zero.
9. **Operational descriptors rejected** (E3); **weather-only rejected** (E5).
10. **Residual learning rejected** (E4, E7): trees ~107 kg, MLP ~104 kg vs direct ~86 kg.
11. **Heuristic mass features rejected** (V4); **vertical embeddings rejected** (V4).
12. **Simple body-class hierarchical routing rejected as universal LOTO solution** (macro MAE worse than global direct).
13. **MoE/experts do not beat ensemble** on standard split; marginal +2–3 kg RMSE single-model only.

### Suggestive / unresolved (Level 2 — unseen aircraft families; LOTO)

14. **Standard-split performance masks catastrophic cross-type failure.** Flight-level MAE ~88 kg vs LOTO macro MAE ~283 kg (direct) — **~3.2× inflation**. Level 1 metrics must not be extrapolated to unseen-airframe deployment.

15. **Fuel-Flow + Energy has a favourable LOTO macro point estimate but is not statistically confirmed.** Global Flow+Energy macro MAE **265.9 kg** vs Global Direct **283.2 kg** (ΔMAE ≈ **−17.4 kg**). However:
    - Flow wins **7 / 12** folds; Direct wins **5 / 12**
    - Median per-type ΔMAE ≈ **−16.3 kg**
    - Hierarchical bootstrap 95% CI: **approximately [−40.3, +18.6] kg** (crosses zero)
    - Type-level bootstrap 95% CI: **approximately [−54.9, +16.9] kg** (crosses zero)
    - Paired Wilcoxon *p* ≈ **0.235**; paired *t*-test *p* ≈ **0.381**
    - Excluding **B77W:** ΔMAE shrinks to **~−4.0 kg**

    **Accurate conclusion:** Fuel-Flow + Energy achieves a lower LOTO macro-average MAE than Direct E+W by approximately 17.4 kg, but the improvement is heterogeneous across aircraft types and is **not statistically robust** under paired type-level or hierarchical bootstrap inference. Both 95% confidence intervals cross zero, and the magnitude of the aggregate advantage is strongly influenced by the B77W fold. The result should be interpreted as **suggestive rather than confirmed**.

16. **No single target formulation wins universally under LOTO.** Large Flow improvements on some folds (B77W ≈ −165 kg; A332 ≈ −72 kg; A321 ≈ −59 kg; A20N ≈ −45 kg; B788 ≈ −43 kg) coexist with Direct advantages on others (B789 ≈ +80 kg; A333 ≈ +48 kg; B738 ≈ +41 kg; A359 ≈ +28 kg; A320 small Direct edge).

17. **The main unresolved question:** *Under what forms of distribution shift does fuel-flow normalization improve cross-aircraft transfer, and when does it fail?* Not: "Flow targets always generalize better."

### Exploratory (influence-sensitive; n = 12 types)

18. **Physical aircraft-specification distance** (OpenAP MTOW, OEW, wing area, cruise Mach, thrust, etc.) shows **apparent** correlation with LOTO error in the full sample (NN distance vs Direct MAE: Pearson *r* ≈ **0.76**, *p* ≈ **0.004**), but **collapses after removing B77W** (Pearson *r* ≈ **0.15**, *p* ≈ **0.67**). Do **not** claim physical distance robustly predicts transfer failure.

19. **Mahalanobis distance is numerically unstable** with 12 types and 10 correlated descriptors (large covariance condition numbers; some folds flagged `mahalanobis_ok = false`). Treat as supplementary/exploratory only.

20. **Partial observability remains structural** (median ~32% labeled flight time; many 2-point intervals).

---

## 13. Current Project Status

| Milestone | Status |
|---|---|
| Dataset ingestion (`AeroDataLoader`) | ✅ Complete |
| Exploratory data analysis (notebooks 01–04) | ✅ Complete |
| Physics baseline (OpenAP pipeline) | ✅ Complete |
| Feature engineering (`featured_dataset.parquet`) | ✅ Complete |
| Flight-level validation (Level 1) | ✅ Complete |
| Physics / sparsity / V2/V3 ablations | ✅ Complete |
| Bootstrap significance framework | ✅ Complete |
| V4 experiments (mass, flow, embeddings) | ✅ Complete |
| Ensemble / stacking / PRC benchmarking | ✅ Complete |
| SHAP explainability | ✅ Complete |
| Aircraft-level analysis | ✅ Complete (exploratory) |
| MoE / aircraft experts | ✅ Complete (exploratory; no ensemble gain) |
| **LOTO core evaluation** | ✅ Complete |
| **LOTO paired robustness analysis** | ✅ Complete |
| **LOTO bootstrap inference** | ✅ Complete |
| **LOTO leave-one-type sensitivity** | ✅ Complete |
| **Physical transfer-distance analysis** | ✅ Complete (**exploratory / influence-sensitive**) |
| **Operational distribution-shift analysis** | ⬜ **Next priority** |
| **External dataset compatibility audit** | ⬜ Next priority (after Priority 1 operational shift) |
| **Second dataset preprocessing pipeline** | ⬜ Pending |
| **Cross-dataset feature alignment** | ✅ Complete |
| **External Direct vs Flow evaluation** | ✅ Done (`physics/external_vs_flow_eval.py`) |
| **External Energy-feature ablation** | ✅ Done (`physics/external_energy_ablation.py`) |
| **External generalization test** | ✅ Done (`tests/test_external_generalization.py`) |
| **Cross-dataset replication analysis** | ✅ Done (`physics/cross_dataset_replication.py`) |
| **Shift-aware routing** | ⬜ Conditional on operational-distance findings |
| **Transformer residual** | ⬜ **Optional / low priority** |
| **Statistical protocol freeze** | ⬜ Pending |
| **Final figure/table consolidation** | ⬜ Pending |
| **Final paper drafting** | ⬜ Pending |
| Optuna / CatBoost tuning | ⬜ Deferred |

### Key artifacts

| File | Description |
|---|---|
| `featured_dataset.parquet` | Base featured dataset |
| `featured_dataset_mass.parquet` | Mass-enriched dataset (LOTO input) |
| `notebooks/15_leave_one_type_out.py` | Comprehensive LOTO evaluation |
| `notebooks/17_loto_significance_and_transfer_distance.py` | LOTO paired significance + transfer distance |
| `figures/table_loto_evaluation_master.csv` | Consolidated LOTO evaluation table |
| `figures/table_loto_paired_per_type.csv` | Per-type Flow vs Direct paired deltas |
| `figures/table_loto_paired_significance_summary.csv` | Pooled LOTO significance summary |
| `figures/table_loto_paired_sensitivity.csv` | B77W / subset sensitivity |
| `figures/table_loto_leave_one_type_robustness.csv` | LOO macro robustness |
| `figures/table_loto_transfer_correlations.csv` | Physical distance correlations |
| `figures/table_loto_transfer_correlations_sensitivity.csv` | B77W-excluded correlations |
| `figures/table_loto_transfer_distance_analysis.csv` | Merged distances + errors |
| `figures/table_loto_transfer_distances.csv` | NN, k3, Mahalanobis per fold |
| `figures/table_loto_transfer_influence.csv` | Leave-one-type correlation influence |
| `figures/table_aircraft_openap_descriptors.csv` | OpenAP physical descriptor table |
| `figures/loto_significance_transfer_conclusions.md` | Analysis conclusions |
| `figures/fig_loto_paired_delta_per_type.png` | Per-type ΔMAE bar chart |
| `figures/fig_loto_paired_bootstrap.png` | Bootstrap uncertainty plots |
| `figures/fig_loto_loo_robustness.png` | Leave-one-type macro robustness |
| `figures/fig_loto_flow_vs_direct.png` | Per-type scatter |
| `figures/fig_loto_distance_vs_mae.png` | Transfer distance vs LOTO MAE |
| `figures/fig_loto_distance_vs_inflation.png` | Transfer distance vs error inflation |
| `figures/table_v3_leaderboard.csv` | V3 model leaderboard |
| `figures/table_flow_vs_prc.csv` | Flow variants vs PRC winner |
| `figures/table_shap_catboost.csv` | SHAP feature importance |

### Strongest results (by generalization level)

| Level | Result | Detail |
|---|---|---|
| **Level 1 MAE** | **83.76 kg** | Energy+Weather Hybrid XGB, flight-level split |
| **Level 1 RMSE** | **202.90 kg** | 5f OOF + LGBM_meta ensemble |
| **Level 2 LOTO macro MAE (best point estimate)** | **265.9 kg** | Global Flow+Energy (vs 283.2 kg direct) — **suggestive, not confirmed** |
| **PRC benchmark** | 202.90 vs 200.83 kg RMSE | Same-dataset competition proximity (~1%); not external validation |

---

## 14. Competition Benchmarking

PRC2025 Winner RMSE = **200.83 kg** (official leaderboard).

| Model | RMSE (kg) |
|---|---|
| PRC2025 Winner | 200.83 |
| AeroTwin Ensemble (5f LGBM_meta) | 202.90 |
| FuelFlow+Energy+Mass | 204.18 |
| FuelFlow+Energy | 206.32 |
| Energy+Weather (direct, single model) | ~219–224 |

**Gap:** 202.90 − 200.83 = **2.07 kg RMSE** (~1.03%).

**Accurate conclusion:** AeroTwin reaches **competition-level RMSE on the same PRC dataset** under the project's ensemble pipeline. Wide flight-clustered bootstrap CIs on flow variants overlap the winner; this supports **benchmark proximity**, not **statistical equivalence** or **external validity**. Being within bootstrap uncertainty of a leaderboard score does not establish that the model would generalize equally on independent data or under aircraft-family shift.

---

## 15. V4 Experiments

V4 focused on fuel-flow targets, mass features, and vertical embeddings while preserving the scientific focus on **which physical inductive biases matter**.

### Fuel Flow Target — **Established under Level 1; suggestive under Level 2**

- Predict `fuel_flow_kgps = actual_fuel_kg / duration_s`; recover `fuel_kg = flow × duration_s`.
- **Standard split:** statistically significant MAE improvement over direct Energy+Weather in targeted ablations; best single-model MAE ≈ 79.5 kg equivalents; ensemble RMSE competitive (206–204 kg).
- **LOTO:** favourable macro point estimate (−17.4 kg vs direct) but **not statistically confirmed** (§18.3.1). Heterogeneous per-type results.

### Mass Features — **Rejected**

No significant gain; bootstrap CIs include zero.

### Vertical Embeddings — **Rejected**

No gain; bootstrap CI includes zero.

**Notebook:** `notebooks/10_fuel_flow_target.py`  
**Artifacts:** `figures/table_fuel_flow.csv`, `figures/fig_fuel_vs_flow.png`

---

## 16. Ensemble Study

**Notebooks:** `notebooks/08_ensemble.py`, `notebooks/11_stacking.py`, `notebooks/12_verify_ensemble.py`

**Best:** LGBM_meta achieving **RMSE = 202.90 kg**.

Stacking improves RMSE substantially over single Energy+Weather models (~212–224 → 202.90). Aircraft experts (~206.8 kg RMSE) **underperform** the global meta-ensemble.

---

## 17. Scientific Narrative

### Paper-level narrative (proposed)

> AeroTwin studies which physics-informed inductive biases remain useful for strong tabular ensembles under partially observed aircraft telemetry. While energy-state representations and fuel-flow formulations improve prediction under conventional held-out-flight evaluation, leave-one-aircraft-type-out testing reveals a large transfer gap and substantial heterogeneity across airframes. Fuel-flow normalization has a favourable but statistically uncertain aggregate effect under type shift, motivating analysis of the operational conditions under which physical normalization improves or harms transfer.

### Supported for paper (standard split)

- Energy-state representation improves flight-level prediction
- Energy + Weather combined representation gives a statistically supported gain
- Fuel-flow target is promising and strong under some settings
- Stacking produces competition-level performance on the PRC dataset

### Rejected for paper

- Raw OpenAP as standalone predictor
- Sparse-physics hypothesis
- Operational summary features as originally constructed
- Weather-only representation
- Residual trees; MLP residual correction
- Heuristic mass features; vertical-rate embeddings
- Simple body-class routing as universal transfer solution

### Suggestive / unresolved for paper

- Flow target improves LOTO macro MAE but lacks robust type-level significance
- Physical distance may relate to transfer error, but evidence is B77W-sensitive
- Conditional target selection may be useful, but must first be explained by operational shift analysis

### Current Research Question

**Under what measurable forms of train–test distribution shift does fuel-flow normalization improve cross-aircraft fuel prediction, and when does direct kilogram regression remain preferable?**

No single inductive bias dominates across unseen aircraft families.

### Immediate Next Experiment

**Operational distribution-shift analysis** (Priority 1; §20): For each LOTO held-out type, measure distributional distance between that type and training aircraft using **operational trajectory distributions** (duration, altitude, speed, VR, phase fractions, energy rates, trajectory density, TAS method, weather proxies). Test correlations with Direct LOTO MAE, Flow LOTO MAE, and **ΔMAE (Flow − Direct)**. The third relationship is the most scientifically important. Fuel labels may be used only in post-hoc diagnostics, not as deployment-time routing features.

**Second major task (execution step 2):** External dataset validation — NASA DASHlink compatibility audit and compact Direct/Flow/Energy replication on a second dataset if scientifically comparable (§20 External Dataset Validation).

---

## 18. Explainability, Specialization & Cross-Type Generalization

### 18.1 SHAP Explainability — ✅ Complete

**Script:** `notebooks/14_shap_explainability.py`

CatBoost on Energy + Weather + Physics (40 features); flight-level split; 5,000-interval SHAP sample.

**Top features:** `duration_s` (mean |SHAP| ≈ 278 kg), `physics_fuel_kg` (≈ 157 kg), `ref_mass_kg` (≈ 142 kg). Energy group collectively third; weather negligible.

**Scope:** Level 1 interpretability only.

**Artifacts:** `figures/table_shap_catboost.csv`, `figures/fig_shap_catboost_top_features.png`, `figures/fig_shap_catboost_summary.png`

---

### 18.2 Aircraft-Level Analysis — ✅ Exploratory

**Scripts:** `notebooks/01_*`, `09_aircraft_experts.py`, `12_verify_ensemble.py`

Per-type errors vary sharply (A20N standard-split MAE ~47 kg vs A306 ~700 kg on small samples). Specialists improve per-group RMSE modestly (~2–3 kg) on **standard split** but do not beat the stacked ensemble.

**Status:** Exploratory; does not establish cross-type transfer.

---

### 18.3 Leave-One-Type-Out (LOTO) — ✅ Complete

**What it is:** Each fold holds out **all flights of one aircraft type** from training and evaluates only on that type. Measures **Level 2** generalization: unseen airframe families.

**Implementation:** `notebooks/15_leave_one_type_out.py` on `featured_dataset_mass.parquet` (115,995 intervals; **12 types** with ≥80 flights). CatBoost; five approaches compared.

#### Master evaluation table (`figures/table_loto_evaluation_master.csv`)

| Experiment | Approach | Split | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|---|---|
| Standard split | Global · Direct · E+W | flight 80/20 | **88.1** | **210.7** | 0.944 |
| LOTO macro-avg | Global · Flow+Energy | LOTO | **265.9** | **445.5** | 0.230 |
| LOTO macro-avg | Global · Direct · E+W | LOTO | 283.2 | 469.4 | 0.136 |
| LOTO macro-avg | Global · Flow/Mass · E+W | LOTO | 321.7 | 505.2 | 0.580 |
| LOTO macro-avg | Body-class · Flow+Energy | LOTO | 378.1 | 600.1 | 0.200 |
| LOTO macro-avg | Body-class · Direct · E+W | LOTO | 386.8 | 562.6 | 0.402 |

#### Core LOTO findings

1. **Large generalization gap (Level 1 → Level 2):** MAE ~88 kg → ~283 kg (~3.2×). Standard flight-level splits overestimate deployment robustness under aircraft-family shift.

2. **Flow+Energy macro point estimate is favourable but not confirmed.** ΔMAE ≈ −17.4 kg (Flow better). See §18.3.1 for inference. **Do not** state that fuel-flow "confirms" or "proves" better unseen-aircraft transfer.

3. **Fold heterogeneity — no universal winner:**

| Flow large improvement (ΔMAE ≈ Flow−Direct) | Direct better (ΔMAE > 0) |
|---|---|
| B77W: **−165 kg** | B789: **+80 kg** |
| A332: **−72 kg** | A333: **+48 kg** |
| A321: **−59 kg** | B738: **+41 kg** |
| A20N: **−45 kg** | A359: **+28 kg** |
| B788: **−43 kg** | A320: small Direct edge |

4. **Body-class hierarchical routing:** macro MAE **worse** than global (+~104 kg direct); fails as universal solution. Fold-specific rescue only (e.g. B77W hier Flow ≈ 578 kg vs global direct ≈ 1,055 kg).

5. **Failures concentrate on heavy wide-bodies** when training is narrow-majority. B77W direct LOTO MAE ≈ 1,055 kg; OpenAP physics alone ≈ 4,343 kg on B77W.

**Artifacts:** `figures/table_loto_comprehensive.csv`, `figures/table_loto_macro_summary.csv`, `figures/table_loto_failure_analysis.csv`, `figures/table_loto_target_comparison.csv`, `figures/fig_loto_macro_comparison.png`, `figures/fig_loto_body_shift.png`, `figures/fig_loto_flow_vs_direct.png`

**Reproduce:** `PYTHONPATH=. python notebooks/15_leave_one_type_out.py`

---

### 18.3.1 LOTO Robustness, Sensitivity, and Paired Inference — ✅ Complete

**Script:** `notebooks/17_loto_significance_and_transfer_distance.py`

Macro-average point estimates alone are **insufficient** when there are only **12 held-out aircraft types**. Paired inference compares **Global Direct E+W** vs **Global Flow+Energy** on identical LOTO folds.

#### Paired comparison summary

| Metric | Value |
|---|---|
| Macro ΔMAE (Flow − Direct) | **≈ −17.4 kg** |
| Median per-type ΔMAE | **≈ −16.3 kg** |
| Flow wins / Direct wins / ties | **7 / 5 / 0** |
| Hierarchical bootstrap 95% CI (type→flight) | **≈ [−40.3, +18.6] kg** |
| Type-level bootstrap 95% CI (12 paired folds) | **≈ [−54.9, +16.9] kg** |
| Paired Wilcoxon (*p*, flow < direct) | **≈ 0.235** |
| Paired *t*-test | **≈ 0.381** |
| Macro ΔMAE excluding B77W | **≈ −3.95 kg** |

#### Interpretation

- Flow+Energy has a **favourable macro point estimate**.
- The direction is **not universal** (5/12 folds favour Direct).
- Aircraft-type **heterogeneity is substantial**.
- The aggregate improvement is **not statistically confirmed** at α = 0.05 (both bootstrap CIs cross zero; paired tests non-significant).
- **B77W contributes strongly** to macro advantage magnitude (−165 kg alone).
- Leave-one-type-out of macro ΔMAE remains negative for all exclusions, but magnitude is **small without B77W** (~−4 kg).

**Correct conclusion:** *Promising but heterogeneous and statistically uncertain.*

**Artifacts:**

- `figures/table_loto_paired_per_type.csv`
- `figures/table_loto_paired_sensitivity.csv`
- `figures/table_loto_paired_significance_summary.csv`
- `figures/table_loto_leave_one_type_robustness.csv`
- `figures/fig_loto_paired_delta_per_type.png`
- `figures/fig_loto_paired_bootstrap.png`
- `figures/fig_loto_loo_robustness.png`
- `figures/fig_loto_flow_vs_direct.png`

---

### 18.3.2 Aircraft Transfer Distance Analysis — ✅ Complete (exploratory)

**Script:** `notebooks/17_loto_significance_and_transfer_distance.py`

**Descriptors:** OpenAP `prop.aircraft()` + `prop.engine()` only (`figures/table_aircraft_openap_descriptors.csv`): MTOW, MLW, OEW, MFC, cruise Mach, cruise range, wing area, wing span, max thrust, MMO — complete for all 12 LOTO types.

**Distances** (held-out type → 11 training types, standardized features):

1. **Nearest-neighbor** Euclidean distance
2. **Mean k=3** nearest-neighbor distance
3. **Mahalanobis** to training centroid (pseudo-inverse) — **exploratory only** due to small *n* and ill-conditioned covariance

#### Full-sample correlations (n = 12) — **influence-sensitive**

| Distance | Outcome | Pearson *r* | *p* |
|---|---|---|---|
| NN | Direct LOTO MAE | **≈ +0.76** | **≈ 0.004** |
| k3-mean | Direct LOTO MAE | **≈ +0.68** | **≈ 0.015** |
| k3-mean | Direct LOTO RMSE | **≈ +0.79** | **≈ 0.002** |
| Mahalanobis | Direct LOTO MAE | ≈ −0.15 | ≈ 0.63 |

Bootstrap Pearson CIs for NN vs MAE are **very wide** (approximately [−0.35, 0.96]), reflecting *n* = 12 instability.

#### After excluding B77W (n = 11)

| Distance | Outcome | Pearson *r* | *p* |
|---|---|---|---|
| NN | LOTO MAE | **≈ +0.15** | **≈ 0.67** |
| k3-mean | LOTO MAE | **≈ +0.31** | **≈ 0.36** |

**Correct interpretation:** Simple aircraft-specification distance is associated with LOTO error in the full 12-type sample, but the relationship is **highly influence-sensitive** and **weakens substantially** after removing B77W. Physical similarity alone is **insufficient** to explain cross-aircraft transfer failure. Mahalanobis results are **not suitable for primary evidence** (large condition numbers; some folds invalid).

**Artifacts:**

- `figures/table_loto_transfer_correlations.csv`
- `figures/table_loto_transfer_correlations_sensitivity.csv`
- `figures/table_loto_transfer_distance_analysis.csv`
- `figures/table_loto_transfer_distances.csv`
- `figures/table_loto_transfer_influence.csv`
- `figures/fig_loto_distance_vs_mae.png`
- `figures/fig_loto_distance_vs_inflation.png`
- `figures/loto_significance_transfer_conclusions.md`

---

### 18.4 Transformer Residual — ⬜ Optional / Low Priority

Sequence-model residual correction has **not** been implemented. Tabular residual learning (trees ~107 kg, MLP ~104 kg) already **failed**. The ensemble is near competition RMSE on **Level 1**. The strongest unresolved issue is **cross-aircraft transfer (Level 2)**, not standard-split model capacity.

**Implement only if testing a specific hypothesis**, e.g.: *"Does preserving within-flight temporal structure improve transfer under aircraft-type shift?"* Do not implement merely because transformers are fashionable.

**Status:** Optional / low priority unless a clear sequence-model hypothesis is formulated.

---

### 18.5 Mixture-of-Experts (MoE) — ✅ Exploratory

**Script:** `notebooks/09_aircraft_experts.py`

Global vs hard experts vs soft MoE on **standard split**. Experts improve single-model RMSE by ~2–3 kg; **underperform** LGBM_meta ensemble (206.8 vs 202.9 kg). Under LOTO, specialization without in-type training data does not solve transfer.

**Do not build shift-aware routing until operational-shift evidence exists** (§20 Priority 7).

---

### 18.6 Implementation Status Summary

| Topic | Status | Primary script | Key verdict |
|---|---|---|---|
| SHAP explainability | ✅ Complete | `14_shap_explainability.py` | Duration, physics, mass, energy dominate |
| Aircraft-level analysis | ✅ Exploratory | `09_aircraft_experts.py` | Modest per-group gains; no ensemble benefit |
| LOTO core evaluation | ✅ Complete | `15_leave_one_type_out.py` | 3× MAE inflation vs standard split |
| LOTO paired robustness | ✅ Complete | `17_loto_significance_and_transfer_distance.py` | Flow macro −17 kg **suggestive, not confirmed** |
| Physical transfer distance | ✅ Exploratory | `17_loto_significance_and_transfer_distance.py` | B77W-sensitive; not robust predictor |
| Operational shift analysis | ⬜ Next | — | Priority 1 |
| Transformer residual | ⬜ Optional | — | Low priority |
| MoE / experts | ✅ Exploratory | `09_aircraft_experts.py` | Marginal single-model; worse than ensemble |

---

## 19. SOTA Position (PRC Dataset Benchmark)

| System | RMSE (kg) | Generalization tested |
|---|---|---|
| OpenAP | 1,582 | Level 1 flights |
| Direct Hybrid | ~224 | Level 1 flights |
| Energy+Weather | ~219–224 | Level 1 flights |
| FuelFlow+Energy | ~206 | Level 1 flights |
| Ensemble | **202.90** | Level 1 flights |
| PRC2025 Winner | **200.83** | Same competition metric |
| LOTO Global Direct | ~469 (macro RMSE) | **Level 2 types** |
| LOTO Global Flow+Energy | ~446 (macro RMSE) | **Level 2 types** |

**Do not conflate** competition RMSE (~203 kg) with LOTO macro RMSE (~446–469 kg). They measure different deployment assumptions.

---

## 20. What Still Needs to Be Done Before Paper Submission

### Priority 1 — Operational Distribution Shift Analysis (**immediate next experiment**)

For every LOTO held-out type, characterize **operational** train–test distance using trajectory distributions, not only static OpenAP specs.

**Candidate variables:** `duration_s`, altitude (mean/median/std), groundspeed, vertical rate, climb/cruise/descent fractions, energy-rate and kinetic/potential energy statistics, start/end flight fraction, `n_traj_pts`, TAS inference `method`, weather proxy distributions.

**Distance metrics:** Wasserstein distance; Jensen–Shannon divergence where appropriate; standardized mean difference; MMD if practical.

**Required analyses:**

- A. operational distance → Direct LOTO MAE
- B. operational distance → Flow LOTO MAE
- C. **operational distance → ΔMAE (Flow − Direct)** ← most important

**Goal:** Determine whether specific operational shifts explain when Flow normalization helps or hurts.

**Avoid leakage:** Fuel labels only in post-hoc diagnostics, not deployment-time routing features.

---

### Priority 2 — Explain Flow-vs-Direct Heterogeneity

Build a fold-level explanatory table (one row per held-out type):

- aircraft type, body class, MTOW, reference mass
- train/test flight counts, narrow/wide training composition
- physical NN / k3 distances
- operational distribution distances (from Priority 1)
- Direct MAE, Flow MAE, ΔMAE, Physics MAE

Then: correlation analysis, robust regression if justified, influence diagnostics, LOO sensitivity.

**Caution:** *n* = 12 types. Do not fit a complex meta-model and present it as reliable.

---

### External Dataset Validation (second dataset workstream)

AeroTwin's current empirical evidence is based primarily on the **EUROCONTROL PRC 2025** dataset. Although the project now includes strict flight-level evaluation, Leave-One-Type-Out evaluation, bootstrap inference, sensitivity analysis, and aircraft-transfer analysis, these are still **internal evaluations within one underlying data source**.

Therefore, validation on a **second independent dataset** is a major remaining requirement for stronger generalization claims.

#### Goal

Test whether AeroTwin's main scientific findings survive **dataset shift** rather than only train/test or aircraft-type shift within the PRC dataset.

The external validation should answer:

1. Do energy-state features improve prediction on another dataset?
2. Does Fuel-Flow target normalization outperform or compete with Direct fuel prediction?
3. Does the Direct-vs-Flow ranking remain heterogeneous under aircraft or operational shift?
4. Does physics-informed feature engineering remain useful when telemetry characteristics differ?
5. Does the large gap between standard evaluation and cross-aircraft evaluation appear in another dataset?

#### Candidate dataset

**Primary candidate:** NASA DASHlink Flight Data Recorder (FDR) dataset.

However, **do not** treat DASHlink as automatically suitable.

Before committing to it, perform a **dataset compatibility audit** covering:

- availability of fuel-flow, fuel-used, or equivalent fuel-consumption ground truth;
- sampling frequency and trajectory completeness;
- aircraft-type identification;
- availability of altitude, speed, vertical rate, time, and mass-related variables;
- whether interval-level fuel targets can be reconstructed;
- whether Fuel-Flow and Direct targets can be compared fairly;
- whether Energy features can be reproduced consistently;
- number of aircraft types and flights available;
- whether the domain differs substantially from the commercial operations represented in PRC2025;
- licensing and reproducibility constraints.

#### Dataset selection rule

Use DASHlink **only if** it supports a scientifically comparable evaluation.

If DASHlink does not provide compatible fuel labels or sufficient aircraft diversity, search for a more suitable public aviation dataset.

**Do not** force AeroTwin's full pipeline onto an incompatible dataset merely to claim multi-dataset validation.

#### Minimum external validation experiment

The second-dataset experiment does **not** need to reproduce every AeroTwin notebook. Run a **compact validation suite**:

| Experiment | Description |
|---|---|
| **A. Direct baseline** | Predict absolute fuel consumption directly |
| **B. Fuel-Flow target** | Predict fuel consumption rate; recover interval fuel via duration |
| **C. Energy feature ablation** | Compare base trajectory features vs base + Energy features |
| **D. Direct vs Flow comparison** | MAE, RMSE, R², bootstrap confidence intervals |
| **E. Generalization test** | Held-out flights, aircraft, aircraft types, routes, temporal split, operator/domain split — whichever the dataset structure supports |

The exact split should reflect the external dataset's real structure.

#### Cross-dataset comparison

Create a final table such as:

| Finding | PRC2025 | External Dataset | Replicated? |
|---|---:|---:|---|
| Energy features improve prediction | Yes | TBD | TBD |
| Flow target improves standard evaluation | Yes / setting-dependent | TBD | TBD |
| Flow target helps all aircraft types | No | TBD | TBD |
| Cross-type shift causes major degradation | Yes | TBD | TBD |
| Raw physics baseline is insufficient | Yes | TBD | TBD |

The purpose is **not** to obtain identical RMSE values across datasets — label scales, aircraft composition, and telemetry differ.

The purpose is to test whether the **qualitative scientific conclusions replicate**.

#### Interpretation policy

- **If the external dataset confirms the same qualitative findings:** state that AeroTwin's main conclusions show evidence of cross-dataset robustness.
- **If only some findings replicate:** report partial replication and explain which inductive biases appear dataset-dependent.
- **If findings fail to replicate:** treat this as scientifically important evidence that the original conclusions were dataset-specific. **Do not hide negative external-validation results.**

#### External validation milestones

| Milestone | Status |
|---|---|
| External dataset compatibility audit | ⬜ Next priority |
| Second dataset preprocessing pipeline | ⬜ Pending |
| Cross-dataset feature alignment | ⬜ Pending |
| External Direct vs Flow evaluation | ✅ Done (`physics/external_vs_flow_eval.py`) |
| External Energy-feature ablation | ⬜ Pending |
| External generalization test | ✅ Done (`tests/test_external_generalization.py`) |
| Cross-dataset replication analysis | ✅ Done (`physics/cross_dataset_replication.py`) |

**Completed external-validation components.** The equivalent AeroTwin Flow-vs-Direct protocol (`physics/external_vs_flow_eval.py`) is runnable on any independent featured-dataset parquet and is covered by `tests/test_external_generalization.py` (15 tests: target transforms, `clean_for_eval` filtering, results-table shaping, internal-baseline normalization, and a `catboost`-gated end-to-end run). The cross-dataset replication analysis (`physics/cross_dataset_replication.py`) runs that protocol across several datasets, decides per dataset whether Flow+Energy still beats Direct at the bootstrap threshold, and aggregates a meta-verdict ("all / partial / failed to replicate"). It is covered by `tests/test_cross_dataset_replication.py` (11 tests; 26 project tests total, all passing). These supply the qualitative-finding comparison table (§20, step 7) once a second dataset is available; they do not yet require one.

---

### Priority 3 — Final Statistical Protocol

Before submission:

- Freeze primary hypotheses; separate confirmatory from exploratory
- Define primary metric per experiment
- Define resampling unit correctly (flight vs type vs hierarchical)
- Report CIs, not only point estimates
- Correct for multiple comparisons where many hypotheses are tested
- Document seeds and split definitions
- Ensure LOTO folds reproducible
- Maintain single master results table

**Do not** retroactively describe exploratory discoveries as preregistered hypotheses.

---

### Priority 4 — Paper Ablation Consolidation

Group findings for the paper:

| Category | Items |
|---|---|
| **Supported** | Energy-state; Energy+Weather; fuel-flow (Level 1); stacking |
| **Rejected** | OpenAP alone; sparse physics; operational; weather-only; residual/MLP; mass; embeddings; body-class universal routing |
| **Suggestive** | Flow LOTO macro gain; physical distance (B77W-sensitive); conditional target selection pending operational analysis |

Move secondary negative experiments and sensitivity tables to appendix.

---

### Priority 5 — Paper Figures and Tables

**Recommended main figures:**

1. AeroTwin system and experimental design (three generalization levels)
2. Standard-split ablation performance
3. Energy / Weather / FuelFlow inductive-bias comparison
4. Standard split vs LOTO generalization gap
5. Per-type Flow vs Direct paired ΔMAE
6. Bootstrap uncertainty for Flow vs Direct LOTO comparison
7. Operational shift distance vs ΔMAE — **only if Priority 1 yields meaningful, robust results**

**Main tables:** dataset statistics; standard-split leaderboard; ablation summary; LOTO macro results; paired LOTO significance; per-type transfer analysis.

---

### Priority 6 — Transformer Work

**Optional / low priority** unless a specific sequence-hypothesis under type shift is formulated (§18.4).

---

### Priority 7 — Shift-Aware Routing (conditional)

**Do not** build a complicated router immediately.

After Priority 1–2, if operational shift explains heterogeneity, test simple routing:

- Always Direct / Always Flow
- **Oracle selector** (upper bound only — not deployable)
- Learned selector using train/validation only (nested evaluation)

Compare against baselines with clear labeling of oracle as upper bound.

---

### Recommended execution order

1. **Operational distribution-shift analysis** on current PRC data (Priority 1).
2. **Dataset compatibility audit** for NASA DASHlink and alternative datasets (External Dataset Validation).
3. **Select second dataset** based on audit — use DASHlink only if scientifically comparable.
4. **Build minimal aligned preprocessing pipeline** for the chosen external dataset.
5. **Reproduce Direct vs Flow and Energy ablation** experiments (compact validation suite A–D).
6. **Run the strongest feasible out-of-distribution evaluation** on the external dataset (experiment E).
7. **Compare qualitative findings** across datasets (cross-dataset replication table).
8. **Only then finalize** broad generalization claims in the paper.

---

## Final Executive Summary

### Established

- Energy-state and Energy+Weather features yield bootstrap-supported gains on **unseen flights** (Level 1).
- Direct hybrid ensembles approach PRC winner RMSE (**202.90 vs 200.83 kg**) on the **same dataset** — benchmarking, not external validation.
- Fuel-flow targets improve MAE under standard-split ablations.
- Multiple inductive biases are **rejected** with preserved negative results (residual learning, sparsity hypothesis, mass, embeddings, weather-only, operational descriptors).

### Suggestive

- Fuel-Flow + Energy lowers LOTO macro MAE by ~17.4 kg vs Direct E+W, but **7/12 wins**, bootstrap CIs **cross zero**, paired tests **non-significant**, and **B77W-dominated** magnitude (ΔMAE ~−4 kg without B77W).
- Physical aircraft distance correlates with LOTO error only with B77W included.

### Failed / rejected (cross-type)

- Universal body-class routing; MoE as ensemble upgrade; Mahalanobis physical distance as primary predictor; claiming Flow "confirms" better unseen-aircraft transfer.

### Must happen next

Two major remaining research tasks:

1. **Explain cross-aircraft transfer heterogeneity** — operational distribution-shift analysis and fold-level explanatory table on PRC data (§20 Priorities 1–2).
2. **External dataset validation** — compatibility audit, compact Direct/Flow/Energy replication suite, and cross-dataset qualitative comparison (§20 External Dataset Validation).

Additional before submission: statistical protocol freeze and figure consolidation (§20 Priorities 3–5).

**Paper readiness:** The project currently has **strong internal validation** under held-out flights and held-out aircraft types, but **external dataset validation remains necessary** before making broad claims that the identified inductive biases generalize across aviation datasets. Cross-type transfer mechanisms are unresolved and LOTO inference is not yet confirmatory. Submission should wait until operational-shift analysis, external validation (or explicit single-dataset scope limitation), and statistical protocol freeze are complete.

---

*Report updated July 2026.*

**Reproduce:**

```bash
PYTHONPATH=. python notebooks/05_baseline_modeling.py
PYTHONPATH=. python notebooks/06_physics_ablation.py
PYTHONPATH=. python notebooks/07_significance_testing.py
PYTHONPATH=. python notebooks/08_physics_features_v2.py
PYTHONPATH=. python notebooks/09_physics_features_v3.py
PYTHONPATH=. python notebooks/09_aircraft_experts.py
PYTHONPATH=. python notebooks/09_mass_features.py
PYTHONPATH=. python notebooks/10_fuel_flow_target.py
PYTHONPATH=. python notebooks/11_stacking.py
PYTHONPATH=. python notebooks/12_verify_ensemble.py
PYTHONPATH=. python notebooks/13_flow_vs_prc.py
PYTHONPATH=. python notebooks/14_shap_explainability.py
PYTHONPATH=. python notebooks/15_leave_one_type_out.py
PYTHONPATH=. python notebooks/17_loto_significance_and_transfer_distance.py
```
