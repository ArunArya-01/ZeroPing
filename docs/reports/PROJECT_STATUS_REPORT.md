# AeroTwin Project Status Report

**Date:** July 2026 (last repo update: **2026-07-30**)  
**Repository:** ZeroPing (AeroTwin)  
**Dataset:** [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) (EUROCONTROL PRC 2025)  
**Current phase:** **Knowledge distillation** — MLP research complete; **official held-out MLP baseline frozen**; next = transformer students

---

## Executive Summary

### Current phase — Knowledge distillation (July 2026)

The official R3 teacher is **frozen**. Distillation Steps 1–5 are complete for the MLP track. The permanent MLP held-out baseline is established for all future student architectures.

| Milestone | Status | Key result |
|-----------|:------:|------------|
| **Step 1** Teacher distillation dataset | ✅ done | 119,032 samples · 60 features · `distillation_dataset.parquet` |
| **Step 2** Baseline MLP (A/B/C) | ✅ done | Teacher-only student beats GT-only; ~1.13M params |
| **Step 3** α/β KD weight sweep | ✅ done | **Best: α=0.1, β=0.9 (KD-1)** |
| **Step 4** Capacity scaling + latency + multi-seed | ✅ done | Best **val** student XLarge ~6.75M (228.1 kg); Large within 2 kg |
| **Step 5** Official held-out Final evaluation | ✅ done | **Official MLP baseline = Large** · Final test RMSE **215.85 kg** · XLarge 218.59 |
| **Combined (Rank+Final)** student eval | ✅ done | Large Combined **225.95** · XLarge **229.10** · teacher **221.33** (Large still best) |
| Step 6+ Transformer students | ⬜ next | α=0.1 / β=0.9; ~3M; beat **Final 215.85** and **Combined 225.95** |

**Recommended supervision for all future students:**  
`L = 0.1 · MSE(gt) + 0.9 · MSE(teacher)` (KD-1).

**Two evaluation protocols (both retained):**

| Protocol | Definition | Use |
|----------|------------|-----|
| **A — Final** | Final holdout only | Architecture research / controlled comparisons |
| **B — Combined** | RMSE(concat Rank, Final) | Official PRC-style parity vs teacher **221.33** |

**Official MLP baselines (do not retrain for comparison):**

| Model | Params | Rank RMSE | Final RMSE | **Combined RMSE** | CPU ms | vs teacher Combined |
|-------|-------:|----------:|-----------:|------------------:|-------:|--------------------:|
| **Large (official)** | **2.89M** | **240.66** | **215.85** | **225.95** | **0.26** | +4.62 kg |
| XLarge | 6.75M | 244.40 | 218.59 | 229.10 | 0.52 | +7.77 kg |
| R3 Teacher | ensemble | 232.53 | 213.62 | **221.33** | ~52 | reference |

- Val ranking (Step 4) favored XLarge; **Final ranking reverses to Large**.
- Generalization: both models **better** on Final than internal val (Large −6.0%; XLarge −4.2%) — **no overfitting**.
- Hard remaining errors: **B77W / B744 / ultra-long / cruise**.

Artifacts: `docs/reports/test_evaluation.md`, `docs/reports/capacity_scaling_report.md`, `results/distillation/test_evaluation/`.

### Official PRC2025 benchmark (canonical — completed July 2026; teacher frozen)

Under the **released official** Rank + Final protocol (train-only fits; no Rank/Final leakage or post-hoc tuning):

| Split | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| **Rank** | 90.9 kg | **239.2 kg** | 0.904 |
| **Final** | 87.3 kg | **220.9 kg** | 0.918 |
| **Combined** | 88.7 kg | **228.3 kg** | 0.913 |

- **Canonical frozen model:** Ensemble (XGB/LGBM/CatBoost × Direct + Fuel-Flow, Energy+Weather) + Ridge meta (chosen on train OOF only). Combined **228.25 kg**.
- **Published winner (combined RMSE):** ≈ **201 kg**. Canonical Δ ≈ **+27 kg** (AeroTwin worse). Combined 95% CI **[207, 249] kg** — does **not** sit entirely below 201 → **no superiority claim**.
- **Gap-closing campaign (v1 → R3, July 2026):** progressive train-only improvements on the official protocol:

| Version | Variant | Combined RMSE | Δ vs 228.25 |
|---------|---------|--------------:|------------:|
| v1.0 | Official ensemble (frozen V4) | **228.25** | reference |
| v1.1 | P1E phase affine + P2 Cat heavy specialist | **227.44** | −0.81 |
| R1 | P1E + OpenAP descriptors in heavy specialist | **226.19** | −2.06 |
| R2 | Fixed B744/B77L/A306 descriptors + R2 features | **225.25** | −3.00 |
| **R3 (current best)** | **P1E + dynamic mass model (21 features)** | **221.33** | **−6.92** |

- **Current best Combined RMSE: 221.33 kg** (Rank **232.53** · Final **213.73**; bias ≈ **+3.9 kg**). Remaining gap to winner ≈ **20 kg**.
- **Teacher Final held-out (distillation parity, audited):** **213.62 kg** on `featured_dataset_final.parquet` via `cache/r3_teacher_distillation_bundle.pkl` — **not** the Combined 221.33 figure. Full audit: `docs/reports/teacher_evaluation_report.md`.
- **Largest single step:** R3 dynamic mass (`physics/mass_model.py`) replaces crude `MTOW × 0.75` with 21 physics-informed mass features.
- **Reports / artifacts:** `official_prc_benchmark_report.md`, `official_gap_closing_report.md`, `CURRENT_MODEL_SUMMARY.md`, `figures/r3_ensemble_summary.json`, `README.md` Results.

**Internal Level-1 holdout RMSE (~196–203 kg) is not the official score.** Rank/Final are harder. Use **canonical Combined 228.25 kg** for frozen baseline comparisons; use **221.33 kg** for current best after gap-closing.

### Established findings (standard flight-level split; unseen flights, known aircraft families)

- **Energy-state representations** (E2) and **Energy + Weather** (E6) yield bootstrap-supported MAE gains over the OpenAP hybrid baseline on strict flight-level holdout (ΔMAE ≈ −1.8 to −2.6 kg; 95% CIs exclude zero).
- **Direct hybrid tree ensembles** generalize to unseen flights at MAE ≈ 84–88 kg and RMSE ≈ 210–224 kg.
- **Ensemble stacking (Direct kg)** reaches **RMSE = 202.90 kg** on the PRC Level-1 flight holdout — within ~1% of the published winner (200.83 kg) on the **same-dataset Direct track only**. This is **not** the official Rank/Final score.
- **Fuel-Flow single models** on the **same** outer flight split reach lower error under a different target: best MAE **79.52 kg** (XGB Flow+Energy) and best single-model RMSE **196.24 kg** (LGBM Flow+Energy). These must **not** be ranked against the Direct stack as one leaderboard (see `figures/LEADERBOARD_AUDIT.md`). On **official** data, Fuel-Flow also beats Direct (best single combined RMSE: LGBM Flow ~230).
- **Rejected under standard-split inference:** raw OpenAP alone; sparse-physics hypothesis; operational descriptors; weather-only; residual trees; MLP residual; early **heuristic** mass features (distinct from kept R3 dynamic mass); vertical-rate embeddings; simple body-class routing as a universal LOTO solution.

### Suggestive / unresolved findings (aircraft-family shift; LOTO)

- **LOTO reveals a large transfer gap:** macro-average MAE rises from ~88 kg (standard split) to ~283 kg (global direct) — standard flight-level metrics substantially overestimate robustness under **unseen-aircraft-family** shift.
- **Fuel-Flow + Energy** achieves a **lower LOTO macro-average MAE** than Direct E+W by approximately **17.4 kg**, but the improvement is **heterogeneous** (7 wins, 5 losses across 12 types), **not statistically robust** under paired type-level or hierarchical bootstrap inference (both 95% CIs cross zero), and **strongly influenced by the B77W fold** (excluding B77W shrinks ΔMAE to ~−4.0 kg). Interpret as **suggestive, not confirmed**.
- **Physical aircraft-specification distance** correlates with LOTO error in the full 12-type sample (Pearson *r* ≈ 0.76 for NN distance vs direct MAE) but **collapses after removing B77W** (*r* ≈ 0.15). Physical similarity alone is insufficient to explain transfer failure.
- **No single inductive bias dominates** across unseen aircraft families. The unresolved question is: *under what forms of distribution shift does fuel-flow normalization improve cross-aircraft transfer, and when does it fail?*

### Failed / rejected hypotheses (preserved)

- Residual learning (trees and MLP); sparsity-conditioned physics gains; MoE/experts as ensemble improvement; body-class hierarchical routing at LOTO macro level; Mahalanobis physical distance as primary transfer predictor.
- **Heuristic Level-1 mass features** (early V4 ablation): rejected under standard flight holdout (bootstrap CI includes zero).
- **Official gap-closing rejections:** global/class/haul affine; isotonic; cruise residual; ensemble reweight. **Keeps:** phase-conditional affine (P1E), heavy FuelFlow specialist (P2), OpenAP heavy descriptors (R1), descriptor fixes (R2), **dynamic mass features (R3)**.

### External validation progress (July 2026)

- **Infrastructure complete:** `physics/external_audit/` (DASHlink MAT loader, OpenSky loader, featured builder, pilot suite A–E), plus `HOW_TO_RUN_AUDIT.md` and unit tests (`tests/test_external_audit.py`).
- **DASHlink Project 85 pilot complete** (tails 686/687, 15 airborne flights, 137 intervals, integrated fuel-flow labels):
  - Energy features **replicate** (Direct Base+Energy vs Base: ΔMAE ≈ −4.85 kg; 95% CI [−6.87, −2.88]).
  - Fuel-Flow target **replicates** (Flow+Energy vs matched Direct: ΔMAE ≈ −2.64 kg; 95% CI [−4.63, −0.75]).
  - ML ≫ raw OpenAP **replicates** (physics MAE ~140 kg vs Direct ~21–26 kg on this pilot scale).
- **Caveats:** pilot is small (4 test flights); labels are reconstructed from `FF_*` (not ACARS FOB); absolute MAE is **not** comparable to PRC Level-1 ~84 kg; aircraft-type diversity for LOTO-style external analysis remains limited; default OpenAP type (`CRJ9`) may not match the FDR fleet.

### Active workstreams (see checklists)

1. **§23 Knowledge distillation (PRIMARY)** — Steps **1–5 complete** (MLP track + official Final held-out). Next: FT/Tab transformer students vs frozen Large baseline (**Final RMSE 215.85**), **α=0.1, β=0.9**.
2. **§21 Official RMSE improvement checklist** — **R1–R3 done** (Combined **228.25 → 221.33**, −6.92 kg). Teacher frozen; further Tier-1 items optional / deferred.
3. **§22 Final project completion checklist** — freeze **221.33** teacher + **215.85 Large Final** student baseline, package repo, write thesis/paper including distillation + held-out eval.

**Paper / major-project submission:** Official evaluation + R1–R3 + distillation MLP track (Steps 1–5) complete. Write-up should cover hybrid teacher, KD weights, capacity, and **official Final student baseline**.

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

**Evidence scope:** Primary ablations and significance tests use the EUROCONTROL PRC dataset. A **pilot-scale** second-dataset run on NASA DASHlink Project 85 is complete (§20 External Dataset Validation); scaled multi-type external validation remains open.

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
| Ensemble / stacking / PRC benchmarking (Level 1) | ✅ Complete |
| **Official Rank+Final featured datasets** | ✅ Complete (`featured_dataset_rank/final.parquet`) |
| **Official frozen ensemble evaluation** | ✅ Complete — Combined RMSE **228.25 kg** (`official_prc_benchmark_report.md`) |
| **Official error analysis (SSE drivers)** | ✅ Complete (`notebooks/18_official_error_analysis.py`) |
| **Gap-closing campaign v1** | ✅ Complete — Combined **227.44 kg** (−0.81 kg) |
| **R1 heavy OpenAP descriptors** | ✅ Complete — Combined **226.19 kg** (`notebooks/21_rmse_r1_heavy_features.py`) |
| **R2 OpenAP descriptor fixes** | ✅ Complete — Combined **225.25 kg** (`notebooks/24_r2_heavy_features.py`) |
| **R3 dynamic mass model** | ✅ Complete — Combined **221.33 kg** (`physics/mass_model.py`, `notebooks/25–26_r3_*.py`) |
| **§21 Official RMSE improvement checklist** | 🔄 **Active** — R1–R3 done; residual gap ≈ **20 kg** |
| SHAP explainability | ✅ Complete |
| Aircraft-level analysis | ✅ Complete (exploratory) |
| MoE / aircraft experts | ✅ Complete (exploratory; no ensemble gain) |
| **LOTO core evaluation** | ✅ Complete |
| **LOTO paired robustness analysis** | ✅ Complete |
| **LOTO bootstrap inference** | ✅ Complete |
| **LOTO leave-one-type sensitivity** | ✅ Complete |
| **Physical transfer-distance analysis** | ✅ Complete (**exploratory / influence-sensitive**) |
| **Operational distribution-shift analysis** | ⬜ Recommended for paper (not required for official RMSE) |
| **External dataset compatibility audit** | ✅ Done (DASHlink Project 85 + audit package; OpenSky path documented) |
| **Second dataset preprocessing pipeline** | ✅ Done (`physics/external_audit/`) |
| **DASHlink MAT loader (struct `.data`)** | ✅ Done (`dashlink_loader.py`; 186 channels/flight) |
| **DASHlink pilot (15 flights)** | ✅ Done — energy + Flow qualitative **replicate** (`audit_results/dashlink_pilot/`) |
| **OpenSky pilot (live Trino)** | ⬜ Pending credentials / query window (code + synthetic fallback ready) |
| **Scaled external validation (≥50–100 flights)** | ⬜ Pending (or scope as pilot-only in write-up) |
| **Cross-dataset feature alignment** | ✅ Complete |
| **External Direct vs Flow evaluation** | ✅ Done (`external_vs_flow_eval.py` + `run_audit_pilot.py`) |
| **External Energy-feature ablation** | ✅ Done (`external_energy_ablation.py` + pilot Exp C) |
| **External generalization test** | ✅ Done (`tests/test_external_generalization.py`, `test_external_audit.py`) |
| **Cross-dataset replication analysis** | ✅ Done (`physics/cross_dataset_replication.py`) |
| **Shift-aware routing** | ✅ Implemented (scaffold; gated, uncalibrated by default — `physics/shift_aware_routing.py`) |
| **Transformer residual** | ✅ Implemented (module scaffold; untested) |
| **Statistical protocol freeze** | ✅ Done (`physics/statistical_protocol.py`, `papers/statistical_protocol.md`) |
| **§22 Final project completion checklist** | ⬜ **Pending** |
| **Final figure/table consolidation** | ⬜ Pending |
| **Final paper / thesis drafting** | ⬜ Pending |
| Optuna / CatBoost tuning (train-OOF nested) | ⬜ Optional Tier-2 RMSE item |

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
| `physics/external_audit/` | Second-dataset loaders + pilot orchestration |
| `HOW_TO_RUN_AUDIT.md` | Step-by-step external audit guide |
| `AeroTwin_External_Dataset_Audit_Package.md` | Compatibility audit design + decision gates |
| `audit_results/dashlink_pilot/` | Real DASHlink pilot metrics, significance, figures |
| `audit_results/dashlink_pilot/featured_dataset_audit.parquet` | Featured intervals from 15 Project 85 flights |
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
| `official_prc_benchmark_report.md` | Canonical official Rank/Final write-up (228.25) |
| `official_gap_closing_report.md` | Gap-closing campaign v1 (−0.81 kg Combined) |
| `CURRENT_MODEL_SUMMARY.md` | Live RMSE audit / model architecture notes |
| `physics/mass_model.py` | R3 dynamic mass feature builder (21 features) |
| `figures/r1_summary.json` / `table_rmse_R1*.csv` | R1 heavy OpenAP descriptors (226.19) |
| `figures/r2_summary.json` / `table_rmse_R2*.csv` | R2 descriptor fixes (225.25) |
| `figures/r3_summary.json` / `r3_ensemble_summary.json` | R3 dynamic mass + ensemble (221.33 best) |
| `figures/table_rmse_R3_*.csv` | R3 full / ensemble leaderboards |
| `figures/fig_r3_dynamic_mass.png` | R3 mass evaluation figure |
| `figures/table_official_leaderboard.csv` | Official per-model Rank/Final/Combined |
| `figures/table_prc_comparison.csv` | Winner vs AeroTwin + CIs |
| `figures/fig_official_leaderboard.png` | Official Rank vs Final bars |
| `figures/fig_prc_vs_aerotwin.png` | Winner comparison figure |
| `figures/official_error_analysis_summary.json` | SSE drivers (type/phase/haul) |
| `figures/table_gap_closing_leaderboard.csv` | Gap-close variants + gates |

### Strongest results (by generalization level) — protocol-separated

| Level / track | Result | Detail |
|---|---|---|
| **Official Combined RMSE (canonical)** | **228.25 kg** | Frozen V4 ensemble Rank+Final; vs winner ≈ 201 (Δ +27) |
| **Official best after gap-close R3** | **221.33 kg** | P1E + 21 dynamic mass features (−6.92 vs canonical) |
| **Official Rank / Final (R3 best)** | **232.53 / 213.73 kg** | From `figures/r3_ensemble_summary.json` |
| **Prior milestones** | 227.44 → 226.19 → 225.25 | v1.1 → R1 → R2 |
| **Level 1 MAE (Fuel-Flow single)** | **79.52 kg** | XGB Flow+Energy — internal flight holdout only |
| **Level 1 RMSE (Fuel-Flow single)** | **196.24 kg** | LGBM Flow+Energy — **not** official Rank/Final |
| **Level 1 MAE (Direct single)** | **83.76 kg** | XGB Energy+Weather Hybrid |
| **Level 1 RMSE (Direct stack)** | **202.90 kg** | 5f OOF LGBM_meta — internal Direct track |
| **Level 2 LOTO macro MAE (best point estimate)** | **265.9 kg** | Global Flow+Energy (vs 283.2 kg direct) — **suggestive, not confirmed** |

Full audit: `figures/LEADERBOARD_AUDIT.md`. Official write-up: `official_prc_benchmark_report.md`.

---

## 14. Competition Benchmarking

### 14.1 Canonical official protocol (Rank + Final) — **use this for competition claims**

| Model | Combined RMSE (kg) | Rank | Final |
|---|---:|---:|---:|
| **Published PRC winner** | ≈ **201** | — | — |
| **AeroTwin R3 best (P1E + dynamic mass)** | **221.33** | 232.53 | 213.73 |
| AeroTwin R2 (descriptor fixes) | **225.25** | 234.81 | 218.82 |
| AeroTwin R1 (OpenAP heavy descriptors) | **226.19** | 235.21 | 220.13 |
| AeroTwin gap-close v1.1 (P1E + heavy Cat) | **227.44** | 235.30 | 222.18 |
| **AeroTwin frozen V4 ensemble (canonical)** | **228.25** | 239.18 | 220.86 |
| Best single (LGBM FuelFlow E+W) | 230.18 | 249.83 | 216.46 |
| OpenAP physics only | 1268.37 | 1191.95 | 1315.65 |

**Δ (canonical ensemble − winner):** **+27.25 kg**. Combined 95% CI **[207.1, 249.4] kg** — no superiority claim.  
**Δ (R3 best − winner):** ≈ **+20 kg**. Still no superiority claim without a CI entirely below 201.  
**Artifacts:** `figures/table_official_leaderboard.csv`, `table_prc_comparison.csv`, `official_prc_benchmark_report.md`, `figures/r3_ensemble_summary.json`.

### 14.2 Internal Level-1 holdout (not official Rank/Final)

Legacy internal Direct stack RMSE **202.90** vs winner cite **200.83** is **train-split benchmarking only**. Do **not** present it as the official competition score.

| Model | Track | RMSE (kg) |
|---|---|---|
| PRC2025 Winner (paper combined) | competition ref | ≈ 201 |
| AeroTwin Ensemble (5f LGBM_meta) | **Internal Direct** stack | **202.90** |
| LGBM Flow+Energy (single) | **Fuel-Flow** single | **196.24** (*not Direct stack*) |
| Optuna CatBoost | Direct single | 204.6 |
| Energy+Weather Direct XGB | Direct single | ~212 |

**Accurate conclusion:** Canonical official Rank/Final Combined RMSE is **228.25 kg** (~27 kg behind the published winner). After train-only gap-closing through **R3**, best Combined is **221.33 kg** (~20 kg behind winner). Internal holdout metrics remain useful for ablations but overstate competition performance relative to the temporal Rank/Final protocol.

---

## 15. V4 Experiments

V4 focused on fuel-flow targets, mass features, and vertical embeddings while preserving the scientific focus on **which physical inductive biases matter**.

### Fuel Flow Target — **Established under Level 1; suggestive under Level 2**

- Predict `fuel_flow_kgps = actual_fuel_kg / duration_s`; recover `fuel_kg = flow × duration_s`.
- **Standard split:** statistically significant MAE improvement over direct Energy+Weather in targeted ablations; best single-model MAE ≈ 79.5 kg equivalents; ensemble RMSE competitive (206–204 kg).
- **LOTO:** favourable macro point estimate (−17.4 kg vs direct) but **not statistically confirmed** (§18.3.1). Heterogeneous per-type results.

### Mass Features — **Level-1 heuristic rejected; official R3 dynamic mass kept**

- **V4 heuristic mass (Level 1):** no significant gain; bootstrap CIs include zero — treated as rejected under the early ablation protocol.
- **R3 dynamic mass (official Rank/Final, July 2026):** **kept**. Twenty-one physics-informed mass features (`physics/mass_model.py`) reduce Combined RMSE to **221.33 kg** (−6.92 vs canonical 228.25). Includes estimated takeoff/landing mass, per-interval mass by flight fraction, mass-scaled energy, wing loading, and phase-aware mass. Bias drops from ~+24 kg to ~**+3.9 kg**.
- **Do not conflate** the rejected Level-1 heuristic with the successful official R3 mass model.

### Vertical Embeddings — **Rejected**

No gain; bootstrap CI includes zero.

**Notebooks:** `notebooks/10_fuel_flow_target.py`, `notebooks/25_r3_dynamic_mass.py`, `notebooks/26_r3_ensemble_mass.py`  
**Artifacts:** `figures/table_fuel_flow.csv`, `figures/fig_fuel_vs_flow.png`, `figures/table_rmse_R3_*.csv`, `figures/r3_summary.json`

---

## 16. Ensemble Study

**Notebooks:** `notebooks/08_ensemble.py`, `notebooks/11_stacking.py`, `notebooks/12_verify_ensemble.py`

**Best on Direct stacking track:** LGBM_meta achieving **RMSE = 202.90 kg** (MAE 84.3).

Stacking improves **Direct** RMSE over single Energy+Weather Direct models (~212 → 202.9). Aircraft experts (~206.8 kg RMSE) **underperform** the global meta-ensemble.

**Do not claim this is the global best AeroTwin model:** on the Fuel-Flow track (same outer test flights), single-model LGBM Flow+Energy reaches RMSE **196.2** and XGB Flow+Energy reaches MAE **79.5**. Different targets → separate leaderboards (`figures/LEADERBOARD_AUDIT.md`).

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
- Early Level-1 **heuristic** mass features; vertical-rate embeddings
- Simple body-class routing as universal transfer solution

### Supported for paper (official gap-closing)

- Heavy specialist + OpenAP descriptors (R1) and descriptor completeness (R2)
- **Dynamic mass estimation (R3)** as largest Combined RMSE gain (−6.92 kg vs canonical)

### Suggestive / unresolved for paper

- Flow target improves LOTO macro MAE but lacks robust type-level significance
- Physical distance may relate to transfer error, but evidence is B77W-sensitive
- Conditional target selection may be useful, but must first be explained by operational shift analysis
- Remaining ~20 kg official gap (heavies / ultra-long / cruise SSE still dominate)

### Current Research Question

**Under what measurable forms of train–test distribution shift does fuel-flow normalization improve cross-aircraft fuel prediction, and when does direct kilogram regression remain preferable?**

No single inductive bias dominates across unseen aircraft families.

### Immediate Next Experiment

**Primary:** Distillation Step 6 — FT-Transformer / TabTransformer student on the **frozen** dataset with **α=0.1, β=0.9**, targeting ~3M params (Large-tier). Must beat **official Large Final RMSE 215.85 kg**. Reuse `runner.py`.

**Secondary:** §22 freeze + write-up (canonical **228.25**, teacher **221.33**, student Final baseline **215.85**, distillation Steps 1–5).

**Deferred:** further teacher RMSE chase (§21 Tier-1) unless reopening the frozen teacher is justified. Further MLP width beyond Large is **not** justified by Step 5 Final results.

**Scientific depth (thesis transfer chapter):** Operational distribution-shift analysis (Priority 1; §20): for each LOTO held-out type, measure operational train–test distance and correlate with Direct MAE, Flow MAE, and **ΔMAE (Flow − Direct)**. Scale DASHlink external pilot or scope it as pilot-only (§20 / F5).

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

### 18.4 Transformer Residual — ✅ Implemented (module scaffold; untested)

**Module:** `physics/transformer_residual.py` exposes `train_transformer_residual(feature_cols, X_train, X_test, y_residual_train, physics_test, ...)`, mirroring `physics/mlp_residual.py`'s interface. It maps each feature to a token and applies a small Transformer encoder (position embeddings, `SmoothL1Loss`, AdamW) to predict `residual_kg`, then returns `physics_fuel_kg + predicted_residual`. A sequence-based flight-level variant also exists at `notebooks/16_transformer_residual.py`.

Tabular residual learning (trees ~107 kg, MLP ~104 kg) already **failed** on standard split, and the ensemble is near competition RMSE on **Level 1**. The strongest unresolved issue remains **cross-aircraft transfer (Level 2)**, not standard-split model capacity.

**Implement/evaluate only if testing a specific hypothesis**, e.g.: *"Does preserving within-flight temporal structure improve transfer under aircraft-type shift?"* Do not pursue merely because transformers are fashionable. The module is available but its empirical contribution is **not yet established**.

**Status:** Implemented scaffold; low priority for evaluation unless a clear sequence-model hypothesis is formulated.

---

### 18.5 Mixture-of-Experts (MoE) — ✅ Exploratory

**Script:** `notebooks/09_aircraft_experts.py`

Global vs hard experts vs soft MoE on **standard split**. Experts improve single-model RMSE by ~2–3 kg; **underperform** LGBM_meta ensemble (206.8 vs 202.9 kg). Under LOTO, specialization without in-type training data does not solve transfer.

**Shift-aware routing is implemented as a guarded scaffold only** (`physics/shift_aware_routing.py`); the learned policy refuses to route until calibrated from operational-shift validation evidence (§20 Priority 7).

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
| Transformer residual | ✅ Implemented | `physics/transformer_residual.py` | Module scaffold; untested |
| MoE / experts | ✅ Exploratory | `09_aircraft_experts.py` | Marginal single-model; worse than ensemble |

---

## 19. SOTA Position (PRC Dataset Benchmark)

| System | RMSE (kg) | Protocol |
|---|---|---|
| OpenAP (official combined) | **1,268** | Official Rank+Final |
| OpenAP (Level 1 holdout) | 1,582 | Internal flight split |
| Official AeroTwin ensemble | **228.25** | **Canonical official Combined** |
| Official gap-close v1.1 | **227.44** | P1E + heavy Cat specialist |
| Official R1 / R2 | **226.19 / 225.25** | OpenAP heavy descriptors + fixes |
| **Official R3 best (current)** | **221.33** | **P1E + dynamic mass** |
| Best official single (LGBM Flow) | ~230 | Official Combined |
| PRC2025 Winner | ≈ **201** | Published combined score |
| Internal Direct ensemble | 202.90 | Level 1 holdout only |
| LOTO Global Direct | ~469 (macro) | **Level 2 types** |
| LOTO Global Flow+Energy | ~446 (macro) | **Level 2 types** |

**Do not conflate** canonical official Combined (**228.25**), gap-close best (**221.33**), internal holdout (~203 kg), and LOTO macro (~446–469 kg). They measure different protocols.

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

AeroTwin's primary empirical evidence remains the **EUROCONTROL PRC 2025** dataset. Strict flight-level evaluation, LOTO, bootstrap inference, and transfer analysis are still largely **within one data source**. External validation is therefore required for stronger cross-dataset claims — and is now **underway** with a completed DASHlink pilot.

#### Goal

Test whether AeroTwin's main scientific findings survive **dataset shift** rather than only train/test or aircraft-type shift within the PRC dataset.

The external validation should answer:

1. Do energy-state features improve prediction on another dataset?
2. Does Fuel-Flow target normalization outperform or compete with Direct fuel prediction?
3. Does the Direct-vs-Flow ranking remain heterogeneous under aircraft or operational shift?
4. Does physics-informed feature engineering remain useful when telemetry characteristics differ?
5. Does the large gap between standard evaluation and cross-aircraft evaluation appear in another dataset?

#### Candidate datasets and audit status

| Dataset | Role | Status |
|---|---|---|
| **NASA DASHlink Project 85** (FDR, tails ~652–687) | Primary external candidate; independent fuel via flow integration | **Go for pilot** — pipeline + 15-flight pilot complete |
| **OpenSky Trino historical** | Telemetry / sparsity shift; **no native fuel** (OpenAP labels) | Code ready; live query pending credentials |

**Compatibility audit outcomes (DASHlink Project 85):**

| Check | Result |
|---|---|
| Fuel signal | ✅ `FF_1…FF_4` (LBS/HR); also `FQTY_*` tank quantities (LBS) |
| Trajectory | ✅ `ALT` (FEET, 4 Hz), `GS` (KNOTS, 4 Hz), `IVV` (FT/MIN, 16 Hz), `CAS`/`MACH`/`TAS`, `LATP`/`LONP` |
| Interval targets | ✅ Fixed windows (e.g. 600 s); integrate fuel flow → `fuel_kg` |
| Energy / phase features | ✅ Reused OpenAP baseline + feature_engineering |
| Aircraft diversity | ⚠ Regional FDR fleet / limited type labels in pilot (default OpenAP type `CRJ9`) |
| Domain shift vs PRC | ✅ FDR-dense onboard samples vs fused ADS-B/ACARS commercial ops |
| Loader note | Each MAT parameter is a **struct** with `.data`, `.Rate`, `.Units` — `dashlink_loader.load_mat_file` extracts series (186 channels/flight) |

**Selection rule:** DASHlink is suitable for **qualitative** Direct / Flow / Energy replication with documented label limitations (integrated flow ≠ ACARS FOB). It is **not** yet sufficient alone for multi-type LOTO external claims without broader type metadata and more flights.

#### Minimum external validation experiment

Implemented in `physics/external_audit/run_audit_pilot.py`:

| Experiment | Description | DASHlink pilot outcome |
|---|---|---|
| **A. Direct baseline** | Predict absolute fuel with base + physics | MAE ≈ 25.5 kg |
| **B. Fuel-Flow target** | Predict rate; recover kg via duration | MAE ≈ **18.1 kg** (best) |
| **C. Energy feature ablation** | Base vs base + Energy | Base+Energy MAE ≈ 20.7; ΔMAE ≈ **−4.85** (sig.) |
| **D. Direct vs Flow comparison** | Bootstrap CIs on flight clusters | Flow better by ≈ **−2.64** kg MAE (sig.) |
| **E. Generalization** | Flight-level holdout | 15 flights → 4 test flights / 40 intervals |

Reproduce:

```bash
python -m physics.external_audit.run_audit_pilot \
  --source dashlink --dashlink-dir data --max-flights 15 \
  --out-dir audit_results/dashlink_pilot
```

#### Cross-dataset comparison (updated after DASHlink pilot)

| Finding | PRC2025 | DASHlink pilot (Project 85) | Replicated? |
|---|---|---|---|
| Energy features improve prediction | Yes (Level 1 bootstrap) | Yes (ΔMAE ≈ −4.85 kg; CI excludes 0) | **Yes (pilot)** |
| Flow target helps under flight-level eval | Yes / setting-dependent | Yes (Flow 18.1 vs Direct 20.7 kg) | **Yes (pilot)** |
| Flow target helps all aircraft types | No (LOTO heterogeneous) | Not tested (single-fleet pilot) | **TBD** |
| Cross-type shift causes major degradation | Yes (LOTO ~3× MAE) | Not tested | **TBD** |
| Raw physics baseline is insufficient | Yes | Yes (physics MAE ~140 vs ML ~18–26) | **Yes (pilot)** |

Absolute error magnitudes are **not** comparable across datasets (label construction, interval length, aircraft, telemetry density). Interpret **qualitative** agreement only.

**Caveats for the DASHlink pilot**

- Fuel labels = integrated multi-engine fuel flow (noisier / differently biased than PRC ACARS FOB deltas).
- Small *n* (15 flights, 4 test flights); bootstrap CIs are indicative, not definitive for publication alone.
- OpenAP type/mass defaults likely mismatch → large physics MAE expected.
- Full archive has thousands of MAT files; pilot used quality-filtered airborne segments from tails 686/687.

#### Interpretation policy

- **If the external dataset confirms the same qualitative findings:** state that AeroTwin's main conclusions show evidence of cross-dataset robustness.
- **If only some findings replicate:** report partial replication and explain which inductive biases appear dataset-dependent.
- **If findings fail to replicate:** treat this as scientifically important evidence that the original conclusions were dataset-specific. **Do not hide negative external-validation results.**

**Current reading:** pilot-scale evidence supports **partial/early cross-dataset robustness** of energy features and flow targets. Do **not** yet claim full external validation or multi-type transfer replication.

#### External validation milestones

| Milestone | Status |
|---|---|
| External dataset compatibility audit | ✅ Done (DASHlink + OpenSky design) |
| Second dataset preprocessing pipeline | ✅ Done (`physics/external_audit/`) |
| Cross-dataset feature alignment | ✅ Done (`cross_dataset_alignment.py`) |
| External Direct vs Flow evaluation | ✅ Done (protocol + DASHlink pilot) |
| External Energy-feature ablation | ✅ Done (protocol + DASHlink pilot Exp C) |
| External generalization test | ✅ Done (flight-level pilot + unit tests) |
| Cross-dataset replication analysis | ✅ Done (`cross_dataset_replication.py`) |
| DASHlink scaled run (≥50–100 flights) | ⬜ Pending |
| OpenSky live pilot | ⬜ Pending (credentials) |
| Multi-type / LOTO on external data | ⬜ Pending (needs type diversity) |

**Infrastructure.** Loaders correctly extract Project 85 parameter structs (`.data` / Rate / Units), resample multi-rate FDR channels, convert units to SI, reconstruct interval fuel from `FF_*`, and run the compact A–E suite into `audit_results/`. Protocol modules (`external_vs_flow_eval.py`, `cross_dataset_replication.py`) remain available for any featured parquet. Tests: `tests/test_external_audit.py` (12), `tests/test_external_generalization.py`, `tests/test_cross_dataset_replication.py`.

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
| **Supported** | Energy-state; Energy+Weather; fuel-flow (Level 1); stacking; heavy specialist (P2/R1/R2); **R3 dynamic mass** (official) |
| **Rejected** | OpenAP alone; sparse physics; operational; weather-only; residual/MLP; early heuristic mass (Level 1); embeddings; body-class universal routing |
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

1. ~~**Official Rank+Final evaluation**~~ ✅ Combined **228.25 kg**; reports frozen.
2. ~~**Gap-closing campaign v1**~~ ✅ best **227.44 kg**; large global calibrations rejected.
3. ~~**§21 R1–R3**~~ ✅ R1 226.19 → R2 225.25 → **R3 221.33** (dynamic mass).
4. **§22 completion package** — freeze **221.33 / 228.25**, sync docs, figures, thesis/paper.
5. **Optional remaining §21** (asymmetric loss, ultra-long / long-interval) *or* **F3 accept residual ~20 kg gap**.
6. **Operational distribution-shift analysis** (Priority 1) — if thesis needs transfer mechanism chapter.
7. ~~Dataset compatibility audit for NASA DASHlink~~ ✅
8. ~~DASHlink pilot~~ ✅ (`audit_results/dashlink_pilot/`)
9. **Scale DASHlink** or scope pilot-only (F5).
10. **Optional OpenSky pilot** for telemetry-shift / physics-label robustness.
11. Figure consolidation + paper drafting (§22 Track 3).

---

## 21. Official RMSE Improvement Checklist

> **Teammate handoff:** open tasks with owners, gates, and PR template live in  
> **[`RMSE_IMPROVEMENT_BACKLOG.md`](RMSE_IMPROVEMENT_BACKLOG.md)** — claim work there.

**Baseline (canonical frozen V4):** Combined RMSE **228.25 kg** · Rank **239.18** · Final **220.86**  
**Best so far (R3 dynamic mass + P1E):** Combined **221.33 kg** (−6.92 kg vs canonical) · Rank **232.53** · Final **213.73**  
**Prior milestones:** v1.1 **227.44** → R1 **226.19** → R2 **225.25** → R3 **221.33**  
**Target reference:** published winner Combined ≈ **201 kg** (Δ from R3 best ≈ **+20 kg**)  
**Primary error drivers:** A359 + B77W + B744 ≈ **72% SSE**; cruise ≈ **87%**; ultra-long-haul ≈ **85%**; R3 bias ≈ **+3.9 kg** (was ~+24–31 kg).

### Protocol rules (do not violate)

- [x] Train-only fitting; **never** tune on Rank/Final labels
- [x] Model selection on train OOF / nested GroupKFold only
- [x] Gate every change on **Combined RMSE** vs baseline; report Rank and Final separately
- [x] No superiority claim vs 201 unless Combined CI sits entirely below 201
- [x] Prefer hypothesis-linked changes (SSE drivers) over blind hyperparameter thrashing
- [x] Preserve leakage-free protocol (`figures/table_overlap_check.csv`, official report §3)

### Already done (do not re-run without new hypothesis)

| ID | Item | Outcome | Status |
|----|------|---------|:------:|
| G0 | Official Rank+Final full evaluation | Combined **228.25**; Δ vs winner **+27.25** | ✅ |
| G1 | Error analysis by type / phase / haul / duration | Heavies + cruise + ultra-long dominate SSE | ✅ |
| G2 | Global / class / haul affine; isotonic calibration | No useful Combined gain; bias is shift-dependent | ❌ reject |
| G3 | Phase-conditional affine (P1E) | Combined **228.16** (−0.10) | ✅ tiny keep |
| G4 | Heavy FuelFlow CatBoost specialist (P2) | Combined **227.44** (−0.81); B744 much better | ✅ keep |
| G5 | Cruise residual (P3); ensemble reweight (P5) | Hurt or no beat of P2 | ❌ reject |
| **R1** | Heavy-only OpenAP descriptors + interactions | Combined **226.19** (−2.06 vs 228.25) | ✅ keep |
| **R2** | OpenAP descriptor fixes (B744/B77L/A306) + R2 features | Combined **225.25** (−3.00 vs 228.25) | ✅ keep |
| **R3** | Dynamic mass model (21 features) + P1E ensemble path | Combined **221.33** (−6.92 vs 228.25) | ✅ keep |

> **Note on IDs:** Early backlog named R2 “asymmetric loss” and R3 “ultra-long specialist”; implementation used **R2 = descriptor completeness** and **R3 = dynamic mass** (largest gain). Original backlog R2/R3 ideas remain open as **R2b** / **R3b** below.

### Tier 1 — Remaining high-impact items

Target the remaining ~20 kg gap via **representation on hard subgroups**, not more global post-hoc maps. Gate vs **221.33** (current best).

| ID | Checklist item | Why | Expected gain | Gate | Status |
|----|----------------|-----|---------------|------|:------:|
| **R1** | Heavy-only OpenAP descriptors + interactions | Heavies ~72% SSE | −3 to −12 kg | Combined ↓ | ✅ **done** (226.19) |
| **R2 (impl.)** | OpenAP missing-descriptor fixes + heavy features | Incomplete aircraft physics | — | Combined ↓ | ✅ **done** (225.25) |
| **R2b** | Quantile / asymmetric / Huber-style loss on heavies | Cut remaining over-predict tail | −2 to −8 kg | Combined ↓ vs 221.33; bias ↓ | ⬜ |
| **R3 (impl.)** | Dynamic mass model (21 features) | Mass is first-order fuel driver | — | Combined ↓ | ✅ **done** (221.33) |
| **R3b** | Ultra-long-haul specialist or haul-conditional FuelFlow path | Ultra-long ≈ 85% SSE; haul affine failed | −2 to −10 kg | Combined ↓; ultra-long ↓ | ⬜ |
| **R4** | Further deploy-safe mass / load proxies | R3 already landed main mass path; only incremental | exploratory | Train-OOF then official | 🔄 largely superseded by R3 |
| **R5** | Long-interval model (iv 10–30 min and ≥ 30 min) | Long intervals ≈ 45% + 34% SSE | −2 to −8 kg | Combined ↓; long-iv ↓ | ⬜ |

### Tier 2 — Medium impact / architecture

| ID | Checklist item | Why | Expected gain | Status |
|----|----------------|-----|---------------|:------:|
| **R6** | **Fuel-Flow-first ensemble redesign** — drop weak Direct bases if train OOF justifies | Flow beats Direct on official; P5 was close | −1 to −5 kg | ⬜ |
| **R7** | **Nested Optuna / deeper trees on FuelFlow only** (train OOF; freeze before Rank/Final) | V4 frozen at 300 trees / lr 0.05 | −1 to −5 kg | ⬜ |
| **R8** | **Temporal / seasonal shift features** train-safe (month-of-year, ISA deviation trends) | Train Apr–Aug → Rank Sep → Final Oct | −1 to −6 kg | ⬜ |
| **R9** | **Promote R3 as documented official floor** (221.33 current best; 228.25 canonical) | Housekeeping so future Δ is honest | decision | 🔄 partial (README updated) |

### Tier 3 — Low priority / closed

| ID | Item | Status |
|----|------|:------:|
| **R10** | More global isotonic / affine without shift-robust design | ❌ stop |
| **R11** | Global cruise residual after stack | ❌ stop |
| **R12** | Transformer residual for official RMSE | ⬜ only if clear sequence hypothesis |
| **R13** | Body-class universal routing as official score fix | ❌ rejected for LOTO / not for RMSE chase |

### Success criteria (honest)

| Goal | Combined RMSE | Claim allowed |
|------|--------------:|---------------|
| Stretch / match winner | ≤ **201** | Match published winner **score** (not code-equivalence) |
| Strong competitive | ≤ **210** | Competitive; CI may still overlap 201 |
| Meaningful improvement | ≤ **220** (≈ −8+ kg from 228) | Report Δ + bootstrap CI |
| **Current best** | **221.33** | No superiority (gap ≈ 20 kg) |
| Canonical reference | **228.25** | Frozen V4 official |

**Stop rule:** If each remaining Tier-1 item fails the gate (Combined gain < ~1 kg or Rank/Final tradeoff is bad), **freeze the model**, accept residual gap with evidence from gap-closing + R1–R3 + error analysis, and complete the project via **§22**.

### Reproduce current baselines

```bash
python notebooks/17_official_prc_evaluation.py --skip-build
python notebooks/18_official_error_analysis.py
python notebooks/19_gap_closing_campaign.py
python notebooks/21_rmse_r1_heavy_features.py
python notebooks/24_r2_heavy_features.py
python notebooks/25_r3_dynamic_mass.py
python notebooks/26_r3_ensemble_mass.py
```

---

## 22. Final Project Completion Checklist

Project is a **major project / thesis-scale** deliverable. Complete **Track 1 + Track 2** minimum; Track 3 is the write-up package.

### Track 1 — Science freeze (minimum for “research done”)

| ID | Checklist item | Notes | Status |
|----|----------------|-------|:------:|
| **F1** | Freeze **canonical + best** official numbers in status report + README | Canonical **228.25**; R3 best **221.33**; Rank/Final; CI; Δ vs winner (~20 kg) | 🔄 README + this report updated; freeze tag still open |
| **F2** | Freeze statistical protocol (confirmatory vs exploratory) | `papers/statistical_protocol.md` + `physics/statistical_protocol.py` | ✅ mostly |
| **F3** | **Scope decision:** (A) run remaining Tier-1 RMSE items **or** (B) accept residual ~20 kg gap and write it up | R1–R3 already run; decide on R2b/R3b/R5 vs freeze | ⬜ decision |
| **F4** | Operational distribution-shift analysis (LOTO ΔMAE drivers) | Recommended if thesis emphasizes transfer; optional if scope is official RMSE only | ⬜ |
| **F5** | External validation: scale DASHlink **or** explicit pilot-only scope in write-up | Do not over-claim from 15 flights | ⬜ |
| **F6** | Consolidate **main** figures/tables for write-up | Official leaderboard, R1–R3 progression, SSE drivers, LOTO gap, ablations | ⬜ |
| **F7** | Limitations + negative-results appendix | Calibration fail, residual fail, residual ~20 kg gap, no superiority | ⬜ |
| **F8** | One-paragraph **canonical one-liner** locked for abstract | Update for R3 best 221.33; copy to thesis abstract | ⬜ |

### Track 2 — Repo & documentation package

| ID | Checklist item | Notes | Status |
|----|----------------|-------|:------:|
| **F9** | Update `README.md` Results with **official** Rank/Final/Combined + R3 | Demote internal 202.9 to “Level-1 holdout” | ✅ R3 table in README (2026-07-23) |
| **F10** | Keep `PROJECT_STATUS_REPORT.md` in sync (this section) | Reflect R1–R3 after 2026-07-23 commits | ✅ updated |
| **F11** | End-to-end reproduce path for official eval | Audit → build Rank/Final features → eval → R1–R3 notebooks | ⬜ polish |
| **F12** | Artifact index for official deliverables | `table_official_*`, `table_rmse_R{1,2,3}_*`, `r3_*.json`, figures | ⬜ |
| **F13** | CI green; no secrets in repo; cache policy documented | `.gitignore`, `HOW_TO_RUN_AUDIT.md` | ⬜ |
| **F14** | Tag release after freeze (e.g. `v1.1-r3-mass`) | Optional but recommended | ⬜ |

### Track 3 — Thesis / major-project submission

| ID | Checklist item | Notes | Status |
|----|----------------|-------|:------:|
| **F15** | Outline: intro → methods → Level 1 → official PRC → LOTO → external pilot → gap-closing → discussion | Use §17 narrative + §14.1 official chapter | ⬜ |
| **F16** | Methods chapter complete | Data, OpenAP, features, models, flight splits, official protocol, leakage checks | ⬜ |
| **F17** | Results chapter complete | Level 1 ablations, official Rank/Final, LOTO, DASHlink pilot, gap-closing | ⬜ |
| **F18** | Discussion: why ~27 kg gap remains | Heavy/ultra-long SSE; temporal shift; unknown winner recipe; honest CI | ⬜ |
| **F19** | Conclusion + future work | Map to remaining Tier-1 items or explicit deferrals | ⬜ |
| **F20** | References + citation (PRC JOAS paper, OpenAP, HF dataset) | doi:10.59490/joas.2026.8750 | ⬜ |
| **F21** | Final PDF + demo/script for viva if required | Reproduce one figure live if asked | ⬜ |

### Recommended execution order (finish the project)

```text
1. F3  Scope decision (chase remaining RMSE vs freeze ~20 kg gap)
2. Remaining §21  (only if F3 = chase: R2b / R3b / R5)  → gate each item
3. F1  Freeze numbers (canonical 228.25 + best 221.33)
4. F6–F7  Figures + limitations (include R1–R3 progression)
5. F9–F12 Repo/docs package (README already has R3)
6. F15–F21  Write-up + submission
```

Optional scientific depth (if thesis needs more than official benchmarking):

```text
F4 operational shift → F5 scaled external → update claims → then write-up
```

### Definition of “project complete”

All of the following:

1. **Canonical + best official metrics frozen** and consistent across report, README, and abstract (**228.25** and **221.33**).
2. Either **(a)** remaining Tier-1 items gated, **or** **(b)** residual ~20 kg gap **explicitly accepted** with gap-closing + R1–R3 + SSE evidence. *(R1–R3 already satisfy “≥1 Tier-1 run.”)*
3. **Write-up** covers methods, results (Level 1 + official + LOTO + external pilot + gap-close R1–R3), and honest limitations.
4. **Reproduce commands** and key artifacts match the frozen numbers.

---

## 23. Knowledge Distillation Stream (PRIMARY)

**Status:** Steps 1–5 **complete** (2026-07-30). Teacher frozen. **Official MLP held-out baseline frozen (Large Final RMSE 215.85 kg).** Next: transformer students with α=0.1 / β=0.9.

### Frozen teacher (do not retrain)

| Component | Detail |
|-----------|--------|
| Ensemble | XGB / LGBM / CatBoost × Direct + Fuel-Flow (6 bases) |
| Meta | Ridge (train-OOF selected) |
| Features | Energy+Weather base + **21 R3 dynamic mass** |
| Calibration | P1E phase-conditional affine |
| Official Combined RMSE | **221.33 kg** (Rank 232.53 · Final 213.73 · bias ~+3.7 kg) |

### Distillation dataset (frozen)

| Field | Value |
|-------|------:|
| Path | `distillation_dataset.parquet` |
| Samples | 119,032 (train intervals) |
| Features | 60 (ensemble input order) |
| Soft labels | OOF base preds + ridge + P1E teacher |
| Report | `docs/reports/distillation_dataset_report.md` |

### Baseline MLP (Step 2)

| Model | Loss | Val RMSE (kg) | Gap vs teacher |
|-------|------|-------------:|---------------:|
| A | GT only | 279.0 | +28.4 |
| B | Teacher only | **258.1** | **+7.5** |
| C | α=β=0.5 | 265.0 | +14.4 |

~1.13M params; flight-level 80/20; seed 42. Report: `docs/reports/mlp_student_report.md`.

### α/β sweep (Step 3) — key table

| Exp | α | β | Val RMSE | Student–Teacher RMSE |
|-----|--:|--:|---------:|---------------------:|
| **KD-1** | **0.1** | **0.9** | **188.31** | 110.80 |
| KD-2 | 0.2 | 0.8 | 189.48 | **110.20** |
| KD-0 | 0.0 | 1.0 | 189.49 | 112.11 |
| KD-4 | 0.5 | 0.5 | 196.78 | 125.15 |
| KD-7 | 1.0 | 0.0 | 221.32 | 162.99 |

- Teacher-heavy mean val RMSE **189.7** ≪ GT-heavy **217.5**.
- Pure teacher beats pure GT by **~32 kg** on val (denoising evidence).
- **Default for future students: α=0.1, β=0.9.**

Full report: `docs/reports/distillation_alpha_beta_sweep.md`  
Runner: `experiments/08_distillation/run_distillation_experiments.py`  
Package: `src/aerotwin/distillation/`

### Distillation directory map

```text
distillation_dataset.parquet          # frozen teacher soft labels
src/aerotwin/distillation/            # data, mlp, trainer, metrics, runner
experiments/08_distillation/
  01_build_teacher_distillation_dataset.py
  02_train_mlp_student.py
  03_alpha_beta_sweep.py
  run_distillation_experiments.py
models/distillation/                  # checkpoints
results/distillation/                 # metrics, preds, alpha_beta_sweep/
logs/distillation/
docs/reports/
  distillation_dataset_report.md
  mlp_student_report.md
  distillation_alpha_beta_sweep.md
```

### Capacity scaling (Step 4) — seed 42, α=0.1 / β=0.9

| Model | Params | Val RMSE | Gap vs teacher soft labels | Checkpoint MB | CPU batch speedup vs teacher |
|-------|-------:|---------:|---------------------------:|--------------:|-----------------------------:|
| Tiny | 0.24M | 270.55 | +26.4 | 0.92 | ~59× |
| Small | 0.50M | 241.73 | −2.4 | 1.93 | ~41× |
| Medium | 1.13M | 235.04 | −9.1 | 4.30 | ~28× |
| Large | 2.89M | 229.70 | −14.4 | 11.0 | ~16× |
| **XLarge** | **6.75M** | **228.14** | **−16.0** | 25.7 | ~7.5× |

Teacher soft-label val RMSE on this flight split: **244.14 kg**.  
Step improvements (kg): Tiny→Small **28.8**, Small→Medium **6.7**, Medium→Large **5.3**, Large→XLarge **1.6** (diminishing returns).

**Smallest within 2 kg of best:** Large (~2.9M).  
**Reproducibility (XLarge, 5 seeds):** mean **228.49 ± 0.92** kg; 95% CI **[227.7, 229.3]** entirely below teacher 244.1; all seeds better than teacher soft labels on this split.

**Inference (highlights):** teacher single-sample CPU ~**52 ms**; students ~**0.2–0.5 ms**. Teacher checkpoint ~**17 MB** (ensemble pickle); students **0.9–26 MB**.

Report: `docs/reports/capacity_scaling_report.md` · `results/distillation/capacity_scaling/`.

### Official held-out Final evaluation (Step 5) — permanent MLP baseline

**Script:** `experiments/08_distillation/05_test_evaluation.py` (eval-only; no training).  
**Data:** `featured_dataset_final.parquet` (37,170 rows · 2,824 flights from `fuel_final`).

| Model | Params | Val RMSE | **Final test RMSE** | Gap (test−val) | Final MAE | Bias | R² |
|-------|-------:|---------:|--------------------:|---------------:|----------:|-----:|---:|
| **Large (official)** | 2.89M | 229.70 | **215.85** | −13.85 (−6.0%) | 76.69 | +5.25 | 0.9220 |
| XLarge | 6.75M | 228.14 | 218.59 | −9.55 (−4.2%) | 77.36 | +6.41 | 0.9201 |
| R3 Teacher | ensemble | — | **213.62** | — | 74.14 | +4.87 | 0.9236 |
| OpenAP | — | — | 1,315.65 | — | 485.40 | +465.5 | −1.90 |

- Ranking **reversed** on Final (val: XLarge; test: **Large**). XLarge does **not** justify +3.9M params on held-out.
- Large is **+2.23 kg** vs teacher Final at ~**200×** lower single-sample CPU latency (0.26 ms vs ~52 ms).
- Dominant failures: **B77W, B744, B772, B789, A359**; ultralong ≥8 h; cruise bulk error.
- Artifacts: `results/distillation/test_evaluation/` · report: `docs/reports/test_evaluation.md`.

**Deployment / comparison baseline for all future students: Large_seed42 Final metrics above.**

### Distillation checklist

| ID | Item | Status |
|----|------|:------:|
| D1 | Export frozen teacher distillation dataset | ✅ |
| D2 | Baseline MLP (GT / teacher / KD 0.5) | ✅ |
| D3 | α/β weight sweep + recommended weights | ✅ |
| D4 | Architecture-agnostic experiment runner | ✅ |
| D5 | Capacity scaling + latency + multi-seed | ✅ |
| D5b | Official Final held-out eval (Large / XLarge) | ✅ |
| D6 | FT-Transformer / TabTransformer student | ⬜ next |
| D7 | Trajectory / sequence student (optional) | ⬜ |
| D8 | Distillation chapter in thesis/paper | ⬜ |

### Open research questions (distillation)

1. Do transformer students beat **Large Final RMSE 215.85** under α=0.1 / β=0.9?
2. Does the same α/β optimum hold under LOTO (unseen aircraft types)?
3. Can multi-teacher heads (per-base GBDT soft labels) help beyond final P1E targets?
4. Can architecture change close residual error on heavies / ultra-long without MLP width?

### Risks

| Risk | Mitigation |
|------|------------|
| Regenerating teacher / dataset breaks student comparability | Treat parquet + R3 as immutable; cache teacher OOF |
| Flight-split non-determinism | Sort flight IDs before split (fixed in `data.py`) |
| Overclaiming student RMSE as official score | Always label flight-holdout val as non-official |
| Architecture search without fixed KD weights | Lock α/β from Step 3 before capacity experiments |

---

## Final Executive Summary

### Official position (July 2026 — teacher frozen after R3)

- **Canonical Combined RMSE: 228.25 kg** (Rank 239.2 · Final 220.9) vs published winner ≈ **201 kg** (Δ ≈ **+27 kg**).
- **Frozen teacher best (R3 dynamic mass + P1E): 221.33 kg** (Rank 232.5 · Final 213.7; Δ vs winner ≈ **+20 kg**).
- Gap-closing ladder: **228.25 → 227.44 → 226.19 → 225.25 → 221.33** (−6.92 kg total). **No superiority claim**.
- **Dynamic mass** is the largest single teacher improvement; bias ~**+3.7–3.9 kg**.

### Distillation position (July 2026 — Steps 1–5 complete)

- Frozen soft-label dataset: **119,032** intervals, **60** features.
- **α/β default: α=0.1, β=0.9**.
- Capacity (val): best **XLarge 6.75M** val RMSE **228.1 kg**; **Large ~2.9M** within **2 kg**.
- **Official Final held-out baseline: Large 215.85 kg** (XLarge 218.59; teacher 213.62). Ranking reversed vs val; XLarge **not** justified on Final.
- Generalization gap val→Final: Large **−6.0%**, XLarge **−4.2%** (test better — no overfit signal).
- Latency: students **≫** faster than R3 ensemble (single-sample CPU ~0.26–0.52 ms vs ~52 ms).
- Ready for FT-Transformer / TabTransformer; comparison target = **Large Final RMSE 215.85**.

### Established (science)

- Energy-state and Energy+Weather features yield bootstrap-supported gains on **unseen flights** (Level 1, PRC).
- Fuel-flow targets improve Level-1 and official error vs Direct; keep protocol-separated leaderboards.
- Official gap-closing **R1–R3 kept** and **frozen as teacher**.
- External DASHlink pilot: energy + Flow **qualitatively replicate** at pilot scale only.
- KD weight structure matters more than balanced 0.5/0.5 for the baseline MLP.
- **MLP capacity saturates by Large (~3M) on held-out Final; extra width does not transfer.**

### Suggestive

- LOTO Flow macro advantage ~17 kg is **not** statistically confirmed; B77W-sensitive.
- Physical aircraft distance is **not** a robust transfer predictor without B77W.
- Small GT anchor (α≈0.1) may slightly help vs pure teacher — re-check on other architectures.

### Must happen next (prioritized)

1. **§23 D6** — FT-Transformer / TabTransformer with **α=0.1, β=0.9** (start near ~3M params); beat **215.85 Final**.
2. **§22** — freeze docs/numbers; thesis chapter on teacher + distillation + capacity + **held-out eval**.
3. Hard-subgroup focus (heavies / ultra-long) rather than more MLP width.
4. Optional / deferred: remaining §21 teacher RMSE items (do not un-freeze without cause).

**Paper / project readiness:** Official evaluation + R1–R3 + distillation Steps 1–5 (MLP + held-out baseline) are **complete and honest**. Do not claim winner-level performance. Quote **Large Final 215.85** as the student baseline, not internal val-only RMSE.

---

*Report updated 2026-07-30 (R3 teacher frozen; distillation Steps 1–5 complete; official Large Final baseline 215.85 kg).*

**Reproduce (core + official + gap-close + distillation):**

```bash
# Official teacher path (reference; do not retrain for distillation)
PYTHONPATH=src python experiments/07_gap_closing/26_r3_ensemble_mass.py

# Distillation
set PYTHONPATH=src
python experiments/08_distillation/01_build_teacher_distillation_dataset.py --train-only
python experiments/08_distillation/02_train_mlp_student.py
python experiments/08_distillation/run_distillation_experiments.py sweep
python experiments/08_distillation/run_distillation_experiments.py capacity
python experiments/08_distillation/05_test_evaluation.py --final-featured featured_dataset_final.parquet
```
