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

- Split by `flight_id` (80/20, seed=42)
- **Train:** 7,980 flights → 92,964 intervals
- **Test:** 1,996 flights → 23,031 intervals
- **Overlap:** 0 flights

### Results (held-out flights)

| Approach | Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|---|
| OpenAP only | — | 668 | 1,582 | −2.16 |
| Direct | Random Forest | 87.1 | 232.8 | 0.93 |
| Direct | XGBoost | 89.5 | 230.6 | 0.93 |
| Direct | LightGBM | 91.8 | 219.6 | 0.94 |
| Residual | XGBoost | 107.1 | 307.6 | 0.88 |
| Residual | LightGBM | 108.7 | 293.3 | 0.89 |
| Residual | Random Forest | 107.5 | 312.5 | 0.88 |

### Comparison to row-level split

| Metric | Row-level | Flight-level | Change |
|---|---|---|---|
| OpenAP MAE | 655 kg | 668 kg | +2% |
| Best residual MAE | 100 kg | 107 kg | +7% |
| Best residual R² | 0.91 | 0.88 | −0.03 |

### Conclusion

**Performance generalizes to unseen flights.** The modest degradation under strict splitting confirms that ML models learn transferable patterns—not flight-specific memorization. Residual learning reduces MAE by **~84%** relative to OpenAP on held-out flights.

**Artifacts:** `figures/table_model_comparison_flight_split.csv`, `figures/fig_actual_vs_predicted.png`

---

## 8. Physics Ablation Study

**Script:** `notebooks/06_physics_ablation.py`  
**Question:** How much does `physics_fuel_kg` contribute as an ML feature?

### Experimental conditions

| Condition | Description |
|---|---|
| **Full Hybrid** | All features including `physics_fuel_kg` |
| **No Physics** | Remove `physics_fuel_kg`; predict from trajectory/metadata only |
| **Physics Only** | Use raw OpenAP prediction (no ML) |

Evaluated on the same flight-level test set. Tree models (RF, XGBoost, LightGBM) tested; best results per condition below.

### Results

| Condition | Best Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|---|
| Full Hybrid | Random Forest | 86.3 | 228.8 | 0.93 |
| No Physics | Random Forest | 87.1 | 232.8 | 0.93 |
| Physics Only | OpenAP | 667.6 | 1,582.4 | −2.16 |

**Removing physics increases MAE by only ~0.9 kg (~1.0%).**

### Interpretation

- **ML models are largely data-driven** when rich trajectory and metadata features are available.
- **`physics_fuel_kg` contributes little as an incremental ML feature** because duration, aircraft type, and kinematic statistics already encode similar information.
- **OpenAP alone performs poorly** (MAE 668 kg)—physics is not replaceable as a standalone predictor.
- **Physics remains useful** as an interpretable baseline and as the foundation of the hybrid/residual architecture, even if it is redundant as an input feature for direct prediction.

**Artifacts:** `figures/table_physics_ablation.csv`, `figures/fig_physics_ablation.png`

---

## 9. Sparsity Study

**Script:** `notebooks/07_sparsity_ablation.py`  
**Question:** Does physics become more valuable when telemetry is limited?

### Sparsity buckets (by `n_traj_pts`)

| Bucket | Definition | Test intervals |
|---|---|---|
| **Dense** | > 1,000 points | 2,432 |
| **Medium** | 100–1,000 points | 11,278 |
| **Sparse** | 10–99 points | 1,141 |
| **Very Sparse** | < 10 points | 8,180 |

Per bucket: train LightGBM (Full Hybrid vs No Physics) on bucket-filtered train data; evaluate on bucket-filtered test data. Same flight-level split.

### Results

| Bucket | Full Hybrid MAE | No Physics MAE | OpenAP MAE | Physics gain |
|---|---|---|---|---|
| Dense | 151.5 kg | 156.7 kg | 1,024 kg | 5.2 kg (3.3%) |
| Medium | 48.3 kg | 50.1 kg | 243 kg | 1.8 kg (3.6%) |
| Sparse | 74.5 kg | 88.9 kg | 248 kg | **14.4 kg (16.2%)** |
| Very Sparse | 129.9 kg | 130.6 kg | 1,205 kg | 0.7 kg (0.5%) |

### Interpretation

- **Physics helps most in the Sparse bucket (10–99 points):** removing physics degrades MAE by 16%, the largest gain across buckets.
- **Benefit is modest overall** and **not monotonic** with sparsity.
- **Physics does not rescue extremely sparse intervals** (<10 points): gain is <1%; metadata features (`duration_s`, `aircraft_type`, `method`) dominate.
- **Medium-density intervals are easiest to predict** (MAE ~48 kg)—sufficient kinematic coverage without the noise of very short windows.
- **ML dramatically outperforms OpenAP in every bucket**, confirming the hybrid architecture's value even when physics-as-feature adds little.

**Artifacts:** `figures/table_sparsity_ablation.csv`

---

## 10. Key Scientific Findings

1. **OpenAP exhibits strong systematic bias** on real ACARS-labeled data (R² ≈ −2.2 flight-level; MAE ≈ 668 kg).

2. **Machine learning predicts fuel burn accurately** when trajectory and metadata features are available (MAE ≈ 87–108 kg; R² ≈ 0.88–0.94 on unseen flights).

3. **Models generalize to unseen flights** under strict flight-level splitting with only modest metric degradation vs. row-level splits.

4. **Physics is largely redundant as an ML input feature** when rich trajectory features exist (~1% MAE difference in ablation), but **essential as a standalone baseline** and residual-learning foundation.

5. **Physics provides the most incremental value under moderate sparsity** (10–99 trajectory points; +16% MAE without physics), not under extreme sparsity (<10 points).

6. **Aircraft type and interval duration are dominant predictors** of residual error (LightGBM gain and permutation importance).

7. **Partial observability is structural:** median ~32% of flight time is labeled; many intervals have only 2 trajectory points.

8. **Residual errors are highly structured,** correlating with physics prediction magnitude (ρ ≈ −0.95), duration, phase fractions, and data quality.

---

## 11. Current Project Status

| Milestone | Status |
|---|---|
| Dataset ingestion (`AeroDataLoader`, HuggingFace remote access) | ✅ Complete |
| Exploratory data analysis (notebooks 01–04) | ✅ Complete |
| Physics baseline (OpenAP per-interval pipeline) | ✅ Complete |
| Feature engineering (`featured_dataset.parquet`, 119k intervals) | ✅ Complete |
| `flight_id` integration for flight-level splits | ✅ Complete |
| ML baselines (LR, RF, XGBoost, LightGBM) | ✅ Complete |
| Flight-level validation | ✅ Complete |
| Physics ablation study | ✅ Complete |
| Sparsity-conditioned ablation | ✅ Complete |
| SHAP explainability analysis | ⬜ Not started |
| Aircraft-level error analysis | ⬜ Not started |
| Cross-aircraft generalization (leave-one-type-out) | ⬜ Not started |
| Neural residual model (MLP / transformer) | ⬜ Not started |
| Paper drafting (Paper 1: characterization; Paper 2: hybrid model) | ⬜ In progress |

### Key artifacts

| File | Description |
|---|---|
| `featured_dataset.parquet` | Training-ready featured dataset |
| `physics/build_featured_dataset.py` | Dataset builder |
| `physics/openap_baseline.py` | OpenAP baseline + feature extraction |
| `notebooks/05_baseline_modeling.py` | Flight-level ML baselines |
| `notebooks/06_physics_ablation.py` | Physics feature ablation |
| `notebooks/07_sparsity_ablation.py` | Sparsity-conditioned ablation |
| `figures/table_model_comparison_flight_split.csv` | Main model comparison |
| `figures/table_physics_ablation.csv` | Physics ablation results |
| `figures/table_sparsity_ablation.csv` | Sparsity ablation results |
| `FEATURED_DATASET.md` | Dataset schema documentation |

---

## 12. Recommended Next Steps

### High priority

**1. Aircraft-level error analysis**

Break down MAE and residual distributions by `aircraft_type`. Wide-bodies (A359, B789) may exhibit distinct error modes due to mass scaling and cruise profiles. Informs whether per-type models or embeddings are needed.

**2. SHAP explainability**

Quantify per-prediction feature contributions beyond global LightGBM importance. Critical for trust, debugging, and Paper 2 narrative—especially for `n_traj_pts`, `method`, and phase fractions.

**3. Cross-aircraft generalization**

Leave-one-aircraft-type-out (LOAO) or hold out rare types (B77W, A388). Tests whether models extrapolate to unseen airframes—a key deployment risk given 26 types with heavy class imbalance.

### Medium priority

**4. Neural baselines**

Train an MLP or small transformer on the featured dataset with aircraft/method embeddings. Compare against tree models; neural nets may capture nonlinear interactions between sparsity and phase more expressively.

**5. Improved physics models**

Better mass estimation (type + route + fraction-of-flight regression), wind correction, or flight-path integration instead of single-point FuelFlow. Could improve both the OpenAP baseline and residual target structure.

### Low priority

**6. API / deployment**

FastAPI inference endpoint wrapping the trained hybrid model for batch interval scoring.

**7. UI / demo work**

Dashboard for visualizing per-flight predictions, residuals, and sparsity indicators. Useful for stakeholder demos but not required for research validation.

---

## 13. Final Assessment

### Does AeroTwin work?

**Yes—with qualifications.** AeroTwin's machine learning layer accurately predicts fuel burn on held-out flights (MAE ~87–108 kg, R² ~0.88–0.94), representing an **~84% error reduction** over raw OpenAP. The core thesis—that structured physics errors can be corrected with trajectory-aware ML—is validated on 10,000 flights and 116,000 intervals.

### What has been validated?

- Remote data pipeline for `aerotwin/aero-data` without full download
- OpenAP per-interval baseline with TAS inference and feature extraction
- Featured dataset construction at scale (119k intervals, 32 columns)
- Structured, learnable residuals correlated with observability and phase
- ML generalization under **strict flight-level** train/test separation
- Physics ablation: ML is data-driven; OpenAP is a weak standalone predictor but supports the hybrid framing
- Sparsity ablation: physics-as-feature helps most at moderate sparsity (10–99 points), not at extreme sparsity

### What remains unproven?

- **Cross-aircraft extrapolation** to rare or unseen types
- **Neural architectures** beyond gradient-boosted trees
- **Production-grade physics** (true mass, wind, engine degradation)
- **Temporal / OOD generalization** on `rank` and `final` splits
- **Explainability** at the per-prediction level (SHAP)
- **Statistical significance** of sparsity-conditioned physics gains across seeds

### Strongest results so far

| Result | Detail |
|---|---|
| **Best overall MAE** | 87.1 kg (Random Forest, direct prediction, flight-level) |
| **Best residual MAE** | 107.1 kg (XGBoost, residual learning, flight-level) |
| **Generalization gap** | +7% MAE vs. row-level split (acceptable) |
| **OpenAP improvement** | 84% MAE reduction (668 → 107 kg) |
| **Dominant features** | `duration_s`, `aircraft_type` |
| **Key sparsity finding** | Physics feature gain peaks at 10–99 traj points (+16% MAE without it) |

### Executive summary

AeroTwin has progressed from dataset characterization through physics baseline validation, feature engineering, and rigorous ML benchmarking. The project demonstrates that **aviation fuel burn can be predicted accurately from partially observable trajectory data**, that **models generalize to unseen flights**, and that **OpenAP provides a useful—but not sufficient—physics foundation**. The path forward prioritizes aircraft-level diagnostics, explainability, and cross-type generalization before investing in neural architectures and deployment infrastructure.

---

*Report generated from experiments in the ZeroPing repository, June 2026. Reproduce results with `python notebooks/05_baseline_modeling.py`, `06_physics_ablation.py`, and `07_sparsity_ablation.py`.*