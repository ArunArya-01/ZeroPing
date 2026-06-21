# AeroTwin: Physics-Informed Hybrid Modeling for Aircraft Fuel Burn Prediction

**Project:** ZeroPing / AeroTwin  
**Focus:** Hybrid OpenAP + ML residual / direct correction on EUROCONTROL PRC 2025 dataset  
**Date:** June 2026 (status)

## Problem

Accurate prediction of fuel burn for commercial aircraft is essential for:
- Emissions accounting and regulatory compliance (e.g., CORSIA, EU ETS)
- Operational efficiency optimization (routes, speeds, altitudes)
- Digital twin and fleet performance analytics

Real-world operational data presents severe challenges:
- Fused ADS-B (dense kinematics: position, groundspeed, altitude, vertical rate) + sparse ACARS reports (fuel-on-board, occasional air data such as Mach/CAS).
- Partial observability: ACARS fuel labels cover only a median of ~32% of flight duration (takeoff to landed).
- Unknown aircraft mass (no TOW or fuel state at start of intervals).
- High heterogeneity in data quality: ~35–46% of labeled intervals are "very sparse" (<5–10 trajectory points, often just the bounding ACARS reports).
- Pure physics models (OpenAP, BADA) rely on assumptions (reference mass, inferred TAS) that produce large systematic errors on real telemetry.

A baseline OpenAP physics model alone achieves MAE ≈ 668 kg and negative R² (−2.16) on held-out commercial flights—worse than predicting the mean fuel burn. This renders standalone physics unusable for operational or regulatory purposes without correction.

## Gap in literature

Existing literature falls into two broad categories with clear limitations for this setting:

1. **Physics-only models** (OpenAP, BADA, etc.): Provide interpretable, physics-grounded fuel flow estimates but are known to degrade under uncertain mass, incomplete air data, and real-world conditions not matching model assumptions. Prior work quantifies baseline errors but rarely explores data-driven correction at scale on hybrid sparse telemetry.

2. **Pure data-driven / ML approaches**: Common on simulated datasets (e.g., NASA turbofan degradation for engine RUL) or high-fidelity flight simulators. These ignore or under-utilize known physics structure (energy conservation, phase-dependent consumption) and often assume rich, fully-observed sensor streams not available in operational ACARS+ADS-B feeds.

The EUROCONTROL PRC 2025 challenge (Sun et al., 2026) released the fused ACARS/ADS-B fuel estimation dataset and some initial baselines. However, there is limited published work that:
- Systematically characterizes partial observability and sparsity effects on real commercial flights at this scale (~10k flights, 119k intervals).
- Performs rigorous ablation of *specific* physics-informed inductive biases (energy state representations, wind/density proxies) versus raw physics point estimates or operational heuristics.
- Uses strict flight-level cross-validation (preventing leakage from correlated intervals within the same flight) combined with flight-clustered bootstrap inference to assess statistical significance of gains.
- Compares direct hybrid prediction against explicit residual learning architectures on this data regime.

The gap is therefore: **how (and whether) targeted physics-derived features can meaningfully and reliably improve modern gradient-boosted ML predictors for fuel burn under the realistic constraints of partial observability, unknown mass, and heterogeneous telemetry quality.**

## Research question

**Primary RQ:** Can the addition of explicit physics-informed features (energy-state representations and atmospheric/wind proxies derived from available kinematics) yield statistically significant improvements in aircraft fuel burn prediction accuracy over a strong OpenAP-hybrid baseline, when evaluated on unseen flights using real-world partially observable ADS-B + ACARS data?

**Secondary questions / hypotheses tested:**
- Does OpenAP provide value primarily as a direct input feature, or do richer energy-derived quantities add independent signal?
- Is the benefit of physics features concentrated in sparse-telemetry regimes (as intuition suggests)?
- Do explicit residual-learning architectures (physics + learned correction) outperform direct prediction of actual fuel (with physics as a feature)?
- Do operational descriptors (cruise variability, holding indicators, path efficiency) or simple weather proxies add value beyond kinematics + energy?

## Methods

### Dataset and preprocessing
- Source: `aerotwin/aero-data` (Hugging Face; EUROCONTROL PRC 2025 challenge data).
- Usable train split: 10,000 flights with complete metadata + trajectory parquet files (out of 11,037 total; ~1k flights lack trajectories and are excluded).
- Labeled targets: 119,032 fuel intervals derived from ACARS FOB differences (after cleaning: 115,995 intervals across 9,976 flights).
- Remote access via Polars + `hf://` (no full local download); `data.AeroDataLoader`.
- Strict flight-level split (random_state=42, 80/20): 7,980 train flights (92,964 intervals) / 1,996 test flights (23,031 intervals). No flight appears in both sets.

### Physics baseline (OpenAP)
- Implemented in `physics/openap_baseline.py`.
- For each labeled fuel interval [start, end]:
  - Extract trajectory window points.
  - Classify dominant phase from median vertical rate (climb > +1.5 m/s, descent < −1.5 m/s, else cruise).
  - Infer true airspeed (TAS) with priority: Mach → CAS → groundspeed fallback.
  - Reference mass = MTOW × 0.75 (documented crude assumption).
  - Call `FuelFlow.enroute(ac_type, mass, tas, alt, vs)` at a representative point in the window.
  - Integrate fuel flow over interval duration → `physics_fuel_kg`.
- Output also includes per-interval `residual_kg = actual_fuel_kg − physics_fuel_kg`, sparsity signals, and method flag (TAS inference path).

### Feature engineering
Base features (trajectory + metadata + physics):
- Interval metadata: `duration_s`, `start_fraction_of_flight`, `end_fraction_of_flight`.
- Trajectory statistics over window: mean/median/max/std of altitude, groundspeed, vertical rate.
- Phase fractions (`climb_fraction`, `cruise_fraction`, `descent_fraction`).
- Data quality: `n_traj_pts`, `has_acars_in_window`, TAS inference `method`.
- Categorical: `aircraft_type` (26 types; A320 family dominant), origin/destination.
- Physics: `physics_fuel_kg`.

Physics-informed augmentations ():
- **Energy-state features (E2)** (`physics/feature_engineering.py`): potential energy, kinetic energy, specific energy (SE = g·h + ½ TAS²), energy change/rate, climb/energy efficiency, cumulative energy change. Computed using reference mass and per-point TAS inference.
- **Operational features (E3)**: climb/descent duration, cruise speed variability, holding indicators (low GS + low VR), path efficiency proxies, altitude stability, segment acceleration.
- **Weather / atmosphere proxies (E5)** (`physics/weather_features.py`): ISA temperature/pressure/density at altitude, density altitude, headwind and crosswind (TAS/GS/track decomposition), ISA deviation proxy. (No direct METAR/GRIB data available.)
- Additional derived: wind-adjusted physics variant for conditional experiments.

Featured dataset materialized once via `physics/build_featured_dataset.py` + enrichment scripts; 32+ columns.

### Models and training
- Primary learners: XGBoost (lr=0.05, max_depth=8, 300 estimators), LightGBM (300 est, lr=0.05), Random Forest (100 trees, max_depth=15), Linear Regression (for baseline).
- Two paradigms:
  - **Direct hybrid**: Predict `actual_fuel_kg` directly; include `physics_fuel_kg` as a strong feature. Model implicitly learns corrections.
  - **Residual learning**: Predict `residual_kg`; final prediction = `physics_fuel_kg + predicted_residual`.
- Preprocessing: One-hot for categoricals (aircraft_type, method, airports), median imputation for numeric, StandardScaler for linear models.
- All models trained on the exact same flight-level split and feature sets for comparability.

### Evaluation and statistical framework
- Primary metrics: MAE (kg), RMSE (kg), R² on the held-out 1,996 flights (23k intervals).
- Rigorous inference (`physics/eval_framework.py`, `notebooks/07_significance_testing.py` and later):
  - **Flight-clustered bootstrap**: 10,000 iterations. Resample *test flights* with replacement (preserves within-flight dependence among intervals).
  - Statistic: ΔMAE = MAE(A) − MAE(B).
  - Report 95% bootstrap CI; one-sided `bootstrap_p = P(ΔMAE > 0)`.
  - Primary decision rule: improvement is supported only if 95% CI excludes 0 *and* `bootstrap_p < 0.05`.
  - Supplementary: Wilcoxon signed-rank on paired per-interval absolute errors (treats intervals as independent; often optimistic).
  - Effect size: Cohen's *d* on paired error differences (negligible / small / medium / large).
- Ablation structure: Compare families (OpenAP only, No Physics / kinematics-only, Energy, Operational, Weather, Energy+Weather, Residual variants, MLP residual) using the same framework.

### Reproducibility
- Fixed seeds, documented splits, scripts in `notebooks/05_–09_*.py` and `physics/`.
- All tables/figures in `figures/`.
- Bootstrap artifacts and significance tables preserved.

## Results

### Core baselines (flight-level test set)
| Approach                  | Model     | MAE (kg) | RMSE (kg) | R²     |
|---------------------------|-----------|----------|-----------|--------|
| OpenAP only               | —         | 667.6    | 1,582.4   | −2.16  |
| Direct hybrid (OpenAP)    | XGBoost   | 86.31    | 224.1     | 0.937  |
| Direct hybrid (OpenAP)    | Random Forest | 86.3 | 228.8 | 0.93 |
| No Physics (kinematics + metadata only) | XGBoost | 89.46 | — | 0.93 |

**Key:** Pure physics is unusable. Strong tree ensembles with trajectory + metadata + physics fuel achieve ~86–90 kg MAE.

### Physics-informed ablations (selected; XGB unless noted)
| Experiment                  | MAE (kg) | ΔMAE vs OpenAP Hybrid | 95% Bootstrap CI          | Verdict                  |
|-----------------------------|----------|-----------------------|---------------------------|--------------------------|
| OpenAP Hybrid (baseline)    | 86.31    | —                     | —                         | —                        |
| Energy Hybrid (E2)          | 84.48    | −1.82                 | [−2.92, −0.67]            | **Significant** (accepted) |
| Energy + Weather (E6)       | **83.76**| **−2.55**             | **[−3.58, −1.50]**        | **Significant (best)**   |
| Weather only (E5)           | 86.59    | +0.28                 | [−0.40, +1.07]            | Not significant          |
| Operational features (E3)   | 86.76    | +0.46                 | [−0.10, +1.01]            | Not significant          |
| No Physics                  | 89.46    | +3.15                 | [0.42, 7.08]              | OpenAP Hybrid better     |

### Architecture comparisons
- **Residual learning (trees):** MAE ≈ 107–108 kg (XGB/LGBM/RF) — significantly **worse** than direct hybrid (+20.8 kg ΔMAE; CI excludes zero).
- **MLP residual correction:** MAE 103.7 kg — significantly worse than OpenAP hybrid (Δ +17.4 kg, CI [7.84, 34.99]).
- **Sparsity-conditioned analysis:** Descriptive gains appeared larger in some sparse buckets, but **all flight-clustered bootstrap CIs overlapped zero**. The hypothesis that "physics helps most under sparse telemetry" is rejected.

### Statistical notes
- All significance tests use 10k flight-resampled bootstraps.
- Effect sizes for accepted gains (energy features) are small/negligible in Cohen's d terms, but the absolute MAE reductions (1.8–2.55 kg) are consistent and CI-supported.
- Negative results (residual architectures, sparsity interaction, operational/weather-only) are reported explicitly.

**Strongest result:** Energy+Weather Hybrid (XGBoost) — **MAE = 83.76 kg** on 1,996 completely unseen flights.

Artifacts: `figures/table_v3_leaderboard.csv`, `table_significance_v3_all.csv`, bootstrap histograms, etc.

## Key takeaways

1. **Hybrid modeling works at scale on real data.** Direct inclusion of a physics baseline feature plus rich kinematics allows tree ensembles to achieve MAE ~84–87 kg on held-out commercial flights—more than 7–8× better than raw OpenAP.

2. **Not all physics priors are equal.** Raw OpenAP point estimates add only modest value (~1–3 kg MAE) when strong trajectory summaries are present. Explicit **energy-state representations** (potential/kinetic/specific energy, rates, efficiency) deliver the largest statistically supported gains.

3. **Energy + lightweight atmospheric proxies is the winning combination** (E6): 83.76 kg MAE, robust bootstrap significance. Weather proxies and operational descriptors alone add little or nothing.

4. **Residual learning underperforms direct hybrid** on this feature set and data regime. Predicting the correction explicitly (trees or MLP) yields worse accuracy than letting the model see physics_fuel_kg and predict total fuel directly.

5. **Intuitive hypotheses can fail rigorous testing.** The expectation that physics corrections would be most valuable on very sparse intervals was **not supported** once flight-level dependence and proper clustered bootstrap inference were applied. Many ablations produced descriptive differences that did not survive significance testing.

6. **Strong ML can recover much signal, but targeted physics features still help.** Even gradient-boosted trees benefit from explicit energy conservation quantities they do not fully reconstruct from summary statistics alone.

7. **Partial observability is fundamental**, not a corner case: median 32% labeled coverage, extreme variance in points per interval. Any operational system must be robust to 2-point "boundary-only" intervals.

8. **Implications for practice and future work:**
   - Energy-aware features are cheap to compute from existing kinematics and worth including.
   - Better mass estimation (or latent mass modeling) remains a high-leverage open direction (current ref mass is crude).
   - Future modeling could explore transformers, physics-informed losses, aircraft-type generalization (leave-one-type-out), and SHAP-based interpretability of energy contributions.
   - The framework (flight-level splits + clustered bootstrap) provides a reproducible template for claiming improvements on correlated aviation trajectory data.

**Best model to date:** Energy+Weather Hybrid XGBoost at **83.76 kg MAE**.

This work demonstrates that carefully chosen, computable physics-informed inductive biases can deliver reliable (if incremental) gains for real-world aviation fuel prediction even when paired with powerful modern ML, while exposing which common assumptions do not hold under operational data constraints.

---

*Sources / reproducibility: See `PROJECT_STATUS_REPORT.md`, `notebooks/05_baseline_modeling.py` through `09_physics_features_v3.py`, `physics/*.py`, and generated tables/figures in `figures/`. Dataset: Hugging Face `aerotwin/aero-data`. Cite the original EUROCONTROL PRC 2025 challenge paper (Sun et al., 2026, JOAS) for the source data.*