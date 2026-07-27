# AeroTwin Research Paper — Complete Writing Package

**Project:** ZeroPing / AeroTwin  
**Working title:** *Physics-Informed Hybrid Modeling for Aircraft Fuel Burn Prediction under Partial Observability*  
**Dataset:** EUROCONTROL PRC 2025 (`aerotwin/aero-data`)  
**Last synced with repo:** 2026-07-27  
**Status:** Ready for draft writing (canonical + R3 gap-closing results frozen)

This file is the **single source of content** for writing the research paper: problem framing, literature gap, methods, results numbers, discussion points, limitations, figure/table names, and a drafting checklist. Expand each section into formal academic prose; do not invent numbers beyond what is listed here.

---

## 0. Paper metadata (fill before submission)

| Field | Value |
|-------|--------|
| **Working title** | Physics-Informed Hybrid Modeling for Aircraft Fuel Burn Prediction under Partial Observability |
| **Alt title A** | Hybrid OpenAP–Gradient Boosting for Interval-Level Aircraft Fuel Burn Estimation |
| **Alt title B** | When Physics Helps: Energy Features, Fuel-Flow Targets, and Cross-Aircraft Transfer on PRC 2025 |
| **Authors** | _[to fill]_ |
| **Affiliation** | _[to fill]_ |
| **Target venues** | Journal of Open Aviation Science (JOAS); Transportation Research Part C; Aerospace Science & Technology; Data-centric AI / Scientific Data (characterization paper) |
| **Keywords** | aircraft fuel burn; hybrid physics–ML; OpenAP; ADS-B; ACARS; residual learning; gradient boosting; partial observability; leave-one-type-out; EUROCONTROL PRC |
| **Code** | https://github.com/ArunArya-01/ZeroPing |
| **Data** | https://huggingface.co/datasets/aerotwin/aero-data |
| **License (code)** | See `LICENSE` |
| **Primary metric (official)** | Combined Rank+Final **RMSE (kg)** |
| **Canonical score** | Combined RMSE **228.25 kg** (frozen V4 ensemble) |
| **Current best** | Combined RMSE **221.33 kg** (R3 dynamic mass + P1E) |
| **Published winner** | ≈ **201 kg** combined RMSE (no superiority claim) |

---

## 1. Abstract (draft — rewrite for venue length)

Accurate estimation of commercial aircraft fuel burn from operational telemetry is central to emissions accounting, operational efficiency, and digital-twin analytics. Real-world data fuse dense ADS-B kinematics with sparse ACARS fuel-on-board reports, yielding partial observability, unknown aircraft mass, and highly variable trajectory density. We present **AeroTwin**, a hybrid physics–machine-learning system that combines an OpenAP fuel-flow baseline with gradient-boosted models trained on interval-level trajectory, energy-state, and atmospheric-proxy features.

On the EUROCONTROL PRC 2025 challenge data (~10,000 training flights; 119,032 fuel intervals), pure OpenAP is unusable (flight-holdout MAE ≈ 668 kg, R² ≈ −2.16). Direct hybrid tree ensembles reach MAE ≈ 84–88 kg on strict flight-level holdout. **Energy-state features** yield bootstrap-supported gains (ΔMAE ≈ −1.8 to −2.6 kg); residual-learning architectures and sparsity-conditioned physics gains are **rejected**. Under the **official Rank+Final** protocol (train-only fits), a six-base ensemble achieves combined RMSE **228.25 kg**; gap-closing with phase-conditional calibration, heavy-aircraft specialists, and **dynamic mass features** reaches **221.33 kg**, remaining ≈20 kg above the published winner (≈201 kg). Leave-one-type-out evaluation inflates error ~3×, showing that standard flight splits overestimate robustness under aircraft-family shift. Pilot external validation on NASA DASHlink replicates energy and fuel-flow benefits at small scale. We release loaders, featured-dataset builders, a frozen statistical protocol, and full experiment artifacts for reproducibility.

**Word target for final abstract:** 150–250 words (venue-dependent).

---

## 2. One-page elevator summary (use for intro / poster)

| Item | Statement |
|------|-----------|
| **Problem** | Predict kg of fuel burned on ACARS-labeled intervals from fused ADS-B + ACARS trajectories when mass is unknown and air-data is incomplete. |
| **Why hard** | Median only ~32% of flight time is labeled; 35–46% of intervals are very sparse (<5 traj points); OpenAP alone has negative R². |
| **Approach** | Hybrid: OpenAP prior + GBDT (XGBoost / LightGBM / CatBoost) on kinematics + **energy** + weather proxies; optional fuel-flow target; stacking ensemble; R3 dynamic mass. |
| **Key result (Level 1)** | Direct hybrid MAE ~84–88 kg; Energy+Weather best ablations with CI-supported ΔMAE. |
| **Key result (official)** | Combined RMSE 228.25 → **221.33** kg after R3; not better than published winner (~201). |
| **Key result (Level 2)** | LOTO macro MAE ~266–283 kg; standard metrics overstate transfer. |
| **Honest claim** | Competitive open hybrid pipeline with rigorous stats and external pilot; **no superiority** over PRC winner. |

---

## 3. Motivation and problem statement

### 3.1 Why fuel burn estimation matters

- **Emissions accounting** and regulatory schemes (CORSIA, EU ETS, airline sustainability reporting).
- **Operational efficiency:** route, speed, altitude, and cost-index optimization.
- **Digital twins / fleet analytics:** performance monitoring without full FDR access.
- **Benchmarking** physics models (OpenAP, BADA) against real commercial telemetry.

### 3.2 Problem formulation

For each labeled fuel interval \(i\) on flight \(f\):

\[
\hat{y}_i = f\!\big(\mathbf{x}_i^{\text{traj}},\; \mathbf{x}_i^{\text{meta}},\; \hat{y}_i^{\text{physics}},\; \mathbf{x}_i^{\text{engineered}}\big)
\]

where:

- \(y_i =\) `actual_fuel_kg` (ACARS FOB difference over `[start, end]`)
- \(\hat{y}_i^{\text{physics}} =\) OpenAP integrated fuel (`physics_fuel_kg`)
- \(\mathbf{x}_i^{\text{traj}}\) = altitude / groundspeed / vertical-rate statistics, phase fractions, density
- \(\mathbf{x}_i^{\text{meta}}\) = aircraft type, origin/destination, timing
- Engineered groups: energy-state, weather/atmosphere proxies, (later) dynamic mass

**Alternate target (fuel flow):**

\[
\hat{y}_i = \widehat{\text{flow}}_i \times \text{duration}_s,\quad
\text{flow}_i = \frac{\text{actual\_fuel\_kg}}{\text{duration}_s}
\]

### 3.3 Core challenges (write these into Introduction)

1. **Partial observability** — labels cover median ~32% of takeoff–landed duration.
2. **Unknown mass** — no TOW / ZFW; crude `MTOW × 0.75` is a major error source (addressed by R3).
3. **Sparse / heterogeneous telemetry** — ADS-B dense but no Mach/TAS; ACARS sparse; many 2-point boundary intervals.
4. **Aircraft heterogeneity** — narrow-body-dominated fleet; heavy wide-bodies dominate RMSE.
5. **Distribution shift** — temporal (Rank/Final months), type shift (LOTO), external datasets (DASHlink / OpenSky).
6. **Evaluation leakage risk** — multiple intervals per flight → must split by `flight_id`.

---

## 4. Literature gap and positioning

### 4.1 Two camps (and why neither is enough)

| Camp | Strength | Limitation for this setting |
|------|----------|------------------------------|
| **Physics-only** (OpenAP, BADA, …) | Interpretable, type-aware fuel flow | Degrades under wrong mass, missing air-data, real operational conditions; alone MAE ~668 kg here |
| **Pure ML** | Flexible on large data | Often assumes rich sensors or simulation; ignores energy / phase structure; weak physics consistency |

### 4.2 Specific gap AeroTwin fills

Limited prior work that **jointly**:

1. Characterizes partial observability and sparsity at PRC scale (~10k flights, 119k intervals).
2. Ablates **specific physics-informed inductive biases** (energy state, weather proxies) with **flight-clustered bootstrap** inference.
3. Contrasts **direct hybrid** vs **explicit residual learning** under the same strict splits.
4. Separates **Level 1** (unseen flights) from **Level 2** (unseen aircraft families / LOTO).
5. Reports **official Rank+Final** scores without leakage, plus honest gap to published winner.
6. Provides **external pilot replication** (not only same-dataset holdout).

### 4.3 Anchor reference (challenge paper)

- Sun, Spinielli & Strohmeier, *Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge*, Journal of Open Aviation Science (2026). doi:10.59490/joas.2026.8750  
- OpenAP: Sun et al. (aircraft performance model library).

### 4.4 Related themes to cite when expanding

- Aircraft performance models and fuel-flow estimation (BADA, OpenAP).
- ADS-B analytics and trajectory mining.
- Physics-informed ML / residual correction in engineering systems.
- Domain shift, hierarchical/clustered bootstrap, leave-one-group-out evaluation.
- Aviation emissions and operational fuel prediction literature.

---

## 5. Research questions and hypotheses

### 5.1 Primary research question

> Can **physics-informed features** (energy-state representations and atmospheric/wind proxies), combined with a hybrid OpenAP + gradient-boosting architecture, yield **statistically significant** improvements in interval-level aircraft fuel burn prediction on real partially observed ADS-B+ACARS data, when evaluated under **flight-level** and **official Rank+Final** protocols?

### 5.2 Secondary questions

| ID | Question | Status |
|----|----------|--------|
| RQ1 | Does raw OpenAP as a feature help when rich kinematics exist? | **Yes, modest** (model-dependent; XGB significant) |
| RQ2 | Do **energy-state** features improve over OpenAP hybrid? | **Yes, significant** |
| RQ3 | Do weather proxies alone help? | **No** (not significant alone; useful with energy) |
| RQ4 | Do operational descriptors (holding, path efficiency) help? | **No** (rejected) |
| RQ5 | Is residual learning better than direct hybrid? | **No** — residual **worse** |
| RQ6 | Is benefit of physics concentrated in sparse intervals? | **No** — CIs cross zero (rejected) |
| RQ7 | Does **fuel-flow** target help vs direct kg? | **Yes** on official & many LOTO folds; heterogeneous |
| RQ8 | Does stacking Direct+Flow improve official RMSE? | **Yes** (ensemble best canonical) |
| RQ9 | Do dynamic mass features close official gap? | **Yes** (largest single step R3, −6.92 kg) |
| RQ10 | Does Level 1 performance transfer to unseen types (LOTO)? | **No** — ~3× MAE inflation |
| RQ11 | Do key findings replicate externally? | **Pilot yes** (DASHlink energy + flow); scale limited |

### 5.3 Hypotheses (for Methods / Results structure)

- **H1 (accepted):** Energy hybrid reduces MAE vs OpenAP hybrid; flight-clustered 95% CI excludes 0.
- **H2 (accepted):** Energy+Weather is the strongest among early feature ablations.
- **H3 (rejected):** Explicit residual learning outperforms direct hybrid.
- **H4 (rejected):** Physics benefits are statistically larger in sparse-telemetry buckets.
- **H5 (partial / suggestive):** Fuel-flow target improves LOTO macro MAE; not robust under type-level paired bootstrap.
- **H6 (accepted under official protocol):** Dynamic mass features reduce combined RMSE vs frozen ensemble baseline.

---

## 6. Contributions (copy into paper bullet list)

1. **Hybrid system** OpenAP + GBDT with energy and weather features for interval fuel prediction under partial observability.
2. **Rigorous evaluation protocol:** flight-level splits; flight-clustered bootstrap; frozen significance rules; official Rank/Final train-only evaluation.
3. **Ablation evidence:** which physics priors help (energy, mass) vs which fail (residual trees/MLP, operational-only, sparsity interaction).
4. **Multi-level generalization analysis:** Level 1 (flights) vs Level 2 (LOTO types) vs Level 3 (transfer mechanisms — partially open).
5. **Official benchmark numbers** and gap-closing ladder (228.25 → 221.33 kg) with honest comparison to published winner (~201 kg).
6. **External audit infrastructure** and pilot replication on DASHlink Project 85.
7. **Open artifacts:** remote loader, featured dataset builder, notebooks, figures/tables, statistical protocol code.

---

## 7. Dataset (write §Dataset / §Data)

### 7.1 Source and access

- Hugging Face: `aerotwin/aero-data`
- Remote load via `data.AeroDataLoader` (`hf://`, Polars); full local download not required
- Components: flightlist, fuel labels, per-flight trajectory parquets, airports

### 7.2 Split counts (official)

| Split | Period | Flights (list) | Usable traj | Fuel labels | Featured intervals |
|-------|--------|---------------:|------------:|------------:|-------------------:|
| **Train** | Apr–Aug 2025 | 11,037 | 10,000 | 131,530 | 119,032 |
| **Rank** | Sep 2025 | 1,888 | 1,888 | 24,289 | 24,158 (1,881 flights) |
| **Final** | Oct 2025 | 2,836 | 2,836 | 37,456 | 37,170 (2,824 flights) |

**Internal Level-1 modeling set (train featured, cleaned):** ~115,995 intervals / 9,976 flights after dropping null physics rows.

### 7.3 Flight-level holdout (Level 1 experiment split)

| Set | Flights | Intervals |
|-----|--------:|----------:|
| Train (80%) | 7,980 | 92,964 |
| Test (20%) | 1,996 | 23,031 |
| Seed | `random_state=42` | 0 flight overlap |

### 7.4 Label and observability facts (must appear in paper)

| Fact | Approx. value |
|------|----------------|
| Median intervals per flight | ~10 (typical 5–25) |
| Median fuel per interval | ~200 kg (heavy right tail) |
| Labeled time share of flight | median **~32%**, mean ~38% |
| Very sparse intervals (`n_traj_pts` < 5) | **~35–46%** |
| Dominant aircraft | A20N, A320 family (~60%+ narrow-body) |
| Types in data | 26 ICAO typecodes (A320 family dominant; A359 large) |
| Missing for ~1,037 train flights | trajectory parquet → excluded from usable set |

### 7.5 Featured dataset construction

Pipeline: `physics/build_featured_dataset.py` + enrichment (`feature_engineering`, `weather_features`, optional mass).

Per interval:

1. Window trajectory points in `[start, end]`
2. OpenAP `FuelFlow.enroute` at representative point (TAS: Mach → CAS → GS)
3. Integrate flow × duration → `physics_fuel_kg`
4. Compute kinematics, phase fractions, quality flags
5. Optional energy / weather / mass features
6. Attach `flight_id` for clustered evaluation

### 7.6 Feature groups (table for paper)

| Group | Examples | Role |
|-------|----------|------|
| Target | `actual_fuel_kg` | Ground truth from ACARS FOB Δ |
| Physics | `physics_fuel_kg`, `method` | OpenAP prior + TAS path |
| Residual (ablation only) | `residual_kg` | actual − physics (**rejected** as architecture) |
| Timing | `duration_s`, start/end fraction of flight | Scale & position |
| Altitude | mean / median / max / std | Cruise/climb context |
| Speed | mean / std / max groundspeed | Energy regime |
| Vertical rate | mean / std | Climb/descent intensity |
| Phase fractions | climb / cruise / descent | Phase mix (vr ±1.5 m/s) |
| Quality | `n_traj_pts`, `has_acars_in_window` | Observability |
| Categorical | aircraft_type, origin, dest, method | Type/route/air-data quality |
| Energy (E2) | PE, KE, SE, ΔE, rates, efficiency | Physics-informed kinematics |
| Weather (E5) | ISA T/p/ρ, density altitude, wind proxies | Atmosphere without METAR |
| Mass (R3) | 21 dynamic mass features | Replace crude MTOW×0.75 |
| Heavy specialist (R1/R2) | OpenAP descriptors + interactions | Wide-body error regime |

**Frozen official feature set (V4):** BASE_NUMERIC + ENERGY + WEATHER + `physics_fuel_kg` + categoricals (~39–47 depending on encoding).  
**R3 path:** base + **21** dynamic mass features.

---

## 8. Methods

### 8.1 Physics baseline (OpenAP)

Implementation: `physics/openap_baseline.py`.

For each interval:

1. Select traj points in window  
2. Phase from median vertical rate (climb > +1.5 m/s, descent < −1.5 m/s, else cruise)  
3. Infer TAS: Mach → CAS → groundspeed fallback  
4. Reference mass: **MTOW × 0.75** (documented limitation; superseded as primary mass signal by R3 features)  
5. `FuelFlow.enroute(ac_type, mass, tas, alt, vs)` at representative point  
6. Integrate over `duration_s` → `physics_fuel_kg`

**OpenAP-only Level-1 holdout:** MAE **668 kg**, RMSE **1,582 kg**, R² **−2.16**.

### 8.2 Modeling paradigms

| Paradigm | Target | How physics enters | Verdict |
|----------|--------|--------------------|---------|
| **Direct hybrid** | `actual_fuel_kg` | Feature `physics_fuel_kg` | **Primary / accepted** |
| **Fuel-flow hybrid** | flow kg/s → × duration | Same features | **Strong** (esp. official MAE/RMSE) |
| **Residual learning** | residual; add to physics | Explicit correction | **Rejected** (~+20 kg MAE) |
| **MLP residual** | residual | Neural correction | **Rejected** |
| **Stacking ensemble** | blend OOF of bases | Ridge meta on train OOF | **Official best** (canonical) |

### 8.3 Learners and frozen hyperparameters

| Model | Key settings |
|-------|----------------|
| XGBoost | n_estimators=300, lr=0.05, max_depth=8 (typical V4) |
| LightGBM | 300 estimators, lr=0.05 |
| CatBoost | iterations=300, lr=0.05 |
| Random Forest | 100 trees, max_depth=15 (early baselines) |
| Linear Regression | scaled numerics (weak baseline) |
| Meta (ensemble) | **Ridge** selected over LGBM by GroupKFold on **train OOF only** |

**Official ensemble bases (6):** {XGB, LGBM, CatBoost} × {Direct kg, Fuel Flow}.

### 8.4 Gap-closing components (post-canonical)

| ID | Component | Combined RMSE | Notes |
|----|-----------|--------------:|-------|
| v1.0 | Frozen V4 ensemble | **228.25** | Canonical paper reference |
| v1.1 | P1E phase-conditional affine + P2 heavy specialist | 227.44 | Keep |
| R1 | OpenAP descriptors in heavy specialist | 226.19 | Keep |
| R2 | Fixed B744/B77L/A306 descriptors | 225.25 | Keep |
| **R3** | **Dynamic mass (21 features) + P1E** | **221.33** | Current best |

**Rejected in gap-closing:** global/class/haul affine; isotonic; cruise residual; simple ensemble reweight (archive R4/R5-style experiments).

### 8.5 Evaluation levels

| Level | Protocol | Held out | What it claims |
|-------|----------|----------|----------------|
| **L1** | Flight-level 80/20 | Unseen flights; types seen | Unseen-flight generalization |
| **Official** | Train fit → Rank + Final | Future months | Challenge / production score |
| **L2 (LOTO)** | Leave-one-type-out | Entire ICAO family | Unseen-type transfer |
| **External** | Train on PRC or external pilot | Other data sources | Cross-dataset replication |

### 8.6 Statistical protocol (frozen — cite `papers/statistical_protocol.md`)

| Constant | Value |
|----------|------:|
| `RANDOM_STATE` | 42 |
| Level-1 `TEST_SIZE` | 0.2 |
| `N_BOOTSTRAP` | 10,000 |
| `ALPHA` | 0.05 |
| CI | 95% (2.5 / 97.5 percentiles) |
| External replication threshold | P(new better) ≥ 0.95 |

**Inference unit must match claim:**

| Claim | Resample unit |
|-------|----------------|
| Unseen-flight generalization | Flight-level bootstrap |
| LOTO macro / type claims | Type-level |
| Paired Flow vs Direct under LOTO | Hierarchical type→flight |
| Wilcoxon on intervals | Supplementary only |

**Decision rule:** significant improvement iff `CI_upper < 0` **and** one-sided bootstrap p < α. If CI crosses 0 → **no significant evidence** (do not claim).

### 8.7 Metrics

| Metric | Use |
|--------|-----|
| **RMSE (kg)** | Primary for **official** Rank/Final/Combined |
| **MAE (kg)** | Primary for many ablations and bootstrap Δ |
| **R²** | Secondary fit quality |
| Bias (signed error) | Diagnostics (R3 reduced bias ~+24 → ~+3.9 kg) |
| Heavy / narrow RMSE | Error stratification |

**Important honesty rule for paper:**  
Internal Level-1 ensemble RMSE (~203 kg) and official Combined (221–228 kg) are **different protocols** — never mix leaderboards. Fuel-Flow and Direct tracks must not be naively ranked as one list without labeling targets.

---

## 9. Results (numbers for tables — do not invent)

### 9.1 Core Level-1 baselines (flight holdout)

| Approach | Model | MAE (kg) | RMSE (kg) | R² |
|----------|-------|---------:|----------:|-----:|
| OpenAP only | — | 667.6 | 1,582 | −2.16 |
| Direct hybrid | XGBoost | ~86–90 | ~224–231 | ~0.93 |
| Direct hybrid | LightGBM | ~92 | ~220 | ~0.94 |
| Direct hybrid | Random Forest | ~86–87 | ~229–233 | ~0.93 |
| Residual trees | XGB/LGBM/RF | ~107–109 | ~293–313 | ~0.88 |
| No physics (kinematics only) | XGB | 89.46 | — | ~0.93 |

Use exact rows from `figures/table_model_comparison_flight_split.csv` when drafting final tables.

### 9.2 Physics-informed ablations (Level 1; selected)

| Experiment | MAE (kg) | ΔMAE vs OpenAP Hybrid | 95% Bootstrap CI | Verdict |
|------------|---------:|----------------------:|------------------|---------|
| OpenAP Hybrid | 86.31 | — | — | baseline |
| Energy Hybrid (E2) | 84.48 | −1.82 | [−2.92, −0.67] | **Significant** |
| Energy + Weather (E6) | **83.76** | **−2.55** | **[−3.58, −1.50]** | **Best early** |
| Weather only (E5) | 86.59 | +0.28 | [−0.40, +1.07] | NS |
| Operational (E3) | 86.76 | +0.46 | [−0.10, +1.01] | NS |
| Residual trees | ~107–108 | ~+21 | excludes 0 | **Worse** |
| MLP residual | 103.7 | +17.4 | [7.84, 34.99] | **Worse** |
| Sparsity × physics | — | — | CIs cross 0 | **Rejected** |

**Strongest Level-1 feature result:** Energy+Weather hybrid XGB **MAE = 83.76 kg** on 1,996 unseen flights.

### 9.3 Fuel-flow vs direct (Level 1 / internal)

- Best fuel-flow single-model MAE ≈ **79.52 kg** (XGB Flow+Energy)  
- Best fuel-flow single-model RMSE ≈ **196.24 kg** (LGBM Flow+Energy)  
- Direct stacking ensemble RMSE ≈ **202.90 kg** (Level-1 holdout) — **not** official score  
- Do not rank Flow and Direct on one unlabeled leaderboard (`figures/LEADERBOARD_AUDIT.md`)

### 9.4 Official PRC Rank + Final (canonical frozen V4)

| Split | MAE (kg) | RMSE (kg) | R² |
|-------|---------:|----------:|----:|
| Rank | 90.89 | **239.18** | 0.904 |
| Final | 87.35 | **220.86** | 0.918 |
| **Combined** | 88.75 | **228.25** | 0.913 |

**Combined RMSE 95% CI (flight bootstrap):** **[207.1, 249.4] kg** — includes values above 201 → **no superiority claim**.

**Official single-model note:** Fuel-Flow bases beat Direct on official RMSE; best single combined ≈ LGBM Flow **230.18**; ensemble **228.25**.

### 9.5 Gap-closing ladder

| Version | Variant | Combined RMSE | Δ vs 228.25 |
|---------|---------|--------------:|------------:|
| v1.0 | Official frozen ensemble | **228.25** | 0 |
| v1.1 | P1E + heavy specialist | 227.44 | −0.81 |
| R1 | + OpenAP heavy descriptors | 226.19 | −2.06 |
| R2 | + descriptor fixes | 225.25 | −3.00 |
| **R3** | **+ dynamic mass (21 feats)** | **221.33** | **−6.92** |

**R3 detail:** Rank RMSE **232.53** · Final **213.73** · Combined bias **+3.85 kg** · Heavy RMSE **416.1** · Narrow **75.0**.  
Remaining gap to published winner ≈ **20 kg**.

### 9.6 LOTO (Level 2) — main messages

- Macro-average MAE rises from ~88 kg (Level 1) to ~**266–283 kg** (global direct LOTO) — **~3× inflation**.
- Fuel-Flow + Energy has **lower LOTO macro MAE** than Direct E+W by ~**17.4 kg**, but:
  - Heterogeneous (≈7 wins / 5 losses across 12 types)
  - Paired type-level and hierarchical bootstrap **CIs cross zero**
  - Strongly influenced by **B77W** fold (exclude B77W → ΔMAE ~−4 kg)
- Physical specification distance correlates with LOTO error in full sample (Pearson *r* ≈ 0.76) but collapses without B77W (*r* ≈ 0.15).
- **Interpretation for paper:** Level 1 metrics overestimate robustness; Flow benefit under type-shift is **suggestive, not confirmatory**.

### 9.7 External pilot (DASHlink Project 85)

- Scale: tails 686/687, **15** airborne flights, **137** intervals (integrated FF_* labels).
- Energy replicates: Direct Base+Energy vs Base ΔMAE ≈ **−4.85 kg**, CI **[−6.87, −2.88]**.
- Flow replicates: Flow+Energy vs matched Direct ΔMAE ≈ **−2.64 kg**, CI **[−4.63, −0.75]**.
- ML ≫ raw OpenAP replicates (physics MAE ~140 kg vs Direct ~21–26 kg on pilot scale).
- **Caveats:** small test set (4 flights); labels not ACARS FOB; absolute MAE **not** comparable to PRC ~84 kg; limited type diversity; default OpenAP type may mismatch FDR fleet.

### 9.8 Error anatomy (discussion fuel)

Largest remaining errors:

- **Heavy aircraft** (A359, B77W, B744, …)
- **Cruise / ultra-long-haul** intervals
- **Long duration** intervals
- Mass / type shift under LOTO

---

## 10. Discussion points (ready-made arguments)

1. **Hybrid > pure physics:** OpenAP alone worse than mean predictor; hybrid recovers operational accuracy.
2. **Not all physics is equal:** point-estimate `physics_fuel_kg` modest; **energy representations** carry independent structure; mass modeling is the largest official gap-close.
3. **Architecture choice matters:** residual = physics + learned correction underperforms end-to-end direct hybrid with physics as feature — trees already absorb the prior.
4. **Fuel-flow normalization** helps scale across interval lengths and often official RMSE; LOTO gains are unstable across types.
5. **Evaluation protocol is part of the science:** flight clustering prevents false discoveries; Rank/Final without leakage is the credible challenge claim.
6. **Three generalization stories:** strong L1, weak L2, pilot L3 external — paper must not oversell “generalization.”
7. **Honest benchmark position:** competitive open system; ~20 kg RMSE remains vs winner; CI does not support beating 201 at canonical freeze.
8. **Negative results are contributions:** residual nets, sparsity hypothesis, operational features, early heuristic mass (distinct from R3).

---

## 11. Limitations (must include)

1. **Unknown true mass** — R3 estimates mass features; still not measured TOW/ZFW.
2. **Partial labels** — not gate-to-gate fuel; median 32% coverage.
3. **Fleet imbalance** — A320-family dominant; rare heavies drive RMSE.
4. **No engine variant / configuration / payload** in public schema.
5. **Weather proxies only** — no METAR/GRIB assimilation.
6. **Official winner pipeline unpublished** — compare to **score**, not reimplemented code.
7. **External validation pilot-scale** — not multi-type full LOTO externally.
8. **Temporal domain** — 2025 European commercial sample; may not transfer to other regions/years without revalidation.
9. **Compute / model class** — GBDTs + simple stacking; limited deep sequence models (transformer residual experimented; not production best).

---

## 12. Conclusions (draft bullets)

1. Hybrid OpenAP + gradient boosting is **necessary and effective** for PRC-style interval fuel prediction.
2. **Energy-state features** provide reliable Level-1 gains; residual architectures do not.
3. Under **official** protocol, a Direct+Flow ensemble reaches **228.25 kg** combined RMSE; **dynamic mass** closes to **221.33 kg**.
4. **LOTO** reveals that flight-level success does not imply type-level transfer.
5. External pilot **replicates** energy and flow findings at small scale.
6. Remaining path to ~201 kg likely needs better **mass**, **heavy/long-haul specialization**, and/or richer air-data or sequence models — without protocol violations.

---

## 13. Future work

- Asymmetric / heavy-focused losses; ultra-long interval specialists  
- Ensemble redesign beyond Ridge meta  
- Sequence / trajectory transformers with strict leakage controls  
- Full multi-fleet external validation (OpenSky + DASHlink at scale)  
- True mass or fuel-state proxies from ops data if available  
- Uncertainty quantification for regulatory use  
- Real-time / streaming interval updates  

---

## 14. Recommended paper structure (IMRaD + extras)

| Section | Content from this file |
|---------|------------------------|
| Title / Authors | §0 |
| Abstract | §1 |
| 1 Introduction | §2–3, contributions §6 |
| 2 Related Work | §4 |
| 3 Dataset | §7 + Figs D* |
| 4 Methods | §8 + Figs M* |
| 5 Experiments & Results | §9 + Figs R* / Tables T* |
| 6 Discussion | §10–11 |
| 7 Conclusion | §12–13 |
| Acknowledgments / Data / Code | §0, §18 |
| Appendix | Protocol constants, full leaderboards, rejected experiments |

**Optional split into 2 papers:**

1. **Dataset characterization** (`papers/dataset_characterization.md`) — JOAS / Scientific Data  
2. **Hybrid modeling + evaluation** (this package) — methods journal  

---

## 15. FIGURE CATALOG — names required for the paper

Use these **canonical paper figure IDs** in the manuscript. Map each to an existing repo artifact when available (path under `figures/` or `audit_results/`). Generate missing ones before camera-ready.

### 15.1 Must-have main-text figures (recommended final set)

| Paper ID | Suggested filename | Caption focus | Source / generate from |
|----------|-------------------|---------------|------------------------|
| **Fig. 1** | `fig_system_architecture.png` | End-to-end AeroTwin pipeline: data → OpenAP → features → models → eval levels | Draw from README architecture; **may need new diagram** |
| **Fig. 2** | `fig_dataset_distribution.png` | Split counts, aircraft mix, route/duration overview | **Exists:** `figures/fig_dataset_distribution.png` |
| **Fig. 3** | `fig_ac_types.png` | Aircraft type distribution (usable train) | **Exists:** `figures/fig_ac_types.png` |
| **Fig. 4** | `fig_fuel_intervals_and_total.png` | Interval fuel distribution + total fuel per flight | **Exists:** `figures/fig_fuel_intervals_and_total.png` |
| **Fig. 5** | `fig_pts_per_interval_cdf.png` | Trajectory density / sparsity (CDF of pts per interval) | **Exists:** `figures/fig_pts_per_interval_cdf.png` (also hist) |
| **Fig. 6** | `fig_profile_example_sparse_dense.png` | Example flight profiles: dense vs sparse labeled windows | **Exists family:** `fig_profile_prc*.png` — pick 2–3 best |
| **Fig. 7** | `fig_physics_vs_actual.png` | OpenAP vs actual (scatter; color by density/phase) | **Exists:** `fig_physics_vs_actual_sample.png`, `fig_physics_vs_actual_demo.png` |
| **Fig. 8** | `fig_actual_vs_predicted.png` | Best hybrid predicted vs actual (Level 1 or official) | **Exists:** `figures/fig_actual_vs_predicted.png` |
| **Fig. 9** | `fig_physics_ablation.png` / `fig_energy_features_bootstrap.png` | Physics & energy ablation + bootstrap ΔMAE | **Exists:** multiple ablation/bootstrap figs |
| **Fig. 10** | `fig_v3_leaderboard.png` or `fig_v4_leaderboard.png` | Level-1 model/feature leaderboard | **Exists** |
| **Fig. 11** | `fig_residual_learning.png` | Residual vs direct (why residual rejected) | **Exists:** `fig_residual_learning.png` (+ bootstrap) |
| **Fig. 12** | `fig_fuel_flow_ablation.png` | Fuel-flow vs direct comparison | **Exists:** `fig_fuel_flow_ablation.png`, `fig_fuel_vs_flow.png`, `fig_prc_comparison.png` |
| **Fig. 13** | `fig_official_leaderboard.png` | Official Rank/Final/Combined leaderboard | **Exists:** `figures/fig_official_leaderboard.png` |
| **Fig. 14** | `fig_gap_closing_rmse.png` | Gap-closing ladder 228.25 → 221.33 | **Exists:** `figures/fig_gap_closing_rmse.png` |
| **Fig. 15** | `fig_error_by_aircraft_type.png` | Error stratified by aircraft type | **Exists** (+ RMSE variant) |
| **Fig. 16** | `fig_error_by_phase.png` | Error by climb/cruise/descent | **Exists** |
| **Fig. 17** | `fig_error_by_haul.png` | Error by haul length | **Exists** |
| **Fig. 18** | `fig_loto_macro_comparison.png` | LOTO macro Direct vs Flow | **Exists** |
| **Fig. 19** | `fig_loto_paired_bootstrap.png` | LOTO significance / paired Δ | **Exists** |
| **Fig. 20** | `fig_loto_distance_vs_mae.png` | Physical distance vs transfer error | **Exists** |
| **Fig. 21** | `fig_shap_catboost_summary.png` | SHAP global explainability | **Exists** (+ top features) |
| **Fig. 22** | `fig_cross_dataset_model_mae.png` | External / cross-dataset comparison | **Exists:** `figures/cross_dataset/` and audit_results |

### 15.2 Strong supplementary / appendix figures

| Paper ID | Suggested filename | Purpose | Repo status |
|----------|-------------------|---------|-------------|
| **Fig. S1** | `fig_pts_per_interval_hist.png` | Sparsity histogram | Exists |
| **Fig. S2** | `fig_fuel_kg_per_interval.png` | Fuel label distribution detail | Exists |
| **Fig. S3** | `fig_audit_phase_pie.png` | Phase composition | Exists |
| **Fig. S4** | `fig_bootstrap_xgb.png` / `lgbm` / `rf` | Bootstrap distributions for significance | Exists |
| **Fig. S5** | `fig_energy_features.png` | Energy feature ablation detail | Exists |
| **Fig. S6** | `fig_mass_ablation.png` | Mass feature ablation (R3 / early) | Exists |
| **Fig. S7** | `fig_r3_dynamic_mass.png` | R3 dynamic mass results | Exists |
| **Fig. S8** | `fig_ensemble.png` / `fig_ensemble_final.png` | Stacking ensemble | Exists |
| **Fig. S9** | `fig_stacking.png` | Stacking detail | Exists |
| **Fig. S10** | `fig_catboost_importance.png` | CatBoost importance | Exists |
| **Fig. S11** | `fig_feature_importance.png` | Generic importance | Exists |
| **Fig. S12** | `fig_shap_catboost_top_features.png` | Top SHAP bars | Exists |
| **Fig. S13** | `fig_prediction_bias_calibration.png` | Bias / calibration (P1E) | Exists |
| **Fig. S14** | `fig_oof_diagnostics.png` | OOF diagnostics | Exists |
| **Fig. S15** | `fig_aircraft_experts.png` | Expert / MoE (if discussed as rejected/kept specialist) | Exists |
| **Fig. S16** | `fig_loto_flow_vs_direct.png` | Per-type Flow vs Direct | Exists |
| **Fig. S17** | `fig_loto_body_shift.png` | Body-class shift | Exists |
| **Fig. S18** | `fig_loto_loo_robustness.png` | Leave-one influence (B77W) | Exists |
| **Fig. S19** | `fig_external_vs_flow.png` | External flow vs direct | Exists |
| **Fig. S20** | `fig_cross_dataset_energy_benefit.png` | Energy benefit across datasets | Exists (`cross_dataset/`) |
| **Fig. S21** | `fig_cross_dataset_flow_advantage.png` | Flow advantage cross-dataset | Exists |
| **Fig. S22** | `fig_cross_dataset_physics_vs_ml.png` | Physics vs ML external | Exists |
| **Fig. S23** | `fig_audit_pred_scatter.png` | External audit scatter | Exists in audit_results |
| **Fig. S24** | `fig_audit_flow_vs_direct_bootstrap.png` | External bootstrap | Exists |
| **Fig. S25** | `fig_optuna_history.png` | HPO (if mentioned) | Exists |
| **Fig. S26** | `fig_transformer_residual` | Sequence residual experiment (if discussed) | table exists; fig may need export |
| **Fig. S27** | `fig_error_vs_density.png` | Error vs traj density | Exists |
| **Fig. S28** | `fig_residual_distributions.png` | Residual error distributions | Exists |
| **Fig. S29** | `fig_verify_predictions.png` | Ensemble verification | Exists |
| **Fig. S30** | `fig_prc_vs_aerotwin.png` | Comparison framing vs challenge baseline | Exists |

### 15.3 Figures that may need to be **created new** for the paper

| Paper ID | Filename to create | Content |
|----------|-------------------|---------|
| **Fig. 1** | `fig_system_architecture.png` | Clean 1-page system diagram (data → physics → ML → eval) |
| **Fig. M1** | `fig_evaluation_protocols.png` | Diagram: Level 1 vs Official vs LOTO vs External |
| **Fig. M2** | `fig_feature_taxonomy.png` | Tree of feature groups (base / energy / weather / mass) |
| **Fig. M3** | `fig_tas_inference_flowchart.png` | Mach → CAS → GS fallback for OpenAP |
| **Fig. R_new1** | `fig_combined_rmse_ci_vs_winner.png` | Point + CI for Combined RMSE vs 201 kg winner line |
| **Fig. R_new2** | `fig_generalization_levels_summary.png` | Bar: L1 MAE vs LOTO macro MAE vs external pilot MAE (with caveat labels) |
| **Fig. R_new3** | `fig_heavy_vs_narrow_rmse.png` | Heavy 416 vs narrow 75 (R3) |
| **Fig. R_new4** | `fig_bias_before_after_mass.png` | Bias drop after R3 mass features |

### 15.4 Complete list of **existing** figure files (repo inventory)

Use this as a master checklist of available PNG assets (names only):

**Core EDA / baseline**

- `fig_ac_types.png`
- `fig_dataset_distribution.png`
- `fig_fuel_intervals_and_total.png`
- `fig_fuel_kg_per_interval.png`
- `fig_pts_per_interval_cdf.png`
- `fig_pts_per_interval_hist.png`
- `fig_physics_vs_actual_demo.png`
- `fig_physics_vs_actual_sample.png`
- `fig_actual_vs_predicted.png`
- `fig_profile_prc770822360.png` … (multiple profile IDs)

**Ablations & significance**

- `fig_physics_ablation.png`
- `fig_energy_features.png`, `fig_energy_features_bootstrap.png`
- `fig_operational_features.png`, `fig_operational_features_bootstrap.png`
- `fig_v3_e5_weather.png`, `fig_v3_e5_weather_bootstrap.png`
- `fig_v3_e6_combined.png`, `fig_v3_e6_combined_bootstrap.png`
- `fig_v3_e7_mlp_bootstrap.png`
- `fig_v3_leaderboard.png`, `fig_v4_leaderboard.png`
- `fig_residual_learning.png`, `fig_residual_learning_bootstrap.png`
- `fig_residual_distributions.png`
- `fig_fuel_flow_ablation.png` (+ per-config bootstrap variants)
- `fig_fuel_flow_bootstrap.png`
- `fig_fuel_vs_flow.png`
- `fig_mass_ablation.png`, `fig_mass_ablation_bootstrap.png`
- `fig_r1_heavy_features.png`, `fig_r3_dynamic_mass.png`
- `fig_vrate_bootstrap.png`
- `fig_vertical_embeddings.png`
- `fig_sparse_bucket_significance.png`
- `fig_bootstrap_rf.png`, `fig_bootstrap_xgb.png`, `fig_bootstrap_lgbm.png`

**Ensemble / official / gap**

- `fig_ensemble.png`, `fig_ensemble_final.png`
- `fig_stacking.png`
- `fig_official_leaderboard.png`
- `fig_gap_closing_rmse.png`
- `fig_prc_comparison.png`, `fig_prc_vs_aerotwin.png`
- `fig_oof_diagnostics.png`
- `fig_prediction_bias_calibration.png`
- `fig_verify_predictions.png`
- `fig_error_rank_vs_final.png`

**Error analysis**

- `fig_error_by_aircraft_type.png`
- `fig_rmse_by_aircraft_type.png`
- `fig_error_by_phase.png`
- `fig_error_by_haul.png`
- `fig_aircraft_errors.png`
- `fig_audit_aircraft_error.png`
- `fig_audit_phase_pie.png`
- `fig_error_vs_density.png`

**Explainability**

- `fig_feature_importance.png`
- `fig_catboost_importance.png`
- `fig_catboost_predictions.png`, `fig_catboost_search.png`
- `fig_shap_catboost_summary.png`
- `fig_shap_catboost_top_features.png`
- `fig_optuna_history.png`

**LOTO / transfer**

- `fig_loto_macro_comparison.png`
- `fig_loto_flow_vs_direct.png`
- `fig_loto_paired_bootstrap.png`
- `fig_loto_paired_delta_per_type.png`
- `fig_loto_distance_vs_mae.png`
- `fig_loto_distance_vs_inflation.png`
- `fig_loto_body_shift.png`
- `fig_loto_loo_robustness.png`
- `fig_loto_residual_vs_direct_matched.png`
- `fig_aircraft_experts.png`

**Cross-dataset / external**

- `cross_dataset/fig_cross_dataset_energy_benefit.png`
- `cross_dataset/fig_cross_dataset_flow_advantage.png`
- `cross_dataset/fig_cross_dataset_model_mae.png`
- `cross_dataset/fig_cross_dataset_physics_vs_ml.png`
- `fig_external_vs_flow.png`
- audit pilots: `fig_audit_pilot_mae.png`, `fig_audit_pred_scatter.png`, `fig_audit_*_bootstrap.png`

---

## 16. TABLE CATALOG — names required for the paper

### 16.1 Main-text tables

| Paper ID | Suggested CSV / content | Columns to include |
|----------|-------------------------|--------------------|
| **Table 1** | `table_dataset_audit.csv` / `table_dataset_summary.csv` | Split, flights, intervals, period |
| **Table 2** | `table_aircraft_types.csv` / distribution | Type, flight count, share |
| **Table 3** | Feature group taxonomy | Group, features, rationale |
| **Table 4** | OpenAP-only vs hybrid baselines | MAE, RMSE, R² |
| **Table 5** | Physics-informed ablation + bootstrap CIs | MAE, ΔMAE, CI, verdict |
| **Table 6** | Architecture comparison (direct / residual / flow / ensemble) | metrics + verdict |
| **Table 7** | Official Rank / Final / Combined leaderboard | from `table_official_leaderboard.csv` |
| **Table 8** | Gap-closing ladder | version, RMSE, Δ |
| **Table 9** | Error by aircraft type / phase / haul | stratified RMSE/MAE |
| **Table 10** | LOTO summary Direct vs Flow | macro MAE, wins/losses, CI |
| **Table 11** | External pilot replication | finding, ΔMAE, CI, replicated? |
| **Table 12** | Statistical protocol constants | seed, B, α, units |

### 16.2 Key existing table files

- `table_dataset_audit.csv`, `table_dataset_summary.csv`, `table_split_statistics.csv`
- `table_aircraft_types.csv`, `table_aircraft_distribution.csv`, `table_route_distribution.csv`
- `table_model_comparison.csv`, `table_model_comparison_flight_split.csv`
- `table_physics_ablation.csv`, `table_energy_results.csv`, `table_operational_results.csv`
- `table_v3_leaderboard.csv`, `table_significance_v3_all.csv` (+ per-experiment significance tables)
- `table_fuel_flow.csv`, `table_fuel_flow_ablation.csv`, `table_flow_vs_prc.csv`
- `table_mass_ablation.csv`, `table_rmse_R1*.csv`, `table_rmse_R3*.csv`
- `table_official_leaderboard.csv`, `table_official_final_results.csv`, `table_official_rank_results.csv`
- `table_gap_closing_leaderboard.csv`, `table_gap_accepted_changes.csv`
- `table_error_by_aircraft_type*.csv`, `table_error_by_phase.csv`, `table_error_by_haul.csv`
- `table_leave_one_type_out.csv`, `table_loto_*.csv` (many)
- `table_shap_catboost*.csv`
- `table_cross_dataset_replication.csv`, `table_external_*.csv`
- `table_audit_pilot_*.csv` (in audit_results)

---

## 17. Claims policy (what you may / may not write)

### Allowed (supported)

- Hybrid ML ≫ raw OpenAP on PRC Level 1 and official.
- Energy (+ with weather) significantly improves Level-1 MAE under flight-clustered bootstrap.
- Residual learning underperforms direct hybrid under same protocol.
- Official combined RMSE **228.25 kg** (canonical); **221.33 kg** after R3 gap-closing.
- LOTO shows large degradation vs Level 1; standard holdout overestimates type robustness.
- DASHlink pilot replicates energy and fuel-flow directional findings (with scale caveats).

### Not allowed without new evidence

- “Beats the PRC winner” / “state-of-the-art overall” (CI does not support; point estimate worse).
- “Physics always helps most when data is sparse” (rejected).
- “Fuel-flow universally better under LOTO” (not significant under paired type inference).
- Treating Level-1 RMSE ~203 as official score.
- Claiming full multi-dataset generalization beyond pilot-scale external results.

---

## 18. Reproducibility block (paste into paper)

**Environment:** Python 3.11+, see `requirements.txt`.  

**Data:**

```python
from data import AeroDataLoader
loader = AeroDataLoader()
flightlist = loader.get_flightlist("train")
fuel = loader.get_fuel_labels("train")
```

**Featured dataset:**

```bash
PYTHONPATH=. python physics/build_featured_dataset.py
```

**Key experiments:**

```bash
PYTHONPATH=. python notebooks/05_baseline_modeling.py
PYTHONPATH=. python notebooks/06_physics_ablation.py
PYTHONPATH=. python notebooks/07_significance_testing.py
PYTHONPATH=. python notebooks/09_physics_features_v3.py
PYTHONPATH=. python notebooks/12_verify_ensemble.py
PYTHONPATH=. python notebooks/14_shap_explainability.py
PYTHONPATH=. python notebooks/15_leave_one_type_out.py
PYTHONPATH=. python notebooks/17_official_prc_evaluation.py --skip-build
PYTHONPATH=. python notebooks/18_official_error_analysis.py
PYTHONPATH=. python notebooks/19_gap_closing_campaign.py
PYTHONPATH=. python notebooks/25_r3_dynamic_mass.py
PYTHONPATH=. python notebooks/26_r3_ensemble_mass.py
```

**Artifacts:** `figures/*.png`, `figures/table_*.csv`, `figures/*summary.json`  
**Protocol code:** `physics/statistical_protocol.py`, `physics/eval_framework.py`  
**Docs:** `official_prc_benchmark_report.md`, `CURRENT_MODEL_SUMMARY.md`, `PROJECT_STATUS_REPORT.md`, `papers/*`

---

## 19. References to seed the bibliography

1. Sun, J., Spinielli, E., & Strohmeier, M. (2026). *Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge*. Journal of Open Aviation Science. https://doi.org/10.59490/joas.2026.8750  
2. OpenAP documentation / Sun et al. — open aircraft performance model.  
3. EUROCONTROL PRC data challenge materials.  
4. Hugging Face dataset card: `aerotwin/aero-data`.  
5. Gradient boosting references as needed (XGBoost, LightGBM, CatBoost papers).  
6. SHAP (Lundberg & Lee) for explainability section.  
7. Domain literature on aviation fuel estimation, ADS-B analytics, physics-informed ML (expand during lit review).  
8. Bootstrap / clustered inference references (e.g., field standards for dependent samples).

---

## 20. Drafting checklist

### Content completeness

- [ ] Abstract matches allowed claims only  
- [ ] Introduction ends with numbered contributions  
- [ ] Related work positions hybrid gap clearly  
- [ ] Dataset section includes partial observability + sparsity numbers  
- [ ] Methods describe OpenAP TAS cascade and mass assumption  
- [ ] Evaluation levels L1 / Official / LOTO / External defined before results  
- [ ] Statistical protocol (flight-clustered bootstrap) described once  
- [ ] Negative results reported (residual, sparsity, operational, early mass)  
- [ ] Official numbers + CI + winner comparison honest  
- [ ] R3 mass presented as gap-closing, distinct from rejected heuristic mass  
- [ ] LOTO caveats (B77W influence, non-significant paired Flow benefit)  
- [ ] External pilot scale caveats  
- [ ] Limitations section complete  
- [ ] Code + data availability statements  

### Figures & tables

- [ ] All **Fig. 1–22** selected or generated  
- [ ] Architecture + protocol diagrams created if missing  
- [ ] Winner comparison CI figure included  
- [ ] Tables 1–12 filled from CSVs (no manual number drift)  
- [ ] Captions state protocol (Level 1 vs Official vs LOTO)  
- [ ] Colorblind-friendly palettes; units (kg) on every axis  

### Consistency pass

- [ ] Single “best official” number used consistently (canonical 228.25 vs current 221.33 labeled)  
- [ ] No mixing of Flow vs Direct leaderboards without labels  
- [ ] Seeds, splits, B=10,000 stated  
- [ ] All metrics say **kg**  

### Submission polish

- [ ] Authors / affiliations / ethics / funding  
- [ ] Anonymization if double-blind  
- [ ] Cite challenge paper and OpenAP  
- [ ] Supplementary zip of tables/figures or OSF/Zenodo deposit  

---

## 21. Suggested narrative thread (story arc for writing)

1. **Hook:** Fuel burn is critical; operational data is messy and partially observed.  
2. **Shock:** Pure physics fails (negative R²).  
3. **Hope:** Hybrid trees work on unseen flights (~85 kg MAE).  
4. **Science:** Which physics features actually help? → energy yes; residual no; sparsity myth no.  
5. **Credibility:** Official Rank+Final without leakage; still ~27 kg (then ~20 kg after R3) behind winner.  
6. **Humility:** LOTO destroys the comfortable Level-1 story.  
7. **External:** Small pilot still replicates energy/flow.  
8. **Close:** Open hybrid baselines + rigorous multi-level evaluation as the contribution.

---

## 22. Quick reference — headline numbers card

| Claim | Number |
|-------|-------:|
| Train usable flights | 10,000 |
| Train featured intervals | 119,032 |
| Level-1 test flights / intervals | 1,996 / 23,031 |
| OpenAP-only MAE | ~668 kg |
| Best early Energy+Weather MAE (L1) | **83.76 kg** |
| Level-1 Direct stack RMSE (internal) | ~202.9 kg |
| Official Combined RMSE (canonical) | **228.25 kg** |
| Official Combined 95% CI | [207.1, 249.4] |
| Official Combined after R3 | **221.33 kg** |
| Published winner Combined RMSE | ≈ **201 kg** |
| Gap after R3 | ≈ **20 kg** |
| LOTO macro MAE (order of magnitude) | ~266–283 kg |
| Label coverage of flight time (median) | ~32% |
| Very sparse intervals | ~35–46% |

---

## 23. Related internal documents (do not lose)

| Document | Role |
|----------|------|
| `README.md` | Public overview |
| `CURRENT_MODEL_SUMMARY.md` | Live best model |
| `PROJECT_STATUS_REPORT.md` | Full project narrative + checklists |
| `official_prc_benchmark_report.md` | Official evaluation write-up |
| `official_gap_closing_report.md` | Gap-closing campaign |
| `papers/hybrid_model_summary.md` | Hybrid paper seed |
| `papers/dataset_characterization.md` | Dataset paper outline |
| `papers/statistical_protocol.md` | Frozen stats rules |
| `papers/shift_aware_routing.md` | Shift/routing notes |
| `HOW_TO_RUN_AUDIT.md` | External audit how-to |
| `figures/LEADERBOARD_AUDIT.md` | Leaderboard hygiene |
| `docs/*` | Parity / gap attribution / verification |

---

*End of research writing package. Expand sections into venue-formatted prose; prefer numbers from `figures/table_*.csv` and summary JSON over retyped values.*
