# Two-Domain Research Writing Package  
## Structural Strategies for Cumulative Energy Prediction under Partial Observation and Entity Shift

**Project framing:** Multi-domain empirical study (AeroTwin aircraft + VED vehicles)  
**Primary domain (fully frozen):** AeroTwin / EUROCONTROL PRC 2025  
**Second domain (numbers locked):** Vehicle Energy Dataset (VED) / VEE project  
**Companion single-domain package:** `research.md` (AeroTwin-only; all numbers preserved here)  
**VED status source:** `../vehicle/VEE/project_status_report.md` (2026-07-25) + `../vehicle/VEE/paper/results.md`  
**Cross-domain diagnostic in-repo:** `docs/VED_PHENOMENA_REPLICATION.md`  
**Last synced:** 2026-07-27 (VED placeholders filled from status report)  
**Tone:** Workshop / specialized-venue empirical paper — careful, non-overclaiming  

This document is the **two-case-study writing package**. Expand into formal prose; do **not** invent metrics beyond what is listed. **AeroTwin numbers are frozen** — identical to `research.md` / official reports. **VED numbers** are locked from the VEE status report (units: liters of fuel, `cum_fuel_l`, unless noted).

---

## Placeholder fill log (this revision)

| Placeholder / gap | Replacement (source) |
|-------------------|----------------------|
| `[VED: table_physics_vs_residual]` | residual_ml − physics_fb: ΔMAE **−0.0688** L; 95% CI **[−0.093, −0.047]**; significant (status §6.3) |
| `[VED: table_physics_only]` | physics_fb LOEO: MAE **0.1270**, RMSE **0.196**, R² **0.761** |
| Table V1 features / train-test | Primary cohort ICE+HEV; LOEO `engine_config` 20 folds; 23,464 test trips |
| Table V1 physics model | Force-balance residual anchor `phys_fb_cum_fuel_l` (not MAF) |
| Table V2 IID MAE | direct **0.0511**, residual **0.0487**, physics_fb **0.1441** (status §6.2) |
| Table V2 IID RMSE (where reported) | From B/V ladder: direct **0.114**, residual **0.089**, rate **0.085** (status §6.6) |
| Table V3 LOEO (primary) | Full primary leaderboard §H.4.0 (user + status §1 / §6.1) |
| Table V3 vehicle holdout MAE | direct **0.0610**, residual **0.0596**, physics **0.1500** |
| Table V4 entity ladder | IID / Vehicle / LOEO family / LOEO config MAE table §6.2 |
| Residual vs Direct MAE claim | ΔMAE **+0.0004**; CI **[−0.002, +0.003]**; **not significant** |
| Rate vs Direct | rate_integrate MAE **0.0618** > direct **0.0578** |
| IID gap | direct **+13.0%** (1.13×); residual **+19.4%** (1.19×) |
| Bias–variance hypothesis | **Rejected** (status §6.6) |
| Domain-specific vehicle features | duration, distance, n_samples, phys_fb_cum_fuel_l, powertrain, displacement (perm. importance) |
| VED path placeholder | `../vehicle/VEE/` |
| “energy” target ambiguity | Primary target **`cum_fuel_l` (L)** on ICE+HEV cohort |

No AeroTwin metrics were modified.

---

# Part A — Paper structure outline (start here)

## Recommended manuscript skeleton

```text
Title / Authors / Abstract
1. Introduction
   1.1 Motivation: cumulative energy under partial observation
   1.2 Structural modeling strategies (Direct / Residual / Rate)
   1.3 Why evaluation protocol and entity shift matter
   1.4 Contributions (honest, multi-domain)
2. Shared problem formulation
   2.1 Cumulative prediction task
   2.2 Three structural strategies
   2.3 Evaluation axes: random / entity / temporal
   2.4 Statistical reporting principles
3. Related work
   (physics–ML hybrids; residual correction; vehicle energy; aviation fuel; domain shift)
4. Case Study 1 — AeroTwin (commercial aircraft fuel intervals)
   4.1 Domain, data, observability
   4.2 Physics prior (OpenAP) and features
   4.3 Methods and protocols
   4.4 Results (L1, ablations, official, LOTO, external pilot)
   4.5 Domain-local conclusions
5. Case Study 2 — VED (vehicle trip fuel; ICE/HEV cohort)
   5.1 Domain, data, scale
   5.2 Physics prior (force-balance) and features
   5.3 Methods and entity protocols (IID / vehicle holdout / LOEO)
   5.4 Results (Direct vs Residual vs Rate; entity difficulty; B/V)
   5.5 Domain-local conclusions
6. Cross-domain analysis (modest)
   6.1 Shared structural comparison table
   6.2 What transfers (and what does not)
   6.3 Protocol lessons
7. Limitations
8. Conclusion
Appendix A — AeroTwin full tables / protocol constants
Appendix B — VED full tables / protocol constants
Appendix C — Artifact index
```

### Design principles for the structure

| Principle | Implication for writing |
|-----------|-------------------------|
| **Shared question first** | Intro frames *strategies*, not “we solve aviation and cars with one model.” |
| **Domain-local rigor** | Each case study stands alone with its own metrics and protocol. |
| **Modest cross-domain section** | One comparison section; no forced universal ranking. |
| **Honesty over narrative neatness** | VED residual beats physics and wins RMSE, but **not** MAE vs Direct; AeroTwin residual loses; Rate is not a universal fix. |
| **Preserve AeroTwin freeze** | Official Combined **221.33 kg** (R3) and canonical **228.25 kg** unchanged. |

---

# Part B — Metadata and titles

## B.0 Paper metadata

| Field | Value |
|-------|--------|
| **Genre** | Two-case-study empirical paper |
| **Primary contribution type** | Structural evaluation + protocol-aware findings (not SOTA claim) |
| **Authors** | _[to fill]_ |
| **Affiliation** | _[to fill]_ |
| **Suggested venues** | Workshop on physics-informed ML / transportation ML; specialized journals open to multi-domain empirical studies; JOAS (if aviation-primary with VED as secondary); TR-C / IEEE ITS (if transportation framing) |
| **Keywords** | cumulative energy prediction; hybrid physics–ML; residual learning; rate-then-integrate; entity shift; leave-one-entity-out; aircraft fuel burn; vehicle fuel; partial observability |
| **AeroTwin code** | https://github.com/ArunArya-01/ZeroPing |
| **AeroTwin data** | https://huggingface.co/datasets/aerotwin/aero-data |
| **VED data** | Vehicle Energy Dataset (Oh et al. / public VED release — cite original) |
| **VED code / report** | `major project/vehicle/VEE/` · `project_status_report.md` |
| **VED study status** | Primary LOEO + generalization ladder + bias–variance **locked** (2026-07-25) |

### Title options (new framing)

| ID | Title |
|----|--------|
| **T1 (recommended)** | Structural Strategies for Cumulative Energy Prediction under Partial Observation and Entity Shift: Aircraft Fuel and Vehicle Energy Case Studies |
| **T2** | Direct, Residual, or Rate-Then-Integrate? An Empirical Two-Domain Study of Physics-Informed Cumulative Prediction |
| **T3** | When Physics-Informed Structure Helps—and When It Does Not: Lessons from Aircraft Fuel Burn and Vehicle Energy Estimation |
| **T4** | Evaluation Protocol Matters: Hybrid Modeling of Cumulative Fuel under Flight-Level and Engine-Config Shift |
| **T5 (aviation-first)** | AeroTwin and Beyond: Hybrid Interval Fuel Prediction with a Comparative Vehicle Energy Case Study |
| **Avoid** | Titles that claim “universal residual learning,” “cross-domain generalization of architecture X,” or “SOTA on both domains” |

---

# Part C — Updated abstract (both domains, honest)

## C.1 Abstract draft (≈220–280 words; trim for venue)

Predicting **cumulative fuel use**—aircraft fuel on ACARS-labeled intervals, or vehicle trip fuel—from partially observed kinematics is a recurring problem in transportation analytics. Practitioners often choose among three **structural strategies**: (i) **Direct** prediction of the cumulative quantity, (ii) **Residual** correction of a physics baseline, and (iii) **Rate-then-integrate** (predict instantaneous consumption, then multiply by duration). Whether these choices transfer across domains is unclear, and **entity-level** evaluation (unseen aircraft types or engine configurations) can change rankings relative to random holdout.

We study this question in **two completed case studies** under domain-appropriate physics priors and protocols.

**Case Study 1 (AeroTwin)** uses the EUROCONTROL PRC 2025 fused ADS-B/ACARS data (~10,000 training flights; 119,032 fuel intervals). Pure OpenAP physics is unusable (flight-holdout MAE ≈ 668 kg; R² ≈ −2.16). Direct hybrid gradient boosting with **energy-state** features yields bootstrap-supported gains; **residual learning underperforms** direct hybrid. Under the official Rank+Final protocol, a Direct+Fuel-Flow ensemble reaches combined RMSE **228.25 kg**, improved to **221.33 kg** with dynamic mass features—still short of the published winner (≈201 kg; no superiority claim). Leave-one-type-out inflates error by roughly **3×**, showing that flight-level metrics overestimate robustness under type shift.

**Case Study 2 (VED)** analyzes **32,536 trips** from **384 vehicles** (primary ICE+HEV cohort: **27,100 trips / 341 vehicles**). Under leave-one-`engine_config`-out (20 folds; **23,464** test trips), all learned models significantly beat fair force-balance physics (physics MAE **0.127 L** vs Direct **0.0578 L**). Residual ML does **not** significantly beat Direct on MAE (0.0582 vs 0.0578; entity-bootstrap CI includes 0) but improves RMSE (**0.095** vs **0.116**) and R² (**0.944** vs **0.916**). Rate-then-integrate does **not** beat Direct on MAE (**0.0618**). IID underestimates LOEO error by **~13–19%** for ML models. A bias–variance hypothesis that structural models mainly reduce variance under shift is **rejected**.

**Cross-domain takeaway:** structural preferences are **domain- and protocol-dependent**. We find **no strong evidence** that residual learning or rate-then-integrate transfer as default recipes from aircraft to cars. Report Direct/Residual/Rate under **matched models**, **both MAE and RMSE**, and entity-aware splits.

**Word target:** 150–250 for short venues; keep the three-strategy framing and the “no clean transfer” sentence in all versions.

## C.2 Elevator summary (poster / intro)

| Item | Statement |
|------|-----------|
| **Shared problem** | Predict cumulative fuel from partial kinematics + imperfect physics priors under entity shift. |
| **Three strategies** | Direct · Residual (physics + correction) · Rate-then-integrate. |
| **AeroTwin headline** | Hybrid works; energy/mass help; residual loses; official **221.33 kg**; LOTO ~3× harder. |
| **VED headline** | LOEO Direct MAE **0.0578 L** (best MAE); residual RMSE **0.095** / residual_rate **0.093** (best RMSE); residual ≰ Direct on MAE; Rate MAE **0.0618**; physics_fb **0.127**; IID gap **13–19%**. |
| **Cross-domain claim** | Results are **domain-dependent**; protocol and metric choice matter; **no universal structural winner**. |
| **What we do *not* claim** | Cross-domain architecture transfer; residual as general best practice; beating PRC winner; “structural models only reduce variance under shift.” |

---

# Part D — Shared problem formulation

## D.1 Cumulative prediction task (domain-agnostic)

For labeled segment \(i\) belonging to entity \(e\) (flight, aircraft type, engine config, vehicle, …):

\[
y_i = \text{cumulative fuel over segment } i
\]

Predict \(\hat{y}_i\) from kinematics \(\mathbf{x}_i\), metadata \(\mathbf{m}_i\), and optional physics baseline \(\hat{y}_i^{\text{phys}}\):

\[
\hat{y}_i = f\!\big(\mathbf{x}_i,\, \mathbf{m}_i,\, \hat{y}_i^{\text{phys}}\big)
\]

**Partial observation:** labels cover only part of operational time or depend on sparse sensors; physics inputs (mass, resistance coefficients, air data) are incomplete or assumed.

**Domain units:** AeroTwin → kg; VED → liters (`cum_fuel_l`). Never plot on one axis without normalization.

## D.2 Three structural strategies (shared vocabulary)

| Strategy | Predict | Recover \(y\) | Physics role |
|----------|---------|---------------|--------------|
| **Direct** | \(y_i\) | identity | Feature and/or ignored |
| **Residual** | \(r_i = y_i - \hat{y}_i^{\text{phys}}\) | \(\hat{y}_i = \hat{y}_i^{\text{phys}} + \hat{r}_i\) | Explicit baseline; model corrects |
| **Rate-then-integrate** | rate \(\rho_i\) (e.g. kg/s or L/s) | \(\hat{y}_i = \hat{\rho}_i \cdot \Delta t_i\) | Optional; normalizes duration scale |

**AeroTwin names:** Direct kg · Residual kg · **Fuel-Flow** (rate).  
**VED names:** `direct_ml` · `residual_ml` (FB + LGBM) · `rate_integrate` · `residual_rate` (residual on rate, then integrate).

Use this shared vocabulary in §6 so readers can compare without equating implementations.

## D.3 Evaluation axes (shared language)

| Axis | AeroTwin analogue | VED analogue | Claim type |
|------|-------------------|--------------|------------|
| **Random / quasi-IID segment groups** | Flight-level 80/20 (types still seen) | Random trip IID | Unseen segments, entities partially seen |
| **Entity holdout** | Leave-one-**type**-out (LOTO) | LOEO on **`engine_config`** (primary); also vehicle, engine_family | Unseen entity generalization |
| **Temporal** | Official Rank/Final months | Not primary VED axis | Time shift (not entity) |
| **External data** | DASHlink pilot | Not in locked VED package | Dataset shift |

**Rule:** Never treat entity-holdout metrics as interchangeable with random-holdout metrics.

## D.4 Shared challenges (intro list)

1. Cumulative targets with heterogeneous segment durations  
2. Imperfect physics baselines (wrong mass, coefficients, or operating regime)  
3. Partial / sparse observability  
4. Entity heterogeneity (fleet types or engine configs / vehicles)  
5. Metric choice (MAE vs RMSE) can flip rankings  
6. Leakage if segments from the same entity cross train/test  

---

# Part E — Updated research questions

## E.1 Primary (two-domain)

> **RQ-Primary.** For cumulative fuel prediction under partial observation, how do **Direct**, **Residual**, and **Rate-then-integrate** strategies behave under **random** versus **entity-level** evaluation—and do structural preferences **transfer** between commercial aircraft fuel intervals (AeroTwin) and vehicle trip fuel (VED)?

**Expected answer style:** preferences are **domain- and protocol-dependent**; residual and rate are **not** reliable universal defaults. On VED, residual is competitive and wins second-moment metrics but not primary MAE.

## E.2 Shared structural questions

| ID | Question | AeroTwin status | VED status (locked) |
|----|----------|-----------------|---------------------|
| **SQ1** | Does residual improve over **pure physics**? | Yes (hybrid ≫ OpenAP), but residual **architecture** loses to Direct | **Yes** — residual_ml vs physics_fb ΔMAE **−0.0688** L, CI excludes 0 |
| **SQ2** | Does residual beat **strong Direct**? | **No** (Level-1 and matched LOTO) | **No on MAE** (0.0582 vs 0.0578; CI includes 0); **Yes on RMSE/R²** |
| **SQ3** | Does rate-then-integrate help vs Direct? | Often yes (official; many LOTO folds) but LOTO significance fragile | **No on MAE** (0.0618 > 0.0578); residual_rate best RMSE (**0.093**) |
| **SQ4** | Is entity-level evaluation harder than random/IID? | **Yes** (~3× MAE under LOTO) | **Yes** — IID→LOEO config gap **+13%** (Direct) / **+19%** (Residual) |
| **SQ5** | Is difficulty strictly monotonic in entity coarseness? | **Not testable** (only one pure entity rung: type LOTO) | **No** — e.g. LOEO family MAE > LOEO config; vehicle vs config not monotone |
| **SQ6** | Do the same structural choices transfer aircraft → cars? | — | **No strong evidence of clean transfer** |

## E.3 Domain-local questions (keep; do not drop AeroTwin science)

### AeroTwin-only (from `research.md`; statuses frozen)

| ID | Question | Status |
|----|----------|--------|
| AQ1 | Raw OpenAP as feature when kinematics exist? | Modest yes (model-dependent) |
| AQ2 | Energy-state features vs OpenAP hybrid? | **Significant yes** |
| AQ3 | Weather alone? | No |
| AQ4 | Operational descriptors? | No |
| AQ5 | Residual better than Direct? | **No** |
| AQ6 | Physics helps most when sparse? | **Rejected** |
| AQ7 | Fuel-flow vs Direct? | Strong on official; LOTO suggestive |
| AQ8 | Stacking Direct+Flow? | Yes (canonical ensemble) |
| AQ9 | Dynamic mass closes official gap? | **Yes** (−6.92 kg Combined) |
| AQ10 | Level-1 → LOTO transfer? | **No** (~3×) |
| AQ11 | External pilot replication? | Pilot yes (energy, flow) |

### VED-only (locked from status report)

| ID | Question | Status |
|----|----------|--------|
| VQ1 | Residual vs pure force-balance physics? | **Yes** — significant (ΔMAE −0.0688 L) |
| VQ2 | Residual vs strong Direct (**MAE**)? | **No** — not significant (ΔMAE +0.0004; CI crosses 0) |
| VQ3 | Residual vs strong Direct (**RMSE / R²**)? | **Yes** — residual_ml RMSE **0.095** / R² **0.944** vs Direct **0.116** / **0.916**; residual_rate RMSE **0.093** / R² **0.946** |
| VQ4 | Rate-then-integrate vs Direct on MAE? | **No** — 0.0618 > 0.0578 |
| VQ5 | Vehicle / LOEO vs IID difficulty? | Entity-level **harder** for ML (+13–19% MAE IID→config LOEO) |
| VQ6 | Monotonic difficulty across entity granularities? | **Not strictly** (family harder than config; vehicle intermediate) |
| VQ7 | Structural models mainly reduce **variance** under shift? | **Rejected** (bias–variance ladder) |

## E.4 Hypotheses (paper-facing)

| ID | Hypothesis | Decision |
|----|------------|----------|
| H-Cross1 | Residual is the best default for cumulative energy across domains | **Rejected** |
| H-Cross2 | Rate-then-integrate is a reliable duration normalization fix across domains | **Rejected** |
| H-Cross3 | Entity holdout is harder than random/IID-style splits in both domains | **Supported** (both domains) |
| H-Cross4 | MAE and RMSE always agree on structural ranking | **Rejected** (VED: Direct best MAE, residual/residual_rate best RMSE) |
| H-AT-Energy | Energy features help AeroTwin Level-1 | **Accepted** (bootstrap) |
| H-AT-Res | Residual beats Direct on AeroTwin | **Rejected** (incl. matched CatBoost LOTO) |
| H-AT-Mass | Dynamic mass improves official Combined RMSE | **Accepted** (221.33) |
| H-VED-Phys | Residual helps over pure physics on VED | **Accepted** |
| H-VED-Dir-MAE | Residual consistently beats Direct on VED MAE | **Rejected** |
| H-VED-Dir-RMSE | Residual improves RMSE/R² vs Direct under LOEO | **Accepted** (point estimates; optional RMSE bootstrap still open in VEE backlog) |
| H-VED-BV | Structural models mainly reduce variance under entity shift | **Rejected** |

---

# Part F — Updated contributions

Write contributions as **empirical + methodological**, not “we win both leaderboards.”

1. **Unified structural framing** of cumulative fuel prediction via Direct / Residual / Rate-then-integrate, applied to two real operational domains with domain-appropriate physics priors (OpenAP; force-balance vehicle model).

2. **Case Study 1 (AeroTwin):** large-scale hybrid aircraft fuel modeling with flight-level and official Rank+Final evaluation; energy-state and dynamic-mass gains; residual architectures rejected; official Combined RMSE **228.25 → 221.33 kg**; LOTO ~3× degradation; pilot external audit.

3. **Case Study 2 (VED):** LOEO on `engine_config` (20 folds, 23,464 test trips) over ICE+HEV cohort (27,100 trips / 341 vehicles) within full VED (32,536 / 384). Learned models ~**2×** better than fair FB physics on MAE; residual **not** significant vs Direct on MAE but better on RMSE/R²; rate_integrate does not win MAE; IID underestimates LOEO by **13–19%**; bias–variance “variance-only under shift” hypothesis **rejected**.

4. **Cross-domain analysis (modest):** side-by-side comparison showing **domain-dependent** structural outcomes and **no strong evidence** of clean architecture transfer from aircraft to cars—while **metric disagreement** (MAE vs RMSE) is itself a transferable reporting lesson.

5. **Protocol contribution:** matched-model comparisons, entity-aware splits, multi-metric reporting, and entity-clustered bootstrap (VED: 2000 entity reps; AeroTwin: 10k flight-clustered where used).

6. **Negative and mixed results as first-class findings:** residual and rate are not universal; AeroTwin residual fails under type shift; VED residual is MAE-neutral but RMSE-helpful; MAF physics circularity documented as validity threat.

7. **Reproducible artifacts:** AeroTwin (`ZeroPing`); VED/VEE (`vehicle/VEE/`, `project_status_report.md`, `results/loeo_engine_config_default/`).

---

# Part G — Case Study 1: AeroTwin  
## (full content condensed; metrics frozen — see `research.md` for expanded tables)

> **Do not change any number in this section.** Source of truth: `research.md`, `CURRENT_MODEL_SUMMARY.md`, `official_prc_benchmark_report.md`.

### G.1 Domain snapshot

| Item | Value |
|------|--------|
| Task | Interval fuel burn (kg) from ADS-B + ACARS |
| Data | `aerotwin/aero-data` (PRC 2025) |
| Train usable flights / intervals | 10,000 / 119,032 |
| Physics | OpenAP `FuelFlow.enroute` |
| Partial observation | Median ~32% of flight time labeled; ~35–46% very sparse intervals |
| Official primary metric | Combined Rank+Final **RMSE (kg)** |

### G.2 Structural strategies (AeroTwin mapping)

| Strategy | Implementation | Main verdict |
|----------|----------------|--------------|
| Direct | Predict `actual_fuel_kg` (+ optional `physics_fuel_kg` feature) | **Primary / best architecture class** |
| Residual | Predict residual; add OpenAP | **Rejected** vs Direct (L1 and matched LOTO) |
| Rate | Fuel-flow kg/s × duration | **Strong** on official; LOTO mixed/suggestive |

### G.3 Headline results (frozen)

#### Physics and Level-1

| Result | Number |
|--------|-------:|
| OpenAP-only MAE / RMSE / R² | ~668 / 1,582 / −2.16 |
| Direct hybrid MAE (order) | ~84–88 kg |
| Energy hybrid ΔMAE (E2) | −1.82; CI [−2.92, −0.67] |
| Energy+Weather MAE (best early) | **83.76 kg** |
| Residual trees MAE | ~107–109 kg (worse) |
| Matched CatBoost Residual vs Direct (flight) | 94.39 vs **88.07** MAE |

#### Official Rank+Final

| Split | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Rank | 90.89 | **239.18** | 0.904 |
| Final | 87.35 | **220.86** | 0.918 |
| Combined (canonical ensemble) | 88.75 | **228.25** | 0.913 |
| Combined 95% CI | — | **[207.1, 249.4]** | — |
| **R3 current best Combined** | — | **221.33** | — |
| Published winner Combined | — | ≈ **201** | — |

Gap-closing ladder: 228.25 → 227.44 → 226.19 → 225.25 → **221.33** (−6.92). Remaining gap to winner ≈ **20 kg**. **No superiority claim.**

#### Entity shift (LOTO)

| Regime | MAE (order) | Note |
|--------|------------:|------|
| Flight Direct (CatBoost ref) | **88.07** | Types still seen |
| LOTO Direct macro | **283.25** | ~3× inflation |
| LOTO Flow macro | **265.86** | Suggestive; CI fragile |
| LOTO Residual matched macro | **523.27** | Much worse |

#### Domain-local conclusions (AeroTwin)

1. Hybrid ML is necessary; pure OpenAP fails.  
2. Energy (+ weather) and dynamic mass help under proper protocols.  
3. Residual is not the right inductive bias when OpenAP is badly scaled for held-out types.  
4. Official **221.33 kg** is competitive open hybrid, not winner.  
5. Entity (type) shift dominates random-flight metrics.

**For full AeroTwin figures/tables, ablations, SHAP, external pilot:** use `research.md` §§7–16 unchanged.

### G.4 AeroTwin figure/table pointer

All **Fig. 1–22** and **Tables 1–12** in `research.md` remain valid for Case Study 1. Prefer labeling them **Fig. AT-*** / **Table AT-*** in the two-domain manuscript to free **Fig. V-*** for VED and **Fig. X-*** for cross-domain.

---

# Part H — Case Study 2: VED  
## (numbers locked from `vehicle/VEE/project_status_report.md`, 2026-07-25)

> **Honesty rule:** All decimals below come from the VEE status report / paper results. Units are **liters** of cumulative trip fuel (`cum_fuel_l`) unless noted. Do not strengthen claims beyond bootstrap results.

### H.1 Domain snapshot

| Item | Value |
|------|--------|
| Dataset | Vehicle Energy Dataset (VED) |
| Full scale | **32,536 trips**, **384 vehicles** (54 weeks) |
| ICE / HEV / PHEV / EV | 18,926 / 9,501 / 3,605 / 504 trips |
| **Primary cohort** | **ICE+HEV, valid fuel >0: 27,100 trips / 341 vehicles** |
| Target | Trip-level cumulative fuel **`cum_fuel_l` (L)** |
| Physics prior (fair) | Force-balance cumulative fuel `phys_fb_cum_fuel_l` |
| Physics (diagnostic only) | MAF / Algorithm-1 path — **circular** with many labels; not a fair baseline |
| Partial observation | Public VED: no OEM manufacturer/model; `vehicle_class` ~92% UNKNOWN; labels often Alg.1 when OEM FuelRate missing |
| Primary entity | **`engine_config`** (structural proxy for type; LOTO-analogue) |
| Primary protocol | LOEO on `engine_config`, **20** multi-vehicle folds, **23,464** pooled test trips |
| Learner | LightGBM for ML models (`direct_ml`, `residual_ml`, …) |

### H.2 Why VED is a valid second case study

| Parallel | AeroTwin | VED |
|----------|----------|-----|
| Cumulative target | Fuel kg over interval | Fuel **L** over trip |
| Physics baseline | OpenAP fuel flow | Force-balance (`physics_fb`) |
| Heterogeneous “entities” | ICAO aircraft types | `engine_config` / vehicle / engine_family |
| Structural menu | Direct / Residual / Rate | `direct_ml` / `residual_ml` / `rate_integrate` (+ `residual_rate`) |
| Risk | Type shift (LOTO) | Engine-config LOEO |

**Do not claim identical data regimes.** Parallelism is **structural**, not physical identity.

### H.3 Methods (VED)

#### H.3.1 Strategies

| Strategy | VED model ID | Definition | Locked claim |
|----------|--------------|------------|--------------|
| **Direct** | `direct_ml` | LGBM: \(f(x)\to y\) | Best **MAE** under primary LOEO |
| **Residual** | `residual_ml` | FB residual + LGBM; add FB | Beats physics; **≈ Direct on MAE**; better RMSE/R² |
| **Rate-then-integrate** | `rate_integrate` | Predict rate × duration | **Does not** beat Direct on MAE |
| **Residual-rate** | `residual_rate` | Residual on rate vs physics rate, then integrate | Best **RMSE / R²** under primary LOEO |
| **Fair physics** | `physics_fb` | Force-balance only | Weak LOEO baseline |
| **MAF physics** | `physics_maf` | Diagnostic only | **Circular** with Alg.1 labels for many types |

#### H.3.2 Evaluation protocols

| Protocol | What is held out | Role |
|----------|------------------|------|
| **IID / random** | Random trips | Upper-bound; **overstates** accuracy |
| **Vehicle holdout** | Unseen vehicles (type overlap allowed) | Intermediate regime |
| **LOEO `engine_family`** | Coarser structural family | Harder for ML than config LOEO |
| **LOEO `engine_config` (primary)** | Exact engine configuration string | Primary paper protocol |
| Inference | Entity paired bootstrap (**2000** reps) for key ΔMAE tests | Significance |

Report **both MAE and RMSE** for every structural comparison.

#### H.3.3 Models and residual anchor

- Residual physics anchor: **`phys_fb_cum_fuel_l` (not MAF)**  
- Matched LGBM comparisons across Direct / Residual / Rate  
- Metrics: MAE, RMSE, R², bias, MedAE, MAPE, macro/micro  

### H.4 Results (locked)

#### H.4.0 Primary LOEO leaderboard (engine_config, 20 folds, n = 23,464)

| Model | MAE (L) | RMSE | R² |
|-------|--------:|-----:|---:|
| **direct_ml** (LGBM) | **0.0578** | 0.116 | 0.916 |
| residual_ml (FB + LGBM) | 0.0582 | **0.095** | **0.944** |
| residual_rate | 0.0594 | **0.093** | **0.946** |
| rate_integrate | 0.0618 | 0.099 | 0.939 |
| physics_fb (fair physics) | 0.1270 | 0.196 | 0.761 |
| physics_maf (diagnostic) | 0.0618 | 0.244 | 0.632 |

**Interpretation (locked):** Learned models roughly **halve** fair force-balance MAE. Ranking is **metric-dependent**: Direct wins MAE by a razor-thin margin; residual / residual_rate win RMSE and R² (better large-error control). **Do not** use physics_maf as a competitive baseline (label circularity).

#### H.4.1 Physics vs hybrid

| Comparison | Result | Source |
|------------|--------|--------|
| residual_ml vs physics_fb | ΔMAE = **−0.0688** L; 95% CI **[−0.093, −0.047]**; **significant** (residual better) | Entity bootstrap, 2000 reps |
| All ML vs physics_fb | All significantly better under entity bootstrap | Status §6.1 |
| physics_fb absolute LOEO | MAE **0.1270**, RMSE **0.196**, R² **0.761** | Primary leaderboard |

#### H.4.2 Residual vs Direct

| Quantity | Value |
|----------|------:|
| residual_ml MAE | 0.0582 |
| direct_ml MAE | **0.0578** |
| ΔMAE (res − dir) | **+0.0004** |
| 95% CI | **[−0.002, +0.003]** |
| Significant on MAE? | **No** |
| residual_ml RMSE / R² | **0.095** / **0.944** |
| direct_ml RMSE / R² | 0.116 / 0.916 |
| Entity wins (lowest MAE among physics/direct/residual) | Direct **11** · Residual **8** · physics_fb **1** |

**Writing language:** “We **cannot** claim residual dominates Direct on primary MAE under LOEO. Residual is competitive and improves second-moment metrics.”

#### H.4.3 Rate-then-integrate

| Model | MAE | vs Direct MAE |
|-------|----:|---------------|
| rate_integrate | **0.0618** | worse than 0.0578 |
| residual_rate | 0.0594 | slightly worse MAE; **best RMSE (0.093)** |

**Locked claim:** Rate structure alone does **not** beat direct cumulative regression on MAE in this trip-level setting.

#### H.4.4 Entity-level difficulty / generalization ladder (MAE)

| Model | IID | Vehicle | LOEO family | LOEO config (primary) |
|-------|----:|--------:|------------:|----------------------:|
| residual_ml | 0.0487 | 0.0596 | **0.0710** | 0.0582 |
| direct_ml | 0.0511 | 0.0610 | **0.0703** | **0.0578** |
| physics_fb | 0.1441 | 0.1500 | 0.1374 | 0.1270 |

**IID → LOEO config gap (ML):**

| Model | IID MAE | LOEO MAE | Gap | Ratio |
|-------|--------:|---------:|----:|------:|
| residual_ml | 0.0487 | 0.0582 | **+19.4%** | **1.19×** |
| direct_ml | 0.0511 | 0.0578 | **+13.0%** | **1.13×** |
| physics_fb | 0.1441 | 0.1270 | −11.9%* | 0.88× |

\*Physics “improves” under LOEO largely due to **different test composition** (only large multi-vehicle entities)—**do not over-interpret** as negative generalization gap.

**Non-monotonicity (locked narrative):**  
For Direct MAE, ordering by regime is **not** a simple “coarser always harder” ladder:  
IID (0.0511) < LOEO config (0.0578) < Vehicle (0.0610) < LOEO family (0.0703).  
Coarser **family** LOEO is hardest for ML; **vehicle** holdout can exceed **config** LOEO error because type overlap and fold composition differ. Document aggregation and entity definition whenever claiming difficulty order.

#### H.4.5 Dataset and protocol summary tables (filled)

**Table V1 — Dataset summary**

| Quantity | Value |
|----------|------:|
| Weeks | 54 |
| All trips / vehicles | **32,536 / 384** |
| Primary cohort (ICE+HEV, fuel>0) | **27,100 trips / 341 vehicles** |
| LOEO test pool | **23,464 trips** (20 `engine_config` entities) |
| Target | `cum_fuel_l` (liters) |
| Features (examples) | trip scale (`duration_s`, `distance_km`, `n_samples`); physics (`phys_fb_cum_fuel_l`); powertrain / displacement; kinematics aggregates from VED preprocess |
| Train/test definition | LOEO: train on all other engine configs; test held-out config (multi-vehicle entities only in primary) |
| Physics model | Force-balance cumulative fuel (`physics_fb`); residual anchor `phys_fb_cum_fuel_l` |
| MAF note | Diagnostic only — circular with Algorithm-1 labels for many types |

**Table V2 — Strategy comparison under IID (MAE locked; RMSE from B/V ladder where available)**

| Strategy | MAE (L) | RMSE (B/V) | vs Direct (MAE) |
|----------|--------:|-----------:|-----------------|
| physics_fb | 0.1441 | — | much worse |
| direct_ml | **0.0511** | 0.114 | ref |
| residual_ml | **0.0487** | 0.089 | slightly better MAE (IID only; not primary claim) |
| rate_integrate | 0.053* | 0.085 | competitive RMSE; not primary LOEO winner |

\*B/V table reports rate IID MAE ≈ 0.053 (rounded); use generalization ladder as primary MAE source for Direct/Residual/Physics.

**Table V3 — Primary entity holdout (LOEO engine_config)**

| Strategy | MAE (L) | RMSE | R² | Notes |
|----------|--------:|-----:|---:|-------|
| direct_ml | **0.0578** | 0.116 | 0.916 | **Best MAE** |
| residual_ml | 0.0582 | **0.095** | **0.944** | MAE ≈ Direct (NS); better RMSE/R² |
| residual_rate | 0.0594 | **0.093** | **0.946** | Best RMSE/R² |
| rate_integrate | 0.0618 | 0.099 | 0.939 | Worse MAE than Direct |
| physics_fb | 0.1270 | 0.196 | 0.761 | Fair physics baseline |

**Vehicle holdout MAE (intermediate):** direct 0.0610 · residual 0.0596 · physics_fb 0.1500.

**Table V4 — Entity granularity ladder (MAE)**

| Granularity / regime | residual_ml | direct_ml | physics_fb | Note |
|----------------------|------------:|----------:|-----------:|------|
| IID (random trips) | 0.0487 | 0.0511 | 0.1441 | Easiest for ML |
| Vehicle holdout | 0.0596 | 0.0610 | 0.1500 | Intermediate; type overlap allowed |
| LOEO engine_config | 0.0582 | **0.0578** | 0.1270 | **Primary** |
| LOEO engine_family | **0.0710** | **0.0703** | 0.1374 | Hardest for ML (coarser blocks) |

Macro MAE on primary LOEO (from paper results, where reported): direct **0.0620**, residual **0.0627**, residual_rate **0.0621**, rate **0.0644**, physics_fb **0.1392**.

**Table V5 — Statistical tests (entity bootstrap, 2000 reps)**

| Comparison | ΔMAE (L) | 95% CI | Significant? |
|------------|---------:|--------|--------------|
| residual_ml − physics_fb | −0.0688 | [−0.093, −0.047] | **Yes** (residual better) |
| residual_ml − direct_ml | +0.0004 | [−0.002, +0.003] | **No** |

**Table V6 — Bias–variance (train bootstrap, 40 reps; selected rungs)**

| Regime | Model | Bias² | Variance | MSE | RMSE |
|--------|-------|------:|---------:|----:|-----:|
| IID | direct | 0.0139 | 0.0015 | 0.0154 | 0.114 |
| IID | residual | 0.0080 | 0.0006 | 0.0087 | 0.089 |
| IID | rate | 0.0073 | 0.0004 | 0.0077 | 0.085 |
| LOEO_config | direct | 0.0160 | 0.0023 | 0.0182 | 0.108 |
| LOEO_config | residual | 0.0088 | 0.0017 | 0.0105 | 0.089 |
| LOEO_config | rate | 0.0094 | 0.0018 | 0.0112 | 0.095 |

**Hypothesis:** “structural models act mainly as variance reducers under entity shift.”  
**Verdict: NOT SUPPORTED.** Residual/Direct variance ratio is **best under IID (0.44×)** and **worse under LOEO (~0.75×)**. Residual improves **both Bias² and Variance** vs Direct at every rung—not variance-only under shift.

#### H.4.6 Per-entity heterogeneity (residual_ml)

- **Hardest:** `5-FI 2.5L` (MAE 0.099), hybrid V6 `6-GAS/ELECTRIC 3.5L`, `4-GAS/ELECTRIC 2.0L`, turbo I4, large V6 ICE.  
- **Easiest:** small turbo `4-FI T/C 1.4L`, high-volume hybrids `4-GAS/ELECTRIC 1.8L/1.5L`.  
- Difficulty concentrates in **rare architectures and large-displacement ICE**.

#### H.4.7 Feature importance under shift (residual_ml, permutation)

Under **LOEO**, `phys_fb_cum_fuel_l` importance **increases** (+0.022 ΔMAE) vs IID, while `duration_s`, `powertrain`, and `engine_displacement_l` collapse. Qualitative support that physics structure matters more when entity identity cues fail—even though residual MAE ≈ direct MAE overall.

### H.5 Domain-local conclusions (VED)

1. Fair force-balance physics alone is weak under LOEO (MAE **0.127 L**); learned models cut error roughly in half.  
2. Residual significantly beats pure physics, but **does not significantly beat Direct on MAE**.  
3. Residual / residual_rate **do** improve RMSE and R² (metric-dependent ranking).  
4. Rate-then-integrate does **not** outperform Direct on MAE.  
5. IID **overstates** ML accuracy by **13–19%** vs engine-config LOEO.  
6. Entity difficulty is **not strictly monotonic** in coarseness (family hardest; vehicle vs config intermediate/non-monotone).  
7. Bias–variance “mainly variance reduction under shift” hypothesis is **rejected**.  
8. MAF physics must not be used as a fair baseline (circular labels).

### H.6 VED figures (export from VEE `results/`)

| Paper ID | Suggested filename / source | Content | Status |
|----------|----------------------------|---------|--------|
| **Fig. V1** | dataset overview | Trips/vehicles, cohort pie, fuel hist | Export from VEE reports |
| **Fig. V2** | physics_fb vs actual | Force-balance scatter | Needed / export |
| **Fig. V3** | `generalization_bars.png` | IID / vehicle / family / config MAE | **Exists** in VEE paper_assets |
| **Fig. V4** | primary LOEO bars | Direct / Residual / Rate / physics MAE+RMSE | Export leaderboard |
| **Fig. V5** | `generalization_gap_pct.png` | IID→LOEO gap % | **Exists** |
| **Fig. V6** | residual vs direct per entity | Win counts 11/8/1 | Needed |
| **Fig. V7** | rate vs direct | MAE comparison | Needed |
| **Fig. V8** | MAE vs RMSE ranking flip | Direct best MAE; residual best RMSE | **High value** |
| **Fig. V9** | `bv_variance_bias_ladder.png` | Bias–variance across rungs | **Exists** in VEE |
| **Fig. V10** | permutation importance IID vs LOEO | Physics feature rises under shift | Export |

### H.7 VED tables for the paper

| Paper ID | Content | Status |
|----------|---------|--------|
| **Table V1** | Dataset summary (§H.4.5) | **Filled** |
| **Table V2** | IID leaderboard | **Filled** (MAE + partial RMSE) |
| **Table V3** | Primary LOEO leaderboard | **Filled** |
| **Table V4** | Entity granularity ladder | **Filled** |
| **Table V5** | Bootstrap significance | **Filled** |
| **Table V6** | Bias–variance summary | **Filled** |
| **Table V7** | Hyperparameters / LGBM config | From `configs/default.yaml` (cite VEE) |

---

# Part I — Cross-domain analysis (modest; do not overclaim)

## I.1 Shared comparison matrix (write as main cross-domain table)

| Structural question | AeroTwin | VED | Transfer? |
|---------------------|----------|-----|-----------|
| Pure physics usable alone? | **No** (MAE ~668 kg, R² −2.16) | **No** (physics_fb MAE 0.127 L under LOEO) | Shared: physics alone inadequate |
| Residual beats pure physics? | Hybrid ≫ physics; residual *form* still loses to Direct | **Yes** (ΔMAE −0.0688 L, significant) | Partial |
| Residual beats strong Direct on **MAE**? | **No** | **No** (NS; 0.0582 vs 0.0578) | **Agreement** |
| Residual better on **RMSE**? | **No** (residual worse on matched LOTO) | **Yes** (0.095 / 0.093 vs 0.116) | **Does not transfer** |
| Rate-then-integrate helps MAE? | Often helpful (official); LOTO stats fragile | **No** (0.0618 > 0.0578) | **Does not transfer cleanly** |
| Entity holdout harder than random? | **Yes** (~3× LOTO) | **Yes** (+13–19% IID→config LOEO) | **Yes — shared lesson** (scale differs) |
| Difficulty monotonic in entity grain? | Untestable (one pure entity rung) | **Not strictly** (family > config; vehicle intermediate) | VED-specific nuance |
| “Structure mainly cuts variance under shift”? | Not the AeroTwin primary hypothesis | **Rejected** on VED B/V ladder | Do not claim for either domain |
| Feature engineering | Energy + dynamic mass | Trip scale + FB physics feature (importance rises under LOEO) | Features **do not** port as-is |
| Best “production” recipe | Direct+Flow ensemble + mass/calibration | Direct for MAE; residual/residual_rate if RMSE prioritized | **Domain- and metric-specific** |

## I.2 What *does* transfer (safe claims)

1. **Physics-only baselines are insufficient** in both regimes.  
2. **Entity-aware evaluation is necessary**; random/IID overestimates robustness (AeroTwin ~3×; VED ~1.13–1.19×).  
3. **Structural choice is not free** and can reverse under shift.  
4. **Metric choice matters**: MAE vs RMSE can disagree (clear on VED; also appears in AeroTwin Direct vs Flow / ensemble).  
5. **Matched-model comparisons** are required.  
6. Residual is **not** an automatic MAE champion once a strong Direct model exists.

## I.3 What does *not* transfer (safe claims)

1. **Residual as universal champion** — AeroTwin residual loses badly under LOTO; VED residual ≈ Direct on MAE.  
2. **Residual RMSE advantage** — present on VED, absent (reversed) on AeroTwin matched residual LOTO.  
3. **Rate-then-integrate as universal duration fix** — helpful in parts of AeroTwin; fails MAE test on VED.  
4. **Feature recipes** — OpenAP energy/mass ≠ force-balance trip features.  
5. **Absolute error scales** — kg ≠ L.  
6. **Magnitude of entity gap** — AeroTwin LOTO inflation ≫ VED IID→LOEO gap; do not equate percentages.

## I.4 Mechanistic intuition (discussion, labeled as hypothesis)

| Observation | Plausible mechanism (not proven universal) |
|-------------|--------------------------------------------|
| AeroTwin residual fails hard on some LOTO types | Broken OpenAP scale for held-out widebodies; residual **inherits** baseline error |
| VED residual helps vs pure physics | Force-balance captures useful structure; residual absorbs coefficient/regime error |
| VED residual ≈ Direct on MAE, better RMSE | Second-moment control / large-error tails; not a first-moment win |
| VED residual_rate best RMSE | Rate residual may stabilize extremes; still not MAE winner |
| Rate fails MAE on VED | Integration amplifies rate bias at trip scale |
| Non-monotonic entity difficulty | Fold composition + structural block size; family removes more shared structure than config |
| B/V hypothesis fails | Residual reduces bias² and variance at every rung; variance ratio best under IID, not LOEO |

Keep language: “suggests,” “consistent with,” not “proves.”

## I.5 Cross-domain figures/tables needed

| Paper ID | Filename | Content |
|----------|----------|---------|
| **Fig. X1** | `fig_cross_strategy_heatmap.png` | 2 domains × strategies × metrics (MAE/RMSE ranks) |
| **Fig. X2** | `fig_cross_entity_difficulty.png` | Relative inflation: AeroTwin LOTO/L1 vs VED LOEO/IID |
| **Fig. X3** | `fig_cross_transfer_summary.png` | Transfer vs no-transfer findings |
| **Table X1** | Comparison matrix (Section I.1) | Main cross-domain table |
| **Table X2** | Protocol dictionary | LOTO ICAO type ↔ LOEO `engine_config` |

**Normalization note:** Fig. X2 uses **relative inflation**, not raw kg/L.

## I.6 Cross-domain discussion paragraph (draft)

> Across aircraft fuel intervals and vehicle trip fuel, we observe a consistent **methodological** pattern rather than a consistent **architectural** winner. Physics baselines alone underperform learned hybrids, and **entity-level** holdout increases error relative to random splits—by roughly **3×** MAE under aircraft type LOTO, and by **13–19%** under vehicle engine-config LOEO. Residual learning significantly improves upon force-balance physics on VED and is dominated by Direct hybrids on AeroTwin; on VED it is **statistically indistinguishable from Direct on MAE** while improving RMSE/R². Rate-then-integrate is helpful in some AeroTwin configurations but does **not** beat Direct on VED MAE. A VED bias–variance analysis further rejects the idea that structural models act mainly as variance reducers under shift. We recommend reporting all three strategies under **matched models**, **both MAE and RMSE**, and **entity-aware splits**, and we caution against transferring structural choices between cumulative prediction domains without re-evaluation.

---

# Part J — Updated claims policy

## J.1 Allowed claims

| # | Claim |
|---|--------|
| A1 | Two real domains share a cumulative prediction structure and the Direct/Residual/Rate design space. |
| A2 | AeroTwin: hybrid ≫ OpenAP; energy features significant on Level-1; residual underperforms Direct; official Combined **221.33 kg** (R3) / **228.25 kg** (canonical); LOTO ~3× harder; no superiority vs ≈201 kg winner. |
| A3 | VED primary LOEO: Direct MAE **0.0578 L** (best MAE); residual_ml MAE **0.0582** (NS vs Direct); residual RMSE **0.095** / residual_rate **0.093** (best second-moment); physics_fb MAE **0.1270**; all ML ≫ fair physics; rate_integrate MAE **0.0618** does not beat Direct. |
| A4 | IID underestimates VED LOEO ML error by **~13–19%**. |
| A5 | VED bias–variance hypothesis that structure mainly reduces variance under shift is **rejected**. |
| A6 | Entity-level evaluation is harder than random/IID-style evaluation in **both** domains (different magnitudes). |
| A7 | Structural preferences are **domain- and protocol- and metric-dependent**. |
| A8 | **No strong evidence** that residual or rate-then-integrate transfer cleanly as default recipes from aircraft to cars. |
| A9 | MAF physics is circular with Alg.1 labels for many VED entities—not a fair baseline. |
| A10 | Matched-model residual comparisons are required (AeroTwin matched CatBoost LOTO; VED matched LGBM). |

## J.2 Forbidden / unsupported claims

| # | Do **not** write |
|---|------------------|
| F1 | “Residual learning generalizes across domains” / “residual is best practice for cumulative energy.” |
| F2 | “Rate-then-integrate solves duration heterogeneity universally.” |
| F3 | “Our method is SOTA on both aircraft and vehicles.” |
| F4 | “AeroTwin beats the PRC winner.” |
| F5 | “LOTO fuel-flow gains are statistically confirmed” (AeroTwin: suggestive; CIs cross zero). |
| F6 | Equating Level-1 ensemble RMSE (~203) with official Combined (221–228). |
| F7 | Plotting AeroTwin kg and VED liters on one unnormalized axis. |
| F8 | Claiming AeroTwin confirms VED non-monotonic entity difficulty (AeroTwin lacks coarser pure entity ladder). |
| F9 | Claiming VED residual RMSE pattern “replicates” on AeroTwin (it does **not**). |
| F10 | Implying multi-domain results validate a single shared trained model. |
| F11 | “Residual significantly beats Direct on VED MAE” (CI includes 0). |
| F12 | “Structural models mainly reduce variance under entity shift” (rejected on VED). |
| F13 | Treating physics_maf as a fair competitive baseline. |
| F14 | Over-interpreting physics_fb “IID → LOEO improvement” (−11.9%) as true negative gap. |

## J.3 Soft claims (allowed with hedging)

| Claim | Required hedge |
|-------|----------------|
| Residual inherits bad physics under aircraft type shift | “consistent with / suggests” + B77W example |
| Direct preferred when physics prior is mis-scaled | “in our aircraft LOTO setting” |
| Multi-metric reporting needed | Cite VED Direct MAE vs residual RMSE; AeroTwin flips where present |
| Residual improves large-error control on VED | Supported by RMSE/R²; optional RMSE bootstrap still listed as future work in VEE |
| Aggregation choices drive non-monotonic difficulty | Cite family vs config vs vehicle MAE order |

## J.4 Metric and leaderboard hygiene

1. Separate **Case Study 1** and **Case Study 2** result sections.  
2. Within AeroTwin: never mix Fuel-Flow and Direct tracks without labels.  
3. Within VED: always state **LOEO engine_config** vs IID vs vehicle vs family.  
4. Always state units (**kg** vs **L**).  
5. Prefer entity-clustered bootstrap for VED ΔMAE claims (2000 reps as locked).

---

# Part K — Updated limitations

## K.1 Shared

1. Two case studies ≠ exhaustive multi-domain proof.  
2. Physics priors differ in fidelity and error structure.  
3. Learners and feature stacks are domain-specific; only **strategy roles** are aligned.  
4. No joint multi-task model was trained across domains.  
5. Workshop-scale scope: depth in two domains rather than breadth across many.

## K.2 AeroTwin-specific (unchanged)

Unknown true mass; partial ACARS labels; fleet imbalance; weather proxies only; winner pipeline unpublished; external pilot small; temporal 2025 European sample; GBDT-centric models.

## K.3 VED-specific

1. Public VED has **no OEM manufacturer/model**; `engine_config` is a structural proxy, not true OEM LOEO.  
2. **MAF / Algorithm-1 label circularity** for many entities—FB residual only is fair.  
3. Primary experiments are **trip-level** (no sequence models yet).  
4. PHEV/EV dual-energy track **not** in primary cohort.  
5. Entity bootstrap uses **2000** reps (not AeroTwin’s 10k flight bootstrap)—state the difference.  
6. Optional residual-vs-direct **RMSE bootstrap** not completed in VEE backlog.  
7. `vehicle_class` ~92% UNKNOWN limits class-conditioned analysis.  
8. Test-set composition differs across regimes (n and entity filters)—especially for physics gap interpretation.

## K.4 Cross-domain comparison limits

1. Different units and scales (kg vs L).  
2. Different entity hierarchies (ICAO type vs engine_config / vehicle).  
3. Different entity-gap magnitudes (~3× vs ~1.1–1.2×).  
4. AeroTwin coarser entity holdout not run (deferred after residual gate).  
5. “Transfer” means **qualitative structural lessons**, not weight transfer.

---

# Part L — Updated conclusions (draft bullets)

1. Cumulative fuel prediction under partial observation admits a common **structural menu** (Direct / Residual / Rate) but **not** a common winner.  
2. **AeroTwin:** hybrid physics–ML is effective; energy and dynamic mass matter; residual loses to Direct; official Combined **221.33 kg**; entity (type) shift is severe (~3×).  
3. **VED:** under LOEO `engine_config`, Direct MAE **0.0578 L** leads first-moment accuracy; residual/residual_rate lead RMSE/R²; residual does not significantly beat Direct on MAE; rate_integrate does not win MAE; fair physics is weak (0.127 L); IID overstates ML by 13–19%; B/V “variance-only under shift” is rejected.  
4. **Cross-domain:** entity-aware protocols and multi-metric reporting transfer as lessons; residual and rate **recipes** do not.  
5. Future work: optional VED RMSE bootstrap; AeroTwin coarser entity ladder; sequence models; physics-quality diagnostics that predict when residual is safe.

---

# Part M — Future work (two-domain aware)

| Priority | Idea |
|----------|------|
| High | Export VED figures from `VEE/results/paper_assets/` into paper |
| High | Optional residual vs direct **RMSE** entity bootstrap (VEE backlog) |
| Medium | Coarser entity holdout on AeroTwin (body/family) for Phenomenon A only |
| Medium | HEV-only vs ICE-only LOEO stratified analysis |
| Medium | Physics-quality diagnostics that predict when residual is safe |
| Low | PHEV/EV dual-energy LOEO track |
| Low | Sequence-level rate models |
| Low | Additional domains (maritime, rail) with same structural menu |

---

# Part N — Figure plan for the **two-domain** paper (integrated)

## N.1 Main text (suggested ~12–14 figures)

| ID | Content | Source |
|----|---------|--------|
| **Fig. 1** | Shared structural strategies diagram (Direct / Residual / Rate) | **New** |
| **Fig. 2** | Two-domain evaluation axes (random vs entity vs temporal) | **New** |
| **Fig. 3–6** | AeroTwin data + physics + hybrid scatter | `research.md` / `figures/` |
| **Fig. 7–8** | AeroTwin ablations + residual rejection | existing |
| **Fig. 9** | AeroTwin official + gap-closing | existing |
| **Fig. 10** | AeroTwin LOTO / entity | existing |
| **Fig. 11** | VED generalization bars (IID→LOEO) | VEE `generalization_bars.png` |
| **Fig. 12** | VED primary LOEO strategy comparison (MAE+RMSE) | Export leaderboard |
| **Fig. 13** | VED MAE vs RMSE ranking flip | **New from locked table** |
| **Fig. 14** | VED bias–variance ladder | VEE `bv_variance_bias_ladder.png` |
| **Fig. 15** | Cross-domain strategy / transfer summary | **New** |

Appendix: remaining AeroTwin SHAP, DASHlink, VED per-entity and importance plots.

## N.2 Main text tables

| ID | Content |
|----|---------|
| **Table 1** | Two-domain task comparison |
| **Table 2** | Strategy definitions (shared + model IDs) |
| **Table 3–6** | AeroTwin core results (official, ablation, LOTO) |
| **Table 7** | VED primary LOEO leaderboard (H.4.0) |
| **Table 8** | VED generalization ladder + IID gap |
| **Table 9** | VED bootstrap + B/V summary |
| **Table 10** | Cross-domain transfer matrix (Section I.1) |
| **Table 11** | Protocol constants (both domains) |

---

# Part O — Drafting checklist (two-domain)

### Framing

- [ ] Title uses multi-domain / structural language (not AeroTwin-only)  
- [ ] Abstract covers both domains + “no clean transfer”  
- [ ] Contributions list includes VED and modest cross-domain analysis  
- [ ] Intro motivates shared structure before domain details  

### AeroTwin integrity

- [ ] All official numbers match freeze (228.25 / 221.33 / CI / winner 201)  
- [ ] Residual rejection includes matched LOTO where discussed  
- [ ] No superiority claim  
- [ ] Fuel-flow LOTO labeled suggestive  

### VED integrity

- [x] 32,536 trips / 384 vehicles stated  
- [x] Primary cohort 27,100 / 341 stated  
- [x] Primary LOEO table filled (0.0578 / 0.0582 / 0.0594 / 0.0618 / 0.1270)  
- [x] Residual > physics; residual ≰ Direct on MAE (NS) stated  
- [x] Residual better RMSE/R² stated without overclaiming MAE win  
- [x] Rate not better than Direct on MAE stated  
- [x] IID gap 13–19% stated  
- [x] B/V hypothesis rejected stated  
- [x] Numeric tables filled (`[VED]` cleared)  
- [ ] VED figures exported into manuscript assets  

### Cross-domain integrity

- [ ] Comparison table present  
- [ ] No universal residual/rate claim  
- [ ] Units not falsely equated  
- [ ] Transfer = qualitative lessons only  
- [ ] Residual RMSE advantage marked VED-only  

### Claims policy pass

- [ ] Section J.1 all supported  
- [ ] Section J.2 none violated  
- [ ] Soft claims hedged  

---

# Part P — Narrative thread (writing order)

1. **Hook:** Cumulative fuel matters in air and on road; data are partial; physics is imperfect.  
2. **Menu:** Direct / Residual / Rate look transferable—are they?  
3. **Case 1 deep dive:** AeroTwin—what works (hybrid, energy, mass) and what fails (residual, sparse myth).  
4. **Case 2 deep dive:** VED—physics weak; Direct wins MAE; residual wins RMSE; Rate fails MAE; IID lies; B/V myth fails.  
5. **Bridge:** Side-by-side matrix; shared protocol lessons; non-transfer of architecture defaults; metric flips.  
6. **Close:** Evaluate structure under entity shift; do not export recipes without re-testing.

---

# Part Q — Quick reference cards

## Q.1 AeroTwin (frozen)

| Item | Value |
|------|------:|
| Train flights / intervals | 10,000 / 119,032 |
| OpenAP MAE | ~668 kg |
| Best early E+W MAE | 83.76 kg |
| Official Combined (canonical) | **228.25 kg** |
| Official Combined (R3 best) | **221.33 kg** |
| Winner | ≈201 kg |
| LOTO Direct macro MAE | ~283 kg |
| Residual vs Direct | Residual worse (matched) |

## Q.2 VED (locked)

| Item | Value |
|------|------:|
| Trips / vehicles (full) | **32,536 / 384** |
| Primary cohort | **27,100 trips / 341 vehicles** |
| Primary protocol | LOEO `engine_config`, 20 folds, **23,464** test trips |
| Target unit | **liters** (`cum_fuel_l`) |
| direct_ml MAE / RMSE / R² | **0.0578** / 0.116 / 0.916 |
| residual_ml MAE / RMSE / R² | 0.0582 / **0.095** / **0.944** |
| residual_rate MAE / RMSE / R² | 0.0594 / **0.093** / **0.946** |
| rate_integrate MAE | 0.0618 |
| physics_fb MAE / RMSE / R² | 0.1270 / 0.196 / 0.761 |
| residual − direct ΔMAE | +0.0004; CI [−0.002, +0.003]; **NS** |
| residual − physics ΔMAE | −0.0688; CI [−0.093, −0.047]; **sig.** |
| IID→LOEO gap | Direct **+13%**; Residual **+19%** |
| B/V “variance under shift” | **Rejected** |
| Cross-domain architecture transfer | No strong evidence |

## Q.3 One-sentence thesis

> **Structural strategies for cumulative fuel prediction must be re-validated per domain, metric, and evaluation protocol; residual correction and rate-then-integrate are useful in some regimes (e.g., VED RMSE) but are not portable defaults from aircraft fuel estimation to vehicle trip fuel estimation—and residual does not significantly beat strong Direct models on VED MAE.**

---

# Part R — Related documents

| Document | Role |
|----------|------|
| `research.md` | Full AeroTwin single-domain package (expand Case Study 1 from here) |
| `docs/VED_PHENOMENA_REPLICATION.md` | Does VED residual/entity pattern replicate on AeroTwin? (partially / no) |
| `CURRENT_MODEL_SUMMARY.md` | AeroTwin live best model |
| `official_prc_benchmark_report.md` | Official evaluation |
| `papers/statistical_protocol.md` | Frozen AeroTwin inference rules |
| `papers/hybrid_model_summary.md` | AeroTwin hybrid narrative seed |
| `../vehicle/VEE/project_status_report.md` | **VED numbers source of truth** |
| `../vehicle/VEE/paper/results.md` | VED paper-facing results tables |
| `../vehicle/VEE/results/loeo_engine_config_default/` | Primary LOEO artifacts |

---

# Part S — Optional venue-specific reshaping

| Venue focus | Emphasize | Compress |
|-------------|-----------|----------|
| Aviation (JOAS) | Case Study 1 full; VED as comparative section | VED B/V detail |
| Transportation / ITS | Equal case studies; fuel framing | PRC winner comparison detail |
| Physics-informed ML workshop | Structural strategies + transfer failure + metric flips | Fleet operational detail |
| Evaluation / ML methodology | Protocol, entity shift, MAE/RMSE disagreement, B/V negative result | Domain physics depth |

---

*End of two-domain writing package. AeroTwin freezes preserved. VED `[VED]` placeholders filled from `vehicle/VEE/project_status_report.md` (2026-07-25). Keep cross-domain claims modest, multi-metric, and protocol-aware.*
