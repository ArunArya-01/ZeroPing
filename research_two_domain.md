# Two-Domain Research Writing Package  
## Structural Strategies for Cumulative Energy Prediction under Partial Observation and Entity Shift

**Project framing:** Multi-domain empirical study (AeroTwin aircraft + VED vehicles)  
**Primary domain (fully frozen):** AeroTwin / EUROCONTROL PRC 2025  
**Second domain (completed case study):** Vehicle Energy Dataset (VED)  
**Companion single-domain package:** `research.md` (AeroTwin-only; all numbers preserved here)  
**Cross-domain diagnostic in-repo:** `docs/VED_PHENOMENA_REPLICATION.md`  
**Last synced:** 2026-07-27  
**Tone:** Workshop / specialized-venue empirical paper — careful, non-overclaiming  

This document is the **two-case-study writing package**. Expand into formal prose; do **not** invent metrics beyond what is listed. **AeroTwin numbers are frozen** — identical to `research.md` / official reports.

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
5. Case Study 2 — VED (vehicle trip energy)
   5.1 Domain, data, scale
   5.2 Physics prior (force-balance) and features
   5.3 Methods and entity protocols (IID / vehicle holdout / LOEO)
   5.4 Results (Direct vs Residual vs Rate; entity difficulty)
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
| **Honesty over narrative neatness** | Residual helps VED physics baseline but does not cleanly beat Direct; AeroTwin residual loses; Rate is unreliable as a universal fix. |
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
| **Keywords** | cumulative energy prediction; hybrid physics–ML; residual learning; rate-then-integrate; entity shift; leave-one-entity-out; aircraft fuel burn; vehicle energy; partial observability |
| **AeroTwin code** | https://github.com/ArunArya-01/ZeroPing |
| **AeroTwin data** | https://huggingface.co/datasets/aerotwin/aero-data |
| **VED data** | Vehicle Energy Dataset (Oh et al. / public VED release — cite original) |
| **VED study status** | Completed case study (summary findings frozen for this package) |

### Title options (new framing)

| ID | Title |
|----|--------|
| **T1 (recommended)** | Structural Strategies for Cumulative Energy Prediction under Partial Observation and Entity Shift: Aircraft Fuel and Vehicle Energy Case Studies |
| **T2** | Direct, Residual, or Rate-Then-Integrate? An Empirical Two-Domain Study of Physics-Informed Cumulative Prediction |
| **T3** | When Physics-Informed Structure Helps—and When It Does Not: Lessons from Aircraft Fuel Burn and Vehicle Energy Estimation |
| **T4** | Evaluation Protocol Matters: Hybrid Modeling of Cumulative Fuel/Energy under Flight-Level and Vehicle-Level Shift |
| **T5 (aviation-first)** | AeroTwin and Beyond: Hybrid Interval Fuel Prediction with a Comparative Vehicle Energy Case Study |
| **Avoid** | Titles that claim “universal residual learning,” “cross-domain generalization of architecture X,” or “SOTA on both domains” |

---

# Part C — Updated abstract (both domains, honest)

## C.1 Abstract draft (≈220–280 words; trim for venue)

Predicting **cumulative energy use**—aircraft fuel on ACARS-labeled intervals, or vehicle energy over trips—from partially observed kinematics is a recurring problem in transportation analytics. Practitioners often choose among three **structural strategies**: (i) **Direct** prediction of the cumulative quantity, (ii) **Residual** correction of a physics baseline, and (iii) **Rate-then-integrate** (predict instantaneous consumption, then multiply by duration). Whether these choices transfer across domains is unclear, and **entity-level** evaluation (unseen aircraft types or vehicles) can change rankings relative to random holdout.

We study this question in **two completed case studies** under domain-appropriate physics priors and protocols.

**Case Study 1 (AeroTwin)** uses the EUROCONTROL PRC 2025 fused ADS-B/ACARS data (~10,000 training flights; 119,032 fuel intervals). Pure OpenAP physics is unusable (flight-holdout MAE ≈ 668 kg; R² ≈ −2.16). Direct hybrid gradient boosting with **energy-state** features yields bootstrap-supported gains; **residual learning underperforms** direct hybrid. Under the official Rank+Final protocol, a Direct+Fuel-Flow ensemble reaches combined RMSE **228.25 kg**, improved to **221.33 kg** with dynamic mass features—still short of the published winner (≈201 kg; no superiority claim). Leave-one-type-out inflates error by roughly **3×**, showing that flight-level metrics overestimate robustness under type shift.

**Case Study 2 (VED)** analyzes **32,536 trips** from **384 vehicles**. Residual correction of a force-balance physics baseline **improves over pure physics**, but residual models do **not consistently beat strong Direct** predictors (especially on MAE). Rate-then-integrate does **not reliably** improve cumulative accuracy. Entity-level protocols (vehicle holdout / leave-one-entity-out) are harder than IID-style splits; difficulty is **not strictly monotonic** across entity granularities when micro-averaging and fold composition effects are considered.

**Cross-domain takeaway:** structural preferences are **domain- and protocol-dependent**. We find **no strong evidence** that residual learning or rate-then-integrate transfer as default recipes from aircraft to cars. The practical message is to report Direct/Residual/Rate under **matched models**, multiple metrics, and entity-aware splits—not to export a single architecture across cumulative prediction tasks.

**Word target:** 150–250 for short venues; keep the three-strategy framing and the “no clean transfer” sentence in all versions.

## C.2 Elevator summary (poster / intro)

| Item | Statement |
|------|-----------|
| **Shared problem** | Predict cumulative energy/fuel from partial kinematics + imperfect physics priors under entity shift. |
| **Three strategies** | Direct · Residual (physics + correction) · Rate-then-integrate. |
| **AeroTwin headline** | Hybrid works; energy/mass help; residual loses; official **221.33 kg**; LOTO ~3× harder. |
| **VED headline** | Residual beats pure physics; residual ≰ Direct (esp. MAE); Rate unreliable; entity eval harder; non-monotonic difficulty possible. |
| **Cross-domain claim** | Results are **domain-dependent**; protocol and metric choice matter; **no universal structural winner**. |
| **What we do *not* claim** | Cross-domain architecture transfer; residual as general best practice; beating PRC winner. |

---

# Part D — Shared problem formulation

## D.1 Cumulative prediction task (domain-agnostic)

For labeled segment \(i\) belonging to entity \(e\) (flight, aircraft type, vehicle, …):

\[
y_i = \text{cumulative energy or fuel over segment } i
\]

Predict \(\hat{y}_i\) from kinematics \(\mathbf{x}_i\), metadata \(\mathbf{m}_i\), and optional physics baseline \(\hat{y}_i^{\text{phys}}\):

\[
\hat{y}_i = f\!\big(\mathbf{x}_i,\, \mathbf{m}_i,\, \hat{y}_i^{\text{phys}}\big)
\]

**Partial observation:** labels cover only part of operational time or depend on sparse sensors; physics inputs (mass, resistance coefficients, air data) are incomplete or assumed.

## D.2 Three structural strategies (shared vocabulary)

| Strategy | Predict | Recover \(y\) | Physics role |
|----------|---------|---------------|--------------|
| **Direct** | \(y_i\) | identity | Feature and/or ignored |
| **Residual** | \(r_i = y_i - \hat{y}_i^{\text{phys}}\) | \(\hat{y}_i = \hat{y}_i^{\text{phys}} + \hat{r}_i\) | Explicit baseline; model corrects |
| **Rate-then-integrate** | rate \(\rho_i\) (e.g. kg/s or energy/s) | \(\hat{y}_i = \hat{\rho}_i \cdot \Delta t_i\) | Optional; normalizes duration scale |

**AeroTwin names:** Direct kg · Residual kg · **Fuel-Flow** (rate).  
**VED names:** Direct · Residual (force-balance) · Rate-then-integrate.

Use this shared vocabulary in §6 so readers can compare without equating implementations.

## D.3 Evaluation axes (shared language)

| Axis | AeroTwin analogue | VED analogue | Claim type |
|------|-------------------|--------------|------------|
| **Random / quasi-IID segment groups** | Flight-level 80/20 (types still seen) | Trip/sample IID or random split | Unseen segments, entities partially seen |
| **Entity holdout** | Leave-one-**type**-out (LOTO) | Vehicle holdout / LOEO | Unseen entity generalization |
| **Temporal** | Official Rank/Final months | _(if used in VED study)_ | Time shift (not entity) |
| **External data** | DASHlink pilot | _(if any)_ | Dataset shift |

**Rule:** Never treat entity-holdout metrics as interchangeable with random-holdout metrics.

## D.4 Shared challenges (intro list)

1. Cumulative targets with heterogeneous segment durations  
2. Imperfect physics baselines (wrong mass, coefficients, or operating regime)  
3. Partial / sparse observability  
4. Entity heterogeneity (fleet types or individual vehicles)  
5. Metric choice (MAE vs RMSE) can flip rankings  
6. Leakage if segments from the same entity cross train/test  

---

# Part E — Updated research questions

## E.1 Primary (two-domain)

> **RQ-Primary.** For cumulative energy/fuel prediction under partial observation, how do **Direct**, **Residual**, and **Rate-then-integrate** strategies behave under **random** versus **entity-level** evaluation—and do structural preferences **transfer** between commercial aircraft fuel intervals (AeroTwin) and vehicle trip energy (VED)?

**Expected answer style (write this into Results/Discussion):** preferences are **domain- and protocol-dependent**; residual and rate are **not** reliable universal defaults.

## E.2 Shared structural questions

| ID | Question | AeroTwin status | VED status (summary) |
|----|----------|-----------------|----------------------|
| **SQ1** | Does residual improve over **pure physics**? | Yes (hybrid ≫ OpenAP), but residual **architecture** loses to Direct | **Yes** — residual improves over force-balance physics |
| **SQ2** | Does residual beat **strong Direct**? | **No** (Level-1 and matched LOTO) | **No** — not consistent (esp. MAE) |
| **SQ3** | Does rate-then-integrate help vs Direct? | Often yes (official; many LOTO folds) but LOTO significance fragile | **No** — does not reliably help |
| **SQ4** | Is entity-level evaluation harder than random/IID? | **Yes** (~3× MAE under LOTO) | **Yes** (vehicle / LOEO harder) |
| **SQ5** | Is difficulty strictly monotonic in entity coarseness? | **Not testable** (only one pure entity rung: type LOTO) | **No** — not strictly monotonic (micro-avg + fold composition) |
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

### VED-only (from completed case study summary)

| ID | Question | Status |
|----|----------|--------|
| VQ1 | Residual vs pure force-balance physics? | Residual **better** |
| VQ2 | Residual vs strong Direct (MAE)? | Residual **not consistently better** |
| VQ3 | Residual vs strong Direct (RMSE)? | Domain-specific; do not claim universal residual RMSE win without tables |
| VQ4 | Rate-then-integrate vs Direct? | **Not reliably helpful** |
| VQ5 | Vehicle / LOEO vs IID difficulty? | Entity-level **harder** |
| VQ6 | Monotonic difficulty across entity granularities? | **Not strictly monotonic** |

## E.4 Hypotheses (paper-facing)

| ID | Hypothesis | Decision |
|----|------------|----------|
| H-Cross1 | Residual is the best default for cumulative energy across domains | **Rejected** |
| H-Cross2 | Rate-then-integrate is a reliable duration normalization fix across domains | **Rejected** |
| H-Cross3 | Entity holdout is harder than random/IID-style splits in both domains | **Supported** (both domains) |
| H-Cross4 | MAE and RMSE always agree on structural ranking | **Rejected** (flips appear; domain-specific) |
| H-AT-Energy | Energy features help AeroTwin Level-1 | **Accepted** (bootstrap) |
| H-AT-Res | Residual beats Direct on AeroTwin | **Rejected** (incl. matched CatBoost LOTO) |
| H-AT-Mass | Dynamic mass improves official Combined RMSE | **Accepted** (221.33) |
| H-VED-Phys | Residual helps over pure physics on VED | **Accepted** (case study) |
| H-VED-Dir | Residual consistently beats Direct on VED | **Rejected** |

---

# Part F — Updated contributions

Write contributions as **empirical + methodological**, not “we win both leaderboards.”

1. **Unified structural framing** of cumulative energy prediction via Direct / Residual / Rate-then-integrate, applied to two real operational domains with domain-appropriate physics priors (OpenAP; force-balance vehicle model).

2. **Case Study 1 (AeroTwin):** large-scale hybrid aircraft fuel modeling with flight-level and official Rank+Final evaluation; energy-state and dynamic-mass gains; residual architectures rejected; official Combined RMSE **228.25 → 221.33 kg**; LOTO ~3× degradation; pilot external audit.

3. **Case Study 2 (VED):** completed vehicle energy study on **32,536 trips / 384 vehicles** comparing Direct, Residual, and Rate under IID and entity-level protocols; residual beats pure physics but not consistently strong Direct; Rate unreliable; entity difficulty non-monotonic under micro-averaging/fold effects.

4. **Cross-domain analysis (modest):** side-by-side comparison showing **domain-dependent** structural outcomes and **no strong evidence** of clean architecture transfer from aircraft to cars.

5. **Protocol contribution:** emphasis on matched-model comparisons, entity-aware splits, multi-metric reporting (MAE and RMSE), and flight-/entity-clustered inference where applicable—so rankings are not artifacts of split choice.

6. **Negative and mixed results as first-class findings:** residual and rate are not universal; sparsity-conditioned physics gains rejected on AeroTwin; LOTO significance for fuel-flow is fragile.

7. **Reproducible AeroTwin artifacts** (loaders, featured dataset, frozen statistical protocol, figures/tables). VED reproducibility pointer to the completed case-study package _[path/citation to fill]_.

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
## (completed case study — write from these findings; fill numeric tables from VED artifacts)

> **Honesty rule:** Use the user’s completed findings as hard constraints. Where exact decimal metrics live in the VED study package, **paste from those tables**—do not invent. Placeholders marked `[VED: …]` must be filled before submission.

### H.1 Domain snapshot

| Item | Value |
|------|--------|
| Dataset | Vehicle Energy Dataset (VED) |
| Scale | **32,536 trips**, **384 vehicles** |
| Task | Cumulative trip / segment **energy** (domain units as in VED study) |
| Physics prior | **Force-balance** (longitudinal dynamics / resistance model) |
| Partial observation | Real driving telemetry; not full lab-grade energy instrumentation on all axes |
| Entity unit | **Vehicle** (and finer/coarser LOEO-style entity definitions used in the study) |

### H.2 Why VED is a valid second case study

| Parallel | AeroTwin | VED |
|----------|----------|-----|
| Cumulative target | Fuel kg over interval | Energy over trip/segment |
| Physics baseline | OpenAP fuel flow | Force-balance power/energy |
| Heterogeneous “entities” | Aircraft types / flights | Vehicles / drivers / trips |
| Structural menu | Direct / Residual / Rate | Direct / Residual / Rate |
| Risk | Type shift (LOTO) | Vehicle holdout / LOEO |

**Do not claim identical data regimes.** Aircraft intervals are ACARS-sparse aviation telemetry; VED is ground-vehicle trips. Parallelism is **structural**, not physical identity.

### H.3 Methods (VED)

#### H.3.1 Strategies

| Strategy | Definition on VED | Expected writing claim |
|----------|-------------------|------------------------|
| **Direct** | Predict cumulative energy end-to-end | Strong baseline |
| **Residual** | Predict \(y - y_{\text{physics}}\) (force-balance) then add back | Beats pure physics; **not** consistent winner vs Direct |
| **Rate-then-integrate** | Predict rate × duration | **Does not reliably help** |

#### H.3.2 Evaluation protocols (must appear)

| Protocol | What is held out | Role in paper |
|----------|------------------|---------------|
| **IID / random** | Random trips or samples (as defined in VED study) | Upper-bound “easy” generalization |
| **Vehicle holdout** | Unseen vehicles | Entity-level difficulty |
| **LOEO** (leave-one-entity-out) | One entity at a time (vehicle or other entity grain) | Stress test |
| **Multiple entity granularities** | Finer vs coarser entity definitions | Non-monotonic difficulty analysis |

Report **both MAE and RMSE** (or domain-standard pair) for every structural comparison.

#### H.3.3 Models

State learner family used in the completed VED study (e.g., GBDT / RF / linear) and keep **matched** when comparing Direct vs Residual vs Rate—same lesson as AeroTwin matched CatBoost residual LOTO.

### H.4 Results (findings-locked; numbers from VED package)

#### H.4.1 Physics vs hybrid

| Comparison | Finding (locked) | Table to cite |
|------------|------------------|---------------|
| Residual vs **pure physics** | Residual **improves** | `[VED: table_physics_vs_residual]` |
| Pure physics absolute error | Large / structured | `[VED: table_physics_only]` |

#### H.4.2 Residual vs Direct

| Comparison | Finding (locked) | Writing language |
|------------|------------------|------------------|
| Residual vs strong Direct (**MAE**) | Residual does **not consistently** beat Direct | “No consistent MAE advantage for residual” |
| Residual vs strong Direct (**RMSE**) | Report actual VED outcome; **do not import AeroTwin residual loss as if identical** | Domain-local only |
| Interpretation | Residual useful as physics correction, not automatic champion | |

#### H.4.3 Rate-then-integrate

| Comparison | Finding (locked) |
|------------|------------------|
| Rate vs Direct | **Does not reliably help** cumulative accuracy |
| When it appears to help | Treat as fold/metric-specific; not a default recommendation |

#### H.4.4 Entity-level difficulty

| Finding (locked) | Implication |
|------------------|-------------|
| Vehicle holdout / LOEO **harder** than IID | Entity shift is first-class |
| Difficulty **not strictly monotonic** across entity granularities | Micro-averaging + fold composition can reverse naive “coarser = harder” stories |
| Do not over-smooth with a single macro number | Show per-granularity + sensitivity |

#### H.4.5 Placeholder result tables (fill from VED study)

**Table V1 — Dataset summary**

| Quantity | Value |
|----------|------:|
| Trips | 32,536 |
| Vehicles | 384 |
| Features / sensors | `[VED]` |
| Train/test definition | `[VED]` |
| Physics model | Force-balance `[VED ref]` |

**Table V2 — Strategy comparison under IID**

| Strategy | MAE | RMSE | vs Direct |
|----------|----:|-----:|-----------|
| Physics only | `[VED]` | `[VED]` | — |
| Direct | `[VED]` | `[VED]` | ref |
| Residual | `[VED]` | `[VED]` | not consistently better (MAE) |
| Rate-then-integrate | `[VED]` | `[VED]` | not reliable |

**Table V3 — Strategy comparison under vehicle holdout / LOEO**

| Strategy | MAE | RMSE | Notes |
|----------|----:|-----:|-------|
| Direct | `[VED]` | `[VED]` | |
| Residual | `[VED]` | `[VED]` | |
| Rate | `[VED]` | `[VED]` | |
| Difficulty vs IID | higher | higher | entity harder |

**Table V4 — Entity granularity ladder**

| Granularity | Macro MAE | Macro RMSE | Micro MAE | Note |
|-------------|----------:|-----------:|----------:|------|
| Fine entity | `[VED]` | `[VED]` | `[VED]` | |
| Medium | `[VED]` | `[VED]` | `[VED]` | |
| Coarse (e.g. vehicle) | `[VED]` | `[VED]` | `[VED]` | not strictly mono. |

### H.5 Domain-local conclusions (VED)

1. Force-balance physics alone is insufficient; **residual correction helps over pure physics**.  
2. Once a **strong Direct** model is available, residual is **not a consistent winner**—especially on **MAE**.  
3. **Rate-then-integrate** is not a reliable structural upgrade.  
4. **Entity-level** evaluation is essential; IID overstates performance.  
5. **Non-monotonic** difficulty across entity grains is a real reporting hazard—document aggregation choices.

### H.6 VED figures still needed for the paper

| Paper ID | Suggested filename | Content | Status |
|----------|-------------------|---------|--------|
| **Fig. V1** | `fig_ved_dataset_overview.png` | Trips/vehicles distribution, energy histogram | **Needed** (or export from VED study) |
| **Fig. V2** | `fig_ved_physics_vs_actual.png` | Force-balance vs ground truth scatter | **Needed** |
| **Fig. V3** | `fig_ved_strategy_iid.png` | Direct / Residual / Rate under IID (MAE+RMSE) | **Needed** |
| **Fig. V4** | `fig_ved_strategy_entity.png` | Same under vehicle holdout / LOEO | **Needed** |
| **Fig. V5** | `fig_ved_entity_difficulty_ladder.png` | Error vs entity granularity (macro + micro) | **Needed** — supports non-monotonic claim |
| **Fig. V6** | `fig_ved_residual_vs_direct_scatter.png` | Per-fold or per-vehicle Δ(Residual−Direct) | **Needed** |
| **Fig. V7** | `fig_ved_rate_ablation.png` | Rate-then-integrate reliability / failures | **Needed** |
| **Fig. V8** | `fig_ved_mae_rmse_ranking.png` | Ranking flips MAE vs RMSE if present | **Needed if flips exist** |
| **Fig. V9** | `fig_ved_error_by_vehicle_cluster.png` | Heterogeneity across vehicles | Optional but strong |

### H.7 VED tables still needed

| Paper ID | Content |
|----------|---------|
| **Table V1** | Dataset summary (32,536 / 384 + splits) |
| **Table V2** | IID leaderboard Direct / Residual / Rate / Physics |
| **Table V3** | Entity-holdout leaderboard |
| **Table V4** | Entity granularity ladder (macro & micro) |
| **Table V5** | Statistical tests / bootstrap / paired tests used in VED study |
| **Table V6** | Hyperparameters and matched-model protocol |

---

# Part I — Cross-domain analysis (modest; do not overclaim)

## I.1 Shared comparison matrix (write as main cross-domain table)

| Structural question | AeroTwin | VED | Transfer? |
|---------------------|----------|-----|-----------|
| Pure physics usable alone? | **No** (MAE ~668, R² −2.16) | Weak / insufficient (residual improves on it) | Shared: physics alone inadequate |
| Residual beats pure physics? | Hybrid ≫ physics; residual *form* still loses to Direct | **Yes** | Partial (physics correction valuable; form differs) |
| Residual beats strong Direct? | **No** (L1 + matched LOTO) | **No** (not consistent; esp. MAE) | **Agreement: residual ≠ automatic winner** |
| Rate-then-integrate helps? | Often helpful (official; many folds); LOTO stats fragile | **Not reliable** | **Does not transfer cleanly** |
| Entity holdout harder than random? | **Yes** (~3× LOTO) | **Yes** | **Yes — shared lesson** |
| Difficulty monotonic in entity grain? | Untestable (one pure entity rung) | **Not strictly** | VED-specific nuance; AeroTwin cannot confirm |
| Energy / mass feature engineering | Strong (energy, R3 mass) | Domain-specific vehicle features `[VED]` | Features **do not** port as-is |
| Best “production” recipe | Direct+Flow ensemble + mass/calibration | Strong Direct (residual optional vs physics) | **Domain-specific pipelines** |

## I.2 What *does* transfer (safe claims)

1. **Physics-only baselines are insufficient** for operational cumulative prediction in both studied regimes.  
2. **Entity-aware evaluation is necessary**; random/IID (or flight-random with types seen) overestimates robustness.  
3. **Structural choice is not free**: Direct vs Residual vs Rate can change error by large margins and can reverse under shift.  
4. **Metric choice matters**: MAE vs RMSE can disagree (document both).  
5. **Matched-model comparisons** are required to avoid confounding architecture with inductive bias (AeroTwin residual LOTO lesson).

## I.3 What does *not* transfer (safe claims)

1. **Residual learning as a universal champion** — loses on AeroTwin vs Direct; fails to consistently beat Direct on VED MAE.  
2. **Rate-then-integrate as a universal duration fix** — helpful in parts of AeroTwin, unreliable on VED.  
3. **Feature recipes** (OpenAP energy, dynamic mass, force-balance terms) — domain physics differ; do not claim portability of feature lists.  
4. **Absolute error scales** — kg aircraft fuel ≠ vehicle energy units; never plot on one axis without normalization.  
5. **“Harder entity = coarser entity”** — VED shows non-monotonic patterns; AeroTwin lacks coarser pure holdout to test.

## I.4 Mechanistic intuition (discussion, labeled as hypothesis)

| Observation | Plausible mechanism (not proven universal) |
|-------------|--------------------------------------------|
| AeroTwin residual fails hard on some LOTO types | Broken OpenAP scale for held-out widebodies; residual **inherits** baseline error; Direct re-learns absolute scale from kinematics |
| VED residual helps vs pure physics | Force-balance captures useful structure; residual absorbs coefficient/regime error |
| VED residual ≰ Direct on MAE | Trees already approximate cumulative mapping; explicit residual adds little once features are rich |
| Rate helps sometimes | Duration heterogeneity; rate normalizes scale—but integration amplifies rate bias |
| Non-monotonic entity difficulty | Micro-averaging weights frequent entities; fold composition can dominate coarseness |

Keep language: “suggests,” “consistent with,” not “proves.”

## I.5 Cross-domain figures/tables needed

| Paper ID | Filename | Content |
|----------|----------|---------|
| **Fig. X1** | `fig_cross_strategy_heatmap.png` | 2 domains × 3 strategies × (IID, entity) qualitative or normalized ranks |
| **Fig. X2** | `fig_cross_entity_difficulty.png` | Side-by-side “random vs entity” error inflation (normalized) |
| **Fig. X3** | `fig_cross_transfer_summary.png` | Simple icon/table graphic: transfer vs no-transfer findings |
| **Table X1** | Comparison matrix (Section I.1) | Main cross-domain table |
| **Table X2** | Protocol dictionary | Align AeroTwin LOTO ↔ VED LOEO terminology |

**Normalization note:** For Fig. X2 use **relative inflation** (entity / random − 1), not raw units.

## I.6 Cross-domain discussion paragraph (draft)

> Across aircraft fuel intervals and vehicle trip energy, we observe a consistent **methodological** pattern rather than a consistent **architectural** winner. Physics baselines alone underperform operational hybrid models, and **entity-level** holdout substantially increases error relative to random or within-entity splits. Residual learning improves upon pure physics in the vehicle setting and is dominated by strong Direct hybrids in the aircraft setting; in neither domain do we obtain robust evidence that residual correction should replace Direct prediction as a default. Rate-then-integrate is helpful in some AeroTwin configurations but does not reliably improve VED cumulative accuracy. We therefore recommend reporting all three strategies under **matched models**, **both MAE and RMSE**, and **entity-aware splits**, and we caution against transferring structural choices between cumulative prediction domains without re-evaluation.

---

# Part J — Updated claims policy

## J.1 Allowed claims

| # | Claim |
|---|--------|
| A1 | Two real domains share a cumulative prediction structure and the Direct/Residual/Rate design space. |
| A2 | AeroTwin: hybrid ≫ OpenAP; energy features significant on Level-1; residual underperforms Direct; official Combined **221.33 kg** (R3) / **228.25 kg** (canonical); LOTO ~3× harder; no superiority vs ≈201 kg winner. |
| A3 | VED: residual improves over pure force-balance physics; residual does **not consistently** beat strong Direct (esp. MAE); rate-then-integrate not reliable; entity eval harder; difficulty not strictly monotonic across entity grains. |
| A4 | Entity-level evaluation is harder than random/IID-style evaluation in **both** domains. |
| A5 | Structural preferences are **domain- and protocol-dependent**. |
| A6 | **No strong evidence** that residual or rate-then-integrate transfer cleanly as default recipes from aircraft to cars. |
| A7 | Negative/mixed results are scientifically valuable. |
| A8 | Matched-model residual comparisons are required (AeroTwin matched CatBoost LOTO). |

## J.2 Forbidden / unsupported claims

| # | Do **not** write |
|---|------------------|
| F1 | “Residual learning generalizes across domains” / “residual is best practice for cumulative energy.” |
| F2 | “Rate-then-integrate solves duration heterogeneity universally.” |
| F3 | “Our method is SOTA on both aircraft and vehicles.” |
| F4 | “AeroTwin beats the PRC winner.” |
| F5 | “LOTO fuel-flow gains are statistically confirmed” (AeroTwin: suggestive; CIs cross zero). |
| F6 | Equating Level-1 ensemble RMSE (~203) with official Combined (221–228). |
| F7 | Plotting AeroTwin kg and VED energy on one unnormalized axis as if comparable. |
| F8 | Claiming AeroTwin confirms VED non-monotonic entity difficulty (insufficient coarser entity ladder). |
| F9 | Claiming VED residual RMSE pattern “replicates” on AeroTwin (it does **not**; see `docs/VED_PHENOMENA_REPLICATION.md`). |
| F10 | Implying multi-domain results validate a single shared trained model (studies are **separate case studies**). |

## J.3 Soft claims (allowed with hedging)

| Claim | Required hedge |
|-------|----------------|
| Residual inherits bad physics under type shift | “consistent with / suggests” + B77W example |
| Direct preferred when physics prior is mis-scaled | “in our aircraft LOTO setting” |
| Multi-metric reporting needed | Cite specific flips (AeroTwin Direct vs Flow; VED if present) |
| Aggregation choices drive non-monotonic difficulty | “micro-averaging and fold composition” |

## J.4 Metric and leaderboard hygiene

1. Separate **Case Study 1** and **Case Study 2** result sections.  
2. Within AeroTwin: never mix Fuel-Flow and Direct tracks without labels (`figures/LEADERBOARD_AUDIT.md`).  
3. Always state protocol: Flight / Official / LOTO / VED-IID / VED-vehicle / LOEO.  
4. Prefer paired/bootstrap tests with correct clustering unit.

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

1. Entity definitions and aggregation (macro vs micro) affect difficulty rankings.  
2. Force-balance assumptions and parameter choices condition residual headroom.  
3. Driving context (urban/highway mix, climate, vehicle class balance) may limit external vehicle fleets.  
4. Exact numeric tables must be locked from the VED study package before camera-ready.  
5. If VED study used different statistical rigor than AeroTwin’s 10k flight bootstrap, state the difference explicitly—do not pretend identical inference.

## K.4 Cross-domain comparison limits

1. Different units and scales.  
2. Different entity hierarchies (ICAO type vs vehicle ID).  
3. AeroTwin coarser entity holdout not run (deferred after residual gate).  
4. “Transfer” here means **qualitative structural lessons**, not weight transfer or identical pipelines.

---

# Part L — Updated conclusions (draft bullets)

1. Cumulative energy prediction under partial observation admits a common **structural menu** (Direct / Residual / Rate) but **not** a common winner.  
2. **AeroTwin:** hybrid physics–ML is effective; energy and dynamic mass matter; residual loses to Direct; official Combined **221.33 kg**; entity (type) shift is severe.  
3. **VED:** residual corrects force-balance physics but does not consistently dominate Direct; rate is unreliable; entity evaluation is essential and can be non-monotonic in grain.  
4. **Cross-domain:** entity-aware protocols and multi-metric reporting transfer as lessons; residual and rate recipes do **not**.  
5. Future work should expand entity ladders, stress-test physics anchors under shift, and treat structural choice as an empirical, protocol-conditioned decision.

---

# Part M — Future work (two-domain aware)

| Priority | Idea |
|----------|------|
| High | Fill VED numeric tables/figures into this package from the completed study |
| High | Standardized reporting template: Direct/Residual/Rate × IID/entity × MAE/RMSE |
| Medium | Coarser entity holdout on AeroTwin (body/family) for Phenomenon A only |
| Medium | Physics-quality diagnostics that predict when residual is safe |
| Medium | Shared synthetic cumulative-prediction stress tests (controlled mass/coeff error) |
| Low | Additional domains (maritime, rail) with same structural menu |
| Low | Deep sequence models with strict entity leakage controls |

---

# Part N — Figure plan for the **two-domain** paper (integrated)

## N.1 Main text (suggested ~12–14 figures)

| ID | Content | Source |
|----|---------|--------|
| **Fig. 1** | Shared structural strategies diagram (Direct / Residual / Rate) | **New** |
| **Fig. 2** | Two-domain evaluation axes (random vs entity vs temporal) | **New** |
| **Fig. 3–6** | AeroTwin data + physics + hybrid scatter (subset of AT figs) | `research.md` / `figures/` |
| **Fig. 7–8** | AeroTwin ablations + residual rejection | existing |
| **Fig. 9** | AeroTwin official + gap-closing | existing |
| **Fig. 10** | AeroTwin LOTO / entity | existing |
| **Fig. 11–13** | VED overview + strategy IID + entity | **VED exports needed** |
| **Fig. 14** | VED entity granularity (non-monotonic) | **Needed** |
| **Fig. 15** | Cross-domain strategy heatmap / transfer summary | **New** |

Appendix: remaining AeroTwin SHAP, DASHlink, full VED diagnostics.

## N.2 Main text tables

| ID | Content |
|----|---------|
| **Table 1** | Two-domain task comparison |
| **Table 2** | Strategy definitions (shared) |
| **Table 3–6** | AeroTwin core results (official, ablation, LOTO) |
| **Table 7–9** | VED core results (IID, entity, granularity) |
| **Table 10** | Cross-domain transfer matrix (Section I.1) |
| **Table 11** | Claims checklist / protocol constants |

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

- [ ] 32,536 trips / 384 vehicles stated  
- [ ] Residual > physics; residual ≰ Direct (MAE) stated  
- [ ] Rate not reliable stated  
- [ ] Entity harder + non-monotonic difficulty stated  
- [ ] Numeric tables filled from VED study (`[VED]` cleared)  
- [ ] VED figures V1–V8 produced  

### Cross-domain integrity

- [ ] Comparison table present  
- [ ] No universal residual/rate claim  
- [ ] Units not falsely equated  
- [ ] Transfer = qualitative lessons only  

### Claims policy pass

- [ ] Section J.1 all supported  
- [ ] Section J.2 none violated  
- [ ] Soft claims hedged  

---

# Part P — Narrative thread (writing order)

1. **Hook:** Cumulative energy matters in air and on road; data are partial; physics is imperfect.  
2. **Menu:** Direct / Residual / Rate look transferable—are they?  
3. **Case 1 deep dive:** AeroTwin—what works (hybrid, energy, mass) and what fails (residual, sparse myth).  
4. **Case 2 deep dive:** VED—residual helps physics, not necessarily Direct; rate falters; entity protocols bite.  
5. **Bridge:** Side-by-side matrix; shared protocol lessons; non-transfer of architecture defaults.  
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

## Q.2 VED (findings-locked)

| Item | Value |
|------|------:|
| Trips | **32,536** |
| Vehicles | **384** |
| Residual vs pure physics | Residual better |
| Residual vs Direct (MAE) | Not consistently better |
| Rate-then-integrate | Not reliable |
| Entity vs IID | Entity harder |
| Entity grain difficulty | Not strictly monotonic |
| Cross-domain architecture transfer | No strong evidence |

## Q.3 One-sentence thesis

> **Structural strategies for cumulative energy prediction must be re-validated per domain and per evaluation protocol; residual correction and rate-then-integrate are useful tools in some regimes but are not portable defaults from aircraft fuel estimation to vehicle energy estimation.**

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
| VED case-study package | _[path/citation to fill — source of VED decimals]_ |

---

# Part S — Optional venue-specific reshaping

| Venue focus | Emphasize | Compress |
|-------------|-----------|----------|
| Aviation (JOAS) | Case Study 1 full; VED as comparative appendix/section | VED methods detail |
| Transportation / ITS | Equal case studies; energy framing | PRC winner comparison detail |
| Physics-informed ML workshop | Structural strategies + transfer failure | Fleet operational detail |
| Evaluation / ML methodology | Protocol, entity shift, metric flips | Domain physics depth |

---

*End of two-domain writing package. Preserve AeroTwin freezes; fill VED `[VED]` numeric cells from the completed vehicle study before submission; keep cross-domain claims modest and protocol-aware.*
