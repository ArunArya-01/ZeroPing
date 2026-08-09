# AeroTwin — Paper Writing Guide

**Purpose:** Master planning document for writing the research paper after all experiments are complete.  
**Status:** Experimentation closed · Method development closed · Mechanism investigation closed  
**Audience:** Authors preparing an empirical ML paper (ICML / NeurIPS / domain venue style)  
**Evidence rule:** Every claim must map to a completed experiment or frozen metric. Do not invent methods or re-open mechanism phases.

**Companion status file:** `docs/reports/PROJECT_STATUS_REPORT.md`

---

# Project Summary

AeroTwin is a physics-informed tabular learning system for aircraft fuel-burn prediction on EUROCONTROL PRC trajectory intervals. A frozen GBDT ensemble teacher is distilled into compact neural students for deployment. Under standard Flight Holdout and Combined evaluation, a Large MLP student nearly matches the teacher. Under aircraft-type (type-macro) evaluation, rankings reverse: an FT-Transformer student is more robust than the MLP despite worse IID accuracy. Systematic investigation rules out teacher-uncertainty adaptive KD, physics-feature reliance, and local smoothness as primary explanations of this architecture-dependent robustness gap, while documenting genuine representation differences that remain insufficient as a full causal account. The paper is an empirical study of evaluation protocols, negative mechanism results, and open questions—not a new algorithm paper.

---

# One-Sentence Contribution

We show that, for distilled tabular fuel models, **architecture ranking reverses under aircraft-type distribution shift**, and that several plausible explanations—adaptive teacher uncertainty, physics-feature reliance, and local smoothness—**do not account for the robustness gap**, leaving the mechanism open while demonstrating that **IID deployment metrics alone can select the wrong robustness ranking**.

---

# Elevator Pitch

Modern aviation fuel models must be accurate under standard holdouts and reliable across aircraft types that appear unevenly in training data. We distill a strong physics-informed ensemble into neural students and obtain excellent Flight Holdout performance with a Large MLP. When we re-evaluate with type-macro RMSE—an equal-weight average across aircraft types with sufficient samples—the ranking flips: an FT-Transformer becomes the best student. This is not a minor metric quirk; the student–teacher gap for the MLP widens substantially under type-macro relative to Flight. We then ask *why*. Adaptive knowledge distillation guided by teacher ensemble disagreement fails. Removing physics-derived features fails to close the MLP’s type-macro gap. Representation analyses show that FT places rare types closer to a common manifold and attributes less to physics, yet geometry does not strongly predict FT’s *relative* advantage, and a causal consistency-regularization intervention does not establish local smoothness as the driver. The contribution is therefore empirical and cautionary: entity-level evaluation changes architecture choices; several natural mechanisms are ruled out; and the residual open question is scientifically useful. We freeze a production MLP for IID deployment and report type-macro as a robustness diagnostic, without claiming a solved causal theory or a new training algorithm.

---

# Candidate Titles

Emphasize empirical investigation, architecture-dependent robustness, and distribution shift. Underspecification appears only as a **positioning option**, not as a proven claim.

1. **Architecture Ranking Reversal under Aircraft-Type Shift in Distilled Tabular Fuel Models**
2. **When IID Deployment Metrics Mislead: Entity-Level Robustness of MLP vs Transformer Students**
3. **An Empirical Study of Architecture-Dependent Robustness under Structured Tabular Distribution Shift**
4. **Knowledge Distillation Meets Entity Shift: Ranking Reversal and Ruled-Out Mechanisms**
5. **Flight Holdout Is Not Enough: Type-Macro Evaluation of Distilled Aircraft Fuel Predictors**
6. **Negative Results on Uncertainty, Physics Features, and Smoothness for Transformer Robustness under Type Shift**
7. **Representation Differences without a Causal Story: MLP vs FT-Transformer under Aircraft-Type Shift** *(honest / conservative)*
8. **Toward Fair Entity-Level Evaluation of Tabular Aviation Models** *(evaluation-angle)*
9. **Underspecification-Style Ranking Sensitivity under Aircraft-Type Evaluation** *(positioning-only; avoid if reviewers demand a formal underspecification theorem)*
10. **What Does Not Explain Transformer Robustness under Type Shift: An Empirical Investigation**

**Recommended working title (conservative):**  
*Architecture Ranking Reversal under Aircraft-Type Distribution Shift: An Empirical Investigation of Distilled Tabular Fuel Models*

---

# Research Question

**Primary:**  
Why does FT-Transformer become a more robust student than Large MLP under aircraft-type distribution shift, despite worse Flight Holdout accuracy—and which candidate mechanisms does the evidence support or reject?

**Secondary (evaluation):**  
Does frequency-weighted Flight Holdout ranking agree with type-equal robustness ranking for deployment model selection?

---

# Motivation

### Why standard IID evaluation can be misleading

Flight Holdout and Combined Rank+Final weight samples by prevalence. Common narrow-body types dominate. A model can look best overall while degrading on rare or heavy types that matter for operational coverage and safety-adjacent monitoring.

### Why deployment robustness matters

Aviation fuel estimates feed planning, emissions accounting, and operational analytics. Systematic errors on underrepresented aircraft types are not absorbed by average RMSE. Entity-level diagnostics (type-macro) stress-test transfer across aircraft identities.

### Why architecture ranking reversals are scientifically interesting

If two architectures trained under the same KD protocol reverse order under a fixed, pre-specified shift protocol, then:

1. Architecture choice is **evaluation-dependent**.
2. The system is **underspecified** with respect to that evaluation axis in the *practical* sense that multiple solutions fit IID data differently under shift (positioning language only—not a claim of the full D’Amour et al. formal setting unless carefully scoped).
3. Mechanism work can proceed without inventing a new model class first.

---

# Story of the Paper

```
Problem
  Physics-informed fuel prediction; need accurate + deployable students
        ↓
Strong teacher + KD
  Large MLP ≈ teacher on Flight / Combined
        ↓
Observation (Phase 0)
  Type-macro: FT best student; ranking reverses; MLP gap widens
        ↓
Hypothesis 1 — Teacher uncertainty
  Disagreement predicts difficulty
  VGKD adaptive weighting
        ↓
Rejected (Phase 1)
  Informative but not a successful robustness lever
        ↓
Hypothesis 2 — Physics-feature reliance
  Large over-relies on OpenAP/mass/energy
        ↓
Rejected (Phase 3 ablation)
  Removing physics does not close type-macro gap
        ↓
Hypothesis 3 — Representation geometry
  FT smoother / rare closer to common manifold
        ↓
Partially supported (Phase 2–3)
  Differences real; weak link to FT *advantage*
        ↓
Causal intervention — Local smoothness (Phase 3.5)
  Consistency regularization on Large
        ↓
Not supported
  Selected model: smoothness not ↑; gains not tied to smoothness
        ↓
Open question
  Residual mechanism of architecture-dependent type-macro robustness
  Paper contribution = phenomenon + evaluation lesson + negative results
```

---

# Timeline of Research

| Order | Phase | Content | Status |
|------:|-------|---------|--------|
| 0 | Data & teacher | PRC data, physics/weather features, R3 teacher freeze | Done |
| 1 | Distillation | α/β sweep, capacity scaling, Large deploy freeze | Done |
| 2 | Protocols | Final, Combined, type-macro, body-macro | Done |
| 3 | Phase 0 | Distribution-shift diagnosis; ranking reversal | Done |
| 4 | Phase 1A | Teacher disagreement vs error | Done |
| 5 | Phase 1B | VGKD negative result | Done |
| 6 | Phase 2 | Geometry, attribution, error localization | Done |
| 7 | Phase 3 | Physics ablation; joint variance decomposition | Done |
| 8 | Phase 3.5 | Consistency / smoothness causal test | Done |
| 9 | **Paper** | Write-up, figures, camera-ready | **Active** |

---

# Main Contributions

State only what the evidence supports.

1. **Empirical finding:** For the same KD teacher and data, **MLP vs FT student ranking reverses** between Flight Holdout and type-macro evaluation.
2. **Evaluation lesson:** Frequency-weighted holdouts can **hide entity-level robustness gaps** that change model selection.
3. **Negative result (uncertainty):** Teacher disagreement correlates with difficulty; **uncertainty-guided adaptive KD (VGKD) does not improve** type-macro robustness.
4. **Negative result (physics):** **Physics-feature ablation** does not explain Large’s type-macro deficit relative to FT.
5. **Partial positive characterization:** FT and Large **differ systematically** in representation geometry and feature attribution; these differences are documented but **not shown to be fully causal**.
6. **Negative / null causal test (smoothness):** Prediction consistency regularization **does not establish** local smoothness as the causal mechanism of the robustness gap.
7. **Deployment honesty:** Report Large as best **IID/production** student and FT as best **type-macro** student without claiming a solved theory.

**Do not claim:** a new SOTA algorithm; a complete causal theory; universal tabular OOD laws; formal underspecification proofs without careful scoping.

---

# Experimental Summary

| Experiment | Objective | Method | Outcome | Conclusion |
|------------|-----------|--------|---------|------------|
| Teacher freeze (R3) | Strong frozen teacher | Ensemble + dynamic mass + P1E | Combined 221.33; Final ~213.6 | Teacher fixed for all students |
| KD α/β + capacity | Best deploy student | MLP sizes; α/β sweep | Large α=0.1 β=0.9 → Final 215.85 | Deploy Large |
| Combined eval | PRC parity protocol | Rank+Final concat | Large Combined 225.95 | Large stable across protocols for IID |
| Phase 0 shift | Entity-level robustness | Type-macro / body-macro | Ranking reverse; Large gap widens | Phenomenon established |
| Phase 1A uncertainty | Is disagreement informative? | Ensemble std vs \|error\| | Spearman ~0.43 | Predictive, not causal proof |
| Phase 1B VGKD | Adaptive KD helps? | β(u) = β exp(−λu) | λ>0 fails type-macro | Reject adaptive uncertainty KD |
| Phase 2 geometry | How do reps differ? | PCA/UMAP, centroids, attribution | Rare closer in FT; attr. differs | Characterization |
| Phase 3 physics ablation | Physics reliance causal? | Retrain w/o 33 phys. feats | Type-macro gap persists | Reject H-B |
| Phase 3 correlations / C1 | Joint factors | Type-level regression | Physics ↛ FT advantage | Supports rejection of H-B |
| Phase 3.5 consistency | Smoothness causal? | λ ∈ {0.01,0.1,1.0} cons. loss | Selected: no smooth ↑; gains not tied to smooth | Not supported as causal |

---

# Hypothesis Summary

| Hypothesis | Prediction | Experiment | Outcome | Conclusion |
|------------|------------|------------|---------|------------|
| **H1 Uncertainty** | High disagreement → hard cases; reweighting teacher helps robustness | Phase 1A/1B VGKD | Correlation yes; VGKD no | **Rejected** as primary fix |
| **H2 Physics reliance** | Removing physics closes Large type-macro gap; physics error ↑ → FT advantage ↑ | Phase 3 A1–A3 | Ablation null; corr. with advantage null | **Rejected** |
| **H3 Representation** | FT geometry explains type-macro advantage | Phase 2–3 B/C | Differences yes; advantage link weak | **Partially supported** |
| **H4 Smoothness causal** | Inducing smoothness in Large improves type-macro | Phase 3.5 | Selected Outcome C; sweep not causal | **Not supported** |
| **H5 Attention routing** | Attention behavior predicts FT type-level *advantage* over Large | Attention analysis (frozen FT) | Primary |ρ|≤0.23 vs advantage; strong vs absolute RMSE | **Rejected** |

---

# Key Results

### Deployment and teacher

| Model | Final RMSE | Combined RMSE | Notes |
|-------|-----------:|--------------:|-------|
| R3 Teacher | 213.62 | **221.33** | Frozen |
| Large MLP | **215.85** | **225.95** | Deploy |
| XLarge MLP | 218.59 | — | Not deploy |
| FT-Transformer | 224.12 | — | ~1.46M params |

### Ranking reversal (type-macro, n≥50)

| Model | Type-macro RMSE | Gap vs teacher type-macro |
|-------|----------------:|--------------------------:|
| Teacher | 256.79 | 0 |
| Large | 270.61 | +13.82 |
| FT | **261.15** | +4.35 |

### Mechanism highlights

| Result | Number / fact |
|--------|----------------|
| Spearman(disagreement, Large \|err\|) | ≈ 0.43 |
| VGKD preferred | λ = 0 only (fixed KD) |
| Large type-macro after physics removal | 269.67 (Δ ≈ −0.9) |
| FT type-macro after physics removal | 263.70 (still better than Large) |
| Spearman(physics RMSE, FT advantage) | ≈ −0.10 (n.s.) |
| Rare→common raw Large / FT | 21.8 / 7.0 |
| Rare→common norm Large / FT | 0.68 / 0.52 |
| Rel. emb. move (continuous) Large / FT | 0.017 / 0.014 |
| Cons. λ=0.1 type-macro | 268.02 (Δ −2.6 vs Large) |
| Cons. λ=0.01 type-macro | 262.39 without smooth ↑ |

---

# Related Work Positioning

Tone: **how this differs**, not “we solve X.”

### Underspecification (D’Amour et al. and follow-ons)

- **Use as positioning:** Multiple models can achieve similar IID metrics yet differ under stress tests.
- **Do not claim:** A formal underspecification audit of the full aviation pipeline, or that we reproduce their exact experimental template.
- **Our difference:** Focused tabular KD students, one structured entity axis (aircraft type), aviation fuel application.

### TableShift / tabular distribution-shift benchmarks

- **Relation:** Share motivation that tabular OOD evaluation matters.
- **Difference:** We study **entity-level** (aircraft type) shift with type-macro aggregation, not only domain/time splits from TableShift datasets; we add **mechanism ablation** sequence.

### Tabular distribution-shift benchmarking generally

- **Difference:** End-to-end **distillation** setting with frozen industrial-style teacher; dual reporting of **deploy metric** vs **type-equal metric**.

### Uncertainty-aware knowledge distillation

- **Relation:** VGKD-style reweighting is in the family of confidence-aware KD.
- **Difference:** **Negative result** under type-macro; disagreement is calibrated to difficulty but **not** a successful training signal here.

### Representation geometry and OOD robustness

- **Relation:** Literature linking representation structure to OOD performance.
- **Difference:** We measure geometry differences and attempt a **causal smoothness intervention**; differences alone do not close the causal story.

### Physics-informed aviation ML

- **Relation:** OpenAP / mass models as features and baselines.
- **Difference:** Physics is a **ruled-out primary cause** of the *relative* robustness gap under our ablation, even though physics predicts absolute difficulty.

---

# Figures

Publication figure list. Prefer redrawing cleanly from existing analysis scripts; paths under `docs/reports/figures/` and `results/distillation/*/plots/`.

| ID | Working title | Communicates |
|----|---------------|--------------|
| **Fig 1** | Pipeline overview | Data → physics/teacher → KD students → dual evaluation |
| **Fig 2** | Ranking reversal bar chart | Final vs type-macro for Teacher, Large, FT (and optionally XLarge) |
| **Fig 3** | Gap inflation | Flight gap vs type-macro gap (student−teacher) |
| **Fig 4** | Error by body / rare–common | Where FT gains appear |
| **Fig 5** | Uncertainty calibration | Disagreement vs \|error\|; optional reliability bins |
| **Fig 6** | VGKD negative | Type-macro (and Final) vs λ |
| **Fig 7** | Representation geometry | PCA/UMAP or rare→common (raw + **normalized**) Large vs FT |
| **Fig 8** | Attribution shares | Physics vs trajectory by model (and heavy/narrow) |
| **Fig 9** | Physics ablation | Full vs no-physics Final and type-macro |
| **Fig 10** | Physics vs FT advantage scatter | Type-level; show null advantage correlation |
| **Fig 11** | Smoothness intervention | Consistency λ: Final, type-macro, rel. emb. move |
| **Fig 12** | (Optional) Decision schematic | Hypotheses tested → reject / partial / open |

**Camera-ready note:** Fig 2 + Fig 9 + Fig 11 are the backbone of the scientific arc.

---

# Tables

| ID | Content |
|----|---------|
| **Table 1** | Dataset splits (train / Rank / Final sizes) |
| **Table 2** | Model cards (params, latency, KD weights) |
| **Table 3** | Main results: Final, Combined, type-macro, body-macro |
| **Table 4** | Phase 0 gap table (teacher / Large / FT) |
| **Table 5** | VGKD summary (λ sweep; preferred λ=0) |
| **Table 6** | Physics ablation (Large & FT) |
| **Table 7** | Geometry metrics (normalized rare→common, purity, continuous stability) |
| **Table 8** | Hypothesis summary (this guide’s table, publication form) |
| **Table 9** | Consistency regularization full λ results |
| **Table A\*** | Appendix: type-level physics table; seed stability if space |

---

# Writing Plan

Recommended order (results-first, claims last):

1. **Figures** — freeze visual story (Fig 2, 9, 11 first)
2. **Tables** — lock numbers against frozen metrics JSON
3. **Results** — Phase 0 → 1 → 2 → 3 → 3.5, each ending with a one-sentence takeaway
4. **Methods** — data, teacher, KD, protocols, analysis definitions (type-macro n≥50)
5. **Introduction** — ranking reversal + evaluation lesson + “we test mechanisms”
6. **Related Work** — short; positioning only
7. **Discussion** — supported / rejected / open; deployment vs research metrics
8. **Limitations**
9. **Abstract** — last; no claims beyond Results
10. **Title** — choose after abstract draft

**Section-length heuristic (conference):** Intro 1.5p · Related 1p · Methods 2p · Results 3–4p · Discussion 1p · Limit 0.5p.

---

# Limitations

State explicitly in the paper:

1. **Single application domain** (aviation fuel intervals; one dataset family).
2. **Mechanism remains unresolved**—we rule out candidates; we do not identify the true cause.
3. **No new learning algorithm** is proposed or claimed as a fix.
4. **Type-macro is a post-hoc entity weighting** of a flight holdout, not leave-one-type-out training (unless separately reported).
5. **Limited type-level sample size** for correlations (≈15 types with n≥50).
6. **Single seed** for many student runs (capacity multi-seed exists for XLarge; consistency sweep is seed 42).
7. **Smoothness intervention** is prediction consistency only—not a full Lipschitz/Jacobian study (by design stopping rule).
8. **Continuous-only** vs all-feature noise changes quantitative smoothness ratios; earlier “4×” must be protocol-qualified.
9. **Combined PRC** not fully re-run for every ablation model.

---

# Future Work

Only items that **follow from the evidence** (not arbitrary methods):

1. **Richer entity-level protocols** (e.g., leave-one-type-out or type-balanced training) to test whether ranking reversal persists under stronger shifts.
2. **Broader tabular domains** with analogous entity axes (device type, product SKU) to test generality of ranking sensitivity.
3. **Causal tools beyond consistency loss** only if a new project charter reopens mechanism work—with pre-registered interventions.
4. **Evaluation standards** for aviation/tabular deployment that **require dual reporting** of frequency-weighted and entity-macro metrics.

**Do not list as future work under this paper’s promise:** new architectures, speculative hybrid KD methods, or unmotivated regularizers.

---

# Final Message

This paper contributes a carefully measured empirical fact—and an equally careful set of negative results—to machine learning practice: **when models are selected by IID holdouts alone, architecture rankings under structured entity-level shift can reverse**, and **several plausible explanations of that reversal fail under controlled tests**. Progress here is scientific honesty: document the phenomenon, rule out what the data reject, freeze a deployment model for what it optimizes, and leave the residual mechanism open rather than inventing an unearned algorithm.

---

# Quick reference — report map

| Paper section fuel | Source report |
|--------------------|---------------|
| Deploy metrics | `test_evaluation.md`, `combined_evaluation.md`, `CURRENT_MODEL_SUMMARY.md` |
| Ranking reversal | `distribution_shift_diagnosis.md` |
| Uncertainty / VGKD | `teacher_uncertainty_analysis.md`, `vgkd_results.md` |
| Geometry / attribution | `transformer_robustness_analysis.md` |
| Physics vs representation | `mechanism_validation.md` |
| Smoothness intervention | `smoothness_causal_intervention.md` |
| Project status | `PROJECT_STATUS_REPORT.md` |

---

*End of paper writing guide. Update only if new completed experiments are added under an explicit new project charter.*
