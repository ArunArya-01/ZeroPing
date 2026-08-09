# AeroTwin Project Status Report

**Date:** July–August 2026 (last update: **2026-08-09**)  
**Repository:** ZeroPing (AeroTwin)  
**Dataset:** [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) (EUROCONTROL PRC 2025)  
**Current phase:** **Paper Writing**

This document is the **single source of truth** for project status after completion of all experiments. It summarizes completed research for paper writing. It does **not** propose new methods or experiments.

---

## Executive Summary

AeroTwin began as a **physics-informed machine learning system** for interval-level aircraft fuel-burn prediction on EUROCONTROL PRC trajectory data. A strong **ensemble teacher** (GBDT bases + Ridge meta-learner, R3 dynamic mass, phase calibration) was frozen for **knowledge distillation** into compact MLP students for deployment.

Under standard **Flight Holdout** and **Combined Rank+Final** evaluation, distillation succeeded: the **Large MLP** student (~2.89M parameters, α=0.1, β=0.9) reached Final RMSE **215.85 kg**, within ~2 kg of the teacher, at far lower latency.

**Distribution-shift evaluation** by aircraft type (type-macro RMSE) revealed a different picture. Student–teacher gaps widened under entity-level shift, and **architecture ranking reversed**: the **FT-Transformer** underperformed Large on Flight Holdout but became the best student under type-macro. Body-macro did not reverse rankings the same way.

A **systematic mechanism investigation** followed:

| Phase | Question | Result |
|-------|----------|--------|
| 1 | Teacher uncertainty / adaptive KD (VGKD)? | Predicts difficulty; **does not** fix robustness |
| 2 | Representation / attribution differences? | FT geometry and features **differ** from Large |
| 3 | Physics-feature reliance vs representation? | Physics reliance **rejected**; representation **partial** |
| 3.5 | Is local smoothness **causal**? | **Not established** (consistency intervention) |
| **Attention routing** | Does FT attention explain type-macro advantage vs Large? | **Rejected** (associates with absolute difficulty, not relative advantage) |

No further experiments or method development are planned under the current charter. The project has transitioned to writing an **empirical research paper** documenting the phenomenon, the evaluation lesson, and the mechanisms tested and ruled out.

---

## Current Status

| Field | Value |
|-------|-------|
| **Current phase** | **Paper Writing** |
| Experimentation | **Complete** |
| Method development | **Complete** (deployment model frozen; no new algorithm) |
| Mechanism investigation | **Complete** |
| Further experiments planned | **None** |
| Phase 4 method work | **Not started** (not recommended by Decision Gate) |

### Official deployment model (frozen)

| Field | Value |
|-------|------|
| Model | **Large MLP** (~2.89M params) |
| KD | α=0.1, β=0.9 |
| Final RMSE | **215.85 kg** |
| Combined RMSE | **225.95 kg** |
| Checkpoint | `results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt` |
| Status | **Frozen** |

### Reference teacher (frozen)

| Field | Value |
|-------|------|
| Combined RMSE | **221.33 kg** |
| Final RMSE (audit) | **213.62 kg** |
| Status | **Frozen** |

---

## Completed Phases

### Engineering foundation (pre–Phase 0)

Physics features, OpenAP baseline, weather/operational features, R3 dynamic mass teacher, KD pipeline, α/β sweep, capacity scaling (Tiny→XLarge), latency benchmarks, Final and Combined evaluation.

**Outcome:** Production-oriented Large MLP student; teacher–student Final parity within ~2 kg.

---

### Phase 0 — Distribution Shift Discovery

**Objective:** Evaluate students under aircraft-type and body-class protocols, not only frequency-weighted Flight Holdout.

**Outcome:**

- Architecture **ranking reversal**: Large best on Final; FT best on type-macro.
- Type-macro student–teacher gap larger than Flight gap for Large (+13.8 kg type-macro vs ~+2 kg Final).
- Body-macro does **not** show the same ranking reversal.

**Report:** `docs/reports/distribution_shift_diagnosis.md`

---

### Phase 1 — Teacher Uncertainty

**Objective:** Test whether teacher ensemble disagreement explains robustness differences and whether adaptive KD helps.

**Phase 1A:** Disagreement correlates with absolute error (Spearman ≈ **0.43** for Large |error|).

**Phase 1B (VGKD):** Uncertainty-weighted teacher strength (λ>0) **worsens or fails to improve** type-macro vs fixed KD.

**Conclusion:** Uncertainty **predicts difficulty** but is **rejected** as a primary lever for closing the robustness gap.

**Reports:** `docs/reports/teacher_uncertainty_analysis.md`, `docs/reports/vgkd_results.md`

---

### Phase 2 — Transformer Robustness Analysis

**Objective:** Characterize *why* FT wins type-macro (no retrain; frozen Large + FT).

**Outcome:**

- FT MAE gains concentrate on rare / heavy segments; Final favors common narrow-body mass.
- FT representations: rare aircraft **closer** to common centroids (raw rare→common ≈ **7.0** vs Large **21.8**).
- FT less type-clustered (silhouette); not “tighter type clusters.”
- Attribution: Large higher physics/mass share; FT higher trajectory/duration share.

**Report:** `docs/reports/transformer_robustness_analysis.md`

---

### Phase 3 — Mechanism Validation

**Objective:** Distinguish Hypothesis A (representation) vs B (physics-feature reliance) with targeted ablations and joint analysis.

**Outcome:**

- Removing 33 physics/mass/energy features: Large type-macro **essentially unchanged** (270.61 → 269.67); FT still better without physics.
- Physics baseline error strongly predicts **absolute** RMSE for both models; does **not** predict FT **advantage** (Spearman ≈ −0.10).
- Representation: normalized rare→common lower for FT (0.52 vs 0.68); geometry weakly predicts FT *advantage* (ρ ≈ 0.09).

**Conclusion:** Physics-feature reliance **rejected**. Representation **partially supported**, not sufficient as a closed causal account.

**Report:** `docs/reports/mechanism_validation.md`

---

### Phase 3.5 — Final Causal Intervention (Smoothness)

**Objective:** One final causal test—does inducing local smoothness in Large via prediction consistency improve type-macro?

**Method:** `L = L_KD + λ ‖f(x) − f(x+ε)‖²` on continuous features only; λ ∈ {0.01, 0.1, 1.0}; select by val RMSE.

**Outcome:**

- Selected λ=0.1: smoothness not meaningfully increased; type-macro Δ ≈ **−2.6 kg** → pre-registered **Outcome C**.
- Continuous-only noise revises prior “~4× smoother” claim to ~**1.25×** (OHE jitter had inflated the gap).
- λ=0.01 can improve type-macro without smoother embeddings → smoothness **not established as causal**.

**Conclusion:** Local smoothness **not supported** as the primary causal mechanism. **No Phase 4 method.** Mechanism investigation **closed.**

**Report:** `docs/reports/smoothness_causal_intervention.md`

---

### Phase — Attention Routing Analysis — **COMPLETE (2026-08-09)**

**Objective:** Test **H-Attention** — whether FT attention-based feature routing is associated with FT’s **relative** type-level advantage over Large (analysis-only; frozen checkpoints).

**Method:** CLS attention extraction (`forward_with_attention`); pre-registered metrics (entropy, top-1 mass, aircraft-cat / physics / trajectory mass, JS shift from common); type-level Spearman + bootstrap CI; body-macro negative control; prediction invariance check (max |Δ| = 0).

**Outcome:**

| Test | Result |
|------|--------|
| Strongest primary metric vs FT advantage | `aircraft_cat_mass` ρ = **−0.23** CI [−0.66, 0.31] p = 0.41 (n=15) |
| Same / related metrics vs FT **absolute** RMSE | e.g. entropy ρ = **−0.79**, top-1 ρ = **+0.81** |
| Body control | No body-level ranking reversal (Large still better on body-group RMSE) |

**Decision:** **C — Rejected** as explanation of the architecture ranking reversal. Attention tracks type **difficulty** for FT, not **relative** FT-vs-Large advantage.

**Report:** `docs/reports/attention_routing_analysis.md`  
**Artifacts:** `results/distillation/attention_routing/`  
**Script:** `experiments/08_distillation/18_attention_routing_analysis.py`

---

## Major Scientific Findings

1. **Architecture rankings reverse** under aircraft-type (entity-level) distribution shift: Large best on Flight Holdout; FT best on type-macro.
2. **Standard Flight Holdout is insufficient** for choosing a deployment model if type-equal robustness matters.
3. **Teacher uncertainty** predicts prediction difficulty but **does not explain** or close the robustness gap (VGKD negative).
4. **Physics-feature reliance** does **not** explain the robustness gap (physics ablation).
5. **Representation geometry differs** between FT and Large (rare–common proximity, scale-normalized metrics, stability under continuous noise) but is **not sufficient** to fully explain FT’s relative advantage.
6. **Local smoothness** was **not established as causal** under the pre-registered consistency intervention.
7. **Attention routing** does **not** explain FT’s type-macro advantage over Large (rejected under pre-registered association tests; attention tracks absolute difficulty).
8. **Deployment recommendation remains Large MLP** under the project’s production criteria (Final / Combined + latency), with type-macro reported as a robustness diagnostic.

### Key quantitative anchors

| Quantity | Value |
|----------|------:|
| Teacher Combined RMSE | 221.33 kg |
| Teacher Final RMSE | 213.62 kg |
| Large Final / Combined | 215.85 / 225.95 kg |
| FT Final | 224.12 kg |
| Large type-macro | 270.61 kg |
| FT type-macro | 261.15 kg |
| Large type-macro gap vs teacher | +13.82 kg |
| FT type-macro gap vs teacher | +4.35 kg |
| Physics ablation Large type-macro Δ | −0.9 kg |
| Cons. reg. selected type-macro Δ | −2.6 kg |
| Rare→common centroid (raw) Large / FT | 21.8 / 7.0 |
| Rel. emb. move continuous Large / FT | 0.017 / 0.014 |

---

## Final Conclusions

### Supported findings

- Distillation achieves near-teacher **IID** accuracy with a compact MLP.
- Under **type-macro**, student robustness is architecture-dependent; FT is the strongest student.
- Evaluation protocol choice (frequency-weighted vs type-equal) **changes model ranking**.
- Teacher disagreement is **informative** about difficulty.
- FT and Large **differ** in latent geometry and feature attribution.

### Rejected hypotheses

| Hypothesis | Status |
|------------|--------|
| Adaptive uncertainty-weighted KD (VGKD) recovers type-macro robustness | **Rejected** |
| Large’s type-macro deficit is primarily due to physics-feature reliance | **Rejected** |
| Local smoothness (as induced by prediction consistency) is the causal driver of FT’s type-macro advantage | **Not supported** |
| Attention-based feature routing explains FT’s relative type-macro advantage over Large | **Rejected** |

### Open questions

- What is the **primary causal** mechanism of the architecture-dependent ranking reversal?
- Would the ranking reverse under other entity-level protocols (e.g., leave-one-type-out training)?
- How general is the phenomenon beyond this fuel-burn tabular setting?

These remain **open**. The paper should state them as open, not invent answers.

---

## Remaining Work

Only the following items remain:

| Task | Status |
|------|--------|
| Paper writing | **Active** |
| Figure preparation (publication set) | Pending |
| Repository cleanup / artifact index | Pending |
| Camera-ready release package | Pending |

**Do not** include future method development, new KD algorithms, or additional mechanism experiments under the current project plan.

Master writing plan: **`docs/reports/PAPER_WRITING_GUIDE.md`**

---

## Key reports & artifacts

| Topic | Path |
|-------|------|
| **This status file** | `docs/reports/PROJECT_STATUS_REPORT.md` |
| **Experiment results catalog (all numbers)** | `docs/reports/EXPERIMENT_RESULTS_CATALOG.md` |
| **Paper writing guide** | `docs/reports/PAPER_WRITING_GUIDE.md` |
| Model summary | `docs/reports/CURRENT_MODEL_SUMMARY.md` |
| Final eval | `docs/reports/test_evaluation.md` |
| Combined eval | `docs/reports/combined_evaluation.md` |
| FT experiment | `docs/reports/ft_transformer_experiment.md` |
| Shift diagnosis | `docs/reports/distribution_shift_diagnosis.md` |
| Uncertainty | `docs/reports/teacher_uncertainty_analysis.md` |
| VGKD | `docs/reports/vgkd_results.md` |
| Phase 2 robustness | `docs/reports/transformer_robustness_analysis.md` |
| Phase 3 mechanism | `docs/reports/mechanism_validation.md` |
| Phase 3.5 smoothness | `docs/reports/smoothness_causal_intervention.md` |
| **Attention routing** | **`docs/reports/attention_routing_analysis.md`** |
| Teacher audit | `docs/reports/teacher_evaluation_report.md` |

### Results roots

| Stream | Path |
|--------|------|
| Capacity / Large deploy | `results/distillation/capacity_scaling/` |
| FT-Transformer | `results/distillation/ft_transformer/` |
| Shift diagnosis | `results/distillation/distribution_shift_diagnosis/` |
| Uncertainty | `results/distillation/uncertainty_analysis/` |
| VGKD | `results/distillation/vgkd/` |
| Phase 2 | `results/distillation/transformer_robustness/` |
| Phase 3 | `results/distillation/mechanism_validation/` |
| Phase 3.5 | `results/distillation/smoothness_causal/` |
| Attention routing | `results/distillation/attention_routing/` |

---

## Decision history (closed)

| Gate | Decision |
|------|----------|
| After Phase 0 | Investigate mechanism; do not only chase Final RMSE |
| After Phase 1B | Do not deploy VGKD |
| After Phase 3 | Physics reliance rejected; representation partial |
| After Phase 3.5 | **No Phase 4 method**; write empirical paper |
| After Attention routing | H-Attention **rejected**; continue paper (no attention-based method) |

---

*All experimental work for this research arc is complete. Documentation above is intended to support paper writing only.*
