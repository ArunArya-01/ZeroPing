# Phase 0 — Distribution Shift Diagnosis

**Date:** 2026-07-30
**Status:** Evaluation only (frozen models; no retrain)

## Scientific question

> Does standard knowledge distillation introduce additional robustness loss under entity-level distribution shift compared to the teacher?

---

## Experimental setup

| Item | Value |
|------|------|
| Models | R3 Teacher, Large MLP, XLarge MLP, FT-Transformer |
| Retrain | **None** — frozen checkpoints only |
| Flight holdout | Final (`featured_dataset_final.parquet`) |
| Type-level (LOTO-style) | Per-type RMSE on Final; unweighted macro (n≥50) |
| Body-class | widebody_heavy / narrowbody / regional_other macros |
| Bootstrap | Flight-clustered for overall RMSE; type-resample for macro gap |

**Important protocol note:** Type-level evaluation is **post-hoc LOTO-style** (entity-level metrics on held-out flights). Models were trained with all types present in the distillation train split. This isolates *relative* robustness of frozen KD students vs teacher under entity heterogeneity without re-fitting.

---

## Metrics — all protocols

| Model | Flight RMSE | Type-macro RMSE | Body-macro RMSE | Deg. ratio (type) | Deg. ratio (body) |
|-------|------------:|----------------:|----------------:|------------------:|------------------:|
| R3 Teacher | 213.62 | 256.79 | 237.55 | 1.202 | 1.112 |
| Large MLP | 215.85 | 270.61 | 239.63 | 1.254 | 1.110 |
| XLarge MLP | 218.59 | 276.01 | 242.08 | 1.263 | 1.107 |
| FT-Transformer | 224.12 | 261.15 | 249.58 | 1.165 | 1.114 |

### Error inflation (shift − flight)

| Model | Inflation type-macro | Inflation body-macro |
|-------|---------------------:|---------------------:|
| R3 Teacher | +43.17 | +23.93 |
| Large MLP | +54.76 | +23.78 |
| XLarge MLP | +57.43 | +23.50 |
| FT-Transformer | +37.02 | +25.46 |

### Teacher–student gap (student − teacher RMSE)

| Student | Gap flight | Gap type-macro | Gap body-macro | Gap increase (type − flight) |
|---------|-----------:|---------------:|---------------:|-----------------------------:|
| Large MLP | +2.23 | +13.82 | +2.08 | +11.59 |
| XLarge MLP | +4.96 | +19.22 | +4.53 | +14.26 |
| FT-Transformer | +10.50 | +4.35 | +12.03 | -6.15 |

### Bootstrap uncertainty (Large vs Teacher)

- Flight gap: **+2.23 kg**, 95% CI **[-3.70, +8.34]** (excludes 0? **False**)
- Type-macro gap: **+13.82 kg**, 95% CI **[+0.05, +35.11]** (excludes 0? **True**)
- Gap increase (type − flight): **+11.59 kg**

---

## Ranking stability

| Protocol | Ranking (best → worst) |
|----------|------------------------|
| Flight | R3 Teacher → Large MLP → XLarge MLP → FT-Transformer |
| Type-macro | R3 Teacher → FT-Transformer → Large MLP → XLarge MLP |
| Body-macro | R3 Teacher → Large MLP → XLarge MLP → FT-Transformer |

---

## Figures

![rmse](figures/fig_shift_rmse_all_protocols.png)

![degradation](figures/fig_shift_degradation_ratio.png)

![gap](figures/fig_shift_teacher_student_gap.png)

![inflation](figures/fig_shift_error_inflation.png)

![rank](figures/fig_shift_ranking_stability.png)

![family](figures/fig_shift_aircraft_family_robustness.png)

---

## Interpretation (evidence only)

1. **Does KD lose robustness under shift?** Degradation ratios (type-macro/flight): Teacher **1.202**, Large **1.254**, XLarge **1.263**, FT **1.165**.
2. **Does the teacher degrade less?** Compare inflation and ratios above.
3. **Does teacher–student gap increase?** Large: flight gap **+2.23** → type-macro gap **+13.82** (Δ **+11.59** kg).
4. **Is Large still the most robust student?** Best student under type-macro: **FT-Transformer** (full order: R3 Teacher → FT-Transformer → Large MLP → XLarge MLP).
5. **Does FT become relatively stronger under shift?** Compare FT rank flight vs type-macro.
6. **Statistically meaningful?** Large flight gap CI excludes 0? **False**. Type-macro gap CI excludes 0? **True**.

---

## Decision gate

| Field | Value |
|-------|------|
| Proceed to Adaptive KD? | **True** |
| Next phase | Phase 1 — Adaptive / Uncertainty-Aware KD |
| Rationale | Large student gap widens under type-level evaluation (flight gap +2.23 → LOTO-macro gap +13.82, Δ=+11.59 kg). Adaptive KD is justified to investigate. |

---

## Artifacts

`results/distillation/distribution_shift_diagnosis/`

*Generated 2026-07-30T12:40:29.044664+00:00*
