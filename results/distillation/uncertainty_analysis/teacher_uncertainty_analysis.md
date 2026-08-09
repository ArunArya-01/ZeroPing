# Phase 1A — Teacher Uncertainty Validation

**Date:** 2026-07-30
**Status:** Diagnostic only (no student training)

## Objective

Determine whether **teacher ensemble disagreement** (std of 6 base kg predictions) is a reliable indicator of prediction difficulty before implementing Adaptive KD.

---

## Methodology

| Item | Value |
|------|------|
| Samples | 37,170 Final intervals / 2,824 flights |
| Disagreement | Std of 6 base ensemble predictions (pre-Ridge/P1E) |
| Teacher error | \|P1E teacher − ground truth\| |
| Student error | \|Large MLP − ground truth\| |
| Bundle | `<project_root>\cache\r3_teacher_distillation_bundle.pkl` |

---

## Descriptives

| Stat | Value |
|------|------:|
| Teacher RMSE | 213.62 |
| Large RMSE | 215.85 |
| Mean disagreement | 28.67 |
| Median disagreement | 13.50 |
| P95 disagreement | 90.86 |

---

## Correlation analysis (Flight / all Final)

### Disagreement → teacher error

| Method | r | 95% CI | p |
|--------|--:|--------|--:|
| Pearson | 0.4721 | [0.426, 0.527] | 0.00e+00 |
| Spearman | 0.4259 | [0.416, 0.435] | 0.00e+00 |
| Kendall | 0.2945 | [0.288, 0.301] | 0.00e+00 |

### Disagreement → Large student error

| Method | r | 95% CI | p |
|--------|--:|--------|--:|
| Pearson | 0.4554 | [0.393, 0.534] | 0.00e+00 |
| Spearman | 0.4352 | [0.426, 0.444] | 0.00e+00 |
| Kendall | 0.2993 | [0.293, 0.306] | 0.00e+00 |

### Type-level (per-type means)

| Relation | Pearson r | Spearman r |
|----------|----------:|-----------:|
| Disagreement vs teacher RMSE | nan | nan |
| Disagreement vs student gap | nan | nan |
| Disagreement vs train frequency | nan | nan |
| Train frequency vs student gap | nan | nan |

---

## Calibration (10 equal-frequency bins)

Bin-level Spearman(disagreement, teacher \|err\|): **0.976**

Bin-level Spearman(disagreement, Large \|err\|): **0.952**

---

## Error localization (top vs bottom 5% disagreement)

| Group | n | Mean std | Teacher RMSE | Large RMSE | Mean fuel | Mean duration |
|-------|--:|---------:|-------------:|-----------:|----------:|--------------:|
| High (top 5%) | 1859 | 216.2 | 778.5 | 781.5 | 2186 | 1475 |
| Low (bottom 5%) | 1859 | 3.0 | 39.1 | 39.3 | 121 | 272 |

High-disagreement body mix: `{'narrowbody': 392, 'regional_other': 1, 'widebody_heavy': 1466}`

High-disagreement phases: `{'climb': 138, 'cruise': 1549, 'descent': 172}`

High-disagreement top types: `[('A359', 405), ('B77W', 403), ('B744', 280), ('A332', 194), ('A20N', 189), ('B789', 84), ('A320', 79), ('B738', 71), ('A333', 36), ('B772', 32)]`

---

## Robustness prediction (type-level)

| Group | Mean student gap | Mean teacher RMSE | Mean Large RMSE |
|-------|-----------------:|------------------:|----------------:|
| Top disagreement types | +47.82 | 585.9 | 633.7 |
| Bottom disagreement types | -1.53 | 68.8 | 67.3 |
| **Δ (top − bottom)** | **+49.35** | | |

Top types: [np.str_('B744'), np.str_('B77W'), np.str_('B772')]

Bottom types: [np.str_('A21N'), np.str_('A320'), np.str_('A20N')]

---

## Figures

![hist](figures/fig_unc_disagreement_hist.png)

![t_err](figures/fig_unc_disagreement_vs_teacher_error.png)

![s_err](figures/fig_unc_disagreement_vs_student_error.png)

![cal](figures/fig_unc_calibration_curve.png)

![ac](figures/fig_unc_aircraft_disagreement_scatter.png)

![gap](figures/fig_unc_gap_vs_disagreement.png)

![freq](figures/fig_unc_train_freq_vs_disagreement.png)

![top](figures/fig_unc_top_uncertain_aircraft.png)

---

## Decision questions (evidence only)

1. **Correlate with teacher error?** Spearman r = **0.426**, Pearson r = **0.472**.
2. **Correlate with student error?** Spearman r = **0.435**, Pearson r = **0.455**.
3. **Identify difficult aircraft?** Type-level analysis and top-uncertain list above.
4. **Predict robustness failures?** Top−bottom type gap Δ = **+49.35 kg**; type-level Spearman(disagreement, gap) = **nan**.
5. **Sufficient for Adaptive KD?** **True**

---

## Recommendation

| Field | Value |
|-------|------|
| Proceed to Adaptive KD (1B)? | **True** |
| Next | Phase 1B — Adaptive Knowledge Distillation |
| Rationale | Ensemble std correlates positively with teacher error (Spearman r=0.426) and Large error (r=0.435); calibration bins are monotonic (ρ_teacher=0.976, ρ_large=0.952); high-disagreement types show gap Δ=+49.35 kg vs low-disagreement types. Adaptive KD is justified. |

---

## Artifacts

`results/distillation/uncertainty_analysis/`

*Generated 2026-07-30T13:06:59.365902+00:00*
