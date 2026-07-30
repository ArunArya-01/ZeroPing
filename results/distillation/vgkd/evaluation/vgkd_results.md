# Phase 1B — Variance-Guided Knowledge Distillation (VGKD)

**Date:** 2026-07-30

## Motivation

Phase 0 showed Large MLP nearly matches the teacher on Flight Final but loses robustness under type-macro evaluation. Phase 1A showed teacher ensemble disagreement correlates with prediction error. VGKD uses that signal to reduce teacher weight on uncertain samples.

## Method

```
u(x)  = std of 6 base ensemble predictions
u_n   = (u − μ_train) / σ_train     # z-score
β(x)  = β_base · exp(−λ · max(u_n, 0))
α(x)  = 1 − β(x)
L     = mean[ α(x)·(ŷ−y)² + β(x)·(ŷ−y_teacher)² ]
```

With β_base = 0.9. Samples with u ≤ train mean keep full teacher weight; uncertain samples shift toward GT.

Architecture: **Large MLP** (~2.89M). No architecture change.

---

## Preferred model

| Field | Value |
|-------|------:|
| Run | `vgkd_exp_lam0.0` |
| Final RMSE | 216.10 |
| Type-macro RMSE | 269.76 |
| Body-macro RMSE | 239.54 |
| Gap flight | +2.48 |
| Gap type-macro | +12.97 |
| Δ Final vs fixed | +0.25 |
| Δ Type-macro vs fixed | -0.85 |
| λ / weight_fn | 0.0 / exp |

### Fixed KD baseline (Large)

- Final RMSE: **215.85**
- Type-macro: **270.61**
- Gap type-macro: **+13.82**

---

## Comparison table

| Run | Final | Type-macro | Body-macro | Gap type | Δtype vs fixed | Δfinal vs fixed |
|-----|------:|-----------:|-----------:|---------:|---------------:|----------------:|
| static_beta0.9 | 216.10 | 269.76 | 239.54 | +12.97 | -0.85 | +0.25 |
| vgkd_exp_lam0.0 | 216.10 | 269.76 | 239.54 | +12.97 | -0.85 | +0.25 |
| fixed_kd_large | 215.85 | 270.61 | 239.63 | +13.82 | +0.00 | +0.00 |
| static_beta0.8 | 216.45 | 273.79 | 240.01 | +17.00 | +3.18 | +0.60 |
| vgkd_random_lam1.0 | 218.64 | 275.10 | 241.84 | +18.31 | +4.49 | +2.79 |
| static_beta0.7 | 218.87 | 280.82 | 242.43 | +24.03 | +10.21 | +3.02 |
| vgkd_exp_lam0.25 | 227.79 | 296.98 | 251.40 | +40.19 | +26.37 | +11.94 |
| vgkd_exp_lam1.0 | 232.92 | 301.05 | 257.01 | +44.26 | +30.44 | +17.07 |
| vgkd_exp_lam0.5 | 231.30 | 302.10 | 255.06 | +45.30 | +31.48 | +15.45 |
| vgkd_lin_lam0.25 | 234.04 | 309.76 | 257.77 | +52.96 | +39.14 | +18.19 |
| vgkd_lin_lam0.5 | 237.39 | 314.01 | 261.17 | +57.22 | +43.40 | +21.54 |
| vgkd_exp_lam2.0 | 236.85 | 315.53 | 261.32 | +58.74 | +44.92 | +21.00 |
| vgkd_lin_lam1.0 | 237.74 | 315.95 | 262.55 | +59.16 | +45.34 | +21.89 |
| vgkd_oracle_lam1.0 | 242.16 | 322.53 | 267.68 | +65.73 | +51.92 | +26.31 |
| vgkd_lin_lam2.0 | 240.88 | 326.58 | 265.99 | +69.79 | +55.97 | +25.03 |

---

## Figures

![beta](figures/fig_vgkd_beta_vs_uncertainty.png)

![lam](figures/fig_vgkd_lambda_sensitivity.png)

![static](figures/fig_vgkd_static_vs_adaptive.png)

![rand](figures/fig_vgkd_random_vs_true.png)

![gap](figures/fig_vgkd_teacher_student_gap.png)

![pareto](figures/fig_vgkd_pareto_final_vs_type.png)

![lin](figures/fig_vgkd_linear_vs_exp.png)

---

## Discussion

Selection prioritizes **type-macro robustness** with bounded Final regression (≤2 kg).
See comparison table for ablations (static β, random u, linear weights, oracle).

*Generated 2026-07-30T15:17:15.966457+00:00*
