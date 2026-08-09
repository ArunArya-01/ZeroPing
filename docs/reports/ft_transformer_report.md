# FT-Transformer Student — Training & Evaluation Report

**Date:** 2026-07-30
**Phase:** 2 — architecture experiment under frozen KD pipeline

Only the student architecture changed. Teacher, data, split, α=0.1, β=0.9, and
preprocessing match the Large MLP baseline.

---

## Architecture

| Item | Value |
|------|------:|
| Architecture | FT-Transformer (Gorishniy et al. 2021) |
| Parameters | 1,458,625 |
| Checkpoint | `<project_root>\results\distillation\ft_transformer\ft_transformer_kd1\best_model.pt` |
| KD weights | α=0.1, β=0.9 |
| Val RMSE (flight holdout) | 236.08 |

Implementation: `src/aerotwin/distillation/models/ft_transformer.py`
Factory: `build_student('ft_transformer', in_dim=…)`

---

## Official metrics

| Model | Rank RMSE | Final RMSE | Combined RMSE | Params |
|-------|----------:|-----------:|--------------:|-------:|
| R3 Teacher | 232.53 | 213.62 | 221.33 | ensemble |
| Large MLP (baseline) | 240.66 | 215.85 | 225.95 | 2887425 |
| FT-Transformer | 246.88 | 224.12 | 233.35 | 1458625 |

### Deltas vs baselines

- Final vs Large MLP: **+8.27 kg** (beats Large? **False**)
- Final vs Teacher: **+10.50 kg**
- Combined vs Large MLP: **+7.40 kg** (beats Large? **False**)
- Combined vs Teacher: **+12.02 kg**

---

## Detailed Final metrics

| Metric | Value |
|--------|------:|
| RMSE | 224.1217 |
| MAE | 74.1653 |
| Bias | -12.0123 |
| R² | 0.915951 |
| n | 37,170 |

---

## Conclusions (evidence only)

1. FT-Transformer Final RMSE = **224.12 kg**.
2. Relative to Large MLP Final **215.85**: **+8.27 kg**.
3. Combined RMSE = **233.35 kg** (vs Large **225.95**, teacher **221.33**).
4. Deployment baseline remains **Large MLP** unless FT beats it on **both** Final and Combined.
5. Future architectures should use `build_student(...)` and the same KD pipeline.

## Artifacts

- Train: `results/distillation/ft_transformer/`
- Eval: `results/distillation/ft_transformer/evaluation/`

*Generated 2026-07-30T09:34:48.920531+00:00*
