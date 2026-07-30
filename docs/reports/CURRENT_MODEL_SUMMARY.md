# Current Model Summary

**Last synced:** 2026-07-30 (R3 teacher frozen; distillation MLP track complete including **Combined Rank+Final**)

## Teacher (frozen — do not retrain)

- **Type:** Ensemble of 6 GBDT base models (XGB/LGBM/CatBoost × Direct kg + Fuel Flow kg/s)
- **Meta-learner:** Ridge regression (chosen by GroupKFold CV on train OOF over LGBM)
- **Base hyperparameters:** n_estimators=300, lr=0.05 (frozen V4)
- **Specialists / calibrators:**
  - P1E: Phase-conditional affine (train OOF, minor keep)
  - P2 / R1 / R2: CatBoost FuelFlow heavy-aircraft specialist with OpenAP descriptors + descriptor fixes
  - **R3: Dynamic mass features (21)** applied on the official ensemble path (`src/aerotwin/engine/mass_model.py`)
- **Calibration:** Phase-conditional affine (P1E)
- **Status:** Frozen teacher for knowledge distillation

## Training Data

- **Source:** `aerotwin/aero-data` (Hugging Face)
- **Train split:** Apr–Aug 2025, 10,000 usable flights, 119,032 intervals
- **Rank split:** Sep 2025, 1,888 flights, 24,158 intervals
- **Final split:** Oct 2025, 2,836 flights → **2,824 flights / 37,170 intervals** after feature build
- **Feature count (base):** ~47 (BASE_NUMERIC + ENERGY_FEATURES + WEATHER_FEATURES + physics + cats)
- **Feature count (R1 specialist):** 57 (base + 10 OpenAP descriptors + 8 interactions)
- **Feature count (R3 mass path):** base + **21** dynamic mass features
- **Final featured:** `featured_dataset_final.parquet` (from `fuel_final.parquet`)

## Official Metrics

### R3_P1E_phase_affine + dynamic mass (current best teacher)

> **Protocol note (teacher audit 2026-07-30):** **Combined ≈ 221.33 kg** is Rank+Final together. **Final-only** is ≈ **213.7 kg**. Do not treat 221 as Final-held-out. Student parity uses Final teacher RMSE **213.62 kg** on `featured_dataset_final.parquet` (see `teacher_evaluation_report.md`).

| Metric | Value |
|--------|------:|
| Rank RMSE | **232.53** kg |
| Final RMSE (official R3 run) | **213.73** kg |
| Final RMSE (held-out audit / Step 5, same bundle) | **213.62** kg |
| **Combined RMSE** | **221.33** kg |
| Δ vs 228.25 (canonical) | **−6.92** kg |
| Δ vs 225.25 (R2) | **−3.92** kg |
| Combined bias | **+3.85** kg |
| Heavy RMSE | **416.1** kg |
| Narrow RMSE | **75.0** kg |

Source: `figures/r3_ensemble_summary.json`, `figures/table_rmse_R3_ensemble_leaderboard.csv`

### Gap-closing ladder

| Version | Variant | Combined RMSE | Δ vs 228.25 |
|---------|---------|--------------:|------------:|
| v1.0 | Official frozen V4 ensemble | **228.25** | reference |
| v1.1 | P1E + P2 Cat heavy specialist | **227.44** | −0.81 |
| R1 | P1E + OpenAP descriptors in heavy specialist | **226.19** | −2.06 |
| R2 | Fixed B744/B77L/A306 descriptors + R2 features | **225.25** | −3.00 |
| **R3** | **P1E + dynamic mass (21 features)** | **221.33** | **−6.92** |

### Canonical official v1 (frozen reference)

- Rank RMSE: **239.18** kg
- Final RMSE: **220.86** kg
- Combined RMSE: **228.25** kg
- Combined 95% CI: **[207.1, 249.4] kg** — does **not** exclude 201 → **no superiority claim**

## Two evaluation protocols

| Protocol | Metric | Use for |
|----------|--------|---------|
| **A — Final** | Final RMSE only | Architecture research, held-out student comparisons |
| **B — Combined** | RMSE(concat Rank, Final) | Official PRC leaderboard parity vs teacher **221.33** |

Both are retained. Transformers must report **both**.

## Distilled MLP — official baselines

**Do not retrain for architecture comparisons.**

### Protocol A — Final (Step 5)

| Model | Params | Val RMSE | **Final RMSE** | MAE | Bias | R² | CPU ms |
|-------|-------:|---------:|---------------:|----:|-----:|---:|-------:|
| **Large (deploy)** | **2.89M** | 229.70 | **215.85** | **76.69** | +5.25 | **0.9220** | **0.26** |
| XLarge | 6.75M | 228.14 | 218.59 | 77.36 | +6.41 | 0.9201 | 0.52 |
| R3 Teacher | ensemble | — | **213.62** | 74.14 | +4.87 | 0.9236 | ~52 |

### Protocol B — Combined Rank+Final

| Model | Rank RMSE | Final RMSE | **Combined RMSE** | Δ teacher Combined | Params |
|-------|----------:|-----------:|------------------:|-------------------:|-------:|
| **Large (deploy)** | **240.66** | **215.85** | **225.95** | **+4.62** | 2.89M |
| XLarge | 244.40 | 218.59 | 229.10 | +7.77 | 6.75M |
| R3 Teacher | 232.53 | 213.62 | **221.33** | — | ensemble |

Report: `docs/reports/combined_evaluation.md` · `results/distillation/combined_evaluation/`

### Phase 2 — FT-Transformer (same KD pipeline)

| Model | Params | Val RMSE | Final | Combined | CPU ms | Beats Large? |
|-------|-------:|---------:|------:|---------:|-------:|:------------:|
| FT-Transformer | 1,458,625 | 236.08 | **224.12** | **233.35** | 9.59 | **No** |
| Large MLP (deploy) | 2,887,425 | 229.70 | **215.85** | **225.95** | 0.26 | — |

FT does **not** replace Large as the deployment baseline. Report: `docs/reports/ft_transformer_experiment.md`.

## Phase 0 — Distribution shift diagnosis (2026-07-30)

| Protocol | Teacher | Large | XLarge | FT |
|----------|--------:|------:|-------:|---:|
| Flight Final RMSE | 213.62 | 215.85 | 218.59 | 224.12 |
| Type-macro RMSE | 256.79 | 270.61 | 276.01 | 261.15 |
| Body-macro RMSE | 237.55 | 239.63 | 242.08 | 249.58 |
| Gap vs teacher (flight) | 0 | +2.23 | +4.96 | +10.50 |
| Gap vs teacher (type-macro) | 0 | **+13.82** | +19.22 | **+4.35** |

**Gate:** Adaptive KD **UNBLOCKED** — Large type-macro gap widens by **+11.59 kg** (CI excludes 0).  
**Deploy baseline unchanged:** Large MLP (Final/Combined).  
Report: `docs/reports/distribution_shift_diagnosis.md`.

## Phase 1A — Teacher uncertainty (2026-07-30)

| Check | Result |
|-------|--------|
| Spearman(disagreement, teacher \|err\|) | **0.426** |
| Spearman(disagreement, Large \|err\|) | **0.435** |
| Bin calibration ρ (teacher / Large) | **0.976 / 0.952** |
| Type-level Spearman(disagreement, teacher RMSE) | **0.757** |
| Top−bottom type student-gap Δ | **+49.4 kg** |
| **Proceed to Adaptive KD (1B)?** | **YES** |

Disagreement = std of 6 base ensemble predictions. Report: `docs/reports/teacher_uncertainty_analysis.md`.

## Phase 1B — VGKD (negative result)

Adaptive \(\beta(x)=\beta_b\exp(-\lambda\max(u_n,0))\) on Large MLP:

| Run | Final | Type-macro | Notes |
|-----|------:|-----------:|-------|
| Fixed KD Large | **215.85** | 270.61 | **Deploy baseline** |
| VGKD λ=0 | 216.10 | 269.76 | Reproduces fixed KD |
| VGKD λ>0 | worse | worse | No robustness gain |

**Do not deploy VGKD.** Report: `docs/reports/vgkd_results.md`.

| Finding | Evidence |
|---------|----------|
| Official student baseline | **Large** (α=0.1, β=0.9) |
| XLarge justifies capacity? | **No** (+2.73 kg worse Final RMSE) |
| Gen. gap val→Final (Large) | **−13.85 kg (−6.0%)** — test better; no overfit |
| Student–teacher Final gap | Large **+2.23 kg** |
| Ranking val vs test | Reversed (val XLarge → test Large) |
| Hard aircraft | B77W, B744, B772, B789, A359 |
| Easy aircraft | A21N, A319, A320, A20N, B738 |

**Checkpoint (official):**  
`results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt`

**Eval artifacts:** `results/distillation/test_evaluation/`  
**Report:** `docs/reports/test_evaluation.md`

## Key Findings

1. **Frozen teacher Combined RMSE: 221.33 kg** (R3 dynamic mass + P1E on official path).
2. Total gap-close from canonical: **−6.92 kg**; remaining gap to published winner (~201 kg): **≈20 kg**.
3. **Largest single teacher step:** R3 mass features — bias falls from ~+24 kg to ~**+3.9 kg**.
4. Largest error source remains **heavy aircraft (A359/B77W/B744)** and **cruise / ultra-long-haul** intervals.
5. **Official MLP Final baseline: Large 215.85 kg** — within ~2 kg of teacher Final, ~200× faster on CPU single-sample.
6. **MLP width beyond Large is not justified on held-out Final** (XLarge 218.59).

## Distillation pipeline summary

| Item | Value |
|------|------:|
| Dataset | `distillation_dataset.parquet` (119,032 × 60) |
| **Recommended KD weights** | **α=0.1, β=0.9** |
| Best capacity (val) | XLarge ~6.75M (228.14 kg) |
| **Best deployment / Final** | **Large ~2.89M (215.85 kg)** |
| Multi-seed mean ± std XLarge (val) | **228.49 ± 0.92 kg** |
| Teacher soft-label val RMSE (capacity split) | **244.14 kg** |

Reports: `distillation_dataset_report.md`, `mlp_student_report.md`, `distillation_alpha_beta_sweep.md`, `capacity_scaling_report.md`, **`test_evaluation.md`**.

## Reproduce

```bash
# Teacher reference (frozen)
PYTHONPATH=src python experiments/07_gap_closing/26_r3_ensemble_mass.py

# Distillation
set PYTHONPATH=src
python experiments/08_distillation/01_build_teacher_distillation_dataset.py --train-only
python experiments/08_distillation/02_train_mlp_student.py
python experiments/08_distillation/run_distillation_experiments.py sweep
python experiments/08_distillation/run_distillation_experiments.py capacity

# Official held-out Final eval (eval-only; no training)
python experiments/08_distillation/05_test_evaluation.py --final-featured featured_dataset_final.parquet
```

## Related docs

- Project status: `PROJECT_STATUS_REPORT.md`
- Official report: `official_prc_benchmark_report.md`
- Gap-closing v1: `official_gap_closing_report.md`
- Distillation: `distillation_alpha_beta_sweep.md`, `capacity_scaling_report.md`
- **Held-out eval:** `test_evaluation.md`
- Backlog: `RMSE_IMPROVEMENT_BACKLOG.md`
- Mass model: `physics/mass_model.py`
