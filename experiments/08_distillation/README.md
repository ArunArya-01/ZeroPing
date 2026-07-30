# Distillation experiments

## Step 1 — Teacher distillation dataset

Frozen R3 teacher export only. No student training.

```bash
# from repo root
set PYTHONPATH=src
python experiments/08_distillation/01_build_teacher_distillation_dataset.py --train-only
```

### Outputs

| Artifact | Path |
|----------|------|
| Dataset | `distillation_dataset.parquet` |
| Report | `docs/reports/distillation_dataset_report.md` |
| Meta JSON | `docs/reports/distillation_dataset_meta.json` |
| Teacher OOF cache | `cache/r3_teacher_distillation_bundle.pkl` |

### Teacher (read-only)

- 6-base ensemble: XGB / LGBM / CatBoost × Direct / Fuel-Flow
- Ridge meta (selected on train OOF)
- R3 dynamic mass features (21)
- P1E phase-conditional affine calibration

### Flags

- `--train-only` — export train split only (default path when rank/final parquet are absent)
- `--force-rebuild` — ignore teacher cache and retrain OOF bases

---

## Step 2 — Baseline MLP student

Trains three compact MLPs on the **frozen** `distillation_dataset.parquet`:

| Model | Loss |
|-------|------|
| A | MSE(student, ground_truth) |
| B | MSE(student, teacher_prediction) |
| C | α·MSE(gt) + β·MSE(teacher) |

```bash
set PYTHONPATH=src
python experiments/08_distillation/02_train_mlp_student.py --alpha 0.5 --beta 0.5
```

Reusable package: `src/aerotwin/distillation/` (`data`, `mlp`, `trainer`, `metrics`, `runner`).

### Outputs

| Artifact | Path |
|----------|------|
| Checkpoints | `models/distillation/<run>/best_model.pt` |
| Metrics / preds / curves | `results/distillation/<run>/` |
| Logs | `logs/distillation/<run>/` |
| Report | `docs/reports/mlp_student_report.md` |

---

## Step 3 — Alpha/beta KD weight sweep

Eight fixed-architecture runs (only alpha/beta change).

```bash
set PYTHONPATH=src
python experiments/08_distillation/run_distillation_experiments.py sweep
# or:
python experiments/08_distillation/03_alpha_beta_sweep.py
```

### Recommended supervision (from sweep)

**alpha = 0.1, beta = 0.9** (`KD-1`) — default for future student architectures.

### Outputs

| Artifact | Path |
|----------|------|
| Metrics / tables / plots | `results/distillation/alpha_beta_sweep/` |
| Report | `docs/reports/distillation_alpha_beta_sweep.md` |
| Best config JSON | `results/distillation/alpha_beta_sweep/best_configuration.json` |

### Reusable runner

```python
from aerotwin.distillation import DistillationData, run_kd_sweep, ExperimentConfig, DEFAULT_KD_SWEEP
from aerotwin.distillation.mlp import StudentMLP  # swap this class for FT-Transformer, etc.

data = DistillationData.from_parquet("distillation_dataset.parquet")
rows = run_kd_sweep(
    data=data,
    model_factory=lambda d: StudentMLP(d, hidden_dims=(1024, 512)),
    weights=DEFAULT_KD_SWEEP,
    exp=ExperimentConfig(),
    results_root=Path("results/distillation/my_student_sweep"),
)
```

---

## Step 4 — Capacity scaling, latency, multi-seed

Fixed **α=0.1, β=0.9**. Tiers: Tiny→XLarge (~250K–6.75M params).

```bash
set PYTHONPATH=src
python experiments/08_distillation/run_distillation_experiments.py capacity
# or:
python experiments/08_distillation/04_capacity_scaling.py
```

### Key results (flight holdout; not official Rank/Final)

| Model | Params | Val RMSE | Gap vs teacher soft labels |
|-------|-------:|---------:|---------------------------:|
| Tiny | 0.24M | 270.55 | +26.4 |
| Small | 0.50M | 241.73 | −2.4 |
| Medium | 1.13M | 235.04 | −9.1 |
| Large | 2.89M | 229.70 | −14.4 |
| **XLarge** | **6.75M** | **228.14** | **−16.0** |

- Multi-seed (XLarge, n=5): **228.49 ± 0.92 kg**; CI entirely below teacher soft-label RMSE.
- Single-sample CPU: teacher ~**52 ms**, students ~**0.2–0.5 ms**.
- Large is within **2 kg** of best → good default capacity for transformers.

### Outputs

| Artifact | Path |
|----------|------|
| Metrics / bench / seeds | `results/distillation/capacity_scaling/` |
| Report | `docs/reports/capacity_scaling_report.md` |

---

## Step 5 — Official held-out Final evaluation (eval only)

**No training.** Loads frozen Large / XLarge checkpoints and evaluates on `featured_dataset_final.parquet`.

```bash
set PYTHONPATH=src
python experiments/08_distillation/05_test_evaluation.py --final-featured featured_dataset_final.parquet
```

### Official results (Final, 37,170 rows / 2,824 flights)

| Model | Params | Final RMSE | vs teacher Final |
|-------|-------:|-----------:|-----------------:|
| **Large (baseline)** | 2.89M | **215.85** | +2.23 kg |
| XLarge | 6.75M | 218.59 | +4.96 kg |
| R3 Teacher | ensemble | **213.62** | — |

Large is the permanent MLP baseline for future architectures. XLarge does not justify extra capacity on Final.

### Outputs

| Artifact | Path |
|----------|------|
| Metrics / preds / plots | `results/distillation/test_evaluation/` |
| Report | `docs/reports/test_evaluation.md` |

---

## Teacher evaluation audit (verification only)

Reproduces frozen R3 teacher Final metrics. Explains Combined **221.33** vs Final **213.62**.

```bash
set PYTHONPATH=src
python experiments/08_distillation/06_teacher_evaluation_audit.py
```

| Artifact | Path |
|----------|------|
| Predictions / metrics | `results/distillation/teacher_audit/` |
| Report | `docs/reports/teacher_evaluation_report.md` |

---

## Combined (Rank + Final) evaluation

Official PRC-style Combined RMSE for frozen Large / XLarge. Requires `featured_dataset_rank.parquet`.

```bash
set PYTHONPATH=src
python experiments/08_distillation/07_combined_evaluation.py
```

| Model | Rank | Final | Combined |
|-------|-----:|------:|---------:|
| Large | 240.66 | 215.85 | **225.95** |
| XLarge | 244.40 | 218.59 | 229.10 |
| R3 Teacher | 232.53 | 213.62 | **221.33** |

Report: `docs/reports/combined_evaluation.md`
