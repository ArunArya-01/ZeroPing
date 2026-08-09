# Teacher Evaluation Audit — Frozen R3 Ensemble

**Audit timestamp (UTC):** 2026-07-30T00:40:00.317800+00:00
**Git commit:** aa6ebf16939d7e7dc94b2a8ba09e0f14f01dbe13

This is a **verification-only** audit. No training, no checkpoint modification, no hyperparameter changes.

---

## 1. Teacher checkpoint identified

| Field | Value |
|-------|------|
| Artifact type | pickle ensemble bundle (not a single neural checkpoint) |
| Filename | `r3_teacher_distillation_bundle.pkl` |
| Path | `<project_root>\cache\r3_teacher_distillation_bundle.pkl` |
| SHA256 | `5be4c65924c33ebb0703b7929bc4f70be3306413020b3f9a86d3d9ccec5a1b42` |
| Size | 17,585,666 bytes |
| Built at | 2026-07-29 19:22:58 |
| File mtime | 2026-07-29T19:22:58.320202 |
| Variant | **R3_P1E_phase_affine** |
| Meta-learner | ridge (Ridge, α=1.0) |
| Calibrator | ConditionalAffineCalibrator groups=['climb', 'cruise', 'descent', 'unknown'] |
| Base models | 6: [['xgb', 'direct'], ['lgbm', 'direct'], ['cat', 'direct'], ['xgb', 'fuel_flow'], ['lgbm', 'fuel_flow'], ['cat', 'fuel_flow']] |
| Feature count | 60 (train-fitted pipelines, n_features_in=[60, 60, 60, 60, 60, 60]) |
| Train rows when built | 119,032 |
| OOF RMSE (pre-P1E / post-P1E) | 252.3175 / 250.2702 |
| Distillation teacher? | **Yes** — same bundle used for soft labels |

### Ensemble members

1. XGB Direct
2. LGBM Direct
3. CatBoost Direct
4. XGB Fuel-Flow
5. LGBM Fuel-Flow
6. CatBoost Fuel-Flow

Stacking: Ridge on the 6 base kg predictions → P1E phase-conditional affine calibrator.

No alternate teacher checkpoint was used in Step 5. The sole inference artifact is this pickle bundle.

---

## 2. Dataset verification

| Field | Value |
|-------|------|
| Featured Final | `featured_dataset_final.parquet` |
| SHA256 | `4509b08399eb32d3dbd7e4315dbdbcd88e9b644e3464290d3cfba8537a4171fb` |
| Rows / flights | **37,170** / **2,824** |
| Source labels | `fuel_final.parquet` (SHA256 `ba274fcafc16097aef25629dc28225812af064a4cdf4ddacd7ed9ff45b44c352`) |
| Same as student Step 5? | **Yes** |
| Mean ground truth | 414.2287 kg |

Note: `fuel_final` has 2,836 flights / 37,456 intervals; feature engineering retains 2,824 / 37,170.

---

## 3. Preprocessing verification

Per-base sklearn Pipeline inside full_models (train-fitted). apply_bases → Ridge meta → ConditionalAffineCalibrator (P1E). No student scaler/OHE. Transform-only.

- **Refit during audit:** No
- **Teacher feature count / order fixed:** 60 columns from bundle
- **All features present after ensure_features:** True
- Missing before ensure (filled/created if any): `[]`

Student MLP uses a separate train-fitted StandardScaler + OHE (582-dim). That does **not** affect teacher inference.
Both models are scored on the **same** Final rows and ground-truth labels → metrics are directly comparable.

---

## 4. Reproduced metrics (Final held-out)

| Metric | Value |
|--------|------:|
| RMSE | **213.6218** |
| MAE | 74.1391 |
| Bias | +4.8680 |
| R² | 0.923642 |
| MAPE % | 39.61 |
| P95 \|err\| | 252.52 |
| Max \|err\| | 5911.52 |
| n | 37,170 |
| Inference time | 2.11 s |

Predictions: `results/distillation/teacher_audit/teacher_predictions.parquet`

---

## 5. Comparison table

| Source | RMSE | MAE | Notes |
|--------|-----:|----:|-------|
| Official R3 Combined (Rank+Final protocol) | 221.3264 | — | From docs/reports/r3_ensemble_summary.json; best_variant=R3_P1E_phase_affine. Rank RMSE=232.53, Final RMSE=213.73. Th... |
| Official R3 Final-only (same R3 run as Combined) | 213.7274 | — | Same evaluation campaign as Combined; Final split component only. |
| Step 5 held-out eval (test_evaluation metrics.json) | 213.6218 | 74.1391 | Teacher inference via same r3_teacher_distillation_bundle.pkl on featured_dataset_final. |
| This audit (reproduced Final inference) | 213.6218 | 74.1391 | Exact re-run on featured_dataset_final; n=37170; n_flights=2824. |
| Distillation meta reference_final_rmse | 213.7300 | — | Documented at dataset build time; expected ~213.73. |
| Distillation meta reference_combined_rmse | 221.3300 | — | Documented Combined; expected ~221.33. |

### Do they match?

- Reproduced vs Step 5: **Δ = 0.0 kg** · match within 0.05 kg? **True**
- Reproduced vs official Final (213.73): **Δ = -0.1056 kg**
- Reproduced vs Combined (221.33): difference is **protocol**, not a bug (see below)

---

## 6–7. Root-cause analysis of ~221 vs ~213.6

### Why project notes say ~221 kg

221.33 kg is the official Combined RMSE (Rank intervals + Final intervals evaluated together), not the Final-only RMSE. Final-only from the same official run is 213.73 kg. Project status docs correctly list both: Combined 221.33 and Final 213.73.

### Why Step 5 / this audit report ~213.6 kg

The held-out evaluation scores **Final only** (Oct 2025), matching the student evaluation protocol.
Official Final-only from the R3 campaign is **213.73 kg**. This audit reproduces **~213.62 kg**.

Official Final RMSE 213.73 used the R3 official Final featured path at gap-closing time. This audit uses featured_dataset_final.parquet (2824 flights, 37170 intervals). fuel_final has 2836 flights / 37456 intervals; feature build retains 2824/37170. Small Δ (~0.1 kg) is expected from row-set / feature-rebuild differences, not a different model.

### Verdict

**There is no contradiction between a correct ~221 Combined figure and a correct ~213.6 Final figure.**
They measure different evaluation aggregates of the same frozen teacher family.

---

## 8. Consistency checks

| Check | Result |
|-------|--------|
| Teacher feature count | 60 |
| Feature order from frozen bundle | fixed list in metrics JSON |
| Preprocessing refit | **No** |
| Checkpoint hash recorded | **Yes** |
| Dataset hash recorded | **Yes** |
| Same Final as student Step 5 | **Yes** |
| Same bundle as distillation soft labels | **Yes** |

---

## 9. Artifacts

| File | Path |
|------|------|
| Predictions | `results/distillation/teacher_audit/teacher_predictions.parquet` |
| Metrics | `results/distillation/teacher_audit/teacher_metrics.json` |
| Comparison CSV | `results/distillation/teacher_audit/comparison_table.csv` |
| This report | `docs/reports/teacher_evaluation_report.md` |

---

## 10. Documentation update decision

Reproduced Final RMSE matches Step 5 within floating-point tolerance.
The ~221 kg number is **correct** as Combined RMSE and is already documented separately from Final.
**No correction to Step 5 Final teacher RMSE is required.**
Docs may add a clarifying note that Combined ≠ Final (optional clarity; not a numeric fix).

---

## Final conclusions (evidence only)

1. **Checkpoint evaluated:** `r3_teacher_distillation_bundle.pkl` (SHA256 `5be4c65924c33ebb0703b7929bc4f70be3306413020b3f9a86d3d9ccec5a1b42`), variant **R3_P1E_phase_affine**, 6 GBDT bases + Ridge + P1E — the distillation teacher bundle.
2. **Reproducible?** **Yes.** Reproduced RMSE **213.6218** matches Step 5 (**Δ=0.0 kg**).
3. **Official held-out Final RMSE of frozen R3 teacher:** **213.62 kg** (this audit / Step 5). Official protocol Final from R3 run: **213.73 kg**. Combined protocol: **221.33 kg**.
4. **Why ~221 kg in notes:** That is **Combined** Rank+Final RMSE from `r3_ensemble_summary.json`, not Final-only.
5. **Student vs teacher comparable?** **Yes** on Final — same `featured_dataset_final.parquet` rows and labels; different internal feature pipelines by design.
6. **Permanent baseline?** **Yes** for Final-held-out student comparisons use **teacher Final ≈ 213.62 kg**. For official PRC Combined reporting continue to cite **221.33 kg**. Do not mix protocols.

### Canonical numbers going forward

| Protocol | Teacher RMSE | Use for |
|----------|-------------:|---------|
| **Final held-out** (student parity) | **213.62** | Distillation / transformer comparisons on Final |
| Official Final (R3 campaign) | **213.73** | Historical R3 report |
| Official Combined Rank+Final | **221.33** | PRC-style combined score |

*Generated 2026-07-30T00:40:00.317800+00:00*
