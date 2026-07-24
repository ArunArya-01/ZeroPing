# Current Model Summary

**Last synced:** 2026-07-24 (repo HEAD: R3 dynamic mass, Combined **221.33**)

## Model Architecture

- **Type:** Ensemble of 6 GBDT base models (XGB/LGBM/CatBoost × Direct kg + Fuel Flow kg/s)
- **Meta-learner:** Ridge regression (chosen by GroupKFold CV on train OOF over LGBM)
- **Base hyperparameters:** n_estimators=300, lr=0.05 (frozen V4)
- **Specialists / calibrators:**
  - P1E: Phase-conditional affine (train OOF, minor keep)
  - P2 / R1 / R2: CatBoost FuelFlow heavy-aircraft specialist with OpenAP descriptors + descriptor fixes
  - **R3: Dynamic mass features (21)** applied on the official ensemble path (`physics/mass_model.py`)
- **Calibration:** Phase-conditional affine (P1E)

## Training Data

- **Source:** `aerotwin/aero-data` (Hugging Face)
- **Train split:** Apr–Aug 2025, 10,000 usable flights, 119,032 intervals
- **Rank split:** Sep 2025, 1,888 flights, 24,158 intervals
- **Final split:** Oct 2025, 2,836 flights, 37,170 intervals
- **Feature count (base):** ~47 (BASE_NUMERIC + ENERGY_FEATURES + WEATHER_FEATURES + physics + cats)
- **Feature count (R1 specialist):** 57 (base + 10 OpenAP descriptors + 8 interactions)
- **Feature count (R3 mass path):** base + **21** dynamic mass features

## Official Metrics

### R3_P1E_phase_affine + dynamic mass (current best)

| Metric | Value |
|--------|------:|
| Rank RMSE | **232.53** kg |
| Final RMSE | **213.73** kg |
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

### R3 single-model mass ablation (reference)

- Best single: `R3_lgbm_dynamic_mass` Combined **224.23** (vs no-mass baseline 230.18)
- Source: `figures/r3_summary.json`

### Canonical official v1 (frozen reference)

- Rank RMSE: **239.18** kg
- Final RMSE: **220.86** kg
- Combined RMSE: **228.25** kg
- Combined 95% CI: **[207.1, 249.4] kg** — does **not** exclude 201 → **no superiority claim**

## Key Findings

1. **Current best Combined RMSE: 221.33 kg** (R3 dynamic mass + P1E on official path).
2. Total gap-close from canonical: **−6.92 kg**; remaining gap to published winner (~201 kg): **≈20 kg**.
3. **Largest single step:** R3 mass features (replacing crude `MTOW × 0.75`) — bias falls from ~+24 kg to ~**+3.9 kg**.
4. Largest error source remains **heavy aircraft (A359/B77W/B744)** and **cruise / ultra-long-haul** intervals.
5. Early Level-1 **heuristic** mass ablation was rejected; official **R3 dynamic mass** is a different formulation and is **kept**.

## Reproduce

```bash
PYTHONPATH=. python notebooks/17_official_prc_evaluation.py --skip-build
PYTHONPATH=. python notebooks/18_official_error_analysis.py
PYTHONPATH=. python notebooks/19_gap_closing_campaign.py
PYTHONPATH=. python notebooks/21_rmse_r1_heavy_features.py
PYTHONPATH=. python notebooks/24_r2_heavy_features.py
PYTHONPATH=. python notebooks/25_r3_dynamic_mass.py
PYTHONPATH=. python notebooks/26_r3_ensemble_mass.py
```

## Related docs

- Project status: `PROJECT_STATUS_REPORT.md`
- Official report: `official_prc_benchmark_report.md`
- Gap-closing v1: `official_gap_closing_report.md`
- Backlog: `RMSE_IMPROVEMENT_BACKLOG.md`
- Mass model: `physics/mass_model.py`
