# Official PRC2025 Benchmark Report — AeroTwin

**Status:** COMPLETE (full Rank + Final evaluation)  
**Date:** July 2026  
**Methodology:** **Frozen** AeroTwin V4 — no tuning after seeing Rank/Final labels  

**Paper:** Sun, Spinielli & Strohmeier, *Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge*, Journal of Open Aviation Science (2026). doi:10.59490/joas.2026.8750

---

## Canonical statement

> **Under the released official PRC2025 benchmark protocol, AeroTwin achieved:**
>
> | Split | MAE (kg) | RMSE (kg) | R² |
> |-------|--------:|----------:|---:|
> | **Rank** (Sep 2025) | **90.89** | **239.18** | 0.904 |
> | **Final** (Oct 2025) | **87.35** | **220.86** | 0.918 |
> | **Combined** (Rank+Final) | **88.75** | **228.25** | 0.913 |
>
> **Best model:** Ensemble (6 bases: XGB/LGBM/CatBoost × Direct + Fuel-Flow, Energy+Weather features) + **Ridge** meta learner selected on train OOF only.  
>
> **Published winner (combined RMSE):** ≈ **201 kg** (paper).  
> **ΔRMSE (AeroTwin − winner):** **+27.25 kg** (AeroTwin higher error).  
> **Combined RMSE 95% CI (flight bootstrap):** **[207.1, 249.4] kg** — does **not** exclude 201; we do **not** claim superiority.
>
> Training used **only** the train split. Rank/Final were never used for feature engineering, hyperparameter search, or model selection. This is the **canonical AeroTwin official benchmark** for the paper.

---

## 1. Dataset audit

Source: Hugging Face `aerotwin/aero-data` (matches official PRC release).

| Split | Period (paper) | Flights (list) | Usable traj | Fuel intervals (labels) | Featured intervals (after pipeline) |
|-------|----------------|---------------:|------------:|------------------------:|------------------------------------:|
| **Train** | Apr–Aug 2025 | 11,037 | 10,000 | 131,530 | 119,032 |
| **Rank** | Sep 2025 | 1,888 | 1,888 | 24,289 | 24,158 (1,881 flights) |
| **Final** | Oct 2025 | 2,836 | 2,836 | 37,456 | 37,170 (2,824 flights) |

Paper Table 1 counts for flights/intervals match HF metadata exactly.

**Artifacts:** `figures/table_dataset_audit.csv`, `table_schema_comparison.csv`, `table_split_statistics.csv`, `table_overlap_check.csv`, `table_aircraft_distribution.csv`, `table_route_distribution.csv`, `fig_dataset_distribution.png`, `table_paper_vs_hf_counts.csv`.

### Schema

Flightlist and fuel schemas are **identical** across train / rank / final. Trajectory columns align with the paper (timestamp, altitude m, groundspeed m/s, vertical_rate m/s, mach, CAS, TAS, source, …).

---

## 2. Protocol verification

| Item | Official (paper) | AeroTwin implementation |
|------|------------------|-------------------------|
| Task | Predict interval fuel (kg) | Same |
| Metric | **RMSE (kg)** | RMSE primary; also MAE, R² |
| Winner | ~**201 kg** combined Rank+Final | Compared to 201 (paper); legacy cite 200.83 |
| Rank | Sep 2025 eval | Full Rank featured set scored |
| Final | Oct 2025 eval | Full Final featured set scored |
| Train | Apr–Aug 2025 only | Train-only for all fits |

**No** hyperparameter search, feature changes, or model selection on Rank/Final.

**Frozen features (39):** `BASE_NUMERIC` + `ENERGY_FEATURES` + `WEATHER_FEATURES` + `physics_fuel_kg` + `CATEGORICAL` (`aircraft_type`, `method`, `origin_icao`, `destination_icao`).

**Frozen tree hyper-parameters:** XGB/LGBM/CatBoost n_estimators/iterations=300, lr=0.05 (from internal V4 defaults).

---

## 3. Leakage verification

| Pair | flight_id | fuel flights | trajectory files | interval keys | Pass |
|------|----------:|-------------:|-----------------:|--------------:|:----:|
| train vs rank | 0 | 0 | 0 | 0 | ✅ |
| train vs final | 0 | 0 | 0 | 0 | ✅ |
| rank vs final | 0 | 0 | 0 | 0 | ✅ |

---

## 4. Training verification

Repository audit: prior experiments used only train-derived `featured_dataset*.parquet`. No historical `split="rank"` / `split="final"` training paths.

Official evaluation entry points:

- `notebooks/16_dataset_audit.py`
- `notebooks/17_official_prc_evaluation.py`
- `physics/official_benchmark.py`

Meta learner (Ridge vs LGBM) chosen by **GroupKFold CV on train OOF only** — Ridge won (train OOF RMSE ~254 vs LGBM ~264).

---

## 5. Official benchmark results

### Full leaderboard (sorted by combined RMSE)

| Model | Target | Rank MAE | Rank RMSE | Rank R² | Final MAE | Final RMSE | Final R² | Comb. MAE | Comb. RMSE |
|-------|--------|--------:|----------:|--------:|----------:|-----------:|---------:|----------:|-----------:|
| **Ensemble (6-base + ridge)** | mixed | **90.89** | **239.18** | 0.904 | **87.35** | **220.86** | 0.918 | **88.75** | **228.25** |
| LGBM FuelFlow E+W | flow | 86.35 | 249.83 | 0.896 | 79.99 | 216.46 | 0.922 | 82.49 | 230.18 |
| CatBoost FuelFlow E+W | flow | 81.53 | 244.90 | 0.900 | 76.73 | 221.94 | 0.918 | 78.62 | 231.26 |
| XGB FuelFlow E+W | flow | 88.86 | 246.39 | 0.899 | 84.84 | 239.23 | 0.904 | 86.42 | 242.08 |
| CatBoost Direct E+W | direct | 117.76 | 253.21 | 0.893 | 121.47 | 256.78 | 0.890 | 120.01 | 255.38 |
| LGBM Direct E+W | direct | 120.43 | 263.26 | 0.884 | 122.14 | 255.50 | 0.891 | 121.47 | 258.59 |
| XGB Direct E+W | direct | 122.93 | 259.75 | 0.887 | 125.20 | 285.75 | 0.863 | 124.30 | 275.80 |
| OpenAP Physics | physics | 451.45 | 1191.95 | −1.38 | 485.40 | 1315.65 | −1.90 | 472.03 | 1268.37 |

**Source:** `figures/table_official_leaderboard.csv`

### Notes

- **Fuel-Flow** models beat **Direct** on official RMSE/MAE (same pattern as internal train holdout).
- Best **single** combined RMSE: LGBM FuelFlow (~230.2).
- Best **MAE** on Final: CatBoost FuelFlow (~76.7).
- **Ensemble** edges combined RMSE (~228.3) via stacking Direct + Flow bases.
- OpenAP alone remains far worse (~1268 combined RMSE).

### Bootstrap (best model = Ensemble)

| Quantity | Point | 95% CI |
|----------|------:|--------|
| Final RMSE | 220.86 | [196.2, 246.6] |
| Final MAE | 87.35 | [82.7, 92.3] |
| Combined RMSE | 228.25 | [207.1, 249.4] |

---

## 6. Comparison against published PRC winner

| Quantity | Value |
|----------|------:|
| Published winner combined RMSE (paper) | **≈ 201 kg** |
| Legacy internal citation | 200.83 kg |
| Best AeroTwin combined RMSE | **228.25 kg** |
| ΔRMSE (AeroTwin − winner) | **+27.25 kg** |
| Combined RMSE 95% CI | [207.1, 249.4] |

### Fairness

- Same task, same public Rank/Final labels, same primary metric (RMSE kg).
- Winner’s exact pipeline is unpublished → comparison is to the **published score**, not a re-run of winning code.
- AeroTwin CI **includes values above 201** and point estimate is **worse** than 201 → **no claim of superiority**.

**Honest conclusion:** Frozen AeroTwin is **competitive with strong open hybrid pipelines** but **does not match the published winner** under the official combined RMSE (~228 vs ~201).

---

## 7. Discussion

1. **Temporal shift is real.** Train (Apr–Aug) → Rank (Sep) → Final (Oct) matches the paper’s observation that month-to-month generalization was hard.
2. **Fuel-flow target helps on official data**, not only internal holdouts.
3. **Internal train-split RMSE (~196–203)** is **not** the official score; Rank/Final are harder (~216–250 for top models).
4. **Ensemble helps modestly** on combined RMSE vs best single Fuel-Flow model.
5. Physics-only is inadequate; hybrid ML remains essential.

---

## 8. Limitations

1. Train trajectories: 10,000 / 11,037 flightlist rows (missing traj files for 1,037).
2. Rank/Final featured builds dropped a few flights with empty OpenAP/feature windows (Rank 1,881/1,888; Final 2,824/2,836).
3. Residual PRC label noise (paper §5.2) affects all methods.
4. Winner recipe unknown → score comparison only.
5. Hyper-parameters frozen at V4 defaults (300 trees); no Rank/Final retuning by design.
6. Combined CI treats published winner RMSE as a fixed constant.

---

## 9. Final conclusion

Under the released official PRC2025 protocol, with methodology frozen before Rank/Final labels:

- **Best AeroTwin:** Ensemble (Direct + Fuel-Flow Energy+Weather bases, Ridge meta).  
- **Rank RMSE 239.2 kg · Final RMSE 220.9 kg · Combined RMSE 228.3 kg.**  
- **Published winner ~201 kg combined** → AeroTwin is **~27 kg RMSE worse** on this run.  
- **Fuel-Flow** dominates Direct; OpenAP alone fails.  
- This evaluation is **reproducible**, **train-only**, and **leakage-free**.

### Reproduce

```bash
python notebooks/16_dataset_audit.py
# Featured sets (slow first time; cached under cache/featured_*_parts/)
python -c "from physics.official_benchmark import build_featured_for_split as b; b('rank'); b('final')"
python notebooks/17_official_prc_evaluation.py --skip-build
```

### Key artifacts

| File | Content |
|------|---------|
| `figures/table_official_rank_results.csv` | Per-model Rank metrics |
| `figures/table_official_final_results.csv` | Per-model Final metrics |
| `figures/table_official_leaderboard.csv` | Full official leaderboard |
| `figures/table_prc_comparison.csv` | Winner comparison + CIs |
| `figures/fig_official_leaderboard.png` | Rank vs Final RMSE bars |
| `figures/fig_prc_vs_aerotwin.png` | Winner vs AeroTwin |
| `figures/official_eval_summary.json` | Machine-readable summary |
| `featured_dataset_rank.parquet` / `_final.parquet` | Official eval features |

---

*Report completed after full Rank+Final featurization and frozen evaluation run (`official_full_run: true`).*
