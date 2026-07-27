# AeroTwin Internal Leaderboard Audit

**Date:** July 2026  
**Question:** What is the true best **internal** model under the final strict flight-level protocol?  
**Focus comparison:** Fuel-Flow + Energy LightGBM (RMSE 196.4) vs stacking ensemble (RMSE 202.9).

---

## 1. Verdict (short)

**These two results are NOT directly comparable** and must **not** share a single “best model” rank.

| Claim | Status |
|-------|--------|
| Flow+Energy LGBM (RMSE **196.4**) beats stacking (RMSE **202.9**) as the overall best model | ❌ Invalid — different **targets** and **training recipes** |
| Stacking is the best **Direct-kg** multi-model ensemble RMSE on Level-1 | ✅ Valid within the Direct / competition track |
| Flow+Energy LGBM has the best **single-model RMSE** on the Fuel-Flow V4 track | ✅ Valid within the Fuel-Flow track |
| Best **single-model MAE** on the Fuel-Flow track is XGB, not LGBM | ✅ Flow+Energy XGB MAE **79.52** < LGBM **80.33** |

**Recommendation:** Maintain **separate leaderboards** by protocol (below). Do not merge Flow single-models and Direct stacking into one ranked table without labels.

---

## 2. Protocol fingerprint comparison

| Dimension | Fuel-Flow LGBM (196.4 RMSE) | Stacking LGBM_meta (202.9 RMSE) | Match? |
|-----------|----------------------------|----------------------------------|--------|
| **Metric** | MAE / RMSE / R² on recovered `fuel_kg` | MAE / RMSE / R² on `fuel_kg` | ✅ Same metric defs |
| **Outer split** | `flight_level_split`, test_size=0.2, seed=42 | Same outer split function | ✅ |
| **Test flights** | 1996 / 9976 | 1996 / 9976 | ✅ Identical set (verified) |
| **Intervals (cleaned)** | 115,995 | 115,995 | ✅ |
| **Parquet** | `featured_dataset_mass.parquet` | `featured_dataset.parquet` | ⚠ Same rows/flights after clean; column set differs |
| **Prediction target** | **Fuel flow** `kg/s`, recover × `duration_s` | **Direct** `actual_fuel_kg` | ❌ **Different** |
| **Features** | Base + Energy + Weather + `physics_fuel_kg` + cats (no `phase` in `eval_framework` path) | Base + Energy + Weather + `physics_fuel_kg` + cats **+ `phase`** | ⚠ Close, not identical |
| **Model** | Single LightGBM | Stack of LGBM/XGB/RF/CatBoost + LGBM meta | ❌ |
| **Training** | One fit on train flights | 5-fold flight GroupKFold/KFold OOF on train → meta → retrain L1 on full train | ❌ |
| **Hyperparameters** | `eval_framework`: LGBM n=300, lr=0.05 | Stack L1: n=600, lr=0.03 (heavier) | ❌ |
| **Source script** | `notebooks/10_fuel_flow_target.py` | `notebooks/11_stacking.py` / `12_verify_ensemble.py` | — |
| **Artifacts** | `table_fuel_flow.csv`, `table_fuel_flow_ablation.csv` | `table_stacking.csv`, `table_ensemble_final.csv`, `final_leaderboard.csv` | — |

### Why RMSE 196.4 ≱ “better than” 202.9

1. **Target space differs.** Flow models optimise rate then recover kg; Direct models optimise kg. Lower RMSE under Flow is a finding about the **target**, not a free upgrade of the stacking architecture.
2. **Architecture differs.** Single LGBM vs multi-model stack.
3. **Capacity / training recipe differ.** Default 300-tree LGBM vs 600-tree L1 bases + meta trained on OOF.
4. **Competition narrative** for RMSE 202.9 was always “Direct hybrid ensemble vs PRC winner 200.83” — that comparison stays on the **Direct** track.

Outer **test flights are the same**, so scientific comparison of *approaches* (Flow vs Direct) is fair **within** the Fuel-Flow notebook (table_fuel_flow.csv). Comparing Flow single-model RMSE to Direct stack RMSE as “which model is best overall” is a category error.

---

## 3. Leaderboard A — Single-model, Fuel-Flow target (V4 scientific track)

**Protocol:** Strict flight-level 80/20 (seed 42) · recover kg · Energy(+Weather) features · `notebooks/10_fuel_flow_target.py`

| Rank | Model | Features | Target | MAE | RMSE | R² | Source |
|-----:|-------|----------|--------|----:|-----:|---:|--------|
| **MAE-best** | **XGB** | Flow + Energy | flow→kg | **79.52** | 208.42 | 0.945 | `table_fuel_flow_ablation.csv` |
| **RMSE-best** | **LGBM** | Flow + Energy | flow→kg | 80.33 | **196.24** | 0.951 | same |
| 3 | LGBM | Flow + Energy+Weather | flow→kg | 80.71 | 196.41 | 0.951 | `table_fuel_flow.csv` |
| 4 | XGB | Flow + Energy+Weather | flow→kg | 80.06 | 210.81 | 0.944 | same |
| 5 | LGBM | Flow + E+W+Mass | flow→kg | 80.23 | 197.20 | 0.951 | ablation |
| — | XGB | Direct E+W (ref) | direct kg | 83.76 | 212.03 | 0.943 | same notebook |

**Primary scientific takeaway (this track):** Fuel-flow target beats Direct on the **same** split/features (ΔMAE ≈ −3.7 kg for XGB E+W; bootstrap significant).  
**Best single model depends on metric:** MAE → XGB; RMSE → LGBM.

---

## 4. Leaderboard B — Single-model, Direct kg target (V3 / hybrid track)

**Protocol:** Strict flight-level 80/20 · Direct `actual_fuel_kg` · Energy/Weather hybrids · primarily XGB in V3

| Rank | Model | Features | MAE | RMSE | R² | Source |
|-----:|-------|----------|----:|-----:|---:|--------|
| 1 (MAE) | **XGB** | Energy+Weather Hybrid | **83.76** | 212.03 | 0.943 | `table_v3_leaderboard.csv`, `table_fuel_flow.csv` |
| 2 | XGB | Energy Hybrid | 84.48 | — | — | V3 E6 |
| 3 | LGBM | Direct E+W | 86.67 | 208.61 | 0.945 | `table_fuel_flow.csv` |
| 4 | XGB | OpenAP Hybrid (base+physics) | 86.31 | — | — | V3 baseline |
| 5 | CatBoost | Direct E+W | 91.32 | 211.45 | 0.944 | `table_fuel_flow.csv` |
| — | OpenAP only | physics | ~655–668 | ~1550+ | <0 | `table_model_comparison*.csv` |

**Primary scientific takeaway:** Energy (+ weather) improves Direct hybrid over OpenAP Hybrid (bootstrap significant). Best Direct single-model **MAE** is XGB E+W **83.76**.

---

## 5. Leaderboard C — Stacking / competition RMSE (Direct kg)

**Protocol:** Same outer flight holdout · **Direct** kg · L1 bases + meta · 5-fold flight-grouped OOF on train · `notebooks/11_stacking.py` / `12_verify_ensemble.py`

| Rank | Model | MAE | RMSE | R² | Source |
|-----:|-------|----:|-----:|---:|--------|
| **1 (RMSE)** | **LGBM_meta (5f OOF)** | 84.3 | **202.9** | 0.948 | `table_ensemble_final.csv`, `final_leaderboard.csv` |
| 2 | Ridge (5f OOF) | 84.5 | 203.4 | 0.948 | same |
| 3 | ElasticNet (5f OOF) | 84.7 | 203.7 | 0.948 | same |
| 4 | XGB_meta | 85.2 | 204.8 | 0.947 | same |
| 5 | Cat_meta | 85.5 | 205.1 | 0.947 | same |
| ref | Optuna CatBoost (Direct) | 84.9 | 204.6 | 0.947 | `final_leaderboard.csv` |
| ref | Aircraft Experts | 85.8 | 206.8 | 0.946 | same |
| ref | L1-XGB alone | 83.76 | 212.03 | 0.943 | `table_stacking.csv` |
| ref | PRC challenge winner | — | **200.83** | — | external ref |

**Primary use:** Same-dataset competition benchmarking (gap to winner ≈ 2.1 kg RMSE).  
**Not** the MAE-optimal scientific model; stack MAE (84.3) is **worse** than Direct XGB E+W MAE (83.76) and **much worse** than Flow XGB MAE (79.5).

---

## 6. What to report where

| Audience / claim | Report |
|------------------|--------|
| “Best Level-1 **MAE** (single model)” | **Flow + Energy XGB, MAE 79.52** (Leaderboard A) |
| “Best Level-1 **RMSE** (single model, Fuel-Flow track)” | **Flow + Energy LGBM, RMSE 196.24** (Leaderboard A) |
| “Best Direct hybrid **MAE**” | **XGB Energy+Weather, MAE 83.76** (Leaderboard B) |
| “Best Direct **ensemble RMSE** / PRC-style” | **LGBM_meta stack, RMSE 202.9** (Leaderboard C) |
| “Beats PRC winner?” | Only discuss Leaderboard C vs 200.83; **do not** put Flow 196.4 in that race without re-running stack under Flow |

---

## 7. Consistency actions (this audit)

1. Replace mixed `final_leaderboard.csv` with **protocol-labeled** master tables (see `master_leaderboard_by_protocol.csv`).
2. Update README / PROJECT_STATUS_REPORT so “best internal” is never a single unlabeled number.
3. Keep Flow and Stack results; **do not delete** either — they answer different questions.

---

## 8. Optional future work (if a single ranked table is required)

To make Flow vs Stack **directly** comparable:

1. Fix parquet, features, and hyperparameter budget.
2. Train the **same** stacking pipeline with **Fuel-Flow** L1 targets (recover kg before meta or meta on recovered kg).
3. Evaluate on the **same** outer test flights.
4. Then rank by pre-registered primary metric (recommend: **MAE** for science, **RMSE** only for competition appendix).

Until that experiment exists, **separate leaderboards are mandatory**.

---

*Audit derived from project tables and scripts; test-set identity verified programmatically on cleaned base vs mass parquets (9976 flights, 1996 test flights, full overlap).*
