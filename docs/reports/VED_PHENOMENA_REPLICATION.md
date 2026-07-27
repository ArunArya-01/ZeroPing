# VED Phenomena Replication on AeroTwin

**Date:** 2026-07-25 (updated: matched CatBoost Residual LOTO)  
**Scope:** Diagnosis + one matched residual LOTO re-run (no architecture search).  
**Sources:** existing Level-1 / LOTO / residual / fuel-flow / official artifacts, plus  
`notebooks/15b_loto_residual_matched.py` outputs under `figures/table_loto_residual_*`.

---

## 0. Matched Residual LOTO (2026-07-25) — confounds fixed

**Why:** Prior Direct-vs-Residual claims mixed **CatBoost LOTO Direct** with **RF/XGB/LGBM Level-1 Residual**, so residual loss could be domain *or* model-family.

**What we ran:** Same LOTO folds (12 types, ≥80 flights), same CatBoost (500 iter, lr 0.05, depth 7, seed 42) as `notebooks/15_leave_one_type_out.py`.

| Setting | Direct | Residual (matched) |
|---------|--------|--------------------|
| Model | CatBoost | CatBoost |
| Target | `actual_fuel_kg` | `residual_kg` → + `physics_fuel_kg` |
| Features | BASE + Energy + Weather + **physics** + cats | BASE + Energy + Weather + cats (**no** physics input) |
| Script | notebook 15 (existing) | `notebooks/15b_loto_residual_matched.py` |

### Matched results (kg)

| Regime | Direct MAE | Direct RMSE | Residual MAE | Residual RMSE | Δ (Res − Dir) MAE | Δ RMSE |
|--------|----------:|------------:|-------------:|--------------:|------------------:|-------:|
| **Flight 80/20** | **88.07** | **210.66** | 94.39 | 232.19 | **+6.32** | **+21.53** |
| **LOTO macro (12 types)** | **283.25** | **469.39** | 523.27 | 790.28 | **+240.02** | **+320.90** |
| LOTO macro Flow (ref) | 265.86 | 445.50 | — | — | — | — |

**Per-type residual wins (matched LOTO):** MAE **5/12**, RMSE **4/12**.  
**MAE↔RMSE ranking flip (Residual vs Direct):** **1/12** (A321: residual better MAE, Direct better RMSE by ~0.6 kg).

Residual wins concentrate on **narrowbodies with usable physics** (A20N, A320, A321 MAE, B738, B788).  
Residual **collapses on bad-physics widebodies** (B77W residual MAE 2761 vs Direct 1055; physics MAE 4343). Macro average is dominated by those failures.

### Interpretation

1. **Model-family confound is closed.** Residual still loses under **matched CatBoost** on Level-1 and especially LOTO macro.
2. **Loss is domain / inductive-bias, not “wrong GBDT.”** When OpenAP is badly mis-scaled for the held-out type, residual learning **inherits** that error; Direct can re-learn absolute fuel scale from kinematics without anchoring to a broken physics prior.
3. **Heterogeneity remains:** residual *can* beat Direct on some folds (A20N residual MAE 86.8 vs Direct 294.5) — so residual is not uniformly dead, but **macro ranking is unambiguous**.
4. **VED Residual-wins-on-RMSE pattern does not replicate** under matched protocol.

### Gate for coarser entity holdout (Phenomenon A)

Per plan: run leave-one-body/family **only if residual becomes competitive under matched setup**.

| Condition | Result |
|-----------|--------|
| Residual macro beats or ties Direct | **No** — loses by +240 MAE / +321 RMSE |
| Residual competitive (within ~10 MAE / ~20 RMSE) | **No** |

**Decision: do not invest in coarser entity holdout for residual-ranking justification.**  
Coarser holdout remains optional later for Phenomenon A (non-monotonic difficulty) alone — not required to close residual confounds.

**Artifacts:**  
`figures/table_loto_residual_matched.csv` · `table_loto_paired_direct_residual_matched.csv` ·  
`table_loto_residual_matched_summary.csv` · `table_loto_residual_level1_matched.csv` ·  
`fig_loto_residual_vs_direct_matched.png`

---

## 1. Evaluation setup inventory

### Entity-level splits that already exist

| Split | What is held out | Status | Primary artifacts |
|-------|------------------|--------|-------------------|
| **Flight-level 80/20** (Level 1) | Random **flights** (seed 42); aircraft types still seen in train | ✅ Core protocol | `physics/eval_framework.flight_level_split`, `table_fuel_flow*.csv`, `table_residual_results.csv`, `table_loto_macro_summary.csv` (standard row) |
| **Leave-One-Type-Out (LOTO)** | Entire **ICAO aircraft type** (≥80 flights; **12 types**) | ✅ Complete | `notebooks/15_leave_one_type_out.py`, `table_loto_*.csv` |
| **Body-class hierarchical routing under LOTO** | Same LOTO type folds, but train only on **same fuselage class** (narrow/wide) | ✅ Exists — **not** leave-one-body-class-out | `table_loto_comprehensive.csv` (`hier_body_*`) |
| **Leave-one-family / leave-one-body-class-out** | Coarser entity pure holdout | ❌ **Not implemented** | — |
| **Airline / operator holdout** | Operator entity | ❌ Not in data path used | — |
| **True interval IID** (shuffle intervals ignoring flight) | Random intervals | ❌ Not used; protocol forbids it | Flight grouping is intentional anti-leakage |
| **Official Rank / Final** | **Temporal** months (Sep / Oct), not entity | ✅ Complete | `table_official_leaderboard.csv` |

**Closest IID baseline:** flight-level 80/20 is the project’s quasi-IID / random baseline. It is **not** pure interval IID (flights are the sampling unit).

### Models with comparable results

| Paradigm | AeroTwin equivalent | Level 1 | LOTO | Official Rank+Final |
|----------|---------------------|:-------:|:----:|:-------------------:|
| **Direct** | Predict `actual_fuel_kg` (hybrid + OpenAP feature) | ✅ | ✅ CatBoost E+W | ✅ XGB/LGBM/Cat + ensemble |
| **Residual** | Predict `residual_kg` + add physics | ✅ (rejected; **matched CatBoost confirms**) | ✅ **matched CatBoost** (15b) | ❌ not official track |
| **Rate / Fuel-Flow** | Predict kg/s then × `duration_s` | ✅ | ✅ CatBoost Flow+Energy | ✅ |

LOTO uses **CatBoost** (depth 7) for Direct, Flow, and **now Residual**. Older RF/XGB/LGBM residual tables remain historical only.

---

## 2. Generalization ladder (closest available)

Entity granularity that can be ordered honestly:

```text
Easiest → hardest (observed error scale)

[1] Flight holdout (types seen)     ≈ “IID / random flights”
[2] Official Rank/Final (temporal)  — different axis, not entity
[3] LOTO ICAO type (entity)         ≈ “finer entity holdout”
[—] Leave-one-body / family         — MISSING (would be coarser entity)
```

**Body-class hierarchical rows are not a coarser holdout.** They retrain under the same type LOTO folds with a restricted training pool. Higher error there reflects **routing / data starvation**, not a coarser entity test.

### Table A — Ladder by regime (macro metrics, kg)

Comparable **CatBoost** LOTO macro averages from `table_loto_macro_summary.csv` / `table_loto_evaluation_master.csv`. Level-1 Flow uses same-notebook CatBoost E+W from `table_fuel_flow.csv` where noted.

| Regime | Model | MAE | RMSE | ΔMAE vs flight Direct | ΔRMSE vs flight Direct |
|--------|-------|----:|-----:|----------------------:|-----------------------:|
| **Flight holdout** | Direct · E+W (CatBoost, LOTO ref) | **88.07** | **210.66** | — | — |
| **Flight holdout** | Direct · E+W (XGB, fuel-flow notebook) | 83.76 | 212.03 | −4.9% | +0.7% |
| **Flight holdout** | Flow+Energy (CatBoost) | 83.49 | 202.40 | −5.2% | −3.9% |
| **Flight holdout** | Flow+Energy (XGB MAE-best) | **79.52** | 208.42 | −9.7% | −1.1% |
| **Flight holdout** | Flow+Energy (LGBM RMSE-best) | 80.33 | **196.24** | −8.8% | **−6.8%** |
| **Flight holdout** | Residual · E+W (CatBoost **matched**) | 94.39 | 232.19 | +7.2% | +10.2% |
| **Flight holdout** | Residual (historical LGBM, unmatched) | 108.72 | 293.33 | +23.4% | +39.2% |
| **LOTO type (macro)** | Global Direct · E+W | **283.25** | **469.39** | **+221.6%** | **+122.8%** |
| **LOTO type (macro)** | Global Flow+Energy | **265.86** | **445.50** | +201.9% | +111.5% |
| **LOTO type (macro)** | Global Residual · E+W (CatBoost **matched**) | **523.27** | **790.28** | **+494.1%** | **+275.1%** |
| **LOTO type (macro)** | Global Flow/Mass · E+W | 321.68 | 505.24 | +265.2% | +139.8% |
| **LOTO + body routing** | Hier Direct · E+W | 386.81 | 562.58 | +339.2% | +167.1% |
| **LOTO + body routing** | Hier Flow+Energy | 378.06 | 600.05 | +329.3% | +184.8% |

Degradation % = \(100 \times (\text{metric}/\text{flight Direct }88.07\text{ or }210.66 - 1)\).  
For Flow Level-1, % is still vs the LOTO notebook’s flight Direct baseline so the ladder stays one reference.

### Table B — Official temporal regime (not entity; Combined Rank+Final)

From `table_official_leaderboard.csv`:

| Model | Combined MAE | Combined RMSE | Rank RMSE | Final RMSE |
|-------|-------------:|--------------:|----------:|-----------:|
| Ensemble (Direct+Flow + ridge) | 88.75 | **228.25** | 239.18 | 220.86 |
| CatBoost FuelFlow | **78.62** | 231.26 | 244.90 | 221.94 |
| LGBM FuelFlow | 82.49 | 230.18 | 249.83 | 216.46 |
| CatBoost Direct | 120.01 | 255.38 | 253.21 | 256.78 |
| LGBM Direct | 121.47 | 258.59 | 263.26 | 255.50 |
| OpenAP only | 472.03 | 1268.37 | 1191.95 | 1315.65 |

Temporal Rank/Final sits **between** flight holdout (~196–212 RMSE) and LOTO (~445–469), as expected for a different shift axis.

### Table C — Residual vs Direct (**matched CatBoost**, 2026-07-25)

| Regime | Direct MAE / RMSE | Residual MAE / RMSE | Winner |
|--------|-------------------:|--------------------:|:------:|
| Flight 80/20 (CatBoost) | **88.07 / 210.66** | 94.39 / 232.19 | Direct both |
| LOTO macro (CatBoost) | **283.25 / 469.39** | 523.27 / 790.28 | Direct both |
| LOTO per-type wins | — | Residual MAE **5/12**, RMSE **4/12** | Mixed |

Historical unmatched RF/XGB/LGBM residual (Level 1) also lost; matched re-run confirms this is not a model-family artifact. See §0.

### Table D — Metric-dependent ranking (Direct vs Rate/Flow/Residual)

| Regime | MAE winner | RMSE winner | Flip? |
|--------|------------|-------------|:-----:|
| Flight holdout, same notebook Direct vs Flow (XGB) | Flow (79.5 vs 83.8) | Flow (208.4 vs 212.0) | No |
| Flight holdout, **within Flow** (XGB vs LGBM) | **XGB** | **LGBM** | **Yes** (models) |
| Flight holdout, matched Direct vs Residual (CatBoost) | Direct | Direct | No |
| LOTO global macro Direct vs Flow | Flow (265.9 vs 283.2) | Flow (445.5 vs 469.4) | No |
| LOTO global macro Direct vs Residual (matched) | Direct | Direct | No |
| LOTO global **per-type** Direct vs Residual | Residual 5 / Direct 7 | Residual 4 / Direct 8 | **1 flip** (A321) |
| LOTO global **per-type** Direct vs Flow | Flow 7 / Direct 5 | Flow 8 / Direct 4 | **1 flip** (A359) |
| LOTO hierarchical body macro | Flow (378.1 vs 386.8) | **Direct** (562.6 vs 600.1) | **Yes** |
| Official Combined, Flow vs Direct singles | Flow always | Flow always | No |
| Official Combined, **Ensemble vs best Flow** | **Cat/LGBM Flow** | **Ensemble** | **Yes** |

**A359 (global LOTO):** Direct better MAE (249.6 vs 278.0), Flow better RMSE (440.8 vs 482.3) — classic MAE/RMSE ranking flip for the same pair of models.

---

## 3. Answers to the research questions

### Phenomenon A — Non-monotonic difficulty across entity granularities

**Is difficulty non-monotonic across available entity granularities?**

**Cannot confirm. Closest reading: not testable with current splits; observed entity axis is monotonic.**

- Only **one true entity holdout granularity** exists: **LOTO by ICAO type**.
- Flight holdout is easier (~3× lower MAE than LOTO Direct) but is **not** a finer *entity* holdout of the same hierarchy — types remain in training.
- There is **no** leave-one-body-class-out / leave-one-family-out / leave-one-vehicle-out analogue to VED’s coarser vehicle holdout.
- Hierarchical body routing looks “harder” than global LOTO, but that is a **training-recipe** comparison under the same type folds, not coarser entity holdout.

**VED pattern:** coarser vehicle holdout harder (by RMSE) than finer LOEO.  
**AeroTwin:** missing the coarser entity rung → **does not replicate / not testable**.

### Phenomenon B — Metric-dependent model ranking

**Does ranking between Direct and Residual (or Rate) ever flip when switching RMSE ↔ MAE?**

| Pair | Flip MAE ↔ RMSE? | Where |
|------|------------------|--------|
| Direct vs **Residual** (matched CatBoost) | **Rare** | Macro: no (Direct both). Per-type: **1/12** (A321). Residual never wins *both* metrics on a fold where Direct wins one except that flip. |
| Direct vs **Rate/Flow** | **Sometimes** | A359 LOTO type; hierarchical body **macro** (Flow wins MAE, Direct wins RMSE) |
| Within Rate or Ensemble vs Flow | **Yes** | Level-1 XGB vs LGBM Flow; official Ensemble (best RMSE) vs Flow singles (best MAE) |

So AeroTwin shows **metric-dependent ranking** mainly for **Direct vs Rate** and architecture variants. **Matched Residual does not dominate RMSE** (opposite of VED).

### How strongly do the VED patterns replicate?

| Phenomenon | Verdict | Strength |
|------------|---------|----------|
| **A Non-monotonic entity difficulty** | **Does not replicate** (insufficient entity ladder) | N/A — structural gap; coarser holdout **deferred** after residual gate |
| **B Metric-dependent ranking** | **Partially replicates** | Moderate for Direct↔Rate; **fails** for Residual↔Direct (matched) |
| **Residual superiority under shift** | **Does not replicate** | Matched CatBoost LOTO: Residual macro **much worse** |

### Overall conclusion

> **Partially replicates** (unchanged at headline; residual claim now **confound-free**).

- **B partially:** MAE/RMSE ranking flips appear for Direct vs Fuel-Flow (A359; hierarchical-body macro). Residual vs Direct does **not** show VED-style Residual-RMSE dominance.
- **A does not:** still missing coarser pure entity holdout; **not run** after residual gate (residual still loses → low ROI for residual paper angle).
- **Residual (matched):** loses Level-1 and LOTO macro under identical CatBoost; failure is **honestly domain/physics-anchor dependent**, not an unmatched-tree artifact.

---

## 4. Limitations (explicit)

1. **No pure coarser entity holdout** — body-class hierarchical is a routing mask, not leave-one-class-out. Deferred after residual gate.
2. ~~No Residual LOTO~~ **Resolved:** matched CatBoost residual LOTO complete (§0).
3. ~~Model family mismatch for residual LOTO~~ **Resolved** for Direct vs Residual. Historical RF/XGB/LGBM residual tables are archival only.
4. **Macro LOTO averages** treat 12 types equally; B77W dominates residual failure (paired table).
5. **Flight holdout ≠ interval IID** — project deliberately uses flight groups.
6. **Official Rank/Final** is temporal shift, not entity granularity.
7. Residual features intentionally **exclude** `physics_fuel_kg` (classic residual); Direct includes it as a hybrid feature — this is the intended inductive-bias contrast, not a bug.
8. Small protocol differences remain across older notebooks (`figures/LEADERBOARD_AUDIT.md`).

---

## 5. Minimal experiments if you want a fairer VED-style test later

| ID | Experiment | Status |
|----|------------|--------|
| ~~R1~~ | Matched CatBoost Residual LOTO | ✅ Done (2026-07-25) |
| A1 | Leave-one-body-class-out | ⬜ Deferred — residual still loses; optional for Phenomenon A only |
| A2 | Leave-one-family-out | ⬜ Optional intermediate granularity |

---

## 6. Artifact index

| File | Use |
|------|-----|
| `figures/table_loto_macro_summary.csv` | Ladder macro Direct/Flow/hier/**residual** |
| `figures/table_loto_comprehensive.csv` | Per-type MAE/RMSE Direct/Flow/hier |
| `figures/table_loto_residual_matched.csv` | Per-type matched residual LOTO |
| `figures/table_loto_paired_direct_residual_matched.csv` | Paired Direct vs Residual (matched) |
| `figures/table_loto_residual_matched_summary.csv` | Flight + LOTO macro summary |
| `figures/fig_loto_residual_vs_direct_matched.png` | Residual vs Direct plot |
| `figures/table_loto_paired_per_type.csv` | Paired Direct vs Flow + flips |
| `figures/table_residual_results.csv` | Historical residual Level 1 (unmatched trees) |
| `figures/table_fuel_flow.csv` / `table_fuel_flow_ablation.csv` | Direct vs Rate Level 1 |
| `figures/table_official_leaderboard.csv` | Temporal Rank/Final |
| `figures/LEADERBOARD_AUDIT.md` | Protocol separation rules |
| `notebooks/15_leave_one_type_out.py` | LOTO Direct/Flow/hier |
| `notebooks/15b_loto_residual_matched.py` | Matched residual LOTO |

*End of diagnostic report.*
