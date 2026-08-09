# AeroTwin — Complete Experiment & Results Catalog

**Last updated:** 2026-08-09  
**Purpose:** Single reference of **every major experiment** and its **quantitative results**, including **pre-R3 work and negative results**.  
**Units:** RMSE/MAE/bias in **kg** unless noted.  
**Sources:** frozen CSVs under `docs/reports/tables/`, phase reports, `results/distillation/`.

This document does **not** invent experiments or reinterpret conclusions.

---

## Table of contents

### Part A — Before R3 (foundation + official gap-closing)

0. [Evaluation regimes (important)](#0-evaluation-regimes-important)  
A1. [Dataset & protocols](#a1-dataset--protocols)  
A2. [Level-1 feature ablations (flight 80/20)](#a2-level-1-feature-ablations-flight-8020)  
A3. [Physics vs hybrid vs residual (Level-1)](#a3-physics-vs-hybrid-vs-residual-level-1)  
A4. [Fuel-flow target ablation](#a4-fuel-flow-target-ablation)  
A5. [Mass / OpenAP ablations](#a5-mass--openap-ablations)  
A6. [Energy + weather stacking (V3 E5–E7)](#a6-energy--weather-stacking-v3-e5e7)  
A7. [Direct stacking / ensemble meta (Level-1)](#a7-direct-stacking--ensemble-meta-level-1)  
A8. [Leave-one-type-out (LOTO) & residual matched](#a8-leave-one-type-out-loto--residual-matched)  
A9. [Official PRC v1 ensemble (canonical)](#a9-official-prc-v1-ensemble-canonical)  
A10. [Gap-closing campaign P1–P5 (all gates)](#a10-gap-closing-campaign-p1p5-all-gates)  
A11. [R1 — OpenAP descriptors on heavy specialist](#a11-r1--openap-descriptors-on-heavy-specialist)  
A12. [R2 — Descriptor fixes + feature families](#a12-r2--descriptor-fixes--feature-families)  

### Part B — R3 and distillation research arc

B0. [R3 — Dynamic mass teacher (current frozen teacher)](#b0-r3--dynamic-mass-teacher-current-frozen-teacher)  
B1. [Distillation: α/β sweep](#b1-distillation-αβ-sweep)  
B2. [Distillation: capacity scaling](#b2-distillation-capacity-scaling)  
B3. [Official Final evaluation (students)](#b3-official-final-evaluation-students)  
B4. [Combined Rank+Final evaluation](#b4-combined-rankfinal-evaluation)  
B5. [FT-Transformer experiment](#b5-ft-transformer-experiment)  
B6. [Phase 0 — Distribution shift](#b6-phase-0--distribution-shift)  
B7. [Phase 1A — Teacher uncertainty](#b7-phase-1a--teacher-uncertainty)  
B8. [Phase 1B — VGKD (negative)](#b8-phase-1b--vgkd-negative)  
B9. [Phase 2 — Transformer robustness](#b9-phase-2--transformer-robustness)  
B10. [Phase 3 — Mechanism validation](#b10-phase-3--mechanism-validation)  
B11. [Phase 3.5 — Smoothness causal intervention](#b11-phase-35--smoothness-causal-intervention)  
B12. [Attention routing analysis (negative)](#b12-attention-routing-analysis-negative)  
B13. [Per-type scoreboard (Final, n≥50)](#b13-per-type-scoreboard-final-n50)  
B14. [All hypothesis outcomes](#b14-all-hypothesis-outcomes)  
B15. [Artifact index](#b15-artifact-index)

---

# Part A — Before R3

## 0. Evaluation regimes (important)

| Regime | What it measures | Typical metrics | Comparability |
|--------|------------------|-----------------|---------------|
| **Level-1 flight 80/20** | Random flight holdout on train-era data | MAE/RMSE | **Not** official Rank+Final |
| **LOTO** | Leave-one-aircraft-type-out | Macro MAE/RMSE over 12 types | Entity shift; harder |
| **Official Rank+Final** | Temporal Sep/Oct holdout | Rank / Final / **Combined** RMSE | PRC leaderboard parity |
| **Type-macro (students)** | Mean per-type RMSE on Final (n≥50) | Type-macro | Distillation-era only |

**Do not** mix Level-1 RMSE (~196–215) with Official Combined (~221–228) as one ranking.

---

## A1. Dataset & protocols

| Split | Period | Flights (usable) | Intervals |
|-------|--------|-----------------:|----------:|
| Train | Apr–Aug 2025 | 10,000 | 119,032 |
| Rank | Sep 2025 | 1,888 | 24,158 |
| Final | Oct 2025 | 2,824 (after feature build) | **37,170** |

**Published PRC winner (paper reference):** Combined RMSE ≈ **201 kg** (external; not AeroTwin).

---

## A2. Level-1 feature ablations (flight 80/20)

### A2.1 Physics ablation (table_physics_ablation)

| Model | Feature set | MAE | RMSE | R² |
|-------|-------------|----:|-----:|---:|
| RF | Full Hybrid (w/ physics) | 86.27 | 228.80 | 0.934 |
| XGB | Full Hybrid | 86.31 | 224.12 | 0.937 |
| LGBM | Full Hybrid | 89.13 | 215.76 | 0.941 |
| RF | No Physics | 87.13 | 232.78 | 0.932 |
| XGB | No Physics | 89.46 | 230.56 | 0.933 |
| LGBM | No Physics | 91.76 | 219.58 | 0.939 |
| — | **Physics Only** | **667.62** | **1582.38** | **−2.16** |

**Bootstrap (XGB Hybrid vs No Physics):** ΔMAE ≈ **−3.15** (CI [−7.08, −0.42], p_boot ≈ 0.006) → physics features help hybrid.  
**Physics Only:** catastrophic — ML hybrid required.

### A2.2 Energy vs OpenAP hybrid (table_energy_results)

| Approach | Best RMSE (LGBM) | Best MAE (XGB) |
|----------|-----------------:|---------------:|
| Energy Hybrid | **208.34** | 84.48 |
| OpenAP Hybrid | 215.76 | 86.31 |
| No Physics | 219.58 | 89.46 |

**Significance (XGB Energy Hybrid vs OpenAP Hybrid):** ΔMAE ≈ **−1.82**, CI [−2.92, −0.67], p_boot ≈ 0.002 → Energy Hybrid better.

### A2.3 Weather (V3 E5)

| Approach | XGB MAE | XGB RMSE | LGBM RMSE |
|----------|--------:|---------:|----------:|
| No Physics | 89.46 | 230.56 | 219.58 |
| OpenAP Hybrid | 86.31 | 224.12 | 215.76 |
| Weather Hybrid | 86.59 | 224.85 | **212.98** |

Weather alone ≈ OpenAP hybrid; not a large leap.

---

## A3. Physics vs hybrid vs residual (Level-1)

### Residual learning rejected (table_residual_results)

| Approach | Best model RMSE | Best MAE | vs OpenAP Hybrid |
|----------|----------------:|---------:|------------------|
| Energy Hybrid | LGBM **208.34** | XGB 84.48 | **better** |
| OpenAP Hybrid | LGBM 215.76 | XGB 86.31 | reference |
| Energy Residual | LGBM 237.25 | XGB 94.01 | **worse** |
| Residual RF/XGB/LGBM | LGBM 293.33 | XGB 107.09 | **much worse** |
| Operational Residual | LGBM 296.90 | XGB 107.22 | **much worse** |
| Operational Hybrid | LGBM 214.90 | RF 86.16 | ≈ hybrid |

**Significance vs OpenAP Hybrid (XGB):**

| Comparison | ΔMAE | Interpretation |
|------------|-----:|----------------|
| Residual-RF/XGB/LGBM | **+20.8** | Hybrid significantly better |
| Energy Residual | **+7.7** | Hybrid significantly better |
| Operational Residual | **+20.9** | Hybrid significantly better |
| Operational Hybrid | +0.46 | No significant evidence |
| Energy Hybrid | **−1.82** | Energy significantly better |

**Decision:** Residual learning (predict residual + add OpenAP) **rejected** under Level-1 hybrid comparison. Direct hybrid preferred.

### Transformer residual (early sequence model)

| Model | Target | MAE | RMSE |
|-------|--------|----:|-----:|
| OpenAP only | fuel kg | 638.75 | 1536.17 |
| Transformer residual/hybrid | residual/fuel | 412.33 | 1317.28 |

**Not competitive** with GBDTs (~210 RMSE). Not pursued as official track.

---

## A4. Fuel-flow target ablation

Predicting **fuel flow (kg/s) × duration** vs **direct fuel (kg)**.

### Core comparison (table_fuel_flow)

| Approach | Model | MAE | RMSE | R² |
|----------|-------|----:|-----:|---:|
| Direct fuel (E+W) | XGB | 83.76 | 212.03 | 0.943 |
| Direct fuel (E+W) | LGBM | 86.67 | 208.61 | 0.945 |
| Direct fuel (E+W) | CatBoost | 91.32 | 211.45 | 0.944 |
| **Fuel flow (E+W)** | XGB | **80.06** | 210.81 | 0.944 |
| **Fuel flow (E+W)** | **LGBM** | **80.71** | **196.41** | **0.951** |
| Fuel flow (E+W) | CatBoost | 83.49 | 202.40 | 0.948 |

### Flow feature groups (table_fuel_flow_ablation) — best LGBM/XGB

| Approach | Best RMSE | Best MAE |
|----------|----------:|---------:|
| Direct (E+W) ref | 208.61 (LGBM) | 83.76 (XGB) |
| **Flow + Energy** | **196.24 (LGBM)** | **79.52 (XGB)** |
| Flow + Energy+Weather | 196.41 (LGBM) | 80.06 (XGB) |
| Flow + Energy+Weather+Mass | 197.20 (LGBM) | 79.57 (XGB) |
| Flow + Mass | 201.49 (LGBM) | 80.87 (XGB) |
| Flow + base | 208.11 (LGBM) | 81.49 (XGB) |

**Decision:** Fuel-flow target **beats** Direct on Level-1 (replicated; ΔMAE Flow−Direct ≈ **−6.0**, bootstrap p≈1.0 for flow better).  
**Best single-model Level-1 RMSE cited:** LGBM Flow+Energy **196.24**.

---

## A5. Mass / OpenAP ablations

| Approach | Best LGBM RMSE | Best XGB MAE | Note |
|----------|---------------:|-------------:|------|
| A: Energy+Weather | 208.61 | **83.76** | Strong baseline |
| B: Energy+Weather+Mass | 214.98 | 84.65 | Mass **does not** clearly help Level-1 Direct |
| C: Mass only | 213.24 | 86.02 | |
| D: Mass+OpenAP | 215.46 | 85.28 | |
| OpenAP Hybrid (ref) | 215.76 | 86.31 | |

**Negative / weak result:** Crude static mass features on Level-1 Direct track often **hurt or stall** RMSE vs Energy+Weather alone. (Later **R3 dynamic mass** succeeds on **official** protocol — different regime.)

---

## A6. Energy + weather stacking (V3 E5–E7)

| Exp | Approach | XGB MAE | Outcome |
|-----|----------|--------:|---------|
| baseline | OpenAP Hybrid | 86.31 | reference |
| E5 | No Physics | 89.46 | worse |
| E5 | Weather Hybrid | 86.59 | ≈ OpenAP |
| E6 | Energy Hybrid | 84.48 | better |
| E6 | **Energy+Weather Hybrid** | **83.76** | **keep** |
| E7 | MLP Residual | **103.73** | **reject** (much worse) |

**E7 MLP residual:** negative result on Level-1 MAE.

---

## A7. Direct stacking / ensemble meta (Level-1)

**Not** official Rank+Final. Stacking Direct bases on train OOF:

| Meta-learner | MAE | RMSE | R² |
|--------------|----:|-----:|---:|
| **LGBM_meta (5f OOF)** | **84.3** | **202.9** | **0.9481** |
| Ridge (5f OOF) | 84.5 | 203.4 | 0.9479 |
| ElasticNet | 84.7 | 203.7 | 0.9477 |
| XGB_meta | 85.2 | 204.8 | 0.9472 |
| Cat_meta | 85.5 | 205.1 | 0.9470 |

**Note (LEADERBOARD_AUDIT):** Level-1 Flow RMSE 196.4 and Direct stack 202.9 are **different targets** — do not rank as one list.

---

## A8. Leave-one-type-out (LOTO) & residual matched

### Macro summary (12 types, CatBoost)

| Approach | Macro MAE | Macro RMSE | R² | n_types |
|----------|----------:|-----------:|---:|--------:|
| Standard flight 80/20 · Direct · E+W | **88.07** | **210.66** | 0.944 | — |
| LOTO Global · Flow+Energy | **265.86** | **445.50** | 0.230 | 12 |
| LOTO Global · Direct · E+W | 283.25 | 469.39 | 0.136 | 12 |
| LOTO Hier body · Direct | 386.81 | 562.58 | 0.402 | 12 |
| LOTO Hier body · Flow+Energy | 378.06 | 600.05 | 0.200 | 12 |
| LOTO Global · Residual (matched CatBoost) | **523.27** | **790.28** | 0.006 | 12 |
| LOTO Global · Flow/Mass | 321.68 | 505.24 | 0.580 | 12 |

### Matched residual vs Direct (confounds closed)

| Regime | Direct MAE | Residual MAE | Δ (Res−Dir) |
|--------|----------:|-------------:|------------:|
| Flight 80/20 | **88.07** | 94.39 | **+6.32** |
| LOTO macro | **283.25** | 523.27 | **+240.02** |

| Regime | Direct RMSE | Residual RMSE | Δ RMSE |
|--------|------------:|--------------:|-------:|
| Flight 80/20 | 210.66 | 232.19 | **+21.53** |
| LOTO macro | 469.39 | 790.28 | **+320.90** |

**Per-type residual MAE wins:** 5/12 types (mainly narrowbodies with usable physics).  
**Macro:** residual **collapses** on bad-physics widebodies (e.g. B77W).

**Decisions (negative / gating):**

1. Residual learning **rejected** for official / macro entity shift.  
2. **Do not** invest in coarser entity holdout for residual ranking justification.  
3. Flow+Energy is the strongest LOTO paradigm among tested.

### Cross-dataset Flow vs Direct replication

| Check | Direct MAE | Flow MAE | Δ | Replicated? |
|-------|----------:|---------:|--:|:-----------:|
| featured_dataset (n_test flights 1996) | 88.20 | 82.20 | **−6.01** | **Yes** |

---

## A9. Official PRC v1 ensemble (canonical)

**Protocol:** Train-only fits; evaluate Rank + Final + Combined. Meta: Ridge on 6 bases.

| Metric | Value |
|--------|------:|
| Rank RMSE | **239.18** |
| Rank MAE | 90.89 |
| Final RMSE | **220.86** |
| Final MAE | 87.35 |
| **Combined RMSE** | **228.25** |
| Combined MAE | 88.75 |
| Combined R² | 0.913 |
| Combined bias (error analysis) | ~**+28 to +31** |
| Δ vs published winner (~201) | **≈ +27.3** |
| Combined 95% CI (bootstrap) | **[207.1, 249.4]** — does **not** exclude 201 → **no superiority claim** |

**Best single FuelFlow baseline (official track):** LGBM FuelFlow Combined **230.18** (worse than ensemble).

---

## A10. Gap-closing campaign P1–P5 (all gates)

**Baseline floor:** Combined **228.25**.  
**Best accepted (v1.1):** Combined **227.44** (−0.81).  
**Report:** `official_gap_closing_report.md` · `table_gap_closing_leaderboard.csv`

### Full leaderboard (Combined ascending)

| Variant | Combined | Rank | Final | Bias | Δ vs 228.25 | Gate |
|---------|---------:|-----:|------:|-----:|------------:|:-----|
| P2b_heavy_cat_flow_on_raw_ensemble | 227.43 | 235.29 | 222.17 | +14.8 | −0.82 | REJECT* |
| **P2_heavy_cat_flow_on_P1base** | **227.44** | **235.30** | **222.18** | **+15.0** | **−0.81** | **KEEP** |
| P5_flow_only_ridge_meta | 227.73 | 242.58 | 217.53 | +6.9 | −0.53 | **REJECT** |
| P5_nonneg_weights_flow3 | 227.78 | 242.56 | 217.63 | +7.0 | −0.47 | **REJECT** |
| P2b_heavy_lgbm_flow_on_raw | 228.15 | 242.96 | 217.99 | +17.9 | −0.10 | REJECT |
| **P1E_affine_by_phase** | **228.16** | 239.02 | 220.80 | +28.2 | −0.10 | **KEEP (minor)** |
| P2_heavy_lgbm_flow_on_P1base | 228.16 | 242.97 | 218.00 | +18.1 | −0.09 | KEEP then superseded |
| P5_nonneg_weights_all6 | 228.18 | 238.76 | 221.04 | +29.8 | −0.07 | **REJECT** |
| baseline_official_v1 | **228.25** | 239.18 | 220.86 | +28.3 | 0 | REFERENCE |
| P1A_global_affine | 228.25 | 239.18 | 220.86 | +28.3 | ~0 | **REJECT** |
| P2c_heavy_lgbm+global_affine | 228.29 | 243.36 | 217.94 | +16.9 | +0.04 | **REJECT** |
| P1C_affine_by_aircraft_class | 228.34 | 239.13 | 221.05 | +28.1 | +0.09 | **REJECT** |
| P1D_affine_by_haul | 228.58 | 239.28 | 221.35 | +27.6 | +0.33 | **REJECT** |
| P1B_isotonic | 228.92 | 241.32 | 220.48 | +28.8 | +0.67 | **REJECT** |
| P1B_isotonic_on_lgbm_flow | 229.08 | 246.76 | 216.81 | +8.1 | +0.83 | **REJECT** |
| P1A_affine_on_lgbm_flow | 230.16 | 249.85 | 216.41 | +7.8 | +1.91 | **REJECT** |
| baseline_lgbm_fuelflow | 230.18 | 249.83 | 216.46 | +7.3 | +1.93 | reference single |
| P1C_class_affine_on_lgbm_flow | 230.38 | 249.61 | 216.97 | +7.3 | +2.13 | **REJECT** |
| P1D_haul_affine_on_lgbm_flow | 230.45 | 249.81 | 216.95 | +6.7 | +2.20 | **REJECT** |
| P2_heavy_xgb_flow_on_P1base | 236.58 | 240.95 | 233.70 | +19.1 | +8.33 | **REJECT** |
| **P3_cruise_residual_lgbm** | **244.87** | 255.45 | 237.74 | +25.2 | **+16.6** | **REJECT** |

\*P2b competitive but accepted stack is P1E + P2 Cat on P1 base.

### Hypothesis outcomes (gap-closing)

| ID | Hypothesis | Expected | Realized | Decision |
|----|------------|----------|----------|----------|
| P1A | Global affine fixes +31 kg bias | −5 to −15 | ~0 | **REJECT** |
| P1B | Isotonic calibration | −5 to −15 | worse | **REJECT** |
| P1C | Class-conditional affine | −5 to −12 | +0.09 | **REJECT** |
| P1D | Haul-conditional affine | −5 to −12 | +0.33 | **REJECT** |
| **P1E** | Phase-conditional affine | −3 to −10 | **−0.10** | **KEEP (tiny)** |
| **P2 Cat** | Heavy FuelFlow specialist | −5 to −12 | **−0.81 Combined** (Rank −3.9) | **KEEP (best)** |
| P2 XGB | Same with XGB | −5 to −12 | +8.3 | **REJECT** |
| P2c | Specialist + global affine | −8 to −18 | +0.04 | **REJECT** |
| **P3** | Cruise residual LGBM | improve | **+16.6** | **REJECT** |
| P5 | Flow-only / nonneg reweight | −2 to −5 | not better than P2 | **REJECT** |

### Subgroup effects (best P2 Cat vs official)

| Subgroup | Baseline | Best P2 Cat | Δ |
|----------|---------:|------------:|--:|
| Combined | 228.25 | **227.44** | **−0.81** |
| Rank | 239.18 | **235.30** | **−3.9** |
| Final | 220.86 | 222.18 | +1.3 |
| B744 | ~1135 | **883** | large drop |
| B77W | ~847 | **831** | better |
| A359 | ~331 | 344 | slightly worse |
| Bias | ~+31 | **+15** | halved |

**Accepted stack v1.1:** Official 6-base ensemble + **P1E** + **P2 Cat heavy FuelFlow**. Combined **227.44**. Remaining gap to winner ≈ **26.4 kg**.

---

## A11. R1 — OpenAP descriptors on heavy specialist

| Quantity | Value |
|----------|------:|
| Official baseline Combined | 228.25 |
| v1.1 Combined | 227.44 |
| **R1 best Combined** | **226.19** |
| Rank | 235.21 |
| Final | 220.13 |
| Bias | +13.71 |
| Δ vs official | **−2.06** |
| Δ vs 227.44 | **−1.25** |
| Heavy RMSE | 423.13 |
| Narrow RMSE | 80.66 |
| A359 / B77W / B744 | 342.0 / 844.5 / 821.0 |
| n variants | 5 |

**Best variant:** `R1_heavy_cat_openap_descriptors`.

---

## A12. R2 — Descriptor fixes + feature families

**Core fix:** B744 / B77L / A306 OpenAP descriptors.

| Quantity | Value |
|----------|------:|
| **R2 Combined (KEEP family)** | **225.25** |
| Rank | 234.81 |
| Final | 218.82 |
| Bias | +11.72 |
| Δ vs official 228.25 | **−3.00** |
| Heavy / Narrow | 421.18 / 80.66 |
| B744 RMSE | **863.3** (improved vs R1 821 still hard) |

Variants R2a–R2e (descriptors ± aircraft chars / mass proxies / cruise / interactions) **tied** at Combined **225.25** in recorded leaderboard (KEEP).

**Negative notes (R2 audits, backlog):** Fuel-flow filtering (<0.05 or >6.5 kg/s) **degrades** RMSE (+1.5 to +3.3). Not accepted.

---

# Part B — R3 and distillation research arc

## B0. R3 — Dynamic mass teacher (current frozen teacher)

### R3 single-model path (LGBM dynamic mass)

| Quantity | Value |
|----------|------:|
| Combined | 224.23 |
| Rank | 238.64 |
| Final | 214.34 |
| Bias | **+0.98** |
| Δ vs 225.25 (R2) | −1.02 |
| Δ vs 228.25 | −4.02 |
| n mass features | **21** |

### R3 ensemble + P1E (frozen teacher for KD)

| Metric | Value |
|--------|------:|
| Rank RMSE | **232.53** |
| Final RMSE (official) | **213.73** |
| Final RMSE (student-parity audit) | **213.62** |
| **Combined RMSE** | **221.33** |
| Combined bias | **+3.85** |
| Heavy RMSE | **416.1** |
| Narrow RMSE | **75.0** |
| Δ vs 228.25 | **−6.92** |
| Δ vs 225.25 (R2) | **−3.92** |

**Teacher type:** 6× GBDT (XGB/LGBM/CatBoost × Direct + Fuel-Flow) + Ridge meta + P1E + **21 R3 dynamic mass features**.

---

## B1. Distillation: α/β sweep

**Script:** `03_alpha_beta_sweep.py` · **Artifacts:** `results/distillation/alpha_beta_sweep/`

| Name | α | β | Val RMSE |
|------|--:|--:|---------:|
| **KD-1** | **0.1** | **0.9** | **188.31** |
| KD-2 | 0.2 | 0.8 | 189.48 |
| KD-0 | 0.0 | 1.0 | 189.49 |
| KD-3 | 0.3 | 0.7 | 191.58 |
| KD-4 | 0.5 | 0.5 | 196.78 |
| KD-5 | 0.7 | 0.3 | 209.22 |
| KD-7 | 1.0 | 0.0 | 221.32 |
| KD-6 | 0.9 | 0.1 | 222.04 |

**Selected:** α=0.1, β=0.9.

---

## B2. Distillation: capacity scaling

**Script:** `04_capacity_scaling.py` · α=0.1, β=0.9, seed 42

| Size | Hidden | Params | Val RMSE | Val MAE | Val bias | Val R² |
|------|--------|-------:|---------:|--------:|---------:|-------:|
| Tiny | 320×160 | 239,041 | 270.55 | 89.65 | −16.42 | 0.904 |
| Small | 576×288 | 504,001 | 241.73 | 83.83 | +0.28 | 0.923 |
| Medium | 1024×512 | 1,125,377 | 235.04 | 82.46 | −1.46 | 0.927 |
| **Large** | **1792×1024** | **2,887,425** | **229.70** | **81.70** | **−4.22** | **0.931** |
| XLarge | 2560×2048 | 6,748,673 | **228.14** | 81.29 | −0.17 | 0.932 |

Teacher val RMSE (capacity split): **244.14**  
XLarge multi-seed val: **228.49 ± 0.92**  
**Deploy decision:** Large wins **held-out Final** vs XLarge → freeze Large.

| Model | CPU ms / sample |
|-------|----------------:|
| Large | **~0.26** |
| XLarge | ~0.52 |
| FT | ~9.59 |
| Teacher | ~52 |

---

## B3. Official Final evaluation (students)

**Script:** `05_test_evaluation.py`

| Model | Params | Val RMSE | **Final RMSE** | MAE | Bias | R² |
|-------|-------:|---------:|---------------:|----:|-----:|---:|
| **Large (deploy)** | **2.89M** | 229.70 | **215.85** | **76.69** | **+5.25** | **0.9220** |
| XLarge | 6.75M | 228.14 | 218.59 | 77.36 | +6.41 | 0.9201 |
| R3 Teacher | ensemble | — | **213.62** | 74.14 | +4.87 | 0.9236 |

Large − Teacher Final: **+2.23**. Val→Final Large: **−13.85** (test better).

---

## B4. Combined Rank+Final evaluation

**Script:** `07_combined_evaluation.py`

| Model | Rank | Final | **Combined** | Δ teacher Comb. |
|-------|-----:|------:|-------------:|----------------:|
| **Large** | **240.66** | **215.85** | **225.95** | **+4.62** |
| XLarge | 244.40 | 218.59 | 229.10 | +7.77 |
| R3 Teacher | 232.53 | 213.62 | **221.33** | 0 |

---

## B5. FT-Transformer experiment

**Params:** 1,458,625 · d_token=192 · 3 blocks · 8 heads · 61 tokens  
**KD:** α=0.1, β=0.9

| Metric | FT | Large | Winner |
|--------|---:|------:|--------|
| Val | 236.08 | 229.70 | Large |
| Final | 224.12 | **215.85** | Large |
| Combined | 233.35 | **225.95** | Large |
| Type-macro | **261.15** | 270.61 | **FT** |
| Body-macro | 249.58 | **239.63** | Large |

---

## B6. Phase 0 — Distribution shift

| Protocol | Teacher | Large | XLarge | FT |
|----------|--------:|------:|-------:|---:|
| Flight Final | 213.62 | 215.85 | 218.59 | 224.12 |
| Type-macro | 256.79 | 270.61 | 276.01 | **261.15** |
| Body-macro | 237.55 | **239.63** | 242.08 | 249.58 |
| Gap vs teacher (flight) | 0 | +2.23 | +4.96 | +10.50 |
| Gap vs teacher (type-macro) | 0 | **+13.82** | +19.22 | **+4.35** |

**Ranking reverse:** Flight Large>FT; Type-macro **FT>Large**. Body-macro does **not** reverse.

---

## B7. Phase 1A — Teacher uncertainty

| Check | Value |
|-------|------:|
| Spearman(disagreement, teacher \|err\|) | **0.426** |
| Spearman(disagreement, Large \|err\|) | **0.435** |
| Bin calibration ρ teacher / Large | **0.976 / 0.952** |
| Type-level Spearman(disagreement, teacher RMSE) | **0.757** |
| Top−bottom type student-gap Δ | **+49.4** |

Disagreement is **informative** → unblocked VGKD.

---

## B8. Phase 1B — VGKD (negative)

| Run | Final | Type-macro | Body-macro | Combined |
|-----|------:|-----------:|-----------:|---------:|
| **fixed_kd_large** | **215.85** | **270.61** | **239.63** | **225.95** |
| static_beta0.7 | 218.87 | 280.82 | 242.43 | 228.71 |
| static_beta0.8 | 216.45 | 273.79 | 240.01 | 226.18 |
| static_beta0.9 / **vgkd_exp_lam0.0** | 216.10 | 269.76 | 239.54 | 226.16 |
| vgkd_exp_lam0.25 | 227.79 | 296.98 | 251.40 | 237.82 |
| vgkd_exp_lam0.5 | 231.30 | 302.10 | 255.06 | 241.27 |
| vgkd_exp_lam1.0 | 232.92 | 301.05 | 257.01 | 240.37 |
| vgkd_exp_lam2.0 | 236.85 | 315.53 | 261.32 | 243.83 |
| vgkd_lin_lam0.25 | 234.04 | 309.76 | 257.77 | 246.04 |
| vgkd_lin_lam0.5 | 237.39 | 314.01 | 261.17 | 244.69 |
| vgkd_lin_lam1.0 | 237.74 | 315.95 | 262.55 | 245.12 |
| vgkd_lin_lam2.0 | 240.88 | 326.58 | 265.99 | 247.45 |
| vgkd_oracle_lam1.0 | 242.16 | 322.53 | 267.68 | 248.60 |
| vgkd_random_lam1.0 | 218.64 | 275.10 | 241.84 | 229.24 |

**Preferred adaptive:** λ=0 only. **λ>0 rejected.** Do not deploy VGKD.

---

## B9. Phase 2 — Transformer robustness

| Evidence | Large | FT |
|----------|------:|---:|
| Final / Type-macro | 215.85 / 270.61 | 224.12 / **261.15** |
| Type silhouette | +0.038 | −0.014 |
| Rare→common centroid | 21.8 | **7.0** |
| Δ\|err\| rare / heavy (FT−Large) | — | **≈−16.5 / −11.5** |
| Physics attr share | ~0.64 | ~0.38 |
| Attr. agreement Spearman | ≈0.55 | |

---

## B10. Phase 3 — Mechanism validation

### Physics ablation (33 features removed)

| Model | Full Final | No-phys Final | Full type-macro | No-phys type-macro | Δ type |
|-------|-----------:|--------------:|----------------:|-------------------:|-------:|
| Large | 215.85 | 219.49 | 270.61 | 269.67 | **−0.94** |
| FT | 224.12 | 225.17 | 261.15 | 263.70 | +2.55 |

### Type-level correlations (n=15)

| Relation | Spearman | p |
|----------|---------:|--:|
| Physics RMSE → FT advantage | **−0.10** | 0.72 |
| Physics RMSE → Large RMSE | **0.85** | 6e−5 |
| Physics RMSE → FT RMSE | **0.91** | 3e−6 |

### Geometry (normalized)

| Metric | Large | FT |
|--------|------:|---:|
| Rare→common norm | 0.681 | **0.524** |
| Centroid dist → FT advantage Spearman | — | **0.09** |

### C1 variance decomp of FT advantage

Full R² **0.654**; unique drops: physics 0.511 (negative coef), centroid 0.195, heavy 0.180, log_n 0.017.

**H-B physics reliance: REJECTED.** H-A representation: partial.

---

## B11. Phase 3.5 — Smoothness causal intervention

| Model | Val | Final | Type-macro | Rel. emb. move (cont.) |
|-------|----:|------:|-----------:|-----------------------:|
| Large | 229.70 | **215.85** | 270.61 | 0.0170 |
| FT | — | 224.12 | **261.15** | 0.0135 |
| Cons λ=0.01 | 230.01 | 214.46 | 262.39 | 0.0171 |
| **Cons λ=0.1 (selected)** | **229.85** | 216.50 | 268.02 | 0.0162 |
| Cons λ=1.0 | 230.18 | 215.94 | 262.36 | 0.0143 |

Selected Δ type-macro vs Large: **−2.59**. Smoothness **not causal**. Outcome **C**.

---

## B12. Attention routing analysis (negative)

**Invariance:** max \|Δ pred\| = **0**.

| Metric | ρ vs FT advantage | ρ vs FT RMSE |
|--------|------------------:|-------------:|
| mean_cls_entropy | −0.14 | **−0.79** |
| top1_mass | +0.07 | **+0.81** |
| aircraft_cat_mass | **−0.23** | +0.30 |
| physics_mass | +0.10 | **+0.70** |
| js_shift_from_common | +0.09 | +0.30 |

**Decision: REJECTED** (tracks difficulty, not relative advantage).

---

## B13. Per-type scoreboard (Final, n≥50)

| Type | Body | n | Physics RMSE | Large | FT | FT adv |
|------|------|--:|-------------:|------:|---:|-------:|
| A20N | narrow | 17056 | 174.6 | 72.10 | 78.79 | −6.69 |
| A21N | narrow | 792 | 395.6 | 59.74 | 60.74 | −0.99 |
| A319 | narrow | 221 | 194.8 | 53.49 | 75.22 | −21.73 |
| A320 | narrow | 7948 | 186.3 | 70.00 | 80.32 | −10.32 |
| A321 | narrow | 146 | 527.7 | 217.95 | 204.16 | **+13.80** |
| A332 | heavy | 1396 | 1660.6 | 254.36 | 259.25 | −4.89 |
| A333 | heavy | 336 | 1330.5 | 245.40 | 210.82 | **+34.58** |
| A359 | heavy | 5545 | 2249.7 | 335.45 | 352.26 | −16.82 |
| B38M | narrow | 79 | 64.0 | 107.45 | 69.22 | **+38.22** |
| B738 | narrow | 1266 | 291.5 | 90.97 | 92.11 | −1.14 |
| B744 | heavy | 334 | 3609.5 | 680.24 | 684.88 | −4.64 |
| B772 | heavy | 122 | 1255.5 | 363.69 | 221.12 | **+142.57** |
| B77W | heavy | 713 | 4986.7 | 857.23 | 906.87 | −49.64 |
| B788 | heavy | 350 | 640.4 | 303.21 | 288.92 | **+14.29** |
| B789 | heavy | 805 | 3214.9 | 347.87 | 332.51 | **+15.36** |

Type-macro Large / FT = **270.61 / 261.15**.

---

## B14. All hypothesis outcomes

### Pre-R3 / foundation

| Hypothesis / method | Result | Key numbers |
|---------------------|--------|-------------|
| Physics-only OpenAP | **Reject** | RMSE ~1582 |
| Residual learning (Level-1) | **Reject** | MAE ~107 vs hybrid ~86 |
| Residual matched LOTO | **Reject macro** | macro MAE 523 vs Direct 283 |
| Weather-only hybrid | Weak / ≈ OpenAP | XGB MAE 86.6 |
| Static mass on Level-1 Direct | Weak / often worse | vs E+W |
| Fuel-flow target | **Accept** | LGBM RMSE **196.24** |
| Energy hybrid | **Accept** | beats OpenAP hybrid |
| Global/isotonic/class/haul affine (official) | **Reject** | Combined ≥228.25 |
| Phase affine P1E | **Keep (tiny)** | Combined −0.10 |
| Heavy Cat specialist P2 | **Keep** | Combined **227.44** (−0.81) |
| Heavy XGB specialist | **Reject** | Combined 236.6 |
| Cruise residual P3 | **Reject** | Combined **244.9** |
| Ensemble reweight P5 alone | **Reject** | not better than P2 |
| MLP residual E7 | **Reject** | MAE 103.7 |
| Sequence transformer residual | **Reject** | RMSE ~1317 |
| R1 OpenAP descriptors | **Accept** | Combined **226.19** |
| R2 descriptor fixes | **Accept** | Combined **225.25** |
| R3 dynamic mass ensemble | **Accept (teacher freeze)** | Combined **221.33** |

### Post-R3 / distillation mechanism

| ID | Hypothesis | Outcome | Key numbers |
|----|------------|---------|-------------|
| — | Deploy Large under IID | **Accept** | Final 215.85 · Combined 225.95 |
| — | Type-macro ranking reverse | **Accept** | FT 261.15 < Large 270.61 |
| H1 | VGKD adaptive KD | **Reject** | λ>0 worse |
| H2 | Physics reliance (students) | **Reject** | type-macro Δ −0.9; corr(phys,adv) −0.10 |
| H3 | Representation geometry | **Partial** | rare→common 7 vs 22; adv ρ 0.09 |
| H4 | Smoothness causal | **Not supported** | selected type Δ −2.6 |
| H5 | Attention routing | **Reject** | \|ρ\| vs adv ≤0.23; vs RMSE up to 0.81 |

---

## B15. Artifact index

### Pre-R3 tables (`docs/reports/tables/`)

| Topic | File |
|-------|------|
| Physics ablation | `table_physics_ablation.csv` |
| Energy / residual / significance | `table_energy_results.csv`, `table_residual_results.csv`, `table_significance_*.csv` |
| Fuel flow | `table_fuel_flow.csv`, `table_fuel_flow_ablation.csv` |
| Mass ablation | `table_mass_ablation.csv` |
| V3 E5–E7 | `table_v3_*.csv` |
| Ensemble Level-1 | `table_ensemble_final.csv` |
| LOTO | `table_loto_*.csv` |
| Gap-closing | `table_gap_closing_leaderboard.csv`, `table_gap_accepted_changes.csv`, `table_gap_p*.csv` |
| R2 | `table_rmse_R2_full_leaderboard.csv` |
| Official | `final_leaderboard.csv`, `leaderboard_v4.csv` |

### Pre-R3 reports

| Report | Path |
|--------|------|
| Gap-closing campaign | `docs/reports/official_gap_closing_report.md` |
| Official PRC | `docs/reports/official_prc_benchmark_report.md` |
| VED / residual LOTO | `docs/reports/VED_PHENOMENA_REPLICATION.md` |
| LOTO conclusions | `docs/reports/loto_significance_transfer_conclusions.md` |
| Leaderboard audit | `docs/reports/LEADERBOARD_AUDIT.md` |
| RMSE gap attribution | `docs/reports/RMSE_GAP_ATTRIBUTION.md` |

### R3 summaries (JSON)

| File | Combined highlight |
|------|-------------------:|
| `docs/reports/r1_summary.json` | 226.19 |
| `docs/reports/tables/table_rmse_R2_*` | 225.25 |
| `docs/reports/r3_summary.json` | 224.23 (single) |
| `docs/reports/r3_ensemble_summary.json` | **221.33** |

### Distillation (`results/distillation/`)

| Stream | Path |
|--------|------|
| α/β | `alpha_beta_sweep/` |
| Capacity / Large | `capacity_scaling/` |
| Final / Combined | `test_evaluation/`, `combined_evaluation/` |
| FT | `ft_transformer/` |
| Phases 0–3.5, attention | `distribution_shift_diagnosis/`, `uncertainty_analysis/`, `vgkd/`, `transformer_robustness/`, `mechanism_validation/`, `smoothness_causal/`, `attention_routing/` |

### Frozen checkpoints

| Model | Path |
|-------|------|
| Large deploy | `results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt` |
| FT KD-1 | `results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt` |

---

## Master timeline (Combined official where applicable)

```text
Physics-only ..................... ~1582 RMSE (Level-1)  → REJECT
Hybrid trees ..................... ~216–224 RMSE (L1)
Energy / Flow tracks ............. best L1 RMSE ~196.2
Residual learning ................ worse than hybrid     → REJECT
Official ensemble v1 ............. Combined 228.25
+ P1E + P2 Cat ................... 227.44                (v1.1)
+ R1 OpenAP descriptors .......... 226.19
+ R2 descriptor fixes ............ 225.25
+ R3 dynamic mass ensemble ....... 221.33                ← frozen teacher
Large MLP student Final .......... 215.85                ← frozen deploy
FT type-macro .................... 261.15 (best student under type-macro)
VGKD / physics-ablation student /
smoothness / attention ........... negative or non-causal for ranking reverse
```

---

*End of catalog. Narrative status: `PROJECT_STATUS_REPORT.md`. Paper plan: `PAPER_WRITING_GUIDE.md`.*
