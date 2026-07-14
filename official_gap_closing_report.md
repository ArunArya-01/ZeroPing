# Official Gap-Closing Campaign Report

**Date:** July 2026  
**Baseline (canonical official v1):** Combined RMSE **228.25 kg** · Rank **239.18** · Final **220.86**  
**Winner (paper):** Combined RMSE ≈ **201 kg**  
**Protocol:** Train-only fits; Rank/Final evaluation only; no retuning of frozen V4 trees except specialist FuelFlow on heavy types.

---

## Executive result

| Variant | Combined RMSE | Rank | Final | Bias | vs official | Gate |
|---------|--------------:|-----:|------:|-----:|------------:|:----:|
| Official ensemble v1 | **228.25** | 239.18 | 220.86 | +28–31* | — | REFERENCE |
| **P1E + P2 Cat heavy specialist** | **227.44** | **235.30** | 222.18 | **+15.0** | **−0.81** | **KEEP** |
| P1E phase affine only | 228.16 | 239.02 | 220.80 | +28.2 | −0.10 | KEEP (minor) |
| P1 affine / isotonic / haul | ≥228.25 | — | — | — | ≥0 | REJECT |
| P3 cruise residual | 244.9 | — | — | — | worse | REJECT |
| P5 reweight alone | ≥228 | — | — | — | not better than best | REJECT |

\*Bias for official v1 from error analysis (~+31 kg).

**Accepted stack:**  
1. Phase-conditional affine calibration (train OOF)  
2. **CatBoost FuelFlow heavy-aircraft specialist** hard-routed for widebodies  

**Net Combined RMSE:** 228.25 → **227.44** (−0.81 kg).  
**Remaining gap to winner:** **≈ 26.4 kg**.

This is a **real but small** official improvement. It does **not** close the ~27 kg gap to the published winner.

---

## Hypotheses and outcomes

### P1 — Remove systematic over-prediction

| ID | Hypothesis | Combined RMSE | Bias | Decision |
|----|------------|--------------:|-----:|----------|
| P1A global affine | +31 kg is affine | 228.25 | +28.3 | REJECT (no gain) |
| P1B isotonic | nonlinear monotone fix | 228.92 | +28.8 | REJECT |
| P1C class affine | heavy vs narrow bias | 228.34 | +28.2 | REJECT |
| P1D haul affine | ultra-long calibration | 228.58 | +27.6 | REJECT |
| **P1E phase affine** | cruise-dominated SSE | **228.16** | +28.2 | **KEEP (tiny)** |
| P1* on LGBM flow | calibrate best single | ≥229 | +7–13 | REJECT vs floor |

**Interpretation:** Train-OOF calibration **barely transfers** to Rank/Final. The +31 kg bias is largely **shift-dependent**, not a fixed train-calibratable offset. Phase-conditional affine is the only weak keep.

**Expected gain was −5 to −15 kg; realized ≈ −0.1 kg.**

---

### P2 — Heavy-aircraft specialists

| ID | Hypothesis | Combined | Rank | Final | B744 RMSE | Decision |
|----|------------|---------:|-----:|------:|----------:|----------|
| P2 LGBM heavy on P1 | FuelFlow specialist on heavies | 228.16 | 243.0 | — | — | KEEP then superseded |
| **P2 Cat heavy on P1** | same, CatBoost | **227.44** | **235.3** | **222.2** | **883** (was 1135) | **KEEP (best)** |
| P2 XGB heavy | same | 236.6 | — | — | — | REJECT |
| P2b Cat on raw ensemble | no phase cal | ~227.4 | 235.3 | — | — | competitive |
| P2c + global affine | residual bias after specialist | 228.3 | — | — | — | REJECT |

**Subgroup effects (best P2 Cat vs official baseline session):**

| Subgroup | Baseline RMSE | Best RMSE | Change |
|----------|-------------:|----------:|-------:|
| Combined | 228.25 | **227.44** | **−0.81** |
| Rank | 239.18 | **235.30** | **−3.9** |
| Final | 220.86 | 222.18 | +1.3 |
| A20N | 93.0 | **86.3** | better |
| A320 | 71.0 | **69.1** | better |
| A359 | 330.7 | 343.8 | slightly worse |
| B77W | 847.4 | **830.5** | better |
| B744 | 1134.7 | **883.0** | **much better** |
| Ultra-long | 454.8 | **437.5** | better |
| Bias | +31 | **+15** | **halved** |

**Interpretation:** Hard routing of a **heavy FuelFlow CatBoost** improves Rank and B744/B77W, cuts bias, and does **not** hurt narrowbodies. Gains are real but **far smaller than the winner gap** because A359/B77W absolute RMSE remains ~330–830 kg.

**Expected −5 to −12 kg; realized −0.8 kg Combined** (Rank −3.9, Final +1.3).

---

### P3 — Cruise residual

| Combined RMSE | Gate |
|--------------:|------|
| 244.9 | REJECT |

Global residual on cruise after P1/P2 **hurts** Rank/Final (distribution shift; residual overfits train OOF).

---

### P5 — Ensemble reweight

Flow-only ridge / nonnegative weights: Combined **≥228**, no beat of best P2 stack.  
Interesting OOF weight signal: **LGBM FuelFlow ~0.39–0.62** dominates when unconstrained nonnegative mix is fit on train OOF — consistent with FuelFlow primacy, but not enough alone for official gain.

---

## Why the ~26 kg gap remains

1. **Calibration does not transfer** across the temporal shift (train Apr–Aug → Rank Sep → Final Oct).  
2. **Heavy RMSE is still huge** after specialists (B744 ~883, B77W ~831, A359 ~344).  
3. **Ultra-long-haul SSE** still dominates; specialist helps B744 bias but not enough on rare extreme intervals.  
4. Winner likely has stronger **long-haul / mass / ops** features not in frozen V4.

Closing to 201 will need **new representation or labels for heavies**, not more global post-hoc maps.

---

## Accepted recommendation (v1.1)

| Step | Method | Fit data |
|------|--------|----------|
| Base | Official 6-model ensemble (Ridge meta) | train OOF |
| + P1E | Phase-conditional affine | train OOF only |
| + P2 | CatBoost FuelFlow on `HEAVY_TYPES`, hard route by `aircraft_type` | train heavy rows only |

**Official Combined RMSE: 227.44 kg** (was 228.25).  
**Do not** claim winner-level performance.

Reproduce:

```bash
python notebooks/19_gap_closing_campaign.py
# artifacts: figures/table_gap_closing_leaderboard.csv
#            figures/table_gap_p1_calibration.csv
#            figures/table_gap_p2_heavy_experts.csv
#            figures/fig_gap_closing_rmse.png
#            figures/gap_closing_summary.json
```

---

## Scientific honesty checklist

| Rule | Status |
|------|--------|
| Train-only fitting | ✅ |
| No Rank/Final tuning | ✅ |
| Hypothesis-linked experiments | ✅ |
| Reject no-gain / regressions | ✅ |
| Report Final regression of best (+1.3 kg) | ✅ disclosed |
| Gap to winner still ~26 kg | ✅ disclosed |

---

## Next bets (if continuing)

In priority order after this campaign:

1. **Heavy-only feature expansion** — OpenAP continuous descriptors + cruise altitude/duration interactions **inside** the specialist only.  
2. **Quantile / asymmetric loss on heavies** — reduce over-prediction tail on B744.  
3. **Do not** invest more in global isotonic/affine without shift-robust validation design.

---

*Campaign implemented in `physics/gap_closing.py` + `notebooks/19_gap_closing_campaign.py`.*
