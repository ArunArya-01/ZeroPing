# RMSE Improvement Backlog

**Purpose:** Open work items to improve **official Combined RMSE**. Claim a task, implement under protocol rules, post metrics, then tick the box here and in `PROJECT_STATUS_REPORT.md` §21.

**Last synced:** 2026-07-24 (R1–R3 complete; best Combined **221.33**)

---

## Current scoreboard (do not invent new baselines)

| Label | Combined RMSE | Rank | Final | Notes |
|-------|--------------:|-----:|------:|-------|
| Official frozen V4 ensemble | **228.25** | 239.18 | 220.86 | Canonical reference |
| Gap-close v1.1 | **227.44** | 235.30 | 222.18 | P1E + heavy Cat FuelFlow |
| R1 OpenAP heavy descriptors | **226.19** | 235.21 | 220.13 | ✅ done |
| R2 descriptor fixes | **225.25** | 234.81 | 218.82 | ✅ done |
| **R3 dynamic mass + P1E (best so far)** | **221.33** | **232.53** | **213.73** | ✅ done |
| Published winner | ≈ **201** | — | — | Paper score only |
| **Remaining gap** | **~20 kg** | | | Do not claim superiority |

**Primary error drivers (SSE share):** A359 ~29% · B77W ~22% · B744 ~21% · cruise ~87% · ultra-long-haul ~85%.  
**R3 bias:** ≈ **+3.9 kg** (was ~+24–31 kg on early stacks).

**Reports:** `official_prc_benchmark_report.md` · `official_gap_closing_report.md` · `CURRENT_MODEL_SUMMARY.md` · `PROJECT_STATUS_REPORT.md`  
**Artifacts:** `figures/table_official_leaderboard.csv` · `figures/r{1,2,3}_summary.json` · `figures/table_rmse_R{1,2,3}_*.csv`

---

## Rules (read before coding)

1. **Train only.** Fit models / calibrators / specialists on **train** (or train OOF). Never use Rank/Final labels for tuning or selection.
2. **Gate on Combined RMSE.** Always report Rank RMSE, Final RMSE, Combined RMSE, and bias.
3. **Compare to the right floor.** Prefer Δ vs **221.33** (R3 best). Also report Δ vs **228.25** (official v1).
4. **Keep / reject.** Keep only if Combined improves **and** narrowbodies are not badly hurt. Reject regressions.
5. **One hypothesis per PR / branch.** Name it `rmse/R2b-heavy-loss`, `rmse/R3b-ultralong`, etc.
6. **No superiority claim** vs 201 unless Combined 95% CI is entirely below 201.

### How to score a candidate

```bash
# Baseline (already run)
python notebooks/17_official_prc_evaluation.py --skip-build
python notebooks/18_official_error_analysis.py
python notebooks/19_gap_closing_campaign.py
python notebooks/25_r3_dynamic_mass.py
python notebooks/26_r3_ensemble_mass.py

# After your change: extend gap_closing path or add notebooks/2x_rmse_<id>.py
# Write: figures/table_rmse_<id>.csv  (+ optional fig)
# Post in PR: Combined / Rank / Final + Δ vs 221.33 + subgroup (A359, B77W, B744, ultra-long)
```

### Code touchpoints

| Area | Path |
|------|------|
| Official eval | `physics/official_benchmark.py`, `notebooks/17_official_prc_evaluation.py` |
| Gap-close / specialists | `physics/gap_closing.py`, `notebooks/19_gap_closing_campaign.py` |
| Dynamic mass (R3) | `physics/mass_model.py`, `notebooks/25_r3_dynamic_mass.py`, `notebooks/26_r3_ensemble_mass.py` |
| Features | `physics/feature_engineering.py`, `physics/enrich_*.py` |
| Ensemble bases | `physics/official_benchmark.py` (`ENSEMBLE` / base models) |
| Heavy type set | `physics/gap_closing.py` → `HEAVY_TYPES` |

---

## Already done — do **not** re-run without a new hypothesis

| ID | What | Result | Owner space |
|----|------|--------|-------------|
| G0 | Full Rank+Final official eval | Combined **228.25** | done |
| G1 | SSE error analysis | Heavies + cruise + ultra-long dominate | done |
| G2 | Global / class / haul affine; isotonic | No real Combined gain | **closed** |
| G3 | Phase-conditional affine (P1E) | −0.10 kg | tiny keep |
| G4 | Heavy CatBoost FuelFlow specialist (P2) | Combined **227.44** (−0.81) | keep |
| G5 | Cruise residual; ensemble reweight | Fail / no beat of P2 | **closed** |
| **R1** | Heavy OpenAP descriptors + interactions | Combined **226.19** (−2.06) | **keep** |
| **R2** | OpenAP descriptor fixes (B744/B77L/A306) + R2 features | Combined **225.25** (−3.00) | **keep** |
| **R3** | Dynamic mass model (21 features) + P1E | Combined **221.33** (−6.92) | **keep** |

> **ID note:** Early backlog named R2 “asymmetric loss” and R3 “ultra-long specialist.” Implementation used **R2 = descriptor completeness** and **R3 = dynamic mass**. The original ideas remain open as **R2b** / **R3b**.

---

## Open tasks — claim one

Put your name in **Owner** when you start. Mark **Status** `in progress` → `done` / `rejected`.

### Priority A (highest impact — pick these first)

#### R2b — Asymmetric / robust loss on heavies
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Train heavy FuelFlow specialist with quantile / Huber / asymmetric loss to cut remaining over-prediction. |
| **Why** | Bias improved under R3 but heavies still dominate SSE. |
| **Expected** | −2 to −8 kg Combined |
| **Start from** | CatBoost/LGBM objective + sample_weight on heavy rows; stack on R3 mass features if possible |
| **Deliverables** | `figures/table_rmse_R2b_heavy_loss.csv` + bias metrics for B744/B77W/A359 |
| **Gate** | Combined < 221.33; mean bias on B744 ↓; narrowbody RMSE flat |
| **Parallel?** | Coordinate with R3b if both edit specialist routing |

#### R3b — Ultra-long-haul FuelFlow path (not global haul affine)
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Specialist or hard route for **ultra-long (≥8h)** flights (FuelFlow). Global haul affine already **failed** — this is a **model**, not a post-hoc map. |
| **Why** | Ultra-long ≈ 85% SSE. |
| **Expected** | −2 to −10 kg Combined |
| **Start from** | Mirror `train_heavy_specialist` with haul mask; compose with heavy route carefully |
| **Deliverables** | `figures/table_rmse_R3b_ultralong.csv` + ultra-long subgroup RMSE |
| **Gate** | Combined < 221.33; ultra-long RMSE ↓; short/medium not worse by >1–2 kg Combined equivalent |
| **Parallel?** | Can run parallel to R2b if routing composition is designed first |

#### R4 — Further mass / load proxies
| | |
|--|--|
| **Status** | 🔄 largely superseded by R3 |
| **Owner** | _unclaimed_ |
| **Goal** | Only pursue **incremental** mass/load features beyond R3 (e.g. better payload proxies without Rank/Final leakage). |
| **Why** | R3 already delivered the main mass gain (−6.92 kg). |
| **Expected** | Unknown; likely small |
| **Start from** | `physics/mass_model.py`; document leakage risk |
| **Deliverables** | Train-OOF ablation first; only then one official Rank/Final score |
| **Gate** | Combined < 221.33; if OOF fails, mark **rejected** and stop |

#### R5 — Long-interval specialist or weights
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Reduce error on `duration` buckets **10–30 min** and **≥30 min**. Options: specialist FuelFlow, sample weights, or duration-conditioned residual **only on train OOF-selected design**. |
| **Why** | Long intervals dominate squared error. |
| **Expected** | −2 to −8 kg Combined |
| **Start from** | Error analysis buckets; routing like P2 |
| **Deliverables** | `figures/table_rmse_R5_long_interval.csv` |
| **Gate** | Combined < 221.33; long-iv RMSE ↓; short-iv not collapsed |
| **Parallel?** | Yes, if routing priority vs heavy/ultra-long is agreed |

---

### Priority B (medium — after A or if A blocked)

| ID | Task | Status | Owner | Expected | Notes |
|----|------|:------:|-------|----------|-------|
| **R6** | Fuel-Flow-first ensemble (drop weak Direct bases if OOF says so) | ⬜ | _unclaimed_ | −1 to −5 kg | Re-check **on top of** R3 |
| **R7** | Nested Optuna on FuelFlow only (train OOF; freeze before official) | ⬜ | _unclaimed_ | −1 to −5 kg | Unlikely to close 20 kg alone |
| **R8** | Train-safe temporal / seasonal features (month, ISA trend) | ⬜ | _unclaimed_ | −1 to −6 kg | Addresses Apr–Aug → Sep → Oct shift |
| **R9** | Promote **221.33** as documented “current floor” in README + reports | 🔄 | partial | 0 kg | README + status report updated 2026-07-24 |

---

### Closed / do not pick

| ID | Task | Why |
|----|------|-----|
| R1 | Heavy OpenAP descriptors | **Done** (226.19) |
| R2 | Descriptor completeness fixes | **Done** (225.25) |
| R3 | Dynamic mass (21 features) | **Done** (221.33) |
| R10 | More global isotonic / affine | Failed transfer Rank/Final |
| R11 | Global cruise residual after stack | Hurt Combined (244+) |
| R13 | Body-class routing as RMSE fix | Rejected as universal LOTO solution; not the official gap |

Optional low priority: **R12** transformer residual — only with a written sequence hypothesis.

---


## PR template (copy into PR body)

```markdown
### RMSE task
- ID: R#
- Hypothesis:
- Baseline compared: 221.33 / 228.25

### Metrics
| Split | RMSE | MAE | Bias |
|-------|-----:|----:|-----:|
| Rank | | | |
| Final | | | |
| Combined | | | |

### Subgroups
- A359 / B77W / B744 / ultra-long RMSE:
- Narrowbody RMSE:

### Decision
- KEEP / REJECT vs 221.33
- Artifacts: figures/table_rmse_R#.csv
```

---

*Parent checklist: `PROJECT_STATUS_REPORT.md` §21. Full scientific status: same report. Official write-up: `official_prc_benchmark_report.md`.*
