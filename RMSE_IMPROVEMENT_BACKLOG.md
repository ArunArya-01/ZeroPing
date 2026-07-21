# RMSE Improvement Backlog 

**Purpose:** Open work items to improve **official Combined RMSE**. Claim a task, implement under protocol rules, post metrics, then tick the box here and in `PROJECT_STATUS_REPORT.md` §21.


---

## Current scoreboard (do not invent new baselines)

| Label | Combined RMSE | Rank | Final | Notes |
|-------|--------------:|-----:|------:|-------|
| Official frozen V4 ensemble | **228.25** | 239.18 | 220.86 | Canonical reference |
| Best so far (gap-close v1.1) | **227.44** | 235.30 | 222.18 | P1E + heavy Cat FuelFlow |
| Published winner | ≈ **201** | — | — | Paper score only |
| **Remaining gap** | **~26 kg** | | | Do not claim superiority |

**Primary error drivers (SSE share):** A359 ~29% · B77W ~22% · B744 ~21% · cruise ~87% · ultra-long-haul ~85%.

**Reports:** `official_prc_benchmark_report.md` · `official_gap_closing_report.md`  
**Artifacts:** `figures/table_official_leaderboard.csv` · `figures/table_gap_closing_leaderboard.csv` · `figures/official_error_analysis_summary.json`

---

## Rules (read before coding)

1. **Train only.** Fit models / calibrators / specialists on **train** (or train OOF). Never use Rank/Final labels for tuning or selection.
2. **Gate on Combined RMSE.** Always report Rank RMSE, Final RMSE, Combined RMSE, and bias.
3. **Compare to the right floor.** Prefer Δ vs **227.44** (v1.1). Also report Δ vs **228.25** (official v1).
4. **Keep / reject.** Keep only if Combined improves **and** narrowbodies are not badly hurt. Reject regressions.
5. **One hypothesis per PR / branch.** Name it `rmse/R1-heavy-features`, `rmse/R2-heavy-loss`, etc.
6. **No superiority claim** vs 201 unless Combined 95% CI is entirely below 201.

### How to score a candidate

```bash
# Baseline (already run)
python notebooks/17_official_prc_evaluation.py --skip-build
python notebooks/18_official_error_analysis.py
python notebooks/19_gap_closing_campaign.py

# After your change: extend gap_closing path or add notebooks/20_rmse_<id>.py
# Write: figures/table_rmse_<id>.csv  (+ optional fig)
# Post in PR: Combined / Rank / Final + Δ vs 227.44 + subgroup (A359, B77W, B744, ultra-long)
```

### Code touchpoints

| Area | Path |
|------|------|
| Official eval | `physics/official_benchmark.py`, `notebooks/17_official_prc_evaluation.py` |
| Gap-close / specialists | `physics/gap_closing.py`, `notebooks/19_gap_closing_campaign.py` |
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

---

## Open tasks — claim one

Put your name in **Owner** when you start. Mark **Status** `in progress` → `done` / `rejected`.

### Priority A (highest impact — pick these first)

#### R1 — Heavy-only feature expansion
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Add features **only inside the heavy specialist** (A359/B77W/B744/…): OpenAP continuous descriptors (MTOW, OEW, wing area, thrust, …) + interactions (e.g. cruise altitude × duration, mean alt × duration_s). Do **not** change the global ensemble feature set first. |
| **Why** | Heavies ≈ 72% SSE; gap report’s #1 next bet. |
| **Expected** | −3 to −12 kg Combined |
| **Start from** | `train_heavy_specialist` / `predict_heavy_routed` in `physics/gap_closing.py` |
| **Deliverables** | Code + `figures/table_rmse_R1_heavy_features.csv` + short note in PR |
| **Gate** | Combined < 227.44; Rank not worse by >2 kg without Final gain; B744/B77W RMSE down |
| **Parallel?** | Conflicts with R2/R3 if both edit specialist training — coordinate or stack on same branch |

#### R2 — Asymmetric / robust loss on heavies
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Train heavy FuelFlow specialist with quantile / Huber / asymmetric loss to cut **over-prediction** (B744 bias was ~+311 kg). |
| **Why** | Systematic over-predict; MSE treats tails symmetrically. |
| **Expected** | −2 to −8 kg Combined |
| **Start from** | CatBoost/LGBM objective + sample_weight on heavy rows; keep hard routing from P2 |
| **Deliverables** | `figures/table_rmse_R2_heavy_loss.csv` + bias metrics for B744/B77W/A359 |
| **Gate** | Combined ↓; mean bias on B744 ↓; narrowbody RMSE flat |
| **Parallel?** | Best after or with R1 (same specialist). Don’t open two independent specialist PRs without sync. |

#### R3 — Ultra-long-haul FuelFlow path (not global haul affine)
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Specialist or hard route for **ultra-long (≥8h)** flights (FuelFlow). Global haul affine already **failed** — this is a **model**, not a post-hoc map. |
| **Why** | Ultra-long ≈ 85% SSE. |
| **Expected** | −2 to −10 kg Combined |
| **Start from** | Mirror `train_heavy_specialist` pattern with haul mask; compose with heavy route carefully (define priority if both match) |
| **Deliverables** | `figures/table_rmse_R3_ultralong.csv` + ultra-long subgroup RMSE |
| **Gate** | Combined ↓; ultra-long RMSE ↓; short/medium not worse by >1–2 kg Combined equivalent |
| **Parallel?** | Can run parallel to R1 if routing composition is designed first (write a short routing matrix in PR). |

#### R4 — Deploy-safe mass / load proxies
| | |
|--|--|
| **Status** | ⬜ open (exploratory) |
| **Owner** | _unclaimed_ |
| **Goal** | Find mass/load features that are available at train time and at Rank/Final **without label leakage**. Heuristic mass was rejected under Level 1 — need something better or prove still dead. |
| **Why** | Winner may use stronger mass/ops; OpenAP uses MTOW×0.75. |
| **Expected** | Unknown; may be 0 |
| **Start from** | `featured_dataset_mass.parquet` / V4 mass ablation notebooks; document leakage risk |
| **Deliverables** | Train-OOF ablation first; only then one official Rank/Final score |
| **Gate** | Train-OOF win **and** Combined ↓; if OOF fails, mark **rejected** and stop |
| **Parallel?** | Yes — independent of specialist routing if features are additive |

#### R5 — Long-interval specialist or weights
| | |
|--|--|
| **Status** | ⬜ open |
| **Owner** | _unclaimed_ |
| **Goal** | Reduce error on `duration` buckets **10–30 min** and **≥30 min** (together ~79% SSE of intervals by SSE share). Options: specialist FuelFlow, sample weights, or duration-conditioned residual **only on train OOF-selected design**. |
| **Why** | Long intervals dominate squared error. |
| **Expected** | −2 to −8 kg Combined |
| **Start from** | Error analysis buckets in `official_error_analysis_summary.json`; routing like P2 |
| **Deliverables** | `figures/table_rmse_R5_long_interval.csv` |
| **Gate** | Combined ↓; long-iv RMSE ↓; short-iv not collapsed |
| **Parallel?** | Yes, if routing priority vs heavy/ultra-long is agreed |

---

### Priority B (medium — after A or if A blocked)

| ID | Task | Status | Owner | Expected | Notes |
|----|------|:------:|-------|----------|-------|
| **R6** | Fuel-Flow-first ensemble (drop weak Direct bases if OOF says so) | ⬜ | _unclaimed_ | −1 to −5 kg | P5 Flow-only was close but lost to P2 — re-check **on top of** v1.1 |
| **R7** | Nested Optuna on FuelFlow only (train OOF; freeze before official) | ⬜ | _unclaimed_ | −1 to −5 kg | Unlikely to close 26 kg alone; still useful |
| **R8** | Train-safe temporal / seasonal features (month, ISA trend) | ⬜ | _unclaimed_ | −1 to −6 kg | Addresses Apr–Aug → Sep → Oct shift |
| **R9** | Promote 227.44 as documented “official floor v1.1” in README + reports | ⬜ | _unclaimed_ | 0 kg | Docs only; do anytime |

---

### Closed / do not pick

| ID | Task | Why |
|----|------|-----|
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
- Baseline compared: 227.44 / 228.25

### Metrics
| Split | RMSE | MAE | Bias |
|-------|-----:|----:|-----:|
| Rank | | | |
| Final | | | |
| Combined | | | |
| Δ Combined vs 227.44 | | | |

### Subgroups
- A359 / B77W / B744 RMSE:
- Ultra-long RMSE:
- Narrowbody (A20N/A320) RMSE:

### Protocol
- [ ] Train-only fits
- [ ] No Rank/Final tuning
- [ ] Artifacts under figures/table_rmse_R#.csv

### Decision
- [ ] KEEP  /  [ ] REJECT
- Reason:
```

---

## Success bars (team goals)

| Goal | Combined RMSE | What we can say |
|------|--------------:|-----------------|
| Stretch | ≤ **201** | Matches published winner **score** |
| Strong | ≤ **210** | Competitive |
| Meaningful | ≤ **220** | Real improvement (~−8+ kg) |
| Now | **227.4–228.3** | No superiority |

---

*Parent checklist: `PROJECT_STATUS_REPORT.md` §21. Full scientific status: same report. Official write-up: `official_prc_benchmark_report.md`.*
