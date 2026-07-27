# LOTO Significance & Transfer-Distance Analysis

**Script:** `notebooks/17_loto_significance_and_transfer_distance.py`  
**Date:** July 2026  
**Prerequisite:** Existing LOTO results in `table_loto_comprehensive.csv` (no new model families).

## 1. Paired significance: Global Flow+Energy vs Global Direct E+W

### Methodology
- Re-ran the **same two CatBoost LOTO configurations** to recover interval-level absolute errors.
- **Per-type inference:** flight-clustered bootstrap (10,000 resamples of flights within each held-out type).
- **Pooled inference:** hierarchical bootstrap resampling types then flights (respects clustering).
- **Macro inference:** bootstrap over 12 type-level paired ΔMAE values.
- **Robustness:** leave-one-type-out macro ΔMAE; sensitivity excluding B77W.

### Results
| Metric | Value |
|---|---|
| Macro ΔMAE (flow − direct) | -17.39 kg |
| Median ΔMAE | -16.28 kg |
| Flow wins / losses / ties | 7 / 5 / 0 |
| Hierarchical flight-bootstrap 95% CI | [-40.3, +18.6] kg |
| P(flow better) hierarchical | 0.727 |
| Type-level bootstrap 95% CI | [-54.9, +16.9] kg |
| Paired t-test p | 0.3814 |
| Paired Wilcoxon p (flow < direct) | 0.2349 |
| Macro ΔMAE excluding B77W | -3.95 kg |

**Interpretation:** Flow+Energy improves macro LOTO MAE by 17.4 kg (7 wins, 5 losses). However, **pooled significance is not established at α=0.05**: hierarchical flight-clustered 95% CI is [−40.3, +18.6] kg and type-level bootstrap CI is [−54.9, +16.9] kg (both include zero). Paired t-test (p=0.38) and Wilcoxon (p=0.23) on 12 type-level MAEs are non-significant. Per-type flight-clustered tests are highly significant for most individual folds (e.g. B77W p≈0), but pooling across types does not yield a significant global claim.

**B77W sensitivity:** Excluding B77W reduces macro ΔMAE from −17.4 kg to **−4.0 kg** — the macro flow advantage is largely driven by one influential wide-body fold (−165 kg on B77W alone). Leave-one-type-out macro ΔMAE remains negative for all exclusions (range −3.9 to −26.2 kg), so the direction is robust, but magnitude is not.

## 2. Aircraft transfer-distance study

### Descriptor table (`table_aircraft_openap_descriptors.csv`)
Built from **OpenAP `prop.aircraft()` and `prop.engine()` only** — no invented values.  
Features used (complete for all 12 types): mtow_kg, mlw_kg, oew_kg, mfc_kg, cruise_mach, cruise_range_km, wing_area_m2, wing_span_m, max_thrust_n, mmo.

### Distance definitions (held-out → 11 training types)
1. **Nearest-neighbor:** min Euclidean distance on standardized descriptors.
2. **k-NN mean (k=3):** mean of 3 smallest NN distances.
3. **Mahalanobis:** distance to training-type centroid using pseudo-inverse covariance (n_train=11 > n_features=10).

### Hypothesis
> Cross-aircraft fuel prediction error increases with physical distance from the training aircraft support.

### Correlation summary (n=12 types)

| Distance | Outcome | Pearson r | p | Spearman ρ | p | Bootstrap 95% CI (Pearson) |
|---|---|---|---|---|---|---|
| NN | LOTO direct MAE | **+0.759** | **0.004** | +0.329 | 0.297 | [−0.35, 0.96] |
| k3-mean | LOTO direct MAE | +0.681 | 0.015 | +0.483 | 0.112 | [−0.11, 0.89] |
| k3-mean | LOTO direct RMSE | +0.789 | 0.002 | +0.706 | 0.010 | [+0.41, 0.94] |
| NN | MAE degradation (LOTO − std) | +0.585 | 0.046 | −0.280 | 0.379 | [−0.78, 0.93] |
| NN | MAE inflation (LOTO / std) | −0.268 | 0.400 | −0.566 | 0.055 | [−0.74, 0.29] |
| Mahalanobis | LOTO direct MAE | −0.155 | 0.631 | −0.063 | 0.846 | — |

**Influence diagnostics:** Dropping **B77W** collapses Pearson r from 0.759 to **0.147** (p=0.666). B77W is a high-leverage point: high transfer distance (NN=1.92) and highest LOTO MAE (1,055 kg).

**B77W-excluded sensitivity (n=11):**
- NN distance vs LOTO MAE: r=0.147, p=0.666 → **not significant**
- k3-mean vs LOTO MAE: r=0.308, p=0.358 → **not significant**
- k3-mean vs MAE inflation: r=−0.773, p=0.005 → significant **negative** correlation (more distant types inflate *less* relative to standard split when B77W removed)

**Verdict:** **PARTIALLY SUPPORTED, B77W-DOMINATED.** With all 12 types, NN/k3 distances correlate positively with absolute LOTO error (Pearson p&lt;0.05 for MAE/RMSE). With B77W excluded, the absolute-error hypothesis **fails** for both Pearson and Spearman. Mahalanobis distance shows no association. Error *inflation* (LOTO/std-split) is not robustly linked to transfer distance. The hypothesis should not be cited as a general law — it holds mainly because B77W is physically distant from training support and catastrophically hard, not because distance monotonically predicts error across the fleet.

## 3. Artifacts

| File | Description |
|---|---|
| `table_loto_paired_per_type.csv` | Per-type paired deltas + flight-bootstrap CIs |
| `table_loto_paired_significance_summary.csv` | Pooled significance summary |
| `table_loto_leave_one_type_robustness.csv` | LOO macro robustness |
| `table_loto_paired_sensitivity.csv` | B77W sensitivity |
| `table_aircraft_openap_descriptors.csv` | OpenAP physical descriptor table |
| `table_loto_transfer_distances.csv` | NN, k-NN, Mahalanobis distances per fold |
| `table_loto_transfer_distance_analysis.csv` | Merged distances + errors + inflation |
| `table_loto_transfer_correlations.csv` | Pearson/Spearman + bootstrap CIs |
| `table_loto_transfer_influence.csv` | Leave-one-type correlation influence |
| `fig_loto_paired_*.png`, `fig_loto_distance_*.png` | Diagnostic plots |
