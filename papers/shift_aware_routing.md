# Shift-Aware Routing (Conditional Scaffold)

**Status:** Implemented as a *scaffold*. Gated — not deployable until operational-shift
evidence exists (PROJECT_STATUS_REPORT §20 Priority 7).

## Purpose

Under leave-one-type-out (LOTO) evaluation, the Fuel-Flow formulation does **not**
win uniformly: Flow+Energy is favourable on some folds (B77W, A332, A321, A20N,
B788) and Direct is better on others (B789, A333, B738, A359). The open research
question is *under what operational shift does flow normalization help or harm*.

Priority 7 of the plan therefore asks, only **after** the operational distribution-shift
analysis (Priority 1–2) explains that heterogeneity, for a **simple** router — not a
complicated one. This module implements those simple policies. It deliberately does
**not** silently produce a deployable router; the learned policy refuses to route until
calibrated from validation evidence.

## Implementation (`physics/shift_aware_routing.py`)

### Operational shift score

`operational_shift_score(train, test, method)` measures the train→test operational
distribution distance using the §20 Priority 1 candidate variables (duration, altitude,
speed, vertical rate, phase fractions, energy rates, trajectory density, start/end
flight fraction). Three metrics:

| Method | Dependency | Notes |
|---|---|---|
| `smd` (default) | numpy only | Standardized mean difference, std units |
| `wasserstein` | scipy (lazy) | Normalized by training std for cross-feature comparability |
| `js` | scipy (lazy) | Jensen–Shannon divergence, [0, 1] |

Returns `(mean_score, per_feature)`; missing features are skipped.

### Router policies (`ShiftAwareRouter`)

| Policy | Deployable? | Behaviour |
|---|---|---|
| `always_direct` | Yes (baseline) | Constant Direct |
| `always_flow` | Yes (baseline) | Constant Flow+Energy |
| `oracle` | **No — upper bound** | Picks lower-MAE formulation per unit using *ground-truth* MAE |
| `learned` | Only after `calibrate()` | Threshold over operational distance, fit on validation folds |

- `fit_reference(train_df)` stores the training operational distributions.
- `calibrate(fold_table)` fits the `learned` threshold by minimizing misrouting error
  over the provided folds (train/validation only). Sets `calibrated=True` and a
  `direction` (`flow_when_high` / `direct_when_high`).
- `route(test_df, ...)` returns `{unit: "direct" | "flow"}`. The `learned` policy
  **raises** if uncalibrated (unless `allow_uncalibrated=True`, which falls back to
  `always_direct` and must not be trusted). `oracle` requires ground-truth MAE and is
  explicitly an upper bound.

### Fold table assembly

`build_fold_table(loto_results, shift_scores)` merges per-type LOTO MAE with
precomputed operational shift scores into the validation fold table consumed by
`calibrate()`. One row per held-out type with `operational_distance` and
`delta_mae_flow_minus_direct`.

## How to use (once evidence exists)

1. Run the operational distribution-shift analysis (Priority 1) and compute
   `shift_scores` per held-out type via `operational_shift_score`.
2. Assemble the validation fold table with `build_fold_table`.
3. `router = ShiftAwareRouter(policy="learned").fit_reference(train).calibrate(fold_table)`.
4. Evaluate the router on held-out folds with clear labeling of `oracle` as upper bound,
   comparing against `always_direct` / `always_flow` baselines.

## Caveats

- `n = 12` LOTO types: the learned threshold is high-variance and B77W-sensitive. Treat
  any routing gain as suggestive, not confirmatory.
- Routing must use only deployment-time-available signals (operational features), never
  fuel labels.
- This scaffold does **not** replace the statistical protocol freeze; inference still uses
  the frozen bootstrap/CI rules from `physics/statistical_protocol.py`.
