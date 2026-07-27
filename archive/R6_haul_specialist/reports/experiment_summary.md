# R6 -- Ultra-Long-Haul FuelFlow Specialist

## Problem Statement

Ultra-long-haul flights (>= 8h) contribute ~59% of train SSE while representing only
~33% of intervals. Their per-interval RMSE (335 kg) is 2.3x higher than 2-4h flights
(143 kg). The baseline ensemble has no haul-aware routing.

## Hypothesis

A dedicated FuelFlow model trained exclusively on >= 8h flights will reduce error in
this dominant regime without affecting shorter flights.

## Implementation

- One new function in `gap_closing.py`: `train_haul_specialist()` (identical pattern to heavy specialist)
- One new routing function: `predict_haul_routed()`
- No new features, no architecture changes
- Train on train data only

## Results

| Metric | Baseline (session) | R6 CatBoost | Delta |
|--------|-------------------|-------------|-------|
| Combined RMSE | 228.31 | 229.52 | +1.21 |
| Rank RMSE | 241.58 | 242.51 | +0.93 |
| Final RMSE | 219.15 | 220.67 | +1.52 |
| Heavy RMSE | 427.5 | 430.1 | +2.6 |
| Narrow RMSE | 83.8 | 84.1 | +0.3 |
| Bias | +24.06 | +12.49 | -11.57 |
| A359 RMSE | 330.7 | 348.0 | +17.3 |
| B77W RMSE | 835.3 | 849.3 | +14.0 |
| B744 RMSE | 900.2 | 875.4 | -24.8 |

## Decision

**NO-GO** (+1.21 kg delta vs baseline).

The haul specialist slightly degrades Combined RMSE. While bias improves (-11.57 kg)
and B744 improves (-24.8 kg), the overall RMSE trend is negative. The LGBM variant is
drastically worse (+22.8 kg).

## Why It Failed

The >= 8h threshold creates a hard boundary. Flights near the boundary (7-8h) receive
no specialist treatment despite sharing similar characteristics. The specialist has only
39,831 of 119,032 training samples -- a much smaller pool than the full ensemble.

Unlike the heavy specialist (which targets aircraft types with genuinely different physics),
the haul split is continuous -- the model already learns duration-related patterns from
the `duration_s` and `cruise_fraction` features. A hard threshold adds a discontinuity
that the continuous features already cover smoothly.

## Recommended Next Experiment

Investigate graduated routing (soft blending by haul bucket) rather than hard thresholds.
Or target the B744 specifically (single worst aircraft at 900+ kg RMSE) rather than all
ultra-long flights.
