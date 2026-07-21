# RMSE Gap Attribution

**Reference model:** v1.1_P1E_R1Cat_descriptors  
**Current Combined RMSE:** 226.19 kg  
**Prior reference (v1.1):** 227.44 kg  
**Winner (paper):** ~201 kg  
**Total remaining gap:** 25.2 kg  
**Delta vs prior best (227.44):** -1.25 kg  

## Error Composition

From audit of Combined Rank+Final predictions:

- Heavy aircraft SSE share: **69.5%** (A359 34.0%, B77W 23.4%, B744 12.1%, other heavies 0.0%)
- Cruise phase SSE share: **86.5%**
- Ultra-long-haul (>=8h) RMSE: **433 kg** vs 94-126 kg for shorter hauls
- Narrowbody RMSE: **80.7 kg** — near-optimal

## Detailed Error by Aircraft

| Aircraft | RMSE (kg) | Bias (kg) | SSE Share | Notes |
|----------|-----------|-----------|-----------|-------|
| A359 | 342 | -10.4 | 34.0% | Under-predicted slightly, large absolute errors |
| B77W | 844 | -57.2 | 23.4% | High variance, under-predicted |
| B744 | 821 | +247.4 | 12.1% | **Severely over-predicted** — systematic bias |
| A332 | 429 | -9.6 | 12.9% | Medium error, nearly unbiased |
| A20N | 83 | +24.6 | 6.4% | Narrowbody baseline |
| B738 | 56 | +22.2 | 0.7% | Best-performing type |

## Flight Phase Breakdown

| Phase | RMSE (kg) | SSE Share | Bias |
|-------|-----------|-----------|------|
| Cruise | 245 | 86.5% | +10.7 |
| Climb | 199 | 8.9% | +41.2 |
| Descent | 126 | 4.6% | +7.5 |

Cruise dominates error. Climb has larger bias (over-prediction) but less SSE contribution due to shorter duration.

## Estimated RMSE Opportunity by Category

| Category | Est. RMSE Reduction (kg) | Confidence | Rationale |
|----------|-------------------------|------------|-----------|
| Mass estimation (TOW/model) | 3-8 | high | MTOW*0.75 is crude. Realistic mass modeling reduces cruise error. |
| Haul-aware routing (ultra-long >=8h) | 2-5 | medium | >=8h dominates. Haul-aware specialist targets dominant error regime. |
| Asymmetric loss (Huber/quantile for heavies) | 1-3 | low | B744 over-predicted by +247 kg. MSE penalizes this symmetrically. |
| Feature: MTOW/OEW/Thrust in base ensemble | 1-3 | medium | R1 proven for heavy specialist. Extend to base. |
| Cruise residual model (heavy+ultra) | 1-4 | medium | P3 rejected globally. Restricted to heavy types may work. |
| Weather data (GRIB/METAR) | 0-2 | low | Weather-only ablation not significant (E5). Actual data unlikely to change. |
| Model architecture (deeper trees, neural nets) | 0-3 | low | GBDT near-optimal for tabular data. Gains come from features, not architecture. |

## Estimated Realistic Path

| Stage | Action | Est. RMSE |
|-------|--------|-----------|
| Now | v1.1_P1E_R1Cat_descriptors | **226.2** |
| Stage 1 | Improved mass estimation (TOW proxy, mass decay) | ~222 kg |
| Stage 2 | Haul-aware specialist + asymmetric loss | ~216 kg |
| Stage 3 | MTOW/OEW in base ensemble + cruise residual | ~213 kg |
| Stage 4 | Interpolation/resampling + comprehensive features | ~209 kg |
| Ceiling (est.) | Irreducible noise (ACARS labels, coverage gaps) | ~207 kg |
| Winner | | **~201 kg** |

**Caveat:** This path is speculative. Each stage must independently pass the accept gate.
Realized gains may be smaller due to distribution shift between train and Rank/Final.
The ~6 kg gap from the ceiling to the winner could be attributed to team-specific techniques
(unknown architectures, external data, or challenge-specific optimizations) not accessible to AeroTwin.

## Highest-Confidence Next Experiment

**Mass estimation improvement.** MTOW*0.75 is the documented largest limitation. Options:
1. Add load-factor proxy: `(ref_mass - OEW) / OEW` — simple, no leakage risk
2. Phase-aware mass: higher for climb, lower for cruise (use fraction-of-flight for linear decay)
3. OEW as minimum mass (reduces over-prediction on short/narrowbody)
4. Wing loading features already added in R1 — extend to base ensemble

Expected: **-2 to -5 kg Combined RMSE.**

## Rejected Improvements

| Improvement | Reason | Evidence |
|-------------|--------|----------|
| Fuel-flow filtering (<0.05 or >6.5 kg/s) | Degrades RMSE | R2 audit: +1.5 to +3.3 kg |
| Global affine/isotonic calibration | No transfer to Rank/Final | P1 audit: >=228.25 |
| Cruise residual (global) | Worsens RMSE | P3: 244.9 kg |
| Ensemble reweight (Flow-only) | No improvement | P5: >=228 kg |
| Residual learning (trees/MLP) | Worse than direct hybrid | Internal: MAE ~107 vs ~84 |
