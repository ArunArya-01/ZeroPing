# RMSE Gap Attribution

**Reference model:** R3 ensemble (mass)  
**Current Combined RMSE:** 221.33 kg  
**Previous best (R2):** 225.25 kg  
**Prior reference (v1.1):** 227.44 kg  
**Official baseline (v1.0):** 228.25 kg  
**Winner (paper):** ~201 kg  
**Total remaining gap:** ~20 kg  
**Delta vs 228.25:** −6.92 kg  

## Error Composition

From audit of Combined Rank+Final predictions (R3 ensemble 221.33):

- Heavy aircraft SSE share: **~70%** (A359 ~29%, B77W ~22%, B744 ~21%)
- Cruise phase SSE share: **~87%**
- Ultra-long-haul (>=8h) RMSE: **~431 kg** vs ~86 kg for medium-haul
- Narrowbody RMSE: **~75 kg** — near-optimal
- Combined bias: **+3.7 kg** (down from +24 kg in v1.0)

The R3 dynamic mass model (21 physics-informed mass features) is the single largest improvement, cutting bias from +24 kg to +3.9 kg on the single-model LGBM variant and improving both heavy (−12 kg) and narrowbody (−6 kg) RMSE.

## Detailed Error by Aircraft (R3 ensemble)

| Aircraft | RMSE (kg) | Bias (kg) | Notes |
|----------|-----------|-----------|-------|
| A359 | ~333 | — | Largest absolute contribution |
| B77W | ~838 | — | High variance |
| B744 | ~866 | — | Over-prediction reduced vs R1 (+247 kg bias) |
| A332 | — | — | Medium error |
| A20N | ~75 | — | Narrowbody baseline |
| B738 | ~70 | — | Best-performing type |

## Flight Phase Breakdown (R3 LGBM mass)

| Phase | RMSE (kg) | SSE Share | Bias |
|-------|-----------|-----------|------|
| Cruise | ~245 | ~87% | ~+1 |
| Climb | ~191 | ~9% | — |
| Descent | ~117 | ~4% | — |

Cruise dominates error. The R3 mass model nearly eliminated cruise bias on the LGBM single model (0.98 kg vs 7.3 kg baseline).

## RMSE Opportunity

The ~20 kg remaining gap to the winner (~201 kg) is attributed to mass estimation refinement, ultra-long-haul specialists, heavy-type loss functions, interpolation/resampling, and potential team-specific optimizations not accessible to AeroTwin.

## Rejected Improvements

| Improvement | Reason | Evidence |
|-------------|--------|----------|
| Fuel-flow filtering (<0.05 or >6.5 kg/s) | Degrades RMSE | R2 audit: +1.5 to +3.3 kg |
| Global affine/isotonic calibration | No transfer to Rank/Final | P1 audit: >=228.25 |
| Cruise residual (global) | Worsens RMSE | P3: 244.9 kg |
| Ensemble reweight (Flow-only) | No improvement | P5: >=228 kg |
| Residual learning (trees/MLP) | Worse than direct hybrid | Internal: MAE ~107 vs ~84 |
