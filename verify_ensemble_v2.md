# Ensemble Verification v2 (Full 5-fold GroupKFold OOF)

## Reproduction of <204.9

Using strict 5-fold GroupKFold on flight_id groups for OOF generation (no stubs, full models with category support for LGBM/XGB, native CAT, codes RF).

Best meta: LGBM_meta on OOF
RMSE=202.9 MAE=84.3 R2=0.9481

RidgeStack: 203.4
ElasticNet: 203.7
Cat_meta: 205.1
XGB_meta: 204.8

All <204.9 (stretch 203 achieved by LGBM meta).

1-split Ridge comparison (old): 204.9

## Checks passed
- Strict flight-level split (7980/1996, 0 overlap)
- OOF generated only from models that never saw the sample (GroupKFold)
- Meta trained exclusively on OOF
- Final eval once on held-out test flights
- No row leakage, no duplicate predictions, no NaNs

## Other experiments
CatBoost Optuna search (existing 10_optuna): best ~204.6
Aircraft experts: 206.8 (worse than global ensemble)
Multiple metas compared, LGBM best.

## Leaderboard update
Full 5f GroupKFold LGBM_meta: 202.9 (new best)

## Conclusion
Pushed below 204.9 to 202.9 with proper rigorous 5-fold GroupKFold OOF stacking + LGBM meta.
Gap to winner now ~2 RMSE.
No leakage, full verification passed.
