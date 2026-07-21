# Current Model Summary

## Model Architecture

- **Type:** Ensemble of 6 GBDT base models (XGB/LGBM/CatBoost x Direct kg + Fuel Flow kg/s)
- **Meta-learner:** Ridge regression (chosen by GroupKFold CV on train OOF over LGBM)
- **Base hyperparameters:** n_estimators=300, lr=0.05 (frozen V4)
- **Specialists:** CatBoost FuelFlow heavy-aircraft specialist (hard-routed for widebodies)
  - P2: Baseline heavy specialist (no extra features)
  - **R1: Heavy specialist with OpenAP descriptors + interactions (proven KEEP at -2.11 kg)**
- **Calibration:** Phase-conditional affine (P1E, train OOF, minor keep)

## Training Data

- **Source:** `aerotwin/aero-data` (Hugging Face)
- **Train split:** Apr-Aug 2025, 10,000 usable flights, 119,032 intervals
- **Rank split:** Sep 2025, 1,888 flights, 24,158 intervals
- **Final split:** Oct 2025, 2,836 flights, 37,170 intervals
- **Feature count (base):** ~47 (BASE_NUMERIC + ENERGY_FEATURES + WEATHER_FEATURES + physics + cats)
- **Feature count (R1 specialist):** 57 (base + 10 OpenAP descriptors + 8 interactions)

## Official Metrics

### v1.1_P1E_R1Cat_descriptors (current best, reference)

- Rank RMSE: **235.21** kg
- Final RMSE: **220.13** kg
- Combined RMSE: **226.19** kg
- Delta vs 227.44: **-1.25** kg
- Delta vs 228.25: **-2.06** kg
- Bias: **+13.7** kg
- Heavy RMSE: **423.1** kg
- Narrow RMSE: **80.7** kg
- A359 RMSE: **342.0** kg
- B77W RMSE: **844.5** kg
- B744 RMSE: **821.0** kg

### v1.1_P1E_R1LGBM_descriptors (close second)

- Combined RMSE: **226.14** kg
- Rank: 241.94, Final: 215.25
- Heavy RMSE: 423.0, Narrow: 80.7

### v1.1_P1E_P2Cat_heavy (prior reference)

- Combined RMSE: **226.74** kg
- Rank: 234.72, Final: 221.40
- Heavy RMSE: 424.3

## Key Findings

1. **Verified current best Combined RMSE: 226.19 kg** (R1 CatBoost heavy specialist with OpenAP descriptors)
2. Previous reference (227.44, v1.1 P2 Cat): beaten by R1 at 226.19 kg (-1.25)
3. Remaining gap to winner (~201 kg): **25.2 kg**
4. Largest error source: **Heavy aircraft (A359/B77W/B744) - ~70% SSE**
5. Dominant phase: **Cruise - 86.5% SSE**
6. Dominant haul: **Ultra-long (>=8h) - highest per-interval RMSE at 433 kg**
