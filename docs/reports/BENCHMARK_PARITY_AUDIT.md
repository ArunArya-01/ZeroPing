# Benchmark Parity Audit

Comparison of AeroTwin implementation vs Sun et al. (JOAS 2026) preprocessing methodology.

| Step | Status | Classification | Notes |
|------|--------|----------------|-------|
| FuelFlow filtering | [MISS] | Missing | R2 audit: all filters degrade RMSE. Correctly omitted. |
| Duplicate removal (traj) | [MISS] | Missing | PRC dataset is pre-QA'd by EUROCONTROL. Not currently needed. |
| Coordinate validation | [MISS] | Missing | Trajectory coords from HF dataset are validated by provider. |
| Trajectory interpolation | [MISS] | Missing | Point-level TAS uses representative points. Full interpolation would help windows. |
| Trajectory resampling | [MISS] | Missing | Raw points used as-is. Uniform resampling could normalize density. |
| TAS reconstruction (Mach->TAS) | [OK] | Implemented | _infer_tas() priority chain matches paper methodology. |
| CAS reconstruction | [OK] | Implemented | CAS->TAS in _infer_tas() when CAS available from ACARS. |
| Mach reconstruction | [N/A] | Not applicable | Mach from ACARS reports only. No reconstruction performed. |
| Statistical embeddings | [OK] | Implemented | Altitude, GS, VR mean/std/min/max per interval window. |
| TOW estimator | [OK] | Implemented (R3) | **R3 dynamic mass model** adds 21 physics-informed mass features: takeoff weight, landing mass, per-interval mass via linear fuel-burn interpolation, mass rate, fuel fraction, phase-aware mass, wing loading. |
| Recursive mass (fuel-burn decay) | [OK] | Implemented (R3) | Per-interval mass via linear fuel-burn interpolation by flight fraction. Mass consumed and remaining fuel tracked through flight. |
| Heuristic mass (MTOW*0.75) | [REPLACED] | Superseded by R3 | _ref_mass() still available as fallback. R3 dynamic mass is the active model. |
| MTOW/OEW/Thrust features | [PART] | Partial | R1 adds for heavy specialist only. R3 adds mass derivatives. Not in base ensemble. |
| Wind interpolation (GRIB/METAR) | [MISS] | Missing | ISA-based proxies from kinematics, not actual weather data. |
| Flight phase detection | [OK] | Implemented | Median VR thresholds (+/-1.5 m/s) in classify_interval_phase(). |
| Split isolation (Train/Rank/Final) | [OK] | Implemented | Strict temporal separation. No cross-contamination. |
| Min interval threshold (60s) | [DIFF] | Different | Labels as _short but does not exclude. |
| Unit conversion (ft->m, kt->m/s) | [OK] | Implemented | Performed in OpenAP/numpy pipeline. |

## Summary

- [OK] Implemented: 11
- [MISS] Missing: 5
- [PART] Partial: 1
- [DIFF] Different: 1

### Key gaps remaining (post-R3)

1. **Interpolation/resampling**: Not performed. Could help normalize data density across intervals.
2. **Actual weather data**: ISA-based proxies only. GRIB/METAR integration could improve wind/temperature estimates.
3. **MTOW/OEW features in base ensemble**: Only in R1 heavy specialist, missing from main feature set.

### Gaps closed by R3 dynamic mass model

- **TOW estimator**: Replaced crude MTOW*0.75 with 21 physics-informed mass features yielding **−6.92 kg Combined RMSE reduction** (221.33 vs 228.25).
- **Recursive mass decay**: Per-interval mass tracking through flight phases.
- **Phase-aware mass**: Climb/cruise/descent differ in fuel state.
- **Wing loading**: Current wing loading per interval added.
