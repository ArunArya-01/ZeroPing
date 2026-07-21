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
| TOW estimator | [MISS] | Missing | MTOW*0.75 is crude cruise mass. No takeoff mass estimation. |
| Recursive mass (fuel-burn decay) | [MISS] | Missing | No mass decay tracking through flight. Each interval uses same ref mass. |
| Heuristic mass (MTOW*0.75) | [OK] | Implemented | _ref_mass(): standard PRC approach. |
| MTOW/OEW/Thrust features | [PART] | Partial | R1 adds for heavy specialist only. Not in base ensemble. |
| Wind interpolation (GRIB/METAR) | [MISS] | Missing | ISA-based proxies from kinematics, not actual weather data. |
| Flight phase detection | [OK] | Implemented | Median VR thresholds (+/-1.5 m/s) in classify_interval_phase(). |
| Split isolation (Train/Rank/Final) | [OK] | Implemented | Strict temporal separation. No cross-contamination. |
| Min interval threshold (60s) | [DIFF] | Different | Labels as _short but does not exclude. |
| Unit conversion (ft->m, kt->m/s) | [OK] | Implemented | Performed in OpenAP/numpy pipeline. |

## Summary

- [OK] Implemented: 9
- [MISS] Missing: 7
- [PART] Partial: 1
- [DIFF] Different: 1

### Key gaps to address

1. **TOW / mass estimation**: MTOW*0.75 is the single largest limitation. Better mass modeling could yield the largest RMSE reduction.
2. **Recursive mass decay**: Not modeling fuel-burn-dependent mass change through flight.
3. **Interpolation/resampling**: Not performed. Could help normalize data density across intervals.
4. **Actual weather data**: ISA-based proxies only. GRIB/METAR integration could improve wind/temperature estimates.
5. **MTOW/OEW features in base ensemble**: Only in R1 heavy specialist, missing from main feature set.
