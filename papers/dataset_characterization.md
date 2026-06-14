# AeroTwin Dataset Characterization Paper (Paper 1) - Outline & Results

**Working title**: "Characterizing a Hybrid ACARS/ADS-B Dataset for Physics-Informed Aircraft Fuel Burn Prediction"

**Authors** (to be filled): ...

**Target**: Journal of Open Aviation Science (or Scientific Data / Data in Brief)

## Abstract (draft)
We present an in-depth characterization of the EUROCONTROL PRC 2025 aircraft fuel burn estimation dataset (aerotwin/aero-data on HF). The data fuses dense ADS-B kinematics with sparse ACARS fuel telemetry and airspeed reports across 15k+ commercial flights. We quantify label linkage, trajectory coverage (median 32% of flight time labeled), source heterogeneity (ACARS air data often partial or absent even on report rows), and the performance of an OpenAP physics baseline. Results highlight challenges for hybrid modeling: highly variable data density per prediction interval (2 to 3000+ points), unknown aircraft mass, and partial observability. These findings directly inform feature engineering and residual learning for the AeroTwin hybrid predictor.

## 1. Introduction
- Motivation: fuel burn prediction for emissions, efficiency, digital twins.
- The PRC 2025 challenge + published paper (Sun et al., 2026).
- Why characterization matters before modeling (physics+NN thesis).
- Contributions: usable sample definition (10k train), coverage/quality metrics, physics error baselines, public loader + EDA.

## 2. Dataset & Access
- HF remote `hf://datasets/aerotwin/aero-data` (no full download; polars + fsspec).
- File structure (observed, not just readme): flightlist_*, fuel_*, flights_* (nested layout), airports.
- Splits + counts (Table 1).
- Loader: `data.AeroDataLoader` (usable filter critical).

**Table 1**: Summary counts per split + usable.

## 3. Flight Metadata & Aircraft
**Table 2**: Aircraft type distribution (A320 family dominant ~60%).
- Date range, origins (273), durations (median 4.1h).
- Fig: ac_types bar.

## 4. Fuel Label Analysis (the prediction targets)
- 1:1 flight_id coverage in fuel.
- Intervals/flight: median 10, 5-95% 5-25.
- fuel_kg: median 200kg/interval, heavy tail.
- **Key finding**: labeled time covers median 32% (mean 38%) of takeoff-landed duration.
- Interval durs 5-60min (filtered).
- Figs: intervals per flight hist, total fuel per flight, fuel_kg distrib.

## 5. Trajectory Data & Source Heterogeneity (core difficulty)
- 13 cols observed (incl. track, source).
- ADSB dense but mach/TAS/CAS always null.
- ACARS sparse + incomplete (samples: mach only / CAS only / none; some flights 0 ACARS rows in traj despite fuel labels).
- **Table**: ACARS completeness, null rates by source.
- Per-interval pts: huge variance (many 2-pt "boundary only").
- Fig: profile examples (2-3 flights) with shaded fuel intervals + ACARS markers (shows gaps in coverage during labeled periods).
- Fig (future full): CDF of pts per fuel interval (highlight thresholds 5/50/500).

## 6. Physics Baseline (OpenAP) Validation
- Method: TAS inference (mach > CAS > GS), ref mass (mtow*0.75), FuelFlow.enroute per interval rep pt, integrate ff * dt.
- Results on demo: dense intervals MAE ~480kg (over ~20-50%); sparse 2-pt MAE ~3k+ (over 2-3x).
- **Table 6**: physics errors overall + by n_pts bin + has_acars.
- Fig: scatter physics vs actual (color by pts density).
- Implication: residuals structured by data availability + phase; NN must learn corrections for missing TAS, mass bias, etc.
- Why before FE: quantifies what "residual" is; guides physics-aware features (acars_avail flag, density, alt/vr proxies).

## 7. Limitations & Recommendations for Downstream Use
- 1037 train flights lack traj (use only usable=10000).
- No mass / TOW / config / weather / engine variant.
- Partial label coverage (not gate-to-gate).
- License CC-BY 4.0; cite challenge paper.
- Suggested: weight loss by 1/sqrt(n_pts) or mask very-sparse; use typecode + route as mass proxy in NN; separate models or embeddings for acars-sparse regimes.
- rank/final for temporal/OOD eval.

## 8. Conclusion & Next (Paper 2)
This characterization provides the empirical foundation for AeroTwin hybrid modeling. Residuals are learnable given the observed kinematics + source flags. Future: full sampling across 10k, feature eng informed by §6, neural residual + physics-consistency loss, benchmarks vs pure ML / tuned OpenAP.

## References
- Sun et al. (2026) ... JOAS.
- OpenAP (Sun et al.).
- Loader + notebooks in https://... (this repo).

## Appendix: Generated Artifacts
- All tables/figs in `figures/`.
- Repro: run `notebooks/0{1,2,3,4}_*.py` (PYTHONPATH=.) with fixed seeds in loader.
- Schemas: see loader.get_schema() output.

*(This md is living; update with full numbers once 03/04 full runs complete. Current numbers from audit probes + stubs.)*
