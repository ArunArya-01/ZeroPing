# AeroTwin External Dataset Audit Package
**Date:** July 2026  
**Purpose:** Structured compatibility audit for second dataset validation (DASHlink + OpenSky)   
**Status:** Ready to execute

---

## 1. Executive Summary & Recommended Execution Order

### Goal
Determine whether NASA DASHlink or OpenSky Network Historical Data can serve as a scientifically meaningful second dataset for testing the robustness of AeroTwin’s key findings:
- Energy-state features improve prediction
- Fuel-Flow target vs Direct target behavior under distribution shift
- Value of physics-informed hybrid correction
- Cross-aircraft generalization behavior

### Recommended Execution Order (Minimal Time Investment)

| Step | Dataset | Action | Estimated Effort | Decision Gate |
|------|---------|--------|------------------|---------------|
| 1 | **DASHlink** (Project 85 + fuel-related projects) | Documentation + sample data audit | 1–2 days | Go / No-Go on fuel target feasibility |
| 2 | **DASHlink** | Minimal pilot (if Step 1 passes) | 3–5 days | Full processing decision |
| 3 | **OpenSky Trino** | Documentation + small query pilot | 1–2 days | Go / No-Go on value without native fuel labels |
| 4 | **OpenSky** | Subset extraction + featured dataset pilot | 4–7 days | Compare value vs DASHlink |
| 5 | Decision | Choose primary external dataset or run both in parallel | — | — |

**Primary recommendation:** Start with **DASHlink** because it offers the best chance of independent fuel labels. Move to OpenSky only if DASHlink fuel reconstruction or aircraft diversity proves insufficient.

---

## 2. Master Compatibility Audit Framework (Applies to Both Datasets)

Use this framework for any candidate dataset.

### Core Evaluation Dimensions

| Dimension | Key Questions | Success Criteria for AeroTwin |
|-----------|---------------|-------------------------------|
| **Fuel Target Construction** | Can we create interval-level fuel burn targets comparable to PRC ACARS FOB differences? | Yes or "reasonable approximation with documented limitations" |
| **Aircraft Type Diversity** | Number of distinct types with ≥80–100 flights each | Ideally ≥6–8 types for meaningful LOTO-style analysis |
| **Trajectory Feature Feasibility** | Can we compute the same core features (altitude stats, groundspeed, vertical rate, phase fractions, energy rates, sparsity signals)? | High fidelity for ≥80% of key features |
| **Physics Baseline** | Can we run OpenAP (or equivalent) to generate `physics_fuel_kg`? | Yes with reasonable effort |
| **Distribution Shift** | How different is operational profile, telemetry characteristics, and aircraft mix from PRC2025? | Meaningful shift (not identical distribution) |
| **Preprocessing Effort** | Estimated person-days to reach "featured_dataset.parquet" equivalent | < 10 days for pilot scale |
| **Statistical Power** | Enough flights/intervals for bootstrap inference and per-type analysis | Yes for at least flight-level splits + limited type analysis |

### Go / No-Go Decision Criteria

**Strong Go:**
- Fuel targets can be constructed with acceptable noise
- ≥5 aircraft types with reasonable volume
- Core trajectory + energy features reproducible

**Conditional Go:**
- Fuel reconstruction is noisy but documented
- Limited type diversity, but still useful for feature robustness testing

**No-Go:**
- No usable fuel signal at all
- Extremely narrow aircraft coverage (1–2 types only)
- Preprocessing effort > 3 weeks for pilot

---

## 3. DASHlink-Specific Audit Plan

### Priority Projects to Audit First

1. **Sample Flight Data (Project 85)** — Highest priority
   - Regional jet fleet (tails ~652–687)
   - Matlab format, 186 parameters per flight
   - 3+ years of data

2. **Analysis of Virtual Sensors for Predicting Aircraft Fuel Consumption** (DASHLINK_620 and related)

3. Any other fuel-related projects on DASHlink

### DASHlink Audit Checklist

#### Phase 1: Documentation & Access (Day 1)

- [ ] Locate and read all documentation for Project 85
- [ ] Confirm download/access method (Matlab .mat files)
- [ ] Identify parameter dictionary or naming convention
- [ ] Check licensing / usage restrictions
- [ ] Download 2–3 sample flights for inspection

#### Phase 2: Fuel Target Feasibility (Critical)

- [ ] Does any parameter contain `fuel_flow`, `fuel_used`, `fuel_quantity`, or equivalent?
- [ ] What is the sampling rate of fuel-related parameters?
- [ ] Can fuel flow be integrated over time windows to create interval burn targets?
- [ ] Are there clean flight phase or event markers to define intervals?
- [ ] How much missing data exists in fuel parameters?
- [ ] Can gross weight / reference mass be derived or used?

**Decision Gate:** If no usable fuel signal → consider DASHlink lower priority and move to OpenSky.

#### Phase 3: Aircraft & Metadata

- [ ] How many distinct aircraft (tail numbers) are present?
- [ ] Is aircraft type/family metadata available and consistent?
- [ ] What is the flight count distribution per tail/type?
- [ ] Can we group into "types" for LOTO-style analysis?

#### Phase 4: Trajectory & Feature Feasibility

- [ ] Confirm availability and quality of:
  - Barometric / pressure altitude
  - Calibrated airspeed or Mach
  - Vertical speed / rate
  - Groundspeed
  - Gross weight
  - Time / timestamp alignment
- [ ] Can we compute phase fractions (climb/cruise/descent) using vertical rate thresholds?
- [ ] Can we derive energy-related features (kinetic + potential)?
- [ ] What is typical sampling rate and data completeness?

#### Phase 5: Pilot Scale Estimation

- [ ] Estimated number of usable flights after cleaning
- [ ] Estimated preprocessing effort (person-days) to reach featured dataset
- [ ] Risk assessment (missing parameters, unit issues, alignment problems)

### Expected Challenges on DASHlink

- Fuel data will likely require integration of fuel flow → noisier labels than PRC ACARS.
- Limited aircraft diversity (regional jet focus) → weak LOTO statistical power.
- Matlab format requires conversion (scipy.io or similar).
- Parameter names may need reverse-engineering from documentation or sample inspection.

---

## 4. OpenSky Network Historical Data (Trino) – Specific Audit Plan

### Access Notes

- Academic/research access to Trino historical database is available (free for qualifying institutions).
- Tools: `pyopensky`, `traffic` library, or `ostk` GUI.
- Data: State vectors (lat, lon, altitude, groundspeed, vertical rate, etc.), flight metadata, aircraft metadata (including type where available).

### OpenSky Audit Checklist

#### Phase 1: Access & Tooling (Day 1)

- [ ] Confirm academic access to Trino interface
- [ ] Set up `pyopensky` or `traffic` library locally
- [ ] Run a small test query (e.g., one day of data for a specific callsign or icao24)
- [ ] Understand table schema (state_vectors, flights, aircraft, etc.)

#### Phase 2: Fuel Label Strategy (Critical)

- [ ] Confirm there are **no native fuel labels** (expected)
- [ ] Decide on physics label strategy:
  - Use OpenAP `FuelFlow.enroute` on reconstructed trajectories (same as current pipeline)
  - Or use BADA if preferred
- [ ] Document that this becomes a **physics-label robustness test** rather than independent fuel validation
- [ ] Assess value: Can we still test energy features + hybrid correction under different telemetry source?

#### Phase 3: Aircraft Type & Diversity

- [ ] Can we reliably get aircraft type (from `aircraft` table or metadata)?
- [ ] How complete is type information across flights?
- [ ] What is realistic type diversity in a manageable query window (e.g., one month Europe or global)?
- [ ] Can we select a subset with ≥6–8 types having reasonable flight counts?

#### Phase 4: Trajectory Feature Feasibility

- [ ] Confirm availability of core fields:
  - `baro_altitude` or `geo_altitude`
  - `groundspeed`
  - `vertical_rate`
  - `lat`, `lon`, `time`
- [ ] How to handle missing values and irregular sampling (OpenSky is event-driven)?
- [ ] Can we compute the same phase fractions, energy rates, and sparsity signals?
- [ ] How does trajectory density / sparsity distribution compare to PRC2025?

#### Phase 5: Operational Distribution Shift Assessment

- [ ] Select a pilot query window (recommend: 7–30 days in a region with good coverage)
- [ ] Compare high-level statistics vs PRC2025:
  - Flight length distribution
  - Altitude band distribution
  - Phase mix
  - Aircraft type mix
- [ ] Document the nature of the shift (geographic, operational, telemetry)

#### Phase 6: Pilot Scale & Effort

- [ ] Time to extract a usable featured dataset subset (e.g., 10k–50k intervals)
- [ ] Preprocessing differences vs current `AeroDataLoader` + `build_featured_dataset.py`
- [ ] Risk: Query performance, rate limits, data volume

### Expected Challenges on OpenSky

- No independent fuel ground truth → labels are physics-derived.
- Event-driven nature of ADS-B → different sparsity patterns than fused PRC data.
- Aircraft type metadata completeness varies.
- Querying large time ranges efficiently requires good SQL / tool usage.

**Value Proposition:** Excellent for testing whether energy features and the hybrid modeling approach are robust to a completely different data source and operational regime.

---

## 5. Minimal Pilot Experiment Design (Once Data is Accessible)

After passing basic feasibility, run this compact suite on the new dataset:

| Experiment | Description | Success Metric |
|------------|-------------|----------------|
| **A. Direct Baseline** | Predict `actual_fuel_kg` directly with trajectory + metadata + physics features | MAE, RMSE, R² vs physics-only |
| **B. Fuel-Flow Target** | Predict fuel flow rate; recover kg via duration | Compare MAE vs Direct |
| **C. Energy Feature Ablation** | Base features vs Base + Energy features | ΔMAE with bootstrap CI |
| **D. Generalization Test** | Flight-level split + limited type-level analysis | Compare to PRC Level 1 / Level 2 gap |
| **E. Qualitative Comparison Table** | Replicate the cross-dataset finding summary table from the project report | Document which findings replicate, partially replicate, or fail |

Use the existing `external_vs_flow_eval.py` and `cross_dataset_replication.py` infrastructure where possible.

---

## 6. Code Structure Recommendations

### Suggested New Files / Modules

```
physics/
├── external_audit/
│   ├── dashlink_loader.py          # Data loading + interval construction for DASHlink
│   ├── opensky_loader.py           # Query + basic cleaning for OpenSky Trino
│   ├── build_featured_audit.py     # Common featured dataset builder
│   ├── audit_utils.py              # Shared functions (phase detection, energy calc, etc.)
│   └── run_audit_pilot.py          # Orchestration script for minimal pilot
```

### Key Design Principles

- Reuse as much logic as possible from `physics/openap_baseline.py` and `physics/build_featured_dataset.py`.
- Make fuel target construction configurable (Direct vs integrated flow).
- Log all assumptions and limitations clearly (especially for OpenSky physics labels).
- Output a standardized `audit_results/` folder with tables and figures.

---

## 7. Final Decision Framework

After completing audits on both datasets, answer:

1. Which dataset allows the cleanest fuel target construction?
2. Which provides the most meaningful distribution shift from PRC2025?
3. Which has acceptable aircraft diversity for at least limited cross-type analysis?
4. What is the total effort to reach a publishable external validation result?
5. Should we run **both** datasets (different strengths) or pick one primary?

**Recommended default path (July 2026):**
- Audit DASHlink first (highest chance of independent labels).
- If fuel reconstruction is feasible → prioritize DASHlink.
- If DASHlink diversity or fuel quality is weak → run OpenSky pilot in parallel and compare value.

---

## Appendix: Quick Start Commands

**DASHlink (after downloading samples):**
```bash
python -c "from scipy.io import loadmat; data = loadmat('sample_flight.mat'); print(list(data.keys())[:20])"
```

**OpenSky Trino (example query structure):**
```python
from pyopensky.trino import Trino
trino = Trino()
df = trino.history(start="2025-01-01", stop="2025-01-02", icao24="abc123")
```

---

**Document Version:** 1.0  
**Next Action:** Execute Phase 1 audit on DASHlink Project 85.

*This package is designed to be executable with low overhead while maintaining scientific rigor.*