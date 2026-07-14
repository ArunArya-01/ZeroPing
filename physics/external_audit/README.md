# AeroTwin External Dataset Audit

Pilot-scale infrastructure for validating AeroTwin findings on **NASA DASHlink** and **OpenSky Network** data, as specified in `AeroTwin_External_Dataset_Audit_Package.md`.

## Module layout

| File | Role |
|------|------|
| `audit_utils.py` | Phase detection, energy rates, sparsity, interval helpers (dataset-agnostic) |
| `dashlink_loader.py` | Load Project 85 MATLAB `.mat` flights; reconstruct fuel from flow/FOB |
| `opensky_loader.py` | Query OpenSky Trino / traffic; **physics-derived** fuel labels |
| `build_featured_audit.py` | Common path → `featured_dataset_audit.parquet` |
| `run_audit_pilot.py` | Experiments A–E (Direct, Fuel-Flow, energy ablation, flight split) |

## Design principles

- Reuse `physics/openap_baseline.py` and `physics/feature_engineering.py` so features match PRC2025.
- Prefer **small samples first** (`--max-flights 5–10`) before full downloads/queries.
- Log all fuel-label assumptions (especially OpenSky physics labels and DASHlink flow integration).
- Offline path: `--source demo` builds synthetic flights for CI and pipeline smoke tests.

## Quick start (always works offline)

From the **project root** (`ZeroPing/`):

```bash
# 1) Smoke test — synthetic demo (no external data)
python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8

# 2) Build featured parquet only
python -m physics.external_audit.build_featured_audit --source demo --max-flights 6 \
  --out audit_results/featured_dataset_audit.parquet

# 3) Re-run pilot on an existing parquet
python -m physics.external_audit.run_audit_pilot \
  --parquet audit_results/featured_dataset_audit.parquet \
  --out-dir audit_results
```

Outputs land in `audit_results/` (tables, figures, meta JSON).

---

## DASHlink (Project 85) — pilot path

**Why first:** best chance of independent (or reconstructed) fuel labels.

### Prerequisites

- `scipy` (for `.mat` I/O) — already in project `requirements.txt`
- Downloaded Sample Flight Data (Project 85) MATLAB files
- Optional: parameter dictionary / notes from DASHlink documentation

### Step-by-step

1. **Download a few sample flights only** (2–5 `.mat` files), not the full multi-year archive.
2. Place them under a local folder, e.g. `data/dashlink/project85/`.
3. **Probe parameters** (Phase 1 audit):

   ```bash
   python -m physics.external_audit.dashlink_loader data/dashlink/project85/some_flight.mat --probe
   ```

4. **Load one flight** (trajectory + reconstructed fuel intervals):

   ```bash
   python -m physics.external_audit.dashlink_loader data/dashlink/project85/some_flight.mat
   ```

5. **Build featured dataset (pilot scale)**:

   ```bash
   python -m physics.external_audit.build_featured_audit --source dashlink \
     --dashlink-dir data/dashlink/project85 \
     --max-flights 5 \
     --out audit_results/featured_dataset_audit.parquet
   ```

6. **Run the pilot experiment suite**:

   ```bash
   python -m physics.external_audit.run_audit_pilot --source dashlink \
     --dashlink-dir data/dashlink/project85 \
     --max-flights 5 \
     --out-dir audit_results/dashlink_pilot
   ```

### Fuel label notes (DASHlink)

| Mode | How target is built | Limitation |
|------|---------------------|------------|
| Fuel flow channels found | Integrate kg/s over fixed intervals | Noisier than PRC ACARS FOB deltas |
| Fuel quantity / FOB found | Interval burn ≈ start − end (or used delta) | Unit heuristics (lb vs kg) |
| No fuel signal | Physics-only features; pilot may fall back to OpenAP as “actual” | **Not** independent validation |

Watch the logs for unit assumptions (ft → m, kt → m/s, lb/h → kg/s).

### Go / No-Go (from audit package)

- **Go:** usable fuel reconstruction on sample flights, core traj features present.
- **No-Go:** no fuel-related parameters → deprioritize DASHlink vs OpenSky robustness tests.

---

## OpenSky (Trino historical) — pilot path

**Value:** different telemetry (ADS-B) and operational mix.  
**Caveat:** **no native fuel labels** — targets are OpenAP physics-derived.

### Prerequisites

- Academic Trino access + `pyopensky` credentials, **or** `traffic` configured for history
- Project deps: `openap`, `traffic` (see root `requirements.txt`)
- Without credentials, the loader falls back to **synthetic** OpenSky-like flights so the code path still runs

### Step-by-step

1. **Confirm access** (interactive):

   ```python
   from pyopensky.trino import Trino
   trino = Trino()
   df = trino.history(start="2024-01-01", stop="2024-01-01 02:00", icao24="...")  # small window
   ```

2. **Pilot query → featured dataset** (short window, few flights):

   ```bash
   python -m physics.external_audit.build_featured_audit --source opensky \
     --start 2024-01-01 --stop 2024-01-01 06:00 \
     --max-flights 10 \
     --out audit_results/featured_dataset_audit_opensky.parquet
   ```

3. **Full pilot suite**:

   ```bash
   python -m physics.external_audit.run_audit_pilot --source opensky \
     --start 2024-01-01 --stop 2024-01-01 06:00 \
     --max-flights 10 \
     --out-dir audit_results/opensky_pilot
   ```

4. Optional filter:

   ```bash
   python -m physics.external_audit.run_audit_pilot --source opensky \
     --icao24 abc123 --start 2024-01-01 --stop 2024-01-02 --max-flights 5
   ```

### Interpreting OpenSky results

Columns `label_source=physics_openap` and `label_is_physics_derived=True` mark physics labels.

- Do **not** treat absolute MAE as independent fuel error.
- Prefer **relative** comparisons: energy ablation, Flow vs Direct ranking, sparsity behaviour.
- Residual vs OpenAP is ~0 by construction when actual is set equal to physics.

---

## Pilot experiments (what `run_audit_pilot` runs)

| ID | Experiment | Success metric |
|----|------------|----------------|
| A | Direct baseline (base + physics) | MAE / RMSE / R² vs physics-only |
| B | Fuel-Flow target (+ matched Direct) | Flow MAE vs Direct MAE |
| C | Energy feature ablation | ΔMAE + bootstrap CI |
| D | Flight-level split generalization | Train/test flight counts + metrics |
| E | Qualitative comparison table | Replicates / partial / fails vs PRC2025 |

Outputs under `--out-dir` (default `audit_results/`):

- `table_audit_pilot_metrics.csv`
- `table_audit_pilot_significance.csv`
- `table_audit_pilot_generalization.csv`
- `table_audit_qualitative_comparison.csv`
- `table_audit_pilot_per_type.csv` (if multiple types in test)
- `figures/fig_audit_*.png`
- `audit_pilot_meta.json`
- `featured_dataset_audit.parquet`

---

## Recommended execution order

1. `--source demo` — prove the pipeline on your machine.
2. **DASHlink** 2–5 flights — fuel feasibility gate.
3. If fuel OK → scale DASHlink pilot (`--max-flights 20–50`).
4. **OpenSky** short window — distribution-shift / physics-label robustness.
5. Compare qualitative tables; decide primary external dataset.

See also project root: **`HOW_TO_RUN_AUDIT.md`** for a condensed checklist.

## Dependencies

Same as the main project (`requirements.txt`): `polars`, `numpy`, `scipy`, `openap`, `scikit-learn`, `lightgbm` (or xgb/catboost), `matplotlib`.  
Optional: `pyopensky` for Trino history (install separately if not pulled in via `traffic`).
