# How to Run the AeroTwin External Dataset Audit

Step-by-step guide for the pilot-scale external validation package in `physics/external_audit/`.

**Full design / decision gates:** `AeroTwin_External_Dataset_Audit_Package.md`  
**Module details:** `physics/external_audit/README.md`  
**Project status (results):** `PROJECT_STATUS_REPORT.md` § External Dataset Validation  

**Status (July 2026):** DASHlink Project 85 pilot complete (15 flights, tails 686/687) — energy features and Fuel-Flow target **replicate** under flight-level holdout. Artifacts in `audit_results/dashlink_pilot/`.

Always start with **small samples**. Scale only after Phase 1 checks pass.

---

## 0. Environment (once)

```bash
cd "path/to/ZeroPing"
pip install -r requirements.txt
# Optional for live OpenSky Trino:
# pip install pyopensky
```

Working directory for all commands below: **project root** (`ZeroPing/`).

---

## 1. Offline smoke test (do this first)

No DASHlink download or OpenSky credentials required.

```bash
python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8 --out-dir audit_results
```

**Expect:**

- `audit_results/featured_dataset_audit.parquet`
- CSV tables (`table_audit_pilot_*.csv`)
- Figures under `audit_results/figures/`

If this fails, fix the local environment before touching real data.

**Unit tests:**

```bash
python -m pytest tests/test_external_audit.py -q
```

---

## 2. DASHlink (Project 85) — recommended first real dataset

### 2.1 Download pilot samples only

1. Obtain NASA DASHlink **Sample Flight Data (Project 85)** access.
2. Download sample flights (e.g. tails **686** and **687**); local layout used in this repo:

   ```text
   data/
     Tail_686_1/*.mat
     Tail_687_1/*.mat
   ```

### 2.2 Phase 1 — parameter probe (fuel feasibility)

```bash
python -m physics.external_audit.dashlink_loader data/Tail_687_1/687200103200323.mat --probe
```

**Expect:** ~186 time-series rows with `size > 1` (e.g. `ALT` thousands of samples, `units=FEET`, `rate_hz=4`).  
If you only see `meta_*` with `size=1`, the loader is outdated — update `dashlink_loader.py` (struct `.data` extraction).

**Project 85 channel names (confirmed):**

| Role | Names | Units (typical) |
|------|--------|-----------------|
| Altitude | `ALT`, `BAL1`/`BAL2` | FEET |
| Groundspeed | `GS` | KNOTS |
| Vertical rate | `IVV`, `ALTR` | FT/MIN |
| Fuel flow | `FF_1`…`FF_4` | LBS/HR |
| Fuel quantity | `FQTY_1`…`FQTY_4` | LBS |
| Position | `LATP`, `LONP` | DEG |
| Air data | `CAS`, `MACH`, `TAS` | KNOTS / MACH |

**Decision gate**

| Result | Action |
|--------|--------|
| Fuel-related channels present | Continue to pilot ✅ (confirmed on Project 85) |
| No fuel signal at all | Document No-Go for independent labels; optionally still use for trajectory-only checks or move to OpenSky |
| Trajectory incomplete | Inspect units / alternate channel names; open an issue note in your audit log |

### 2.3 Phase 2 — load one flight end-to-end

```bash
python -m physics.external_audit.dashlink_loader data/Tail_686_1/686200104111724.mat
```

Confirm logs mention `integrated_fuel_flow` (or `quantity_delta`) and unit conversions (FEET→m, KNOTS→m/s, LBS/HR→kg/s).

### 2.4 Phase 3 — featured dataset (pilot scale)

```bash
python -m physics.external_audit.build_featured_audit --source dashlink \
  --dashlink-dir data \
  --max-flights 5 \
  --out audit_results/dashlink_pilot/featured_dataset_audit.parquet
```

### 2.5 Phase 4 — pilot experiment suite

```bash
python -m physics.external_audit.run_audit_pilot --source dashlink \
  --dashlink-dir data \
  --max-flights 15 \
  --out-dir audit_results/dashlink_pilot \
  --model lgbm \
  --n-bootstrap 500
```

**Recorded pilot (15 flights):** Flow MAE ≈ 18.1 kg; Direct+Energy ≈ 20.7; energy ΔMAE ≈ −4.9 (sig.); Flow vs Direct ΔMAE ≈ −2.6 (sig.). See `PROJECT_STATUS_REPORT.md`.

### 2.6 What to read after the run

| File | Use |
|------|-----|
| `table_audit_pilot_metrics.csv` | MAE/RMSE/R² for Direct, Flow, physics-only, energy ablation |
| `table_audit_pilot_significance.csv` | Bootstrap ΔMAE + CI |
| `table_audit_qualitative_comparison.csv` | replicates / partial / fails vs PRC2025 findings |
| `audit_pilot_meta.json` | flight counts, label sources, feature list |
| console logs | unit conversions and fuel reconstruction assumptions |

### 2.7 Scale only if pilot is Go

```bash
# Example next step: more flights, same pipeline
python -m physics.external_audit.run_audit_pilot --source dashlink \
  --dashlink-dir data/dashlink/project85 \
  --max-flights 50 \
  --out-dir audit_results/dashlink_scale \
  --n-bootstrap 2000
```

**DASHlink caveats**

- Integrated fuel flow ≠ PRC ACARS FOB deltas (noisier labels).
- Fleet is regional-jet heavy → limited type diversity for LOTO-style analysis.
- Unit heuristics (ft, kt, fpm, lb) are logged; verify against Project 85 docs when possible.

---

## 3. OpenSky Network (Trino) — physics-label robustness

Use when you need a **different telemetry source / operational mix**, or when DASHlink fuel quality is weak.

### 3.1 Access setup

1. Confirm academic access to the OpenSky historical Trino database.
2. Install/configure `pyopensky` (or use `traffic` history with your credentials).
3. Run a **tiny** test query first (one aircraft, a few hours):

```python
from pyopensky.trino import Trino
trino = Trino()
df = trino.history(start="2024-01-01", stop="2024-01-01 02:00", icao24="YOUR_ICAO24")
print(len(df), df.columns if df is not None else None)
```

If access fails, the loader can still exercise the pipeline with **synthetic** OpenSky-like data (`synthetic_fallback=True` by default in the build helpers). Synthetic data is only for code validation, not scientific conclusions.

### 3.2 Pilot extraction + featured dataset

Keep the time window short and flight count low:

```bash
python -m physics.external_audit.build_featured_audit --source opensky \
  --start 2024-01-01 \
  --stop 2024-01-01 06:00 \
  --max-flights 10 \
  --out audit_results/opensky_pilot/featured_dataset_audit.parquet
```

Optional single aircraft:

```bash
python -m physics.external_audit.run_audit_pilot --source opensky \
  --start 2024-01-01 --stop 2024-01-02 \
  --icao24 abc123 \
  --max-flights 5 \
  --out-dir audit_results/opensky_pilot
```

### 3.3 Full pilot suite

```bash
python -m physics.external_audit.run_audit_pilot --source opensky \
  --start 2024-01-01 \
  --stop 2024-01-01 06:00 \
  --max-flights 10 \
  --out-dir audit_results/opensky_pilot \
  --model lgbm
```

### 3.4 Critical interpretation rule

OpenSky has **no independent fuel ground truth**.

- Labels come from OpenAP (`FuelFlow.enroute`) via the existing baseline.
- Parquet columns: `label_source=physics_openap`, `label_is_physics_derived=True`.
- `actual_fuel_kg` is set equal to `physics_fuel_kg` for schema compatibility → residual ≈ 0.
- Report this as a **physics-label / telemetry-shift robustness test**, not external fuel validation.

Useful comparisons still include:

- Energy features help (or not) under ADS-B sparsity  
- Flow target vs Direct ranking under a different operational mix  
- Sparsity distributions vs PRC2025  

---

## 4. Re-run experiments on a saved parquet

```bash
python -m physics.external_audit.run_audit_pilot \
  --parquet audit_results/dashlink_pilot/featured_dataset_audit.parquet \
  --out-dir audit_results/dashlink_pilot_rerun \
  --model lgbm \
  --test-size 0.25
```

---

## 5. CLI cheat sheet

| Goal | Command |
|------|---------|
| Demo pilot | `python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8` |
| Probe one MAT | `python -m physics.external_audit.dashlink_loader PATH.mat --probe` |
| DASHlink pilot | `... run_audit_pilot --source dashlink --dashlink-dir DIR --max-flights 5` |
| OpenSky pilot | `... run_audit_pilot --source opensky --start DATE --stop DATE --max-flights 10` |
| Build parquet only | `python -m physics.external_audit.build_featured_audit --source demo\|dashlink\|opensky ...` |
| Existing parquet | `... run_audit_pilot --parquet PATH.parquet --out-dir DIR` |

Common flags:

- `--max-flights N` — pilot cap (start small)  
- `--out-dir DIR` — results folder  
- `--model lgbm|xgb|rf|cat`  
- `--n-bootstrap N` — significance iterations (default 2000; use 500 for quick pilots)  
- `--test-size 0.25` — flight-level holdout fraction  

---

## 6. Decision after both pilots

Answer (from the audit package):

1. Cleanest fuel target construction?  
2. Most meaningful shift from PRC2025?  
3. Enough aircraft diversity for limited type analysis?  
4. Effort to a publishable external result?  
5. Run both datasets or pick one primary?

**Default path (July 2026):** DASHlink first for labels → OpenSky if fuel/diversity is insufficient → optionally both for complementary strengths.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Demo pilot fails on import | Missing deps | `pip install -r requirements.txt` |
| Empty DASHlink load | Wrong path / corrupt mat | Check `--dashlink-dir`, open with `scipy.io.loadmat` manually |
| All fuel null | No matching fuel param names | `--probe` keys; extend patterns in `dashlink_loader.py` if docs give names |
| OpenSky empty | No Trino auth | Configure `pyopensky`; or accept synthetic fallback for code only |
| Unstable MAE / CI | Too few test flights | Raise `--max-flights`; ensure ≥5–8 flights for a meaningful split |
| “physics-derived” warning | OpenSky path | Expected — do not over-claim absolute error |

---

*Document version: 1.0 — matches `physics/external_audit` pilot package.*
