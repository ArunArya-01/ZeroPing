<div align="center">

# ZeroPing / AeroTwin

**Physics-informed aircraft fuel-burn prediction — hybrid OpenAP + machine-learning modeling, evaluation, and cross-dataset validation.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why ZeroPing](#why-zeroping)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Access](#data-access)
- [Quick Start](#quick-start)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Cross-Dataset Validation](#cross-dataset-validation)
- [Testing & Quality Gates](#testing--quality-gates)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)
- [Citation](#citation)

---

## Overview

**AeroTwin** predicts interval-level aircraft fuel burn from real-world, fused ADS-B and ACARS telemetry (EUROCONTROL PRC 2025 challenge data). It evaluates a **hybrid physics + machine-learning** paradigm:

```text
predicted_fuel_kg = f(trajectory, aircraft, route, physics_fuel_kg, engineered_features)
```

- **Physics baseline.** [OpenAP](https://github.com/junzis/openap) provides an interpretable fuel-flow estimate from aircraft type, inferred true airspeed, altitude, vertical rate, and reference mass.
- **Residual learning.** Gradient-boosted models (XGBoost, LightGBM, CatBoost) learn the structure remaining in operational data — sparse telemetry, missing air-data, and unknown aircraft mass.
- **Rigorous evaluation.** Flight-level train/test splits prevent interval leakage; flight-clustered bootstrap tests quantify significance.
- **Credibility via external validation.** A second-dataset audit pipeline (NASA DASHlink + OpenSky) tests whether findings *replicate* outside the training distribution.

Dataset: [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) on Hugging Face.

---

## Why AeroTwin

| Capability | Description |
|---|---|
| **Hybrid physics–ML** | Combines a first-principles OpenAP baseline with gradient-boosted residual correction. |
| **Reproducible pipelines** | Deterministic data loader, featured-dataset builder, and frozen statistical protocol. |
| **Leakage-safe evaluation** | Flight-level splits and flight-clustered bootstrap significance testing. |
| **External validation** | Independent DASHlink / OpenSky auditors to check generalization, not just in-sample fit. |
| **Explainability** | Native SHAP attribution for the production CatBoost hybrid. |
| **Modular & tested** | Clean `data` / `physics` / `notebooks` separation with CI and unit tests. |

---

## Architecture

```text
               ┌────────────────────────────┐
  ADS-B/ACARS  │       data/loader.py       │  (Hugging Face: hf://)
  telemetry ─▶ │  AeroDataLoader (remote)   │
               └──────────────┬─────────────┘
                              │ flightlist, fuel labels, traj parquet
                              ▼
               ┌────────────────────────────┐
               │   physics/openap_baseline   │  OpenAP interval fuel-flow
               │   + feature_engineering     │  energy / operational features
               │   + weather_features        │  ISA & wind proxies
               └──────────────┬─────────────┘
                              │ featured_dataset.parquet
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │  eval_framework  →  Direct / Residual / Stacking /     │
   │  Experts  →  SHAP  →  flight-clustered bootstrap       │
   └──────────────┬───────────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────┐
   │  external_audit/  (DASHlink + OpenSky)                │
   │  cross_dataset_replication → generalization verdict    │
   └──────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
ZeroPing/
├── data/
│   ├── __init__.py
│   └── loader.py                      # Remote Hugging Face dataset loader
├── physics/
│   ├── openap_baseline.py             # OpenAP interval fuel baseline
│   ├── feature_engineering.py         # Energy and operational features
│   ├── weather_features.py            # ISA and wind proxy features
│   ├── mass_model.py                  # Dynamic mass estimation model (R3)
│   ├── gap_closing.py                 # Calibrators, heavy specialist, R1/R2 features
│   ├── official_benchmark.py          # Frozen ensemble training + OOF matrix
│   ├── build_featured_dataset.py      # Main featured dataset builder (PRC)
│   ├── eval_framework.py              # Evaluation and bootstrap utilities
│   ├── statistical_protocol.py        # Frozen inference/significance protocol
│   ├── shift_aware_routing.py         # Conditional shift-aware router
│   ├── cross_dataset_alignment.py     # Schema/scale harmonization
│   ├── external_vs_flow_eval.py       # Flow+Energy vs Direct on any parquet
│   ├── external_energy_ablation.py    # Energy-feature ablation on 2nd dataset
│   ├── cross_dataset_replication.py   # Multi-dataset replication verdict
│   ├── transformer_residual.py        # Transformer feature-token residual corrector
│   └── external_audit/                # Second-dataset validation package
│       ├── audit_utils.py             # Phase, energy, sparsity, intervals
│       ├── dashlink_loader.py         # Project 85 MAT → traj + fuel intervals
│       ├── opensky_loader.py          # OpenSky Trino + physics labels
│       ├── build_featured_audit.py    # featured_dataset_audit.parquet
│       ├── run_audit_pilot.py         # Pilot experiments A–E
│       └── README.md
├── notebooks/                         # Reproducible experiment scripts
├── tests/                             # Unit tests (incl. external_audit)
├── figures/                           # PRC plots, leaderboards, tables
├── docs/                              # Audit deliverables (parity, gap attribution)
├── audit_results/                     # External audit outputs
├── papers/                            # Research summaries
├── AeroTwin_External_Dataset_Audit_Package.md
├── HOW_TO_RUN_AUDIT.md                # Step-by-step external audit guide
├── PROJECT_STATUS_REPORT.md
├── requirements.txt
├── setup.cfg
└── pyproject.toml
```

---

## Installation

ZeroPing targets **Python 3.11+** (current `openap` releases require it).

```bash
git clone https://github.com/ArunArya-01/ZeroPing.git
cd ZeroPing

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some experimental scripts import optional ML packages (e.g. `xgboost`) that are not required for the core loader and audit pipeline. Install them only when running those experiments:

```bash
pip install xgboost       # for XGBoost-based experiments
```

---

## Data Access

The project reads the dataset remotely from Hugging Face via `hf://` paths, Polars, and `huggingface_hub`. A full local download is **not** required for the loader.

```python
from data import AeroDataLoader

loader = AeroDataLoader()
flightlist = loader.get_flightlist("train")
fuel = loader.get_fuel_labels("train")
paths = loader.sample_flight_files(split="train", n=5)
```

If authentication is required, set `HF_TOKEN` in your environment.

---

## Quick Start

Build (or refresh) the featured dataset:

```bash
PYTHONPATH=. python physics/build_featured_dataset.py
```

Run baseline and feature-ablation experiments:

```bash
PYTHONPATH=. python notebooks/05_baseline_modeling.py
PYTHONPATH=. python notebooks/09_physics_features_v3.py
PYTHONPATH=. python notebooks/12_verify_ensemble.py
PYTHONPATH=. python notebooks/14_shap_explainability.py
PYTHONPATH=. python notebooks/19_gap_closing_campaign.py
PYTHONPATH=. python notebooks/23_rmse_audit_agent.py
PYTHONPATH=. python notebooks/25_r3_dynamic_mass.py
PYTHONPATH=. python notebooks/26_r3_ensemble_mass.py
```

Offline audit smoke test (no external data required):

```bash
python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8
```

Run the test suite:

```bash
PYTHONPATH=. pytest tests/ -q
```

---

## Modeling Approach

The experiments compare three model families on interval-level fuel burn:

1. **OpenAP-only physics baseline** — interpretable, no learning.
2. **Direct hybrid** — predicts `actual_fuel_kg` with `physics_fuel_kg` as an input feature.
3. **Residual models** — predict `actual_fuel_kg − physics_fuel_kg`.

Additional studies cover feature ablations (energy-state, operational, weather proxies, mass, vertical embeddings), stacking ensembles, aircraft-type experts, leave-one-type-out (LOTO) generalization, and CatBoost SHAP explainability. Evaluation uses flight-level splits and flight-clustered bootstrap significance testing.

---

## Results

### Official PRC2025 Benchmark (Rank + Final)

| Split | MAE (kg) | RMSE (kg) | R² |
|-------|--------:|----------:|---:|
| **Rank** (Sep 2025) | 90.89 | 239.18 | 0.904 |
| **Final** (Oct 2025) | 87.35 | 220.86 | 0.918 |
| **Combined** | 88.75 | **228.25** | 0.913 |

**Best model:** Ensemble (XGB/LGBM/CatBoost × Direct + Fuel-Flow, Energy+Weather) + Ridge meta.  
**Published winner:** ≈201 kg Combined RMSE.  

### Gap-Closing Campaign

| Version | Variant | Combined RMSE | Δ vs 228.25 | Improvement |
|---------|---------|--------------:|-------------|-------------|
| v1.0 | Official ensemble (frozen V4) | **228.25** | reference | baseline |
| v1.1 | P1E phase affine + P2 Cat heavy specialist | **227.44** | −0.81 | heavy specialist |
| R1 | P1E + OpenAP descriptors in heavy specialist | **226.19** | −2.06 | aircraft physics |
| R2 | Fixed B744/B77L/A306 descriptors + R2 features | **225.25** | −3.00 | missing descriptor fix |
| **R3** | **P1E + dynamic mass model (21 features)** | **221.33** | **−6.92** | **mass estimation** |

> Remaining gap to winner (201 kg): **≈20 kg**.  
> Full leaderboard: `figures/table_current_rmse.csv`. RMSE audit: `CURRENT_MODEL_SUMMARY.md`.

### R3 Dynamic Mass Model

The single largest improvement comes from replacing the crude `mass = MTOW × 0.75` with 21 physics-informed mass features:
- **Takeoff weight** and **landing mass** estimated from aircraft specs + flight duration
- **Per-interval mass** via linear fuel-burn interpolation by flight fraction
- **Mass consumed**, **mass rate**, **fuel fraction**, **remaining fuel**
- Mass-scaled **potential/kinetic energy**, current **wing loading**
- **Phase-aware mass** (climb/cruise/descent differing fuel states)

Mass features reduce bias from +24 kg to **+3.9 kg** and narrow both heavy (−12 kg) and narrowbody (−6 kg) RMSE.

> Ablation results: `figures/table_rmse_R3_mass.csv`. Ensemble results: `figures/table_rmse_R3_mass_ensemble.csv`.  
> Source: `physics/mass_model.py`, `notebooks/25_r3_dynamic_mass.py`, `notebooks/26_r3_ensemble_mass.py`.

### Internal (PRC) leaderboard — protocol-separated

| Track | Best model | MAE | RMSE | Notes |
|-------|------------|----:|-----:|-------|
| **A · Fuel-Flow single model** | XGB Flow+Energy / LGBM Flow+Energy | **79.5** / 80.3 | 208.4 / **196.2** | MAE-best vs RMSE-best differ |
| **B · Direct single model** | XGB Energy+Weather | **83.8** | 212.0 | Main hybrid MAE story |
| **C · Direct stacking (competition)** | LGBM_meta 5-fold OOF | 84.3 | **202.9** | vs PRC winner RMSE 200.83 |

> Flow RMSE 196 and stack RMSE 203 are **not** ranked against each other (different targets / training). See `figures/LEADERBOARD_AUDIT.md`.

### SHAP explainability

Native CatBoost SHAP values on held-out flights identify `physics_fuel_kg`, energy-state, and operational features as the dominant drivers of the hybrid prediction. Outputs: `figures/table_shap_catboost.csv`, `figures/fig_shap_catboost_top_features.png`.

---

## Cross-Dataset Validation

Generalization is tested on an **independent** dataset rather than relying on in-sample fit alone.

### DASHlink pilot (NASA Project 85)

Real FDR data, tails 686/687, 15 airborne flights, 137 intervals, LightGBM, flight-level 75/25 split. Fuel targets integrated from `FF_1…FF_4` (LBS/HR → kg).

| Experiment | MAE (kg) | Notes |
|---|---:|---|
| Physics-only (OpenAP) | 140.1 | Type/mass defaults may not match fleet |
| Direct · base + physics | 25.5 | |
| Direct · base + energy + physics | 20.7 | Energy ablation ΔMAE ≈ **−4.9** (95% CI excludes 0) |
| Flow · base + energy + physics | **18.1** | vs matched Direct ΔMAE ≈ **−2.6** (95% CI excludes 0) |

**Qualitative verdict:** energy features **replicate**, Fuel-Flow target **replicates**, ML ≫ raw physics **replicates**. Absolute MAE is not comparable to PRC (~84 kg) due to differing interval scales, aircraft mix, and label construction. Details in `audit_results/dashlink_pilot/`. Full guide: [HOW_TO_RUN_AUDIT.md](HOW_TO_RUN_AUDIT.md).

---

## Testing & Quality Gates

### CI & Quality Gates

GitHub Actions (`.github/workflows/ci.yml`) runs:

- Python 3.11 environment setup.
- Syntax compilation for `data`, `physics`, and `notebooks`.
- Strict `flake8` fatal-error checks (E9, F63, F7, F82) for `data` and `physics`.
- Dependency installation from `requirements.txt`.
- Smoke imports for lightweight core modules.
- Security scanning with `safety`.

Heavier experimental modules requiring optional packages (e.g. `xgboost`) are syntax-checked but not smoke-imported.

Useful local checks:

```bash
python -m compileall -q data physics notebooks
flake8 data physics --count --select=E9,F63,F7,F82 --show-source --statistics
PYTHONPATH=. pytest tests/ -q
PYTHONPATH=. pytest tests/ -m slow    # protocol runs only
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `openap` will not install on Python 3.10 | Use Python 3.11 or newer. |
| `ModuleNotFoundError` for local packages | Run commands from the repo root with `PYTHONPATH=.`. |
| Hugging Face access fails | Check network access and set `HF_TOKEN` if required. |
| Experiment script cannot find parquet files | Generate the featured dataset first or confirm artifacts exist in the repo root. |
| `ModuleNotFoundError: xgboost` | Install the optional experiment dependency: `pip install xgboost`. |
| DASHlink `No trajectory channels found` | Use the updated `dashlink_loader` (extracts struct `.data`); probe with `--probe`. |
| OpenSky empty result | Configure `pyopensky` Trino credentials, or use the demo/synthetic fallback for code checks only. |

---

## References

- Dataset: [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data)
- OpenAP: <https://github.com/junzis/openap>
- Project status: [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md)
- Current RMSE audit: [CURRENT_MODEL_SUMMARY.md](CURRENT_MODEL_SUMMARY.md)
- Dynamic mass model: [physics/mass_model.py](physics/mass_model.py)
- R3 mass evaluation: [figures/table_rmse_R3_mass.csv](figures/table_rmse_R3_mass.csv), [figures/r3_summary.json](figures/r3_summary.json)
- Benchmark parity: [docs/BENCHMARK_PARITY_AUDIT.md](docs/BENCHMARK_PARITY_AUDIT.md)
- Gap attribution: [docs/RMSE_GAP_ATTRIBUTION.md](docs/RMSE_GAP_ATTRIBUTION.md)
- RMSE improvement backlog: [RMSE_IMPROVEMENT_BACKLOG.md](RMSE_IMPROVEMENT_BACKLOG.md)
- External audit how-to: [HOW_TO_RUN_AUDIT.md](HOW_TO_RUN_AUDIT.md)
- Audit package design: [AeroTwin_External_Dataset_Audit_Package.md](AeroTwin_External_Dataset_Audit_Package.md)
- Hybrid model summary: [papers/hybrid_model_summary.md](papers/hybrid_model_summary.md)

---

## License

ZeroPing / AeroTwin is released under the [MIT License](LICENSE).
