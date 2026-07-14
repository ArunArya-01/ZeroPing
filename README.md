<div align="center">

# AeroTwin

<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"></a>

**Physics-informed aircraft fuel burn prediction with OpenAP and machine learning**

</div>

---

## Overview

AeroTwin predicts aircraft fuel burn for labeled ACARS fuel intervals using real-world fused ADS-B and ACARS telemetry from the EUROCONTROL PRC 2025 challenge dataset.

The project evaluates a hybrid modeling approach:

```text
predicted_fuel_kg = f(trajectory, aircraft, route, physics_fuel_kg, engineered_features)
```

OpenAP provides a physics baseline from aircraft type, inferred true airspeed, altitude, vertical rate, and reference mass. Machine learning models then learn the remaining structure in real operational data, including sparse telemetry, missing air-data, and unknown aircraft mass.

Dataset: [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data) on Hugging Face.

## Current Status

As of July 2026, this repository contains:

- A Hugging Face backed data loader for remote dataset access (EUROCONTROL PRC 2025).
- OpenAP fuel-flow baseline generation and hybrid ML pipelines.
- Feature engineering for trajectory, phase, energy, operational, and weather-proxy features.
- Experiment scripts for baseline modeling, ablations, stacking, aircraft experts, LOTO, and verification.
- SHAP explainability for the CatBoost energy/weather/physics hybrid model.
- **External validation infrastructure** under `physics/external_audit/` (DASHlink + OpenSky pilots).
- A completed **NASA DASHlink Project 85 pilot** with qualitative replication of energy features and Fuel-Flow targets (see below).
- Generated tables and figures under `figures/` and `audit_results/`.
- Paper-oriented summaries under `papers/`.

**Best internal (PRC) result:** Energy + Weather Hybrid XGBoost at about **83.76 kg MAE** on a held-out flight-level split; ensemble RMSE **202.90 kg** vs PRC winner 200.83 kg (same-dataset benchmarking).

**External pilot (DASHlink Project 85, tails 686/687, 15 flights):** Energy features and Fuel-Flow target both **replicate** under flight-level holdout with integrated fuel-flow labels. See [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md), [HOW_TO_RUN_AUDIT.md](HOW_TO_RUN_AUDIT.md), and [physics/external_audit/README.md](physics/external_audit/README.md).

## Repository Structure

```text
ZeroPing/
├── data/
│   ├── __init__.py
│   └── loader.py                      # Remote Hugging Face dataset loader
│   # Optional local DASHlink: data/Tail_686_1/*.mat, data/Tail_687_1/*.mat
├── physics/
│   ├── openap_baseline.py             # OpenAP interval fuel baseline
│   ├── feature_engineering.py         # Energy and operational features
│   ├── weather_features.py            # ISA and wind proxy features
│   ├── build_featured_dataset.py      # Main featured dataset builder (PRC)
│   ├── eval_framework.py              # Evaluation and bootstrap utilities
│   ├── external_vs_flow_eval.py       # Flow+Energy vs Direct on any parquet
│   ├── cross_dataset_replication.py   # Multi-dataset replication verdict
│   ├── external_audit/                # Second-dataset validation package
│   │   ├── audit_utils.py             # Phase, energy, sparsity, intervals
│   │   ├── dashlink_loader.py         # Project 85 MAT → traj + fuel intervals
│   │   ├── opensky_loader.py          # OpenSky Trino + physics labels
│   │   ├── build_featured_audit.py    # featured_dataset_audit.parquet
│   │   ├── run_audit_pilot.py         # Pilot experiments A–E
│   │   └── README.md
│   └── ...
├── notebooks/                         # Reproducible experiment scripts
├── tests/                             # Unit tests (incl. external_audit)
├── figures/                           # PRC plots, leaderboards, tables
├── audit_results/                     # External audit outputs
│   └── dashlink_pilot/                # Real DASHlink pilot tables + figures
├── papers/                            # Research summaries
├── AeroTwin_External_Dataset_Audit_Package.md
├── HOW_TO_RUN_AUDIT.md                # Step-by-step external audit guide
├── PROJECT_STATUS_REPORT.md
├── featured_dataset*.parquet
└── requirements.txt
```

## Key Entry Points

| Path | Purpose |
|---|---|
| `data/loader.py` | Load flight lists, fuel labels, and per-flight trajectory parquet files from Hugging Face. |
| `physics/openap_baseline.py` | Build interval-level OpenAP fuel estimates and base trajectory features. |
| `physics/build_featured_dataset.py` | Materialize the main featured dataset. |
| `physics/feature_engineering.py` | Add energy-state and operational features. |
| `physics/weather_features.py` | Add atmosphere and wind proxy features. |
| `physics/eval_framework.py` | Train/evaluate models and run flight-clustered bootstrap tests. |
| `physics/statistical_protocol.py` | Frozen inference protocol: bootstrap/significance constants and the shared interpretation policy. |
| `physics/shift_aware_routing.py` | Conditional shift-aware router (Direct vs Flow+Energy) gated on operational-shift calibration. |
| `physics/cross_dataset_alignment.py` | Harmonize schemas and feature scales across multiple datasets. |
| `physics/external_vs_flow_eval.py` | Run the equivalent AeroTwin protocol on an independent dataset to test whether Flow+Energy still beats Direct. |
| `physics/external_energy_ablation.py` | Run the equivalent AeroTwin Energy-feature ablation (V3 E6) on an independent dataset. |
| `physics/cross_dataset_replication.py` | Multi-dataset Flow+Energy-vs-Direct replication verdict. |
| `physics/external_audit/` | Full second-dataset pipeline (DASHlink MAT loader, OpenSky, featured builder, pilot suite). |
| `physics/external_audit/dashlink_loader.py` | Load Project 85 FDR structs (`.data` / Rate / Units); reconstruct fuel from `FF_*`. |
| `physics/external_audit/run_audit_pilot.py` | Compact pilot: Direct, Fuel-Flow, energy ablation, flight-level split. |
| `HOW_TO_RUN_AUDIT.md` | End-to-end instructions for demo, DASHlink, and OpenSky pilots. |
| `notebooks/05_baseline_modeling.py` | Baseline modeling experiments. |
| `notebooks/09_physics_features_v3.py` | Energy/weather feature ablations. |
| `notebooks/12_verify_ensemble.py` | Ensemble verification workflow. |
| `notebooks/14_shap_explainability.py` | CatBoost SHAP explainability tables and plots. |
| `physics/transformer_residual.py` | Reusable Transformer feature-token residual corrector (`train_transformer_residual`). |

## Installation

Use Python 3.11. Some dependencies, including current `openap` releases, require Python 3.11 or newer.

```bash
git clone https://github.com/ArunArya-01/ZeroPing.git
cd ZeroPing

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some experimental scripts import optional ML packages that may not be installed by `requirements.txt` in every environment, such as `xgboost`. Install those only when running the corresponding experiments.

## Data Access

The project is designed to read the AeroTwin dataset remotely from Hugging Face using `hf://` paths, Polars, and `huggingface_hub`. A full local dataset download is not required for the loader.

```python
from data import AeroDataLoader

loader = AeroDataLoader()
flightlist = loader.get_flightlist("train")
fuel = loader.get_fuel_labels("train")
paths = loader.sample_flight_files(split="train", n=5)
```

If Hugging Face authentication is required, set `HF_TOKEN` in your environment.

## Typical Workflows

Build or refresh the featured dataset:

```bash
PYTHONPATH=. python physics/build_featured_dataset.py
```

Align multiple featured datasets (e.g. different versions or splits) onto a
shared schema and feature scale so they can be combined or transferred between
models without schema mismatch or distribution shift:

```bash
PYTHONPATH=. python physics/cross_dataset_alignment.py \
    featured_dataset.parquet featured_dataset_mass.parquet featured_dataset_vrate.parquet
```

Run feature enrichment:

```bash
PYTHONPATH=. python physics/enrich_featured_dataset.py
PYTHONPATH=. python physics/enrich_v3_features.py
```

### External dataset audit (DASHlink / OpenSky)

Offline smoke test (no external data):

```bash
python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8
```

Probe and run a **real DASHlink Project 85** pilot (MAT files under `data/`):

```bash
python -m physics.external_audit.dashlink_loader data/Tail_687_1/687200103200323.mat --probe

python -m physics.external_audit.run_audit_pilot \
  --source dashlink \
  --dashlink-dir data \
  --max-flights 15 \
  --out-dir audit_results/dashlink_pilot
```

OpenSky short-window pilot (physics-derived labels; needs Trino credentials or synthetic fallback):

```bash
python -m physics.external_audit.run_audit_pilot \
  --source opensky \
  --start 2024-01-01 --stop 2024-01-01 06:00 \
  --max-flights 10 \
  --out-dir audit_results/opensky_pilot
```

Full checklist: [HOW_TO_RUN_AUDIT.md](HOW_TO_RUN_AUDIT.md).

### Protocol helpers on an existing featured parquet

```bash
PYTHONPATH=. python physics/external_vs_flow_eval.py \
    --external /path/to/independent_featured_dataset.parquet \
    --internal figures/table_loto_evaluation_master.csv

PYTHONPATH=. python physics/external_energy_ablation.py \
    --external /path/to/independent_featured_dataset.parquet \
    --internal figures/table_significance_v3_e6.csv

PYTHONPATH=. python physics/cross_dataset_replication.py \
    /path/to/dataset_a.parquet /path/to/dataset_b.parquet \
    --outdir figures
```

### Tests

```bash
PYTHONPATH=. pytest tests/ -q
PYTHONPATH=. pytest tests/test_external_audit.py -q
PYTHONPATH=. pytest tests/ -m slow    # protocol runs only
```


```bash
PYTHONPATH=. python notebooks/05_baseline_modeling.py
PYTHONPATH=. python notebooks/09_physics_features_v3.py
PYTHONPATH=. python notebooks/12_verify_ensemble.py
PYTHONPATH=. python notebooks/14_shap_explainability.py
```

Many scripts expect the generated parquet artifacts in the repository root and write outputs to `figures/`.

## Modeling Summary

The main experiments compare:

- OpenAP-only physics baseline.
- Direct hybrid models that predict `actual_fuel_kg` with `physics_fuel_kg` as an input.
- Residual models that predict `actual_fuel_kg - physics_fuel_kg`.
- Feature ablations for energy-state features, operational descriptors, weather proxies, mass features, vertical embeddings, stacking, and aircraft-type experts.
- SHAP explainability for the direct CatBoost hybrid model, using CatBoost native SHAP values on held-out flights.
- **External pilots** on DASHlink (reconstructed fuel flow) and optionally OpenSky (physics-derived labels).

Evaluation uses flight-level train/test splits to avoid leakage between intervals from the same flight. Statistical comparisons use flight-clustered bootstrap tests where applicable.

### DASHlink pilot snapshot (July 2026)

Real NASA DASHlink Sample Flight Data (Project 85), tails 686/687, 15 airborne flights, 137 intervals, LightGBM, flight-level 75/25 split. Fuel targets from integrated `FF_1…FF_4` (LBS/HR → kg).

| Experiment | MAE (kg) | Notes |
|---|---:|---|
| Physics-only (OpenAP) | 140.1 | Type/mass defaults may not match fleet |
| Direct · base + physics | 25.5 | |
| Direct · base + energy + physics | 20.7 | Energy ablation ΔMAE ≈ **−4.9** (95% CI excludes 0) |
| Flow · base + energy + physics | **18.1** | vs matched Direct ΔMAE ≈ **−2.6** (95% CI excludes 0) |

**Qualitative:** energy features **replicate**; Fuel-Flow target **replicates**; ML ≫ raw physics **replicates**. Absolute MAE is not comparable to PRC (~84 kg) because interval scales, aircraft mix, and label construction differ. Details: `audit_results/dashlink_pilot/`.

## Generated Artifacts

Important generated files include:

- `featured_dataset.parquet`
- `featured_dataset_mass.parquet`
- `featured_dataset_vrate.parquet`
- `figures/final_leaderboard.csv`
- `figures/table_v3_leaderboard.csv`
- `figures/table_significance_v3_all.csv`
- `figures/table_verify_ensemble.csv`
- `figures/table_shap_catboost.csv`
- `figures/fig_shap_catboost_top_features.png`
- `figures/table_external_flow_vs_direct.csv`
- `figures/table_external_vs_internal.csv`
- `figures/fig_external_vs_flow.png`
- `figures/table_external_energy_ablation.csv`
- `figures/table_external_energy_ablation_significance.csv`
- `figures/table_external_energy_ablation_vs_internal.csv`
- `figures/fig_external_energy_ablation.png`
- `audit_results/dashlink_pilot/featured_dataset_audit.parquet`
- `audit_results/dashlink_pilot/table_audit_pilot_metrics.csv`
- `audit_results/dashlink_pilot/table_audit_pilot_significance.csv`
- `audit_results/dashlink_pilot/table_audit_qualitative_comparison.csv`
- `audit_results/dashlink_pilot/figures/fig_audit_*.png`

These files are experiment artifacts and may be regenerated by the scripts above.

## CI/CD

GitHub Actions currently runs:

- Python 3.11 setup.
- Syntax compilation for `data`, `physics`, and `notebooks`.
- Strict `flake8` fatal-error checks for importable source under `data` and `physics`.
- Dependency installation from `requirements.txt`.
- Smoke imports for lightweight core modules.
- Security scanning with `safety`.
- Build verification for core modules and script syntax.

The CI intentionally does not smoke-import heavier experimental modules that require optional packages such as `xgboost`; those modules are still syntax-checked.

Useful local checks:

```bash
python -m compileall -q data physics notebooks
flake8 data physics --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `openap` will not install on Python 3.10 | Use Python 3.11 or newer. |
| `ModuleNotFoundError` for local packages | Run commands from the repo root with `PYTHONPATH=.`. |
| Hugging Face access fails | Check network access and set `HF_TOKEN` if required. |
| Experiment script cannot find parquet files | Generate the featured dataset first or confirm the parquet artifacts exist in the repo root. |
| `ModuleNotFoundError: xgboost` | Install optional experiment dependency with `pip install xgboost`. |
| DASHlink `No trajectory channels found` | Use updated `dashlink_loader` (extracts struct `.data`); probe with `--probe`. |
| DASHlink channels all `meta_*` size 1 | Old loader bug; upgrade `physics/external_audit/dashlink_loader.py`. |
| OpenSky empty result | Configure `pyopensky` Trino credentials, or use demo/synthetic fallback for code checks only. |

## References

- Dataset: [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data)
- OpenAP: <https://github.com/junzis/openap>
- Project status: [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md)
- External audit how-to: [HOW_TO_RUN_AUDIT.md](HOW_TO_RUN_AUDIT.md)
- Audit package design: [AeroTwin_External_Dataset_Audit_Package.md](AeroTwin_External_Dataset_Audit_Package.md)
- Hybrid model summary: [papers/hybrid_model_summary.md](papers/hybrid_model_summary.md)

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
