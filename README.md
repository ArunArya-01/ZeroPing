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

As of June 2026, this repository contains:

- A Hugging Face backed data loader for remote dataset access.
- OpenAP fuel-flow baseline generation.
- Feature engineering for trajectory, phase, energy, operational, and weather-proxy features.
- Experiment scripts for baseline modeling, ablations, stacking, aircraft experts, and verification.
- Generated tables and figures under `figures/`.
- Paper-oriented summaries under `papers/`.

Best documented result so far: Energy + Weather Hybrid XGBoost at about **83.76 kg MAE** on a held-out flight-level split. See [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md) and [papers/hybrid_model_summary.md](papers/hybrid_model_summary.md) for details.

## Repository Structure

```text
ZeroPing/
├── data/
│   ├── __init__.py
│   └── loader.py                    # Remote Hugging Face dataset loader
├── physics/
│   ├── openap_baseline.py           # OpenAP interval fuel baseline
│   ├── feature_engineering.py       # Energy and operational features
│   ├── weather_features.py          # ISA and wind proxy features
│   ├── build_featured_dataset.py    # Main featured dataset builder
│   ├── enrich_featured_dataset.py   # Additional feature enrichment
│   ├── enrich_v3_features.py        # V3 feature set enrichment
│   ├── eval_framework.py            # Evaluation and bootstrap utilities
│   └── mlp_residual.py              # Neural residual experiments
├── notebooks/                       # Reproducible experiment scripts
├── figures/                         # Generated plots, leaderboards, tables
├── papers/                          # Research summaries and paper drafts
├── featured_dataset*.parquet         # Materialized modeling datasets
├── requirements.txt                 # Runtime and experiment dependencies
├── setup.cfg                        # Tool configuration
└── .github/workflows/ci.yml         # CI/CD workflow
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
| `notebooks/05_baseline_modeling.py` | Baseline modeling experiments. |
| `notebooks/09_physics_features_v3.py` | Energy/weather feature ablations. |
| `notebooks/12_verify_ensemble.py` | Ensemble verification workflow. |

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

Run feature enrichment:

```bash
PYTHONPATH=. python physics/enrich_featured_dataset.py
PYTHONPATH=. python physics/enrich_v3_features.py
```

Run experiment scripts:

```bash
PYTHONPATH=. python notebooks/05_baseline_modeling.py
PYTHONPATH=. python notebooks/09_physics_features_v3.py
PYTHONPATH=. python notebooks/12_verify_ensemble.py
```

Many scripts expect the generated parquet artifacts in the repository root and write outputs to `figures/`.

## Modeling Summary

The main experiments compare:

- OpenAP-only physics baseline.
- Direct hybrid models that predict `actual_fuel_kg` with `physics_fuel_kg` as an input.
- Residual models that predict `actual_fuel_kg - physics_fuel_kg`.
- Feature ablations for energy-state features, operational descriptors, weather proxies, mass features, vertical embeddings, stacking, and aircraft-type experts.

Evaluation uses flight-level train/test splits to avoid leakage between intervals from the same flight. Statistical comparisons use flight-clustered bootstrap tests where applicable.

## Generated Artifacts

Important generated files include:

- `featured_dataset.parquet`
- `featured_dataset_mass.parquet`
- `featured_dataset_vrate.parquet`
- `figures/final_leaderboard.csv`
- `figures/table_v3_leaderboard.csv`
- `figures/table_significance_v3_all.csv`
- `figures/table_verify_ensemble.csv`

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

## References

- Dataset: [`aerotwin/aero-data`](https://huggingface.co/datasets/aerotwin/aero-data)
- OpenAP: <https://github.com/junzis/openap>
- Project status: [PROJECT_STATUS_REPORT.md](PROJECT_STATUS_REPORT.md)
- Hybrid model summary: [papers/hybrid_model_summary.md](papers/hybrid_model_summary.md)

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
