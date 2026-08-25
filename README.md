<div align="center">

# AeroFlux

**An Aviation Physics-Informed Machine Learning Framework for Fuel-Flow Prediction**

</div>

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

**Frozen teacher Combined RMSE: 221.33 kg** (R3 ensemble with dynamic mass; Rank 232.53 · Final 213.73). **Final held-out teacher (student parity): 213.62 kg** — do not confuse Combined with Final. Remaining Combined gap to winner (~201 kg): ~20 kg.

---

## Why This Project Exists

Aviation fuel burn is notoriously hard to predict at the interval (segment) level because operational telemetry — such as ADS-B and ACARS — is sparse, noisy, and incomplete. Aircraft mass is often unknown, air-data fields are frequently missing, and flights are subject to weather and routing variability that cannot be captured by physics theory alone.

This project exists to answer a practical question: *can combining a first-principles physics baseline with modern machine learning close the gap to operational reality?* Instead of relying on either approach alone, it uses the physics model to capture the interpretable bulk of fuel consumption and trains ML models to learn whatever remains unexplained — the residual structure in real flight data.

## What the Project Does

At a high level, the project:

1. **Loads real telemetry** — fused ADS-B and ACARS data from the PRC 2025 challenge, covering flights, aircraft, routes, and fuel labels.
2. **Builds physics baselines** — computes an interpretable fuel-flow estimate using OpenAP, given aircraft type, inferred true airspeed, altitude, vertical rate, and reference mass.
3. **Engineers features** — derives energy-state, operational, weather-proxy, and dynamic-mass features that help explain fuel burn beyond basic physics.
4. **Trains hybrid models** — gradient-boosted models (XGBoost, LightGBM, CatBoost) learn direct fuel predictions and fuel-flow residuals, assembled into a stacked ensemble with a ridge meta-learner.
5. **Guards against leakage** — splits data by flight and uses flight-clustered bootstrap testing so reported metrics reflect genuine generalization, not accidental information leakage.
6. **Validates externally** — an independent audit pipeline on NASA DASHlink and OpenSky data checks whether learned patterns replicate outside the training set.
7. **Distills into efficient students** — the frozen teacher ensemble's soft labels train smaller neural students (MLP and FT-Transformer) for deployment.
8. **Deploys as universal models** — students are exportable to open ONNX format and served by a compiled Rust binary with no Python runtime at inference.

## Key Results

- **Best ensemble (R3, dynamic mass):** 221.33 kg Combined RMSE.
- **Official baselines:** Large MLP deploy baseline at 225.95 kg Combined (0.26 ms CPU inference) vs the R3 teacher at 221.33 kg.
- **Cross-dataset validation** confirms the hybrid (physics + features) approach reduces error significantly versus physics alone, with statistically significant improvements from energy features and fuel-flow modeling.

## Design Principles

| Principle | What it means here |
|---|---|
| **Hybrid physics–ML** | A first-principles OpenAP baseline combined with gradient-boosted residual correction. |
| **Reproducible pipelines** | Deterministic data loading, featured-dataset building, and a frozen statistical/testing protocol. |
| **Leakage-safe evaluation** | Flight-level splits and flight-clustered bootstrap significance testing. |
| **External validation** | Independent DASHlink / OpenSky audits to check generalization, not just in-sample fit. |
| **Explainability** | Native SHAP attribution for the production CatBoost hybrid. |
| **Modular & extensible** | An installable `aerotwin` package with clearly separated data, engine, models, and validation layers. |

---

## Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input — ADS-B + ACARS Telemetry"]
        A1["EUROCONTROL PRC 2025<br/>(fused ADS-B + ACARS)"]
    end

    subgraph DATA["Data Layer — src/aerotwin/data"]
        B1["AeroDataLoader<br/>Hugging Face remote"]
    end

    subgraph FEATURES["Feature Engineering — src/aerotwin/engine"]
        C1["OpenAP Baseline<br/>Fuel-flow (kg/s)"]
        C2["Energy-State Features<br/>PE + KE + rate"]
        C3["Operational Features<br/>phases / altitude / speed"]
        C4["Weather Proxies<br/>ISA + wind"]
        C5["R3 Dynamic Mass<br/>21 features"]
        C6["OpenAP Descriptors<br/>MTOW / OEW / thrust"]
    end

    subgraph MODELING["Modeling — stacked GBDT ensemble"]
        D1["XGBoost"]
        D2["LightGBM"]
        D3["CatBoost"]
        D4["Direct (kg) / Fuel-Flow (kg/s)"]
        D5["Ridge Meta-Learner<br/>5-fold GroupKFold OOF"]
        D6["Heavy-Aircraft Specialist (R1)<br/>+ P1E phase calibration"]
    end

    subgraph EVAL["Evaluation — src/aerotwin/validation"]
        E1["Flight-Level Split<br/>80/20"]
        E2["LOTO<br/>12 aircraft types"]
        E3["Temporal Rank / Final"]
        E4["Flight-Clustered Bootstrap<br/>significance"]
        E5["External Validation<br/>DASHlink + OpenSky"]
    end

    A1 --> B1
    B1 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> C6
    C6 --> D1
    C6 --> D2
    C6 --> D3
    D1 & D2 & D3 --> D4
    D4 --> D5
    D5 --> D6
    D6 --> E1
    D6 --> E2
    D6 --> E3
    E1 & E2 & E3 --> E4
    E4 --> E5

    style DATA fill:#1f4e79,color:#fff,stroke:#0d2f4e
    style FEATURES fill:#2e7d32,color:#fff,stroke:#1b5e20
    style MODELING fill:#b26a00,color:#fff,stroke:#7f4f00
    style EVAL fill:#6a1b9a,color:#fff,stroke:#4a148c
```

---

## ONNX Deployment (Universal Models)

Trained distillation students (e.g. the Large MLP) can be exported to the open
**ONNX** format and served by a **compiled Rust binary** — no Python, PyTorch, or
scikit-learn at inference time. The same `.onnx` artifact runs on any ONNX
Runtime, making the models framework-agnostic and portable across Python,
Rust, web/JavaScript, C++, and edge devices.

```text
best_model.pt (PyTorch)
      │  export_onnx.py  (run once where torch is installed)
      ▼
model.onnx  +  model.preproc.json  +  model.meta.json
      │  cargo build --release
      ▼
aerotwin-onnx-serving  (native binary · axum HTTP · ONNX Runtime)
```

### Why ONNX + Rust

| Concern | Python/PyTorch inference | ONNX + Rust host |
|---|---|---|
| Runtime deps | torch, sklearn, numpy, venv | single native binary + model file |
| Latency | interpreter + tensor copies | compiled, zero-copy array input |
| Portability | platform-locked | cross-platform, framework-agnostic |
| Model durability | tied to `.pt` checkpoints | open, self-describing `.onnx` |

The Rust server also reproduces the exact training-time preprocessing (median
impute → `StandardScaler` → `OneHotEncoder`) from a side-car `*.preproc.json`,
so raw feature rows map to the same model input the student saw in training.

### Relevant paths

- **`experiments/11_onnx_deploy/export_onnx.py`** — PyTorch → ONNX exporter + preprocessing JSON.
- **`rust/aerotwin-onnx-serving/`** — compiled inference server (axum + `ort`/ONNX Runtime).
- Generated artifacts (`*.onnx`, `*.pt`, `*.pkl`, `target/`) are git-ignored.

See `rust/aerotwin-onnx-serving/README.md` for full build/run details.

### AeroSim — Interactive Web Simulator

**`aero_sim/`** is a browser-based 3D flight simulation built with CesiumJS,
Three.js, and ONNX Runtime Web. It animates an aircraft flying between two
airports along a real route while predicting **per-segment fuel burn directly
from the trained AeroTwin model in the browser**. CesiumJS renders the globe,
route, and an auto-following aircraft; per-segment markers are color-coded by
predicted burn (green → red) with fuel-kg labels; and a Three.js fuel-tank
overlay animates remaining fuel as the flight progresses. A live HUD shows
progress, fuel used, fuel remaining, and distance. It reuses the same
`large_mlp.onnx` + `preproc.json` artifacts as the Rust server (placed under
`aero_sim/public/models/`); when they are absent it falls back to a physics
approximation so the demo always runs.

See `aero_sim/README.md` for full setup details.

---

## Repository Layout

The repository is organized into focused domains:

- **`src/aerotwin/`** — the installable Python package, split into:
  - **data** — loading fused ADS-B + ACARS telemetry from Hugging Face.
  - **engine** — the core pipeline: OpenAP baseline, feature engineering, weather features, dynamic mass model, evaluation framework, statistical protocol, and the official ensemble.
  - **models** — specialized architectures including shift-aware routing, a transformer feature-token corrector, and an MLP residual regressor.
  - **validation** — cross-dataset validation and the DASHlink + OpenSky audit pipeline.
- **`experiments/`** — numbered, reproducible experiment scripts, from data exploration through ensemble building, generalization checks, gap-closing, distillation, interpretability, and advanced studies.
- **`tests/`** — unit and integration tests guarding the core logic.
- **`docs/`** — status reports, RMSE audits, benchmark-parity checks, research paper drafts, and external-validation guides.
- **`scripts/`** — build and run utilities.
- **`aero_sim/`** — browser-based 3D flight fuel-burn simulator (CesiumJS + Three.js + ONNX Runtime Web).

---

## License

ZeroPing / AeroTwin is released under the [MIT License](LICENSE).
