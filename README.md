<div align="center">

## EngineSentinel

<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"></a>

**Intelligent Aircraft Engine Health Monitoring & Emergency Decision Support System**

</div>

---

## Overview

EngineSentinel analyzes turbofan engine sensor data from the NASA C-MAPSS dataset to predict Remaining Useful Life (RUL), detect anomalies, explain predictions using SHAP, compute health indices, and provides an interactive React + FastAPI dashboard.

## Features

| Feature | Description |
|---------|-------------|
| **RUL Prediction** | Predicts remaining cycles before failure using Random Forest, XGBoost, and LSTM |
| **Anomaly Detection** | Identifies unusual engine behavior using Isolation Forest |
| **Explainable AI** | SHAP values explain which sensors influence predictions |
| **Health Score** | Engine Health Index (0–100) based on RUL, anomalies, and degradation |
| **Risk Assessment** | Risk levels (Green/Yellow/Red) with advisory messages |
| **Digital Twin** | Simulates real-time engine degradation trends |
| **Interactive Dashboard** | React + TypeScript + Tailwind CSS visualization |

## Dashboard Overview

The frontend dashboard consists of several components:

- **EngineSelector** — Allows selecting different engines for monitoring
- **RemainingLife** — Displays the predicted Remaining Useful Life (RUL) for the selected engine
- **HealthGauge** — Shows the engine health index as a gauge
- **SensorMonitoring** — Visualizes real-time sensor data
- **DegradationSimulation** — Simulates engine degradation trends (digital twin)
- **ExplainableAI** — Shows SHAP explanations for the predictions
- **RiskStatus** — Displays the risk level (Green/Yellow/Red) and advisory messages
- **SystemArchitecture** — Provides an overview of the system architecture

## Project Structure

```
EngineSentinel/
├── anomaly_detection/       # Isolation Forest anomaly detection
├── api/                     # FastAPI REST API backend
├── data_processing/         # Data loading, preprocessing, feature engineering
├── digital_twin/            # Digital twin simulation
├── explainable_ai/          # SHAP-based explanations
├── frontend/                # React + TypeScript frontend
├── health_risk/             # Health scoring & risk assessment
├── ml_prediction/           # ML models (RF, XGBoost, LSTM)
├── requirements.txt         # Python dependencies
└── requirements-dev.txt     # Development & CI dependencies
```

## Prerequisites

- **Python**: 3.8+ | **Node.js**: 16+ | **OS**: Windows, macOS, Linux | **RAM**: 8 GB+ recommended

## Installation

```bash
git clone https://github.com/ArunArya-01/EngineSentinel.git
cd EngineSentinel
pip install -r requirements.txt

# Create Dataset directory and add C-MAPSS files
mkdir -p Dataset
# Download from: https://www.kaggle.com/datasets/bishal098/nasa-turbofan-engine-degradation-simulation
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in Dataset/
```

## How to Run

```bash
# Step 1: Train models
python -m ml_prediction.trainer

# Step 2: Start FastAPI backend
uvicorn api.main:app --reload     # API docs at http://localhost:8000/docs

# Step 3: Launch frontend
cd frontend && npm install && npm run dev    # Frontend at http://localhost:5173
```

## Risk Levels

| Risk Level | Health Index | Meaning | Recommended Action |
|------------|--------------|---------|--------|
| 🟢 Green | 70–100 | Healthy | Normal operation |
| 🟡 Yellow | 40–69 | Warning | Monitor closely, schedule maintenance |
| 🔴 Red | 0–39 | Critical | Immediate inspection required |

## Key Metrics

- **RUL** — Remaining operational cycles until failure
- **Health Index** — Composite score (0–100) based on RUL, anomalies, and degradation
- **Anomaly Score** — Higher values indicate more unusual sensor behaviour

## CI/CD

GitHub Actions runs linting (flake8, black, isort), dependency validation, security scans (safety), and tests.

```bash
# Run locally
pip install -r requirements-dev.txt
flake8 . --count --show-source --statistics
black --check .
pytest tests/ -v
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run all commands from the project root |
| `Dataset not found` | Ensure C-MAPSS files are placed in `Dataset/` |
| `Import errors` | Run `pip install -r requirements.txt` |
| `Backend not starting` | Try `uvicorn api.main:app --reload --log-level debug` |
| `Frontend not loading` | Run `npm install` inside the `frontend/` directory |

## License

MIT License — See [LICENSE](LICENSE) file.

## Acknowledgments

- **NASA** for the C-MAPSS turbofan engine degradation simulation dataset
- **Open-source community** for FastAPI, React, scikit-learn, XGBoost, TensorFlow, and SHAP
