<div align="center">

## EngineSentinel

<sup>
<a href="#"><img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python"></a>
<a href="#"><img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-orange?style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/Deep%20Learning-LSTM-red?style=flat-square"></a>
<a href="#"><img src="https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?style=flat-square&logo=streamlit"></a>
<a href="#"><img src="https://img.shields.io/badge/Visualization-Plotly-purple?style=flat-square"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"></a>
</sup>

<sub><strong>Intelligent Aircraft Engine Health Monitoring and Emergency Decision Support System</strong></sub>
<br>
<sub><em>A simulation-based AI system for predictive aircraft engine monitoring and pilot decision support.</em></sub>

</div>

---

## Overview

EngineSentinel is a comprehensive **aircraft engine health monitoring system** that analyzes turbofan engine sensor data from the NASA C-MAPSS dataset to predict Remaining Useful Life (RUL), detect abnormal patterns, explain model predictions using SHAP, compute engine health indices, and visualize everything through an interactive Streamlit dashboard.

The system demonstrates how **machine learning and explainable AI** can enhance aircraft engine health monitoring and decision support in real-world scenarios.

---

## Features

| Feature | Description |
|---------|-------------|
| **RUL Prediction** | Predicts remaining operational cycles before engine failure using Random Forest, XGBoost, and LSTM models |
| **Anomaly Detection** | Identifies unusual engine behavior using Isolation Forest algorithm |
| **Explainable AI** | Provides interpretable explanations using SHAP values to understand which sensors influence predictions |
| **Health Score** | Computes a comprehensive Engine Health Index (0-100) based on RUL, anomalies, and degradation trends |
| **Risk Assessment** | Translates health metrics into actionable risk levels (Green/Yellow/Red) with advisory messages |
| **Digital Twin** | Simulates real-time engine degradation and health trends over time |
| **Interactive Dashboard** | Visualizes all metrics with interactive charts using Streamlit and Plotly |

---

## Project Structure

```
ZeroPing/
├── anomaly_detection/          # Anomaly detection using Isolation Forest
│   └── isolation_forest_detector.py
├── data_processing/            # Data loading, preprocessing, and feature engineering
│   ├── data_loader.py          # C-MAPSS dataset loader
│   ├── data_pipeline.py        # Complete data pipeline
│   ├── feature_engineering.py  # RUL label generation, sequence creation
│   └── preprocessor.py         # Sensor normalization, constant sensor removal
├── digital_twin/              # Digital twin simulation layer
│   └── digital_twin_simulator.py
├── explainable_ai/            # SHAP-based model explanations
│   └── shap_explainer.py
├── health_risk/               # Health scoring and risk assessment
│   ├── health_calculator.py
│   └── risk_evaluator.py
├── ml_prediction/             # ML models for RUL prediction
│   ├── random_forest_model.py
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── trainer.py             # Model training script
│   └── evaluator.py
├── dashboard/                # Streamlit dashboard
│   └── dashboard_app.py
├── Dataset/                  # C-MAPSS dataset files (you need to add these)
└── requirements.txt         # Python dependencies
```

---

## Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM recommended for training models
- **Disk Space**: At least 2GB for dataset and model storage

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ZeroPing
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The following packages will be installed:
- pandas, numpy
- scikit-learn
- tensorflow
- xgboost
- shap
- plotly
- streamlit
- joblib

### 3. Download the Dataset

The project uses the **NASA C-MAPSS turbofan engine degradation simulation dataset**.

1. Create the Dataset directory:
```bash
mkdir -p Dataset
```

2. Download the dataset from: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation

3. Place the following files in the `Dataset` folder:
   - `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` (for FD001)
   - Or use FD002, FD003, FD004 for more advanced datasets

---

## How to Run

### Step 1: Train the Models

Train the machine learning models (this also runs the data pipeline):

```bash
python -m ml_prediction.trainer
```

This will:
- Load and process the C-MAPSS dataset
- Train Random Forest, XGBoost, and LSTM models
- Train the Isolation Forest anomaly detector
- Save models to `ml_prediction/models/FD001/`

### Step 2: Launch the Dashboard

```bash
streamlit run dashboard/dashboard_app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Dashboard Features

- **Select Dataset**: Choose FD001, FD002, FD003, or FD004
- **Select Model**: Choose Random Forest or XGBoost for RUL prediction
- **Select Engine ID**: View specific engine data and predictions
- **View Metrics**: See health score, RUL, anomaly scores
- **SHAP Explanations**: Understand which sensors influence predictions
- **Trend Charts**: Visualize degradation over time

---

##### Risk Levels Understanding the Output



| Risk Level | Health Index | Meaning | Action Required |
|------------|--------------|---------|-----------------|
| 🟢 Green | 70-100 | Healthy | Normal operation |
| 🟡 Yellow | 40-69 | Warning | Monitor closely, schedule maintenance |
| 🔴 Red | 0-39 | Critical | Immediate inspection required |

### Key Metrics

- **RUL (Remaining Useful Life)**: Number of cycles until engine failure
- **Health Index**: Composite score (0-100) based on RUL, anomalies, and degradation
- **Anomaly Score**: Higher values indicate more unusual behavior
- **Sensor Importance**: Which sensors contribute most to RUL predictions (via SHAP)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run from project root directory |
| `Dataset not found` | Ensure C-MAPSS files are in `Dataset/` folder |
| `Import errors` | Ensure all dependencies are installed: `pip install -r requirements.txt` |
| `Dashboard not loading` | Verify Streamlit is installed; try `streamlit run dashboard/dashboard_app.py --clear-cache` |
| `Memory errors during training` | Reduce model complexity or use smaller dataset subset |
| `SHAP errors` | Check that feature columns match between training and prediction |

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- **NASA** for providing the C-MAPSS turbofan engine degradation simulation dataset
- **Open-source community** for Streamlit, scikit-learn, XGBoost, TensorFlow, and SHAP libraries
