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

This project aims to develop a **simulation-based aircraft engine monitoring system** that analyzes turbofan engine sensor data to predict Remaining Useful Life (RUL), detect abnormal patterns, explain model predictions, compute engine health risk levels, and display the results through an interactive dashboard.

The system demonstrates how **machine learning and explainable AI techniques** can enhance aircraft engine health monitoring and decision support.

EngineSentinel integrates multiple technologies including predictive machine learning models, anomaly detection algorithms, explainable AI methods, and digital twin simulation to provide a comprehensive monitoring framework.

The system transforms complex engine telemetry data into clear operational insights that can assist engineers and pilots in understanding engine degradation trends and potential risks.

---

## Project Objective

The primary objective of this project is to demonstrate how **machine learning models and explainable AI techniques** can improve aircraft engine monitoring systems.

The project focuses on:

- Predicting engine **Remaining Useful Life (RUL)**
- Detecting **anomalous engine behavior**
- Explaining machine learning predictions
- Calculating **engine health scores**
- Assessing **risk levels**
- Providing a visual dashboard for monitoring

The goal is to convert complex engine sensor data into **interpretable insights that support better decision making**.

---

## Features

### Remaining Useful Life (RUL) Prediction

Predicts how many more operational cycles an engine can run before failure.

The system uses multiple machine learning models including:

- Random Forest
- XGBoost / Gradient Boosting
- LSTM (Long Short-Term Memory)

These models analyze historical sensor data to estimate degradation trends.

---

### Anomaly Detection

Detects unusual engine behavior using the **Isolation Forest algorithm**.

This module identifies abnormal patterns in sensor data that may indicate potential issues or unexpected engine performance changes.

---

### Explainable AI

The system integrates **SHAP (SHapley Additive Explanations)** to interpret machine learning predictions.

Explainable AI helps determine:

- Which sensors influence the model's predictions
- Why a particular engine is predicted to fail earlier
- What features contribute most to degradation trends

---

### Health Score Calculation

A comprehensive **Engine Health Index (0–100)** is computed based on:

- predicted RUL
- anomaly detection scores
- sensor degradation trends

This health index provides a simplified indicator of overall engine condition.

---

### Risk Assessment

The engine health index is translated into operational risk levels.

| Risk Level | Health Index | Meaning | Action |
|------------|-------------|--------|--------|
| Green | 70–100 | Healthy | Normal operation |
| Yellow | 40–69 | Warning | Monitor closely |
| Red | 0–39 | Critical | Immediate inspection |

---

### Digital Twin Simulation

The digital twin module simulates **real-time engine degradation trends**.

It integrates predictions, anomaly scores, and health indices to simulate how engine conditions evolve across operational cycles.

---

### Interactive Dashboard

An interactive monitoring dashboard is built using **Streamlit and Plotly**.

The dashboard provides real-time visualization of:

- Engine health scores
- Remaining useful life
- Sensor importance
- Anomaly detection results
- Risk levels
- Historical trends

---

## Modules

### 1. Data Processing Module

Handles loading, cleaning, and preprocessing of the C-MAPSS dataset.

Responsibilities include:

- Data loading
- Sensor normalization
- Feature engineering
- Sliding window creation for time series models

---

### 2. Machine Learning Prediction Module

Implements predictive models including:

- Random Forest
- XGBoost
- LSTM neural networks

These models estimate engine degradation and predict RUL.

---

### 3. Anomaly Detection Module

Detects abnormal engine behavior using Isolation Forest.

It produces anomaly scores that help identify unusual sensor patterns.

---

### 4. Explainable AI Module

Uses SHAP to interpret predictions made by machine learning models.

It highlights which sensors contribute most to the predictions.

---

### 5. Engine Health Score Module

Combines multiple signals including RUL and anomaly scores to compute a simplified engine health index.

---

### 6. Risk Assessment Module

Converts health indices into operational risk levels with advisory messages.

---

### 7. Digital Twin Simulation Layer

Simulates engine degradation trends over time and visualizes health metrics dynamically.

---

### 8. Interactive Dashboard

Provides a graphical interface for monitoring all system outputs including predictions, explanations, health scores, and risk alerts.

---

## System Architecture

The EngineSentinel system follows a modular architecture where each component performs a specific role in the monitoring pipeline.

```
Engine Sensor Data (C-MAPSS Dataset)
        ↓
Data Processing & Feature Engineering
        ↓
Machine Learning Prediction Models
(Random Forest / XGBoost / LSTM)
        ↓
Remaining Useful Life Prediction
        ↓
Anomaly Detection (Isolation Forest)
        ↓
Explainable AI (SHAP)
        ↓
Engine Health Score Calculation
        ↓
Risk Assessment Module
        ↓
Digital Twin Simulation
        ↓
Interactive Dashboard
```

---

## Prerequisites

- Python 3.8 or higher
- Windows / macOS / Linux
- Minimum 8GB RAM recommended
- At least 2GB disk space

---

## Directory Structure

```
EngineSentinel/
├── anomaly_detection/
├── data_processing/
├── digital_twin/
├── explainable_ai/
├── health_risk/
├── ml_prediction/
├── dashboard/
├── Dataset/
└── requirements.txt
```

---

## Setup

### Install Dependencies

```
pip install -r requirements.txt
```

---

### Download Dataset

Download the **NASA C-MAPSS turbofan engine degradation dataset**.

Place the files inside the `Dataset` directory.

Dataset source:

https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

---

### Train Models

Run the training script:

```
python -m EngineSentinel.ml_prediction.trainer
```

---

### Launch Dashboard

```
streamlit run EngineSentinel/dashboard/dashboard_app.py
```

The dashboard will open in your web browser.

---

## Key Metrics

| Metric | Description |
|------|-------------|
| RUL | Remaining Useful Life before failure |
| Health Index | Composite engine health score |
| Anomaly Score | Degree of abnormal behavior |
| Sensor Importance | Key features influencing predictions |

---

## Troubleshooting

| Issue | Solution |
|------|----------|
| Import errors | Ensure dependencies are installed |
| Dataset missing | Verify files exist in Dataset folder |
| Slow training | Reduce dataset size or adjust parameters |
| Dashboard not loading | Ensure Streamlit is installed |

---

## Acknowledgements

- NASA for the C-MAPSS turbofan engine dataset
- Open-source libraries including Streamlit, Scikit-learn, XGBoost, TensorFlow, and SHAP

---

## License

MIT License