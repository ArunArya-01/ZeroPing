# ZeroPing
# AI-Based Aircraft Engine Health Monitoring and Emergency Pilot Decision Support System

## Overview
This project aims to develop an intelligent system that monitors aircraft engine health and assists pilots during abnormal situations. The system uses machine learning to analyze engine telemetry data and detect early signs of engine degradation. It also provides decision support to pilots by translating complex engine data into clear warnings and suggested actions.

The goal of this project is to improve aviation safety by enabling early detection of potential engine problems and enhancing pilot situational awareness during emergencies.

---

## Problem Statement
Modern aircraft engines generate large amounts of sensor data such as vibration, temperature, fuel flow, and pressure. While existing aircraft systems monitor these parameters, many alerts are triggered only when a threshold limit is exceeded. Subtle degradation patterns may remain unnoticed until they become critical.

This project addresses this problem by using machine learning to detect early anomalies in engine performance and provide predictive insights before serious failures occur.

---

## Objectives
- Develop a machine learning model to analyze aircraft engine telemetry data.
- Detect early degradation patterns and potential engine anomalies.
- Provide interpretable insights using explainable AI techniques.
- Build a simulation dashboard that visualizes engine health status.
- Provide decision support recommendations for pilots during abnormal conditions.

---

## System Components

### 1. Engine Health Monitoring
The system analyzes engine parameters such as:
- Vibration
- Exhaust Gas Temperature (EGT)
- Fuel Flow
- Compressor Speed
- Oil Pressure

Machine learning models are used to detect anomalies and predict possible engine degradation.

### 2. Explainable AI
Explainable AI techniques help identify which parameters contribute to engine health degradation. This improves trust and interpretability in safety-critical environments.

### 3. Risk Assessment
The system assigns a health score and risk level based on the detected engine conditions.

Example risk levels:
- Normal
- Warning
- Critical

### 4. Emergency Pilot Decision Support
Based on the detected anomalies, the system provides recommendations such as:
- Monitor engine parameters
- Reduce engine thrust
- Consider diversion to the nearest airport

---

## Technologies Used
- Python
- Machine Learning (Scikit-learn / TensorFlow / PyTorch)
- Data Analysis (Pandas, NumPy)
- Explainable AI (SHAP / LIME)
- Data Visualization (Matplotlib / Plotly / Dash)

---

## Dataset
The system can be trained using publicly available aircraft engine datasets such as:
- NASA Turbofan Engine Degradation Dataset

---

## Expected Outcomes
- Early detection of engine anomalies
- Improved interpretability of engine health data
- Enhanced pilot situational awareness
- A prototype AI-based aviation safety monitoring system

---

## Future Scope
- Integration with real-time aircraft telemetry
- Digital twin simulation of aircraft engines
- Integration with cockpit decision support systems
- Real-time monitoring dashboards for aircraft operations

---

## Conclusion
This project explores how artificial intelligence can enhance aircraft engine health monitoring and assist pilots during abnormal situations. By combining predictive analytics, explainable AI, and decision support systems, the project demonstrates a new approach to improving aviation safety.
