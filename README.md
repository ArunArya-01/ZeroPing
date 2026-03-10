# EngineSentinel: Intelligent Aircraft Engine Health Monitoring and Emergency Decision Support System

This project aims to develop a simulation-based aircraft engine monitoring system that analyzes turbofan engine sensor data to predict Remaining Useful Life (RUL), detect abnormal patterns, explain model predictions, compute engine health risk levels, and display the results through an interactive dashboard.

## Project Objective

The system will demonstrate how machine learning and explainable AI can enhance aircraft engine health monitoring and decision support.

## Modules:

1.  **Data Processing Module:** Handles loading, cleaning, normalizing, and feature engineering of the C-MAPSS dataset.
2.  **Machine Learning Prediction Module:** Implements models (Random Forest, XGBoost/Gradient Boosting, LSTM) for RUL prediction.
3.  **Anomaly Detection Module:** Detects abnormal engine behavior using methods like Isolation Forest.
4.  **Explainable AI Module:** Uses SHAP to explain model predictions and identify influential sensors.
5.  **Engine Health Score Module:** Computes a simplified Engine Health Index (0-100).
6.  **Risk Assessment Module:** Converts health index into operational risk levels (Green, Yellow, Red) with advisory messages.
7.  **Digital Twin Simulation Layer:** Simulates and displays engine degradation, health score, and RUL trends over time.
8.  **Interactive Dashboard:** A Streamlit and Plotly-based dashboard for real-time monitoring and visualization of insights.

## System Architecture

The EngineSentinel system is designed with a modular architecture, where each component handles a specific aspect of the engine health monitoring process. The data flows sequentially from raw sensor data through processing, ML prediction, anomaly detection, explainable AI, health scoring, and risk assessment, culminating in an interactive dashboard for visualization and decision support.

-   **Data Processing Module**: Responsible for ingesting raw C-MAPSS turbofan engine data, performing necessary cleaning, normalization of sensor values, removal of irrelevant (constant) features, generation of RUL labels, and creation of time-series windows for sequential models.
-   **Machine Learning Prediction Module**: Houses various RUL prediction models (Random Forest, XGBoost, LSTM). It trains these models on processed data and provides predicted RUL values for engine cycles.
-   **Anomaly Detection Module**: Utilizes techniques like Isolation Forest to identify unusual patterns or abnormal behavior in sensor data, outputting an anomaly score.
-   **Explainable AI Module**: Employs SHAP (SHapley Additive Explanations) to provide insights into model predictions, highlighting the most influential sensors and explaining individual degradation forecasts.
-   **Engine Health Score Module**: Aggregates predicted RUL, anomaly scores, and (simulated) sensor degradation trends to compute a comprehensive Engine Health Index (0-100).
-   **Risk Assessment Module**: Translates the Engine Health Index into operational risk levels (Green, Yellow, Red) and generates actionable advisory messages.
-   **Digital Twin Simulation Layer**: A core component that simulates the real-time health state of an engine using a stream of sensor data. It integrates outputs from the prediction, anomaly, health score, and risk modules to provide a dynamic view of engine degradation over time.
-   **Interactive Dashboard**: Built with Streamlit and Plotly, this module serves as the user interface, displaying all critical insights: current health score, predicted RUL, sensor importance, risk level, and historical trends for various metrics.

## Setup

To set up and run the EngineSentinel project locally, follow these steps:

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository-url>
    cd ZeroPing/EngineSentinel
    ```

2.  **Install Python dependencies:**
    Ensure you have Python 3.8+ installed. Then install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Dataset:**
    The NASA C-MAPSS turbofan engine degradation dataset is expected to be in the `Dataset/` directory relative to the project root. Please ensure the dataset files (`train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`) are present in this directory.

## Usage

Follow these steps to train the models and run the interactive dashboard:

1.  **Train Machine Learning Models:**
    First, you need to train the RUL prediction and anomaly detection models. This script will process the data, train the models, evaluate them, and save the trained models to `./EngineSentinel/ml_prediction/models/FD00x/` and `./EngineSentinel/anomaly_detection/` respectively.
    Navigate to the project root directory and run:
    ```bash
    python -m EngineSentinel.ml_prediction.trainer
    ```
    By default, this trains models for FD001. You can modify `trainer.py` to train for other datasets if needed.

2.  **Run the Interactive Dashboard:**
    Once the models are trained, you can launch the Streamlit dashboard to visualize the engine health monitoring system.
    Navigate to the project root directory and run:
    ```bash
    streamlit run EngineSentinel/dashboard/dashboard_app.py
    ```
    The dashboard will open in your web browser, allowing you to select an engine and observe its simulated health state, RUL predictions, anomaly scores, and SHAP explanations.

