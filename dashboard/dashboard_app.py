import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from our project
from data_processing.data_pipeline import run_data_pipeline
from ml_prediction.random_forest_model import load_model_rf
from ml_prediction.xgboost_model import load_model_xgb
# from ml_prediction.lstm_model import load_model_lstm # Uncomment if LSTM is fully implemented
from anomaly_detection.isolation_forest_detector import load_model_if
from explainable_ai.shap_explainer import initialize_explainer, compute_shap_values, get_feature_importance, explain_single_prediction
from health_risk.health_calculator import compute_engine_health_index
from health_risk.risk_evaluator import assess_risk_level
from digital_twin.digital_twin_simulator import DigitalTwinSimulator

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="EngineSentinel Dashboard")
st.title("EngineSentinel: Aircraft Engine Health Monitoring")
st.markdown("--- ")

# --- Global Variables and Model Loading (Cached) ---@
st.cache_resource(show_spinner=False)
def load_all_resources(dataset_number, sequence_length, model_type):
    # Initialize the simulator which loads data and models
    simulator = DigitalTwinSimulator(dataset_number=dataset_number, sequence_length=sequence_length, model_type=model_type)
    
    # For SHAP, we need a background dataset from the training data
    # The simulator already loads X_train_non_seq (from run_data_pipeline)
    # We need to expose it or reload a small part for SHAP explainer
    # (This is inefficient; in a real app, simulator should expose it more cleanly)
    (
        X_train_non_seq, _, _, _,
        _, _, _, _,
        feature_cols, _
    ) = run_data_pipeline(dataset_number, sequence_length)
    
    # For SHAP background, sample a smaller portion if training data is too large
    if X_train_non_seq.shape[0] > 1000:
        shap_background_data = X_train_non_seq[feature_cols].sample(n=500, random_state=42)
    else:
        shap_background_data = X_train_non_seq[feature_cols]

    # Initialize SHAP explainer for the RUL prediction model
    shap_explainer_obj = initialize_explainer(simulator.rul_prediction_model, shap_background_data)

    return simulator, shap_explainer_obj, feature_cols

# --- Sidebar for User Inputs ---
st.sidebar.header("Configuration")
dataset_choice = st.sidebar.selectbox("Select C-MAPSS Dataset", [1, 2, 3, 4], index=0)
model_choice = st.sidebar.selectbox("Select RUL Prediction Model", ["random_forest", "xgboost"], index=0)
sequence_length = st.sidebar.slider("Sequence Length (for LSTM, if implemented)", 10, 100, 50)

# Load resources based on selections
simulator, shap_explainer, feature_cols = load_all_resources(dataset_choice, sequence_length, model_choice)

# Get available engine IDs for the selected dataset
available_engine_ids = simulator.X_test_non_seq["engine_id"].unique()
selected_engine_id = st.sidebar.selectbox("Select Engine ID", available_engine_ids)

# --- Simulate and Get Engine Data ---
@st.cache_data
def get_simulated_data(engine_id, _simulator_instance):
    return _simulator_instance.simulate_engine_degradation(engine_id)

simulated_data = get_simulated_data(selected_engine_id, simulator)

# --- Main Dashboard Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Engine Health Score")
    latest_health_index = simulated_data["health_index"].iloc[-1]
    st.metric(label="Current Health Index", value=f"{latest_health_index:.2f}")

    latest_risk_level, latest_advisory = assess_risk_level(latest_health_index)
    
    color_map = {"Green": "#28a745", "Yellow": "#ffc107", "Red": "#dc3545"}
    risk_color = color_map.get(latest_risk_level, "#6c757d")
    st.markdown(f"### Risk Level: <span style=\"color:{risk_color}\">{latest_risk_level}</span>", unsafe_allow_html=True)
    st.info(latest_advisory)

with col2:
    st.subheader("Remaining Useful Life")
    latest_predicted_rul = simulated_data["predicted_RUL"].iloc[-1]
    latest_true_rul = simulated_data["true_RUL"].iloc[-1]
    st.metric(label="Predicted RUL (Cycles)", value=f"{latest_predicted_rul:.2f}")
    st.markdown(f"_True RUL (End of Test): {latest_true_rul:.0f} cycles_ ")

with col3:
    st.subheader("Anomaly Detection")
    latest_anomaly_score = simulated_data["anomaly_score"].iloc[-1]
    st.metric(label="Current Anomaly Score", value=f"{latest_anomaly_score:.4f}")
    if latest_anomaly_score > 0.5: # Simple threshold for visual alert
        st.warning("High anomaly detected!")

st.markdown("--- ")

# --- Trends Over Time ---
st.subheader(f"Trends for Engine ID: {selected_engine_id}")

# RUL Trend
fig_rul = px.line(
    simulated_data, x="time_cycle", y=["predicted_RUL", "true_RUL"],
    title="RUL Prediction Trend",
    labels={
        "time_cycle": "Time Cycle",
        "value": "Remaining Useful Life (Cycles)",
        "variable": "Type"
    },
    height=400
)
fig_rul.update_layout(hovermode="x unified")
st.plotly_chart(fig_rul, use_container_width=True)

# Health Index Trend
fig_health = px.line(
    simulated_data, x="time_cycle", y="health_index",
    title="Engine Health Index Trend (0-100)",
    labels={
        "time_cycle": "Time Cycle",
        "health_index": "Health Index"
    },
    height=400
)
fig_health.update_layout(hovermode="x unified")
fig_health.add_hline(y=80, line_dash="dot", annotation_text="Healthy Threshold (80)", annotation_position="top left")
fig_health.add_hline(y=50, line_dash="dot", annotation_text="Warning Threshold (50)", annotation_position="top left", line_color="orange")
st.plotly_chart(fig_health, use_container_width=True)

# Anomaly Score Trend
fig_anomaly = px.line(
    simulated_data, x="time_cycle", y="anomaly_score",
    title="Anomaly Score Trend (Higher is More Anomalous)",
    labels={
        "time_cycle": "Time Cycle",
        "anomaly_score": "Anomaly Score"
    },
    height=400
)
fig_anomaly.update_layout(hovermode="x unified")
st.plotly_chart(fig_anomaly, use_container_width=True)

st.markdown("--- ")

# --- SHAP Feature Importance ---
st.subheader("SHAP Feature Importance")
st.write("Understanding which sensors contribute most to the RUL prediction.")

# Get the last row of the simulated data for individual prediction explanation
latest_features = simulated_data[feature_cols].iloc[-1]

# Compute SHAP values for the last prediction
shap_values_latest = explain_single_prediction(shap_explainer, latest_features, feature_cols)

# Plot global feature importance
fig_feature_importance = px.bar(
    shap_values_latest.abs().sort_values(ascending=False),
    orientation="h",
    title="Overall Feature Importance (Mean Absolute SHAP)",
    labels={
        "value": "Mean Absolute SHAP Value",
        "index": "Feature"
    },
    height=500
)
fig_feature_importance.update_layout(showlegend=False, yaxis_autorange="reversed")
st.plotly_chart(fig_feature_importance, use_container_width=True)

# Display individual prediction explanation (force plot is not directly supported by Streamlit/Plotly)
st.write("**Individual Prediction Explanation (Latest Cycle):**")
st.dataframe(shap_values_latest.to_frame(name="SHAP Value"))

st.markdown("--- ")
st.info("To run this dashboard: \n1. Ensure all Python dependencies are installed (`pip install -r requirements.txt`).\n2. Run the `trainer.py` script once to train and save the models (`python -m ml_prediction.trainer`).\n3. Navigate to the project root directory in your terminal.\n4. Run `streamlit run dashboard/dashboard_app.py`.")
