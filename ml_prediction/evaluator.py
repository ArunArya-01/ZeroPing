import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp # For Drift Detection

# 1.1 Existing Metric Functions
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# NEW: Section 1.2 - Requirement #5 (Model Ensemble)
def get_ensemble_predictions(rf_preds, xgb_preds, lstm_preds, weights=[0.2, 0.3, 0.5]):
    """
    Combines predictions from multiple models using a weighted average.
    Weights give more importance to the LSTM (usually better for sequences).
    """
    print("Calculating Ensemble Predictions (RF + XGBoost + LSTM)...")
    
    # Ensure all predictions are numpy arrays
    rf_preds = np.array(rf_preds)
    xgb_preds = np.array(xgb_preds)
    lstm_preds = np.array(lstm_preds)
    
    # Weighted average: (W1*P1 + W2*P2 + W3*P3)
    ensemble_pred = (weights[0] * rf_preds) + (weights[1] * xgb_preds) + (weights[2] * lstm_preds)
    return ensemble_pred

# NEW: Section 1.2 - Requirement #6 (Monitoring for Drift)
def detect_data_drift(train_data, current_data, threshold=0.05):
    """
    Uses the KS-Test to check if the new data distribution matches the training data.
    If p-value < threshold, drift is detected.
    """
    print("\n--- Running Model Monitoring: Drift Detection ---")
    drift_detected = False
    
    # In a real scenario, we check each sensor column
    # For dummy testing, we'll check the first column
    p_value = ks_2samp(train_data.flatten(), current_data.flatten()).pvalue
    
    if p_value < threshold:
        print(f"⚠️ ALERT: Data Drift Detected! (p-value: {p_value:.4f})")
        drift_detected = True
    else:
        print(f"✅ Data Stable. No significant drift detected. (p-value: {p_value:.4f})")
        
    return drift_detected

def monitor_performance_degradation(current_rmse, baseline_rmse=50.0):
    """Checks if the model performance has dropped significantly."""
    if current_rmse > baseline_rmse * 1.5:
        print(f"⚠️ ALERT: Performance Degradation! Current RMSE ({current_rmse:.2f}) is 50% higher than baseline.")
        return True
    return False

# UPDATED: Unified Evaluation Function
def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    print(f"\n--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Check for degradation
    monitor_performance_degradation(rmse)
    
    return rmse, mae

if __name__ == "__main__":
    # 1. Dummy data for Ensemble test
    y_true = np.array([100, 80, 60, 40])
    rf_p = np.array([95, 85, 55, 45])
    xgb_p = np.array([98, 82, 58, 42])
    lstm_p = np.array([101, 79, 61, 39])

    # Test Ensemble
    ensemble_p = get_ensemble_predictions(rf_p, xgb_p, lstm_p)
    evaluate_model(y_true, ensemble_p, "Ensemble Super-Model")

    # 2. Test Drift Detection
    train_dist = np.random.normal(0, 1, 1000)
    stable_new_data = np.random.normal(0, 1, 100) # Similar to train
    drifted_new_data = np.random.normal(5, 2, 100) # Very different (Drift!)

    print("\nChecking Stable Data:")
    detect_data_drift(train_dist, stable_new_data)

    print("\nChecking Drifting Data:")
    detect_data_drift(train_dist, drifted_new_data)