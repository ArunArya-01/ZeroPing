import numpy as np
import os
import sys
import pandas as pd
import mlflow

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing modules
from data_processing.data_pipeline import run_data_pipeline
from ml_prediction.random_forest_model import train_random_forest_model, predict_rul_rf, save_model_rf
from ml_prediction.xgboost_model import train_xgboost_model, predict_rul_xgb, save_model_xgb
from ml_prediction.lstm_model import train_lstm_model, predict_rul_lstm, save_model_lstm
from ml_prediction.evaluator import evaluate_model, get_ensemble_predictions, detect_data_drift

# REQUIREMENT #1: Import the new Anomaly Detection Suite
from anomaly_detection.isolation_forest_detector import (
    train_isolation_forest, train_ocsvm, train_lof, train_autoencoder, 
    get_ensemble_anomaly_score, save_anomaly_suite
)

def log_metrics_to_last_run(rmse, mae):
    """Helper to push test metrics to the MLflow run that just finished."""
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)

def train_and_evaluate_all_models(dataset_number=1, sequence_length=50, data_path="./Dataset", tune=True):
    print(f"\n{'='*60}")
    print(f"🚀 ORCHESTRATING FULL STACK FOR: EngineSentinel")
    print(f"Status: Integrating Prediction & Anomaly Modules")
    print(f"{'='*60}")

    # 1. DATA INGESTION (NASA C-MAPSS FD00{dataset_number})
    (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    ) = run_data_pipeline(dataset_number, sequence_length, data_path)

    # Directories for results
    model_dir = f"./ml_prediction/models/FD00{dataset_number}"
    anomaly_dir = f"./anomaly_detection/models/FD00{dataset_number}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(anomaly_dir, exist_ok=True)
    
    # 2. MODULE 1.2: RUL PREDICTION (Regression)
    print("\n--- PHASE 1: Training Prediction Models ---")
    
    # Random Forest
    rf_model = train_random_forest_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    
    # XGBoost
    xgb_model = train_xgboost_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    
    # LSTM
    lstm_model = train_lstm_model(X_train_seq, y_train_seq, tune_hyperparameters=tune, n_trials=3)

    # 3. MODULE 1.3: ANOMALY DETECTION (Requirement #1 & #4)
    print("\n--- PHASE 2: Training Anomaly Detection Suite ---")
    
    # We use MLflow to track the anomaly training too
    mlflow.set_experiment("Engine_Anomaly_Detection")
    with mlflow.start_run(run_name=f"Anomaly_Suite_FD00{dataset_number}"):
        anomaly_suite = {
            'if': train_isolation_forest(X_train_non_seq),
            'ocsvm': train_ocsvm(X_train_non_seq),
            'lof': train_lof(X_train_non_seq),
            'ae': train_autoencoder(X_train_non_seq)
        }
        
        # Calculate a sample anomaly score for the first batch of test data
        sample_anom_scores = get_ensemble_anomaly_score(anomaly_suite, X_test_non_seq.iloc[:100])
        avg_anom = np.mean(sample_anom_scores)
        mlflow.log_metric("avg_test_anomaly_score", avg_anom)
        
        # Save the full suite (Requirement #7)
        save_anomaly_suite(anomaly_suite, anomaly_dir)

    # 4. FINAL ENSEMBLE & ALIGNMENT
    print(f"\n--- PHASE 3: Computing Final Predictions & Metrics ---")
    X_test_aligned_2d = X_test_seq[:, -1, :] 
    
    rf_preds = predict_rul_rf(rf_model, X_test_aligned_2d)
    xgb_preds = predict_rul_xgb(xgb_model, X_test_aligned_2d)
    lstm_preds = predict_rul_lstm(lstm_model, X_test_seq)
    
    # Log individual test metrics to MLflow
    # Note: These helper calls find the most recent 'finished' model run
    # (LSTM is last, so we log its specific metric there)
    evaluate_model(y_test_seq, lstm_preds, "LSTM")
    log_metrics_to_last_run(np.sqrt(np.mean((y_test_seq - lstm_preds)**2)), np.mean(np.abs(y_test_seq - lstm_preds)))

    # Compute Ensemble Prediction (Requirement #5)
    ensemble_preds = get_ensemble_predictions(rf_preds, xgb_preds, lstm_preds, weights=[0.2, 0.3, 0.5])
    
    mlflow.set_experiment("EngineSentinel_Final_Results")
    with mlflow.start_run(run_name=f"Full_Stack_FD00{dataset_number}"):
        e_rmse, e_mae = evaluate_model(y_test_seq, ensemble_preds, "FINAL ENSEMBLE")
        mlflow.log_metrics({"final_rmse": e_rmse, "final_mae": e_mae})

    # 5. MONITORING
    detect_data_drift(X_train_non_seq.values, X_test_non_seq.values)
    
    # Physical saving of predictors
    save_model_rf(rf_model, os.path.join(model_dir, "random_forest_model.joblib"))
    save_model_xgb(xgb_model, os.path.join(model_dir, "xgboost_model.joblib"))
    save_model_lstm(lstm_model, os.path.join(model_dir, "lstm_model.keras"))

    print(f"\n✅ PIPELINE SUCCESS: Prediction & Anomaly Modules are ready for EngineSentinel.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    train_and_evaluate_all_models(dataset_number=1, sequence_length=50, tune=True)