import numpy as np
import os
import sys
import pandas as pd
import mlflow

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_pipeline import run_data_pipeline
from ml_prediction.random_forest_model import train_random_forest_model, predict_rul_rf, save_model_rf
from ml_prediction.xgboost_model import train_xgboost_model, predict_rul_xgb, save_model_xgb
from ml_prediction.lstm_model import train_lstm_model, predict_rul_lstm, save_model_lstm
from ml_prediction.evaluator import evaluate_model, get_ensemble_predictions, detect_data_drift

def log_metrics_to_last_run(rmse, mae):
    """Helper to push test metrics to the MLflow run that just finished."""
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)

def train_and_evaluate_all_models(dataset_number=1, sequence_length=50, data_path="./Dataset", tune=True):
    print(f"\n{'='*60}")
    print(f"🚀 ORCHESTRATING PIPELINE FOR: SkyNex Aero Ltd")
    print(f"Project: The Zero-Hardware Aviation Safety Stack")
    print(f"{'='*60}")

    # 1. DATA INGESTION
    (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    ) = run_data_pipeline(dataset_number, sequence_length, data_path)

    model_dir = f"./ml_prediction/models/FD00{dataset_number}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. TRAIN, PREDICT & LOG INDIVIDUAL MODELS
    
    # --- Random Forest ---
    rf_model = train_random_forest_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    rf_preds = predict_rul_rf(rf_model, X_test_non_seq)
    rf_rmse, rf_mae = evaluate_model(y_test_non_seq, rf_preds, "Random Forest")
    log_metrics_to_last_run(rf_rmse, rf_mae)

    # --- XGBoost ---
    xgb_model = train_xgboost_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    xgb_preds = predict_rul_xgb(xgb_model, X_test_non_seq)
    xgb_rmse, xgb_mae = evaluate_model(y_test_non_seq, xgb_preds, "XGBoost")
    log_metrics_to_last_run(xgb_rmse, xgb_mae)

    # --- LSTM ---
    lstm_model = train_lstm_model(X_train_seq, y_train_seq, tune_hyperparameters=tune, n_trials=3)
    lstm_preds = predict_rul_lstm(lstm_model, X_test_seq)
    lstm_rmse, lstm_mae = evaluate_model(y_test_seq, lstm_preds, "LSTM")
    log_metrics_to_last_run(lstm_rmse, lstm_mae)

    # 3. ALIGNMENT & ENSEMBLE
    print(f"\n[STEP 4/5] Computing Final Ensemble...")
    X_test_aligned_2d = X_test_seq[:, -1, :] 
    
    # Re-predict on aligned data for the ensemble
    rf_final_preds = predict_rul_rf(rf_model, X_test_aligned_2d)
    xgb_final_preds = predict_rul_xgb(xgb_model, X_test_aligned_2d)
    
    ensemble_preds = get_ensemble_predictions(rf_final_preds, xgb_final_preds, lstm_preds, weights=[0.2, 0.3, 0.5])
    
    print("\n" + "-"*30)
    e_rmse, e_mae = evaluate_model(y_test_seq, ensemble_preds, "FINAL ENSEMBLE MODEL")
    print("-" * 30)

    # Log Ensemble results to a dedicated MLflow experiment
    mlflow.set_experiment("Aviation_Ensemble_Results")
    with mlflow.start_run(run_name=f"Ensemble_FD00{dataset_number}"):
        mlflow.log_params({"rf_weight": 0.2, "xgb_weight": 0.3, "lstm_weight": 0.5})
        mlflow.log_metrics({"ensemble_rmse": e_rmse, "ensemble_mae": e_mae})

    # 4. MONITORING & SAVING
    detect_data_drift(X_train_non_seq.values, X_test_non_seq.values)
    
    save_model_rf(rf_model, os.path.join(model_dir, "random_forest_model.joblib"))
    save_model_xgb(xgb_model, os.path.join(model_dir, "xgboost_model.joblib"))
    save_model_lstm(lstm_model, os.path.join(model_dir, "lstm_model.keras"))

    print(f"\n✅ SUCCESS: Results logged to MLflow for SkyNex Aero Ltd.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    train_and_evaluate_all_models(dataset_number=1, sequence_length=50, tune=True)