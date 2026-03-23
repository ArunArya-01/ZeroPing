import numpy as np
import os
import sys
import pandas as pd

# Add project root to Python path to ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project Imports from your ZeroPing structure
from data_processing.data_pipeline import run_data_pipeline
from ml_prediction.random_forest_model import train_random_forest_model, predict_rul_rf, save_model_rf
from ml_prediction.xgboost_model import train_xgboost_model, predict_rul_xgb, save_model_xgb
from ml_prediction.lstm_model import train_lstm_model, predict_rul_lstm, save_model_lstm
from ml_prediction.evaluator import evaluate_model, get_ensemble_predictions, detect_data_drift

def train_and_evaluate_all_models(dataset_number=1, sequence_length=50, data_path="./Dataset", tune=True):
    """
    Orchestrates the full ML pipeline: Ingestion -> Tuning -> Training -> Ensembling -> Monitoring.
    """
    print(f"\n{'='*60}")
    print(f"🚀 STARTING END-TO-END PIPELINE: FD00{dataset_number}")
    print(f"{'='*60}")

    # 1. DATA INGESTION (Uses logic from Section 1.1)
    (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    ) = run_data_pipeline(dataset_number, sequence_length, data_path)

    # Setup local model storage path
    model_dir = f"./ml_prediction/models/FD00{dataset_number}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. TRAIN & TUNE INDIVIDUAL MODELS (Section 1.2 Requirements #1, #2, #3)
    # These functions automatically log to MLflow and use Optuna tuning
    
    print("\n[STEP 1/5] Training Tuned Random Forest...")
    rf_model = train_random_forest_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    
    print("\n[STEP 2/5] Training Tuned XGBoost...")
    xgb_model = train_xgboost_model(X_train_non_seq, y_train_non_seq, tune_hyperparameters=tune, n_trials=5)
    
    print("\n[STEP 3/5] Training Tuned LSTM (Deep Learning)...")
    lstm_model = train_lstm_model(X_train_seq, y_train_seq, tune_hyperparameters=tune, n_trials=3)

    # 3. ALIGNMENT STEP (Fixes shape mismatch error)
    # LSTM data is shorter because it requires a 50-step 'window'. 
    # We flatten the LSTM test data to 2D by taking only the 'last' timestep.
    print(f"\n[STEP 4/5] Aligning model shapes for Ensemble...")
    X_test_aligned_2d = X_test_seq[:, -1, :] 
    
    rf_preds = predict_rul_rf(rf_model, X_test_aligned_2d)
    xgb_preds = predict_rul_xgb(xgb_model, X_test_aligned_2d)
    lstm_preds = predict_rul_lstm(lstm_model, X_test_seq)
    
    # Verification of shapes
    print(f"-> Prediction Shapes: RF:{rf_preds.shape}, XGB:{xgb_preds.shape}, LSTM:{lstm_preds.shape}")

    # 4. ENSEMBLE (Section 1.2 Requirement #5)
    # Weighted average: We trust the LSTM more for time-series aviation data.
    ensemble_preds = get_ensemble_predictions(rf_preds, xgb_preds, lstm_preds, weights=[0.2, 0.3, 0.5])
    
    print("\n" + "-"*30)
    # Using y_test_seq as the ground truth as it matches the aligned prediction count
    evaluate_model(y_test_seq, ensemble_preds, "FINAL ENSEMBLE MODEL")
    print("-" * 30)

    # 5. MONITORING & SAVING (Section 1.2 Requirements #6, #7)
    detect_data_drift(X_train_non_seq.values, X_test_non_seq.values)
    
    # Save physical models locally as a backup to the MLflow Registry
    save_model_rf(rf_model, os.path.join(model_dir, "random_forest_model.joblib"))
    save_model_xgb(xgb_model, os.path.join(model_dir, "xgboost_model.joblib"))
    save_model_lstm(lstm_model, os.path.join(model_dir, "lstm_model.keras"))

    print(f"\n✅ PIPELINE COMPLETE: All models versioned and ensemble verified.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Ensure you are in your ZeroPing root directory on your Mac
    train_and_evaluate_all_models(dataset_number=1, sequence_length=50, tune=True)