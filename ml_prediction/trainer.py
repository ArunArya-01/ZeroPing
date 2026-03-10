import numpy as np
from EngineSentinel.data_processing.data_pipeline import run_data_pipeline
from EngineSentinel.ml_prediction.random_forest_model import train_random_forest_model, predict_rul_rf, save_model_rf
from EngineSentinel.ml_prediction.xgboost_model import train_xgboost_model, predict_rul_xgb, save_model_xgb
from EngineSentinel.ml_prediction.lstm_model import build_lstm_model, train_lstm_model, predict_rul_lstm, save_model_lstm
from EngineSentinel.ml_prediction.evaluator import evaluate_model

def train_and_evaluate_all_models(dataset_number=1, sequence_length=50, data_path="./Dataset"):
    print(f"\n--- Starting Model Training and Evaluation for FD00{dataset_number} ---")

    # 1. Run Data Pipeline
    (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    ) = run_data_pipeline(dataset_number, sequence_length, data_path)

    print(f"\nFeature columns used for training: {feature_cols}")

    # Define paths for saving models
    model_dir = f"./EngineSentinel/ml_prediction/models/FD00{dataset_number}"
    import os
    os.makedirs(model_dir, exist_ok=True)
    rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
    xgb_model_path = os.path.join(model_dir, "xgboost_model.joblib")
    lstm_model_path = os.path.join(model_dir, "lstm_model.h5")

    # 2. Train and Evaluate Random Forest Model
    print("\n--- Random Forest Model ---")
    rf_model = train_random_forest_model(X_train_non_seq, y_train_non_seq)
    rf_predictions = predict_rul_rf(rf_model, X_test_non_seq)
    evaluate_model(y_test_non_seq, rf_predictions, "Random Forest")
    save_model_rf(rf_model, rf_model_path)

    # 3. Train and Evaluate XGBoost Model
    print("\n--- XGBoost Model ---")
    xgb_model = train_xgboost_model(X_train_non_seq, y_train_non_seq)
    xgb_predictions = predict_rul_xgb(xgb_model, X_test_non_seq)
    evaluate_model(y_test_non_seq, xgb_predictions, "XGBoost")
    save_model_xgb(xgb_model, xgb_model_path)

    # 4. Train and Evaluate LSTM Model
    print("\n--- LSTM Model ---")
    # Ensure X_train_seq and X_test_seq are not None before proceeding
    if X_train_seq is not None and y_train_seq is not None and X_test_seq is not None and y_test_seq is not None:
        lstm_model = build_lstm_model(input_shape=(sequence_length, len(feature_cols)))
        train_lstm_model(lstm_model, X_train_seq, y_train_seq, epochs=50, batch_size=256, model_save_path=lstm_model_path)
        lstm_predictions = predict_rul_lstm(lstm_model, X_test_seq)
        evaluate_model(y_test_seq, lstm_predictions, "LSTM")
    else:
        print("Skipping LSTM training: Sequential data not properly generated.")

    print(f"\n--- Model Training and Evaluation for FD00{dataset_number} Completed ---")

if __name__ == "__main__":
    # You can change the dataset number and sequence length here
    train_and_evaluate_all_models(dataset_number=1, sequence_length=50)
