import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

# NEW: Section 1.2 - Optuna Objective using Proper Time-Series Cross Validation
def optimize_rf(trial, X_train, y_train):
    """Optuna objective function to find the best hyperparameters."""
    # 1. Define the search space
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    # 2. Proper Time-Series Cross Validation (Prevents data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    
    # Convert to numpy arrays for reliable indexing
    X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

    for train_idx, val_idx in tscv.split(X_arr):
        X_cv_train, X_cv_val = X_arr[train_idx], X_arr[val_idx]
        y_cv_train, y_cv_val = y_arr[train_idx], y_arr[val_idx]
        
        model.fit(X_cv_train, y_cv_train)
        preds = model.predict(X_cv_val)
        rmse = np.sqrt(mean_squared_error(y_cv_val, preds))
        cv_scores.append(rmse)
        
    return np.mean(cv_scores)

# UPDATED: Section 1.2 - Implemented Model Versioning & Tuning
def train_random_forest_model(X_train, y_train, tune_hyperparameters=True, n_trials=5, run_name="RF_Baseline"):
    """Trains the RF model with optional Optuna tuning and mandatory MLflow tracking."""
    print(f"\n--- Starting Random Forest Training ({'With Tuning' if tune_hyperparameters else 'Default'} ) ---")
    
    # Set up MLflow Experiment Tracking
    mlflow.set_experiment("RUL_Prediction_RandomForest")
    
    with mlflow.start_run(run_name=run_name):
        if tune_hyperparameters:
            print("Running Optuna Hyperparameter Tuning (TimeSeries CV)...")
            # We minimize because the metric returned is RMSE
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: optimize_rf(trial, X_train, y_train), n_trials=n_trials)
            best_params = study.best_params
            print(f"Best Optuna Parameters Found: {best_params}")
        else:
            best_params = {'n_estimators': 100}
            print(f"Using Default Parameters: {best_params}")

        # Train final model on ALL training data using best parameters
        print("Training final Random Forest Regressor...")
        final_params = {**best_params, 'random_state': 42, 'n_jobs': -1}
        model = RandomForestRegressor(**final_params)
        model.fit(X_train, y_train)
        
        # Log parameters and the trained model natively to MLflow Registry
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print("Random Forest Regressor training complete. Logged to MLflow.")
        return model

# KEPT ORIGINAL LOGIC: Prediction and Saving/Loading
def predict_rul_rf(model, X_test):
    print("Making predictions with Random Forest Regressor...")
    predictions = model.predict(X_test)
    return predictions

def save_model_rf(model, filepath):
    dump(model, filepath)
    print(f"Random Forest model saved to {filepath}")

def load_model_rf(filepath):
    model = load(filepath)
    print(f"Random Forest model loaded from {filepath}")
    return model

if __name__ == "__main__":
    # Dummy data for demonstration
    X_train_dummy = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train_dummy = pd.Series(np.random.randint(0, 200, 100))
    X_test_dummy = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y_test_dummy = pd.Series(np.random.randint(0, 200, 50))

    # Train and evaluate model (Now includes Optuna tuning & MLflow)
    rf_model = train_random_forest_model(X_train_dummy, y_train_dummy, tune_hyperparameters=True, n_trials=3)
    predictions = predict_rul_rf(rf_model, X_test_dummy)

    print("\nExample predictions (first 5):", predictions[:5])

    rmse = np.sqrt(mean_squared_error(y_test_dummy, predictions))
    mae = mean_absolute_error(y_test_dummy, predictions)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Log metrics to the active MLflow run (re-opening the last run for the test data)
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)

    # Save and load example
    model_path = "./random_forest_model.joblib"
    save_model_rf(rf_model, model_path)
    loaded_model = load_model_rf(model_path)

    # Clean up dummy model file
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")