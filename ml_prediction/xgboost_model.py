import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

def train_xgboost_model(X_train, y_train, **kwargs):
    print("Training XGBoost Regressor...")
    # Default parameters, can be overridden by kwargs
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    params = {**default_params, **kwargs}

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    print("XGBoost Regressor training complete.")
    return model

def predict_rul_xgb(model, X_test):
    print("Making predictions with XGBoost Regressor...")
    predictions = model.predict(X_test)
    return predictions

def save_model_xgb(model, filepath):
    dump(model, filepath)
    print(f"XGBoost model saved to {filepath}")

def load_model_xgb(filepath):
    model = load(filepath)
    print(f"XGBoost model loaded from {filepath}")
    return model

if __name__ == "__main__":
    # Dummy data for demonstration
    X_train_dummy = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train_dummy = pd.Series(np.random.randint(0, 200, 100))
    X_test_dummy = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y_test_dummy = pd.Series(np.random.randint(0, 200, 50))

    # Train and evaluate model
    xgb_model = train_xgboost_model(X_train_dummy, y_train_dummy)
    predictions = predict_rul_xgb(xgb_model, X_test_dummy)

    print("\nExample predictions (first 5):", predictions[:5])

    rmse = np.sqrt(mean_squared_error(y_test_dummy, predictions))
    mae = mean_absolute_error(y_test_dummy, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Save and load example
    model_path = "./xgboost_model.joblib"
    save_model_xgb(xgb_model, model_path)
    loaded_model = load_model_xgb(model_path)

    # Clean up dummy model file
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")
