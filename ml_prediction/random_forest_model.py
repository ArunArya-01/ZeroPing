import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

def train_random_forest_model(X_train, y_train, n_estimators=100, random_state=42):
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Random Forest Regressor training complete.")
    return model

def predict_rul_rf(model, X_test):
    print("Making predictions with Random Forest Regressor...")
    predictions = model.predict(X_test)
    return predictions

def save_model_rf(model, filepath):
    joblib.dump(model, filepath)
    print(f"Random Forest model saved to {filepath}")

def load_model_rf(filepath):
    model = joblib.load(filepath)
    print(f"Random Forest model loaded from {filepath}")
    return model

if __name__ == "__main__":
    # Dummy data for demonstration
    X_train_dummy = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(10)])
    y_train_dummy = pd.Series(np.random.randint(0, 200, 100))
    X_test_dummy = pd.DataFrame(np.random.rand(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y_test_dummy = pd.Series(np.random.randint(0, 200, 50))

    # Train and evaluate model
    rf_model = train_random_forest_model(X_train_dummy, y_train_dummy)
    predictions = predict_rul_rf(rf_model, X_test_dummy)

    print("\nExample predictions (first 5):", predictions[:5])

    rmse = np.sqrt(mean_squared_error(y_test_dummy, predictions))
    mae = mean_absolute_error(y_test_dummy, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Save and load example
    model_path = "./random_forest_model.joblib"
    save_model_rf(rf_model, model_path)
    loaded_model = load_model_rf(model_path)

    # Clean up dummy model file
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")
