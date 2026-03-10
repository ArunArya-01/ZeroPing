import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

def train_isolation_forest(X_train, contamination='auto', random_state=42):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    print("Isolation Forest model training complete.")
    return model

def predict_anomaly_score_if(model, X_data):
    print("Predicting anomaly scores with Isolation Forest...")

    anomaly_scores = -model.decision_function(X_data)

    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    if max_score == min_score:
        normalized_scores = np.zeros_like(anomaly_scores)
    else:
        normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)

    return normalized_scores

def save_model_if(model, filepath):
 
    joblib.dump(model, filepath)
    print(f"Isolation Forest model saved to {filepath}")

def load_model_if(filepath):
 
    model = joblib.load(filepath)
    print(f"Isolation Forest model loaded from {filepath}")
    return model

if __name__ == "__main__":
  
    np.random.seed(42)
    X_normal = np.random.randn(1000, 5) * 0.5 + 5 # Mean 5, std 0.5
    # Generate some outliers
    X_outliers = np.random.randn(50, 5) * 2 + 15 # Mean 15, std 2

    X_train_dummy = pd.DataFrame(np.vstack([X_normal, X_outliers]))

    # Train Isolation Forest
    if_model = train_isolation_forest(X_train_dummy, contamination=0.05) # Assume 5% outliers

    # Predict anomaly scores on some test data (can be similar to train or new data)
    X_test_normal = np.random.randn(100, 5) * 0.5 + 5
    X_test_outlier = np.random.randn(10, 5) * 2 + 15
    X_test_dummy = pd.DataFrame(np.vstack([X_test_normal, X_test_outlier]))

    anomaly_scores = predict_anomaly_score_if(if_model, X_test_dummy)

    print("\nAnomaly scores (first 10, should be low for normal data):", anomaly_scores[:10])
    print("Anomaly scores (last 10, should be high for outlier data):", anomaly_scores[-10:])

    # Save and load example
    model_path = "./isolation_forest_model.joblib"
    save_model_if(if_model, model_path)
    loaded_model = load_model_if(model_path)

    # Clean up dummy model file
    import os
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up {model_path}")
