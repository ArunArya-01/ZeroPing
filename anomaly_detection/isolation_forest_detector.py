import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from joblib import dump, load
import os

# --- Isolation Forest ---
def train_isolation_forest(X_train, contamination='auto', random_state=42):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    # Store training scores range for global normalization
    model.train_scores_ = -model.decision_function(X_train)
    print("Isolation Forest training complete.")
    return model

# --- Requirement #1: New Algorithms ---
def train_ocsvm(X_train, kernel='rbf', nu=0.05):
    print("Training One-Class SVM...")
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(X_train)
    model.train_scores_ = -model.decision_function(X_train)
    return model

def train_lof(X_train, n_neighbors=20, contamination=0.05):
    print("Training Local Outlier Factor...")
    # novelty=True allows predicting on new data
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True, n_jobs=-1)
    model.fit(X_train)
    model.train_scores_ = -model.score_samples(X_train)
    return model

def train_autoencoder(X_train, epochs=10, batch_size=32):
    print("Training Autoencoder...")
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(input_dim // 2, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Pre-calculate reconstruction error for scaling
    reconstructed = autoencoder.predict(X_train, verbose=0)
    autoencoder.train_scores_ = np.mean(np.power(X_train - reconstructed, 2), axis=1)
    return autoencoder

# --- Requirement #4: Ensemble Logic ---
def get_ensemble_anomaly_score(models, X_data, weights={'if': 0.25, 'ocsvm': 0.25, 'lof': 0.25, 'ae': 0.25}):
    s_if = -models['if'].decision_function(X_data)
    s_ocsvm = -models['ocsvm'].decision_function(X_data)
    s_lof = -models['lof'].score_samples(X_data)
    
    reconstructed = models['ae'].predict(X_data, verbose=0)
    s_ae = np.mean(np.power(X_data - reconstructed, 2), axis=1)
    
    # Global Normalization relative to training data
    ns_if = _global_normalize(s_if, models['if'].train_scores_)
    ns_ocsvm = _global_normalize(s_ocsvm, models['ocsvm'].train_scores_)
    ns_lof = _global_normalize(s_lof, models['lof'].train_scores_)
    ns_ae = _global_normalize(s_ae, models['ae'].train_scores_)
    
    ensemble_score = (weights['if'] * ns_if + 
                      weights['ocsvm'] * ns_ocsvm + 
                      weights['lof'] * ns_lof + 
                      weights['ae'] * ns_ae)
    return ensemble_score

# --- Requirement #3: Dynamic Thresholds ---
def calculate_dynamic_threshold(score_history, sensitivity=2.5):
    if len(score_history) < 5:
        return 0.5 
    return np.mean(score_history) + (sensitivity * np.std(score_history))

# --- THE FIX: save_anomaly_suite utility ---
def save_anomaly_suite(models, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    dump(models['if'], os.path.join(folder_path, "if_model.joblib"))
    dump(models['ocsvm'], os.path.join(folder_path, "ocsvm_model.joblib"))
    dump(models['lof'], os.path.join(folder_path, "lof_model.joblib"))
    models['ae'].save(os.path.join(folder_path, "ae_model.keras"))
    print(f"✅ Anomaly Detection Suite saved to: {folder_path}")

# --- Helpers ---
def _global_normalize(current_score, train_scores):
    min_s, max_s = train_scores.min(), train_scores.max()
    return (current_score - min_s) / (max_s - min_s + 1e-6)

if __name__ == "__main__":
    # Test block
    X_train_dummy = pd.DataFrame(np.random.randn(500, 10))
    suite = {
        'if': train_isolation_forest(X_train_dummy),
        'ocsvm': train_ocsvm(X_train_dummy),
        'lof': train_lof(X_train_dummy),
        'ae': train_autoencoder(X_train_dummy)
    }
    
    print("\n--- Real-Time Stream Simulation ---")
    scores = []
    for i in range(10):
        new_data = np.random.randn(1, 10)
        if i == 7: new_data *= 5
        score = get_ensemble_anomaly_score(suite, new_data)[0]
        scores.append(score)
        thresh = calculate_dynamic_threshold(scores)
        status = "⚠️ ANOMALY" if score > thresh else "✅ NORMAL"
        print(f"Step {i}: Score={score:.3f} | Threshold={thresh:.3f} | {status}")