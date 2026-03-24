import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from joblib import dump, load
import os

# --- Isolation Forest ---
def train_isolation_forest(X_train, contamination='auto', random_state=42):
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    
    # Added from your version (important for normalization)
    model.train_scores_ = -model.decision_function(X_train)
    
    print("Isolation Forest training complete.")
    return model


# ✅ From MAIN VERSION (IMPORTANT - your app needs this)
def predict_anomaly_score_if(model, X_data):
    anomaly_scores = -model.decision_function(X_data)

    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()

    if max_score == min_score:
        normalized_scores = np.zeros_like(anomaly_scores)
    else:
        normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)

    return normalized_scores


# --- Requirement #1: New Algorithms ---
def train_ocsvm(X_train, kernel='rbf', nu=0.05):
    print("Training One-Class SVM...")
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(X_train)
    model.train_scores_ = -model.decision_function(X_train)
    return model


def train_lof(X_train, n_neighbors=20, contamination=0.05):
    print("Training Local Outlier Factor...")
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

    reconstructed = autoencoder.predict(X_train, verbose=0)
    autoencoder.train_scores_ = np.mean(np.power(X_train - reconstructed, 2), axis=1)

    return autoencoder


# --- Ensemble Logic ---
def get_ensemble_anomaly_score(models, X_data, weights={'if': 0.25, 'ocsvm': 0.25, 'lof': 0.25, 'ae': 0.25}):
    s_if = -models['if'].decision_function(X_data)
    s_ocsvm = -models['ocsvm'].decision_function(X_data)
    s_lof = -models['lof'].score_samples(X_data)

    reconstructed = models['ae'].predict(X_data, verbose=0)
    s_ae = np.mean(np.power(X_data - reconstructed, 2), axis=1)

    ns_if = _global_normalize(s_if, models['if'].train_scores_)
    ns_ocsvm = _global_normalize(s_ocsvm, models['ocsvm'].train_scores_)
    ns_lof = _global_normalize(s_lof, models['lof'].train_scores_)
    ns_ae = _global_normalize(s_ae, models['ae'].train_scores_)

    ensemble_score = (
        weights['if'] * ns_if +
        weights['ocsvm'] * ns_ocsvm +
        weights['lof'] * ns_lof +
        weights['ae'] * ns_ae
    )

    return ensemble_score


# --- Dynamic Threshold ---
def calculate_dynamic_threshold(score_history, sensitivity=2.5):
    if len(score_history) < 5:
        return 0.5
    return np.mean(score_history) + (sensitivity * np.std(score_history))


# --- Save/Load (Merged correctly) ---
def save_model_if(model, filepath):
    dump(model, filepath)
    print(f"Isolation Forest model saved to {filepath}")


def load_model_if(filepath):
    model = load(filepath)
    print(f"Isolation Forest model loaded from {filepath}")
    return model


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