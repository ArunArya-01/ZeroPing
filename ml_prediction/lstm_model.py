import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
import mlflow
import mlflow.keras
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error 

# NEW: Section 1.2 - Optuna Objective for LSTM
def optimize_lstm(trial, X_train_seq, y_train_seq):
    """Optuna objective function to find the best LSTM hyperparameters."""
    units = trial.suggest_int('units', 50, 150, step=50)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))

    # FIX: Using full name 'mean_squared_error' for better serialization
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2, 
        shuffle=False, 
        epochs=5,     
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0      
    )

    val_loss = min(history.history['val_loss'])
    return val_loss

# UPDATED: Main Training Function with MLflow
def train_lstm_model(X_train_seq, y_train_seq, tune_hyperparameters=True, n_trials=3, run_name="LSTM_Baseline"):
    print(f"\n--- Starting LSTM Training ({'With Tuning' if tune_hyperparameters else 'Default'} ) ---")
    
    mlflow.set_experiment("RUL_Prediction_LSTM")
    
    with mlflow.start_run(run_name=run_name):
        if tune_hyperparameters:
            print("Running Optuna Hyperparameter Tuning for LSTM...")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: optimize_lstm(trial, X_train_seq, y_train_seq), n_trials=n_trials)
            best_params = study.best_params
            print(f"Best Optuna Parameters Found: {best_params}")
        else:
            best_params = {'units': 100, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 128}

        print("Training final LSTM model...")
        model = Sequential()
        model.add(LSTM(units=best_params['units'], return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
        model.add(Dropout(best_params['dropout_rate']))
        model.add(LSTM(units=best_params['units']))
        model.add(Dropout(best_params['dropout_rate']))
        model.add(Dense(units=1))
        
        optimizer = Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=20,
            batch_size=best_params['batch_size'],
            validation_split=0.2,
            shuffle=False, 
            callbacks=[early_stop],
            verbose=1
        )
        
        mlflow.log_params(best_params)
        mlflow.keras.log_model(model, "lstm_model")
        
        print("LSTM training complete. Logged to MLflow.")
        return model

def predict_rul_lstm(model, X_test_seq):
    print('Making predictions with LSTM model...')
    predictions = model.predict(X_test_seq)
    return predictions.flatten()

def save_model_lstm(model, filepath):
    model.save(filepath)
    print(f'LSTM model saved to {filepath}')

def load_model_lstm(filepath):
    # FIX: Added compile=False to avoid the deserialization error during simple prediction loads
    model = load_model(filepath, compile=False)
    print(f'LSTM model loaded from {filepath}')
    return model

if __name__ == "__main__":
    sequence_length = 10
    num_features = 5
    num_samples_train = 1000
    num_samples_test = 200

    X_train_dummy_seq = np.random.rand(num_samples_train, sequence_length, num_features)
    y_train_dummy_seq = np.random.randint(0, 200, num_samples_train)
    X_test_dummy_seq = np.random.rand(num_samples_test, sequence_length, num_features)
    y_test_dummy_seq = np.random.randint(0, 200, num_samples_test)

    # Train, Tune, and Log
    lstm_model = train_lstm_model(X_train_dummy_seq, y_train_dummy_seq, tune_hyperparameters=True, n_trials=2)

    predictions = predict_rul_lstm(lstm_model, X_test_dummy_seq)

    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(y_test_dummy_seq, predictions))
    mae = mean_absolute_error(y_test_dummy_seq, predictions)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")

    # Log metrics to MLflow
    last_run = mlflow.last_active_run()
    if last_run:
        with mlflow.start_run(run_id=last_run.info.run_id):
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)

    # FIX: Changed to .keras extension for modern compatibility
    model_save_path = "best_lstm_model.keras"
    save_model_lstm(lstm_model, model_save_path)
    loaded_lstm_model = load_model_lstm(model_save_path)

    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"Cleaned up {model_save_path}")