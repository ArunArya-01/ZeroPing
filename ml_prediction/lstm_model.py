import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_lstm_model(input_shape, units=100, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    print('LSTM model built and compiled.')
    model.summary()
    return model

def train_lstm_model(model, X_train_seq, y_train_seq, epochs=50, batch_size=256, validation_split=0.2, patience=10, model_save_path=None):
    print('Training LSTM model...')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=1),
    ]
    if model_save_path:
        callbacks.append(ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss', mode='min'))

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2
    )
    print('LSTM model training complete.')
    return history

def predict_rul_lstm(model, X_test_seq):
    print('Making predictions with LSTM model...')
    predictions = model.predict(X_test_seq)
    return predictions.flatten()

def save_model_lstm(model, filepath):
    model.save(filepath)
    print(f'LSTM model saved to {filepath}')

def load_model_lstm(filepath):
    model = load_model(filepath)
    print(f'LSTM model loaded from {filepath}')
    return model

if __name__ == "__main__":
    # Dummy data for demonstration
    sequence_length = 10
    num_features = 5
    num_samples_train = 1000
    num_samples_test = 200

    X_train_dummy_seq = np.random.rand(num_samples_train, sequence_length, num_features)
    y_train_dummy_seq = np.random.randint(0, 200, num_samples_train)
    X_test_dummy_seq = np.random.rand(num_samples_test, sequence_length, num_features)
    y_test_dummy_seq = np.random.randint(0, 200, num_samples_test)

    # Build, train, and evaluate LSTM model
    lstm_model = build_lstm_model(input_shape=(sequence_length, num_features))
    
    # Define a path for saving the best model during training
    model_save_path = "best_lstm_model.h5"
    
    train_lstm_model(lstm_model, X_train_dummy_seq, y_train_dummy_seq, epochs=5, model_save_path=model_save_path)

    predictions = predict_rul_lstm(lstm_model, X_test_dummy_seq)

    print("\nExample predictions (first 5):", predictions[:5])

    rmse = np.sqrt(tf.keras.metrics.mean_squared_error(y_test_dummy_seq, predictions).numpy())
    mae = tf.keras.metrics.mean_absolute_error(y_test_dummy_seq, predictions).numpy()
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Load example
    loaded_lstm_model = load_model_lstm(model_save_path)
    print("Loaded model predictions (first 5):", predict_rul_lstm(loaded_lstm_model, X_test_dummy_seq)[:5])

    # Clean up dummy model file
    import os
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"Cleaned up {model_save_path}")
