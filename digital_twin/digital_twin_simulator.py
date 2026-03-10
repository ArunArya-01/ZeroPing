import pandas as pd
import numpy as np
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_pipeline import run_data_pipeline
from ml_prediction.random_forest_model import load_model_rf, predict_rul_rf
from ml_prediction.xgboost_model import load_model_xgb, predict_rul_xgb
from anomaly_detection.isolation_forest_detector import load_model_if, predict_anomaly_score_if, save_model_if
from health_risk.health_calculator import compute_engine_health_index
from health_risk.risk_evaluator import assess_risk_level
import os

class DigitalTwinSimulator:
    def __init__(self, dataset_number=1, sequence_length=50, model_type: str = "random_forest"):
        self.dataset_number = dataset_number
        self.sequence_length = sequence_length
        self.model_type = model_type

        self.feature_cols = None
        self.scaler = None
        self.rul_prediction_model = None
        self.anomaly_detection_model = None

        # Load data and models
        self._load_data_and_models()

        self.engine_states = {}

    def _load_data_and_models(self):
        print(f"Loading data and models for FD00{self.dataset_number}...")
        # Load data using the pipeline
        (
            X_train_non_seq, y_train_non_seq,
            self.X_test_non_seq, self.y_test_non_seq,
            X_train_seq, y_train_seq,
            X_test_seq, y_test_seq,
            self.feature_cols, self.scaler
        ) = run_data_pipeline(self.dataset_number, self.sequence_length)

        # Also load raw test data for getting engine_ids and apply same preprocessing
        from data_processing.data_loader import load_cmapss_data
        from data_processing.preprocessor import remove_constant_sensors
        from data_processing.feature_engineering import generate_rul_labels
        
        train_df_raw, test_df_raw, rul_df = load_cmapss_data(self.dataset_number, f"./Dataset")
        
        # Get constant sensors from training to apply to test
        _, constant_sensors = remove_constant_sensors(train_df_raw.copy())
        test_df_processed = test_df_raw.drop(columns=constant_sensors, errors='ignore')
        
        # Normalize using the same scaler
        test_df_normalized = test_df_processed.copy()
        test_df_normalized[self.feature_cols] = self.scaler.transform(test_df_processed[self.feature_cols])
        
        # Calculate true RUL for each engine
        test_rul_list = []
        for i, engine_id in enumerate(test_df_normalized['engine_id'].unique()):
            true_rul = rul_df.iloc[i][0]
            engine_df = test_df_normalized[test_df_normalized['engine_id'] == engine_id].copy()
            max_time = engine_df['time_cycle'].max()
            engine_df['RUL'] = true_rul + (max_time - engine_df['time_cycle'])
            test_rul_list.append(engine_df)
        
        self.test_df_with_engine = pd.concat(test_rul_list, ignore_index=True)

        # Determine model paths
        model_dir = f"./ml_prediction/models/FD00{self.dataset_number}"
        rf_model_path = os.path.join(model_dir, "random_forest_model.joblib")
        xgb_model_path = os.path.join(model_dir, "xgboost_model.joblib")
        # lstm_model_path = os.path.join(model_dir, "lstm_model.h5") # Uncomment and implement if LSTM is to be used

        anomaly_model_path = f"./anomaly_detection/isolation_forest_model_FD00{self.dataset_number}.joblib"

        # Load RUL prediction model
        if self.model_type == 'random_forest':
            self.rul_prediction_model = load_model_rf(rf_model_path)
        elif self.model_type == 'xgboost':
            self.rul_prediction_model = load_model_xgb(xgb_model_path)
        elif self.model_type == 'lstm':
            from ml_prediction.lstm_model import load_model_lstm
            lstm_model_path = os.path.join(model_dir, "lstm_model.h5")
            self.rul_prediction_model = load_model_lstm(lstm_model_path)
        # else: # Add LSTM loading if needed

        # Load Anomaly Detection model
        # For anomaly detection, we train IF on the *training data features*.
        # Let's ensure a model is saved by `trainer.py` or train it here for demo purposes.
        # For now, we will create a dummy if not found.
        if not os.path.exists(anomaly_model_path):
            print("Anomaly detection model not found. Training a new one for simulation.")
            from anomaly_detection.isolation_forest_detector import train_isolation_forest
            self.anomaly_detection_model = train_isolation_forest(X_train_non_seq[self.feature_cols], contamination=0.01)
            save_model_if(self.anomaly_detection_model, anomaly_model_path)
        else:
            self.anomaly_detection_model = load_model_if(anomaly_model_path)

        print("Data and models loaded.")

    def simulate_engine_degradation(self, engine_id):
        print(f"Simulating degradation for engine ID: {engine_id}")

        # Use the normalized test data with engine_id and time_cycle
        engine_df = self.test_df_with_engine[self.test_df_with_engine["engine_id"] == engine_id].copy()
        
        results = []
        initial_sensor_values = engine_df[self.feature_cols].iloc[0]

        # Placeholder for sensor degradation thresholds (these would ideally be learned or defined)
        sensor_degradation_thresholds = {col: (0.1, 0.1) for col in self.feature_cols if "sensor_measure" in col}

        for i in range(len(engine_df)):
            current_cycle_data = engine_df.iloc[i]
            current_features = current_cycle_data[self.feature_cols].to_frame().T

            # Predict RUL
            print(f"Using model type: {self.model_type}")
            if self.model_type == 'random_forest':
                predicted_rul = predict_rul_rf(self.rul_prediction_model, current_features)[0]
            elif self.model_type == 'xgboost':
                predicted_rul = predict_rul_xgb(self.rul_prediction_model, current_features)[0]
            elif self.model_type == 'lstm':
                # LSTM requires sequence data - needs proper implementation
                # For now, show message that LSTM needs training
                print("Note: LSTM model needs to be trained separately")
                predicted_rul = 0  # Placeholder - would need sequence data
            else:
                predicted_rul = np.nan

            # Predict Anomaly Score
            anomaly_score = predict_anomaly_score_if(self.anomaly_detection_model, current_features)[0]

            # Compute Engine Health Index
            health_index = compute_engine_health_index(
                predicted_rul,
                anomaly_score,
                current_sensor_values=current_cycle_data[self.feature_cols],
                initial_sensor_values=initial_sensor_values,
                sensor_degradation_thresholds=sensor_degradation_thresholds
            )

            # Assess Risk Level
            risk_level, advisory_message = assess_risk_level(health_index)

            results.append({
                "engine_id": engine_id,
                "time_cycle": current_cycle_data["time_cycle"],
                "predicted_RUL": predicted_rul,
                "true_RUL": current_cycle_data["RUL"],
                "anomaly_score": anomaly_score,
                "health_index": health_index,
                "risk_level": risk_level,
                "advisory_message": advisory_message,
                **{f: current_cycle_data[f] for f in self.feature_cols} # Include raw features for plotting
            })

        results_df = pd.DataFrame(results)
        self.engine_states[engine_id] = results_df
        print(f"Simulation for engine ID {engine_id} complete.")
        return results_df

    def get_engine_state(self, engine_id):
        """
        Retrieves the simulated state for a given engine.
        """
        return self.engine_states.get(engine_id, None)

if __name__ == "__main__":
    simulator = DigitalTwinSimulator(dataset_number=1, model_type="random_forest")

    engine_id_to_simulate = 1
    simulated_data = simulator.simulate_engine_degradation(engine_id_to_simulate)

    print(f"\nSimulated data for Engine {engine_id_to_simulate}:\n")
    print(simulated_data.head())
    print(simulated_data.tail())
