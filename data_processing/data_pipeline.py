import pandas as pd
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import load_cmapss_data
from data_processing.preprocessor import remove_constant_sensors, normalize_data
from data_processing.feature_engineering import generate_rul_labels, create_sequences

def run_data_pipeline(dataset_number, sequence_length=50, data_path="./Dataset"):

    print(f"--- Starting Data Pipeline for FD00{dataset_number} ---")

    # 1. Load Data
    train_df, test_df, rul_df = load_cmapss_data(dataset_number, data_path)

    # 2. Generate RUL labels for training data
    train_df = generate_rul_labels(train_df)

    # Identify sensor columns before any removal
    all_sensor_cols = [col for col in train_df.columns if 'sensor_measure' in col]

    # 3. Remove Constant Sensors (fit on train, transform on both)
    train_df_processed, constant_sensors = remove_constant_sensors(train_df.copy())
    test_df_processed = test_df.drop(columns=constant_sensors, errors='ignore')

    active_sensor_cols = [col for col in all_sensor_cols if col not in constant_sensors]
    op_setting_cols = [col for col in train_df.columns if 'op_setting' in col]
    feature_cols = op_setting_cols + active_sensor_cols

    # 4. Normalize Data
    train_df_normalized, test_df_normalized, scaler = normalize_data(
        train_df_processed.copy(), test_df_processed.copy(), feature_cols
    )

    # For non-sequential models, X_train and y_train are simply the features and RUL
    X_train_non_seq = train_df_normalized[feature_cols]
    y_train_non_seq = train_df_normalized["RUL"]

    # For test data, align RUL correctly for non-sequential models
    # The rul_df provides RUL for the *last* cycle of each engine in test_df
    test_rul_aligned = pd.DataFrame(test_df_normalized.groupby('engine_id')['time_cycle'].max()).reset_index()
    test_rul_aligned = test_rul_aligned.merge(rul_df, left_index=True, right_index=True)
    test_rul_aligned['RUL'] = test_rul_aligned['RUL'].apply(lambda x: x + test_rul_aligned['time_cycle'].max() - test_rul_aligned['time_cycle'].iloc[0]) # Incorrect RUL alignment logic

    # Correct RUL alignment for test data. For each engine in test_df, the RUL applies to its LAST cycle.
    # We need to calculate RUL for all preceding cycles based on this final RUL.
    def calculate_test_rul(df_test_engine, true_rul_at_last_cycle):
        max_time_cycle = df_test_engine['time_cycle'].max()
        # RUL at the last observed cycle in test_df is true_rul_at_last_cycle
        df_test_engine['RUL'] = true_rul_at_last_cycle + (max_time_cycle - df_test_engine['time_cycle'])
        return df_test_engine

    # Apply RUL calculation for each engine in the test set
    test_df_with_rul = test_df_normalized.copy()
    y_test_non_seq = []
    X_test_non_seq_list = []

    # Reset RUL dataframe index to ensure proper indexing
    rul_df = rul_df.reset_index(drop=True)
    
    for i, engine_id in enumerate(test_df_normalized['engine_id'].unique()):
        engine_test_df = test_df_normalized[test_df_normalized['engine_id'] == engine_id].copy()
        # Get RUL value safely
        true_rul = rul_df['RUL'].iloc[i]
        engine_test_df = calculate_test_rul(engine_test_df, true_rul)
        y_test_non_seq.extend(engine_test_df['RUL'].tolist())
        X_test_non_seq_list.append(engine_test_df[feature_cols])

    X_test_non_seq = pd.concat(X_test_non_seq_list)
    y_test_non_seq = pd.Series(y_test_non_seq)


    # 5. Create Time-Series Sequences for LSTM
    # For training, `generate_rul_labels` adds RUL to train_df already.
    X_train_seq, y_train_seq = create_sequences(train_df_normalized, feature_cols, sequence_length)

    X_test_seq, y_test_seq = create_sequences(test_df_with_rul, feature_cols, sequence_length)

    print(f"--- Data Pipeline for FD00{dataset_number} Completed ---")

    # Return both sequential and non-sequential data for flexibility
    return (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    )

if __name__ == "__main__":
    # Example usage for FD001
    (
        X_train_non_seq, y_train_non_seq,
        X_test_non_seq, y_test_non_seq,
        X_train_seq, y_train_seq,
        X_test_seq, y_test_seq,
        feature_cols, scaler
    ) = run_data_pipeline(dataset_number=1, sequence_length=50)

    print("\nShapes for non-sequential data (FD001):")
    print(f"X_train_non_seq shape: {X_train_non_seq.shape}, y_train_non_seq shape: {y_train_non_seq.shape}")
    print(f"X_test_non_seq shape: {X_test_non_seq.shape}, y_test_non_seq shape: {y_test_non_seq.shape}")

    print("\nShapes for sequential data (FD001):")
    print(f"X_train_seq shape: {X_train_seq.shape}, y_train_seq shape: {y_train_seq.shape}")
    print(f"X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")
    print(f"Feature columns: {feature_cols}")
