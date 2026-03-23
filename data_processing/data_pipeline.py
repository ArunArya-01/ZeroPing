import pandas as pd
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import load_cmapss_data

# Section 1.1 functions from your updated files
from data_processing.preprocessor import (
    remove_constant_sensors, 
    normalize_data, 
    validate_and_clean_data, 
    apply_smote_augmentation, 
    generate_data_version
)
from data_processing.feature_engineering import (
    generate_rul_labels, 
    create_sequences, 
    advanced_feature_engineering
)

def run_data_pipeline(dataset_number, sequence_length=50, data_path="./Dataset"):

    print(f"--- Starting Data Pipeline for FD00{dataset_number} ---")

    # 1. Load Data
    train_df, test_df, rul_df = load_cmapss_data(dataset_number, data_path)

    # Section 1.1: Validate and clean data immediately after loading (Fixes missing values & outliers)
    train_df = validate_and_clean_data(train_df)
    test_df = validate_and_clean_data(test_df)

    # 2. Generate RUL labels for training data
    train_df = generate_rul_labels(train_df)

    # Identify sensor columns before any removal
    all_sensor_cols = [col for col in train_df.columns if 'sensor_measure' in col]

    # 3. Remove Constant Sensors (fit on train, transform on both)
    train_df_processed, constant_sensors = remove_constant_sensors(train_df.copy())
    test_df_processed = test_df.drop(columns=constant_sensors, errors='ignore')

    active_sensor_cols = [col for col in all_sensor_cols if col not in constant_sensors]
    op_setting_cols = [col for col in train_df.columns if 'op_setting' in col]

    # Section 1.1: Apply Rolling Averages, Trend, and FFT
    print("Applying Advanced Feature Engineering (Rolling Means, FFT)...")
    train_df_processed, new_cols = advanced_feature_engineering(train_df_processed)
    test_df_processed, _ = advanced_feature_engineering(test_df_processed)

    # Update feature columns to include the new engineered columns
    feature_cols = op_setting_cols + active_sensor_cols + new_cols

    # 4. Normalize Data
    train_df_normalized, test_df_normalized, scaler = normalize_data(
        train_df_processed.copy(), test_df_processed.copy(), feature_cols
    )

    # Section 1.1: Generate Data Version Hash
    generate_data_version(train_df_normalized)

    # For non-sequential models, X_train and y_train are simply the features and RUL
    X_train_non_seq = train_df_normalized[feature_cols]
    y_train_non_seq = train_df_normalized["RUL"]

    # Section 1.1: Apply SMOTE Augmentation (Logged for imbalanced classes)
    _ = apply_smote_augmentation(X_train_non_seq, y_train_non_seq)

    # Correct RUL alignment for test data.
    def calculate_test_rul(df_test_engine, true_rul_at_last_cycle):
        max_time_cycle = df_test_engine['time_cycle'].max()
        df_test_engine['RUL'] = true_rul_at_last_cycle + (max_time_cycle - df_test_engine['time_cycle'])
        return df_test_engine

    # FIX: Create lists to properly reconstruct the test DataFrame with RULs for sequences
    y_test_non_seq = []
    X_test_non_seq_list = []
    test_df_with_rul_list = []

    # Reset RUL dataframe index to ensure proper indexing
    rul_df = rul_df.reset_index(drop=True)
    
    for i, engine_id in enumerate(test_df_normalized['engine_id'].unique()):
        engine_test_df = test_df_normalized[test_df_normalized['engine_id'] == engine_id].copy()
        true_rul = rul_df['RUL'].iloc[i]
        
        # Calculate RUL
        engine_test_df = calculate_test_rul(engine_test_df, true_rul)
        
        # Save to lists
        y_test_non_seq.extend(engine_test_df['RUL'].tolist())
        X_test_non_seq_list.append(engine_test_df[feature_cols])
        test_df_with_rul_list.append(engine_test_df) # This fixes the NoneType error!

    X_test_non_seq = pd.concat(X_test_non_seq_list)
    y_test_non_seq = pd.Series(y_test_non_seq)
    
    # Reconstruct the full test DataFrame so the sequence creator can find the 'RUL' column
    test_df_with_rul = pd.concat(test_df_with_rul_list)

    # 5. Create Time-Series Sequences for LSTM
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
    print(f"Number of Feature columns: {len(feature_cols)}")