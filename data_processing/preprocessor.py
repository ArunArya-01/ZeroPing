import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def remove_constant_sensors(df):
 
    sensor_cols = [col for col in df.columns if 'sensor_measure' in col]
    constant_sensors = [col for col in sensor_cols if df[col].nunique() == 1]
    if constant_sensors:
        print(f"Removing constant sensors: {constant_sensors}")
        df = df.drop(columns=constant_sensors)
    return df, constant_sensors

def normalize_data(train_df, test_df, sensor_cols):

    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    return train_df, test_df, scaler

if __name__ == "__main__":
    # Example usage (assuming you have a way to load dummy data or use data_loader):
    # For demonstration, let's create a dummy DataFrame
    data = {
        'engine_id': [1, 1, 1, 2, 2, 2],
        'time_cycle': [1, 2, 3, 1, 2, 3],
        'op_setting_1': [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        'sensor_measure_1': [100, 101, 102, 105, 106, 107],
        'sensor_measure_2': [0.5, 0.5, 0.5, 0.8, 0.8, 0.8], # Constant sensor for engine 1
        'sensor_measure_3': [10, 11, 12, 13, 14, 15]
    }
    dummy_train_df = pd.DataFrame(data)
    dummy_test_df = pd.DataFrame(data)

    print("Original training data:\n", dummy_train_df)

    # Remove constant sensors
    processed_train_df, removed_sensors = remove_constant_sensors(dummy_train_df.copy())
    print("\nProcessed training data after removing constant sensors:\n", processed_train_df)
    print("Removed sensors:", removed_sensors)

    # Identify sensor columns for normalization after removing constant ones
    active_sensor_cols = [col for col in processed_train_df.columns if 'sensor_measure' in col]

    # Normalize data
    normalized_train_df, normalized_test_df, scaler = normalize_data(
        processed_train_df.copy(), dummy_test_df.copy(), active_sensor_cols
    )
    print("\nNormalized training data:\n", normalized_train_df)
    print("\nNormalized test data:\n", normalized_test_df)
