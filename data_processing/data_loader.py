import pandas as pd
import numpy as np

def load_cmapss_data(dataset_number, data_path="./Dataset"):
 
    # Define column names based on the readme file
    cols = [
        "engine_id", "time_cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
        *[f"sensor_measure_{i}" for i in range(1, 22)]
    ]

    # Load training data
    train_df = pd.read_csv(
        f"{data_path}/train_FD00{dataset_number}.txt", sep="\\s+", header=None, names=cols
    )

    # Load test data
    test_df = pd.read_csv(
        f"{data_path}/test_FD00{dataset_number}.txt", sep="\\s+", header=None, names=cols
    )

    # Load RUL data
    rul_df = pd.read_csv(
        f"{data_path}/RUL_FD00{dataset_number}.txt", sep="\\s+", header=None, names=["RUL"]
    )

    print(f"Loaded FD00{dataset_number}:")
    print(f"  Training data shape: {train_df.shape}")
    print(f"  Test data shape: {test_df.shape}")
    print(f"  RUL data shape: {rul_df.shape}")

    return train_df, test_df, rul_df

if __name__ == "__main__":
    # Example usage:
    train_df1, test_df1, rul_df1 = load_cmapss_data(1)
    print("\nFirst few rows of training data (FD001):")
    print(train_df1.head())
    print("\nFirst few rows of RUL data (FD001):")
    print(rul_df1.head())

    train_df4, test_df4, rul_df4 = load_cmapss_data(4)
    print("\nFirst few rows of training data (FD004):")
    print(train_df4.head())
    print("\nFirst few rows of RUL data (FD004):")
    print(rul_df4.head())
