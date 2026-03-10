import pandas as pd
import numpy as np

def generate_rul_labels(df):

    max_cycles = df.groupby("engine_id")["time_cycle"].max()
    merged = df.merge(max_cycles.rename("max_time_cycle"), on="engine_id")
    df["RUL"] = merged["max_time_cycle"] - df["time_cycle"]
    return df

def create_sequences(df, sensor_cols, sequence_length):
 
    X, y = [], []
    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id]
        features = engine_df[sensor_cols].values
        if "RUL" in engine_df.columns:
            rul_labels = engine_df["RUL"].values
        else:
            rul_labels = None

        for i in range(len(engine_df) - sequence_length + 1):
            X.append(features[i : i + sequence_length])
            if rul_labels is not None:
                # For RUL prediction, we predict the RUL at the end of the sequence
                y.append(rul_labels[i + sequence_length - 1])

    return np.array(X), np.array(y) if y else None

if __name__ == "__main__":
    # Example usage with dummy data
    data = {
        'engine_id': [1, 1, 1, 1, 2, 2, 2, 2],
        'time_cycle': [1, 2, 3, 4, 1, 2, 3, 4],
        'sensor_measure_1': [10, 20, 30, 40, 50, 60, 70, 80],
        'sensor_measure_2': [1, 2, 3, 4, 5, 6, 7, 8]
    }
    dummy_df = pd.DataFrame(data)

    print("Original DataFrame:\n", dummy_df)

    # Generate RUL labels
    df_with_rul = generate_rul_labels(dummy_df.copy())
    print("\nDataFrame with RUL labels:\n", df_with_rul)

    # Create sequences
    sensor_cols = ["sensor_measure_1", "sensor_measure_2"]
    sequence_length = 2
    X_sequences, y_labels = create_sequences(df_with_rul.copy(), sensor_cols, sequence_length)

    print(f"\nCreated sequences (X) with length {sequence_length}:\n", X_sequences)
    print(f"Shape of X_sequences: {X_sequences.shape}")
    print(f"\nCorresponding RUL labels (y):\n", y_labels)
    print(f"Shape of y_labels: {y_labels.shape}")

    # Example for test data (without RUL in input df)
    X_test_sequences, _ = create_sequences(dummy_df.copy(), sensor_cols, sequence_length)
    print(f"\nTest sequences (X) without RUL labels:\n", X_test_sequences)
    print(f"Shape of X_test_sequences: {X_test_sequences.shape}")
