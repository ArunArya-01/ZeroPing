import pandas as pd
import numpy as np

def generate_rul_labels(df):
    max_cycles = df.groupby("engine_id")["time_cycle"].max()
    merged = df.merge(max_cycles.rename("max_time_cycle"), on="engine_id")
    df["RUL"] = merged["max_time_cycle"] - df["time_cycle"]
    return df

def advanced_feature_engineering(df):
    """Section 1.1: Enhance feature engineering (rolling averages, FFT, trend)"""
    df_out = df.copy()
    sensor_cols = [col for col in df.columns if 'sensor_measure' in col]
    grouped = df_out.groupby('engine_id')
    
    new_cols = []
    fft_targets = ['sensor_measure_2', 'sensor_measure_3', 'sensor_measure_15']
    
    for col in sensor_cols:
        if col in df_out.columns:
            # Rolling average (5 cycles)
            df_out[f'{col}_roll_mean'] = grouped[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
            new_cols.append(f'{col}_roll_mean')
            
            # Trend (Difference from previous cycle)
            df_out[f'{col}_trend'] = grouped[col].transform(lambda x: x.diff().fillna(0))
            new_cols.append(f'{col}_trend')
            
            # FFT (Fast Fourier Transform for specific volatile sensors)
            if col in fft_targets:
                df_out[f'{col}_fft'] = grouped[col].transform(lambda x: np.abs(np.fft.fft(x)))
                new_cols.append(f'{col}_fft')
                
    return df_out, new_cols

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