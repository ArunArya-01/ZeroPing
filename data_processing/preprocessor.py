import pandas as pd
import numpy as np
import hashlib
import time
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def validate_and_clean_data(df):
    """Section 1.1: Data validation (missing values, outliers)"""
    sensor_cols = [col for col in df.columns if 'sensor_measure' in col]
    
    # 1. Fill missing values
    df = df.ffill().bfill()
    
    # 2. Clip outliers to 1st and 99th percentiles
    for col in sensor_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
            
    return df

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

def apply_smote_augmentation(X, y, critical_threshold=30):
    """Section 1.1: Data augmentation for imbalanced classes (Critical vs Healthy)"""
    print("Applying SMOTE Augmentation for imbalanced critical states...")
    try:
        # Create a binary mask for SMOTE: 1 if critical, 0 if healthy
        critical_mask = (y <= critical_threshold).astype(int)
        smote = SMOTE(sampling_strategy='minority', random_state=42)
        X_resampled, _ = smote.fit_resample(X, critical_mask)
        return X_resampled
    except Exception as e:
        print(f"Skipping SMOTE (requires more diverse data): {e}")
        return X

def generate_data_version(df):
    """Section 1.1: Implement data versioning"""
    data_string = pd.util.hash_pandas_object(df).values.tobytes()
    version_hash = hashlib.md5(data_string).hexdigest()[:8]
    version_tag = f"v_{time.strftime('%Y%m%d')}_{version_hash}"
    print(f"Data Version Locked: {version_tag}")
    return version_tag