import pandas as pd
import numpy as np

def calculate_rul_score(predicted_rul, max_rul=150):
    # Cap RUL at max_rul to prevent scores above 100 for very high RULs
    capped_rul = min(predicted_rul, max_rul)
    score = (capped_rul / max_rul) * 100
    return max(0, min(100, score)) # Ensure score is between 0 and 100

def calculate_anomaly_impact_score(anomaly_score, max_anomaly_score=1.0):
    # Normalize anomaly score to 0-100 range
    score = (anomaly_score / max_anomaly_score) * 100
    return max(0, min(100, score)) # Ensure score is between 0 and 100

def calculate_degradation_trend_score(current_sensor_values, initial_sensor_values, sensor_degradation_thresholds):
    if initial_sensor_values is None or current_sensor_values is None or sensor_degradation_thresholds is None:
        return 100.0 # Assume healthy if no baseline or thresholds provided

    degradation_penalties = []
    for sensor, (lower_thresh, upper_thresh) in sensor_degradation_thresholds.items():
        if sensor in current_sensor_values.index and sensor in initial_sensor_values.index:
            current_val = current_sensor_values[sensor]
            initial_val = initial_sensor_values[sensor]

            # Simple linear penalty based on deviation from initial within thresholds
            # This can be made more sophisticated.
            penalty = 0.0
            if current_val < initial_val * (1 - lower_thresh):
                penalty = abs(current_val - initial_val * (1 - lower_thresh)) / (initial_val * lower_thresh) * 50 # Example penalty scaling
            elif current_val > initial_val * (1 + upper_thresh):
                penalty = abs(current_val - initial_val * (1 + upper_thresh)) / (initial_val * upper_thresh) * 50 # Example penalty scaling

            degradation_penalties.append(min(penalty, 100.0)) # Cap individual penalty at 100

    if not degradation_penalties:
        return 100.0 # No relevant sensors for degradation

    # Average penalties, then convert to a score (100 - average_penalty)
    average_penalty = np.mean(degradation_penalties)
    degradation_score = 100 - average_penalty

    return max(0, min(100, degradation_score))

def compute_engine_health_index(predicted_rul, anomaly_score, current_sensor_values=None,
                                initial_sensor_values=None, sensor_degradation_thresholds=None,
                                weight_rul=0.5, weight_anomaly=0.3, weight_degradation=0.2,
                                max_rul_for_score=150, max_anomaly_score_for_impact=1.0):
    rul_score = calculate_rul_score(predicted_rul, max_rul=max_rul_for_score)
    anomaly_impact = calculate_anomaly_impact_score(anomaly_score, max_anomaly_score=max_anomaly_score_for_impact)
    # Invert anomaly_impact for health index calculation: higher health for lower anomaly
    anomaly_health_contribution = 100 - anomaly_impact

    degradation_score = calculate_degradation_trend_score(
        current_sensor_values, initial_sensor_values, sensor_degradation_thresholds
    )

    # Ensure weights sum to 1
    total_weight = weight_rul + weight_anomaly + weight_degradation
    if total_weight != 1.0:
        print("Warning: Weights do not sum to 1. Normalizing weights.")
        weight_rul /= total_weight
        weight_anomaly /= total_weight
        weight_degradation /= total_weight

    health_index = (
        (rul_score * weight_rul) +
        (anomaly_health_contribution * weight_anomaly) +
        (degradation_score * weight_degradation)
    )

    # Final clamp to ensure health index is always between 0 and 100
    return float(max(0.0, min(100.0, health_index)))

if __name__ == "__main__":
    # Example usage:
    predicted_rul = 75
    anomaly_score = 0.2
    current_sensors = pd.Series({
        'sensor_measure_1': 0.5,
        'sensor_measure_2': 0.6,
        'sensor_measure_3': 0.7
    })
    initial_sensors = pd.Series({
        'sensor_measure_1': 0.4,
        'sensor_measure_2': 0.5,
        'sensor_measure_3': 0.6
    })
    degradation_thresholds = {
        'sensor_measure_1': (0.1, 0.1), # 10% deviation allowed
        'sensor_measure_2': (0.1, 0.1),
        'sensor_measure_3': (0.05, 0.05)
    }

    health_index = compute_engine_health_index(
        predicted_rul,
        anomaly_score,
        current_sensor_values=current_sensors,
        initial_sensor_values=initial_sensors,
        sensor_degradation_thresholds=degradation_thresholds
    )
    print(f"\nComputed Engine Health Index: {health_index:.2f}")

    # Example with different values
    predicted_rul_low = 10
    anomaly_score_high = 0.9
    health_index_critical = compute_engine_health_index(
        predicted_rul_low,
        anomaly_score_high,
        current_sensor_values=current_sensors,
        initial_sensor_values=initial_sensors,
        sensor_degradation_thresholds=degradation_thresholds
    )
    print(f"Computed Engine Health Index (Critical): {health_index_critical:.2f}")

    predicted_rul_high = 120
    anomaly_score_low = 0.05
    health_index_healthy = compute_engine_health_index(
        predicted_rul_high,
        anomaly_score_low,
        current_sensor_values=current_sensors,
        initial_sensor_values=initial_sensors,
        sensor_degradation_thresholds=degradation_thresholds
    )
    print(f"Computed Engine Health Index (Healthy): {health_index_healthy:.2f}")
