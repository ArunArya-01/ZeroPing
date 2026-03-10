import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def calculate_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)

    print(f"--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print("------------------------------")

    return rmse, mae

if __name__ == "__main__":
    # Dummy data for demonstration
    y_true_dummy = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    y_pred_dummy_good = np.array([98, 88, 81, 72, 59, 51, 39, 31, 19, 11])
    y_pred_dummy_bad = np.array([110, 80, 70, 60, 50, 60, 50, 40, 30, 20])

    print("Evaluating a good prediction set:")
    rmse_good, mae_good = evaluate_model(y_true_dummy, y_pred_dummy_good, "Good Predictor")

    print("\nEvaluating a bad prediction set:")
    rmse_bad, mae_bad = evaluate_model(y_true_dummy, y_pred_dummy_bad, "Bad Predictor")
