import shap
import pandas as pd
import numpy as np

def initialize_explainer(model, X_background):

    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, X_background)
    print("SHAP TreeExplainer initialized.")
    return explainer

def compute_shap_values(explainer, X_data):

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_data)
    print("SHAP values computed.")
    return shap_values

def get_feature_importance(shap_values, feature_names):

    # For tree models, shap_values can be a single array or a list of arrays for multi-output
    # For regression, it's typically a single array.
    if isinstance(shap_values, list):
        # If multi-output, take the mean absolute SHAP values across outputs
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    feature_importance = pd.Series(mean_abs_shap, index=feature_names)
    feature_importance = feature_importance.sort_values(ascending=False)
    return feature_importance

def explain_single_prediction(explainer, X_sample, feature_names):
    if isinstance(X_sample, pd.Series):
        X_sample = X_sample.to_frame().T # Reshape for explainer
    elif isinstance(X_sample, np.ndarray):
        X_sample = X_sample.reshape(1, -1) # Reshape for explainer

    single_shap_values = explainer.shap_values(X_sample)[0]
    explanation = pd.Series(single_shap_values, index=feature_names)
    return explanation.sort_values(key=abs, ascending=False)

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    # Dummy data for demonstration
    np.random.seed(42)
    X_dummy = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(1, 6)])
    y_dummy = pd.Series(np.random.rand(100))

    # Train a dummy model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_dummy, y_dummy)

    # Initialize explainer with a background dataset (e.g., a subset of training data)
    X_background_sample = X_dummy.sample(frac=0.1, random_state=42)
    explainer = initialize_explainer(model, X_background_sample)

    # Compute SHAP values for a subset of data
    X_test_sample = X_dummy.sample(n=10, random_state=42)
    shap_values = compute_shap_values(explainer, X_test_sample)

    print("\nSHAP values shape:", shap_values.shape)

    # Get global feature importance
    feature_importance = get_feature_importance(shap_values, X_dummy.columns.tolist())
    print("\nGlobal Feature Importance (mean absolute SHAP values):\n", feature_importance)

    # Explain a single prediction
    single_sample = X_test_sample.iloc[0]
    single_explanation = explain_single_prediction(explainer, single_sample, X_dummy.columns.tolist())
    print("\nExplanation for a single prediction (most influential features):\n", single_explanation)
