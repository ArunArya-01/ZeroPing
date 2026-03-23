import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import json


def initialize_explainer(model, X_background):
    """Initialize SHAP TreeExplainer with background data."""
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, X_background)
    print("SHAP TreeExplainer initialized.")
    return explainer


def compute_shap_values(explainer, X_data):
    """Compute SHAP values for given data."""
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_data, check_additivity=False)
    print("SHAP values computed.")
    return shap_values


def get_feature_importance(shap_values, feature_names) -> pd.Series:
    """Calculate global feature importance from SHAP values."""
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    feature_importance = pd.Series(mean_abs_shap, index=feature_names)
    feature_importance = feature_importance.sort_values(ascending=False)
    return feature_importance


def explain_single_prediction(explainer, X_sample, feature_names) -> pd.Series:
    """Explain a single prediction with SHAP values."""
    if isinstance(X_sample, pd.Series):
        X_sample = X_sample.to_frame().T
    elif isinstance(X_sample, np.ndarray):
        X_sample = X_sample.reshape(1, -1)

    single_shap_values = explainer.shap_values(X_sample, check_additivity=False)[0]
    explanation = pd.Series(single_shap_values, index=feature_names)
    return explanation.sort_values(key=abs, ascending=False)


def generate_force_plot(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    sample_idx: int = 0
) -> str:
    """Generate interactive force plot visualization (HTML)."""
    if isinstance(shap_values, list):
        sv = shap_values[0][sample_idx:sample_idx+1]
    else:
        sv = shap_values[sample_idx:sample_idx+1]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[0]

    sample_data = X_data.iloc[sample_idx:sample_idx+1]

    # Create force plot
    try:
        force_plot = shap.force_plot(
            base_value,
            sv,
            sample_data,
            matplotlib=False
        )
        return force_plot.data
    except Exception as e:
        print(f"Error generating force plot: {e}")
        return ""


def generate_waterfall_plot(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    sample_idx: int = 0,
    max_display: int = 10
) -> Tuple[str, Dict[str, float]]:
    """Generate waterfall plot showing feature contributions (HTML + JSON data)."""
    if isinstance(shap_values, list):
        sv = shap_values[0][sample_idx]
    else:
        sv = shap_values[sample_idx]

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[0]

    feature_names = X_data.columns.tolist()
    sample_data = X_data.iloc[sample_idx]

    # Create explanation object for waterfall plot
    explanation = shap.Explanation(
        values=sv,
        base_values=base_value,
        data=sample_data.values,
        feature_names=feature_names
    )

    try:
        waterfall_plot = shap.plots.waterfall(explanation, show=False)

        # Extract data for JSON response
        waterfall_data = {
            "base_value": float(base_value),
            "prediction": float(base_value + np.sum(sv)),
            "features": []
        }

        # Sort by absolute impact
        impacts = [(i, sv[i], feature_names[i], sample_data[i]) for i in range(len(sv))]
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)

        for idx, impact, fname, fvalue in impacts[:max_display]:
            waterfall_data["features"].append({
                "name": fname,
                "value": float(fvalue),
                "impact": float(impact),
                "contribution": "positive" if impact > 0 else "negative"
            })

        return str(waterfall_plot), waterfall_data
    except Exception as e:
        print(f"Error generating waterfall plot: {e}")
        return "", {}


def generate_dependence_plot(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    feature_name: str,
    max_display: int = 100
) -> Dict[str, Any]:
    """Generate dependence plot data (showing feature interaction)."""
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    if feature_name not in X_data.columns:
        return {}

    feature_idx = X_data.columns.get_loc(feature_name)

    # Sample data if too large
    if len(X_data) > max_display:
        indices = np.random.choice(len(X_data), max_display, replace=False)
        feature_vals = X_data.iloc[indices, feature_idx].values
        shap_vals = sv[indices, feature_idx]
        sample_X = X_data.iloc[indices]
    else:
        feature_vals = X_data.iloc[:, feature_idx].values
        shap_vals = sv[:, feature_idx]
        sample_X = X_data

    # Find interaction feature (highest correlation with SHAP values)
    shap_corr = np.corrcoef(sv, X_data.T)
    interaction_idx = np.argmax(np.abs(shap_corr[feature_idx, len(sv):-1]))
    interaction_feature = X_data.columns[interaction_idx]
    interaction_vals = sample_X.iloc[:, interaction_idx].values

    dependence_data = {
        "feature": feature_name,
        "interaction_feature": interaction_feature,
        "data_points": [
            {
                "feature_value": float(fv),
                "shap_value": float(sv),
                "interaction_value": float(iv)
            }
            for fv, sv, iv in zip(feature_vals, shap_vals, interaction_vals)
        ]
    }

    return dependence_data


def generate_summary_plot_data(
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    max_features: int = 15
) -> Dict[str, Any]:
    """Generate summary plot data (feature importance with value effects)."""
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    # Calculate mean absolute impact
    mean_abs_impact = np.mean(np.abs(sv), axis=0)

    # Sort features by importance
    feature_importance = pd.Series(mean_abs_impact, index=X_data.columns)
    feature_importance = feature_importance.nlargest(max_features)

    summary_data = {
        "title": "SHAP Summary Plot (Mean |SHAP value|)",
        "features": []
    }

    for feature in feature_importance.index:
        feature_idx = X_data.columns.get_loc(feature)
        feature_values = X_data.iloc[:, feature_idx].values
        shap_feature_values = sv[:, feature_idx]

        summary_data["features"].append({
            "name": feature,
            "mean_impact": float(feature_importance[feature]),
            "value_range": {
                "min": float(np.min(feature_values)),
                "max": float(np.max(feature_values)),
                "mean": float(np.mean(feature_values))
            },
            "shap_range": {
                "min": float(np.min(shap_feature_values)),
                "max": float(np.max(shap_feature_values))
            }
        })

    return summary_data


def generate_decision_plot_data(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    sample_count: int = 10
) -> Dict[str, Any]:
    """Generate decision plot data (showing prediction path)."""
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[0]

    # Sample if needed
    if len(X_data) > sample_count:
        indices = np.random.choice(len(X_data), sample_count, replace=False)
        sv_sample = sv[indices]
        X_sample = X_data.iloc[indices]
    else:
        sv_sample = sv
        X_sample = X_data

    decision_data = {
        "base_value": float(base_value),
        "paths": []
    }

    for idx in range(len(sv_sample)):
        shap_vals = sv_sample[idx]
        cumsum = np.cumsum([base_value] + shap_vals.tolist())

        path = {
            "prediction": float(cumsum[-1]),
            "steps": [
                {
                    "feature": X_sample.columns[i],
                    "value": float(X_sample.iloc[idx, i]),
                    "contribution": float(shap_vals[i]),
                    "cumulative": float(cumsum[i+1])
                }
                for i in range(len(shap_vals))
            ]
        }
        decision_data["paths"].append(path)

    return decision_data


def create_shap_visualization_bundle(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    sample_idx: int = 0,
    max_features: int = 15
) -> Dict[str, Any]:
    """Create a complete SHAP visualization bundle for a single prediction."""
    bundle = {
        "sample_index": sample_idx,
        "timestamp": pd.Timestamp.now().isoformat(),
        "visualizations": {}
    }

    try:
        bundle["visualizations"]["waterfall"] = generate_waterfall_plot(
            explainer, shap_values, X_data, sample_idx
        )[1]
    except Exception as e:
        print(f"Waterfall plot error: {e}")
        bundle["visualizations"]["waterfall"] = {}

    try:
        bundle["visualizations"]["summary"] = generate_summary_plot_data(
            shap_values, X_data, max_features
        )
    except Exception as e:
        print(f"Summary plot error: {e}")
        bundle["visualizations"]["summary"] = {}

    try:
        # Get top features for dependence plots
        feature_importance = get_feature_importance(shap_values, X_data.columns.tolist())
        top_features = feature_importance.head(3).index.tolist()

        bundle["visualizations"]["dependence"] = {}
        for feature in top_features:
            try:
                bundle["visualizations"]["dependence"][feature] = generate_dependence_plot(
                    explainer, shap_values, X_data, feature
                )
            except Exception as e:
                print(f"Dependence plot error for {feature}: {e}")
    except Exception as e:
        print(f"Dependence plots error: {e}")
        bundle["visualizations"]["dependence"] = {}

    try:
        bundle["visualizations"]["decision"] = generate_decision_plot_data(
            explainer, shap_values, X_data, sample_count=5
        )
    except Exception as e:
        print(f"Decision plot error: {e}")
        bundle["visualizations"]["decision"] = {}

    return bundle

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
