"""
Tests for enhanced SHAP visualizations
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pytest

from explainable_ai.shap_explainer import (
    initialize_explainer,
    compute_shap_values,
    get_feature_importance,
    explain_single_prediction,
    generate_waterfall_plot,
    generate_dependence_plot,
    generate_summary_plot_data,
    generate_decision_plot_data,
    create_shap_visualization_bundle
)


@pytest.fixture
def sample_data():
    """Create dummy dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = 100 + np.sum(X * np.array(list(range(1, n_features + 1))), axis=1) + np.random.randn(n_samples) * 5

    # Train model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Create explainer
    X_background = X.sample(frac=0.1, random_state=42)
    explainer = initialize_explainer(model, X_background)

    # Compute SHAP values
    shap_values = compute_shap_values(explainer, X)

    return {
        "X": X,
        "y": y,
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values
    }


class TestShapExplainer:
    """Test basic SHAP functionality"""

    def test_explainer_initialization(self, sample_data):
        """Test SHAP explainer initializes correctly"""
        assert sample_data["explainer"] is not None
        assert hasattr(sample_data["explainer"], "expected_value")

    def test_shap_values_computation(self, sample_data):
        """Test SHAP values are computed correctly"""
        shap_vals = sample_data["shap_values"]
        assert shap_vals.shape[0] == 100  # n_samples
        assert shap_vals.shape[1] == 10   # n_features

    def test_feature_importance(self, sample_data):
        """Test feature importance calculation"""
        importance = get_feature_importance(
            sample_data["shap_values"],
            sample_data["X"].columns.tolist()
        )

        assert isinstance(importance, pd.Series)
        assert len(importance) == 10
        assert importance.sum() > 0
        # Features with higher coefficients should have higher importance
        assert importance.index[0] in sample_data["X"].columns

    def test_single_prediction_explanation(self, sample_data):
        """Test explanation for single prediction"""
        explanation = explain_single_prediction(
            sample_data["explainer"],
            sample_data["X"].iloc[0],
            sample_data["X"].columns.tolist()
        )

        assert isinstance(explanation, pd.Series)
        assert len(explanation) == 10
        assert all(explanation.index.isin(sample_data["X"].columns))


class TestWaterfallPlot:
    """Test waterfall plot generation"""

    def test_waterfall_generation(self, sample_data):
        """Test waterfall plot generation"""
        html, data = generate_waterfall_plot(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_idx=0,
            max_display=10
        )

        assert data is not None
        assert "base_value" in data
        assert "prediction" in data
        assert "features" in data
        assert len(data["features"]) <= 10

    def test_waterfall_consistency(self, sample_data):
        """Test waterfall prediction sum matches expected value"""
        _, data = generate_waterfall_plot(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_idx=0
        )

        base = data["base_value"]
        contributions = sum(f["impact"] for f in data["features"])
        expected_pred = base + contributions

        assert abs(expected_pred - data["prediction"]) < 1.0  # Allow small numerical error


class TestDependencePlot:
    """Test dependence plot generation"""

    def test_dependence_generation(self, sample_data):
        """Test dependence plot data generation"""
        dependence = generate_dependence_plot(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            feature_name="feature_0"
        )

        assert dependence is not None
        assert "feature" in dependence
        assert "interaction_feature" in dependence
        assert "data_points" in dependence
        assert len(dependence["data_points"]) > 0

    def test_dependence_data_structure(self, sample_data):
        """Test dependence plot data structure"""
        dependence = generate_dependence_plot(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            feature_name="feature_5",
            max_display=50
        )

        for point in dependence["data_points"]:
            assert "feature_value" in point
            assert "shap_value" in point
            assert "interaction_value" in point
            assert isinstance(point["feature_value"], (int, float))

    def test_dependence_max_display(self, sample_data):
        """Test max_display parameter"""
        dependence = generate_dependence_plot(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            feature_name="feature_0",
            max_display=20
        )

        assert len(dependence["data_points"]) <= 20


class TestSummaryPlot:
    """Test summary plot generation"""

    def test_summary_generation(self, sample_data):
        """Test summary plot generation"""
        summary = generate_summary_plot_data(
            sample_data["shap_values"],
            sample_data["X"],
            max_features=5
        )

        assert summary is not None
        assert "title" in summary
        assert "features" in summary
        assert len(summary["features"]) <= 5

    def test_summary_feature_info(self, sample_data):
        """Test summary plot feature information"""
        summary = generate_summary_plot_data(
            sample_data["shap_values"],
            sample_data["X"],
            max_features=10
        )

        for feature in summary["features"]:
            assert "name" in feature
            assert "mean_impact" in feature
            assert "value_range" in feature
            assert "shap_range" in feature
            assert feature["mean_impact"] >= 0

    def test_summary_ranking(self, sample_data):
        """Test features are ranked by importance"""
        summary = generate_summary_plot_data(
            sample_data["shap_values"],
            sample_data["X"],
            max_features=10
        )

        impacts = [f["mean_impact"] for f in summary["features"]]
        assert impacts == sorted(impacts, reverse=True)


class TestDecisionPlot:
    """Test decision plot generation"""

    def test_decision_generation(self, sample_data):
        """Test decision plot generation"""
        decision = generate_decision_plot_data(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_count=10
        )

        assert decision is not None
        assert "base_value" in decision
        assert "paths" in decision
        assert len(decision["paths"]) <= 10

    def test_decision_path_structure(self, sample_data):
        """Test decision path data structure"""
        decision = generate_decision_plot_data(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_count=5
        )

        for path in decision["paths"]:
            assert "prediction" in path
            assert "steps" in path
            assert len(path["steps"]) == 10  # n_features

            for step in path["steps"]:
                assert "feature" in step
                assert "value" in step
                assert "contribution" in step
                assert "cumulative" in step


class TestVisualizationBundle:
    """Test complete visualization bundle"""

    def test_bundle_generation(self, sample_data):
        """Test complete visualization bundle generation"""
        bundle = create_shap_visualization_bundle(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_idx=0
        )

        assert bundle is not None
        assert "sample_index" in bundle
        assert "timestamp" in bundle
        assert "visualizations" in bundle

    def test_bundle_visualizations(self, sample_data):
        """Test all visualizations in bundle"""
        bundle = create_shap_visualization_bundle(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"]
        )

        viz = bundle["visualizations"]
        assert "waterfall" in viz
        assert "summary" in viz
        assert "dependence" in viz
        assert "decision" in viz

    def test_bundle_error_handling(self, sample_data):
        """Test bundle handles errors gracefully"""
        # Test with invalid sample index
        bundle = create_shap_visualization_bundle(
            sample_data["explainer"],
            sample_data["shap_values"],
            sample_data["X"],
            sample_idx=1000  # Out of range
        )

        assert bundle is not None
        assert "visualizations" in bundle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
