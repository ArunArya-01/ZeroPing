"""
SHAP Visualizations API Module
Provides enhanced SHAP visualizations via REST endpoints
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel

from .shap_explainer import (
    generate_force_plot,
    generate_waterfall_plot,
    generate_dependence_plot,
    generate_summary_plot_data,
    generate_decision_plot_data,
    create_shap_visualization_bundle,
    get_feature_importance
)


class ShapVisualizationRequest(BaseModel):
    """Request model for SHAP visualization"""
    sample_index: int = 0
    max_features: int = 15
    include_force_plot: bool = True
    include_waterfall: bool = True
    include_dependence: bool = True
    include_decision: bool = True
    include_summary: bool = True


class FeatureContribution(BaseModel):
    """Model for single feature contribution in explanation"""
    name: str
    value: float
    impact: float
    contribution: str  # "positive" or "negative"


class WaterfallData(BaseModel):
    """Model for waterfall plot data"""
    base_value: float
    prediction: float
    features: List[FeatureContribution]


class DependencePoint(BaseModel):
    """Model for single point in dependence plot"""
    feature_value: float
    shap_value: float
    interaction_value: float


class DependencePlotData(BaseModel):
    """Model for dependence plot data"""
    feature: str
    interaction_feature: str
    data_points: List[DependencePoint]


class SummaryFeature(BaseModel):
    """Model for feature in summary plot"""
    name: str
    mean_impact: float
    value_range: Dict[str, float]
    shap_range: Dict[str, float]


class SummaryPlotData(BaseModel):
    """Model for summary plot data"""
    title: str
    features: List[SummaryFeature]


class DecisionStep(BaseModel):
    """Model for single decision step"""
    feature: str
    value: float
    contribution: float
    cumulative: float


class DecisionPath(BaseModel):
    """Model for single decision path"""
    prediction: float
    steps: List[DecisionStep]


class DecisionPlotData(BaseModel):
    """Model for decision plot data"""
    base_value: float
    paths: List[DecisionPath]


class ShapVisualizationResponse(BaseModel):
    """Complete SHAP visualization response"""
    sample_index: int
    timestamp: str
    feature_importance: Dict[str, float]
    waterfall: Optional[WaterfallData] = None
    summary: Optional[SummaryPlotData] = None
    dependence: Optional[Dict[str, DependencePlotData]] = None
    decision: Optional[DecisionPlotData] = None


def prepare_shap_visualizations(
    explainer,
    shap_values: np.ndarray,
    X_data: pd.DataFrame,
    request: ShapVisualizationRequest
) -> ShapVisualizationResponse:
    """Prepare complete SHAP visualization response"""

    sample_idx = request.sample_index
    if sample_idx >= len(X_data):
        sample_idx = 0

    # Always include feature importance
    feature_importance_series = get_feature_importance(shap_values, X_data.columns.tolist())
    feature_importance = {
        name: float(value)
        for name, value in feature_importance_series.items()
    }

    response_data = {
        "sample_index": sample_idx,
        "timestamp": pd.Timestamp.now().isoformat(),
        "feature_importance": feature_importance
    }

    # Generate requested visualizations
    if request.include_waterfall:
        try:
            _, waterfall_dict = generate_waterfall_plot(
                explainer, shap_values, X_data, sample_idx, request.max_features
            )
            if waterfall_dict:
                response_data["waterfall"] = waterfall_dict
        except Exception as e:
            print(f"Error generating waterfall: {e}")

    if request.include_summary:
        try:
            summary = generate_summary_plot_data(shap_values, X_data, request.max_features)
            if summary:
                response_data["summary"] = summary
        except Exception as e:
            print(f"Error generating summary: {e}")

    if request.include_dependence:
        try:
            feature_importance_series = get_feature_importance(shap_values, X_data.columns.tolist())
            top_features = feature_importance_series.head(3).index.tolist()

            dependence = {}
            for feature in top_features:
                try:
                    dep_data = generate_dependence_plot(explainer, shap_values, X_data, feature)
                    if dep_data:
                        dependence[feature] = dep_data
                except Exception as e:
                    print(f"Error generating dependence for {feature}: {e}")

            if dependence:
                response_data["dependence"] = dependence
        except Exception as e:
            print(f"Error generating dependence plots: {e}")

    if request.include_decision:
        try:
            decision = generate_decision_plot_data(explainer, shap_values, X_data, sample_count=10)
            if decision:
                response_data["decision"] = decision
        except Exception as e:
            print(f"Error generating decision plot: {e}")

    return ShapVisualizationResponse(**response_data)
