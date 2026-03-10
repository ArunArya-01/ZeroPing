import sys
import os

import pytest


class TestImports:

    def test_data_processing_imports(self):
        from data_processing import data_loader
        from data_processing import data_pipeline
        from data_processing import feature_engineering
        from data_processing import preprocessor
        
        assert data_loader is not None
        assert data_pipeline is not None
        assert feature_engineering is not None
        assert preprocessor is not None
    
    def test_ml_prediction_imports(self):
        from ml_prediction import random_forest_model
        from ml_prediction import xgboost_model
        from ml_prediction import lstm_model
        from ml_prediction import trainer
        from ml_prediction import evaluator
        
        assert random_forest_model is not None
        assert xgboost_model is not None
        assert lstm_model is not None
        assert trainer is not None
        assert evaluator is not None
    
    def test_anomaly_detection_imports(self):
        from anomaly_detection import isolation_forest_detector
        
        assert isolation_forest_detector is not None
    
    def test_explainable_ai_imports(self):
        from explainable_ai import shap_explainer
        
        assert shap_explainer is not None
    
    def test_health_risk_imports(self):
        from health_risk import health_calculator
        from health_risk import risk_evaluator
        
        assert health_calculator is not None
        assert risk_evaluator is not None
    
    @pytest.mark.skip(reason="Requires dataset files - tested via syntax check")
    def test_digital_twin_imports(self):
        from digital_twin import digital_twin_simulator
        
        assert digital_twin_simulator is not None
    
    @pytest.mark.skip(reason="Requires dataset files - tested via syntax check")
    def test_dashboard_imports(self):
        from dashboard import dashboard_app
        
        assert dashboard_app is not None


class TestDataProcessing:
   
    @pytest.mark.unit
    def test_preprocessor_class(self):
        from data_processing.preprocessor import Preprocessor
        
        # Just check class exists and is callable
        assert callable(Preprocessor)
    
    @pytest.mark.unit
    def test_feature_engineering_class(self):
        from data_processing.feature_engineering import FeatureEngineering
        
        assert callable(FeatureEngineering)


class TestMLModels:

    
    @pytest.mark.unit
    def test_random_forest_model_class(self):
        from ml_prediction.random_forest_model import RandomForestModel
        
        assert callable(RandomForestModel)
    
    @pytest.mark.unit
    def test_xgboost_model_class(self):
        from ml_prediction.xgboost_model import XGBoostModel
        
        assert callable(XGBoostModel)
    
    @pytest.mark.unit
    def test_lstm_model_class(self):
        from ml_prediction.lstm_model import LSTMModel
        
        assert callable(LSTMModel)


class TestAnomalyDetection:
    
    @pytest.mark.unit
    def test_isolation_forest_class(self):
        from anomaly_detection.isolation_forest_detector import IsolationForestDetector
        
        assert callable(IsolationForestDetector)


class TestHealthRisk:

    @pytest.mark.unit
    def test_health_calculator_class(self):
        from health_risk.health_calculator import HealthCalculator
        
        assert callable(HealthCalculator)
    
    @pytest.mark.unit
    def test_risk_evaluator_class(self):
        from health_risk.risk_evaluator import RiskEvaluator
        
        assert callable(RiskEvaluator)


class TestDigitalTwin:

    @pytest.mark.unit
    def test_digital_twin_simulator_class(self):
        from digital_twin.digital_twin_simulator import DigitalTwinSimulator
        
        assert callable(DigitalTwinSimulator)

class TestExplainableAI:

    @pytest.mark.unit
    def test_shap_explainer_class(self):
        from explainable_ai.shap_explainer import SHAPExplainer
        
        assert callable(SHAPExplainer)
