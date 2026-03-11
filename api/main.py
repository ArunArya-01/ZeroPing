"""
ZeroPing API - FastAPI backend for the Engine Health Monitoring System
"""
import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Try to import ML modules - make them optional for graceful degradation
try:
    from data_processing.data_pipeline import run_data_pipeline
    from anomaly_detection.isolation_forest_detector import load_model_if
    from explainable_ai.shap_explainer import initialize_explainer, explain_single_prediction
    from health_risk.health_calculator import compute_engine_health_index
    from health_risk.risk_evaluator import assess_risk_level
    from digital_twin.digital_twin_simulator import DigitalTwinSimulator
    
    ML_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML modules could not be loaded: {e}")
    ML_MODELS_AVAILABLE = False

# Try to import optional ML models
try:
    from ml_prediction.random_forest_model import load_model_rf
    from ml_prediction.xgboost_model import load_model_xgb
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# Global state for models
class ModelState:
    simulator: Optional[Any] = None
    shap_explainer: Optional[Any] = None
    feature_cols: Optional[List[str]] = None
    initialized: bool = False


model_state = ModelState()


def initialize_models():
    """Initialize all ML models and resources"""
    if model_state.initialized or not ML_MODELS_AVAILABLE:
        return
    
    try:
        dataset_number = 1
        sequence_length = 50
        model_type = "random_forest"
        
        # Initialize the digital twin simulator
        model_state.simulator = DigitalTwinSimulator(
            dataset_number=dataset_number,
            sequence_length=sequence_length,
            model_type=model_type
        )
        
        # Get training data for SHAP explainer
        (
            X_train_non_seq, _, _, _,
            _, _, _, _,
            model_state.feature_cols, _
        ) = run_data_pipeline(dataset_number, sequence_length)
        
        # Sample background data for SHAP
        if X_train_non_seq.shape[0] > 1000:
            shap_background_data = X_train_non_seq[model_state.feature_cols].sample(n=500, random_state=42)
        else:
            shap_background_data = X_train_non_seq[model_state.feature_cols]
        
        # Initialize SHAP explainer
        model_state.shap_explainer = initialize_explainer(
            model_state.simulator.rul_prediction_model,
            shap_background_data
        )
        
        model_state.initialized = True
        print("ML models loaded successfully!")
    except Exception as e:
        print(f"Error initializing models: {e}")
        model_state.initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    initialize_models()
    yield
    # Cleanup on shutdown


app = FastAPI(
    title="ZeroPing API",
    description="Engine Health Monitoring System API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class EngineInfo(BaseModel):
    id: str
    healthIndex: float
    status: str
    flightCycle: int
    rul: int
    advisory: str


class RulTrendPoint(BaseModel):
    cycle: int
    rul: int


class SensorDataPoint(BaseModel):
    cycle: int
    temperature: float
    pressure: float
    fanSpeed: float
    coreSpeed: float
    vibration: float


class FeatureImportancePoint(BaseModel):
    name: str
    importance: float
    impact: str


class DegradationDataPoint(BaseModel):
    cycle: int
    health: float


class EngineDetails(BaseModel):
    engine: EngineInfo
    rulTrend: List[RulTrendPoint]
    sensorData: List[SensorDataPoint]
    featureImportance: List[FeatureImportancePoint]
    degradationData: List[DegradationDataPoint]


def generate_fallback_engine_data(engine_id: int, engine_num: int = 0) -> EngineDetails:
    """Generate fallback data when ML models are not available"""
    np.random.seed(engine_id * 100 + engine_num)
    
    # Define engine states based on engine number to get a mix
    # Engine 1-3: nominal (high health)
    # Engine 4-6: warning (medium health)
    # Engine 7-10: critical (low health)
    engine_states = [
        ("nominal", 85, 98),   # Engine 1-3
        ("nominal", 80, 95),   # 
        ("nominal", 88, 100),  #
        ("warning", 55, 70),   # Engine 4-6
        ("warning", 60, 75),   #
        ("warning", 50, 65),   #
        ("critical", 20, 40),  # Engine 7-10
        ("critical", 25, 38),  #
        ("critical", 15, 35),  #
        ("critical", 30, 45),  #
    ]
    
    state_type, health_min, health_max = engine_states[engine_num % len(engine_states)]
    
    # Generate current health based on state
    current_health = np.random.uniform(health_min, health_max)
    
    # Calculate max cycles and current cycle based on health
    # Higher health = fewer cycles completed
    if state_type == "nominal":
        max_cycles = np.random.randint(200, 280)
        current_cycle = int(max_cycles * (100 - current_health) / 100 * np.random.uniform(0.3, 0.7))
    elif state_type == "warning":
        max_cycles = np.random.randint(180, 260)
        current_cycle = int(max_cycles * (100 - current_health) / 100 * np.random.uniform(0.5, 0.85))
    else:  # critical
        max_cycles = np.random.randint(150, 240)
        current_cycle = int(max_cycles * (100 - current_health) / 100 * np.random.uniform(0.7, 1.0))
    
    current_cycle = max(1, min(current_cycle, max_cycles - 1))
    
    rul_trend = []
    degradation_data = []
    sensor_data = []
    
    for cycle in range(1, max_cycles + 1):
        progress = cycle / max_cycles
        
        # Calculate health based on position in lifecycle
        if cycle <= current_cycle:
            # Historical data (before current cycle)
            health = current_health + (100 - current_health) * (1 - cycle / current_cycle)
        else:
            # Future prediction
            remaining_progress = (cycle - current_cycle) / (max_cycles - current_cycle)
            health = current_health * (1 - remaining_progress * 0.9)
        
        health = max(0, min(100, health + np.random.randn() * 1.5))
        
        # RUL calculation
        rul = max(0, max_cycles - cycle)
        
        rul_trend.append(RulTrendPoint(cycle=cycle, rul=int(rul)))
        degradation_data.append(DegradationDataPoint(cycle=cycle, health=float(health)))
        
        # Generate sensor data
        base = health / 100
        sensor_data.append(SensorDataPoint(
            cycle=cycle,
            temperature=250 + 80 * (1 - base) + np.random.randn() * 3,
            pressure=80 + 40 * (1 - base) + np.random.randn() * 2,
            fanSpeed=4000 + 1500 * base + np.random.randn() * 30,
            coreSpeed=8000 + 2500 * base + np.random.randn() * 50,
            vibration=0.005 + 0.025 * (1 - base) + np.random.randn() * 0.002,
        ))
    
    # Get current values at the engine's current cycle
    if current_cycle <= len(degradation_data):
        latest_health = degradation_data[current_cycle - 1].health
    else:
        latest_health = degradation_data[-1].health
    
    if current_cycle <= len(rul_trend):
        latest_rul = rul_trend[current_cycle - 1].rul
    else:
        latest_rul = rul_trend[-1].rul
    
    # Determine status based on health
    if latest_health < 40:
        status = "critical"
        advisory = "Immediate maintenance recommended. Critical degradation detected."
    elif latest_health < 70:
        status = "warning"
        advisory = "Monitor closely - gradual degradation detected."
    else:
        status = "nominal"
        advisory = "Engine operating within normal parameters."
    
    # Feature importance
    feature_importance = [
        FeatureImportancePoint(name="Total Temperature", importance=0.85, impact="high"),
        FeatureImportancePoint(name="Pressure Ratio", importance=0.72, impact="high"),
        FeatureImportancePoint(name="Fan Speed", importance=0.58, impact="medium"),
        FeatureImportancePoint(name="Core Speed", importance=0.45, impact="medium"),
        FeatureImportancePoint(name="Oil Temp", importance=0.32, impact="low"),
        FeatureImportancePoint(name="Vibration", importance=0.25, impact="low"),
    ]
    
    engine_info = EngineInfo(
        id=f"Engine {engine_id}",
        healthIndex=float(latest_health),
        status=status,
        flightCycle=current_cycle,
        rul=int(latest_rul),
        advisory=advisory
    )
    
    return EngineDetails(
        engine=engine_info,
        rulTrend=rul_trend,
        sensorData=sensor_data,
        featureImportance=feature_importance,
        degradationData=degradation_data
    )


# API Endpoints
@app.get("/api/engines", response_model=List[EngineInfo])
async def get_engines():
    """Get list of all available engines"""
    engines = []
    
    if model_state.initialized and model_state.simulator:
        try:
            available_engine_ids = model_state.simulator.test_df_with_engine["engine_id"].unique()
            
            for i, engine_id in enumerate(available_engine_ids[:20]):
                simulated_data = model_state.simulator.simulate_engine_degradation(engine_id)
                
                latest_row = simulated_data.iloc[-1]
                health_index = latest_row["health_index"]
                risk_level, advisory = assess_risk_level(health_index)
                
                status = "nominal"
                if risk_level == "Red":
                    status = "critical"
                elif risk_level == "Yellow":
                    status = "warning"
                
                engines.append(EngineInfo(
                    id=f"Engine {engine_id}",
                    healthIndex=health_index,
                    status=status,
                    flightCycle=int(latest_row["time_cycle"]),
                    rul=int(latest_row["predicted_RUL"]),
                    advisory=advisory
                ))
        except Exception as e:
            print(f"Error getting engines from simulator: {e}")
    
    # If no engines from simulator, generate fallback data with proper distribution
    if not engines:
        for i in range(1, 11):
            details = generate_fallback_engine_data(i, i - 1)
            engines.append(details.engine)
    
    return engines


@app.get("/api/engines/{engine_id}", response_model=EngineDetails)
async def get_engine_details(engine_id: int):
    """Get detailed information for a specific engine"""
    # Find which fallback engine number this is
    engine_num = (engine_id - 1) % 10
    
    if not model_state.initialized or not model_state.simulator:
        # Return fallback data
        return generate_fallback_engine_data(engine_id, engine_num)
    
    try:
        # Get simulated data
        simulated_data = model_state.simulator.simulate_engine_degradation(engine_id)
        
        # Engine basic info
        latest_row = simulated_data.iloc[-1]
        health_index = latest_row["health_index"]
        risk_level, advisory = assess_risk_level(health_index)
        
        status = "nominal"
        if risk_level == "Red":
            status = "critical"
        elif risk_level == "Yellow":
            status = "warning"
        
        engine_info = EngineInfo(
            id=f"Engine {engine_id}",
            healthIndex=health_index,
            status=status,
            flightCycle=int(latest_row["time_cycle"]),
            rul=int(latest_row["predicted_RUL"]),
            advisory=advisory
        )
        
        # RUL Trend
        rul_trend = [
            RulTrendPoint(
                cycle=int(row["time_cycle"]),
                rul=int(row["predicted_RUL"])
            )
            for _, row in simulated_data.iterrows()
        ]
        
        # Sensor Data
        sensor_cols = [col for col in simulated_data.columns if 'temp' in col.lower() or 'pressure' in col.lower() or 'speed' in col.lower()]
        
        if sensor_cols:
            sensor_data = []
            for _, row in simulated_data.iterrows():
                point = SensorDataPoint(
                    cycle=int(row["time_cycle"]),
                    temperature=float(row.get(sensor_cols[0], 300)) if sensor_cols else 300.0,
                    pressure=float(row.get(sensor_cols[1], 100)) if len(sensor_cols) > 1 else 100.0,
                    fanSpeed=float(row.get(sensor_cols[2], 5000)) if len(sensor_cols) > 2 else 5000.0,
                    coreSpeed=float(row.get(sensor_cols[3], 10000)) if len(sensor_cols) > 3 else 10000.0,
                    vibration=float(row.get(sensor_cols[4], 0.01)) if len(sensor_cols) > 4 else 0.01,
                )
                sensor_data.append(point)
        else:
            sensor_data = []
            for i, row in simulated_data.iterrows():
                cycle = int(row["time_cycle"])
                base = 1 - (cycle / 300)
                sensor_data.append(SensorDataPoint(
                    cycle=cycle,
                    temperature=300 + 20 * base + np.random.randn() * 5,
                    pressure=100 + 10 * base + np.random.randn() * 2,
                    fanSpeed=5000 - 500 * base + np.random.randn() * 50,
                    coreSpeed=10000 - 1000 * base + np.random.randn() * 100,
                    vibration=0.01 + 0.02 * (1 - base) + np.random.randn() * 0.005,
                ))
        
        # Feature Importance (SHAP)
        if model_state.feature_cols and model_state.shap_explainer:
            latest_features = simulated_data[model_state.feature_cols].iloc[-1]
            shap_values = explain_single_prediction(model_state.shap_explainer, latest_features, model_state.feature_cols)
            
            abs_shap = shap_values.abs().sort_values(ascending=False)
            max_val = abs_shap.max() if abs_shap.max() > 0 else 1
            
            feature_importance = []
            for feature, value in abs_shap.head(10).items():
                normalized = value / max_val
                impact = "high" if normalized > 0.6 else "medium" if normalized > 0.3 else "low"
                feature_importance.append(FeatureImportancePoint(
                    name=feature[:20],
                    importance=normalized,
                    impact=impact
                ))
        else:
            feature_importance = [
                FeatureImportancePoint(name="Total Temperature", importance=0.85, impact="high"),
                FeatureImportancePoint(name="Pressure Ratio", importance=0.72, impact="high"),
                FeatureImportancePoint(name="Fan Speed", importance=0.58, impact="medium"),
                FeatureImportancePoint(name="Core Speed", importance=0.45, impact="medium"),
            ]
        
        # Degradation Data (Health Index over time)
        degradation_data = [
            DegradationDataPoint(
                cycle=int(row["time_cycle"]),
                health=float(row["health_index"])
            )
            for _, row in simulated_data.iterrows()
        ]
        
        return EngineDetails(
            engine=engine_info,
            rulTrend=rul_trend,
            sensorData=sensor_data,
            featureImportance=feature_importance,
            degradationData=degradation_data
        )
        
    except Exception as e:
        print(f"Error getting engine details: {e}")
        return generate_fallback_engine_data(engine_id, engine_num)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ZeroPing API",
        "ml_models_available": ML_MODELS_AVAILABLE,
        "initialized": model_state.initialized
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
