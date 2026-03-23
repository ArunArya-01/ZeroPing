/**
 * Engine Data API Client
 * Fetches engine data from the ZeroPing API backend
 */

const API_BASE_URL = "/api";

export interface EngineData {
  id: string;
  healthIndex: number;
  status: "nominal" | "warning" | "critical";
  flightCycle: number;
  rul: number;
  advisory: string;
}

export interface RulTrendPoint {
  cycle: number;
  rul: number;
}

export interface SensorDataPoint {
  cycle: number;
  temperature: number;
  pressure: number;
  fanSpeed: number;
  coreSpeed: number;
  vibration: number;
}

export interface FeatureImportancePoint {
  name: string;
  importance: number;
  impact: "high" | "medium" | "low";
}

export interface DegradationDataPoint {
  cycle: number;
  health: number;
}

// SHAP Visualization Interfaces
export interface FeatureContribution {
  name: string;
  value: number;
  impact: number;
  contribution: "positive" | "negative";
}

export interface WaterfallData {
  base_value: number;
  prediction: number;
  features: FeatureContribution[];
}

export interface DependencePoint {
  feature_value: number;
  shap_value: number;
  interaction_value: number;
}

export interface DependencePlotData {
  feature: string;
  interaction_feature: string;
  data_points: DependencePoint[];
}

export interface SummaryFeature {
  name: string;
  mean_impact: number;
  value_range: { min: number; max: number };
  shap_range: { min: number; max: number };
}

export interface SummaryPlotData {
  title: string;
  features: SummaryFeature[];
}

export interface DecisionStep {
  feature: string;
  value: number;
  contribution: number;
  cumulative: number;
}

export interface DecisionPath {
  prediction: number;
  steps: DecisionStep[];
}

export interface DecisionPlotData {
  base_value: number;
  paths: DecisionPath[];
}

export interface ShapVisualizationResponse {
  sample_index: number;
  timestamp: string;
  feature_importance: { [key: string]: number };
  waterfall?: WaterfallData;
  summary?: SummaryPlotData;
  dependence?: { [key: string]: DependencePlotData };
  decision?: DecisionPlotData;
}

export interface EngineDetails {
  engine: EngineData;
  rulTrend: RulTrendPoint[];
  sensorData: SensorDataPoint[];
  featureImportance: FeatureImportancePoint[];
  degradationData: DegradationDataPoint[];
}

// Cache for engine details
const engineDetailsCache: Map<string, EngineDetails> = new Map();
let enginesCache: EngineData[] = [];

// Fetch all available engines
export async function fetchEngines(): Promise<EngineData[]> {
  if (enginesCache.length > 0) {
    return enginesCache;
  }

  try {
    const response = await fetch(`${API_BASE_URL}/engines`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    // Ensure healthIndex is always within 0-100 range
    enginesCache = data.map((engine: EngineData) => ({
      ...engine,
      healthIndex: Math.max(0, Math.min(100, engine.healthIndex))
    }));
    return enginesCache;
  } catch (error) {
    console.error("Failed to fetch engines:", error);
    // Return fallback data if API is not available
    enginesCache = getFallbackEngines();
    return enginesCache;
  }
}

// Fetch detailed data for a specific engine
export async function fetchEngineDetails(engineId: string): Promise<EngineDetails | null> {
  const cacheKey = engineId;
  
  if (engineDetailsCache.has(cacheKey)) {
    return engineDetailsCache.get(cacheKey)!;
  }

  try {
    // Extract numeric ID from "Engine X" format
    const numericId = engineId.replace("Engine ", "");
    const response = await fetch(`${API_BASE_URL}/engines/${numericId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const details = await response.json();
    // Ensure healthIndex is always within 0-100 range
    if (details.engine) {
      details.engine.healthIndex = Math.max(0, Math.min(100, details.engine.healthIndex));
    }
    engineDetailsCache.set(cacheKey, details);
    return details;
  } catch (error) {
    console.error(`Failed to fetch engine details for ${engineId}:`, error);
    return null;
  }
}

// Helper functions for UI
export function getStatusColor(status: string): string {
  switch (status) {
    case "nominal":
      return "hsl(120, 100%, 35%)"; // Green
    case "warning":
      return "hsl(45, 100%, 50%)";  // Yellow/Amber
    case "critical":
      return "hsl(0, 100%, 50%)";  // Red
    default:
      return "hsl(220, 10%, 50%)"; // Gray
  }
}

export function getStatusClass(status: string): string {
  switch (status) {
    case "nominal":
      return "text-green-500";
    case "warning":
      return "text-yellow-500";
    case "critical":
      return "text-red-500";
    default:
      return "text-gray-500";
  }
}

// Get RUL trend data for an engine
export async function getRulTrend(engineId: string): Promise<RulTrendPoint[]> {
  const details = await fetchEngineDetails(engineId);
  if (!details) {
    return getFallbackRulTrend();
  }
  return details.rulTrend;
}

// Get sensor data for an engine
export async function getSensorData(engineId: string): Promise<SensorDataPoint[]> {
  const details = await fetchEngineDetails(engineId);
  if (!details) {
    return getFallbackSensorData();
  }
  return details.sensorData;
}

// Get feature importance data for an engine
export async function getFeatureImportance(engineId: string): Promise<FeatureImportancePoint[]> {
  const details = await fetchEngineDetails(engineId);
  if (!details) {
    return getFallbackFeatureImportance();
  }
  return details.featureImportance;
}

// Get degradation data for an engine
export async function getDegradationData(engineId: string): Promise<DegradationDataPoint[]> {
  const details = await fetchEngineDetails(engineId);
  if (!details) {
    return getFallbackDegradationData();
  }
  return details.degradationData;
}

// Get SHAP visualizations for an engine
export async function getShapVisualizations(engineId: string, sampleIndex: number = 0): Promise<ShapVisualizationResponse | null> {
  try {
    const numericId = engineId.replace("Engine ", "");
    const response = await fetch(`${API_BASE_URL}/shap/visualizations/${numericId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_index: sampleIndex,
        max_features: 15,
        include_force_plot: false,
        include_waterfall: true,
        include_dependence: true,
        include_decision: true,
        include_summary: true,
      }),
    });

    if (!response.ok) {
      console.warn(`Failed to fetch SHAP visualizations: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch SHAP visualizations for ${engineId}:`, error);
    return null;
  }
}

// Get summary plot data
export async function getSummaryPlot(maxFeatures: number = 15): Promise<SummaryPlotData | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/shap/summary-plot?max_features=${maxFeatures}`);
    if (!response.ok) {
      console.warn(`Failed to fetch summary plot: ${response.status}`);
      return null;
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch summary plot:", error);
    return null;
  }
}

// Get dependence plot for a specific feature
export async function getDependencePlot(engineId: string, featureName: string, maxDisplay: number = 100): Promise<DependencePlotData | null> {
  try {
    const numericId = engineId.replace("Engine ", "");
    const params = new URLSearchParams({
      engine_id: numericId,
      feature_name: featureName,
      max_display: maxDisplay.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/shap/dependence-plot?${params}`, {
      method: "POST",
    });

    if (!response.ok) {
      console.warn(`Failed to fetch dependence plot: ${response.status}`);
      return null;
    }

    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch dependence plot for ${featureName}:`, error);
    return null;
  }
}

// Fallback data when API is not available
function getFallbackEngines(): EngineData[] {
  return [
    { id: "Engine 1", healthIndex: 85.2, status: "nominal", flightCycle: 128, rul: 172, advisory: "Engine operating within normal parameters." },
    { id: "Engine 2", healthIndex: 62.5, status: "warning", flightCycle: 189, rul: 85, advisory: "Monitor closely - gradual degradation detected." },
    { id: "Engine 3", healthIndex: 38.7, status: "critical", flightCycle: 245, rul: 23, advisory: "Immediate maintenance recommended." },
    { id: "Engine 4", healthIndex: 78.4, status: "nominal", flightCycle: 156, rul: 144, advisory: "Engine operating within normal parameters." },
    { id: "Engine 5", healthIndex: 55.8, status: "warning", flightCycle: 201, rul: 98, advisory: "Monitor closely - some sensor readings abnormal." },
  ];
}

function getFallbackRulTrend(): RulTrendPoint[] {
  const data: RulTrendPoint[] = [];
  for (let i = 0; i <= 50; i++) {
    data.push({ cycle: i, rul: Math.max(0, 200 - i * 2) });
  }
  return data;
}

function getFallbackSensorData(): SensorDataPoint[] {
  const data: SensorDataPoint[] = [];
  for (let i = 0; i <= 50; i++) {
    const base = 1 - (i / 300);
    data.push({
      cycle: i,
      temperature: 300 + 20 * base + Math.random() * 5,
      pressure: 100 + 10 * base + Math.random() * 2,
      fanSpeed: 5000 - 500 * base + Math.random() * 50,
      coreSpeed: 10000 - 1000 * base + Math.random() * 100,
      vibration: 0.01 + 0.02 * (1 - base) + Math.random() * 0.005,
    });
  }
  return data;
}

function getFallbackFeatureImportance(): FeatureImportancePoint[] {
  return [
    { name: "Total Temperature", importance: 0.85, impact: "high" },
    { name: "Pressure Ratio", importance: 0.72, impact: "high" },
    { name: "Fan Speed", importance: 0.58, impact: "medium" },
    { name: "Core Speed", importance: 0.45, impact: "medium" },
    { name: "Oil Temp", importance: 0.32, impact: "low" },
    { name: "Vibration", importance: 0.25, impact: "low" },
    { name: "Fuel Flow", importance: 0.18, impact: "low" },
  ];
}

function getFallbackDegradationData(): DegradationDataPoint[] {
  const data: DegradationDataPoint[] = [];
  for (let i = 0; i <= 50; i++) {
    data.push({
      cycle: i,
      health: Math.max(0, 100 - (i * 0.5) + Math.random() * 2),
    });
  }
  return data;
}

// Export a default engines array for initial render
export const engines: EngineData[] = getFallbackEngines();
