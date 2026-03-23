import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell, ComposedChart, Line, ScatterChart, Scatter } from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getFeatureImportance, getShapVisualizations, getSummaryPlot, getDependencePlot, type EngineData, type ShapVisualizationResponse, type SummaryPlotData } from "@/lib/engine-data";

interface ExplainableAIProps {
  engine: EngineData;
  featureImportance?: { name: string; importance: number; impact: string }[];
}

const impactColor = (impact: string | number): string => {
  if (typeof impact === "number") {
    const absImpact = Math.abs(impact);
    if (absImpact > 0.6) return "hsl(349, 100%, 50%)";
    if (absImpact > 0.3) return "hsl(53, 100%, 50%)";
    return "hsl(222, 18%, 30%)";
  }
  switch (impact) {
    case "high":
      return "hsl(349, 100%, 50%)";
    case "medium":
      return "hsl(53, 100%, 50%)";
    default:
      return "hsl(222, 18%, 30%)";
  }
};

const WaterfallPlot = ({ data }: { data?: any }) => {
  if (!data?.features) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No waterfall data available
      </div>
    );
  }

  const chartData = [
    { name: "Base Value", value: data.base_value },
    ...data.features.map((f: any) => ({
      name: f.name,
      value: f.value,
      contribution: f.contribution,
      impact: f.impact,
    })),
    { name: "Prediction", value: data.prediction },
  ];

  return (
    <div className="h-96">
      <div className="text-xs text-muted-foreground mb-3">
        <p>Base: {data.base_value.toFixed(3)} → Prediction: {data.prediction.toFixed(3)}</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} layout="vertical" margin={{ left: 120 }}>
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(230, 25%, 9%)",
              border: "1px solid hsl(222, 18%, 18%)",
              borderRadius: "2px",
            }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="value" barSize={20}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={impactColor(entry.impact || entry.contribution)} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

const SummaryPlot = ({ data }: { data?: SummaryPlotData }) => {
  if (!data?.features || data.features.length === 0) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No summary data available
      </div>
    );
  }

  const chartData = data.features.map((f) => ({
    name: f.name,
    meanImpact: Math.abs(f.mean_impact),
    range: f.shap_range.max - f.shap_range.min,
  }));

  return (
    <div className="h-96">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 120 }} >
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(230, 25%, 9%)",
              border: "1px solid hsl(222, 18%, 18%)",
              borderRadius: "2px",
            }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="meanImpact" fill="hsl(53, 100%, 50%)" barSize={16} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const DependencePlot = ({ data }: { data?: any }) => {
  if (!data?.data_points || data.data_points.length === 0) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No dependence data available
      </div>
    );
  }

  return (
    <div className="h-96">
      <div className="text-xs text-muted-foreground mb-3">
        <p>Feature: {data.feature} | Interaction: {data.interaction_feature}</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ left: 80, right: 20 }}>
          <XAxis dataKey="feature_value" type="number" tick={{ fontSize: 10 }} />
          <YAxis dataKey="shap_value" type="number" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(230, 25%, 9%)",
              border: "1px solid hsl(222, 18%, 18%)",
              borderRadius: "2px",
            }}
            cursor={{ strokeDasharray: "3 3" }}
          />
          <Scatter name={data.feature} data={data.data_points} fill="hsl(53, 100%, 50%)" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

const DecisionPlot = ({ data }: { data?: any }) => {
  if (!data?.paths || data.paths.length === 0) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No decision data available
      </div>
    );
  }

  const firstPath = data.paths[0];
  if (!firstPath?.steps || firstPath.steps.length === 0) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No decision steps available
      </div>
    );
  }

  const chartData = firstPath.steps.map((step: any) => ({
    name: step.feature,
    cumulative: step.cumulative,
    contribution: step.contribution,
  }));

  return (
    <div className="h-96">
      <div className="text-xs text-muted-foreground mb-3">
        <p>Base: {data.base_value.toFixed(3)} → Prediction: {firstPath.prediction.toFixed(3)}</p>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} layout="vertical" margin={{ left: 120 }}>
          <XAxis type="number" />
          <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(230, 25%, 9%)",
              border: "1px solid hsl(222, 18%, 18%)",
              borderRadius: "2px",
            }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="cumulative" barSize={20}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.contribution >= 0 ? "hsl(120, 100%, 35%)" : "hsl(349, 100%, 50%)"} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

const FeatureImportanceChart = ({ data }: { data?: any[] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-56 text-muted-foreground">
        No feature importance data available
      </div>
    );
  }

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 85 }}>
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10 }} axisLine={{ stroke: "hsl(222, 18%, 18%)" }} tickLine={false} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "hsl(220, 20%, 90%)" }} axisLine={false} tickLine={false} width={85} />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(230, 25%, 9%)",
              border: "1px solid hsl(222, 18%, 18%)",
              borderRadius: "2px",
            }}
            formatter={(value: number) => [(value * 100).toFixed(0) + "%", "Importance"]}
          />
          <Bar dataKey="importance" radius={[0, 2, 2, 0]} barSize={16}>
            {data.map((entry, index) => (
              <Cell key={index} fill={impactColor(entry.impact)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const getFallbackShapData = (): ShapVisualizationResponse => ({
  sample_index: 0,
  timestamp: new Date().toISOString(),
  feature_importance: {
    "Total Temperature": 0.85,
    "Pressure Ratio": 0.72,
    "Fan Speed": 0.58,
    "Core Speed": 0.45,
    "Oil Temp": 0.32,
    "Vibration": 0.25,
    "Fuel Flow": 0.18,
  },
  waterfall: {
    base_value: 0.45,
    prediction: 0.72,
    features: [
      { name: "Total Temperature", value: 0.32, impact: 0.32, contribution: "positive" },
      { name: "Pressure Ratio", value: 0.18, impact: 0.18, contribution: "positive" },
      { name: "Fan Speed", value: -0.12, impact: -0.12, contribution: "negative" },
      { name: "Core Speed", value: 0.08, impact: 0.08, contribution: "positive" },
      { name: "Oil Temp", value: -0.05, impact: -0.05, contribution: "negative" },
      { name: "Vibration", value: -0.14, impact: -0.14, contribution: "negative" },
    ],
  },
  summary: {
    title: "SHAP Summary Plot (Mean |SHAP value|)",
    features: [
      { name: "Total Temperature", mean_impact: 0.32, value_range: { min: 250, max: 350 }, shap_range: { min: -0.5, max: 0.8 } },
      { name: "Pressure Ratio", mean_impact: 0.25, value_range: { min: 80, max: 120 }, shap_range: { min: -0.4, max: 0.5 } },
      { name: "Fan Speed", mean_impact: 0.18, value_range: { min: 4000, max: 5500 }, shap_range: { min: -0.3, max: 0.4 } },
      { name: "Core Speed", mean_impact: 0.15, value_range: { min: 8000, max: 10500 }, shap_range: { min: -0.25, max: 0.35 } },
      { name: "Oil Temp", mean_impact: 0.12, value_range: { min: 280, max: 320 }, shap_range: { min: -0.2, max: 0.3 } },
    ],
  },
  dependence: {
    "Total Temperature": {
      feature: "Total Temperature",
      interaction_feature: "Pressure Ratio",
      data_points: Array.from({ length: 50 }, (_, i) => ({
        feature_value: 250 + i * 2,
        shap_value: Math.sin(i * 0.2) * 0.3 + (i / 50) * 0.2,
        interaction_value: 80 + Math.random() * 40,
      })),
    },
    "Pressure Ratio": {
      feature: "Pressure Ratio",
      interaction_feature: "Fan Speed",
      data_points: Array.from({ length: 50 }, (_, i) => ({
        feature_value: 80 + i * 0.8,
        shap_value: Math.cos(i * 0.15) * 0.25 - (i / 50) * 0.15,
        interaction_value: 4000 + Math.random() * 1500,
      })),
    },
  },
  decision: {
    base_value: 0.45,
    paths: [
      {
        prediction: 0.72,
        steps: [
          { feature: "Base Value", value: 0.45, contribution: 0.45, cumulative: 0.45 },
          { feature: "Total Temperature", value: 310, contribution: 0.32, cumulative: 0.77 },
          { feature: "Pressure Ratio", value: 105, contribution: 0.18, cumulative: 0.95 },
          { feature: "Fan Speed", value: 4800, contribution: -0.12, cumulative: 0.83 },
          { feature: "Core Speed", value: 9500, contribution: 0.08, cumulative: 0.91 },
          { feature: "Oil Temp", value: 295, contribution: -0.05, cumulative: 0.86 },
          { feature: "Vibration", value: 0.015, contribution: -0.14, cumulative: 0.72 },
        ],
      },
    ],
  },
});

const getFallbackSummaryData = (): SummaryPlotData => ({
  title: "SHAP Summary Plot (Mean |SHAP value|)",
  features: [
    { name: "Total Temperature", mean_impact: 0.32, value_range: { min: 250, max: 350 }, shap_range: { min: -0.5, max: 0.8 } },
    { name: "Pressure Ratio", mean_impact: 0.25, value_range: { min: 80, max: 120 }, shap_range: { min: -0.4, max: 0.5 } },
    { name: "Fan Speed", mean_impact: 0.18, value_range: { min: 4000, max: 5500 }, shap_range: { min: -0.3, max: 0.4 } },
    { name: "Core Speed", mean_impact: 0.15, value_range: { min: 8000, max: 10500 }, shap_range: { min: -0.25, max: 0.35 } },
    { name: "Oil Temp", mean_impact: 0.12, value_range: { min: 280, max: 320 }, shap_range: { min: -0.2, max: 0.3 } },
  ],
});

const ExplainableAI = ({ engine, featureImportance: propFeatureImportance }: ExplainableAIProps) => {
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [shapData, setShapData] = useState<ShapVisualizationResponse | null>(null);
  const [summaryData, setSummaryData] = useState<SummaryPlotData | null>(null);
  const [activeDependenceFeature, setActiveDependenceFeature] = useState<string>("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        if (propFeatureImportance && propFeatureImportance.length > 0) {
          setFeatureImportance(propFeatureImportance);
        } else {
          const importance = await getFeatureImportance(engine.id);
          setFeatureImportance(importance.length > 0 ? importance : [
            { name: "Total Temperature", importance: 0.85, impact: "high" },
            { name: "Pressure Ratio", importance: 0.72, impact: "high" },
            { name: "Fan Speed", importance: 0.58, impact: "medium" },
            { name: "Core Speed", importance: 0.45, impact: "medium" },
            { name: "Oil Temp", importance: 0.32, impact: "low" },
            { name: "Vibration", importance: 0.25, impact: "low" },
            { name: "Fuel Flow", importance: 0.18, impact: "low" },
          ]);
        }

        const shap = await getShapVisualizations(engine.id);
        if (shap && (shap.waterfall || shap.summary || shap.dependence || shap.decision)) {
          setShapData(shap);
          if (shap.dependence && Object.keys(shap.dependence).length > 0) {
            setActiveDependenceFeature(Object.keys(shap.dependence)[0]);
          }
        } else {
          const fallbackData = getFallbackShapData();
          setShapData(fallbackData);
          setActiveDependenceFeature("Total Temperature");
        }

        const summary = await getSummaryPlot();
        if (summary && summary.features && summary.features.length > 0) {
          setSummaryData(summary);
        } else {
          setSummaryData(getFallbackSummaryData());
        }
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [engine.id, propFeatureImportance]);

  return (
    <div className="panel">
      <p className="label-text mb-4">Explainable AI — SHAP Visualizations</p>

      {loading && (
        <div className="flex items-center justify-center h-56 text-muted-foreground">
          Loading visualizations...
        </div>
      )}

      {!loading && (
        <Tabs defaultValue="importance" className="w-full">
          <TabsList className="grid w-full grid-cols-5 mb-4 bg-transparent" style={{ borderBottom: "1px solid hsl(222, 18%, 18%)" }}>
            <TabsTrigger value="importance" className="text-xs">Feature Importance</TabsTrigger>
            <TabsTrigger value="waterfall" className="text-xs">Waterfall Plot</TabsTrigger>
            <TabsTrigger value="summary" className="text-xs">Summary Plot</TabsTrigger>
            <TabsTrigger value="dependence" className="text-xs">Dependence</TabsTrigger>
            <TabsTrigger value="decision" className="text-xs">Decision Path</TabsTrigger>
          </TabsList>

          <TabsContent value="importance" className="mt-0">
            <FeatureImportanceChart data={featureImportance} />
            <div className="flex gap-6 mt-3">
              {["high", "medium", "low"].map((level) => (
                <div key={level} className="flex items-center gap-2">
                  <div className="w-2 h-2" style={{ backgroundColor: impactColor(level) }} />
                  <span className="font-mono text-[10px] text-muted-foreground uppercase">{level}</span>
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="waterfall" className="mt-0">
            <WaterfallPlot data={shapData?.waterfall} />
          </TabsContent>

          <TabsContent value="summary" className="mt-0">
            <SummaryPlot data={summaryData || shapData?.summary} />
          </TabsContent>

          <TabsContent value="dependence" className="mt-0">
            {shapData?.dependence && Object.keys(shapData.dependence).length > 0 ? (
              <>
                <div className="mb-3 flex gap-2 flex-wrap">
                  {Object.keys(shapData.dependence).map((feature) => (
                    <button
                      key={feature}
                      onClick={() => setActiveDependenceFeature(feature)}
                      className="px-3 py-1 rounded text-xs font-mono transition-colors"
                      style={{
                        backgroundColor: activeDependenceFeature === feature ? "hsl(53, 100%, 50%)" : "hsl(222, 18%, 18%)",
                        color: activeDependenceFeature === feature ? "hsl(230, 25%, 9%)" : "hsl(220, 20%, 90%)",
                      }}
                      onMouseEnter={(e) => {
                        if (activeDependenceFeature !== feature) {
                          (e.target as HTMLButtonElement).style.backgroundColor = "hsl(222, 18%, 25%)";
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (activeDependenceFeature !== feature) {
                          (e.target as HTMLButtonElement).style.backgroundColor = "hsl(222, 18%, 18%)";
                        }
                      }}
                    >
                      {feature}
                    </button>
                  ))}
                </div>
                <DependencePlot data={shapData.dependence[activeDependenceFeature]} />
              </>
            ) : (
              <div className="flex items-center justify-center h-56 text-muted-foreground">
                No dependence data available
              </div>
            )}
          </TabsContent>

          <TabsContent value="decision" className="mt-0">
            <DecisionPlot data={shapData?.decision} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default ExplainableAI;
