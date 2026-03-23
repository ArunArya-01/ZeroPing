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
        // Load feature importance
        if (propFeatureImportance && propFeatureImportance.length > 0) {
          setFeatureImportance(propFeatureImportance);
        } else {
          const importance = await getFeatureImportance(engine.id);
          setFeatureImportance(importance);
        }

        // Load SHAP visualizations
        const shap = await getShapVisualizations(engine.id);
        if (shap) {
          setShapData(shap);
          if (shap.dependence) {
            const firstFeature = Object.keys(shap.dependence)[0];
            setActiveDependenceFeature(firstFeature);
          }
        }

        // Load summary plot
        const summary = await getSummaryPlot();
        if (summary) {
          setSummaryData(summary);
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
