import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { getFeatureImportance, type EngineData } from "@/lib/engine-data";

interface ExplainableAIProps {
  engine: EngineData;
  featureImportance?: { name: string; importance: number; impact: string }[];
}

const impactColor = (impact: string) => {
  switch (impact) {
    case "high": return "hsl(349, 100%, 50%)";
    case "medium": return "hsl(53, 100%, 50%)";
    default: return "hsl(222, 18%, 30%)";
  }
};

const ExplainableAI = ({ engine, featureImportance: propFeatureImportance }: ExplainableAIProps) => {
  const [data, setData] = useState<{ name: string; importance: number; impact: string }[]>([]);

  useEffect(() => {
    if (propFeatureImportance && propFeatureImportance.length > 0) {
      setData(propFeatureImportance);
    } else {
      getFeatureImportance(engine.id).then(setData);
    }
  }, [engine.id, propFeatureImportance]);

  return (
    <div className="panel">
      <p className="label-text mb-4">Explainable AI — Feature Importance</p>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 10 }}>
            <XAxis
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={{ stroke: "hsl(222, 18%, 18%)" }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 11, fill: "hsl(220, 20%, 90%)", fontFamily: "IBM Plex Mono" }}
              axisLine={false}
              tickLine={false}
              width={85}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(230, 25%, 9%)",
                border: "1px solid hsl(222, 18%, 18%)",
                borderRadius: "2px",
                fontFamily: "IBM Plex Mono",
                fontSize: "11px",
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
      <div className="flex gap-6 mt-3">
        {["high", "medium", "low"].map((level) => (
          <div key={level} className="flex items-center gap-2">
            <div className="w-2 h-2" style={{ backgroundColor: impactColor(level) }} />
            <span className="font-mono text-[10px] text-muted-foreground uppercase">{level}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExplainableAI;
