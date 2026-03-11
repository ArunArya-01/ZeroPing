import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { type EngineData, getRulTrend, getStatusColor } from "@/lib/engine-data";
import { useState, useEffect } from "react";

interface RemainingLifeProps {
  engine: EngineData;
  rulTrend?: { cycle: number; rul: number }[];
}

const RemainingLife = ({ engine, rulTrend }: RemainingLifeProps) => {
  const [trendData, setTrendData] = useState<{ cycle: number; rul: number }[]>([]);
  const color = getStatusColor(engine.status);

  useEffect(() => {
    if (rulTrend && rulTrend.length > 0) {
      setTrendData(rulTrend);
    } else {
      getRulTrend(engine.id).then(setTrendData);
    }
  }, [engine.id, rulTrend]);

  return (
    <div className="panel">
      <p className="label-text mb-2">Remaining Useful Life</p>
      <div className="flex items-baseline gap-2 mb-4">
        <span className="data-value text-3xl font-bold">{engine.rul}</span>
        <span className="label-text">Cycles</span>
      </div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trendData}>
            <XAxis
              dataKey="cycle"
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={{ stroke: "hsl(222, 18%, 18%)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={false}
              tickLine={false}
              width={30}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(230, 25%, 9%)",
                border: "1px solid hsl(222, 18%, 18%)",
                borderRadius: "2px",
                fontFamily: "IBM Plex Mono",
                fontSize: "11px",
              }}
              labelStyle={{ color: "hsl(222, 18%, 30%)" }}
            />
            <Line
              type="monotone"
              dataKey="rul"
              stroke={color}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default RemainingLife;
