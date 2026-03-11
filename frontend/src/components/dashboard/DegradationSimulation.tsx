import { useState, useEffect } from "react";
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip, ReferenceLine } from "recharts";
import { getDegradationData, getStatusColor, type EngineData } from "@/lib/engine-data";

interface DegradationSimulationProps {
  engine: EngineData;
  degradationData?: { cycle: number; health: number }[];
}

const DegradationSimulation = ({ engine, degradationData: propDegradationData }: DegradationSimulationProps) => {
  const [data, setData] = useState<{ cycle: number; health: number }[]>([]);
  const color = getStatusColor(engine.status);

  useEffect(() => {
    if (propDegradationData && propDegradationData.length > 0) {
      setData(propDegradationData);
    } else {
      getDegradationData(engine.id).then(setData);
    }
  }, [engine.id, propDegradationData]);

  return (
    <div className="panel">
      <p className="label-text mb-4">Engine Degradation Simulation</p>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.15} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="cycle"
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={{ stroke: "hsl(222, 18%, 18%)" }}
              tickLine={false}
              label={{ value: "Flight Cycles", position: "insideBottom", offset: -5, style: { fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" } }}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={false}
              tickLine={false}
              width={30}
              label={{ value: "Health %", angle: -90, position: "insideLeft", style: { fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" } }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(230, 25%, 9%)",
                border: "1px solid hsl(222, 18%, 18%)",
                borderRadius: "2px",
                fontFamily: "IBM Plex Mono",
                fontSize: "11px",
              }}
            />
            <ReferenceLine y={70} stroke="hsl(53, 100%, 50%)" strokeDasharray="4 4" strokeOpacity={0.5} />
            <ReferenceLine y={40} stroke="hsl(349, 100%, 50%)" strokeDasharray="4 4" strokeOpacity={0.5} />
            <ReferenceLine x={engine.flightCycle} stroke="hsl(220, 20%, 90%)" strokeDasharray="4 4" strokeOpacity={0.3} />
            <Area
              type="monotone"
              dataKey="health"
              stroke={color}
              strokeWidth={2}
              fill="url(#healthGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-6 mt-3">
        <div className="flex items-center gap-2">
          <div className="w-4 border-t-2 border-dashed" style={{ borderColor: "hsl(53, 100%, 50%)" }} />
          <span className="font-mono text-[10px] text-muted-foreground">WARNING THRESHOLD</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 border-t-2 border-dashed" style={{ borderColor: "hsl(349, 100%, 50%)" }} />
          <span className="font-mono text-[10px] text-muted-foreground">CRITICAL THRESHOLD</span>
        </div>
      </div>
    </div>
  );
};

export default DegradationSimulation;
