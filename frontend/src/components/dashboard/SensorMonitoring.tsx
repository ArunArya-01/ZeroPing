import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { getSensorData, type EngineData } from "@/lib/engine-data";

interface SensorMonitoringProps {
  engine: EngineData;
  sensorData?: { cycle: number; temperature: number; pressure: number; fanSpeed: number; coreSpeed: number; vibration: number }[];
}

const sensors = [
  { key: "temperature", label: "TEMPERATURE", unit: "°C" },
  { key: "pressure", label: "PRESSURE", unit: "PSI" },
  { key: "fanSpeed", label: "FAN SPEED", unit: "RPM" },
  { key: "coreSpeed", label: "CORE SPEED", unit: "RPM" },
  { key: "vibration", label: "VIBRATION", unit: "g" },
] as const;

const SensorMonitoring = ({ engine, sensorData: propSensorData }: SensorMonitoringProps) => {
  const [data, setData] = useState<{ cycle: number; temperature: number; pressure: number; fanSpeed: number; coreSpeed: number; vibration: number }[]>([]);
  const [activeSensor, setActiveSensor] = useState<string>("temperature");

  useEffect(() => {
    if (propSensorData && propSensorData.length > 0) {
      setData(propSensorData);
    } else {
      getSensorData(engine.id).then(setData);
    }
  }, [engine.id, propSensorData]);

  return (
    <div className="panel">
      <p className="label-text mb-4">Sensor Monitoring</p>
      <div className="flex gap-2 mb-4 flex-wrap">
        {sensors.map((s) => (
          <button
            key={s.key}
            onClick={() => setActiveSensor(s.key)}
            className={`px-3 py-1.5 rounded-sm font-mono text-[11px] transition-colors ${
              activeSensor === s.key
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis
              dataKey="cycle"
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={{ stroke: "hsl(222, 18%, 18%)" }}
              tickLine={false}
              label={{ value: "Cycle", position: "insideBottom", offset: -5, style: { fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" } }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "hsl(222, 18%, 30%)", fontFamily: "IBM Plex Mono" }}
              axisLine={false}
              tickLine={false}
              width={50}
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
              dataKey={activeSensor}
              stroke="hsl(153, 100%, 50%)"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SensorMonitoring;
