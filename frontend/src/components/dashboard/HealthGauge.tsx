import { type EngineData, getStatusColor, getStatusClass } from "@/lib/engine-data";

interface HealthGaugeProps {
  engine: EngineData;
}

const HealthGauge = ({ engine }: HealthGaugeProps) => {
  const radius = 80;
  const stroke = 8;
  const circumference = 2 * Math.PI * radius;
  const progress = (engine.healthIndex / 100) * circumference;
  const color = getStatusColor(engine.status);

  return (
    <div className="panel flex flex-col items-center py-8">
      <p className="label-text mb-6">Engine Health Index</p>
      <div className="relative">
        <svg width="200" height="200" viewBox="0 0 200 200">
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth={stroke}
          />
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            strokeLinecap="butt"
            transform="rotate(-90 100 100)"
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`data-value text-5xl font-bold ${getStatusClass(engine.status)}`}>
            {engine.healthIndex}
          </span>
          <span className="label-text mt-1">/ 100</span>
        </div>
      </div>
      <p className="label-text mt-4">{engine.id} — CYCLE {engine.flightCycle}</p>
    </div>
  );
};

export default HealthGauge;
