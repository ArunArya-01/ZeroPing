import { type EngineData, getStatusClass } from "@/lib/engine-data";

interface RiskStatusProps {
  engine: EngineData;
}

const RiskStatus = ({ engine }: RiskStatusProps) => {
  const statusLabel = engine.status === "nominal" ? "NORMAL" : engine.status === "warning" ? "DEGRADING" : "CRITICAL";

  return (
    <div className="panel">
      <p className="label-text mb-3">Risk Status</p>
      <p className={`data-value text-2xl font-bold ${getStatusClass(engine.status)}`}>
        {statusLabel}
      </p>
      <p className="text-sm text-muted-foreground mt-3 font-body leading-relaxed">
        {engine.advisory}
      </p>
    </div>
  );
};

export default RiskStatus;
