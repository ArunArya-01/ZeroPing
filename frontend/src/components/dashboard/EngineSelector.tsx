import { type EngineData, getStatusClass } from "@/lib/engine-data";

interface EngineSelectorProps {
  engines: EngineData[];
  selectedEngine: EngineData;
  onSelect: (engine: EngineData) => void;
}

const EngineSelector = ({ engines, selectedEngine, onSelect }: EngineSelectorProps) => {
  return (
    <div className="panel">
      <p className="label-text mb-3">Engine Selection</p>
      <div className="space-y-2">
        {engines.map((engine) => (
          <button
            key={engine.id}
            onClick={() => onSelect(engine)}
            className={`w-full text-left px-3 py-2.5 rounded-sm font-mono text-sm transition-colors ${
              selectedEngine.id === engine.id
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
            }`}
          >
            <div className="flex items-center justify-between">
              <span>{engine.id}</span>
              <span className={`text-xs ${getStatusClass(engine.status)}`}>
                {engine.status.toUpperCase()}
              </span>
            </div>
            <div className="text-[11px] text-muted-foreground mt-0.5">
              Cycle {engine.flightCycle}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default EngineSelector;
