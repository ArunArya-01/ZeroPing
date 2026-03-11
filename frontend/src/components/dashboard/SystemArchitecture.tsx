const steps = [
  { label: "SENSOR DATA", desc: "Real-time telemetry" },
  { label: "AI PREDICTION", desc: "Health & RUL model" },
  { label: "EXPLAINABLE AI", desc: "Feature attribution" },
  { label: "RISK SCORING", desc: "Classification" },
  { label: "DASHBOARD", desc: "Visualization" },
];

const SystemArchitecture = () => {
  return (
    <div className="panel">
      <p className="label-text mb-5">System Architecture Pipeline</p>
      <div className="flex items-center justify-between gap-1">
        {steps.map((step, i) => (
          <div key={step.label} className="flex items-center gap-1 flex-1">
            <div className="flex flex-col items-center text-center flex-1">
              <div className="w-full py-3 px-2 bg-muted rounded-sm">
                <p className="font-mono text-[10px] font-semibold text-foreground leading-tight">
                  {step.label}
                </p>
                <p className="font-mono text-[9px] text-muted-foreground mt-1">{step.desc}</p>
              </div>
            </div>
            {i < steps.length - 1 && (
              <span className="font-mono text-muted-foreground text-xs shrink-0">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SystemArchitecture;
