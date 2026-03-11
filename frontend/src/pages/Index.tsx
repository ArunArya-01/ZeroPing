import { useState, useEffect } from "react";
import { fetchEngines, type EngineData } from "@/lib/engine-data";
import { useEngineTelemetry } from "../hooks/useEngineTelemetry"; // <-- Our new WebSocket Hook
import NavBar from "@/components/dashboard/NavBar";
import EngineSelector from "@/components/dashboard/EngineSelector";
import HealthGauge from "@/components/dashboard/HealthGauge";
import RemainingLife from "@/components/dashboard/RemainingLife";
import RiskStatus from "@/components/dashboard/RiskStatus";
import SensorMonitoring from "@/components/dashboard/SensorMonitoring";
import ExplainableAI from "@/components/dashboard/ExplainableAI";
import DegradationSimulation from "@/components/dashboard/DegradationSimulation";
import SystemArchitecture from "@/components/dashboard/SystemArchitecture";

const Index = () => {
  const [engines, setEngines] = useState<EngineData[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<EngineData | null>(null);
  const [loading, setLoading] = useState(true);

  // 1. Initial REST API Load (Keeps your teammate's original functionality)
  useEffect(() => {
    const loadData = async () => {
      try {
        const engineList = await fetchEngines();
        setEngines(engineList);
        if (engineList.length > 0) {
          setSelectedEngine(engineList[0]);
        }
      } catch (err) {
        console.error("Failed to load engines:", err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // 2. Extract the numeric ID from the selected engine (e.g., "Engine 1" -> 1)
  const engineNumericId = selectedEngine 
    ? parseInt(selectedEngine.id.replace(/[^0-9]/g, "")) || 1 
    : 1;

  // 3. Connect to the FastAPI WebSocket pipeline!
  const { telemetryData, isConnected } = useEngineTelemetry(engineNumericId);

  // 4. The Magic Swap: Use live data if connected, otherwise use the initial REST data
  // We cast it to any here to seamlessly blend the WebSocket JSON with their EngineData type
  const displayEngine = (telemetryData as any) || selectedEngine;

  if (loading) {
    return (
      <div className="flex flex-col h-screen bg-background overflow-hidden">
        <NavBar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-muted-foreground">Loading engine data...</div>
        </div>
      </div>
    );
  }

  if (!selectedEngine || engines.length === 0) {
    return (
      <div className="flex flex-col h-screen bg-background overflow-hidden">
        <NavBar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-muted-foreground">No engines available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Optional: You can pass `isConnected` to NavBar later if you want a live green dot! */}
      <NavBar /> 
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column — Conclusions (sticky) */}
        <div className="w-80 shrink-0 p-4 space-y-4 overflow-y-auto border-r">
          <EngineSelector 
            engines={engines} 
            selectedEngine={selectedEngine} // Keep original here so the dropdown works normally
            onSelect={setSelectedEngine} 
          />
          {/* Feed the LIVE data into the visual components */}
          <HealthGauge engine={displayEngine} />
          <RemainingLife engine={displayEngine} />
          <RiskStatus engine={displayEngine} />
        </div>

        {/* Right Column — Evidence (scrollable) */}
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          {/* Feed the LIVE data into the charts */}
          <SensorMonitoring engine={displayEngine} />
          <ExplainableAI engine={displayEngine} />
          <DegradationSimulation engine={displayEngine} />
          <SystemArchitecture />
        </div>
      </div>
    </div>
  );
};

export default Index;