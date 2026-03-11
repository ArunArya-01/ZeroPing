import { useState, useEffect } from "react";
import { fetchEngines, type EngineData } from "@/lib/engine-data";
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
      <NavBar />
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column — Conclusions (sticky) */}
        <div className="w-80 shrink-0 p-4 space-y-4 overflow-y-auto border-r">
          <EngineSelector 
            engines={engines} 
            selectedEngine={selectedEngine} 
            onSelect={setSelectedEngine} 
          />
          <HealthGauge engine={selectedEngine} />
          <RemainingLife engine={selectedEngine} />
          <RiskStatus engine={selectedEngine} />
        </div>

        {/* Right Column — Evidence (scrollable) */}
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          <SensorMonitoring engine={selectedEngine} />
          <ExplainableAI engine={selectedEngine} />
          <DegradationSimulation engine={selectedEngine} />
          <SystemArchitecture />
        </div>
      </div>
    </div>
  );
};

export default Index;
