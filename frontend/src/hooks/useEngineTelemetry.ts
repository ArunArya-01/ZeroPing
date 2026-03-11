import { useState, useEffect } from 'react';

// Define the shape of our data based on the FastAPI backend
export interface EngineData {
    engine: {
        id: string;
        healthIndex: number;
        status: string;
        flightCycle: number;
        rul: number;
        advisory: string;
    };
}

export function useEngineTelemetry(engineId: number) {
    const [telemetryData, setTelemetryData] = useState<EngineData | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    useEffect(() => {
        const ws = new WebSocket(`ws://localhost:8000/ws/telemetry/${engineId}`);

        ws.onopen = () => {
            setIsConnected(true);
            console.log(`🟢 Connected to Engine ${engineId} telemetry`);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setTelemetryData(data);
        };

        ws.onerror = (error) => console.error("🔴 WebSocket error:", error);
        
        ws.onclose = () => {
            setIsConnected(false);
            console.log("WebSocket disconnected");
        };

        return () => ws.close();
    }, [engineId]);

    return { telemetryData, isConnected };
}