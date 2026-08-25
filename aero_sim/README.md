# AeroSim — Flight Fuel-Burn Simulation

Interactive 3D flight simulation showing an aircraft flying from one airport to
another, with **per-segment fuel burn predicted by the trained AeroTwin model**
running in the browser via ONNX Runtime Web. A Three.js tank overlay visualizes
fuel remaining in real time.

Stack: **CesiumJS** (globe, route, animated aircraft) + **Three.js** (fuel tank
HUD) + **onnxruntime-web** (model inference) + **Vite**.

## Features

- 3D globe with terrain, lighting, and a real great-circle-ish route
- Animated aircraft following the route; camera auto-follows
- Route polyline plus per-segment markers **colored by predicted fuel burn**
  (green = low, orange = medium, red = high) with fuel-kg labels
- HUD: progress %, fuel used, fuel remaining, distance, prediction engine
- Three.js fuel-tank gauge animating with remaining fuel
- **Two prediction engines:**
  1. **ONNX Runtime Web** — the real exported student model (`public/models/`)
  2. **Physics fallback** — used automatically when the ONNX files are absent

## Using the real model

The simulator auto-detects `public/models/large_mlp.onnx` +
`large_mlp.preproc.json`. See [`public/models/README.md`](public/models/README.md)
for how to export them from the project's distillation checkpoints
(`experiments/11_onnx_deploy/export_onnx.py`). Without them, the physics
fallback keeps the demo running.

## Project layout

```
aero_sim/
├── index.html                 # App shell + HUD
├── package.json
├── vite.config.js             # Vite + Cesium plugin
├── public/
│   └── models/                # Drop exported .onnx + preproc.json here
└── src/
    ├── main.js                # Cesium viewer, route, animation, wiring
    ├── fuel.js                # FuelPredictor: ONNX + physics fallback
    ├── threeScene.js          # Three.js fuel-gauge overlay
    └── data/
        └── routes.js          # Sample routes (EGLL→KJFK, KSFO→KORD)
```
