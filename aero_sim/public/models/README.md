# Browser Models

Drop the exported AeroTwin model here to run the simulator on the real trained
student (ONNX Runtime Web). Expected files:

```
public/models/large_mlp.onnx
public/models/large_mlp.preproc.json
```

## How to produce them

From the project root, on a machine with PyTorch installed:

```bash
PYTHONPATH=src python experiments/11_onnx_deploy/export_onnx.py \
  --checkpoint results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt \
  --out models/onnx/large_mlp.onnx \
  --name large_mlp \
  --preproc-distillation-parquet distillation_dataset.parquet

cp models/onnx/large_mlp.onnx aero_sim/public/models/
cp models/onnx/large_mlp.preproc.json aero_sim/public/models/
```

The `fuel.js` module in `src/` builds the model input row from the segment
geometry (altitude, speed, duration) and fills the remaining feature columns with
the preproc medians. When the model files are absent, the simulator falls back
to a simple physics approximation so the demo always runs.
