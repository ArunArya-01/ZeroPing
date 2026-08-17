# AeroTwin ONNX Inference Host (Rust)

A **compiled, language-neutral** inference service for AeroTwin distillation
students. Once a model is exported to ONNX, it runs as a single native binary
via [ONNX Runtime](https://onnxruntime.ai/) — no Python, no PyTorch, no
scikit-learn at inference time.

This makes the models **universal**: the exact same `*.onnx` artifact can be
served by any ONNX Runtime (Python, C++, Rust, JavaScript/Web, embedded, mobile)
and preserved independent of the training framework.

---

## Why ONNX + Rust?

| Concern | Python/PyTorch inference | This host |
|---|---|---|
| Runtime deps | torch, sklearn, numpy, venv | single native binary + model file |
| Latency | interpreter + tensor copies | compiled, zero-copy array input |
| Portability | platform-locked | cross-platform, framework-agnostic ONNX |
| Model durability | tied to checkpoints | open, self-describing `.onnx` |

The Rust server also reproduces the exact training-time preprocessing
(median impute → `StandardScaler` → `OneHotEncoder`) from a side-car
`*.preproc.json`, so raw feature rows map to the same model input the student
was trained on.

---

## Pipeline

```text
best_model.pt (PyTorch)
      │  export_onnx.py          (run once, where torch is installed)
      ▼
model.onnx  +  model.preproc.json +  model.meta.json
      │  cargo build --release   (once)
      ▼
aerotwin-onnx-serving  (native binary, axum HTTP server, ort/ONNX Runtime)
```

---

## 1. Export the model to ONNX

Run where PyTorch is installed (e.g. the repo training venv):

```bash
PYTHONPATH=src python experiments/11_onnx_deploy/export_onnx.py \
    --checkpoint results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt \
    --out models/onnx/large_mlp.onnx \
    --name large_mlp \
    --opset 17 \
    --preproc-distillation-parquet distillation_dataset.parquet
```

The `distillation_dataset.parquet` and `docs/reports/distillation_dataset_meta.json`
are required only for rebuilding the preprocessing JSON; they are the same files the
distillation step already uses. This emits:

- `models/onnx/large_mlp.onnx` — the exported graph (dynamic batch dimension).
- `models/onnx/large_mlp.preproc.json` — impute medians, scaler mean/scale, OHE categories.
- `models/onnx/large_mlp.meta.json` — input dimension and model metadata.

> Note: `*.onnx`, `*.pt`, `*.pkl`, `distillation_dataset.parquet`, and `cache/`
> are already git-ignored (regenerable artifacts).

## 2. Build the Rust server

Requires a Rust toolchain (`rustup`, [`install.cargo`](https://rustup.rs/)) — no
Python needed after this point.

```bash
cd rust/aerotwin-onnx-serving
cargo build --release
```

The first build downloads a prebuilt ONNX Runtime library via the `ort`
crate (`download-binaries` feature).

## 3. Run

```bash
AEROTWIN_MODEL=models/onnx/large_mlp.onnx \
AEROTWIN_PREPROC=models/onnx/large_mlp.preproc.json \
AEROTWIN_BIND=127.0.0.1:8080 \
./target/release/aerotwin-onnx-serving
```

Environment variables (all optional, with defaults):

| Variable | Default |
|---|---|
| `AEROTWIN_MODEL` | `models/onnx/large_mlp.onnx` |
| `AEROTWIN_PREPROC` | `<model>.preproc.json` (derived) |
| `AEROTWIN_BIND` | `127.0.0.1:8080` |

## 4. Use

Get the input contract:

```bash
curl http://127.0.0.1:8080/v1/meta
```

Predict fuel burn for one raw feature row. `numeric` is ordered as
`numeric_columns` from `/v1/meta` (use `null` for a missing value → column
median); `categories` is ordered as `categorical_columns`:

```bash
curl -X POST http://127.0.0.1:8080/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "rows": [
      {
        "numeric": [120.0, 0.4, ..., null],
        "categories": [
          "A320",          # aircraft_type
          "acars",         # method
          "EGLL",          # origin_icao
          "KJFK"           # destination_icao
        ]
      }
    ]
  }'
```

Response:

```json
{ "predictions_kg": [ 1234.56 ], "count": 1 }
```

For a whole batch, put multiple objects in the `rows` array; each row carries the
raw features and is preprocessed identically.

---

## Layout

```text
rust/aerotwin-onnx-serving/
├── Cargo.toml
└── src/
    ├── main.rs        # axum server, /v1/predict, /v1/meta, /healthz
    └── preproc.rs     # median-impute + scaler + one-hot, mirror of distillation/data.py
```

The `export_onnx.py` source lives at
`experiments/11_onnx_deploy/export_onnx.py`.

## Testing

- `/healthz` returns `{"status": "ok"}` for liveness.
- Compare a `/v1/predict` batch against the same rows through
  `DistillationData` in Python to confirm bit-level preprocessing parity before
  trusting the compiled path in production.
- Use `AEROTWIN_BIND=0.0.0.0:8080` to expose on a network (add TLS/auth for
  anything public; this server is unauthenticated by design).
