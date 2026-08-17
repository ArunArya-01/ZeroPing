//! Run AeroTwin ONNX models as a compiled, language-neutral inference service.
//!
//! The server loads an exported `*.onnx` (from `export_onnx.py`) plus its
//! `*.preproc.json`, and exposes:
//!   * `POST /v1/predict`  — JSON row(s) -> fuel_kg prediction(s)
//!   * `GET  /v1/meta`     — model input contract
//!   * `GET  /healthz`     — liveness probe
//!
//! This replaces a Python/PyTorch inference path with a single native binary
//! using ONNX Runtime (the `ort` crate). The model itself is framework-agnostic
//! ONNX, so the same artifact can also be served by any ONNX runtime elsewhere.

mod preproc;

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use preproc::{Preproc, RawRow};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct AppState {
    session: Arc<Mutex<Session>>,
    preproc: Arc<Preproc>,
}

#[derive(Debug, Deserialize)]
struct PredictBody {
    rows: Vec<RawRow>,
}

async fn predict(State(state): State<AppState>, Json(body): Json<PredictBody>) -> (StatusCode, Json<Value>) {
    let n: usize = body.rows.len();
    if n == 0 {
        return (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"error": "no rows"})));
    }

    let in_dim = state.preproc.in_dim();
    let mut flat = Vec::with_capacity(n * in_dim);
    for row in &body.rows {
        flat.extend(row.encode(&state.preproc));
    }
    let arr = match Array2::from_shape_vec((n, in_dim), flat) {
        Ok(a) => a,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))),
    };

    let input = match TensorRef::from_array_view(&arr) {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    };

    let mut guard = match state.session.lock() {
        Ok(g) => g,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    };
    let outputs = match guard.run(ort::inputs![input]) {
        Ok(out) => out,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    };
    let (_, data) = match outputs[0].try_extract_tensor::<f32>() {
        Ok(d) => d,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    };
    let preds: Vec<f64> = data.iter().map(|x| *x as f64).collect();

    (StatusCode::OK, Json(json!({ "predictions_kg": preds, "count": n })))
}

async fn meta(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "input_dim": state.preproc.in_dim(),
        "numeric_columns": state.preproc.numeric_columns,
        "categorical_columns": state.preproc.categorical_columns,
        "onehot_categories": state.preproc.onehot_categories,
        "model_output": "fuel_kg",
    }))
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model_path = std::env::var("AEROTWIN_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/onnx/large_mlp.onnx"));
    let preproc_path = std::env::var("AEROTWIN_PREPROC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| preproc_path_for(&model_path));

    tracing::info!("loading model from {}", model_path.display());
    if !model_path.exists() {
        anyhow::bail!(
            "model not found at {}; export it first with export_onnx.py",
            model_path.display()
        );
    }

    let preproc = Preproc::from_json(
        &std::fs::read_to_string(&preproc_path)
            .map_err(|e| anyhow::anyhow!("cannot read preproc {}: {e}", preproc_path.display()))?,
    )?;
    tracing::info!("preproc input_dim = {}", preproc.in_dim());

    let session = Session::builder()?.commit_from_file(model_path)?;
    let state = AppState {
        session: Arc::new(Mutex::new(session)),
        preproc: Arc::new(preproc),
    };

    let app = Router::new()
        .route("/v1/predict", axum::routing::post(predict))
        .route("/v1/meta", get(meta))
        .route("/healthz", get(healthz))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state);

    let bind = std::env::var("AEROTWIN_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!("AeroTwin ONNX server listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn preproc_path_for(model: &std::path::Path) -> PathBuf {
    model.with_extension("preproc.json")
}
