//! Preprocessing mirror of `aerotwin/distillation/data.py` so inference is
//! end-to-end in Rust with no scikit-learn dependency.
//!
//! The companion `*.preproc.json` emitted by `export_onnx.py` carries every
//! learned parameter (median impute, StandardScaler mean/scale, OneHotEncoder
//! categories). This module applies exactly the same transform so that the
//! tensor fed to ONNX matches the training-time encoding.

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Preproc {
    pub numeric_columns: Vec<String>,
    pub categorical_columns: Vec<String>,
    #[serde(default)]
    pub median_impute: Vec<f64>,
    #[serde(default)]
    pub scaler_mean: Vec<f64>,
    #[serde(default)]
    pub scaler_scale: Vec<f64>,
    pub onehot_categories: Vec<Vec<String>>,
    #[serde(default)]
    pub n_numeric: usize,
    #[serde(default)]
    pub ohe_start: usize,
}

impl Preproc {
    /// Load from the JSON produced by `export_onnx.py --preproc-distillation-parquet`.
    pub fn from_json(json: &str) -> Result<Self> {
        let p: Preproc = serde_json::from_str(json)?;
        if p.numeric_columns.len() != p.median_impute.len()
            || p.numeric_columns.len() != p.scaler_mean.len()
            || p.numeric_columns.len() != p.scaler_scale.len()
            || (p.n_numeric != 0 && p.n_numeric != p.numeric_columns.len())
            || (p.ohe_start != 0 && p.ohe_start != p.numeric_columns.len())
        {
            anyhow::bail!(
                "preproc length mismatch: numeric={} impute={} mean={} scale={} n_numeric={} ohe_start={}",
                p.numeric_columns.len(),
                p.median_impute.len(),
                p.scaler_mean.len(),
                p.scaler_scale.len(),
                p.n_numeric,
                p.ohe_start,
            );
        }
        Ok(p)
    }

    pub fn in_dim(&self) -> usize {
        let ohe_dim: usize = self.onehot_categories.iter().map(|c| c.len()).sum();
        self.numeric_columns.len() + ohe_dim
    }
}

/// A single row of raw, pre-encoding features (matching the training columns).
///
/// Serialized as:
/// ```json
/// {
///   "visual": null,
///   "numeric": [120.0, 0.4, null, ...],          // ordered by numeric_columns
///   "categories": ["A320", "acars", "EGLL", "KJFK"] // ordered by categorical_columns
/// }
/// ```
/// Use `null` in `numeric` for a missing value (encoded as the column median).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawRow {
    pub numeric: Vec<Option<f64>>,
    pub categories: Vec<String>,
}

impl RawRow {
    /// Encode into the model's input vector.
    ///
    /// Numeric: null/non-finite -> column median, then (x - mean) / scale.
    /// Categorical: unknown label -> all-zero OHE slice (handle_unknown="ignore").
    pub fn encode(&self, p: &Preproc) -> Vec<f32> {
        let n_num = p.numeric_columns.len();
        let mut out = Vec::with_capacity(p.in_dim());

        for j in 0..n_num {
            let value = self.numeric.get(j).copied().flatten();
            let x = match value {
                Some(v) if v.is_finite() => v,
                _ => p.median_impute.get(j).copied().unwrap_or(0.0),
            };
            let mean = p.scaler_mean.get(j).copied().unwrap_or(0.0);
            let scale = p.scaler_scale.get(j).copied().unwrap_or(1.0);
            let normalized = if scale.abs() < f64::EPSILON {
                0.0
            } else {
                (x - mean) / scale
            };
            out.push(normalized as f32);
        }

        for (cat_idx, categories) in p.onehot_categories.iter().enumerate() {
            let label = self.categories.get(cat_idx).cloned().unwrap_or_default();
            let matched = categories.iter().position(|cat| cat == &label);
            if let Some(k) = matched {
                for i in 0..categories.len() {
                    out.push(if i == k { 1.0 } else { 0.0 });
                }
            } else {
                // handle_unknown="ignore" -> all-zero slice
                for _ in 0..categories.len() {
                    out.push(0.0);
                }
            }
        }
        debug_assert_eq!(out.len(), p.in_dim());
        out
    }
}
