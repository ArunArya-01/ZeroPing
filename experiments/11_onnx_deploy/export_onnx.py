"""Step 11.1 — Export a trained distillation student (MLP) to a portable ONNX model.

Best feature of a compiled deployment: once exported, the model runs anywhere —
Python, Rust, JavaScript (ONNX Runtime Web), embedded — with no PyTorch dependency
and a single forward pass.

This script:
  1. Loads a ``best_model.pt`` checkpoint produced by the distillation pipeline
     (see ``experiments/08_distillation/02_train_mlp_student.py``).
  2. Reconstructs the :class:`aerotwin.distillation.mlp.StudentMLP` architecture
     from the in_dim / hidden_dims stored in the checkpoint.
  3. Optionally rebuilds the preprocessing (median impute + StandardScaler +
     OneHotEncoder) from ``distillation_dataset.parquet`` and writes it to a
     companion ``preproc.json`` so inference is end-to-end in any runtime.
  4. Exports the fused graph to ``<name>.onnx`` (Float32, batch dimension
     dynamic). Output opset 17.

The saved checkpoint carries ``in_dim`` and ``hidden_dims``; dropout is applied
only during training and is dropped on export (already ``model.eval()``).

Run (from repo root, in the venv that has torch):
    PYTHONPATH=src python experiments/11_onnx_deploy/export_onnx.py \
        --checkpoint results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt \
        --out models/onnx/large_mlp.onnx \
        --name large_mlp \
        --preproc-distillation-parquet distillation_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _infer_architecture(blob: dict) -> tuple[int, list[int]]:
    """Return (in_dim, hidden_dims) from a trainer checkpoint blob."""
    in_dim = blob.get("in_dim")
    hidden = blob.get("hidden_dims")
    if not in_dim or not hidden:
        raise ValueError(
            "Checkpoint lacks in_dim/hidden_dims metadata; export requires them to "
            "reconstruct the architecture. Check the checkpoint was written by the "
            "distillation trainer."
        )
    return int(in_dim), [int(h) for h in hidden]


def export_checkpoint(
    checkpoint: Path,
    out_path: Path,
    *,
    name: str,
    opset: int = 17,
) -> None:
    """Export a PyTorch distillation checkpoint to ONNX."""
    if not torch.__version__:
        raise RuntimeError("torch is required for export")
    import torch.nn as nn

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    in_dim, hidden_dims = _infer_architecture(blob["model_config"] or blob)

    from aerotwin.distillation.mlp import StudentMLP

    model = StudentMLP(in_dim=in_dim, hidden_dims=hidden_dims, dropout=0.0)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, in_dim, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            out_path,
            input_names=["input"],
            output_names=["fuel_kg"],
            dynamic_axes={"input": {0: "batch"}, "fuel_kg": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )

    # Emit side-car metadata alongside the .onnx so consumers know the input contract.
    meta = {
        "name": name,
        "input_dim": in_dim,
        "hidden_dims": hidden_dims,
        "output": "fuel_kg",
        "opset": opset,
        "source_checkpoint": str(checkpoint),
        "preproc_path": str(out_path.with_suffix(".preproc.json")),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported ONNX -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Metadata    -> {meta_path}")


def _numeric_columns(feature_cols: list[str]) -> list[str]:
    cats = {"aircraft_type", "method", "origin_icao", "destination_icao"}
    return [c for c in feature_cols if c not in cats]


def build_preproc(
    parquet: Path,
    meta_json: Path,
    out_path: Path,
    *,
    seed: int = 42,
) -> None:
    """Rebuild median-impute + StandardScaler + OneHotEncoder and persist to JSON.

    Mirrors the fit performed in
    ``aerotwin/distillation/data.py::DistillationData.from_parquet`` so a runtime
    that does not ship scikit-learn can reproduce the exact feature encoding.
    """
    import polars as pl
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pl.read_parquet(parquet)
    feature_cols = json.loads(meta_json.read_text(encoding="utf-8")).get("feature_cols")
    if not feature_cols:
        from aerotwin.distillation.data import FEATURE_COLS_DEFAULT

        feature_cols = list(FEATURE_COLS_DEFAULT)

    cat_cols = [c for c in ("aircraft_type", "method", "origin_icao", "destination_icao") if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    df = df.filter(pl.col("ground_truth").is_finite() & pl.col("teacher_prediction").is_finite())
    flights = np.sort(df["flight_id"].unique().to_numpy())
    tr_f, _ = train_test_split(flights, test_size=0.2, random_state=seed)
    tr_set = set(map(str, tr_f))
    fids = df["flight_id"].cast(pl.Utf8).to_numpy()
    train_idx = np.array([f in tr_set for f in fids], dtype=bool)

    num_all = np.column_stack(
        [df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64) for c in num_cols]
    )
    medians = np.nanmedian(num_all[train_idx], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(num_all.shape[1]):
        col = num_all[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col = col.copy()
            col[bad] = medians[j]
            num_all[:, j] = col

    scaler = StandardScaler()
    scaler.fit(num_all[train_idx])

    cat_frame = df.select([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in cat_cols]).to_pandas()
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    ohe.fit(cat_frame.iloc[train_idx])

    payload = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "median_impute": [float(x) for x in medians],
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "onehot_categories": [list(ohe.categories_[i]) for i in range(len(cat_cols))],
        "n_numeric": len(num_cols),
        "ohe_start": len(num_cols),  # first ohe_start-... columns after numeric block
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote preprocessing params -> {out_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to best_model.pt")
    p.add_argument("--out", required=True, type=Path, help="Output .onnx path")
    p.add_argument("--name", default="student_mlp", help="Model name for metadata")
    p.add_argument("--opset", default=17, type=int)
    p.add_argument(
        "--preproc-distillation-parquet",
        type=Path,
        default=None,
        help="Optional distillation_dataset.parquet to rebuild preprocessing JSON.",
    )
    p.add_argument(
        "--meta-json",
        type=Path,
        default=ROOT / "docs/reports/distillation_dataset_meta.json",
        help="Feature metadata JSON (only needed with --preproc-distillation-parquet).",
    )
    args = p.parse_args(argv)

    export_checkpoint(args.checkpoint, args.out, name=args.name, opset=args.opset)

    if args.preproc_distillation_parquet is not None:
        build_preproc(
            args.preproc_distillation_parquet,
            args.meta_json,
            args.out.with_suffix(".preproc.json"),
        )


if __name__ == "__main__":
    main()
