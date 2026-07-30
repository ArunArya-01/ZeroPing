"""Evaluate a trained FT-Transformer (or any factory student) on Final + Combined.

Evaluation only — reuses Step-5 Final features and Rank features when present.
Does not train or modify checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.engine.gap_closing import clean_featured, ensure_features, group_phase, rmse as kg_rmse
from aerotwin.engine.mass_model import enrich_mass_from_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("eval_ft")

TEACHER_FINAL = 213.62
TEACHER_COMBINED = 221.33
TEACHER_RANK = 232.53
LARGE_FINAL = 215.85
LARGE_COMBINED = 225.95
LARGE_RANK = 240.66


def _prepare(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "actual_fuel_kg" not in df.columns and "fuel_kg" in df.columns:
        df = df.with_columns(pl.col("fuel_kg").alias("actual_fuel_kg"))
    return enrich_mass_from_columns(clean_featured(df))


def _transform(df: pl.DataFrame, data: DistillationData) -> tuple[np.ndarray, np.ndarray]:
    feats = data.feature_cols
    numeric_cols = data.numeric_cols
    cat_cols = data.cat_cols
    df = ensure_features(df, feats)
    train_df = pl.read_parquet(data.parquet_path).filter(
        pl.col("ground_truth").is_finite()
        & pl.col("teacher_prediction").is_finite()
        & pl.col("flight_id").is_not_null()
    )
    train_num = np.column_stack(
        [
            train_df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64)
            for c in numeric_cols
        ]
    )
    medians = np.nanmedian(train_num[data.train_idx], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    num = np.column_stack(
        [df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64) for c in numeric_cols]
    )
    for j in range(num.shape[1]):
        bad = ~np.isfinite(num[:, j])
        if bad.any():
            col = num[:, j].copy()
            col[bad] = medians[j]
            num[:, j] = col
    x_num = data.scaler.transform(num).astype(np.float32)
    cat_pdf = df.select([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in cat_cols]).to_pandas()
    x_cat = data.ohe.transform(cat_pdf).astype(np.float32)
    x = np.hstack([x_num, x_cat]).astype(np.float32)
    y = df["actual_fuel_kg"].to_numpy().astype(np.float64)
    return x, y


@torch.no_grad()
def _predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), 512):
        out.append(model(xt[i : i + 512].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _full(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = regression_metrics(y, p)
    err = p - y
    m.update(
        {
            "mean_residual": float(np.mean(err)),
            "p95_abs_error": float(np.percentile(np.abs(err), 95)),
            "n": int(len(y)),
        }
    )
    return m


def _load_model(ckpt: Path, student_cfg: StudentConfig, in_dim: int, device: torch.device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    # Prefer config stored at train time
    sc = student_cfg
    if blob.get("model_config") and isinstance(blob["model_config"], dict):
        mc = blob["model_config"]
        sc = StudentConfig(
            architecture=mc.get("architecture") or student_cfg.architecture,
            in_dim=in_dim,
            d_token=int(mc.get("d_token", student_cfg.d_token)),
            n_blocks=int(mc.get("n_blocks", student_cfg.n_blocks)),
            n_heads=int(mc.get("n_heads", student_cfg.n_heads)),
            n_num_features=mc.get("n_num_features", student_cfg.n_num_features),
            cat_cardinalities=mc.get("cat_cardinalities", student_cfg.cat_cardinalities),
        )
    elif blob.get("config", {}).get("extras", {}).get("student_config"):
        sc = StudentConfig.from_mapping(blob["config"]["extras"]["student_config"])
        sc.in_dim = in_dim
    else:
        sc = student_cfg
        sc.in_dim = in_dim
    model = build_student(sc, in_dim=in_dim)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device).eval()
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, n_params, blob


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt",
    )
    ap.add_argument(
        "--student-config",
        type=Path,
        default=ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json",
    )
    ap.add_argument("--rank-featured", type=Path, default=ROOT / "featured_dataset_rank.parquet")
    ap.add_argument("--final-featured", type=Path, default=ROOT / "featured_dataset_final.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/distillation/ft_transformer/evaluation",
    )
    args = ap.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.final_featured.exists():
        raise FileNotFoundError(args.final_featured)

    student_cfg = StudentConfig(architecture="ft_transformer")
    if args.student_config.exists():
        student_cfg = StudentConfig.from_mapping(
            json.loads(args.student_config.read_text(encoding="utf-8"))
        )

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    # Ensure native layout matches training even if config omitted cards
    if student_cfg.n_num_features is None:
        student_cfg.n_num_features = len(data.numeric_cols)
    if student_cfg.cat_cardinalities is None:
        student_cfg.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    model, n_params, blob = _load_model(args.checkpoint, student_cfg, data.in_dim, device)
    LOGGER.info("Loaded %s params=%s device=%s", args.checkpoint, f"{n_params:,}", device)

    # Final
    final_df = _prepare(args.final_featured)
    x_f, y_f = _transform(final_df, data)
    p_f = _predict(model, x_f, device)
    m_final = _full(y_f, p_f)
    LOGGER.info("Final RMSE=%.2f MAE=%.2f R2=%.4f", m_final["rmse"], m_final["mae"], m_final["r2"])

    pl.DataFrame(
        {
            "flight_id": final_df["flight_id"].cast(pl.Utf8).to_list()
            if "flight_id" in final_df.columns
            else [str(i) for i in range(len(y_f))],
            "interval_idx": final_df["interval_idx"].to_list()
            if "interval_idx" in final_df.columns
            else list(range(len(y_f))),
            "aircraft_type": final_df["aircraft_type"].cast(pl.Utf8).fill_null("?").to_list()
            if "aircraft_type" in final_df.columns
            else ["?"] * len(y_f),
            "phase": group_phase(final_df).astype(str).tolist(),
            "ground_truth": y_f,
            "predicted_fuel": p_f,
            "residual": p_f - y_f,
            "absolute_error": np.abs(p_f - y_f),
        }
    ).write_parquet(out / "predictions_final.parquet")

    # Rank + Combined
    m_rank = None
    m_comb = None
    if args.rank_featured.exists():
        rank_df = _prepare(args.rank_featured)
        x_r, y_r = _transform(rank_df, data)
        p_r = _predict(model, x_r, device)
        m_rank = _full(y_r, p_r)
        y_c = np.concatenate([y_r, y_f])
        p_c = np.concatenate([p_r, p_f])
        m_comb = _full(y_c, p_c)
        m_comb["rmse"] = kg_rmse(y_c, p_c)
        LOGGER.info(
            "Rank RMSE=%.2f Combined RMSE=%.2f", m_rank["rmse"], m_comb["rmse"]
        )
        pl.DataFrame(
            {
                "split": ["rank"] * len(y_r) + ["final"] * len(y_f),
                "ground_truth": y_c,
                "predicted_fuel": p_c,
                "residual": p_c - y_c,
                "absolute_error": np.abs(p_c - y_c),
            }
        ).write_parquet(out / "predictions_combined.parquet")
        pl.DataFrame(
            {
                "ground_truth": y_r,
                "predicted_fuel": p_r,
                "residual": p_r - y_r,
                "absolute_error": np.abs(p_r - y_r),
            }
        ).write_parquet(out / "predictions_rank.parquet")
    else:
        LOGGER.warning("Rank features missing — Combined not computed")

    comparison = [
        {
            "model": "R3 Teacher",
            "rank_rmse": TEACHER_RANK,
            "final_rmse": TEACHER_FINAL,
            "combined_rmse": TEACHER_COMBINED,
            "parameters": "ensemble",
        },
        {
            "model": "Large MLP (baseline)",
            "rank_rmse": LARGE_RANK,
            "final_rmse": LARGE_FINAL,
            "combined_rmse": LARGE_COMBINED,
            "parameters": 2_887_425,
        },
        {
            "model": "FT-Transformer",
            "rank_rmse": m_rank["rmse"] if m_rank else None,
            "final_rmse": m_final["rmse"],
            "combined_rmse": m_comb["rmse"] if m_comb else None,
            "parameters": n_params,
        },
    ]
    pl.DataFrame(comparison).write_csv(out / "comparison_table.csv")

    val_rmse = float(blob.get("best_val_rmse") or float("nan"))
    metrics = {
        "architecture": "ft_transformer",
        "checkpoint": str(args.checkpoint.resolve()),
        "n_params": n_params,
        "alpha": 0.1,
        "beta": 0.9,
        "val_rmse": val_rmse,
        "final": m_final,
        "rank": m_rank,
        "combined": m_comb,
        "comparison": comparison,
        "vs_large_final": m_final["rmse"] - LARGE_FINAL,
        "vs_teacher_final": m_final["rmse"] - TEACHER_FINAL,
        "vs_large_combined": (m_comb["rmse"] - LARGE_COMBINED) if m_comb else None,
        "vs_teacher_combined": (m_comb["rmse"] - TEACHER_COMBINED) if m_comb else None,
        "beats_large_final": m_final["rmse"] < LARGE_FINAL,
        "beats_large_combined": (m_comb["rmse"] < LARGE_COMBINED) if m_comb else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    report = _report(metrics)
    (out / "ft_transformer_evaluation.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "ft_transformer_report.md").write_text(report, encoding="utf-8")

    print("\n=== FT-TRANSFORMER EVALUATION ===")
    print(f"  Final RMSE={m_final['rmse']:.2f} (Large {LARGE_FINAL}, Teacher {TEACHER_FINAL})")
    if m_comb:
        print(
            f"  Combined RMSE={m_comb['rmse']:.2f} (Large {LARGE_COMBINED}, Teacher {TEACHER_COMBINED})"
        )
    print(f"  beats Large Final? {metrics['beats_large_final']}")
    print(f"  results={out}")


def _report(m: dict[str, Any]) -> str:
    mf = m["final"]
    lines = [
        "# FT-Transformer Student — Training & Evaluation Report",
        "",
        f"**Date:** {m['timestamp_utc'][:10]}",
        "**Phase:** 2 — architecture experiment under frozen KD pipeline",
        "",
        "Only the student architecture changed. Teacher, data, split, α=0.1, β=0.9, and",
        "preprocessing match the Large MLP baseline.",
        "",
        "---",
        "",
        "## Architecture",
        "",
        "| Item | Value |",
        "|------|------:|",
        f"| Architecture | FT-Transformer (Gorishniy et al. 2021) |",
        f"| Parameters | {m['n_params']:,} |",
        f"| Checkpoint | `{m['checkpoint']}` |",
        f"| KD weights | α={m['alpha']}, β={m['beta']} |",
        f"| Val RMSE (flight holdout) | {m['val_rmse']:.2f} |",
        "",
        "Implementation: `src/aerotwin/distillation/models/ft_transformer.py`",
        "Factory: `build_student('ft_transformer', in_dim=…)`",
        "",
        "---",
        "",
        "## Official metrics",
        "",
        "| Model | Rank RMSE | Final RMSE | Combined RMSE | Params |",
        "|-------|----------:|-----------:|--------------:|-------:|",
    ]
    for row in m["comparison"]:
        rr = f"{row['rank_rmse']:.2f}" if row.get("rank_rmse") is not None else "—"
        fr = f"{row['final_rmse']:.2f}" if row.get("final_rmse") is not None else "—"
        cr = f"{row['combined_rmse']:.2f}" if row.get("combined_rmse") is not None else "—"
        lines.append(
            f"| {row['model']} | {rr} | {fr} | {cr} | {row['parameters']} |"
        )
    lines += [
        "",
        "### Deltas vs baselines",
        "",
        f"- Final vs Large MLP: **{m['vs_large_final']:+.2f} kg** (beats Large? **{m['beats_large_final']}**)",
        f"- Final vs Teacher: **{m['vs_teacher_final']:+.2f} kg**",
    ]
    if m.get("vs_large_combined") is not None:
        lines += [
            f"- Combined vs Large MLP: **{m['vs_large_combined']:+.2f} kg** (beats Large? **{m['beats_large_combined']}**)",
            f"- Combined vs Teacher: **{m['vs_teacher_combined']:+.2f} kg**",
        ]
    lines += [
        "",
        "---",
        "",
        "## Detailed Final metrics",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| RMSE | {mf['rmse']:.4f} |",
        f"| MAE | {mf['mae']:.4f} |",
        f"| Bias | {mf['bias']:+.4f} |",
        f"| R² | {mf['r2']:.6f} |",
        f"| n | {mf['n']:,} |",
        "",
        "---",
        "",
        "## Conclusions (evidence only)",
        "",
        f"1. FT-Transformer Final RMSE = **{mf['rmse']:.2f} kg**.",
        f"2. Relative to Large MLP Final **215.85**: **{m['vs_large_final']:+.2f} kg**.",
    ]
    if m.get("combined"):
        lines.append(
            f"3. Combined RMSE = **{m['combined']['rmse']:.2f} kg** "
            f"(vs Large **225.95**, teacher **221.33**)."
        )
    lines += [
        "4. Deployment baseline remains **Large MLP** unless FT beats it on **both** Final and Combined.",
        "5. Future architectures should use `build_student(...)` and the same KD pipeline.",
        "",
        "## Artifacts",
        "",
        "- Train: `results/distillation/ft_transformer/`",
        "- Eval: `results/distillation/ft_transformer/evaluation/`",
        "",
        f"*Generated {m['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
