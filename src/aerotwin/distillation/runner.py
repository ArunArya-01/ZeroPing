"""Architecture-agnostic distillation experiment runner.

Future students (FT-Transformer, TabTransformer, …) only need to supply a
``model_factory(in_dim) -> nn.Module``. Data loading, KD loss, optimizers,
early stopping, metrics, and artifact layout stay fixed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.trainer import TrainConfig, set_seed, train_student

LOGGER = logging.getLogger(__name__)

ModelFactory = Callable[[int], nn.Module]


@dataclass
class KDWeightConfig:
    """One (α, β) supervision setting."""

    name: str
    alpha: float
    beta: float

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha, "beta": self.beta}


# Default sweep used for Step 3 (and re-usable defaults for later students).
DEFAULT_KD_SWEEP: tuple[KDWeightConfig, ...] = (
    KDWeightConfig("KD-0", 0.0, 1.0),
    KDWeightConfig("KD-1", 0.1, 0.9),
    KDWeightConfig("KD-2", 0.2, 0.8),
    KDWeightConfig("KD-3", 0.3, 0.7),
    KDWeightConfig("KD-4", 0.5, 0.5),
    KDWeightConfig("KD-5", 0.7, 0.3),
    KDWeightConfig("KD-6", 0.9, 0.1),
    KDWeightConfig("KD-7", 1.0, 0.0),
)


@dataclass
class ExperimentConfig:
    """Shared training knobs — only α/β should vary within a KD sweep."""

    seed: int = 42
    val_fraction: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 80
    patience: int = 12
    min_delta: float = 0.05
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    grad_clip: float = 1.0
    device: str = "auto"
    num_workers: int = 0
    # Architecture defaults for the baseline MLP (ignored if factory overrides).
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.1
    extras: dict[str, Any] = field(default_factory=dict)


def make_train_config(
    *,
    run_name: str,
    alpha: float,
    beta: float,
    exp: ExperimentConfig,
) -> TrainConfig:
    """Always use KD mode: loss = α·MSE(gt) + β·MSE(teacher)."""
    extras = dict(exp.extras)
    return TrainConfig(
        mode="kd",
        alpha=float(alpha),
        beta=float(beta),
        lr=exp.lr,
        weight_decay=exp.weight_decay,
        batch_size=exp.batch_size,
        max_epochs=exp.max_epochs,
        patience=exp.patience,
        min_delta=exp.min_delta,
        scheduler_factor=exp.scheduler_factor,
        scheduler_patience=exp.scheduler_patience,
        grad_clip=exp.grad_clip,
        seed=exp.seed,
        device=exp.device,
        num_workers=exp.num_workers,
        hidden_dims=exp.hidden_dims,
        dropout=exp.dropout,
        run_name=run_name,
        consistency_lambda=float(extras.get("consistency_lambda", 0.0) or 0.0),
        consistency_noise_scale=float(extras.get("consistency_noise_scale", 0.015) or 0.015),
        n_num_features=extras.get("n_num_features"),
        extras=extras,
    )


def run_single_experiment(
    *,
    data: DistillationData,
    model_factory: ModelFactory,
    weight: KDWeightConfig,
    exp: ExperimentConfig,
    out_dir: Path,
    log_dir: Path | None = None,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one student under a fixed (α, β) and write artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    set_seed(exp.seed)
    model = model_factory(data.in_dim)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_loader, train_eval_loader, val_loader = data.loaders(
        batch_size=exp.batch_size,
        num_workers=exp.num_workers,
    )
    cfg = make_train_config(
        run_name=weight.name,
        alpha=weight.alpha,
        beta=weight.beta,
        exp=exp,
    )
    cfg.extras.update(
        {
            "kd_name": weight.name,
            "alpha": weight.alpha,
            "beta": weight.beta,
            "n_params": n_params,
            "in_dim": data.in_dim,
        }
    )

    LOGGER.info(
        "=== %s  α=%.2f  β=%.2f  params=%s ===",
        weight.name,
        weight.alpha,
        weight.beta,
        f"{n_params:,}",
    )
    metrics = train_student(
        model,
        train_loader,
        val_loader,
        cfg,
        out_dir=out_dir,
        log_dir=log_dir,
        train_eval_loader=train_eval_loader,
    )
    metrics["kd_name"] = weight.name
    metrics["alpha"] = weight.alpha
    metrics["beta"] = weight.beta
    metrics["n_params"] = n_params
    metrics["in_dim"] = data.in_dim

    # Flatten primary metrics for tabular export.
    flat = flatten_run_metrics(metrics)
    (out_dir / "flat_metrics.json").write_text(
        json.dumps(flat, indent=2, default=str), encoding="utf-8"
    )

    if model_dir is not None:
        src = out_dir / "best_model.pt"
        if src.exists():
            (Path(model_dir) / "best_model.pt").write_bytes(src.read_bytes())
        (Path(model_dir) / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )

    return metrics


def flatten_run_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Row-friendly metrics for CSV / comparison tables."""
    val_s = metrics.get("val", {}).get("student", {})
    val_t = metrics.get("val", {}).get("teacher", {})
    tr_s = metrics.get("train", {}).get("student", {})
    return {
        "name": metrics.get("kd_name") or metrics.get("run_name"),
        "run_name": metrics.get("run_name"),
        "alpha": metrics.get("alpha", metrics.get("config", {}).get("alpha")),
        "beta": metrics.get("beta", metrics.get("config", {}).get("beta")),
        "val_rmse": val_s.get("rmse"),
        "val_mae": val_s.get("mae"),
        "val_bias": val_s.get("bias"),
        "val_r2": val_s.get("r2"),
        "teacher_val_rmse": val_t.get("rmse"),
        "teacher_student_rmse_gap": metrics.get("val", {}).get("teacher_student_rmse_gap"),
        "student_vs_teacher_rmse": metrics.get("val", {}).get("student_vs_teacher_rmse"),
        "train_rmse": tr_s.get("rmse"),
        "train_mae": tr_s.get("mae"),
        "train_bias": tr_s.get("bias"),
        "train_r2": tr_s.get("r2"),
        "best_epoch": metrics.get("best_epoch"),
        "epochs_ran": metrics.get("epochs_ran"),
        "train_seconds": metrics.get("train_seconds"),
        "n_params": metrics.get("n_params"),
        "device": metrics.get("device"),
        "best_val_rmse": metrics.get("best_val_rmse"),
    }


def run_kd_sweep(
    *,
    data: DistillationData,
    model_factory: ModelFactory,
    weights: Sequence[KDWeightConfig],
    exp: ExperimentConfig,
    results_root: Path,
    logs_root: Path | None = None,
    models_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Iterate α/β configs; return list of flat metric rows."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    for w in weights:
        run_dir = results_root / w.name
        log_dir = Path(logs_root) / w.name if logs_root else None
        model_dir = Path(models_root) / w.name if models_root else None
        metrics = run_single_experiment(
            data=data,
            model_factory=model_factory,
            weight=w,
            exp=exp,
            out_dir=run_dir,
            log_dir=log_dir,
            model_dir=model_dir,
        )
        rows.append(flatten_run_metrics(metrics))

    elapsed = time.time() - t0
    LOGGER.info("KD sweep finished: %d runs in %.1f min", len(rows), elapsed / 60.0)
    return rows


def analyze_kd_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive best/worst and teacher-vs-GT conclusions from metrics only."""
    if not rows:
        return {"error": "no rows"}

    def key_rmse(r: dict[str, Any]) -> float:
        return float(r["val_rmse"])

    def key_imit(r: dict[str, Any]) -> float:
        return float(r["student_vs_teacher_rmse"])

    best = min(rows, key=key_rmse)
    worst = max(rows, key=key_rmse)
    best_imit = min(rows, key=key_imit)
    worst_imit = max(rows, key=key_imit)

    # Teacher-heavy: β > α; GT-heavy: α > β
    teacher_heavy = [r for r in rows if float(r["beta"]) > float(r["alpha"])]
    gt_heavy = [r for r in rows if float(r["alpha"]) > float(r["beta"])]
    balanced = [r for r in rows if float(r["alpha"]) == float(r["beta"])]

    def mean_rmse(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        return float(np.mean([float(r["val_rmse"]) for r in group]))

    def mean_imit(group: list[dict[str, Any]]) -> float | None:
        if not group:
            return None
        return float(np.mean([float(r["student_vs_teacher_rmse"]) for r in group]))

    pure_teacher = next((r for r in rows if float(r["alpha"]) == 0.0), None)
    pure_gt = next((r for r in rows if float(r["beta"]) == 0.0), None)

    # Does adding GT ever improve teacher imitation vs pure teacher (KD-0)?
    imit_vs_pure_teacher: list[dict[str, Any]] = []
    if pure_teacher is not None:
        base_imit = float(pure_teacher["student_vs_teacher_rmse"])
        for r in rows:
            if r is pure_teacher:
                continue
            delta = float(r["student_vs_teacher_rmse"]) - base_imit
            imit_vs_pure_teacher.append(
                {
                    "name": r["name"],
                    "alpha": r["alpha"],
                    "beta": r["beta"],
                    "student_vs_teacher_rmse": r["student_vs_teacher_rmse"],
                    "delta_vs_pure_teacher": delta,
                    "improved_imitation": delta < -1e-9,
                }
            )
        any_gt_improves_imit = any(x["improved_imitation"] for x in imit_vs_pure_teacher)
    else:
        any_gt_improves_imit = None

    # Label denoiser evidence: pure teacher (or teacher-heavy) better val RMSE than pure GT.
    denoiser_evidence = None
    if pure_teacher is not None and pure_gt is not None:
        denoiser_evidence = {
            "pure_teacher_val_rmse": pure_teacher["val_rmse"],
            "pure_gt_val_rmse": pure_gt["val_rmse"],
            "delta_rmse_teacher_minus_gt": float(pure_teacher["val_rmse"])
            - float(pure_gt["val_rmse"]),
            "teacher_better_on_gt": float(pure_teacher["val_rmse"])
            < float(pure_gt["val_rmse"]),
        }

    teacher_heavy_mean = mean_rmse(teacher_heavy)
    gt_heavy_mean = mean_rmse(gt_heavy)
    teacher_heavy_outperforms = None
    if teacher_heavy_mean is not None and gt_heavy_mean is not None:
        teacher_heavy_outperforms = teacher_heavy_mean < gt_heavy_mean

    analysis = {
        "best_by_val_rmse": {
            "name": best["name"],
            "alpha": best["alpha"],
            "beta": best["beta"],
            "val_rmse": best["val_rmse"],
            "val_mae": best["val_mae"],
            "val_bias": best["val_bias"],
            "val_r2": best["val_r2"],
            "student_vs_teacher_rmse": best["student_vs_teacher_rmse"],
            "teacher_student_rmse_gap": best["teacher_student_rmse_gap"],
        },
        "worst_by_val_rmse": {
            "name": worst["name"],
            "alpha": worst["alpha"],
            "beta": worst["beta"],
            "val_rmse": worst["val_rmse"],
        },
        "best_teacher_imitation": {
            "name": best_imit["name"],
            "alpha": best_imit["alpha"],
            "beta": best_imit["beta"],
            "student_vs_teacher_rmse": best_imit["student_vs_teacher_rmse"],
        },
        "worst_teacher_imitation": {
            "name": worst_imit["name"],
            "alpha": worst_imit["alpha"],
            "beta": worst_imit["beta"],
            "student_vs_teacher_rmse": worst_imit["student_vs_teacher_rmse"],
        },
        "group_means": {
            "teacher_heavy_val_rmse": teacher_heavy_mean,
            "gt_heavy_val_rmse": gt_heavy_mean,
            "balanced_val_rmse": mean_rmse(balanced),
            "teacher_heavy_imit_rmse": mean_imit(teacher_heavy),
            "gt_heavy_imit_rmse": mean_imit(gt_heavy),
        },
        "teacher_heavy_outperforms_gt_heavy": teacher_heavy_outperforms,
        "adding_gt_ever_improves_imitation": any_gt_improves_imit,
        "imitation_deltas_vs_pure_teacher": imit_vs_pure_teacher,
        "label_denoiser_evidence": denoiser_evidence,
        "recommended_alpha_beta": {
            "name": best["name"],
            "alpha": best["alpha"],
            "beta": best["beta"],
            "reason": "lowest validation RMSE among fixed-architecture KD weight configs",
        },
    }
    return analysis


def write_sweep_tables(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    analysis: dict[str, Any] | None = None,
    exp: ExperimentConfig | None = None,
    total_seconds: float | None = None,
) -> dict[str, Path]:
    """Write metrics.csv, comparison_table.csv, summary.json, best_configuration.json."""
    import polars as pl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = analysis or analyze_kd_sweep(rows)

    # Sort by alpha for readability
    rows_sorted = sorted(rows, key=lambda r: (float(r["alpha"]), float(r["beta"])))
    df = pl.DataFrame(rows_sorted)

    metrics_path = out_dir / "metrics.csv"
    df.write_csv(metrics_path)

    # Publication comparison table (subset of columns, friendly names).
    cmp_cols = [
        "name",
        "alpha",
        "beta",
        "val_rmse",
        "val_mae",
        "val_bias",
        "val_r2",
        "student_vs_teacher_rmse",
        "teacher_student_rmse_gap",
        "teacher_val_rmse",
        "best_epoch",
        "epochs_ran",
        "train_seconds",
    ]
    present = [c for c in cmp_cols if c in df.columns]
    cmp_df = df.select(present).sort("val_rmse")
    comparison_path = out_dir / "comparison_table.csv"
    cmp_df.write_csv(comparison_path)

    summary = {
        "n_runs": len(rows),
        "total_seconds": total_seconds,
        "experiment": asdict(exp) if exp is not None else None,
        "analysis": analysis,
        "rows": rows_sorted,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    best_path = out_dir / "best_configuration.json"
    best_path.write_text(
        json.dumps(analysis["recommended_alpha_beta"], indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "metrics": metrics_path,
        "comparison_table": comparison_path,
        "summary": summary_path,
        "best_configuration": best_path,
    }


def plot_kd_sweep(
    rows: list[dict[str, Any]],
    plots_dir: Path,
    *,
    teacher_val_rmse: float | None = None,
) -> dict[str, Path]:
    """Publication-quality plots vs α."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_sorted = sorted(rows, key=lambda r: float(r["alpha"]))
    alphas = np.array([float(r["alpha"]) for r in rows_sorted], dtype=np.float64)
    val_rmse = np.array([float(r["val_rmse"]) for r in rows_sorted], dtype=np.float64)
    imit = np.array(
        [float(r["student_vs_teacher_rmse"]) for r in rows_sorted], dtype=np.float64
    )
    bias = np.array([float(r["val_bias"]) for r in rows_sorted], dtype=np.float64)
    r2 = np.array([float(r["val_r2"]) for r in rows_sorted], dtype=np.float64)
    names = [str(r["name"]) for r in rows_sorted]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 160,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    paths: dict[str, Path] = {}

    def _style_ax(ax, ylabel: str, title: str) -> None:
        ax.set_xlabel(r"$\alpha$ (ground-truth weight)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.1f}" for a in alphas])

    # 1. RMSE vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(alphas, val_rmse, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Student val RMSE")
    if teacher_val_rmse is not None:
        ax.axhline(
            teacher_val_rmse,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Teacher val RMSE ({teacher_val_rmse:.2f})",
        )
    best_i = int(np.argmin(val_rmse))
    ax.scatter(
        [alphas[best_i]],
        [val_rmse[best_i]],
        s=120,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2,
        zorder=5,
        label=f"Best: {names[best_i]}",
    )
    for a, y, n in zip(alphas, val_rmse, names):
        ax.annotate(n, (a, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _style_ax(ax, "Validation RMSE (kg)", "Knowledge distillation: validation RMSE vs α")
    ax.legend(loc="best")
    fig.tight_layout()
    p1 = plots_dir / "rmse_vs_alpha.png"
    fig.savefig(p1)
    plt.close(fig)
    paths["rmse_vs_alpha"] = p1

    # 2. Student-Teacher RMSE vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(alphas, imit, "s-", color="#ff7f0e", linewidth=2, markersize=7, label="Student↔Teacher RMSE")
    best_i = int(np.argmin(imit))
    ax.scatter(
        [alphas[best_i]],
        [imit[best_i]],
        s=120,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2,
        zorder=5,
        label=f"Closest: {names[best_i]}",
    )
    for a, y, n in zip(alphas, imit, names):
        ax.annotate(n, (a, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _style_ax(
        ax,
        "Student ↔ Teacher RMSE (kg)",
        "Teacher imitation quality vs α",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    p2 = plots_dir / "student_teacher_rmse_vs_alpha.png"
    fig.savefig(p2)
    plt.close(fig)
    paths["student_teacher_rmse_vs_alpha"] = p2

    # 3. Bias vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(alphas, bias, "D-", color="#2ca02c", linewidth=2, markersize=7, label="Val bias (pred − true)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    for a, y, n in zip(alphas, bias, names):
        ax.annotate(n, (a, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _style_ax(ax, "Validation bias (kg)", "Prediction bias vs α")
    ax.legend(loc="best")
    fig.tight_layout()
    p3 = plots_dir / "bias_vs_alpha.png"
    fig.savefig(p3)
    plt.close(fig)
    paths["bias_vs_alpha"] = p3

    # 4. R2 vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(alphas, r2, "^-", color="#9467bd", linewidth=2, markersize=7, label="Val R²")
    best_i = int(np.argmax(r2))
    ax.scatter(
        [alphas[best_i]],
        [r2[best_i]],
        s=120,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2,
        zorder=5,
        label=f"Best R²: {names[best_i]}",
    )
    for a, y, n in zip(alphas, r2, names):
        ax.annotate(n, (a, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    _style_ax(ax, "Validation R²", "Coefficient of determination vs α")
    ax.legend(loc="best")
    fig.tight_layout()
    p4 = plots_dir / "r2_vs_alpha.png"
    fig.savefig(p4)
    plt.close(fig)
    paths["r2_vs_alpha"] = p4

    # Combined 2x2 panel for the report
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        (axes[0, 0], val_rmse, "Validation RMSE (kg)", "RMSE vs α", "#1f77b4", "o"),
        (axes[0, 1], imit, "Student↔Teacher RMSE (kg)", "Imitation vs α", "#ff7f0e", "s"),
        (axes[1, 0], bias, "Bias (kg)", "Bias vs α", "#2ca02c", "D"),
        (axes[1, 1], r2, "R²", "R² vs α", "#9467bd", "^"),
    ]
    for ax, y, ylabel, title, color, marker in specs:
        ax.plot(alphas, y, f"{marker}-", color=color, linewidth=2, markersize=6)
        if ylabel.startswith("Bias"):
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        if title.startswith("RMSE") and teacher_val_rmse is not None:
            ax.axhline(teacher_val_rmse, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(alphas)
    fig.suptitle("AeroTwin distillation: α/β weight sweep (baseline MLP)", fontsize=13, y=1.01)
    fig.tight_layout()
    p5 = plots_dir / "alpha_beta_sweep_panel.png"
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    paths["panel"] = p5

    LOGGER.info("Wrote %d sweep plots under %s", len(paths), plots_dir)
    return paths
