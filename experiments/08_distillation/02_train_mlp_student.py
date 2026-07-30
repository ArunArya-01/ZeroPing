"""Step 2 — Train the first student (baseline MLP) on the frozen distillation dataset.

Three experiments:
  A) ground-truth only     MSE(student, ground_truth)
  B) teacher only          MSE(student, teacher_prediction)
  C) knowledge distillation alpha*MSE(gt) + beta*MSE(teacher)

The distillation_dataset.parquet must already exist and is never regenerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.trainer import TrainConfig, set_seed, train_student

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("mlp_student")

RUN_SPECS = {
    "A": {
        "run_name": "model_a_gt_only",
        "mode": "gt",
        "label": "Model A — Ground Truth Only",
        "description": "MSE(student, ground_truth) — neural baseline",
    },
    "B": {
        "run_name": "model_b_teacher_only",
        "mode": "teacher",
        "label": "Model B — Teacher Only",
        "description": "MSE(student, teacher_prediction) — pure imitation",
    },
    "C": {
        "run_name": "model_c_kd",
        "mode": "kd",
        "label": "Model C — Knowledge Distillation",
        "description": "alpha*MSE(gt) + beta*MSE(teacher)",
    },
}


def _dirs(root: Path) -> dict[str, Path]:
    d = {
        "models": root / "models" / "distillation",
        "logs": root / "logs" / "distillation",
        "results": root / "results" / "distillation",
        "figures": root / "docs" / "reports" / "figures",
        "report": root / "docs" / "reports" / "mlp_student_report.md",
    }
    for k in ("models", "logs", "results", "figures"):
        d[k].mkdir(parents=True, exist_ok=True)
    return d


def _plot_learning_curves(
    all_metrics: dict[str, dict[str, Any]],
    results_root: Path,
    fig_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, m in all_metrics.items():
        curve_path = results_root / m["run_name"] / "training_curve.csv"
        if not curve_path.exists():
            continue
        curve = pl.read_csv(curve_path)
        label = RUN_SPECS[key]["run_name"]
        axes[0].plot(curve["epoch"], curve["train_rmse"], label=f"{label} train")
        axes[0].plot(
            curve["epoch"],
            curve["val_rmse"],
            linestyle="--",
            label=f"{label} val",
        )
        axes[1].plot(curve["epoch"], curve["val_rmse"], label=label)
        if "val_teacher_rmse" in curve.columns:
            # Teacher is constant — plot once
            if key == list(all_metrics.keys())[0]:
                axes[1].axhline(
                    float(curve["val_teacher_rmse"][0]),
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    label="teacher val RMSE",
                )

    axes[0].set_title("Train / Val RMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE (kg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].set_title("Validation RMSE vs Teacher")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val RMSE (kg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    LOGGER.info("Wrote learning-curve figure %s", fig_path)


def _write_report(
    report_path: Path,
    *,
    data: DistillationData,
    all_metrics: dict[str, dict[str, Any]],
    hidden_dims: tuple[int, ...],
    dropout: float,
    alpha: float,
    beta: float,
    total_seconds: float,
    fig_rel: str,
) -> None:
    # Use first model for architecture / param count (same for all).
    any_m = next(iter(all_metrics.values()))
    n_params = any_m["n_params"]
    device = any_m["device"]

    def row(key: str) -> dict[str, Any]:
        return all_metrics[key]

    lines = [
        "# MLP Student Distillation Report",
        "",
        "**Stage:** AeroTwin Distillation - Step 2 (baseline MLP student)",
        "",
        "Question: *Can a small neural network absorb the knowledge of the frozen AeroTwin R3 ensemble?*",
        "",
        "The teacher distillation dataset is **frozen** and was not regenerated.",
        "",
        "---",
        "",
        "## Setup",
        "",
        f"| Item | Value |",
        f"|------|------:|",
        f"| Dataset | `distillation_dataset.parquet` |",
        f"| Samples | {data.n_samples:,} |",
        f"| Flights | {data.n_flights:,} |",
        f"| Raw features | {len(data.feature_cols)} |",
        f"| Model input dim (after OHE + scale) | {data.in_dim} |",
        f"| Train rows / Val rows | {len(data.train_idx):,} / {len(data.val_idx):,} |",
        f"| Val fraction (flight-level) | {data.val_fraction} |",
        f"| Seed | {data.seed} |",
        f"| Device | `{device}` |",
        f"| Total wall time (A+B+C) | {total_seconds/60:.1f} min ({total_seconds:.0f}s) |",
        "",
        "### Architecture",
        "",
        "```",
        f"Input ({data.in_dim})",
        "  -> Linear -> ReLU -> LayerNorm -> Dropout",
        f"  -> Linear -> ReLU -> LayerNorm -> Dropout   # hidden = {list(hidden_dims)}",
        "  -> Linear -> scalar (kg)",
        "```",
        "",
        f"- Hidden dims: `{list(hidden_dims)}`",
        f"- Dropout: `{dropout}`",
        f"- Parameters: **{n_params:,}** (~{n_params/1e6:.2f}M)",
        f"- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)",
        f"- Scheduler: ReduceLROnPlateau on val RMSE",
        f"- Early stopping: patience on val RMSE",
        f"- KD weights (Model C): alpha={alpha}, beta={beta}",
        "",
        "### Preprocessing (train-fit only)",
        "",
        "- Numeric: median impute + `StandardScaler`",
        "- Categorical (`aircraft_type`, `method`, `origin_icao`, `destination_icao`): `OneHotEncoder`",
        "- Split: Group by `flight_id` (80/20 train/val)",
        "",
        "---",
        "",
        "## Experiments",
        "",
        "| Model | Loss |",
        "|-------|------|",
        "| A | MSE(student, ground_truth) |",
        "| B | MSE(student, teacher_prediction) |",
        f"| C | {alpha}·MSE(gt) + {beta}·MSE(teacher) |",
        "",
        "---",
        "",
        "## Final metrics",
        "",
        "### Validation (primary)",
        "",
        "| Model | Student RMSE | Student MAE | Bias | R² | Teacher RMSE | Student−Teacher gap | Student↔Teacher RMSE | Best epoch | Train time (s) |",
        "|-------|-------------:|------------:|-----:|---:|-------------:|--------------------:|---------------------:|-----------:|---------------:|",
    ]

    for key in ("A", "B", "C"):
        if key not in all_metrics:
            continue
        m = row(key)
        vs = m["val"]["student"]
        vt = m["val"]["teacher"]
        lines.append(
            f"| {key} | {vs['rmse']:.2f} | {vs['mae']:.2f} | {vs['bias']:+.2f} | {vs['r2']:.4f} "
            f"| {vt['rmse']:.2f} | {m['val']['teacher_student_rmse_gap']:+.2f} "
            f"| {m['val']['student_vs_teacher_rmse']:.2f} | {m['best_epoch']} | {m['train_seconds']:.0f} |"
        )

    lines += [
        "",
        "### Train set",
        "",
        "| Model | Student RMSE | Teacher RMSE | Gap | R² |",
        "|-------|-------------:|-------------:|----:|---:|",
    ]
    for key in ("A", "B", "C"):
        if key not in all_metrics:
            continue
        m = row(key)
        ts = m["train"]["student"]
        tt = m["train"]["teacher"]
        lines.append(
            f"| {key} | {ts['rmse']:.2f} | {tt['rmse']:.2f} | "
            f"{m['train']['teacher_student_rmse_gap']:+.2f} | {ts['r2']:.4f} |"
        )

    lines += [
        "",
        "### Learning curves",
        "",
        f"![MLP student learning curves]({fig_rel})",
        "",
        "---",
        "",
        "## Comparison and observations",
        "",
    ]

    # Auto observations
    if set(all_metrics) >= {"A", "B", "C"}:
        va = {k: all_metrics[k]["val"]["student"]["rmse"] for k in ("A", "B", "C")}
        gaps = {k: all_metrics[k]["val"]["teacher_student_rmse_gap"] for k in ("A", "B", "C")}
        best_key = min(va, key=va.get)
        closest_key = min(gaps, key=lambda k: abs(gaps[k]) if gaps[k] == gaps[k] else 1e9)
        # closest to teacher by student_vs_teacher_rmse
        imitate = {
            k: all_metrics[k]["val"]["student_vs_teacher_rmse"] for k in ("A", "B", "C")
        }
        best_imitate = min(imitate, key=imitate.get)
        teacher_rmse = all_metrics["A"]["val"]["teacher"]["rmse"]

        lines += [
            f"1. **Teacher val RMSE** on this flight split is **{teacher_rmse:.2f} kg** "
            f"(train-OOF soft labels from the frozen R3 path; not Rank/Final).",
            f"2. **Best student vs ground truth** on val: **Model {best_key}** "
            f"(RMSE {va[best_key]:.2f} kg).",
            f"3. **Closest imitation of the teacher** (Student↔Teacher RMSE): **Model {best_imitate}** "
            f"({imitate[best_imitate]:.2f} kg).",
            f"4. **Teacher→Student RMSE gaps** (student_rmse − teacher_rmse): "
            f"A {gaps['A']:+.2f}, B {gaps['B']:+.2f}, C {gaps['C']:+.2f} kg.",
            "",
        ]

        if va["B"] <= va["A"] * 1.05 or imitate["B"] < imitate["A"]:
            lines.append(
                "5. Model B (teacher-only) demonstrates that the MLP can **track the frozen teacher** "
                "soft labels; a lower Student↔Teacher RMSE than Model A supports knowledge absorption."
            )
        else:
            lines.append(
                "5. Model B imitation is mixed relative to A — check capacity, training length, "
                "and feature scaling if Student↔Teacher RMSE remains large."
            )

        if va["C"] <= min(va["A"], va["B"]) + 1.0:
            lines.append(
                "6. Model C (KD) is competitive with or better than pure GT / pure teacher losses, "
                "suggesting a useful blend of hard labels and teacher soft targets."
            )
        elif va["C"] < va["A"]:
            lines.append(
                "6. Model C improves over pure GT (A) but does not dominate teacher-only (B) — "
                "α/β may need tuning in a later step (not done here)."
            )
        else:
            lines.append(
                "6. Model C did not beat A/B with α=β=0.5; this baseline still answers capacity, "
                "not optimal distillation weights."
            )

        absorbed = imitate[best_imitate] < teacher_rmse * 0.35 or gaps[best_key] < 50
        lines += [
            "",
            "### Answer (this stage only)",
            "",
        ]
        if absorbed or va[best_key] < teacher_rmse * 1.25:
            lines.append(
                f"**Yes, partially.** A ~{n_params/1e6:.1f}M MLP reaches val RMSE "
                f"**{va[best_key]:.1f} kg** vs teacher **{teacher_rmse:.1f} kg** "
                f"(gap {gaps[best_key]:+.1f} kg) and can imitate the teacher "
                f"(best Student↔Teacher RMSE {imitate[best_imitate]:.1f} kg). "
                "It absorbs a large fraction of the ensemble signal under a simple tabular MLP, "
                "with remaining gap expected vs a 6-base GBDT + Ridge + P1E stack."
            )
        else:
            lines.append(
                f"**Not fully.** Best val RMSE is **{va[best_key]:.1f} kg** vs teacher "
                f"**{teacher_rmse:.1f} kg**. The MLP learns useful structure but does not yet "
                "match teacher quality; later students (FT-Transformer, etc.) can reuse this pipeline."
            )

    lines += [
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "| Kind | Path pattern |",
        "|------|--------------|",
        "| Checkpoints | `models/distillation/<run>/best_model.pt` |",
        "| Metrics | `results/distillation/<run>/metrics.json` |",
        "| Predictions | `results/distillation/<run>/predictions.parquet` |",
        "| Curves | `results/distillation/<run>/training_curve.csv` |",
        "| Logs | `logs/distillation/<run>/` |",
        "| Comparison | `results/distillation/comparison.json` |",
        "",
        "### Out of scope (intentionally)",
        "",
        "- FT-Transformer / TabTransformer / Trajectory Transformer",
        "- Multi-task heads, temperature scaling, hidden-state matching",
        "- Feature ablations, architecture search, hyperparameter optimization",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote report %s", report_path)


def train_one(
    key: str,
    data: DistillationData,
    dirs: dict[str, Path],
    *,
    alpha: float,
    beta: float,
    hidden_dims: tuple[int, ...],
    dropout: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    spec = RUN_SPECS[key]
    run = spec["run_name"]
    model_dir = dirs["models"] / run
    result_dir = dirs["results"] / run
    log_dir = dirs["logs"] / run
    for d in (model_dir, result_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Mirror best_model into both models/ and results/ as requested
    out_dir = result_dir
    set_seed(seed)
    model = StudentMLP(
        in_dim=data.in_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    train_loader, train_eval_loader, val_loader = data.loaders(batch_size=batch_size)

    cfg = TrainConfig(
        mode=spec["mode"],  # type: ignore[arg-type]
        alpha=alpha,
        beta=beta,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        hidden_dims=hidden_dims,
        dropout=dropout,
        run_name=run,
        extras={"experiment": key, "label": spec["label"]},
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

    # Copy / save checkpoint under models/distillation/<run>/
    ckpt_src = out_dir / "best_model.pt"
    ckpt_dst = model_dir / "best_model.pt"
    if ckpt_src.exists():
        ckpt_dst.write_bytes(ckpt_src.read_bytes())
    # Also ensure metrics/predictions/curve live under results (already)
    # and a pointer json in models/
    (model_dir / "metrics.json").write_text(
        (out_dir / "metrics.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "distillation_dataset.parquet",
        help="Path to frozen distillation_dataset.parquet",
    )
    p.add_argument("--models", type=str, default="A,B,C", help="Comma list: A,B,C")
    p.add_argument("--alpha", type=float, default=0.5, help="KD weight on ground truth")
    p.add_argument("--beta", type=float, default=0.5, help="KD weight on teacher")
    p.add_argument("--hidden", type=str, default="1024,512", help="Hidden dims, comma-sep")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Missing {args.dataset}. Run Step 1 first; do not regenerate here."
        )

    hidden_dims = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    keys = [k.strip().upper() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in RUN_SPECS:
            raise SystemExit(f"Unknown model key {k}; choose from A,B,C")

    dirs = _dirs(ROOT)
    LOGGER.info("Loading frozen distillation dataset from %s", args.dataset)
    data = DistillationData.from_parquet(
        args.dataset,
        root=ROOT,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    LOGGER.info(
        "Prepared data: in_dim=%d train=%d val=%d params_budget hidden=%s",
        data.in_dim,
        len(data.train_idx),
        len(data.val_idx),
        hidden_dims,
    )

    # Quick param count sanity
    probe = StudentMLP(data.in_dim, hidden_dims=hidden_dims, dropout=args.dropout)
    n_params = probe.count_parameters()
    LOGGER.info("StudentMLP parameters: %s (%.2fM)", f"{n_params:,}", n_params / 1e6)
    if n_params > 5_000_000:
        LOGGER.warning("Parameter count exceeds 5M suggested budget")

    t0 = time.time()
    all_metrics: dict[str, dict[str, Any]] = {}
    for key in keys:
        LOGGER.info("========== %s ==========", RUN_SPECS[key]["label"])
        all_metrics[key] = train_one(
            key,
            data,
            dirs,
            alpha=args.alpha,
            beta=args.beta,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    total_seconds = time.time() - t0

    comparison = {
        "dataset": str(args.dataset),
        "in_dim": data.in_dim,
        "n_params": n_params,
        "hidden_dims": list(hidden_dims),
        "alpha": args.alpha,
        "beta": args.beta,
        "seed": args.seed,
        "total_seconds": total_seconds,
        "models": {k: all_metrics[k] for k in all_metrics},
    }
    cmp_path = dirs["results"] / "comparison.json"
    cmp_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    fig_path = dirs["figures"] / "fig_mlp_student_learning_curves.png"
    _plot_learning_curves(all_metrics, dirs["results"], fig_path)

    _write_report(
        dirs["report"],
        data=data,
        all_metrics=all_metrics,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        alpha=args.alpha,
        beta=args.beta,
        total_seconds=total_seconds,
        fig_rel="figures/fig_mlp_student_learning_curves.png",
    )

    print("\n=== MLP STUDENT TRAINING COMPLETE ===")
    for key in keys:
        m = all_metrics[key]
        print(
            f"  {key}: val_rmse={m['val']['student']['rmse']:.2f} "
            f"teacher={m['val']['teacher']['rmse']:.2f} "
            f"gap={m['val']['teacher_student_rmse_gap']:+.2f} "
            f"params={m['n_params']:,}"
        )
    print(f"report: {dirs['report']}")
    print(f"comparison: {cmp_path}")


if __name__ == "__main__":
    main()
