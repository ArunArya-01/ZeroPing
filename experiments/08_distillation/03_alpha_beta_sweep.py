"""Step 3 — α/β knowledge-distillation weight sweep (baseline MLP).

Only α and β change. Architecture, seed, split, optimizer, and scheduler match
Step 2. The frozen teacher dataset is never regenerated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.runner import (
    DEFAULT_KD_SWEEP,
    ExperimentConfig,
    KDWeightConfig,
    analyze_kd_sweep,
    plot_kd_sweep,
    run_kd_sweep,
    write_sweep_tables,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("alpha_beta_sweep")


def _write_report(
    report_path: Path,
    *,
    rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    exp: ExperimentConfig,
    data: DistillationData,
    total_seconds: float,
    n_params: int,
    plot_rels: dict[str, str],
) -> None:
    best = analysis["best_by_val_rmse"]
    worst = analysis["worst_by_val_rmse"]
    den = analysis.get("label_denoiser_evidence") or {}
    gm = analysis.get("group_means") or {}

    lines = [
        "# Distillation α/β Weight Sweep Report",
        "",
        "**Stage:** AeroTwin Distillation - Step 3",
        "",
        "Goal: determine how knowledge should be transferred from the frozen R3 "
        "teacher to neural students by varying only the KD weights "
        r"$\alpha$ (ground truth) and $\beta$ (teacher).",
        "",
        "The teacher, `distillation_dataset.parquet`, feature engineering, and "
        "data splits are **unchanged**.",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "### Loss",
        "",
        r"$$L = \alpha \cdot \mathrm{MSE}(\hat{y}, y_{\mathrm{gt}}) + "
        r"\beta \cdot \mathrm{MSE}(\hat{y}, y_{\mathrm{teacher}})$$",
        "",
        "All other training settings are identical to Step 2 (baseline MLP).",
        "",
        "### Fixed settings",
        "",
        "| Setting | Value |",
        "|---------|------:|",
        f"| Student | MLP `[1024, 512]` + LayerNorm + Dropout |",
        f"| Parameters | **{n_params:,}** |",
        f"| Input dim | {data.in_dim} |",
        f"| Seed | {exp.seed} |",
        f"| Flight-level val fraction | {exp.val_fraction} |",
        f"| Train / val rows | {len(data.train_idx):,} / {len(data.val_idx):,} |",
        f"| Optimizer | AdamW lr={exp.lr}, wd={exp.weight_decay} |",
        f"| Batch size | {exp.batch_size} |",
        f"| Max epochs | {exp.max_epochs} |",
        f"| Early-stopping patience | {exp.patience} |",
        f"| Scheduler | ReduceLROnPlateau factor={exp.scheduler_factor}, patience={exp.scheduler_patience} |",
        f"| Wall time (full sweep) | {total_seconds/60:.1f} min |",
        "",
        "### Configurations",
        "",
        "| Experiment | α (GT) | β (Teacher) |",
        "|------------|-------:|------------:|",
    ]
    for r in sorted(rows, key=lambda x: float(x["alpha"])):
        lines.append(f"| {r['name']} | {float(r['alpha']):.1f} | {float(r['beta']):.1f} |")

    lines += [
        "",
        "---",
        "",
        "## Results",
        "",
        "### Comparison table (sorted by val RMSE)",
        "",
        "| Exp | α | β | Val RMSE | MAE | Bias | R² | Student↔Teacher | Gap vs teacher | Epochs | Time (s) |",
        "|-----|--:|--:|---------:|----:|-----:|---:|----------------:|---------------:|-------:|---------:|",
    ]
    for r in sorted(rows, key=lambda x: float(x["val_rmse"])):
        lines.append(
            f"| {r['name']} | {float(r['alpha']):.1f} | {float(r['beta']):.1f} "
            f"| {float(r['val_rmse']):.2f} | {float(r['val_mae']):.2f} "
            f"| {float(r['val_bias']):+.2f} | {float(r['val_r2']):.4f} "
            f"| {float(r['student_vs_teacher_rmse']):.2f} "
            f"| {float(r['teacher_student_rmse_gap']):+.2f} "
            f"| {int(r['epochs_ran'])} | {float(r['train_seconds']):.0f} |"
        )

    t_rmse = rows[0].get("teacher_val_rmse")
    lines += [
        "",
        f"Teacher validation RMSE (fixed soft labels on this split): "
        f"**{float(t_rmse):.2f} kg**." if t_rmse is not None else "",
        "",
        "### Figures",
        "",
        f"![RMSE vs α]({plot_rels.get('rmse_vs_alpha', '')})",
        "",
        f"![Student–Teacher RMSE vs α]({plot_rels.get('student_teacher_rmse_vs_alpha', '')})",
        "",
        f"![Bias vs α]({plot_rels.get('bias_vs_alpha', '')})",
        "",
        f"![R² vs α]({plot_rels.get('r2_vs_alpha', '')})",
        "",
        f"![Panel]({plot_rels.get('panel', '')})",
        "",
        "---",
        "",
        "## Analysis",
        "",
        f"- **Best α/β (val RMSE):** `{best['name']}` with α={best['alpha']}, β={best['beta']} "
        f"→ val RMSE **{float(best['val_rmse']):.2f} kg** "
        f"(MAE {float(best['val_mae']):.2f}, bias {float(best['val_bias']):+.2f}, "
        f"R² {float(best['val_r2']):.4f}).",
        f"- **Worst α/β (val RMSE):** `{worst['name']}` (α={worst['alpha']}, β={worst['beta']}) "
        f"→ **{float(worst['val_rmse']):.2f} kg**.",
        f"- **Best teacher imitation:** `{analysis['best_teacher_imitation']['name']}` "
        f"(Student↔Teacher RMSE {float(analysis['best_teacher_imitation']['student_vs_teacher_rmse']):.2f} kg).",
        "",
    ]

    th = analysis.get("teacher_heavy_outperforms_gt_heavy")
    if th is True:
        lines.append(
            f"- **Teacher-heavy vs GT-heavy:** teacher-heavy mean val RMSE "
            f"**{gm.get('teacher_heavy_val_rmse'):.2f}** < GT-heavy "
            f"**{gm.get('gt_heavy_val_rmse'):.2f}** → teacher-heavy supervision "
            "outperforms GT-heavy on this sweep."
        )
    elif th is False:
        lines.append(
            f"- **Teacher-heavy vs GT-heavy:** GT-heavy mean val RMSE "
            f"**{gm.get('gt_heavy_val_rmse'):.2f}** ≤ teacher-heavy "
            f"**{gm.get('teacher_heavy_val_rmse'):.2f}** → GT-heavy is better or tied "
            "on mean val RMSE in this sweep."
        )
    else:
        lines.append("- **Teacher-heavy vs GT-heavy:** insufficient groups to compare.")

    agi = analysis.get("adding_gt_ever_improves_imitation")
    if agi is True:
        lines.append(
            "- **Adding GT and imitation:** at least one α>0 configuration improved "
            "Student↔Teacher RMSE vs pure teacher (KD-0)."
        )
    elif agi is False:
        lines.append(
            "- **Adding GT and imitation:** no configuration with α>0 improved "
            "Student↔Teacher RMSE vs pure teacher (KD-0). Adding ground truth did "
            "**not** improve teacher imitation under these settings."
        )

    if den:
        if den.get("teacher_better_on_gt"):
            lines.append(
                f"- **Label-denoiser evidence:** pure teacher (α=0) val RMSE "
                f"**{float(den['pure_teacher_val_rmse']):.2f}** < pure GT (β=0) "
                f"**{float(den['pure_gt_val_rmse']):.2f}** "
                f"(Δ {float(den['delta_rmse_teacher_minus_gt']):+.2f} kg). "
                "Training on teacher soft labels yields better agreement with ground "
                "truth than training on ground truth alone — consistent with the "
                "teacher acting as a **label denoiser / regularizer** for this student."
            )
        else:
            lines.append(
                f"- **Label-denoiser evidence:** pure teacher val RMSE "
                f"**{float(den['pure_teacher_val_rmse']):.2f}** is not better than pure GT "
                f"**{float(den['pure_gt_val_rmse']):.2f}**. No clear denoising advantage "
                "of teacher-only vs GT-only under this metric."
            )

    rec = analysis["recommended_alpha_beta"]
    lines += [
        "",
        "### Recommended α/β for future students",
        "",
        f"**`{rec['name']}`: α = {rec['alpha']}, β = {rec['beta']}**",
        "",
        f"Reason: {rec['reason']}.",
        "",
        "Use this pair as the default supervision strategy for subsequent "
        "architectures (FT-Transformer, TabTransformer, trajectory models) unless "
        "a new sweep on that architecture contradicts it.",
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "| File | Path |",
        "|------|------|",
        "| Metrics | `results/distillation/alpha_beta_sweep/metrics.csv` |",
        "| Comparison table | `results/distillation/alpha_beta_sweep/comparison_table.csv` |",
        "| Summary | `results/distillation/alpha_beta_sweep/summary.json` |",
        "| Best config | `results/distillation/alpha_beta_sweep/best_configuration.json` |",
        "| Plots | `results/distillation/alpha_beta_sweep/plots/` |",
        "| Per-run checkpoints | `results/distillation/alpha_beta_sweep/KD-*/` |",
        "",
        "### Out of scope",
        "",
        "- Architecture changes or hyperparameter search",
        "- Regenerating the teacher dataset",
        "- Official Rank/Final re-evaluation of the teacher",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote report %s", report_path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "distillation_dataset.parquet",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--hidden", type=str, default="1024,512")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma list of KD names to run (e.g. KD-0,KD-4)",
    )
    args = p.parse_args(argv)

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Missing {args.dataset}. Do not regenerate — restore from Step 1."
        )

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    exp = ExperimentConfig(
        seed=args.seed,
        val_fraction=args.val_fraction,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        hidden_dims=hidden,
        dropout=args.dropout,
        extras={"stage": "step3_alpha_beta_sweep"},
    )

    LOGGER.info("Loading frozen distillation dataset (read-only): %s", args.dataset)
    data = DistillationData.from_parquet(
        args.dataset,
        root=ROOT,
        val_fraction=exp.val_fraction,
        seed=exp.seed,
    )

    def model_factory(in_dim: int) -> StudentMLP:
        return StudentMLP(in_dim, hidden_dims=exp.hidden_dims, dropout=exp.dropout)

    probe = model_factory(data.in_dim)
    n_params = probe.count_parameters()
    LOGGER.info("Student params=%s in_dim=%d", f"{n_params:,}", data.in_dim)

    weights: list[KDWeightConfig] = list(DEFAULT_KD_SWEEP)
    if args.only.strip():
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        weights = [w for w in weights if w.name in want]
        if not weights:
            raise SystemExit(f"No matching KD names in --only={args.only}")

    results_root = ROOT / "results" / "distillation" / "alpha_beta_sweep"
    logs_root = ROOT / "logs" / "distillation" / "alpha_beta_sweep"
    models_root = ROOT / "models" / "distillation" / "alpha_beta_sweep"
    plots_dir = results_root / "plots"
    for d in (results_root, logs_root, models_root, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = run_kd_sweep(
        data=data,
        model_factory=model_factory,
        weights=weights,
        exp=exp,
        results_root=results_root,
        logs_root=logs_root,
        models_root=models_root,
    )
    total_seconds = time.time() - t0

    analysis = analyze_kd_sweep(rows)
    write_sweep_tables(
        rows,
        results_root,
        analysis=analysis,
        exp=exp,
        total_seconds=total_seconds,
    )

    teacher_val = float(rows[0]["teacher_val_rmse"]) if rows else None
    plot_paths = plot_kd_sweep(rows, plots_dir, teacher_val_rmse=teacher_val)

    # Also copy key plots into docs/reports/figures for the markdown report
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rels: dict[str, str] = {}
    for key, path in plot_paths.items():
        dest = fig_dir / f"fig_kd_sweep_{path.name}"
        dest.write_bytes(path.read_bytes())
        plot_rels[key] = f"figures/{dest.name}"

    report_path = ROOT / "docs" / "reports" / "distillation_alpha_beta_sweep.md"
    _write_report(
        report_path,
        rows=rows,
        analysis=analysis,
        exp=exp,
        data=data,
        total_seconds=total_seconds,
        n_params=n_params,
        plot_rels=plot_rels,
    )

    # Persist analysis next to results
    (results_root / "analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== α/β SWEEP COMPLETE ===")
    for r in sorted(rows, key=lambda x: float(x["val_rmse"])):
        print(
            f"  {r['name']}: α={r['alpha']} β={r['beta']} "
            f"val_rmse={float(r['val_rmse']):.2f} "
            f"imit={float(r['student_vs_teacher_rmse']):.2f} "
            f"gap={float(r['teacher_student_rmse_gap']):+.2f}"
        )
    rec = analysis["recommended_alpha_beta"]
    print(f"recommended: {rec['name']} α={rec['alpha']} β={rec['beta']}")
    print(f"report: {report_path}")
    print(f"results: {results_root}")


if __name__ == "__main__":
    main()
