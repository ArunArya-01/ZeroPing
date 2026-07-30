"""Step 4 — MLP capacity scaling, inference benchmarks, multi-seed reproducibility.

Fixed from Steps 1–3 (do not change):
  * frozen R3 teacher + distillation_dataset.parquet
  * feature engineering / preprocessing / flight split protocol
  * training loop (AdamW, scheduler, early stopping)
  * KD weights alpha=0.1, beta=0.9
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.benchmark import (
    benchmark_callable,
    benchmark_torch_model,
    checkpoint_size_mb,
    efficiency_metrics,
)
from aerotwin.distillation.capacity import (
    CAPACITY_TIERS,
    FIXED_ALPHA,
    FIXED_BETA,
    REPRO_SEEDS,
    CapacityTier,
)
from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.runner import ExperimentConfig, run_single_experiment
from aerotwin.distillation.trainer import set_seed
from aerotwin.engine.gap_closing import ensure_features
from aerotwin.engine.official_benchmark import apply_bases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("capacity_scaling")


def _results_root(root: Path) -> Path:
    p = root / "results" / "distillation" / "capacity_scaling"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _csv_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested fields so Polars can write CSV."""
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        hd = d.get("hidden_dims")
        if isinstance(hd, (list, tuple)):
            d["hidden_dims"] = "x".join(str(h) for h in hd)
        out.append(d)
    return out


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pl.DataFrame(_csv_safe_rows(rows)).write_csv(path)


def _parse_hidden(val: Any) -> tuple[int, ...]:
    if isinstance(val, (list, tuple)):
        return tuple(int(x) for x in val)
    s = str(val)
    if "x" in s:
        return tuple(int(x) for x in s.split("x") if x)
    # fallback "[1024, 512]"
    s = s.strip("[]() ")
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _load_val_dataframe(data: DistillationData, root: Path) -> pl.DataFrame:
    """Reconstruct val-split rows from the frozen parquet (same indices as data)."""
    df = pl.read_parquet(data.parquet_path)
    df = df.filter(
        pl.col("ground_truth").is_finite()
        & pl.col("teacher_prediction").is_finite()
        & pl.col("flight_id").is_not_null()
    )
    # Indices are positional on the filtered frame used by DistillationData
    return df[data.val_idx.tolist()]


def train_capacity(
    *,
    data: DistillationData,
    tier: CapacityTier,
    exp: ExperimentConfig,
    out_root: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    seed = exp.seed if seed is None else int(seed)
    run_name = f"{tier.name}_seed{seed}"
    exp_run = ExperimentConfig(
        seed=seed,
        val_fraction=exp.val_fraction,
        lr=exp.lr,
        weight_decay=exp.weight_decay,
        batch_size=exp.batch_size,
        max_epochs=exp.max_epochs,
        patience=exp.patience,
        min_delta=exp.min_delta,
        scheduler_factor=exp.scheduler_factor,
        scheduler_patience=exp.scheduler_patience,
        grad_clip=exp.grad_clip,
        device=exp.device,
        num_workers=exp.num_workers,
        hidden_dims=tier.hidden_dims,
        dropout=exp.dropout,
        extras={
            "tier": tier.name,
            "target_params": tier.target_params,
            "alpha": FIXED_ALPHA,
            "beta": FIXED_BETA,
            "stage": "step4_capacity",
        },
    )

    from aerotwin.distillation.runner import KDWeightConfig

    weight = KDWeightConfig(run_name, FIXED_ALPHA, FIXED_BETA)

    def factory(in_dim: int) -> StudentMLP:
        return StudentMLP(in_dim, hidden_dims=tier.hidden_dims, dropout=exp.dropout)

    out_dir = out_root / "runs" / run_name
    metrics = run_single_experiment(
        data=data,
        model_factory=factory,
        weight=weight,
        exp=exp_run,
        out_dir=out_dir,
        log_dir=ROOT / "logs" / "distillation" / "capacity_scaling" / run_name,
        model_dir=ROOT / "models" / "distillation" / "capacity_scaling" / run_name,
    )
    ckpt = out_dir / "best_model.pt"
    size_mb = checkpoint_size_mb(ckpt)
    row = {
        "name": tier.name,
        "run_name": run_name,
        "seed": seed,
        "hidden_dims": list(tier.hidden_dims),
        "hidden_dims_str": "x".join(str(h) for h in tier.hidden_dims),
        "target_params": tier.target_params,
        "n_params": metrics["n_params"],
        "checkpoint_mb": size_mb,
        "alpha": FIXED_ALPHA,
        "beta": FIXED_BETA,
        "val_rmse": metrics["val"]["student"]["rmse"],
        "val_mae": metrics["val"]["student"]["mae"],
        "val_bias": metrics["val"]["student"]["bias"],
        "val_r2": metrics["val"]["student"]["r2"],
        "teacher_val_rmse": metrics["val"]["teacher"]["rmse"],
        "student_vs_teacher_rmse": metrics["val"]["student_vs_teacher_rmse"],
        "teacher_student_rmse_gap": metrics["val"]["teacher_student_rmse_gap"],
        "train_rmse": metrics["train"]["student"]["rmse"],
        "best_epoch": metrics["best_epoch"],
        "epochs_ran": metrics["epochs_ran"],
        "train_seconds": metrics["train_seconds"],
        "device": metrics["device"],
        "checkpoint": str(ckpt),
    }
    return row


def _load_student(ckpt_path: Path, in_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> StudentMLP:
    model = StudentMLP(in_dim, hidden_dims=hidden_dims, dropout=dropout)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model


def benchmark_students_and_teacher(
    *,
    data: DistillationData,
    capacity_rows: list[dict[str, Any]],
    root: Path,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Benchmark each capacity checkpoint + frozen teacher ensemble."""
    x_val = data.x_val
    y_teacher = data.y_teacher_val
    teacher_rmse = float(np.sqrt(np.mean((y_teacher - data.y_gt_val) ** 2)))

    # --- Teacher ---
    cache_path = root / "cache" / "r3_teacher_distillation_bundle.pkl"
    teacher_rows: list[dict[str, Any]] = []
    teacher_size_mb = float("nan")
    teacher_cpu_single_ms = float("nan")
    teacher_cpu_batch_ms = float("nan")
    teacher_ram = None

    if cache_path.exists():
        teacher_size_mb = checkpoint_size_mb(cache_path)
        val_df = _load_val_dataframe(data, root)
        # prepare_xy expects official column names.
        if "actual_fuel_kg" not in val_df.columns and "ground_truth" in val_df.columns:
            val_df = val_df.with_columns(pl.col("ground_truth").alias("actual_fuel_kg"))
        with open(cache_path, "rb") as f:
            bundle = pickle.load(f)
        feat_cols = list(bundle["feat_cols"])
        val_df = ensure_features(val_df, feat_cols)
        full_models = bundle["full_models"]
        meta = bundle["meta"]
        cal = bundle["cal_phase"]
        n_val = len(val_df)
        positions = np.arange(n_val, dtype=np.int64)

        def teacher_predict_idx(idx: np.ndarray) -> np.ndarray:
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            sub = val_df[idx.tolist()]
            P = apply_bases(full_models, sub, feat_cols)
            ridge = np.asarray(meta.predict(P), dtype=np.float64)
            return np.asarray(cal.transform(sub, ridge), dtype=np.float64)

        # Warm + time using index batches (same batching structure as students)
        LOGGER.info("Benchmarking frozen R3 teacher ensemble (CPU)...")
        # fewer iters for teacher (slow sklearn stack)
        bench = benchmark_callable(
            teacher_predict_idx,
            positions,  # 1d indices
            batch_size=batch_size,
            n_warmup=2,
            n_iters=3,
            single_samples=50,
            label="R3_teacher",
        )
        # Override: callable expects indices; our benchmark_callable with 1d works
        teacher_cpu_single_ms = bench["single_latency_ms"]
        teacher_cpu_batch_ms = bench["batch_latency_ms"]
        teacher_ram = bench.get("peak_ram_mb_batch")

        teacher_rows.append(
            {
                "name": "R3_teacher",
                "kind": "teacher",
                "n_params": None,
                "checkpoint_mb": teacher_size_mb,
                "device": "cpu",
                "single_latency_ms": teacher_cpu_single_ms,
                "batch_latency_ms": teacher_cpu_batch_ms,
                "single_throughput_sps": bench["single_throughput_sps"],
                "batch_throughput_sps": bench["batch_throughput_sps"],
                "peak_ram_mb": teacher_ram,
                "peak_gpu_mb": None,
                "val_rmse": teacher_rmse,
            }
        )
        # Teacher has no GPU path (sklearn)
        LOGGER.info(
            "Teacher CPU single=%.3f ms/sample batch=%.3f ms/sample size=%.1f MB",
            teacher_cpu_single_ms,
            teacher_cpu_batch_ms,
            teacher_size_mb,
        )
    else:
        LOGGER.warning("Teacher cache missing at %s — skip teacher bench", cache_path)

    # --- Students ---
    bench_rows: list[dict[str, Any]] = list(teacher_rows)
    has_cuda = torch.cuda.is_available()

    for row in capacity_rows:
        ckpt = Path(row["checkpoint"])
        if not ckpt.exists():
            LOGGER.warning("Missing checkpoint %s", ckpt)
            continue
        hidden = _parse_hidden(row["hidden_dims"])
        model = _load_student(ckpt, data.in_dim, hidden, dropout=0.1)
        size_mb = float(row["checkpoint_mb"])

        for dev_name in (["cpu"] + (["cuda"] if has_cuda else [])):
            device = torch.device(dev_name)
            # lighter on CPU for large models
            n_warmup = 10 if dev_name == "cuda" else 3
            n_iters = 30 if dev_name == "cuda" else 5
            n_single = 200 if dev_name == "cuda" else 50
            LOGGER.info("Benchmarking %s on %s...", row["name"], dev_name)
            b = benchmark_torch_model(
                model,
                x_val,
                device=device,
                batch_size=batch_size,
                n_warmup=n_warmup,
                n_iters=n_iters,
                single_samples=n_single,
            )
            lat_ms = b["batch_latency_ms"]
            eff = efficiency_metrics(
                val_rmse=float(row["val_rmse"]),
                n_params=int(row["n_params"]),
                size_mb=size_mb,
                latency_ms=lat_ms,
                teacher_val_rmse=teacher_rmse,
                teacher_latency_ms=teacher_cpu_batch_ms
                if np.isfinite(teacher_cpu_batch_ms)
                else None,
                teacher_size_mb=teacher_size_mb if np.isfinite(teacher_size_mb) else None,
                teacher_ram_mb=teacher_ram,
                student_ram_mb=b.get("peak_ram_mb_batch"),
            )
            bench_rows.append(
                {
                    "name": row["name"],
                    "kind": "student",
                    "n_params": row["n_params"],
                    "checkpoint_mb": size_mb,
                    "device": dev_name,
                    "single_latency_ms": b["single_latency_ms"],
                    "batch_latency_ms": b["batch_latency_ms"],
                    "single_throughput_sps": b["single_throughput_sps"],
                    "batch_throughput_sps": b["batch_throughput_sps"],
                    "peak_ram_mb": b.get("peak_ram_mb_batch"),
                    "peak_gpu_mb": b.get("peak_gpu_mb_batch"),
                    "val_rmse": row["val_rmse"],
                    **eff,
                }
            )

    return bench_rows


def analyze_scaling(
    capacity_rows: list[dict[str, Any]],
    bench_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = sorted(capacity_rows, key=lambda r: int(r["n_params"]))
    best = min(rows, key=lambda r: float(r["val_rmse"]))
    best_rmse = float(best["val_rmse"])

    def smallest_within(delta: float) -> dict[str, Any] | None:
        cands = [r for r in rows if float(r["val_rmse"]) <= best_rmse + delta]
        if not cands:
            return None
        return min(cands, key=lambda r: int(r["n_params"]))

    # Saturation: improvement from largest step (Medium→Large, Large→XLarge)
    rmses = [float(r["val_rmse"]) for r in rows]
    params = [int(r["n_params"]) for r in rows]
    deltas = [rmses[i] - rmses[i + 1] for i in range(len(rmses) - 1)]
    saturates = all(d < 1.0 for d in deltas[-2:]) if len(deltas) >= 2 else False

    # Latency scaling (CPU batch)
    cpu_bench = [b for b in bench_rows if b.get("kind") == "student" and b.get("device") == "cpu"]
    lat_by_name = {b["name"]: float(b["batch_latency_ms"]) for b in cpu_bench}
    lat_vals = [lat_by_name[r["name"]] for r in rows if r["name"] in lat_by_name]
    lat_params = [int(r["n_params"]) for r in rows if r["name"] in lat_by_name]
    latency_linear = None
    if len(lat_vals) >= 3:
        # correlation of log params vs log latency
        lp = np.log(np.asarray(lat_params, dtype=np.float64))
        ll = np.log(np.asarray(lat_vals, dtype=np.float64) + 1e-12)
        corr = float(np.corrcoef(lp, ll)[0, 1])
        latency_linear = {"log_param_log_latency_corr": corr, "near_linear": abs(corr - 1.0) < 0.25 or corr > 0.9}

    # Best efficiency: minimize RMSE * latency (CPU batch) among students with both
    tradeoff = None
    scored = []
    for r in rows:
        if r["name"] in lat_by_name:
            scored.append(
                {
                    "name": r["name"],
                    "val_rmse": r["val_rmse"],
                    "batch_latency_ms": lat_by_name[r["name"]],
                    "score_rmse_x_ms": float(r["val_rmse"]) * lat_by_name[r["name"]],
                    "n_params": r["n_params"],
                }
            )
    if scored:
        tradeoff = min(scored, key=lambda x: x["score_rmse_x_ms"])

    # Seed statistics
    seed_stats = None
    if seed_rows:
        arr = np.array([float(r["val_rmse"]) for r in seed_rows], dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        # 95% CI (t approx; n small)
        if len(arr) > 1:
            # simple normal approx: 1.96 * se
            se = std / np.sqrt(len(arr))
            ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
        else:
            ci_lo = ci_hi = mean
        teacher_rmse = float(seed_rows[0]["teacher_val_rmse"])
        best_s = min(seed_rows, key=lambda r: float(r["val_rmse"]))
        worst_s = max(seed_rows, key=lambda r: float(r["val_rmse"]))
        # consistent improvement: all seeds better than teacher?
        all_better = all(float(r["val_rmse"]) < teacher_rmse for r in seed_rows)
        ci_excludes_teacher = ci_hi < teacher_rmse
        seed_stats = {
            "n": len(arr),
            "mean_val_rmse": mean,
            "std_val_rmse": std,
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
            "best_seed": best_s["seed"],
            "best_val_rmse": best_s["val_rmse"],
            "worst_seed": worst_s["seed"],
            "worst_val_rmse": worst_s["val_rmse"],
            "teacher_val_rmse": teacher_rmse,
            "all_seeds_better_than_teacher": all_better,
            "ci95_entirely_below_teacher": ci_excludes_teacher,
            "mean_gap_vs_teacher": mean - teacher_rmse,
        }

    return {
        "best_model": {
            "name": best["name"],
            "n_params": best["n_params"],
            "val_rmse": best["val_rmse"],
            "student_vs_teacher_rmse": best["student_vs_teacher_rmse"],
            "teacher_student_rmse_gap": best["teacher_student_rmse_gap"],
        },
        "smallest_within_1kg": smallest_within(1.0),
        "smallest_within_2kg": smallest_within(2.0),
        "rmse_by_capacity": [
            {"name": r["name"], "n_params": r["n_params"], "val_rmse": r["val_rmse"]} for r in rows
        ],
        "step_improvements_kg": [
            {
                "from": rows[i]["name"],
                "to": rows[i + 1]["name"],
                "delta_rmse": deltas[i],
            }
            for i in range(len(deltas))
        ],
        "performance_saturates": saturates,
        "latency_scaling": latency_linear,
        "best_accuracy_efficiency_tradeoff": tradeoff,
        "seed_statistics": seed_stats,
    }


def plot_all(
    capacity_rows: list[dict[str, Any]],
    bench_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    plots_dir: Path,
    teacher_rmse: float,
) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(capacity_rows, key=lambda r: int(r["n_params"]))
    params_m = np.array([r["n_params"] / 1e6 for r in rows], dtype=np.float64)
    names = [r["name"] for r in rows]
    rmse = np.array([r["val_rmse"] for r in rows], dtype=np.float64)
    imit = np.array([r["student_vs_teacher_rmse"] for r in rows], dtype=np.float64)
    tsec = np.array([r["train_seconds"] for r in rows], dtype=np.float64)
    size = np.array([r["checkpoint_mb"] for r in rows], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
            "savefig.dpi": 160,
        }
    )
    paths: dict[str, Path] = {}

    def _save(fig, key: str) -> None:
        p = plots_dir / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths[key] = p

    def _annotate(ax, xs, ys):
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    # 1 params vs rmse
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(params_m, rmse, "o-", color="#1f77b4", lw=2, ms=8)
    ax.axhline(teacher_rmse, color="k", ls=":", lw=1.5, label=f"Teacher ({teacher_rmse:.2f})")
    _annotate(ax, params_m, rmse)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Validation RMSE (kg)")
    ax.set_title("Capacity scaling: parameters vs validation RMSE")
    ax.legend()
    _save(fig, "params_vs_val_rmse")

    # 2 params vs student-teacher
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(params_m, imit, "s-", color="#ff7f0e", lw=2, ms=8)
    _annotate(ax, params_m, imit)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Student ↔ Teacher RMSE (kg)")
    ax.set_title("Capacity scaling: teacher imitation")
    _save(fig, "params_vs_student_teacher_rmse")

    # 3 train time
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(params_m, tsec / 60.0, "D-", color="#2ca02c", lw=2, ms=8)
    _annotate(ax, params_m, tsec / 60.0)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Training time (minutes)")
    ax.set_title("Capacity scaling: training time")
    _save(fig, "params_vs_train_time")

    # 4 model size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(params_m, size, "^-", color="#9467bd", lw=2, ms=8)
    _annotate(ax, params_m, size)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Checkpoint size (MB)")
    ax.set_title("Capacity scaling: model size")
    _save(fig, "params_vs_model_size")

    # latency plots
    cpu = {b["name"]: b for b in bench_rows if b.get("kind") == "student" and b.get("device") == "cpu"}
    gpu = {b["name"]: b for b in bench_rows if b.get("kind") == "student" and b.get("device") == "cuda"}
    teacher_b = next((b for b in bench_rows if b.get("name") == "R3_teacher"), None)

    def _lat_plot(device_map: dict, key: str, title: str, color: str) -> None:
        xs, ys, ns = [], [], []
        for r in rows:
            if r["name"] in device_map:
                xs.append(r["n_params"] / 1e6)
                ys.append(device_map[r["name"]]["batch_latency_ms"])
                ns.append(r["name"])
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=8)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        if teacher_b and device_map is cpu:
            ax.axhline(
                teacher_b["batch_latency_ms"],
                color="k",
                ls=":",
                label=f"Teacher CPU ({teacher_b['batch_latency_ms']:.3f} ms)",
            )
            ax.legend()
        ax.set_xlabel("Parameters (millions)")
        ax.set_ylabel("Batch latency (ms/sample)")
        ax.set_title(title)
        _save(fig, key)

    _lat_plot(cpu, "params_vs_cpu_latency", "Capacity scaling: CPU batch latency", "#d62728")
    _lat_plot(gpu, "params_vs_gpu_latency", "Capacity scaling: GPU batch latency", "#17becf")

    # Pareto accuracy vs latency (CPU)
    if cpu:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [cpu[r["name"]]["batch_latency_ms"] for r in rows if r["name"] in cpu]
        ys = [r["val_rmse"] for r in rows if r["name"] in cpu]
        ns = [r["name"] for r in rows if r["name"] in cpu]
        ax.plot(xs, ys, "o-", color="#1f77b4", lw=2, ms=9)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
        if teacher_b:
            ax.scatter(
                [teacher_b["batch_latency_ms"]],
                [teacher_rmse],
                marker="*",
                s=180,
                color="k",
                label="R3 teacher",
                zorder=5,
            )
            ax.legend()
        ax.set_xlabel("CPU batch latency (ms/sample)")
        ax.set_ylabel("Validation RMSE (kg)")
        ax.set_title("Accuracy–latency Pareto (lower-left is better)")
        _save(fig, "pareto_accuracy_vs_latency")

    # accuracy vs model size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(size, rmse, "o-", color="#8c564b", lw=2, ms=8)
    _annotate(ax, size, rmse)
    ax.axhline(teacher_rmse, color="k", ls=":", lw=1.5, label="Teacher RMSE")
    ax.set_xlabel("Checkpoint size (MB)")
    ax.set_ylabel("Validation RMSE (kg)")
    ax.set_title("Accuracy vs model size")
    ax.legend()
    _save(fig, "accuracy_vs_model_size")

    # seed stability
    if seed_rows:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        vals = [float(r["val_rmse"]) for r in seed_rows]
        seeds = [str(r["seed"]) for r in seed_rows]
        ax.boxplot(vals, vert=True, widths=0.4)
        ax.scatter(np.ones(len(vals)), vals, color="#1f77b4", zorder=3)
        for i, (s, v) in enumerate(zip(seeds, vals)):
            ax.annotate(s, (1.05, v), fontsize=8)
        ax.axhline(teacher_rmse, color="k", ls=":", label=f"Teacher ({teacher_rmse:.2f})")
        ax.set_xticks([1])
        ax.set_xticklabels([seed_rows[0]["name"] if "name" in seed_rows[0] else "best tier"])
        ax.set_ylabel("Validation RMSE (kg)")
        ax.set_title("Seed stability (5 independent runs)")
        ax.legend()
        _save(fig, "seed_stability")

    LOGGER.info("Wrote %d figures to %s", len(paths), plots_dir)
    return paths


def write_report(
    path: Path,
    *,
    capacity_rows: list[dict[str, Any]],
    bench_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    total_seconds: float,
    data: DistillationData,
) -> None:
    best = analysis["best_model"]
    seed_stats = analysis.get("seed_statistics") or {}
    teacher_rmse = float(capacity_rows[0]["teacher_val_rmse"]) if capacity_rows else float("nan")

    lines = [
        "# MLP Capacity Scaling Report",
        "",
        "**Stage:** AeroTwin Distillation - Step 4",
        "",
        "Fixed KD supervision: **alpha=0.1, beta=0.9** (from Step 3).",
        "Teacher, dataset, split protocol, and training loop are unchanged.",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "| Setting | Value |",
        "|---------|------:|",
        f"| Dataset | `distillation_dataset.parquet` |",
        f"| Samples | {data.n_samples:,} |",
        f"| Train / val rows | {len(data.train_idx):,} / {len(data.val_idx):,} |",
        f"| Input dim | {data.in_dim} |",
        f"| alpha / beta | {FIXED_ALPHA} / {FIXED_BETA} |",
        f"| Seed (capacity) | 42 |",
        f"| Repro seeds | {list(REPRO_SEEDS)} |",
        f"| Max epochs / patience | 80 / 12 |",
        f"| Optimizer | AdamW 1e-3 |",
        f"| Wall time | {total_seconds/60:.1f} min |",
        "",
        "Architecture family: Linear → ReLU → LayerNorm → Dropout (×2) → Linear head.",
        "",
        "### Capacity tiers",
        "",
        "| Model | Hidden | Target | Actual params |",
        "|-------|--------|-------:|--------------:|",
    ]
    for r in sorted(capacity_rows, key=lambda x: int(x["n_params"])):
        hd = r.get("hidden_dims_str") or r.get("hidden_dims")
        lines.append(
            f"| {r['name']} | {hd} | {r['target_params']} | {int(r['n_params']):,} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Scaling results",
        "",
        "| Model | Params | Val RMSE | MAE | Bias | R2 | Student-Teacher | Gap | Epochs | Train s | Size MB |",
        "|-------|-------:|---------:|----:|-----:|---:|----------------:|----:|-------:|--------:|--------:|",
    ]
    for r in sorted(capacity_rows, key=lambda x: int(x["n_params"])):
        lines.append(
            f"| {r['name']} | {int(r['n_params']):,} | {float(r['val_rmse']):.2f} "
            f"| {float(r['val_mae']):.2f} | {float(r['val_bias']):+.2f} | {float(r['val_r2']):.4f} "
            f"| {float(r['student_vs_teacher_rmse']):.2f} | {float(r['teacher_student_rmse_gap']):+.2f} "
            f"| {int(r['epochs_ran'])} | {float(r['train_seconds']):.0f} | {float(r['checkpoint_mb']):.2f} |"
        )

    lines += [
        "",
        f"Teacher val RMSE (soft labels on this split): **{teacher_rmse:.2f} kg**.",
        "",
        "### Inference benchmarks",
        "",
        "| Name | Device | Single ms/sample | Batch ms/sample | Single sps | Batch sps | Peak RAM MB | Peak GPU MB | Size MB |",
        "|------|--------|-----------------:|----------------:|-----------:|----------:|------------:|------------:|--------:|",
    ]
    for b in bench_rows:
        lines.append(
            f"| {b.get('name')} | {b.get('device')} "
            f"| {float(b.get('single_latency_ms') or float('nan')):.4f} "
            f"| {float(b.get('batch_latency_ms') or float('nan')):.4f} "
            f"| {float(b.get('single_throughput_sps') or float('nan')):.1f} "
            f"| {float(b.get('batch_throughput_sps') or float('nan')):.1f} "
            f"| {b.get('peak_ram_mb')} | {b.get('peak_gpu_mb')} "
            f"| {float(b.get('checkpoint_mb') or float('nan')):.2f} |"
        )

    lines += [
        "",
        "### Efficiency (students, CPU batch latency)",
        "",
        "| Model | RMSE/Mparams | RMSE/MB | RMSE/ms | Speedup vs teacher |",
        "|-------|-------------:|--------:|--------:|-------------------:|",
    ]
    for b in bench_rows:
        if b.get("kind") != "student" or b.get("device") != "cpu":
            continue
        lines.append(
            f"| {b['name']} | {float(b.get('rmse_per_million_params', float('nan'))):.2f} "
            f"| {float(b.get('rmse_per_mb', float('nan'))):.2f} "
            f"| {float(b.get('rmse_per_ms', float('nan'))):.2f} "
            f"| {float(b.get('latency_speedup_vs_teacher', float('nan'))):.1f}x |"
        )

    lines += [
        "",
        "### Figures",
        "",
        "![params vs RMSE](figures/fig_cap_params_vs_val_rmse.png)",
        "",
        "![params vs imitation](figures/fig_cap_params_vs_student_teacher_rmse.png)",
        "",
        "![params vs train time](figures/fig_cap_params_vs_train_time.png)",
        "",
        "![params vs size](figures/fig_cap_params_vs_model_size.png)",
        "",
        "![CPU latency](figures/fig_cap_params_vs_cpu_latency.png)",
        "",
        "![GPU latency](figures/fig_cap_params_vs_gpu_latency.png)",
        "",
        "![Pareto](figures/fig_cap_pareto_accuracy_vs_latency.png)",
        "",
        "![size](figures/fig_cap_accuracy_vs_model_size.png)",
        "",
        "![seeds](figures/fig_cap_seed_stability.png)",
        "",
        "---",
        "",
        "## Scaling analysis",
        "",
        f"- **Best model:** {best['name']} ({int(best['n_params']):,} params) "
        f"val RMSE **{float(best['val_rmse']):.2f} kg** "
        f"(gap vs teacher {float(best['teacher_student_rmse_gap']):+.2f} kg).",
    ]
    w1 = analysis.get("smallest_within_1kg")
    w2 = analysis.get("smallest_within_2kg")
    if w1:
        lines.append(
            f"- **Smallest within 1 kg of best:** {w1['name']} "
            f"({int(w1['n_params']):,} params, RMSE {float(w1['val_rmse']):.2f})."
        )
    if w2:
        lines.append(
            f"- **Smallest within 2 kg of best:** {w2['name']} "
            f"({int(w2['n_params']):,} params, RMSE {float(w2['val_rmse']):.2f})."
        )
    lines.append(
        f"- **Saturation:** "
        + (
            "Yes — late capacity steps improve RMSE by <1 kg each."
            if analysis.get("performance_saturates")
            else "Not fully — later capacity steps still move RMSE by ≥1 kg, or improvement remains material."
        )
    )
    if analysis.get("latency_scaling"):
        ls = analysis["latency_scaling"]
        lines.append(
            f"- **Latency vs capacity:** log-param vs log-latency correlation "
            f"{ls.get('log_param_log_latency_corr', float('nan')):.3f} "
            f"(near-linear={ls.get('near_linear')})."
        )
    if analysis.get("best_accuracy_efficiency_tradeoff"):
        t = analysis["best_accuracy_efficiency_tradeoff"]
        lines.append(
            f"- **Best accuracy/efficiency tradeoff (min RMSE×CPU ms):** {t['name']} "
            f"(score={float(t['score_rmse_x_ms']):.2f})."
        )

    lines += ["", "## Reproducibility (best capacity, 5 seeds)", ""]
    if seed_stats:
        lines += [
            f"| Stat | Value |",
            f"|------|------:|",
            f"| Tier | {seed_rows[0].get('name', best['name'])} |",
            f"| Mean val RMSE | {seed_stats['mean_val_rmse']:.2f} |",
            f"| Std | {seed_stats['std_val_rmse']:.2f} |",
            f"| 95% CI | [{seed_stats['ci95_low']:.2f}, {seed_stats['ci95_high']:.2f}] |",
            f"| Best seed | {seed_stats['best_seed']} ({seed_stats['best_val_rmse']:.2f}) |",
            f"| Worst seed | {seed_stats['worst_seed']} ({seed_stats['worst_val_rmse']:.2f}) |",
            f"| Teacher val RMSE | {seed_stats['teacher_val_rmse']:.2f} |",
            f"| Mean gap vs teacher | {seed_stats['mean_gap_vs_teacher']:+.2f} |",
            f"| All seeds better than teacher | {seed_stats['all_seeds_better_than_teacher']} |",
            f"| 95% CI entirely below teacher | {seed_stats['ci95_entirely_below_teacher']} |",
            "",
        ]
        if seed_stats["ci95_entirely_below_teacher"]:
            lines.append(
                "**Conclusion:** Across five seeds, the mean student RMSE is below the teacher "
                "and the 95% CI does not include the teacher RMSE — the improvement is consistent "
                "on this flight-holdout split (not official Rank/Final)."
            )
        elif seed_stats["all_seeds_better_than_teacher"]:
            lines.append(
                "**Conclusion:** All five seeds beat the teacher point estimate, but the 95% CI "
                "is not entirely below the teacher — treat the improvement as promising but not "
                "over-claim statistical separation."
            )
        else:
            lines.append(
                "**Conclusion:** Not all seeds beat the teacher (or CIs overlap). Do not claim "
                "consistent superiority over the teacher from this experiment alone."
            )
        lines.append("")
        lines.append("| Seed | Val RMSE | Gap | Student-Teacher |")
        lines.append("|-----:|---------:|----:|----------------:|")
        for r in sorted(seed_rows, key=lambda x: int(x["seed"])):
            lines.append(
                f"| {r['seed']} | {float(r['val_rmse']):.2f} "
                f"| {float(r['teacher_student_rmse_gap']):+.2f} "
                f"| {float(r['student_vs_teacher_rmse']):.2f} |"
            )
    else:
        lines.append("Seed study not run.")

    lines += [
        "",
        "---",
        "",
        "## Discussion",
        "",
        "1. Capacity scaling isolates neural width under fixed KD weights (0.1 / 0.9).",
        "2. If RMSE flattens while parameters grow, the teacher soft labels are largely absorbed "
        "and further MLP capacity yields diminishing returns — a signal that architecture "
        "(e.g. transformers) may matter more than raw width.",
        "3. Inference benchmarks compare deployable student speed/size against the multi-model "
        "R3 ensemble on CPU; GPU numbers apply to students only.",
        "4. Student flight-holdout RMSE is **not** official Rank/Final Combined RMSE.",
        "",
        "## Conclusions",
        "",
        f"- Best capacity under seed 42: **{best['name']}**.",
        f"- Minimum capacity within 1 kg of best: "
        + (
            f"**{w1['name']}**"
            if w1
            else "n/a"
        )
        + ".",
        "- Reproducibility: see seed section above.",
        "- Proceed to FT-Transformer / TabTransformer only if capacity saturates or the "
        "accuracy–latency frontier justifies architecture change; use α=0.1, β=0.9.",
        "",
        "## Artifacts",
        "",
        "| Path | Content |",
        "|------|---------|",
        "| `results/distillation/capacity_scaling/metrics.csv` | Capacity metrics |",
        "| `results/distillation/capacity_scaling/benchmark_results.csv` | Latency / memory |",
        "| `results/distillation/capacity_scaling/seed_statistics.json` | Multi-seed stats |",
        "| `results/distillation/capacity_scaling/best_model.json` | Best tier |",
        "| `results/distillation/capacity_scaling/plots/` | Figures |",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote report %s", path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "distillation_dataset.parquet")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--bench-batch-size", type=int, default=256)
    p.add_argument("--skip-train", action="store_true", help="Only re-bench / replot from existing runs")
    p.add_argument("--skip-seeds", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument(
        "--only-tiers",
        type=str,
        default="",
        help="Comma list of tier names, e.g. Tiny,Medium",
    )
    args = p.parse_args(argv)

    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    out = _results_root(ROOT)
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading frozen distillation dataset (read-only)")
    data = DistillationData.from_parquet(
        args.dataset, root=ROOT, val_fraction=0.2, seed=42
    )
    exp = ExperimentConfig(
        seed=42,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_dims=(1024, 512),
        dropout=0.1,
        extras={"stage": "step4"},
    )

    tiers = list(CAPACITY_TIERS)
    if args.only_tiers.strip():
        want = {x.strip() for x in args.only_tiers.split(",") if x.strip()}
        tiers = [t for t in tiers if t.name in want]

    t0 = time.time()
    capacity_rows: list[dict[str, Any]] = []

    metrics_path = out / "metrics.csv"
    if args.skip_train and metrics_path.exists():
        capacity_rows = pl.read_csv(metrics_path).to_dicts()
        LOGGER.info("Loaded %d capacity rows from %s", len(capacity_rows), metrics_path)
    else:
        for tier in tiers:
            run_name = f"{tier.name}_seed42"
            existing = out / "runs" / run_name / "metrics.json"
            if existing.exists():
                LOGGER.info("===== Capacity %s (resume metrics) =====", tier.name)
                m = json.loads(existing.read_text(encoding="utf-8"))
                ckpt = out / "runs" / run_name / "best_model.pt"
                row = {
                    "name": tier.name,
                    "run_name": run_name,
                    "seed": 42,
                    "hidden_dims": list(tier.hidden_dims),
                    "hidden_dims_str": "x".join(str(h) for h in tier.hidden_dims),
                    "target_params": tier.target_params,
                    "n_params": m["n_params"],
                    "checkpoint_mb": checkpoint_size_mb(ckpt),
                    "alpha": FIXED_ALPHA,
                    "beta": FIXED_BETA,
                    "val_rmse": m["val"]["student"]["rmse"],
                    "val_mae": m["val"]["student"]["mae"],
                    "val_bias": m["val"]["student"]["bias"],
                    "val_r2": m["val"]["student"]["r2"],
                    "teacher_val_rmse": m["val"]["teacher"]["rmse"],
                    "student_vs_teacher_rmse": m["val"]["student_vs_teacher_rmse"],
                    "teacher_student_rmse_gap": m["val"]["teacher_student_rmse_gap"],
                    "train_rmse": m["train"]["student"]["rmse"],
                    "best_epoch": m["best_epoch"],
                    "epochs_ran": m["epochs_ran"],
                    "train_seconds": m["train_seconds"],
                    "device": m["device"],
                    "checkpoint": str(ckpt),
                }
            else:
                LOGGER.info("===== Capacity %s %s =====", tier.name, tier.hidden_dims)
                row = train_capacity(data=data, tier=tier, exp=exp, out_root=out, seed=42)
            capacity_rows.append(row)
            _write_rows_csv(capacity_rows, metrics_path)

    # Multi-seed on best capacity
    seed_rows: list[dict[str, Any]] = []
    seed_path = out / "seed_runs.csv"
    best_tier_name = min(capacity_rows, key=lambda r: float(r["val_rmse"]))["name"]
    best_tier = next(t for t in CAPACITY_TIERS if t.name == best_tier_name)

    if not args.skip_seeds:
        for seed in REPRO_SEEDS:
            # Reuse capacity seed-42 run if present
            if seed == 42:
                base = next(r for r in capacity_rows if r["name"] == best_tier_name and int(r["seed"]) == 42)
                seed_rows.append(dict(base))
                continue
            LOGGER.info("===== Repro %s seed=%d =====", best_tier.name, seed)
            row = train_capacity(data=data, tier=best_tier, exp=exp, out_root=out, seed=seed)
            seed_rows.append(row)
        _write_rows_csv(seed_rows, seed_path)
    elif seed_path.exists():
        seed_rows = pl.read_csv(seed_path).to_dicts()

    # Benchmarks
    bench_rows: list[dict[str, Any]] = []
    bench_path = out / "benchmark_results.csv"
    if not args.skip_bench:
        # capacity seed-42 rows only for scaling bench
        cap42 = [r for r in capacity_rows if int(r.get("seed", 42)) == 42]
        bench_rows = benchmark_students_and_teacher(
            data=data,
            capacity_rows=cap42,
            root=ROOT,
            batch_size=args.bench_batch_size,
        )
        _write_rows_csv(bench_rows, bench_path)
    elif bench_path.exists():
        bench_rows = pl.read_csv(bench_path).to_dicts()

    total_seconds = time.time() - t0
    analysis = analyze_scaling(capacity_rows, bench_rows, seed_rows)
    teacher_rmse = float(capacity_rows[0]["teacher_val_rmse"])

    # Tables
    _write_rows_csv(capacity_rows, out / "metrics.csv")
    cmp_rows = sorted(capacity_rows, key=lambda r: float(r["val_rmse"]))
    _write_rows_csv(cmp_rows, out / "comparison_table.csv")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "total_seconds": total_seconds,
                "alpha": FIXED_ALPHA,
                "beta": FIXED_BETA,
                "capacity": capacity_rows,
                "analysis": analysis,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if analysis.get("seed_statistics"):
        (out / "seed_statistics.json").write_text(
            json.dumps(analysis["seed_statistics"], indent=2, default=str),
            encoding="utf-8",
        )
    (out / "best_model.json").write_text(
        json.dumps(analysis["best_model"], indent=2, default=str),
        encoding="utf-8",
    )

    plot_paths = plot_all(capacity_rows, bench_rows, seed_rows, plots_dir, teacher_rmse)
    # Copy plots to docs figures
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, pth in plot_paths.items():
        dest = fig_dir / f"fig_cap_{pth.name}"
        dest.write_bytes(pth.read_bytes())

    write_report(
        ROOT / "docs" / "reports" / "capacity_scaling_report.md",
        capacity_rows=capacity_rows,
        bench_rows=bench_rows,
        seed_rows=seed_rows,
        analysis=analysis,
        total_seconds=total_seconds,
        data=data,
    )

    print("\n=== CAPACITY SCALING COMPLETE ===")
    for r in sorted(capacity_rows, key=lambda x: int(x["n_params"])):
        print(
            f"  {r['name']}: params={int(r['n_params']):,} "
            f"val_rmse={float(r['val_rmse']):.2f} "
            f"gap={float(r['teacher_student_rmse_gap']):+.2f}"
        )
    print("best:", analysis["best_model"])
    if analysis.get("seed_statistics"):
        print("seed mean±std:", analysis["seed_statistics"]["mean_val_rmse"], analysis["seed_statistics"]["std_val_rmse"])
    print("results:", out)


if __name__ == "__main__":
    main()
