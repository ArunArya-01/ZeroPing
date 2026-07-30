"""Official Combined (Rank + Final) evaluation of frozen Large / XLarge MLPs.

Evaluation only — no training, no checkpoint or preprocessing changes.

Protocol B (PRC Combined):
  combined_rmse = RMSE(concat(y_rank, y_final), concat(p_rank, p_final))
  same aggregation as aerotwin.engine.gap_closing.full_scorecard / R3 teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
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

from aerotwin.distillation.capacity import FIXED_ALPHA, FIXED_BETA
from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.engine.gap_closing import clean_featured, ensure_features, group_phase, rmse as kg_rmse
from aerotwin.engine.mass_model import enrich_mass_from_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("combined_evaluation")

CHECKPOINTS = {
    "Large": {
        "path": ROOT
        / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt",
        "hidden_dims": (1792, 1024),
        "n_params": 2_887_425,
        "label": "Large MLP (~3M)",
    },
    "XLarge": {
        "path": ROOT
        / "results/distillation/capacity_scaling/runs/XLarge_seed42/best_model.pt",
        "hidden_dims": (2560, 2048),
        "n_params": 6_748_673,
        "label": "XLarge MLP (~6.75M)",
    },
}

VAL_METRICS_PATHS = {
    "Large": ROOT
    / "results/distillation/capacity_scaling/runs/Large_seed42/metrics.json",
    "XLarge": ROOT
    / "results/distillation/capacity_scaling/runs/XLarge_seed42/metrics.json",
}

STEP5_METRICS = ROOT / "results/distillation/test_evaluation/metrics.json"
STEP5_PRED = {
    "Large": ROOT / "results/distillation/test_evaluation/predictions_large.parquet",
    "XLarge": ROOT / "results/distillation/test_evaluation/predictions_xlarge.parquet",
}

# Canonical teacher (audit + official Combined)
TEACHER_FINAL_RMSE = 213.62
TEACHER_COMBINED_RMSE = 221.33
TEACHER_RANK_RMSE = 232.53  # official R3 campaign Rank component


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(1024 * 1024)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _prepare(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "actual_fuel_kg" not in df.columns and "fuel_kg" in df.columns:
        df = df.with_columns(pl.col("fuel_kg").alias("actual_fuel_kg"))
    df = clean_featured(df)
    df = enrich_mass_from_columns(df)
    return df


def _fit_preprocessors() -> DistillationData:
    return DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet",
        root=ROOT,
        val_fraction=0.2,
        seed=42,
    )


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
        [
            df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64)
            for c in numeric_cols
        ]
    )
    for j in range(num.shape[1]):
        bad = ~np.isfinite(num[:, j])
        if bad.any():
            col = num[:, j].copy()
            col[bad] = medians[j]
            num[:, j] = col

    x_num = data.scaler.transform(num).astype(np.float32)
    cat_pdf = df.select(
        [pl.col(c).cast(pl.Utf8).fill_null("missing") for c in cat_cols]
    ).to_pandas()
    x_cat = data.ohe.transform(cat_pdf).astype(np.float32)
    x = np.hstack([x_num, x_cat]).astype(np.float32)
    y = df["actual_fuel_kg"].to_numpy().astype(np.float64)
    return x, y


@torch.no_grad()
def _predict(
    ckpt: Path, hidden: tuple[int, ...], x: np.ndarray, device: torch.device
) -> np.ndarray:
    model = StudentMLP(int(x.shape[1]), hidden_dims=hidden, dropout=0.1)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device).eval()
    out: list[np.ndarray] = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), 4096):
        out.append(model(xt[i : i + 4096].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _full(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    base = regression_metrics(y, p)
    base.update(
        {
            "mean_residual": float(np.mean(err)),
            "median_residual": float(np.median(err)),
            "std_residual": float(np.std(err)),
            "p95_abs_error": float(np.percentile(np.abs(err), 95)),
            "max_abs_error": float(np.max(np.abs(err))),
            "n": int(len(y)),
        }
    )
    return base


def _load_val(name: str) -> dict[str, float]:
    m = json.loads(VAL_METRICS_PATHS[name].read_text(encoding="utf-8"))
    return {
        "val_rmse": float(m["val"]["student"]["rmse"]),
        "val_mae": float(m["val"]["student"]["mae"]),
        "val_bias": float(m["val"]["student"]["bias"]),
        "val_r2": float(m["val"]["student"]["r2"]),
    }


def _load_final_step5() -> dict[str, dict[str, float]]:
    m = json.loads(STEP5_METRICS.read_text(encoding="utf-8"))
    out = {}
    for name in ["Large", "XLarge"]:
        t = m["models"][name]["test"]
        out[name] = {
            "rmse": float(t["rmse"]),
            "mae": float(t["mae"]),
            "bias": float(t["bias"]),
            "r2": float(t["r2"]),
            "cpu_latency_ms": float(m["models"][name]["cpu_latency_ms"]),
            "n_params": int(m["models"][name]["n_params"]),
        }
    return out


def _pred_table(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, split: str) -> pl.DataFrame:
    err = p - y
    phases = group_phase(df).astype(str)
    return pl.DataFrame(
        {
            "split": [split] * len(y),
            "flight_id": df["flight_id"].cast(pl.Utf8).to_list()
            if "flight_id" in df.columns
            else [str(i) for i in range(len(y))],
            "interval_idx": df["interval_idx"].to_list()
            if "interval_idx" in df.columns
            else list(range(len(y))),
            "start": df["start"].to_list() if "start" in df.columns else [None] * len(y),
            "aircraft_type": df["aircraft_type"].cast(pl.Utf8).fill_null("unknown").to_list()
            if "aircraft_type" in df.columns
            else ["unknown"] * len(y),
            "phase": phases.tolist(),
            "ground_truth": y,
            "predicted_fuel": p,
            "residual": err,
            "absolute_error": np.abs(err),
        }
    )


def _plots(
    rank_y: np.ndarray,
    rank_preds: dict[str, np.ndarray],
    combined_y: np.ndarray,
    combined_preds: dict[str, np.ndarray],
    final_metrics: dict[str, dict[str, float]],
    rank_metrics: dict[str, dict[str, float]],
    combined_metrics: dict[str, dict[str, float]],
    plots: Path,
) -> dict[str, str]:
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rels: dict[str, str] = {}
    colors = {"Large": "#1f77b4", "XLarge": "#d62728", "Teacher": "#2ca02c"}
    plt.rcParams.update(
        {"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150}
    )

    def save(fig, key: str) -> None:
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        dest = fig_dir / f"fig_comb_{key}.png"
        dest.write_bytes(p.read_bytes())
        plt.close(fig)
        rels[key] = f"figures/{dest.name}"

    # 1 Rank pred vs truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axes, ["Large", "XLarge"]):
        p = rank_preds[name]
        ax.scatter(rank_y, p, s=4, alpha=0.25, c=colors[name], rasterized=True)
        lim = [0, max(rank_y.max(), p.max()) * 1.02]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ground truth (kg)")
        ax.set_ylabel("Prediction (kg)")
        ax.set_title(f"Rank: {name} RMSE={rank_metrics[name]['rmse']:.1f}")
    save(fig, "rank_pred_vs_truth")

    # 2 Rank residual hist
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["Large", "XLarge"]:
        ax.hist(
            rank_preds[name] - rank_y,
            bins=80,
            alpha=0.5,
            label=name,
            color=colors[name],
            density=True,
        )
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Residual (pred − truth) kg")
    ax.set_ylabel("Density")
    ax.set_title("Rank residual histogram")
    ax.legend()
    save(fig, "rank_residual_hist")

    # 3 Combined error distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["Large", "XLarge"]:
        ax.hist(
            np.abs(combined_preds[name] - combined_y),
            bins=80,
            alpha=0.5,
            label=f"{name} |err|",
            color=colors[name],
            density=True,
        )
    ax.set_xlabel("Absolute error (kg)")
    ax.set_ylabel("Density")
    ax.set_title("Combined absolute-error distribution")
    ax.legend()
    save(fig, "combined_error_dist")

    # 4 Teacher vs student comparison bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ["R3 Teacher", "Large", "XLarge"]
    final_vals = [TEACHER_FINAL_RMSE, final_metrics["Large"]["rmse"], final_metrics["XLarge"]["rmse"]]
    comb_vals = [
        TEACHER_COMBINED_RMSE,
        combined_metrics["Large"]["rmse"],
        combined_metrics["XLarge"]["rmse"],
    ]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w / 2, final_vals, w, label="Final RMSE", color="#4c72b0")
    ax.bar(x + w / 2, comb_vals, w, label="Combined RMSE", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Teacher vs student — Final and Combined")
    ax.legend()
    for i, (fv, cv) in enumerate(zip(final_vals, comb_vals)):
        ax.text(i - w / 2, fv + 1.5, f"{fv:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, cv + 1.5, f"{cv:.1f}", ha="center", fontsize=8)
    save(fig, "teacher_vs_student")

    # 5 Final vs Combined performance
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name in ["Large", "XLarge"]:
        ax.scatter(
            final_metrics[name]["rmse"],
            combined_metrics[name]["rmse"],
            s=120,
            c=colors[name],
            label=name,
            zorder=3,
        )
        ax.annotate(
            name,
            (final_metrics[name]["rmse"], combined_metrics[name]["rmse"]),
            textcoords="offset points",
            xytext=(6, 6),
        )
    ax.scatter(
        TEACHER_FINAL_RMSE,
        TEACHER_COMBINED_RMSE,
        s=140,
        c=colors["Teacher"],
        marker="*",
        label="R3 Teacher",
        zorder=3,
    )
    ax.set_xlabel("Final RMSE (kg)")
    ax.set_ylabel("Combined RMSE (kg)")
    ax.set_title("Final vs Combined performance")
    ax.legend()
    save(fig, "final_vs_combined")

    # 6 Large vs XLarge multi-protocol
    fig, ax = plt.subplots(figsize=(8, 4.5))
    protocols = ["Val", "Rank", "Final", "Combined"]
    large_vals = [
        rank_metrics["Large"].get("val_rmse") or _load_val("Large")["val_rmse"],
        rank_metrics["Large"]["rmse"],
        final_metrics["Large"]["rmse"],
        combined_metrics["Large"]["rmse"],
    ]
    # fix val
    large_vals[0] = _load_val("Large")["val_rmse"]
    xlarge_vals = [
        _load_val("XLarge")["val_rmse"],
        rank_metrics["XLarge"]["rmse"],
        final_metrics["XLarge"]["rmse"],
        combined_metrics["XLarge"]["rmse"],
    ]
    x = np.arange(len(protocols))
    w = 0.35
    ax.bar(x - w / 2, large_vals, w, label="Large", color=colors["Large"])
    ax.bar(x + w / 2, xlarge_vals, w, label="XLarge", color=colors["XLarge"])
    ax.axhline(TEACHER_COMBINED_RMSE, color=colors["Teacher"], ls="--", lw=1.2, label="Teacher Combined")
    ax.axhline(TEACHER_FINAL_RMSE, color=colors["Teacher"], ls=":", lw=1.0, label="Teacher Final")
    ax.set_xticks(x)
    ax.set_xticklabels(protocols)
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Large vs XLarge across protocols")
    ax.legend(fontsize=9)
    save(fig, "large_vs_xlarge_protocols")

    return rels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rank-featured",
        type=Path,
        default=ROOT / "featured_dataset_rank.parquet",
    )
    ap.add_argument(
        "--final-featured",
        type=Path,
        default=ROOT / "featured_dataset_final.parquet",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--rebuild-final-preds",
        action="store_true",
        help="Re-infer Final instead of reusing Step 5 predictions",
    )
    args = ap.parse_args(argv)

    if not args.rank_featured.exists():
        raise FileNotFoundError(
            f"Missing {args.rank_featured}. Build with:\n"
            "  PYTHONPATH=src python -c \"from aerotwin.engine.official_benchmark import "
            "build_featured_for_split; from pathlib import Path; "
            "build_featured_for_split('rank', out_path=Path('featured_dataset_rank.parquet'))\""
        )
    if not args.final_featured.exists():
        raise FileNotFoundError(f"Missing {args.final_featured}")
    if not STEP5_METRICS.exists():
        raise FileNotFoundError(f"Missing Step 5 metrics: {STEP5_METRICS}")
    for name, spec in CHECKPOINTS.items():
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing checkpoint {name}: {spec['path']}")

    out = ROOT / "results" / "distillation" / "combined_evaluation"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    LOGGER.info("Device %s", device)
    t0 = time.time()

    final_step5 = _load_final_step5()
    data = _fit_preprocessors()

    # --- Rank ---
    rank_df = _prepare(args.rank_featured)
    x_rank, y_rank = _transform(rank_df, data)
    LOGGER.info(
        "Rank: %d rows, %d flights, X=%s",
        len(rank_df),
        rank_df["flight_id"].n_unique() if "flight_id" in rank_df.columns else -1,
        x_rank.shape,
    )

    # --- Final (reuse Step 5 preds by default) ---
    final_df = _prepare(args.final_featured)
    y_final = final_df["actual_fuel_kg"].to_numpy().astype(np.float64)
    if not args.rebuild_final_preds:
        for name in ["Large", "XLarge"]:
            if not STEP5_PRED[name].exists():
                raise FileNotFoundError(f"Missing Step 5 preds {STEP5_PRED[name]}")
        LOGGER.info("Reusing Step 5 Final predictions")
    else:
        x_final, y_final_chk = _transform(final_df, data)
        assert len(y_final_chk) == len(y_final)

    rank_preds: dict[str, np.ndarray] = {}
    final_preds: dict[str, np.ndarray] = {}
    rank_metrics: dict[str, dict[str, float]] = {}
    final_metrics: dict[str, dict[str, float]] = {}
    combined_metrics: dict[str, dict[str, float]] = {}
    combined_preds: dict[str, np.ndarray] = {}

    for name, spec in CHECKPOINTS.items():
        LOGGER.info("Rank inference %s", name)
        p_r = _predict(spec["path"], tuple(spec["hidden_dims"]), x_rank, device)
        rank_preds[name] = p_r
        rank_metrics[name] = _full(y_rank, p_r)

        if args.rebuild_final_preds:
            p_f = _predict(spec["path"], tuple(spec["hidden_dims"]), x_final, device)
        else:
            tbl = pl.read_parquet(STEP5_PRED[name])
            # Align by order — Step 5 wrote same order as featured final after clean/enrich
            p_f = tbl["predicted_fuel"].to_numpy().astype(np.float64)
            y_f_stored = tbl["ground_truth"].to_numpy().astype(np.float64)
            if len(p_f) != len(y_final) or abs(float(np.mean(y_f_stored)) - float(np.mean(y_final))) > 1e-3:
                LOGGER.warning("Step 5 Final pred length/mean mismatch — re-inferring Final for %s", name)
                x_f, y_f = _transform(final_df, data)
                p_f = _predict(spec["path"], tuple(spec["hidden_dims"]), x_f, device)
                y_final = y_f
            else:
                # prefer stored y for exact Step 5 parity
                y_final = y_f_stored
        final_preds[name] = p_f
        final_metrics[name] = _full(y_final, p_f)
        # Prefer Step 5 reported Final metrics for table consistency when reused
        if not args.rebuild_final_preds and abs(final_metrics[name]["rmse"] - final_step5[name]["rmse"]) < 0.05:
            final_metrics[name] = {
                **final_metrics[name],
                "rmse": final_step5[name]["rmse"],
                "mae": final_step5[name]["mae"],
                "bias": final_step5[name]["bias"],
                "r2": final_step5[name]["r2"],
            }

        # Combined = concat Rank then Final (official protocol)
        y_c = np.concatenate([y_rank, y_final])
        p_c = np.concatenate([p_r, p_f])
        combined_preds[name] = p_c
        combined_metrics[name] = _full(y_c, p_c)
        # Explicit official Combined RMSE via same kg_rmse helper as teacher scorecard
        combined_metrics[name]["rmse"] = kg_rmse(y_c, p_c)

        LOGGER.info(
            "%s Rank RMSE=%.2f Final=%.2f Combined=%.2f",
            name,
            rank_metrics[name]["rmse"],
            final_metrics[name]["rmse"],
            combined_metrics[name]["rmse"],
        )

        # Rank predictions
        _pred_table(rank_df, y_rank, p_r, "rank").write_parquet(
            out / f"predictions_rank_{name.lower()}.parquet"
        )
        # Combined predictions
        rank_tbl = _pred_table(rank_df, y_rank, p_r, "rank")
        final_tbl = _pred_table(final_df, y_final, p_f, "final")
        # If final_df length mismatch after y_final from step5, rebuild final_tbl from step5
        if len(final_tbl) != len(y_final):
            st = pl.read_parquet(STEP5_PRED[name])
            final_tbl = st.with_columns(pl.lit("final").alias("split")).select(
                [
                    "split",
                    "flight_id",
                    "interval_idx",
                    "start",
                    "aircraft_type",
                    "phase",
                    pl.col("ground_truth"),
                    pl.col("predicted_fuel"),
                    pl.col("residual"),
                    pl.col("absolute_error"),
                ]
            )
        pl.concat([rank_tbl, final_tbl], how="diagonal_relaxed").write_parquet(
            out / f"combined_predictions_{name.lower()}.parquet"
        )

    # Comparison table
    comparison = [
        {
            "model": "R3 Teacher (frozen)",
            "rank_rmse": TEACHER_RANK_RMSE,
            "final_rmse": TEACHER_FINAL_RMSE,
            "combined_rmse": TEACHER_COMBINED_RMSE,
            "mae": None,
            "r2": None,
            "parameters": "ensemble",
            "cpu_latency_ms": 52.0,
            "notes": "Rank/Combined from official R3 campaign; Final held-out audit 213.62",
        }
    ]
    for name in ["Large", "XLarge"]:
        comparison.append(
            {
                "model": CHECKPOINTS[name]["label"],
                "rank_rmse": rank_metrics[name]["rmse"],
                "final_rmse": final_metrics[name]["rmse"],
                "combined_rmse": combined_metrics[name]["rmse"],
                "mae": combined_metrics[name]["mae"],
                "r2": combined_metrics[name]["r2"],
                "parameters": CHECKPOINTS[name]["n_params"],
                "cpu_latency_ms": final_step5[name]["cpu_latency_ms"],
                "notes": "Combined MAE/R2 on concat Rank+Final",
            }
        )
    pl.DataFrame(comparison).write_csv(out / "comparison_table.csv")

    # Rankings
    best_final = "Large" if final_metrics["Large"]["rmse"] <= final_metrics["XLarge"]["rmse"] else "XLarge"
    best_combined = (
        "Large"
        if combined_metrics["Large"]["rmse"] <= combined_metrics["XLarge"]["rmse"]
        else "XLarge"
    )
    best_rank = (
        "Large" if rank_metrics["Large"]["rmse"] <= rank_metrics["XLarge"]["rmse"] else "XLarge"
    )

    val_map = {n: _load_val(n) for n in ["Large", "XLarge"]}

    plot_rels = _plots(
        y_rank,
        rank_preds,
        np.concatenate([y_rank, y_final]),
        combined_preds,
        final_metrics,
        rank_metrics,
        combined_metrics,
        plots,
    )

    metrics_blob = {
        "protocol": {
            "A_final": "Held-out Final only (Step 5)",
            "B_combined": "Official PRC Combined = RMSE(concat Rank, Final)",
            "alpha": FIXED_ALPHA,
            "beta": FIXED_BETA,
        },
        "datasets": {
            "rank": {
                "path": str(args.rank_featured.resolve()),
                "sha256": _sha256(args.rank_featured),
                "n_rows": len(rank_df),
                "n_flights": int(rank_df["flight_id"].n_unique())
                if "flight_id" in rank_df.columns
                else None,
            },
            "final": {
                "path": str(args.final_featured.resolve()),
                "sha256": _sha256(args.final_featured),
                "n_rows": int(len(y_final)),
                "n_flights": int(final_df["flight_id"].n_unique())
                if "flight_id" in final_df.columns
                else None,
            },
        },
        "teacher": {
            "rank_rmse": TEACHER_RANK_RMSE,
            "final_rmse": TEACHER_FINAL_RMSE,
            "combined_rmse": TEACHER_COMBINED_RMSE,
        },
        "models": {
            name: {
                "checkpoint": str(CHECKPOINTS[name]["path"]),
                "n_params": CHECKPOINTS[name]["n_params"],
                "validation": val_map[name],
                "rank": rank_metrics[name],
                "final": final_metrics[name],
                "combined": combined_metrics[name],
                "gap_to_teacher_final": final_metrics[name]["rmse"] - TEACHER_FINAL_RMSE,
                "gap_to_teacher_combined": combined_metrics[name]["rmse"] - TEACHER_COMBINED_RMSE,
                "cpu_latency_ms": final_step5[name]["cpu_latency_ms"],
            }
            for name in ["Large", "XLarge"]
        },
        "ranking": {
            "validation_best": "XLarge"
            if val_map["XLarge"]["val_rmse"] < val_map["Large"]["val_rmse"]
            else "Large",
            "rank_best": best_rank,
            "final_best": best_final,
            "combined_best": best_combined,
            "ranking_final_vs_combined_consistent": best_final == best_combined,
        },
        "comparison": comparison,
        "deployment_recommendation": best_combined
        if abs(combined_metrics["Large"]["rmse"] - combined_metrics["XLarge"]["rmse"]) >= 2.0
        else "Large",
        "wall_seconds": time.time() - t0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "plots": plot_rels,
    }
    # Prefer Large when combined delta small
    d_comb = combined_metrics["XLarge"]["rmse"] - combined_metrics["Large"]["rmse"]
    metrics_blob["deployment_recommendation"] = (
        "Large" if d_comb >= -2.0 else "XLarge"
    )  # Large unless XLarge better by >2 kg

    (out / "metrics.json").write_text(json.dumps(metrics_blob, indent=2, default=str), encoding="utf-8")
    (out / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "timestamp_utc": metrics_blob["timestamp_utc"],
                "git_commit": metrics_blob["git_commit"],
                "checkpoints": {
                    n: {
                        "path": str(CHECKPOINTS[n]["path"].resolve()),
                        "sha256": _sha256(CHECKPOINTS[n]["path"]),
                        "n_params": CHECKPOINTS[n]["n_params"],
                    }
                    for n in ["Large", "XLarge"]
                },
                "preprocessing": "train-fitted DistillationData scaler+OHE seed=42; transform-only",
                "final_preds_source": "step5_reuse" if not args.rebuild_final_preds else "re_inferred",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = ROOT / "docs" / "reports" / "combined_evaluation.md"
    report_path.write_text(
        _write_report(metrics_blob, plot_rels),
        encoding="utf-8",
    )
    (out / "combined_evaluation.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("\n=== OFFICIAL COMBINED EVALUATION ===")
    for name in ["Large", "XLarge"]:
        print(
            f"  {name}: Rank={rank_metrics[name]['rmse']:.2f} "
            f"Final={final_metrics[name]['rmse']:.2f} "
            f"Combined={combined_metrics[name]['rmse']:.2f} "
            f"(Δ teacher Combined {combined_metrics[name]['rmse'] - TEACHER_COMBINED_RMSE:+.2f})"
        )
    print(f"  best Rank={best_rank} Final={best_final} Combined={best_combined}")
    print(f"  deploy={metrics_blob['deployment_recommendation']}")
    print(f"  results={out}")


def _write_report(blob: dict[str, Any], plot_rels: dict[str, str]) -> str:
    L = blob["models"]["Large"]
    X = blob["models"]["XLarge"]
    lines = [
        "# Official Combined (Rank + Final) Student Evaluation",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "**Stage:** Evaluation only — frozen Large / XLarge MLPs, α=0.1, β=0.9",
        "",
        "No training. No checkpoint changes. Preprocessing = train-fitted transform-only.",
        "",
        "---",
        "",
        "## Two evaluation protocols",
        "",
        "### Protocol A — Final",
        "",
        "Controlled held-out evaluation on Oct 2025 Final intervals only.",
        "Used for architecture research and student comparisons under a fixed unseen-flight holdout.",
        "",
        "### Protocol B — Combined (Rank + Final)",
        "",
        "Official PRC-style aggregate:",
        "",
        "```text",
        "combined_rmse = RMSE( concat(y_rank, y_final), concat(p_rank, p_final) )",
        "```",
        "",
        "Identical aggregation to `full_scorecard` / R3 teacher Combined reporting.",
        "Used for direct comparison with the official PRC leaderboard and teacher Combined **221.33 kg**.",
        "",
        "**Both protocols are retained** — they answer different questions.",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "| Item | Value |",
        "|------|------:|",
        f"| Rank features | `{blob['datasets']['rank']['path']}` |",
        f"| Rank rows / flights | {blob['datasets']['rank']['n_rows']:,} / {blob['datasets']['rank']['n_flights']} |",
        f"| Final features | `{blob['datasets']['final']['path']}` |",
        f"| Final rows / flights | {blob['datasets']['final']['n_rows']:,} / {blob['datasets']['final']['n_flights']} |",
        "| Models | Large_seed42, XLarge_seed42 |",
        "| Final preds | Reused Step 5 when available |",
        "| Combined formula | concat Rank then Final, single RMSE |",
        "",
        "---",
        "",
        "## Rank evaluation",
        "",
        "| Model | RMSE | MAE | Bias | R² | n |",
        "|-------|-----:|----:|-----:|---:|--:|",
        f"| Large | {L['rank']['rmse']:.2f} | {L['rank']['mae']:.2f} | {L['rank']['bias']:+.2f} | {L['rank']['r2']:.4f} | {L['rank']['n']:,} |",
        f"| XLarge | {X['rank']['rmse']:.2f} | {X['rank']['mae']:.2f} | {X['rank']['bias']:+.2f} | {X['rank']['r2']:.4f} | {X['rank']['n']:,} |",
        f"| R3 Teacher (official campaign) | {blob['teacher']['rank_rmse']:.2f} | — | — | — | — |",
        "",
        "---",
        "",
        "## Final evaluation (Step 5 verified)",
        "",
        "| Model | RMSE | MAE | Bias | R² |",
        "|-------|-----:|----:|-----:|---:|",
        f"| Large | {L['final']['rmse']:.2f} | {L['final']['mae']:.2f} | {L['final']['bias']:+.2f} | {L['final']['r2']:.4f} |",
        f"| XLarge | {X['final']['rmse']:.2f} | {X['final']['mae']:.2f} | {X['final']['bias']:+.2f} | {X['final']['r2']:.4f} |",
        f"| R3 Teacher (held-out audit) | {blob['teacher']['final_rmse']:.2f} | — | — | — |",
        "",
        "---",
        "",
        "## Combined evaluation (Protocol B)",
        "",
        "| Model | Combined RMSE | Combined MAE | Combined Bias | Combined R² | n |",
        "|-------|-------------:|-------------:|--------------:|------------:|--:|",
        f"| **Large** | **{L['combined']['rmse']:.2f}** | {L['combined']['mae']:.2f} | {L['combined']['bias']:+.2f} | {L['combined']['r2']:.4f} | {L['combined']['n']:,} |",
        f"| XLarge | {X['combined']['rmse']:.2f} | {X['combined']['mae']:.2f} | {X['combined']['bias']:+.2f} | {X['combined']['r2']:.4f} | {X['combined']['n']:,} |",
        f"| R3 Teacher | **{blob['teacher']['combined_rmse']:.2f}** | — | — | — | — |",
        "",
        f"- Large gap to teacher Combined: **{L['gap_to_teacher_combined']:+.2f} kg**",
        f"- XLarge gap to teacher Combined: **{X['gap_to_teacher_combined']:+.2f} kg**",
        "",
        "---",
        "",
        "## Final comparison table",
        "",
        "| Model | Rank RMSE | Final RMSE | Combined RMSE | MAE | R² | Parameters | CPU Latency (ms) |",
        "|-------|----------:|-----------:|--------------:|----:|---:|-----------:|-----------------:|",
        f"| R3 Teacher | {blob['teacher']['rank_rmse']:.2f} | {blob['teacher']['final_rmse']:.2f} | **{blob['teacher']['combined_rmse']:.2f}** | — | — | ensemble | ~52 |",
        f"| Large MLP | {L['rank']['rmse']:.2f} | {L['final']['rmse']:.2f} | **{L['combined']['rmse']:.2f}** | {L['combined']['mae']:.2f} | {L['combined']['r2']:.4f} | {L['n_params']:,} | {L['cpu_latency_ms']:.2f} |",
        f"| XLarge MLP | {X['rank']['rmse']:.2f} | {X['final']['rmse']:.2f} | **{X['combined']['rmse']:.2f}** | {X['combined']['mae']:.2f} | {X['combined']['r2']:.4f} | {X['n_params']:,} | {X['cpu_latency_ms']:.2f} |",
        "",
        "Teacher MAE/R² Combined not re-derived in this student run; Rank teacher RMSE is the official R3 campaign component.",
        "",
        "---",
        "",
        "## Generalization analysis",
        "",
        "| Model | Val RMSE | Rank RMSE | Final RMSE | Combined RMSE |",
        "|-------|---------:|----------:|-----------:|--------------:|",
        f"| Large | {L['validation']['val_rmse']:.2f} | {L['rank']['rmse']:.2f} | {L['final']['rmse']:.2f} | {L['combined']['rmse']:.2f} |",
        f"| XLarge | {X['validation']['val_rmse']:.2f} | {X['rank']['rmse']:.2f} | {X['final']['rmse']:.2f} | {X['combined']['rmse']:.2f} |",
        "",
        f"- Best on validation: **{blob['ranking']['validation_best']}**",
        f"- Best on Rank: **{blob['ranking']['rank_best']}**",
        f"- Best on Final: **{blob['ranking']['final_best']}**",
        f"- Best on Combined: **{blob['ranking']['combined_best']}**",
        f"- Final vs Combined ranking consistent: **{blob['ranking']['ranking_final_vs_combined_consistent']}**",
        f"- Deployment recommendation: **{blob['deployment_recommendation']}**",
        "",
        "---",
        "",
        "## Teacher comparison",
        "",
        "Canonical teacher:",
        "",
        f"- Final: **{blob['teacher']['final_rmse']:.2f} kg**",
        f"- Combined: **{blob['teacher']['combined_rmse']:.2f} kg**",
        f"- Rank (official campaign): **{blob['teacher']['rank_rmse']:.2f} kg**",
        "",
        "---",
        "",
        "## Large vs XLarge",
        "",
        f"- Combined RMSE delta (XLarge − Large): **{X['combined']['rmse'] - L['combined']['rmse']:+.2f} kg**",
        f"- Final delta (XLarge − Large): **{X['final']['rmse'] - L['final']['rmse']:+.2f} kg**",
        f"- Rank delta (XLarge − Large): **{X['rank']['rmse'] - L['rank']['rmse']:+.2f} kg**",
        "",
        "---",
        "",
        "## Figures",
        "",
    ]
    for key, rel in plot_rels.items():
        lines += [f"### {key}", "", f"![{key}]({rel})", ""]

    lines += [
        "---",
        "",
        "## Official PRC comparison",
        "",
        f"Published winner Combined ≈ **201 kg**. Teacher Combined **{blob['teacher']['combined_rmse']:.2f}**. "
        f"Best student Combined **{min(L['combined']['rmse'], X['combined']['rmse']):.2f}** "
        f"({blob['ranking']['combined_best']}).",
        "",
        "---",
        "",
        "## Deployment recommendation",
        "",
        f"**{blob['deployment_recommendation']}** remains preferred unless XLarge improves Combined by >2 kg "
        f"(measured Combined delta XLarge−Large = {X['combined']['rmse'] - L['combined']['rmse']:+.2f} kg).",
        "",
        "---",
        "",
        "## Final conclusions (evidence only)",
        "",
        f"1. **Large Combined RMSE:** **{L['combined']['rmse']:.2f} kg**",
        f"2. **XLarge Combined RMSE:** **{X['combined']['rmse']:.2f} kg**",
        f"3. **Closest to teacher under PRC Combined:** **{blob['ranking']['combined_best']}** "
        f"(gap {min(L['gap_to_teacher_combined'], X['gap_to_teacher_combined']):+.2f} kg)",
        f"4. **Preferred deployment model:** **{blob['deployment_recommendation']}**",
        f"5. **Ranking change Final→Combined?** "
        f"{'No' if blob['ranking']['ranking_final_vs_combined_consistent'] else 'Yes'}",
        "6. **Future transformers should report both protocols:** Final (research holdout) **and** Combined (PRC parity).",
        "",
        "## Artifacts",
        "",
        "`results/distillation/combined_evaluation/` — metrics, rank/combined predictions, plots, metadata.",
        "",
        f"*Generated {blob['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
