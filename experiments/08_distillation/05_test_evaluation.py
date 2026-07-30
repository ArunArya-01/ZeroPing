"""Step 5 — Official held-out Final evaluation of frozen Large / XLarge MLPs.

STRICT evaluation only:
  * load Step-4 checkpoints (no retrain / fine-tune)
  * featured_dataset_final.parquet (already built from fuel_final)
  * train-fitted preprocessing only (no re-fit of model weights)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
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
from aerotwin.engine.gap_closing import clean_featured, ensure_features, group_phase
from aerotwin.engine.mass_model import enrich_mass_from_columns
from aerotwin.engine.official_benchmark import apply_bases

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("test_evaluation")

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


def _file_sha256(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    size = path.stat().st_size
    tag = "full" if size <= max_bytes else f"first_{max_bytes}_bytes"
    return f"{h.hexdigest()}:{tag}:size={size}"


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return None


def _pkg_versions() -> dict[str, str]:
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "polars", "torch", "sklearn", "matplotlib"):
        try:
            m = __import__(mod if mod != "sklearn" else "sklearn")
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = "missing"
    return vers


def _load_val_metrics(name: str) -> dict[str, float] | None:
    p = VAL_METRICS_PATHS[name]
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {
        "val_rmse": float(m["val"]["student"]["rmse"]),
        "val_mae": float(m["val"]["student"]["mae"]),
        "val_bias": float(m["val"]["student"]["bias"]),
        "val_r2": float(m["val"]["student"]["r2"]),
        "n_params": int(m["n_params"]),
    }


def _prepare_test(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "actual_fuel_kg" not in df.columns and "fuel_kg" in df.columns:
        df = df.with_columns(pl.col("fuel_kg").alias("actual_fuel_kg"))
    df = clean_featured(df)
    df = enrich_mass_from_columns(df)
    LOGGER.info(
        "Test: %d rows, %d flights",
        len(df),
        df["flight_id"].n_unique() if "flight_id" in df.columns else -1,
    )
    return df


def _fit_train_preprocessors(root: Path) -> DistillationData:
    """Identical train-fit path as capacity runs (seed=42, 20% flight val)."""
    return DistillationData.from_parquet(
        root / "distillation_dataset.parquet",
        root=root,
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
    ckpt: Path,
    hidden: tuple[int, ...],
    x: np.ndarray,
    device: torch.device,
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


def _mape(y: np.ndarray, p: np.ndarray, eps: float = 1.0) -> float:
    m = np.isfinite(y) & np.isfinite(p) & (np.abs(y) >= eps)
    if m.sum() < 10:
        return float("nan")
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100.0)


def _full_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    abs_e = np.abs(err)
    base = regression_metrics(y, p)
    base.update(
        {
            "mape_pct": _mape(y, p),
            "mean_residual": float(np.mean(err)),
            "median_residual": float(np.median(err)),
            "std_residual": float(np.std(err)),
            "p95_abs_error": float(np.percentile(abs_e, 95)),
            "p99_abs_error": float(np.percentile(abs_e, 99)),
            "max_abs_error": float(np.max(abs_e)),
            "mean_prediction": float(np.mean(p)),
            "mean_ground_truth": float(np.mean(y)),
            "var_prediction": float(np.var(p)),
            "var_ground_truth": float(np.var(y)),
            "pct_overpredict": float(np.mean(err > 0) * 100),
            "pct_underpredict": float(np.mean(err < 0) * 100),
        }
    )
    return base


def _duration_bin(h: float) -> str:
    if not np.isfinite(h):
        return "unknown"
    if h < 2:
        return "short_<2h"
    if h < 5:
        return "medium_2-5h"
    if h < 8:
        return "long_5-8h"
    return "ultralong_>=8h"


def _fuel_bin(kg: float) -> str:
    if not np.isfinite(kg):
        return "unknown"
    if kg < 200:
        return "fuel_<200"
    if kg < 500:
        return "fuel_200-500"
    if kg < 1000:
        return "fuel_500-1000"
    if kg < 2000:
        return "fuel_1000-2000"
    return "fuel_>=2000"


def _est_hours(df: pl.DataFrame) -> np.ndarray:
    if "start_fraction_of_flight" in df.columns and "end_fraction_of_flight" in df.columns:
        frac = np.clip(
            (
                df["end_fraction_of_flight"].to_numpy()
                - df["start_fraction_of_flight"].to_numpy()
            ).astype(np.float64),
            1e-3,
            None,
        )
        return (df["duration_s"].to_numpy().astype(np.float64) / frac) / 3600.0
    return df["duration_s"].to_numpy().astype(np.float64) / 3600.0


def _group_metrics(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, min_n: int = 20
) -> pl.DataFrame:
    rows = []
    for g in np.unique(groups.astype(str)):
        m = groups.astype(str) == str(g)
        if int(m.sum()) < min_n:
            continue
        met = regression_metrics(y[m], p[m])
        rows.append({"group": str(g), "n": int(m.sum()), **met})
    if not rows:
        return pl.DataFrame({"group": [], "n": [], "rmse": [], "mae": [], "bias": [], "r2": []})
    return pl.DataFrame(rows).sort("rmse")


def _teacher_predict(df: pl.DataFrame, root: Path) -> np.ndarray | None:
    cache = root / "cache" / "r3_teacher_distillation_bundle.pkl"
    if not cache.exists():
        return None
    with open(cache, "rb") as f:
        bundle = pickle.load(f)
    cols = list(bundle["feat_cols"])
    sub = ensure_features(df, cols)
    P = apply_bases(bundle["full_models"], sub, cols)
    ridge = np.asarray(bundle["meta"].predict(P), dtype=np.float64)
    return np.asarray(bundle["cal_phase"].transform(sub, ridge), dtype=np.float64)


def _bench_latency_cpu_ms(model: StudentMLP, x: np.ndarray, n: int = 100) -> float:
    model.eval().cpu()
    xt = torch.as_tensor(x[: min(len(x), 2048)], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10):
            _ = model(xt[:1])
        t0 = time.perf_counter()
        for i in range(n):
            _ = model(xt[i % len(xt) : (i % len(xt)) + 1])
        t1 = time.perf_counter()
    return 1000.0 * (t1 - t0) / n


def _plot_suite(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    val_map: dict[str, dict[str, float]],
    test_metrics: dict[str, dict[str, float]],
    failures: dict[str, list[dict[str, Any]]],
    plots: Path,
) -> dict[str, Path]:
    plots.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    colors = {"Large": "#1f77b4", "XLarge": "#d62728"}
    plt.rcParams.update(
        {"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 160}
    )

    def save(fig, key: str) -> None:
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths[key] = p

    # 1 pred vs truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axes, ["Large", "XLarge"]):
        p = preds[name]
        ax.scatter(y, p, s=3, alpha=0.12, c=colors[name], rasterized=True)
        lim = [0, float(max(np.percentile(y, 99.5), np.percentile(p, 99.5)))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Ground truth (kg)")
        ax.set_ylabel("Prediction (kg)")
        ax.set_title(f"{name} RMSE={test_metrics[name]['rmse']:.1f}")
    save(fig, "pred_vs_truth")

    # 2 residual hist
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ["Large", "XLarge"]:
        ax.hist(preds[name] - y, bins=80, density=True, alpha=0.45, label=name, color=colors[name])
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("Residual (pred−true) kg")
    ax.set_ylabel("Density")
    ax.set_title("Residual histogram (Final test)")
    ax.legend()
    save(fig, "residual_hist")

    # 3 residual distribution (box)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [np.clip(preds[n] - y, np.percentile(preds[n] - y, 1), np.percentile(preds[n] - y, 99)) for n in ["Large", "XLarge"]]
    ax.boxplot(data, labels=["Large", "XLarge"], showfliers=False)
    ax.axhline(0, color="k", ls="--")
    ax.set_ylabel("Residual (kg)")
    ax.set_title("Residual distribution (1–99%)")
    save(fig, "residual_distribution")

    # residual vs truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, ["Large", "XLarge"]):
        ax.scatter(y, preds[name] - y, s=2, alpha=0.1, c=colors[name], rasterized=True)
        ax.axhline(0, color="k", ls="--")
        ax.set_xlabel("Ground truth (kg)")
        ax.set_ylabel("Residual (kg)")
        ax.set_title(f"{name}: residual vs truth")
    save(fig, "residual_vs_truth")

    # QQ residuals
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ["Large", "XLarge"]):
        r = np.sort(preds[name] - y)
        # theoretical normal quantiles
        n = len(r)
        probs = (np.arange(1, n + 1) - 0.5) / n
        from scipy import stats

        theo = stats.norm.ppf(probs, loc=np.mean(r), scale=np.std(r) + 1e-12)
        # subsample for plot speed
        idx = np.linspace(0, n - 1, min(3000, n)).astype(int)
        ax.scatter(theo[idx], r[idx], s=4, alpha=0.3, c=colors[name])
        lim = [np.min(theo[idx]), np.max(theo[idx])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample residual quantiles")
        ax.set_title(f"{name} residual QQ")
    save(fig, "residual_qq")

    # abs error hist
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in ["Large", "XLarge"]:
        ax.hist(np.abs(preds[name] - y), bins=80, density=True, alpha=0.45, label=name, color=colors[name])
    ax.set_xlabel("|Error| (kg)")
    ax.set_ylabel("Density")
    ax.set_title("Absolute error distribution")
    ax.legend()
    save(fig, "abs_error_hist")

    # 4 aircraft
    ac = meta["aircraft_type"].astype(str)
    top = [u for u, c in sorted(zip(*np.unique(ac, return_counts=True)), key=lambda t: -t[1])[:15]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(top))
    for i, name in enumerate(["Large", "XLarge"]):
        vals = [
            float(np.sqrt(np.mean((preds[name][ac == t] - y[ac == t]) ** 2))) for t in top
        ]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=name, color=colors[name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top, rotation=45, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Test RMSE by aircraft type (top 15)")
    ax.legend()
    save(fig, "error_by_aircraft")

    # 5 duration
    order = ["short_<2h", "medium_2-5h", "long_5-8h", "ultralong_>=8h"]
    present = [g for g in order if g in set(meta["duration_bin"].astype(str))]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(present))
    for i, name in enumerate(["Large", "XLarge"]):
        vals = [
            float(
                np.sqrt(
                    np.mean(
                        (preds[name][meta["duration_bin"].astype(str) == g] - y[meta["duration_bin"].astype(str) == g])
                        ** 2
                    )
                )
            )
            for g in present
        ]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=name, color=colors[name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Test RMSE by flight duration")
    ax.legend()
    save(fig, "error_by_duration")

    # 6 fuel
    forder = ["fuel_<200", "fuel_200-500", "fuel_500-1000", "fuel_1000-2000", "fuel_>=2000"]
    present = [g for g in forder if g in set(meta["fuel_bin"].astype(str))]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(present))
    for i, name in enumerate(["Large", "XLarge"]):
        vals = [
            float(
                np.sqrt(
                    np.mean(
                        (preds[name][meta["fuel_bin"].astype(str) == g] - y[meta["fuel_bin"].astype(str) == g]) ** 2
                    )
                )
            )
            for g in present
        ]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=name, color=colors[name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=15, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Test RMSE by fuel consumption")
    ax.legend()
    save(fig, "error_by_fuel")

    # 7 calibration
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in ["Large", "XLarge"]:
        edges = np.unique(np.percentile(y, np.linspace(0, 100, 21)))
        mt, mp = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (y >= lo) & (y < hi) if hi > lo else (y >= lo) & (y <= hi)
            if m.sum() < 30:
                continue
            mt.append(float(np.mean(y[m])))
            mp.append(float(np.mean(preds[name][m])))
        ax.plot(mt, mp, "o-", label=name, color=colors[name], lw=2)
    lim = [0, float(np.percentile(y, 99))]
    ax.plot(lim, lim, "k--", label="ideal")
    ax.set_xlabel("Mean ground truth (kg)")
    ax.set_ylabel("Mean prediction (kg)")
    ax.set_title("Calibration curve")
    ax.legend()
    save(fig, "calibration")

    # 8 large vs xlarge abs error
    fig, ax = plt.subplots(figsize=(6.5, 6))
    eL = np.abs(preds["Large"] - y)
    eX = np.abs(preds["XLarge"] - y)
    ax.scatter(eL, eX, s=3, alpha=0.1, c="#555", rasterized=True)
    mmax = float(np.percentile(np.concatenate([eL, eX]), 99))
    ax.plot([0, mmax], [0, mmax], "k--")
    ax.set_xlim(0, mmax)
    ax.set_ylim(0, mmax)
    ax.set_xlabel("|Error| Large (kg)")
    ax.set_ylabel("|Error| XLarge (kg)")
    ax.set_title("Large vs XLarge absolute error")
    save(fig, "large_vs_xlarge")

    # 9 val vs test bars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = ["Large", "XLarge"]
    x = np.arange(len(names))
    val_r = [val_map[n]["val_rmse"] for n in names]
    te_r = [test_metrics[n]["rmse"] for n in names]
    ax.bar(x - 0.18, val_r, 0.35, label="Validation", color="#7f7f7f")
    ax.bar(x + 0.18, te_r, 0.35, label="Final test", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Validation vs held-out Final test")
    ax.legend()
    save(fig, "val_vs_test")

    # 10 generalization gap
    fig, ax = plt.subplots(figsize=(6, 4.2))
    gaps = [test_metrics[n]["rmse"] - val_map[n]["val_rmse"] for n in names]
    ax.bar(names, gaps, color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.axhline(0, color="k", ls="--")
    ax.set_ylabel("Test − Val RMSE (kg)")
    ax.set_title("Generalization gap")
    for i, g in enumerate(gaps):
        ax.text(i, g, f"{g:+.1f}", ha="center", va="bottom" if g >= 0 else "top")
    save(fig, "generalization_gap")

    # 11 top failure cases (XLarge abs error)
    fig, ax = plt.subplots(figsize=(9, 5))
    fails = failures["XLarge"][:15]
    labels = [f"{f['flight_id'][-6:]}|{f['aircraft_type']}" for f in fails]
    vals = [f["abs_error_kg"] for f in fails]
    ax.barh(range(len(vals))[::-1], vals[::-1], color="#d62728", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("|Error| (kg)")
    ax.set_title("Top failure cases (XLarge, abs error)")
    save(fig, "top_failures")

    LOGGER.info("Wrote %d figures", len(paths))
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--final-featured",
        type=Path,
        default=ROOT / "featured_dataset_final.parquet",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    if not args.final_featured.exists():
        raise FileNotFoundError(
            f"Missing {args.final_featured}. Build from fuel_final first; do not retrain models."
        )
    for name, spec in CHECKPOINTS.items():
        if not spec["path"].exists():
            raise FileNotFoundError(f"Missing frozen checkpoint for {name}: {spec['path']}")

    out = ROOT / "results" / "distillation" / "test_evaluation"
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

    test_df = _prepare_test(args.final_featured)
    data = _fit_train_preprocessors(ROOT)
    x_test, y = _transform(test_df, data)
    LOGGER.info("X_test %s in_dim match train %s", x_test.shape, data.in_dim)

    hours = _est_hours(test_df)
    phases = group_phase(test_df).astype(str)
    meta = {
        "aircraft_type": test_df["aircraft_type"].cast(pl.Utf8).fill_null("unknown").to_numpy()
        if "aircraft_type" in test_df.columns
        else np.array(["unknown"] * len(y)),
        "phase": phases,
        "duration_bin": np.array([_duration_bin(h) for h in hours]),
        "fuel_bin": np.array([_fuel_bin(v) for v in y]),
        "flight_hours": hours,
    }
    fids = (
        test_df["flight_id"].cast(pl.Utf8).to_numpy()
        if "flight_id" in test_df.columns
        else np.array([str(i) for i in range(len(y))])
    )
    interval_idx = (
        test_df["interval_idx"].to_numpy()
        if "interval_idx" in test_df.columns
        else np.arange(len(y))
    )
    starts = test_df["start"].to_list() if "start" in test_df.columns else [None] * len(y)

    preds: dict[str, np.ndarray] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    val_map: dict[str, dict[str, float]] = {}
    latencies: dict[str, float] = {}
    sizes: dict[str, float] = {}

    for name, spec in CHECKPOINTS.items():
        LOGGER.info("Inference %s <- %s", name, spec["path"])
        p = _predict(spec["path"], tuple(spec["hidden_dims"]), x_test, device)
        preds[name] = p
        test_metrics[name] = _full_metrics(y, p)
        vm = _load_val_metrics(name)
        if vm is None:
            raise RuntimeError(f"Missing validation metrics for {name}")
        val_map[name] = vm
        sizes[name] = spec["path"].stat().st_size / (1024 * 1024)
        # quick CPU single-sample latency (reload cpu)
        m = StudentMLP(int(x_test.shape[1]), hidden_dims=tuple(spec["hidden_dims"]), dropout=0.1)
        blob = torch.load(spec["path"], map_location="cpu", weights_only=False)
        m.load_state_dict(blob["model_state_dict"])
        latencies[name] = _bench_latency_cpu_ms(m, x_test, n=80)
        LOGGER.info(
            "%s test RMSE=%.2f MAE=%.2f bias=%+.2f R2=%.4f (val RMSE=%.2f)",
            name,
            test_metrics[name]["rmse"],
            test_metrics[name]["mae"],
            test_metrics[name]["bias"],
            test_metrics[name]["r2"],
            val_map[name]["val_rmse"],
        )

        # predictions parquet
        err = p - y
        pred_tbl = pl.DataFrame(
            {
                "flight_id": fids.tolist(),
                "interval_idx": interval_idx,
                "start": starts,
                "aircraft_type": meta["aircraft_type"].tolist(),
                "phase": meta["phase"].tolist(),
                "duration_bin": meta["duration_bin"].tolist(),
                "fuel_bin": meta["fuel_bin"].tolist(),
                "flight_hours": hours,
                "ground_truth": y,
                "predicted_fuel": p,
                "residual": err,
                "absolute_error": np.abs(err),
            }
        )
        pred_tbl.write_parquet(out / f"predictions_{name.lower()}.parquet")
        pred_tbl.select(
            ["flight_id", "interval_idx", "ground_truth", "predicted_fuel", "residual", "absolute_error"]
        ).write_parquet(out / f"residuals_{name.lower()}.parquet")

        # breakdown CSVs (per model prefix later merge for Large primary + both)
        g_ac = _group_metrics(y, p, meta["aircraft_type"])
        g_ph = _group_metrics(y, p, meta["phase"], min_n=10)
        g_du = _group_metrics(y, p, meta["duration_bin"], min_n=10)
        g_ac.write_csv(out / f"metrics_by_aircraft_{name.lower()}.csv")
        g_ph.write_csv(out / f"metrics_by_phase_{name.lower()}.csv")
        g_du.write_csv(out / f"metrics_by_duration_{name.lower()}.csv")

    # Combined breakdown (XLarge as primary official tables + both)
    for name in ["Large", "XLarge"]:
        pass
    # Official required names use XLarge for main tables and also write dual-model comparison
    ac_rows = []
    for g in np.unique(meta["aircraft_type"].astype(str)):
        m = meta["aircraft_type"].astype(str) == g
        if m.sum() < 20:
            continue
        row = {"aircraft_type": g, "n": int(m.sum())}
        for name in ["Large", "XLarge"]:
            met = regression_metrics(y[m], preds[name][m])
            row[f"{name.lower()}_rmse"] = met["rmse"]
            row[f"{name.lower()}_mae"] = met["mae"]
            row[f"{name.lower()}_bias"] = met["bias"]
        ac_rows.append(row)
    pl.DataFrame(ac_rows).sort("xlarge_rmse").write_csv(out / "metrics_by_aircraft.csv")

    ph_rows = []
    for g in np.unique(meta["phase"].astype(str)):
        m = meta["phase"].astype(str) == g
        if m.sum() < 10:
            continue
        row = {"phase": g, "n": int(m.sum())}
        for name in ["Large", "XLarge"]:
            met = regression_metrics(y[m], preds[name][m])
            row[f"{name.lower()}_rmse"] = met["rmse"]
        ph_rows.append(row)
    pl.DataFrame(ph_rows).sort("xlarge_rmse").write_csv(out / "metrics_by_phase.csv")

    du_rows = []
    for g in ["short_<2h", "medium_2-5h", "long_5-8h", "ultralong_>=8h"]:
        m = meta["duration_bin"].astype(str) == g
        if m.sum() < 10:
            continue
        row = {"duration_bin": g, "n": int(m.sum())}
        for name in ["Large", "XLarge"]:
            met = regression_metrics(y[m], preds[name][m])
            row[f"{name.lower()}_rmse"] = met["rmse"]
        du_rows.append(row)
    pl.DataFrame(du_rows).write_csv(out / "metrics_by_duration.csv")

    # Large vs XLarge pairwise analysis
    d_abs = np.abs(preds["XLarge"] - y) - np.abs(preds["Large"] - y)
    improve = d_abs < -1.0  # XLarge better by >1kg abs
    regress = d_abs > 1.0
    both_bad = (np.abs(preds["Large"] - y) > 500) & (np.abs(preds["XLarge"] - y) > 500)
    pairwise = {
        "mean_abs_err_delta_xlarge_minus_large": float(np.mean(d_abs)),
        "median_abs_err_delta": float(np.median(d_abs)),
        "frac_xlarge_better_gt1kg": float(np.mean(improve)),
        "frac_large_better_gt1kg": float(np.mean(regress)),
        "frac_both_abs_err_gt500": float(np.mean(both_bad)),
        "max_xlarge_improvement": float(-np.min(d_abs)),
        "max_xlarge_regression": float(np.max(d_abs)),
        "test_rmse_delta_xlarge_minus_large": float(
            test_metrics["XLarge"]["rmse"] - test_metrics["Large"]["rmse"]
        ),
    }

    # Failure / success cases
    failures: dict[str, list[dict[str, Any]]] = {}
    case_summaries: dict[str, Any] = {}
    for name in ["Large", "XLarge"]:
        err = preds[name] - y
        abs_e = np.abs(err)
        order_worst = np.argsort(-abs_e)
        order_best = np.argsort(abs_e)
        def pack(idxs):
            rows = []
            for i in idxs:
                rows.append(
                    {
                        "flight_id": str(fids[i]),
                        "interval_idx": int(interval_idx[i]),
                        "aircraft_type": str(meta["aircraft_type"][i]),
                        "phase": str(meta["phase"][i]),
                        "duration_bin": str(meta["duration_bin"][i]),
                        "true_kg": float(y[i]),
                        "pred_kg": float(preds[name][i]),
                        "error_kg": float(err[i]),
                        "abs_error_kg": float(abs_e[i]),
                    }
                )
            return rows

        failures[name] = pack(order_worst[:100])
        case_summaries[name] = {
            "top100_worst": failures[name],
            "top100_best": pack(order_best[:100]),
            "largest_overpredictions": pack(np.argsort(-err)[:50]),
            "largest_underpredictions": pack(np.argsort(err)[:50]),
        }

    # Aircraft difficulty ranking (XLarge)
    ac_tbl = _group_metrics(y, preds["XLarge"], meta["aircraft_type"])
    easiest = ac_tbl.head(5).to_dicts() if len(ac_tbl) else []
    hardest = ac_tbl.sort("rmse", descending=True).head(5).to_dicts() if len(ac_tbl) else []

    # Baselines
    comparison = []
    for name in ["Large", "XLarge"]:
        comparison.append(
            {
                "model": CHECKPOINTS[name]["label"],
                "parameters": CHECKPOINTS[name]["n_params"],
                "rmse": test_metrics[name]["rmse"],
                "mae": test_metrics[name]["mae"],
                "bias": test_metrics[name]["bias"],
                "r2": test_metrics[name]["r2"],
                "cpu_latency_ms_per_sample": latencies[name],
                "model_size_mb": sizes[name],
                "status": "evaluated",
            }
        )

    y_teacher = _teacher_predict(test_df, ROOT)
    if y_teacher is not None:
        m = np.isfinite(y_teacher)
        tm = _full_metrics(y[m], y_teacher[m])
        t_size = (ROOT / "cache" / "r3_teacher_distillation_bundle.pkl").stat().st_size / (1024 * 1024)
        comparison.append(
            {
                "model": "R3 Teacher (frozen ensemble)",
                "parameters": "ensemble",
                "rmse": tm["rmse"],
                "mae": tm["mae"],
                "bias": tm["bias"],
                "r2": tm["r2"],
                "cpu_latency_ms_per_sample": "see Step4 bench (~52ms single)",
                "model_size_mb": t_size,
                "status": "evaluated",
            }
        )
    else:
        comparison.append(
            {
                "model": "R3 Teacher (frozen ensemble)",
                "parameters": "ensemble",
                "rmse": None,
                "mae": None,
                "bias": None,
                "r2": None,
                "cpu_latency_ms_per_sample": None,
                "model_size_mb": None,
                "status": "unavailable (cache missing)",
            }
        )

    if "physics_fuel_kg" in test_df.columns:
        op = test_df["physics_fuel_kg"].cast(pl.Float64, strict=False).to_numpy().astype(np.float64)
        m = np.isfinite(op)
        om = _full_metrics(y[m], op[m])
        comparison.append(
            {
                "model": "OpenAP baseline",
                "parameters": "—",
                "rmse": om["rmse"],
                "mae": om["mae"],
                "bias": om["bias"],
                "r2": om["r2"],
                "cpu_latency_ms_per_sample": "n/a",
                "model_size_mb": "n/a",
                "status": "evaluated",
            }
        )

    comparison.append(
        {
            "model": "Best LightGBM (standalone)",
            "parameters": "n/a",
            "rmse": None,
            "mae": None,
            "bias": None,
            "r2": None,
            "cpu_latency_ms_per_sample": None,
            "model_size_mb": None,
            "status": "unavailable (no frozen single-LGBM Final checkpoint in distillation path)",
        }
    )

    pl.DataFrame(comparison).write_csv(out / "comparison_table.csv")

    # Plots
    plot_paths = _plot_suite(y, preds, meta, val_map, test_metrics, failures, plots)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rels = {}
    for k, pth in plot_paths.items():
        dest = fig_dir / f"fig_test_{pth.name}"
        dest.write_bytes(pth.read_bytes())
        plot_rels[k] = f"figures/{dest.name}"

    # Generalization
    gen = {}
    for name in ["Large", "XLarge"]:
        vr = val_map[name]["val_rmse"]
        tr = test_metrics[name]["rmse"]
        gen[name] = {
            "val_rmse": vr,
            "test_rmse": tr,
            "gap_test_minus_val": tr - vr,
            "pct_change": 100.0 * (tr - vr) / max(vr, 1e-9),
        }

    ranking_val = "XLarge" if val_map["XLarge"]["val_rmse"] < val_map["Large"]["val_rmse"] else "Large"
    ranking_test = "XLarge" if test_metrics["XLarge"]["rmse"] < test_metrics["Large"]["rmse"] else "Large"
    delta_test = test_metrics["XLarge"]["rmse"] - test_metrics["Large"]["rmse"]
    deploy = "Large" if abs(delta_test) < 2.0 or delta_test > 0 else "XLarge"

    metrics_blob = {
        "n_test": len(y),
        "n_flights": int(test_df["flight_id"].n_unique()) if "flight_id" in test_df.columns else None,
        "alpha": FIXED_ALPHA,
        "beta": FIXED_BETA,
        "models": {
            name: {
                "checkpoint": str(CHECKPOINTS[name]["path"]),
                "n_params": CHECKPOINTS[name]["n_params"],
                "test": test_metrics[name],
                "validation": val_map[name],
                "generalization": gen[name],
                "cpu_latency_ms": latencies[name],
                "model_size_mb": sizes[name],
            }
            for name in ["Large", "XLarge"]
        },
        "comparison": comparison,
        "pairwise_large_vs_xlarge": pairwise,
        "easiest_aircraft_xlarge": easiest,
        "hardest_aircraft_xlarge": hardest,
        "ranking_validation": ranking_val,
        "ranking_test": ranking_test,
        "ranking_consistent": ranking_val == ranking_test,
        "deployment_recommendation": deploy,
        "case_summaries": {
            n: {
                "top5_worst": case_summaries[n]["top100_worst"][:5],
                "top5_best": case_summaries[n]["top100_best"][:5],
            }
            for n in ["Large", "XLarge"]
        },
    }
    # full case lists separate to keep metrics.json readable
    (out / "case_analysis.json").write_text(
        json.dumps(case_summaries, indent=2, default=str), encoding="utf-8"
    )
    (out / "metrics.json").write_text(json.dumps(metrics_blob, indent=2, default=str), encoding="utf-8")

    meta = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(ROOT),
        "final_featured_path": str(args.final_featured.resolve()),
        "final_featured_sha256": _file_sha256(args.final_featured),
        "fuel_final_path": str((ROOT / "fuel_final.parquet").resolve())
        if (ROOT / "fuel_final.parquet").exists()
        else None,
        "checkpoints": {
            n: {
                "path": str(CHECKPOINTS[n]["path"].resolve()),
                "sha256": _file_sha256(CHECKPOINTS[n]["path"]),
                "hidden_dims": list(CHECKPOINTS[n]["hidden_dims"]),
                "n_params": CHECKPOINTS[n]["n_params"],
            }
            for n in ["Large", "XLarge"]
        },
        "distillation_dataset": str((ROOT / "distillation_dataset.parquet").resolve()),
        "distillation_dataset_sha256": _file_sha256(ROOT / "distillation_dataset.parquet")
        if (ROOT / "distillation_dataset.parquet").exists()
        else None,
        "preprocessing": "train-fitted StandardScaler+OHE from DistillationData(seed=42,val=0.2); transform-only on Final",
        "package_versions": _pkg_versions(),
        "wall_seconds": time.time() - t0,
    }
    (out / "evaluation_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    # Report
    _write_report(
        ROOT / "docs" / "reports" / "test_evaluation.md",
        metrics_blob=metrics_blob,
        pairwise=pairwise,
        gen=gen,
        plot_rels=plot_rels,
        n_test=len(y),
        n_flights=int(test_df["flight_id"].n_unique()) if "flight_id" in test_df.columns else 0,
        easiest=easiest,
        hardest=hardest,
        deploy=deploy,
        ranking_val=ranking_val,
        ranking_test=ranking_test,
    )

    print("\n=== OFFICIAL FINAL TEST EVALUATION ===")
    for name in ["Large", "XLarge"]:
        print(
            f"  {name}: test_rmse={test_metrics[name]['rmse']:.2f} "
            f"val_rmse={val_map[name]['val_rmse']:.2f} "
            f"gap={gen[name]['gap_test_minus_val']:+.2f} "
            f"({gen[name]['pct_change']:+.1f}%)"
        )
    print(f"  ranking val={ranking_val} test={ranking_test} consistent={ranking_val==ranking_test}")
    print(f"  deploy_recommendation={deploy}")
    print(f"  results={out}")


def _write_report(
    path: Path,
    *,
    metrics_blob: dict[str, Any],
    pairwise: dict[str, Any],
    gen: dict[str, Any],
    plot_rels: dict[str, str],
    n_test: int,
    n_flights: int,
    easiest: list[dict],
    hardest: list[dict],
    deploy: str,
    ranking_val: str,
    ranking_test: str,
) -> None:
    L = metrics_blob["models"]["Large"]
    X = metrics_blob["models"]["XLarge"]
    lines = [
        "# Official Held-Out Final Test Evaluation — Distilled MLP Baseline",
        "",
        "**Stage:** AeroTwin Distillation Step 5 (evaluation only)",
        "",
        "Frozen Step-4 checkpoints. No training, no hyperparameter changes, no preprocessing refits of model weights.",
        f"Training KD weights were α={FIXED_ALPHA}, β={FIXED_BETA}.",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "| Item | Value |",
        "|------|------:|",
        f"| Test features | `featured_dataset_final.parquet` (from `fuel_final`) |",
        f"| Test rows / flights | {n_test:,} / {n_flights:,} |",
        f"| Models | Large_seed42, XLarge_seed42 |",
        f"| Preprocessing | Train-fitted scaler/OHE (distillation train, seed 42); transform-only |",
        f"| Teacher comparison | Frozen R3 ensemble cache if present |",
        "",
        "## Models evaluated",
        "",
        "| Model | Params | Checkpoint |",
        "|-------|-------:|------------|",
        f"| Large | {L['n_params']:,} | `{L['checkpoint']}` |",
        f"| XLarge | {X['n_params']:,} | `{X['checkpoint']}` |",
        "",
        "---",
        "",
        "## Overall metrics (Final test)",
        "",
        "| Model | Params | RMSE | MAE | Bias | R² | MAPE % | P95 |abs| |",
        "|-------|-------:|-----:|----:|-----:|---:|-------:|----------:|",
        f"| Large | {L['n_params']:,} | {L['test']['rmse']:.2f} | {L['test']['mae']:.2f} "
        f"| {L['test']['bias']:+.2f} | {L['test']['r2']:.4f} | {L['test']['mape_pct']:.2f} "
        f"| {L['test']['p95_abs_error']:.2f} |",
        f"| XLarge | {X['n_params']:,} | {X['test']['rmse']:.2f} | {X['test']['mae']:.2f} "
        f"| {X['test']['bias']:+.2f} | {X['test']['r2']:.4f} | {X['test']['mape_pct']:.2f} "
        f"| {X['test']['p95_abs_error']:.2f} |",
        "",
        "### Residual / prediction stats",
        "",
        "| Model | Mean res | Median res | Std res | Mean pred | Mean truth |",
        "|-------|---------:|-----------:|--------:|----------:|-----------:|",
        f"| Large | {L['test']['mean_residual']:+.2f} | {L['test']['median_residual']:+.2f} "
        f"| {L['test']['std_residual']:.2f} | {L['test']['mean_prediction']:.2f} "
        f"| {L['test']['mean_ground_truth']:.2f} |",
        f"| XLarge | {X['test']['mean_residual']:+.2f} | {X['test']['median_residual']:+.2f} "
        f"| {X['test']['std_residual']:.2f} | {X['test']['mean_prediction']:.2f} "
        f"| {X['test']['mean_ground_truth']:.2f} |",
        "",
        "---",
        "",
        "## Validation vs Test",
        "",
        "| Model | Val RMSE | Test RMSE | Gap | % change |",
        "|-------|---------:|----------:|----:|---------:|",
        f"| Large | {gen['Large']['val_rmse']:.2f} | {gen['Large']['test_rmse']:.2f} "
        f"| {gen['Large']['gap_test_minus_val']:+.2f} | {gen['Large']['pct_change']:+.1f}% |",
        f"| XLarge | {gen['XLarge']['val_rmse']:.2f} | {gen['XLarge']['test_rmse']:.2f} "
        f"| {gen['XLarge']['gap_test_minus_val']:+.2f} | {gen['XLarge']['pct_change']:+.1f}% |",
        "",
        f"- Validation ranking: **{ranking_val}** better",
        f"- Test ranking: **{ranking_test}** better",
        f"- Ranking consistent: **{ranking_val == ranking_test}**",
        f"- Test RMSE (XLarge − Large): **{X['test']['rmse'] - L['test']['rmse']:+.2f} kg**",
        "",
        "---",
        "",
        "## Large vs XLarge",
        "",
        f"- Mean |err| delta (XLarge − Large): **{pairwise['mean_abs_err_delta_xlarge_minus_large']:+.2f} kg**",
        f"- Fraction XLarge better by >1 kg abs: **{100*pairwise['frac_xlarge_better_gt1kg']:.1f}%**",
        f"- Fraction Large better by >1 kg abs: **{100*pairwise['frac_large_better_gt1kg']:.1f}%**",
        f"- Max XLarge improvement / regression: **{pairwise['max_xlarge_improvement']:.1f}** / **{pairwise['max_xlarge_regression']:.1f}** kg",
        f"- Both fail (|err|>500 kg): **{100*pairwise['frac_both_abs_err_gt500']:.2f}%** of intervals",
        "",
        f"**Deployment recommendation (accuracy–cost): `{deploy}`**",
        "",
        "---",
        "",
        "## Baseline comparison",
        "",
        "| Model | Parameters | RMSE | MAE | Bias | R² | CPU latency | Size MB | Status |",
        "|-------|------------|-----:|----:|-----:|---:|-------------|---------|--------|",
    ]
    for row in metrics_blob["comparison"]:
        rmse = row["rmse"]
        rmse_s = f"{float(rmse):.2f}" if isinstance(rmse, (int, float)) else "—"
        mae_s = f"{float(row['mae']):.2f}" if isinstance(row.get("mae"), (int, float)) else "—"
        bias_s = f"{float(row['bias']):+.2f}" if isinstance(row.get("bias"), (int, float)) else "—"
        r2_s = f"{float(row['r2']):.4f}" if isinstance(row.get("r2"), (int, float)) else "—"
        lines.append(
            f"| {row['model']} | {row['parameters']} | {rmse_s} | {mae_s} | {bias_s} | {r2_s} "
            f"| {row.get('cpu_latency_ms_per_sample')} | {row.get('model_size_mb')} | {row.get('status')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Error analysis",
        "",
        "### Easiest aircraft (XLarge, lowest RMSE)",
        "",
    ]
    for r in easiest:
        lines.append(f"- `{r['group']}`: RMSE {r['rmse']:.2f} (n={r['n']})")
    lines += ["", "### Hardest aircraft (XLarge, highest RMSE)", ""]
    for r in hardest:
        lines.append(f"- `{r['group']}`: RMSE {r['rmse']:.2f} (n={r['n']})")

    lines += [
        "",
        "See `case_analysis.json` for top-100 best/worst and over/under-prediction lists.",
        "",
        "---",
        "",
        "## Figures",
        "",
        f"![pred vs truth]({plot_rels.get('pred_vs_truth', '')})",
        "",
        f"![residual hist]({plot_rels.get('residual_hist', '')})",
        "",
        f"![residual dist]({plot_rels.get('residual_distribution', '')})",
        "",
        f"![by aircraft]({plot_rels.get('error_by_aircraft', '')})",
        "",
        f"![by duration]({plot_rels.get('error_by_duration', '')})",
        "",
        f"![by fuel]({plot_rels.get('error_by_fuel', '')})",
        "",
        f"![calibration]({plot_rels.get('calibration', '')})",
        "",
        f"![L vs XL]({plot_rels.get('large_vs_xlarge', '')})",
        "",
        f"![val vs test]({plot_rels.get('val_vs_test', '')})",
        "",
        f"![gap]({plot_rels.get('generalization_gap', '')})",
        "",
        f"![failures]({plot_rels.get('top_failures', '')})",
        "",
        "---",
        "",
        "## Final conclusions (evidence-only)",
        "",
        f"1. **Official MLP baseline / deployment pick:** **{deploy}** "
        f"(test RMSE Large={L['test']['rmse']:.2f}, XLarge={X['test']['rmse']:.2f}).",
        f"2. **Does XLarge justify +3.9M params?** Test delta {X['test']['rmse']-L['test']['rmse']:+.2f} kg; "
        + (
            "gap is small — **cost does not clearly justify capacity** on Final."
            if abs(X["test"]["rmse"] - L["test"]["rmse"]) < 2
            else "see magnitude of RMSE delta vs Step-4 latency/size cost."
        ),
        f"3. **Generalization gap:** Large {gen['Large']['gap_test_minus_val']:+.2f} kg "
        f"({gen['Large']['pct_change']:+.1f}%); XLarge {gen['XLarge']['gap_test_minus_val']:+.2f} kg "
        f"({gen['XLarge']['pct_change']:+.1f}%).",
        "4. **Failure modes:** see hardest aircraft + ultralong/high-fuel bins in breakdown CSVs and top failures figure.",
        "5. **Challenging aircraft:** listed under hardest aircraft above.",
        "6. **Baseline for transformers?** Yes — these frozen Final metrics are the permanent MLP reference.",
        "7. **Next directions:** architecture change (FT/Tab transformer) under α=0.1/β=0.9; "
        "hard-subgroup focus (heavies / ultra-long) rather than more MLP width.",
        "",
        "## Artifacts",
        "",
        "`results/distillation/test_evaluation/` — metrics, predictions, residuals, breakdowns, plots, metadata.",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
