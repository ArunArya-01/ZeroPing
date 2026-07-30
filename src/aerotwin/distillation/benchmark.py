"""Inference latency / throughput / memory benchmarks for distillation models."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _try_rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _gpu_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)


def checkpoint_size_mb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return float(path.stat().st_size) / (1024 * 1024)


@torch.no_grad()
def benchmark_torch_model(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
    n_warmup: int = 20,
    n_iters: int = 50,
    single_samples: int = 200,
) -> dict[str, Any]:
    """Time a torch student on CPU or GPU.

    Parameters
    ----------
    x:
        Feature matrix (n, d), float32 preferred.
    batch_size:
        Batch size for batched throughput measurement.
    n_warmup / n_iters:
        Warmup and timed iterations for batch inference.
    single_samples:
        Number of sequential single-sample forwards for single-sample latency.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if isinstance(x, np.ndarray):
        x_t = torch.as_tensor(x, dtype=torch.float32)
    else:
        x_t = x.detach().float()

    n = int(x_t.shape[0])
    if n < 1:
        raise ValueError("empty feature matrix")

    # ---- single-sample ----
    n_single = min(single_samples, n)
    idx = torch.arange(n_single)
    # warmup
    for i in range(min(10, n_single)):
        _ = model(x_t[i : i + 1].to(device))
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rss0 = _try_rss_mb()
    t0 = time.perf_counter()
    for i in range(n_single):
        _ = model(x_t[i : i + 1].to(device))
    _sync(device)
    t1 = time.perf_counter()
    single_total_s = t1 - t0
    single_ms = 1000.0 * single_total_s / n_single
    single_tps = n_single / single_total_s if single_total_s > 0 else float("nan")
    peak_gpu_single = _gpu_peak_mb(device)
    rss1 = _try_rss_mb()
    peak_ram_single = None if rss0 is None or rss1 is None else max(rss0, rss1)

    # ---- batch ----
    bs = min(batch_size, n)
    # use repeated full passes over the matrix
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_warmup):
        for start in range(0, n, bs):
            end = min(start + bs, n)
            _ = model(x_t[start:end].to(device))
    _sync(device)

    rss0 = _try_rss_mb()
    n_seen = 0
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for start in range(0, n, bs):
            end = min(start + bs, n)
            _ = model(x_t[start:end].to(device))
            n_seen += end - start
    _sync(device)
    t1 = time.perf_counter()
    batch_total_s = t1 - t0
    batch_ms = 1000.0 * batch_total_s / max(n_seen, 1)
    batch_tps = n_seen / batch_total_s if batch_total_s > 0 else float("nan")
    peak_gpu_batch = _gpu_peak_mb(device)
    rss1 = _try_rss_mb()
    peak_ram_batch = None if rss0 is None or rss1 is None else max(rss0, rss1)

    return {
        "device": str(device),
        "n_rows": n,
        "batch_size": bs,
        "single_latency_ms": single_ms,
        "single_throughput_sps": single_tps,
        "batch_latency_ms": batch_ms,
        "batch_throughput_sps": batch_tps,
        "peak_gpu_mb_single": peak_gpu_single,
        "peak_gpu_mb_batch": peak_gpu_batch,
        "peak_ram_mb_single": peak_ram_single,
        "peak_ram_mb_batch": peak_ram_batch,
        "n_single": n_single,
        "n_batch_samples_timed": n_seen,
        "n_warmup": n_warmup,
        "n_batch_iters": n_iters,
    }


def benchmark_callable(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    batch_size: int = 256,
    n_warmup: int = 5,
    n_iters: int = 10,
    single_samples: int = 100,
    label: str = "callable",
) -> dict[str, Any]:
    """Time a numpy in/out callable (e.g. frozen teacher ensemble)."""
    n = int(x.shape[0])
    n_single = min(single_samples, n)
    bs = min(batch_size, n)

    for i in range(min(5, n_single)):
        _ = predict_fn(x[i : i + 1])

    rss0 = _try_rss_mb()
    t0 = time.perf_counter()
    for i in range(n_single):
        _ = predict_fn(x[i : i + 1])
    t1 = time.perf_counter()
    single_total_s = t1 - t0
    single_ms = 1000.0 * single_total_s / n_single
    single_tps = n_single / single_total_s if single_total_s > 0 else float("nan")
    rss1 = _try_rss_mb()
    peak_ram_single = None if rss0 is None or rss1 is None else max(rss0, rss1)

    for _ in range(n_warmup):
        for start in range(0, n, bs):
            _ = predict_fn(x[start : min(start + bs, n)])

    rss0 = _try_rss_mb()
    n_seen = 0
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for start in range(0, n, bs):
            end = min(start + bs, n)
            _ = predict_fn(x[start:end])
            n_seen += end - start
    t1 = time.perf_counter()
    batch_total_s = t1 - t0
    batch_ms = 1000.0 * batch_total_s / max(n_seen, 1)
    batch_tps = n_seen / batch_total_s if batch_total_s > 0 else float("nan")
    rss1 = _try_rss_mb()
    peak_ram_batch = None if rss0 is None or rss1 is None else max(rss0, rss1)

    return {
        "device": "cpu",
        "label": label,
        "n_rows": n,
        "batch_size": bs,
        "single_latency_ms": single_ms,
        "single_throughput_sps": single_tps,
        "batch_latency_ms": batch_ms,
        "batch_throughput_sps": batch_tps,
        "peak_gpu_mb_single": None,
        "peak_gpu_mb_batch": None,
        "peak_ram_mb_single": peak_ram_single,
        "peak_ram_mb_batch": peak_ram_batch,
        "n_single": n_single,
        "n_batch_samples_timed": n_seen,
        "n_warmup": n_warmup,
        "n_batch_iters": n_iters,
    }


def load_teacher_predict_fn(
    cache_path: Path,
    val_df,  # polars DataFrame with feat cols
    feat_cols: list[str] | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Build a predict function for the frozen R3 teacher on *rows of val_df*.

    The callable accepts a 2D index array (or boolean mask rows) via integer
    row indices into ``val_df``. For simplicity we expose predict on feature
    row indices: ``predict_fn(row_index_array)`` where indices select rows of
    ``val_df``.

    Actually for fair comparison with student numpy X matrix, we map row
    positions 0..n-1 of val_df to teacher predictions.
    """
    from aerotwin.engine.gap_closing import ensure_features
    from aerotwin.engine.official_benchmark import apply_bases

    with open(cache_path, "rb") as f:
        bundle = pickle.load(f)

    cols = list(feat_cols or bundle["feat_cols"])
    df = ensure_features(val_df, cols)
    # Precompute full-val teacher preds once; timing uses sliced recompute for fairness
    # For latency we recompute on subsets to match real inference.

    full_models = bundle["full_models"]
    meta = bundle["meta"]
    cal = bundle["cal_phase"]

    def predict_rows(row_idx: np.ndarray) -> np.ndarray:
        # row_idx: 1d integer indices into df
        idx = np.asarray(row_idx, dtype=np.int64).reshape(-1)
        sub = df[idx.tolist()]
        P = apply_bases(full_models, sub, cols)
        ridge = np.asarray(meta.predict(P), dtype=np.float64)
        return np.asarray(cal.transform(sub, ridge), dtype=np.float64)

    # Also provide a version that takes dummy X but uses positional indices 0..len
    # for the common case of timing contiguous batches of val rows:
    n = len(df)
    positions = np.arange(n, dtype=np.int64)

    def predict_batch_positions(batch_x_or_idx: np.ndarray) -> np.ndarray:
        """If 2D features, interpret as consecutive rows by length starting at 0
        is wrong. Better API: always pass 1d indices.
        """
        arr = np.asarray(batch_x_or_idx)
        if arr.ndim == 1:
            return predict_rows(arr)
        # 2D: treat as selecting first arr.shape[0] rows of a provided order —
        # callers should use the wrapper below with explicit indices.
        raise TypeError("pass 1d row indices for teacher predict")

    meta_out = {
        "feat_cols": cols,
        "n_val": n,
        "meta_kind": bundle.get("meta_kind"),
        "cache_path": str(cache_path),
        "positions": positions,
    }
    return predict_rows, meta_out


def efficiency_metrics(
    *,
    val_rmse: float,
    n_params: int,
    size_mb: float,
    latency_ms: float,
    teacher_val_rmse: float | None = None,
    teacher_latency_ms: float | None = None,
    teacher_size_mb: float | None = None,
    teacher_ram_mb: float | None = None,
    student_ram_mb: float | None = None,
) -> dict[str, float]:
    """Derived efficiency / Pareto metrics."""
    params_m = max(n_params / 1e6, 1e-12)
    size = max(size_mb, 1e-12)
    lat = max(latency_ms, 1e-12)
    out = {
        "rmse_per_million_params": float(val_rmse) / params_m,
        "rmse_per_mb": float(val_rmse) / size,
        "rmse_per_ms": float(val_rmse) / lat,
        "param_efficiency_inv_rmse_per_m": params_m / max(float(val_rmse), 1e-12),
    }
    if teacher_val_rmse is not None:
        out["delta_rmse_vs_teacher"] = float(val_rmse) - float(teacher_val_rmse)
    if teacher_latency_ms is not None and teacher_latency_ms > 0:
        out["latency_speedup_vs_teacher"] = float(teacher_latency_ms) / lat
        out["latency_improvement_ms"] = float(teacher_latency_ms) - lat
    if teacher_size_mb is not None and teacher_size_mb > 0:
        out["size_reduction_factor"] = float(teacher_size_mb) / size
    if teacher_ram_mb is not None and student_ram_mb is not None and student_ram_mb > 0:
        out["memory_reduction_factor"] = float(teacher_ram_mb) / float(student_ram_mb)
    return out
