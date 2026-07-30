"""Generic student training loop with KD loss, early stopping, and checkpointing."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aerotwin.distillation.metrics import compare_to_teacher, regression_metrics

LOGGER = logging.getLogger(__name__)

LossMode = Literal["gt", "teacher", "kd"]


@dataclass
class TrainConfig:
    mode: LossMode = "kd"
    alpha: float = 0.5
    beta: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 80
    patience: int = 12
    min_delta: float = 0.05  # kg RMSE improvement required to reset patience
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    hidden_dims: tuple[int, ...] = (1024, 512)
    dropout: float = 0.1
    run_name: str = "student"
    extras: dict[str, Any] = field(default_factory=dict)

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    pred: torch.Tensor,
    y_gt: torch.Tensor,
    y_teacher: torch.Tensor,
    *,
    mode: LossMode,
    alpha: float,
    beta: float,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_gt = criterion(pred, y_gt)
    loss_t = criterion(pred, y_teacher)
    if mode == "gt":
        loss = loss_gt
    elif mode == "teacher":
        loss = loss_t
    elif mode == "kd":
        loss = alpha * loss_gt + beta * loss_t
    else:
        raise ValueError(f"Unknown loss mode: {mode}")
    parts = {
        "loss": float(loss.detach().cpu()),
        "loss_gt": float(loss_gt.detach().cpu()),
        "loss_teacher": float(loss_t.detach().cpu()),
    }
    return loss, parts


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    teachers: list[np.ndarray] = []
    sids: list[np.ndarray] = []
    has_sid = False
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x)
        preds.append(pred.detach().cpu().numpy())
        gts.append(batch["y_gt"].numpy())
        teachers.append(batch["y_teacher"].numpy())
        if "sample_id" in batch:
            has_sid = True
            sids.append(batch["sample_id"].numpy())
    y_pred = np.concatenate(preds)
    y_gt = np.concatenate(gts)
    y_teacher = np.concatenate(teachers)
    sample_ids = np.concatenate(sids) if has_sid else None
    return y_pred, y_gt, y_teacher, sample_ids


def train_student(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    *,
    out_dir: Path,
    log_dir: Path | None = None,
    train_eval_loader: DataLoader | None = None,
    model_builder: Callable[[], nn.Module] | None = None,
) -> dict[str, Any]:
    """Train a student model and write checkpoints + metrics under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else out_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_train_loader = train_eval_loader or train_loader

    set_seed(config.seed)
    device = config.resolved_device()
    model = model.to(device)

    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    LOGGER.info(
        "Training %s on %s | mode=%s alpha=%.3f beta=%.3f | params=%s",
        config.run_name,
        device,
        config.mode,
        config.alpha,
        config.beta,
        f"{n_params:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = -1
    patience_left = config.patience
    history: list[dict[str, float]] = []
    ckpt_path = out_dir / "best_model.pt"
    t0 = time.time()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y_gt = batch["y_gt"].to(device)
            y_t = batch["y_teacher"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = compute_loss(
                pred,
                y_gt,
                y_t,
                mode=config.mode,
                alpha=config.alpha,
                beta=config.beta,
                criterion=criterion,
            )
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        # Epoch-end evaluation (non-shuffled train loader when provided)
        tr_pred, tr_gt, tr_teacher, _ = predict_loader(model, eval_train_loader, device)
        va_pred, va_gt, va_teacher, _ = predict_loader(model, val_loader, device)
        tr_m = regression_metrics(tr_gt, tr_pred)
        va_m = regression_metrics(va_gt, va_pred)
        tr_t = regression_metrics(tr_gt, tr_teacher)
        va_t = regression_metrics(va_gt, va_teacher)
        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "train_rmse": tr_m["rmse"],
            "train_mae": tr_m["mae"],
            "train_bias": tr_m["bias"],
            "train_r2": tr_m["r2"],
            "val_rmse": va_m["rmse"],
            "val_mae": va_m["mae"],
            "val_bias": va_m["bias"],
            "val_r2": va_m["r2"],
            "val_teacher_rmse": va_t["rmse"],
            "train_teacher_rmse": tr_t["rmse"],
            "lr": lr_now,
        }
        history.append(row)
        LOGGER.info(
            "epoch %03d | loss=%.4f train_rmse=%.2f val_rmse=%.2f "
            "val_mae=%.2f val_bias=%+.2f r2=%.4f lr=%.2e",
            epoch,
            row["train_loss"],
            row["train_rmse"],
            row["val_rmse"],
            row["val_mae"],
            row["val_bias"],
            row["val_r2"],
            lr_now,
        )

        scheduler.step(row["val_rmse"])

        improved = row["val_rmse"] < best_val_rmse - config.min_delta
        if improved:
            best_val_rmse = row["val_rmse"]
            best_epoch = epoch
            patience_left = config.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                    "n_params": n_params,
                    "in_dim": getattr(model, "in_dim", None),
                    "hidden_dims": getattr(model, "hidden_dims", None),
                    "architecture": getattr(model, "architecture", None),
                    "model_config": getattr(model, "config_dict", lambda: None)()
                    if callable(getattr(model, "config_dict", None))
                    else None,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info(
                    "Early stopping at epoch %d (best epoch %d, best val RMSE %.2f)",
                    epoch,
                    best_epoch,
                    best_val_rmse,
                )
                break

    train_seconds = time.time() - t0

    # Reload best weights for final evaluation
    if ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])

    tr_pred, tr_gt, tr_teacher, tr_sid = predict_loader(model, eval_train_loader, device)
    va_pred, va_gt, va_teacher, va_sid = predict_loader(model, val_loader, device)

    train_cmp = compare_to_teacher(tr_gt, tr_pred, tr_teacher)
    val_cmp = compare_to_teacher(va_gt, va_pred, va_teacher)

    # Training curve CSV
    curve_path = out_dir / "training_curve.csv"
    import polars as pl

    pl.DataFrame(history).write_csv(curve_path)

    # Predictions parquet (val + train tagged)
    pred_frames = []
    for split_name, y_p, y_g, y_t, sid in [
        ("train", tr_pred, tr_gt, tr_teacher, tr_sid),
        ("val", va_pred, va_gt, va_teacher, va_sid),
    ]:
        d = {
            "split": [split_name] * len(y_p),
            "ground_truth": y_g,
            "teacher_prediction": y_t,
            "student_prediction": y_p,
        }
        if sid is not None:
            d["sample_id"] = sid
        pred_frames.append(pl.DataFrame(d))
    pred_path = out_dir / "predictions.parquet"
    pl.concat(pred_frames).write_parquet(pred_path)

    metrics: dict[str, Any] = {
        "run_name": config.run_name,
        "mode": config.mode,
        "alpha": config.alpha,
        "beta": config.beta,
        "n_params": n_params,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "train": train_cmp,
        "val": val_cmp,
        "config": asdict(config),
        "paths": {
            "checkpoint": str(ckpt_path),
            "training_curve": str(curve_path),
            "predictions": str(pred_path),
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    # Also mirror metrics/curve into log_dir if distinct
    if log_dir.resolve() != out_dir.resolve():
        (log_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
        pl.DataFrame(history).write_csv(log_dir / "training_curve.csv")

    LOGGER.info(
        "Done %s | best_epoch=%d val_rmse=%.2f teacher_val_rmse=%.2f gap=%+.2f (%.1fs)",
        config.run_name,
        best_epoch,
        val_cmp["student"]["rmse"],
        val_cmp["teacher"]["rmse"],
        val_cmp["teacher_student_rmse_gap"],
        train_seconds,
    )
    return metrics
