"""Variance-Guided Knowledge Distillation (VGKD).

Adaptive KD weights from frozen teacher ensemble disagreement (Phase 1A):

    u(x)  = std of 6 base ensemble kg predictions
    u_n   = z-score of u on the training split
    β(x)  = β_base · exp(-λ · u_n⁺)   or linear variant
    α(x)  = 1 − β(x)

with u_n⁺ = max(u_n, 0) for exp (only up-weight GT when uncertainty ≥ train mean).
Actually the user formula is exp(-λ · u_norm). With z-score, negative u_norm
increases β above β_base which violates 0 ≤ β ≤ β_base.

We clamp:
    β(x) = clip(β_base · f(u_norm), 0, β_base)

For exp: f = exp(-λ · max(u_norm, 0)) so confident samples (u < mean) keep β=β_base,
uncertain samples reduce β. This matches the scientific intent from Phase 1A.

For linear: β = β_base · clip(1 − λ · max(u_norm, 0), 0, 1)

λ=0 → β=β_base everywhere (fixed KD baseline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

BASE_PRED_COLS = [
    "xgb_direct_prediction",
    "lgbm_direct_prediction",
    "cat_direct_prediction",
    "xgb_flow_prediction",
    "lgbm_flow_prediction",
    "cat_flow_prediction",
]

WeightFn = Literal["exp", "linear"]
UncertaintySource = Literal["ensemble_std", "random", "oracle_abs_error"]


@dataclass
class VGKDConfig:
    beta_base: float = 0.9
    lam: float = 0.0
    weight_fn: WeightFn = "exp"
    uncertainty_source: UncertaintySource = "ensemble_std"
    # Z-score stats (fit on train only)
    u_mean: float = 0.0
    u_std: float = 1.0
    # For static-β ablation: if set, ignore adaptive formula
    static_beta: float | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensemble_std_from_bases(base_matrix: np.ndarray) -> np.ndarray:
    """base_matrix: (n, 6) → per-row std."""
    return np.std(base_matrix.astype(np.float64), axis=1)


def fit_zscore(u: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(u))
    sig = float(np.std(u))
    if sig < 1e-8:
        sig = 1.0
    return mu, sig


def zscore(u: np.ndarray, mu: float, sig: float) -> np.ndarray:
    return (u.astype(np.float64) - mu) / sig


def adaptive_beta(
    u_norm: np.ndarray | torch.Tensor,
    *,
    beta_base: float = 0.9,
    lam: float = 0.0,
    weight_fn: WeightFn = "exp",
    static_beta: float | None = None,
) -> np.ndarray | torch.Tensor:
    """Compute β(x) with bounds [0, β_base]."""
    if static_beta is not None:
        if isinstance(u_norm, torch.Tensor):
            return torch.full_like(u_norm, float(static_beta), dtype=torch.float32)
        return np.full_like(u_norm, float(static_beta), dtype=np.float64)

    if isinstance(u_norm, torch.Tensor):
        # only reduce teacher weight when uncertainty ≥ train mean
        u_pos = torch.clamp(u_norm, min=0.0)
        if weight_fn == "exp":
            beta = beta_base * torch.exp(-float(lam) * u_pos)
        else:
            beta = beta_base * torch.clamp(1.0 - float(lam) * u_pos, min=0.0, max=1.0)
        return torch.clamp(beta, min=0.0, max=float(beta_base))

    u_pos = np.maximum(u_norm.astype(np.float64), 0.0)
    if weight_fn == "exp":
        beta = beta_base * np.exp(-float(lam) * u_pos)
    else:
        beta = beta_base * np.clip(1.0 - float(lam) * u_pos, 0.0, 1.0)
    return np.clip(beta, 0.0, float(beta_base))


def adaptive_alpha(beta: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(beta, torch.Tensor):
        return 1.0 - beta
    return 1.0 - beta.astype(np.float64)


def vgkd_loss(
    pred: torch.Tensor,
    y_gt: torch.Tensor,
    y_teacher: torch.Tensor,
    u_norm: torch.Tensor,
    *,
    cfg: VGKDConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-sample VGKD loss, mean-reduced."""
    beta = adaptive_beta(
        u_norm,
        beta_base=cfg.beta_base,
        lam=cfg.lam,
        weight_fn=cfg.weight_fn,
        static_beta=cfg.static_beta,
    )
    alpha = adaptive_alpha(beta)
    err_gt = (pred - y_gt) ** 2
    err_t = (pred - y_teacher) ** 2
    per = alpha * err_gt + beta * err_t
    loss = per.mean()
    parts = {
        "loss": float(loss.detach().cpu()),
        "mean_alpha": float(alpha.detach().mean().cpu()),
        "mean_beta": float(beta.detach().mean().cpu()),
        "min_beta": float(beta.detach().min().cpu()),
        "max_beta": float(beta.detach().max().cpu()),
    }
    return loss, parts


class VGKDDataset(torch.utils.data.Dataset):
    """KD dataset with per-sample normalized uncertainty."""

    def __init__(
        self,
        x: np.ndarray,
        y_gt: np.ndarray,
        y_teacher: np.ndarray,
        u_norm: np.ndarray,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y_gt = torch.as_tensor(y_gt, dtype=torch.float32)
        self.y_teacher = torch.as_tensor(y_teacher, dtype=torch.float32)
        self.u_norm = torch.as_tensor(u_norm, dtype=torch.float32)
        self.sample_ids = (
            None if sample_ids is None else torch.as_tensor(sample_ids, dtype=torch.int64)
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "x": self.x[idx],
            "y_gt": self.y_gt[idx],
            "y_teacher": self.y_teacher[idx],
            "u_norm": self.u_norm[idx],
        }
        if self.sample_ids is not None:
            item["sample_id"] = self.sample_ids[idx]
        return item
