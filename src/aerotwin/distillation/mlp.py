"""Compact MLP student for tabular fuel-burn regression."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class StudentMLP(nn.Module):
    """Baseline MLP student.

    Architecture (per block, repeated for each hidden width)::

        Linear -> ReLU -> LayerNorm -> Dropout

    followed by a final linear head to a scalar prediction (kg).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (1024, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        dims = [int(in_dim), *[int(h) for h in hidden_dims]]
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 1):
            blocks.extend(
                [
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(dims[i + 1]),
                    nn.Dropout(p=dropout),
                ]
            )
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], 1)
        self.in_dim = int(in_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation (post-backbone, pre-head)."""
        return self.backbone(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return int(sum(p.numel() for p in params))
