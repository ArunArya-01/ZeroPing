"""MLP capacity tiers for Step-4 scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityTier:
    name: str
    hidden_dims: tuple[int, ...]
    target_params: str


# Approximate targets for in_dim ≈ 580–590 (OHE-expanded distillation features).
# Measured on in_dim=582: Tiny 239k, Small 504k, Medium 1.13M, Large 2.89M, XLarge 6.75M.
CAPACITY_TIERS: tuple[CapacityTier, ...] = (
    CapacityTier("Tiny", (320, 160), "~250K"),
    CapacityTier("Small", (576, 288), "~500K"),
    CapacityTier("Medium", (1024, 512), "~1M"),
    CapacityTier("Large", (1792, 1024), "~3M"),
    CapacityTier("XLarge", (2560, 2048), "~5-10M"),
)

# Fixed KD weights from Step 3 (immutable for Step 4).
FIXED_ALPHA = 0.1
FIXED_BETA = 0.9

REPRO_SEEDS: tuple[int, ...] = (42, 123, 3407, 2025, 9999)
