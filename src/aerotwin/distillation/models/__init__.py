"""Pluggable student architectures for AeroTwin distillation.

Use :func:`build_student` to construct models by name. The training loop
(:mod:`aerotwin.distillation.trainer`) stays architecture-agnostic.
"""

from __future__ import annotations

from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models.factory import (
    ARCHITECTURES,
    StudentConfig,
    build_student,
    list_architectures,
)
from aerotwin.distillation.models.ft_transformer import FTTransformer

__all__ = [
    "ARCHITECTURES",
    "StudentConfig",
    "StudentMLP",
    "FTTransformer",
    "build_student",
    "list_architectures",
]
