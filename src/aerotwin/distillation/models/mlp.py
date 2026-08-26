"""MLP student re-export (canonical implementation remains ``aerotwin.distillation.mlp``).

Kept so callers can import from ``aerotwin.distillation.models`` uniformly
without duplicating or modifying the original Large/XLarge MLP code.
"""

from __future__ import annotations

from aerotwin.distillation.mlp import StudentMLP

__all__ = ["StudentMLP"]
