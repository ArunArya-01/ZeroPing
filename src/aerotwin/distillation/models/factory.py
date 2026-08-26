"""Student model factory — architecture selected by name / config.

Example
-------
>>> model = build_student("ft_transformer", in_dim=582)
>>> model = build_student(StudentConfig(architecture="large_mlp", in_dim=582))
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Mapping

import torch.nn as nn

from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models.ft_transformer import FTTransformer

# Registry of architecture names → builder(in_dim, **kwargs) -> nn.Module
ARCHITECTURES: dict[str, Callable[..., nn.Module]] = {}


def register_architecture(name: str):
    """Decorator to register a student architecture under ``name``."""

    def deco(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        key = name.lower().strip()
        if key in ARCHITECTURES:
            raise ValueError(f"Architecture already registered: {key}")
        ARCHITECTURES[key] = fn
        return fn

    return deco


def list_architectures() -> list[str]:
    return sorted(ARCHITECTURES.keys())


@dataclass
class StudentConfig:
    """Architecture + hyperparameters for :func:`build_student`.

    Shared training knobs (α, β, lr, …) live in ``ExperimentConfig`` / ``TrainConfig``.
    Only architecture-specific fields belong here.
    """

    architecture: str = "large_mlp"
    in_dim: int | None = None

    # MLP
    hidden_dims: tuple[int, ...] | list[int] = (1792, 1024)
    dropout: float = 0.1

    # FT-Transformer (Gorishniy et al. baselines)
    d_token: int = 192
    n_blocks: int = 3
    n_heads: int = 8
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    ffn_d_hidden: int | None = None
    # Native tabular layout decoded from dense [num | OHE] vector
    n_num_features: int | None = None
    cat_cardinalities: list[int] | tuple[int, ...] | None = None

    # Escape hatch for future architectures (TabTransformer, SAINT, TabM, …)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "StudentConfig":
        """Build from a dict / YAML ``student:`` section (unknown keys → extras)."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        extras: dict[str, Any] = dict(m.get("extras") or {})
        for k, v in m.items():
            if k == "extras":
                continue
            if k in known:
                kwargs[k] = v
            else:
                extras[k] = v
        if extras:
            kwargs["extras"] = extras
        if "hidden_dims" in kwargs and isinstance(kwargs["hidden_dims"], list):
            kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
        return cls(**kwargs)


# ---- Registered builders ----------------------------------------------------


@register_architecture("mlp")
@register_architecture("student_mlp")
def _build_mlp(
    in_dim: int,
    *,
    hidden_dims: tuple[int, ...] = (1024, 512),
    dropout: float = 0.1,
    **_: Any,
) -> StudentMLP:
    return StudentMLP(in_dim, hidden_dims=hidden_dims, dropout=dropout)


@register_architecture("large_mlp")
def _build_large_mlp(
    in_dim: int,
    *,
    dropout: float = 0.1,
    **kwargs: Any,
) -> StudentMLP:
    hidden = kwargs.get("hidden_dims") or (1792, 1024)
    return StudentMLP(in_dim, hidden_dims=tuple(hidden), dropout=dropout)


@register_architecture("xlarge_mlp")
def _build_xlarge_mlp(
    in_dim: int,
    *,
    dropout: float = 0.1,
    **kwargs: Any,
) -> StudentMLP:
    hidden = kwargs.get("hidden_dims") or (2560, 2048)
    return StudentMLP(in_dim, hidden_dims=tuple(hidden), dropout=dropout)


@register_architecture("tiny_mlp")
def _build_tiny_mlp(in_dim: int, *, dropout: float = 0.1, **_: Any) -> StudentMLP:
    return StudentMLP(in_dim, hidden_dims=(512, 256), dropout=dropout)


@register_architecture("small_mlp")
def _build_small_mlp(in_dim: int, *, dropout: float = 0.1, **_: Any) -> StudentMLP:
    return StudentMLP(in_dim, hidden_dims=(768, 384), dropout=dropout)


@register_architecture("medium_mlp")
def _build_medium_mlp(in_dim: int, *, dropout: float = 0.1, **_: Any) -> StudentMLP:
    return StudentMLP(in_dim, hidden_dims=(1024, 512), dropout=dropout)


@register_architecture("ft_transformer")
@register_architecture("ft-transformer")
@register_architecture("fttransformer")
def _build_ft_transformer(
    in_dim: int,
    *,
    d_token: int = 192,
    n_blocks: int = 3,
    n_heads: int = 8,
    attention_dropout: float = 0.2,
    ffn_dropout: float = 0.1,
    residual_dropout: float = 0.0,
    ffn_d_hidden: int | None = None,
    n_num_features: int | None = None,
    cat_cardinalities: list[int] | tuple[int, ...] | None = None,
    **_: Any,
) -> FTTransformer:
    return FTTransformer(
        in_dim,
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        ffn_d_hidden=ffn_d_hidden,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
    )


def build_student(
    architecture: str | StudentConfig = "large_mlp",
    in_dim: int | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Construct a student network by architecture name or :class:`StudentConfig`.

    Parameters
    ----------
    architecture:
        Name (``large_mlp``, ``xlarge_mlp``, ``ft_transformer``, …) or a full
        :class:`StudentConfig`.
    in_dim:
        Input feature dimension (required unless provided on the config).
    **kwargs:
        Overrides forwarded to the architecture builder (and/or config fields).

    Returns
    -------
    torch.nn.Module
        Student regressor with ``forward(x) -> (batch,)`` fuel predictions.
    """
    if isinstance(architecture, StudentConfig):
        cfg = architecture
        # kwargs override config fields
        if kwargs:
            data = cfg.to_dict()
            data.update(kwargs)
            if "hidden_dims" in data and isinstance(data["hidden_dims"], list):
                data["hidden_dims"] = tuple(data["hidden_dims"])
            cfg = StudentConfig.from_mapping(data)
        arch = cfg.architecture
        dim = in_dim if in_dim is not None else cfg.in_dim
        if dim is None:
            raise ValueError("in_dim must be set on StudentConfig or passed explicitly")
        builder_kwargs = {
            "hidden_dims": tuple(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "d_token": cfg.d_token,
            "n_blocks": cfg.n_blocks,
            "n_heads": cfg.n_heads,
            "attention_dropout": cfg.attention_dropout,
            "ffn_dropout": cfg.ffn_dropout,
            "residual_dropout": cfg.residual_dropout,
            "ffn_d_hidden": cfg.ffn_d_hidden,
            "n_num_features": cfg.n_num_features,
            "cat_cardinalities": list(cfg.cat_cardinalities)
            if cfg.cat_cardinalities is not None
            else None,
            **cfg.extras,
        }
    else:
        arch = str(architecture)
        dim = in_dim
        if dim is None:
            raise ValueError("in_dim is required when architecture is a string")
        builder_kwargs = dict(kwargs)

    key = arch.lower().strip()
    if key not in ARCHITECTURES:
        known = ", ".join(list_architectures())
        raise KeyError(f"Unknown architecture {arch!r}. Registered: {known}")

    model = ARCHITECTURES[key](int(dim), **builder_kwargs)
    # Stamp architecture name for checkpoint metadata
    if not hasattr(model, "architecture"):
        setattr(model, "architecture", key)
    return model
