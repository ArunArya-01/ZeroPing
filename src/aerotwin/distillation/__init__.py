"""Reusable training utilities for AeroTwin knowledge-distillation students.

The frozen teacher dataset (`distillation_dataset.parquet`) is the sole data
source. Feature engineering is not modified here.

Swap student architectures via::

    from aerotwin.distillation.models import build_student
    model = build_student("ft_transformer", in_dim=data.in_dim)

or by providing a ``model_factory`` to ``run_kd_sweep`` / ``run_single_experiment``.
The training loop stays fixed.
"""

from aerotwin.distillation.data import FEATURE_COLS_DEFAULT, DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models import (
    FTTransformer,
    StudentConfig,
    build_student,
    list_architectures,
)
from aerotwin.distillation.runner import (
    DEFAULT_KD_SWEEP,
    ExperimentConfig,
    KDWeightConfig,
    analyze_kd_sweep,
    run_kd_sweep,
    run_single_experiment,
)
from aerotwin.distillation.trainer import TrainConfig, train_student

__all__ = [
    "DistillationData",
    "FEATURE_COLS_DEFAULT",
    "StudentMLP",
    "FTTransformer",
    "StudentConfig",
    "build_student",
    "list_architectures",
    "TrainConfig",
    "train_student",
    "regression_metrics",
    "DEFAULT_KD_SWEEP",
    "ExperimentConfig",
    "KDWeightConfig",
    "analyze_kd_sweep",
    "run_kd_sweep",
    "run_single_experiment",
]
