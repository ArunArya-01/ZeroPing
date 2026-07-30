"""Regression metrics for distillation evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(err**2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean (pred - true). Positive => over-prediction."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_to_teacher(
    y_true: np.ndarray,
    y_student: np.ndarray,
    y_teacher: np.ndarray,
) -> dict[str, Any]:
    m_s = regression_metrics(y_true, y_student)
    m_t = regression_metrics(y_true, y_teacher)
    return {
        "student": m_s,
        "teacher": m_t,
        "teacher_student_rmse_gap": m_s["rmse"] - m_t["rmse"],
        "student_vs_teacher_rmse": rmse(y_teacher, y_student),
    }
