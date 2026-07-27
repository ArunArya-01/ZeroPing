from __future__ import annotations

"""Frozen statistical protocol for AeroTwin.

This module is the **single source of truth** for the inference methodology used
across every AeroTwin experiment (Level 1 flight-level, Level 2 LOTO, and
external / cross-dataset generalization). It exists so that the statistical
protocol is *frozen* rather than re-tuned per notebook:

  * All numeric thresholds (bootstrap count, split size, significance level,
    confidence level, replication threshold) live here as constants.
  * The interpretation policy (how a bootstrap CI maps to a plain-language
    verdict) is implemented once in :func:`classify_significance` and matches
    the rule already applied in ``physics.eval_framework.significance_test``.
  * The correct resampling unit for each scientific claim is enforced by
    :func:`frozen_significance`.

Changing any constant here is a **protocol change** and must be reflected in
``papers/statistical_protocol.md`` and bumped in ``PROTOCOL_VERSION``.

The module deliberately imports no heavy ML dependencies at module load time
(``catboost`` / ``lightgbm`` / ``xgboost`` / ``sklearn``), so it can be imported
and its constants inspected in a minimal environment. Actual bootstrap
computation is delegated lazily to ``physics.eval_framework`` at call time.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Frozen constants — change only via a documented protocol revision.
# --------------------------------------------------------------------------- #
PROTOCOL_VERSION = "1.0"

RANDOM_STATE = 42
"""Seed for every split, bootstrap, and model that consumes randomness."""

TEST_SIZE = 0.2
"""Held-out fraction for strict flight-level (Level 1) evaluation."""

N_BOOTSTRAP = 10_000
"""Bootstrap resamples for flight-clustered MAE-difference inference."""

ALPHA = 0.05
"""Nominal significance level (two-sided family, one-sided at ALPHA)."""

CI_LEVEL = 0.95
"""Bootstrap confidence level; mapped to the 2.5 / 97.5 percentiles."""

REPLICATION_P_THRESHOLD = 0.95
"""P(new better) at or above this counts a qualitative finding as replicated
on a single external dataset (mirrors the one-sided ALPHA rule)."""


# --------------------------------------------------------------------------- #
# Inference units — the resampling unit must match the scientific claim.
# --------------------------------------------------------------------------- #
INFERENCE_UNITS: dict[str, str] = {
    "flight_level": (
        "Unseen-flight generalization (Level 1): resample held-out flights."
    ),
    "type_level": (
        "Unseen-aircraft-family generalization (Level 2 LOTO): resample the "
        "12 held-out aircraft types."
    ),
    "hierarchical_type_to_flight": (
        "LOTO paired comparison: hierarchical type -> flight resampling."
    ),
}

ALLOWED_INFERENCE_UNITS = tuple(INFERENCE_UNITS.keys())


# --------------------------------------------------------------------------- #
# Protocol functions
# --------------------------------------------------------------------------- #
def bootstrap_ci(boot_dist: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    """Return the (lower, upper) percentile bounds for ``level`` confidence.

    ``boot_dist`` is the per-resample MAE-difference distribution
    (MAE(new) - MAE(baseline)); negative values favour the new approach.
    """
    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = (1.0 + level) / 2.0 * 100.0
    lo, hi = np.percentile(np.asarray(boot_dist, dtype=np.float64), [lo_q, hi_q])
    return float(lo), float(hi)


def classify_significance(
    delta_mae: float,
    ci_lower: float,
    ci_upper: float,
    bootstrap_p: float,
    new_name: str = "new",
    baseline_name: str = "baseline",
) -> str:
    """Map frozen bootstrap evidence to a plain-language verdict.

    Implements the project interpretation policy (PROJECT_STATUS_REPORT §11):

      * ``ci_upper < 0`` AND ``bootstrap_p < ALPHA`` -> new significantly better
        (``bootstrap_p`` is P(MAE new > MAE baseline), i.e. P(new worse)).
      * ``ci_lower > 0`` -> baseline significantly better.
      * otherwise -> no significant evidence (do not claim an effect).

    This rule is identical to the one in ``physics.eval_framework.significance_test``;
    centralizing it here prevents per-notebook drift.
    """
    if ci_upper < 0 and float(bootstrap_p) < ALPHA:
        return f"{new_name} significantly better than {baseline_name}"
    if ci_lower > 0:
        return f"{baseline_name} significantly better than {new_name}"
    return "No significant evidence"


def frozen_significance(
    err_new: np.ndarray,
    err_baseline: np.ndarray,
    flight_ids: np.ndarray,
    new_name: str,
    baseline_name: str,
    unit: str,
) -> dict:
    """Run a protocol-compliant significance test.

    Delegates the bootstrap computation to ``physics.eval_framework.significance_test``
    (lazy import, so heavy ML deps are only required when this is actually run),
    then tags the result with the frozen protocol metadata and **enforces** that
    ``unit`` is one of :data:`ALLOWED_INFERENCE_UNITS`.

    Returns the ``significance_test`` dict augmented with ``inference_unit``,
    ``inference_unit_desc``, ``alpha``, and ``protocol_version``.
    """
    if unit not in ALLOWED_INFERENCE_UNITS:
        raise ValueError(
            f"Unknown inference unit {unit!r}. Allowed: {ALLOWED_INFERENCE_UNITS}. "
            "The resampling unit must match the scientific claim."
        )
    from aerotwin.engine.eval_framework import significance_test

    sig = significance_test(
        np.asarray(err_new, dtype=np.float64),
        np.asarray(err_baseline, dtype=np.float64),
        np.asarray(flight_ids),
        new_name,
        baseline_name,
    )
    sig["inference_unit"] = unit
    sig["inference_unit_desc"] = INFERENCE_UNITS[unit]
    sig["alpha"] = ALPHA
    sig["protocol_version"] = PROTOCOL_VERSION
    return sig


def replication_decision(
    delta_mae: float,
    p_new_better: float,
    threshold: float = REPLICATION_P_THRESHOLD,
) -> bool:
    """Decide whether one dataset replicates a qualitative finding.

    Mirrors ``physics.cross_dataset_replication.dataset_replicates_flow_better``:
    the new approach must have strictly lower MAE *and* meet ``threshold`` on
    P(new better). Centralizing the threshold keeps replication claims frozen.
    """
    return (delta_mae < 0) and (float(p_new_better) >= threshold)


def assert_protocol_constants() -> None:
    """Lightweight guard that the frozen constants stay in their locked values.

    Useful as a documentation anchor / sanity check; not a replacement for the
    documented values in ``papers/statistical_protocol.md``.
    """
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.2
    assert N_BOOTSTRAP == 10_000
    assert ALPHA == 0.05
    assert abs(CI_LEVEL - 0.95) < 1e-9
    assert REPLICATION_P_THRESHOLD == 0.95
