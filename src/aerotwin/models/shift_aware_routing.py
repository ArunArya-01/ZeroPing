from __future__ import annotations

"""Shift-aware routing (conditional scaffold).

Background
----------
Under leave-one-type-out (LOTO) evaluation, the Fuel-Flow vs Direct kilogram
formulation does **not** win uniformly: Flow+Energy is favourable on some folds
(B77W, A332, A321, …) and Direct is better on others (B789, A333, B738, …).
Priority 7 of the project plan therefore asks, *only after* the operational
distribution-shift analysis explains that heterogeneity, for a **simple** router
— not a complicated one — comparing:

  * Always Direct / Always Flow (baselines)
  * Oracle selector (upper bound only — needs ground-truth MAE, not deployable)
  * Learned selector, calibrated on train/validation only (nested evaluation)

This module implements exactly those simple policies. It is intentionally a
**scaffold**: the learned selector refuses to route until it has been calibrated
from a validation fold table, and the oracle is explicitly flagged as a
non-deployable upper bound. Building or deploying a router before operational-shift
evidence exists is contrary to the project plan; this code makes the gate explicit
rather than silently producing a router.

The operational shift score reuses the candidate variables from §20 Priority 1
(duration, altitude, speed, vertical rate, phase fractions, energy rates,
trajectory density, TAS method, start/end flight fraction).

Runtime dependencies: only ``numpy`` / ``polars`` at import time. ``scipy`` is
imported lazily inside the Wasserstein / Jensen–Shannon paths so the module
compiles and imports in a minimal environment.
"""

import numpy as np
import polars as pl

from aerotwin.engine.statistical_protocol import PROTOCOL_VERSION  # noqa: E402

# Candidate operational-distribution variables (PROJECT_STATUS_REPORT §20 P1).
OPERATIONAL_FEATURES: list[str] = [
    "duration_s",
    "start_fraction_of_flight",
    "end_fraction_of_flight",
    "n_traj_pts",
    "mean_altitude",
    "median_altitude",
    "std_altitude",
    "mean_groundspeed",
    "std_groundspeed",
    "max_groundspeed",
    "mean_vertical_rate",
    "std_vertical_rate",
    "climb_fraction",
    "cruise_fraction",
    "descent_fraction",
    "energy_rate_jpkg_s",
]

VALID_POLICIES = ("always_direct", "always_flow", "oracle", "learned")


# --------------------------------------------------------------------------- #
# Operational shift distances (per feature, two 1-D samples)
# --------------------------------------------------------------------------- #
def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference: |mean diff| / pooled std (std units)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    pool = np.sqrt((a.var() + b.var()) / 2.0)
    if pool <= 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pool)


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(wasserstein_distance(a, b))


def _jensen_shannon(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import entropy

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, 20)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    # Smooth to avoid log(0); renormalize.
    ha = ha + 1e-9
    hb = hb + 1e-9
    ha /= ha.sum()
    hb /= hb.sum()
    m = 0.5 * (ha + hb)
    return float(0.5 * entropy(ha, m) + 0.5 * entropy(hb, m))


def _feature_distance(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if method == "smd":
        return _smd(a, b)
    if method == "wasserstein":
        # Normalize by training std so the magnitude is comparable across features.
        scale = float(np.std(b)) if np.std(b) > 0 else 1.0
        return _wasserstein(a, b) / scale
    if method == "js":
        return _jensen_shannon(a, b)
    raise ValueError(f"Unknown distance method: {method!r}")


def operational_shift_score(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    method: str = "smd",
    features: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Aggregate operational distribution shift between train and test.

    Returns ``(mean_score, per_feature)`` where ``mean_score`` is the mean of the
    per-feature distances and ``per_feature`` is the breakdown. Missing features
    are skipped (``avail``-style). Both must be present in both frames.
    """
    feats = [f for f in (features or OPERATIONAL_FEATURES) if f in train_df.columns and f in test_df.columns]
    if not feats:
        raise ValueError("No operational features shared by train and test frames.")
    per_feature: dict[str, float] = {}
    for f in feats:
        ta = train_df[f].drop_nulls().to_numpy()
        tb = test_df[f].drop_nulls().to_numpy()
        per_feature[f] = _feature_distance(tb, ta, method)
    mean_score = float(np.mean(list(per_feature.values())))
    return mean_score, per_feature


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class ShiftAwareRouter:
    """Simple, gated shift-aware router over Direct vs Flow+Energy formulations.

    Policies
    --------
    * ``always_direct`` / ``always_flow`` — constant baselines (always routable).
    * ``oracle`` — picks the lower-MAE formulation per unit using *ground-truth*
      MAE. **Upper bound only; not deployable.** Requires ``mae_direct`` and
      ``mae_flow`` per unit at routing time.
    * ``learned`` — calibrated from a validation fold table; refuses to route
      until :meth:`calibrate` has run (enforces the project's "no router before
      operational-shift evidence" gate).

    The shift score uses ``OPERATIONAL_FEATURES`` and the frozen random seed from
    the statistical protocol for any randomized tie-breaking.
    """

    def __init__(self, policy: str = "always_direct", method: str = "smd", unit_col: str = "flight_id"):
        if policy not in VALID_POLICIES:
            raise ValueError(f"policy must be one of {VALID_POLICIES}; got {policy!r}")
        self.policy = policy
        self.method = method
        self.unit_col = unit_col
        self.calibrated = False
        self.threshold: float | None = None
        self.direction: str | None = None  # "flow_when_high" | "direct_when_high"
        self.protocol_version = PROTOCOL_VERSION
        self._ref: dict[str, np.ndarray] = {}

    # -- reference / calibration ------------------------------------------ #
    def fit_reference(self, train_df: pl.DataFrame) -> "ShiftAwareRouter":
        """Store the training operational reference distributions."""
        for f in OPERATIONAL_FEATURES:
            if f in train_df.columns:
                self._ref[f] = train_df[f].drop_nulls().to_numpy()
        return self

    def calibrate(self, fold_table: pl.DataFrame) -> "ShiftAwareRouter":
        """Fit the learned selector from a validation fold table.

        ``fold_table`` must carry ``operational_distance`` (already computed via
        :func:`operational_shift_score`) and ``delta_mae_flow_minus_direct``
        (Flow MAE − Direct MAE per held-out type). A simple threshold over the
        operational distance is chosen by minimizing misrouting error on these
        folds (train/validation only). The direction is set from the sign of the
        relationship at the chosen threshold.
        """
        if self.policy != "learned":
            raise ValueError("calibrate() is only meaningful for policy='learned'.")
        if "operational_distance" not in fold_table.columns or "delta_mae_flow_minus_direct" not in fold_table.columns:
            raise ValueError(
                "fold_table must contain 'operational_distance' and "
                "'delta_mae_flow_minus_direct'."
            )
        d = fold_table["operational_distance"].to_numpy().astype(np.float64)
        delta = fold_table["delta_mae_flow_minus_direct"].to_numpy().astype(np.float64)
        if d.size < 2:
            raise ValueError("Need at least two folds to calibrate a threshold.")

        # Candidate thresholds: midpoints between sorted unique distances.
        order = np.argsort(d)
        d_s, delta_s = d[order], delta[order]
        cand = (d_s[:-1] + d_s[1:]) / 2.0
        if cand.size == 0:
            cand = np.array([float(d_s.mean())])

        best_err = np.inf
        best_t = float(d_s.mean())
        best_dir = "flow_when_high"
        for t in cand:
            pred_flow = d_s >= t  # True => route Flow
            # Predicted better formulation MAE advantage: if pred_flow, use delta;
            # else use -delta (Direct better by delta).
            err = np.where(pred_flow, delta_s, -delta_s)
            total = float(np.sum(np.abs(err)))
            if total < best_err:
                best_err = total
                best_t = float(t)
                # Direction by mean delta among the two sides.
                hi = delta_s[d_s >= t]
                lo = delta_s[d_s < t]
                mean_hi = hi.mean() if hi.size else 0.0
                mean_lo = lo.mean() if lo.size else 0.0
                best_dir = "flow_when_high" if mean_hi < mean_lo else "direct_when_high"

        self.threshold = best_t
        self.direction = best_dir
        self.calibrated = True
        return self

    # -- scoring ----------------------------------------------------------- #
    def score(self, test_df: pl.DataFrame) -> dict[str, float]:
        """Per-unit operational shift vs the stored training reference."""
        if not self._ref:
            raise RuntimeError("Call fit_reference(train_df) before score().")
        if self.unit_col not in test_df.columns:
            raise ValueError(f"unit_col {self.unit_col!r} not present in test frame.")
        out: dict[str, float] = {}
        for unit, sub in test_df.group_by(self.unit_col):
            unit_key = unit[0] if isinstance(unit, tuple) else unit
            scores = []
            for f, ref in self._ref.items():
                if f not in sub.columns:
                    continue
                tb = sub[f].drop_nulls().to_numpy()
                if tb.size == 0:
                    continue
                scores.append(_feature_distance(tb, ref, self.method))
            out[str(unit_key)] = float(np.mean(scores)) if scores else 0.0
        return out

    # -- routing ----------------------------------------------------------- #
    def route(
        self,
        test_df: pl.DataFrame,
        allow_uncalibrated: bool = False,
        mae_direct: dict[str, float] | None = None,
        mae_flow: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Return ``{unit: "direct" | "flow"}`` decisions.

        * ``always_direct`` / ``always_flow`` always return that constant.
        * ``learned`` requires a prior :meth:`calibrate`; otherwise raises unless
          ``allow_uncalibrated`` is set (then it falls back to ``always_direct``
          and the caller is responsible for not trusting the output).
        * ``oracle`` requires ``mae_direct`` and ``mae_flow`` per unit (ground
          truth) and is an **upper bound only**.
        """
        if self.policy == "always_direct":
            return self._constant(test_df, "direct")
        if self.policy == "always_flow":
            return self._constant(test_df, "flow")

        if self.policy == "oracle":
            return self._route_oracle(test_df, mae_direct, mae_flow)

        # learned
        if not self.calibrated:
            if allow_uncalibrated:
                return self._constant(test_df, "direct")
            raise RuntimeError(
                "ShiftAwareRouter(policy='learned') is not calibrated. Call "
                "calibrate(fold_table) with operational-shift validation evidence "
                "before routing. (Project gate: no router before operational-shift "
                "evidence exists.)"
            )
        scores = self.score(test_df)
        out: dict[str, str] = {}
        for unit, s in scores.items():
            if self.direction == "flow_when_high":
                out[unit] = "flow" if s >= self.threshold else "direct"
            else:
                out[unit] = "direct" if s >= self.threshold else "flow"
        return out

    # -- helpers ----------------------------------------------------------- #
    def _constant(self, test_df: pl.DataFrame, choice: str) -> dict[str, str]:
        units = test_df[self.unit_col].unique().to_list()
        return {str(u): choice for u in units}

    def _route_oracle(
        self,
        test_df: pl.DataFrame,
        mae_direct: dict[str, float] | None,
        mae_flow: dict[str, float] | None,
    ) -> dict[str, str]:
        if mae_direct is None or mae_flow is None:
            raise ValueError("Oracle routing requires ground-truth mae_direct and mae_flow per unit (upper bound only).")
        units = [str(u) for u in test_df[self.unit_col].unique().to_list()]
        out: dict[str, str] = {}
        for u in units:
            if u not in mae_direct or u not in mae_flow:
                raise ValueError(f"Oracle missing MAE for unit {u!r}.")
            out[u] = "flow" if mae_flow[u] < mae_direct[u] else "direct"
        return out


# --------------------------------------------------------------------------- #
# Fold-level calibration table assembly
# --------------------------------------------------------------------------- #
def build_fold_table(
    loto_results: pl.DataFrame,
    shift_scores: dict[str, float],
    direct_col: str = "direct_mae",
    flow_col: str = "flow_mae",
    type_col: str = "aircraft_type",
) -> pl.DataFrame:
    """Merge per-type LOTO MAE with precomputed operational shift scores.

    Produces the validation fold table consumed by :meth:`ShiftAwareRouter.calibrate`:
    one row per held-out type with ``operational_distance`` and
    ``delta_mae_flow_minus_direct`` (= ``flow_mae`` − ``direct_mae``).
    """
    if direct_col not in loto_results.columns or flow_col not in loto_results.columns:
        raise ValueError(f"loto_results must contain {direct_col!r} and {flow_col!r}.")
    rows = []
    for r in loto_results.iter_rows(named=True):
        t = str(r[type_col])
        if t not in shift_scores:
            continue
        rows.append(
            {
                type_col: t,
                "operational_distance": shift_scores[t],
                "direct_mae": float(r[direct_col]),
                "flow_mae": float(r[flow_col]),
                "delta_mae_flow_minus_direct": float(r[flow_col]) - float(r[direct_col]),
            }
        )
    return pl.DataFrame(rows)
