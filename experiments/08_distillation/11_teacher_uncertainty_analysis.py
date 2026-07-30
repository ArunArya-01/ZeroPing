"""Phase 1A — Validate teacher ensemble disagreement as uncertainty signal.

Diagnostic only: no student training. Requires frozen teacher bundle
(cache/r3_teacher_distillation_bundle.pkl) with full_models for base predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.engine.gap_closing import (
    HEAVY_TYPES,
    NARROW_TYPES,
    aircraft_class,
    clean_featured,
    ensure_features,
    group_phase,
)
from aerotwin.engine.mass_model import enrich_mass_from_columns
from aerotwin.engine.official_benchmark import apply_bases
from aerotwin.engine.statistical_protocol import RANDOM_STATE, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("uncertainty")

BUNDLE = ROOT / "cache" / "r3_teacher_distillation_bundle.pkl"
LARGE_CKPT = ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt"
FINAL_PATH = ROOT / "featured_dataset_final.parquet"
OUT = ROOT / "results" / "distillation" / "uncertainty_analysis"
MIN_TYPE_N = 50
N_BOOT = 1000


def _body(ac: str) -> str:
    c = aircraft_class(str(ac))
    if c == "heavy":
        return "widebody_heavy"
    if c == "narrow":
        return "narrowbody"
    return "regional_other"


def _prepare(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "actual_fuel_kg" not in df.columns and "fuel_kg" in df.columns:
        df = df.with_columns(pl.col("fuel_kg").alias("actual_fuel_kg"))
    return enrich_mass_from_columns(clean_featured(df))


def _transform(df: pl.DataFrame, data: DistillationData) -> tuple[np.ndarray, np.ndarray]:
    feats = data.feature_cols
    numeric_cols = data.numeric_cols
    cat_cols = data.cat_cols
    df = ensure_features(df, feats)
    train_df = pl.read_parquet(data.parquet_path).filter(
        pl.col("ground_truth").is_finite()
        & pl.col("teacher_prediction").is_finite()
        & pl.col("flight_id").is_not_null()
    )
    train_num = np.column_stack(
        [
            train_df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64)
            for c in numeric_cols
        ]
    )
    medians = np.nanmedian(train_num[data.train_idx], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    num = np.column_stack(
        [df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64) for c in numeric_cols]
    )
    for j in range(num.shape[1]):
        bad = ~np.isfinite(num[:, j])
        if bad.any():
            col = num[:, j].copy()
            col[bad] = medians[j]
            num[:, j] = col
    x_num = data.scaler.transform(num).astype(np.float32)
    cat_pdf = df.select([pl.col(c).cast(pl.Utf8).fill_null("missing") for c in cat_cols]).to_pandas()
    x_cat = data.ohe.transform(cat_pdf).astype(np.float32)
    x = np.hstack([x_num, x_cat]).astype(np.float32)
    y = df["actual_fuel_kg"].to_numpy().astype(np.float64)
    return x, y


@torch.no_grad()
def _predict_large(x: np.ndarray, device: torch.device) -> np.ndarray:
    m = StudentMLP(x.shape[1], hidden_dims=(1792, 1024), dropout=0.1)
    blob = torch.load(LARGE_CKPT, map_location=device, weights_only=False)
    m.load_state_dict(blob["model_state_dict"])
    m.to(device).eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), 2048):
        out.append(m(xt[i : i + 2048].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _corr_with_ci(
    x: np.ndarray, y: np.ndarray, method: str = "pearson", n_boot: int = N_BOOT
) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return {"r": float("nan"), "p": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(len(x))}
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.kendalltau(x, y)
    rng = np.random.default_rng(RANDOM_STATE)
    boots = []
    n = len(x)
    n_boot_eff = n_boot if n >= 30 else min(n_boot, 400)
    for _ in range(n_boot_eff):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        try:
            if method == "pearson":
                rb, _ = stats.pearsonr(xb, yb)
            elif method == "spearman":
                rb, _ = stats.spearmanr(xb, yb)
            else:
                rb, _ = stats.kendalltau(xb, yb)
        except Exception:
            continue
        if np.isfinite(rb):
            boots.append(float(rb))
    if boots:
        lo, hi = bootstrap_ci(np.asarray(boots, dtype=np.float64))
    else:
        lo, hi = float("nan"), float("nan")
    return {
        "r": float(r) if np.isfinite(r) else float("nan"),
        "p": float(p) if np.isfinite(p) else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": int(n),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-featured", type=Path, default=FINAL_PATH)
    ap.add_argument("--bundle", type=Path, default=BUNDLE)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    if not args.bundle.exists():
        raise FileNotFoundError(
            f"Missing teacher bundle {args.bundle}. Rebuild with:\n"
            "  PYTHONPATH=src python experiments/08_distillation/01_build_teacher_distillation_dataset.py --train-only\n"
            "(restores frozen teacher cache; does not train students)"
        )
    if not args.final_featured.exists():
        raise FileNotFoundError(args.final_featured)
    if not LARGE_CKPT.exists():
        raise FileNotFoundError(LARGE_CKPT)

    out = Path(args.out)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    )
    t0 = time.time()

    LOGGER.info("Loading teacher bundle %s", args.bundle)
    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    feat_cols = list(bundle["feat_cols"])
    base_cols = list(bundle.get("base_pred_cols") or [f"base_{i}" for i in range(6)])

    LOGGER.info("Loading Final + student preprocessors")
    final_df = _prepare(args.final_featured)
    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    x_student, y = _transform(final_df, data)

    LOGGER.info("Teacher base ensemble inference (apply_bases)")
    sub = ensure_features(final_df, feat_cols)
    P = apply_bases(bundle["full_models"], sub, feat_cols)  # (n, 6)
    ridge = np.asarray(bundle["meta"].predict(P), dtype=np.float64)
    teacher = np.asarray(bundle["cal_phase"].transform(sub, ridge), dtype=np.float64)

    teacher_mean_bases = P.mean(axis=1)
    teacher_std = P.std(axis=1)
    teacher_var = P.var(axis=1)
    # CV with floor to avoid div by zero
    teacher_cv = teacher_std / np.maximum(np.abs(teacher_mean_bases), 1.0)

    LOGGER.info("Large MLP inference")
    large_pred = _predict_large(x_student, device)

    teacher_abs_err = np.abs(teacher - y)
    large_abs_err = np.abs(large_pred - y)
    residual_teacher = teacher - y
    residual_large = large_pred - y

    types = (
        final_df["aircraft_type"].cast(pl.Utf8).fill_null("unknown").to_numpy()
        if "aircraft_type" in final_df.columns
        else np.array(["unknown"] * len(y))
    )
    bodies = np.array([_body(t) for t in types])
    phases = group_phase(final_df).astype(str)
    duration = final_df["duration_s"].to_numpy().astype(np.float64) if "duration_s" in final_df.columns else np.full(len(y), np.nan)
    fids = final_df["flight_id"].cast(pl.Utf8).to_numpy() if "flight_id" in final_df.columns else np.array([str(i) for i in range(len(y))])
    interval_idx = final_df["interval_idx"].to_numpy() if "interval_idx" in final_df.columns else np.arange(len(y))

    # Training frequency of aircraft type (from distillation train)
    train_df = pl.read_parquet(ROOT / "distillation_dataset.parquet")
    if "aircraft_type" in train_df.columns:
        freq = train_df.group_by("aircraft_type").len().rename({"len": "train_n"})
        freq_map = {str(r["aircraft_type"]): int(r["train_n"]) for r in freq.iter_rows(named=True)}
    else:
        freq_map = {}
    train_freq = np.array([freq_map.get(str(t), 0) for t in types], dtype=np.float64)

    # Build parquet
    rec: dict[str, Any] = {
        "flight_id": fids.tolist(),
        "interval_idx": interval_idx,
        "aircraft_type": types.tolist(),
        "body_class": bodies.tolist(),
        "phase": phases.tolist(),
        "duration_s": duration,
        "train_type_n": train_freq,
        "ground_truth": y,
        "teacher_prediction": teacher,
        "teacher_ridge": ridge,
        "teacher_mean_bases": teacher_mean_bases,
        "teacher_std": teacher_std,
        "teacher_var": teacher_var,
        "teacher_cv": teacher_cv,
        "teacher_abs_error": teacher_abs_err,
        "teacher_residual": residual_teacher,
        "large_prediction": large_pred,
        "large_abs_error": large_abs_err,
        "large_residual": residual_large,
        "student_gap_abs": large_abs_err - teacher_abs_err,
    }
    for j, name in enumerate(base_cols[: P.shape[1]]):
        rec[str(name)] = P[:, j]
    tbl = pl.DataFrame(rec)
    tbl.write_parquet(out / "teacher_uncertainty.parquet")
    LOGGER.info("Wrote %s (%d rows)", out / "teacher_uncertainty.parquet", len(tbl))

    # ---- Correlations (overall / by body / type-weighted emphasis via per-type aggregates) ----
    corr_results: dict[str, Any] = {}
    for label, mask in [
        ("flight_all", np.ones(len(y), dtype=bool)),
        ("narrowbody", bodies == "narrowbody"),
        ("widebody_heavy", bodies == "widebody_heavy"),
    ]:
        if mask.sum() < 100:
            continue
        corr_results[label] = {
            "disagreement_vs_teacher_error": {
                "pearson": _corr_with_ci(teacher_std[mask], teacher_abs_err[mask], "pearson"),
                "spearman": _corr_with_ci(teacher_std[mask], teacher_abs_err[mask], "spearman"),
                "kendall": _corr_with_ci(teacher_std[mask], teacher_abs_err[mask], "kendall"),
            },
            "disagreement_vs_large_error": {
                "pearson": _corr_with_ci(teacher_std[mask], large_abs_err[mask], "pearson"),
                "spearman": _corr_with_ci(teacher_std[mask], large_abs_err[mask], "spearman"),
                "kendall": _corr_with_ci(teacher_std[mask], large_abs_err[mask], "kendall"),
            },
            "n": int(mask.sum()),
        }

    # Type-macro style: correlate type-mean disagreement with type RMSE
    type_rows = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < MIN_TYPE_N:
            continue
        type_rows.append(
            {
                "aircraft_type": t,
                "body_class": _body(t),
                "n": int(m.sum()),
                "mean_disagreement": float(np.mean(teacher_std[m])),
                "teacher_rmse": float(np.sqrt(np.mean((teacher[m] - y[m]) ** 2))),
                "large_rmse": float(np.sqrt(np.mean((large_pred[m] - y[m]) ** 2))),
                "student_gap": float(
                    np.sqrt(np.mean((large_pred[m] - y[m]) ** 2))
                    - np.sqrt(np.mean((teacher[m] - y[m]) ** 2))
                ),
                "train_n": float(freq_map.get(t, 0)),
                "mean_teacher_abs_err": float(np.mean(teacher_abs_err[m])),
                "mean_large_abs_err": float(np.mean(large_abs_err[m])),
            }
        )
    type_df = pl.DataFrame(type_rows)
    type_df.write_csv(out / "uncertainty_by_type.csv")

    if len(type_rows) >= 5:
        md = np.array([r["mean_disagreement"] for r in type_rows])
        tg = np.array([r["student_gap"] for r in type_rows])
        tr = np.array([r["teacher_rmse"] for r in type_rows])
        tn = np.array([r["train_n"] for r in type_rows])
        corr_results["type_level"] = {
            "disagreement_vs_teacher_rmse": {
                "pearson": _corr_with_ci(md, tr, "pearson"),
                "spearman": _corr_with_ci(md, tr, "spearman"),
            },
            "disagreement_vs_student_gap": {
                "pearson": _corr_with_ci(md, tg, "pearson"),
                "spearman": _corr_with_ci(md, tg, "spearman"),
            },
            "disagreement_vs_train_n": {
                "pearson": _corr_with_ci(md, tn, "pearson"),
                "spearman": _corr_with_ci(md, tn, "spearman"),
            },
            "train_n_vs_student_gap": {
                "pearson": _corr_with_ci(tn, tg, "pearson"),
                "spearman": _corr_with_ci(tn, tg, "spearman"),
            },
            "n_types": len(type_rows),
        }

    # ---- Calibration bins (equal-frequency) ----
    n_bins = 10
    order = np.argsort(teacher_std)
    bin_ids = np.array_split(order, n_bins)
    calib = []
    for bi, idxs in enumerate(bin_ids):
        if len(idxs) == 0:
            continue
        calib.append(
            {
                "bin": bi,
                "n": int(len(idxs)),
                "mean_disagreement": float(np.mean(teacher_std[idxs])),
                "mean_teacher_abs_error": float(np.mean(teacher_abs_err[idxs])),
                "mean_large_abs_error": float(np.mean(large_abs_err[idxs])),
                "teacher_rmse": float(np.sqrt(np.mean((teacher[idxs] - y[idxs]) ** 2))),
                "large_rmse": float(np.sqrt(np.mean((large_pred[idxs] - y[idxs]) ** 2))),
                "mean_student_gap_abs": float(np.mean(large_abs_err[idxs] - teacher_abs_err[idxs])),
            }
        )
    pl.DataFrame(calib).write_csv(out / "calibration_bins.csv")
    # monotonicity: spearman between bin mean disagreement and bin mean error
    calib_mono = {
        "spearman_disagreement_vs_teacher_err": float(
            stats.spearmanr([c["mean_disagreement"] for c in calib], [c["mean_teacher_abs_error"] for c in calib]).correlation
        ),
        "spearman_disagreement_vs_large_err": float(
            stats.spearmanr([c["mean_disagreement"] for c in calib], [c["mean_large_abs_error"] for c in calib]).correlation
        ),
    }

    # ---- Top/bottom 5% localization ----
    thr_hi = np.quantile(teacher_std, 0.95)
    thr_lo = np.quantile(teacher_std, 0.05)
    hi = teacher_std >= thr_hi
    lo = teacher_std <= thr_lo

    def _profile(mask: np.ndarray) -> dict[str, Any]:
        return {
            "n": int(mask.sum()),
            "mean_disagreement": float(np.mean(teacher_std[mask])),
            "mean_teacher_abs_err": float(np.mean(teacher_abs_err[mask])),
            "mean_large_abs_err": float(np.mean(large_abs_err[mask])),
            "teacher_rmse": float(np.sqrt(np.mean((teacher[mask] - y[mask]) ** 2))),
            "large_rmse": float(np.sqrt(np.mean((large_pred[mask] - y[mask]) ** 2))),
            "mean_duration_s": float(np.nanmean(duration[mask])),
            "mean_fuel_kg": float(np.mean(y[mask])),
            "body_counts": {str(b): int((bodies[mask] == b).sum()) for b in np.unique(bodies[mask])},
            "phase_counts": {str(p): int((phases[mask] == p).sum()) for p in np.unique(phases[mask])},
            "top_types": sorted(
                (
                    (str(t), int((types[mask] == t).sum()))
                    for t in np.unique(types[mask])
                ),
                key=lambda x: -x[1],
            )[:10],
        }

    localization = {
        "high_disagreement_top5pct": _profile(hi),
        "low_disagreement_bottom5pct": _profile(lo),
        "thresholds": {"p95": float(thr_hi), "p05": float(thr_lo)},
    }

    # ---- Robustness prediction: high-disagreement types vs student gap ----
    type_rows_sorted = sorted(type_rows, key=lambda r: -r["mean_disagreement"])
    n_top = max(3, len(type_rows_sorted) // 4)
    top_unc = type_rows_sorted[:n_top]
    bot_unc = type_rows_sorted[-n_top:]
    robustness_pred = {
        "top_disagreement_types": [r["aircraft_type"] for r in top_unc],
        "bottom_disagreement_types": [r["aircraft_type"] for r in bot_unc],
        "mean_gap_top_disagreement_types": float(np.mean([r["student_gap"] for r in top_unc])),
        "mean_gap_bottom_disagreement_types": float(np.mean([r["student_gap"] for r in bot_unc])),
        "mean_teacher_rmse_top": float(np.mean([r["teacher_rmse"] for r in top_unc])),
        "mean_teacher_rmse_bottom": float(np.mean([r["teacher_rmse"] for r in bot_unc])),
        "mean_large_rmse_top": float(np.mean([r["large_rmse"] for r in top_unc])),
        "mean_large_rmse_bottom": float(np.mean([r["large_rmse"] for r in bot_unc])),
    }
    robustness_pred["gap_delta_top_minus_bottom"] = (
        robustness_pred["mean_gap_top_disagreement_types"]
        - robustness_pred["mean_gap_bottom_disagreement_types"]
    )

    # ---- Decision ----
    sp_t = corr_results["flight_all"]["disagreement_vs_teacher_error"]["spearman"]["r"]
    sp_s = corr_results["flight_all"]["disagreement_vs_large_error"]["spearman"]["r"]
    pe_t = corr_results["flight_all"]["disagreement_vs_teacher_error"]["pearson"]["r"]
    pe_s = corr_results["flight_all"]["disagreement_vs_large_error"]["pearson"]["r"]
    type_gap_sp = (
        corr_results.get("type_level", {})
        .get("disagreement_vs_student_gap", {})
        .get("spearman", {})
        .get("r", float("nan"))
    )
    type_gap_pe = (
        corr_results.get("type_level", {})
        .get("disagreement_vs_student_gap", {})
        .get("pearson", {})
        .get("r", float("nan"))
    )

    # Criteria thresholds (pre-registered style)
    meaningful_corr = 0.15  # weak but positive; 0.3 moderate
    strong_enough = (
        sp_t >= meaningful_corr
        and sp_s >= meaningful_corr
        and calib_mono["spearman_disagreement_vs_teacher_err"] > 0.5
        and calib_mono["spearman_disagreement_vs_large_err"] > 0.5
        and (
            (np.isfinite(type_gap_sp) and type_gap_sp >= 0.2)
            or robustness_pred["gap_delta_top_minus_bottom"] > 2.0
        )
    )
    # Soft pass: positive correlations + calibration monotonicity even if type-level weak
    soft_pass = (
        sp_t > 0.1
        and sp_s > 0.1
        and calib_mono["spearman_disagreement_vs_teacher_err"] > 0.7
        and calib_mono["spearman_disagreement_vs_large_err"] > 0.7
    )
    proceed = bool(strong_enough or (soft_pass and robustness_pred["gap_delta_top_minus_bottom"] > 1.0))

    alternatives = []
    if not proceed:
        alternatives = [
            "Residual magnitude proxy from a cheap model",
            "Distance to training feature distribution (Mahalanobis / kNN)",
            "Epistemic uncertainty via MC-dropout on the student",
            "Phase / body-class hard routing without continuous disagreement",
            "Teacher–student absolute residual as pseudo-label difficulty (post-hoc only)",
        ]

    decision = {
        "proceed_to_adaptive_kd": proceed,
        "spearman_disagreement_vs_teacher_error": sp_t,
        "spearman_disagreement_vs_large_error": sp_s,
        "pearson_disagreement_vs_teacher_error": pe_t,
        "pearson_disagreement_vs_large_error": pe_s,
        "type_level_spearman_disagreement_vs_gap": type_gap_sp,
        "type_level_pearson_disagreement_vs_gap": type_gap_pe,
        "calibration_monotonicity": calib_mono,
        "robustness_pred": robustness_pred,
        "criteria": {
            "meaningful_corr_threshold": meaningful_corr,
            "strong_enough": strong_enough,
            "soft_pass": soft_pass,
        },
        "rationale": "",
        "alternative_signals": alternatives,
    }
    if proceed:
        decision["rationale"] = (
            f"Ensemble std correlates positively with teacher error (Spearman r={sp_t:.3f}) "
            f"and Large error (r={sp_s:.3f}); calibration bins are monotonic "
            f"(ρ_teacher={calib_mono['spearman_disagreement_vs_teacher_err']:.3f}, "
            f"ρ_large={calib_mono['spearman_disagreement_vs_large_err']:.3f}); "
            f"high-disagreement types show gap Δ={robustness_pred['gap_delta_top_minus_bottom']:+.2f} kg "
            f"vs low-disagreement types. Adaptive KD is justified."
        )
        decision["next"] = "Phase 1B — Adaptive Knowledge Distillation"
    else:
        decision["rationale"] = (
            f"Disagreement–error correlations (Spearman teacher={sp_t:.3f}, student={sp_s:.3f}) "
            f"and/or type-level gap link (ρ={type_gap_sp:.3f}, top−bottom gap Δ="
            f"{robustness_pred['gap_delta_top_minus_bottom']:+.2f}) are insufficient under "
            f"pre-set criteria. Do not implement Adaptive KD yet; try alternative signals."
        )
        decision["next"] = "Investigate alternative uncertainty signals (no Adaptive KD yet)"

    # Plots
    _plots(
        teacher_std,
        teacher_abs_err,
        large_abs_err,
        calib,
        type_rows,
        train_freq,
        plots,
        fig_dir,
    )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(y),
        "n_flights": int(len(np.unique(fids))),
        "teacher_bundle": str(args.bundle.resolve()),
        "disagreement_definition": "std of 6 base ensemble kg predictions (pre-meta)",
        "correlations": corr_results,
        "calibration_bins": calib,
        "calibration_monotonicity": calib_mono,
        "localization": localization,
        "robustness_prediction": robustness_pred,
        "decision": decision,
        "wall_seconds": time.time() - t0,
        "descriptives": {
            "teacher_std_mean": float(np.mean(teacher_std)),
            "teacher_std_p50": float(np.median(teacher_std)),
            "teacher_std_p95": float(np.percentile(teacher_std, 95)),
            "teacher_rmse": float(np.sqrt(np.mean((teacher - y) ** 2))),
            "large_rmse": float(np.sqrt(np.mean((large_pred - y) ** 2))),
        },
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    report = _report(summary)
    (out / "teacher_uncertainty_analysis.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "teacher_uncertainty_analysis.md").write_text(report, encoding="utf-8")

    print("\n=== PHASE 1A TEACHER UNCERTAINTY ===")
    print(f"  Spearman r (disagreement vs teacher err) = {sp_t:.3f}")
    print(f"  Spearman r (disagreement vs Large err)   = {sp_s:.3f}")
    print(f"  Type-level ρ (disagreement vs student gap)= {type_gap_sp:.3f}")
    print(f"  Calib mono teacher/large = {calib_mono}")
    print(f"  Top−bottom type gap Δ = {robustness_pred['gap_delta_top_minus_bottom']:+.2f} kg")
    print(f"  proceed_to_adaptive_kd = {proceed}")
    print(f"  next = {decision['next']}")
    print(f"  results = {out}")


def _plots(std, t_err, s_err, calib, type_rows, train_freq, plots, fig_dir):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_unc_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    # 1 histogram
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(std, bins=80, color="#4c72b0", alpha=0.85, density=True)
    ax.set_xlabel("Teacher ensemble std (kg)")
    ax.set_ylabel("Density")
    ax.set_title("Teacher disagreement histogram")
    save(fig, "disagreement_hist")

    # 2 scatter teacher err (subsample for plot)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(std), size=min(8000, len(std)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(std[idx], t_err[idx], s=4, alpha=0.2, rasterized=True)
    ax.set_xlabel("Teacher disagreement (std, kg)")
    ax.set_ylabel("Teacher |error| (kg)")
    ax.set_title("Disagreement vs teacher error")
    save(fig, "disagreement_vs_teacher_error")

    # 3 student error
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(std[idx], s_err[idx], s=4, alpha=0.2, c="#d62728", rasterized=True)
    ax.set_xlabel("Teacher disagreement (std, kg)")
    ax.set_ylabel("Large |error| (kg)")
    ax.set_title("Disagreement vs Large student error")
    save(fig, "disagreement_vs_student_error")

    # 4 calibration
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = [c["mean_disagreement"] for c in calib]
    ax.plot(xs, [c["mean_teacher_abs_error"] for c in calib], "o-", label="Teacher |err|")
    ax.plot(xs, [c["mean_large_abs_error"] for c in calib], "s-", label="Large |err|")
    ax.set_xlabel("Mean disagreement in bin (kg)")
    ax.set_ylabel("Mean absolute error (kg)")
    ax.set_title("Reliability: error vs disagreement bins")
    ax.legend()
    save(fig, "calibration_curve")

    # 5 aircraft scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    md = [r["mean_disagreement"] for r in type_rows]
    tr = [r["teacher_rmse"] for r in type_rows]
    lr = [r["large_rmse"] for r in type_rows]
    ax.scatter(md, tr, s=60, label="Teacher RMSE", c="#2ca02c")
    ax.scatter(md, lr, s=60, label="Large RMSE", c="#1f77b4")
    for r in type_rows:
        ax.annotate(r["aircraft_type"], (r["mean_disagreement"], r["large_rmse"]), fontsize=7, alpha=0.8)
    ax.set_xlabel("Mean teacher disagreement (kg)")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Aircraft-level disagreement vs RMSE")
    ax.legend()
    save(fig, "aircraft_disagreement_scatter")

    # 6 gap vs disagreement
    fig, ax = plt.subplots(figsize=(6.5, 5))
    gaps = [r["student_gap"] for r in type_rows]
    ax.scatter(md, gaps, s=70, c="#d62728")
    for r in type_rows:
        ax.annotate(r["aircraft_type"], (r["mean_disagreement"], r["student_gap"]), fontsize=7)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Mean teacher disagreement (kg)")
    ax.set_ylabel("Large − Teacher RMSE (kg)")
    ax.set_title("Student gap vs disagreement (by type)")
    save(fig, "gap_vs_disagreement")

    # 7 train freq vs disagreement
    fig, ax = plt.subplots(figsize=(6.5, 5))
    tn = [r["train_n"] for r in type_rows]
    ax.scatter(tn, md, s=70)
    for r in type_rows:
        ax.annotate(r["aircraft_type"], (r["train_n"], r["mean_disagreement"]), fontsize=7)
    ax.set_xlabel("Training type frequency (intervals)")
    ax.set_ylabel("Mean teacher disagreement (kg)")
    ax.set_title("Training frequency vs disagreement")
    ax.set_xscale("log")
    save(fig, "train_freq_vs_disagreement")

    # 8 top uncertain aircraft
    top = sorted(type_rows, key=lambda r: -r["mean_disagreement"])[:12]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["aircraft_type"] for r in top]
    x = np.arange(len(names))
    ax.bar(x, [r["mean_disagreement"] for r in top], color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean disagreement (kg)")
    ax.set_title("Top uncertain aircraft types")
    save(fig, "top_uncertain_aircraft")


def _report(s: dict[str, Any]) -> str:
    d = s["decision"]
    c = s["correlations"]["flight_all"]
    lines = [
        "# Phase 1A — Teacher Uncertainty Validation",
        "",
        f"**Date:** {s['timestamp_utc'][:10]}",
        "**Status:** Diagnostic only (no student training)",
        "",
        "## Objective",
        "",
        "Determine whether **teacher ensemble disagreement** (std of 6 base kg predictions) "
        "is a reliable indicator of prediction difficulty before implementing Adaptive KD.",
        "",
        "---",
        "",
        "## Methodology",
        "",
        "| Item | Value |",
        "|------|------|",
        f"| Samples | {s['n_samples']:,} Final intervals / {s['n_flights']:,} flights |",
        f"| Disagreement | Std of 6 base ensemble predictions (pre-Ridge/P1E) |",
        f"| Teacher error | \\|P1E teacher − ground truth\\| |",
        f"| Student error | \\|Large MLP − ground truth\\| |",
        f"| Bundle | `{s['teacher_bundle']}` |",
        "",
        "---",
        "",
        "## Descriptives",
        "",
        f"| Stat | Value |",
        f"|------|------:|",
        f"| Teacher RMSE | {s['descriptives']['teacher_rmse']:.2f} |",
        f"| Large RMSE | {s['descriptives']['large_rmse']:.2f} |",
        f"| Mean disagreement | {s['descriptives']['teacher_std_mean']:.2f} |",
        f"| Median disagreement | {s['descriptives']['teacher_std_p50']:.2f} |",
        f"| P95 disagreement | {s['descriptives']['teacher_std_p95']:.2f} |",
        "",
        "---",
        "",
        "## Correlation analysis (Flight / all Final)",
        "",
        "### Disagreement → teacher error",
        "",
        f"| Method | r | 95% CI | p |",
        f"|--------|--:|--------|--:|",
        f"| Pearson | {c['disagreement_vs_teacher_error']['pearson']['r']:.4f} | "
        f"[{c['disagreement_vs_teacher_error']['pearson']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_teacher_error']['pearson']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_teacher_error']['pearson']['p']:.2e} |",
        f"| Spearman | {c['disagreement_vs_teacher_error']['spearman']['r']:.4f} | "
        f"[{c['disagreement_vs_teacher_error']['spearman']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_teacher_error']['spearman']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_teacher_error']['spearman']['p']:.2e} |",
        f"| Kendall | {c['disagreement_vs_teacher_error']['kendall']['r']:.4f} | "
        f"[{c['disagreement_vs_teacher_error']['kendall']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_teacher_error']['kendall']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_teacher_error']['kendall']['p']:.2e} |",
        "",
        "### Disagreement → Large student error",
        "",
        f"| Method | r | 95% CI | p |",
        f"|--------|--:|--------|--:|",
        f"| Pearson | {c['disagreement_vs_large_error']['pearson']['r']:.4f} | "
        f"[{c['disagreement_vs_large_error']['pearson']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_large_error']['pearson']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_large_error']['pearson']['p']:.2e} |",
        f"| Spearman | {c['disagreement_vs_large_error']['spearman']['r']:.4f} | "
        f"[{c['disagreement_vs_large_error']['spearman']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_large_error']['spearman']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_large_error']['spearman']['p']:.2e} |",
        f"| Kendall | {c['disagreement_vs_large_error']['kendall']['r']:.4f} | "
        f"[{c['disagreement_vs_large_error']['kendall']['ci_lo']:.3f}, "
        f"{c['disagreement_vs_large_error']['kendall']['ci_hi']:.3f}] | "
        f"{c['disagreement_vs_large_error']['kendall']['p']:.2e} |",
        "",
    ]
    if "type_level" in s["correlations"]:
        tl = s["correlations"]["type_level"]
        lines += [
            "### Type-level (per-type means)",
            "",
            f"| Relation | Pearson r | Spearman r |",
            f"|----------|----------:|-----------:|",
            f"| Disagreement vs teacher RMSE | {tl['disagreement_vs_teacher_rmse']['pearson']['r']:.3f} | {tl['disagreement_vs_teacher_rmse']['spearman']['r']:.3f} |",
            f"| Disagreement vs student gap | {tl['disagreement_vs_student_gap']['pearson']['r']:.3f} | {tl['disagreement_vs_student_gap']['spearman']['r']:.3f} |",
            f"| Disagreement vs train frequency | {tl['disagreement_vs_train_n']['pearson']['r']:.3f} | {tl['disagreement_vs_train_n']['spearman']['r']:.3f} |",
            f"| Train frequency vs student gap | {tl['train_n_vs_student_gap']['pearson']['r']:.3f} | {tl['train_n_vs_student_gap']['spearman']['r']:.3f} |",
            "",
        ]
    rp = s["robustness_prediction"]
    loc = s["localization"]
    lines += [
        "---",
        "",
        "## Calibration (10 equal-frequency bins)",
        "",
        f"Bin-level Spearman(disagreement, teacher \\|err\\|): **{s['calibration_monotonicity']['spearman_disagreement_vs_teacher_err']:.3f}**",
        "",
        f"Bin-level Spearman(disagreement, Large \\|err\\|): **{s['calibration_monotonicity']['spearman_disagreement_vs_large_err']:.3f}**",
        "",
        "---",
        "",
        "## Error localization (top vs bottom 5% disagreement)",
        "",
        f"| Group | n | Mean std | Teacher RMSE | Large RMSE | Mean fuel | Mean duration |",
        f"|-------|--:|---------:|-------------:|-----------:|----------:|--------------:|",
        f"| High (top 5%) | {loc['high_disagreement_top5pct']['n']} | {loc['high_disagreement_top5pct']['mean_disagreement']:.1f} | "
        f"{loc['high_disagreement_top5pct']['teacher_rmse']:.1f} | {loc['high_disagreement_top5pct']['large_rmse']:.1f} | "
        f"{loc['high_disagreement_top5pct']['mean_fuel_kg']:.0f} | {loc['high_disagreement_top5pct']['mean_duration_s']:.0f} |",
        f"| Low (bottom 5%) | {loc['low_disagreement_bottom5pct']['n']} | {loc['low_disagreement_bottom5pct']['mean_disagreement']:.1f} | "
        f"{loc['low_disagreement_bottom5pct']['teacher_rmse']:.1f} | {loc['low_disagreement_bottom5pct']['large_rmse']:.1f} | "
        f"{loc['low_disagreement_bottom5pct']['mean_fuel_kg']:.0f} | {loc['low_disagreement_bottom5pct']['mean_duration_s']:.0f} |",
        "",
        f"High-disagreement body mix: `{loc['high_disagreement_top5pct']['body_counts']}`",
        "",
        f"High-disagreement phases: `{loc['high_disagreement_top5pct']['phase_counts']}`",
        "",
        f"High-disagreement top types: `{loc['high_disagreement_top5pct']['top_types']}`",
        "",
        "---",
        "",
        "## Robustness prediction (type-level)",
        "",
        f"| Group | Mean student gap | Mean teacher RMSE | Mean Large RMSE |",
        f"|-------|-----------------:|------------------:|----------------:|",
        f"| Top disagreement types | {rp['mean_gap_top_disagreement_types']:+.2f} | {rp['mean_teacher_rmse_top']:.1f} | {rp['mean_large_rmse_top']:.1f} |",
        f"| Bottom disagreement types | {rp['mean_gap_bottom_disagreement_types']:+.2f} | {rp['mean_teacher_rmse_bottom']:.1f} | {rp['mean_large_rmse_bottom']:.1f} |",
        f"| **Δ (top − bottom)** | **{rp['gap_delta_top_minus_bottom']:+.2f}** | | |",
        "",
        f"Top types: {rp['top_disagreement_types']}",
        "",
        f"Bottom types: {rp['bottom_disagreement_types']}",
        "",
        "---",
        "",
        "## Figures",
        "",
        "![hist](figures/fig_unc_disagreement_hist.png)",
        "",
        "![t_err](figures/fig_unc_disagreement_vs_teacher_error.png)",
        "",
        "![s_err](figures/fig_unc_disagreement_vs_student_error.png)",
        "",
        "![cal](figures/fig_unc_calibration_curve.png)",
        "",
        "![ac](figures/fig_unc_aircraft_disagreement_scatter.png)",
        "",
        "![gap](figures/fig_unc_gap_vs_disagreement.png)",
        "",
        "![freq](figures/fig_unc_train_freq_vs_disagreement.png)",
        "",
        "![top](figures/fig_unc_top_uncertain_aircraft.png)",
        "",
        "---",
        "",
        "## Decision questions (evidence only)",
        "",
        f"1. **Correlate with teacher error?** Spearman r = **{d['spearman_disagreement_vs_teacher_error']:.3f}**, "
        f"Pearson r = **{d['pearson_disagreement_vs_teacher_error']:.3f}**.",
        f"2. **Correlate with student error?** Spearman r = **{d['spearman_disagreement_vs_large_error']:.3f}**, "
        f"Pearson r = **{d['pearson_disagreement_vs_large_error']:.3f}**.",
        f"3. **Identify difficult aircraft?** Type-level analysis and top-uncertain list above.",
        f"4. **Predict robustness failures?** Top−bottom type gap Δ = **{rp['gap_delta_top_minus_bottom']:+.2f} kg**; "
        f"type-level Spearman(disagreement, gap) = **{d['type_level_spearman_disagreement_vs_gap']:.3f}**.",
        f"5. **Sufficient for Adaptive KD?** **{d['proceed_to_adaptive_kd']}**",
        "",
        "---",
        "",
        "## Recommendation",
        "",
        f"| Field | Value |",
        f"|-------|------|",
        f"| Proceed to Adaptive KD (1B)? | **{d['proceed_to_adaptive_kd']}** |",
        f"| Next | {d['next']} |",
        f"| Rationale | {d['rationale']} |",
        "",
    ]
    if d.get("alternative_signals"):
        lines += ["### Alternative signals to consider", ""]
        for a in d["alternative_signals"]:
            lines.append(f"- {a}")
        lines.append("")
    lines += [
        "---",
        "",
        "## Artifacts",
        "",
        "`results/distillation/uncertainty_analysis/`",
        "",
        f"*Generated {s['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
