"""Minimal pilot experiment suite for external dataset audit.

Experiments (AeroTwin_External_Dataset_Audit_Package §5):

* **A. Direct baseline** – predict ``actual_fuel_kg`` with base + physics features
* **B. Fuel-Flow target** – predict rate; recover kg via duration
* **C. Energy feature ablation** – Base vs Base+Energy (bootstrap ΔMAE)
* **D. Generalization** – flight-level split metrics (+ optional type summary)
* **E. Comparison table** – standardized CSV/figures under ``audit_results/``

Reuses ``physics.eval_framework`` and feature lists from the main pipeline.
Designed to run on small samples (demo data or a handful of real flights).

Example
-------
::

    python -m physics.external_audit.run_audit_pilot --source demo --max-flights 8
    python -m physics.external_audit.run_audit_pilot --parquet featured_dataset_audit.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    CATEGORICAL,
    evaluate,
    flight_level_split,
    plot_bootstrap_hist,
    project_root,
    significance_test,
    train_predict,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES  # noqa: E402

LOGGER = logging.getLogger(__name__)

RANDOM_STATE = 42
DEFAULT_MODEL = "lgbm"
# Lighter than full notebooks for pilot speed
N_BOOTSTRAP_PILOT = 2_000


def _avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    required = ["actual_fuel_kg", "duration_s", "flight_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Audit dataset missing required columns: {missing}")
    out = df.drop_nulls(subset=required).filter(
        pl.col("actual_fuel_kg").is_finite()
        & pl.col("duration_s").is_finite()
        & (pl.col("duration_s") > 0)
    )
    if "physics_fuel_kg" in out.columns:
        out = out.filter(pl.col("physics_fuel_kg").is_finite())
    return out


def _feature_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    """Feature groups aligned with PRC notebooks.

    Only columns listed in ``eval_framework.CATEGORICAL`` are treated as
    categorical by ``train_predict`` / sklearn pipelines. Do **not** append
    extra string columns (e.g. ``phase``) here — they would be fed to the
    median imputer and fail.
    """
    base = _avail(list(BASE_NUMERIC), df)
    energy = _avail(list(ENERGY_FEATURES), df)
    cats = _avail(list(CATEGORICAL), df)
    physics = ["physics_fuel_kg"] if "physics_fuel_kg" in df.columns else []

    return {
        "base": base + physics + cats,
        "base_energy": base + energy + physics + cats,
        "energy_only_extra": energy,  # for reporting which energy cols exist
    }


def _run_model(
    pdf,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    duration_test: np.ndarray,
    *,
    target: str = "direct",
    model_key: str = DEFAULT_MODEL,
    physics_test: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Train and return recovered fuel-kg predictions + metrics."""
    if not feature_cols:
        raise ValueError("No feature columns available for model")

    X_tr = pdf[feature_cols].iloc[train_idx]
    X_te = pdf[feature_cols].iloc[test_idx]
    dur_tr = pdf["duration_s"].to_numpy()[train_idx].astype(np.float64)
    dur_tr = np.clip(dur_tr, 1.0, None)
    dur_te = np.clip(duration_test.astype(np.float64), 1.0, None)

    if target == "fuel_flow":
        y_tr = (y_train / dur_tr).astype(np.float64)
    else:
        y_tr = y_train.astype(np.float64)

    # Prefer sklearn pipeline models; catboost path via eval_framework if requested
    try:
        pred_raw = train_predict(model_key, feature_cols, X_tr, X_te, y_tr)
    except Exception as exc:
        LOGGER.warning("Model %s failed (%s); falling back to lgbm", model_key, exc)
        pred_raw = train_predict("lgbm", feature_cols, X_tr, X_te, y_tr)

    if target == "fuel_flow":
        pred_fuel = pred_raw * dur_te
    else:
        pred_fuel = pred_raw

    mets = evaluate(y_test, pred_fuel)
    return pred_fuel.astype(np.float64), mets


def run_pilot_suite(
    df: pl.DataFrame,
    *,
    out_dir: Path,
    model_key: str = DEFAULT_MODEL,
    test_size: float = 0.25,
    n_bootstrap: int = N_BOOTSTRAP_PILOT,
) -> dict[str, Any]:
    """Execute experiments A–E and write tables/figures to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = _clean(df)
    n_flights = df["flight_id"].n_unique()
    if n_flights < 3:
        LOGGER.warning(
            "Only %d unique flights — pilot metrics will be unstable (need ≥4–5 for split)",
            n_flights,
        )

    label_note = ""
    if "label_is_physics_derived" in df.columns and df["label_is_physics_derived"].any():
        label_note = (
            "WARNING: some/all labels are physics-derived (OpenAP). "
            "Absolute MAE is not independent fuel error."
        )
        LOGGER.warning(label_note)

    feats = _feature_sets(df)
    pdf = df.to_pandas()
    # categorical as string for OHE (must match CATEGORICAL used by make_pipeline)
    for c in _avail(list(CATEGORICAL), df):
        pdf[c] = pdf[c].astype(str).fillna("missing")

    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy().astype(np.float64)
    duration = df["duration_s"].to_numpy().astype(np.float64)
    physics = (
        df["physics_fuel_kg"].to_numpy().astype(np.float64)
        if "physics_fuel_kg" in df.columns
        else None
    )

    train_idx, test_idx, train_fids, test_fids = flight_level_split(fids, test_size=test_size)
    LOGGER.info(
        "Flight split: %d train / %d test flights, %d / %d intervals",
        len(train_fids),
        len(test_fids),
        len(train_idx),
        len(test_idx),
    )

    y_tr, y_te = y[train_idx], y[test_idx]
    dur_te = duration[test_idx]
    fid_te = fids[test_idx]
    phys_te = physics[test_idx] if physics is not None else None

    results_rows: list[dict[str, Any]] = []
    preds: dict[str, np.ndarray] = {}

    # ----- A. Direct baseline (base + physics) -----
    LOGGER.info("Experiment A: Direct baseline")
    pred_a, met_a = _run_model(
        pdf,
        feats["base"],
        train_idx,
        test_idx,
        y_tr,
        y_te,
        dur_te,
        target="direct",
        model_key=model_key,
        physics_test=phys_te,
    )
    preds["direct_base"] = pred_a
    results_rows.append(
        {
            "experiment": "A_direct_baseline",
            "approach": "direct",
            "features": "base+physics",
            "target": "direct_fuel",
            **met_a,
            "n_features": len(feats["base"]),
        }
    )

    # Physics-only baseline on test
    if phys_te is not None:
        phys_mets = evaluate(y_te, phys_te)
        results_rows.append(
            {
                "experiment": "A_physics_only",
                "approach": "physics",
                "features": "openap",
                "target": "physics_fuel_kg",
                **phys_mets,
                "n_features": 0,
            }
        )

    # ----- B. Fuel-flow target -----
    LOGGER.info("Experiment B: Fuel-flow target")
    pred_b, met_b = _run_model(
        pdf,
        feats["base_energy"] if feats["base_energy"] else feats["base"],
        train_idx,
        test_idx,
        y_tr,
        y_te,
        dur_te,
        target="fuel_flow",
        model_key=model_key,
    )
    preds["flow_energy"] = pred_b
    results_rows.append(
        {
            "experiment": "B_fuel_flow",
            "approach": "fuel_flow",
            "features": "base+energy+physics",
            "target": "fuel_flow",
            **met_b,
            "n_features": len(feats["base_energy"] or feats["base"]),
        }
    )

    # Direct with same energy feature set for fair Direct vs Flow compare
    pred_direct_e, met_direct_e = _run_model(
        pdf,
        feats["base_energy"] if feats["base_energy"] else feats["base"],
        train_idx,
        test_idx,
        y_tr,
        y_te,
        dur_te,
        target="direct",
        model_key=model_key,
    )
    preds["direct_energy"] = pred_direct_e
    results_rows.append(
        {
            "experiment": "B_direct_energy_matched",
            "approach": "direct",
            "features": "base+energy+physics",
            "target": "direct_fuel",
            **met_direct_e,
            "n_features": len(feats["base_energy"] or feats["base"]),
        }
    )

    # ----- C. Energy ablation -----
    LOGGER.info("Experiment C: Energy feature ablation")
    pred_base_c, met_base_c = _run_model(
        pdf,
        feats["base"],
        train_idx,
        test_idx,
        y_tr,
        y_te,
        dur_te,
        target="direct",
        model_key=model_key,
    )
    pred_energy_c, met_energy_c = _run_model(
        pdf,
        feats["base_energy"] if feats["base_energy"] else feats["base"],
        train_idx,
        test_idx,
        y_tr,
        y_te,
        dur_te,
        target="direct",
        model_key=model_key,
    )
    preds["ablation_base"] = pred_base_c
    preds["ablation_energy"] = pred_energy_c
    results_rows.append(
        {
            "experiment": "C_energy_ablation_base",
            "approach": "direct",
            "features": "base+physics",
            "target": "direct_fuel",
            **met_base_c,
            "n_features": len(feats["base"]),
        }
    )
    results_rows.append(
        {
            "experiment": "C_energy_ablation_base_energy",
            "approach": "direct",
            "features": "base+energy+physics",
            "target": "direct_fuel",
            **met_energy_c,
            "n_features": len(feats["base_energy"] or feats["base"]),
        }
    )

    err_base = np.abs(pred_base_c - y_te)
    err_energy = np.abs(pred_energy_c - y_te)
    # Temporarily lower bootstrap iterations via monkey-patch of significance path
    from aerotwin.engine import eval_framework as ef

    old_n = ef.N_BOOTSTRAP
    ef.N_BOOTSTRAP = n_bootstrap
    try:
        sig_energy = significance_test(
            err_energy, err_base, fid_te, "Base+Energy", "Base"
        )
        sig_flow = significance_test(
            np.abs(pred_b - y_te),
            np.abs(pred_direct_e - y_te),
            fid_te,
            "Flow+Energy",
            "Direct+Energy",
        )
    finally:
        ef.N_BOOTSTRAP = old_n

    # ----- D. Generalization summary -----
    gen_rows = [
        {
            "split": "flight_level",
            "n_train_flights": int(len(train_fids)),
            "n_test_flights": int(len(test_fids)),
            "n_train_intervals": int(len(train_idx)),
            "n_test_intervals": int(len(test_idx)),
            "direct_base_mae": met_a["mae"],
            "direct_energy_mae": met_direct_e["mae"],
            "flow_energy_mae": met_b["mae"],
            "physics_mae": float(np.mean(np.abs(phys_te - y_te))) if phys_te is not None else None,
        }
    ]

    type_table = None
    if "aircraft_type" in df.columns:
        # Per-type test MAE for direct_energy
        type_rows = []
        types_te = df["aircraft_type"].to_numpy()[test_idx]
        for t in sorted(set(str(x) for x in types_te if x is not None)):
            mask = np.array([str(x) == t for x in types_te])
            if mask.sum() < 2:
                continue
            type_rows.append(
                {
                    "aircraft_type": t,
                    "n_test": int(mask.sum()),
                    "mae_direct_energy": float(
                        np.mean(np.abs(pred_direct_e[mask] - y_te[mask]))
                    ),
                    "mae_flow_energy": float(np.mean(np.abs(pred_b[mask] - y_te[mask]))),
                }
            )
        if type_rows:
            type_table = pl.DataFrame(type_rows)

    # ----- Write tables -----
    results_df = pl.DataFrame(results_rows)
    results_path = out_dir / "table_audit_pilot_metrics.csv"
    results_df.write_csv(results_path)

    sig_rows = [
        {
            "comparison": sig_energy["comparison"],
            "delta_mae": sig_energy["delta_mae"],
            "ci_lower": sig_energy["ci_lower"],
            "ci_upper": sig_energy["ci_upper"],
            "bootstrap_p": sig_energy["bootstrap_p"],
            "cohens_d": sig_energy["cohens_d"],
            "effect_size": sig_energy["effect_size"],
            "interpretation": sig_energy["interpretation"],
        },
        {
            "comparison": sig_flow["comparison"],
            "delta_mae": sig_flow["delta_mae"],
            "ci_lower": sig_flow["ci_lower"],
            "ci_upper": sig_flow["ci_upper"],
            "bootstrap_p": sig_flow["bootstrap_p"],
            "cohens_d": sig_flow["cohens_d"],
            "effect_size": sig_flow["effect_size"],
            "interpretation": sig_flow["interpretation"],
        },
    ]
    sig_df = pl.DataFrame(sig_rows)
    sig_df.write_csv(out_dir / "table_audit_pilot_significance.csv")

    gen_df = pl.DataFrame(gen_rows)
    gen_df.write_csv(out_dir / "table_audit_pilot_generalization.csv")

    if type_table is not None:
        type_table.write_csv(out_dir / "table_audit_pilot_per_type.csv")

    # Qualitative comparison template (Experiment E)
    comparison = pl.DataFrame(
        [
            {
                "finding": "Energy features improve Direct MAE",
                "prc2025_expected": "Yes (ΔMAE < 0 with CI)",
                "audit_result": (
                    "replicates"
                    if sig_energy["delta_mae"] < 0 and sig_energy["ci_upper"] < 0
                    else (
                        "partial"
                        if sig_energy["delta_mae"] < 0
                        else "fails"
                    )
                ),
                "delta_mae": sig_energy["delta_mae"],
                "interpretation": sig_energy["interpretation"],
            },
            {
                "finding": "Fuel-Flow target beats matched Direct",
                "prc2025_expected": "Often yes under LOTO / distribution shift",
                "audit_result": (
                    "replicates"
                    if sig_flow["delta_mae"] < 0 and sig_flow["ci_upper"] < 0
                    else ("partial" if sig_flow["delta_mae"] < 0 else "fails")
                ),
                "delta_mae": sig_flow["delta_mae"],
                "interpretation": sig_flow["interpretation"],
            },
            {
                "finding": "ML improves on physics-only",
                "prc2025_expected": "Yes on independent fuel labels",
                "audit_result": (
                    "n/a_physics_labels"
                    if label_note
                    else (
                        "replicates"
                        if phys_te is not None and met_direct_e["mae"] < float(np.mean(np.abs(phys_te - y_te)))
                        else "partial"
                    )
                ),
                "delta_mae": (
                    met_direct_e["mae"] - float(np.mean(np.abs(phys_te - y_te)))
                    if phys_te is not None
                    else None
                ),
                "interpretation": label_note or "Compare Direct MAE to physics MAE",
            },
        ]
    )
    comparison.write_csv(out_dir / "table_audit_qualitative_comparison.csv")

    # ----- Figures -----
    try:
        plot_bootstrap_hist(
            sig_energy["bootstrap_dist"],
            "Energy ablation ΔMAE (Base+Energy − Base)",
            fig_dir / "fig_audit_energy_ablation_bootstrap.png",
            color="steelblue",
        )
        plot_bootstrap_hist(
            sig_flow["bootstrap_dist"],
            "Flow vs Direct ΔMAE (Flow+Energy − Direct+Energy)",
            fig_dir / "fig_audit_flow_vs_direct_bootstrap.png",
            color="darkseagreen",
        )
        _plot_mae_bars(results_df, fig_dir / "fig_audit_pilot_mae.png")
        _plot_pred_scatter(y_te, pred_direct_e, pred_b, fig_dir / "fig_audit_pred_scatter.png")
    except Exception as exc:
        LOGGER.warning("Figure generation failed: %s", exc)

    meta = {
        "n_intervals": len(df),
        "n_flights": n_flights,
        "n_test_flights": int(len(test_fids)),
        "n_test_intervals": int(len(test_idx)),
        "model": model_key,
        "test_size": test_size,
        "n_bootstrap": n_bootstrap,
        "energy_features_used": feats["energy_only_extra"],
        "label_note": label_note,
        "dataset_source": (
            df["dataset_source"].unique().to_list()
            if "dataset_source" in df.columns
            else []
        ),
        "label_source": (
            df["label_source"].unique().to_list() if "label_source" in df.columns else []
        ),
    }
    (out_dir / "audit_pilot_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    LOGGER.info("Pilot suite complete → %s", out_dir)
    print("\n=== Audit Pilot Metrics ===")
    print(results_df)
    print("\n=== Significance ===")
    print(sig_df)
    print("\n=== Qualitative comparison ===")
    print(comparison)
    if label_note:
        print("\n" + label_note)

    return {
        "metrics": results_df,
        "significance": sig_df,
        "generalization": gen_df,
        "comparison": comparison,
        "meta": meta,
        "predictions": preds,
        "y_test": y_te,
        "flight_ids_test": fid_te,
    }


def _plot_mae_bars(results_df: pl.DataFrame, path: Path) -> None:
    pdf = results_df.to_pandas()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pdf))
    ax.bar(x, pdf["mae"], color="cornflowerblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(pdf["experiment"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("MAE [kg]")
    ax.set_title("External audit pilot — MAE by experiment")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_scatter(
    y_true: np.ndarray,
    pred_direct: np.ndarray,
    pred_flow: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title in zip(
        axes,
        [pred_direct, pred_flow],
        ["Direct + Energy", "Flow + Energy"],
    ):
        ax.scatter(y_true, pred, s=12, alpha=0.5, c="steelblue")
        lims = [
            min(y_true.min(), pred.min()),
            max(y_true.max(), pred.max()),
        ]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Actual fuel [kg]")
        ax.set_ylabel("Predicted fuel [kg]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Audit pilot predictions (test flights)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_or_build_dataset(args: argparse.Namespace) -> pl.DataFrame:
    """Load parquet or build from source for the pilot."""
    if args.parquet:
        path = Path(args.parquet)
        if not path.exists():
            raise SystemExit(f"Parquet not found: {path}")
        LOGGER.info("Loading %s", path)
        return pl.read_parquet(path)

    from aerotwin.validation.audit.build_featured_audit import (
        build_demo_featured,
        build_from_dashlink,
        build_from_opensky,
        write_featured_audit,
    )

    out_parq = Path(args.out_dir) / "featured_dataset_audit.parquet"
    if args.source == "demo":
        return build_demo_featured(n_flights=args.max_flights, out_path=out_parq)
    if args.source == "dashlink":
        if not args.dashlink_dir:
            raise SystemExit("--dashlink-dir required for --source dashlink")
        return build_from_dashlink(
            args.dashlink_dir,
            max_flights=args.max_flights,
            out_path=out_parq,
        )
    if args.source == "opensky":
        return build_from_opensky(
            args.start,
            args.stop,
            max_flights=args.max_flights,
            out_path=out_parq,
            icao24=args.icao24,
            synthetic_fallback=True,
        )
    raise SystemExit(f"Unknown source {args.source}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AeroTwin external dataset audit pilot experiments"
    )
    p.add_argument(
        "--source",
        choices=["demo", "dashlink", "opensky"],
        default="demo",
        help="Build dataset from this source (ignored if --parquet set)",
    )
    p.add_argument("--parquet", type=str, default=None, help="Use existing featured audit parquet")
    p.add_argument("--dashlink-dir", type=str, default=None)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--stop", default="2024-01-02")
    p.add_argument("--icao24", default=None)
    p.add_argument("--max-flights", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <project>/audit_results)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["lgbm", "xgb", "rf", "cat"])
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_PILOT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else project_root() / "audit_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_build_dataset(args)
    if df.is_empty():
        raise SystemExit("Empty dataset — cannot run pilot")

    # Persist a copy next to results for reproducibility
    cache = out_dir / "featured_dataset_audit.parquet"
    if not args.parquet or Path(args.parquet).resolve() != cache.resolve():
        try:
            df.write_parquet(cache)
        except Exception as exc:
            LOGGER.warning("Could not cache parquet: %s", exc)

    return run_pilot_suite(
        df,
        out_dir=out_dir,
        model_key=args.model,
        test_size=args.test_size,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
