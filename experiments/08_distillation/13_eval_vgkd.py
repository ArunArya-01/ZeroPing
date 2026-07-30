"""Evaluate all VGKD runs on Flight / Type-macro / Body-macro / Combined."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.vgkd import adaptive_beta, zscore
from aerotwin.engine.gap_closing import (
    HEAVY_TYPES,
    NARROW_TYPES,
    aircraft_class,
    clean_featured,
    ensure_features,
    rmse as kg_rmse,
)
from aerotwin.engine.mass_model import enrich_mass_from_columns
from aerotwin.engine.official_benchmark import apply_bases
from aerotwin.engine.statistical_protocol import RANDOM_STATE, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("vgkd_eval")

VGKD_ROOT = ROOT / "results" / "distillation" / "vgkd"
TEACHER_FINAL = 213.62
MIN_TYPE_N = 50
N_BOOT = 1000

# Fixed-KD Large reference (Phase 5)
FIXED_LARGE = {
    "final_rmse": 215.85,
    "combined_rmse": 225.95,
    "rank_rmse": 240.66,
    "n_params": 2_887_425,
    "cpu_ms": 0.26,
}


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
    feats, numeric_cols, cat_cols = data.feature_cols, data.numeric_cols, data.cat_cols
    df = ensure_features(df, feats)
    train_df = pl.read_parquet(data.parquet_path).filter(
        pl.col("ground_truth").is_finite()
        & pl.col("teacher_prediction").is_finite()
        & pl.col("flight_id").is_not_null()
    )
    train_num = np.column_stack(
        [train_df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64) for c in numeric_cols]
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
    y = df["actual_fuel_kg"].to_numpy().astype(np.float64)
    return np.hstack([x_num, x_cat]).astype(np.float32), y


@torch.no_grad()
def _predict(model: StudentMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), 2048):
        out.append(model(xt[i : i + 2048].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _full(y, p):
    m = regression_metrics(y, p)
    m["n"] = int(len(y))
    return m


def _type_macro(y, p, types, min_n=MIN_TYPE_N):
    rmses = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < min_n:
            continue
        rmses.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return float(np.mean(rmses)) if rmses else float("nan"), len(rmses)


def _body_macro(y, p, bodies, min_n=100):
    rmses = []
    for b in np.unique(bodies.astype(str)):
        m = bodies.astype(str) == b
        if m.sum() < min_n:
            continue
        rmses.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return float(np.mean(rmses)) if rmses else float("nan"), len(rmses)


def _boot_delta(y, p_new, p_base, fids, n_boot=N_BOOT):
    rng = np.random.default_rng(RANDOM_STATE)
    unique = np.unique(fids.astype(str))
    groups = {u: np.flatnonzero(fids.astype(str) == u) for u in unique}
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[u] for u in samp])
        boots.append(
            float(np.sqrt(np.mean((p_new[idx] - y[idx]) ** 2))
            - np.sqrt(np.mean((p_base[idx] - y[idx]) ** 2)))
        )
    lo, hi = bootstrap_ci(np.asarray(boots, dtype=np.float64))
    point = float(np.sqrt(np.mean((p_new - y) ** 2)) - np.sqrt(np.mean((p_base - y) ** 2)))
    return {"delta_rmse": point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": not (lo <= 0 <= hi)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    )

    out = VGKD_ROOT / "evaluation"
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = VGKD_ROOT / "runs"
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and (p / "best_model.pt").exists()])
    if not run_dirs:
        raise FileNotFoundError(f"No VGKD runs in {runs_dir}")

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    final_df = _prepare(ROOT / "featured_dataset_final.parquet")
    x_f, y_f = _transform(final_df, data)
    types = final_df["aircraft_type"].cast(pl.Utf8).fill_null("?").to_numpy()
    bodies = np.array([_body(t) for t in types])
    fids = final_df["flight_id"].cast(pl.Utf8).to_numpy()

    # Teacher Final preds
    bundle_path = ROOT / "cache" / "r3_teacher_distillation_bundle.pkl"
    if bundle_path.exists():
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        sub = ensure_features(final_df, list(bundle["feat_cols"]))
        P = apply_bases(bundle["full_models"], sub, list(bundle["feat_cols"]))
        ridge = np.asarray(bundle["meta"].predict(P), dtype=np.float64)
        y_teacher = np.asarray(bundle["cal_phase"].transform(sub, ridge), dtype=np.float64)
    else:
        audit = ROOT / "results/distillation/teacher_audit/teacher_predictions.parquet"
        y_teacher = pl.read_parquet(audit)["teacher_prediction"].to_numpy().astype(np.float64)

    teacher_final = _full(y_f, y_teacher)
    teacher_type, n_types = _type_macro(y_f, y_teacher, types)
    teacher_body, n_bodies = _body_macro(y_f, y_teacher, bodies)

    # Rank for Combined
    y_r = p_r_teacher = None
    rank_path = ROOT / "featured_dataset_rank.parquet"
    if rank_path.exists():
        rank_df = _prepare(rank_path)
        x_r, y_r = _transform(rank_df, data)
        if bundle_path.exists():
            sub_r = ensure_features(rank_df, list(bundle["feat_cols"]))
            Pr = apply_bases(bundle["full_models"], sub_r, list(bundle["feat_cols"]))
            rr = np.asarray(bundle["meta"].predict(Pr), dtype=np.float64)
            p_r_teacher = np.asarray(bundle["cal_phase"].transform(sub_r, rr), dtype=np.float64)

    # Fixed Large baseline preds (from checkpoint)
    large_fixed = StudentMLP(data.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
    blob_fixed = torch.load(
        ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt",
        map_location=device,
        weights_only=False,
    )
    large_fixed.load_state_dict(blob_fixed["model_state_dict"])
    large_fixed.to(device)
    p_fixed = _predict(large_fixed, x_f, device)
    fixed_metrics = {
        "run_name": "fixed_kd_large",
        "final": _full(y_f, p_fixed),
        "type_macro_rmse": _type_macro(y_f, p_fixed, types)[0],
        "body_macro_rmse": _body_macro(y_f, p_fixed, bodies)[0],
        "gap_flight": float(np.sqrt(np.mean((p_fixed - y_f) ** 2)) - teacher_final["rmse"]),
        "gap_type": _type_macro(y_f, p_fixed, types)[0] - teacher_type,
        "degradation_ratio_type": _type_macro(y_f, p_fixed, types)[0]
        / float(np.sqrt(np.mean((p_fixed - y_f) ** 2))),
    }
    if y_r is not None:
        p_fixed_r = _predict(large_fixed, x_r, device)
        y_c = np.concatenate([y_r, y_f])
        p_c = np.concatenate([p_fixed_r, p_fixed])
        fixed_metrics["combined_rmse"] = kg_rmse(y_c, p_c)
        fixed_metrics["rank_rmse"] = float(np.sqrt(np.mean((p_fixed_r - y_r) ** 2)))
    del large_fixed

    rows = []
    all_preds = {"fixed_kd_large": p_fixed, "teacher": y_teacher}

    for rd in run_dirs:
        name = rd.name
        LOGGER.info("Eval %s", name)
        ckpt = torch.load(rd / "best_model.pt", map_location=device, weights_only=False)
        model = StudentMLP(data.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        p = _predict(model, x_f, device)
        all_preds[name] = p
        fin = _full(y_f, p)
        tm, _ = _type_macro(y_f, p, types)
        bm, _ = _body_macro(y_f, p, bodies)
        rf = fin["rmse"]
        row = {
            "run_name": name,
            "final_rmse": rf,
            "final_mae": fin["mae"],
            "final_bias": fin["bias"],
            "final_r2": fin["r2"],
            "type_macro_rmse": tm,
            "body_macro_rmse": bm,
            "teacher_final_rmse": teacher_final["rmse"],
            "teacher_type_macro": teacher_type,
            "gap_flight": rf - teacher_final["rmse"],
            "gap_type": tm - teacher_type,
            "degradation_ratio_type": tm / rf if rf > 0 else float("nan"),
            "delta_final_vs_fixed": rf - fixed_metrics["final"]["rmse"],
            "delta_type_vs_fixed": tm - fixed_metrics["type_macro_rmse"],
            "n_params": int(ckpt.get("n_params") or 2_887_425),
        }
        vg = ckpt.get("vgkd_config") or {}
        row.update(
            {
                "lam": vg.get("lam"),
                "weight_fn": vg.get("weight_fn"),
                "static_beta": vg.get("static_beta"),
                "uncertainty_source": vg.get("uncertainty_source"),
            }
        )
        if y_r is not None:
            p_rank = _predict(model, x_r, device)
            y_c = np.concatenate([y_r, y_f])
            p_c = np.concatenate([p_rank, p])
            row["combined_rmse"] = kg_rmse(y_c, p_c)
            row["rank_rmse"] = float(np.sqrt(np.mean((p_rank - y_r) ** 2)))
            row["delta_combined_vs_fixed"] = row["combined_rmse"] - fixed_metrics.get(
                "combined_rmse", FIXED_LARGE["combined_rmse"]
            )
        # bootstrap vs fixed on Final
        boot = _boot_delta(y_f, p, p_fixed, fids)
        row["boot_delta_final"] = boot["delta_rmse"]
        row["boot_delta_final_ci_lo"] = boot["ci_lo"]
        row["boot_delta_final_ci_hi"] = boot["ci_hi"]
        row["boot_delta_final_sig"] = boot["excludes_zero"]
        # type-macro gap increase vs fixed
        row["gap_increase_type_vs_fixed"] = row["gap_type"] - fixed_metrics["gap_type"]
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Add fixed row
    fixed_row = {
        "run_name": "fixed_kd_large",
        "final_rmse": fixed_metrics["final"]["rmse"],
        "final_mae": fixed_metrics["final"]["mae"],
        "final_bias": fixed_metrics["final"]["bias"],
        "final_r2": fixed_metrics["final"]["r2"],
        "type_macro_rmse": fixed_metrics["type_macro_rmse"],
        "body_macro_rmse": fixed_metrics["body_macro_rmse"],
        "gap_flight": fixed_metrics["gap_flight"],
        "gap_type": fixed_metrics["gap_type"],
        "degradation_ratio_type": fixed_metrics["degradation_ratio_type"],
        "delta_final_vs_fixed": 0.0,
        "delta_type_vs_fixed": 0.0,
        "lam": 0.0,
        "weight_fn": "fixed",
        "static_beta": 0.9,
        "uncertainty_source": "none",
        "n_params": FIXED_LARGE["n_params"],
        "combined_rmse": fixed_metrics.get("combined_rmse"),
        "rank_rmse": fixed_metrics.get("rank_rmse"),
    }
    rows_all = [fixed_row] + rows
    pl.DataFrame(rows_all).write_csv(out / "comparison_table.csv")

    # Select preferred: minimize type_macro among those with final within +2 kg of fixed
    fixed_final = fixed_metrics["final"]["rmse"]
    candidates = [
        r
        for r in rows
        if r["final_rmse"] <= fixed_final + 2.0
        and r.get("uncertainty_source") == "ensemble_std"
        and r.get("static_beta") is None
    ]
    if not candidates:
        candidates = [r for r in rows if r.get("uncertainty_source") == "ensemble_std"]
    preferred = min(candidates, key=lambda r: (r["type_macro_rmse"], r["final_rmse"]))

    # β vs u curves for plotting
    u_grid = np.linspace(-2, 4, 200)
    beta_curves = {}
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        beta_curves[f"exp_lam{lam}"] = adaptive_beta(u_grid, beta_base=0.9, lam=lam, weight_fn="exp")
        if lam > 0:
            beta_curves[f"lin_lam{lam}"] = adaptive_beta(
                u_grid, beta_base=0.9, lam=lam, weight_fn="linear"
            )

    _plots(rows_all, preferred, beta_curves, u_grid, plots, fig_dir)

    blob = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "final_rmse": teacher_final["rmse"],
            "type_macro_rmse": teacher_type,
            "body_macro_rmse": teacher_body,
        },
        "fixed_kd_large": fixed_metrics,
        "runs": rows_all,
        "preferred": preferred,
        "selection_rule": "min type_macro among ensemble_std adaptive runs with Final RMSE <= fixed+2kg",
        "n_types_in_macro": n_types,
        "n_body_classes": n_bodies,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    (out / "preferred.json").write_text(json.dumps(preferred, indent=2, default=str), encoding="utf-8")

    report = _report(blob)
    (out / "vgkd_results.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "vgkd_results.md").write_text(report, encoding="utf-8")

    print("\n=== VGKD EVALUATION ===")
    print(
        f"  Fixed Large: Final={fixed_metrics['final']['rmse']:.2f} "
        f"type_macro={fixed_metrics['type_macro_rmse']:.2f} gap_type={fixed_metrics['gap_type']:+.2f}"
    )
    print(
        f"  Preferred: {preferred['run_name']} Final={preferred['final_rmse']:.2f} "
        f"type_macro={preferred['type_macro_rmse']:.2f} "
        f"Δtype={preferred['delta_type_vs_fixed']:+.2f} Δfinal={preferred['delta_final_vs_fixed']:+.2f}"
    )
    print(f"  results={out}")


def _plots(rows, preferred, beta_curves, u_grid, plots, fig_dir):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_vgkd_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    # 1 β vs u
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, beta in beta_curves.items():
        if name.startswith("exp"):
            ax.plot(u_grid, beta, label=name)
    ax.set_xlabel("u_norm (z-score, clipped ≥0 in formula)")
    ax.set_ylabel("β(x)")
    ax.set_title("Adaptive β vs normalized uncertainty (exponential)")
    ax.legend(fontsize=8)
    save(fig, "beta_vs_uncertainty")

    # 2 λ sensitivity — type macro and final
    exp_runs = [r for r in rows if r["run_name"].startswith("vgkd_exp_lam")]
    exp_runs = sorted(exp_runs, key=lambda r: float(r.get("lam") or 0))
    if exp_runs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lams = [float(r["lam"]) for r in exp_runs]
        ax.plot(lams, [r["final_rmse"] for r in exp_runs], "o-", label="Final RMSE")
        ax.plot(lams, [r["type_macro_rmse"] for r in exp_runs], "s-", label="Type-macro RMSE")
        ax.axhline(FIXED_LARGE["final_rmse"], color="gray", ls="--", label="Fixed Final")
        ax.set_xlabel("λ")
        ax.set_ylabel("RMSE (kg)")
        ax.set_title("λ sensitivity (exponential VGKD)")
        ax.legend()
        save(fig, "lambda_sensitivity")

    # 3 static vs adaptive
    static = [r for r in rows if str(r["run_name"]).startswith("static_beta")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r["run_name"] for r in static] + [preferred["run_name"], "fixed_kd_large"]
    # rebuild list carefully
    show = static + [preferred] + [r for r in rows if r["run_name"] == "fixed_kd_large"]
    # unique by name
    seen = set()
    show2 = []
    for r in show:
        if r["run_name"] not in seen:
            seen.add(r["run_name"])
            show2.append(r)
    x = np.arange(len(show2))
    ax.bar(x - 0.2, [r["final_rmse"] for r in show2], 0.4, label="Final")
    ax.bar(x + 0.2, [r["type_macro_rmse"] for r in show2], 0.4, label="Type-macro")
    ax.set_xticks(x)
    ax.set_xticklabels([r["run_name"] for r in show2], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Static β vs preferred adaptive")
    ax.legend()
    save(fig, "static_vs_adaptive")

    # 4 random vs true
    rand = [r for r in rows if "random" in r["run_name"]]
    true = [r for r in rows if r["run_name"] == "vgkd_exp_lam1.0"]
    if rand and true:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        labs = ["True u λ=1", "Random u λ=1", "Fixed KD"]
        finals = [true[0]["final_rmse"], rand[0]["final_rmse"], FIXED_LARGE["final_rmse"]]
        types = [true[0]["type_macro_rmse"], rand[0]["type_macro_rmse"],
                 next(r["type_macro_rmse"] for r in rows if r["run_name"] == "fixed_kd_large")]
        x = np.arange(3)
        ax.bar(x - 0.2, finals, 0.4, label="Final")
        ax.bar(x + 0.2, types, 0.4, label="Type-macro")
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel("RMSE (kg)")
        ax.set_title("True vs random uncertainty")
        ax.legend()
        save(fig, "random_vs_true")

    # 5 teacher-student gap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # top runs: fixed + exp sweep + preferred
    show = [r for r in rows if r["run_name"] in (
        "fixed_kd_large", "vgkd_exp_lam0.0", "vgkd_exp_lam0.25", "vgkd_exp_lam0.5",
        "vgkd_exp_lam1.0", "vgkd_exp_lam2.0", preferred["run_name"],
        "vgkd_random_lam1.0", "vgkd_oracle_lam1.0",
    )]
    # dedupe
    seen = set()
    show2 = []
    for r in show:
        if r["run_name"] not in seen:
            seen.add(r["run_name"])
            show2.append(r)
    x = np.arange(len(show2))
    ax.bar(x - 0.2, [r["gap_flight"] for r in show2], 0.4, label="Gap flight")
    ax.bar(x + 0.2, [r["gap_type"] for r in show2], 0.4, label="Gap type-macro")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r["run_name"] for r in show2], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Student − teacher RMSE (kg)")
    ax.set_title("Teacher–student gap")
    ax.legend()
    save(fig, "teacher_student_gap")

    # 6 robustness comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        if r["run_name"].startswith("vgkd_exp") or r["run_name"] == "fixed_kd_large":
            ax.scatter(r["final_rmse"], r["type_macro_rmse"], s=80)
            ax.annotate(r["run_name"].replace("vgkd_exp_", ""), (r["final_rmse"], r["type_macro_rmse"]), fontsize=7)
    ax.set_xlabel("Final RMSE (kg)")
    ax.set_ylabel("Type-macro RMSE (kg)")
    ax.set_title("Pareto: Final vs Type-macro")
    save(fig, "pareto_final_vs_type")

    # 7 linear vs exp at lam=1
    exp1 = next((r for r in rows if r["run_name"] == "vgkd_exp_lam1.0"), None)
    lin1 = next((r for r in rows if r["run_name"] == "vgkd_lin_lam1.0"), None)
    if exp1 and lin1:
        fig, ax = plt.subplots(figsize=(6, 4))
        labs = ["Exp λ=1", "Linear λ=1", "Fixed"]
        fin = [exp1["final_rmse"], lin1["final_rmse"], FIXED_LARGE["final_rmse"]]
        typ = [
            exp1["type_macro_rmse"],
            lin1["type_macro_rmse"],
            next(r["type_macro_rmse"] for r in rows if r["run_name"] == "fixed_kd_large"),
        ]
        x = np.arange(3)
        ax.bar(x - 0.2, fin, 0.4, label="Final")
        ax.bar(x + 0.2, typ, 0.4, label="Type-macro")
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.legend()
        ax.set_title("Linear vs exponential weight function")
        save(fig, "linear_vs_exp")


def _report(blob: dict[str, Any]) -> str:
    pref = blob["preferred"]
    fixed = blob["fixed_kd_large"]
    lines = [
        "# Phase 1B — Variance-Guided Knowledge Distillation (VGKD)",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "",
        "## Motivation",
        "",
        "Phase 0 showed Large MLP nearly matches the teacher on Flight Final but loses robustness under type-macro evaluation. "
        "Phase 1A showed teacher ensemble disagreement correlates with prediction error. "
        "VGKD uses that signal to reduce teacher weight on uncertain samples.",
        "",
        "## Method",
        "",
        "```",
        "u(x)  = std of 6 base ensemble predictions",
        "u_n   = (u − μ_train) / σ_train     # z-score",
        "β(x)  = β_base · exp(−λ · max(u_n, 0))",
        "α(x)  = 1 − β(x)",
        "L     = mean[ α(x)·(ŷ−y)² + β(x)·(ŷ−y_teacher)² ]",
        "```",
        "",
        "With β_base = 0.9. Samples with u ≤ train mean keep full teacher weight; uncertain samples shift toward GT.",
        "",
        "Architecture: **Large MLP** (~2.89M). No architecture change.",
        "",
        "---",
        "",
        "## Preferred model",
        "",
        f"| Field | Value |",
        f"|-------|------:|",
        f"| Run | `{pref['run_name']}` |",
        f"| Final RMSE | {pref['final_rmse']:.2f} |",
        f"| Type-macro RMSE | {pref['type_macro_rmse']:.2f} |",
        f"| Body-macro RMSE | {pref['body_macro_rmse']:.2f} |",
        f"| Gap flight | {pref['gap_flight']:+.2f} |",
        f"| Gap type-macro | {pref['gap_type']:+.2f} |",
        f"| Δ Final vs fixed | {pref['delta_final_vs_fixed']:+.2f} |",
        f"| Δ Type-macro vs fixed | {pref['delta_type_vs_fixed']:+.2f} |",
        f"| λ / weight_fn | {pref.get('lam')} / {pref.get('weight_fn')} |",
        "",
        "### Fixed KD baseline (Large)",
        "",
        f"- Final RMSE: **{fixed['final']['rmse']:.2f}**",
        f"- Type-macro: **{fixed['type_macro_rmse']:.2f}**",
        f"- Gap type-macro: **{fixed['gap_type']:+.2f}**",
        "",
        "---",
        "",
        "## Comparison table",
        "",
        "| Run | Final | Type-macro | Body-macro | Gap type | Δtype vs fixed | Δfinal vs fixed |",
        "|-----|------:|-----------:|-----------:|---------:|---------------:|----------------:|",
    ]
    for r in sorted(blob["runs"], key=lambda x: x.get("type_macro_rmse") or 1e9):
        lines.append(
            f"| {r['run_name']} | {r['final_rmse']:.2f} | {r['type_macro_rmse']:.2f} | "
            f"{r.get('body_macro_rmse', float('nan')):.2f} | {r['gap_type']:+.2f} | "
            f"{r.get('delta_type_vs_fixed', float('nan')):+.2f} | "
            f"{r.get('delta_final_vs_fixed', float('nan')):+.2f} |"
        )
    lines += [
        "",
        "---",
        "",
        "## Figures",
        "",
        "![beta](figures/fig_vgkd_beta_vs_uncertainty.png)",
        "",
        "![lam](figures/fig_vgkd_lambda_sensitivity.png)",
        "",
        "![static](figures/fig_vgkd_static_vs_adaptive.png)",
        "",
        "![rand](figures/fig_vgkd_random_vs_true.png)",
        "",
        "![gap](figures/fig_vgkd_teacher_student_gap.png)",
        "",
        "![pareto](figures/fig_vgkd_pareto_final_vs_type.png)",
        "",
        "![lin](figures/fig_vgkd_linear_vs_exp.png)",
        "",
        "---",
        "",
        "## Discussion",
        "",
        "Selection prioritizes **type-macro robustness** with bounded Final regression (≤2 kg).",
        "See comparison table for ablations (static β, random u, linear weights, oracle).",
        "",
        f"*Generated {blob['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
