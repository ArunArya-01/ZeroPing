"""Phase 3.5 — Final mechanism experiment: consistency regularization on Large MLP.

Single causal intervention on local prediction smoothness. No architecture change.
λ ∈ {0.01, 0.1, 1.0}. Select best by validation RMSE, then evaluate + geometry.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.distillation.runner import ExperimentConfig, KDWeightConfig, run_single_experiment
from aerotwin.distillation.trainer import set_seed
from aerotwin.engine.gap_closing import aircraft_class, clean_featured, ensure_features, group_phase
from aerotwin.engine.mass_model import enrich_mass_from_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("smoothness_causal")

OUT = ROOT / "results" / "distillation" / "smoothness_causal"
LAMBDAS = (0.01, 0.1, 1.0)
NOISE_SCALE = 0.015  # ~1.5% of standardized continuous feature std
MIN_TYPE_N = 50

LARGE_CKPT = ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt"
FT_CKPT = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt"
FT_CFG = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json"


def _body(ac: str) -> str:
    c = aircraft_class(str(ac))
    return {"heavy": "widebody_heavy", "narrow": "narrowbody"}.get(c, "regional_other")


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
def _encode_predict(model, x, device, bs=1024):
    model.eval()
    embs, preds = [], []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), bs):
        xb = xt[i : i + bs].to(device)
        embs.append(model.encode(xb).cpu().numpy())
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(embs), np.concatenate(preds).astype(np.float64)


def _type_macro(y, p, types, min_n=MIN_TYPE_N):
    vals = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < min_n:
            continue
        vals.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def _body_macro(y, p, bodies, min_n=100):
    vals = []
    for b in np.unique(bodies.astype(str)):
        m = bodies.astype(str) == b
        if m.sum() < min_n:
            continue
        vals.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def _load_large(in_dim: int, ckpt: Path, device) -> StudentMLP:
    m = StudentMLP(in_dim, hidden_dims=(1792, 1024), dropout=0.1)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(blob["model_state_dict"])
    return m.to(device).eval()


def _load_ft(data: DistillationData, device):
    sc = StudentConfig.from_mapping(json.loads(FT_CFG.read_text(encoding="utf-8")))
    sc.in_dim = data.in_dim
    sc.n_num_features = len(data.numeric_cols)
    sc.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    ft = build_student(sc, in_dim=data.in_dim)
    ft.load_state_dict(torch.load(FT_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    return ft.to(device).eval()


def _stability(model, x, n_num, device, eps=0.05, n=1500):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    xb = x[idx].copy()
    noise = rng.normal(0, eps, size=xb.shape).astype(np.float32)
    # Prefer continuous-only noise for fairness with training intervention
    noise[:, n_num:] = 0.0
    x2 = xb + noise
    with torch.no_grad():
        e1 = model.encode(torch.as_tensor(xb, device=device)).cpu().numpy()
        e2 = model.encode(torch.as_tensor(x2, device=device)).cpu().numpy()
        p1 = model(torch.as_tensor(xb, device=device)).cpu().numpy()
        p2 = model(torch.as_tensor(x2, device=device)).cpu().numpy()
    move = np.linalg.norm(e1 - e2, axis=1)
    scale = np.linalg.norm(e1, axis=1) + 1e-9
    return {
        "mean_abs_move": float(np.mean(move)),
        "mean_rel_move": float(np.mean(move / scale)),
        "median_rel_move": float(np.median(move / scale)),
        "mean_pred_abs_delta": float(np.mean(np.abs(p1 - p2))),
        "eps": eps,
        "n": int(len(idx)),
        "noise_on": "continuous_only",
    }


def _geometry(z, types, rare, freq_map):
    cents = {}
    within = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < 10:
            continue
        c = z[m].mean(axis=0)
        cents[t] = c
        within.append(float(np.mean(np.linalg.norm(z[m] - c, axis=1))))
    keys = list(cents.keys())
    inter = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter.append(float(np.linalg.norm(cents[keys[i]] - cents[keys[j]])))
    common = [t for t, n in sorted(freq_map.items(), key=lambda kv: -kv[1])[:5] if t in cents]
    if rare.any() and common:
        C = np.stack([cents[t] for t in common if t in cents], axis=0)
        d = np.linalg.norm(z[rare][:, None, :] - C[None, :, :], axis=-1).min(axis=1)
        rare_common = float(np.mean(d))
    else:
        rare_common = float("nan")
    mid = float(np.mean(inter)) if inter else float("nan")
    # silhouette on subsample
    rng = np.random.default_rng(42)
    n = min(5000, len(z))
    idx = rng.choice(len(z), size=n, replace=False)
    le_types = types[idx].astype(str)
    # need ≥2 samples per label for silhouette
    uniq, counts = np.unique(le_types, return_counts=True)
    valid = set(uniq[counts >= 2])
    mask = np.array([t in valid for t in le_types])
    sil = float("nan")
    if mask.sum() >= 100 and len(valid) >= 2:
        try:
            sil = float(silhouette_score(z[idx][mask], le_types[mask], metric="euclidean"))
        except Exception:
            sil = float("nan")
    # neighborhood purity
    nn = NearestNeighbors(n_neighbors=11).fit(z[idx])
    _, ind = nn.kneighbors(z[idx])
    purity = float(np.mean([np.mean(le_types[ind[i, 1:]] == le_types[i]) for i in range(n)]))
    return {
        "mean_within_type": float(np.mean(within)) if within else float("nan"),
        "mean_inter_type_centroid": mid,
        "rare_to_common_raw": rare_common,
        "rare_to_common_norm": rare_common / mid if mid and mid > 0 else float("nan"),
        "silhouette": sil,
        "type_purity_k10": purity,
        "n_types_centroid": len(cents),
    }


def train_sweep(data: DistillationData, device_str: str, force: bool) -> list[dict[str, Any]]:
    OUT.mkdir(parents=True, exist_ok=True)
    weight = KDWeightConfig(name="kd1", alpha=0.1, beta=0.9)
    summary = []
    for lam in LAMBDAS:
        run_name = f"large_cons_lam{lam}"
        out_dir = OUT / "runs" / run_name
        if (out_dir / "best_model.pt").exists() and not force:
            LOGGER.info("Skip existing %s", run_name)
            m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8")) if (out_dir / "metrics.json").exists() else {}
            summary.append(
                {
                    "run": run_name,
                    "lambda": lam,
                    "best_val_rmse": m.get("best_val_rmse"),
                    "skipped": True,
                }
            )
            continue

        def factory(in_dim: int):
            return StudentMLP(in_dim, hidden_dims=(1792, 1024), dropout=0.1)

        exp = ExperimentConfig(
            seed=42,
            val_fraction=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=2048,
            max_epochs=80,
            patience=12,
            min_delta=0.05,
            device=device_str,
            extras={
                "consistency_lambda": lam,
                "consistency_noise_scale": NOISE_SCALE,
                "n_num_features": len(data.numeric_cols),
                "intervention": "prediction_consistency",
                "architecture": "large_mlp",
            },
        )
        LOGGER.info("=== Train %s λ=%.3f n_num=%d ===", run_name, lam, len(data.numeric_cols))
        t0 = time.time()
        metrics = run_single_experiment(
            data=data,
            model_factory=factory,
            weight=weight,
            exp=exp,
            out_dir=out_dir,
            log_dir=OUT / "logs" / run_name,
            model_dir=OUT / "models" / run_name,
        )
        metrics["wall_seconds"] = time.time() - t0
        metrics["consistency_lambda"] = lam
        metrics["consistency_noise_scale"] = NOISE_SCALE
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        summary.append(
            {
                "run": run_name,
                "lambda": lam,
                "best_val_rmse": metrics.get("best_val_rmse"),
                "best_epoch": metrics.get("best_epoch"),
                "n_params": metrics.get("n_params"),
                "wall_seconds": metrics["wall_seconds"],
            }
        )
        LOGGER.info("%s done val_rmse=%.2f", run_name, metrics.get("best_val_rmse", float("nan")))
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def analyze(data: DistillationData, device, train_summary: list[dict]) -> dict[str, Any]:
    final = _prepare(ROOT / "featured_dataset_final.parquet")
    x, y = _transform(final, data)
    n_num = len(data.numeric_cols)
    types = final["aircraft_type"].cast(pl.Utf8).fill_null("?").to_numpy()
    bodies = np.array([_body(t) for t in types])
    train_df = pl.read_parquet(ROOT / "distillation_dataset.parquet")
    freq_map = {
        str(r["aircraft_type"]): int(r["len"])
        for r in train_df.group_by("aircraft_type").len().iter_rows(named=True)
    }
    train_n = np.array([freq_map.get(str(t), 0) for t in types], dtype=np.float64)
    thr = float(np.percentile(list({t: freq_map.get(str(t), 0) for t in np.unique(types)}.values()), 33))
    rare = train_n <= thr

    # Baselines
    LOGGER.info("Load baselines Large + FT")
    large = _load_large(data.in_dim, LARGE_CKPT, device)
    ft = _load_ft(data, device)
    emb_l, pred_l = _encode_predict(large, x, device)
    emb_f, pred_f = _encode_predict(ft, x, device)

    def pack_metrics(pred):
        return {
            "final_rmse": regression_metrics(y, pred)["rmse"],
            "final_mae": regression_metrics(y, pred)["mae"],
            "type_macro": _type_macro(y, pred, types)[0],
            "body_macro": _body_macro(y, pred, bodies)[0],
        }

    results = {
        "large_baseline": pack_metrics(pred_l),
        "ft_baseline": pack_metrics(pred_f),
        "lambda_runs": {},
    }
    results["large_baseline"]["val_rmse"] = None  # from frozen deploy
    # attach val from capacity metrics if present
    large_m_path = LARGE_CKPT.parent / "metrics.json"
    if large_m_path.exists():
        lm = json.loads(large_m_path.read_text(encoding="utf-8"))
        results["large_baseline"]["val_rmse"] = lm.get("best_val_rmse") or lm.get("val", {}).get("student", {}).get("rmse")

    z_l = StandardScaler().fit_transform(emb_l)
    z_f = StandardScaler().fit_transform(emb_f)
    geom = {
        "large": _geometry(z_l, types, rare, freq_map),
        "ft": _geometry(z_f, types, rare, freq_map),
    }
    stab = {
        "large": _stability(large, x, n_num, device),
        "ft": _stability(ft, x, n_num, device),
    }

    # Consistency runs
    best_lam = None
    best_val = float("inf")
    for s in train_summary:
        lam = s["lambda"]
        run = s["run"]
        ckpt = OUT / "runs" / run / "best_model.pt"
        if not ckpt.exists():
            continue
        m = _load_large(data.in_dim, ckpt, device)
        emb, pred = _encode_predict(m, x, device)
        met = pack_metrics(pred)
        # val from training metrics
        tm = json.loads((OUT / "runs" / run / "metrics.json").read_text(encoding="utf-8"))
        met["val_rmse"] = tm.get("best_val_rmse")
        met["best_epoch"] = tm.get("best_epoch")
        met["lambda"] = lam
        z = StandardScaler().fit_transform(emb)
        met["geometry"] = _geometry(z, types, rare, freq_map)
        met["stability"] = _stability(m, x, n_num, device)
        results["lambda_runs"][str(lam)] = met
        if met["val_rmse"] is not None and met["val_rmse"] < best_val:
            best_val = met["val_rmse"]
            best_lam = lam
        del m

    results["selected_lambda"] = best_lam
    results["selected_by"] = "best_validation_rmse"
    results["geometry_baselines"] = geom
    results["stability_baselines"] = stab

    # Outcome classification
    if best_lam is None:
        outcome = "C"
        interpretation = "No trained consistency models found."
        phase4 = "Do not draw causal conclusions; re-run training."
    else:
        sel = results["lambda_runs"][str(best_lam)]
        large_type = results["large_baseline"]["type_macro"]
        ft_type = results["ft_baseline"]["type_macro"]
        sel_type = sel["type_macro"]
        large_final = results["large_baseline"]["final_rmse"]
        sel_final = sel["final_rmse"]

        smooth_up = sel["stability"]["mean_rel_move"] < stab["large"]["mean_rel_move"] * 0.85
        geom_ftlike = (
            sel["geometry"]["rare_to_common_norm"] < geom["large"]["rare_to_common_norm"] * 0.9
            or abs(sel["geometry"]["rare_to_common_norm"] - geom["ft"]["rare_to_common_norm"])
            < abs(geom["large"]["rare_to_common_norm"] - geom["ft"]["rare_to_common_norm"]) * 0.7
        )
        # Robustness improvement: type-macro decreases by ≥ 3 kg (meaningful) toward FT
        type_improve = (large_type - sel_type) >= 3.0
        type_near_ft = sel_type <= large_type - 0.5 * max(large_type - ft_type, 1.0)
        iid_changed = abs(sel_final - large_final) >= 2.0
        type_flat = abs(sel_type - large_type) < 3.0

        if not smooth_up:
            outcome = "C"
            interpretation = (
                "Consistency regularization did not meaningfully increase embedding smoothness "
                f"(rel_move {sel['stability']['mean_rel_move']:.4f} vs Large {stab['large']['mean_rel_move']:.4f}). "
                "Causal conclusions about smoothness are not supported."
            )
            phase4 = "Do not draw causal conclusions about smoothness. Document unresolved mechanism."
        elif type_improve or type_near_ft:
            outcome = "A"
            interpretation = (
                f"Smoothness increased (rel_move {sel['stability']['mean_rel_move']:.4f} vs "
                f"{stab['large']['mean_rel_move']:.4f}); type-macro improved "
                f"({large_type:.1f} → {sel_type:.1f}). Local smoothness is a plausible causal mechanism."
            )
            phase4 = "Phase 4: smoothness-aware training method (not representation distillation)."
        elif smooth_up and type_flat and not iid_changed:
            outcome = "B"
            interpretation = (
                f"Smoothness increased (rel_move {sel['stability']['mean_rel_move']:.4f}) but type-macro "
                f"did not improve meaningfully ({large_type:.1f} → {sel_type:.1f}). "
                "Smoothness is not sufficient for the robustness advantage."
            )
            phase4 = (
                "Reject smoothness as primary causal mechanism. Do not pursue representation distillation. "
                "Document architecture-dependent robustness reversal as unresolved mechanism."
            )
        elif smooth_up and iid_changed and type_flat:
            outcome = "D"
            interpretation = (
                f"Smoothness increased; Final changed ({large_final:.1f} → {sel_final:.1f}) but type-macro "
                f"unchanged ({large_type:.1f} → {sel_type:.1f}). Smoothness affects optimization/IID, not transfer."
            )
            phase4 = (
                "Reject smoothness as principal explanation of type-macro robustness. "
                "Do not pursue representation distillation."
            )
        else:
            # smooth up, type not improved enough, geometry maybe mixed
            outcome = "B"
            interpretation = (
                f"Smoothness increased (rel_move {sel['stability']['mean_rel_move']:.4f}) but type-macro "
                f"({large_type:.1f} → {sel_type:.1f}) did not show a clear robustness gain. "
                "Treat as Outcome B (smoothness not sufficient)."
            )
            phase4 = (
                "Reject smoothness as primary causal mechanism. Do not pursue representation distillation. "
                "Write empirical paper documenting tested/ruled-out mechanisms."
            )

        results["selected_metrics"] = {
            "lambda": best_lam,
            "val_rmse": sel["val_rmse"],
            "final_rmse": sel_final,
            "type_macro": sel_type,
            "body_macro": sel["body_macro"],
            "delta_type_vs_large": sel_type - large_type,
            "delta_final_vs_large": sel_final - large_final,
            "smoothness_increased": smooth_up,
            "geometry_more_ftlike": geom_ftlike,
            "rel_move": sel["stability"]["mean_rel_move"],
            "rare_to_common_norm": sel["geometry"]["rare_to_common_norm"],
        }

    results["outcome"] = outcome
    results["interpretation"] = interpretation
    results["phase4_recommendation"] = phase4
    results["decision_gate"] = {
        "outcome": outcome,
        "pursue_smoothness_method": outcome == "A",
        "pursue_representation_distillation": False,  # only if A would have been smoothness; gate says no rep distill if fail
        "mechanism_resolved": outcome == "A",
        "next_step": (
            "Develop smoothness-aware method"
            if outcome == "A"
            else "Write empirical paper; mechanism remains unresolved (smoothness rejected or intervention failed)"
        ),
    }
    return results


def _plots(results: dict, plots: Path, fig_dir: Path):
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_p35_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    # Benchmark bars
    labels = ["Large", "FT"]
    finals = [results["large_baseline"]["final_rmse"], results["ft_baseline"]["final_rmse"]]
    types = [results["large_baseline"]["type_macro"], results["ft_baseline"]["type_macro"]]
    bodies = [results["large_baseline"]["body_macro"], results["ft_baseline"]["body_macro"]]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        labels.append(f"Cons λ={lam}")
        finals.append(r["final_rmse"])
        types.append(r["type_macro"])
        bodies.append(r["body_macro"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    ax.bar(x - w, finals, w, label="Final RMSE")
    ax.bar(x, types, w, label="Type-macro")
    ax.bar(x + w, bodies, w, label="Body-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Phase 3.5 — Consistency regularization vs baselines")
    ax.legend()
    save(fig, "benchmark")

    # Smoothness
    names = ["Large", "FT"]
    rel = [
        results["stability_baselines"]["large"]["mean_rel_move"],
        results["stability_baselines"]["ft"]["mean_rel_move"],
    ]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        names.append(f"λ={lam}")
        rel.append(r["stability"]["mean_rel_move"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, rel)
    ax.set_ylabel("Mean relative embedding movement")
    ax.set_title("Representation smoothness (ε=0.05 continuous noise)")
    plt.xticks(rotation=15, ha="right")
    save(fig, "smoothness")

    # Geometry rare→common norm
    names = ["Large", "FT"]
    rc = [
        results["geometry_baselines"]["large"]["rare_to_common_norm"],
        results["geometry_baselines"]["ft"]["rare_to_common_norm"],
    ]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        names.append(f"λ={lam}")
        rc.append(r["geometry"]["rare_to_common_norm"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, rc)
    ax.set_ylabel("Rare→common centroid (normalized)")
    ax.set_title("Geometry: rare→common proximity")
    plt.xticks(rotation=15, ha="right")
    save(fig, "geometry_rare_common")


def _report(results: dict, train_summary: list) -> str:
    sel = results.get("selected_lambda")
    sm = results.get("selected_metrics") or {}
    lines = [
        "# Phase 3.5 — Final Mechanism Experiment (Causal Validation)",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "**Status:** Complete — single causal intervention on local smoothness",
        "**Stopping rule:** No further mechanism hypotheses after this experiment",
        "",
        "---",
        "",
        "## 1. Motivation",
        "",
        "Prior phases established:",
        "",
        "- Large MLP wins Flight Holdout; FT wins Type-Macro (ranking reversal).",
        "- Teacher uncertainty (VGKD) **rejected** as a robustness lever.",
        "- Physics-feature reliance **rejected** by targeted ablation (Phase 3).",
        "- Representation geometry **partially supported** but geometry ↛ FT *advantage* (ρ≈0.09).",
        "",
        "FT embeddings are ~4× smoother under input noise. This experiment tests whether that "
        "smoothness is **causal** for type-macro robustness by inducing smoothness in Large MLP.",
        "",
        "---",
        "",
        "## 2. Scientific hypothesis",
        "",
        "> If local smoothness is causal for FT’s robustness advantage, then training Large MLP "
        "with prediction consistency regularization will improve Type-Macro RMSE.",
        "",
        "If Type-Macro does not improve despite increased smoothness → smoothness is not sufficient.",
        "",
        "---",
        "",
        "## 3. Experimental setup",
        "",
        "| Held fixed | Value |",
        "|------------|-------|",
        "| Architecture | Large MLP (1792, 1024), dropout 0.1 |",
        "| KD | α=0.1, β=0.9 |",
        "| Optimizer / LR / schedule | AdamW 1e-3, ReduceLROnPlateau (unchanged) |",
        "| Split / seed / data | Flight split 0.2, seed 42, full feature set |",
        "| Teacher | Frozen R3 |",
        "",
        "**Only change:** consistency loss on continuous (standardized numeric) features.",
        "",
        "---",
        "",
        "## 4. Regularization details",
        "",
        "```",
        "L_total = α·MSE(f(x), y) + β·MSE(f(x), y_teacher) + λ · ||f(x) − f(x+ε)||²",
        "```",
        "",
        "- ε ~ N(0, σ²) on continuous columns only (indices `0:n_num`); OHE categoricals untouched.",
        f"- σ = **{NOISE_SCALE}** on StandardScaler-normalized features (~1.5% of unit std).",
        "- λ ∈ {0.01, 0.1, 1.0}; select **best validation RMSE** checkpoint.",
        "",
        "---",
        "",
        "## 5. Hyperparameters",
        "",
        f"| λ | Val RMSE | Best epoch |",
        f"|--:|---------:|-----------:|",
    ]
    for s in train_summary:
        lines.append(
            f"| {s['lambda']} | {s.get('best_val_rmse', float('nan'))} | {s.get('best_epoch', '—')} |"
        )
    lines += [
        "",
        f"**Selected λ:** `{sel}` (by validation RMSE)",
        "",
        "---",
        "",
        "## 6. Benchmark results",
        "",
        "| Model | Val RMSE | Final RMSE | Type-Macro | Body-Macro |",
        "|-------|---------:|-----------:|-----------:|-----------:|",
        f"| Large (baseline) | {results['large_baseline'].get('val_rmse') or '—'} | "
        f"{results['large_baseline']['final_rmse']:.2f} | "
        f"{results['large_baseline']['type_macro']:.2f} | "
        f"{results['large_baseline']['body_macro']:.2f} |",
        f"| FT (baseline) | — | "
        f"{results['ft_baseline']['final_rmse']:.2f} | "
        f"{results['ft_baseline']['type_macro']:.2f} | "
        f"{results['ft_baseline']['body_macro']:.2f} |",
    ]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        mark = " **← selected**" if sel is not None and float(lam) == float(sel) else ""
        lines.append(
            f"| Cons λ={lam}{mark} | {r.get('val_rmse', float('nan')):.2f} | "
            f"{r['final_rmse']:.2f} | {r['type_macro']:.2f} | {r['body_macro']:.2f} |"
        )
    if sm:
        lines += [
            "",
            f"**Δ Type-Macro vs Large (selected):** {sm.get('delta_type_vs_large', float('nan')):+.2f} kg  ",
            f"**Δ Final vs Large (selected):** {sm.get('delta_final_vs_large', float('nan')):+.2f} kg",
        ]
    lines += [
        "",
        "![benchmark](figures/fig_p35_benchmark.png)",
        "",
        "---",
        "",
        "## 7. Representation analysis",
        "",
        "### Smoothness (embedding movement, ε=0.05 continuous noise)",
        "",
        "| Model | Mean rel move | Mean \|Δpred\| |",
        "|-------|-------------:|---------------:|",
        f"| Large | {results['stability_baselines']['large']['mean_rel_move']:.4f} | "
        f"{results['stability_baselines']['large']['mean_pred_abs_delta']:.2f} |",
        f"| FT | {results['stability_baselines']['ft']['mean_rel_move']:.4f} | "
        f"{results['stability_baselines']['ft']['mean_pred_abs_delta']:.2f} |",
    ]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        lines.append(
            f"| Cons λ={lam} | {r['stability']['mean_rel_move']:.4f} | "
            f"{r['stability']['mean_pred_abs_delta']:.2f} |"
        )
    lines += [
        "",
        "![smooth](figures/fig_p35_smoothness.png)",
        "",
        "### Geometry",
        "",
        "| Model | Rare→common (norm) | Within-type | Inter-centroid | Silhouette | Type purity k=10 |",
        "|-------|-------------------:|------------:|---------------:|-----------:|-----------------:|",
        f"| Large | {results['geometry_baselines']['large']['rare_to_common_norm']:.4f} | "
        f"{results['geometry_baselines']['large']['mean_within_type']:.3f} | "
        f"{results['geometry_baselines']['large']['mean_inter_type_centroid']:.3f} | "
        f"{results['geometry_baselines']['large']['silhouette']:.4f} | "
        f"{results['geometry_baselines']['large']['type_purity_k10']:.4f} |",
        f"| FT | {results['geometry_baselines']['ft']['rare_to_common_norm']:.4f} | "
        f"{results['geometry_baselines']['ft']['mean_within_type']:.3f} | "
        f"{results['geometry_baselines']['ft']['mean_inter_type_centroid']:.3f} | "
        f"{results['geometry_baselines']['ft']['silhouette']:.4f} | "
        f"{results['geometry_baselines']['ft']['type_purity_k10']:.4f} |",
    ]
    for lam, r in sorted(results["lambda_runs"].items(), key=lambda kv: float(kv[0])):
        g = r["geometry"]
        lines.append(
            f"| Cons λ={lam} | {g['rare_to_common_norm']:.4f} | {g['mean_within_type']:.3f} | "
            f"{g['mean_inter_type_centroid']:.3f} | {g['silhouette']:.4f} | {g['type_purity_k10']:.4f} |"
        )
    lines += [
        "",
        "![geom](figures/fig_p35_geometry_rare_common.png)",
        "",
        "---",
        "",
        "## 8. Comparison with FT",
        "",
        "Target properties of FT: lower relative embedding movement; moderate rare→common norm; "
        "not tighter type clusters (similar purity/silhouette).",
        "",
        "---",
        "",
        "## 9. Interpretation",
        "",
        f"**Outcome label:** `{results['outcome']}`",
        "",
        f"{results['interpretation']}",
        "",
        "### Outcome key (pre-registered)",
        "",
        "| Code | Condition | Meaning |",
        "|------|-----------|---------|",
        "| A | Smoothness ↑ + type-macro improves | Smoothness plausible causal |",
        "| B | Smoothness ↑ + type-macro flat | Smoothness not sufficient |",
        "| C | Smoothness not ↑ | Intervention failed |",
        "| D | Smoothness ↑ + IID shifts + type flat | Affects optimization not transfer |",
        "",
        "---",
        "",
        "## 10. Decision Gate",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Outcome | `{results['outcome']}` |",
        f"| Pursue smoothness-aware Phase 4 method | **{results['decision_gate']['pursue_smoothness_method']}** |",
        f"| Pursue representation distillation | **{results['decision_gate']['pursue_representation_distillation']}** |",
        f"| Mechanism resolved | **{results['decision_gate']['mechanism_resolved']}** |",
        f"| Next step | {results['decision_gate']['next_step']} |",
        "",
        f"**Phase 4 / paper recommendation:** {results['phase4_recommendation']}",
        "",
        "### Stopping rule",
        "",
        "This is the **final mechanism experiment**. Do not test Jacobian penalties, spectral "
        "normalization, Lipschitz constraints, adversarial smoothing, or further mechanism phases.",
        "",
        "---",
        "",
        "## Artifacts",
        "",
        f"- Results: `results/distillation/smoothness_causal/`",
        f"- Report: `docs/reports/smoothness_causal_intervention.md`",
        f"- Train script: `experiments/08_distillation/17_smoothness_causal_intervention.py`",
        "",
        f"*Generated {datetime.now(timezone.utc).isoformat()}*",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Only analyze existing checkpoints")
    args = ap.parse_args(argv)

    device_str = args.device
    device = torch.device(
        "cuda" if device_str == "auto" and torch.cuda.is_available() else (device_str if device_str != "auto" else "cpu")
    )

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    LOGGER.info("in_dim=%d n_num=%d", data.in_dim, len(data.numeric_cols))

    if args.skip_train:
        summary = json.loads((OUT / "training_summary.json").read_text(encoding="utf-8"))
    else:
        summary = train_sweep(data, device_str, args.force)

    results = analyze(data, device, summary)
    results["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    results["train_summary"] = summary
    results["noise_scale"] = NOISE_SCALE
    results["lambdas"] = list(LAMBDAS)

    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / "plots"
    fig_dir = ROOT / "docs" / "reports" / "figures"
    _plots(results, plots, fig_dir)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (OUT / "decision.json").write_text(
        json.dumps(
            {
                "outcome": results["outcome"],
                "interpretation": results["interpretation"],
                "phase4_recommendation": results["phase4_recommendation"],
                "decision_gate": results["decision_gate"],
                "selected_lambda": results.get("selected_lambda"),
                "selected_metrics": results.get("selected_metrics"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    report = _report(results, summary)
    (OUT / "smoothness_causal_intervention.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "smoothness_causal_intervention.md").write_text(report, encoding="utf-8")

    print("\n=== PHASE 3.5 SMOOTHNESS CAUSAL INTERVENTION ===")
    print(json.dumps(results["decision_gate"], indent=2))
    print(f"outcome={results['outcome']}")
    print(f"selected_lambda={results.get('selected_lambda')}")
    if results.get("selected_metrics"):
        print(json.dumps(results["selected_metrics"], indent=2, default=str))
    print(f"results={OUT}")


if __name__ == "__main__":
    main()
