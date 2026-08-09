"""Phase — Attention Routing Analysis (analysis-only; frozen FT + Large).

Tests H-Attention: whether FT attention behavior is associated with FT's
*relative* type-level advantage over Large MLP under aircraft-type shift.

No training. No checkpoint modification. Predictions must match instrumented path.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import CAT_FEATURES, DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.distillation.physics_features import classify_numeric
from aerotwin.engine.gap_closing import aircraft_class, clean_featured, ensure_features
from aerotwin.engine.mass_model import enrich_mass_from_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("attention_routing")

OUT = ROOT / "results" / "distillation" / "attention_routing"
LARGE_CKPT = ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt"
FT_CKPT = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt"
FT_CFG = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json"
TYPE_TABLE = ROOT / "results/distillation/mechanism_validation/type_level_physics_table.csv"
MIN_TYPE_N = 50
N_BOOT = 2000
# Pre-registered primary attention metrics (hypothesis-driven; not post-hoc fishing)
PRIMARY_METRICS = (
    "mean_cls_entropy",
    "top1_mass",
    "aircraft_cat_mass",
    "physics_mass",
    "trajectory_mass",
    "js_shift_from_common",
)


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


def _token_meta(data: DistillationData) -> dict[str, Any]:
    """Map token indices → names / families. Order: CLS, numeric…, categorical…"""
    n_num = len(data.numeric_cols)
    cat_names = list(data.cat_cols)
    names = ["CLS"] + list(data.numeric_cols) + cat_names
    families = ["cls"]
    for c in data.numeric_cols:
        families.append(classify_numeric(c))
    for c in cat_names:
        if c == "aircraft_type":
            families.append("aircraft_cat")
        else:
            families.append("other_cat")
    n_tokens = 1 + n_num + len(cat_names)
    assert len(names) == n_tokens
    family_to_idx: dict[str, list[int]] = {}
    for i, f in enumerate(families):
        family_to_idx.setdefault(f, []).append(i)
    aircraft_tok = 1 + n_num + cat_names.index("aircraft_type") if "aircraft_type" in cat_names else None
    return {
        "n_tokens": n_tokens,
        "n_num": n_num,
        "n_cat": len(cat_names),
        "names": names,
        "families": families,
        "family_to_idx": family_to_idx,
        "aircraft_token_index": aircraft_tok,
        "cat_names": cat_names,
    }


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (base e), symmetric, ≥0."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> dict[str, float]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(len(x))
    if n < 5:
        return {
            "spearman": float("nan"),
            "pearson": float("nan"),
            "p_spearman": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": n,
        }
    rho, pval = stats.spearmanr(x, y)
    pr, _ = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            boots.append(float(r))
    boots = np.asarray(boots) if boots else np.array([float("nan")])
    return {
        "spearman": float(rho),
        "pearson": float(pr),
        "p_spearman": float(pval),
        "ci_low": float(np.nanpercentile(boots, 2.5)),
        "ci_high": float(np.nanpercentile(boots, 97.5)),
        "n": n,
    }


@torch.no_grad()
def _predict_batch(model, x: np.ndarray, device, bs: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), bs):
        outs.append(model(xt[i : i + bs].to(device)).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


@torch.no_grad()
def _predict_with_attn(model, x: np.ndarray, device, bs: int = 256):
    """Returns pred (N,), and list of per-layer CLS attention (N, H, L)."""
    model.eval()
    preds = []
    # accumulate layer -> list of (B,H,L) cls rows
    layer_cls: list[list[np.ndarray]] = None  # type: ignore
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), bs):
        xb = xt[i : i + bs].to(device)
        pred, attns = model.forward_with_attention(xb)
        preds.append(pred.cpu().numpy())
        if layer_cls is None:
            layer_cls = [[] for _ in attns]
        for li, w in enumerate(attns):
            # w: (B, H, L, L) — CLS query row
            cls_row = w[:, :, 0, :].cpu().numpy()  # (B, H, L)
            layer_cls[li].append(cls_row)
    pred = np.concatenate(preds).astype(np.float64)
    layers = [np.concatenate(parts, axis=0) for parts in layer_cls]
    return pred, layers


def _cls_metrics_from_layers(
    layers: list[np.ndarray],
    meta: dict[str, Any],
) -> dict[str, np.ndarray]:
    """layers[i]: (N, H, L) CLS attention. Returns per-sample primary metrics."""
    n = layers[0].shape[0]
    n_layers = len(layers)
    n_heads = layers[0].shape[1]
    L = layers[0].shape[2]
    fam_idx = meta["family_to_idx"]
    ac_idx = meta["aircraft_token_index"]

    # Stack (N, n_layers, H, L)
    stack = np.stack(layers, axis=1)
    # mean over layers & heads for distribution: (N, L)
    mean_attn = stack.mean(axis=(1, 2))
    ent_all = _entropy(stack, axis=-1)  # (N, n_layers, H)
    mean_ent = ent_all.mean(axis=(1, 2))
    top1 = stack.max(axis=-1).mean(axis=(1, 2))
    # top-3 mass
    part = np.partition(stack, -3, axis=-1)[..., -3:]
    top3 = part.sum(axis=-1).mean(axis=(1, 2))
    # effective number of tokens: exp(entropy)
    eff_n = np.exp(ent_all).mean(axis=(1, 2))

    def mass_for(idxs: list[int]) -> np.ndarray:
        if not idxs:
            return np.zeros(n, dtype=np.float64)
        return mean_attn[:, idxs].sum(axis=1)

    aircraft_mass = mean_attn[:, ac_idx] if ac_idx is not None else np.zeros(n)
    physics_mass = mass_for(fam_idx.get("physics", []))
    traj_mass = mass_for(fam_idx.get("trajectory", []))
    weather_mass = mass_for(fam_idx.get("weather", []))
    ops_mass = mass_for(fam_idx.get("operational", []))
    other_cat_mass = mass_for(fam_idx.get("other_cat", []))
    cls_self = mean_attn[:, 0]

    return {
        "mean_cls_entropy": mean_ent.astype(np.float64),
        "top1_mass": top1.astype(np.float64),
        "top3_mass": top3.astype(np.float64),
        "effective_n_tokens": eff_n.astype(np.float64),
        "aircraft_cat_mass": aircraft_mass.astype(np.float64),
        "physics_mass": physics_mass.astype(np.float64),
        "trajectory_mass": traj_mass.astype(np.float64),
        "weather_mass": weather_mass.astype(np.float64),
        "operational_mass": ops_mass.astype(np.float64),
        "other_cat_mass": other_cat_mass.astype(np.float64),
        "cls_self_mass": cls_self.astype(np.float64),
        "mean_attn_dist": mean_attn.astype(np.float64),  # (N, L)
        "entropy_by_layer_head": ent_all.astype(np.float64),  # (N, layers, heads)
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_tokens": L,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    )
    out = OUT
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    meta = _token_meta(data)
    LOGGER.info(
        "Tokens: n=%d n_num=%d n_cat=%d aircraft_tok=%s",
        meta["n_tokens"],
        meta["n_num"],
        meta["n_cat"],
        meta["aircraft_token_index"],
    )
    (out / "token_meta.json").write_text(
        json.dumps({k: v for k, v in meta.items() if k != "family_to_idx"} | {
            "family_to_idx": {kk: vv for kk, vv in meta["family_to_idx"].items()}
        }, indent=2),
        encoding="utf-8",
    )

    final = _prepare(ROOT / "featured_dataset_final.parquet")
    x, y = _transform(final, data)
    types = final["aircraft_type"].cast(pl.Utf8).fill_null("?").to_numpy()
    bodies = np.array([_body(t) for t in types])

    # Load models
    LOGGER.info("Load frozen FT + Large")
    sc = StudentConfig.from_mapping(json.loads(FT_CFG.read_text(encoding="utf-8")))
    sc.in_dim = data.in_dim
    sc.n_num_features = len(data.numeric_cols)
    sc.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    ft = build_student(sc, in_dim=data.in_dim)
    ft.load_state_dict(torch.load(FT_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    ft.to(device).eval()

    large = StudentMLP(data.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
    large.load_state_dict(torch.load(LARGE_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    large.to(device).eval()

    # ---- Prediction invariance ----
    LOGGER.info("Prediction invariance check (subsample)")
    rng = np.random.default_rng(42)
    inv_n = min(2048, len(x))
    inv_idx = rng.choice(len(x), size=inv_n, replace=False)
    p_normal = _predict_batch(ft, x[inv_idx], device, bs=args.bs)
    p_instr, _ = _predict_with_attn(ft, x[inv_idx], device, bs=args.bs)
    inv = {
        "n": inv_n,
        "max_abs_diff": float(np.max(np.abs(p_normal - p_instr))),
        "mean_abs_diff": float(np.mean(np.abs(p_normal - p_instr))),
        "rmse_diff": float(np.sqrt(np.mean((p_normal - p_instr) ** 2))),
        "normal_rmse_vs_y": float(np.sqrt(np.mean((p_normal - y[inv_idx]) ** 2))),
        "instr_rmse_vs_y": float(np.sqrt(np.mean((p_instr - y[inv_idx]) ** 2))),
    }
    LOGGER.info("Invariance: max|Δ|=%.3e mean|Δ|=%.3e", inv["max_abs_diff"], inv["mean_abs_diff"])
    if inv["max_abs_diff"] > 1e-3:
        LOGGER.error("Prediction invariance FAILED — aborting analysis")
        (out / "prediction_invariance.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
        raise SystemExit(2)
    (out / "prediction_invariance.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")

    # ---- Full attention extraction ----
    LOGGER.info("Extract attention on Final holdout (N=%d)", len(x))
    pred_f, layers = _predict_with_attn(ft, x, device, bs=args.bs)
    pred_l = _predict_batch(large, x, device, bs=512)
    sample_m = _cls_metrics_from_layers(layers, meta)

    # rare definition: 33rd percentile of train frequency among types present
    train_df = pl.read_parquet(ROOT / "distillation_dataset.parquet")
    freq_map = {
        str(r["aircraft_type"]): int(r["len"])
        for r in train_df.group_by("aircraft_type").len().iter_rows(named=True)
    }
    train_n = np.array([freq_map.get(str(t), 0) for t in types], dtype=np.float64)
    thr_rare = float(np.percentile(list({t: freq_map.get(str(t), 0) for t in np.unique(types)}.values()), 33))
    rare = train_n <= thr_rare

    # ---- Type-level table (canonical metrics from Phase 3 + attention) ----
    type_rows_src = pl.read_csv(TYPE_TABLE) if TYPE_TABLE.exists() else None
    type_rows = []
    # common reference: top-5 train frequency types present in Final
    common_types = [t for t, _ in sorted(freq_map.items(), key=lambda kv: -kv[1])[:5]]
    common_mask = np.isin(types.astype(str), common_types)
    if common_mask.sum() < 50:
        common_mask = ~rare
    ref_attn = sample_m["mean_attn_dist"][common_mask].mean(axis=0)
    ref_attn = ref_attn / max(ref_attn.sum(), 1e-12)

    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < MIN_TYPE_N:
            continue
        l_rmse = float(np.sqrt(np.mean((pred_l[m] - y[m]) ** 2)))
        f_rmse = float(np.sqrt(np.mean((pred_f[m] - y[m]) ** 2)))
        adv = l_rmse - f_rmse
        mean_dist = sample_m["mean_attn_dist"][m].mean(axis=0)
        mean_dist = mean_dist / max(mean_dist.sum(), 1e-12)
        js = _js_divergence(mean_dist, ref_attn)
        row = {
            "aircraft_type": t,
            "body_class": _body(t),
            "n": int(m.sum()),
            "train_n": int(freq_map.get(t, 0)),
            "is_rare": bool(freq_map.get(t, 0) <= thr_rare),
            "large_rmse": l_rmse,
            "ft_rmse": f_rmse,
            "ft_advantage": adv,
            "mean_cls_entropy": float(sample_m["mean_cls_entropy"][m].mean()),
            "top1_mass": float(sample_m["top1_mass"][m].mean()),
            "top3_mass": float(sample_m["top3_mass"][m].mean()),
            "effective_n_tokens": float(sample_m["effective_n_tokens"][m].mean()),
            "aircraft_cat_mass": float(sample_m["aircraft_cat_mass"][m].mean()),
            "physics_mass": float(sample_m["physics_mass"][m].mean()),
            "trajectory_mass": float(sample_m["trajectory_mass"][m].mean()),
            "weather_mass": float(sample_m["weather_mass"][m].mean()),
            "operational_mass": float(sample_m["operational_mass"][m].mean()),
            "other_cat_mass": float(sample_m["other_cat_mass"][m].mean()),
            "cls_self_mass": float(sample_m["cls_self_mass"][m].mean()),
            "js_shift_from_common": js,
        }
        # per-layer mean entropy
        ent_lh = sample_m["entropy_by_layer_head"][m].mean(axis=0)  # (layers, heads)
        for li in range(ent_lh.shape[0]):
            row[f"entropy_layer{li}"] = float(ent_lh[li].mean())
            for hi in range(ent_lh.shape[1]):
                row[f"entropy_L{li}H{hi}"] = float(ent_lh[li, hi])
        type_rows.append(row)

    type_df = pl.DataFrame(type_rows)
    type_df.write_csv(out / "per_type_attention_metrics.csv")

    # ---- Group aggregates ----
    def group_stats(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() < 10:
            return {"n": int(mask.sum())}
        return {
            "n": int(mask.sum()),
            "mean_cls_entropy": float(sample_m["mean_cls_entropy"][mask].mean()),
            "top1_mass": float(sample_m["top1_mass"][mask].mean()),
            "aircraft_cat_mass": float(sample_m["aircraft_cat_mass"][mask].mean()),
            "physics_mass": float(sample_m["physics_mass"][mask].mean()),
            "trajectory_mass": float(sample_m["trajectory_mass"][mask].mean()),
            "ft_rmse": float(np.sqrt(np.mean((pred_f[mask] - y[mask]) ** 2))),
            "large_rmse": float(np.sqrt(np.mean((pred_l[mask] - y[mask]) ** 2))),
            "ft_advantage": float(
                np.sqrt(np.mean((pred_l[mask] - y[mask]) ** 2))
                - np.sqrt(np.mean((pred_f[mask] - y[mask]) ** 2))
            ),
        }

    groups = {
        "all": np.ones(len(y), dtype=bool),
        "common": ~rare,
        "rare": rare,
        "heavy": bodies == "widebody_heavy",
        "narrow": bodies == "narrowbody",
        "ft_wins_types": np.isin(
            types.astype(str),
            [r["aircraft_type"] for r in type_rows if r["ft_advantage"] > 0],
        ),
        "large_wins_types": np.isin(
            types.astype(str),
            [r["aircraft_type"] for r in type_rows if r["ft_advantage"] <= 0],
        ),
    }
    group_table = {k: group_stats(v) for k, v in groups.items()}

    # ---- Primary correlations (type-level) ----
    adv = np.array([r["ft_advantage"] for r in type_rows], dtype=float)
    ft_rmse = np.array([r["ft_rmse"] for r in type_rows], dtype=float)
    corr_advantage = {}
    corr_absolute = {}
    for metric in PRIMARY_METRICS:
        xm = np.array([r[metric] for r in type_rows], dtype=float)
        corr_advantage[metric] = _spearman_boot(xm, adv)
        corr_absolute[metric] = _spearman_boot(xm, ft_rmse)

    # exploratory: per-layer mean entropy vs advantage (labeled)
    exploratory = {}
    for li in range(sample_m["n_layers"]):
        key = f"entropy_layer{li}"
        xm = np.array([r[key] for r in type_rows], dtype=float)
        exploratory[key] = {
            "vs_ft_advantage": _spearman_boot(xm, adv),
            "vs_ft_rmse": _spearman_boot(xm, ft_rmse),
        }

    # strongest head (exploratory): max |rho| vs advantage among L*H
    head_corrs = []
    for li in range(sample_m["n_layers"]):
        for hi in range(sample_m["n_heads"]):
            key = f"entropy_L{li}H{hi}"
            xm = np.array([r[key] for r in type_rows], dtype=float)
            c = _spearman_boot(xm, adv)
            head_corrs.append({"layer": li, "head": hi, "metric": key, **c})
    head_corrs_sorted = sorted(head_corrs, key=lambda d: abs(d["spearman"]) if np.isfinite(d["spearman"]) else -1, reverse=True)

    # ---- Body-macro negative control ----
    body_rows = []
    for b in np.unique(bodies.astype(str)):
        m = bodies.astype(str) == b
        if m.sum() < 100:
            continue
        l_rmse = float(np.sqrt(np.mean((pred_l[m] - y[m]) ** 2)))
        f_rmse = float(np.sqrt(np.mean((pred_f[m] - y[m]) ** 2)))
        mean_dist = sample_m["mean_attn_dist"][m].mean(axis=0)
        mean_dist = mean_dist / max(mean_dist.sum(), 1e-12)
        body_rows.append(
            {
                "body_class": b,
                "n": int(m.sum()),
                "large_rmse": l_rmse,
                "ft_rmse": f_rmse,
                "ft_advantage": l_rmse - f_rmse,
                "mean_cls_entropy": float(sample_m["mean_cls_entropy"][m].mean()),
                "top1_mass": float(sample_m["top1_mass"][m].mean()),
                "aircraft_cat_mass": float(sample_m["aircraft_cat_mass"][m].mean()),
                "physics_mass": float(sample_m["physics_mass"][m].mean()),
                "trajectory_mass": float(sample_m["trajectory_mass"][m].mean()),
                "js_shift_from_common": _js_divergence(mean_dist, ref_attn),
            }
        )
    pl.DataFrame(body_rows).write_csv(out / "per_body_attention_metrics.csv")

    # Body-level correlations (n small — report but do not overclaim)
    body_corr = {}
    if len(body_rows) >= 3:
        b_adv = np.array([r["ft_advantage"] for r in body_rows], dtype=float)
        for metric in PRIMARY_METRICS:
            if metric not in body_rows[0]:
                continue
            xm = np.array([r[metric] for r in body_rows], dtype=float)
            body_corr[metric] = _spearman_boot(xm, b_adv, n_boot=500)
    else:
        # descriptive only
        body_corr = {"note": "n_body_classes < 3; correlations not estimated", "rows": body_rows}

    # ---- Layer/head summary table ----
    lh_summary = []
    ent_lh_all = sample_m["entropy_by_layer_head"].mean(axis=0)  # (L, H)
    for li in range(ent_lh_all.shape[0]):
        for hi in range(ent_lh_all.shape[1]):
            lh_summary.append(
                {
                    "layer": li,
                    "head": hi,
                    "mean_entropy": float(ent_lh_all[li, hi]),
                    "spearman_vs_ft_advantage": next(
                        (h["spearman"] for h in head_corrs if h["layer"] == li and h["head"] == hi),
                        float("nan"),
                    ),
                }
            )
    pl.DataFrame(lh_summary).write_csv(out / "layer_head_summary.csv")

    # ---- Decision (pre-registered logic) ----
    # Strongest primary metric by |Spearman| vs FT advantage
    best_name, best_c = max(
        corr_advantage.items(),
        key=lambda kv: abs(kv[1]["spearman"]) if np.isfinite(kv[1]["spearman"]) else -1,
    )
    best_abs = corr_absolute[best_name]
    rho = best_c["spearman"]
    ci_lo, ci_hi = best_c["ci_low"], best_c["ci_high"]
    # Meaningful: |ρ| ≥ 0.4 and CI excludes 0 (or p < 0.1 with |ρ|≥0.35), n small
    meaningful = (
        np.isfinite(rho)
        and abs(rho) >= 0.40
        and np.isfinite(ci_lo)
        and np.isfinite(ci_hi)
        and (ci_lo * ci_hi > 0)  # CI excludes 0
    )
    weaker_abs = abs(best_abs["spearman"]) if np.isfinite(best_abs["spearman"]) else 0.0
    # Advantage correlation stronger than absolute? (for support)
    stronger_than_abs = abs(rho) > weaker_abs + 0.05 if np.isfinite(rho) else False

    # Body control: if only 2 bodies, compare whether the same metric orders with advantage
    body_control = "ambiguous"
    if len(body_rows) >= 2:
        # Under body-macro, Large typically still wins overall; check if best metric varies with body advantage
        b_metrics = [r.get(best_name, float("nan")) for r in body_rows]
        b_advs = [r["ft_advantage"] for r in body_rows]
        # If FT advantage is negative for all bodies (Large better), mechanism should not claim type-specific routing as general win
        if all(a < 0 for a in b_advs):
            body_control = "no_ranking_reversal_at_body_level"
        if isinstance(body_corr, dict) and best_name in body_corr and isinstance(body_corr[best_name], dict):
            br = body_corr[best_name].get("spearman", float("nan"))
            if np.isfinite(br) and abs(br) >= abs(rho) - 0.05 and abs(br) >= 0.4:
                body_control = "similar_association_under_body"
            elif np.isfinite(br) and abs(br) < 0.25:
                body_control = "weaker_under_body"

    if meaningful and stronger_than_abs and body_control in ("weaker_under_body", "no_ranking_reversal_at_body_level"):
        decision = "A_supported"
        decision_note = (
            "Primary attention metric shows meaningful association with FT type-level advantage; "
            "stronger than absolute-error link; body control does not show the same ranking reversal pattern."
        )
    elif meaningful and not stronger_than_abs:
        decision = "B_suggestive"
        decision_note = (
            "Attention associates with FT advantage but also (or more) with absolute FT error — "
            "may reflect difficulty rather than relative robustness."
        )
    elif abs(rho) >= 0.30 and np.isfinite(rho):
        decision = "B_suggestive"
        decision_note = (
            f"Moderate association (ρ={rho:.2f}) with n={best_c['n']} types; CI or magnitude insufficient for support."
        )
    elif not np.isfinite(rho) or abs(rho) < 0.25:
        decision = "C_rejected"
        decision_note = (
            f"Weak/null association between primary attention metrics and FT type-level advantage "
            f"(strongest |ρ|={abs(rho) if np.isfinite(rho) else float('nan'):.2f} for {best_name})."
        )
    else:
        decision = "D_inconclusive"
        decision_note = "Evidence ambiguous under small-n type-level design."

    # Overall metrics for report
    full_metrics = {
        "ft_final_rmse": regression_metrics(y, pred_f)["rmse"],
        "large_final_rmse": regression_metrics(y, pred_l)["rmse"],
    }

    # ---- Plots ----
    _plots(
        type_rows,
        body_rows,
        group_table,
        sample_m,
        meta,
        types,
        rare,
        corr_advantage,
        best_name,
        plots,
        fig_dir,
        layers,
        y,
        pred_f,
        pred_l,
    )

    blob = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "H-Attention",
        "prediction_invariance": inv,
        "full_metrics": full_metrics,
        "token_meta_summary": {
            "n_tokens": meta["n_tokens"],
            "aircraft_token_index": meta["aircraft_token_index"],
            "families": sorted(meta["family_to_idx"].keys()),
        },
        "group_table": group_table,
        "primary_correlations_vs_ft_advantage": corr_advantage,
        "primary_correlations_vs_ft_rmse": corr_absolute,
        "exploratory_layer_entropy": exploratory,
        "strongest_heads_exploratory": head_corrs_sorted[:5],
        "body_rows": body_rows,
        "body_correlations": body_corr if not isinstance(body_corr.get("note"), str) else body_corr,
        "body_control_label": body_control,
        "best_primary_metric": best_name,
        "best_primary_vs_advantage": best_c,
        "best_primary_vs_absolute": best_abs,
        "decision": decision,
        "decision_note": decision_note,
        "rare_threshold_train_n": thr_rare,
        "common_reference_types": common_types,
        "n_types": len(type_rows),
        "wall_seconds": time.time() - t0,
        "primary_metrics_preregistered": list(PRIMARY_METRICS),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    (out / "decision.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "decision_note": decision_note,
                "best_primary_metric": best_name,
                "spearman_vs_ft_advantage": best_c,
                "spearman_vs_ft_rmse": best_abs,
                "body_control": body_control,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    # correlation table CSV
    corr_rows = []
    for mname in PRIMARY_METRICS:
        corr_rows.append(
            {
                "metric": mname,
                "target": "ft_advantage",
                **corr_advantage[mname],
            }
        )
        corr_rows.append(
            {
                "metric": mname,
                "target": "ft_rmse",
                **corr_absolute[mname],
            }
        )
    pl.DataFrame(corr_rows).write_csv(out / "correlation_table.csv")

    report = _report(blob, type_rows, body_rows, group_table)
    (out / "attention_routing_analysis.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "attention_routing_analysis.md").write_text(report, encoding="utf-8")

    print("\n=== ATTENTION ROUTING ANALYSIS ===")
    print(f"decision={decision}")
    print(f"best_metric={best_name} ρ={rho:.3f} CI=[{ci_lo:.3f},{ci_hi:.3f}] p={best_c['p_spearman']:.3g}")
    print(f"vs absolute FT RMSE: ρ={best_abs['spearman']:.3f}")
    print(f"body_control={body_control}")
    print(f"invariance max|Δ|={inv['max_abs_diff']:.3e}")
    print(f"results={out}")


def _plots(
    type_rows,
    body_rows,
    group_table,
    sample_m,
    meta,
    types,
    rare,
    corr_advantage,
    best_name,
    plots,
    fig_dir,
    layers,
    y,
    pred_f,
    pred_l,
):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_attn_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    # Fig 1: entropy by layer/head (global mean)
    ent = sample_m["entropy_by_layer_head"].mean(axis=0)  # (L,H)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(ent, aspect="auto", cmap="viridis")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("CLS attention entropy (mean over Final samples)")
    ax.set_yticks(range(ent.shape[0]))
    ax.set_xticks(range(ent.shape[1]))
    fig.colorbar(im, ax=ax, label="Entropy")
    save(fig, "entropy_layer_head")

    # Fig 2: concentration by group
    keys = [k for k in ["common", "rare", "heavy", "narrow", "ft_wins_types", "large_wins_types"] if k in group_table]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(keys))
    top1 = [group_table[k].get("top1_mass", np.nan) for k in keys]
    ac = [group_table[k].get("aircraft_cat_mass", np.nan) for k in keys]
    ax.bar(x - 0.2, top1, 0.4, label="Top-1 mass")
    ax.bar(x + 0.2, ac, 0.4, label="Aircraft-cat mass")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Attention mass")
    ax.set_title("Attention concentration by group")
    ax.legend()
    save(fig, "concentration_by_group")

    # Fig 3: feature-family attention (all)
    fams = ["physics", "trajectory", "weather", "operational", "aircraft_cat", "other_cat", "cls"]
    masses = []
    for f in fams:
        if f == "aircraft_cat":
            masses.append(float(sample_m["aircraft_cat_mass"].mean()))
        elif f == "other_cat":
            masses.append(float(sample_m["other_cat_mass"].mean()))
        elif f == "cls":
            masses.append(float(sample_m["cls_self_mass"].mean()))
        else:
            masses.append(float(sample_m.get(f"{f}_mass", np.zeros(1)).mean()) if f"{f}_mass" in sample_m else np.nan)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fams, masses)
    ax.set_ylabel("Mean CLS attention mass")
    ax.set_title("Feature-family attention (Final, all samples)")
    plt.xticks(rotation=20, ha="right")
    save(fig, "feature_family_mass")

    # Fig 4: redistribution common vs rare (family bars)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fam_keys = ["physics_mass", "trajectory_mass", "aircraft_cat_mass", "weather_mass"]
    labs = ["physics", "trajectory", "aircraft_cat", "weather"]
    x = np.arange(len(labs))
    common_v = [group_table["common"].get(k, np.nan) for k in fam_keys]
    rare_v = [group_table["rare"].get(k, np.nan) for k in fam_keys]
    ax.bar(x - 0.2, common_v, 0.4, label="common")
    ax.bar(x + 0.2, rare_v, 0.4, label="rare")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.legend()
    ax.set_title("Feature-family mass: common vs rare")
    ax.set_ylabel("Mean CLS attention mass")
    save(fig, "family_common_vs_rare")

    # Fig 5: PRIMARY — best metric vs FT advantage
    xs = np.array([r[best_name] for r in type_rows], dtype=float)
    ys = np.array([r["ft_advantage"] for r in type_rows], dtype=float)
    labels = [r["aircraft_type"] for r in type_rows]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(xs, ys, s=70, zorder=3)
    for xi, yi, lab in zip(xs, ys, labels):
        ax.annotate(lab, (xi, yi), fontsize=8, alpha=0.9)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    c = corr_advantage[best_name]
    ax.set_xlabel(best_name)
    ax.set_ylabel("FT advantage (Large RMSE − FT RMSE)")
    ax.set_title(
        f"Primary test: {best_name} vs FT advantage\n"
        f"Spearman ρ={c['spearman']:.2f} [{c['ci_low']:.2f},{c['ci_high']:.2f}] n={c['n']}"
    )
    if np.isfinite(c["spearman"]) and abs(c["spearman"]) >= 0.3 and c["n"] >= 8:
        coef = np.polyfit(xs[np.isfinite(xs) & np.isfinite(ys)], ys[np.isfinite(xs) & np.isfinite(ys)], 1)
        xr = np.linspace(np.nanmin(xs), np.nanmax(xs), 50)
        ax.plot(xr, np.polyval(coef, xr), color="C1", lw=1.5, alpha=0.8, label="OLS trend")
        ax.legend()
    save(fig, "primary_metric_vs_ft_advantage")

    # Fig 6: type vs body for strongest metric
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar([r["aircraft_type"] for r in type_rows], [r[best_name] for r in type_rows])
    axes[0].tick_params(axis="x", rotation=75, labelsize=7)
    axes[0].set_title(f"Type-level {best_name}")
    axes[0].set_ylabel(best_name)
    if body_rows:
        axes[1].bar([r["body_class"] for r in body_rows], [r.get(best_name, np.nan) for r in body_rows])
        axes[1].set_title(f"Body-level {best_name}")
    save(fig, "type_vs_body_metric")

    # Representative heatmaps: pick types by advantage
    sorted_by_adv = sorted(type_rows, key=lambda r: r["ft_advantage"], reverse=True)
    sorted_by_freq = sorted(type_rows, key=lambda r: -r["train_n"])
    pick = []
    if sorted_by_freq:
        pick.append(("common", sorted_by_freq[0]["aircraft_type"]))
    if sorted_by_adv:
        pick.append(("ft_wins", sorted_by_adv[0]["aircraft_type"]))
    if sorted_by_adv:
        pick.append(("large_wins", sorted_by_adv[-1]["aircraft_type"]))
    for tag, tname in pick:
        m = types.astype(str) == tname
        if m.sum() < 5:
            continue
        # mean entropy-like: mean CLS attention over samples (avg heads) per layer: use layer mean attn
        # layers[li][m]: (n, H, L) -> mean (H, L) then mean heads -> (L,) for bar; or heatmap heads x tokens
        fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.5))
        if len(layers) == 1:
            axes = [axes]
        for li, ax in enumerate(axes):
            mat = layers[li][m].mean(axis=0)  # (H, L)
            im = ax.imshow(mat, aspect="auto", cmap="magma")
            ax.set_title(f"{tname} L{li}")
            ax.set_xlabel("Token")
            ax.set_ylabel("Head")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"CLS attention heatmap ({tag}: {tname})")
        save(fig, f"heatmap_{tag}_{tname}")


def _report(blob: dict, type_rows: list, body_rows: list, group_table: dict) -> str:
    c = blob["best_primary_vs_advantage"]
    ca = blob["best_primary_vs_absolute"]
    lines = [
        "# Attention Routing Analysis",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "**Status:** Complete — analysis-only (frozen FT + Large; no training)",
        f"**Decision:** `{blob['decision']}`",
        "",
        "---",
        "",
        "## 1. Research question",
        "",
        "Does **attention-based feature routing** in the FT-Transformer help explain why FT beats "
        "Large MLP under **aircraft-type macro** evaluation despite worse Flight Holdout RMSE?",
        "",
        "---",
        "",
        "## 2. Hypothesis",
        "",
        "> **H-Attention:** Under aircraft-type distribution shift, FT dynamically changes how it routes "
        "information across feature tokens, allowing more effective use of features shared across "
        "aircraft types than the MLP.",
        "",
        "We test association of attention metrics with **FT advantage** "
        "`(Large RMSE_t − FT RMSE_t)`, not merely whether attention maps look interesting.",
        "",
        "---",
        "",
        "## 3. Existing evidence (not reinterpreted)",
        "",
        "| Model | Final RMSE | Type-macro RMSE |",
        "|-------|----------:|----------------:|",
        "| Large MLP | 215.85 | 270.61 |",
        "| FT-Transformer | 224.12 | 261.15 |",
        "",
        "Prior mechanisms rejected/not established: teacher uncertainty (VGKD), physics-feature "
        "reliance, representation geometry as sufficient causal account, local smoothness.",
        "",
        "---",
        "",
        "## 4. Experimental setup",
        "",
        "- Frozen FT checkpoint: `results/distillation/ft_transformer/ft_transformer_kd1/`",
        "- Frozen Large: `results/distillation/capacity_scaling/runs/Large_seed42/`",
        "- Evaluation: `featured_dataset_final.parquet` (same pipeline as prior phases)",
        "- Type unit: aircraft types with n≥50 on Final",
        "- **No training**, no feature/split changes",
        "",
        "---",
        "",
        "## 5. Attention extraction method",
        "",
        "- Instrument `MultiheadAttention` / `TransformerBlock` with optional `need_weights`.",
        "- Residual path still uses the same SDPA (or fallback) output as production.",
        "- Analytic softmax(QKᵀ/√d) weights are computed **after** the residual output for analysis only.",
        "- Primary readout: **CLS query row** attention over tokens [CLS + 56 num + 4 cat].",
        "- API: `FTTransformer.forward_with_attention(x) → (pred, [attn_layer…])`.",
        "",
        "### Prediction invariance",
        "",
        f"| Check | Value |",
        f"|-------|------:|",
        f"| n | {blob['prediction_invariance']['n']} |",
        f"| max \\|Δ\\| | {blob['prediction_invariance']['max_abs_diff']:.3e} |",
        f"| mean \\|Δ\\| | {blob['prediction_invariance']['mean_abs_diff']:.3e} |",
        f"| RMSE(Δ) | {blob['prediction_invariance']['rmse_diff']:.3e} |",
        "",
        "Pass criterion: max |Δ| ≤ 1e−3 (numerical noise).",
        "",
        "---",
        "",
        "## 6. Metrics (pre-registered primary set)",
        "",
        "Hypothesis-driven primary metrics (not selected after fishing):",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| `mean_cls_entropy` | Mean entropy of CLS attention over tokens (avg layers×heads) |",
        "| `top1_mass` | Mean max attention weight (concentration) |",
        "| `aircraft_cat_mass` | CLS mass on aircraft_type categorical token |",
        "| `physics_mass` | CLS mass on physics/mass/energy numeric tokens |",
        "| `trajectory_mass` | CLS mass on trajectory numeric tokens |",
        "| `js_shift_from_common` | JS divergence of type mean CLS attention vs common-type reference |",
        "",
        "Per-layer / per-head entropy correlations are **exploratory**.",
        "",
        "---",
        "",
        "## 7. Results",
        "",
        "### 7.1 Group-level attention",
        "",
        "| Group | n | Entropy | Top-1 | Aircraft-cat | Physics | Trajectory | FT adv (group RMSE) |",
        "|-------|--:|--------:|------:|-------------:|--------:|-----------:|--------------------:|",
    ]
    for k, v in group_table.items():
        if "mean_cls_entropy" not in v:
            continue
        lines.append(
            f"| {k} | {v['n']} | {v['mean_cls_entropy']:.3f} | {v['top1_mass']:.3f} | "
            f"{v['aircraft_cat_mass']:.3f} | {v['physics_mass']:.3f} | {v['trajectory_mass']:.3f} | "
            f"{v.get('ft_advantage', float('nan')):.2f} |"
        )
    lines += [
        "",
        "### 7.2 Primary correlations (aircraft type = unit)",
        "",
        "#### Attention metric ↔ FT advantage",
        "",
        "| Metric | Spearman ρ | 95% CI | p | n |",
        "|--------|----------:|-------:|--:|--:|",
    ]
    for mname, cdict in blob["primary_correlations_vs_ft_advantage"].items():
        lines.append(
            f"| {mname} | {cdict['spearman']:.3f} | [{cdict['ci_low']:.3f}, {cdict['ci_high']:.3f}] | "
            f"{cdict['p_spearman']:.3g} | {cdict['n']} |"
        )
    lines += [
        "",
        "#### Attention metric ↔ FT absolute RMSE (critical comparison)",
        "",
        "| Metric | Spearman ρ | 95% CI | p | n |",
        "|--------|----------:|-------:|--:|--:|",
    ]
    for mname, cdict in blob["primary_correlations_vs_ft_rmse"].items():
        lines.append(
            f"| {mname} | {cdict['spearman']:.3f} | [{cdict['ci_low']:.3f}, {cdict['ci_high']:.3f}] | "
            f"{cdict['p_spearman']:.3g} | {cdict['n']} |"
        )
    lines += [
        "",
        f"**Strongest primary vs advantage:** `{blob['best_primary_metric']}` "
        f"ρ={c['spearman']:.3f} CI=[{c['ci_low']:.3f},{c['ci_high']:.3f}] p={c['p_spearman']:.3g}",
        "",
        f"**Same metric vs absolute FT RMSE:** ρ={ca['spearman']:.3f} "
        f"CI=[{ca['ci_low']:.3f},{ca['ci_high']:.3f}]",
        "",
        "![primary](figures/fig_attn_primary_metric_vs_ft_advantage.png)",
        "",
        "### 7.3 Body-macro negative control",
        "",
        "| Body | n | Large RMSE | FT RMSE | FT adv | Entropy | Aircraft-cat |",
        "|------|--:|-----------:|--------:|-------:|--------:|-------------:|",
    ]
    for r in body_rows:
        lines.append(
            f"| {r['body_class']} | {r['n']} | {r['large_rmse']:.2f} | {r['ft_rmse']:.2f} | "
            f"{r['ft_advantage']:.2f} | {r['mean_cls_entropy']:.3f} | {r['aircraft_cat_mass']:.3f} |"
        )
    lines += [
        "",
        f"**Body-control label:** `{blob['body_control_label']}`",
        "",
        "![type_body](figures/fig_attn_type_vs_body_metric.png)",
        "",
        "### 7.4 Layer / head (exploratory)",
        "",
        "Top heads by |Spearman| vs FT advantage (exploratory; multiple comparisons):",
        "",
    ]
    for h in blob["strongest_heads_exploratory"][:5]:
        lines.append(
            f"- L{h['layer']}H{h['head']}: ρ={h['spearman']:.3f} "
            f"CI=[{h['ci_low']:.3f},{h['ci_high']:.3f}] p={h['p_spearman']:.3g}"
        )
    lines += [
        "",
        "![entropy](figures/fig_attn_entropy_layer_head.png)",
        "",
        "![family](figures/fig_attn_feature_family_mass.png)",
        "",
        "![conc](figures/fig_attn_concentration_by_group.png)",
        "",
        "---",
        "",
        "## 8. Statistical analysis",
        "",
        f"- Unit of inference: aircraft type (n={blob['n_types']}).",
        f"- Bootstrap: {N_BOOT} resamples of types for Spearman CI.",
        "- Small n: do not treat p-values as strong confirmatory evidence.",
        "- Primary metric set fixed before looking at advantage correlations.",
        "",
        "---",
        "",
        "## 9. Negative control",
        "",
        "Body-macro does not reverse Large vs FT ranking in established results. "
        f"Here body-control label = `{blob['body_control_label']}`. "
        "A mechanism that only tracks type-level FT advantage should not be required to "
        "produce body-level ranking reversal; conversely, if the same attention–advantage "
        "link appears equally under body grouping with no ranking flip, that weakens "
        "specificity to the type-macro phenomenon.",
        "",
        "---",
        "",
        "## 10. Interpretation",
        "",
        blob["decision_note"],
        "",
        "**Language:** Attention behavior is discussed as **association**, not causation.",
        "",
        "---",
        "",
        "## 11. Limitations",
        "",
        "1. Analytic attention weights may differ slightly from fused SDPA internals (predictions verified invariant).",
        "2. Small number of aircraft types (n≈15) limits power.",
        "3. CLS-row attention is one readout; other query positions not exhaustively tested.",
        "4. No causal intervention on attention in this phase.",
        "5. Feature-family mapping uses project `classify_numeric` + categorical names; residual \"other\" bucket exists.",
        "6. Per-head results are exploratory (multiple comparisons).",
        "",
        "---",
        "",
        "## 12. Decision",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Classification | **{blob['decision']}** |",
        f"| Best primary metric | `{blob['best_primary_metric']}` |",
        f"| ρ (vs FT advantage) | {c['spearman']:.3f} |",
        f"| 95% CI | [{c['ci_low']:.3f}, {c['ci_high']:.3f}] |",
        f"| ρ (vs FT RMSE) | {ca['spearman']:.3f} |",
        f"| Body control | {blob['body_control_label']} |",
        "",
        "---",
        "",
        "## 13. Recommended next step",
        "",
    ]
    d = blob["decision"]
    if d.startswith("A"):
        next_step = (
            "Design **one** pre-registered attention intervention (e.g., soft mask on aircraft-type "
            "token or feature-family attention reweighting) on a *copy* of FT to test causality — "
            "only if a new project charter reopens method work."
        )
    elif d.startswith("B"):
        next_step = (
            "Treat H-Attention as **suggestive only**. Do not base a new method on attention routing. "
            "If paper space allows, report the association with explicit small-n caveats; otherwise "
            "fold into appendix. Prefer completing the empirical paper over new mechanism phases."
        )
    elif d.startswith("C"):
        next_step = (
            "Record H-Attention as **rejected** (no association with relative FT advantage). "
            "Do not pursue attention-based methods. Continue **paper writing** documenting "
            "the ranking reversal and ruled-out mechanisms."
        )
    else:
        next_step = (
            "Mark H-Attention **inconclusive**. Do not launch methods. Prefer paper write-up "
            "with a short appendix on attention extraction limitations."
        )
    lines += [
        next_step,
        "",
        "---",
        "",
        "## Artifacts",
        "",
        f"- Results: `results/distillation/attention_routing/`",
        f"- Script: `experiments/08_distillation/18_attention_routing_analysis.py`",
        f"- Instrumentation: `src/aerotwin/distillation/models/ft_transformer.py` "
        f"(`forward_with_attention`)",
        "",
        f"*Generated {blob['timestamp_utc']}*",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
