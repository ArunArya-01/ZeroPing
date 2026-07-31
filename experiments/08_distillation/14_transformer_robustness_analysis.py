"""Phase 2 — Understanding Transformer Robustness (analysis only, no training).

Compares frozen Large MLP vs FT-Transformer on Final holdout:
  A) Representation geometry (PCA / UMAP / t-SNE + clustering metrics)
  B) Error localization (type, body, phase, duration, fuel, rare/common)
  C) Feature utilization (gradient×input attribution on numeric features)

Does not modify checkpoints or retrain models.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.engine.gap_closing import (
    HEAVY_TYPES,
    NARROW_TYPES,
    aircraft_class,
    clean_featured,
    ensure_features,
    group_phase,
)
from aerotwin.engine.mass_model import enrich_mass_from_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("phase2_robustness")

LARGE_CKPT = ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt"
FT_CKPT = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt"
FT_CFG = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json"
FINAL = ROOT / "featured_dataset_final.parquet"
OUT = ROOT / "results" / "distillation" / "transformer_robustness"
MIN_TYPE_N = 50
# Subsample for expensive viz / attribution
VIZ_N = 6000
ATTR_N = 2500
TSNE_N = 3000


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
def _encode_predict(model: torch.nn.Module, x: np.ndarray, device: torch.device, bs: int = 1024):
    model.eval()
    embs, preds = [], []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), bs):
        xb = xt[i : i + bs].to(device)
        h = model.encode(xb)
        p = model(xb)
        embs.append(h.cpu().numpy())
        preds.append(p.cpu().numpy())
    return np.concatenate(embs).astype(np.float64), np.concatenate(preds).astype(np.float64)


def _geometry_metrics(emb: np.ndarray, labels: np.ndarray, min_class: int = 20) -> dict[str, float]:
    """Clustering geometry metrics for type labels."""
    le = LabelEncoder()
    y = le.fit_transform(labels.astype(str))
    # drop rare classes for silhouette stability
    counts = np.bincount(y)
    keep = np.isin(y, np.where(counts >= min_class)[0])
    emb_k, y_k = emb[keep], y[keep]
    if len(np.unique(y_k)) < 2 or len(emb_k) < 100:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "n": int(len(emb))}

    # standardize embeddings for distance metrics
    z = StandardScaler().fit_transform(emb_k)
    sil = float(silhouette_score(z, y_k, sample_size=min(5000, len(z)), random_state=42))
    db = float(davies_bouldin_score(z, y_k))

    # intra / inter type mean pairwise (sample for speed)
    rng = np.random.default_rng(42)
    intra, inter = [], []
    classes = np.unique(y_k)
    for c in classes:
        idx = np.flatnonzero(y_k == c)
        if len(idx) < 2:
            continue
        samp = rng.choice(idx, size=min(80, len(idx)), replace=False)
        d = np.linalg.norm(z[samp, None, :] - z[None, samp, :], axis=-1)
        tri = d[np.triu_indices(len(samp), k=1)]
        if len(tri):
            intra.append(float(np.mean(tri)))
        # inter: to other class centroids
        others = z[y_k != c]
        if len(others) == 0:
            continue
        osamp = others[rng.choice(len(others), size=min(200, len(others)), replace=False)]
        inter.append(float(np.mean(np.linalg.norm(z[samp, None, :] - osamp[None, :, :], axis=-1))))

    # nearest other-type neighbor fraction (local consistency)
    nn = NearestNeighbors(n_neighbors=6).fit(z)
    dist, ind = nn.kneighbors(z)
    # exclude self (first neighbor)
    same = []
    for i in range(len(z)):
        neigh = ind[i, 1:]
        same.append(float(np.mean(y_k[neigh] == y_k[i])))
    local_consistency = float(np.mean(same))

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "mean_intra_type_dist": float(np.mean(intra)) if intra else float("nan"),
        "mean_inter_type_dist": float(np.mean(inter)) if inter else float("nan"),
        "inter_intra_ratio": float(np.mean(inter) / max(np.mean(intra), 1e-9)) if intra and inter else float("nan"),
        "local_type_consistency": local_consistency,
        "n_for_metrics": int(len(emb_k)),
        "n_types_used": int(len(np.unique(y_k))),
    }


def _grad_input_attr(
    model: torch.nn.Module,
    x: np.ndarray,
    n_num: int,
    device: torch.device,
    max_n: int = ATTR_N,
) -> np.ndarray:
    """Mean |grad * input| over samples for each numeric feature (first n_num cols)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=min(max_n, len(x)), replace=False)
    xs = torch.as_tensor(x[idx], dtype=torch.float32, device=device)
    xs.requires_grad_(True)
    model.eval()
    pred = model(xs)
    # sum of predictions so each sample contributes
    pred.sum().backward()
    g = xs.grad.detach().cpu().numpy()
    xi = xs.detach().cpu().numpy()
    # |g * x| mean over batch for numeric columns
    attr = np.mean(np.abs(g[:, :n_num] * xi[:, :n_num]), axis=0)
    return attr.astype(np.float64)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    )
    out = Path(args.out)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    LOGGER.info("Device %s | loading data (no training)", device)

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    final_df = _prepare(FINAL)
    x, y = _transform(final_df, data)
    n_num = len(data.numeric_cols)
    types = final_df["aircraft_type"].cast(pl.Utf8).fill_null("?").to_numpy()
    bodies = np.array([_body(t) for t in types])
    phases = group_phase(final_df).astype(str)
    duration = final_df["duration_s"].to_numpy().astype(np.float64)
    # altitude if available
    alt = (
        final_df["mean_altitude"].to_numpy().astype(np.float64)
        if "mean_altitude" in final_df.columns
        else np.full(len(y), np.nan)
    )
    temp = (
        final_df["temperature_k"].to_numpy().astype(np.float64)
        if "temperature_k" in final_df.columns
        else np.full(len(y), np.nan)
    )
    wind = (
        final_df["headwind_mps"].to_numpy().astype(np.float64)
        if "headwind_mps" in final_df.columns
        else np.full(len(y), np.nan)
    )

    # train frequency
    train_df = pl.read_parquet(ROOT / "distillation_dataset.parquet")
    freq_map = {
        str(r["aircraft_type"]): int(r["len"])
        for r in train_df.group_by("aircraft_type").len().iter_rows(named=True)
    }
    train_n = np.array([freq_map.get(str(t), 0) for t in types], dtype=np.float64)
    # rare = bottom third of types by training frequency
    type_train = {t: freq_map.get(str(t), 0) for t in np.unique(types.astype(str))}
    thr_rare = float(np.percentile(list(type_train.values()), 33)) if type_train else 0.0
    rare = np.array([type_train[str(t)] <= thr_rare for t in types])

    # Load models
    LOGGER.info("Loading frozen Large MLP")
    large = StudentMLP(data.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
    large.load_state_dict(
        torch.load(LARGE_CKPT, map_location=device, weights_only=False)["model_state_dict"]
    )
    large.to(device).eval()

    LOGGER.info("Loading frozen FT-Transformer")
    sc = StudentConfig.from_mapping(json.loads(FT_CFG.read_text(encoding="utf-8")))
    sc.in_dim = data.in_dim
    sc.n_num_features = n_num
    sc.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    ft = build_student(sc, in_dim=data.in_dim)
    ft.load_state_dict(torch.load(FT_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    ft.to(device).eval()

    LOGGER.info("Encoding Full Final set")
    emb_l, pred_l = _encode_predict(large, x, device)
    emb_f, pred_f = _encode_predict(ft, x, device)
    err_l = np.abs(pred_l - y)
    err_f = np.abs(pred_f - y)
    sq_l = (pred_l - y) ** 2
    sq_f = (pred_f - y) ** 2

    # Save predictions + meta
    pl.DataFrame(
        {
            "aircraft_type": types.tolist(),
            "body_class": bodies.tolist(),
            "phase": phases.tolist(),
            "duration_s": duration,
            "mean_altitude": alt,
            "temperature_k": temp,
            "headwind_mps": wind,
            "train_type_n": train_n,
            "is_rare_type": rare.astype(int),
            "ground_truth": y,
            "pred_large": pred_l,
            "pred_ft": pred_f,
            "abs_err_large": err_l,
            "abs_err_ft": err_f,
            "sq_err_large": sq_l,
            "sq_err_ft": sq_f,
            "ft_better": (err_f < err_l).astype(int),
        }
    ).write_parquet(out / "error_table.parquet")

    # Overall metrics
    overall = {
        "large": regression_metrics(y, pred_l),
        "ft": regression_metrics(y, pred_f),
        "type_macro_large": _type_macro(y, pred_l, types),
        "type_macro_ft": _type_macro(y, pred_f, types),
        "body_macro_large": _body_macro(y, pred_l, bodies),
        "body_macro_ft": _body_macro(y, pred_f, bodies),
    }

    # Geometry on subsample
    rng = np.random.default_rng(42)
    viz_idx = rng.choice(len(x), size=min(VIZ_N, len(x)), replace=False)
    LOGGER.info("Geometry metrics + projections (n=%d)", len(viz_idx))

    geom = {
        "large": _geometry_metrics(emb_l[viz_idx], types[viz_idx]),
        "ft": _geometry_metrics(emb_f[viz_idx], types[viz_idx]),
    }

    # PCA
    pca_l = PCA(n_components=2, random_state=42)
    pca_f = PCA(n_components=2, random_state=42)
    z_l = StandardScaler().fit_transform(emb_l[viz_idx])
    z_f = StandardScaler().fit_transform(emb_f[viz_idx])
    pca2_l = pca_l.fit_transform(z_l)
    pca2_f = pca_f.fit_transform(z_f)
    geom["large"]["pca_var_explained"] = [float(v) for v in pca_l.explained_variance_ratio_]
    geom["ft"]["pca_var_explained"] = [float(v) for v in pca_f.explained_variance_ratio_]

    # UMAP
    umap2_l = umap2_f = None
    try:
        import umap

        um = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        umap2_l = um.fit_transform(z_l)
        um2 = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        umap2_f = um2.fit_transform(z_f)
        LOGGER.info("UMAP done")
    except Exception as e:
        LOGGER.warning("UMAP unavailable: %s", e)

    # t-SNE
    tsne_idx = viz_idx[: min(TSNE_N, len(viz_idx))]
    # map tsne_idx positions in viz_idx
    pos = {int(v): i for i, v in enumerate(viz_idx)}
    tsne_pos = [pos[int(i)] for i in tsne_idx if int(i) in pos]
    tsne_l = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(z_l[tsne_pos])
    tsne_f = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(z_f[tsne_pos])

    # Distance to nearest common-type centroid (transfer proxy)
    common_types = [t for t, n in sorted(freq_map.items(), key=lambda kv: -kv[1])[:5]]
    geom["large"]["mean_dist_to_common_centroid"] = _dist_to_group_centroid(
        emb_l, types, common_types, rare
    )
    geom["ft"]["mean_dist_to_common_centroid"] = _dist_to_group_centroid(
        emb_f, types, common_types, rare
    )

    # Error localization tables
    err_loc = {
        "by_type": _group_err(y, pred_l, pred_f, types, min_n=MIN_TYPE_N, train_freq=freq_map),
        "by_body": _group_err(y, pred_l, pred_f, bodies, min_n=100),
        "by_phase": _group_err(y, pred_l, pred_f, phases, min_n=100),
        "by_duration_bin": _group_err(
            y, pred_l, pred_f, _duration_bins(duration), min_n=100
        ),
        "by_fuel_bin": _group_err(y, pred_l, pred_f, _fuel_bins(y), min_n=100),
        "by_altitude_bin": _group_err(y, pred_l, pred_f, _alt_bins(alt), min_n=100),
        "by_rare": _group_err(
            y, pred_l, pred_f, np.where(rare, "rare", "common"), min_n=100
        ),
    }
    pl.DataFrame(err_loc["by_type"]).write_csv(out / "error_by_type.csv")
    pl.DataFrame(err_loc["by_body"]).write_csv(out / "error_by_body.csv")
    pl.DataFrame(err_loc["by_phase"]).write_csv(out / "error_by_phase.csv")
    pl.DataFrame(err_loc["by_rare"]).write_csv(out / "error_by_rare.csv")

    # Where FT wins: fraction better + mean error reduction
    ft_gain = {
        "frac_ft_lower_abs_err": float(np.mean(err_f < err_l)),
        "mean_abs_err_delta_ft_minus_large": float(np.mean(err_f - err_l)),
        "mean_abs_err_delta_on_rare": float(np.mean(err_f[rare] - err_l[rare])) if rare.any() else float("nan"),
        "mean_abs_err_delta_on_common": float(np.mean(err_f[~rare] - err_l[~rare])) if (~rare).any() else float("nan"),
        "mean_abs_err_delta_heavy": float(
            np.mean(err_f[bodies == "widebody_heavy"] - err_l[bodies == "widebody_heavy"])
        ),
        "mean_abs_err_delta_narrow": float(
            np.mean(err_f[bodies == "narrowbody"] - err_l[bodies == "narrowbody"])
        ),
        "types_where_ft_type_rmse_better": [
            r["group"]
            for r in err_loc["by_type"]
            if r["ft_rmse"] < r["large_rmse"]
        ],
        "types_where_large_type_rmse_better": [
            r["group"]
            for r in err_loc["by_type"]
            if r["large_rmse"] <= r["ft_rmse"]
        ],
    }

    # Feature attribution
    LOGGER.info("Gradient×input attribution (numeric features)")
    attr_l = _grad_input_attr(large, x, n_num, device)
    attr_f = _grad_input_attr(ft, x, n_num, device)
    # normalize to sum 1
    attr_l_n = attr_l / max(attr_l.sum(), 1e-12)
    attr_f_n = attr_f / max(attr_f.sum(), 1e-12)
    feat_names = list(data.numeric_cols)
    attr_table = pl.DataFrame(
        {
            "feature": feat_names,
            "large_attr": attr_l_n,
            "ft_attr": attr_f_n,
            "abs_diff": np.abs(attr_l_n - attr_f_n),
        }
    ).sort("abs_diff", descending=True)
    attr_table.write_csv(out / "feature_attribution.csv")

    # Spearman of attributions across models
    attr_corr = float(stats.spearmanr(attr_l_n, attr_f_n).correlation)
    # stability: attribution on two random halves for each model
    half = len(x) // 2
    attr_l1 = _grad_input_attr(large, x[:half], n_num, device, max_n=min(ATTR_N, half))
    attr_l2 = _grad_input_attr(large, x[half:], n_num, device, max_n=min(ATTR_N, half))
    attr_f1 = _grad_input_attr(ft, x[:half], n_num, device, max_n=min(ATTR_N, half))
    attr_f2 = _grad_input_attr(ft, x[half:], n_num, device, max_n=min(ATTR_N, half))
    stability = {
        "large_half_spearman": float(stats.spearmanr(attr_l1, attr_l2).correlation),
        "ft_half_spearman": float(stats.spearmanr(attr_f1, attr_f2).correlation),
        "cross_model_spearman": attr_corr,
    }

    # Physics feature share
    phys_keys = [i for i, n in enumerate(feat_names) if "physics" in n or n.startswith("r3_") or "energy" in n]
    aircraft_id_keys = []  # OHE not in numeric attr
    phys_share = {
        "large_physics_share": float(attr_l_n[phys_keys].sum()) if phys_keys else float("nan"),
        "ft_physics_share": float(attr_f_n[phys_keys].sum()) if phys_keys else float("nan"),
        "n_physics_like_features": len(phys_keys),
    }

    # Plots
    meta_viz = {
        "types": types[viz_idx],
        "bodies": bodies[viz_idx],
        "err_l": err_l[viz_idx],
        "err_f": err_f[viz_idx],
        "train_n": train_n[viz_idx],
        "rare": rare[viz_idx],
    }
    _plots(
        pca2_l,
        pca2_f,
        umap2_l,
        umap2_f,
        tsne_l,
        tsne_f,
        meta_viz,
        tsne_pos,
        err_loc,
        attr_table,
        overall,
        geom,
        ft_gain,
        plots,
        fig_dir,
    )

    # Hypotheses from evidence
    hypotheses = _hypotheses(overall, geom, ft_gain, err_loc, stability, phys_share)

    blob = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_training": True,
        "n_final": len(y),
        "overall": overall,
        "geometry": geom,
        "error_localization_summary": {
            k: (v[:15] if isinstance(v, list) else v) for k, v in err_loc.items()
        },
        "ft_gain": ft_gain,
        "feature_attribution_top": attr_table.head(15).to_dicts(),
        "attribution_stability": stability,
        "physics_share": phys_share,
        "hypotheses": hypotheses,
        "wall_seconds": time.time() - t0,
        "common_types_for_centroid": common_types,
        "rare_train_n_threshold": float(thr_rare),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")

    # Save embeddings subsample for reproducibility
    np.savez_compressed(
        out / "embeddings_subsample.npz",
        idx=viz_idx,
        emb_large=emb_l[viz_idx],
        emb_ft=emb_f[viz_idx],
        types=types[viz_idx],
        bodies=bodies[viz_idx],
        err_large=err_l[viz_idx],
        err_ft=err_f[viz_idx],
    )

    report = _report(blob)
    (out / "transformer_robustness_analysis.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "transformer_robustness_analysis.md").write_text(
        report, encoding="utf-8"
    )

    print("\n=== PHASE 2 ROBUSTNESS ANALYSIS ===")
    print(f"  Large Final RMSE={overall['large']['rmse']:.2f} type_macro={overall['type_macro_large']['rmse']:.2f}")
    print(f"  FT    Final RMSE={overall['ft']['rmse']:.2f} type_macro={overall['type_macro_ft']['rmse']:.2f}")
    print(f"  Silhouette Large={geom['large']['silhouette']:.3f} FT={geom['ft']['silhouette']:.3f}")
    print(f"  Inter/intra Large={geom['large']['inter_intra_ratio']:.3f} FT={geom['ft']['inter_intra_ratio']:.3f}")
    print(f"  FT better abs-err fraction={ft_gain['frac_ft_lower_abs_err']:.3f}")
    print(f"  results={out}")


def _type_macro(y, p, types, min_n=MIN_TYPE_N):
    rows = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < min_n:
            continue
        rows.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return {"rmse": float(np.mean(rows)) if rows else float("nan"), "n_types": len(rows)}


def _body_macro(y, p, bodies, min_n=100):
    rows = []
    for b in np.unique(bodies.astype(str)):
        m = bodies.astype(str) == b
        if m.sum() < min_n:
            continue
        rows.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
    return {"rmse": float(np.mean(rows)) if rows else float("nan"), "n_classes": len(rows)}


def _group_err(y, p_l, p_f, groups, min_n=50, train_freq=None):
    out = []
    for g in np.unique(np.asarray(groups).astype(str)):
        m = np.asarray(groups).astype(str) == g
        if m.sum() < min_n:
            continue
        row = {
            "group": g,
            "n": int(m.sum()),
            "large_rmse": float(np.sqrt(np.mean((p_l[m] - y[m]) ** 2))),
            "ft_rmse": float(np.sqrt(np.mean((p_f[m] - y[m]) ** 2))),
            "large_mae": float(np.mean(np.abs(p_l[m] - y[m]))),
            "ft_mae": float(np.mean(np.abs(p_f[m] - y[m]))),
            "delta_rmse_ft_minus_large": float(
                np.sqrt(np.mean((p_f[m] - y[m]) ** 2)) - np.sqrt(np.mean((p_l[m] - y[m]) ** 2))
            ),
            "frac_ft_better": float(np.mean(np.abs(p_f[m] - y[m]) < np.abs(p_l[m] - y[m]))),
        }
        if train_freq is not None and g in train_freq:
            row["train_n"] = train_freq[g]
        out.append(row)
    return sorted(out, key=lambda r: r["delta_rmse_ft_minus_large"])


def _duration_bins(d):
    h = d / 3600.0
    out = []
    for v in h:
        if not np.isfinite(v):
            out.append("unknown")
        elif v < 2:
            out.append("short_<2h")
        elif v < 5:
            out.append("medium_2-5h")
        elif v < 8:
            out.append("long_5-8h")
        else:
            out.append("ultralong_>=8h")
    return np.array(out)


def _fuel_bins(y):
    out = []
    for v in y:
        if v < 200:
            out.append("fuel_<200")
        elif v < 500:
            out.append("fuel_200-500")
        elif v < 1000:
            out.append("fuel_500-1000")
        elif v < 2000:
            out.append("fuel_1000-2000")
        else:
            out.append("fuel_>=2000")
    return np.array(out)


def _alt_bins(alt):
    out = []
    for v in alt:
        if not np.isfinite(v):
            out.append("alt_unknown")
        elif v < 3000:
            out.append("alt_<3km")
        elif v < 8000:
            out.append("alt_3-8km")
        elif v < 11000:
            out.append("alt_8-11km")
        else:
            out.append("alt_>=11km")
    return np.array(out)


def _dist_to_group_centroid(emb, types, common_types, rare_mask):
    z = StandardScaler().fit_transform(emb)
    cents = []
    for t in common_types:
        m = types.astype(str) == t
        if m.sum() < 10:
            continue
        cents.append(z[m].mean(axis=0))
    if not cents or not rare_mask.any():
        return float("nan")
    C = np.stack(cents, axis=0)
    zr = z[rare_mask]
    # mean distance to nearest common centroid
    d = np.linalg.norm(zr[:, None, :] - C[None, :, :], axis=-1).min(axis=1)
    return float(np.mean(d))


def _hypotheses(overall, geom, ft_gain, err_loc, stability, phys_share):
    hyps = []
    # H1 representation structure
    if geom["ft"]["silhouette"] > geom["large"]["silhouette"] + 0.02:
        hyps.append(
            {
                "id": "H1_type_clustered_ft",
                "claim": "FT embeddings cluster more by aircraft type than Large (higher silhouette).",
                "support": f"silhouette FT={geom['ft']['silhouette']:.3f} vs Large={geom['large']['silhouette']:.3f}",
                "status": "supported" if geom["ft"]["silhouette"] > geom["large"]["silhouette"] else "not_supported",
            }
        )
    else:
        hyps.append(
            {
                "id": "H1_type_clustered_ft",
                "claim": "FT embeddings are more type-separated than Large.",
                "support": f"silhouette FT={geom['ft']['silhouette']:.3f} vs Large={geom['large']['silhouette']:.3f}",
                "status": "not_supported"
                if geom["ft"]["silhouette"] <= geom["large"]["silhouette"]
                else "supported",
            }
        )

    hyps.append(
        {
            "id": "H2_smoother_geometry",
            "claim": "FT has higher inter/intra type distance ratio (more separable types).",
            "support": f"inter/intra FT={geom['ft']['inter_intra_ratio']:.3f} Large={geom['large']['inter_intra_ratio']:.3f}",
            "status": "supported"
            if geom["ft"]["inter_intra_ratio"] > geom["large"]["inter_intra_ratio"]
            else "not_supported",
        }
    )
    hyps.append(
        {
            "id": "H3_ft_helps_rare_or_heavy",
            "claim": "FT reduces error more on rare and/or heavy types than on common/narrow.",
            "support": (
                f"Δ|err| FT-Large rare={ft_gain['mean_abs_err_delta_on_rare']:+.2f} "
                f"common={ft_gain['mean_abs_err_delta_on_common']:+.2f} "
                f"heavy={ft_gain['mean_abs_err_delta_heavy']:+.2f} "
                f"narrow={ft_gain['mean_abs_err_delta_narrow']:+.2f}"
            ),
            "status": "supported"
            if (
                ft_gain["mean_abs_err_delta_on_rare"] < ft_gain["mean_abs_err_delta_on_common"]
                or ft_gain["mean_abs_err_delta_heavy"] < ft_gain["mean_abs_err_delta_narrow"]
            )
            else "not_supported",
        }
    )
    hyps.append(
        {
            "id": "H4_type_macro_from_hard_types",
            "claim": "FT type-macro gain comes from improving a subset of hard aircraft types.",
            "support": f"types FT better type-RMSE: {ft_gain['types_where_ft_type_rmse_better']}",
            "status": "supported" if len(ft_gain["types_where_ft_type_rmse_better"]) > 0 else "not_supported",
        }
    )
    hyps.append(
        {
            "id": "H5_feature_use_differs",
            "claim": "MLP and FT emphasize different numeric features (attribution rank correlation < 0.7).",
            "support": f"Spearman attr Large vs FT={stability['cross_model_spearman']:.3f}; "
            f"physics share L={phys_share['large_physics_share']:.3f} FT={phys_share['ft_physics_share']:.3f}",
            "status": "supported"
            if stability["cross_model_spearman"] < 0.7
            else "not_supported",
        }
    )
    return hyps


def _plots(
    pca_l, pca_f, umap_l, umap_f, tsne_l, tsne_f, meta, tsne_pos, err_loc, attr_table,
    overall, geom, ft_gain, plots, fig_dir,
):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_p2_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    def scatter2(xy, color_key, title, fname, cmap="tab20"):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        c = meta[color_key]
        if color_key in ("err_l", "err_f", "train_n"):
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=4, alpha=0.5, cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046)
        elif color_key == "rare":
            ax.scatter(xy[:, 0], xy[:, 1], c=np.where(c, 1, 0), s=4, alpha=0.5, cmap="coolwarm", rasterized=True)
        else:
            # categorical: encode top types
            labs = c.astype(str)
            top = [t for t, _ in sorted(
                ((t, (labs == t).sum()) for t in np.unique(labs)), key=lambda x: -x[1]
            )[:12]]
            lab_map = {t: i for i, t in enumerate(top)}
            cols = np.array([lab_map.get(t, -1) for t in labs])
            ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=4, alpha=0.5, cmap=cmap, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        save(fig, fname)

    scatter2(pca_l, "types", "Large MLP PCA colored by type", "pca_large_type")
    scatter2(pca_f, "types", "FT PCA colored by type", "pca_ft_type")
    scatter2(pca_l, "bodies", "Large MLP PCA by body class", "pca_large_body")
    scatter2(pca_f, "bodies", "FT PCA by body class", "pca_ft_body")
    scatter2(pca_l, "err_l", "Large MLP PCA by |error|", "pca_large_error")
    scatter2(pca_f, "err_f", "FT PCA by |error|", "pca_ft_error")
    scatter2(pca_l, "train_n", "Large MLP PCA by train frequency", "pca_large_freq")
    scatter2(pca_f, "train_n", "FT PCA by train frequency", "pca_ft_freq")

    if umap_l is not None:
        scatter2(umap_l, "types", "Large MLP UMAP by type", "umap_large_type")
        scatter2(umap_f, "types", "FT UMAP by type", "umap_ft_type")
        scatter2(umap_l, "bodies", "Large MLP UMAP by body", "umap_large_body")
        scatter2(umap_f, "bodies", "FT UMAP by body", "umap_ft_body")

    # t-SNE uses subset meta
    meta_tsne = {k: (v[tsne_pos] if hasattr(v, "__getitem__") else v) for k, v in meta.items()}
    # fix scatter2 to use external meta - redefine local

    def scatter_tsne(xy, color_key, title, fname):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        c = meta_tsne[color_key]
        if color_key in ("err_l", "err_f", "train_n"):
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=4, alpha=0.5, cmap="viridis", rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=0.046)
        else:
            labs = c.astype(str)
            top = [t for t, _ in sorted(((t, (labs == t).sum()) for t in np.unique(labs)), key=lambda x: -x[1])[:12]]
            lab_map = {t: i for i, t in enumerate(top)}
            cols = np.array([lab_map.get(t, -1) for t in labs])
            ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=4, alpha=0.5, cmap="tab20", rasterized=True)
        ax.set_title(title)
        save(fig, fname)

    scatter_tsne(tsne_l, "types", "Large MLP t-SNE by type", "tsne_large_type")
    scatter_tsne(tsne_f, "types", "FT t-SNE by type", "tsne_ft_type")

    # Error by type bar
    by_t = err_loc["by_type"][:16]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(by_t))
    ax.bar(x - 0.2, [r["large_rmse"] for r in by_t], 0.4, label="Large")
    ax.bar(x + 0.2, [r["ft_rmse"] for r in by_t], 0.4, label="FT")
    ax.set_xticks(x)
    ax.set_xticklabels([r["group"] for r in by_t], rotation=45, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Per-type RMSE (sorted by FT−Large)")
    ax.legend()
    save(fig, "error_by_type")

    # body / phase / rare
    for key, title in [
        ("by_body", "Body class RMSE"),
        ("by_phase", "Phase RMSE"),
        ("by_rare", "Rare vs common type RMSE"),
        ("by_duration_bin", "Duration bin RMSE"),
    ]:
        rows = err_loc[key]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["large_rmse"] for r in rows], 0.4, label="Large")
        ax.bar(x + 0.2, [r["ft_rmse"] for r in rows], 0.4, label="FT")
        ax.set_xticks(x)
        ax.set_xticklabels([r["group"] for r in rows], rotation=20, ha="right")
        ax.set_ylabel("RMSE (kg)")
        ax.set_title(title)
        ax.legend()
        save(fig, f"error_{key}")

    # attribution top 12
    top = attr_table.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(top))
    ax.barh(ypos - 0.2, top["large_attr"].to_numpy(), 0.4, label="Large")
    ax.barh(ypos + 0.2, top["ft_attr"].to_numpy(), 0.4, label="FT")
    ax.set_yticks(ypos)
    ax.set_yticklabels(top["feature"].to_list(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized |grad×input|")
    ax.set_title("Top features by attribution difference")
    ax.legend()
    save(fig, "feature_attribution")

    # geometry bars
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["silhouette", "inter_intra_ratio", "local_type_consistency"]
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, [geom["large"][m] for m in metrics], 0.4, label="Large")
    ax.bar(x + 0.2, [geom["ft"][m] for m in metrics], 0.4, label="FT")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_title("Representation geometry metrics")
    ax.legend()
    save(fig, "geometry_metrics")


def _report(blob: dict[str, Any]) -> str:
    o = blob["overall"]
    g = blob["geometry"]
    f = blob["ft_gain"]
    s = blob["attribution_stability"]
    p = blob["physics_share"]
    lines = [
        "# Phase 2 — Understanding Transformer Robustness",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "**Status:** Analysis only — no training; frozen Large MLP + FT-Transformer + Teacher",
        "",
        "## Central question",
        "",
        "Why does architecture ranking reverse?",
        "",
        "| Protocol | Ranking |",
        "|----------|---------|",
        "| Flight Holdout | Teacher > Large MLP > XLarge > FT |",
        "| Type-macro | Teacher > **FT** > Large MLP > XLarge |",
        "",
        "---",
        "",
        "## Overall metrics (this re-run on Final)",
        "",
        f"| Model | Final RMSE | Type-macro | Body-macro |",
        f"|-------|-----------:|-----------:|-----------:|",
        f"| Large MLP | {o['large']['rmse']:.2f} | {o['type_macro_large']['rmse']:.2f} | {o['body_macro_large']['rmse']:.2f} |",
        f"| FT-Transformer | {o['ft']['rmse']:.2f} | {o['type_macro_ft']['rmse']:.2f} | {o['body_macro_ft']['rmse']:.2f} |",
        "",
        f"Fraction of intervals with lower |error| for FT: **{f['frac_ft_lower_abs_err']:.3f}**",
        "",
        f"Mean |err| FT − Large: overall **{f['mean_abs_err_delta_ft_minus_large']:+.2f}**, "
        f"rare **{f['mean_abs_err_delta_on_rare']:+.2f}**, common **{f['mean_abs_err_delta_on_common']:+.2f}**, "
        f"heavy **{f['mean_abs_err_delta_heavy']:+.2f}**, narrow **{f['mean_abs_err_delta_narrow']:+.2f}**",
        "",
        "---",
        "",
        "## Study A — Representation geometry",
        "",
        f"| Metric | Large MLP | FT-Transformer |",
        f"|--------|----------:|---------------:|",
        f"| Silhouette (type) | {g['large']['silhouette']:.4f} | {g['ft']['silhouette']:.4f} |",
        f"| Davies–Bouldin (↓ better) | {g['large']['davies_bouldin']:.4f} | {g['ft']['davies_bouldin']:.4f} |",
        f"| Mean intra-type dist | {g['large']['mean_intra_type_dist']:.4f} | {g['ft']['mean_intra_type_dist']:.4f} |",
        f"| Mean inter-type dist | {g['large']['mean_inter_type_dist']:.4f} | {g['ft']['mean_inter_type_dist']:.4f} |",
        f"| Inter/intra ratio | {g['large']['inter_intra_ratio']:.4f} | {g['ft']['inter_intra_ratio']:.4f} |",
        f"| Local type consistency (5-NN) | {g['large']['local_type_consistency']:.4f} | {g['ft']['local_type_consistency']:.4f} |",
        f"| Mean dist (rare → common centroids) | {g['large']['mean_dist_to_common_centroid']:.4f} | {g['ft']['mean_dist_to_common_centroid']:.4f} |",
        f"| PCA var explained (PC1, PC2) | {g['large']['pca_var_explained']} | {g['ft']['pca_var_explained']} |",
        "",
        "Embeddings: penultimate layer (MLP backbone; FT CLS after LayerNorm). Metrics on standardized embeddings; silhouette uses up to 5k subsample.",
        "",
        "### Figures (geometry)",
        "",
        "![pca L type](figures/fig_p2_pca_large_type.png)",
        "",
        "![pca FT type](figures/fig_p2_pca_ft_type.png)",
        "",
        "![umap L](figures/fig_p2_umap_large_type.png)",
        "",
        "![umap FT](figures/fig_p2_umap_ft_type.png)",
        "",
        "![tsne L](figures/fig_p2_tsne_large_type.png)",
        "",
        "![tsne FT](figures/fig_p2_tsne_ft_type.png)",
        "",
        "![geom](figures/fig_p2_geometry_metrics.png)",
        "",
        "---",
        "",
        "## Study B — Error localization",
        "",
        "### Types where FT has lower type-RMSE",
        "",
        f"{f['types_where_ft_type_rmse_better']}",
        "",
        "### Types where Large has lower type-RMSE",
        "",
        f"{f['types_where_large_type_rmse_better']}",
        "",
        "Full tables: `results/distillation/transformer_robustness/error_by_*.csv`",
        "",
        "![err type](figures/fig_p2_error_by_type.png)",
        "",
        "![err body](figures/fig_p2_error_by_body.png)",
        "",
        "![err phase](figures/fig_p2_error_by_phase.png)",
        "",
        "![err rare](figures/fig_p2_error_by_rare.png)",
        "",
        "![err dur](figures/fig_p2_error_by_duration_bin.png)",
        "",
        "---",
        "",
        "## Study C — Feature utilization",
        "",
        f"Method: mean |grad × input| on numeric features (n≈{ATTR_N} samples), L1-normalized.",
        "",
        f"| Stability / agreement | Value |",
        f"|----------------------|------:|",
        f"| Large half-split Spearman | {s['large_half_spearman']:.3f} |",
        f"| FT half-split Spearman | {s['ft_half_spearman']:.3f} |",
        f"| Cross-model Spearman | {s['cross_model_spearman']:.3f} |",
        f"| Large physics-like share | {p['large_physics_share']:.3f} |",
        f"| FT physics-like share | {p['ft_physics_share']:.3f} |",
        "",
        "Top features by |Large−FT| attribution:",
        "",
        "| Feature | Large | FT | |diff| |",
        "|---------|------:|---:|------:|",
    ]
    for r in blob["feature_attribution_top"][:12]:
        lines.append(
            f"| {r['feature']} | {r['large_attr']:.4f} | {r['ft_attr']:.4f} | {r['abs_diff']:.4f} |"
        )
    lines += [
        "",
        "![attr](figures/fig_p2_feature_attribution.png)",
        "",
        "---",
        "",
        "## Evidence-supported hypotheses",
        "",
    ]
    for h in blob["hypotheses"]:
        lines.append(f"### {h['id']} — **{h['status']}**")
        lines.append("")
        lines.append(f"**Claim:** {h['claim']}")
        lines.append("")
        lines.append(f"**Evidence:** {h['support']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Synthesis: why the ranking reverses",
        "",
        "Evidence-based synthesis (not speculation beyond measurements):",
        "",
        "1. **Overall vs entity-equal metrics.** Large wins on frequency-weighted Final RMSE; FT wins when each aircraft type is weighted equally (type-macro). That alone can reverse rankings if FT is relatively better on lower-frequency / higher-error types even when worse on dominant narrow-bodies (A20N/A320 mass).",
        "",
        "2. **Where FT gains.** See type list and rare/heavy Δ|err| above — interpret relative to Large, not absolute teacher.",
        "",
        "3. **Representation structure.** Compare silhouette / inter-intra / local consistency between Large and FT (Study A table). Higher type separation or consistency supports a geometry story; lower does not.",
        "",
        "4. **Feature use.** Cross-model attribution correlation and physics-share differences (Study C) indicate whether the models rely on different numeric cues.",
        "",
        "5. **Not explained by training new weights.** Both models are frozen KD students on the same α/β and data; differences are architectural inductive bias + representation geometry under the same supervision.",
        "",
        "---",
        "",
        "## Open questions & limitations",
        "",
        "- Post-hoc type-macro ≠ re-trained leave-one-type-out.",
        "- Grad×input is a local linearization; not causal feature importance.",
        "- Attention maps not fully dissected (FT token attention avg left for follow-up).",
        "- Embedding metrics are sensitive to standardization and class imbalance handling.",
        "- XLarge not re-analyzed (ranking already known from Phase 0).",
        "- No new models trained; mechanisms are descriptive.",
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "`results/distillation/transformer_robustness/`",
        "",
        f"*Generated {blob['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
