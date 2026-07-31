"""Phase 3 — Mechanism validation (A/B/C workstreams).

Trains nothing itself (expects physics ablations from 15_train_physics_ablation.py
if available). Uses frozen full-feature Large/FT + optional nophysics models.
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData, load_feature_cols
from aerotwin.distillation.metrics import regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.distillation.physics_features import (
    classify_numeric,
    is_physics_feature,
    nophysics_feature_cols,
    split_features,
)
from aerotwin.engine.gap_closing import aircraft_class, clean_featured, ensure_features, group_phase
from aerotwin.engine.mass_model import enrich_mass_from_columns
from aerotwin.engine.statistical_protocol import RANDOM_STATE, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("mechanism")

LARGE_CKPT = ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt"
FT_CKPT = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt"
FT_CFG = ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json"
ABL_ROOT = ROOT / "results/distillation/mechanism_validation/physics_ablation"
OUT = ROOT / "results/distillation/mechanism_validation"
MIN_TYPE_N = 50
N_BOOT = 500


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


def _corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(m.sum())}
    pr, pp = stats.pearsonr(x[m], y[m])
    sr, sp = stats.spearmanr(x[m], y[m])
    return {
        "pearson": float(pr),
        "pearson_p": float(pp),
        "spearman": float(sr),
        "spearman_p": float(sp),
        "n": int(m.sum()),
    }


def _grad_attr(model, x, n_num, device, idx=None, max_n=2000):
    rng = np.random.default_rng(42)
    if idx is None:
        idx = rng.choice(len(x), size=min(max_n, len(x)), replace=False)
    xs = torch.tensor(x[idx], dtype=torch.float32, device=device, requires_grad=True)
    model.eval()
    pred = model(xs)
    pred.sum().backward()
    g = xs.grad.detach().cpu().numpy()
    xi = xs.detach().cpu().numpy()
    attr = np.mean(np.abs(g[:, :n_num] * xi[:, :n_num]), axis=0)
    s = attr.sum()
    return attr / s if s > 0 else attr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
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

    # ---- Data full features ----
    data_full = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )
    final = _prepare(ROOT / "featured_dataset_final.parquet")
    x_full, y = _transform(final, data_full)
    n_num_full = len(data_full.numeric_cols)
    types = final["aircraft_type"].cast(pl.Utf8).fill_null("?").to_numpy()
    bodies = np.array([_body(t) for t in types])
    phases = group_phase(final).astype(str)
    duration = final["duration_s"].to_numpy().astype(np.float64)
    physics_fuel = (
        final["physics_fuel_kg"].to_numpy().astype(np.float64)
        if "physics_fuel_kg" in final.columns
        else np.full(len(y), np.nan)
    )

    train_df = pl.read_parquet(ROOT / "distillation_dataset.parquet")
    freq_map = {
        str(r["aircraft_type"]): int(r["len"])
        for r in train_df.group_by("aircraft_type").len().iter_rows(named=True)
    }
    train_n = np.array([freq_map.get(str(t), 0) for t in types], dtype=np.float64)
    thr_rare = float(np.percentile(list({t: freq_map.get(str(t), 0) for t in np.unique(types)}.values()), 33))
    rare = train_n <= thr_rare

    # Load full models
    LOGGER.info("Load full-feature Large + FT")
    large = StudentMLP(data_full.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
    large.load_state_dict(torch.load(LARGE_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    large.to(device).eval()
    sc = StudentConfig.from_mapping(json.loads(FT_CFG.read_text(encoding="utf-8")))
    sc.in_dim = data_full.in_dim
    sc.n_num_features = n_num_full
    sc.cat_cardinalities = [len(c) for c in data_full.ohe.categories_]
    ft = build_student(sc, in_dim=data_full.in_dim)
    ft.load_state_dict(torch.load(FT_CKPT, map_location=device, weights_only=False)["model_state_dict"])
    ft.to(device).eval()

    emb_l, pred_l = _encode_predict(large, x_full, device)
    emb_f, pred_f = _encode_predict(ft, x_full, device)

    full_metrics = {
        "large": {
            "final": regression_metrics(y, pred_l),
            "type_macro": _type_macro(y, pred_l, types)[0],
            "body_macro": _body_macro(y, pred_l, bodies)[0],
        },
        "ft": {
            "final": regression_metrics(y, pred_f),
            "type_macro": _type_macro(y, pred_f, types)[0],
            "body_macro": _body_macro(y, pred_f, bodies)[0],
        },
    }

    # ============================================================
    # A1/A2 — Physics ablation models (if trained)
    # ============================================================
    keep_feats = nophysics_feature_cols(load_feature_cols(ROOT))
    data_np = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet",
        root=ROOT,
        feature_cols=keep_feats,
        val_fraction=0.2,
        seed=42,
    )
    x_np, y_np = _transform(final, data_np)
    assert np.allclose(y_np, y)

    ablation = {"available": {}, "metrics": {}}
    for run, arch in [("large_nophysics", "large"), ("ft_nophysics", "ft")]:
        ckpt = ABL_ROOT / run / "best_model.pt"
        if not ckpt.exists():
            LOGGER.warning("Missing ablation checkpoint %s — skip A1/A2 for %s", ckpt, run)
            continue
        LOGGER.info("Eval ablation %s", run)
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        if arch == "large":
            m = StudentMLP(data_np.in_dim, hidden_dims=(1792, 1024), dropout=0.1)
            m.load_state_dict(blob["model_state_dict"])
        else:
            cfg = StudentConfig(
                architecture="ft_transformer",
                d_token=192,
                n_blocks=3,
                n_heads=8,
                n_num_features=len(data_np.numeric_cols),
                cat_cardinalities=[len(c) for c in data_np.ohe.categories_],
            )
            m = build_student(cfg, in_dim=data_np.in_dim)
            m.load_state_dict(blob["model_state_dict"])
        m.to(device).eval()
        _, p = _encode_predict(m, x_np, device)
        ablation["available"][run] = True
        ablation["metrics"][run] = {
            "final": regression_metrics(y, p),
            "type_macro": _type_macro(y, p, types)[0],
            "body_macro": _body_macro(y, p, bodies)[0],
            "n_params": int(blob.get("n_params") or sum(pp.numel() for pp in m.parameters())),
        }
        del m

    # Compare full vs nophysics
    a1_a2 = {}
    if "large_nophysics" in ablation["metrics"]:
        a1_a2["large"] = {
            "full_final": full_metrics["large"]["final"]["rmse"],
            "nophys_final": ablation["metrics"]["large_nophysics"]["final"]["rmse"],
            "delta_final": ablation["metrics"]["large_nophysics"]["final"]["rmse"]
            - full_metrics["large"]["final"]["rmse"],
            "full_type_macro": full_metrics["large"]["type_macro"],
            "nophys_type_macro": ablation["metrics"]["large_nophysics"]["type_macro"],
            "delta_type_macro": ablation["metrics"]["large_nophysics"]["type_macro"]
            - full_metrics["large"]["type_macro"],
        }
    if "ft_nophysics" in ablation["metrics"]:
        a1_a2["ft"] = {
            "full_final": full_metrics["ft"]["final"]["rmse"],
            "nophys_final": ablation["metrics"]["ft_nophysics"]["final"]["rmse"],
            "delta_final": ablation["metrics"]["ft_nophysics"]["final"]["rmse"]
            - full_metrics["ft"]["final"]["rmse"],
            "full_type_macro": full_metrics["ft"]["type_macro"],
            "nophys_type_macro": ablation["metrics"]["ft_nophysics"]["type_macro"],
            "delta_type_macro": ablation["metrics"]["ft_nophysics"]["type_macro"]
            - full_metrics["ft"]["type_macro"],
        }
    if "large" in a1_a2 and "ft" in a1_a2:
        a1_a2["relative"] = {
            "large_delta_type": a1_a2["large"]["delta_type_macro"],
            "ft_delta_type": a1_a2["ft"]["delta_type_macro"],
            "large_more_sensitive_to_physics_removal": a1_a2["large"]["delta_type_macro"]
            > a1_a2["ft"]["delta_type_macro"] + 1.0,
        }

    # ============================================================
    # A3 — Physics reliability correlation (per type)
    # ============================================================
    type_rows = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < MIN_TYPE_N:
            continue
        phys_err = np.abs(physics_fuel[m] - y[m])
        phys_rmse = float(np.sqrt(np.mean((physics_fuel[m] - y[m]) ** 2)))
        l_rmse = float(np.sqrt(np.mean((pred_l[m] - y[m]) ** 2)))
        f_rmse = float(np.sqrt(np.mean((pred_f[m] - y[m]) ** 2)))
        type_rows.append(
            {
                "aircraft_type": t,
                "body_class": _body(t),
                "n": int(m.sum()),
                "train_n": int(freq_map.get(t, 0)),
                "physics_rmse": phys_rmse,
                "physics_mae": float(np.mean(phys_err)),
                "large_rmse": l_rmse,
                "ft_rmse": f_rmse,
                "ft_advantage": l_rmse - f_rmse,  # positive => FT better
                "large_minus_physics": l_rmse - phys_rmse,
                "log_train_n": float(np.log1p(freq_map.get(t, 0))),
            }
        )
    type_df = pl.DataFrame(type_rows)
    type_df.write_csv(out / "type_level_physics_table.csv")

    phys = np.array([r["physics_rmse"] for r in type_rows])
    ft_adv = np.array([r["ft_advantage"] for r in type_rows])
    large_rmse = np.array([r["large_rmse"] for r in type_rows])
    train_freq = np.array([r["train_n"] for r in type_rows], dtype=float)

    a3 = {
        "physics_error_vs_ft_advantage": _corr(phys, ft_adv),
        "physics_error_vs_large_rmse": _corr(phys, large_rmse),
        "physics_error_vs_ft_rmse": _corr(phys, np.array([r["ft_rmse"] for r in type_rows])),
        "physics_error_vs_log_train_n": _corr(phys, np.log1p(train_freq)),
        "ft_advantage_vs_log_train_n": _corr(ft_adv, np.log1p(train_freq)),
        "n_types": len(type_rows),
    }

    # ============================================================
    # A4 — Attribution shift by subgroup
    # ============================================================
    LOGGER.info("Attribution shift by subgroup")
    feat_names = list(data_full.numeric_cols)
    buckets = [classify_numeric(n) for n in feat_names]
    subgroups = {
        "all": np.ones(len(y), dtype=bool),
        "common": ~rare,
        "rare": rare,
        "heavy": bodies == "widebody_heavy",
        "narrow": bodies == "narrowbody",
    }
    a4 = {}
    for name, mask in subgroups.items():
        if mask.sum() < 200:
            continue
        idx = np.flatnonzero(mask)
        al = _grad_attr(large, x_full, n_num_full, device, idx=idx[:2000] if len(idx) > 2000 else idx)
        af = _grad_attr(ft, x_full, n_num_full, device, idx=idx[:2000] if len(idx) > 2000 else idx)
        share_l = {b: float(al[[i for i, bb in enumerate(buckets) if bb == b]].sum()) for b in set(buckets)}
        share_f = {b: float(af[[i for i, bb in enumerate(buckets) if bb == b]].sum()) for b in set(buckets)}
        a4[name] = {
            "n": int(mask.sum()),
            "large_bucket_share": share_l,
            "ft_bucket_share": share_f,
            "large_physics_share": share_l.get("physics", 0.0),
            "ft_physics_share": share_f.get("physics", 0.0),
            "large_trajectory_share": share_l.get("trajectory", 0.0),
            "ft_trajectory_share": share_f.get("trajectory", 0.0),
        }

    # ============================================================
    # B1 — Normalized representation geometry
    # ============================================================
    LOGGER.info("Representation geometry (normalized)")
    z_l = StandardScaler().fit_transform(emb_l)
    z_f = StandardScaler().fit_transform(emb_f)
    b1 = {
        "large": _rep_geometry(z_l, types, rare, freq_map),
        "ft": _rep_geometry(z_f, types, rare, freq_map),
    }
    # scale-free: divide rare-common by mean inter-type centroid distance
    for k in ("large", "ft"):
        mid = b1[k]["mean_inter_type_centroid_dist"]
        b1[k]["rare_to_common_norm"] = b1[k]["rare_to_common_centroid_dist"] / max(mid, 1e-9)
        b1[k]["common_to_common_norm"] = b1[k]["common_to_common_centroid_dist"] / max(mid, 1e-9)

    # ============================================================
    # B2 — NN transfer for rare samples
    # ============================================================
    LOGGER.info("Nearest-neighbor transfer")
    b2 = {
        "large": _nn_transfer(z_l, types, rare, train_n, y, pred_l),
        "ft": _nn_transfer(z_f, types, rare, train_n, y, pred_f),
    }

    # ============================================================
    # B3 — Local neighborhood structure
    # ============================================================
    b3 = {
        "large": _neighborhood(z_l, types, k=10),
        "ft": _neighborhood(z_f, types, k=10),
    }

    # ============================================================
    # B4 — Representation stability under perturbation
    # ============================================================
    LOGGER.info("Representation stability")
    b4 = {
        "large": _stability(large, x_full, n_num_full, device),
        "ft": _stability(ft, x_full, n_num_full, device),
    }

    # ============================================================
    # B5 — Geometry vs robustness per type
    # ============================================================
    b5_rows = []
    for r in type_rows:
        t = r["aircraft_type"]
        m = types.astype(str) == t
        # distance of this type centroid to common centroids
        for model_name, z in [("large", z_l), ("ft", z_f)]:
            cent = z[m].mean(axis=0)
            common = [tt for tt, n in sorted(freq_map.items(), key=lambda kv: -kv[1])[:5]]
            cds = []
            for tt in common:
                mm = types.astype(str) == tt
                if mm.sum() < 10:
                    continue
                cds.append(float(np.linalg.norm(cent - z[mm].mean(axis=0))))
            b5_rows.append(
                {
                    "aircraft_type": t,
                    "model": model_name,
                    "centroid_dist_to_common": float(np.mean(cds)) if cds else float("nan"),
                    "type_rmse": r["large_rmse"] if model_name == "large" else r["ft_rmse"],
                    "ft_advantage": r["ft_advantage"],
                    "train_n": r["train_n"],
                    "physics_rmse": r["physics_rmse"],
                }
            )
    b5_df = pl.DataFrame(b5_rows)
    b5_df.write_csv(out / "geometry_vs_robustness.csv")
    # FT advantage vs FT centroid distance / vs train_n / vs physics
    ft_only = [r for r in b5_rows if r["model"] == "ft"]
    b5 = {
        "ft_centroid_dist_vs_ft_rmse": _corr(
            np.array([r["centroid_dist_to_common"] for r in ft_only]),
            np.array([r["type_rmse"] for r in ft_only]),
        ),
        "ft_centroid_dist_vs_ft_advantage": _corr(
            np.array([r["centroid_dist_to_common"] for r in ft_only]),
            np.array([r["ft_advantage"] for r in ft_only]),
        ),
        "train_n_vs_ft_advantage": _corr(
            np.log1p(np.array([r["train_n"] for r in ft_only], dtype=float)),
            np.array([r["ft_advantage"] for r in ft_only]),
        ),
        "physics_rmse_vs_ft_advantage": a3["physics_error_vs_ft_advantage"],
    }

    # ============================================================
    # C1 — Variance decomposition of FT advantage
    # ============================================================
    LOGGER.info("Variance decomposition")
    # per-type regression
    X_list = []
    y_adv = []
    for r in type_rows:
        t = r["aircraft_type"]
        # get ft centroid dist
        cd = next(
            (row["centroid_dist_to_common"] for row in b5_rows if row["aircraft_type"] == t and row["model"] == "ft"),
            np.nan,
        )
        X_list.append(
            [
                r["physics_rmse"],
                np.log1p(r["train_n"]),
                1.0 if r["body_class"] == "widebody_heavy" else 0.0,
                cd if np.isfinite(cd) else 0.0,
            ]
        )
        y_adv.append(r["ft_advantage"])
    Xmat = np.asarray(X_list, dtype=float)
    yvec = np.asarray(y_adv, dtype=float)
    # standardize predictors for comparable coefs
    Xs = StandardScaler().fit_transform(Xmat)
    reg = LinearRegression().fit(Xs, yvec)
    yhat = reg.predict(Xs)
    # unique variance: drop one predictor at a time
    names = ["physics_rmse", "log_train_n", "is_heavy", "centroid_dist"]
    full_r2 = float(r2_score(yvec, yhat))
    unique = {}
    for i, nm in enumerate(names):
        mask = [j for j in range(len(names)) if j != i]
        r = LinearRegression().fit(Xs[:, mask], yvec)
        unique[nm] = full_r2 - float(r2_score(yvec, r.predict(Xs[:, mask])))
    c1 = {
        "n_types": len(yvec),
        "full_r2": full_r2,
        "coefficients_standardized": {nm: float(c) for nm, c in zip(names, reg.coef_)},
        "intercept": float(reg.intercept_),
        "unique_r2_drop": unique,
        "predictor_order": names,
    }

    # ============================================================
    # C2/C3 — Evidence table + conclusion
    # ============================================================
    evidence_table, conclusion = _decide(a1_a2, a3, a4, b1, b2, b3, b4, b5, c1, full_metrics)

    # Plots
    _plots(type_rows, a4, b1, b4, a1_a2, plots, fig_dir)

    blob = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_metrics": full_metrics,
        "A_physics_ablation": a1_a2,
        "A_ablation_raw": ablation,
        "A3_physics_reliability": a3,
        "A4_attribution_shift": a4,
        "B1_geometry": b1,
        "B2_nn_transfer": b2,
        "B3_neighborhood": b3,
        "B4_stability": b4,
        "B5_geometry_vs_robustness": b5,
        "C1_variance_decomposition": c1,
        "evidence_table": evidence_table,
        "conclusion": conclusion,
        "wall_seconds": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(conclusion, indent=2, default=str), encoding="utf-8")

    report = _report(blob)
    (out / "mechanism_validation.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "mechanism_validation.md").write_text(report, encoding="utf-8")

    print("\n=== PHASE 3 MECHANISM VALIDATION ===")
    print(json.dumps(conclusion, indent=2, default=str))
    print(f"results={out}")


def _rep_geometry(z, types, rare, freq_map):
    # type centroids
    cents = {}
    within = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if m.sum() < 10:
            continue
        c = z[m].mean(axis=0)
        cents[t] = c
        within.append(float(np.mean(np.linalg.norm(z[m] - c, axis=1))))
    # inter centroid distances
    keys = list(cents.keys())
    inter = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter.append(float(np.linalg.norm(cents[keys[i]] - cents[keys[j]])))
    common = [t for t, n in sorted(freq_map.items(), key=lambda kv: -kv[1])[:5] if t in cents]
    rare_types = [t for t in cents if freq_map.get(t, 0) <= np.percentile(list(freq_map.values()), 33)]
    # rare sample to common centroids
    if rare.any() and common:
        C = np.stack([cents[t] for t in common if t in cents], axis=0)
        d = np.linalg.norm(z[rare][:, None, :] - C[None, :, :], axis=-1).min(axis=1)
        rare_common = float(np.mean(d))
    else:
        rare_common = float("nan")
    # common-common mean centroid distance
    if len(common) >= 2:
        cc = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                if common[i] in cents and common[j] in cents:
                    cc.append(float(np.linalg.norm(cents[common[i]] - cents[common[j]])))
        common_common = float(np.mean(cc)) if cc else float("nan")
    else:
        common_common = float("nan")
    return {
        "mean_within_type_var_proxy": float(np.mean(within)) if within else float("nan"),
        "mean_inter_type_centroid_dist": float(np.mean(inter)) if inter else float("nan"),
        "rare_to_common_centroid_dist": rare_common,
        "common_to_common_centroid_dist": common_common,
        "n_types_with_centroid": len(cents),
    }


def _nn_transfer(z, types, rare, train_n, y, pred, k=5):
    # for each rare sample, NN among common samples
    common_mask = ~rare
    if rare.sum() < 10 or common_mask.sum() < 50:
        return {"error": "insufficient samples"}
    nn = NearestNeighbors(n_neighbors=k).fit(z[common_mask])
    dist, ind = nn.kneighbors(z[rare])
    common_types = types[common_mask]
    rare_types = types[rare]
    # neighbor type mode matches same body class?
    same_body = []
    for i, neigh in enumerate(ind):
        rt = rare_types[i]
        ntypes = common_types[neigh]
        # body match fraction
        rb = _body(rt)
        same_body.append(float(np.mean([_body(t) == rb for t in ntypes])))
    # mean distance
    return {
        "mean_nn_dist": float(np.mean(dist)),
        "median_nn_dist": float(np.median(dist)),
        "mean_neighbor_same_body_frac": float(np.mean(same_body)),
        "n_rare": int(rare.sum()),
        "n_common": int(common_mask.sum()),
    }


def _neighborhood(z, types, k=10):
    # subsample for speed
    rng = np.random.default_rng(42)
    n = min(8000, len(z))
    idx = rng.choice(len(z), size=n, replace=False)
    zz = z[idx]
    tt = types[idx].astype(str)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(zz)
    dist, ind = nn.kneighbors(zz)
    purity = []
    for i in range(n):
        neigh = ind[i, 1:]
        purity.append(float(np.mean(tt[neigh] == tt[i])))
    # trustworthiness-like: fraction of k-NN that are same type (already purity)
    return {
        "k": k,
        "mean_type_purity": float(np.mean(purity)),
        "mean_nn_dist": float(np.mean(dist[:, 1:])),
        "n": n,
    }


def _stability(model, x, n_num, device, eps=0.05, n=1500):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x), size=min(n, len(x)), replace=False)
    xb = x[idx].copy()
    # perturb continuous (all columns slightly)
    noise = rng.normal(0, eps, size=xb.shape).astype(np.float32)
    x2 = xb + noise
    with torch.no_grad():
        e1 = model.encode(torch.as_tensor(xb, device=device)).cpu().numpy()
        e2 = model.encode(torch.as_tensor(x2, device=device)).cpu().numpy()
    # normalize by embedding scale
    move = np.linalg.norm(e1 - e2, axis=1)
    scale = np.linalg.norm(e1, axis=1) + 1e-9
    return {
        "mean_abs_move": float(np.mean(move)),
        "mean_rel_move": float(np.mean(move / scale)),
        "median_rel_move": float(np.median(move / scale)),
        "eps": eps,
        "n": int(len(idx)),
    }


def _decide(a1_a2, a3, a4, b1, b2, b3, b4, b5, c1, full_metrics):
    table = []
    # Hypothesis B — physics
    phys_support = []
    phys_contra = []
    if a1_a2.get("large"):
        if a1_a2["large"]["delta_type_macro"] < -2:
            phys_support.append(
                f"Removing physics improves Large type-macro by {-a1_a2['large']['delta_type_macro']:.1f} kg"
            )
        elif a1_a2["large"]["delta_type_macro"] > 2:
            phys_contra.append(
                f"Removing physics *worsens* Large type-macro by {a1_a2['large']['delta_type_macro']:.1f} kg"
            )
        else:
            phys_contra.append(
                f"Large type-macro change after physics removal is small ({a1_a2['large']['delta_type_macro']:+.1f} kg)"
            )
    if a1_a2.get("relative"):
        if a1_a2["relative"]["large_more_sensitive_to_physics_removal"]:
            phys_support.append("Large type-macro more sensitive to physics removal than FT")
        else:
            phys_contra.append("FT not clearly less sensitive to physics removal than Large")
    pr = a3["physics_error_vs_ft_advantage"]
    if np.isfinite(pr["spearman"]) and pr["spearman"] > 0.3 and pr["spearman_p"] < 0.1:
        phys_support.append(
            f"Type-level Spearman(physics_rmse, FT advantage)={pr['spearman']:.2f} (p={pr['spearman_p']:.3f})"
        )
    elif np.isfinite(pr["spearman"]):
        phys_contra.append(
            f"Weak/non-sig Spearman(physics_rmse, FT advantage)={pr['spearman']:.2f} (p={pr['spearman_p']:.3f})"
        )
    if a4.get("rare") and a4.get("common"):
        if a4["rare"]["large_physics_share"] > a4["common"]["large_physics_share"] + 0.05:
            phys_support.append(
                f"Large physics attr share higher on rare ({a4['rare']['large_physics_share']:.2f}) "
                f"than common ({a4['common']['large_physics_share']:.2f})"
            )
        else:
            phys_contra.append(
                f"Large physics attr share rare={a4['rare']['large_physics_share']:.2f} "
                f"vs common={a4['common']['large_physics_share']:.2f} (no large increase)"
            )

    # Hypothesis A — representation
    rep_support = []
    rep_contra = []
    if b1["ft"]["rare_to_common_norm"] < b1["large"]["rare_to_common_norm"] * 0.7:
        rep_support.append(
            f"Normalized rare→common distance lower for FT "
            f"({b1['ft']['rare_to_common_norm']:.3f} vs {b1['large']['rare_to_common_norm']:.3f})"
        )
    else:
        rep_contra.append("FT rare→common normalized distance not substantially smaller")
    if b2["ft"].get("mean_nn_dist", 1e9) < b2["large"].get("mean_nn_dist", 0):
        rep_support.append(
            f"Rare samples closer to common NN in FT space "
            f"({b2['ft'].get('mean_nn_dist'):.3f} vs {b2['large'].get('mean_nn_dist'):.3f})"
        )
    if b3["ft"]["mean_type_purity"] < b3["large"]["mean_type_purity"] - 0.02:
        rep_contra.append(
            f"FT lower type purity ({b3['ft']['mean_type_purity']:.3f} vs {b3['large']['mean_type_purity']:.3f}) "
            "— not tighter type clusters"
        )
    elif b3["ft"]["mean_type_purity"] > b3["large"]["mean_type_purity"] + 0.02:
        rep_support.append("FT higher local type purity")
    if b4["ft"]["mean_rel_move"] < b4["large"]["mean_rel_move"]:
        rep_support.append(
            f"FT more stable under input noise (rel move {b4['ft']['mean_rel_move']:.4f} "
            f"vs {b4['large']['mean_rel_move']:.4f})"
        )
    else:
        rep_contra.append(
            f"FT not more stable (rel move {b4['ft']['mean_rel_move']:.4f} vs {b4['large']['mean_rel_move']:.4f})"
        )
    gd = b5["ft_centroid_dist_vs_ft_advantage"]
    if np.isfinite(gd["spearman"]) and abs(gd["spearman"]) > 0.3:
        rep_support.append(
            f"Centroid distance correlates with FT advantage (Spearman={gd['spearman']:.2f})"
        )
    else:
        rep_contra.append(
            f"Weak centroid-distance vs FT-advantage link (Spearman={gd.get('spearman', float('nan')):.2f})"
        )

    # scores
    def score(sup, con):
        return len(sup) - 0.5 * len(con)

    s_phys = score(phys_support, phys_contra)
    s_rep = score(rep_support, rep_contra)

    table = [
        {
            "mechanism": "A_representation",
            "supporting_evidence": phys_support and rep_support,  # fix below
            "supporting": rep_support,
            "contradicting": rep_contra,
            "score": s_rep,
        },
        {
            "mechanism": "B_physics_reliance",
            "supporting": phys_support,
            "contradicting": phys_contra,
            "score": s_phys,
        },
    ]
    # clean table for JSON
    table = [
        {
            "mechanism": "Representation (Hypothesis A)",
            "supporting_evidence": rep_support,
            "contradicting_evidence": rep_contra,
            "score": s_rep,
            "confidence": "high" if abs(s_rep) >= 3 else ("medium" if abs(s_rep) >= 1.5 else "low"),
        },
        {
            "mechanism": "Physics-feature reliance (Hypothesis B)",
            "supporting_evidence": phys_support,
            "contradicting_evidence": phys_contra,
            "score": s_phys,
            "confidence": "high" if abs(s_phys) >= 3 else ("medium" if abs(s_phys) >= 1.5 else "low"),
        },
    ]

    if s_rep > s_phys + 1 and s_rep > 0:
        label = "mostly_representation"
        conf = table[0]["confidence"]
    elif s_phys > s_rep + 1 and s_phys > 0:
        label = "mostly_physics_reliance"
        conf = table[1]["confidence"]
    elif s_rep > 0 and s_phys > 0:
        label = "hybrid"
        conf = "medium"
    else:
        label = "inconclusive"
        conf = "low"

    # Phase 4 recommendation
    if label == "mostly_representation":
        phase4 = "Representation distillation / align student latents to FT-like geometry"
    elif label == "mostly_physics_reliance":
        phase4 = "Physics-reliability-aware MLP (downweight physics where OpenAP fails)"
    elif label == "hybrid":
        phase4 = "Hybrid method addressing both representation transfer and physics reliability"
    else:
        phase4 = "Return to hypothesis generation; do not implement a new learning method yet"

    # variance decomposition note
    top_pred = max(c1["unique_r2_drop"], key=c1["unique_r2_drop"].get) if c1.get("unique_r2_drop") else None

    conclusion = {
        "most_likely_mechanism": label,
        "confidence": conf,
        "representation_score": s_rep,
        "physics_score": s_phys,
        "variance_decomp_full_r2": c1.get("full_r2"),
        "variance_decomp_top_unique": top_pred,
        "variance_decomp_unique_r2": c1.get("unique_r2_drop"),
        "phase4_recommendation": phase4,
        "summary": "",
    }
    conclusion["summary"] = (
        f"Mechanism label={label} (rep_score={s_rep:.1f}, phys_score={s_phys:.1f}). "
        f"Type-level regression R²={c1.get('full_r2', float('nan')):.3f}; "
        f"top unique R² drop predictor={top_pred}. "
        f"Phase 4: {phase4}."
    )
    return table, conclusion


def _plots(type_rows, a4, b1, b4, a1_a2, plots, fig_dir):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_m3_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    phys = np.array([r["physics_rmse"] for r in type_rows])
    adv = np.array([r["ft_advantage"] for r in type_rows])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(phys, adv, s=60)
    for r in type_rows:
        ax.annotate(r["aircraft_type"], (r["physics_rmse"], r["ft_advantage"]), fontsize=7)
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("OpenAP / physics RMSE (kg)")
    ax.set_ylabel("FT advantage (Large RMSE − FT RMSE)")
    ax.set_title("Physics error vs FT type-level advantage")
    save(fig, "physics_vs_ft_advantage")

    # geometry bars
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labs = ["rare→common\n(raw)", "rare→common\n(norm)", "within-type", "inter-centroid"]
    large_v = [
        b1["large"]["rare_to_common_centroid_dist"],
        b1["large"]["rare_to_common_norm"],
        b1["large"]["mean_within_type_var_proxy"],
        b1["large"]["mean_inter_type_centroid_dist"],
    ]
    ft_v = [
        b1["ft"]["rare_to_common_centroid_dist"],
        b1["ft"]["rare_to_common_norm"],
        b1["ft"]["mean_within_type_var_proxy"],
        b1["ft"]["mean_inter_type_centroid_dist"],
    ]
    x = np.arange(len(labs))
    ax.bar(x - 0.2, large_v, 0.4, label="Large")
    ax.bar(x + 0.2, ft_v, 0.4, label="FT")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.legend()
    ax.set_title("Representation geometry (scale-sensitive + normalized)")
    save(fig, "geometry_normalized")

    # attribution physics share
    if a4:
        keys = [k for k in ["common", "rare", "heavy", "narrow"] if k in a4]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [a4[k]["large_physics_share"] for k in keys], 0.4, label="Large physics")
        ax.bar(x + 0.2, [a4[k]["ft_physics_share"] for k in keys], 0.4, label="FT physics")
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        ax.set_ylabel("Attribution share")
        ax.set_title("Physics feature attribution by subgroup")
        ax.legend()
        save(fig, "attribution_physics_share")

    # ablation deltas if present
    if a1_a2.get("large") and a1_a2.get("ft"):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labs = ["Large ΔFinal", "Large ΔType", "FT ΔFinal", "FT ΔType"]
        vals = [
            a1_a2["large"]["delta_final"],
            a1_a2["large"]["delta_type_macro"],
            a1_a2["ft"]["delta_final"],
            a1_a2["ft"]["delta_type_macro"],
        ]
        ax.bar(labs, vals, color=["#1f77b4", "#1f77b4", "#d62728", "#d62728"])
        ax.axhline(0, color="k", ls="--")
        ax.set_ylabel("RMSE change after removing physics (kg)")
        ax.set_title("Physics ablation impact (positive = worse without physics)")
        plt.xticks(rotation=15, ha="right")
        save(fig, "physics_ablation_deltas")

    # stability
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Large", "FT"], [b4["large"]["mean_rel_move"], b4["ft"]["mean_rel_move"]])
    ax.set_ylabel("Mean relative embedding movement")
    ax.set_title("Stability under input noise (ε=0.05)")
    save(fig, "stability")


def _report(blob: dict[str, Any]) -> str:
    c = blob["conclusion"]
    a3 = blob["A3_physics_reliability"]
    b1 = blob["B1_geometry"]
    lines = [
        "# Phase 3 — Mechanism Validation",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "**Status:** Complete — targeted physics ablations + representation analysis (no new architectures/KD methods)",
        "",
        "## 1. Motivation",
        "",
        "FT-Transformer underperforms Large MLP on Flight Holdout but **wins type-macro**. "
        "Two explanations compete: smoother representations (A) vs lower reliance on unreliable physics (B).",
        "",
        "## 2. Competing hypotheses",
        "",
        "| ID | Claim | Prediction |",
        "|----|-------|------------|",
        "| **A** | Representation mechanism | Geometry metrics + NN transfer explain FT type-macro gains |",
        "| **B** | Physics-feature reliance | Removing physics hurts Large more; physics error correlates with FT advantage |",
        "",
        "---",
        "",
        "## 3. Experimental design",
        "",
        "- Frozen full-feature Large + FT (deployment/reference).",
        "- **Targeted ablations:** retrain Large & FT **without physics/mass/energy features** only (same α/β, split, arch).",
        "- Type-level OpenAP RMSE vs FT advantage correlations.",
        "- Grad×input attribution by rare/common/heavy/narrow.",
        "- Normalized representation geometry, NN transfer, neighborhood purity, perturbation stability.",
        "- Type-level linear variance decomposition of FT advantage.",
        "",
        "---",
        "",
        "## 4. Physics-reliance analysis (Workstream A)",
        "",
        "### A1/A2 — Feature ablation",
        "",
    ]
    a = blob["A_physics_ablation"]
    if a.get("large"):
        lines += [
            f"| Model | Full Final | No-phys Final | Δ Final | Full type-macro | No-phys type-macro | Δ type-macro |",
            f"|-------|-----------:|--------------:|--------:|----------------:|-------------------:|-------------:|",
            f"| Large | {a['large']['full_final']:.2f} | {a['large']['nophys_final']:.2f} | {a['large']['delta_final']:+.2f} | "
            f"{a['large']['full_type_macro']:.2f} | {a['large']['nophys_type_macro']:.2f} | {a['large']['delta_type_macro']:+.2f} |",
        ]
        if a.get("ft"):
            lines.append(
                f"| FT | {a['ft']['full_final']:.2f} | {a['ft']['nophys_final']:.2f} | {a['ft']['delta_final']:+.2f} | "
                f"{a['ft']['full_type_macro']:.2f} | {a['ft']['nophys_type_macro']:.2f} | {a['ft']['delta_type_macro']:+.2f} |"
            )
        if a.get("relative"):
            lines.append(
                f"\nLarge more sensitive to physics removal on type-macro? "
                f"**{a['relative']['large_more_sensitive_to_physics_removal']}** "
                f"(ΔL={a['relative']['large_delta_type']:+.2f}, ΔFT={a['relative']['ft_delta_type']:+.2f})"
            )
    else:
        lines.append(
            "_Physics ablation checkpoints not found — run "
            "`python experiments/08_distillation/15_train_physics_ablation.py` then re-run this script._"
        )

    lines += [
        "",
        "### A3 — Physics reliability correlations (type-level)",
        "",
        f"| Relation | Pearson | Spearman | p(Spearman) | n |",
        f"|----------|--------:|---------:|------------:|--:|",
        f"| Physics RMSE → FT advantage | {a3['physics_error_vs_ft_advantage']['pearson']:.3f} | "
        f"{a3['physics_error_vs_ft_advantage']['spearman']:.3f} | "
        f"{a3['physics_error_vs_ft_advantage']['spearman_p']:.3g} | {a3['physics_error_vs_ft_advantage']['n']} |",
        f"| Physics RMSE → Large RMSE | {a3['physics_error_vs_large_rmse']['pearson']:.3f} | "
        f"{a3['physics_error_vs_large_rmse']['spearman']:.3f} | "
        f"{a3['physics_error_vs_large_rmse']['spearman_p']:.3g} | {a3['physics_error_vs_large_rmse']['n']} |",
        f"| Physics RMSE → log train n | {a3['physics_error_vs_log_train_n']['pearson']:.3f} | "
        f"{a3['physics_error_vs_log_train_n']['spearman']:.3f} | "
        f"{a3['physics_error_vs_log_train_n']['spearman_p']:.3g} | {a3['physics_error_vs_log_train_n']['n']} |",
        "",
        "![phys](figures/fig_m3_physics_vs_ft_advantage.png)",
        "",
        "### A4 — Attribution shift",
        "",
    ]
    a4 = blob["A4_attribution_shift"]
    if a4:
        lines.append("| Subgroup | Large physics share | FT physics share | Large traj share | FT traj share |")
        lines.append("|----------|--------------------:|-----------------:|-----------------:|--------------:|")
        for k, v in a4.items():
            lines.append(
                f"| {k} | {v['large_physics_share']:.3f} | {v['ft_physics_share']:.3f} | "
                f"{v['large_trajectory_share']:.3f} | {v['ft_trajectory_share']:.3f} |"
            )
    lines += [
        "",
        "![attr](figures/fig_m3_attribution_physics_share.png)",
        "",
        "---",
        "",
        "## 5. Representation analysis (Workstream B)",
        "",
        "### B1 Geometry (raw + normalized by mean inter-type centroid distance)",
        "",
        f"| Metric | Large | FT |",
        f"|--------|------:|---:|",
        f"| Rare→common centroid (raw) | {b1['large']['rare_to_common_centroid_dist']:.4f} | {b1['ft']['rare_to_common_centroid_dist']:.4f} |",
        f"| Rare→common (normalized) | {b1['large']['rare_to_common_norm']:.4f} | {b1['ft']['rare_to_common_norm']:.4f} |",
        f"| Common↔common (norm) | {b1['large']['common_to_common_norm']:.4f} | {b1['ft']['common_to_common_norm']:.4f} |",
        f"| Within-type proxy | {b1['large']['mean_within_type_var_proxy']:.4f} | {b1['ft']['mean_within_type_var_proxy']:.4f} |",
        f"| Inter-type centroid | {b1['large']['mean_inter_type_centroid_dist']:.4f} | {b1['ft']['mean_inter_type_centroid_dist']:.4f} |",
        "",
        "![geom](figures/fig_m3_geometry_normalized.png)",
        "",
        "### B2 Nearest-neighbor transfer (rare → common)",
        "",
        f"| Metric | Large | FT |",
        f"|--------|------:|---:|",
        f"| Mean NN dist | {blob['B2_nn_transfer']['large'].get('mean_nn_dist', float('nan')):.4f} | "
        f"{blob['B2_nn_transfer']['ft'].get('mean_nn_dist', float('nan')):.4f} |",
        f"| Neighbor same-body frac | {blob['B2_nn_transfer']['large'].get('mean_neighbor_same_body_frac', float('nan')):.4f} | "
        f"{blob['B2_nn_transfer']['ft'].get('mean_neighbor_same_body_frac', float('nan')):.4f} |",
        "",
        "### B3 Local neighborhood",
        "",
        f"| Metric | Large | FT |",
        f"|--------|------:|---:|",
        f"| Type purity (k=10) | {blob['B3_neighborhood']['large']['mean_type_purity']:.4f} | "
        f"{blob['B3_neighborhood']['ft']['mean_type_purity']:.4f} |",
        f"| Mean NN dist | {blob['B3_neighborhood']['large']['mean_nn_dist']:.4f} | "
        f"{blob['B3_neighborhood']['ft']['mean_nn_dist']:.4f} |",
        "",
        "### B4 Stability (ε=0.05 noise)",
        "",
        f"| Metric | Large | FT |",
        f"|--------|------:|---:|",
        f"| Mean rel embedding move | {blob['B4_stability']['large']['mean_rel_move']:.4f} | "
        f"{blob['B4_stability']['ft']['mean_rel_move']:.4f} |",
        "",
        "![stab](figures/fig_m3_stability.png)",
        "",
        "### B5 Geometry vs robustness",
        "",
        f"| Relation | Spearman | p |",
        f"|----------|---------:|--:|",
        f"| FT centroid dist → FT type RMSE | {blob['B5_geometry_vs_robustness']['ft_centroid_dist_vs_ft_rmse']['spearman']:.3f} | "
        f"{blob['B5_geometry_vs_robustness']['ft_centroid_dist_vs_ft_rmse']['spearman_p']:.3g} |",
        f"| FT centroid dist → FT advantage | {blob['B5_geometry_vs_robustness']['ft_centroid_dist_vs_ft_advantage']['spearman']:.3f} | "
        f"{blob['B5_geometry_vs_robustness']['ft_centroid_dist_vs_ft_advantage']['spearman_p']:.3g} |",
        f"| log train n → FT advantage | {blob['B5_geometry_vs_robustness']['train_n_vs_ft_advantage']['spearman']:.3f} | "
        f"{blob['B5_geometry_vs_robustness']['train_n_vs_ft_advantage']['spearman_p']:.3g} |",
        "",
        "---",
        "",
        "## 6. Joint statistical analysis (Workstream C)",
        "",
        f"Type-level linear model for FT advantage (standardized predictors).",
        "",
        f"| Item | Value |",
        f"|------|------:|",
        f"| n types | {blob['C1_variance_decomposition']['n_types']} |",
        f"| Full R² | {blob['C1_variance_decomposition']['full_r2']:.3f} |",
        "",
        "Standardized coefficients:",
        "",
    ]
    for k, v in blob["C1_variance_decomposition"]["coefficients_standardized"].items():
        lines.append(f"- `{k}`: **{v:+.3f}**")
    lines += [
        "",
        "Unique R² drop (leave-one-predictor):",
        "",
    ]
    for k, v in blob["C1_variance_decomposition"]["unique_r2_drop"].items():
        lines.append(f"- `{k}`: **{v:.3f}**")
    lines += [
        "",
        "---",
        "",
        "## 7. Competing hypothesis table",
        "",
        "| Mechanism | Supporting evidence | Contradicting evidence | Score | Confidence |",
        "|-----------|---------------------|--------------------------|------:|------------|",
    ]
    for row in blob["evidence_table"]:
        sup = "; ".join(row["supporting_evidence"]) if row["supporting_evidence"] else "—"
        con = "; ".join(row["contradicting_evidence"]) if row["contradicting_evidence"] else "—"
        lines.append(
            f"| {row['mechanism']} | {sup} | {con} | {row['score']:.1f} | {row['confidence']} |"
        )
    lines += [
        "",
        "---",
        "",
        "## 8. Scientific conclusion",
        "",
        f"**Most likely mechanism:** `{c['most_likely_mechanism']}`  ",
        f"**Confidence:** {c['confidence']}  ",
        f"**Representation score:** {c['representation_score']:.1f} · **Physics score:** {c['physics_score']:.1f}",
        "",
        c["summary"],
        "",
        "### Phase 4 recommendation (decision gate)",
        "",
        f"**{c['phase4_recommendation']}**",
        "",
        "---",
        "",
        "## 9. Limitations",
        "",
        "- Physics ablation retrains students (targeted only); stochastic training adds noise.",
        "- Grad×input is local and incomplete (no categorical OHE attribution).",
        "- Type-macro is post-hoc entity weighting, not re-trained LOTO.",
        "- Linear variance model is associative; n types is small (~15).",
        "- Attention-map mechanisms not exhaustively analyzed.",
        "",
        "## 10. Recommendation for Phase 4",
        "",
        "Do **not** implement a new algorithm until the conclusion label above is accepted. "
        "If hybrid/inconclusive, prefer further measurement over method invention.",
        "",
        f"*Generated {blob['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
