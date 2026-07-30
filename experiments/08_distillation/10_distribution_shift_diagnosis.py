"""Phase 0 — Distribution-shift diagnosis (evaluation only, no training).

Scientific question
-------------------
Does standard knowledge distillation introduce additional robustness loss under
entity-level distribution shift compared to the frozen teacher?

Protocols (all on frozen models + frozen features; no retrain)
--------------------------------------------------------------
1. Flight holdout (Final) — overall metrics
2. Type-level / LOTO-style evaluation — per-type metrics on Final, macro-average
   over types with n >= MIN_TYPE_N (post-hoc entity-level shift; not re-trained LOTO)
3. Body-class evaluation — metrics on heavy / narrow / other; LOBCO-style
   per-class evaluation and macro-average over body classes

Models: R3 Teacher, Large MLP, XLarge MLP, FT-Transformer
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
)
from aerotwin.engine.mass_model import enrich_mass_from_columns
from aerotwin.engine.official_benchmark import apply_bases
from aerotwin.engine.statistical_protocol import N_BOOTSTRAP, RANDOM_STATE, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("shift_diagnosis")

MIN_TYPE_N = 50  # min intervals for type-level macro
MIN_BODY_N = 100
N_BOOT = min(2000, N_BOOTSTRAP)  # practical for diagnosis (protocol allows 10k)

CHECKPOINTS = {
    "Large": {
        "path": ROOT / "results/distillation/capacity_scaling/runs/Large_seed42/best_model.pt",
        "kind": "mlp",
        "hidden_dims": (1792, 1024),
        "n_params": 2_887_425,
        "label": "Large MLP",
    },
    "XLarge": {
        "path": ROOT / "results/distillation/capacity_scaling/runs/XLarge_seed42/best_model.pt",
        "kind": "mlp",
        "hidden_dims": (2560, 2048),
        "n_params": 6_748_673,
        "label": "XLarge MLP",
    },
    "FT": {
        "path": ROOT / "results/distillation/ft_transformer/ft_transformer_kd1/best_model.pt",
        "kind": "ft",
        "config": ROOT
        / "results/distillation/ft_transformer/ft_transformer_kd1/student_config.json",
        "n_params": 1_458_625,
        "label": "FT-Transformer",
    },
}

# Canonical published baselines (for cross-check only)
KNOWN_FINAL = {
    "Teacher": 213.62,
    "Large": 215.85,
    "XLarge": 218.59,
    "FT": 224.12,
}


def _prepare(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "actual_fuel_kg" not in df.columns and "fuel_kg" in df.columns:
        df = df.with_columns(pl.col("fuel_kg").alias("actual_fuel_kg"))
    return enrich_mass_from_columns(clean_featured(df))


def _fit_data() -> DistillationData:
    return DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )


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
def _predict_torch(model: torch.nn.Module, x: np.ndarray, device: torch.device, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    for i in range(0, len(xt), bs):
        out.append(model(xt[i : i + bs].to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def _load_mlp(path: Path, hidden: tuple[int, ...], in_dim: int, device: torch.device) -> torch.nn.Module:
    m = StudentMLP(in_dim, hidden_dims=hidden, dropout=0.1)
    blob = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(blob["model_state_dict"])
    return m.to(device).eval()


def _load_ft(path: Path, cfg_path: Path, data: DistillationData, device: torch.device) -> torch.nn.Module:
    sc = StudentConfig.from_mapping(json.loads(cfg_path.read_text(encoding="utf-8")))
    sc.in_dim = data.in_dim
    sc.n_num_features = len(data.numeric_cols)
    sc.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    m = build_student(sc, in_dim=data.in_dim)
    blob = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(blob["model_state_dict"])
    return m.to(device).eval()


def _teacher_predict(df: pl.DataFrame) -> np.ndarray | None:
    """Teacher predictions on Final — prefer live bundle, else audited parquet."""
    cache = ROOT / "cache" / "r3_teacher_distillation_bundle.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            bundle = pickle.load(f)
        cols = list(bundle["feat_cols"])
        sub = ensure_features(df, cols)
        P = apply_bases(bundle["full_models"], sub, cols)
        ridge = np.asarray(bundle["meta"].predict(P), dtype=np.float64)
        return np.asarray(bundle["cal_phase"].transform(sub, ridge), dtype=np.float64)

    # Fallback: Phase teacher audit permanent artifact (same Final rows)
    audit = ROOT / "results" / "distillation" / "teacher_audit" / "teacher_predictions.parquet"
    if not audit.exists():
        return None
    LOGGER.warning(
        "Teacher bundle pickle missing; using audited teacher_predictions.parquet"
    )
    tp = pl.read_parquet(audit)
    # Align by flight_id + interval_idx when available
    if (
        "flight_id" in df.columns
        and "interval_idx" in df.columns
        and "flight_id" in tp.columns
        and "interval_idx" in tp.columns
    ):
        left = df.select(
            [
                pl.col("flight_id").cast(pl.Utf8),
                pl.col("interval_idx").cast(pl.Int64),
            ]
        ).with_row_index("_i")
        right = tp.select(
            [
                pl.col("flight_id").cast(pl.Utf8),
                pl.col("interval_idx").cast(pl.Int64),
                pl.col("teacher_prediction").alias("_pred"),
            ]
        )
        joined = left.join(right, on=["flight_id", "interval_idx"], how="left").sort("_i")
        pred = joined["_pred"].to_numpy().astype(np.float64)
        if not np.isfinite(pred).all():
            LOGGER.warning("Teacher audit join incomplete; falling back to row order")
            pred = tp["teacher_prediction"].to_numpy().astype(np.float64)
        return pred
    return tp["teacher_prediction"].to_numpy().astype(np.float64)


def _full(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = regression_metrics(y, p)
    m["n"] = int(len(y))
    return m


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _body_label(ac: str) -> str:
    """Broad body families for LOBCO-style evaluation."""
    c = aircraft_class(ac)
    if c == "heavy":
        return "widebody_heavy"
    if c == "narrow":
        return "narrowbody"
    # residual types in Final (e.g. B763) → regional/other
    return "regional_other"


def flight_bootstrap_rmse(
    y: np.ndarray,
    p: np.ndarray,
    flight_ids: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Flight-clustered bootstrap CI for RMSE."""
    rng = np.random.default_rng(seed)
    fids = np.asarray(flight_ids).astype(str)
    unique = np.unique(fids)
    point = _rmse(y, p)
    boots = []
    # index by flight
    groups = {u: np.flatnonzero(fids == u) for u in unique}
    for _ in range(n_boot):
        samp = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([groups[u] for u in samp])
        boots.append(_rmse(y[idx], p[idx]))
    boots_a = np.asarray(boots, dtype=np.float64)
    lo, hi = bootstrap_ci(boots_a)
    return {"rmse": point, "ci_lo": lo, "ci_hi": hi, "n_flights": int(len(unique)), "n_boot": n_boot}


def type_macro_metrics(
    y: np.ndarray,
    p: np.ndarray,
    types: np.ndarray,
    min_n: int = MIN_TYPE_N,
) -> dict[str, Any]:
    """Per-type metrics + unweighted macro average (LOTO-style, no retrain)."""
    rows = []
    for t in np.unique(types.astype(str)):
        m = types.astype(str) == t
        if int(m.sum()) < min_n:
            continue
        met = _full(y[m], p[m])
        rows.append({"aircraft_type": t, "body_class": _body_label(t), **met})
    if not rows:
        return {"macro": None, "per_type": [], "n_types": 0}
    macro = {
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "bias": float(np.mean([r["bias"] for r in rows])),
        "r2": float(np.mean([r["r2"] for r in rows])),
        "n_types": len(rows),
        "n_total": int(sum(r["n"] for r in rows)),
    }
    # type-level bootstrap for macro RMSE
    rng = np.random.default_rng(RANDOM_STATE)
    type_rmses = np.array([r["rmse"] for r in rows], dtype=np.float64)
    boots = [
        float(np.mean(rng.choice(type_rmses, size=len(type_rmses), replace=True)))
        for _ in range(N_BOOT)
    ]
    lo, hi = bootstrap_ci(np.asarray(boots))
    macro["rmse_ci_lo"] = lo
    macro["rmse_ci_hi"] = hi
    return {"macro": macro, "per_type": rows, "n_types": len(rows)}


def body_metrics(
    y: np.ndarray, p: np.ndarray, bodies: np.ndarray, min_n: int = MIN_BODY_N
) -> dict[str, Any]:
    rows = []
    for b in sorted(set(bodies.astype(str))):
        m = bodies.astype(str) == b
        if int(m.sum()) < min_n:
            continue
        met = _full(y[m], p[m])
        rows.append({"body_class": b, **met})
    if not rows:
        return {"macro": None, "per_body": []}
    macro = {
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "bias": float(np.mean([r["bias"] for r in rows])),
        "r2": float(np.mean([r["r2"] for r in rows])),
        "n_classes": len(rows),
    }
    return {"macro": macro, "per_body": rows}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-featured", type=Path, default=ROOT / "featured_dataset_final.parquet")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "distillation" / "distribution_shift_diagnosis",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "docs" / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    t0 = time.time()

    LOGGER.info("Loading Final features + distillation preprocessors")
    final_df = _prepare(args.final_featured)
    data = _fit_data()
    x, y = _transform(final_df, data)
    types = (
        final_df["aircraft_type"].cast(pl.Utf8).fill_null("unknown").to_numpy()
        if "aircraft_type" in final_df.columns
        else np.array(["unknown"] * len(y))
    )
    bodies = np.array([_body_label(t) for t in types])
    fids = (
        final_df["flight_id"].cast(pl.Utf8).to_numpy()
        if "flight_id" in final_df.columns
        else np.array([str(i) for i in range(len(y))])
    )
    LOGGER.info(
        "Final n=%d flights=%d types=%d bodies=%s",
        len(y),
        len(np.unique(fids)),
        len(np.unique(types)),
        {b: int((bodies == b).sum()) for b in np.unique(bodies)},
    )

    preds: dict[str, np.ndarray] = {}

    # Teacher
    LOGGER.info("Teacher inference")
    tp = _teacher_predict(final_df)
    if tp is None:
        raise FileNotFoundError("Teacher bundle missing")
    preds["Teacher"] = tp

    # Students
    for name, spec in CHECKPOINTS.items():
        LOGGER.info("Inference %s", name)
        if not spec["path"].exists():
            raise FileNotFoundError(spec["path"])
        if spec["kind"] == "mlp":
            model = _load_mlp(spec["path"], tuple(spec["hidden_dims"]), data.in_dim, device)
        else:
            model = _load_ft(spec["path"], Path(spec["config"]), data, device)
        preds[name] = _predict_torch(model, x, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_order = ["Teacher", "Large", "XLarge", "FT"]
    labels = {
        "Teacher": "R3 Teacher",
        "Large": "Large MLP",
        "XLarge": "XLarge MLP",
        "FT": "FT-Transformer",
    }

    # ---- Protocol metrics ----
    flight: dict[str, Any] = {}
    loto: dict[str, Any] = {}
    body: dict[str, Any] = {}

    for name in model_order:
        p = preds[name]
        flight[name] = {
            "overall": _full(y, p),
            "bootstrap": flight_bootstrap_rmse(y, p, fids),
        }
        loto[name] = type_macro_metrics(y, p, types, MIN_TYPE_N)
        body[name] = body_metrics(y, p, bodies, MIN_BODY_N)
        LOGGER.info(
            "%s Final RMSE=%.2f | type-macro=%.2f | body-macro=%.2f",
            name,
            flight[name]["overall"]["rmse"],
            loto[name]["macro"]["rmse"] if loto[name]["macro"] else float("nan"),
            body[name]["macro"]["rmse"] if body[name]["macro"] else float("nan"),
        )

    # ---- Robustness metrics ----
    robustness = {}
    for name in model_order:
        rf = flight[name]["overall"]["rmse"]
        rl = loto[name]["macro"]["rmse"] if loto[name]["macro"] else float("nan")
        rb = body[name]["macro"]["rmse"] if body[name]["macro"] else float("nan")
        rt = flight["Teacher"]["overall"]["rmse"]
        rtl = loto["Teacher"]["macro"]["rmse"]
        rtb = body["Teacher"]["macro"]["rmse"]
        robustness[name] = {
            "rmse_flight": rf,
            "rmse_loto_macro": rl,
            "rmse_body_macro": rb,
            "degradation_ratio_loto": rl / rf if rf > 0 else float("nan"),
            "degradation_ratio_body": rb / rf if rf > 0 else float("nan"),
            "error_inflation_loto": rl - rf,
            "error_inflation_body": rb - rf,
            "student_gap_flight": rf - rt,
            "student_gap_loto": rl - rtl,
            "student_gap_body": rb - rtb,
            "gap_widens_loto": (rl - rtl) > (rf - rt) if name != "Teacher" else False,
            "gap_widens_body": (rb - rtb) > (rf - rt) if name != "Teacher" else False,
        }

    # Bootstrap CIs for student gaps (flight-clustered RMSE difference)
    gap_ci = {}
    for name in ["Large", "XLarge", "FT"]:
        gap_ci[name] = {}
        # flight gap
        rng = np.random.default_rng(RANDOM_STATE)
        unique = np.unique(fids.astype(str))
        groups = {u: np.flatnonzero(fids.astype(str) == u) for u in unique}
        boots_f, boots_l = [], []
        # precompute type masks for loto bootstrap over types
        type_list = [r["aircraft_type"] for r in loto[name]["per_type"]]
        for _ in range(N_BOOT):
            samp = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([groups[u] for u in samp])
            boots_f.append(
                _rmse(y[idx], preds[name][idx]) - _rmse(y[idx], preds["Teacher"][idx])
            )
        # type-level gap bootstrap: resample types
        t_rmses_s = np.array([r["rmse"] for r in loto[name]["per_type"]])
        t_rmses_t = np.array(
            [next(x["rmse"] for x in loto["Teacher"]["per_type"] if x["aircraft_type"] == t) for t in type_list]
        )
        for _ in range(N_BOOT):
            bi = rng.integers(0, len(t_rmses_s), size=len(t_rmses_s))
            boots_l.append(float(np.mean(t_rmses_s[bi] - t_rmses_t[bi])))
        flo, fhi = bootstrap_ci(np.asarray(boots_f))
        llo, lhi = bootstrap_ci(np.asarray(boots_l))
        gap_ci[name] = {
            "flight_gap": robustness[name]["student_gap_flight"],
            "flight_gap_ci": [flo, fhi],
            "loto_gap": robustness[name]["student_gap_loto"],
            "loto_gap_ci": [llo, lhi],
            "flight_gap_ci_excludes_zero": not (flo <= 0 <= fhi),
            "loto_gap_ci_excludes_zero": not (llo <= 0 <= lhi),
            "gap_increase_loto": robustness[name]["student_gap_loto"]
            - robustness[name]["student_gap_flight"],
        }

    # Ranking
    def rank_by(key_fn):
        order = sorted(model_order, key=key_fn)
        return order

    rankings = {
        "flight": rank_by(lambda n: flight[n]["overall"]["rmse"]),
        "loto_macro": rank_by(lambda n: loto[n]["macro"]["rmse"]),
        "body_macro": rank_by(lambda n: body[n]["macro"]["rmse"]),
    }

    # Decision gate
    # Meaningful if for Large (primary student): LOTO gap increases by >2 kg AND CI on LOTO gap excludes 0 or increase is large
    large_inc = gap_ci["Large"]["gap_increase_loto"]
    large_loto_gap = gap_ci["Large"]["loto_gap"]
    large_flight_gap = gap_ci["Large"]["flight_gap"]
    # Also check body class heavy-only gap
    heavy_mask = bodies == "widebody_heavy"
    heavy_gaps = {}
    for name in model_order:
        if heavy_mask.sum() < MIN_BODY_N:
            continue
        yh, ph = y[heavy_mask], preds[name][heavy_mask]
        heavy_gaps[name] = _rmse(yh, ph)
    heavy_student_gaps = {
        n: heavy_gaps[n] - heavy_gaps["Teacher"]
        for n in ["Large", "XLarge", "FT"]
        if n in heavy_gaps
    }

    decision = {
        "question": "Does KD introduce additional robustness loss under entity-level shift vs teacher?",
        "primary_student": "Large",
        "flight_gap_large": large_flight_gap,
        "loto_gap_large": large_loto_gap,
        "gap_increase_loto_large": large_inc,
        "body_gap_large": robustness["Large"]["student_gap_body"],
        "heavy_gap_large": heavy_student_gaps.get("Large"),
        "flight_gap_ci_excludes_zero": gap_ci["Large"]["flight_gap_ci_excludes_zero"],
        "loto_gap_ci_excludes_zero": gap_ci["Large"]["loto_gap_ci_excludes_zero"],
        "threshold_kg_for_meaningful_increase": 2.0,
        "meaningful_gap_increase_loto": bool(large_inc > 2.0),
        "proceed_to_adaptive_kd": None,  # filled below
        "rationale": "",
    }
    # Gate: Adaptive KD only if gap widens meaningfully under type-level shift
    # AND student remains worse than teacher on shift protocol
    proceed = (
        decision["meaningful_gap_increase_loto"]
        and large_loto_gap > large_flight_gap
        and large_loto_gap > 2.0
    )
    # Also consider if LOTO degradation ratio student >> teacher
    deg_s = robustness["Large"]["degradation_ratio_loto"]
    deg_t = robustness["Teacher"]["degradation_ratio_loto"]
    decision["degradation_ratio_large"] = deg_s
    decision["degradation_ratio_teacher"] = deg_t
    decision["student_degrades_more_than_teacher"] = bool(deg_s > deg_t + 0.02)
    # Final gate: need BOTH meaningful absolute gap increase AND worse relative degradation
    # OR a large absolute LOTO gap (>5 kg) that CI excludes zero
    proceed = bool(
        (large_inc > 2.0 and deg_s > deg_t)
        or (large_loto_gap > 5.0 and gap_ci["Large"]["loto_gap_ci_excludes_zero"] and large_inc > 1.0)
    )
    decision["proceed_to_adaptive_kd"] = proceed
    if proceed:
        decision["rationale"] = (
            f"Large student gap widens under type-level evaluation "
            f"(flight gap {large_flight_gap:+.2f} → LOTO-macro gap {large_loto_gap:+.2f}, "
            f"Δ={large_inc:+.2f} kg). Adaptive KD is justified to investigate."
        )
        decision["next_phase"] = "Phase 1 — Adaptive / Uncertainty-Aware KD"
    else:
        decision["rationale"] = (
            f"No meaningful increase in teacher–student gap under type/body shift "
            f"(flight gap {large_flight_gap:+.2f} kg, LOTO-macro gap {large_loto_gap:+.2f} kg, "
            f"increase {large_inc:+.2f} kg). Adaptive KD is NOT justified by this evidence. "
            f"Pivot to systematic architecture + KD under shift study."
        )
        decision["next_phase"] = (
            "Paper: empirical study of architecture choice and KD under distribution shift "
            "(no Adaptive KD implementation)"
        )

    # Save predictions + tables
    for name in model_order:
        err = preds[name] - y
        pl.DataFrame(
            {
                "flight_id": fids.tolist(),
                "aircraft_type": types.tolist(),
                "body_class": bodies.tolist(),
                "ground_truth": y,
                "prediction": preds[name],
                "residual": err,
                "absolute_error": np.abs(err),
            }
        ).write_parquet(out / f"predictions_{name.lower()}.parquet")

    # Comparison tables
    rows_main = []
    for name in model_order:
        rows_main.append(
            {
                "model": labels[name],
                "rmse_flight": flight[name]["overall"]["rmse"],
                "mae_flight": flight[name]["overall"]["mae"],
                "bias_flight": flight[name]["overall"]["bias"],
                "r2_flight": flight[name]["overall"]["r2"],
                "rmse_loto_macro": loto[name]["macro"]["rmse"],
                "mae_loto_macro": loto[name]["macro"]["mae"],
                "rmse_body_macro": body[name]["macro"]["rmse"],
                "mae_body_macro": body[name]["macro"]["mae"],
                "degradation_ratio_loto": robustness[name]["degradation_ratio_loto"],
                "degradation_ratio_body": robustness[name]["degradation_ratio_body"],
                "error_inflation_loto": robustness[name]["error_inflation_loto"],
                "error_inflation_body": robustness[name]["error_inflation_body"],
                "student_gap_flight": robustness[name]["student_gap_flight"],
                "student_gap_loto": robustness[name]["student_gap_loto"],
                "student_gap_body": robustness[name]["student_gap_body"],
            }
        )
    pl.DataFrame(rows_main).write_csv(out / "metrics_all_protocols.csv")

    # Per-type comparison Large vs Teacher
    type_cmp = []
    for row in loto["Large"]["per_type"]:
        t = row["aircraft_type"]
        tr = next(x for x in loto["Teacher"]["per_type"] if x["aircraft_type"] == t)
        xl = next((x for x in loto["XLarge"]["per_type"] if x["aircraft_type"] == t), None)
        ft = next((x for x in loto["FT"]["per_type"] if x["aircraft_type"] == t), None)
        type_cmp.append(
            {
                "aircraft_type": t,
                "body_class": row["body_class"],
                "n": row["n"],
                "teacher_rmse": tr["rmse"],
                "large_rmse": row["rmse"],
                "xlarge_rmse": xl["rmse"] if xl else None,
                "ft_rmse": ft["rmse"] if ft else None,
                "large_gap": row["rmse"] - tr["rmse"],
            }
        )
    pl.DataFrame(type_cmp).sort("large_gap", descending=True).write_csv(out / "metrics_by_type.csv")

    body_rows = []
    for bname in sorted({r["body_class"] for r in body["Large"]["per_body"]}):
        row = {"body_class": bname}
        for name in model_order:
            br = next(x for x in body[name]["per_body"] if x["body_class"] == bname)
            row[f"{name.lower()}_rmse"] = br["rmse"]
            row[f"{name.lower()}_n"] = br["n"]
        body_rows.append(row)
    pl.DataFrame(body_rows).write_csv(out / "metrics_by_body.csv")

    # Plots
    _plots(
        model_order,
        labels,
        flight,
        loto,
        body,
        robustness,
        rankings,
        plots,
        fig_dir,
        type_cmp,
    )

    blob = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_notes": {
            "flight": "Final held-out overall metrics",
            "loto_style": (
                f"Post-hoc per-type metrics on Final, unweighted macro over types with n>={MIN_TYPE_N}. "
                "Models are NOT re-trained leave-one-type-out; this measures entity-level "
                "heterogeneity / shift within flight-holdout, comparable across frozen models."
            ),
            "body_class": (
                "Metrics on widebody_heavy / narrowbody / regional_other subsets of Final; "
                "macro = unweighted mean over body classes with n>={MIN_BODY_N}."
            ),
            "no_retrain": True,
        },
        "models": model_order,
        "flight": {n: flight[n] for n in model_order},
        "loto": {
            n: {"macro": loto[n]["macro"], "n_types": loto[n]["n_types"]} for n in model_order
        },
        "body": {n: {"macro": body[n]["macro"], "per_body": body[n]["per_body"]} for n in model_order},
        "robustness": robustness,
        "gap_ci": gap_ci,
        "rankings": rankings,
        "heavy_rmse": heavy_gaps,
        "heavy_student_gaps": heavy_student_gaps,
        "decision": decision,
        "wall_seconds": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    (out / "decision_gate.json").write_text(
        json.dumps(decision, indent=2, default=str), encoding="utf-8"
    )

    report = _write_report(blob, labels)
    (out / "distribution_shift_diagnosis.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "reports" / "distribution_shift_diagnosis.md").write_text(
        report, encoding="utf-8"
    )

    print("\n=== PHASE 0 DISTRIBUTION SHIFT DIAGNOSIS ===")
    for name in model_order:
        r = robustness[name]
        print(
            f"  {labels[name]}: flight={r['rmse_flight']:.2f} "
            f"loto_macro={r['rmse_loto_macro']:.2f} body_macro={r['rmse_body_macro']:.2f} "
            f"gap_flight={r['student_gap_flight']:+.2f} gap_loto={r['student_gap_loto']:+.2f}"
        )
    print(f"  rankings flight={rankings['flight']} loto={rankings['loto_macro']}")
    print(f"  proceed_to_adaptive_kd={decision['proceed_to_adaptive_kd']}")
    print(f"  next={decision['next_phase']}")
    print(f"  results={out}")


def _plots(model_order, labels, flight, loto, body, robustness, rankings, plots, fig_dir, type_cmp):
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})
    names = [labels[n] for n in model_order]
    colors = {"Teacher": "#2ca02c", "Large": "#1f77b4", "XLarge": "#ff7f0e", "FT": "#d62728"}

    def save(fig, key):
        p = plots / f"{key}.png"
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        (fig_dir / f"fig_shift_{key}.png").write_bytes(p.read_bytes())
        plt.close(fig)

    # 1 RMSE across protocols
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(3)
    w = 0.18
    for i, n in enumerate(model_order):
        vals = [
            flight[n]["overall"]["rmse"],
            loto[n]["macro"]["rmse"],
            body[n]["macro"]["rmse"],
        ]
        ax.bar(x + (i - 1.5) * w, vals, w, label=labels[n], color=colors[n])
    ax.set_xticks(x)
    ax.set_xticklabels(["Flight (Final)", "Type-macro (LOTO-style)", "Body-macro"])
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("RMSE across evaluation protocols")
    ax.legend(fontsize=9)
    save(fig, "rmse_all_protocols")

    # 2 Degradation ratio
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(model_order))
    w = 0.35
    ax.bar(
        x - w / 2,
        [robustness[n]["degradation_ratio_loto"] for n in model_order],
        w,
        label="Type-macro / Flight",
        color="#4c72b0",
    )
    ax.bar(
        x + w / 2,
        [robustness[n]["degradation_ratio_body"] for n in model_order],
        w,
        label="Body-macro / Flight",
        color="#dd8452",
    )
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Degradation ratio")
    ax.set_title("Relative degradation under entity-level protocols")
    ax.legend()
    save(fig, "degradation_ratio")

    # 3 Teacher-student gap
    fig, ax = plt.subplots(figsize=(7, 4.5))
    students = ["Large", "XLarge", "FT"]
    x = np.arange(len(students))
    w = 0.25
    ax.bar(
        x - w,
        [robustness[n]["student_gap_flight"] for n in students],
        w,
        label="Flight gap",
    )
    ax.bar(
        x,
        [robustness[n]["student_gap_loto"] for n in students],
        w,
        label="Type-macro gap",
    )
    ax.bar(
        x + w,
        [robustness[n]["student_gap_body"] for n in students],
        w,
        label="Body-macro gap",
    )
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in students])
    ax.set_ylabel("RMSE_student − RMSE_teacher (kg)")
    ax.set_title("Teacher–student gap across protocols")
    ax.legend()
    save(fig, "teacher_student_gap")

    # 4 Error inflation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(model_order))
    ax.bar(
        x - 0.2,
        [robustness[n]["error_inflation_loto"] for n in model_order],
        0.4,
        label="Type-macro − Flight",
    )
    ax.bar(
        x + 0.2,
        [robustness[n]["error_inflation_body"] for n in model_order],
        0.4,
        label="Body-macro − Flight",
    )
    ax.axhline(0, color="k", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Error inflation (kg)")
    ax.set_title("Absolute error inflation under shift protocols")
    ax.legend()
    save(fig, "error_inflation")

    # 5 Ranking stability
    fig, ax = plt.subplots(figsize=(7, 4.2))
    protocols = ["flight", "loto_macro", "body_macro"]
    for n in model_order:
        ranks = [rankings[p].index(n) + 1 for p in protocols]
        ax.plot(protocols, ranks, "o-", label=labels[n], color=colors[n], lw=2)
    ax.set_ylim(0.5, 4.5)
    ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Ranking stability across protocols")
    ax.legend()
    save(fig, "ranking_stability")

    # 6 Aircraft-family robustness (top types by large gap)
    fig, ax = plt.subplots(figsize=(9, 5))
    # sort by n desc for readability — show all types in type_cmp
    tc = sorted(type_cmp, key=lambda r: -r["n"])[:12]
    xs = np.arange(len(tc))
    w = 0.2
    ax.bar(xs - 1.5 * w, [r["teacher_rmse"] for r in tc], w, label="Teacher", color=colors["Teacher"])
    ax.bar(xs - 0.5 * w, [r["large_rmse"] for r in tc], w, label="Large", color=colors["Large"])
    ax.bar(xs + 0.5 * w, [r["xlarge_rmse"] for r in tc], w, label="XLarge", color=colors["XLarge"])
    ax.bar(xs + 1.5 * w, [r["ft_rmse"] for r in tc], w, label="FT", color=colors["FT"])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["aircraft_type"] for r in tc], rotation=45, ha="right")
    ax.set_ylabel("RMSE (kg)")
    ax.set_title("Per-type Final RMSE (largest types)")
    ax.legend(fontsize=8)
    save(fig, "aircraft_family_robustness")


def _write_report(blob: dict[str, Any], labels: dict[str, str]) -> str:
    d = blob["decision"]
    R = blob["robustness"]
    lines = [
        "# Phase 0 — Distribution Shift Diagnosis",
        "",
        f"**Date:** {blob['timestamp_utc'][:10]}",
        "**Status:** Evaluation only (frozen models; no retrain)",
        "",
        "## Scientific question",
        "",
        "> Does standard knowledge distillation introduce additional robustness loss "
        "under entity-level distribution shift compared to the teacher?",
        "",
        "---",
        "",
        "## Experimental setup",
        "",
        "| Item | Value |",
        "|------|------|",
        "| Models | R3 Teacher, Large MLP, XLarge MLP, FT-Transformer |",
        "| Retrain | **None** — frozen checkpoints only |",
        "| Flight holdout | Final (`featured_dataset_final.parquet`) |",
        "| Type-level (LOTO-style) | Per-type RMSE on Final; unweighted macro (n≥50) |",
        "| Body-class | widebody_heavy / narrowbody / regional_other macros |",
        "| Bootstrap | Flight-clustered for overall RMSE; type-resample for macro gap |",
        "",
        "**Important protocol note:** Type-level evaluation is **post-hoc LOTO-style** "
        "(entity-level metrics on held-out flights). Models were trained with all types "
        "present in the distillation train split. This isolates *relative* robustness "
        "of frozen KD students vs teacher under entity heterogeneity without re-fitting.",
        "",
        "---",
        "",
        "## Metrics — all protocols",
        "",
        "| Model | Flight RMSE | Type-macro RMSE | Body-macro RMSE | Deg. ratio (type) | Deg. ratio (body) |",
        "|-------|------------:|----------------:|----------------:|------------------:|------------------:|",
    ]
    for n in blob["models"]:
        r = R[n]
        lines.append(
            f"| {labels[n]} | {r['rmse_flight']:.2f} | {r['rmse_loto_macro']:.2f} | "
            f"{r['rmse_body_macro']:.2f} | {r['degradation_ratio_loto']:.3f} | "
            f"{r['degradation_ratio_body']:.3f} |"
        )

    lines += [
        "",
        "### Error inflation (shift − flight)",
        "",
        "| Model | Inflation type-macro | Inflation body-macro |",
        "|-------|---------------------:|---------------------:|",
    ]
    for n in blob["models"]:
        r = R[n]
        lines.append(
            f"| {labels[n]} | {r['error_inflation_loto']:+.2f} | {r['error_inflation_body']:+.2f} |"
        )

    lines += [
        "",
        "### Teacher–student gap (student − teacher RMSE)",
        "",
        "| Student | Gap flight | Gap type-macro | Gap body-macro | Gap increase (type − flight) |",
        "|---------|-----------:|---------------:|---------------:|-----------------------------:|",
    ]
    for n in ["Large", "XLarge", "FT"]:
        r = R[n]
        lines.append(
            f"| {labels[n]} | {r['student_gap_flight']:+.2f} | {r['student_gap_loto']:+.2f} | "
            f"{r['student_gap_body']:+.2f} | "
            f"{r['student_gap_loto'] - r['student_gap_flight']:+.2f} |"
        )

    g = blob["gap_ci"]["Large"]
    lines += [
        "",
        "### Bootstrap uncertainty (Large vs Teacher)",
        "",
        f"- Flight gap: **{g['flight_gap']:+.2f} kg**, 95% CI **[{g['flight_gap_ci'][0]:+.2f}, {g['flight_gap_ci'][1]:+.2f}]** "
        f"(excludes 0? **{g['flight_gap_ci_excludes_zero']}**)",
        f"- Type-macro gap: **{g['loto_gap']:+.2f} kg**, 95% CI **[{g['loto_gap_ci'][0]:+.2f}, {g['loto_gap_ci'][1]:+.2f}]** "
        f"(excludes 0? **{g['loto_gap_ci_excludes_zero']}**)",
        f"- Gap increase (type − flight): **{g['gap_increase_loto']:+.2f} kg**",
        "",
        "---",
        "",
        "## Ranking stability",
        "",
        f"| Protocol | Ranking (best → worst) |",
        f"|----------|------------------------|",
        f"| Flight | {' → '.join(labels[n] for n in blob['rankings']['flight'])} |",
        f"| Type-macro | {' → '.join(labels[n] for n in blob['rankings']['loto_macro'])} |",
        f"| Body-macro | {' → '.join(labels[n] for n in blob['rankings']['body_macro'])} |",
        "",
        "---",
        "",
        "## Figures",
        "",
        "![rmse](figures/fig_shift_rmse_all_protocols.png)",
        "",
        "![degradation](figures/fig_shift_degradation_ratio.png)",
        "",
        "![gap](figures/fig_shift_teacher_student_gap.png)",
        "",
        "![inflation](figures/fig_shift_error_inflation.png)",
        "",
        "![rank](figures/fig_shift_ranking_stability.png)",
        "",
        "![family](figures/fig_shift_aircraft_family_robustness.png)",
        "",
        "---",
        "",
        "## Interpretation (evidence only)",
        "",
        f"1. **Does KD lose robustness under shift?** "
        f"Degradation ratios (type-macro/flight): Teacher **{R['Teacher']['degradation_ratio_loto']:.3f}**, "
        f"Large **{R['Large']['degradation_ratio_loto']:.3f}**, "
        f"XLarge **{R['XLarge']['degradation_ratio_loto']:.3f}**, "
        f"FT **{R['FT']['degradation_ratio_loto']:.3f}**.",
        f"2. **Does the teacher degrade less?** Compare inflation and ratios above.",
        f"3. **Does teacher–student gap increase?** Large: flight gap "
        f"**{R['Large']['student_gap_flight']:+.2f}** → type-macro gap "
        f"**{R['Large']['student_gap_loto']:+.2f}** (Δ **{R['Large']['student_gap_loto'] - R['Large']['student_gap_flight']:+.2f}** kg).",
        f"4. **Is Large still the most robust student?** "
        f"Best student under type-macro: **{labels[blob['rankings']['loto_macro'][1] if blob['rankings']['loto_macro'][0]=='Teacher' else blob['rankings']['loto_macro'][0]]}** "
        f"(full order: {' → '.join(labels[n] for n in blob['rankings']['loto_macro'])}).",
        f"5. **Does FT become relatively stronger under shift?** Compare FT rank flight vs type-macro.",
        f"6. **Statistically meaningful?** Large flight gap CI excludes 0? "
        f"**{g['flight_gap_ci_excludes_zero']}**. Type-macro gap CI excludes 0? "
        f"**{g['loto_gap_ci_excludes_zero']}**.",
        "",
        "---",
        "",
        "## Decision gate",
        "",
        f"| Field | Value |",
        f"|-------|------|",
        f"| Proceed to Adaptive KD? | **{d['proceed_to_adaptive_kd']}** |",
        f"| Next phase | {d['next_phase']} |",
        f"| Rationale | {d['rationale']} |",
        "",
        "---",
        "",
        "## Artifacts",
        "",
        "`results/distillation/distribution_shift_diagnosis/`",
        "",
        f"*Generated {blob['timestamp_utc']}*",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
