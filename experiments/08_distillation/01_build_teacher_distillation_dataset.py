"""Build the frozen R3 teacher distillation dataset.

Exports teacher knowledge for every available training sample so neural
students can be trained later with knowledge distillation.

Teacher (frozen, read-only methodology):
  * 6-base ensemble: {XGB, LGBM, CatBoost} x {Direct kg, Fuel-Flow kg/s}
  * Ridge (or LGBM) meta-learner chosen on train OOF
  * R3 dynamic mass features (21 physics-informed)
  * P1E phase-conditional affine calibration

This script does NOT train any student, modify feature engineering, or
optimize hyperparameters. It only re-runs the existing frozen inference path
and writes:

  * distillation_dataset.parquet
  * docs/reports/distillation_dataset_report.md

Train rows use GroupKFold OOF base predictions (no in-fold leakage).
Rank/Final rows (if present) use full-train base models.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.engine.eval_framework import project_root
from aerotwin.engine.gap_closing import (
    ENSEMBLE_BASES,
    ConditionalAffineCalibrator,
    clean_featured,
    ensure_features,
    group_phase,
)
from aerotwin.engine.mass_model import R3_MASS_FEATURES, enrich_mass_from_columns
from aerotwin.engine.official_benchmark import (
    apply_bases,
    build_oof_matrix,
    choose_meta_on_train_folds,
    ew_feature_cols,
    featured_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("distillation_export")

# Human-readable column names for the 6 frozen ensemble bases (ENSEMBLE_BASES order).
BASE_PRED_COLS = [
    "xgb_direct_prediction",
    "lgbm_direct_prediction",
    "cat_direct_prediction",
    "xgb_flow_prediction",
    "lgbm_flow_prediction",
    "cat_flow_prediction",
]

# Optional convenience aliases (first Direct model of each family).
ALIAS_PRED_COLS = {
    "xgb_prediction": "xgb_direct_prediction",
    "lgbm_prediction": "lgbm_direct_prediction",
    "cat_prediction": "cat_direct_prediction",
}

# Physics / intermediate signals already present in the featured pipeline.
# Only exported when the column exists — never recomputed here.
AUX_SOURCE_COLS = {
    "openap_prediction": "physics_fuel_kg",
    "residual": "residual_kg",
    "phase": "phase",
    "dynamic_mass": "r3_mean_mass_kg",
    "ref_mass_kg": "ref_mass_kg",
    "r3_mass_start_kg": "r3_mass_start_kg",
    "r3_mass_end_kg": "r3_mass_end_kg",
    "r3_mass_consumed_kg": "r3_mass_consumed_kg",
    "r3_phase_mass_kg": "r3_phase_mass_kg",
    "r3_fuel_fraction": "r3_fuel_fraction",
}

# Identity / bookkeeping only. Columns that are also ensemble features
# (e.g. aircraft_type, origin_icao) are exported once via the feature block.
ID_META_COLS = [
    "flight_id",
    "interval_idx",
    "start",
    "end",
]

CACHE_NAME = "r3_teacher_distillation_bundle.pkl"
OUT_PARQUET_NAME = "distillation_dataset.parquet"
OUT_REPORT_NAME = "distillation_dataset_report.md"
OUT_META_NAME = "distillation_dataset_meta.json"


def _root() -> Path:
    """Prefer cwd/project that holds featured_dataset.parquet."""
    pr = project_root()
    if (pr / "featured_dataset.parquet").exists():
        return pr
    if (ROOT / "featured_dataset.parquet").exists():
        return ROOT
    return pr


def _cache_path(root: Path) -> Path:
    return root / "cache" / CACHE_NAME


def _feature_cols(df: pl.DataFrame) -> list[str]:
    base = ew_feature_cols(df)
    mass = [c for c in R3_MASS_FEATURES if c in df.columns]
    return list(dict.fromkeys(base + mass))


def _load_split(name: str, root: Path) -> pl.DataFrame | None:
    path = featured_path(name, root=root)  # type: ignore[arg-type]
    if not path.exists():
        LOGGER.info("Split %s not found at %s — skipping", name, path)
        return None
    df = clean_featured(pl.read_parquet(path))
    LOGGER.info("Loaded %s: %d rows from %s", name, len(df), path)
    return df


def _build_or_load_teacher(
    train: pl.DataFrame,
    feat_cols: list[str],
    cache_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build OOF teacher predictions or load a prior cache.

    Cache stores only arrays + sklearn/xgb/lgbm/cat objects needed to apply
    the teacher to other splits. Feature engineering is not altered.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        LOGGER.info("Loading teacher cache %s", cache_path)
        with open(cache_path, "rb") as f:
            bundle = pickle.load(f)
        if (
            bundle.get("feat_cols") == feat_cols
            and len(bundle.get("P_oof", [])) == len(train)
            and len(bundle.get("y_kg", [])) == len(train)
        ):
            LOGGER.info(
                "Cache hit: meta=%s, OOF RMSE=%.2f",
                bundle.get("meta_kind"),
                bundle.get("oof_rmse"),
            )
            return bundle
        LOGGER.warning("Teacher cache mismatch; rebuilding")

    LOGGER.info(
        "Building frozen R3 teacher OOF (bases=%d, features=%d, rows=%d) — this is slow once...",
        len(ENSEMBLE_BASES),
        len(feat_cols),
        len(train),
    )
    t0 = time.time()
    P_oof, y_kg, full_models = build_oof_matrix(
        train, feat_cols, ENSEMBLE_BASES, n_splits=5
    )
    groups = train["flight_id"].to_numpy()
    meta_kind, meta = choose_meta_on_train_folds(P_oof, y_kg, groups, n_splits=5)
    ridge_oof = np.asarray(meta.predict(P_oof), dtype=np.float64)
    oof_rmse = float(np.sqrt(np.mean((ridge_oof - y_kg) ** 2)))
    LOGGER.info(
        "Meta=%s, train OOF RMSE (pre-P1E)=%.2f  [%.1f min]",
        meta_kind,
        oof_rmse,
        (time.time() - t0) / 60.0,
    )

    cal_phase = ConditionalAffineCalibrator(group_phase).fit(train, y_kg, ridge_oof)
    teacher_oof = cal_phase.transform(train, ridge_oof)
    p1e_rmse = float(np.sqrt(np.mean((teacher_oof - y_kg) ** 2)))
    p1e_bias = float(np.mean(teacher_oof - y_kg))
    LOGGER.info("P1E OOF RMSE=%.2f bias=%.2f", p1e_rmse, p1e_bias)

    bundle: dict[str, Any] = {
        "feat_cols": feat_cols,
        "base_specs": list(ENSEMBLE_BASES),
        "base_pred_cols": list(BASE_PRED_COLS),
        "P_oof": P_oof,
        "y_kg": y_kg,
        "full_models": full_models,
        "meta_kind": meta_kind,
        "meta": meta,
        "ridge_oof": ridge_oof,
        "teacher_oof": teacher_oof,
        "cal_phase": cal_phase,
        "oof_rmse": oof_rmse,
        "p1e_oof_rmse": p1e_rmse,
        "p1e_oof_bias": p1e_bias,
        "n_train": len(train),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Cached teacher bundle -> %s", cache_path)
    return bundle


def _predict_holdout(
    bundle: dict[str, Any],
    df: pl.DataFrame,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-model base matrix, ridge meta, and P1E teacher predictions."""
    df = ensure_features(df, feat_cols)
    P = apply_bases(bundle["full_models"], df, feat_cols)
    ridge = np.asarray(bundle["meta"].predict(P), dtype=np.float64)
    teacher = bundle["cal_phase"].transform(df, ridge)
    return P, ridge, teacher


def _assemble_split_frame(
    df: pl.DataFrame,
    feat_cols: list[str],
    *,
    split: str,
    P: np.ndarray,
    ridge: np.ndarray,
    teacher: np.ndarray,
    sample_id_offset: int,
) -> pl.DataFrame:
    """Build one split's distillation rows."""
    n = len(df)
    assert P.shape == (n, len(BASE_PRED_COLS)), (P.shape, n, len(BASE_PRED_COLS))

    parts: list[pl.DataFrame] = []

    # Build via horizontal concat of polars frames so nulls stay null (not NaN).
    blocks: list[pl.DataFrame] = []

    id_cols: dict[str, Any] = {
        "sample_id": np.arange(sample_id_offset, sample_id_offset + n, dtype=np.int64),
        "split": [split] * n,
    }
    blocks.append(pl.DataFrame(id_cols))

    meta_present = [c for c in ID_META_COLS if c in df.columns]
    if meta_present:
        meta_df = df.select(meta_present)
        # Normalize string-like columns.
        casts = []
        for c in meta_present:
            if meta_df[c].dtype in (pl.Categorical,):
                casts.append(pl.col(c).cast(pl.Utf8))
            else:
                casts.append(pl.col(c))
        blocks.append(meta_df.select(casts))

    # Feature vector exactly as fed to the ensemble (order preserved).
    feat_df = ensure_features(df, feat_cols).select(feat_cols)
    feat_casts = []
    for c in feat_cols:
        if feat_df[c].dtype == pl.Categorical:
            feat_casts.append(pl.col(c).cast(pl.Utf8))
        elif feat_df[c].dtype == pl.Boolean:
            feat_casts.append(pl.col(c).cast(pl.Float64))
        else:
            feat_casts.append(pl.col(c))
    blocks.append(feat_df.select(feat_casts))

    # Ground truth + teacher outputs (dense arrays, no nulls).
    target_cols: dict[str, Any] = {
        "ground_truth": df["actual_fuel_kg"].to_numpy().astype(np.float64),
        "teacher_prediction": teacher.astype(np.float64),
        "ridge_prediction": ridge.astype(np.float64),
    }
    for j, col in enumerate(BASE_PRED_COLS):
        target_cols[col] = P[:, j].astype(np.float64)
    for alias, src in ALIAS_PRED_COLS.items():
        target_cols[alias] = target_cols[src]
    target_cols["calibrated_prediction"] = target_cols["teacher_prediction"]
    blocks.append(pl.DataFrame(target_cols))

    # Intermediate physics signals already on the frame (aliases only).
    aux_exprs: list[pl.Expr] = []
    used_names = set()
    for b in blocks:
        used_names.update(b.columns)

    for out_name, src_name in AUX_SOURCE_COLS.items():
        if src_name not in df.columns or out_name in used_names:
            # Missing source, or already exported under the target name.
            continue
        s = df[src_name]
        if s.dtype in (pl.Utf8, pl.Categorical, pl.String):
            aux_exprs.append(pl.col(src_name).cast(pl.Utf8).alias(out_name))
        else:
            aux_exprs.append(pl.col(src_name).alias(out_name))

    if aux_exprs:
        blocks.append(df.select(aux_exprs))

    # Dominant phase used by P1E (always export; may equal `phase`).
    blocks.append(
        pl.DataFrame({"p1e_phase_group": group_phase(df).astype(str)})
    )

    out = blocks[0]
    for b in blocks[1:]:
        # Drop any accidental overlapping columns from the right block.
        overlap = [c for c in b.columns if c in out.columns]
        if overlap:
            b = b.drop(overlap)
        if b.width == 0:
            continue
        out = pl.concat([out, b], how="horizontal")
    return out


def _missing_report(df: pl.DataFrame) -> dict[str, Any]:
    """Count nulls and NaNs (float columns may contain either after I/O)."""
    n = len(df)
    missing: dict[str, dict[str, float | int]] = {}
    for c in df.columns:
        nc = int(df[c].null_count())
        nan_c = 0
        if df[c].dtype in (pl.Float32, pl.Float64):
            nan_c = int(df[c].is_nan().sum())
        total = nc + nan_c
        if total:
            missing[c] = {
                "null_count": nc,
                "nan_count": nan_c,
                "missing_count": total,
                "missing_frac": total / max(n, 1),
            }
    return {
        "columns_with_missing": len(missing),
        "total_missing_cells": int(sum(int(v["missing_count"]) for v in missing.values())),
        "per_column": missing,
    }


def _write_report(
    path: Path,
    *,
    df: pl.DataFrame,
    feat_cols: list[str],
    aux_targets: list[str],
    parquet_path: Path,
    bundle: dict[str, Any],
    split_counts: dict[str, int],
) -> None:
    size_bytes = parquet_path.stat().st_size if parquet_path.exists() else 0
    size_mb = size_bytes / (1024 * 1024)
    miss = _missing_report(df)

    # Feature-only missing summary.
    feat_nulls = {
        c: miss["per_column"][c]
        for c in feat_cols
        if c in miss["per_column"]
    }
    top_feat_nulls = sorted(
        feat_nulls.items(), key=lambda kv: int(kv[1]["missing_count"]), reverse=True
    )[:15]

    y = df["ground_truth"].to_numpy()
    t = df["teacher_prediction"].to_numpy()
    train_mask = (df["split"] == "train").to_numpy()
    if train_mask.any():
        tr_rmse = float(np.sqrt(np.mean((t[train_mask] - y[train_mask]) ** 2)))
        tr_bias = float(np.mean(t[train_mask] - y[train_mask]))
    else:
        tr_rmse = float("nan")
        tr_bias = float("nan")

    lines = [
        "# Teacher Distillation Dataset Report",
        "",
        "**Stage:** AeroTwin Distillation - Step 1 (export only)",
        "",
        "**Teacher (frozen):** R3 dynamic mass (21 features) + 6-base GBDT ensemble "
        "+ Ridge/LGBM meta + P1E phase-conditional affine",
        "",
        "Reference published metrics (Rank/Final): Combined RMSE **221.33 kg**, "
        "Final RMSE **213.73 kg**, bias **~+3.7 kg** "
        "(`R3_P1E_phase_affine`).",
        "",
        "---",
        "",
        "## Dataset summary",
        "",
        f"| Field | Value |",
        f"|-------|------:|",
        f"| Number of samples | **{len(df):,}** |",
        f"| Number of features (ensemble input) | **{len(feat_cols)}** |",
        f"| Dataset size (parquet) | **{size_mb:.2f} MB** ({size_bytes:,} bytes) |",
        f"| Output path | `{parquet_path.as_posix()}` |",
        f"| Columns with any missing (null/NaN) | {miss['columns_with_missing']} |",
        f"| Total missing cells | {miss['total_missing_cells']:,} |",
        f"| Meta learner | `{bundle.get('meta_kind')}` |",
        f"| Train OOF RMSE (pre-P1E) | {bundle.get('oof_rmse'):.2f} kg |",
        f"| Train OOF RMSE (teacher / P1E) | {bundle.get('p1e_oof_rmse'):.2f} kg |",
        f"| Train OOF bias (teacher) | {bundle.get('p1e_oof_bias'):+.2f} kg |",
        f"| Exported train teacher RMSE (sanity) | {tr_rmse:.2f} kg |",
        f"| Exported train teacher bias (sanity) | {tr_bias:+.2f} kg |",
        "",
        "### Split counts",
        "",
        "| Split | Rows |",
        "|-------|-----:|",
    ]
    for k, v in split_counts.items():
        lines.append(f"| {k} | {v:,} |")

    lines += [
        "",
        "### Missing values (top feature columns)",
        "",
    ]
    if not top_feat_nulls:
        lines.append("No missing values in feature columns.")
    else:
        lines += [
            "| Feature | Missing count | Missing fraction |",
            "|---------|--------------:|-----------------:|",
        ]
        for c, info in top_feat_nulls:
            lines.append(
                f"| `{c}` | {int(info['missing_count']):,} | {float(info['missing_frac']):.4f} |"
            )

    lines += [
        "",
        "### Exported auxiliary targets / signals",
        "",
        "These are teacher soft labels or intermediate physics quantities already "
        "present in the pipeline (not recomputed beyond the frozen teacher path):",
        "",
    ]
    for name in aux_targets:
        lines.append(f"- `{name}`")

    lines += [
        "",
        "### Feature columns (ensemble input order)",
        "",
        "```",
        ", ".join(feat_cols),
        "```",
        "",
        "### Schema notes",
        "",
        "- `ground_truth` = `actual_fuel_kg` (interval fuel burn, kg)",
        "- `teacher_prediction` = final R3 teacher after P1E calibration",
        "- `ridge_prediction` = meta-ensemble output before P1E",
        "- `xgb_*` / `lgbm_*` / `cat_*` = base model kg predictions (Direct and Fuel-Flow)",
        "- `xgb_prediction` / `lgbm_prediction` / `cat_prediction` = Direct-target aliases",
        "- `openap_prediction` = existing `physics_fuel_kg` (OpenAP baseline)",
        "- `dynamic_mass` = existing `r3_mean_mass_kg`",
        "- `residual` = existing `residual_kg` (actual − OpenAP)",
        "- Train rows use GroupKFold OOF base predictions (leakage-safe soft labels)",
        "",
        "### What this stage does *not* do",
        "",
        "- Train any neural student",
        "- Implement distillation loss",
        "- Change AeroTwin feature engineering or hyperparameters",
        "",
        f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote report %s", path)


def build_distillation_dataset(
    *,
    force_rebuild: bool = False,
    include_eval_splits: bool = True,
) -> tuple[Path, Path]:
    root = _root()
    LOGGER.info("Project root: %s", root)

    train = _load_split("train", root)
    if train is None:
        raise FileNotFoundError(
            "featured_dataset.parquet (train) is required. "
            "Place it at the project root or build via aerotwin.engine.build_featured_dataset."
        )

    LOGGER.info("Enriching train with frozen R3 mass features...")
    train = enrich_mass_from_columns(train)
    feat_cols = _feature_cols(train)
    mass_n = sum(1 for c in R3_MASS_FEATURES if c in train.columns)
    LOGGER.info(
        "Feature vector: %d columns (base+energy+weather+physics+cats + %d mass)",
        len(feat_cols),
        mass_n,
    )

    bundle = _build_or_load_teacher(
        train, feat_cols, _cache_path(root), force=force_rebuild
    )

    frames: list[pl.DataFrame] = []
    split_counts: dict[str, int] = {}

    # Train: OOF teacher knowledge
    frames.append(
        _assemble_split_frame(
            train,
            feat_cols,
            split="train",
            P=bundle["P_oof"],
            ridge=bundle["ridge_oof"],
            teacher=bundle["teacher_oof"],
            sample_id_offset=0,
        )
    )
    split_counts["train"] = len(train)
    offset = len(train)

    if include_eval_splits:
        for split in ("rank", "final"):
            raw = _load_split(split, root)
            if raw is None:
                continue
            enriched = enrich_mass_from_columns(raw)
            enriched = ensure_features(enriched, feat_cols)
            P, ridge, teacher = _predict_holdout(bundle, enriched, feat_cols)
            frames.append(
                _assemble_split_frame(
                    enriched,
                    feat_cols,
                    split=split,
                    P=P,
                    ridge=ridge,
                    teacher=teacher,
                    sample_id_offset=offset,
                )
            )
            split_counts[split] = len(enriched)
            offset += len(enriched)

    dataset = pl.concat(frames, how="vertical_relaxed")
    out_parquet = root / OUT_PARQUET_NAME
    dataset.write_parquet(out_parquet)
    LOGGER.info("Wrote %s (%d rows, %d cols)", out_parquet, len(dataset), len(dataset.columns))

    aux_targets = [
        "teacher_prediction",
        "ridge_prediction",
        *BASE_PRED_COLS,
        *ALIAS_PRED_COLS.keys(),
        "calibrated_prediction",
    ]
    for out_name, src_name in AUX_SOURCE_COLS.items():
        if out_name in dataset.columns:
            aux_targets.append(out_name)
    if "p1e_phase_group" in dataset.columns:
        aux_targets.append("p1e_phase_group")

    report_path = root / "docs" / "reports" / OUT_REPORT_NAME
    _write_report(
        report_path,
        df=dataset,
        feat_cols=feat_cols,
        aux_targets=aux_targets,
        parquet_path=out_parquet,
        bundle=bundle,
        split_counts=split_counts,
    )

    meta = {
        "n_samples": len(dataset),
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "split_counts": split_counts,
        "aux_targets": aux_targets,
        "parquet": str(out_parquet),
        "report": str(report_path),
        "teacher": {
            "variant": "R3_P1E_phase_affine",
            "meta_kind": bundle.get("meta_kind"),
            "bases": [list(b) for b in ENSEMBLE_BASES],
            "oof_rmse_pre_p1e": bundle.get("oof_rmse"),
            "oof_rmse_teacher": bundle.get("p1e_oof_rmse"),
            "oof_bias_teacher": bundle.get("p1e_oof_bias"),
            "reference_combined_rmse": 221.33,
            "reference_final_rmse": 213.73,
        },
        "missing": _missing_report(dataset),
        "parquet_bytes": out_parquet.stat().st_size,
    }
    meta_path = root / "docs" / "reports" / OUT_META_NAME
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Wrote meta %s", meta_path)
    return out_parquet, report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore teacher OOF cache and rebuild base models",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Export only the train split (skip rank/final even if present)",
    )
    args = parser.parse_args(argv)

    out_parquet, report_path = build_distillation_dataset(
        force_rebuild=args.force_rebuild,
        include_eval_splits=not args.train_only,
    )
    print("\n=== DISTILLATION DATASET READY ===")
    print(f"parquet: {out_parquet}")
    print(f"report:  {report_path}")


if __name__ == "__main__":
    main()
