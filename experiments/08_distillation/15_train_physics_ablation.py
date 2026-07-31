"""Phase 3A — Train Large MLP and FT without physics-derived features.

Targeted ablations only: same KD (α=0.1, β=0.9), same split, same architecture
templates; feature set excludes physics/mass/energy columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData, load_feature_cols
from aerotwin.distillation.models import StudentConfig, build_student
from aerotwin.distillation.physics_features import nophysics_feature_cols, split_features
from aerotwin.distillation.runner import ExperimentConfig, KDWeightConfig, run_single_experiment
from aerotwin.distillation.trainer import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("physics_ablation")

OUT = ROOT / "results" / "distillation" / "mechanism_validation" / "physics_ablation"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", choices=("large", "ft", "both"), default="both")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    full_feats = load_feature_cols(ROOT)
    phys, keep = split_features(full_feats)
    LOGGER.info("Full features=%d physics=%d keep=%d", len(full_feats), len(phys), len(keep))
    LOGGER.info("Removed physics: %s", phys)

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet",
        root=ROOT,
        feature_cols=keep,
        val_fraction=0.2,
        seed=42,
    )
    LOGGER.info("in_dim without physics = %d", data.in_dim)

    device = args.device
    weight = KDWeightConfig(name="kd1", alpha=0.1, beta=0.9)
    exp = ExperimentConfig(
        seed=42,
        val_fraction=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=2048,
        max_epochs=80,
        patience=12,
        min_delta=0.05,
        device=device,
        extras={"ablation": "nophysics", "removed_features": phys},
    )

    meta = {
        "feature_cols_kept": keep,
        "feature_cols_removed": phys,
        "in_dim": data.in_dim,
        "n_num": len(data.numeric_cols),
        "n_cat": len(data.cat_cols),
        "cat_cardinalities": [len(c) for c in data.ohe.categories_],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_sets.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    runs = []
    if args.only in ("large", "both"):
        runs.append(("large_nophysics", "large_mlp", dict(hidden_dims=(1792, 1024), dropout=0.1)))
    if args.only in ("ft", "both"):
        # FT paper baseline; need cat cards for remaining cats
        runs.append(
            (
                "ft_nophysics",
                "ft_transformer",
                dict(
                    d_token=192,
                    n_blocks=3,
                    n_heads=8,
                    attention_dropout=0.2,
                    ffn_dropout=0.1,
                    n_num_features=len(data.numeric_cols),
                    cat_cardinalities=[len(c) for c in data.ohe.categories_],
                ),
            )
        )

    summary = []
    for run_name, arch, kwargs in runs:
        out_dir = OUT / run_name
        if (out_dir / "best_model.pt").exists() and not args.force:
            LOGGER.info("Skip existing %s", run_name)
            if (out_dir / "metrics.json").exists():
                m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
                summary.append({"run": run_name, "best_val_rmse": m.get("best_val_rmse")})
            continue

        def factory(in_dim: int, _arch=arch, _kw=kwargs):
            return build_student(_arch, in_dim=in_dim, **_kw)

        # FT needs smaller batch if many tokens - with fewer num features still ok at 1024
        exp_run = ExperimentConfig(
            seed=42,
            val_fraction=0.2,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=1024 if arch == "ft_transformer" else 2048,
            max_epochs=80,
            patience=12,
            min_delta=0.05,
            device=device,
            extras={
                "ablation": "nophysics",
                "architecture": arch,
                "removed_features": phys,
                "kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
            },
        )
        LOGGER.info("=== Training %s arch=%s in_dim=%d ===", run_name, arch, data.in_dim)
        t0 = time.time()
        metrics = run_single_experiment(
            data=data,
            model_factory=factory,
            weight=weight,
            exp=exp_run,
            out_dir=out_dir,
            log_dir=OUT / "logs" / run_name,
            model_dir=OUT / "models" / run_name,
        )
        metrics["wall_seconds"] = time.time() - t0
        metrics["feature_meta"] = meta
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        summary.append(
            {
                "run": run_name,
                "architecture": arch,
                "best_val_rmse": metrics.get("best_val_rmse"),
                "n_params": metrics.get("n_params"),
                "wall_seconds": metrics["wall_seconds"],
            }
        )
        LOGGER.info("%s done val_rmse=%.2f", run_name, metrics.get("best_val_rmse", float("nan")))

    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== PHYSICS ABLATION TRAINING DONE ===")
    for s in summary:
        print(f"  {s}")


if __name__ == "__main__":
    main()
