"""Phase 2 — Train FT-Transformer student under frozen KD pipeline.

Identical to Large MLP setup except architecture:
  teacher, dataset, split, α=0.1, β=0.9, optimizer (AdamW), early stopping.

Usage
-----
  set PYTHONPATH=src
  python experiments/08_distillation/08_train_ft_transformer.py
  python experiments/08_distillation/08_train_ft_transformer.py --config configs/distillation/ft_transformer.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.models import StudentConfig, build_student, list_architectures
from aerotwin.distillation.runner import (
    ExperimentConfig,
    KDWeightConfig,
    flatten_run_metrics,
    run_single_experiment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("ft_transformer_train")

DEFAULT_CONFIG = ROOT / "configs" / "distillation" / "ft_transformer.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--architecture",
        default=None,
        help=f"Override architecture (registered: {list_architectures()})",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d-token", type=int, default=None)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results root (default from config)",
    )
    args = ap.parse_args(argv)

    cfg_path = args.config if args.config.exists() else None
    file_cfg: dict[str, Any] = _load_yaml(cfg_path) if cfg_path else {}
    if cfg_path:
        LOGGER.info("Loaded config %s", cfg_path)
    else:
        LOGGER.warning("Config not found at %s — using defaults", args.config)

    student_section = dict(file_cfg.get("student") or {})
    if args.architecture:
        student_section["architecture"] = args.architecture
    if args.d_token is not None:
        student_section["d_token"] = args.d_token
    if args.n_blocks is not None:
        student_section["n_blocks"] = args.n_blocks
    if args.n_heads is not None:
        student_section["n_heads"] = args.n_heads

    student_cfg = StudentConfig.from_mapping(student_section)
    arch = student_cfg.architecture

    alpha = float(args.alpha if args.alpha is not None else _deep_get(file_cfg, "kd", "alpha", default=0.1))
    beta = float(args.beta if args.beta is not None else _deep_get(file_cfg, "kd", "beta", default=0.9))

    seed = int(args.seed if args.seed is not None else _deep_get(file_cfg, "train", "seed", default=42))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else _deep_get(file_cfg, "train", "batch_size", default=512)
    )
    max_epochs = int(
        args.max_epochs
        if args.max_epochs is not None
        else _deep_get(file_cfg, "train", "max_epochs", default=80)
    )
    lr = float(args.lr if args.lr is not None else _deep_get(file_cfg, "train", "lr", default=1e-3))
    device = str(args.device or _deep_get(file_cfg, "train", "device", default="auto"))
    weight_decay = float(_deep_get(file_cfg, "train", "weight_decay", default=1e-4))
    patience = int(_deep_get(file_cfg, "train", "patience", default=12))
    min_delta = float(_deep_get(file_cfg, "train", "min_delta", default=0.05))
    val_fraction = float(_deep_get(file_cfg, "train", "val_fraction", default=0.2))
    grad_clip = float(_deep_get(file_cfg, "train", "grad_clip", default=1.0))
    num_workers = int(_deep_get(file_cfg, "train", "num_workers", default=0))

    run_name = str(_deep_get(file_cfg, "run", "name", default=f"{arch}_kd1"))
    results_root = Path(
        args.results_dir
        or _deep_get(file_cfg, "run", "results_dir", default="results/distillation/ft_transformer")
    )
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    models_root = Path(
        _deep_get(file_cfg, "run", "models_dir", default="models/distillation/ft_transformer")
    )
    if not models_root.is_absolute():
        models_root = ROOT / models_root
    logs_root = Path(
        _deep_get(file_cfg, "run", "logs_dir", default="logs/distillation/ft_transformer")
    )
    if not logs_root.is_absolute():
        logs_root = ROOT / logs_root

    parquet = ROOT / str(
        _deep_get(file_cfg, "data", "distillation_parquet", default="distillation_dataset.parquet")
    )
    if not parquet.exists():
        raise FileNotFoundError(f"Missing frozen distillation dataset: {parquet}")

    LOGGER.info(
        "Architecture=%s  α=%.2f β=%.2f  seed=%d  batch=%d  d_token=%d n_blocks=%d",
        arch,
        alpha,
        beta,
        seed,
        batch_size,
        student_cfg.d_token,
        student_cfg.n_blocks,
    )

    data = DistillationData.from_parquet(
        parquet, root=ROOT, val_fraction=val_fraction, seed=seed
    )
    student_cfg.in_dim = data.in_dim
    # Paper-native token layout decoded from dense [num | OHE] (same preprocessing)
    student_cfg.n_num_features = len(data.numeric_cols)
    student_cfg.cat_cardinalities = [len(c) for c in data.ohe.categories_]
    LOGGER.info(
        "Tabular tokens: n_num=%d cat_cards=%s (n_tokens≈%d incl. CLS)",
        student_cfg.n_num_features,
        student_cfg.cat_cardinalities,
        1 + student_cfg.n_num_features + len(student_cfg.cat_cardinalities),
    )

    # Smoke-build to log param count
    probe = build_student(student_cfg, in_dim=data.in_dim)
    n_params = int(sum(p.numel() for p in probe.parameters() if p.requires_grad))
    LOGGER.info("Student params: %s  in_dim=%d", f"{n_params:,}", data.in_dim)
    del probe

    def model_factory(in_dim: int):
        return build_student(student_cfg, in_dim=in_dim)

    exp = ExperimentConfig(
        seed=seed,
        val_fraction=val_fraction,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        grad_clip=grad_clip,
        device=device,
        num_workers=num_workers,
        dropout=student_cfg.dropout,
        extras={
            "architecture": arch,
            "student_config": student_cfg.to_dict(),
            "stage": "phase2_ft_transformer",
            "config_path": str(cfg_path) if cfg_path else None,
        },
    )
    weight = KDWeightConfig(name=run_name, alpha=alpha, beta=beta)

    out_dir = results_root / run_name
    t0 = time.time()
    metrics = run_single_experiment(
        data=data,
        model_factory=model_factory,
        weight=weight,
        exp=exp,
        out_dir=out_dir,
        log_dir=logs_root / run_name,
        model_dir=models_root / run_name,
    )
    elapsed = time.time() - t0

    flat = flatten_run_metrics(metrics)
    flat["architecture"] = arch
    flat["n_params"] = n_params
    summary = {
        "architecture": arch,
        "alpha": alpha,
        "beta": beta,
        "n_params": n_params,
        "in_dim": data.in_dim,
        "student_config": student_cfg.to_dict(),
        "train_config": {
            "seed": seed,
            "batch_size": batch_size,
            "lr": lr,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        "metrics": metrics,
        "flat": flat,
        "wall_seconds": elapsed,
        "checkpoint": str((out_dir / "best_model.pt").resolve()),
        "baselines": {
            "large_mlp_final_rmse": 215.85,
            "large_mlp_combined_rmse": 225.95,
            "teacher_final_rmse": 213.62,
            "teacher_combined_rmse": 221.33,
            "note": "Val RMSE is flight-holdout only; run 09_eval_ft_transformer for Final/Combined",
        },
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "student_config.json").write_text(
        json.dumps(student_cfg.to_dict(), indent=2), encoding="utf-8"
    )

    print("\n=== FT-TRANSFORMER TRAINING COMPLETE ===")
    print(f"  architecture={arch}")
    print(f"  params={n_params:,}")
    print(f"  best_val_rmse={metrics.get('best_val_rmse'):.2f}")
    print(f"  best_epoch={metrics.get('best_epoch')}")
    print(f"  checkpoint={out_dir / 'best_model.pt'}")
    print(f"  wall_seconds={elapsed:.1f}")
    print(f"  Large MLP val reference ~229.70 (not Final)")


if __name__ == "__main__":
    main()
