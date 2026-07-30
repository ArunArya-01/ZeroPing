"""Phase 1B — Train Variance-Guided KD (VGKD) Large MLP students.

Only the KD weighting changes. Architecture = Large MLP (~2.89M).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aerotwin.distillation.data import DistillationData
from aerotwin.distillation.metrics import compare_to_teacher, regression_metrics
from aerotwin.distillation.mlp import StudentMLP
from aerotwin.distillation.trainer import set_seed
from aerotwin.distillation.vgkd import (
    BASE_PRED_COLS,
    VGKDConfig,
    VGKDDataset,
    adaptive_beta,
    ensemble_std_from_bases,
    fit_zscore,
    vgkd_loss,
    zscore,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("vgkd_train")

OUT_ROOT = ROOT / "results" / "distillation" / "vgkd"


def _load_uncertainty(
    data: DistillationData,
    *,
    source: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, Any]]:
    """Return u_train_raw, u_val_raw, mu, sig, meta."""
    df = pl.read_parquet(data.parquet_path).filter(
        pl.col("ground_truth").is_finite()
        & pl.col("teacher_prediction").is_finite()
        & pl.col("flight_id").is_not_null()
    )
    missing = [c for c in BASE_PRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base prediction columns: {missing}")

    if source == "ensemble_std":
        P = np.column_stack([df[c].to_numpy().astype(np.float64) for c in BASE_PRED_COLS])
        u_all = ensemble_std_from_bases(P)
    elif source == "oracle_abs_error":
        # Analysis only — uses train GT (not for deployment)
        u_all = np.abs(
            df["teacher_prediction"].to_numpy().astype(np.float64)
            - df["ground_truth"].to_numpy().astype(np.float64)
        )
    elif source == "random":
        rng = np.random.default_rng(seed)
        # Match train ensemble_std distribution (fit on train after split)
        P = np.column_stack([df[c].to_numpy().astype(np.float64) for c in BASE_PRED_COLS])
        u_ref = ensemble_std_from_bases(P)
        u_tr_ref = u_ref[data.train_idx]
        # sample with replacement from train ref dist for all rows
        u_all = rng.choice(u_tr_ref, size=len(df), replace=True)
    else:
        raise ValueError(source)

    u_train = u_all[data.train_idx]
    u_val = u_all[data.val_idx]
    mu, sig = fit_zscore(u_train)
    meta = {
        "source": source,
        "u_mean_train": mu,
        "u_std_train": sig,
        "u_train_mean": float(np.mean(u_train)),
        "u_train_p95": float(np.percentile(u_train, 95)),
    }
    return u_train, u_val, mu, sig, meta


def train_one(
    *,
    run_name: str,
    data: DistillationData,
    u_train: np.ndarray,
    u_val: np.ndarray,
    cfg: VGKDConfig,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 2048,
    max_epochs: int = 80,
    patience: int = 12,
    min_delta: float = 0.05,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    u_n_tr = zscore(u_train, cfg.u_mean, cfg.u_std)
    u_n_va = zscore(u_val, cfg.u_mean, cfg.u_std)

    # Diagnostics for β distribution on train
    beta_tr = adaptive_beta(
        u_n_tr,
        beta_base=cfg.beta_base,
        lam=cfg.lam,
        weight_fn=cfg.weight_fn,
        static_beta=cfg.static_beta,
    )
    LOGGER.info(
        "%s | mean β=%.4f min=%.4f max=%.4f | mean α=%.4f",
        run_name,
        float(np.mean(beta_tr)),
        float(np.min(beta_tr)),
        float(np.max(beta_tr)),
        float(np.mean(1.0 - beta_tr)),
    )

    train_ds = VGKDDataset(
        data.x_train, data.y_gt_train, data.y_teacher_train, u_n_tr, data.sample_id_train
    )
    val_ds = VGKDDataset(
        data.x_val, data.y_gt_val, data.y_teacher_val, u_n_va, data.sample_id_val
    )
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin
    )
    train_eval = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )

    model = StudentMLP(data.in_dim, hidden_dims=(1792, 1024), dropout=0.1).to(device)
    n_params = model.count_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    best_val = float("inf")
    best_epoch = -1
    patience_left = patience
    history: list[dict[str, float]] = []
    ckpt = out_dir / "best_model.pt"
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        mean_betas = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y_gt = batch["y_gt"].to(device)
            y_t = batch["y_teacher"].to(device)
            u = batch["u_norm"].to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss, parts = vgkd_loss(pred, y_gt, y_t, u, cfg=cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(parts["loss"])
            mean_betas.append(parts["mean_beta"])

        # eval
        model.eval()
        with torch.no_grad():
            def _pred(loader):
                ps, gs, ts = [], [], []
                for b in loader:
                    p = model(b["x"].to(device)).cpu().numpy()
                    ps.append(p)
                    gs.append(b["y_gt"].numpy())
                    ts.append(b["y_teacher"].numpy())
                return np.concatenate(ps), np.concatenate(gs), np.concatenate(ts)

            tr_p, tr_g, tr_t = _pred(train_eval)
            va_p, va_g, va_t = _pred(val_loader)
        tr_m = regression_metrics(tr_g, tr_p)
        va_m = regression_metrics(va_g, va_p)
        va_t_m = regression_metrics(va_g, va_t)
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "mean_beta": float(np.mean(mean_betas)),
            "train_rmse": tr_m["rmse"],
            "val_rmse": va_m["rmse"],
            "val_mae": va_m["mae"],
            "val_bias": va_m["bias"],
            "val_r2": va_m["r2"],
            "val_teacher_rmse": va_t_m["rmse"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        LOGGER.info(
            "%s ep%03d loss=%.4f β=%.3f train=%.2f val=%.2f",
            run_name,
            epoch,
            row["train_loss"],
            row["mean_beta"],
            row["train_rmse"],
            row["val_rmse"],
        )
        sched.step(row["val_rmse"])
        if row["val_rmse"] < best_val - min_delta:
            best_val = row["val_rmse"]
            best_epoch = epoch
            patience_left = patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vgkd_config": cfg.to_dict(),
                    "best_val_rmse": best_val,
                    "best_epoch": best_epoch,
                    "n_params": n_params,
                    "in_dim": data.in_dim,
                    "hidden_dims": (1792, 1024),
                    "architecture": "large_mlp_vgkd",
                    "run_name": run_name,
                },
                ckpt,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                LOGGER.info("Early stop at %d (best %d val=%.2f)", epoch, best_epoch, best_val)
                break

    train_seconds = time.time() - t0
    if ckpt.exists():
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])

    model.eval()
    with torch.no_grad():
        def _pred(loader):
            ps, gs, ts = [], [], []
            for b in loader:
                p = model(b["x"].to(device)).cpu().numpy()
                ps.append(p)
                gs.append(b["y_gt"].numpy())
                ts.append(b["y_teacher"].numpy())
            return np.concatenate(ps), np.concatenate(gs), np.concatenate(ts)

        tr_p, tr_g, tr_t = _pred(train_eval)
        va_p, va_g, va_t = _pred(val_loader)

    metrics = {
        "run_name": run_name,
        "vgkd": cfg.to_dict(),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val,
        "train_seconds": train_seconds,
        "train": compare_to_teacher(tr_g, tr_p, tr_t),
        "val": compare_to_teacher(va_g, va_p, va_t),
        "beta_train_stats": {
            "mean": float(np.mean(beta_tr)),
            "std": float(np.std(beta_tr)),
            "min": float(np.min(beta_tr)),
            "max": float(np.max(beta_tr)),
            "p05": float(np.percentile(beta_tr, 5)),
            "p50": float(np.percentile(beta_tr, 50)),
            "p95": float(np.percentile(beta_tr, 95)),
        },
    }
    pl.DataFrame(history).write_csv(out_dir / "training_curve.csv")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "vgkd_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return metrics


def _run_specs() -> list[dict[str, Any]]:
    """Full experimental grid."""
    specs: list[dict[str, Any]] = []
    # λ sweep (exp)
    for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
        specs.append(
            {
                "run_name": f"vgkd_exp_lam{lam}",
                "lam": lam,
                "weight_fn": "exp",
                "uncertainty_source": "ensemble_std",
                "static_beta": None,
                "group": "lambda_sweep",
            }
        )
    # A1 static β
    for b in [0.7, 0.8, 0.9]:
        specs.append(
            {
                "run_name": f"static_beta{b}",
                "lam": 0.0,
                "weight_fn": "exp",
                "uncertainty_source": "ensemble_std",
                "static_beta": b,
                "group": "static_beta",
            }
        )
    # A3 linear weight (same λ grid except 0)
    for lam in [0.25, 0.5, 1.0, 2.0]:
        specs.append(
            {
                "run_name": f"vgkd_lin_lam{lam}",
                "lam": lam,
                "weight_fn": "linear",
                "uncertainty_source": "ensemble_std",
                "static_beta": None,
                "group": "linear",
            }
        )
    # A2 random uncertainty (use λ=1.0 default; also re-run best after if needed)
    specs.append(
        {
            "run_name": "vgkd_random_lam1.0",
            "lam": 1.0,
            "weight_fn": "exp",
            "uncertainty_source": "random",
            "static_beta": None,
            "group": "random",
        }
    )
    # A4 oracle
    specs.append(
        {
            "run_name": "vgkd_oracle_lam1.0",
            "lam": 1.0,
            "weight_fn": "exp",
            "uncertainty_source": "oracle_abs_error",
            "static_beta": None,
            "group": "oracle",
        }
    )
    return specs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default=None, help="Comma-separated run names to train")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    )
    LOGGER.info("Device %s", device)

    data = DistillationData.from_parquet(
        ROOT / "distillation_dataset.parquet", root=ROOT, val_fraction=0.2, seed=42
    )

    # Cache uncertainty by source
    u_cache: dict[str, tuple[np.ndarray, np.ndarray, float, float, dict]] = {}

    specs = _run_specs()
    if args.only:
        allow = set(args.only.split(","))
        specs = [s for s in specs if s["run_name"] in allow]

    summary_rows = []
    for spec in specs:
        run = spec["run_name"]
        out_dir = OUT_ROOT / "runs" / run
        if (out_dir / "best_model.pt").exists() and args.skip_existing and not args.force:
            LOGGER.info("Skip existing %s", run)
            if (out_dir / "metrics.json").exists():
                m = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
                summary_rows.append(
                    {
                        "run_name": run,
                        "group": spec["group"],
                        "best_val_rmse": m.get("best_val_rmse"),
                        "n_params": m.get("n_params"),
                    }
                )
            continue

        src = spec["uncertainty_source"]
        if src not in u_cache:
            u_cache[src] = _load_uncertainty(data, source=src, seed=42)
        u_tr, u_va, mu, sig, umeta = u_cache[src]

        cfg = VGKDConfig(
            beta_base=0.9,
            lam=float(spec["lam"]),
            weight_fn=spec["weight_fn"],
            uncertainty_source=src,
            u_mean=mu,
            u_std=sig,
            static_beta=spec["static_beta"],
            seed=42,
        )
        LOGGER.info("=== Training %s ===", run)
        metrics = train_one(
            run_name=run,
            data=data,
            u_train=u_tr,
            u_val=u_va,
            cfg=cfg,
            out_dir=out_dir,
            device=device,
        )
        metrics["uncertainty_meta"] = umeta
        metrics["group"] = spec["group"]
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        summary_rows.append(
            {
                "run_name": run,
                "group": spec["group"],
                "lam": cfg.lam,
                "weight_fn": cfg.weight_fn,
                "static_beta": cfg.static_beta,
                "uncertainty_source": src,
                "best_val_rmse": metrics["best_val_rmse"],
                "best_epoch": metrics["best_epoch"],
                "mean_beta": metrics["beta_train_stats"]["mean"],
                "n_params": metrics["n_params"],
                "train_seconds": metrics["train_seconds"],
            }
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(OUT_ROOT / "training_summary.csv")
    (OUT_ROOT / "training_summary.json").write_text(
        json.dumps(summary_rows, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== VGKD TRAINING DONE ===")
    for r in summary_rows:
        print(f"  {r['run_name']}: val_rmse={r.get('best_val_rmse')}")


if __name__ == "__main__":
    main()
