#!/usr/bin/env python
"""Unified entry point for AeroTwin distillation student experiments.

Examples
--------
# Full α/β KD weight sweep (Step 3 default)
python experiments/08_distillation/run_distillation_experiments.py sweep

# Baseline A/B/C MLP (Step 2)
python experiments/08_distillation/run_distillation_experiments.py baseline

# Forward extra flags to the underlying script, e.g.:
python experiments/08_distillation/run_distillation_experiments.py sweep --only KD-0,KD-4
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_and_run(script_name: str, rest: list[str]) -> None:
    path = HERE / script_name
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        raise RuntimeError(f"{script_name} has no main()")
    mod.main(rest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AeroTwin distillation experiment runner",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="sweep",
        choices=(
            "sweep",
            "baseline",
            "alpha-beta",
            "capacity",
            "ft-transformer",
            "ft_transformer",
        ),
        help="sweep/baseline/capacity (MLP) or ft-transformer (Phase 2 student)",
    )
    args, rest = parser.parse_known_args(argv)

    if args.command in ("sweep", "alpha-beta"):
        _load_and_run("03_alpha_beta_sweep.py", rest)
    elif args.command == "baseline":
        _load_and_run("02_train_mlp_student.py", rest)
    elif args.command == "capacity":
        _load_and_run("04_capacity_scaling.py", rest)
    elif args.command in ("ft-transformer", "ft_transformer"):
        _load_and_run("08_train_ft_transformer.py", rest)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
