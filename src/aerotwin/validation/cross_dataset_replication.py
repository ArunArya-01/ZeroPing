from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import project_root  # noqa: E402
from aerotwin.validation.external_vs_flow_eval import (  # noqa: E402
    external_results_table,
    run_protocol,
)

# P(Flow+Energy better) at or above this counts the Flow-over-Direct finding as
# "replicated" on a single dataset. Mirrors the bootstrap significance threshold
# used elsewhere in the project (α = 0.05, one-sided).
REPLICATION_P_THRESHOLD = 0.95


def run_protocol_on_datasets(
    datasets: Sequence[tuple[str, pl.DataFrame]],
    test_size: float = 0.2,
    iterations: int = 500,
) -> list[dict]:
    """Run the equivalent Flow-vs-Direct protocol on several datasets.

    ``datasets`` is an iterable of ``(name, dataframe)`` pairs. Each dataset is
    evaluated with ``run_protocol`` (which requires CatBoost) and the resulting
    approach-level table is attached so a cross-dataset comparison can be built.

    Returns a list of dicts with ``name``, the raw ``result`` from
    ``run_protocol``, and the flattened ``table`` from ``external_results_table``.
    """
    out: list[dict] = []
    for name, df in datasets:
        result = run_protocol(df, test_size=test_size, iterations=iterations)
        out.append(
            {
                "name": name,
                "result": result,
                "table": external_results_table(result),
            }
        )
    return out


def dataset_replicates_flow_better(table: pl.DataFrame, threshold: float = REPLICATION_P_THRESHOLD) -> dict:
    """Decide whether one dataset replicates the Flow+Energy-over-Direct finding.

    A dataset replicates when Flow+Energy has a strictly lower MAE than Direct
    *and* the bootstrap probability that Flow is better meets ``threshold``.

    Returns a flat dict with the comparison metrics and a boolean ``replicated``.
    """
    flow = table.filter(pl.col("approach") == "flow")
    direct = table.filter(pl.col("approach") == "direct")
    if flow.is_empty() or direct.is_empty():
        raise ValueError("Replication table must contain both 'flow' and 'direct' rows.")

    direct_mae = float(direct["mae_kg"][0])
    flow_mae = float(flow["mae_kg"][0])
    delta_mae = float(flow["flow_minus_direct_delta_mae"][0])
    p_better = float(flow["bootstrap_p_flow_better"][0])
    replicated = (flow_mae < direct_mae) and (p_better >= threshold)

    return {
        "direct_mae_kg": direct_mae,
        "flow_mae_kg": flow_mae,
        "delta_mae_kg": delta_mae,
        "p_flow_better": p_better,
        "n_test_flights": int(flow["n_test_flights"][0]),
        "n_test_intervals": int(flow["n_test_intervals"][0]),
        "replicated": bool(replicated),
        "interpretation": str(flow["interpretation"][0]),
    }


def build_replication_table(results: Iterable[dict]) -> pl.DataFrame:
    """Combine per-dataset protocol results into a cross-dataset table.

    Each row reports whether that dataset replicated the qualitative finding
    (Flow+Energy generalizes better than Direct) so findings can be compared
    across datasets rather than within a single source.
    """
    rows = []
    for entry in results:
        name = entry["name"]
        decision = dataset_replicates_flow_better(entry["table"])
        rows.append(
            {
                "dataset": name,
                "n_test_flights": decision["n_test_flights"],
                "n_test_intervals": decision["n_test_intervals"],
                "direct_mae_kg": decision["direct_mae_kg"],
                "flow_mae_kg": decision["flow_mae_kg"],
                "delta_mae_kg": decision["delta_mae_kg"],
                "p_flow_better": decision["p_flow_better"],
                "replicated": decision["replicated"],
                "interpretation": decision["interpretation"],
            }
        )
    return pl.DataFrame(rows)


def aggregate_replication(table: pl.DataFrame) -> dict:
    """Summarize how many datasets replicated the finding.

    Produces a meta-verdict consistent with the project interpretation policy:
    consistent replication across datasets supports cross-dataset robustness,
    while failures (or only partial replication) are reported explicitly rather
    than hidden.
    """
    n = table.height
    if n == 0:
        return {
            "n_datasets": 0,
            "n_replicated": 0,
            "fraction_replicated": float("nan"),
            "verdict": "No datasets evaluated",
        }
    n_rep = int(table["replicated"].sum())
    frac = n_rep / n
    if n_rep == n:
        verdict = "Finding replicated across all datasets"
    elif n_rep == 0:
        verdict = "Finding failed to replicate on any dataset"
    else:
        verdict = "Partial replication across datasets"
    return {
        "n_datasets": n,
        "n_replicated": n_rep,
        "fraction_replicated": float(frac),
        "verdict": verdict,
    }


def write_replication_outputs(table: pl.DataFrame, outdir: Path | None = None) -> Path:
    """Persist the replication table and return its path.

    The CSV is the cross-dataset replication deliverable described in the
    project status report (§20, cross-dataset comparison table).
    """
    outdir = outdir or (project_root() / "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "table_cross_dataset_replication.csv"
    table.write_csv(path)
    return path


def main(
    external_paths: Sequence[str],
    outdir: Path | None = None,
    test_size: float = 0.2,
    iterations: int = 500,
) -> None:
    outdir = outdir or (project_root() / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CROSS-DATASET REPLICATION ANALYSIS — Flow+Energy vs Direct")
    print("=" * 72)

    import polars as _pl

    datasets = []
    for p in external_paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"Dataset not found: {path}")
        df = _pl.read_parquet(path)
        datasets.append((path.stem, df))

    results = run_protocol_on_datasets(datasets, test_size=test_size, iterations=iterations)
    table = build_replication_table(results)
    agg = aggregate_replication(table)

    print(f"\nDatasets evaluated: {agg['n_datasets']}")
    print(f"Replicated finding : {agg['n_replicated']} / {agg['n_datasets']} "
          f"({agg['fraction_replicated']:.0%})")
    print(f"Verdict            : {agg['verdict']}")
    print("\nPer-dataset replication:")
    for row in table.iter_rows(named=True):
        flag = "REPLICATED" if row["replicated"] else "not replicated"
        print(f"  {row['dataset']:28s} ΔMAE={row['delta_mae_kg']:+7.1f} kg "
              f"P(Flow better)={row['p_flow_better']:.3f} -> {flag}")

    path = write_replication_outputs(table, outdir)
    print(f"\nSaved {path}")
    print("=" * 72)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python physics/cross_dataset_replication.py PATH1 [PATH2 ...]")
        print("       [--outdir DIR] [--test-size 0.2] [--iterations 500]")
        raise SystemExit(0)

    paths: list[str] = []
    outdir = None
    test_size = 0.2
    iterations = 500
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--outdir":
            outdir = Path(args[i + 1]); i += 2
        elif a == "--test-size":
            test_size = float(args[i + 1]); i += 2
        elif a == "--iterations":
            iterations = int(args[i + 1]); i += 2
        else:
            paths.append(a); i += 1

    if not paths:
        raise SystemExit("Provide at least one external dataset parquet path.")
    main(paths, outdir=outdir, test_size=test_size, iterations=iterations)
