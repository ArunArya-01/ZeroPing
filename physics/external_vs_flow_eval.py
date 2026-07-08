from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    CATEGORICAL,
    evaluate,
    flight_level_split,
    project_root,
    significance_test,
)
from physics.feature_engineering import ENERGY_FEATURES  # noqa: E402
from physics.weather_features import WEATHER_FEATURES  # noqa: E402

# Categorical features mirror notebooks/15 (the source protocol) so the
# external run is directly comparable to the internal AeroTwin result.
CAT_FEATURES = CATEGORICAL + ["phase"]

RANDOM_STATE = 42

# Feature groupings copied verbatim from the AeroTwin Flow-vs-Direct protocol
# (notebooks/15_leave_one_type_out.py). Keeping them identical is what makes
# the external run "equivalent" to the internal one.
EW_FEATURES = list(BASE_NUMERIC) + list(ENERGY_FEATURES) + list(WEATHER_FEATURES) + ["physics_fuel_kg"] + list(CATEGORICAL)
FLOW_ENERGY_FEATURES = list(BASE_NUMERIC) + list(ENERGY_FEATURES) + list(CATEGORICAL)


def clean_for_eval(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows unusable for evaluation without requiring every internal column.

    An independent external dataset may lack ``residual_kg``; we only require
    the columns the protocol actually consumes (the fuel label, duration,
    flight id, and any present physics baseline).
    """
    required = ["actual_fuel_kg", "duration_s", "flight_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"External dataset missing required columns: {missing}")
    out = df.drop_nulls(subset=required)
    if "physics_fuel_kg" in out.columns:
        out = out.filter(pl.col("physics_fuel_kg").is_finite())
    out = out.filter(
        pl.col("actual_fuel_kg").is_finite()
        & pl.col("duration_s").is_finite()
        & (pl.col("duration_s") > 0)
    )
    return out


def avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    """Subset ``cols`` to those actually present in ``df``."""
    return [c for c in cols if c in df.columns]


def transform_y(name: str, actual: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """Map interval fuel (kg) to the modeling target space."""
    dur = np.clip(duration.astype(np.float64), 1.0, None)
    actual = actual.astype(np.float64)
    if name == "direct_fuel":
        return actual
    if name == "fuel_flow":
        return actual / dur
    raise ValueError(name)


def recover_fuel(name: str, pred: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """Map predictions back to interval fuel (kg) for a fair comparison."""
    dur = np.clip(duration.astype(np.float64), 1.0, None)
    pred = pred.astype(np.float64)
    if name == "direct_fuel":
        return pred
    if name == "fuel_flow":
        return pred * dur
    raise ValueError(name)


def train_catboost(X_train, y_train, feat_cols, cat_names, iterations: int = 500):
    """Train a CatBoost regressor; imported lazily (optional dependency)."""
    from catboost import CatBoostRegressor, Pool

    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    pool = Pool(X_train, y_train, cat_features=cat_idx, feature_names=feat_cols)
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=7,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )
    model.fit(pool)
    return model


def predict_cat(model, X, feat_cols, cat_names) -> np.ndarray:
    from catboost import Pool

    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    pool = Pool(X, cat_features=cat_idx, feature_names=feat_cols)
    return np.asarray(model.predict(pool), dtype=np.float64)


def run_protocol(
    df: pl.DataFrame,
    test_size: float = 0.2,
    iterations: int = 500,
):
    """Run the equivalent Direct-vs-Flow protocol on one dataset.

    Returns a dict with per-approach metrics, recovered-fuel predictions,
    flight ids, and a significance test of Flow+Energy vs Direct.
    """
    df = clean_for_eval(df)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy()
    duration = df["duration_s"].to_numpy()

    feat_sets = {
        "ew": avail(EW_FEATURES, df),
        "flow_energy": avail(FLOW_ENERGY_FEATURES, df),
    }
    cat_names = [c for c in CAT_FEATURES if c in df.columns]

    train_idx, test_idx, _, _ = flight_level_split(fids, test_size=test_size)

    approaches = {
        "direct": ("direct_fuel", "ew", "Direct · E+W"),
        "flow": ("fuel_flow", "flow_energy", "Flow+Energy"),
    }

    out: dict[str, dict] = {}
    preds_fuel: dict[str, np.ndarray] = {}
    for key, (target, group, label) in approaches.items():
        cols = feat_sets[group]
        y_train = transform_y(target, y[train_idx], duration[train_idx])
        model = train_catboost(
            pdf[cols].iloc[train_idx], y_train, cols, cat_names, iterations=iterations
        )
        raw = predict_cat(model, pdf[cols].iloc[test_idx], cols, cat_names)
        fuel = recover_fuel(target, raw, duration[test_idx])
        mets = evaluate(y[test_idx], fuel)
        out[key] = {"label": label, "target": target, "feature_group": group, **mets}
        preds_fuel[key] = fuel

    err_direct = np.abs(preds_fuel["direct"] - y[test_idx])
    err_flow = np.abs(preds_fuel["flow"] - y[test_idx])
    sig = significance_test(err_flow, err_direct, fids[test_idx], "Flow+Energy", "Direct")

    return {
        "n_intervals": len(df),
        "n_test_intervals": int(len(test_idx)),
        "n_test_flights": int(np.unique(fids[test_idx]).size),
        "metrics": out,
        "significance": sig,
        "y_test": y[test_idx],
        "flight_ids_test": fids[test_idx],
    }


def external_results_table(result: dict) -> pl.DataFrame:
    """Flatten one external run into an approach-level results table."""
    rows = []
    for key, m in result["metrics"].items():
        rows.append(
            {
                "approach": key,
                "label": m["label"],
                "target": m["target"],
                "feature_group": m["feature_group"],
                "mae_kg": m["mae"],
                "rmse_kg": m["rmse"],
                "r2": m["r2"],
            }
        )
    sig = result["significance"]
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.lit(sig["delta_mae"]).alias("flow_minus_direct_delta_mae"),
        pl.lit(sig["ci_lower"]).alias("ci_lower"),
        pl.lit(sig["ci_upper"]).alias("ci_upper"),
        # significance_test's bootstrap_p is P(new worse); flow better is 1 - p.
        pl.lit(1.0 - sig["bootstrap_p"]).alias("bootstrap_p_flow_better"),
        pl.lit(sig["interpretation"]).alias("interpretation"),
        pl.lit(result["n_test_flights"]).alias("n_test_flights"),
        pl.lit(result["n_test_intervals"]).alias("n_test_intervals"),
    )
    return df


def load_internal_baseline(path: Path) -> pl.DataFrame | None:
    """Load the current AeroTwin results for contrast.

    Accepts the LOTO evaluation master table (``table_loto_evaluation_master.csv``)
    which already reports Direct/Energy vs Flow/Energy at the internal protocol.
    Classification is done by substring so it works whether the file stores raw
    approach keys (``global_direct_ew``) or human labels (``Global · Direct · E+W``).
    """
    if not Path(path).exists():
        return None
    df = pl.read_csv(path)
    if "approach" not in df.columns:
        return None

    mae_col = "mae_kg" if "mae_kg" in df.columns else ("mae" if "mae" in df.columns else None)
    if mae_col is None:
        return None

    low = df["approach"].cast(pl.Utf8).str.to_lowercase()
    kind = (
        pl.when(low.str.contains("flow") & ~low.str.contains("direct"))
        .then(pl.lit("flow"))
        .when(low.str.contains("direct"))
        .then(pl.lit("direct"))
        .otherwise(pl.lit(None))
        .alias("approach_norm")
    )
    norm = df.with_columns(kind).filter(pl.col("approach_norm").is_not_null())
    if norm.is_empty():
        return None
    return norm.group_by("approach_norm").agg(
        pl.col(mae_col).mean().alias("mae_kg")
    ).rename({"approach_norm": "approach"})


def contrast_table(external_df: pl.DataFrame, internal_df: pl.DataFrame | None) -> pl.DataFrame:
    """Side-by-side external vs internal Direct/Energy and Flow/Energy MAE.

    ``internal_df`` is expected to already carry a normalized ``approach``
    column (``"direct"`` / ``"flow"``) produced by ``load_internal_baseline``.
    """
    if internal_df is None:
        return external_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("internal_mae_kg"))

    intl = internal_df.select(["approach", "mae_kg"]).rename({"mae_kg": "internal_mae_kg"})

    return external_df.join(intl, on="approach", how="left")


def plot_external_vs_flow(
    result: dict, contrast: pl.DataFrame | None, path: Path
) -> None:
    import matplotlib.pyplot as plt

    m = result["metrics"]
    labels = ["Direct · E+W", "Flow+Energy"]
    keys = ["direct", "flow"]
    maes = [m[k]["mae"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, maes, color=["#e74c3c", "#27ae60"], alpha=0.85, edgecolor="white")
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} kg", ha="center", va="bottom", fontsize=10)

    sig = result["significance"]
    ax.set_ylabel("MAE on recovered interval fuel (kg)")
    ax.set_title(
        "External dataset: Flow+Energy vs Direct\n"
        f"ΔMAE={sig['delta_mae']:+.1f} kg | P(Flow better)={1.0 - sig['bootstrap_p']:.3f}"
    )

    if contrast is not None:
        pdf = contrast.to_pandas()
        if "internal_mae_kg" in pdf.columns and pdf["internal_mae_kg"].notna().any():
            x = np.arange(len(labels))
            internal = pdf.set_index("approach").loc[keys, "internal_mae_kg"].to_numpy()
            ax.scatter(x, internal, color="black", marker="D", zorder=5, label="Internal (AeroTwin)")
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(
    external_path: str,
    internal_path: str | None = None,
    outdir: Path | None = None,
    test_size: float = 0.2,
    iterations: int = 500,
) -> None:
    outdir = outdir or (project_root() / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXTERNAL vs FLOW — generalization check of AeroTwin methodology")
    print("=" * 72)
    print(f"External dataset: {external_path}")

    if not Path(external_path).exists():
        raise SystemExit(f"External dataset not found: {external_path}")

    ext_df = pl.read_parquet(external_path)
    required = {"actual_fuel_kg", "duration_s", "flight_id"}
    missing = required - set(ext_df.columns)
    if missing:
        raise SystemExit(f"External dataset missing required columns: {missing}")

    result = run_protocol(ext_df, test_size=test_size, iterations=iterations)
    print(f"External intervals: {result['n_intervals']:,} | test flights: {result['n_test_flights']:,}")
    for key, m in result["metrics"].items():
        print(f"  {m['label']:22s} MAE={m['mae']:7.1f} RMSE={m['rmse']:7.1f} R2={m['r2']:.4f}")
    sig = result["significance"]
    print(f"\nFlow+Energy vs Direct on external data:")
    print(f"  ΔMAE={sig['delta_mae']:+.2f} kg | 95% CI [{sig['ci_lower']:.2f}, {sig['ci_upper']:.2f}]")
    print(f"  bootstrap P(Flow+Energy better)={1.0 - sig['bootstrap_p']:.3f}")
    print(f"  → {sig['interpretation']}")

    ext_table = external_results_table(result)
    ext_path = outdir / "table_external_flow_vs_direct.csv"
    ext_table.write_csv(ext_path)
    print(f"\nSaved {ext_path}")

    internal = load_internal_baseline(internal_path) if internal_path else None
    if internal is not None:
        contrast = contrast_table(ext_table, internal)
        cpath = outdir / "table_external_vs_internal.csv"
        contrast.write_csv(cpath)
        print(f"Saved {cpath}")
        plot_external_vs_flow(result, contrast, outdir / "fig_external_vs_flow.png")
    else:
        plot_external_vs_flow(result, None, outdir / "fig_external_vs_flow.png")
    print("Saved figure fig_external_vs_flow.png")
    print("=" * 72)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = sys.argv[1:]
    external = None
    internal = None
    outdir = None
    test_size = 0.2
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--external":
            external = args[i + 1]; i += 2
        elif a == "--internal":
            internal = args[i + 1]; i += 2
        elif a == "--outdir":
            outdir = Path(args[i + 1]); i += 2
        elif a == "--test-size":
            test_size = float(args[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print("Usage: python physics/external_vs_flow_eval.py --external PATH [--internal PATH]")
            print("       [--outdir DIR] [--test-size 0.2]")
            raise SystemExit(0)
        else:
            i += 1

    if not external:
        raise SystemExit("Provide --external PATH to an independent featured-dataset parquet.")
    main(external, internal_path=internal, outdir=outdir, test_size=test_size)
