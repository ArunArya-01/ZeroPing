from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    CATEGORICAL,
    evaluate,
    flight_level_split,
    project_root,
    significance_test,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES, enrich_from_columns  # noqa: E402
from aerotwin.engine.weather_features import WEATHER_FEATURES, enrich_weather_from_columns  # noqa: E402
from aerotwin.validation.external_vs_flow_eval import (  # noqa: E402
    avail,
    clean_for_eval,
    predict_cat,
    train_catboost,
)

# Categorical features mirror notebooks/15 (the source protocol) so the external
# run is directly comparable to the internal AeroTwin result.
CAT_FEATURES = CATEGORICAL + ["phase"]

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Enrichment (robust to partial external schemas)
# --------------------------------------------------------------------------- #
def enrich_energy_weather(df: pl.DataFrame) -> pl.DataFrame:
    """Add energy + weather columns when the required base columns exist.

    An independent external dataset may already carry these columns (e.g. if it
    was produced by the same featured-dataset builder) or may lack them entirely.
    Enrichment is attempted per-group and degrades gracefully: if the source
    columns needed for a group are missing, that group is simply left out and the
    downstream ``avail`` filter drops any column that did not land.
    """
    out = df
    if "mean_specific_energy_jpkg" not in out.columns:
        try:
            out = enrich_from_columns(out)
        except Exception:
            pass
    if "headwind_mps" not in out.columns:
        try:
            out = enrich_weather_from_columns(out)
        except Exception:
            pass
    return out


def base_feature_cols(df: pl.DataFrame) -> list[str]:
    """OpenAP-Hybrid-equivalent baseline feature set on external data.

    Mirrors the internal 'OpenAP Hybrid' baseline (trajectory + physics + cats).
    ``physics_fuel_kg`` is included only when present so the ablation remains
    runnable on datasets that carry no OpenAP baseline.
    """
    cols = list(avail(BASE_NUMERIC, df))
    if "physics_fuel_kg" in df.columns:
        cols.append("physics_fuel_kg")
    cols += avail(CAT_FEATURES, df)
    return list(dict.fromkeys(cols))


# --------------------------------------------------------------------------- #
# Ablation core
# --------------------------------------------------------------------------- #
def run_energy_ablation(
    df: pl.DataFrame,
    test_size: float = 0.2,
    iterations: int = 500,
):
    """Run the External Energy-feature ablation (mirrors internal V3 E6).

    Compares three feature sets, all predicting absolute interval fuel (kg) with
    CatBoost on a strict flight-level split:

      * ``base``        — trajectory + physics + categorical (OpenAP Hybrid equiv.)
      * ``energy``      — base + energy-state features
      * ``energy_weather`` — base + energy-state + weather-proxy features

    Returns a dict with per-approach metrics, recovered-fuel predictions, test
    flight ids, and flight-clustered bootstrap significance of each augmentation
    relative to the base.
    """
    df = clean_for_eval(df)
    df = enrich_energy_weather(df)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy()

    base_cols = base_feature_cols(df)
    energy_cols = list(dict.fromkeys(base_cols + avail(ENERGY_FEATURES, df)))
    ew_cols = list(dict.fromkeys(energy_cols + avail(WEATHER_FEATURES, df)))

    cat_names = [c for c in CAT_FEATURES if c in df.columns]

    train_idx, test_idx, _, _ = flight_level_split(fids, test_size=test_size)

    y_train = y[train_idx]
    y_test = y[test_idx]
    test_fids = fids[test_idx]

    approaches = {
        "base": ("OpenAP Hybrid equiv.", base_cols),
        "energy": ("Energy Hybrid", energy_cols),
        "energy_weather": ("Energy+Weather Hybrid", ew_cols),
    }

    out: dict[str, dict] = {}
    preds_fuel: dict[str, np.ndarray] = {}
    for key, (label, cols) in approaches.items():
        # Skip empty/degenerate feature sets so partial schemas still evaluate.
        usable = [c for c in cols if c in pdf.columns]
        if not usable:
            continue
        model = train_catboost(
            pdf[usable].iloc[train_idx], y_train, usable, cat_names, iterations=iterations
        )
        fuel = np.asarray(
            predict_cat(model, pdf[usable].iloc[test_idx], usable, cat_names),
            dtype=np.float64,
        )
        mets = evaluate(y_test, fuel)
        out[key] = {
            "label": label,
            "feature_group": key,
            "n_features": len(usable),
            **mets,
        }
        preds_fuel[key] = fuel

    if "base" not in preds_fuel:
        raise SystemExit("External Energy ablation: no usable baseline feature set.")

    base_err = np.abs(preds_fuel["base"] - y_test)
    sig_rows: list[dict] = []
    for key in ("energy", "energy_weather"):
        if key not in preds_fuel:
            continue
        err = np.abs(preds_fuel[key] - y_test)
        sig = significance_test(
            err, base_err, test_fids, out[key]["label"], out["base"]["label"]
        )
        # ΔMAE here is (augmented − base); negative => energy features help.
        sig_rows.append(sig)

    return {
        "n_intervals": len(df),
        "n_test_intervals": int(len(test_idx)),
        "n_test_flights": int(np.unique(test_fids).size),
        "metrics": out,
        "significance": sig_rows,
        "y_test": y_test,
        "flight_ids_test": test_fids,
    }


# --------------------------------------------------------------------------- #
# Tabular outputs
# --------------------------------------------------------------------------- #
def ablation_results_table(result: dict) -> pl.DataFrame:
    """Flatten one external ablation run into an approach-level results table."""
    rows = []
    for key, m in result["metrics"].items():
        rows.append(
            {
                "approach": key,
                "label": m["label"],
                "feature_group": m["feature_group"],
                "n_features": m["n_features"],
                "mae_kg": m["mae"],
                "rmse_kg": m["rmse"],
                "r2": m["r2"],
            }
        )
    df = pl.DataFrame(rows)
    base_mae = result["metrics"]["base"]["mae"]
    df = df.with_columns(
        (pl.col("mae_kg") - base_mae).alias("delta_mae_vs_base"),
        pl.lit(result["n_test_flights"]).alias("n_test_flights"),
        pl.lit(result["n_test_intervals"]).alias("n_test_intervals"),
    )
    return df


def ablation_significance_table(result: dict) -> pl.DataFrame:
    """Per-augmentation bootstrap significance vs the base feature set."""
    sig = result["significance"]
    if not sig:
        return pl.DataFrame()
    rows = []
    for s in sig:
        rows.append(
            {
                "comparison": s["comparison"],
                "mae_new": s["mae_new"],
                "mae_baseline": s["mae_baseline"],
                "delta_mae": s["delta_mae"],
                "ci_lower": s["ci_lower"],
                "ci_upper": s["ci_upper"],
                "bootstrap_p": s["bootstrap_p"],
                "cohens_d": s["cohens_d"],
                "effect_size": s["effect_size"],
                "interpretation": s["interpretation"],
            }
        )
    return pl.DataFrame(rows)


def load_internal_e6(path: Path) -> pl.DataFrame | None:
    """Load the internal V3 E6 significance table for contrast.

    Expected columns include ``comparison`` and ``delta_mae`` (produced by
    ``notebooks/09_physics_features_v3.py`` -> ``table_significance_v3_e6.csv``).
    Returns a normalized frame with ``approach`` and ``delta_mae`` columns, or
    ``None`` when the file is absent or not the expected shape.
    """
    if not Path(path).exists():
        return None
    df = pl.read_csv(path)
    if "comparison" not in df.columns or "delta_mae" not in df.columns:
        return None

    low = df["comparison"].cast(pl.Utf8).str.to_lowercase()
    kind = (
        # Match on separate substrings so the '+' in "Energy+Weather" (a regex
        # metacharacter) never breaks the classification.
        pl.when(low.str.contains("energy") & low.str.contains("weather"))
        .then(pl.lit("energy_weather"))
        .when(low.str.contains("energy"))
        .then(pl.lit("energy"))
        .otherwise(pl.lit(None))
        .alias("approach")
    )
    norm = df.with_columns(kind).filter(pl.col("approach").is_not_null())
    if norm.is_empty():
        return None
    return norm.group_by("approach").agg(
        pl.col("delta_mae").mean().alias("internal_delta_mae")
    )


def contrast_ablation(external_df: pl.DataFrame, internal_df: pl.DataFrame | None) -> pl.DataFrame:
    """Side-by-side external vs internal ΔMAE for energy augmentations."""
    if internal_df is None:
        return external_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("internal_delta_mae")
        )
    intl = internal_df.rename({"internal_delta_mae": "internal_delta_mae"})
    return external_df.join(intl, on="approach", how="left")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_ablation(result: dict, contrast: pl.DataFrame | None, path: Path) -> None:
    import matplotlib.pyplot as plt

    m = result["metrics"]
    order = [k for k in ("base", "energy", "energy_weather") if k in m]
    labels = [m[k]["label"] for k in order]
    maes = [m[k]["mae"] for k in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, maes, color=["#2980b9", "#27ae60", "#8e44ad"], alpha=0.85, edgecolor="white")
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} kg", ha="center", va="bottom", fontsize=10)

    base_mae = m["base"]["mae"]
    ax.axhline(base_mae, color="#c0392b", ls="--", lw=1.4, label=f"Base MAE={base_mae:.1f} kg")

    ax.set_ylabel("MAE on interval fuel (kg)")
    ax.set_title("External dataset: Energy-feature ablation (CatBoost, flight-level split)")

    if contrast is not None:
        pdf = contrast.to_pandas()
        if "internal_delta_mae" in pdf.columns and pdf["internal_delta_mae"].notna().any():
            for k in order:
                row = pdf[pdf["approach"] == k]
                if row.empty or row["internal_delta_mae"].isna().all():
                    continue
                ext_d = float(row["delta_mae_vs_base"].iloc[0])
                int_d = float(row["internal_delta_mae"].iloc[0])
                ax.annotate(
                    f"Δext={ext_d:+.1f}\nΔint={int_d:+.1f}",
                    xy=(labels.index(m[k]["label"]), m[k]["mae"]),
                    xytext=(0, -38),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#555555",
                )

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
    print("EXTERNAL ENERGY-FEATURE ABLATION — does energy-state representation")
    print("help prediction on an independent dataset?")
    print("=" * 72)
    print(f"External dataset: {external_path}")

    if not Path(external_path).exists():
        raise SystemExit(f"External dataset not found: {external_path}")

    ext_df = pl.read_parquet(external_path)
    required = {"actual_fuel_kg", "duration_s", "flight_id"}
    missing = required - set(ext_df.columns)
    if missing:
        raise SystemExit(f"External dataset missing required columns: {missing}")

    result = run_energy_ablation(ext_df, test_size=test_size, iterations=iterations)
    print(f"External intervals: {result['n_intervals']:,} | test flights: {result['n_test_flights']:,}")
    for key, m in result["metrics"].items():
        print(f"  {m['label']:22s} MAE={m['mae']:7.1f} RMSE={m['rmse']:7.1f} R2={m['r2']:.4f}")
    print("\nEnergy augmentation significance vs OpenAP-Hybrid-equiv baseline:")
    for s in result["significance"]:
        print(f"  {s['comparison']}")
        print(f"    ΔMAE={s['delta_mae']:+.2f} kg | 95% CI [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]")
        print(f"    → {s['interpretation']}")

    res_table = ablation_results_table(result)
    res_path = outdir / "table_external_energy_ablation.csv"
    res_table.write_csv(res_path)
    print(f"\nSaved {res_path}")

    sig_table = ablation_significance_table(result)
    if not sig_table.is_empty():
        sig_path = outdir / "table_external_energy_ablation_significance.csv"
        sig_table.write_csv(sig_path)
        print(f"Saved {sig_path}")

    internal = load_internal_e6(internal_path) if internal_path else None
    if internal is not None:
        contrast = contrast_ablation(res_table, internal)
        cpath = outdir / "table_external_energy_ablation_vs_internal.csv"
        contrast.write_csv(cpath)
        print(f"Saved {cpath}")
        plot_ablation(result, contrast, outdir / "fig_external_energy_ablation.png")
    else:
        plot_ablation(result, None, outdir / "fig_external_energy_ablation.png")
    print("Saved figure fig_external_energy_ablation.png")
    print("=" * 72)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = sys.argv[1:]
    external = None
    internal = None
    outdir = None
    test_size = 0.2
    iterations = 500
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
        elif a == "--iterations":
            iterations = int(args[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print("Usage: python physics/external_energy_ablation.py --external PATH [--internal PATH]")
            print("       [--outdir DIR] [--test-size 0.2] [--iterations 500]")
            raise SystemExit(0)
        else:
            i += 1

    if not external:
        raise SystemExit("Provide --external PATH to an independent featured-dataset parquet.")
    main(external, internal_path=internal, outdir=outdir, test_size=test_size, iterations=iterations)
