"""Error analysis of frozen AeroTwin official Rank/Final predictions.

Answers:
  1. Which aircraft types contribute most to RMSE?
  2. Which phases (climb/cruise/descent) contribute most?
  3. Is the gap concentrated in long-haul flights?
  4. Rank vs Final underperformance
  5. Systematic over/under-prediction bias

Trains ONLY on train (same frozen recipe as notebook 17). Does not tune.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import project_root
from physics.official_benchmark import (
    apply_bases,
    build_oof_matrix,
    choose_meta_on_train_folds,
    ew_feature_cols,
    featured_path,
    predict_fuel_kg,
    prepare_xy,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("error_analysis")
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

ENSEMBLE_BASES = [
    ("xgb", "direct"),
    ("lgbm", "direct"),
    ("cat", "direct"),
    ("xgb", "fuel_flow"),
    ("lgbm", "fuel_flow"),
    ("cat", "fuel_flow"),
]

# Duration buckets (interval-level duration_s, and flight-level total from fractions)
# Long-haul proxy: high max altitude + long total flight duration estimate
# total_flight_s ≈ duration_s / max(end_frac - start_frac, eps) when fractions available


def dominant_phase(row) -> str:
    """Phase label for analysis: prefer discrete phase, else max fraction."""
    ph = row.get("phase")
    if ph in ("climb", "cruise", "descent"):
        return ph
    c = float(row.get("climb_fraction") or 0)
    cr = float(row.get("cruise_fraction") or 0)
    d = float(row.get("descent_fraction") or 0)
    m = max(c, cr, d)
    if m < 1e-9:
        return "unknown"
    if m == c:
        return "climb"
    if m == d:
        return "descent"
    return "cruise"


def flight_duration_s(df: pl.DataFrame) -> pl.Series:
    """Estimate full-flight duration from interval window fractions."""
    # duration_interval / (end - start fraction) ≈ full flight seconds
    frac = (pl.col("end_fraction_of_flight") - pl.col("start_fraction_of_flight")).clip(
        lower_bound=1e-3
    )
    return (pl.col("duration_s") / frac).alias("est_flight_duration_s")


def haul_bucket(flight_hours: float) -> str:
    if flight_hours < 2:
        return "short_<2h"
    if flight_hours < 5:
        return "medium_2-5h"
    if flight_hours < 8:
        return "long_5-8h"
    return "ultralong_>=8h"


def sse_contrib(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum((p - y) ** 2))


def group_metrics(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    rows = []
    total_sse = float(((df["pred"] - df["actual_fuel_kg"]) ** 2).sum())
    total_n = len(df)
    for key, g in df.group_by(group_col):
        k = key[0] if isinstance(key, tuple) else key
        y = g["actual_fuel_kg"].to_numpy()
        p = g["pred"].to_numpy()
        n = len(g)
        err = p - y
        sse = float(np.sum(err**2))
        rows.append(
            {
                group_col: str(k),
                "n": n,
                "n_frac": n / total_n,
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias_mean_pred_minus_actual": float(np.mean(err)),
                "median_bias": float(np.median(err)),
                "sse": sse,
                "sse_share": sse / total_sse if total_sse > 0 else 0.0,
                "mean_actual_kg": float(np.mean(y)),
                "mean_pred_kg": float(np.mean(p)),
            }
        )
    return pl.DataFrame(rows).sort("sse_share", descending=True)


def main() -> None:
    train = pl.read_parquet(featured_path("train"))
    rank = pl.read_parquet(featured_path("rank"))
    final_ = pl.read_parquet(featured_path("final"))
    for name, df in ("train", train), ("rank", rank), ("final", final_):
        df = df.drop_nulls(subset=["actual_fuel_kg", "physics_fuel_kg", "flight_id"]).filter(
            pl.col("actual_fuel_kg").is_finite()
            & pl.col("physics_fuel_kg").is_finite()
            & (pl.col("duration_s") > 0)
        )
        if name == "train":
            train = df
        elif name == "rank":
            rank = df
        else:
            final_ = df

    feat_cols = ew_feature_cols(train)
    for name, df in ("rank", rank), ("final", final_):
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(None).alias(c) for c in missing])
            if name == "rank":
                rank = df
            else:
                final_ = df

    LOGGER.info("Building train OOF ensemble (frozen)...")
    P_oof, y_oof, full_models = build_oof_matrix(train, feat_cols, ENSEMBLE_BASES, n_splits=5)
    groups = train["flight_id"].to_numpy()
    meta_kind, meta = choose_meta_on_train_folds(P_oof, y_oof, groups, n_splits=5)
    LOGGER.info("Meta=%s", meta_kind)

    # Also train best single (LGBM flow) for comparison
    X_tr, _, y_kg, dur_tr = prepare_xy(train, feat_cols, "direct")
    y_flow = y_kg / dur_tr
    lgbm_flow = train_model("lgbm", X_tr, y_flow, feat_cols)

    panels = {}
    for split_name, df in ("rank", rank), ("final", final_), ("combined", None):
        if split_name == "combined":
            df = pl.concat([rank, final_], how="diagonal_relaxed")
        P = apply_bases(full_models, df, feat_cols)
        pred_ens = np.asarray(meta.predict(P), dtype=np.float64)
        X_te, _, y_te, dur_te = prepare_xy(df, feat_cols, "direct")
        pred_lgbm = predict_fuel_kg(lgbm_flow, X_te, dur_te, "fuel_flow")
        pred_phys = df["physics_fuel_kg"].to_numpy()

        # annotate
        pdf = df.with_columns(
            pl.Series("pred", pred_ens),
            pl.Series("pred_lgbm_flow", pred_lgbm),
            pl.Series("pred_physics", pred_phys),
            pl.Series("resid", pred_ens - df["actual_fuel_kg"].to_numpy()),
            pl.Series("abs_err", np.abs(pred_ens - df["actual_fuel_kg"].to_numpy())),
            pl.Series("sq_err", (pred_ens - df["actual_fuel_kg"].to_numpy()) ** 2),
        )
        # dominant phase
        phases = [dominant_phase(r) for r in pdf.iter_rows(named=True)]
        pdf = pdf.with_columns(pl.Series("phase_dom", phases))
        # flight duration estimate
        pdf = pdf.with_columns(flight_duration_s(pdf))
        pdf = pdf.with_columns(
            (pl.col("est_flight_duration_s") / 3600.0).alias("est_flight_hours")
        )
        hauls = [haul_bucket(float(h)) if h == h else "unknown" for h in pdf["est_flight_hours"].to_list()]
        pdf = pdf.with_columns(pl.Series("haul_bucket", hauls))
        # altitude band (cruise proxy)
        pdf = pdf.with_columns(
            pl.when(pl.col("mean_altitude") < 3000)
            .then(pl.lit("low_<3km"))
            .when(pl.col("mean_altitude") < 9000)
            .then(pl.lit("mid_3-9km"))
            .otherwise(pl.lit("high_>=9km"))
            .alias("alt_band")
        )
        panels[split_name] = pdf

        y = pdf["actual_fuel_kg"].to_numpy()
        LOGGER.info(
            "%s ensemble MAE=%.2f RMSE=%.2f bias=%.2f | LGBM-flow RMSE=%.2f",
            split_name,
            float(np.mean(np.abs(pred_ens - y))),
            float(np.sqrt(np.mean((pred_ens - y) ** 2))),
            float(np.mean(pred_ens - y)),
            float(np.sqrt(np.mean((pred_lgbm - y) ** 2))),
        )

    # ----- 1. Aircraft type -----
    ac_rank = group_metrics(panels["rank"], "aircraft_type")
    ac_final = group_metrics(panels["final"], "aircraft_type")
    ac_comb = group_metrics(panels["combined"], "aircraft_type")
    ac_comb.write_csv(OUT / "table_error_by_aircraft_type.csv")
    ac_rank.write_csv(OUT / "table_error_by_aircraft_type_rank.csv")
    ac_final.write_csv(OUT / "table_error_by_aircraft_type_final.csv")

    # ----- 2. Phase -----
    ph_rank = group_metrics(panels["rank"], "phase_dom")
    ph_final = group_metrics(panels["final"], "phase_dom")
    ph_comb = group_metrics(panels["combined"], "phase_dom")
    ph_comb.write_csv(OUT / "table_error_by_phase.csv")

    # ----- 3. Haul / long-haul -----
    haul_rank = group_metrics(panels["rank"], "haul_bucket")
    haul_final = group_metrics(panels["final"], "haul_bucket")
    haul_comb = group_metrics(panels["combined"], "haul_bucket")
    haul_comb.write_csv(OUT / "table_error_by_haul.csv")

    # Interval duration buckets
    for split_name, pdf in panels.items():
        panels[split_name] = pdf.with_columns(
            pl.when(pl.col("duration_s") < 600)
            .then(pl.lit("iv_<10min"))
            .when(pl.col("duration_s") < 1800)
            .then(pl.lit("iv_10-30min"))
            .otherwise(pl.lit("iv_>=30min"))
            .alias("interval_dur_bucket")
        )
    iv_comb = group_metrics(panels["combined"], "interval_dur_bucket")
    iv_comb.write_csv(OUT / "table_error_by_interval_duration.csv")

    alt_comb = group_metrics(panels["combined"], "alt_band")
    alt_comb.write_csv(OUT / "table_error_by_altitude_band.csv")

    # ----- 4. Rank vs Final -----
    split_rows = []
    for s in ("rank", "final"):
        pdf = panels[s]
        y = pdf["actual_fuel_kg"].to_numpy()
        p = pdf["pred"].to_numpy()
        err = p - y
        split_rows.append(
            {
                "split": s,
                "n": len(pdf),
                "n_flights": pdf["flight_id"].n_unique(),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "median_bias": float(np.median(err)),
                "pct_overpredict": float(np.mean(err > 0) * 100),
                "mean_actual": float(np.mean(y)),
            }
        )
    split_df = pl.DataFrame(split_rows)
    split_df.write_csv(OUT / "table_error_rank_vs_final.csv")

    # ----- 5. Bias overall + residual histogram -----
    bias_rows = []
    for s, pdf in panels.items():
        err = pdf["resid"].to_numpy()
        bias_rows.append(
            {
                "split": s,
                "mean_bias_pred_minus_actual": float(np.mean(err)),
                "median_bias": float(np.median(err)),
                "pct_overpredict": float(np.mean(err > 0) * 100),
                "pct_underpredict": float(np.mean(err < 0) * 100),
                "bias_p10": float(np.percentile(err, 10)),
                "bias_p90": float(np.percentile(err, 90)),
            }
        )
    bias_df = pl.DataFrame(bias_rows)
    bias_df.write_csv(OUT / "table_prediction_bias.csv")

    # Calibration by actual fuel decile
    comb = panels["combined"]
    try:
        comb = comb.with_columns(
            pl.col("actual_fuel_kg").qcut(10, labels=[str(i) for i in range(10)]).alias("actual_decile")
        )
    except Exception:
        # fallback rank-based
        comb = comb.with_columns(
            (pl.col("actual_fuel_kg").rank() / len(comb) * 10).floor().clip(0, 9).cast(pl.Int32).cast(pl.Utf8).alias("actual_decile")
        )
    cal = group_metrics(comb, "actual_decile").sort("actual_decile")
    cal.write_csv(OUT / "table_error_by_actual_decile.csv")

    # ----- Summary JSON for report -----
    top_ac = ac_comb.head(8).to_dicts()
    summary = {
        "meta": meta_kind,
        "rank_vs_final": split_df.to_dicts(),
        "bias": bias_df.to_dicts(),
        "top_aircraft_by_sse_share": top_ac,
        "phase_combined": ph_comb.to_dicts(),
        "haul_combined": haul_comb.to_dicts(),
        "interval_duration_combined": iv_comb.to_dicts(),
        "altitude_band_combined": alt_comb.to_dicts(),
        "answers": {
            "aircraft_types_most_sse": [
                f"{r['aircraft_type']} (SSE share {r['sse_share']*100:.1f}%, RMSE {r['rmse']:.1f}, n={r['n']})"
                for r in top_ac[:5]
            ],
            "phase_most_sse": [
                f"{r['phase_dom']}: SSE {r['sse_share']*100:.1f}%, RMSE {r['rmse']:.1f}, n={r['n']}"
                for r in ph_comb.to_dicts()
            ],
            "long_haul": haul_comb.to_dicts(),
            "rank_vs_final_worse": (
                "rank" if split_df.filter(pl.col("split") == "rank")["rmse"][0]
                > split_df.filter(pl.col("split") == "final")["rmse"][0]
                else "final"
            ),
            "systematic_bias": bias_df.filter(pl.col("split") == "combined").to_dicts()[0],
        },
    }
    (OUT / "official_error_analysis_summary.json").write_text(json.dumps(summary, indent=2))

    # ----- Figures -----
    sns.set_theme(style="whitegrid")

    # Fig 1: aircraft SSE share top 12
    fig, ax = plt.subplots(figsize=(9, 5))
    top = ac_comb.head(12).to_pandas()
    ax.barh(top["aircraft_type"][::-1], top["sse_share"][::-1] * 100, color="#2E5A88")
    ax.set_xlabel("Share of total SSE (%)")
    ax.set_title("Aircraft types contributing most to Combined SSE (Ensemble)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_by_aircraft_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: phase RMSE + SSE
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ph = ph_comb.to_pandas()
    axes[0].bar(ph["phase_dom"], ph["rmse"], color="#C45C26")
    axes[0].set_ylabel("RMSE [kg]")
    axes[0].set_title("RMSE by dominant phase")
    axes[1].bar(ph["phase_dom"], ph["sse_share"] * 100, color="#2F7D4F")
    axes[1].set_ylabel("SSE share [%]")
    axes[1].set_title("SSE contribution by phase")
    fig.suptitle("Phase error breakdown (Combined Rank+Final)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_by_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: haul
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = haul_comb.to_pandas()
    order = ["short_<2h", "medium_2-5h", "long_5-8h", "ultralong_>=8h"]
    h["ord"] = h["haul_bucket"].map({k: i for i, k in enumerate(order)})
    h = h.sort_values("ord")
    axes[0].bar(h["haul_bucket"], h["rmse"], color="#2E5A88")
    axes[0].set_ylabel("RMSE [kg]")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_title("RMSE by estimated flight length")
    axes[1].bar(h["haul_bucket"], h["sse_share"] * 100, color="#C45C26")
    axes[1].set_ylabel("SSE share [%]")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_title("SSE share by flight length")
    fig.suptitle("Long-haul concentration? (est. from interval fractions)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_by_haul.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Rank vs Final
    fig, ax = plt.subplots(figsize=(6, 4))
    s = split_df.to_pandas()
    x = np.arange(2)
    ax.bar(x - 0.2, s["rmse"], 0.4, label="RMSE", color="#2E5A88")
    ax.bar(x + 0.2, s["mae"], 0.4, label="MAE", color="#C45C26")
    ax.set_xticks(x)
    ax.set_xticklabels(s["split"])
    ax.set_ylabel("kg")
    ax.set_title("Ensemble error: Rank vs Final")
    ax.legend()
    for i, row in s.iterrows():
        ax.text(i - 0.2, row["rmse"] + 2, f"{row['rmse']:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, row["mae"] + 2, f"{row['mae']:.0f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_rank_vs_final.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Fig 5: residual histogram + bias
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s, color in ("rank", "#2E5A88"), ("final", "#C45C26"):
        axes[0].hist(
            panels[s]["resid"].to_numpy(),
            bins=80,
            alpha=0.5,
            density=True,
            label=s,
            color=color,
            range=(-1500, 1500),
        )
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_xlabel("pred − actual [kg]")
    axes[0].set_title("Residual distribution")
    axes[0].legend()
    # calibration: mean pred vs mean actual by decile
    cal_pd = cal.to_pandas()
    # need mean actual per decile from comb
    dec_stats = (
        comb.group_by("actual_decile")
        .agg(
            pl.col("actual_fuel_kg").mean().alias("mean_actual"),
            pl.col("pred").mean().alias("mean_pred"),
        )
        .sort("actual_decile")
        .to_pandas()
    )
    axes[1].plot(dec_stats["mean_actual"], dec_stats["mean_pred"], "o-", color="#2F7D4F")
    lims = [
        0,
        max(dec_stats["mean_actual"].max(), dec_stats["mean_pred"].max()) * 1.05,
    ]
    axes[1].plot(lims, lims, "k--", lw=1)
    axes[1].set_xlabel("Mean actual [kg] (decile)")
    axes[1].set_ylabel("Mean prediction [kg]")
    axes[1].set_title("Calibration by actual-fuel decile")
    fig.suptitle("Systematic over/under-prediction (Ensemble, Combined)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_prediction_bias_calibration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Aircraft type RMSE for top SSE types
    fig, ax = plt.subplots(figsize=(9, 5))
    t = ac_comb.head(12).to_pandas()
    ax.barh(t["aircraft_type"][::-1], t["rmse"][::-1], color="#6B7280")
    ax.set_xlabel("RMSE [kg]")
    ax.set_title("RMSE by aircraft type (top 12 by SSE share)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rmse_by_aircraft_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n=== ERROR ANALYSIS SUMMARY ===")
    print(json.dumps(summary["answers"], indent=2))
    print("\nRank vs Final:")
    print(split_df)
    print("\nTop aircraft by SSE share:")
    print(ac_comb.head(8))
    print("\nPhase:")
    print(ph_comb)
    print("\nHaul:")
    print(haul_comb)
    print("\nBias:")
    print(bias_df)


if __name__ == "__main__":
    # Fix invalid walrus import leftover if any
    main()
