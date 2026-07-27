"""Matched CatBoost Residual LOTO — fixes model-family confound vs Direct.

Runs residual learning under the *same* LOTO folds and CatBoost hyperparameters
as experiments/06_loto_generalization/15_leave_one_type_out.py (iterations=500, lr=0.05, depth=7).

Protocol (classic residual, matches experiments/03_baselines/05_baseline_modeling.py):
  - Target: residual_kg = actual_fuel_kg - physics_fuel_kg
  - Features: Energy+Weather hybrid *without* physics_fuel_kg as an input
  - Recover: pred_fuel_kg = physics_fuel_kg + pred_residual

Joins against existing Global · Direct · E+W rows in table_loto_comprehensive.csv
for paired Direct-vs-Residual comparison (same held-out types).

Does *not* retrain Flow / hierarchical approaches.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    evaluate,
    flight_level_split,
    load_and_clean,
    project_root,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES  # noqa: E402
from aerotwin.engine.weather_features import WEATHER_FEATURES  # noqa: E402

PARQUET = project_root() / "featured_dataset_mass.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_LOTO_FLIGHTS = 80
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]


def avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_cols_direct_ew(df: pl.DataFrame) -> list[str]:
    """Match LOTO Global · Direct · E+W feature set (includes physics)."""
    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    return list(dict.fromkeys(c for c in cols if c in df.columns))


def feature_cols_residual_ew(df: pl.DataFrame) -> list[str]:
    """Classic residual: same E+W signal as Direct, but no physics_fuel_kg input."""
    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    cats = [c for c in CAT_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather + cats
    return list(dict.fromkeys(c for c in cols if c in df.columns))


def train_catboost(
    X_train,
    y_train,
    feat_cols: list[str],
    cat_names: list[str],
    iterations: int = 500,
) -> CatBoostRegressor:
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


def predict_cat(model, X, feat_cols: list[str], cat_names: list[str]) -> np.ndarray:
    cat_idx = [i for i, c in enumerate(feat_cols) if c in cat_names]
    pool = Pool(X, cat_features=cat_idx, feature_names=feat_cols)
    return np.asarray(model.predict(pool), dtype=np.float64)


def body_class(ac_type: str, mtow: float | None = None) -> str:
    ac = str(ac_type)
    if ac in {
        "A20N", "A21N", "A318", "A319", "A320", "A321", "A306",
        "B738", "B38M", "B39M", "B37M", "E190", "E195",
    }:
        return "narrow"
    if ac in {
        "A332", "A333", "A359", "A388",
        "B772", "B77L", "B77W", "B788", "B789", "MD11",
    }:
        return "wide"
    if mtow is not None and np.isfinite(mtow):
        return "wide" if mtow >= 200_000 else "narrow"
    return "other"


def flight_type_lookup(df: pl.DataFrame) -> dict[str, str]:
    rows = (
        df.select(["flight_id", "aircraft_type"])
        .unique()
        .group_by("flight_id")
        .agg(pl.col("aircraft_type").first())
    )
    data = rows.to_dict(as_series=False)
    return dict(zip(data["flight_id"], data["aircraft_type"]))


def train_mask_excluding_type(
    fids: np.ndarray,
    fid_to_type: dict[str, str],
    held_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    held_fids = {f for f in np.unique(fids) if fid_to_type[f] == held_type}
    train_mask = np.array([f not in held_fids for f in fids])
    test_mask = ~train_mask
    return train_mask, test_mask


def main() -> None:
    print("=" * 72)
    print("LOTO RESIDUAL MATCHED — CatBoost same as notebook 15 Direct")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy()
    physics = df["physics_fuel_kg"].to_numpy()
    residual = df["residual_kg"].to_numpy()

    direct_cols = feature_cols_direct_ew(df)
    residual_cols = feature_cols_residual_ew(df)
    cat_names = [c for c in CAT_FEATURES if c in residual_cols]

    print(f"Intervals: {len(df):,}")
    print(f"Direct E+W features:    {len(direct_cols)} (incl. physics)")
    print(f"Residual E+W features:  {len(residual_cols)} (excl. physics)")

    # --- Level-1 matched reference (flight split) ---
    train_idx, test_idx, _, _ = flight_level_split(fids)
    print("\n[1] Standard split · matched CatBoost Direct vs Residual")

    t0 = time.perf_counter()
    m_d = train_catboost(
        pdf[direct_cols].iloc[train_idx], y[train_idx], direct_cols, cat_names
    )
    p_d = predict_cat(m_d, pdf[direct_cols].iloc[test_idx], direct_cols, cat_names)
    met_direct = evaluate(y[test_idx], p_d)
    print(
        f"  Direct  MAE={met_direct['mae']:.2f} RMSE={met_direct['rmse']:.2f} "
        f"R2={met_direct['r2']:.4f} ({time.perf_counter() - t0:.0f}s)"
    )

    t0 = time.perf_counter()
    m_r = train_catboost(
        pdf[residual_cols].iloc[train_idx],
        residual[train_idx],
        residual_cols,
        cat_names,
    )
    p_res = predict_cat(m_r, pdf[residual_cols].iloc[test_idx], residual_cols, cat_names)
    p_r_fuel = physics[test_idx] + p_res
    met_residual = evaluate(y[test_idx], p_r_fuel)
    print(
        f"  Residual MAE={met_residual['mae']:.2f} RMSE={met_residual['rmse']:.2f} "
        f"R2={met_residual['r2']:.4f} ({time.perf_counter() - t0:.0f}s)"
    )

    level1 = pl.DataFrame(
        [
            {
                "split": "flight_80_20",
                "approach": "global_direct_ew",
                "label": "Standard split · Global · Direct · E+W · CatBoost",
                "mae": met_direct["mae"],
                "rmse": met_direct["rmse"],
                "r2": met_direct["r2"],
                "model": "CatBoost",
                "matched_protocol": True,
            },
            {
                "split": "flight_80_20",
                "approach": "global_residual_ew",
                "label": "Standard split · Global · Residual · E+W · CatBoost",
                "mae": met_residual["mae"],
                "rmse": met_residual["rmse"],
                "r2": met_residual["r2"],
                "model": "CatBoost",
                "matched_protocol": True,
            },
        ]
    )
    level1.write_csv(OUT / "table_loto_residual_level1_matched.csv")

    # --- LOTO residual folds ---
    fid_to_type = flight_type_lookup(df)
    unique_fids = np.unique(fids)
    flight_counts = (
        pl.DataFrame({"flight_id": unique_fids})
        .join(
            pl.DataFrame(
                [{"flight_id": f, "aircraft_type": fid_to_type[f]} for f in unique_fids]
            ),
            on="flight_id",
            how="left",
        )
        .group_by("aircraft_type")
        .agg(pl.len().alias("n_flights"))
        .sort("n_flights", descending=True)
    )
    loto_types = flight_counts.filter(pl.col("n_flights") >= MIN_LOTO_FLIGHTS)[
        "aircraft_type"
    ].to_list()
    print(f"\n[2] LOTO residual folds ({len(loto_types)} types): {loto_types}")

    rows: list[dict] = []
    for i, held_type in enumerate(loto_types, 1):
        train_mask, test_mask = train_mask_excluding_type(fids, fid_to_type, held_type)
        n_held_fl = int(np.unique(fids[test_mask]).size)
        n_held_int = int(test_mask.sum())
        if n_held_int < 50:
            continue

        held_body = body_class(held_type)
        physics_mae = float(
            mean_absolute_error(y[test_mask], physics[test_mask])
        )

        print(
            f"  [{i}/{len(loto_types)}] {held_type} ({held_body}): "
            f"{n_held_fl} fl, {n_held_int} int | physics MAE={physics_mae:.1f}",
            flush=True,
        )

        t0 = time.perf_counter()
        model = train_catboost(
            pdf[residual_cols].iloc[train_mask],
            residual[train_mask],
            residual_cols,
            cat_names,
        )
        raw_res = predict_cat(
            model, pdf[residual_cols].iloc[test_mask], residual_cols, cat_names
        )
        fuel_pred = physics[test_mask] + raw_res
        mets = evaluate(y[test_mask], fuel_pred)
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "held_out_type": held_type,
                "approach": "global_residual_ew",
                "approach_label": "Global · Residual · E+W · CatBoost",
                "routing": "global",
                "target": "residual_kg",
                "feature_group": "ew_no_physics",
                "body_class": held_body,
                "n_held_flights": n_held_fl,
                "n_held_intervals": n_held_int,
                "physics_mae": physics_mae,
                "mae": mets["mae"],
                "rmse": mets["rmse"],
                "r2": mets["r2"],
                "delta_mae_vs_physics": mets["mae"] - physics_mae,
                "train_seconds": elapsed,
                "model": "CatBoost",
                "matched_protocol": True,
            }
        )
        print(
            f"      residual MAE={mets['mae']:7.1f} RMSE={mets['rmse']:7.1f} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    residual_df = pl.DataFrame(rows).sort("held_out_type")
    residual_path = OUT / "table_loto_residual_matched.csv"
    residual_df.write_csv(residual_path)
    print(f"\nWrote {residual_path}")

    # --- Join with existing Direct (and Flow if present) ---
    comprehensive_path = OUT / "table_loto_comprehensive.csv"
    if not comprehensive_path.exists():
        print("WARNING: table_loto_comprehensive.csv missing; skipping paired join")
        return

    comp = pl.read_csv(comprehensive_path)
    direct = comp.filter(pl.col("approach") == "global_direct_ew").select(
        [
            "held_out_type",
            pl.col("mae").alias("direct_mae"),
            pl.col("rmse").alias("direct_rmse"),
            pl.col("r2").alias("direct_r2"),
            pl.col("n_held_flights"),
            pl.col("n_held_intervals"),
            pl.col("body_class"),
            pl.col("physics_mae"),
        ]
    )
    flow = None
    if "global_flow_energy" in comp["approach"].to_list():
        flow = comp.filter(pl.col("approach") == "global_flow_energy").select(
            [
                "held_out_type",
                pl.col("mae").alias("flow_mae"),
                pl.col("rmse").alias("flow_rmse"),
            ]
        )

    paired = residual_df.select(
        [
            "held_out_type",
            pl.col("mae").alias("residual_mae"),
            pl.col("rmse").alias("residual_rmse"),
            pl.col("r2").alias("residual_r2"),
            "body_class",
            "n_held_flights",
            "n_held_intervals",
            "physics_mae",
        ]
    ).join(direct.drop(["body_class", "n_held_flights", "n_held_intervals", "physics_mae"]), on="held_out_type", how="left")

    if flow is not None:
        paired = paired.join(flow, on="held_out_type", how="left")

    paired = paired.with_columns(
        (pl.col("residual_mae") - pl.col("direct_mae")).alias("delta_mae_res_minus_dir"),
        (pl.col("residual_rmse") - pl.col("direct_rmse")).alias(
            "delta_rmse_res_minus_dir"
        ),
        (pl.col("residual_mae") < pl.col("direct_mae")).alias("residual_wins_mae"),
        (pl.col("residual_rmse") < pl.col("direct_rmse")).alias("residual_wins_rmse"),
    )
    if "flow_mae" in paired.columns:
        paired = paired.with_columns(
            (pl.col("residual_mae") < pl.col("flow_mae")).alias("residual_beats_flow_mae"),
            (pl.col("residual_rmse") < pl.col("flow_rmse")).alias(
                "residual_beats_flow_rmse"
            ),
        )

    paired_path = OUT / "table_loto_paired_direct_residual_matched.csv"
    paired.write_csv(paired_path)
    print(f"Wrote {paired_path}")

    # Macro summary
    macro = {
        "approach": "global_residual_ew",
        "label": "Global · Residual · E+W · CatBoost (matched)",
        "routing": "global",
        "target": "residual_kg",
        "feature_group": "ew_no_physics",
        "mae": float(residual_df["mae"].mean()),
        "rmse": float(residual_df["rmse"].mean()),
        "r2": float(residual_df["r2"].mean()),
        "n_types": len(residual_df),
        "model": "CatBoost",
        "matched_protocol": True,
    }
    direct_macro = {
        "approach": "global_direct_ew",
        "label": "Global · Direct · E+W · CatBoost (from comprehensive)",
        "mae": float(paired["direct_mae"].mean()),
        "rmse": float(paired["direct_rmse"].mean()),
        "n_types": len(paired),
    }
    level1_macro_rows = [
        {
            "regime": "flight_80_20",
            "approach": "direct",
            "mae": met_direct["mae"],
            "rmse": met_direct["rmse"],
            "model": "CatBoost",
        },
        {
            "regime": "flight_80_20",
            "approach": "residual",
            "mae": met_residual["mae"],
            "rmse": met_residual["rmse"],
            "model": "CatBoost",
        },
        {
            "regime": "loto_macro",
            "approach": "direct",
            "mae": direct_macro["mae"],
            "rmse": direct_macro["rmse"],
            "model": "CatBoost",
        },
        {
            "regime": "loto_macro",
            "approach": "residual",
            "mae": macro["mae"],
            "rmse": macro["rmse"],
            "model": "CatBoost",
        },
    ]
    if flow is not None:
        level1_macro_rows.append(
            {
                "regime": "loto_macro",
                "approach": "flow",
                "mae": float(paired["flow_mae"].mean()),
                "rmse": float(paired["flow_rmse"].mean()),
                "model": "CatBoost",
            }
        )

    summary = pl.DataFrame(level1_macro_rows)
    summary_path = OUT / "table_loto_residual_matched_summary.csv"
    summary.write_csv(summary_path)
    print(f"Wrote {summary_path}")

    n_res_mae = int(paired["residual_wins_mae"].sum())
    n_res_rmse = int(paired["residual_wins_rmse"].sum())
    n = len(paired)
    n_flip = int(
        (
            (paired["residual_wins_mae"] != paired["residual_wins_rmse"])
        ).sum()
    )

    print("\n" + "=" * 72)
    print("MATCHED DIRECT vs RESIDUAL RESULTS")
    print("=" * 72)
    print(
        f"Level-1 flight:  Direct MAE={met_direct['mae']:.2f} RMSE={met_direct['rmse']:.2f}"
    )
    print(
        f"                 Residual MAE={met_residual['mae']:.2f} RMSE={met_residual['rmse']:.2f}"
    )
    print(
        f"LOTO macro:      Direct MAE={direct_macro['mae']:.2f} RMSE={direct_macro['rmse']:.2f}"
    )
    print(
        f"                 Residual MAE={macro['mae']:.2f} RMSE={macro['rmse']:.2f}"
    )
    print(
        f"Δ (Res − Dir) LOTO macro: "
        f"MAE={macro['mae'] - direct_macro['mae']:+.2f}  "
        f"RMSE={macro['rmse'] - direct_macro['rmse']:+.2f}"
    )
    print(f"Residual wins MAE:  {n_res_mae}/{n}")
    print(f"Residual wins RMSE: {n_res_rmse}/{n}")
    print(f"Per-type MAE↔RMSE ranking flips: {n_flip}/{n}")

    # Append residual to evaluation master if present
    master_path = OUT / "table_loto_evaluation_master.csv"
    if master_path.exists():
        master = pl.read_csv(master_path)
        # drop previous residual row if re-run
        master = master.filter(~pl.col("approach").str.contains("(?i)residual"))
        new_row = pl.DataFrame(
            [
                {
                    "experiment": "loto",
                    "approach": "Global · Residual · E+W · CatBoost (matched)",
                    "routing": "global",
                    "target": "residual_kg",
                    "features": "ew_no_physics",
                    "split": "loto_macro_avg",
                    "mae_kg": macro["mae"],
                    "rmse_kg": macro["rmse"],
                    "r2": macro["r2"],
                    "n_folds": macro["n_types"],
                }
            ]
        )
        master = pl.concat([master, new_row], how="diagonal_relaxed").sort("mae_kg")
        master.write_csv(master_path)
        print(f"Updated {master_path}")

    # Macro summary file update
    macro_path = OUT / "table_loto_macro_summary.csv"
    if macro_path.exists():
        mdf = pl.read_csv(macro_path)
        mdf = mdf.filter(pl.col("approach") != "global_residual_ew")
        mdf = pl.concat(
            [
                mdf,
                pl.DataFrame(
                    [
                        {
                            "approach": macro["approach"],
                            "label": macro["label"],
                            "routing": macro["routing"],
                            "target": macro["target"],
                            "feature_group": macro["feature_group"],
                            "mae": macro["mae"],
                            "rmse": macro["rmse"],
                            "r2": macro["r2"],
                            "n_types": macro["n_types"],
                        }
                    ]
                ),
            ],
            how="diagonal_relaxed",
        )
        mdf.write_csv(macro_path)
        print(f"Updated {macro_path}")

    # Figure: paired ΔMAE
    pdf_p = paired.sort("delta_mae_res_minus_dir").to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#c0392b" if d > 0 else "#27ae60" for d in pdf_p["delta_mae_res_minus_dir"]]
    axes[0].barh(pdf_p["held_out_type"], pdf_p["delta_mae_res_minus_dir"], color=colors)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("ΔMAE (Residual − Direct) kg")
    axes[0].set_title("Matched CatBoost LOTO: Residual vs Direct MAE")
    axes[1].scatter(pdf_p["direct_mae"], pdf_p["residual_mae"], s=60)
    lim = max(pdf_p["direct_mae"].max(), pdf_p["residual_mae"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    for _, r in pdf_p.iterrows():
        axes[1].annotate(r["held_out_type"], (r["direct_mae"], r["residual_mae"]), fontsize=8)
    axes[1].set_xlabel("Direct MAE")
    axes[1].set_ylabel("Residual MAE")
    axes[1].set_title("Per-type MAE (above diagonal = residual worse)")
    fig.tight_layout()
    fig_path = OUT / "fig_loto_residual_vs_direct_matched.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    # Decision gate text for coarser entity experiment
    residual_wins_macro = macro["mae"] < direct_macro["mae"] or macro["rmse"] < direct_macro["rmse"]
    residual_competitive = (
        abs(macro["mae"] - direct_macro["mae"]) < 10
        or abs(macro["rmse"] - direct_macro["rmse"]) < 20
    )
    print("\n[GATE for coarser entity holdout]")
    if residual_wins_macro:
        print("  Residual beats Direct on macro MAE or RMSE under matched setup.")
        print("  → Coarser entity holdout is HIGH VALUE — recommend running.")
    elif residual_competitive:
        print("  Residual is close to Direct under matched setup.")
        print("  → Coarser entity holdout is OPTIONAL / medium value.")
    else:
        print("  Residual still clearly loses under matched CatBoost LOTO.")
        print("  → Domain-dependent residual failure is honest; coarser holdout")
        print("    less urgent for residual ranking, still useful for Phenomenon A.")


if __name__ == "__main__":
    main()
