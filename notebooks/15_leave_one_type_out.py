"""Comprehensive leave-one-type-out (LOTO) evaluation suite.

Experiments:
  - Direct vs fuel-flow vs mass-normalized flow targets
  - Global vs body-class hierarchical (transfer-aware) routing
  - Failure analysis by aircraft mass class and narrow/wide-body shift
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    CATEGORICAL,
    evaluate,
    flight_level_split,
    load_and_clean,
    project_root,
)
from physics.feature_engineering import ENERGY_FEATURES  # noqa: E402
from physics.weather_features import WEATHER_FEATURES  # noqa: E402

PARQUET = project_root() / "featured_dataset_mass.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_LOTO_FLIGHTS = 80
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]

# ICAO type → fuselage class (for transfer-aware routing)
NARROW_BODY_TYPES = {
    "A20N", "A21N", "A318", "A319", "A320", "A321", "A306",
    "B738", "B38M", "B39M", "B37M", "E190", "E195",
}
WIDE_BODY_TYPES = {
    "A332", "A333", "A359", "A388",
    "B772", "B77L", "B77W", "B788", "B789",
    "MD11",
}

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


@dataclass(frozen=True)
class TargetSpec:
    name: str
    label: str

    def transform_y(
        self,
        actual: np.ndarray,
        duration: np.ndarray,
        ref_mass: np.ndarray | None = None,
    ) -> np.ndarray:
        dur = np.clip(duration, 1.0, None)
        if self.name == "direct_fuel":
            return actual.astype(np.float64)
        if self.name == "fuel_flow":
            return (actual / dur).astype(np.float64)
        if self.name == "flow_per_mass":
            mass = np.clip(ref_mass, 1.0, None)
            return (actual / (dur * mass)).astype(np.float64)
        raise ValueError(self.name)

    def recover_fuel(
        self,
        pred: np.ndarray,
        duration: np.ndarray,
        ref_mass: np.ndarray | None = None,
    ) -> np.ndarray:
        dur = np.clip(duration, 1.0, None)
        if self.name == "direct_fuel":
            return pred.astype(np.float64)
        if self.name == "fuel_flow":
            return (pred * dur).astype(np.float64)
        if self.name == "flow_per_mass":
            mass = np.clip(ref_mass, 1.0, None)
            return (pred * dur * mass).astype(np.float64)
        raise ValueError(self.name)


@dataclass(frozen=True)
class ApproachSpec:
    key: str
    label: str
    routing: str  # global | hierarchical_body
    feature_group: str  # ew | flow_energy
    target: TargetSpec


TARGETS = {
    "direct_fuel": TargetSpec("direct_fuel", "Direct fuel (kg)"),
    "fuel_flow": TargetSpec("fuel_flow", "Fuel flow (kg/s)"),
    "flow_per_mass": TargetSpec("flow_per_mass", "Specific flow (kg/s/kg)"),
}

APPROACHES = [
    ApproachSpec(
        "global_direct_ew",
        "Global · Direct · E+W",
        "global",
        "ew",
        TARGETS["direct_fuel"],
    ),
    ApproachSpec(
        "global_flow_energy",
        "Global · Flow+Energy",
        "global",
        "flow_energy",
        TARGETS["fuel_flow"],
    ),
    ApproachSpec(
        "global_flow_per_mass_ew",
        "Global · Flow/Mass · E+W",
        "global",
        "ew",
        TARGETS["flow_per_mass"],
    ),
    ApproachSpec(
        "hier_body_direct_ew",
        "Body-class · Direct · E+W",
        "hierarchical_body",
        "ew",
        TARGETS["direct_fuel"],
    ),
    ApproachSpec(
        "hier_body_flow_energy",
        "Body-class · Flow+Energy",
        "hierarchical_body",
        "flow_energy",
        TARGETS["fuel_flow"],
    ),
]


def body_class(ac_type: str, mtow: float | None = None) -> str:
    ac = str(ac_type)
    if ac in NARROW_BODY_TYPES:
        return "narrow"
    if ac in WIDE_BODY_TYPES:
        return "wide"
    if mtow is not None and np.isfinite(mtow):
        return "wide" if mtow >= 200_000 else "narrow"
    return "other"


def mass_class(mtow: float) -> str:
    if not np.isfinite(mtow):
        return "unknown"
    if mtow < 90_000:
        return "light"
    if mtow < 200_000:
        return "medium"
    return "heavy"


def avail(cols: list[str], df: pl.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def feature_cols(df: pl.DataFrame, group: str) -> list[str]:
    energy = avail(ENERGY_FEATURES, df)
    weather = avail(WEATHER_FEATURES, df)
    cats = [c for c in CAT_FEATURES if c in df.columns]
    if group == "ew":
        cols = list(BASE_NUMERIC) + energy + weather + ["physics_fuel_kg"] + cats
    elif group == "flow_energy":
        cols = list(BASE_NUMERIC) + energy + cats
    else:
        raise ValueError(group)
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


def flight_type_lookup(df: pl.DataFrame) -> dict[str, str]:
    rows = (
        df.select(["flight_id", "aircraft_type"])
        .unique()
        .group_by("flight_id")
        .agg(pl.col("aircraft_type").first())
    )
    data = rows.to_dict(as_series=False)
    return dict(zip(data["flight_id"], data["aircraft_type"]))


def type_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Per-aircraft-type body class, mass class, and MTOW."""
    if "mtow" in df.columns:
        agg = df.group_by("aircraft_type").agg(
            pl.col("mtow").first().alias("mtow"),
            pl.len().alias("n_intervals"),
        )
    else:
        agg = df.group_by("aircraft_type").agg(pl.len().alias("n_intervals"))
        agg = agg.with_columns(pl.lit(200_000.0).alias("mtow"))

    pdf = agg.to_pandas()
    rows = []
    for _, r in pdf.iterrows():
        ac = str(r["aircraft_type"])
        mtow = float(r["mtow"])
        rows.append(
            {
                "aircraft_type": ac,
                "mtow_kg": mtow,
                "body_class": body_class(ac, mtow),
                "mass_class": mass_class(mtow),
                "n_intervals": int(r["n_intervals"]),
            }
        )
    return pl.DataFrame(rows)


def train_mask_excluding_type(
    fids: np.ndarray,
    fid_to_type: dict[str, str],
    held_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    held_fids = {f for f in np.unique(fids) if fid_to_type[f] == held_type}
    train_mask = np.array([f not in held_fids for f in fids])
    test_mask = ~train_mask
    return train_mask, test_mask


def body_train_mask(
    train_mask: np.ndarray,
    body_values: np.ndarray,
    body_label: str,
) -> np.ndarray:
    return train_mask & (body_values == body_label)


def run_loto_fold(
    approach: ApproachSpec,
    pdf,
    feat_cols: list[str],
    cat_names: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    y_actual: np.ndarray,
    duration: np.ndarray,
    ref_mass: np.ndarray,
    body_values: np.ndarray,
    held_body: str,
) -> tuple[np.ndarray, dict[str, float]]:
    X_all = pdf[feat_cols]
    y_test_actual = y_actual[test_mask]
    dur_test = duration[test_mask]
    mass_test = ref_mass[test_mask]

    y_all = approach.target.transform_y(y_actual, duration, ref_mass)

    if approach.routing == "global":
        model = train_catboost(
            X_all.iloc[train_mask],
            y_all[train_mask],
            feat_cols,
            cat_names,
        )
        raw = predict_cat(model, X_all.iloc[test_mask], feat_cols, cat_names)
    elif approach.routing == "hierarchical_body":
        if held_body not in ("narrow", "wide"):
            held_body = "narrow"
        body_mask = body_train_mask(train_mask, body_values, held_body)
        if body_mask.sum() < 100:
            body_mask = train_mask
        model = train_catboost(
            X_all.iloc[body_mask],
            y_all[body_mask],
            feat_cols,
            cat_names,
        )
        raw = predict_cat(model, X_all.iloc[test_mask], feat_cols, cat_names)
    else:
        raise ValueError(approach.routing)

    fuel_pred = approach.target.recover_fuel(raw, dur_test, mass_test)
    return fuel_pred, evaluate(y_test_actual, fuel_pred)


def summarize_macro(per_type: pl.DataFrame, approach_key: str) -> dict:
    sub = per_type.filter(pl.col("approach") == approach_key)
    if sub.is_empty():
        return {"approach": approach_key, "mae": None, "rmse": None, "r2": None, "n_types": 0}
    return {
        "approach": approach_key,
        "label": sub["approach_label"][0],
        "routing": sub["routing"][0],
        "target": sub["target"][0],
        "feature_group": sub["feature_group"][0],
        "mae": float(sub["mae"].mean()),
        "rmse": float(sub["rmse"].mean()),
        "r2": float(sub["r2"].mean()),
        "n_types": len(sub),
    }


def main() -> None:
    print("=" * 72)
    print("LOTO COMPREHENSIVE — FuelFlow+Energy, normalized targets, hierarchical")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy()
    duration = df["duration_s"].to_numpy()
    ref_mass = (
        df["ref_mass_kg"].to_numpy()
        if "ref_mass_kg" in df.columns
        else np.full(len(df), 150_000.0)
    )

    meta = type_metadata(df)
    meta_lut = {r["aircraft_type"]: r for r in meta.to_dicts()}

    feat_sets = {g: feature_cols(df, g) for g in ("ew", "flow_energy")}
    cat_names = [c for c in CAT_FEATURES if c in feat_sets["ew"]]

    pdf["body_class"] = pdf["aircraft_type"].map(
        lambda ac: meta_lut.get(ac, {}).get("body_class", body_class(ac))
    )
    body_values = pdf["body_class"].to_numpy()

    fid_to_type = flight_type_lookup(df)
    unique_fids = np.unique(fids)

    print(f"Intervals: {len(df):,} | E+W feats: {len(feat_sets['ew'])} | Flow+E: {len(feat_sets['flow_energy'])}")

    # --- Standard split reference ---
    train_idx, test_idx, _, _ = flight_level_split(fids)
    print("\n[1] Standard split reference (global direct E+W)")
    ref_approach = APPROACHES[0]
    ref_cols = feat_sets[ref_approach.feature_group]
    y_ref = ref_approach.target.transform_y(y, duration, ref_mass)
    global_ref = train_catboost(pdf[ref_cols].iloc[train_idx], y_ref[train_idx], ref_cols, cat_names)
    p_ref = predict_cat(global_ref, pdf[ref_cols].iloc[test_idx], ref_cols, cat_names)
    fuel_ref = ref_approach.target.recover_fuel(
        p_ref, duration[test_idx], ref_mass[test_idx]
    )
    m_ref = evaluate(y[test_idx], fuel_ref)
    print(f"  MAE={m_ref['mae']:.2f} RMSE={m_ref['rmse']:.2f} R2={m_ref['r2']:.4f}")

    # --- LOTO folds ---
    print(f"\n[2] LOTO folds (min {MIN_LOTO_FLIGHTS} flights per held-out type)")
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
    loto_types = flight_counts.filter(pl.col("n_flights") >= MIN_LOTO_FLIGHTS)["aircraft_type"].to_list()
    print(f"  Types: {loto_types}")

    loto_rows: list[dict] = []
    t_start = time.perf_counter()

    for i, held_type in enumerate(loto_types, 1):
        train_mask, test_mask = train_mask_excluding_type(fids, fid_to_type, held_type)
        n_held_fl = int(np.unique(fids[test_mask]).size)
        n_held_int = int(test_mask.sum())
        if n_held_int < 50:
            continue

        held_meta = meta_lut.get(held_type, {})
        held_body = held_meta.get("body_class", body_class(held_type))
        held_mass = held_meta.get("mass_class", "unknown")
        held_mtow = float(held_meta.get("mtow_kg", np.nan))

        train_bodies = body_values[train_mask]
        train_narrow_frac = float((train_bodies == "narrow").mean())
        train_wide_frac = float((train_bodies == "wide").mean())
        body_shift = (
            "narrow_held_wide_train"
            if held_body == "narrow" and train_wide_frac > 0.35
            else "wide_held_narrow_train"
            if held_body == "wide" and train_narrow_frac > 0.55
            else "matched"
        )

        physics_preds = pdf["physics_fuel_kg"].to_numpy()[test_mask]
        physics_mae = float(mean_absolute_error(y[test_mask], physics_preds))

        print(
            f"  [{i}/{len(loto_types)}] {held_type} ({held_body}/{held_mass}): "
            f"{n_held_fl} fl, {n_held_int} int | physics MAE={physics_mae:.1f}",
            flush=True,
        )

        for approach in APPROACHES:
            fcols = feat_sets[approach.feature_group]
            t0 = time.perf_counter()
            preds, mets = run_loto_fold(
                approach,
                pdf,
                fcols,
                cat_names,
                train_mask,
                test_mask,
                y,
                duration,
                ref_mass,
                body_values,
                held_body,
            )
            elapsed = time.perf_counter() - t0
            loto_rows.append(
                {
                    "held_out_type": held_type,
                    "approach": approach.key,
                    "approach_label": approach.label,
                    "routing": approach.routing,
                    "target": approach.target.name,
                    "feature_group": approach.feature_group,
                    "body_class": held_body,
                    "mass_class": held_mass,
                    "mtow_kg": held_mtow,
                    "body_shift": body_shift,
                    "train_narrow_frac": train_narrow_frac,
                    "train_wide_frac": train_wide_frac,
                    "n_held_flights": n_held_fl,
                    "n_held_intervals": n_held_int,
                    "physics_mae": physics_mae,
                    "mae": mets["mae"],
                    "rmse": mets["rmse"],
                    "r2": mets["r2"],
                    "delta_mae_vs_physics": mets["mae"] - physics_mae,
                    "train_seconds": elapsed,
                }
            )
            print(
                f"      {approach.key:28s} MAE={mets['mae']:7.1f} "
                f"RMSE={mets['rmse']:7.1f} ({elapsed:.0f}s)",
                flush=True,
            )

    loto_df = pl.DataFrame(loto_rows).sort(["held_out_type", "approach"])
    loto_path = OUT / "table_loto_comprehensive.csv"
    loto_df.write_csv(loto_path)

    # Legacy-compatible per-type table (global direct only)
    legacy = loto_df.filter(pl.col("approach") == "global_direct_ew").select(
        [
            "held_out_type",
            "n_held_flights",
            "n_held_intervals",
            "mae",
            "rmse",
            "r2",
        ]
    )
    legacy.write_csv(OUT / "table_leave_one_type_out.csv")

    # --- Macro summaries ---
    print("\n[3] Macro-averaged LOTO summaries")
    macro_rows = [summarize_macro(loto_df, a.key) for a in APPROACHES]
    macro_rows.insert(
        0,
        {
            "approach": "standard_split_global_direct_ew",
            "label": "Standard split · Global · Direct · E+W",
            "routing": "global",
            "target": "direct_fuel",
            "feature_group": "ew",
            "mae": m_ref["mae"],
            "rmse": m_ref["rmse"],
            "r2": m_ref["r2"],
            "n_types": 0,
        },
    )
    macro_df = pl.DataFrame(macro_rows)
    macro_path = OUT / "table_loto_macro_summary.csv"
    macro_df.write_csv(macro_path)

    # Master evaluation table (single consistent schema)
    master_rows = []
    for r in macro_rows:
        master_rows.append(
            {
                "experiment": "loto" if r["approach"] != "standard_split_global_direct_ew" else "standard_split",
                "approach": r.get("label", r["approach"]),
                "routing": r.get("routing", ""),
                "target": r.get("target", ""),
                "features": r.get("feature_group", ""),
                "split": "loto_macro_avg" if "loto" in r["approach"] or r["approach"].startswith("global_") or r["approach"].startswith("hier_") else "flight_80_20",
                "mae_kg": r["mae"],
                "rmse_kg": r["rmse"],
                "r2": r["r2"],
                "n_folds": r.get("n_types", 0),
            }
        )
    master_df = pl.DataFrame(master_rows).sort("mae_kg")
    master_path = OUT / "table_loto_evaluation_master.csv"
    master_df.write_csv(master_path)

    # Legacy summary (two-row)
    g = macro_df.filter(pl.col("approach") == "global_direct_ew")
    loto_g = g.row(0, named=True) if len(g) else None
    pl.DataFrame(
        [
            {
                "method": "global_standard_split",
                "mae": m_ref["mae"],
                "rmse": m_ref["rmse"],
                "r2": m_ref["r2"],
            },
            {
                "method": "loto_macro_avg",
                "mae": loto_g["mae"] if loto_g else None,
                "rmse": loto_g["rmse"] if loto_g else None,
                "r2": loto_g["r2"] if loto_g else None,
            },
        ]
    ).write_csv(OUT / "table_loto_summary.csv")

    # --- Failure analysis ---
    print("\n[4] Failure analysis by mass class and body shift")
    direct = loto_df.filter(pl.col("approach") == "global_direct_ew")
    flow_e = loto_df.filter(pl.col("approach") == "global_flow_energy")

    failure_rows = []
    for dim, col in [("mass_class", "mass_class"), ("body_class", "body_class"), ("body_shift", "body_shift")]:
        for grp in direct[col].unique().sort().to_list():
            d_sub = direct.filter(pl.col(col) == grp)
            f_sub = flow_e.filter(pl.col(col) == grp)
            if d_sub.is_empty():
                continue
            failure_rows.append(
                {
                    "stratification": dim,
                    "group": grp,
                    "n_types": len(d_sub),
                    "direct_mae": float(d_sub["mae"].mean()),
                    "flow_energy_mae": float(f_sub["mae"].mean()) if len(f_sub) else None,
                    "physics_mae": float(d_sub["physics_mae"].mean()),
                    "direct_rmse": float(d_sub["rmse"].mean()),
                    "flow_energy_rmse": float(f_sub["rmse"].mean()) if len(f_sub) else None,
                    "flow_vs_direct_delta_mae": float(f_sub["mae"].mean() - d_sub["mae"].mean()) if len(f_sub) else None,
                    "worst_type": d_sub.sort("mae", descending=True)["held_out_type"][0],
                    "worst_mae": float(d_sub["mae"].max()),
                }
            )

    failure_df = pl.DataFrame(failure_rows)
    failure_path = OUT / "table_loto_failure_analysis.csv"
    failure_df.write_csv(failure_path)

    # Flow vs direct per-type deltas
    pivot_rows = []
    for held in loto_types:
        d = loto_df.filter(
            (pl.col("held_out_type") == held) & (pl.col("approach") == "global_direct_ew")
        )
        f = loto_df.filter(
            (pl.col("held_out_type") == held) & (pl.col("approach") == "global_flow_energy")
        )
        h = loto_df.filter(
            (pl.col("held_out_type") == held) & (pl.col("approach") == "hier_body_flow_energy")
        )
        if d.is_empty():
            continue
        dr = d.row(0, named=True)
        fr = f.row(0, named=True) if len(f) else None
        hr = h.row(0, named=True) if len(h) else None
        pivot_rows.append(
            {
                "held_out_type": held,
                "body_class": dr["body_class"],
                "mass_class": dr["mass_class"],
                "body_shift": dr["body_shift"],
                "direct_mae": dr["mae"],
                "flow_energy_mae": fr["mae"] if fr else None,
                "hier_flow_energy_mae": hr["mae"] if hr else None,
                "flow_delta_mae": (fr["mae"] - dr["mae"]) if fr else None,
                "hier_delta_mae": (hr["mae"] - dr["mae"]) if hr else None,
                "physics_mae": dr["physics_mae"],
            }
        )
    pivot_df = pl.DataFrame(pivot_rows).sort("direct_mae", descending=True)
    pivot_path = OUT / "table_loto_target_comparison.csv"
    pivot_df.write_csv(pivot_path)

    # --- Figures ---
    print("\n[5] Figures")
    macro_plot = macro_df.filter(pl.col("approach") != "standard_split_global_direct_ew").to_pandas()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=macro_plot, x="label", y="mae", ax=axes[0], color="steelblue")
    axes[0].axhline(m_ref["mae"], color="crimson", ls="--", label=f"Standard split ({m_ref['mae']:.0f})")
    axes[0].set_title("LOTO macro-average MAE by approach")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=8)

    sns.barplot(data=macro_plot, x="label", y="rmse", ax=axes[1], color="seagreen")
    axes[1].axhline(m_ref["rmse"], color="crimson", ls="--", label=f"Standard split ({m_ref['rmse']:.0f})")
    axes[1].set_title("LOTO macro-average RMSE by approach")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_macro_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fail_plot = failure_df.filter(pl.col("stratification") == "body_shift").to_pandas()
    if not fail_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(fail_plot))
        w = 0.25
        ax.bar(x - w, fail_plot["direct_mae"], w, label="Direct E+W", color="#e74c3c")
        ax.bar(x, fail_plot["flow_energy_mae"], w, label="Flow+Energy", color="#27ae60")
        ax.bar(x + w, fail_plot["physics_mae"], w, label="OpenAP physics", color="#95a5a6")
        ax.set_xticks(x)
        ax.set_xticklabels(fail_plot["group"], rotation=15)
        ax.set_ylabel("MAE (kg)")
        ax.set_title("LOTO errors by train/test body-class shift")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "fig_loto_body_shift.png", bbox_inches="tight")
        plt.close(fig)

    pivot_plot = pivot_df.to_pandas()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(pivot_plot["direct_mae"], pivot_plot["flow_energy_mae"], s=80, alpha=0.85)
    lim = max(pivot_plot["direct_mae"].max(), pivot_plot["flow_energy_mae"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    for _, r in pivot_plot.iterrows():
        ax.annotate(r["held_out_type"], (r["direct_mae"], r["flow_energy_mae"]), fontsize=8, alpha=0.8)
    ax.set_xlabel("Direct E+W MAE (kg)")
    ax.set_ylabel("Flow+Energy MAE (kg)")
    ax.set_title("Per-type LOTO: flow target vs direct (below diagonal = flow wins)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_flow_vs_direct.png", bbox_inches="tight")
    plt.close(fig)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nTotal LOTO runtime: {elapsed_total / 60:.1f} min")
    print(f"Saved {loto_path}")
    print(f"Saved {macro_path}")
    print(f"Saved {master_path}")
    print(f"Saved {failure_path}")
    print(f"Saved {pivot_path}")

    print("\nLOTO macro-average MAE:")
    for r in macro_rows[1:]:
        print(f"  {r.get('label', r['approach']):35s} {r['mae']:7.1f} kg")

    flow_macro = next((r for r in macro_rows if r["approach"] == "global_flow_energy"), None)
    direct_macro = next((r for r in macro_rows if r["approach"] == "global_direct_ew"), None)
    if flow_macro and direct_macro:
        delta = flow_macro["mae"] - direct_macro["mae"]
        print(f"\nFlow+Energy vs Direct under LOTO: ΔMAE={delta:+.1f} kg")
        if delta < 0:
            print("  → Physically normalized flow target HELPS unseen-aircraft transfer.")
        else:
            print("  → Flow target does NOT improve LOTO; direct fuel remains better.")

    print("=" * 72)


if __name__ == "__main__":
    main()