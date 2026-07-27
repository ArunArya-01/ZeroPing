"""
LOTO post-hoc analysis: paired significance (Direct E+W vs Flow+Energy) and
aircraft transfer-distance hypothesis testing.

Does NOT train new model families — re-runs only the two existing global LOTO
approaches to recover interval-level errors for flight-clustered inference.
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
from scipy import stats
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerotwin.engine.eval_framework import (  # noqa: E402
    BASE_NUMERIC,
    N_BOOTSTRAP,
    evaluate,
    flight_level_split,
    load_and_clean,
    project_root,
)
from aerotwin.engine.feature_engineering import ENERGY_FEATURES  # noqa: E402
from aerotwin.engine.weather_features import WEATHER_FEATURES  # noqa: E402

try:
    from openap import prop
except ImportError as e:
    raise RuntimeError("openap required") from e

PARQUET = project_root() / "featured_dataset_mass.parquet"
LOTO_CSV = project_root() / "figures" / "table_loto_comprehensive.csv"
STANDARD_PERTYPE = project_root() / "figures" / "table_aircraft_level_pertype.csv"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_LOTO_FLIGHTS = 80
N_BOOT = N_BOOTSTRAP
CAT_FEATURES = ["aircraft_type", "method", "origin_icao", "destination_icao", "phase"]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# Shared LOTO training helpers (mirrors 15_leave_one_type_out.py, two approaches)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    name: str

    def transform_y(self, actual, duration, ref_mass=None):
        dur = np.clip(duration, 1.0, None)
        if self.name == "direct_fuel":
            return actual.astype(np.float64)
        if self.name == "fuel_flow":
            return (actual / dur).astype(np.float64)
        raise ValueError(self.name)

    def recover_fuel(self, pred, duration, ref_mass=None):
        dur = np.clip(duration, 1.0, None)
        if self.name == "direct_fuel":
            return pred.astype(np.float64)
        if self.name == "fuel_flow":
            return (pred * dur).astype(np.float64)
        raise ValueError(self.name)


DIRECT = TargetSpec("direct_fuel")
FLOW = TargetSpec("fuel_flow")


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


def train_catboost(X_train, y_train, feat_cols, cat_names, iterations=500):
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


def predict_cat(model, X, feat_cols, cat_names):
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
    d = rows.to_dict(as_series=False)
    return dict(zip(d["flight_id"], d["aircraft_type"]))


def loto_types_list(df: pl.DataFrame, fid_to_type: dict[str, str], min_flights: int) -> list[str]:
    unique_fids = np.unique(df["flight_id"].to_numpy())
    counts = (
        pl.DataFrame({"flight_id": unique_fids})
        .with_columns(
            pl.col("flight_id").replace_strict(fid_to_type).alias("aircraft_type")
        )
        .group_by("aircraft_type")
        .agg(pl.len().alias("n_flights"))
        .filter(pl.col("n_flights") >= min_flights)
        .sort("n_flights", descending=True)
    )
    return counts["aircraft_type"].to_list()


def run_loto_fold_errors(
    pdf,
    feat_cols: list[str],
    cat_names: list[str],
    target: TargetSpec,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    y_actual: np.ndarray,
    duration: np.ndarray,
    flight_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return abs errors on fuel kg, y_true, flight_ids for held-out intervals."""
    y_tr = target.transform_y(y_actual, duration)
    model = train_catboost(
        pdf[feat_cols].iloc[train_mask],
        y_tr[train_mask],
        feat_cols,
        cat_names,
    )
    raw = predict_cat(model, pdf[feat_cols].iloc[test_mask], feat_cols, cat_names)
    fuel_pred = target.recover_fuel(raw, duration[test_mask])
    y_te = y_actual[test_mask]
    err = np.abs(y_te - fuel_pred)
    return err, y_te, flight_ids[test_mask]


def flight_clustered_mae_diff(err_a, err_b, flight_ids, n_iter=N_BOOT, seed=RANDOM_STATE):
    """Bootstrap MAE(Flow) - MAE(Direct); negative => Flow better."""
    _, codes = np.unique(flight_ids, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    sa, sb = err_a[order], err_b[order]
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(flight_ids)]))
    sums_a = np.add.reduceat(sa, starts)
    sums_b = np.add.reduceat(sb, starts)
    counts = ends - starts
    n_fl = len(counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_fl, size=(n_iter, n_fl))
    boot_a = sums_a[idx].sum(axis=1)
    boot_b = sums_b[idx].sum(axis=1)
    boot_n = counts[idx].sum(axis=1)
    return boot_b / boot_n - boot_a / boot_n  # flow - direct


def hierarchical_type_flight_bootstrap(
    per_type_data: dict[str, dict],
    n_iter=N_BOOT,
    seed=RANDOM_STATE,
) -> np.ndarray:
    """Resample types, then flights within type; return flow-direct MAE delta."""
    types = list(per_type_data.keys())
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_iter)
    for b in range(n_iter):
        sampled_types = rng.choice(types, size=len(types), replace=True)
        total_err_d = 0.0
        total_err_f = 0.0
        total_n = 0
        for t in sampled_types:
            d = per_type_data[t]
            fids = d["flight_ids"]
            unique_f, codes = np.unique(fids, return_inverse=True)
            n_fl = len(unique_f)
            pick = rng.integers(0, n_fl, size=n_fl)
            mask = np.isin(codes, pick)
            total_err_d += d["err_direct"][mask].sum()
            total_err_f += d["err_flow"][mask].sum()
            total_n += mask.sum()
        deltas[b] = (total_err_f - total_err_d) / max(total_n, 1)
    return deltas


def bootstrap_correlation(x, y, method="pearson", n_iter=5000, seed=RANDOM_STATE):
    """Bootstrap CI for correlation by resampling paired (x,y) observations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    cors = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if method == "pearson":
            cors[b], _ = stats.pearsonr(x[idx], y[idx])
        else:
            cors[b], _ = stats.spearmanr(x[idx], y[idx])
    return cors


def leave_one_out_correlation(x, y, method="pearson") -> list[dict]:
    """Influence: correlation when each point is excluded."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    full = stats.pearsonr(x, y)[0] if method == "pearson" else stats.spearmanr(x, y)[0]
    rows = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        if method == "pearson":
            r, _ = stats.pearsonr(x[mask], y[mask])
        else:
            r, _ = stats.spearmanr(x[mask], y[mask])
        rows.append({"dropped_type_index": i, "r_full": full, "r_loo": r, "delta_r": r - full})
    return rows


# ---------------------------------------------------------------------------
# OpenAP aircraft descriptor table
# ---------------------------------------------------------------------------

DESCRIPTOR_COLS = [
    "mtow_kg",
    "mlw_kg",
    "oew_kg",
    "mfc_kg",
    "cruise_mach",
    "cruise_range_km",
    "wing_area_m2",
    "wing_span_m",
    "max_thrust_n",
    "mmo",
]


def extract_openap_descriptors(ac_type: str) -> dict:
    """Pull descriptors from OpenAP only; NaN if field missing (no invented values)."""
    row = {"aircraft_type": ac_type}
    try:
        ac = prop.aircraft(ac_type)
    except Exception:
        return {**{c: np.nan for c in DESCRIPTOR_COLS}, "aircraft_type": ac_type, "source": "missing"}

    row["source"] = "openap"
    row["mtow_kg"] = float(ac.get("mtow") or np.nan)
    row["mlw_kg"] = float(ac.get("mlw") or np.nan)
    row["oew_kg"] = float(ac.get("oew") or np.nan)
    limits = ac.get("limits") or {}
    row["mfc_kg"] = float(limits.get("MFC") or ac.get("mfc") or np.nan)
    cruise = ac.get("cruise") or {}
    row["cruise_mach"] = float(cruise.get("mach") or np.nan)
    row["cruise_range_km"] = float(cruise.get("range") or np.nan)
    wing = ac.get("wing") or {}
    row["wing_area_m2"] = float(wing.get("area") or np.nan)
    row["wing_span_m"] = float(wing.get("span") or np.nan)
    row["mmo"] = float(ac.get("mmo") or limits.get("MMO") or np.nan)

    eng_default = (ac.get("engine") or {}).get("default")
    row["max_thrust_n"] = np.nan
    if eng_default:
        try:
            eng = prop.engine(eng_default)
            row["max_thrust_n"] = float(eng.get("max_thrust") or np.nan)
            row["engine_default"] = eng_default
        except Exception:
            row["engine_default"] = eng_default
    return row


def build_descriptor_table(types: list[str]) -> pl.DataFrame:
    rows = [extract_openap_descriptors(t) for t in types]
    return pl.DataFrame(rows)


def compute_transfer_distances(
    desc: pl.DataFrame,
    held_type: str,
    feature_cols_used: list[str],
) -> dict:
    """Distance from held-out type to training-type support (11 types)."""
    train = desc.filter(pl.col("aircraft_type") != held_type)
    held = desc.filter(pl.col("aircraft_type") == held_type)
    if held.is_empty() or train.is_empty():
        return {}

    X_train = train.select(feature_cols_used).to_numpy().astype(float)
    x_held = held.select(feature_cols_used).to_numpy().astype(float).ravel()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    x_h_s = scaler.transform(x_held.reshape(1, -1)).ravel()

    dists = cdist(x_h_s.reshape(1, -1), X_tr_s, metric="euclidean").ravel()
    k = min(3, len(dists))
    sorted_d = np.sort(dists)

    out = {
        "held_out_type": held_type,
        "n_train_types": len(train),
        "nn_distance": float(sorted_d[0]),
        f"k{k}_mean_distance": float(sorted_d[:k].mean()),
    }

    # Mahalanobis to training centroid (pseudo-inverse if singular)
    if len(train) > len(feature_cols_used):
        cov = np.cov(X_tr_s, rowvar=False)
        mu = X_tr_s.mean(axis=0)
        try:
            cov_inv = np.linalg.pinv(cov)
            md = float(mahalanobis(x_h_s, mu, cov_inv))
            cond = float(np.linalg.cond(cov)) if cov.ndim == 2 else np.nan
            out["mahalanobis_distance"] = md
            out["mahalanobis_cond"] = cond
            out["mahalanobis_ok"] = cond < 1e6
        except Exception:
            out["mahalanobis_distance"] = np.nan
            out["mahalanobis_ok"] = False
    else:
        out["mahalanobis_distance"] = np.nan
        out["mahalanobis_ok"] = False
        out["mahalanobis_note"] = "n_train <= n_features; skipped"

    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("LOTO SIGNIFICANCE + TRANSFER-DISTANCE ANALYSIS")
    print("=" * 72)

    # --- Load existing aggregate LOTO results ---
    loto_existing = pl.read_csv(LOTO_CSV)
    direct_agg = loto_existing.filter(pl.col("approach") == "global_direct_ew")
    flow_agg = loto_existing.filter(pl.col("approach") == "global_flow_energy")
    types = sorted(direct_agg["held_out_type"].unique().to_list())
    print(f"Loaded existing LOTO results for {len(types)} types")

    # --- Part 1: Re-extract interval-level errors (same models, analysis only) ---
    print("\n[1] Re-running two global LOTO approaches for interval-level errors ...")
    df = load_and_clean(PARQUET)
    pdf = df.to_pandas()
    fids = df["flight_id"].to_numpy()
    y = df["actual_fuel_kg"].to_numpy()
    duration = df["duration_s"].to_numpy()

    ew_cols = feature_cols(df, "ew")
    fe_cols = feature_cols(df, "flow_energy")
    cat_names = [c for c in CAT_FEATURES if c in ew_cols]
    fid_to_type = flight_type_lookup(df)
    loto_types = loto_types_list(df, fid_to_type, MIN_LOTO_FLIGHTS)

    per_type_data: dict[str, dict] = {}
    per_type_rows: list[dict] = []

    t0_all = time.perf_counter()
    for i, held_type in enumerate(loto_types, 1):
        held_fids = {f for f in np.unique(fids) if fid_to_type[f] == held_type}
        train_mask = np.array([f not in held_fids for f in fids])
        test_mask = ~train_mask

        print(f"  [{i}/{len(loto_types)}] {held_type} ...", flush=True)
        err_d, y_te, f_te = run_loto_fold_errors(
            pdf, ew_cols, cat_names, DIRECT, train_mask, test_mask, y, duration, fids
        )
        err_f, _, _ = run_loto_fold_errors(
            pdf, fe_cols, cat_names, FLOW, train_mask, test_mask, y, duration, fids
        )

        m_d = evaluate(y_te, y_te - err_d)  # reconstruct pred implicit
        m_f = evaluate(y_te, y_te - err_f)
        delta_mae = m_f["mae"] - m_d["mae"]

        boot = flight_clustered_mae_diff(err_f, err_d, f_te)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        try:
            w_stat, w_p = stats.wilcoxon(err_f, err_d, alternative="less")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        per_type_data[held_type] = {
            "err_direct": err_d,
            "err_flow": err_f,
            "flight_ids": f_te,
            "y_true": y_te,
        }
        per_type_rows.append(
            {
                "held_out_type": held_type,
                "n_intervals": int(test_mask.sum()),
                "n_flights": int(np.unique(f_te).size),
                "direct_mae": m_d["mae"],
                "flow_mae": m_f["mae"],
                "delta_mae_flow_minus_direct": delta_mae,
                "direct_rmse": m_d["rmse"],
                "flow_rmse": m_f["rmse"],
                "delta_rmse": m_f["rmse"] - m_d["rmse"],
                "flow_wins": delta_mae < 0,
                "per_type_boot_ci_lower": float(ci_lo),
                "per_type_boot_ci_upper": float(ci_hi),
                "per_type_boot_p_flow_better": float((boot < 0).mean()),
                "wilcoxon_p_interval": float(w_p),
                "existing_direct_mae": float(
                    direct_agg.filter(pl.col("held_out_type") == held_type)["mae"][0]
                ),
                "existing_flow_mae": float(
                    flow_agg.filter(pl.col("held_out_type") == held_type)["mae"][0]
                ),
            }
        )

    per_type_df = pl.DataFrame(per_type_rows).sort("delta_mae_flow_minus_direct")
    per_type_path = OUT / "table_loto_paired_per_type.csv"
    per_type_df.write_csv(per_type_path)

    # Pooled hierarchical bootstrap (type → flight clustering)
    hier_boot = hierarchical_type_flight_bootstrap(per_type_data)
    hier_ci_lo, hier_ci_hi = np.percentile(hier_boot, [2.5, 97.5])

    # Type-level paired analysis (12 paired MAE deltas)
    deltas = per_type_df["delta_mae_flow_minus_direct"].to_numpy()
    macro_delta = float(deltas.mean())
    median_delta = float(np.median(deltas))
    wins = int((deltas < 0).sum())
    losses = int((deltas > 0).sum())
    ties = int((deltas == 0).sum())

    try:
        t_stat, t_p = stats.ttest_rel(
            per_type_df["flow_mae"].to_numpy(),
            per_type_df["direct_mae"].to_numpy(),
        )
    except Exception:
        t_stat, t_p = float("nan"), float("nan")
    try:
        w_signed, w_signed_p = stats.wilcoxon(
            per_type_df["flow_mae"].to_numpy(),
            per_type_df["direct_mae"].to_numpy(),
            alternative="less",
        )
    except ValueError:
        w_signed, w_signed_p = float("nan"), float("nan")

    # Type-level bootstrap (resample 12 types)
    rng = np.random.default_rng(RANDOM_STATE)
    type_boot = np.empty(N_BOOT)
    n_t = len(deltas)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_t, size=n_t)
        type_boot[b] = deltas[idx].mean()
    type_ci_lo, type_ci_hi = np.percentile(type_boot, [2.5, 97.5])

    # Leave-one-type-out macro robustness
    loo_rows = []
    for held in loto_types:
        sub = per_type_df.filter(pl.col("held_out_type") != held)
        loo_rows.append(
            {
                "excluded_type": held,
                "macro_delta_mae": float(sub["delta_mae_flow_minus_direct"].mean()),
                "median_delta_mae": float(sub["delta_mae_flow_minus_direct"].median()),
                "wins": int((sub["delta_mae_flow_minus_direct"] < 0).sum()),
                "losses": int((sub["delta_mae_flow_minus_direct"] > 0).sum()),
            }
        )
    loo_df = pl.DataFrame(loo_rows)

    # B77W sensitivity
    no_b77w = per_type_df.filter(pl.col("held_out_type") != "B77W")
    sens_rows = [
        {
            "subset": "all_12_types",
            "macro_delta_mae": macro_delta,
            "median_delta_mae": median_delta,
            "wins": wins,
            "losses": losses,
            "hier_boot_ci_lower": float(hier_ci_lo),
            "hier_boot_ci_upper": float(hier_ci_hi),
            "hier_boot_p_flow_better": float((hier_boot < 0).mean()),
            "type_boot_ci_lower": float(type_ci_lo),
            "type_boot_ci_upper": float(type_ci_hi),
        },
        {
            "subset": "exclude_B77W",
            "macro_delta_mae": float(no_b77w["delta_mae_flow_minus_direct"].mean()),
            "median_delta_mae": float(no_b77w["delta_mae_flow_minus_direct"].median()),
            "wins": int((no_b77w["delta_mae_flow_minus_direct"] < 0).sum()),
            "losses": int((no_b77w["delta_mae_flow_minus_direct"] > 0).sum()),
            "hier_boot_ci_lower": np.nan,
            "hier_boot_ci_upper": np.nan,
            "hier_boot_p_flow_better": np.nan,
            "type_boot_ci_lower": float(np.percentile(
                [deltas[i] for i in range(len(deltas)) if loto_types[i] != "B77W"], 2.5
            )) if len(no_b77w) else np.nan,
            "type_boot_ci_upper": np.nan,
        },
    ]
    # Re-bootstrap excluding B77W for type-level CI
    deltas_no_b77 = no_b77w["delta_mae_flow_minus_direct"].to_numpy()
    boot_no = np.empty(N_BOOT)
    n_nb = len(deltas_no_b77)
    for b in range(N_BOOT):
        boot_no[b] = deltas_no_b77[rng.integers(0, n_nb, size=n_nb)].mean()
    sens_rows[1]["type_boot_ci_lower"] = float(np.percentile(boot_no, 2.5))
    sens_rows[1]["type_boot_ci_upper"] = float(np.percentile(boot_no, 97.5))

    sig_summary = {
        "comparison": "Global Flow+Energy vs Global Direct E+W",
        "n_types": len(loto_types),
        "macro_delta_mae_flow_minus_direct": macro_delta,
        "median_delta_mae": median_delta,
        "wins_flow_better": wins,
        "losses_flow_worse": losses,
        "ties": ties,
        "paired_ttest_p": float(t_p),
        "paired_wilcoxon_p": float(w_signed_p),
        "hierarchical_flight_boot_ci_lower": float(hier_ci_lo),
        "hierarchical_flight_boot_ci_upper": float(hier_ci_hi),
        "hierarchical_flight_boot_p_flow_better": float((hier_boot < 0).mean()),
        "type_level_boot_ci_lower": float(type_ci_lo),
        "type_level_boot_ci_upper": float(type_ci_hi),
        "exclude_B77W_macro_delta": float(deltas_no_b77.mean()),
        "exclude_B77W_median_delta": float(np.median(deltas_no_b77)),
        "runtime_min": (time.perf_counter() - t0_all) / 60,
    }
    pl.DataFrame([sig_summary]).write_csv(OUT / "table_loto_paired_significance_summary.csv")
    loo_df.write_csv(OUT / "table_loto_leave_one_type_robustness.csv")
    pl.DataFrame(sens_rows).write_csv(OUT / "table_loto_paired_sensitivity.csv")

    print(f"\n  Paired significance (12 types):")
    print(f"    Macro ΔMAE (flow-direct): {macro_delta:+.2f} kg")
    print(f"    Median ΔMAE: {median_delta:+.2f} kg | Wins: {wins} Losses: {losses}")
    print(f"    Hierarchical flight-bootstrap 95% CI: [{hier_ci_lo:+.1f}, {hier_ci_hi:+.1f}]")
    print(f"    Type-level bootstrap 95% CI: [{type_ci_lo:+.1f}, {type_ci_hi:+.1f}]")
    print(f"    Exclude B77W macro ΔMAE: {deltas_no_b77.mean():+.2f} kg")

    # --- Part 2: Transfer distance study ---
    print("\n[2] Aircraft transfer-distance study (OpenAP descriptors)")
    desc = build_descriptor_table(loto_types)
    desc_path = OUT / "table_aircraft_openap_descriptors.csv"
    desc.write_csv(desc_path)

    # Use descriptors with complete data across all 12 types
    complete_cols = []
    for c in DESCRIPTOR_COLS:
        col = desc[c].to_numpy()
        if np.isfinite(col).all():
            complete_cols.append(c)
    print(f"  Complete descriptors ({len(complete_cols)}): {complete_cols}")

    dist_rows = []
    for held in loto_types:
        d = compute_transfer_distances(desc, held, complete_cols)
        dist_rows.append(d)
    dist_df = pl.DataFrame(dist_rows)
    dist_path = OUT / "table_loto_transfer_distances.csv"
    dist_df.write_csv(dist_path)

    # Merge with LOTO errors + standard-split per-type baseline
    std_pertype = pl.read_csv(STANDARD_PERTYPE)
    analysis = (
        direct_agg.select(
            [
                "held_out_type",
                "mae",
                "rmse",
                "body_class",
                "mass_class",
                "mtow_kg",
            ]
        )
        .rename({"mae": "loto_direct_mae", "rmse": "loto_direct_rmse"})
        .join(
            flow_agg.select(["held_out_type", pl.col("mae").alias("loto_flow_mae"), pl.col("rmse").alias("loto_flow_rmse")]),
            on="held_out_type",
        )
        .join(dist_df, on="held_out_type", how="left")
        .join(
            std_pertype.select(["aircraft_type", pl.col("mae").alias("standard_split_mae"), pl.col("rmse").alias("standard_split_rmse")]),
            left_on="held_out_type",
            right_on="aircraft_type",
            how="left",
        )
        .with_columns(
            (pl.col("loto_direct_mae") / pl.col("standard_split_mae")).alias("mae_inflation_direct"),
            (pl.col("loto_flow_mae") / pl.col("standard_split_mae")).alias("mae_inflation_flow"),
            (pl.col("loto_direct_mae") - pl.col("standard_split_mae")).alias("mae_degradation_direct"),
            (pl.col("loto_flow_mae") - pl.col("standard_split_mae")).alias("mae_degradation_flow"),
        )
    )
    analysis_path = OUT / "table_loto_transfer_distance_analysis.csv"
    analysis.write_csv(analysis_path)

    # Correlation tests
    dist_metrics = ["nn_distance", "k3_mean_distance", "mahalanobis_distance"]
    outcome_cols = [
        "loto_direct_mae",
        "loto_flow_mae",
        "loto_direct_rmse",
        "mae_inflation_direct",
        "mae_degradation_direct",
    ]

    corr_rows = []
    adf = analysis.to_pandas()
    for dmet in dist_metrics:
        if dmet not in adf.columns or adf[dmet].isna().all():
            continue
        x = adf[dmet].to_numpy()
        valid = np.isfinite(x)
        for outcome in outcome_cols:
            y = adf[outcome].to_numpy()
            v = valid & np.isfinite(y)
            if v.sum() < 5:
                continue
            xv, yv = x[v], y[v]
            pr, pp = stats.pearsonr(xv, yv)
            sr, sp = stats.spearmanr(xv, yv)
            boot_p = bootstrap_correlation(xv, yv, "pearson")
            boot_s = bootstrap_correlation(xv, yv, "spearman")
            corr_rows.append(
                {
                    "distance_metric": dmet,
                    "outcome": outcome,
                    "n": int(v.sum()),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "pearson_boot_ci_lower": float(np.percentile(boot_p, 2.5)),
                    "pearson_boot_ci_upper": float(np.percentile(boot_p, 97.5)),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                    "spearman_boot_ci_lower": float(np.percentile(boot_s, 2.5)),
                    "spearman_boot_ci_upper": float(np.percentile(boot_s, 97.5)),
                }
            )

    corr_df = pl.DataFrame(corr_rows)
    corr_path = OUT / "table_loto_transfer_correlations.csv"
    corr_df.write_csv(corr_path)

    # Influence diagnostics (leave-one-type-out correlation)
    infl_rows = []
    for dmet in dist_metrics:
        if dmet not in adf.columns:
            continue
        x = adf[dmet].to_numpy()
        for outcome in ["loto_direct_mae", "mae_inflation_direct"]:
            y = adf[outcome].to_numpy()
            v = np.isfinite(x) & np.isfinite(y)
            if v.sum() < 5:
                continue
            types_v = adf["held_out_type"].to_numpy()[v]
            for method in ("pearson", "spearman"):
                loo = leave_one_out_correlation(x[v], y[v], method)
                for j, row in enumerate(loo):
                    infl_rows.append(
                        {
                            "distance_metric": dmet,
                            "outcome": outcome,
                            "method": method,
                            "dropped_type": types_v[j],
                            "r_loo": row["r_loo"],
                            "delta_r": row["delta_r"],
                        }
                    )
    infl_df = pl.DataFrame(infl_rows)
    infl_path = OUT / "table_loto_transfer_influence.csv"
    infl_df.write_csv(infl_path)

    # Sensitivity: exclude B77W correlations
    sens_corr_rows = []
    adf_no = adf[adf["held_out_type"] != "B77W"]
    for dmet in dist_metrics:
        if dmet not in adf_no.columns:
            continue
        x = adf_no[dmet].to_numpy()
        for outcome in ["loto_direct_mae", "mae_inflation_direct"]:
            y = adf_no[outcome].to_numpy()
            v = np.isfinite(x) & np.isfinite(y)
            if v.sum() < 4:
                continue
            pr, pp = stats.pearsonr(x[v], y[v])
            sr, sp = stats.spearmanr(x[v], y[v])
            sens_corr_rows.append(
                {
                    "subset": "exclude_B77W",
                    "distance_metric": dmet,
                    "outcome": outcome,
                    "n": int(v.sum()),
                    "pearson_r": float(pr),
                    "pearson_p": float(pp),
                    "spearman_r": float(sr),
                    "spearman_p": float(sp),
                }
            )
    pl.DataFrame(sens_corr_rows).write_csv(OUT / "table_loto_transfer_correlations_sensitivity.csv")

    # --- Figures ---
    print("\n[3] Figures")

    # Paired per-type delta
    fig, ax = plt.subplots(figsize=(10, 6))
    pt = per_type_df.to_pandas()
    colors = ["#27ae60" if w else "#e74c3c" for w in pt["flow_wins"]]
    ax.barh(pt["held_out_type"], pt["delta_mae_flow_minus_direct"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(macro_delta, color="navy", ls="--", label=f"Macro avg ({macro_delta:+.1f})")
    ax.set_xlabel("ΔMAE (Flow − Direct) [kg]  [negative = Flow better]")
    ax.set_title("Per-type LOTO paired comparison: Flow+Energy vs Direct E+W")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_paired_delta_per_type.png", bbox_inches="tight")
    plt.close(fig)

    # Bootstrap distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(hier_boot, bins=50, color="steelblue", alpha=0.85, density=True)
    axes[0].axvline(0, color="black", lw=1.5)
    axes[0].axvspan(hier_ci_lo, hier_ci_hi, alpha=0.2, color="green")
    axes[0].set_title("Hierarchical bootstrap ΔMAE (type→flight)")
    axes[0].set_xlabel("Flow − Direct MAE [kg]")
    axes[1].hist(type_boot, bins=50, color="darkorange", alpha=0.85, density=True)
    axes[1].axvline(0, color="black", lw=1.5)
    axes[1].axvspan(type_ci_lo, type_ci_hi, alpha=0.2, color="green")
    axes[1].set_title("Type-level bootstrap ΔMAE (12 paired folds)")
    axes[1].set_xlabel("Flow − Direct MAE [kg]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_paired_bootstrap.png", bbox_inches="tight")
    plt.close(fig)

    # Transfer distance vs LOTO error
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dmet, title in zip(
        axes,
        ["nn_distance", "mahalanobis_distance"],
        ["Nearest-neighbor distance", "Mahalanobis distance"],
    ):
        sub = adf[np.isfinite(adf[dmet])]
        ax.scatter(sub[dmet], sub["loto_direct_mae"], s=90, alpha=0.85, c="#3498db")
        for _, r in sub.iterrows():
            ax.annotate(r["held_out_type"], (r[dmet], r["loto_direct_mae"]), fontsize=8)
        ax.set_xlabel(f"{title} (standardized features)")
        ax.set_ylabel("LOTO Direct E+W MAE (kg)")
        ax.set_title(f"{title} vs LOTO MAE")
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_distance_vs_mae.png", bbox_inches="tight")
    plt.close(fig)

    # Transfer distance vs error inflation
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = adf[np.isfinite(adf["nn_distance"])]
    ax.scatter(sub["nn_distance"], sub["mae_inflation_direct"], s=90, alpha=0.85, c="#9b59b6")
    for _, r in sub.iterrows():
        ax.annotate(r["held_out_type"], (r["nn_distance"], r["mae_inflation_direct"]), fontsize=8)
    ax.set_xlabel("Nearest-neighbor transfer distance")
    ax.set_ylabel("MAE inflation (LOTO / standard-split)")
    ax.set_title("Transfer distance vs error inflation")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_distance_vs_inflation.png", bbox_inches="tight")
    plt.close(fig)

    # LOO robustness
    fig, ax = plt.subplots(figsize=(10, 5))
    lp = loo_df.to_pandas()
    ax.bar(lp["excluded_type"], lp["macro_delta_mae"], color="teal", alpha=0.85)
    ax.axhline(macro_delta, color="crimson", ls="--", label=f"Full macro ({macro_delta:+.1f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Macro ΔMAE (flow − direct)")
    ax.set_title("Leave-one-type-out robustness of macro improvement")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_loto_loo_robustness.png", bbox_inches="tight")
    plt.close(fig)

    # --- Conclusions markdown ---
    # Determine hypothesis verdict
    hyp_lines = []
    key_corr = corr_df.filter(
        (pl.col("distance_metric") == "nn_distance") & (pl.col("outcome") == "loto_direct_mae")
    )
    if len(key_corr):
        kc = key_corr.row(0, named=True)
        hyp_lines.append(
            f"Nearest-neighbor distance vs LOTO direct MAE: "
            f"Pearson r={kc['pearson_r']:.3f} (95% CI [{kc['pearson_boot_ci_lower']:.3f}, "
            f"{kc['pearson_boot_ci_upper']:.3f}], p={kc['pearson_p']:.4f}); "
            f"Spearman ρ={kc['spearman_r']:.3f} (p={kc['spearman_p']:.4f})."
        )
        if kc["pearson_p"] < 0.05 and kc["pearson_r"] > 0:
            hyp_verdict = "SUPPORTED (positive association at α=0.05)"
        elif kc["pearson_p"] >= 0.05:
            hyp_verdict = "NOT SUPPORTED (association not significant at α=0.05)"
        else:
            hyp_verdict = "NOT SUPPORTED (negative or non-monotonic association)"
    else:
        hyp_verdict = "INCONCLUSIVE (insufficient data)"

    conclusions = f"""# LOTO Significance & Transfer-Distance Analysis

**Script:** `experiments/06_loto_generalization/17_loto_significance_and_transfer_distance.py`  
**Date:** July 2026  
**Prerequisite:** Existing LOTO results in `table_loto_comprehensive.csv` (no new model families).

## 1. Paired significance: Global Flow+Energy vs Global Direct E+W

### Methodology
- Re-ran the **same two CatBoost LOTO configurations** to recover interval-level absolute errors.
- **Per-type inference:** flight-clustered bootstrap (10,000 resamples of flights within each held-out type).
- **Pooled inference:** hierarchical bootstrap resampling types then flights (respects clustering).
- **Macro inference:** bootstrap over 12 type-level paired ΔMAE values.
- **Robustness:** leave-one-type-out macro ΔMAE; sensitivity excluding B77W.

### Results
| Metric | Value |
|---|---|
| Macro ΔMAE (flow − direct) | {macro_delta:+.2f} kg |
| Median ΔMAE | {median_delta:+.2f} kg |
| Flow wins / losses / ties | {wins} / {losses} / {ties} |
| Hierarchical flight-bootstrap 95% CI | [{hier_ci_lo:+.1f}, {hier_ci_hi:+.1f}] kg |
| P(flow better) hierarchical | {(hier_boot < 0).mean():.3f} |
| Type-level bootstrap 95% CI | [{type_ci_lo:+.1f}, {type_ci_hi:+.1f}] kg |
| Paired t-test p | {t_p:.4f} |
| Paired Wilcoxon p (flow < direct) | {w_signed_p:.4f} |
| Macro ΔMAE excluding B77W | {deltas_no_b77.mean():+.2f} kg |

**Interpretation:** Flow+Energy improves macro LOTO MAE by {abs(macro_delta):.1f} kg. Hierarchical flight-clustered CI {'excludes zero (flow significantly better)' if hier_ci_hi < 0 else 'includes zero (not significant at 95%)'}. B77W exclusion {'changes' if abs(deltas_no_b77.mean() - macro_delta) > 5 else 'does not materially change'} the macro conclusion.

## 2. Aircraft transfer-distance study

### Descriptor table (`table_aircraft_openap_descriptors.csv`)
Built from **OpenAP `prop.aircraft()` and `prop.engine()` only** — no invented values.  
Features used (complete for all 12 types): {', '.join(complete_cols)}.

### Distance definitions (held-out → 11 training types)
1. **Nearest-neighbor:** min Euclidean distance on standardized descriptors.
2. **k-NN mean (k=3):** mean of 3 smallest NN distances.
3. **Mahalanobis:** distance to training-type centroid using pseudo-inverse covariance (n_train=11 > n_features={len(complete_cols)}).

### Hypothesis
> Cross-aircraft fuel prediction error increases with physical distance from the training aircraft support.

### Correlation summary
{chr(10).join('- ' + line for line in hyp_lines) if hyp_lines else '- See table_loto_transfer_correlations.csv'}

**Verdict:** {hyp_verdict}

Negative results are preserved if correlations are weak or non-significant. See `table_loto_transfer_correlations_sensitivity.csv` for B77W-excluded analysis.

## 3. Artifacts

| File | Description |
|---|---|
| `table_loto_paired_per_type.csv` | Per-type paired deltas + flight-bootstrap CIs |
| `table_loto_paired_significance_summary.csv` | Pooled significance summary |
| `table_loto_leave_one_type_robustness.csv` | LOO macro robustness |
| `table_loto_paired_sensitivity.csv` | B77W sensitivity |
| `table_aircraft_openap_descriptors.csv` | OpenAP physical descriptor table |
| `table_loto_transfer_distances.csv` | NN, k-NN, Mahalanobis distances per fold |
| `table_loto_transfer_distance_analysis.csv` | Merged distances + errors + inflation |
| `table_loto_transfer_correlations.csv` | Pearson/Spearman + bootstrap CIs |
| `table_loto_transfer_influence.csv` | Leave-one-type correlation influence |
| `fig_loto_paired_*.png`, `fig_loto_distance_*.png` | Diagnostic plots |
"""
    concl_path = OUT / "loto_significance_transfer_conclusions.md"
    concl_path.write_text(conclusions, encoding="utf-8")

    print(f"\nSaved conclusions to {concl_path}")
    print(f"Transfer hypothesis verdict: {hyp_verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()