from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import polars as pl

try:
    from openap import FuelFlow, aero, prop
except ImportError as e:  # pragma: no cover
    raise RuntimeError("openap is required for physics baseline. pip install -r requirements.txt") from e

# For phase detection
import numpy as np  # available via requirements


DEFAULT_REF_MASS_FRAC = 0.75  # crude cruise mass assumption; tune or replace with better estimator
MIN_INTERVAL_SECONDS = 60  # below this, predictions are noisy/unreliable per data filters


def _infer_tas(
    row: dict[str, Any] | pl.Series,
    alt: float,
) -> float | None:
    mach = row.get("mach") if isinstance(row, dict) else row["mach"]
    cas = row.get("CAS") if isinstance(row, dict) else row["CAS"]
    gs = row.get("groundspeed") if isinstance(row, dict) else row["groundspeed"]

    if mach is not None and not (isinstance(mach, float) and (mach != mach)):  # not nan
        try:
            return float(aero.mach2tas(float(mach), float(alt)))
        except Exception:
            pass
    if cas is not None and not (isinstance(cas, float) and (cas != cas)):
        try:
            return float(aero.cas2tas(float(cas), float(alt)))
        except Exception:
            pass
    if gs is not None and not (isinstance(gs, float) and (gs != gs)):
        # last resort; note: this ignores wind + compressibility at high mach/alt
        return float(gs)
    return None


def _ref_mass(ac_type: str) -> float:
    try:
        ac = prop.aircraft(ac_type)
        mtow = ac.get("mtow") or ac.get("MTOW") or 200_000.0
        return float(mtow) * DEFAULT_REF_MASS_FRAC
    except Exception:
        # fallback for unknown/synonym types
        return 180_000.0 * DEFAULT_REF_MASS_FRAC


def classify_interval_phase(traj_win: pl.DataFrame) -> str:
    if traj_win.is_empty() or "vertical_rate" not in traj_win.columns:
        return "unknown"
    med_vr = traj_win["vertical_rate"].median()
    if med_vr is None or med_vr != med_vr:  # nan check
        return "unknown"
    if med_vr > 1.5:
        return "climb"
    elif med_vr < -1.5:
        return "descent"
    else:
        return "cruise"


def predict_fuel_intervals(
    traj: pl.DataFrame,
    fuel: pl.DataFrame,
    ac_type: str | None = None,
) -> pl.DataFrame:
    if traj.is_empty() or fuel.is_empty():
        return pl.DataFrame()

    if ac_type is None:
        ac_type = traj["typecode"].drop_nulls().first() if "typecode" in traj.columns else "A320"
    ac_type = str(ac_type)

    try:
        ff = FuelFlow(ac=ac_type)
    except Exception:
        ff = FuelFlow(ac="A320")  # safe synonym fallback

    mass = _ref_mass(ac_type)

    # ensure sorted
    traj = traj.sort("timestamp")
    fuel = fuel.sort("start")

    results: list[dict[str, Any]] = []

    tr_time = traj["timestamp"]
    has_source = "source" in traj.columns

    for row in fuel.iter_rows(named=True):
        s = row["start"]
        e = row["end"]
        actual = float(row["fuel_kg"]) if row.get("fuel_kg") is not None else None
        idx = row.get("idx")

        # window subset (inclusive)
        mask = (tr_time >= s) & (tr_time <= e)
        win = traj.filter(mask)
        n_pts = len(win)

        if n_pts == 0:
            # no coverage at all; cannot predict meaningfully
            results.append(
                {
                    "interval_idx": idx,
                    "start": s,
                    "end": e,
                    "actual_fuel_kg": actual,
                    "physics_fuel_kg": None,
                    "tas_used": None,
                    "alt_used": None,
                    "vs_used": None,
                    "n_traj_pts": 0,
                    "has_acars_in_window": False,
                    "method": "no_coverage",
                }
            )
            continue

        # representative point: prefer ACARS row if present in window, else median row
        if has_source:
            ac_win = win.filter(pl.col("source") == "acars")
            rep = ac_win.row(0, named=True) if len(ac_win) > 0 else win.row(len(win) // 2, named=True)
            has_acars = len(ac_win) > 0
        else:
            rep = win.row(len(win) // 2, named=True)
            has_acars = False

        alt = float(rep.get("altitude") or 10000.0)
        vs = float(rep.get("vertical_rate") or 0.0)

        tas = _infer_tas(rep, alt)
        if tas is None or tas <= 0:
            # extreme fallback
            tas = 200.0
            method = "fallback_tas_200"
        else:
            method = "tas_from_mach" if (rep.get("mach") and not (isinstance(rep.get("mach"), float) and rep["mach"] != rep["mach"])) else \
                     "tas_from_cas" if (rep.get("CAS") and not (isinstance(rep.get("CAS"), float) and rep["CAS"] != rep["CAS"])) else \
                     "tas_from_gs"

        # duration factor (simple): use enroute ff * hours. For climb/descent vs may matter but enroute handles vs.
        dur_s = max(0.0, (e - s).total_seconds())
        if dur_s < MIN_INTERVAL_SECONDS:
            method += "_short"

        try:
            # ff returns kg/s (confirmed via openap + manual cross-check on B789 cruise)
            ff_rate = float(ff.enroute(mass=mass, tas=tas, alt=alt, vs=vs))
            physics_kg = ff_rate * dur_s if dur_s > 0 else 0.0
        except Exception:
            physics_kg = None
            method += "_ff_failed"

        # Phase for this window (using the same win subset)
        phase = classify_interval_phase(win)

        results.append(
            {
                "interval_idx": idx,
                "start": s,
                "end": e,
                "actual_fuel_kg": actual,
                "physics_fuel_kg": physics_kg,
                "tas_used": tas,
                "alt_used": alt,
                "vs_used": vs,
                "n_traj_pts": n_pts,
                "has_acars_in_window": has_acars,
                "phase": phase,
                "method": method,
            }
        )

    return pl.DataFrame(results)


def compute_physics_errors(
    preds: pl.DataFrame,
    fuel: pl.DataFrame | None = None,
) -> dict[str, Any]:
    if "actual_fuel_kg" not in preds.columns or "physics_fuel_kg" not in preds.columns:
        return {"error": "missing columns"}

    df = preds.filter(
        (pl.col("actual_fuel_kg").is_not_null()) & (pl.col("physics_fuel_kg").is_not_null())
    )
    if df.is_empty():
        return {"n_valid": 0}

    err = (df["physics_fuel_kg"] - df["actual_fuel_kg"]).abs()
    se = (df["physics_fuel_kg"] - df["actual_fuel_kg"]) ** 2

    overall = {
        "n_valid": len(df),
        "mae_kg": float(err.mean()),
        "rmse_kg": float(se.mean() ** 0.5),
        "median_abs_err_kg": float(err.median()),
        "mean_actual_kg": float(df["actual_fuel_kg"].mean()),
    }

    # simple sparsity bins
    bins = []
    for label, lo, hi in [("very_sparse", 0, 5), ("sparse", 5, 50), ("medium", 50, 500), ("dense", 500, 10**9)]:
        sub = df.filter((pl.col("n_traj_pts") >= lo) & (pl.col("n_traj_pts") < hi))
        if len(sub) > 0:
            e = (sub["physics_fuel_kg"] - sub["actual_fuel_kg"]).abs()
            bins.append({
                "bin": label,
                "n": len(sub),
                "mae": float(e.mean()),
                "rmse": float(((sub["physics_fuel_kg"] - sub["actual_fuel_kg"])**2).mean()**0.5),
            })

    by_acars = {}
    for has in [True, False]:
        sub = df.filter(pl.col("has_acars_in_window") == has)
        if len(sub) > 0:
            e = (sub["physics_fuel_kg"] - sub["actual_fuel_kg"]).abs()
            by_acars[f"has_acars_{has}"] = {"n": len(sub), "mae": float(e.mean())}

    return {"overall": overall, "by_n_pts_bin": bins, "by_acars": by_acars}


if __name__ == "__main__":
    # Demo on two flights we probed during audit (known to exist + have labels)
    import logging
    from data import AeroDataLoader

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    loader = AeroDataLoader()
    # these two from live probes
    demo_fids = ["prc770822360", "prc770831136"]

    fuel_all = loader.get_fuel_labels()
    for fid in demo_fids:
        print(f"\n=== Physics baseline demo: {fid} ===")
        try:
            traj = loader.load_flight_by_id(fid)
            ac = traj["typecode"][0] if "typecode" in traj.columns else "B789"
            fu = fuel_all.filter(pl.col("flight_id") == fid).sort("start")
            preds = predict_fuel_intervals(traj, fu, ac_type=ac)
            print(preds.select(["interval_idx", "actual_fuel_kg", "physics_fuel_kg", "n_traj_pts", "has_acars_in_window", "method"]))
            errs = compute_physics_errors(preds)
            print("Errors:", errs.get("overall"))
        except Exception as exc:
            print(f"  demo failed for {fid}: {exc}")
