"""Load NASA DASHlink Sample Flight Data (Project 85) MATLAB ``.mat`` files.

Project 85 stores per-flight recordings with ~186 parameters (regional jet
fleet, tails ~652–687). Fuel targets are reconstructed from fuel-flow or
fuel-used parameters when present; the reconstruction path is logged because
integrated flow labels are noisier than PRC ACARS FOB differences.

Output trajectory schema is aligned with AeroTwin / ``predict_fuel_intervals``:
timestamp, altitude, groundspeed, vertical_rate, latitude, longitude, typecode.

References
----------
* AeroTwin External Dataset Audit Package (July 2026)
* DASHlink Project 85 – Sample Flight Data
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from physics.external_audit.audit_utils import (
    DEFAULT_INTERVAL_S,
    MIN_INTERVAL_S,
    construct_fixed_intervals,
    construct_intervals_from_events,
    ensure_standard_traj_columns,
    integrate_rate,
)

LOGGER = logging.getLogger(__name__)

# Candidate parameter name patterns (case-insensitive substring / regex).
# Project 85 naming is not fully standardized publicly; we probe common aliases.
# Project 85 uses short FDR names (ALT, GS, IVV, FF_1, FQTY_1, …) as well as
# longer aliases. Patterns are matched against channel names (case-insensitive).
FUEL_FLOW_PATTERNS: list[str] = [
    r"^ff_\d+$",  # Project 85: FF_1 … FF_4 (LBS/HR)
    r"^ff\d+$",
    r"fuel.?flow",
    r"wff",
    r"fuel_flow",
    r"fuelflow",
    r"ffl",
    r"eng.*fuel",
    r"fuel.*rate",
]
FUEL_QTY_PATTERNS: list[str] = [
    r"^fqty",  # Project 85: FQTY_1 … FQTY_4 (LBS)
    r"fuel.?qty",
    r"fuel.?quantity",
    r"fuel.?used",
    r"total.?fuel",
    r"fob",
    r"fuel.?on.?board",
    r"fuel.?remaining",
    r"fuel.?mass",
    r"fuel.?weight",
]
TIME_PATTERNS: list[str] = [
    r"^time$",
    r"timestamp",
    r"utc",
    r"gmt",
    r"rel.?time",
    r"elapsed",
    r"^t$",
    r"^acmt$",  # ACMS timing
]
ALT_PATTERNS: list[str] = [
    r"^alt$",  # Project 85 pressure altitude (FEET)
    r"^bal[12]$",  # baro-corrected alt
    r"altitude",
    r"pressure.?alt",
    r"baro.?alt",
    r"alt_ft",
    r"alt_m",
    r"hp$",
]
GS_PATTERNS: list[str] = [
    r"^gs$",  # Project 85 ground speed (KNOTS)
    r"ground.?speed",
    r"gnd.?spd",
    r"groundspeed",
]
VR_PATTERNS: list[str] = [
    r"^ivv$",  # Project 85 inertial vertical speed (FT/MIN)
    r"^altr$",  # altitude rate (FT/MIN)
    r"vertical.?speed",
    r"vertical.?rate",
    r"^vs$",
    r"^vr$",
    r"rocd",
    r"baro.?rate",
]
LAT_PATTERNS: list[str] = [
    r"^latp$",  # Project 85 latitude position
    r"^lat$",
    r"latitude",
]
LON_PATTERNS: list[str] = [
    r"^lonp$",  # Project 85 longitude position
    r"^lon$",
    r"^lng$",
    r"longitude",
]
CAS_PATTERNS: list[str] = [
    r"^cas$",  # Project 85 computed airspeed (KNOTS)
    r"calibrated.?air",
    r"computed.?airspeed",
    r"airspeed",
]
MACH_PATTERNS: list[str] = [r"^mach$", r"^m$"]
MASS_PATTERNS: list[str] = [r"gross.?weight", r"gw$", r"mass", r"weight"]
TAS_PATTERNS: list[str] = [r"^tas$", r"true.?airspeed"]

# Unit conversion helpers
FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444
FPM_TO_MPS = 0.00508
LB_TO_KG = 0.45359237


def _match_key(keys: Iterable[str], patterns: list[str]) -> str | None:
    """Return first key matching any regex pattern (case-insensitive)."""
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for k in keys:
        for cre in compiled:
            if cre.search(str(k)):
                return str(k)
    return None


def _match_all_keys(keys: Iterable[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    out: list[str] = []
    for k in keys:
        for cre in compiled:
            if cre.search(str(k)):
                out.append(str(k))
                break
    return out


def _as_1d(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim > 1:
        # take first column of multi-channel parameters
        a = a.reshape(a.shape[0], -1)[:, 0]
    return a.astype(np.float64, copy=False)


def _unwrap_mat_value(obj: Any) -> Any:
    """Unwrap scipy ``(1,1) object`` cells and nested mat containers."""
    # (1,1) object array holding a mat_struct / ndarray
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        try:
            return _unwrap_mat_value(obj.reshape(-1)[0])
        except Exception:
            return obj
    if isinstance(obj, np.ndarray) and obj.shape == (1, 1) and obj.dtype.names:
        try:
            return obj[0, 0]
        except Exception:
            return obj
    return obj


def _scalarish(val: Any) -> Any:
    """Reduce 0-d / length-1 arrays to a Python scalar when possible."""
    if val is None:
        return None
    if isinstance(val, (str, bytes, int, float, bool, np.generic)):
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        if isinstance(val, np.generic):
            return val.item()
        return val
    try:
        a = np.asarray(val)
        if a.size == 0:
            return None
        if a.size == 1:
            item = a.reshape(-1)[0]
            if isinstance(item, bytes):
                return item.decode("utf-8", errors="replace")
            if isinstance(item, np.generic):
                return item.item()
            return item
    except Exception:
        pass
    return val


def _mat_struct_fields(obj: Any) -> list[str] | None:
    """Return field names for a scipy mat_struct or structured ndarray."""
    if hasattr(obj, "_fieldnames") and obj._fieldnames:
        return list(obj._fieldnames)
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return list(obj.dtype.names)
    return None


def _get_field(obj: Any, name: str) -> Any:
    """Get a struct field by name (mat_struct attribute or structured array)."""
    if hasattr(obj, name):
        return getattr(obj, name)
    # case-insensitive fallback (Rate vs rate)
    fields = _mat_struct_fields(obj) or []
    lower = {f.lower(): f for f in fields}
    key = lower.get(name.lower())
    if key is None:
        raise AttributeError(name)
    if hasattr(obj, key):
        return getattr(obj, key)
    if hasattr(obj, "dtype") and obj.dtype.names:
        val = obj[key]
        if isinstance(val, np.ndarray) and val.size == 1:
            return val.reshape(-1)[0]
        return val
    raise AttributeError(name)


def _extract_param_bundle(obj: Any) -> dict[str, Any] | None:
    """Extract ``data`` / ``Rate`` / ``Units`` from a Project 85 parameter struct.

    DASHlink Project 85 stores each FDR parameter as a MATLAB struct with fields
    approximately::

        .data          – time-series samples (N×1 or N,)
        .Rate          – sample rate in Hz
        .Units         – unit string (e.g. FEET, KNOTS, FT/MIN, LBS/HR)
        .Description   – human-readable label
        .Alpha         – short name

    Returns None if ``obj`` is not a parameter struct with usable ``data``.
    """
    obj = _unwrap_mat_value(obj)
    fields = _mat_struct_fields(obj)
    if not fields:
        return None

    field_l = {f.lower(): f for f in fields}
    if "data" not in field_l:
        return None

    try:
        data_raw = _get_field(obj, field_l["data"])
        data = _as_1d(data_raw)
    except Exception:
        return None

    if data.size <= 1:
        return None

    rate = None
    units = None
    description = None
    alpha = None
    if "rate" in field_l:
        rate = _scalarish(_get_field(obj, field_l["rate"]))
        try:
            rate = float(rate) if rate is not None else None
        except (TypeError, ValueError):
            rate = None
    if "units" in field_l:
        units = _scalarish(_get_field(obj, field_l["units"]))
        if units is not None:
            units = str(units)
    if "description" in field_l:
        description = _scalarish(_get_field(obj, field_l["description"]))
        if description is not None:
            description = str(description)
    if "alpha" in field_l:
        alpha = _scalarish(_get_field(obj, field_l["alpha"]))
        if alpha is not None:
            alpha = str(alpha)

    return {
        "data": data.astype(np.float64, copy=False),
        "rate": rate,
        "units": units,
        "description": description,
        "alpha": alpha,
    }


def _flatten_mat_struct(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten MATLAB structs / nested dicts into dotted keys.

    Project 85 parameter structs (with a ``data`` field) are **not** fully
    expanded here — prefer :func:`_extract_param_bundle` via :func:`load_mat_file`.
    This helper remains for nested containers and non-parameter payloads.
    """
    out: dict[str, Any] = {}
    obj = _unwrap_mat_value(obj)

    # Project 85 parameter struct: keep as leaf so load_mat_file can extract .data
    bundle = _extract_param_bundle(obj) if prefix else None
    if bundle is not None and prefix:
        out[prefix] = obj
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_mat_struct(v, key))
        return out

    # scipy.io.loadmat structured arrays
    if hasattr(obj, "dtype") and obj.dtype.names:
        for name in obj.dtype.names:
            key = f"{prefix}.{name}" if prefix else name
            try:
                val = obj[name]
                if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
                    val = val.item()
                out.update(_flatten_mat_struct(val, key))
            except Exception:
                continue
        return out

    # mat_struct without treated-as-parameter (or top-level handled elsewhere)
    if hasattr(obj, "_fieldnames") and obj._fieldnames:
        for name in obj._fieldnames:
            key = f"{prefix}.{name}" if prefix else name
            try:
                out.update(_flatten_mat_struct(getattr(obj, name), key))
            except Exception:
                continue
        return out

    if prefix:
        out[prefix] = obj
    return out


def _resample_series(
    y: np.ndarray,
    rate_src: float | None,
    n_dst: int,
    rate_dst: float | None,
) -> np.ndarray:
    """Resample ``y`` sampled at ``rate_src`` onto ``n_dst`` samples at ``rate_dst``.

    Uses linear interpolation in time. If rates are missing, falls back to
    truncating / padding to ``n_dst``.
    """
    y = _as_1d(y)
    if n_dst <= 0:
        return y[:0]
    if len(y) == n_dst:
        return y.astype(np.float64, copy=False)

    if rate_src and rate_src > 0 and rate_dst and rate_dst > 0 and len(y) > 1:
        t_src = np.arange(len(y), dtype=np.float64) / float(rate_src)
        t_dst = np.arange(n_dst, dtype=np.float64) / float(rate_dst)
        t_end = t_src[-1]
        t_dst = np.clip(t_dst, t_src[0], t_end)
        return np.interp(t_dst, t_src, y.astype(np.float64))

    if len(y) >= n_dst:
        return y[:n_dst].astype(np.float64, copy=False)
    out = np.full(n_dst, np.nan, dtype=np.float64)
    out[: len(y)] = y
    if len(y) > 0:
        out[len(y) :] = y[-1]
    return out


def load_mat_file(path: Path | str) -> dict[str, Any]:
    """Load a DASHlink Project 85 ``.mat`` file into a channel map.

    Returns
    -------
    dict
        * ``<PARAM>`` → 1-D ``float64`` numpy array (the ``.data`` time series)
        * ``meta_<PARAM>_rate`` → sample rate in Hz (if present)
        * ``meta_<PARAM>_units`` → unit string (if present)
        * ``meta_<PARAM>_description`` → description string (if present)

    Project 85 parameters are MATLAB structs with fields ``data``, ``Rate``,
    ``Units``, ``Description``, ``Alpha``. Older loaders that only flattened
    structs treated each parameter as a scalar object and produced only
    ``meta_*`` keys with ``size=1``.
    """
    try:
        from scipy.io import loadmat
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "scipy is required to load DASHlink .mat files. pip install scipy"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    LOGGER.info("Loading DASHlink MAT: %s", path)
    # squeeze_me=True yields mat_struct objects directly; struct_as_record=False
    # exposes fields as attributes (.data, .Rate, …).
    try:
        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except TypeError:
        raw = loadmat(str(path), squeeze_me=True)

    channels: dict[str, Any] = {}
    n_series = 0
    for k, v in raw.items():
        if str(k).startswith("__"):
            continue
        name = str(k)

        # Primary path: Project 85 parameter struct with .data
        bundle = _extract_param_bundle(v)
        if bundle is not None:
            channels[name] = bundle["data"]
            n_series += 1
            if bundle["rate"] is not None:
                channels[f"meta_{name}_rate"] = bundle["rate"]
            if bundle["units"] is not None:
                channels[f"meta_{name}_units"] = bundle["units"]
            if bundle["description"] is not None:
                channels[f"meta_{name}_description"] = bundle["description"]
            if bundle.get("alpha") is not None:
                channels[f"meta_{name}_alpha"] = bundle["alpha"]
            continue

        # Nested dict / struct containers
        unwrapped = _unwrap_mat_value(v)
        if isinstance(unwrapped, dict) or _mat_struct_fields(unwrapped):
            flat = _flatten_mat_struct({name: unwrapped})
            for fk, fv in flat.items():
                b2 = _extract_param_bundle(fv)
                if b2 is not None:
                    channels[fk] = b2["data"]
                    n_series += 1
                    if b2["rate"] is not None:
                        channels[f"meta_{fk}_rate"] = b2["rate"]
                    if b2["units"] is not None:
                        channels[f"meta_{fk}_units"] = b2["units"]
                    continue
                try:
                    a = _as_1d(fv)
                    if a.size > 1 and np.issubdtype(a.dtype, np.number):
                        channels[fk] = a
                        n_series += 1
                    elif a.size <= 1:
                        channels[f"meta_{fk}"] = _scalarish(fv)
                except Exception:
                    continue
            continue

        # Plain numeric array already
        try:
            a = np.asarray(unwrapped)
            if a.dtype != object and a.size > 1 and np.issubdtype(a.dtype, np.number):
                channels[name] = _as_1d(a)
                n_series += 1
            else:
                channels[f"meta_{name}"] = _scalarish(unwrapped)
        except Exception:
            channels[f"meta_{name}"] = unwrapped

    series_keys = [k for k in channels if not str(k).startswith("meta_")]
    LOGGER.info(
        "MAT %s: %d time-series channels (e.g. %s)",
        path.name,
        n_series,
        series_keys[:12],
    )
    if n_series == 0:
        LOGGER.error(
            "No time-series channels extracted from %s. "
            "Expected Project 85 structs with a .data field.",
            path.name,
        )
    return channels


def _channel_units(channels: dict[str, Any], name: str) -> str:
    u = channels.get(f"meta_{name}_units")
    return str(u).strip() if u is not None else ""


def _channel_rate(channels: dict[str, Any], name: str, default: float = 1.0) -> float:
    r = channels.get(f"meta_{name}_rate")
    try:
        return float(r) if r is not None and float(r) > 0 else default
    except (TypeError, ValueError):
        return default


def _detect_units_and_scale(
    name: str,
    series: np.ndarray,
    kind: str,
    units: str | None = None,
) -> np.ndarray:
    """Heuristic unit conversion to SI (m, m/s, kg/s).

    Prefers explicit ``units`` from the MAT struct (Project 85: FEET, KNOTS,
    FT/MIN, LBS/HR, LBS). Falls back to name / magnitude heuristics.
    """
    s = series.astype(np.float64)
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return s
    med = float(np.median(np.abs(finite)))
    lname = name.lower()
    u = (units or "").lower()

    if kind == "altitude":
        if "feet" in u or u == "ft" or "ft" in lname or "feet" in lname or med > 2000:
            if med > 500:  # avoid converting already-metres cruise (~10 km)
                LOGGER.info(
                    "Altitude %r units=%r (median abs=%.1f) → metres",
                    name,
                    units,
                    med,
                )
                return s * FT_TO_M
    if kind == "groundspeed":
        if "knot" in u or "kt" in u or "kt" in lname or "knot" in lname or (30 < med < 600):
            LOGGER.info(
                "Groundspeed %r units=%r (median abs=%.1f) → m/s",
                name,
                units,
                med,
            )
            return s * KTS_TO_MPS
    if kind == "vertical_rate":
        if (
            "ft/min" in u
            or "fpm" in u
            or "ft/min" in lname
            or "fpm" in lname
            or med > 20
        ):
            LOGGER.info(
                "Vertical rate %r units=%r (median abs=%.1f) → m/s",
                name,
                units,
                med,
            )
            return s * FPM_TO_MPS
    if kind == "fuel_flow":
        if "lb" in u or "lbs" in u or "pph" in u or "lb" in lname or "pph" in lname:
            LOGGER.info("Fuel flow %r units=%r → kg/s (from lb/h)", name, units)
            return s * LB_TO_KG / 3600.0
        if "kg/h" in u or "kgh" in lname or med > 5:
            LOGGER.info("Fuel flow %r units=%r → kg/s (from kg/h)", name, units)
            return s / 3600.0
    if kind == "fuel_qty":
        if "lb" in u or "lbs" in u or "lsb" in u or "lb" in lname or med > 5000:
            LOGGER.info("Fuel quantity %r units=%r → kg (from lb)", name, units)
            return s * LB_TO_KG
    return s


def _build_time_axis(
    channels: dict[str, Any],
    n: int,
    sample_hz: float | None = None,
    epoch: datetime | None = None,
) -> np.ndarray:
    """Return unix-second timestamps of length n."""
    keys = [k for k in channels if not k.startswith("meta_")]
    tkey = _match_key(keys, TIME_PATTERNS)
    epoch = epoch or datetime(2010, 1, 1, 0, 0, 0)

    if tkey is not None:
        t = _as_1d(channels[tkey])
        if len(t) >= n:
            t = t[:n]
            # relative seconds starting near 0
            if np.nanmax(t) < 1e9:
                base = epoch.timestamp()
                return base + (t - np.nanmin(t))
            if np.nanmax(t) > 1e12:
                return t / 1000.0
            return t

    hz = sample_hz or 1.0
    LOGGER.info(
        "No time channel found; synthesizing time at %.3f Hz from epoch %s",
        hz,
        epoch.isoformat(),
    )
    return epoch.timestamp() + np.arange(n, dtype=np.float64) / hz


def channels_to_trajectory(
    channels: dict[str, Any],
    flight_id: str,
    aircraft_type: str = "CRJ9",
    sample_hz: float | None = None,
    epoch: datetime | None = None,
) -> pl.DataFrame:
    """Map a flat channel dict to a standard AeroTwin trajectory DataFrame.

    Project 85 channels are recorded at different rates (e.g. ALT/GS @ 4 Hz,
    IVV @ 16 Hz, LATP @ 1 Hz). We pick a reference channel (prefer ALT), then
    resample others onto that timeline before unit conversion to SI.

    Default aircraft_type ``CRJ9`` reflects Project 85 regional-jet fleet;
    override when tail/type metadata is known.
    """
    keys = [k for k in channels if not str(k).startswith("meta_")]

    def pick(patterns: list[str]) -> tuple[str | None, np.ndarray | None]:
        k = _match_key(keys, patterns)
        if k is None:
            return None, None
        return k, _as_1d(channels[k])

    alt_k, alt = pick(ALT_PATTERNS)
    gs_k, gs = pick(GS_PATTERNS)
    vr_k, vr = pick(VR_PATTERNS)
    lat_k, lat = pick(LAT_PATTERNS)
    lon_k, lon = pick(LON_PATTERNS)
    cas_k, cas = pick(CAS_PATTERNS)
    mach_k, mach = pick(MACH_PATTERNS)
    tas_k, tas = pick(TAS_PATTERNS)

    # Reference timeline: prefer pressure altitude (ALT @ 4 Hz on Project 85)
    ref_k = alt_k or gs_k or vr_k
    if ref_k is None:
        raise ValueError(
            f"No trajectory channels found for {flight_id}. "
            f"Series keys sample: {keys[:40]}"
        )

    ref_series = _as_1d(channels[ref_k])
    n = int(len(ref_series))
    if n < 2:
        raise ValueError(f"{flight_id}: reference channel {ref_k} too short (n={n})")

    rate_dst = sample_hz or _channel_rate(channels, ref_k, default=4.0)
    LOGGER.info(
        "%s: reference channel=%s n=%d rate=%.3f Hz; series=%s",
        flight_id,
        ref_k,
        n,
        rate_dst,
        [k for k in (alt_k, gs_k, vr_k, lat_k, lon_k, cas_k) if k],
    )

    def align(
        name: str | None,
        series: np.ndarray | None,
        kind: str,
        default: float,
    ) -> np.ndarray:
        if name is None or series is None:
            return np.full(n, default, dtype=np.float64)
        rate_src = _channel_rate(channels, name, default=rate_dst)
        aligned = _resample_series(series, rate_src, n, rate_dst)
        units = _channel_units(channels, name)
        return _detect_units_and_scale(name, aligned, kind, units=units)

    if alt_k is None:
        LOGGER.warning("%s: missing altitude; filling 10000 m", flight_id)
    if gs_k is None:
        LOGGER.warning("%s: missing groundspeed; filling 200 m/s", flight_id)
    if vr_k is None:
        LOGGER.warning("%s: missing vertical_rate; filling 0", flight_id)

    alt_out = align(alt_k, alt, "altitude", 10000.0)
    gs_out = align(gs_k, gs, "groundspeed", 200.0)
    vr_out = align(vr_k, vr, "vertical_rate", 0.0)

    t_unix = _build_time_axis(channels, n, sample_hz=rate_dst, epoch=epoch)
    # Prefer rate-based synthesis when ACMT/DATE are not true timestamps
    if sample_hz is None:
        t_unix = (epoch or datetime(2010, 1, 1, 0, 0, 0)).timestamp() + (
            np.arange(n, dtype=np.float64) / rate_dst
        )
    ts = [datetime.utcfromtimestamp(float(x)) for x in t_unix]

    data: dict[str, Any] = {
        "timestamp": ts,
        "altitude": alt_out.astype(np.float64),
        "groundspeed": gs_out.astype(np.float64),
        "vertical_rate": vr_out.astype(np.float64),
        "typecode": [aircraft_type] * n,
        "source": ["dashlink"] * n,
        "flight_id": [flight_id] * n,
    }
    if lat is not None and lat_k:
        data["latitude"] = align(lat_k, lat, "latitude", 0.0)
    if lon is not None and lon_k:
        data["longitude"] = align(lon_k, lon, "longitude", 0.0)
    if cas is not None and cas_k:
        # CAS in Project 85 is KNOTS — convert to m/s for OpenAP cas2tas
        cas_aligned = _resample_series(
            cas, _channel_rate(channels, cas_k, rate_dst), n, rate_dst
        )
        cas_units = _channel_units(channels, cas_k)
        if "knot" in cas_units.lower() or "kt" in cas_units.lower() or (
            30 < float(np.nanmedian(np.abs(cas_aligned))) < 600
        ):
            cas_aligned = cas_aligned * KTS_TO_MPS
        data["CAS"] = cas_aligned
    if mach is not None and mach_k:
        data["mach"] = _resample_series(
            mach, _channel_rate(channels, mach_k, rate_dst), n, rate_dst
        )
    if tas is not None and tas_k:
        tas_aligned = _resample_series(
            tas, _channel_rate(channels, tas_k, rate_dst), n, rate_dst
        )
        tas_units = _channel_units(channels, tas_k)
        if "knot" in tas_units.lower() or (
            30 < float(np.nanmedian(np.abs(tas_aligned))) < 600
        ):
            tas_aligned = tas_aligned * KTS_TO_MPS
        # stash as groundspeed fallback is already set; keep TAS via mach/CAS path
        data["tas"] = tas_aligned

    traj = pl.DataFrame(data).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    return ensure_standard_traj_columns(traj)


def extract_fuel_series(
    channels: dict[str, Any],
) -> dict[str, Any]:
    """Locate fuel-related channels and return raw series + interpretation.

    Returns dict with keys:
    ``mode`` (flow | quantity | none), ``series`` (list of arrays),
    ``names``, ``unit_guess``, ``notes``.
    """
    keys = [k for k in channels if not str(k).startswith("meta_")]
    flow_keys = _match_all_keys(keys, FUEL_FLOW_PATTERNS)
    qty_keys = _match_all_keys(keys, FUEL_QTY_PATTERNS)

    # Prefer multi-engine flow sum
    if flow_keys:
        series = [_as_1d(channels[k]) for k in flow_keys]
        LOGGER.info(
            "Fuel FLOW channels detected: %s — will integrate for interval targets. "
            "ASSUMPTION: sum engines if multiple; units auto-detected.",
            flow_keys,
        )
        return {
            "mode": "flow",
            "series": series,
            "names": flow_keys,
            "unit_guess": "auto",
            "notes": "integrated fuel flow → interval burn (noisier than FOB delta)",
        }

    if qty_keys:
        series = [_as_1d(channels[k]) for k in qty_keys]
        LOGGER.info(
            "Fuel QUANTITY channels detected: %s — interval burn = start−end delta.",
            qty_keys,
        )
        return {
            "mode": "quantity",
            "series": series,
            "names": qty_keys,
            "unit_guess": "auto",
            "notes": "FOB-like quantity differences per interval",
        }

    LOGGER.warning(
        "No fuel-related channels matched. Keys sample: %s. "
        "Fuel targets will be unavailable (or physics-only).",
        keys[:40],
    )
    return {
        "mode": "none",
        "series": [],
        "names": [],
        "unit_guess": None,
        "notes": "no usable fuel signal",
    }


def reconstruct_fuel_intervals(
    channels: dict[str, Any],
    traj: pl.DataFrame,
    interval_s: float = DEFAULT_INTERVAL_S,
    min_interval_s: float = MIN_INTERVAL_S,
    fuel_rate_unit: str = "auto",
) -> pl.DataFrame:
    """Build interval-level ``fuel_kg`` labels from DASHlink fuel channels.

    Modes
    -----
    * **flow** – sum engine fuel-flow series, integrate with trapezoidal rule
      over each fixed interval.
    * **quantity** – difference of fuel-on-board (or used) at interval endpoints.
    * **none** – empty fuel frame (caller may still run physics-only features).

    Output columns: ``idx``, ``start``, ``end``, ``fuel_kg`` (plus optional
    ``fuel_flow_mean_kg_s``, ``label_method``).
    """
    if traj.is_empty() or "timestamp" not in traj.columns:
        return pl.DataFrame(
            schema={
                "idx": pl.Int64,
                "start": pl.Datetime("us"),
                "end": pl.Datetime("us"),
                "fuel_kg": pl.Float64,
            }
        )

    t0 = traj["timestamp"][0]
    t1 = traj["timestamp"][-1]
    intervals = construct_fixed_intervals(t0, t1, interval_s=interval_s, min_interval_s=min_interval_s)
    if intervals.is_empty():
        return intervals.with_columns(pl.lit(None).cast(pl.Float64).alias("fuel_kg"))

    fuel_info = extract_fuel_series(channels)
    mode = fuel_info["mode"]

    if mode == "none":
        return intervals.with_columns(
            pl.lit(None).cast(pl.Float64).alias("fuel_kg"),
            pl.lit("no_fuel_signal").alias("label_method"),
        )

    times = traj["timestamp"].to_list()
    rows: list[dict[str, Any]] = []
    n = len(traj)
    # Trajectory sample rate (Project 85 ALT/GS typically 4 Hz)
    if n >= 2:
        dt0 = (times[1] - times[0]).total_seconds()
        traj_hz = 1.0 / dt0 if dt0 > 0 else 4.0
    else:
        traj_hz = 4.0

    if mode == "flow":
        # Resample each engine FF onto the trajectory timeline; sum → kg/s
        flows = []
        for name, ser in zip(fuel_info["names"], fuel_info["series"]):
            rate_src = _channel_rate(channels, name, default=traj_hz)
            s = _resample_series(ser, rate_src, n, traj_hz)
            units = _channel_units(channels, name)
            if fuel_rate_unit == "auto":
                s = _detect_units_and_scale(name, s, "fuel_flow", units=units)
            elif fuel_rate_unit == "kg_h":
                s = s / 3600.0
            elif fuel_rate_unit == "lb_h":
                s = s * LB_TO_KG / 3600.0
            flows.append(s)
        total_ff = np.nansum(np.vstack(flows), axis=0)  # kg/s
        LOGGER.info(
            "Fuel flow sum: mean=%.4f kg/s over %d samples (%d engines)",
            float(np.nanmean(total_ff)),
            n,
            len(flows),
        )

        for row in intervals.iter_rows(named=True):
            s, e = row["start"], row["end"]
            idx = np.where([(t >= s and t <= e) for t in times])[0]
            if len(idx) < 2:
                fuel_kg = None
                mean_ff = None
            else:
                t_sub = [times[i] for i in idx]
                r_sub = total_ff[idx]
                fuel_kg = integrate_rate(t_sub, r_sub, unit="kg_s")
                mean_ff = float(np.nanmean(r_sub))
                # Guard pathological integration (sensor zeros / spikes)
                if fuel_kg is not None and (fuel_kg < 0 or fuel_kg > 50_000):
                    LOGGER.debug(
                        "Dropping implausible integrated fuel_kg=%.1f at interval %s",
                        fuel_kg,
                        row["idx"],
                    )
                    fuel_kg = None
            rows.append(
                {
                    "idx": row["idx"],
                    "start": s,
                    "end": e,
                    "fuel_kg": fuel_kg,
                    "fuel_flow_mean_kg_s": mean_ff,
                    "label_method": "integrated_fuel_flow",
                }
            )

    elif mode == "quantity":
        # Sum all tank quantities when multiple FQTY_* present (total FOB)
        qty_parts = []
        for name, ser in zip(fuel_info["names"], fuel_info["series"]):
            rate_src = _channel_rate(channels, name, default=1.0)
            q = _resample_series(ser, rate_src, n, traj_hz)
            units = _channel_units(channels, name)
            q = _detect_units_and_scale(name, q, "fuel_qty", units=units)
            qty_parts.append(q)
        qty = np.nansum(np.vstack(qty_parts), axis=0)
        LOGGER.info(
            "Fuel quantity total FOB: mean=%.1f kg from %s",
            float(np.nanmean(qty)),
            fuel_info["names"],
        )

        for row in intervals.iter_rows(named=True):
            s, e = row["start"], row["end"]
            mask_idx = [i for i, t in enumerate(times) if s <= t <= e]
            if len(mask_idx) < 2:
                fuel_kg = None
            else:
                q0, q1 = qty[mask_idx[0]], qty[mask_idx[-1]]
                if not (np.isfinite(q0) and np.isfinite(q1)):
                    fuel_kg = None
                else:
                    # FOB decreases over the interval
                    delta = float(q0 - q1)
                    names_l = " ".join(fuel_info["names"]).lower()
                    if "used" in names_l:
                        delta = float(q1 - q0)
                    if delta < 0:
                        # refuel / sensor glitch — skip rather than abs()
                        fuel_kg = None
                    else:
                        fuel_kg = delta
            rows.append(
                {
                    "idx": row["idx"],
                    "start": s,
                    "end": e,
                    "fuel_kg": fuel_kg,
                    "fuel_flow_mean_kg_s": None,
                    "label_method": "quantity_delta",
                }
            )

    out = pl.DataFrame(rows).with_columns(
        pl.col("start").cast(pl.Datetime("us")),
        pl.col("end").cast(pl.Datetime("us")),
    )
    n_valid = out.filter(pl.col("fuel_kg").is_not_null() & pl.col("fuel_kg").is_finite()).height
    LOGGER.info(
        "Reconstructed %d intervals (%d with valid fuel_kg) via %s",
        len(out),
        n_valid,
        mode,
    )
    return out


def _flight_id_from_path(path: Path) -> str:
    stem = path.stem
    # common patterns: flight_652_... or 652.mat
    return f"dashlink_{stem}"


def _infer_tail_type(path: Path, channels: dict[str, Any]) -> tuple[str | None, str]:
    """Best-effort tail number and aircraft type from path / meta channels."""
    tail = None
    m = re.search(r"(65[2-9]|66[0-9]|67[0-9]|68[0-7])", path.stem)
    if m:
        tail = m.group(1)
    ac_type = "CRJ9"  # Project 85 default regional jet family
    for k, v in channels.items():
        if "type" in k.lower() or "acft" in k.lower():
            try:
                ac_type = str(np.asarray(v).ravel()[0])
            except Exception:
                pass
    return tail, ac_type


def load_dashlink_flight(
    path: Path | str,
    aircraft_type: str | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    sample_hz: float | None = None,
) -> dict[str, Any]:
    """Load one DASHlink flight into trajectory + fuel + metadata.

    Returns
    -------
    dict with keys:
        ``flight_id``, ``traj``, ``fuel``, ``meta``, ``channels_summary``
    """
    path = Path(path)
    channels = load_mat_file(path)
    flight_id = _flight_id_from_path(path)
    tail, ac_guess = _infer_tail_type(path, channels)
    ac_type = aircraft_type or ac_guess

    traj = channels_to_trajectory(
        channels,
        flight_id=flight_id,
        aircraft_type=ac_type,
        sample_hz=sample_hz,
    )
    fuel = reconstruct_fuel_intervals(channels, traj, interval_s=interval_s)

    t0 = traj["timestamp"][0] if len(traj) else None
    t1 = traj["timestamp"][-1] if len(traj) else None
    meta = {
        "flight_id": flight_id,
        "aircraft_type": ac_type,
        "origin_icao": None,
        "destination_icao": None,
        "takeoff": t0,
        "landed": t1,
        "tail": tail,
        "source": "dashlink_project85",
        "mat_path": str(path),
        "label_source": (
            fuel["label_method"][0]
            if "label_method" in fuel.columns and len(fuel)
            else "unknown"
        ),
    }
    summary = {
        "n_channels": len([k for k in channels if not str(k).startswith("meta_")]),
        "n_traj_pts": len(traj),
        "n_intervals": len(fuel),
        "fuel_valid": int(
            fuel.filter(pl.col("fuel_kg").is_not_null()).height
            if "fuel_kg" in fuel.columns
            else 0
        ),
    }
    return {
        "flight_id": flight_id,
        "traj": traj,
        "fuel": fuel,
        "meta": meta,
        "channels_summary": summary,
    }


def load_dashlink_directory(
    directory: Path | str,
    max_flights: int | None = None,
    aircraft_type: str | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    pattern: str = "*.mat",
    min_duration_s: float = 300.0,
    min_max_gs_mps: float = 40.0,
) -> list[dict[str, Any]]:
    """Load ``.mat`` flights under ``directory`` (recursive).

    Skips files that fail to parse and ground/taxi stubs that lack usable
    airborne trajectory (duration / max groundspeed filters). Stops once
    ``max_flights`` good flights are collected (pilot-friendly).
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"DASHlink directory not found: {directory}. "
            "Download Project 85 samples and pass --dashlink-dir."
        )

    files = sorted(directory.rglob(pattern))
    # Scan further than max_flights so short stubs can be skipped
    scan_cap = None if max_flights is None else max(max_flights * 5, max_flights + 20)
    if scan_cap is not None:
        files = files[:scan_cap]

    LOGGER.info(
        "Found MAT files under %s; scanning up to %d (target good flights=%s)",
        directory,
        len(files),
        max_flights,
    )
    out: list[dict[str, Any]] = []
    n_skip_quality = 0
    for fp in files:
        if max_flights is not None and len(out) >= max_flights:
            break
        try:
            rec = load_dashlink_flight(fp, aircraft_type=aircraft_type, interval_s=interval_s)
            traj = rec["traj"]
            if traj.is_empty() or "timestamp" not in traj.columns:
                n_skip_quality += 1
                continue
            t0, t1 = traj["timestamp"][0], traj["timestamp"][-1]
            dur = (t1 - t0).total_seconds()
            max_gs = float(traj["groundspeed"].max()) if "groundspeed" in traj.columns else 0.0
            if dur < min_duration_s or max_gs < min_max_gs_mps:
                LOGGER.info(
                    "Skip %s (quality): duration=%.0fs max_gs=%.1f m/s "
                    "(need duration>=%.0fs and max_gs>=%.1f)",
                    fp.name,
                    dur,
                    max_gs,
                    min_duration_s,
                    min_max_gs_mps,
                )
                n_skip_quality += 1
                continue
            if rec["channels_summary"].get("fuel_valid", 0) == 0:
                LOGGER.warning("%s: loaded but no valid fuel intervals", fp.name)
            out.append(rec)
        except Exception as exc:
            LOGGER.exception("Failed to load %s: %s", fp, exc)
    LOGGER.info(
        "Loaded %d good DASHlink flights (skipped %d low-quality/short; scanned %d files)",
        len(out),
        n_skip_quality,
        len(files),
    )
    return out


def probe_mat_parameters(path: Path | str, max_keys: int = 80) -> pl.DataFrame:
    """Diagnostic table of parameter names and shapes for audit Phase 1.

    Prefer listing time-series channels first (size > 1), then meta.
    """
    channels = load_mat_file(path)
    series_items = [(k, v) for k, v in channels.items() if not str(k).startswith("meta_")]
    meta_items = [(k, v) for k, v in channels.items() if str(k).startswith("meta_")]
    # Series first so --probe clearly shows trajectory arrays
    ordered = series_items + meta_items

    rows = []
    for k, v in ordered[:max_keys]:
        try:
            a = np.asarray(v)
            finite_frac = None
            if a.size and np.issubdtype(a.dtype, np.number):
                finite_frac = float(np.isfinite(a.astype(np.float64)).mean())
            rows.append(
                {
                    "name": k,
                    "shape": str(a.shape),
                    "dtype": str(a.dtype),
                    "size": int(a.size),
                    "finite_frac": finite_frac,
                    "rate_hz": channels.get(f"meta_{k}_rate") if not str(k).startswith("meta_") else None,
                    "units": channels.get(f"meta_{k}_units") if not str(k).startswith("meta_") else None,
                }
            )
        except Exception:
            rows.append(
                {
                    "name": k,
                    "shape": "?",
                    "dtype": type(v).__name__,
                    "size": 0,
                    "finite_frac": None,
                    "rate_hz": None,
                    "units": None,
                }
            )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Probe / load DASHlink MAT flights")
    p.add_argument("path", type=str, help="Path to .mat file or directory")
    p.add_argument("--probe", action="store_true", help="Print parameter table only")
    p.add_argument("--max-flights", type=int, default=3)
    args = p.parse_args()
    path = Path(args.path)
    if path.is_file():
        if args.probe:
            print(probe_mat_parameters(path))
        else:
            rec = load_dashlink_flight(path)
            print(rec["channels_summary"])
            print(rec["traj"].head())
            print(rec["fuel"].head())
    elif path.is_dir():
        recs = load_dashlink_directory(path, max_flights=args.max_flights)
        print(f"Loaded {len(recs)} flights")
        for r in recs:
            print(r["flight_id"], r["channels_summary"])
    else:
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)
