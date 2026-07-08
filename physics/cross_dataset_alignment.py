from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

# Columns that carry targets, identifiers, or metadata and must NOT be
# standardized or treated as modeling features during alignment.
TARGET_COLS = ("actual_fuel_kg", "physics_fuel_kg", "residual_kg")
ID_COLS = ("flight_id", "interval_idx", "start", "end")
META_COLS = ("aircraft_type", "origin_icao", "destination_icao", "method")

# Cast alignment: numeric features are coerced to this dtype so datasets
# produced by different builders share a single numeric representation.
COMMON_FLOAT_DTYPE = pl.Float64


def _is_excluded(col: str) -> bool:
    """True for target / id / metadata columns that are never standardized."""
    return col in TARGET_COLS or col in ID_COLS or col in META_COLS


def _feature_columns(df: pl.DataFrame) -> list[str]:
    """Numeric, non-excluded columns usable as aligned modeling features."""
    out: list[str] = []
    for name, dtype in df.schema.items():
        if _is_excluded(name):
            continue
        if dtype.is_numeric() and not dtype.is_temporal():
            out.append(name)
    return out


def align_schemas(*frames: pl.DataFrame) -> list[pl.DataFrame]:
    """Bring several datasets onto a single shared schema.

    All frames are projected to the union of their columns, ordered with
    targets first, then ids, then metadata, then the remaining features
    sorted alphabetically. Columns absent from a frame are filled with null
    and numeric columns are coerced to ``COMMON_FLOAT_DTYPE`` so a model
    trained on one dataset can consume another without schema mismatch.
    """
    if not frames:
        return []

    all_cols: list[str] = []
    seen: set[str] = set()
    for f in frames:
        for c in f.columns:
            if c not in seen:
                seen.add(c)
                all_cols.append(c)

    # Stable, meaningful column ordering across datasets.
    ordered: list[str] = [c for c in TARGET_COLS if c in all_cols]
    ordered += [c for c in ID_COLS if c in all_cols]
    ordered += [c for c in META_COLS if c in all_cols]
    rest = sorted(c for c in all_cols if c not in set(ordered))
    ordered += rest

    aligned: list[pl.DataFrame] = []
    for f in frames:
        missing = [c for c in ordered if c not in f.columns]
        out = f
        if missing:
            out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
        # Coerce numeric columns to the common float dtype for consistency.
        cast_exprs = []
        for c in ordered:
            dtype = out.schema.get(c)
            if dtype is not None and dtype.is_numeric() and not dtype.is_temporal():
                cast_exprs.append(pl.col(c).cast(COMMON_FLOAT_DTYPE, strict=False))
        if cast_exprs:
            out = out.with_columns(cast_exprs)
        aligned.append(out.select(ordered))
    return aligned


class FeatureAligner:
    """Align one or more datasets to a reference dataset's schema and scale.

    The aligner is fit on a reference dataset (typically the training set or
    the canonical featured dataset). It records the union schema and, when
    standardization is enabled, the per-feature mean and standard deviation.
    ``transform`` then brings any other dataset onto that schema and scale so
    feature distributions from different data sources line up.

    Example
    -------
    >>> aligner = FeatureAligner().fit(reference_df)
    >>> train_aligned, test_aligned = aligner.transform(train_df, test_df)
    """

    def __init__(self, standardize: bool = True, eps: float = 1e-9) -> None:
        self.standardize = standardize
        self.eps = eps
        self.columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}

    def fit(self, reference: pl.DataFrame) -> "FeatureAligner":
        """Learn the shared schema and, optionally, feature statistics."""
        self.columns_ = align_schemas(reference)[0].columns
        features = _feature_columns(reference)
        self.feature_columns_ = features
        if self.standardize:
            for col in features:
                vals = reference[col].drop_nulls().cast(pl.Float64)
                if len(vals) == 0:
                    self._mean[col], self._std[col] = 0.0, 1.0
                    continue
                mean = float(vals.mean())
                std = float(vals.std())
                std = std if std and std > self.eps else 1.0
                self._mean[col], self._std[col] = mean, std
        return self

    def _scale(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.standardize:
            return df
        exprs = []
        for col in self.feature_columns_:
            if col in df.columns:
                m = self._mean.get(col, 0.0)
                s = self._std.get(col, 1.0)
                exprs.append(((pl.col(col) - m) / s).alias(col))
        return df.with_columns(exprs)

    def transform(self, *frames: pl.DataFrame) -> list[pl.DataFrame]:
        """Align and (optionally) standardize the supplied datasets."""
        if not self.columns_:
            raise RuntimeError("FeatureAligner must be fit before transform.")
        aligned = align_schemas(*frames)
        return [self._scale(df) for df in aligned]

    def fit_transform(self, reference: pl.DataFrame, *others: pl.DataFrame) -> list[pl.DataFrame]:
        self.fit(reference)
        return self.transform(reference, *others)


def cross_align(
    reference: pl.DataFrame,
    others: Iterable[pl.DataFrame],
    standardize: bool = True,
) -> list[pl.DataFrame]:
    """Convenience wrapper: align a reference plus several other datasets."""
    aligner = FeatureAligner(standardize=standardize)
    return aligner.fit_transform(reference, *others)


def main(paths: Sequence[str], standardize: bool = True, outdir: Path | None = None) -> None:
    """Align multiple featured-dataset parquet files onto a common schema.

    The first path is treated as the reference dataset; its schema and
    feature statistics are the alignment target. Aligned copies are written
    with an ``aligned__`` prefix so the originals are never overwritten.
    """
    if len(paths) < 2:
        raise SystemExit("Provide at least two parquet paths (reference + others).")

    root = Path(__file__).resolve().parents[1]
    outdir = outdir or root
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Aligning {len(paths)} datasets (reference={paths[0]})...")
    reference = pl.read_parquet(paths[0])
    others = [pl.read_parquet(p) for p in paths[1:]]

    aligned = cross_align(reference, others, standardize=standardize)

    for p, df in zip(paths, aligned):
        stem = Path(p).stem
        _write(outdir / f"aligned__{stem}.parquet", df)

    print(f"Wrote aligned datasets to {outdir} (prefix 'aligned__').")
    print("Shared schema:")
    for c in aligned[0].columns:
        print(f"  - {c} ({aligned[0].schema[c]})")


def _write(path: Path, df: pl.DataFrame) -> None:
    df.write_parquet(path)
    print(f"  {path.name}: {len(df):,} rows / {len(df.columns)} cols")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        print("Usage: python physics/cross_dataset_alignment.py PATH1 PATH2 [PATH3 ...] [--no-std]")
        raise SystemExit(0)

    standardize = True
    if "--no-std" in args:
        standardize = False
        args.remove("--no-std")

    main(args, standardize=standardize)
