from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from aerotwin.engine.feature_engineering import ENERGY_FEATURES, OPERATIONAL_FEATURES, enrich_from_columns


def main(parquet_path: Path | None = None) -> None:
    path = parquet_path or root / "featured_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    print(f"Loading {path} ...")
    df = pl.read_parquet(path)
    n_before = len(df.columns)
    print(f"  {len(df):,} rows, {n_before} columns")

    enriched = enrich_from_columns(df)
    new_cols = [c for c in ENERGY_FEATURES + OPERATIONAL_FEATURES if c in enriched.columns]
    print(f"  Added/updated {len(new_cols)} physics-informed feature columns")

    enriched.write_parquet(path)
    print(f"Saved -> {path} ({len(enriched.columns)} columns)")


if __name__ == "__main__":
    main()