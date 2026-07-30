"""Load and prepare the frozen teacher distillation dataset for student training."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

# Fallback if meta JSON is missing (matches Step-1 export order).
FEATURE_COLS_DEFAULT: list[str] = [
    "duration_s",
    "start_fraction_of_flight",
    "end_fraction_of_flight",
    "n_traj_pts",
    "has_acars_in_window",
    "mean_altitude",
    "median_altitude",
    "max_altitude",
    "std_altitude",
    "mean_groundspeed",
    "std_groundspeed",
    "max_groundspeed",
    "mean_vertical_rate",
    "std_vertical_rate",
    "climb_fraction",
    "cruise_fraction",
    "descent_fraction",
    "ref_mass_kg",
    "mean_potential_energy_j",
    "mean_kinetic_energy_j",
    "mean_specific_energy_jpkg",
    "specific_energy_start",
    "specific_energy_end",
    "energy_change_jpkg",
    "energy_rate_jpkg_s",
    "climb_efficiency",
    "energy_efficiency",
    "cumulative_energy_change_jpkg",
    "headwind_mps",
    "crosswind_mps",
    "temperature_k",
    "pressure_pa",
    "isa_deviation_k",
    "density_altitude_m",
    "physics_fuel_kg",
    "aircraft_type",
    "method",
    "origin_icao",
    "destination_icao",
    "r3_tow_kg",
    "r3_landing_mass_kg",
    "r3_mass_start_kg",
    "r3_mass_end_kg",
    "r3_mean_mass_kg",
    "r3_min_mass_kg",
    "r3_max_mass_kg",
    "r3_mass_std_kg",
    "r3_mass_consumed_kg",
    "r3_mass_rate_kgps",
    "r3_fuel_fraction",
    "r3_remaining_fuel_frac",
    "r3_phase_mass_kg",
    "r3_cruise_mass_kg",
    "r3_wing_loading_cur",
    "r3_oew_base_kg",
    "r3_mean_pe_j",
    "r3_mean_ke_j",
    "r3_fuel_mass_efficiency",
    "r3_tow_mtow_ratio",
    "r3_cruise_mass_fuel_ratio",
]

CAT_FEATURES = ("aircraft_type", "method", "origin_icao", "destination_icao")


def load_feature_cols(root: Path) -> list[str]:
    meta_path = root / "docs" / "reports" / "distillation_dataset_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cols = list(meta.get("feature_cols") or [])
        if cols:
            return cols
    return list(FEATURE_COLS_DEFAULT)


class TensorDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_gt: np.ndarray,
        y_teacher: np.ndarray,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y_gt = torch.as_tensor(y_gt, dtype=torch.float32)
        self.y_teacher = torch.as_tensor(y_teacher, dtype=torch.float32)
        self.sample_ids = (
            None
            if sample_ids is None
            else torch.as_tensor(sample_ids, dtype=torch.int64)
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "x": self.x[idx],
            "y_gt": self.y_gt[idx],
            "y_teacher": self.y_teacher[idx],
        }
        if self.sample_ids is not None:
            item["sample_id"] = self.sample_ids[idx]
        return item


@dataclass
class DistillationData:
    """Flight-level train/val split of the frozen distillation parquet."""

    feature_cols: list[str]
    numeric_cols: list[str]
    cat_cols: list[str]
    in_dim: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    x_train: np.ndarray
    x_val: np.ndarray
    y_gt_train: np.ndarray
    y_gt_val: np.ndarray
    y_teacher_train: np.ndarray
    y_teacher_val: np.ndarray
    sample_id_train: np.ndarray
    sample_id_val: np.ndarray
    flight_id_train: np.ndarray
    flight_id_val: np.ndarray
    scaler: StandardScaler
    ohe: OneHotEncoder
    n_samples: int
    n_flights: int
    val_fraction: float
    seed: int
    parquet_path: str
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Path,
        *,
        root: Path | None = None,
        feature_cols: Sequence[str] | None = None,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> DistillationData:
        root = root or parquet_path.resolve().parent
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Distillation dataset not found: {parquet_path}. "
                "Run Step 1 first; do not regenerate in this step."
            )

        df = pl.read_parquet(parquet_path)
        feats = list(feature_cols) if feature_cols else load_feature_cols(root)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns missing from dataset: {missing[:10]}")
        for req in ("ground_truth", "teacher_prediction", "flight_id"):
            if req not in df.columns:
                raise ValueError(f"Required column missing: {req}")

        # Keep only rows with finite targets.
        df = df.filter(
            pl.col("ground_truth").is_finite()
            & pl.col("teacher_prediction").is_finite()
            & pl.col("flight_id").is_not_null()
        )

        cat_cols = [c for c in CAT_FEATURES if c in feats]
        numeric_cols = [c for c in feats if c not in cat_cols]

        # Flight-level split (no interval leakage).
        # Sort for deterministic train_test_split across platforms/runs.
        flights = np.sort(df["flight_id"].unique().to_numpy())
        tr_f, va_f = train_test_split(
            flights, test_size=val_fraction, random_state=seed
        )
        tr_set = set(map(str, tr_f))
        va_set = set(map(str, va_f))
        fids = df["flight_id"].cast(pl.Utf8).to_numpy()
        train_mask = np.array([f in tr_set for f in fids], dtype=bool)
        val_mask = np.array([f in va_set for f in fids], dtype=bool)
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        LOGGER.info(
            "Flight split: train_flights=%d val_flights=%d | train_rows=%d val_rows=%d",
            len(tr_f),
            len(va_f),
            len(train_idx),
            len(val_idx),
        )

        # Numeric matrix: null/NaN -> column median (fit on train only).
        num_all = np.column_stack(
            [
                df[c].cast(pl.Float64, strict=False).to_numpy().astype(np.float64)
                for c in numeric_cols
            ]
        )
        medians = np.nanmedian(num_all[train_idx], axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        for j in range(num_all.shape[1]):
            col = num_all[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col = col.copy()
                col[bad] = medians[j]
                num_all[:, j] = col

        scaler = StandardScaler()
        x_num_train = scaler.fit_transform(num_all[train_idx])
        x_num_val = scaler.transform(num_all[val_idx])

        # Categorical one-hot (handle_unknown for robustness).
        cat_frame = df.select(
            [pl.col(c).cast(pl.Utf8).fill_null("missing") for c in cat_cols]
        ).to_pandas()
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        x_cat_train = ohe.fit_transform(cat_frame.iloc[train_idx])
        x_cat_val = ohe.transform(cat_frame.iloc[val_idx])

        x_train = np.hstack([x_num_train, x_cat_train]).astype(np.float32)
        x_val = np.hstack([x_num_val, x_cat_val]).astype(np.float32)

        y_gt = df["ground_truth"].to_numpy().astype(np.float64)
        y_teacher = df["teacher_prediction"].to_numpy().astype(np.float64)
        sample_ids = (
            df["sample_id"].to_numpy().astype(np.int64)
            if "sample_id" in df.columns
            else np.arange(len(df), dtype=np.int64)
        )
        flight_ids = df["flight_id"].cast(pl.Utf8).to_numpy()

        return cls(
            feature_cols=feats,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
            in_dim=int(x_train.shape[1]),
            train_idx=train_idx,
            val_idx=val_idx,
            x_train=x_train,
            x_val=x_val,
            y_gt_train=y_gt[train_idx],
            y_gt_val=y_gt[val_idx],
            y_teacher_train=y_teacher[train_idx],
            y_teacher_val=y_teacher[val_idx],
            sample_id_train=sample_ids[train_idx],
            sample_id_val=sample_ids[val_idx],
            flight_id_train=flight_ids[train_idx],
            flight_id_val=flight_ids[val_idx],
            scaler=scaler,
            ohe=ohe,
            n_samples=len(df),
            n_flights=int(len(flights)),
            val_fraction=val_fraction,
            seed=seed,
            parquet_path=str(parquet_path),
            extras={
                "n_numeric": len(numeric_cols),
                "n_categorical": len(cat_cols),
                "ohe_dim": int(x_cat_train.shape[1]),
            },
        )

    def loaders(
        self,
        *,
        batch_size: int = 2048,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train_shuffle, train_eval, val_eval) loaders."""
        train_ds = TensorDataset(
            self.x_train,
            self.y_gt_train,
            self.y_teacher_train,
            self.sample_id_train,
        )
        val_ds = TensorDataset(
            self.x_val,
            self.y_gt_val,
            self.y_teacher_val,
            self.sample_id_val,
        )
        pin = torch.cuda.is_available()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin,
        )
        train_eval_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin,
        )
        return train_loader, train_eval_loader, val_loader
