"""Flight-sequence Transformer residual corrector (physics + learned residual)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

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

PARQUET = project_root() / "featured_dataset.parquet"
OUT = project_root() / "figures"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
MAX_SEQ_LEN = 24
BATCH_SIZE = 256
EPOCHS = 12
LR = 1e-3
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def numeric_features(df: pl.DataFrame) -> list[str]:
    energy = [c for c in ENERGY_FEATURES if c in df.columns]
    weather = [c for c in WEATHER_FEATURES if c in df.columns]
    cols = list(BASE_NUMERIC) + energy + weather
    return [c for c in cols if c in df.columns]


class FlightSequenceDataset(Dataset):
    def __init__(
        self,
        flight_ids: list[str],
        seq_features: np.ndarray,
        seq_physics: np.ndarray,
        seq_residual: np.ndarray,
        seq_mask: np.ndarray,
        ac_type_idx: np.ndarray,
    ) -> None:
        self.flight_ids = flight_ids
        self.seq_features = seq_features
        self.seq_physics = seq_physics
        self.seq_residual = seq_residual
        self.seq_mask = seq_mask
        self.ac_type_idx = ac_type_idx

    def __len__(self) -> int:
        return len(self.flight_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.seq_features[idx], dtype=torch.float32),
            torch.tensor(self.seq_physics[idx], dtype=torch.float32),
            torch.tensor(self.seq_residual[idx], dtype=torch.float32),
            torch.tensor(self.seq_mask[idx], dtype=torch.bool),
            torch.tensor(self.ac_type_idx[idx], dtype=torch.long),
        )


class TransformerResidual(nn.Module):
    def __init__(self, n_features: int, n_types: int, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(n_features, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=N_HEADS,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        ac_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.feature_proj(x)
        h = h + self.type_emb(ac_idx).unsqueeze(1)
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = h + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h).squeeze(-1)


def build_sequences(
    df: pl.DataFrame,
    flight_ids: np.ndarray,
    idx: np.ndarray,
    feat_cols: list[str],
    le: LabelEncoder,
    scaler: StandardScaler,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    held_flights = np.unique(flight_ids[idx])
    sub = (
        df.filter(pl.col("flight_id").is_in(held_flights.tolist()))
        .sort(["flight_id", "interval_idx"])
        .to_pandas()
    )

    flights = []
    seq_x, seq_phys, seq_res, seq_mask, ac_idx = [], [], [], [], []

    for fid, pdf in sub.groupby("flight_id", sort=False):
        n = len(pdf)
        t = min(n, MAX_SEQ_LEN)
        x = np.zeros((MAX_SEQ_LEN, len(feat_cols)), dtype=np.float32)
        phys = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        res = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=bool)

        raw = pdf[feat_cols].to_numpy(dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw = scaler.transform(raw).astype(np.float32)

        x[:t] = raw[:t]
        phys[:t] = pdf["physics_fuel_kg"].to_numpy()[:t].astype(np.float32)
        res[:t] = pdf["residual_kg"].to_numpy()[:t].astype(np.float32)
        mask[:t] = True

        ac = str(pdf["aircraft_type"].iloc[0])
        ac_i = int(le.transform([ac])[0])

        flights.append(str(fid))
        seq_x.append(x)
        seq_phys.append(phys)
        seq_res.append(res)
        seq_mask.append(mask)
        ac_idx.append(ac_i)

    return (
        flights,
        np.stack(seq_x),
        np.stack(seq_phys),
        np.stack(seq_res),
        np.stack(seq_mask),
        np.asarray(ac_idx),
    )


def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, phys, res, mask, ac in loader:
        x, res, mask, ac = x.to(DEVICE), res.to(DEVICE), mask.to(DEVICE), ac.to(DEVICE)
        optimizer.zero_grad()
        pred_res = model(x, ac, mask)
        loss = criterion(pred_res[mask], res[mask])
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * int(mask.sum().item())
        n += int(mask.sum().item())
    return total / max(n, 1)


@torch.no_grad()
def predict_intervals(
    model,
    loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_res_pred, all_res_true, all_phys = [], [], []
    for x, phys, res, mask, ac in loader:
        x, phys, res, mask, ac = x.to(DEVICE), phys.to(DEVICE), res.to(DEVICE), mask.to(DEVICE), ac.to(DEVICE)
        pred_res = model(x, ac, mask)
        valid = mask.cpu().numpy()
        pr = pred_res.cpu().numpy()
        for i in range(pr.shape[0]):
            m = valid[i]
            all_res_pred.append(pr[i][m])
            all_res_true.append(res.cpu().numpy()[i][m])
            all_phys.append(phys.cpu().numpy()[i][m])
    return (
        np.concatenate(all_res_pred),
        np.concatenate(all_res_true),
        np.concatenate(all_phys),
    )


def main() -> None:
    print("=" * 72)
    print(f"TRANSFORMER RESIDUAL (device={DEVICE})")
    print("=" * 72)

    df = load_and_clean(PARQUET)
    fids = df["flight_id"].to_numpy()
    feat_cols = numeric_features(df)
    print(f"Intervals: {len(df):,} | Sequence features: {len(feat_cols)}")

    train_idx, test_idx, train_fids, test_fids = flight_level_split(fids)

    pdf = df.to_pandas()
    le = LabelEncoder()
    le.fit(pdf["aircraft_type"].astype(str).unique())

    raw = pdf[feat_cols].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    scaler.fit(raw[train_idx])

    print("Building flight sequences ...", flush=True)
    t0 = time.perf_counter()
    tr_pack = build_sequences(df, fids, train_idx, feat_cols, le, scaler)
    te_pack = build_sequences(df, fids, test_idx, feat_cols, le, scaler)
    print(f"  Train flights: {len(tr_pack[0]):,} | Test flights: {len(te_pack[0]):,} ({time.perf_counter() - t0:.1f}s)")

    train_loader = DataLoader(
        FlightSequenceDataset(*tr_pack),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        FlightSequenceDataset(*te_pack),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TransformerResidual(len(feat_cols), len(le.classes_)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    print(f"Training {EPOCHS} epochs ...", flush=True)
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % 3 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}/{EPOCHS} loss={loss:.4f}", flush=True)
    print(f"  done ({time.perf_counter() - t0:.1f}s)")

    res_pred, res_true, phys = predict_intervals(model, test_loader)
    fuel_pred = phys + res_pred
    fuel_true = phys + res_true

    m_residual = evaluate(res_true, res_pred)
    m_fuel = evaluate(fuel_true, fuel_pred)
    m_physics = evaluate(fuel_true, phys)

    # Direct hybrid reference: physics as strong feature is already in tree models;
    # report OpenAP-only baseline on same interval subset for context.
    rows = [
        {"model": "OpenAP_only", "target": "actual_fuel_kg", **m_physics},
        {"model": "Transformer_residual", "target": "residual_kg", **m_residual},
        {"model": "Transformer_hybrid", "target": "actual_fuel_kg", **m_fuel},
    ]
    table = pl.DataFrame(rows)
    out_path = OUT / "table_transformer_residual.csv"
    table.write_csv(out_path)

    meta = pl.DataFrame(
        [
            {
                "n_train_flights": len(tr_pack[0]),
                "n_test_flights": len(te_pack[0]),
                "n_test_intervals": len(res_pred),
                "max_seq_len": MAX_SEQ_LEN,
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "epochs": EPOCHS,
                "device": DEVICE,
            }
        ]
    )
    meta.write_csv(OUT / "table_transformer_residual_meta.csv")

    print(f"\nSaved {out_path}")
    print("\nResults (held-out flight sequences):")
    for r in rows:
        print(f"  {r['model']:22s} MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} R2={r['r2']:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()