from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from physics.eval_framework import RANDOM_STATE


class _FeatureTokenTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1)) + self.pos_emb
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


def train_transformer_residual(
    feature_cols: list[str],
    X_train,
    X_test,
    y_residual_train: np.ndarray,
    physics_test: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_STATE)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
    X_te = scaler.transform(np.asarray(X_test, dtype=np.float64))
    y_tr = np.asarray(y_residual_train, dtype=np.float64)

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = _FeatureTokenTransformer(X_tr.shape[1], d_model, n_heads, n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_res = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()

    return physics_test + pred_res
