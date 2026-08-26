"""FT-Transformer student for tabular regression.

Faithful to Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
(NeurIPS 2021): Feature Tokenizer → [CLS] + PreNorm Transformer stack → linear head.

Input contract
--------------
The frozen distillation pipeline still produces a dense matrix::

    x = [ scaled_numeric (n_num) | one-hot categories (sum cards) ]

When ``n_num_features`` and ``cat_cardinalities`` are provided (recommended),
the model **decodes** that vector into paper-native continuous tokens +
categorical embeddings (~60 tokens). Preprocessing artifacts (scaler, OHE)
are unchanged — only the architecture changes.

If cardinalities are omitted, every column is treated as a continuous feature
(dense tokenizer over ``in_dim`` columns) — correct but much slower for large
OHE widths.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalFeatureTokenizer(nn.Module):
    """token_f = x_f * W_f + b_f  for each continuous feature."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        if n_features < 1 or d_token < 1:
            raise ValueError("n_features and d_token must be positive")
        self.n_features = int(n_features)
        self.d_token = int(d_token)
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(1)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) → (B, F, d)
        return x.unsqueeze(-1) * self.weight + self.bias


class CategoricalFeatureTokenizer(nn.Module):
    """One embedding table per categorical feature (paper §3.1)."""

    def __init__(self, cardinalities: Sequence[int], d_token: int) -> None:
        super().__init__()
        self.cardinalities = [int(c) for c in cardinalities]
        self.d_token = int(d_token)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in self.cardinalities]
        )
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight, a=math.sqrt(5))

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, n_cat) long indices
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)  # (B, n_cat, d)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token={d_token} must be divisible by n_heads={n_heads}")
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.W_q = nn.Linear(d_token, d_token, bias=bias)
        self.W_k = nn.Linear(d_token, d_token, bias=bias)
        self.W_v = nn.Linear(d_token, d_token, bias=bias)
        self.W_out = nn.Linear(d_token, d_token, bias=bias)
        self.dropout_p = float(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, l, _ = t.shape
        return t.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Self-attention. When ``need_weights`` is True, also return analytic softmax weights.

        The residual output path is unchanged when ``need_weights=False`` (identical to
        the production forward). Analytic weights are computed *after* the same SDPA
        (or fallback) output and do not affect predictions.
        """
        q = self._split(self.W_q(x))
        k = self._split(self.W_k(x))
        v = self._split(self.W_v(x))
        drop = self.dropout_p if self.training else 0.0
        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        except Exception:
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn = F.softmax(attn, dim=-1)
            if drop > 0:
                attn = F.dropout(attn, p=drop, training=self.training)
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.d_token)
        out = self.W_out(out)
        if not need_weights:
            return out
        # Analysis-only: softmax(QK^T/√d) at dropout=0 (eval). Does not modify ``out``.
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            weights = F.softmax(scores, dim=-1)
        return out, weights


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        *,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        ffn_d_hidden: int | None = None,
    ) -> None:
        super().__init__()
        if ffn_d_hidden is None:
            ffn_d_hidden = int(round((8 / 3) * d_token))
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = MultiheadAttention(
            d_token, n_heads, dropout=attention_dropout, bias=True
        )
        self.drop_path1 = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden * 2),
            GEGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.drop_path2 = nn.Dropout(residual_dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        if need_weights:
            a, w = self.attn(h, need_weights=True)
            x = x + self.drop_path1(a)
            x = x + self.drop_path2(self.ffn(self.norm2(x)))
            return x, w
        x = x + self.drop_path1(self.attn(h, need_weights=False))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for scalar fuel-burn regression."""

    def __init__(
        self,
        in_dim: int,
        *,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        ffn_d_hidden: int | None = None,
        d_out: int = 1,
        n_num_features: int | None = None,
        cat_cardinalities: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if d_token % n_heads != 0:
            raise ValueError(f"d_token ({d_token}) must be divisible by n_heads ({n_heads})")

        self.in_dim = int(in_dim)
        self.d_token = int(d_token)
        self.n_blocks = int(n_blocks)
        self.n_heads = int(n_heads)
        self.architecture = "ft_transformer"

        cards = list(cat_cardinalities) if cat_cardinalities is not None else []
        self.cat_cardinalities = [int(c) for c in cards]
        ohe_width = int(sum(self.cat_cardinalities))

        if n_num_features is not None:
            self.n_num_features = int(n_num_features)
        elif self.cat_cardinalities:
            self.n_num_features = self.in_dim - ohe_width
        else:
            # Dense continuous-only tokenizer over all columns
            self.n_num_features = self.in_dim

        if self.n_num_features < 0:
            raise ValueError("n_num_features cannot be negative")
        if self.cat_cardinalities and self.n_num_features + ohe_width != self.in_dim:
            raise ValueError(
                f"in_dim={self.in_dim} != n_num ({self.n_num_features}) + ohe ({ohe_width}). "
                "Pass matching n_num_features / cat_cardinalities from DistillationData."
            )

        self.num_tokenizer = (
            NumericalFeatureTokenizer(self.n_num_features, d_token)
            if self.n_num_features > 0
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(self.cat_cardinalities, d_token)
            if self.cat_cardinalities
            else None
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.01)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_token,
                    n_heads,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout,
                    ffn_d_hidden=ffn_d_hidden,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, d_out)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight)

    def _decode_ohe_categories(self, x: torch.Tensor) -> torch.Tensor:
        """Convert OHE block of dense ``x`` to integer category indices (B, n_cat)."""
        idxs = []
        offset = self.n_num_features
        for card in self.cat_cardinalities:
            block = x[:, offset : offset + card]
            idxs.append(block.argmax(dim=1))
            offset += card
        return torch.stack(idxs, dim=1)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.num_tokenizer is not None:
            parts.append(self.num_tokenizer(x[:, : self.n_num_features]))
        if self.cat_tokenizer is not None:
            cat_idx = self._decode_ohe_categories(x)
            parts.append(self.cat_tokenizer(cat_idx))
        if not parts:
            raise RuntimeError("FTTransformer has no feature tokenizers")
        return torch.cat(parts, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """CLS representation after transformer stack + head LayerNorm (pre-head)."""
        tokens = self._tokenize(x)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x_t = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x_t = block(x_t, need_weights=False)
        return self.head_norm(x_t[:, 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, in_dim) dense preprocessed features (num + OHE)

        Returns
        -------
        (batch,) predicted fuel kg
        """
        return self.head(self.encode(x)).squeeze(-1)

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Prediction + per-layer attention weights for analysis.

        Returns
        -------
        pred : (B,) fuel kg — same computation graph residual path as ``forward``
        attns : list of length ``n_blocks``, each (B, n_heads, L, L) softmax weights
                (analytic; for inspection only). Token order: [CLS, num…, cat…].
        """
        tokens = self._tokenize(x)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x_t = torch.cat([cls, tokens], dim=1)
        attns: list[torch.Tensor] = []
        for block in self.blocks:
            x_t, w = block(x_t, need_weights=True)
            attns.append(w)
        pred = self.head(self.head_norm(x_t[:, 0])).squeeze(-1)
        return pred, attns

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return int(sum(p.numel() for p in params))

    def config_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "in_dim": self.in_dim,
            "d_token": self.d_token,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "n_num_features": self.n_num_features,
            "cat_cardinalities": list(self.cat_cardinalities),
            "n_tokens": 1
            + self.n_num_features
            + len(self.cat_cardinalities),  # CLS + features
            "n_params": self.count_parameters(),
        }


def make_ft_transformer_baseline(
    in_dim: int,
    *,
    d_token: int = 192,
    n_blocks: int = 3,
    n_heads: int = 8,
    attention_dropout: float = 0.2,
    ffn_dropout: float = 0.1,
    residual_dropout: float = 0.0,
    n_num_features: int | None = None,
    cat_cardinalities: Sequence[int] | None = None,
) -> FTTransformer:
    return FTTransformer(
        in_dim,
        d_token=d_token,
        n_blocks=n_blocks,
        n_heads=n_heads,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
    )
