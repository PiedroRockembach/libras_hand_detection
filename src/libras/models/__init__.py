"""Models for isolated signs (ADR-0014): a per-frame temporal encoder, temporal pooling to an
embedding, and a cosine classifier whose weights can be imprinted from class prototypes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CosineClassifier(nn.Module):
    """logits = scale * cos(embedding, class weight). A new class can be added without training
    by setting its weight to the mean embedding of a few examples (weight imprinting)."""

    def __init__(self, dim: int, n_classes: int, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.scale = scale

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * F.normalize(emb, dim=-1) @ F.normalize(self.weight, dim=-1).T

    @torch.no_grad()
    def imprint(self, prototypes: torch.Tensor) -> None:
        self.weight.copy_(F.normalize(prototypes, dim=-1))


class StatsEncoder(nn.Module):
    """Sanity baseline: per-feature mean, std, min and max over time -> MLP. Ignores order."""

    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * n_features), nn.Linear(4 * n_features, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.dim = hidden

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        stats = torch.cat([x.mean(1), x.std(1), x.amin(1), x.amax(1)], dim=-1)
        return self.mlp(stats), None


class GRUEncoder(nn.Module):
    """Per-frame projection -> GRU. Returns the pooled embedding and the per-frame outputs
    (the latter feed the CTC head of phase C)."""

    def __init__(self, n_features: int, hidden: int, layers: int, bidirectional: bool,
                 dropout: float, embedding: int):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, hidden),
                                  nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out), nn.Dropout(dropout), nn.Linear(out, embedding))
        self.dim = embedding

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames, _ = self.gru(self.proj(x))
        return self.head(frames.mean(1)), frames


class SignModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int, scale: float):
        super().__init__()
        self.encoder = encoder
        self.classifier = CosineClassifier(encoder.dim, n_classes, scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb, _ = self.encoder(x)
        return self.classifier(emb), emb


def build_model(cfg: dict, n_features: int, n_classes: int) -> SignModel:
    kind = cfg["encoder"]
    if kind == "stats_mlp":
        enc = StatsEncoder(n_features, cfg["hidden"], cfg["dropout"])
    elif kind == "gru":
        enc = GRUEncoder(n_features, cfg["hidden"], cfg["layers"], cfg["bidirectional"],
                         cfg["dropout"], cfg["embedding"])
    else:
        raise ValueError(f"unknown encoder {kind!r}")
    return SignModel(enc, n_classes, cfg["cosine_scale"])
