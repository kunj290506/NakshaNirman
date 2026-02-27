"""
Encoder modules for plot shape, condition vector, and occupancy grid.

These produce latent embeddings that condition the generator.

Architecture overview:

  PolygonEncoder        Boundary polygon (P,2) → (polygon_feat_dim,)
  ConditionEncoder      Condition vector  (D,) → (cond_dim,)
  OccupancyGridEncoder  Binary grid    (1,H,W) → (polygon_feat_dim,)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Positional encoding (for polygon vertices)
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for sequences of 2-D coordinates."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))           # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Polygon encoder — Transformer over boundary vertices
# ---------------------------------------------------------------------------

class PolygonEncoder(nn.Module):
    """
    Encode an irregular plot boundary polygon into a fixed-length vector.

    Input:  boundary (B, P, 2),  boundary_mask (B, P) bool
    Output: (B, polygon_feat_dim)

    Architecture:
      Linear(2 → D) + SinusoidalPE → 4-layer Transformer encoder → mean pool (masked)
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        cfg = cfg or PipelineConfig()
        d = cfg.polygon_feat_dim  # 128

        self.proj = nn.Linear(2, d)
        self.pe = SinusoidalPE(d, max_len=cfg.max_boundary_pts)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(d)

    def forward(self, boundary: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        boundary : (B, P, 2) float — normalised polygon vertices
        mask : (B, P) bool — True for valid vertices

        Returns
        -------
        (B, D) latent embedding
        """
        x = self.proj(boundary)               # (B, P, D)
        x = self.pe(x)
        # Transformer expects key_padding_mask = True for IGNORED positions
        pad_mask = ~mask.bool()                # (B, P)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()    # (B, P, 1)
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return x                               # (B, D)


# ---------------------------------------------------------------------------
# Condition encoder — small MLP
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    Project the structured condition vector to an embedding.

    Input:  (B, raw_cond_dim)
    Output: (B, cond_dim)
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        cfg = cfg or PipelineConfig()
        d = cfg.cond_dim

        self.net = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


# ---------------------------------------------------------------------------
# Occupancy grid encoder (alternative to polygon encoder)
# ---------------------------------------------------------------------------

class OccupancyGridEncoder(nn.Module):
    """
    Encode a binary boundary occupancy grid into a latent vector.

    Input:  (B, 1, H, W) float  — 1 inside polygon, 0 outside
    Output: (B, polygon_feat_dim)

    Architecture: 5-layer strided ConvNet → global average pool → linear.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        cfg = cfg or PipelineConfig()
        ch = cfg.gen_base_ch
        out_d = cfg.polygon_feat_dim

        self.convs = nn.Sequential(
            # 256 → 128
            nn.Conv2d(1, ch, 4, 2, 1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 → 64
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.InstanceNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 32
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.InstanceNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 → 16
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.InstanceNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 → 8
            nn.Conv2d(ch * 8, ch * 8, 4, 2, 1),
            nn.InstanceNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch * 8, out_d)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.convs(grid)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
