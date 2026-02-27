"""
Conditional Floor Plan Generator.

Encoder-Decoder architecture:
  Encoder:  plot shape embedding + condition embedding + noise z  →  bottleneck
  Decoder:  bottleneck  →  room-layout mask  (B, C, H, W)

The generator outputs a per-pixel class logit map (room mask) that is
supervised with cross-entropy (reconstruction) and adversarial loss.

Also emits an adjacency prediction head for graph-based supervision.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.config import PipelineConfig, NUM_ROOM_CLASSES
from ml_pipeline.models.encoders import (
    PolygonEncoder,
    ConditionEncoder,
    OccupancyGridEncoder,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with InstanceNorm + LeakyReLU."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Instance Norm — modulates features with condition vector."""

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gain = nn.Linear(cond_dim, num_features)
        self.bias = nn.Linear(cond_dim, num_features)
        nn.init.ones_(self.gain.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        g = self.gain(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        b = self.bias(cond).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + g) + b


class CondResBlock(nn.Module):
    """Residual block with conditional normalisation."""

    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.cn1 = ConditionalBatchNorm2d(ch, cond_dim)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.cn2 = ConditionalBatchNorm2d(ch, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.cn1(self.conv1(x), cond), 0.2)
        h = self.cn2(self.conv2(h), cond)
        return x + h


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FloorPlanGenerator(nn.Module):
    """
    Conditional floor-plan generator.

    Inputs
    ------
    boundary      : (B, P, 2)   plot polygon vertices
    boundary_mask : (B, P)      valid-vertex mask
    condition     : (B, D)      structured condition vector
    z             : (B, Z)      noise vector (optional; sampled if None)

    Outputs
    -------
    mask_logits   : (B, C, H, W)  per-pixel room-class logits
    adjacency_pred: (B, C, C)     predicted adjacency matrix (sigmoid)

    Architecture
    ────────────
    1.  Encode plot shape  → (B, F_poly)
    2.  Encode condition   → (B, F_cond)
    3.  Concat [poly; cond; z] → Linear → spatial reshape → (B, ch*8, 8, 8)
    4.  Decode via 5 up-conv stages  8→16→32→64→128→256  with conditional ResBlocks
    5.  Final 1×1 conv → (B, C, 256, 256)
    6.  Side head: global-pool decoder features → MLP → (B, C, C) adjacency
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        cfg = cfg or PipelineConfig()
        self.cfg = cfg
        ch = cfg.gen_base_ch          # 64
        C = NUM_ROOM_CLASSES
        Z = cfg.latent_dim            # 128
        F_poly = cfg.polygon_feat_dim # 128
        F_cond = cfg.cond_dim         # 64
        self.Z = Z

        # --- Encoders ---
        if cfg.plot_encoder == "polygon":
            self.plot_enc = PolygonEncoder(cfg)
        else:
            self.plot_enc = OccupancyGridEncoder(cfg)
        self.cond_enc = ConditionEncoder(cfg)

        # --- Bottleneck projection ---
        total_in = F_poly + F_cond + Z
        self.cond_total_dim = F_cond          # for CondResBlocks
        self.fc = nn.Sequential(
            nn.Linear(total_in, ch * 8 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Decoder (up-conv) ---
        # 8 → 16
        self.up1 = nn.ConvTranspose2d(ch * 8, ch * 8, 4, 2, 1)
        self.cr1 = CondResBlock(ch * 8, F_cond)
        # 16 → 32
        self.up2 = nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1)
        self.cr2 = CondResBlock(ch * 4, F_cond)
        # 32 → 64
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1)
        self.cr3 = CondResBlock(ch * 2, F_cond)
        # 64 → 128
        self.up4 = nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1)
        self.cr4 = CondResBlock(ch, F_cond)
        # 128 → 256
        self.up5 = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
        self.cr5 = CondResBlock(ch, F_cond)

        # --- Mask head ---
        self.mask_head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, C, 1),
        )

        # --- Adjacency head ---
        self.adj_pool = nn.AdaptiveAvgPool2d(1)
        self.adj_head = nn.Sequential(
            nn.Linear(ch, cfg.graph_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.graph_hidden, C * C),
        )
        self._C = C

    def forward(
        self,
        boundary: torch.Tensor,
        boundary_mask: torch.Tensor,
        condition: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        occ_grid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = boundary.size(0)
        device = boundary.device

        # --- Encode plot ---
        if isinstance(self.plot_enc, OccupancyGridEncoder):
            assert occ_grid is not None, "OccupancyGridEncoder needs occ_grid"
            f_poly = self.plot_enc(occ_grid)
        else:
            f_poly = self.plot_enc(boundary, boundary_mask)

        # --- Encode condition ---
        f_cond = self.cond_enc(condition)

        # --- Noise ---
        if z is None:
            z = torch.randn(B, self.Z, device=device)

        # --- Bottleneck ---
        h = torch.cat([f_poly, f_cond, z], dim=1)
        h = self.fc(h)
        h = h.view(B, self.cfg.gen_base_ch * 8, 8, 8)

        # --- Decode ---
        h = F.leaky_relu(self.up1(h), 0.2)
        h = self.cr1(h, f_cond)
        h = F.leaky_relu(self.up2(h), 0.2)
        h = self.cr2(h, f_cond)
        h = F.leaky_relu(self.up3(h), 0.2)
        h = self.cr3(h, f_cond)
        h = F.leaky_relu(self.up4(h), 0.2)
        h = self.cr4(h, f_cond)
        h = F.leaky_relu(self.up5(h), 0.2)
        h = self.cr5(h, f_cond)

        # --- Mask logits ---
        mask_logits = self.mask_head(h)      # (B, C, 256, 256)

        # --- Adjacency prediction ---
        pooled = self.adj_pool(h).view(B, -1)
        adj_flat = self.adj_head(pooled)     # (B, C*C)
        adj_pred = torch.sigmoid(adj_flat.view(B, self._C, self._C))

        return {
            "mask_logits": mask_logits,
            "adjacency": adj_pred,
            "f_cond": f_cond,
        }

    def sample(
        self,
        boundary: torch.Tensor,
        boundary_mask: torch.Tensor,
        condition: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0,
        occ_grid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate multiple diverse samples from the same inputs."""
        B = boundary.size(0)
        assert B == 1, "sample() expects B=1; tile externally for batches"

        all_masks = []
        all_adj = []
        for _ in range(n_samples):
            z = torch.randn(1, self.Z, device=boundary.device) * temperature
            out = self.forward(boundary, boundary_mask, condition, z, occ_grid)
            all_masks.append(out["mask_logits"])
            all_adj.append(out["adjacency"])

        return {
            "mask_logits": torch.cat(all_masks, dim=0),
            "adjacency": torch.cat(all_adj, dim=0),
        }
