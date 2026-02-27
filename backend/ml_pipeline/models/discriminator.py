"""
Conditional discriminator — judges real vs. generated floor plan masks.

Architecture:
  Input:  room mask (B, C, H, W) + condition embedding (B, D) → real/fake score.

  Condition embedding is spatially broadcast and concatenated to the mask.
  Five strided-conv layers downsample 256→128→64→32→16→8.
  PatchGAN-style output → (B, 1, 8, 8)  or  scalar via global pool.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.config import PipelineConfig, NUM_ROOM_CLASSES


class SpectralNormConv(nn.Module):
    """Conv2d + SpectralNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 4, s: int = 2, p: int = 1):
        super().__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, k, s, p))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class FloorPlanDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation.

    Input channels = NUM_ROOM_CLASSES + cond_dim (broadcast spatially).

    Parameters
    ----------
    cfg : PipelineConfig

    Forward
    -------
    mask       : (B, C, H, W)  one-hot or soft room mask
    condition  : (B, D)         condition vector

    Returns
    -------
    score      : (B, 1)         real/fake logit
    features   : list of intermediate feature maps (for feature matching loss)
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        cfg = cfg or PipelineConfig()
        C = NUM_ROOM_CLASSES
        D = cfg.cond_dim
        ch = cfg.disc_base_ch  # 64

        in_ch = C + D  # room mask channels + spatially broadcast condition

        self.blocks = nn.ModuleList([
            # 256 → 128
            SpectralNormConv(in_ch, ch, 4, 2, 1),
            # 128 → 64
            SpectralNormConv(ch, ch * 2, 4, 2, 1),
            # 64 → 32
            SpectralNormConv(ch * 2, ch * 4, 4, 2, 1),
            # 32 → 16
            SpectralNormConv(ch * 4, ch * 8, 4, 2, 1),
            # 16 → 8
            SpectralNormConv(ch * 8, ch * 8, 4, 2, 1),
        ])

        # PatchGAN head → single scalar
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ch * 8, 1)),
        )

    def forward(
        self,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ):
        B, C, H, W = mask.shape
        D = condition.size(1)

        # Broadcast condition spatially and concat
        cond_map = condition.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)
        x = torch.cat([mask, cond_map], dim=1)  # (B, C+D, H, W)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        score = self.head(x)  # (B, 1)
        return score, features

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        condition: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """WGAN-GP gradient penalty."""
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        score, _ = self.forward(interp, condition)
        grad = torch.autograd.grad(
            outputs=score,
            inputs=interp,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad = grad.view(B, -1)
        penalty = lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty
