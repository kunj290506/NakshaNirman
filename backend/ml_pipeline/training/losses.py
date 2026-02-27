"""
Loss functions for conditional floor plan generation.

Loss components
───────────────
1.  Adversarial         — WGAN-GP hinge loss (disc) + generator fooling
2.  Mask reconstruction — cross-entropy between predicted & GT room mask
3.  Adjacency           — BCE between predicted & GT adjacency matrix
4.  Boundary fitting    — L2 penalty for rooms outside plot boundary
5.  Diversity           — encourage different outputs for different z
6.  Structural          — combined architectural constraint penalties
7.  Feature matching    — L1 between disc features of real & fake

Total G loss = λ_adv·L_adv + λ_recon·L_recon + λ_adj·L_adj
             + λ_boundary·L_boundary + λ_diversity·L_diversity
             + λ_structural·L_structural
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.config import PipelineConfig, NUM_ROOM_CLASSES


class FloorPlanLoss(nn.Module):
    """
    Complete loss computation for generator and discriminator.

    Usage:
        loss_fn = FloorPlanLoss(cfg)
        g_losses = loss_fn.generator_loss(gen_out, real_batch, disc, constraint_net)
        d_losses = loss_fn.discriminator_loss(gen_out, real_batch, disc)
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        self.cfg = cfg or PipelineConfig()

    # =====================================================================
    # Generator losses
    # =====================================================================

    def generator_loss(
        self,
        gen_out: Dict[str, torch.Tensor],
        real_batch: Dict[str, torch.Tensor],
        disc: nn.Module,
        constraint_net: Optional[nn.Module] = None,
        boundary_grid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all generator loss terms.

        Parameters
        ----------
        gen_out       : dict from generator forward() — mask_logits, adjacency, f_cond
        real_batch    : dict from dataloader — room_mask, adjacency, condition, ...
        disc          : discriminator module
        constraint_net: architectural constraint network (optional)
        boundary_grid : (B, 1, H, W) plot occupancy grid

        Returns
        -------
        dict of loss name → scalar tensor.  "total" key is the weighted sum.
        """
        cfg = self.cfg
        losses = {}

        mask_logits = gen_out["mask_logits"]        # (B, C, H, W)
        adj_pred    = gen_out["adjacency"]           # (B, C, C)
        f_cond      = gen_out["f_cond"]              # (B, D)

        real_mask = real_batch["room_mask"]           # (B, H, W) long
        real_adj  = real_batch["adjacency"]           # (B, C, C) float

        # --- 1. Adversarial (generator wants disc to say "real") ---
        fake_mask_soft = F.softmax(mask_logits, dim=1)          # (B, C, H, W)
        fake_score, fake_features = disc(fake_mask_soft, real_batch["condition"])
        losses["adv"] = -fake_score.mean()

        # --- 2. Mask reconstruction (cross-entropy) ---
        losses["recon"] = F.cross_entropy(mask_logits, real_mask)

        # --- 3. Adjacency correctness (BCE) ---
        losses["adj"] = F.binary_cross_entropy(
            adj_pred, real_adj, reduction="mean"
        )

        # --- 4. Boundary fitting ---
        if boundary_grid is not None:
            room_prob = 1.0 - fake_mask_soft[:, -1:]  # everything except exterior
            outside = (1.0 - boundary_grid)
            losses["boundary"] = (room_prob * outside).pow(2).mean()
        else:
            losses["boundary"] = torch.tensor(0.0, device=mask_logits.device)

        # --- 5. Diversity regulariser ---
        #     If two different z produce similar outputs for same input → penalise
        losses["diversity"] = self._diversity_loss(mask_logits)

        # --- 6. Feature matching ---
        with torch.no_grad():
            real_mask_oh = F.one_hot(real_mask, NUM_ROOM_CLASSES).permute(0, 3, 1, 2).float()
            _, real_features = disc(real_mask_oh, real_batch["condition"])
        fm_loss = torch.tensor(0.0, device=mask_logits.device)
        for rf, ff in zip(real_features, fake_features):
            fm_loss = fm_loss + F.l1_loss(ff, rf.detach())
        losses["feature_match"] = fm_loss / max(len(real_features), 1)

        # --- 7. Structural / architectural constraints ---
        if constraint_net is not None and boundary_grid is not None:
            c_losses = constraint_net(mask_logits, boundary_grid, adj_pred)
            structural = torch.tensor(0.0, device=mask_logits.device)
            for v in c_losses.values():
                structural = structural + v
            losses["structural"] = structural
        else:
            losses["structural"] = torch.tensor(0.0, device=mask_logits.device)

        # --- Weighted total ---
        total = (
            cfg.lambda_adv       * losses["adv"]
            + cfg.lambda_recon   * losses["recon"]
            + cfg.lambda_adj     * losses["adj"]
            + cfg.lambda_boundary* losses["boundary"]
            + cfg.lambda_diversity * losses["diversity"]
            + cfg.lambda_structural * losses["structural"]
            + 2.0               * losses["feature_match"]
        )
        losses["total"] = total
        return losses

    # =====================================================================
    # Discriminator losses (WGAN-GP / hinge)
    # =====================================================================

    def discriminator_loss(
        self,
        gen_out: Dict[str, torch.Tensor],
        real_batch: Dict[str, torch.Tensor],
        disc: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss with gradient penalty.

        Returns
        -------
        dict with "real", "fake", "gp", "total".
        """
        cfg = self.cfg
        mask_logits = gen_out["mask_logits"].detach()
        fake_mask_soft = F.softmax(mask_logits, dim=1)

        real_mask = real_batch["room_mask"]
        real_mask_oh = F.one_hot(real_mask, NUM_ROOM_CLASSES).permute(0, 3, 1, 2).float()
        cond = real_batch["condition"]

        # Real
        real_score, _ = disc(real_mask_oh, cond)
        # Fake
        fake_score, _ = disc(fake_mask_soft, cond)

        # Hinge loss
        d_loss_real = F.relu(1.0 - real_score).mean()
        d_loss_fake = F.relu(1.0 + fake_score).mean()

        # Gradient penalty
        gp = disc.gradient_penalty(real_mask_oh, fake_mask_soft, cond, cfg.lambda_gp)

        total = d_loss_real + d_loss_fake + gp

        return {
            "real": d_loss_real,
            "fake": d_loss_fake,
            "gp": gp,
            "total": total,
        }

    # =====================================================================
    # Diversity loss
    # =====================================================================

    @staticmethod
    def _diversity_loss(mask_logits: torch.Tensor) -> torch.Tensor:
        """
        Encourage diversity: if batch > 1, variance across samples should be high.

        Penalise low variance in the softmax output across the batch.
        """
        if mask_logits.size(0) < 2:
            return torch.tensor(0.0, device=mask_logits.device)

        soft = F.softmax(mask_logits, dim=1)          # (B, C, H, W)
        # Per-pixel variance across batch
        var = soft.var(dim=0).mean()                   # scalar
        # We want HIGH variance → penalise low variance
        return 1.0 / (var + 1e-6)
