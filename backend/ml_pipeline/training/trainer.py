"""
Training loop with EMA, gradient clipping, mixed-precision,
checkpoint save/resume, and periodic evaluation.

Usage:
    from ml_pipeline.training import FloorPlanTrainer
    trainer = FloorPlanTrainer(cfg)
    trainer.fit()
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    TORCH = True
except ImportError:
    TORCH = False

from ml_pipeline.config import PipelineConfig, CHECKPOINT_DIR, LOG_DIR, NUM_ROOM_CLASSES
from ml_pipeline.data.combined import build_dataloaders
from ml_pipeline.data.preprocessing import polygon_to_occupancy_grid
from ml_pipeline.models.generator import FloorPlanGenerator
from ml_pipeline.models.discriminator import FloorPlanDiscriminator
from ml_pipeline.models.constraints import ArchitecturalConstraintNet
from ml_pipeline.training.losses import FloorPlanLoss

logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = sd


class FloorPlanTrainer:
    """
    Full training loop for the conditional floor-plan GAN.

    Parameters
    ----------
    cfg : PipelineConfig — all hyper-parameters
    cubicasa_root, rplan_root : override dataset locations
    resume : path to a checkpoint to resume from
    """

    def __init__(
        self,
        cfg: Optional[PipelineConfig] = None,
        cubicasa_root: Optional[str] = None,
        rplan_root: Optional[str] = None,
        resume: Optional[str] = None,
    ):
        if not TORCH:
            raise RuntimeError("PyTorch is required for training")

        self.cfg = cfg or PipelineConfig()
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() and "cuda" in self.cfg.device
            else "cpu"
        )

        # --- Data ---
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            self.cfg, cubicasa_root, rplan_root,
        )

        # --- Models ---
        self.gen = FloorPlanGenerator(self.cfg).to(self.device)
        self.disc = FloorPlanDiscriminator(self.cfg).to(self.device)
        self.constraint = ArchitecturalConstraintNet(self.cfg).to(self.device)
        self.loss_fn = FloorPlanLoss(self.cfg)

        # --- Optimisers ---
        self.opt_g = torch.optim.Adam(
            list(self.gen.parameters()) + list(self.constraint.parameters()),
            lr=self.cfg.lr_g,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.disc.parameters(),
            lr=self.cfg.lr_d,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )

        # --- Schedulers ---
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=self.cfg.epochs, eta_min=1e-6,
        )
        self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=self.cfg.epochs, eta_min=1e-6,
        )

        # --- EMA ---
        self.ema = EMAModel(self.gen, self.cfg.ema_decay)

        # --- Mixed precision ---
        self.scaler_g = GradScaler(enabled=(self.device.type == "cuda"))
        self.scaler_d = GradScaler(enabled=(self.device.type == "cuda"))

        # --- Logging ---
        self.epoch = 0
        self.global_step = 0
        self.history: List[Dict] = []

        # --- Resume ---
        if resume:
            self._load_checkpoint(resume)

        # Log model sizes
        g_params = sum(p.numel() for p in self.gen.parameters())
        d_params = sum(p.numel() for p in self.disc.parameters())
        c_params = sum(p.numel() for p in self.constraint.parameters())
        logger.info(
            "Models created — G: %.2f M, D: %.2f M, C: %.2f M params  [%s]",
            g_params / 1e6, d_params / 1e6, c_params / 1e6, self.device,
        )

    # ==================================================================
    # Main training loop
    # ==================================================================

    def fit(self):
        """Run the full training loop."""
        cfg = self.cfg

        if self.train_loader is None:
            logger.error("No training data.  Place datasets in ml_pipeline/datasets/")
            return

        logger.info("Starting training for %d epochs  (batch=%d)", cfg.epochs, cfg.batch_size)

        for epoch in range(self.epoch, cfg.epochs):
            self.epoch = epoch
            t0 = time.time()

            metrics = self._train_one_epoch()

            self.sched_g.step()
            self.sched_d.step()

            elapsed = time.time() - t0
            metrics["epoch"] = epoch
            metrics["time_s"] = elapsed
            metrics["lr_g"] = self.opt_g.param_groups[0]["lr"]
            self.history.append(metrics)

            # Log
            logger.info(
                "Epoch %3d/%d  G=%.4f  D=%.4f  recon=%.4f  adj=%.4f  "
                "struct=%.4f  [%.1fs]",
                epoch + 1, cfg.epochs,
                metrics.get("g_total", 0),
                metrics.get("d_total", 0),
                metrics.get("g_recon", 0),
                metrics.get("g_adj", 0),
                metrics.get("g_structural", 0),
                elapsed,
            )

            # Evaluate
            if (epoch + 1) % cfg.eval_every == 0 and self.val_loader is not None:
                val_metrics = self._evaluate()
                metrics.update(val_metrics)
                logger.info(
                    "  Val — IoU=%.3f  AdjAcc=%.3f  Diversity=%.3f  Violations=%.3f",
                    val_metrics.get("val_iou", 0),
                    val_metrics.get("val_adj_acc", 0),
                    val_metrics.get("val_diversity", 0),
                    val_metrics.get("val_violations", 0),
                )

            # Checkpoint
            if (epoch + 1) % cfg.save_every == 0:
                self._save_checkpoint(epoch)

        # Final save
        self._save_checkpoint(cfg.epochs - 1, tag="final")
        self._save_history()
        logger.info("Training complete.")

    # ==================================================================
    # One epoch
    # ==================================================================

    def _train_one_epoch(self) -> Dict[str, float]:
        self.gen.train()
        self.disc.train()
        self.constraint.train()

        accum = {}
        count = 0

        for batch in self.train_loader:
            if batch is None:
                continue

            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["boundary"].size(0)

            # Build boundary occupancy grid for constraint net
            boundary_grid = self._batch_boundary_grid(batch)

            # --- Discriminator step ---
            self.opt_d.zero_grad()
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                gen_out = self.gen(
                    batch["boundary"], batch["boundary_mask"],
                    batch["condition"],
                )
                d_losses = self.loss_fn.discriminator_loss(gen_out, batch, self.disc)

            self.scaler_d.scale(d_losses["total"]).backward()
            self.scaler_d.unscale_(self.opt_d)
            nn.utils.clip_grad_norm_(self.disc.parameters(), self.cfg.grad_clip)
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

            # --- Generator step ---
            self.opt_g.zero_grad()
            with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                gen_out = self.gen(
                    batch["boundary"], batch["boundary_mask"],
                    batch["condition"],
                )
                g_losses = self.loss_fn.generator_loss(
                    gen_out, batch, self.disc,
                    self.constraint, boundary_grid,
                )

            self.scaler_g.scale(g_losses["total"]).backward()
            self.scaler_g.unscale_(self.opt_g)
            nn.utils.clip_grad_norm_(self.gen.parameters(), self.cfg.grad_clip)
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()

            # EMA update
            self.ema.update(self.gen)
            self.global_step += 1

            # Accumulate metrics
            for k, v in g_losses.items():
                key = f"g_{k}"
                accum[key] = accum.get(key, 0) + v.item()
            for k, v in d_losses.items():
                key = f"d_{k}"
                accum[key] = accum.get(key, 0) + v.item()
            count += 1

        # Average
        if count > 0:
            accum = {k: v / count for k, v in accum.items()}
        return accum

    # ==================================================================
    # Evaluation
    # ==================================================================

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        from ml_pipeline.evaluation.metrics import compute_metrics

        self.gen.eval()
        all_pred_masks = []
        all_gt_masks = []
        all_pred_adj = []
        all_gt_adj = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if batch is None or i >= self.cfg.num_eval_samples:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                gen_out = self.gen(
                    batch["boundary"], batch["boundary_mask"],
                    batch["condition"],
                )
                pred_mask = gen_out["mask_logits"].argmax(dim=1)  # (B, H, W)
                all_pred_masks.append(pred_mask.cpu().numpy())
                all_gt_masks.append(batch["room_mask"].cpu().numpy())
                all_pred_adj.append(gen_out["adjacency"].cpu().numpy())
                all_gt_adj.append(batch["adjacency"].cpu().numpy())

        if not all_pred_masks:
            return {}

        pred_masks = np.concatenate(all_pred_masks, axis=0)
        gt_masks = np.concatenate(all_gt_masks, axis=0)
        pred_adj = np.concatenate(all_pred_adj, axis=0)
        gt_adj = np.concatenate(all_gt_adj, axis=0)

        metrics = compute_metrics(pred_masks, gt_masks, pred_adj, gt_adj)
        return {f"val_{k}": v for k, v in metrics.items()}

    # ==================================================================
    # Boundary grid construction
    # ==================================================================

    def _batch_boundary_grid(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build (B, 1, H, W) occupancy grid from batch boundaries."""
        B = batch["boundary"].size(0)
        size = self.cfg.img_size
        grids = []
        for i in range(B):
            coords = batch["boundary"][i].cpu().numpy()
            mask = batch["boundary_mask"][i].cpu().numpy()
            valid = coords[mask]
            if len(valid) < 3:
                grids.append(np.ones((size, size), dtype=np.float32))
            else:
                grids.append(polygon_to_occupancy_grid(valid, size))
        grid_np = np.stack(grids)[:, np.newaxis]  # (B, 1, H, W)
        return torch.from_numpy(grid_np).to(self.device)

    # ==================================================================
    # Checkpointing
    # ==================================================================

    def _save_checkpoint(self, epoch: int, tag: str = ""):
        fname = f"checkpoint_epoch{epoch + 1}"
        if tag:
            fname += f"_{tag}"
        fname += ".pt"
        path = CHECKPOINT_DIR / fname

        torch.save({
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "gen": self.gen.state_dict(),
            "disc": self.disc.state_dict(),
            "constraint": self.constraint.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "ema": self.ema.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)
        logger.info("Checkpoint saved → %s", path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.gen.load_state_dict(ckpt["gen"])
        self.disc.load_state_dict(ckpt["disc"])
        self.constraint.load_state_dict(ckpt["constraint"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        self.ema.load_state_dict(ckpt["ema"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        logger.info("Resumed from checkpoint %s (epoch %d)", path, self.epoch)

    def _save_history(self):
        path = LOG_DIR / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info("Training history → %s", path)
