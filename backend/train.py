#!/usr/bin/env python3
"""
train.py — CLI entry point for the ML Floor Plan Training Pipeline.

Usage:
    python train.py --data-dir ./data --epochs 200 --batch-size 16
    python train.py --resume checkpoints/checkpoint_epoch_100.pt
    python train.py --eval-only --checkpoint checkpoints/best_model.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a conditional GAN for floor plan generation"
    )

    # Data
    p.add_argument("--data-dir", type=str, default="./data",
                    help="Root directory containing CubiCasa5K and/or RPLAN datasets")
    p.add_argument("--cubicasa-dir", type=str, default=None,
                    help="CubiCasa5K root (overrides data-dir/cubicasa5k)")
    p.add_argument("--rplan-dir", type=str, default=None,
                    help="RPLAN root (overrides data-dir/rplan)")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-g", type=float, default=2e-4, help="Generator LR")
    p.add_argument("--lr-d", type=float, default=4e-4, help="Discriminator LR")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--d-steps", type=int, default=2,
                    help="Discriminator steps per generator step")
    p.add_argument("--workers", type=int, default=4,
                    help="DataLoader num_workers")
    p.add_argument("--amp", action="store_true", default=True,
                    help="Use mixed precision training")
    p.add_argument("--no-amp", action="store_true",
                    help="Disable mixed precision training")

    # Checkpoint
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--log-dir", type=str, default="./logs")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=10,
                    help="Save checkpoint every N epochs")
    p.add_argument("--eval-every", type=int, default=5,
                    help="Run evaluation every N epochs")

    # Eval only
    p.add_argument("--eval-only", action="store_true",
                    help="Only run evaluation on test set")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint for eval-only mode")

    # Loss weights
    p.add_argument("--lambda-adv", type=float, default=1.0)
    p.add_argument("--lambda-recon", type=float, default=10.0)
    p.add_argument("--lambda-adj", type=float, default=5.0)
    p.add_argument("--lambda-boundary", type=float, default=2.0)
    p.add_argument("--lambda-diversity", type=float, default=1.0)
    p.add_argument("--lambda-feat", type=float, default=2.0)
    p.add_argument("--lambda-structural", type=float, default=3.0)

    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  ML Floor Plan Training Pipeline")
    logger.info("=" * 60)

    # Build config
    from ml_pipeline.config import PipelineConfig

    cfg = PipelineConfig(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        d_steps=args.d_steps,
        use_amp=args.amp and not args.no_amp,
        lambda_adv=args.lambda_adv,
        lambda_recon=args.lambda_recon,
        lambda_adj=args.lambda_adj,
        lambda_boundary=args.lambda_boundary,
        lambda_diversity=args.lambda_diversity,
        lambda_structural=args.lambda_structural,
    )

    logger.info("Config: %s", cfg)

    # Override directories
    import ml_pipeline.config as mlcfg
    if args.data_dir:
        mlcfg.DATA_DIR = Path(args.data_dir)
    if args.checkpoint_dir:
        mlcfg.CHECKPOINT_DIR = Path(args.checkpoint_dir)
    if args.log_dir:
        mlcfg.LOG_DIR = Path(args.log_dir)

    # Create directories
    mlcfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    mlcfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Check torch
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("PyTorch %s | Device: %s", torch.__version__, device)
        if device == "cuda":
            logger.info("GPU: %s | VRAM: %.1f GB",
                         torch.cuda.get_device_name(0),
                         torch.cuda.get_device_properties(0).total_mem / 1e9)
    except ImportError:
        logger.error("PyTorch not available. Install with: pip install torch")
        sys.exit(1)

    # ---- Eval-only mode ----
    if args.eval_only:
        _run_eval(args, cfg, device)
        return

    # ---- Build datasets ----
    logger.info("Building datasets...")
    from ml_pipeline.data.combined import build_dataloaders

    cubicasa = args.cubicasa_dir or str(mlcfg.DATA_DIR / "cubicasa5k")
    rplan = args.rplan_dir or str(mlcfg.DATA_DIR / "rplan")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        cubicasa_root=cubicasa,
        rplan_root=rplan,
        num_workers=args.workers,
    )

    logger.info("Train: %d batches | Val: %d batches | Test: %d batches",
                len(train_loader), len(val_loader), len(test_loader))

    # ---- Build models ----
    logger.info("Building models...")
    from ml_pipeline.models.generator import FloorPlanGenerator
    from ml_pipeline.models.discriminator import FloorPlanDiscriminator

    gen = FloorPlanGenerator(cfg).to(device)
    disc = FloorPlanDiscriminator(cfg).to(device)

    g_params = sum(p.numel() for p in gen.parameters())
    d_params = sum(p.numel() for p in disc.parameters())
    logger.info("Generator:     %.2f M params", g_params / 1e6)
    logger.info("Discriminator: %.2f M params", d_params / 1e6)

    # ---- Build trainer ----
    from ml_pipeline.training.trainer import FloorPlanTrainer

    trainer = FloorPlanTrainer(cfg, gen, disc, device=device)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info("Resumed from %s", args.resume)

    # ---- Train ----
    logger.info("Starting training for %d epochs...", cfg.epochs)
    trainer.fit(train_loader, val_loader)

    # ---- Final evaluation ----
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer._evaluate(test_loader)
    logger.info("Test metrics: %s", test_metrics)

    logger.info("Training complete!")


def _run_eval(args, cfg, device):
    """Run evaluation only mode."""
    import torch
    from ml_pipeline.models.generator import FloorPlanGenerator
    from ml_pipeline.models.discriminator import FloorPlanDiscriminator
    from ml_pipeline.training.trainer import FloorPlanTrainer
    from ml_pipeline.data.combined import build_dataloaders
    import ml_pipeline.config as mlcfg

    ckpt = args.checkpoint or args.resume
    if not ckpt:
        logger.error("--checkpoint or --resume required for eval-only mode")
        sys.exit(1)

    gen = FloorPlanGenerator(cfg).to(device)
    disc = FloorPlanDiscriminator(cfg).to(device)
    trainer = FloorPlanTrainer(cfg, gen, disc, device=device)
    trainer.load_checkpoint(ckpt)

    cubicasa = args.cubicasa_dir or str(mlcfg.DATA_DIR / "cubicasa5k")
    rplan = args.rplan_dir or str(mlcfg.DATA_DIR / "rplan")

    _, _, test_loader = build_dataloaders(
        cfg, cubicasa_root=cubicasa, rplan_root=rplan,
        num_workers=args.workers,
    )

    metrics = trainer._evaluate(test_loader)
    logger.info("=" * 40)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 40)
    for k, v in metrics.items():
        logger.info("  %-25s  %.4f", k, v)


if __name__ == "__main__":
    main()
