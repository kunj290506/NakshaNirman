"""
Combined dataset & PyTorch DataLoader factory.

Merges CubiCasa5K and RPLAN into a single training stream, handles
collation of variable-length samples, and provides convenience builders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH = True
except ImportError:
    TORCH = False
    Dataset = object

from ml_pipeline.config import PipelineConfig, DATA_DIR
from ml_pipeline.data.cubicasa import CubiCasa5KDataset
from ml_pipeline.data.rplan import RPLANDataset
from ml_pipeline.data.preprocessing import FloorPlanSample

logger = logging.getLogger(__name__)


class CombinedFloorPlanDataset(Dataset):
    """
    PyTorch Dataset combining CubiCasa5K and RPLAN.

    Falls back gracefully if only one source exists.
    """

    def __init__(
        self,
        split: str = "train",
        cfg: Optional[PipelineConfig] = None,
        cubicasa_root: Optional[str | Path] = None,
        rplan_root: Optional[str | Path] = None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.split = split
        self.sources: List = []

        # Try CubiCasa5K
        croot = Path(cubicasa_root) if cubicasa_root else DATA_DIR / "cubicasa5k"
        if croot.exists():
            ds = CubiCasa5KDataset(croot, split, self.cfg, augment=(split == "train"))
            if len(ds) > 0:
                self.sources.append(ds)
                logger.info("CubiCasa5K [%s]: %d samples", split, len(ds))

        # Try RPLAN
        rroot = Path(rplan_root) if rplan_root else DATA_DIR / "rplan"
        if rroot.exists():
            ds = RPLANDataset(rroot, split, self.cfg, augment=(split == "train"))
            if len(ds) > 0:
                self.sources.append(ds)
                logger.info("RPLAN [%s]: %d samples", split, len(ds))

        # Pre-compute cumulative lengths
        self._cumlen: List[int] = []
        total = 0
        for src in self.sources:
            total += len(src)
            self._cumlen.append(total)

        if total == 0:
            logger.warning(
                "No dataset samples found.  "
                "Place CubiCasa5K in %s and/or RPLAN in %s.",
                croot, rroot,
            )

    def __len__(self) -> int:
        return self._cumlen[-1] if self._cumlen else 0

    def __getitem__(self, idx: int) -> Optional[Dict]:
        # Locate which source
        for si, cl in enumerate(self._cumlen):
            if idx < cl:
                src = self.sources[si]
                local_idx = idx - (self._cumlen[si - 1] if si > 0 else 0)
                sample = src[local_idx]
                if sample is None:
                    return None
                return self._to_tensors(sample)
        return None

    @staticmethod
    def _to_tensors(s: FloorPlanSample) -> Dict[str, "torch.Tensor"]:
        """Convert numpy FloorPlanSample → dict of tensors for collation."""
        if not TORCH:
            raise RuntimeError("PyTorch required for tensor conversion")
        return {
            "boundary":      torch.from_numpy(s.boundary),           # (P, 2)
            "boundary_mask": torch.from_numpy(s.boundary_mask),      # (P,)
            "room_mask":     torch.from_numpy(s.room_mask).long(),   # (H, W)
            "room_boxes":    torch.from_numpy(s.room_boxes),         # (K, 5)
            "adjacency":     torch.from_numpy(s.adjacency.astype(np.float32)),  # (C, C)
            "condition":     torch.from_numpy(s.condition),          # (D,)
            "entry_side":    torch.tensor(s.entry_side, dtype=torch.long),
            "north_dir":     torch.tensor(s.north_dir, dtype=torch.float32),
            "budget_level":  torch.tensor(s.budget_level, dtype=torch.long),
        }


def _collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, "torch.Tensor"]]:
    """Custom collator that skips None samples."""
    if not TORCH:
        return None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_dataloaders(
    cfg: Optional[PipelineConfig] = None,
    cubicasa_root: Optional[str] = None,
    rplan_root: Optional[str] = None,
) -> Tuple[Optional["DataLoader"], Optional["DataLoader"], Optional["DataLoader"]]:
    """
    Build train / val / test DataLoaders.

    Returns (train_loader, val_loader, test_loader).
    Any may be None if dataset is not found.
    """
    if not TORCH:
        logger.error("PyTorch not installed — cannot build DataLoaders.")
        return None, None, None

    cfg = cfg or PipelineConfig()

    loaders = []
    for split in ("train", "val", "test"):
        ds = CombinedFloorPlanDataset(
            split=split, cfg=cfg,
            cubicasa_root=cubicasa_root, rplan_root=rplan_root,
        )
        if len(ds) == 0:
            loaders.append(None)
            continue

        loaders.append(DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=0,            # safe default on Windows
            collate_fn=_collate_fn,
            drop_last=(split == "train"),
            pin_memory=torch.cuda.is_available(),
        ))

    return tuple(loaders)  # type: ignore
