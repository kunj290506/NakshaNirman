"""
RPLAN dataset loader.

RPLAN provides ~80 K floor plan images with per-pixel room labels stored
as PNG index maps.  Each image is a 256×256 single-channel PNG where
pixel value encodes room type.

Expected layout after download:
  rplan/
    0.png            — floor plan index image
    1.png
    ...
    floorplan_list.txt (optional — train/val/test split)

Reference: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml_pipeline.config import (
    PipelineConfig,
    ROOM_TO_IDX,
    LABEL_ALIASES,
    NUM_ROOM_CLASSES,
)
from ml_pipeline.data.preprocessing import (
    FloorPlanSample,
    normalise_polygon,
    extract_room_masks,
    rooms_to_boxes,
    build_adjacency_matrix,
    encode_condition,
    augment_sample,
)

logger = logging.getLogger(__name__)

# RPLAN pixel-value → room label (from the dataset documentation)
RPLAN_INDEX_MAP: Dict[int, str] = {
    0:  "exterior",
    1:  "wall",
    2:  "living room",
    3:  "master bedroom",
    4:  "bedroom",
    5:  "bedroom",       # bedroom2
    6:  "bedroom",       # bedroom3
    7:  "kitchen",
    8:  "bathroom",
    9:  "bathroom",      # bath2
    10: "dining",
    11: "study",
    12: "hallway",
    13: "closet",
    14: "balcony",
    15: "storage",
    16: "utility",
    17: "door",
    18: "window",
}


class RPLANDataset:
    """
    Lazy-loading dataset for RPLAN.

    Parameters
    ----------
    root : path to the ``rplan/`` directory containing numbered PNGs.
    split : 'train', 'val', or 'test'.
    cfg : pipeline hyper-parameters.
    augment : apply random augmentations.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        cfg: Optional[PipelineConfig] = None,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.cfg = cfg or PipelineConfig()
        self.augment = augment and (split == "train")
        self.rng = np.random.default_rng(42)

        self.samples: List[Path] = []
        self._discover_samples()

    def _discover_samples(self):
        """Collect all numbered PNGs and apply train/val/test split."""
        all_pngs = sorted(self.root.glob("*.png"), key=lambda p: p.stem)
        # Filter out non-numeric filenames
        all_pngs = [p for p in all_pngs if p.stem.isdigit()]

        n = len(all_pngs)
        t80 = int(n * 0.8)
        t90 = int(n * 0.9)
        if self.split == "train":
            self.samples = all_pngs[:t80]
        elif self.split == "val":
            self.samples = all_pngs[t80:t90]
        else:
            self.samples = all_pngs[t90:]

        logger.info("RPLAN [%s]: %d samples discovered", self.split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[FloorPlanSample]:
        path = self.samples[idx]
        try:
            sample = self._load_sample(path)
            if sample is not None and self.augment:
                sample = augment_sample(sample, self.rng)
            return sample
        except Exception as e:
            logger.warning("Failed to load RPLAN sample %s: %s", path, e)
            return None

    def _load_sample(self, png_path: Path) -> Optional[FloorPlanSample]:
        """Load one RPLAN index-map PNG → FloorPlanSample."""
        from PIL import Image
        cfg = self.cfg

        img = Image.open(png_path)
        if img.mode == "RGB":
            arr = np.array(img)
            # Convert RGB-encoded index → single channel via red channel
            idx_map = arr[:, :, 0].astype(np.int32)
        elif img.mode in ("L", "P"):
            idx_map = np.array(img, dtype=np.int32)
        else:
            idx_map = np.array(img.convert("L"), dtype=np.int32)

        h, w = idx_map.shape

        # Build boundary from wall mask
        wall_mask = (idx_map == 1)
        exterior_mask = (idx_map == 0)
        interior = ~exterior_mask & ~wall_mask

        if not interior.any():
            return None

        # Find bounding rectangle of interior
        ys, xs = np.where(interior | wall_mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        # Boundary polygon (from bounding box of the plan)
        boundary_coords = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)

        boundary, b_mask, scale = normalise_polygon(
            boundary_coords, cfg.max_boundary_pts, cfg.normalize_scale
        )

        # Build semantic mask — map RPLAN indices → our canonical classes
        mask = np.full((h, w), ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1),
                       dtype=np.int32)
        rooms_info = []
        label_map = {}

        for pval, label_str in RPLAN_INDEX_MAP.items():
            canonical = LABEL_ALIASES.get(label_str.lower().strip(), label_str.lower().strip())
            if canonical not in ROOM_TO_IDX:
                canonical = "exterior"
            cidx = ROOM_TO_IDX[canonical]
            label_map[pval] = label_str

            region = (idx_map == pval)
            if not region.any():
                continue
            mask[region] = cidx

        # Resize mask
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((cfg.img_size, cfg.img_size), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int32)

        # Extract room bounding boxes
        sx, sy = cfg.img_size / w, cfg.img_size / h
        for cidx in range(NUM_ROOM_CLASSES):
            region = (mask == cidx)
            if not region.any():
                continue
            cname = list(ROOM_TO_IDX.keys())[cidx]
            if cname in ("exterior", "wall", "door", "window"):
                continue
            ys_r, xs_r = np.where(region)
            rooms_info.append({
                "class_idx": cidx,
                "name": cname,
                "bbox": (int(xs_r.min()), int(ys_r.min()),
                         int(xs_r.max()), int(ys_r.max())),
                "area_frac": float(region.sum()) / (cfg.img_size ** 2),
            })

        if not rooms_info:
            return None

        # Count rooms
        num_beds = sum(1 for r in rooms_info if r["name"] in ("bedroom", "master_bedroom"))
        num_baths = sum(1 for r in rooms_info if r["name"] in ("bathroom", "toilet"))
        num_kits = sum(1 for r in rooms_info if r["name"] == "kitchen")

        boxes = rooms_to_boxes(rooms_info, cfg.img_size, cfg.max_rooms)
        adj = build_adjacency_matrix(mask, NUM_ROOM_CLASSES)
        aspect = float(x2 - x1) / max(float(y2 - y1), 1.0)

        cond = encode_condition(
            num_bedrooms=num_beds,
            num_bathrooms=num_baths,
            num_kitchens=num_kits,
            plot_area=float(interior.sum()) / (h * w) * 5000,
            plot_aspect=aspect,
            cfg=cfg,
        )

        return FloorPlanSample(
            boundary=boundary,
            boundary_mask=b_mask,
            room_mask=mask,
            room_boxes=boxes,
            adjacency=adj,
            condition=cond,
            sample_id=png_path.stem,
            source="rplan",
            original_scale=1.0 / scale,
        )
