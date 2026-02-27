"""
CubiCasa5K dataset loader.

CubiCasa5K structure (after download & extraction):
  cubicasa5k/
    high_quality/
      <id>/
        model.svg       — vector floor plan
        F1_scaled.png   — rasterised floor plan image
        wall.png        — wall mask
        rooms.png       — room instance segmentation map
    high_quality_architectural/
      ...
    colorful/
      ...
    ...

We parse ``model.svg`` for room polygons & labels, and ``rooms.png`` for
pixel-level masks.  Each sample is normalised, augmented, and converted
to a ``FloorPlanSample``.

Reference: https://zenodo.org/record/2613548
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
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
    polygon_to_occupancy_grid,
    extract_room_masks,
    rooms_to_boxes,
    build_adjacency_matrix,
    encode_condition,
    augment_sample,
)

logger = logging.getLogger(__name__)

# CubiCasa5K colour → room label mapping (from their README)
CUBICASA_COLOUR_MAP: Dict[Tuple[int, int, int], str] = {
    (192, 192, 224): "living room",
    (192, 255, 255): "bedroom",
    (224, 255, 192): "kitchen",
    (255, 224, 128): "bathroom",
    (255, 160, 96):  "hallway",
    (255, 224, 224): "dining",
    (224, 224, 128): "study",
    (224, 224, 224): "closet",
    (255, 255, 255): "exterior",
    (128, 128, 128): "wall",
}


class CubiCasa5KDataset:
    """
    Lazy-loading dataset for CubiCasa5K.

    Parameters
    ----------
    root : path to the ``cubicasa5k/`` directory.
    split : 'train', 'val', or 'test' — based on folder lists.
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

    # ---- Discovery --------------------------------------------------------

    def _discover_samples(self):
        """Walk the directory tree and collect sample paths."""
        quality_dirs = ["high_quality", "high_quality_architectural", "colorful"]
        for qd in quality_dirs:
            qdir = self.root / qd
            if not qdir.exists():
                continue
            for entry in sorted(qdir.iterdir()):
                if entry.is_dir():
                    # Must have either model.svg or rooms.png
                    if (entry / "model.svg").exists() or (entry / "rooms.png").exists():
                        self.samples.append(entry)

        # Split 80/10/10
        n = len(self.samples)
        t80 = int(n * 0.8)
        t90 = int(n * 0.9)
        if self.split == "train":
            self.samples = self.samples[:t80]
        elif self.split == "val":
            self.samples = self.samples[t80:t90]
        else:
            self.samples = self.samples[t90:]

        logger.info("CubiCasa5K [%s]: %d samples discovered", self.split, len(self.samples))

    # ---- Length & indexing -------------------------------------------------

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
            logger.warning("Failed to load CubiCasa sample %s: %s", path, e)
            return None

    # ---- Loading logic -----------------------------------------------------

    def _load_sample(self, sample_dir: Path) -> Optional[FloorPlanSample]:
        """Load one CubiCasa5K sample directory → FloorPlanSample."""
        cfg = self.cfg

        # --- Try SVG first (vector data with polygons) ---
        svg_path = sample_dir / "model.svg"
        rooms_png = sample_dir / "rooms.png"

        boundary_coords = None
        room_polys: Dict[str, List[np.ndarray]] = {}

        if svg_path.exists():
            boundary_coords, room_polys = self._parse_svg(svg_path)

        # --- Fallback to rooms.png ---
        room_mask_raw = None
        label_map = {}
        if rooms_png.exists():
            from PIL import Image
            img = np.array(Image.open(rooms_png).convert("RGB"))
            # Build colour → label map
            unique_colours = np.unique(img.reshape(-1, 3), axis=0)
            for c in unique_colours:
                ct = tuple(c.tolist())
                if ct in CUBICASA_COLOUR_MAP:
                    hval = c[0] * 65536 + c[1] * 256 + c[2]
                    label_map[hval] = CUBICASA_COLOUR_MAP[ct]
            room_mask_raw = img

        # Must have at least a room mask
        if room_mask_raw is None and not room_polys:
            return None

        # --- Build boundary ---
        if boundary_coords is None:
            # Estimate boundary from image extent
            h, w = (room_mask_raw.shape[:2] if room_mask_raw is not None
                     else (cfg.img_size, cfg.img_size))
            boundary_coords = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)

        boundary, b_mask, scale = normalise_polygon(
            boundary_coords, cfg.max_boundary_pts, cfg.normalize_scale
        )

        # --- Build room mask ---
        if room_mask_raw is not None:
            mask, rooms_info = extract_room_masks(
                room_mask_raw, label_map, cfg.img_size
            )
        else:
            # Build from SVG polygons (rasterise each room)
            mask = np.full(
                (cfg.img_size, cfg.img_size),
                ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1),
                dtype=np.int32,
            )
            rooms_info = []
            for label, polys in room_polys.items():
                canonical = LABEL_ALIASES.get(label.lower(), label.lower())
                cidx = ROOM_TO_IDX.get(canonical, ROOM_TO_IDX.get("exterior", 0))
                for poly in polys:
                    occ = polygon_to_occupancy_grid(poly * scale, cfg.img_size)
                    mask[occ > 0.5] = cidx

            # Re-extract bounding boxes from built mask
            for cidx in range(NUM_ROOM_CLASSES):
                region = (mask == cidx)
                if not region.any() or cidx == ROOM_TO_IDX.get("exterior", 0):
                    continue
                ys, xs = np.where(region)
                rooms_info.append({
                    "class_idx": cidx,
                    "name": list(ROOM_TO_IDX.keys())[cidx],
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "area_frac": float(region.sum()) / (cfg.img_size ** 2),
                })

        # --- Count rooms for condition vector ---
        num_beds = sum(1 for r in rooms_info if r["name"] in ("bedroom", "master_bedroom"))
        num_baths = sum(1 for r in rooms_info if r["name"] in ("bathroom", "toilet"))
        num_kits = sum(1 for r in rooms_info if r["name"] == "kitchen")

        boxes = rooms_to_boxes(rooms_info, cfg.img_size, cfg.max_rooms)
        adj = build_adjacency_matrix(mask, NUM_ROOM_CLASSES)

        cond = encode_condition(
            num_bedrooms=num_beds,
            num_bathrooms=num_baths,
            num_kitchens=num_kits,
            plot_area=float(np.sum(mask != ROOM_TO_IDX.get("exterior", 0)))
                      / (cfg.img_size ** 2) * 5000,
            plot_aspect=1.0,
            cfg=cfg,
        )

        return FloorPlanSample(
            boundary=boundary,
            boundary_mask=b_mask,
            room_mask=mask,
            room_boxes=boxes,
            adjacency=adj,
            condition=cond,
            sample_id=sample_dir.name,
            source="cubicasa",
            original_scale=1.0 / scale,
        )

    # ---- SVG parsing -------------------------------------------------------

    def _parse_svg(self, svg_path: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Parse CubiCasa5K model.svg for boundary and room polygons.

        Returns (boundary_coords, {room_label: [np.array of coords]}).
        """
        try:
            tree = ET.parse(svg_path)
        except ET.ParseError:
            return None, {}

        root = tree.getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}

        room_polys: Dict[str, List[np.ndarray]] = {}
        all_points = []

        for g in root.iter("{http://www.w3.org/2000/svg}g"):
            gid = g.get("id", "").lower()
            label = None

            # Try to identify room type from group id
            for alias, canonical in LABEL_ALIASES.items():
                if alias in gid:
                    label = canonical
                    break

            for poly in g.iter("{http://www.w3.org/2000/svg}polygon"):
                pts_str = poly.get("points", "")
                coords = self._parse_svg_points(pts_str)
                if coords is not None and len(coords) >= 3:
                    all_points.extend(coords.tolist())
                    if label:
                        room_polys.setdefault(label, []).append(coords)

            for path in g.iter("{http://www.w3.org/2000/svg}path"):
                d_attr = path.get("d", "")
                coords = self._parse_svg_path_d(d_attr)
                if coords is not None and len(coords) >= 3:
                    all_points.extend(coords.tolist())
                    if label:
                        room_polys.setdefault(label, []).append(coords)

        # Build outer boundary from convex hull of all points
        boundary = None
        if all_points:
            pts = np.array(all_points, dtype=np.float32)
            if SHAPELY:
                from shapely.geometry import MultiPoint
                hull = MultiPoint(pts.tolist()).convex_hull
                boundary = np.array(hull.exterior.coords, dtype=np.float32)
            else:
                # Simple bounding box fallback
                mn = pts.min(axis=0)
                mx = pts.max(axis=0)
                boundary = np.array([
                    mn, [mx[0], mn[1]], mx, [mn[0], mx[1]]
                ], dtype=np.float32)

        return boundary, room_polys

    @staticmethod
    def _parse_svg_points(points_str: str) -> Optional[np.ndarray]:
        """Parse SVG polygon points attribute → (N, 2) array."""
        try:
            pairs = points_str.strip().split()
            coords = []
            for p in pairs:
                x, y = p.split(",")
                coords.append([float(x), float(y)])
            return np.array(coords, dtype=np.float32) if coords else None
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_svg_path_d(d: str) -> Optional[np.ndarray]:
        """Parse a minimal SVG path 'd' attribute (M/L/Z only)."""
        coords = []
        tokens = re.findall(r"[MLZmlz]|[-+]?\d*\.?\d+", d)
        cmd = "M"
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.isalpha():
                cmd = t.upper()
                i += 1
                continue
            if cmd in ("M", "L"):
                try:
                    x, y = float(tokens[i]), float(tokens[i + 1])
                    coords.append([x, y])
                    i += 2
                except (ValueError, IndexError):
                    i += 1
            elif cmd == "Z":
                i += 1
            else:
                i += 1
        return np.array(coords, dtype=np.float32) if len(coords) >= 3 else None
