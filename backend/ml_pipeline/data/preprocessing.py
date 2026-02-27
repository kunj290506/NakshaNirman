"""
Core preprocessing utilities shared across all datasets.

Responsibilities:
  1. Normalise polygon coordinates to a canonical scale.
  2. Convert polygons → occupancy grids (binary masks).
  3. Extract per-room semantic masks from annotated images.
  4. Build room adjacency matrices from mask overlaps / shared edges.
  5. Encode architectural condition vectors (bedrooms, budget, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY = True
except ImportError:
    SHAPELY = False

from ml_pipeline.config import (
    PipelineConfig,
    ROOM_TYPES,
    ROOM_TO_IDX,
    NUM_ROOM_CLASSES,
    LABEL_ALIASES,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FloorPlanSample:
    """One training / eval sample — everything the model sees."""

    # Plot boundary (N×2 float32, normalised)
    boundary: np.ndarray                          # (max_pts, 2)
    boundary_mask: np.ndarray                     # (max_pts,) bool — valid verts

    # Target room layout mask — H×W int with class indices
    room_mask: np.ndarray                         # (H, W) int32

    # Per-room boxes  (K, 5) — [class_idx, cx, cy, w, h] normalised 0‥1
    room_boxes: np.ndarray

    # Adjacency matrix (K, K) bool
    adjacency: np.ndarray

    # Condition vector — structured inputs
    condition: np.ndarray                         # (cond_dim,) float32

    # Meta (not fed to model)
    sample_id: str = ""
    source: str = ""                              # cubicasa | rplan
    original_scale: float = 1.0                   # undo normalisation

    # Optional extras
    entry_side: int = 0                           # 0=S 1=E 2=N 3=W
    north_dir: float = 0.0                        # degrees from +Y
    budget_level: int = 1                         # 0=low 1=mid 2=high


# ---------------------------------------------------------------------------
# Polygon normalisation
# ---------------------------------------------------------------------------

def normalise_polygon(
    coords: np.ndarray,
    max_pts: int = 32,
    target_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalise a boundary polygon.

    1. Centre at origin.
    2. Scale so max bounding dimension == ``target_scale``.
    3. Pad / truncate to ``max_pts`` vertices.
    4. Return (coords, valid_mask, scale_factor).
    """
    coords = np.asarray(coords, dtype=np.float32)

    # Remove duplicate closing vertex
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]

    # Centre
    centroid = coords.mean(axis=0)
    coords = coords - centroid

    # Scale to target bounding box
    bbox = coords.max(axis=0) - coords.min(axis=0)
    max_dim = max(bbox[0], bbox[1], 1e-6)
    scale = target_scale / max_dim
    coords = coords * scale

    # Pad / truncate
    n = len(coords)
    out = np.zeros((max_pts, 2), dtype=np.float32)
    mask = np.zeros(max_pts, dtype=bool)
    k = min(n, max_pts)
    out[:k] = coords[:k]
    mask[:k] = True

    return out, mask, float(scale)


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

def polygon_to_occupancy_grid(
    coords: np.ndarray,
    size: int = 256,
) -> np.ndarray:
    """
    Rasterise a normalised polygon into a binary occupancy grid.

    coords: (N, 2) normalised to [-0.5, 0.5].
    Returns: (size, size) float32 mask — 1 inside, 0 outside.
    """
    grid = np.zeros((size, size), dtype=np.float32)

    # Map normalised coords → pixel coords
    px = ((coords[:, 0] + 0.5) * size).astype(int).clip(0, size - 1)
    py = ((coords[:, 1] + 0.5) * size).astype(int).clip(0, size - 1)

    if SHAPELY:
        poly = ShapelyPolygon(list(zip(px, py)))
        if not poly.is_valid:
            poly = poly.buffer(0)
        minx, miny, maxx, maxy = poly.bounds
        for y in range(max(int(miny), 0), min(int(maxy) + 1, size)):
            for x in range(max(int(minx), 0), min(int(maxx) + 1, size)):
                from shapely.geometry import Point
                if poly.contains(Point(x, y)):
                    grid[y, x] = 1.0
    else:
        # Scanline rasterisation fallback
        from matplotlib.path import Path as MplPath
        path = MplPath(list(zip(px, py)))
        yy, xx = np.mgrid[0:size, 0:size]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        inside = path.contains_points(pts).reshape(size, size)
        grid[inside] = 1.0

    return grid


# ---------------------------------------------------------------------------
# Room mask extraction
# ---------------------------------------------------------------------------

def extract_room_masks(
    annotation: np.ndarray,
    label_map: Dict[int, str],
    img_size: int = 256,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Convert a colour-labelled annotation image into:
      1. Semantic mask  (H, W)  — each pixel = class index from ROOM_TYPES.
      2. List of room info dicts with bounding boxes.

    Parameters
    ----------
    annotation : (H, W) or (H, W, 3) label image
    label_map  : {pixel_value_or_colour_hash → room label string}
    img_size   : resize to this square resolution

    Returns
    -------
    mask : (img_size, img_size) int32
    rooms : list of {class_idx, name, bbox: (x1, y1, x2, y2), area_frac}
    """
    from PIL import Image

    if annotation.ndim == 3:
        # Hash RGB → single int for lookup
        flat = (annotation[:, :, 0].astype(np.int32) * 65536
                + annotation[:, :, 1].astype(np.int32) * 256
                + annotation[:, :, 2].astype(np.int32))
    else:
        flat = annotation.astype(np.int32)

    h, w = flat.shape
    mask = np.full((h, w), ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1),
                   dtype=np.int32)

    rooms = []
    for pval, label_str in label_map.items():
        canonical = LABEL_ALIASES.get(label_str.lower().strip(), label_str.lower().strip())
        if canonical not in ROOM_TO_IDX:
            canonical = "exterior"
        cidx = ROOM_TO_IDX[canonical]

        region = (flat == pval)
        if not region.any():
            continue
        mask[region] = cidx

        ys, xs = np.where(region)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        area_frac = region.sum() / (h * w)

        rooms.append({
            "class_idx": cidx,
            "name": canonical,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "area_frac": float(area_frac),
        })

    # Resize to target
    if h != img_size or w != img_size:
        img = Image.fromarray(mask.astype(np.uint8))
        img = img.resize((img_size, img_size), Image.NEAREST)
        mask = np.array(img, dtype=np.int32)
        sx, sy = img_size / w, img_size / h
        for r in rooms:
            x1, y1, x2, y2 = r["bbox"]
            r["bbox"] = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    return mask, rooms


# ---------------------------------------------------------------------------
# Room bounding boxes → normalised (K, 5) tensor
# ---------------------------------------------------------------------------

def rooms_to_boxes(
    rooms: List[Dict],
    img_size: int,
    max_rooms: int = 16,
) -> np.ndarray:
    """
    Convert room info list → (max_rooms, 5) array.

    Each row: [class_idx, cx, cy, w, h] normalised to [0, 1].
    Unused slots are zero-filled.
    """
    out = np.zeros((max_rooms, 5), dtype=np.float32)
    for i, r in enumerate(rooms[:max_rooms]):
        x1, y1, x2, y2 = r["bbox"]
        cx = (x1 + x2) / 2 / img_size
        cy = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        out[i] = [r["class_idx"], cx, cy, w, h]
    return out


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def build_adjacency_matrix(
    mask: np.ndarray,
    num_classes: int = NUM_ROOM_CLASSES,
    dilation_px: int = 3,
) -> np.ndarray:
    """
    Compute room adjacency from semantic mask.

    Two rooms are adjacent if their dilated masks overlap.
    Returns (num_classes, num_classes) bool.
    """
    from scipy.ndimage import binary_dilation

    adj = np.zeros((num_classes, num_classes), dtype=bool)
    struct = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1))

    for ci in range(num_classes):
        region_i = (mask == ci)
        if not region_i.any():
            continue
        dilated_i = binary_dilation(region_i, structure=struct)
        for cj in range(ci + 1, num_classes):
            region_j = (mask == cj)
            if not region_j.any():
                continue
            if (dilated_i & region_j).any():
                adj[ci, cj] = True
                adj[cj, ci] = True

    return adj


# ---------------------------------------------------------------------------
# Condition vector encoder
# ---------------------------------------------------------------------------

def encode_condition(
    num_bedrooms: int = 2,
    num_bathrooms: int = 1,
    num_kitchens: int = 1,
    entry_side: int = 0,            # 0=S 1=E 2=N 3=W
    north_dir: float = 0.0,         # degrees
    budget_level: int = 1,          # 0=low 1=mid 2=high
    kitchen_type: int = 0,          # 0=open 1=closed 2=semi
    parking: int = 0,               # 0=none 1=covered 2=basement
    plot_area: float = 1200.0,
    plot_aspect: float = 1.0,       # width / length
    cfg: Optional[PipelineConfig] = None,
) -> np.ndarray:
    """
    Build a fixed-length condition vector.

    Layout:
      [0:8]   bedrooms  one-hot
      [8:14]  bathrooms one-hot
      [14:17] kitchens  one-hot
      [17:21] entry side one-hot
      [21:23] north dir  (sin, cos)
      [23:26] budget one-hot
      [26:29] kitchen type one-hot
      [29:32] parking one-hot
      [32]    plot_area (normalised)
      [33]    plot_aspect (normalised)
      [34:cond_dim] zeros (padding / future use)
    """
    if cfg is None:
        cfg = PipelineConfig()
    vec = np.zeros(cfg.cond_dim, dtype=np.float32)

    # One-hot bedrooms (max 8)
    vec[min(num_bedrooms, 7)] = 1.0
    # One-hot bathrooms (max 6)
    vec[8 + min(num_bathrooms, 5)] = 1.0
    # One-hot kitchens (max 3)
    vec[14 + min(num_kitchens, 2)] = 1.0
    # One-hot entry side
    vec[17 + (entry_side % 4)] = 1.0
    # North direction (sin, cos encoding)
    rad = math.radians(north_dir)
    vec[21] = math.sin(rad)
    vec[22] = math.cos(rad)
    # Budget
    vec[23 + min(budget_level, 2)] = 1.0
    # Kitchen type
    vec[26 + min(kitchen_type, 2)] = 1.0
    # Parking
    vec[29 + min(parking, 2)] = 1.0
    # Continuous features — normalised
    vec[32] = min(plot_area / 5000.0, 1.0)
    vec[33] = min(plot_aspect / 3.0, 1.0)

    return vec


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def augment_sample(sample: FloorPlanSample, rng: np.random.Generator) -> FloorPlanSample:
    """
    Apply random augmentations:
      - 90°/180°/270° rotation
      - horizontal / vertical flip
      - small boundary vertex jitter
    """
    import copy
    s = copy.deepcopy(sample)

    # Random 90° rotation (0, 1, 2 or 3 times)
    k = rng.integers(0, 4)
    if k > 0:
        s.room_mask = np.rot90(s.room_mask, k).copy()
        # Rotate boundary coords
        for _ in range(k):
            new_b = np.empty_like(s.boundary)
            new_b[:, 0] = -s.boundary[:, 1]
            new_b[:, 1] = s.boundary[:, 0]
            s.boundary = new_b

    # Random flip
    if rng.random() > 0.5:
        s.room_mask = np.flip(s.room_mask, axis=1).copy()
        s.boundary[:, 0] = -s.boundary[:, 0]
    if rng.random() > 0.5:
        s.room_mask = np.flip(s.room_mask, axis=0).copy()
        s.boundary[:, 1] = -s.boundary[:, 1]

    # Small vertex jitter (1% of scale)
    jitter = rng.normal(0, 0.005, size=s.boundary.shape).astype(np.float32)
    jitter[~s.boundary_mask] = 0
    s.boundary += jitter

    return s
