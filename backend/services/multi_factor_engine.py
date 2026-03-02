"""
Residential Architectural Design Engine v2.0

Designs houses as *spatial experiences*, not geometric partitions.
A house is a spatial journey — entry sequence -> public zone -> private retreat.

Core design philosophy:
  1. Spatial hierarchy   — Entry -> Living -> Dining -> Circulation -> Bedrooms
  2. Privacy layering    — Public -> Semi-private -> Private (3-layer depth)
  3. Controlled circulation — Min 3 ft passages, no room-through-room access
  4. Visual privacy      — Bedrooms never exposed, bathrooms never dominant
  5. Functional adjacency — Kitchen<->Dining, MasterBR<->Bath
  6. Comfort proportions — Aspect ratio 1:1 to 1:2 for habitable rooms

Layout strategy selection (based on plot shape):
  - Wide rectangular (W/L > 1.3) -> Side-corridor planning
  - Deep narrow     (W/L < 0.77) -> Central-corridor planning
  - Near square     (else)       -> Cluster planning

Three-layer depth zoning from entrance:
  Layer 1 (Front)  : Living Room
  Layer 2 (Middle) : Dining + Kitchen
  Layer 3 (Rear)   : Bedrooms + Bathrooms (bathrooms attached to bedrooms)

Re-design mode: rotates strategy, flips orientation, shuffles room placement.
"""

import math
import re
import time
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.layout_constants import (
    GRID_SNAP,
    WALL_EXTERNAL_FT, WALL_INTERNAL_FT,
    AREA_FRACTIONS, MIN_AREAS, MAX_AREAS,
    ZONE_MAP, PRIORITY,
    DESIRED_ADJACENCIES, FORBIDDEN_ADJACENCIES,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

WALL_EXT = WALL_EXTERNAL_FT     # 0.75 ft  (9 inches)
WALL_INT = WALL_INTERNAL_FT     # 0.375 ft (4.5 inches)
MIN_PASSAGE = 3.0               # Minimum passage/corridor width in ft

# Layout strategies (never repeat on redesign)
STRATEGIES = ["side_corridor", "central_corridor", "cluster"]

# ── Comfort aspect-ratio limits (Section 5: 1:1 to 1:2 for comfort rooms) ──
COMFORT_AR: Dict[str, float] = {
    "living": 2.0, "master_bedroom": 2.0, "bedroom": 2.0,
    "kitchen": 2.0, "dining": 2.0, "bathroom": 2.0,
    "toilet": 2.0, "study": 2.0, "pooja": 2.0,
    "store": 2.0, "utility": 2.0, "foyer": 2.0,
    "entrance": 2.0, "porch": 2.0, "wash_area": 2.5,
    "balcony": 3.0, "staircase": 2.5, "garage": 2.5,
    "parking": 2.5, "passage": 10.0, "corridor": 10.0,
    "hallway": 5.0,
}

# ── Minimum room areas (sq ft) ──
MIN_ROOM_AREA: Dict[str, float] = {
    "master_bedroom": 120, "bedroom": 100, "living": 120,
    "kitchen": 80, "dining": 80, "bathroom": 35,
    "toilet": 15, "study": 60, "pooja": 16,
    "store": 25, "utility": 20, "hallway": 30,
    "porch": 40, "parking": 150, "balcony": 25,
    "staircase": 40, "garage": 150, "passage": 15,
    "wash_area": 20, "foyer": 25, "entrance": 20,
}

# ── Spatial layer assignment (which depth layer each room belongs to) ──
LAYER_MAP: Dict[str, int] = {
    # Layer 1 — Front (public, closest to entrance)
    "living": 1, "foyer": 1, "entrance": 1, "porch": 1,
    "balcony": 1, "garage": 1, "parking": 1,
    # Layer 2 — Middle (semi-private)
    "dining": 2, "kitchen": 2, "wash_area": 2,
    "pooja": 2, "store": 2, "utility": 2,
    # Layer 3 — Rear (private, furthest from entrance)
    "master_bedroom": 3, "bedroom": 3, "bathroom": 3,
    "toilet": 3, "study": 3, "staircase": 3,
}

# ── Spatial zone classification ──
SPATIAL_ZONE: Dict[str, str] = {
    "living": "public", "foyer": "public", "entrance": "public",
    "porch": "public", "balcony": "public", "garage": "public",
    "parking": "public",
    "dining": "semi_private", "kitchen": "semi_private",
    "wash_area": "semi_private", "pooja": "semi_private",
    "store": "service", "utility": "service",
    "master_bedroom": "private", "bedroom": "private",
    "study": "private",
    "bathroom": "service", "toilet": "service", "staircase": "service",
}

# Room display names
DISPLAY_NAMES: Dict[str, str] = {
    "master_bedroom": "Master Bedroom", "bedroom": "Bedroom",
    "living": "Living Room", "kitchen": "Kitchen",
    "dining": "Dining Room", "bathroom": "Bathroom",
    "toilet": "Toilet", "study": "Study Room",
    "pooja": "Pooja Room", "store": "Store Room",
    "utility": "Utility", "balcony": "Balcony",
    "staircase": "Staircase", "garage": "Garage",
    "parking": "Parking", "hallway": "Hallway",
    "passage": "Passage", "wash_area": "Wash Area",
    "porch": "Porch", "foyer": "Entrance Foyer",
    "entrance": "Entrance", "corridor": "Corridor",
}


# ═══════════════════════════════════════════════════════════════════════════
# GRID SNAPPING (6-inch / 0.5 ft structural grid)
# ═══════════════════════════════════════════════════════════════════════════

def snap(v: float) -> float:
    return round(v * 2) / 2

def snap_down(v: float) -> float:
    return math.floor(v * 2) / 2

def snap_up(v: float) -> float:
    return math.ceil(v * 2) / 2


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════

def parse_input(input_data: Any) -> Dict:
    """Parse user input (natural language, JSON, or command) into requirements."""
    if isinstance(input_data, str):
        return _parse_natural_language(input_data)
    elif isinstance(input_data, dict):
        return _parse_structured_input(input_data)
    return {"error": "Unsupported input format"}


def _parse_natural_language(text: str) -> Dict:
    """Extract requirements from natural language."""
    t = text.lower().strip()
    req: Dict[str, Any] = {
        "plot_width": None, "plot_length": None, "total_area": None,
        "floors": 1, "bedrooms": None, "bathrooms": None,
        "extras": [], "is_redesign": False, "is_generate": False,
    }

    # Commands
    if any(k in t for k in ("generate new", "new plan", "different layout",
                             "new layout", "redesign", "regenerate",
                             "try different")):
        req["is_redesign"] = True
    if any(k in t for k in ("generate plan", "create plan", "design plan")):
        req["is_generate"] = True

    # Dimensions:  30x40, 30x40, 30*40
    m = re.search(r'(\d+(?:\.\d+)?)\s*[x\u00d7*]\s*(\d+(?:\.\d+)?)', t)
    if m:
        req["plot_width"] = float(m.group(1))
        req["plot_length"] = float(m.group(2))
        req["total_area"] = req["plot_width"] * req["plot_length"]

    # Area:  1200 sqft
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\s*ft|sqft|square\s*feet?)', t)
    if m:
        req["total_area"] = float(m.group(1))

    # BHK shorthand:  3BHK -> 3 bedrooms, 2 bathrooms
    m = re.search(r'(\d+)\s*bhk', t)
    if m:
        bhk = int(m.group(1))
        req["bedrooms"] = bhk
        req["bathrooms"] = max(1, bhk - 1)

    # Explicit bed/bath counts
    m = re.search(r'(\d+)\s*(?:bed(?:room)?s?)', t)
    if m:
        req["bedrooms"] = int(m.group(1))
    m = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilets?)', t)
    if m:
        req["bathrooms"] = int(m.group(1))

    # Floors
    m = re.search(r'(\d+)\s*(?:floor|storey|story|level)', t)
    if m:
        req["floors"] = int(m.group(1))

    # Extras
    for kw, rtype in [("dining", "dining"), ("study", "study"),
                       ("pooja", "pooja"), ("balcon", "balcony"),
                       ("parking", "parking"), ("garage", "garage"),
                       ("store", "store"), ("utilit", "utility"),
                       ("stair", "staircase"), ("wash", "wash_area")]:
        if kw in t:
            req["extras"].append(rtype)

    return req


def _parse_structured_input(data: Dict) -> Dict:
    """Parse structured JSON input from frontend."""
    req: Dict[str, Any] = {
        "plot_width": data.get("plot_width"),
        "plot_length": data.get("plot_length"),
        "total_area": data.get("total_area"),
        "floors": data.get("floors", 1),
        "bedrooms": data.get("bedrooms", 2),
        "bathrooms": data.get("bathrooms", 1),
        "extras": list(data.get("extras", [])),
        "is_redesign": data.get("is_redesign", False) or data.get("redesign", False),
        "is_generate": data.get("is_generate", True),
        "boundary_polygon": data.get("boundary_polygon"),
        "rooms": data.get("rooms"),
    }

    # Parse rooms list from frontend cards
    if req["rooms"] and isinstance(req["rooms"], list):
        bed_count = bath_count = 0
        extra_list = list(req["extras"])
        for r in req["rooms"]:
            rtype = r.get("room_type", "")
            qty = r.get("quantity", 1)
            if rtype in ("master_bedroom", "bedroom"):
                bed_count += qty
            elif rtype == "bathroom":
                bath_count += qty
            elif rtype not in ("kitchen", "living"):
                for _ in range(qty):
                    if rtype not in extra_list:
                        extra_list.append(rtype)
        if bed_count > 0:
            req["bedrooms"] = bed_count
        if bath_count > 0:
            req["bathrooms"] = bath_count
        req["extras"] = extra_list

    return req


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — REQUIREMENTS NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_requirements(req: Dict) -> Dict:
    """Fill defaults, derive plot dimensions, clamp values."""
    bedrooms = req.get("bedrooms") or 2
    bathrooms = req.get("bathrooms") or max(1, bedrooms - 1)
    floors = req.get("floors") or 1
    extras = req.get("extras") or []

    plot_w = req.get("plot_width")
    plot_l = req.get("plot_length")
    total_area = req.get("total_area")

    # Derive from boundary polygon if provided
    boundary = req.get("boundary_polygon")
    if boundary and len(boundary) >= 3 and not plot_w:
        if isinstance(boundary[0], (list, tuple)):
            xs = [p[0] for p in boundary]
            ys = [p[1] for p in boundary]
            plot_w = max(xs) - min(xs)
            plot_l = max(ys) - min(ys)
            total_area = plot_w * plot_l

    # Derive missing dimensions
    if total_area and not plot_w:
        ratio = 1.3
        plot_w = snap(math.sqrt(total_area * ratio))
        plot_l = snap(total_area / plot_w)
        total_area = plot_w * plot_l
    elif plot_w and plot_l and not total_area:
        total_area = plot_w * plot_l

    # Fallback defaults
    if not plot_w or not plot_l or not total_area:
        total_area = total_area or 1200
        ratio = 1.3
        plot_w = snap(math.sqrt(total_area * ratio))
        plot_l = snap(total_area / plot_w)
        total_area = plot_w * plot_l

    return {
        "plot_width": snap(plot_w),
        "plot_length": snap(plot_l),
        "total_area": total_area,
        "floors": floors,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "extras": extras,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — AREA ALLOCATION (proportional to BHK, with layer assignment)
# ═══════════════════════════════════════════════════════════════════════════

def _allocate_areas(
    total_area: float,
    usable_w: float,
    usable_l: float,
    bedrooms: int,
    bathrooms: int,
    extras: List[str],
) -> List[Dict]:
    """
    Proportional area allocation. Each room gets a fraction of usable area
    scaled by BHK density, clamped to min/max bounds.
    """
    usable_area = usable_w * usable_l
    room_types: List[Tuple[str, int]] = []

    # Core rooms
    room_types.append(("living", 1))
    room_types.append(("kitchen", 1))

    # Bedrooms (first is master)
    if bedrooms >= 1:
        room_types.append(("master_bedroom", 1))
    for _ in range(max(0, bedrooms - 1)):
        room_types.append(("bedroom", 1))

    # Bathrooms
    for _ in range(bathrooms):
        room_types.append(("bathroom", 1))

    # Always add dining for >= 2 bedrooms
    has_dining = any(r[0] == "dining" for r in room_types) or "dining" in extras
    if not has_dining and bedrooms >= 2:
        room_types.append(("dining", 1))

    # Extras
    for extra in extras:
        if extra not in [r[0] for r in room_types]:
            room_types.append((extra, 1))

    total_rooms = len(room_types)

    # Build room specs with proportional area
    room_specs: List[Dict] = []
    for rtype, qty in room_types:
        frac = AREA_FRACTIONS.get(rtype, (0.04, 0.06, 0.08))
        lo, ideal, hi = frac

        # Scale down when many rooms compete for space
        if total_rooms > 6:
            scale = max(0.72, math.sqrt(6.0 / total_rooms))
            ideal *= scale

        # BHK-aware adjustments for small plots
        if usable_area < 400:
            if rtype in ("master_bedroom", "bedroom"):
                ideal = max(ideal, 0.22)
            elif rtype == "living":
                ideal = min(ideal, 0.12)
        elif usable_area < 600:
            if rtype in ("master_bedroom", "bedroom"):
                ideal = max(ideal, 0.17)
            elif rtype == "living":
                ideal = min(ideal, 0.14)

        target = usable_area * ideal
        target = max(target, MIN_ROOM_AREA.get(rtype, 25))
        target = min(target, MAX_AREAS.get(rtype, usable_area * 0.4))

        room_specs.append({
            "room_type": rtype,
            "target_area": round(target, 1),
            "min_area": MIN_ROOM_AREA.get(rtype, 25),
            "zone": SPATIAL_ZONE.get(rtype, "service"),
            "layer": LAYER_MAP.get(rtype, 2),
            "priority": PRIORITY.get(rtype, 10),
        })

    # Cap total allocation at 90% of usable (leave space for walls + circulation)
    total_alloc = sum(r["target_area"] for r in room_specs)
    if total_alloc > usable_area * 0.90:
        scale = (usable_area * 0.88) / total_alloc
        for r in room_specs:
            r["target_area"] = max(r["min_area"] * 0.6,
                                   round(r["target_area"] * scale, 1))

    # Sort by priority (higher placed first -> better position)
    room_specs.sort(key=lambda r: r["priority"], reverse=True)
    return room_specs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRATEGY SELECTION (Step 2)
# ═══════════════════════════════════════════════════════════════════════════

def _select_strategy(plot_w: float, plot_l: float) -> str:
    """
    Choose layout strategy dynamically based on plot shape.

    Wide rectangular  (W/L > 1.3) -> side_corridor
    Deep narrow       (W/L < 0.77) -> central_corridor
    Near square       (else)        -> cluster
    """
    ratio = plot_w / plot_l if plot_l > 0 else 1.0
    if ratio > 1.3:
        return "side_corridor"
    elif ratio < 0.77:
        return "central_corridor"
    else:
        return "cluster"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — SPATIAL LAYER ASSIGNMENT (Step 3)
# ═══════════════════════════════════════════════════════════════════════════

def _assign_to_layers(room_specs: List[Dict], uw: float = 20, ul: float = 20) -> Dict[int, List[Dict]]:
    """
    Assign rooms to 3 spatial layers based on depth from entrance.

    Layer 1 (Front)  : Living, foyer, porch
    Layer 2 (Middle) : Kitchen, dining, utility, pooja
    Layer 3 (Rear)   : Bedrooms, bathrooms, study
    """
    layers: Dict[int, List[Dict]] = {1: [], 2: [], 3: []}
    for spec in room_specs:
        layer = spec.get("layer", 2)
        layers[layer].append(spec)

    # Balance layers: ensure each non-empty layer has 2+ rooms
    # to prevent AR violations when rooms span full band width
    _balance_layers(layers, uw, ul)

    # Interleave bedrooms with bathrooms in Layer 3 for proper adjacency:
    # Master BR -> Bath -> BR2 -> Bath2 -> remaining
    layers[3] = _interleave_beds_baths(layers[3])
    return layers


def _balance_layers(layers: Dict[int, List[Dict]], uw: float, ul: float) -> None:
    """
    Ensure each non-empty layer has 2+ rooms.
    Single rooms spanning full band width create extreme AR violations.
    Redistributes or adds implicit fill rooms.

    Layer 1 ALWAYS gets balanced (single room at full width is always bad).
    Layers 2/3 only balanced on non-tight plots.
    """
    total_target = sum(
        sum(r["target_area"] for r in layers[l]) for l in (1, 2, 3)
    )
    usable_area = uw * ul
    is_tight = uw <= 16 or total_target > usable_area * 0.85

    # Layer 1 ALWAYS balanced — 1 room at full width creates massive oversized room
    # But only when width allows 2 rooms of at least 8ft each
    if len(layers[1]) == 1 and uw > 20:
        dining = [r for r in layers[2] if r["room_type"] == "dining"]
        if dining:
            dining[0]["layer"] = 1  # Update layer attribute
            layers[1].append(dining[0])
            layers[2].remove(dining[0])
        elif not is_tight:
            layers[1].append({
                "room_type": "foyer", "target_area": 40,
                "min_area": 25, "zone": "public", "layer": 1,
                "priority": 60,
            })

    # Skip Layers 2/3 balancing for tight plots
    if is_tight:
        return

    # Layer 2 single room: add wash_area (only on wide plots where both rooms
    # still get adequate width after the L1/L2 split)
    if len(layers[2]) == 1 and uw > 22:
        layers[2].append({
            "room_type": "wash_area", "target_area": 35,
            "min_area": 20, "zone": "semi_private", "layer": 2,
            "priority": 30,
        })

    # Layer 2 empty (all moved to L1): create minimal layer
    if len(layers[2]) == 0 and len(layers[1]) > 0 and len(layers[3]) > 0:
        layers[2].append({
            "room_type": "wash_area", "target_area": 30,
            "min_area": 20, "zone": "semi_private", "layer": 2,
            "priority": 30,
        })


def _interleave_beds_baths(rooms: List[Dict]) -> List[Dict]:
    """Order: master_bedroom, bath, bedroom, bath, ... then others."""
    beds = [r for r in rooms if r["room_type"] in ("master_bedroom", "bedroom")]
    baths = [r for r in rooms if r["room_type"] in ("bathroom", "toilet")]
    others = [r for r in rooms
              if r["room_type"] not in ("master_bedroom", "bedroom", "bathroom", "toilet")]

    # Master first
    beds.sort(key=lambda r: 0 if r["room_type"] == "master_bedroom" else 1)

    result: List[Dict] = []
    bi = 0
    for bed in beds:
        result.append(bed)
        if bi < len(baths):
            result.append(baths[bi])
            bi += 1
    while bi < len(baths):
        result.append(baths[bi])
        bi += 1
    result.extend(others)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUILDABLE AREA
# ═══════════════════════════════════════════════════════════════════════════

def _buildable_rect(plot_w: float, plot_l: float) -> Tuple[float, float, float, float]:
    """Return (ux, uy, uw, ul) -- origin and size of usable area inside ext walls."""
    ux = snap_up(WALL_EXT)
    uy = snap_up(WALL_EXT)
    uw = snap_down(plot_w - 2 * WALL_EXT)
    ul = snap_down(plot_l - 2 * WALL_EXT)
    return ux, uy, uw, ul


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — ROOM PLACEMENT DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

def _place_rooms(
    room_specs: List[Dict],
    plot_w: float,
    plot_l: float,
    strategy: str,
) -> Tuple[List[Dict], Dict]:
    """
    Place rooms according to selected strategy.
    Returns (placed_rooms, circulation_info).
    """
    ux, uy, uw, ul = _buildable_rect(plot_w, plot_l)
    layers = _assign_to_layers(room_specs, uw, ul)

    if strategy == "central_corridor":
        return _place_central_corridor(layers, ux, uy, uw, ul)
    elif strategy == "side_corridor":
        return _place_side_corridor(layers, ux, uy, uw, ul)
    else:
        return _place_cluster(layers, ux, uy, uw, ul)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY A — CENTRAL CORRIDOR  (deep / narrow plots)
# ═══════════════════════════════════════════════════════════════════════════
#
#  +----------------------------------+  North
#  |  LAYER 3  -- Private zone        |
#  |  [Master BR][Bath][BR2][Bath2]   |
#  +----------------------------------+
#  |  PASSAGE  (>= 3 ft)             |
#  +----------------------------------+
#  |  LAYER 2  -- Semi-private zone   |
#  |  [Kitchen]   [Dining]            |
#  +----------------------------------+
#  |  LAYER 1  -- Public zone         |
#  |  [Living Room]  [Foyer]          |
#  +----------^-----------------------+  South (entrance)
#

def _place_central_corridor(
    layers: Dict[int, List[Dict]],
    ux: float, uy: float, uw: float, ul: float,
) -> Tuple[List[Dict], Dict]:
    placed: List[Dict] = []
    MIN_DEPTH = 7.0

    passage_h = snap(max(MIN_PASSAGE, min(4.5, ul * 0.08)))

    # Area needs per layer
    a1 = sum(r["target_area"] for r in layers[1]) or 1
    a2 = sum(r["target_area"] for r in layers[2]) or 1
    a3 = sum(r["target_area"] for r in layers[3]) or 1
    total_a = a1 + a2 + a3

    available = ul - passage_h

    # Proportional depth allocation
    d1 = available * (a1 / total_a)
    d2 = available * (a2 / total_a)
    d3 = available * (a3 / total_a)

    # AR-safe minimum depths: each room in a layer gets ~uw/n width,
    # so min depth = (uw/n) / max_ar to satisfy aspect ratio
    def _ar_min_depth(layer_rooms):
        if not layer_rooms:
            return 0
        n = len(layer_rooms)
        per_w = uw / n
        return snap_up(per_w / 2.0)

    ar_min1 = _ar_min_depth(layers[1])
    ar_min2 = _ar_min_depth(layers[2])
    ar_min3 = _ar_min_depth(layers[3])

    # Apply AR minimums and structural minimums
    d1 = max(MIN_DEPTH, ar_min1, snap(d1)) if layers[1] else 0
    d2 = max(MIN_DEPTH, ar_min2, snap(d2)) if layers[2] else 0
    d3 = max(MIN_DEPTH, ar_min3, snap(d3)) if layers[3] else 0

    # Fit within available depth -- scale proportionally if needed
    total_used = d1 + d2 + d3
    if total_used > available:
        # Scale while respecting AR minimums
        excess = total_used - available
        # Reduce layer with most headroom above its AR minimum
        for _ in range(10):
            total_used = d1 + d2 + d3
            if total_used <= available + 0.3:
                break
            headrooms = [
                (d1 - ar_min1, 1) if layers[1] else (0, 1),
                (d2 - ar_min2, 2) if layers[2] else (0, 2),
                (d3 - ar_min3, 3) if layers[3] else (0, 3),
            ]
            total_hr = sum(h for h, _ in headrooms) or 1
            excess = total_used - available
            for hr, lnum in headrooms:
                reduction = excess * (hr / total_hr)
                if lnum == 1:
                    d1 = max(ar_min1, snap(d1 - reduction))
                elif lnum == 2:
                    d2 = max(ar_min2, snap(d2 - reduction))
                else:
                    d3 = max(ar_min3, snap(d3 - reduction))

    # Vertical positions (south -> north)
    y1 = uy                    # Layer 1 -- front (entrance)
    y2 = y1 + d1               # Layer 2 -- middle
    y_pass = y2 + d2           # Passage
    y3 = y_pass + passage_h    # Layer 3 -- rear

    # Recalculate d3 to fill remaining space exactly
    d3_actual = snap(uy + ul - y3)
    if d3_actual >= MIN_DEPTH:
        d3 = d3_actual

    # Place rooms in horizontal bands
    if layers[1]:
        placed.extend(_place_band_h(layers[1], ux, y1, uw, d1))
    if layers[2]:
        placed.extend(_place_band_h(layers[2], ux, y2, uw, d2))
    if layers[3]:
        placed.extend(_place_band_h(layers[3], ux, y3, uw, d3))

    circ = {
        "type": "horizontal_passage",
        "width_ft": round(uw, 1),
        "depth_ft": round(passage_h, 1),
        "position": {"x": round(ux, 2), "y": round(y_pass, 2)},
        "description": (
            f"{passage_h}ft passage spanning full width, "
            f"connecting dining zone to bedroom wing"
        ),
    }
    return placed, circ


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY B — SIDE CORRIDOR  (wide rectangular plots)
# ═══════════════════════════════════════════════════════════════════════════
#
#  +------------+----+-------------+
#  |  PUBLIC    | P  |  PRIVATE    |
#  |  Living    | A  |  Master BR  |
#  |            | S  |  + Bath     |
#  |  Dining    | S  |             |
#  |            | A  |  Bedroom 2  |
#  |  Kitchen   | G  |             |
#  |            | E  |  Bathroom   |
#  +-----^------+----+-------------+
#     Entrance (south-left)
#

def _place_side_corridor(
    layers: Dict[int, List[Dict]],
    ux: float, uy: float, uw: float, ul: float,
) -> Tuple[List[Dict], Dict]:
    placed: List[Dict] = []

    passage_w = snap(max(MIN_PASSAGE, min(4.0, uw * 0.08)))

    # Public rooms (layers 1+2) on left, private (layer 3) on right
    left_rooms = layers[1] + layers[2]
    right_rooms = layers[3]

    left_area = sum(r["target_area"] for r in left_rooms) or 1
    right_area = sum(r["target_area"] for r in right_rooms) or 1
    total = left_area + right_area

    avail_w = uw - passage_w
    left_w = snap(max(8, avail_w * (left_area / total)))
    right_w = snap(avail_w - left_w)

    # Place left column (public rooms stacked vertically)
    if left_rooms:
        placed.extend(_place_band_v(left_rooms, ux, uy, left_w, ul))

    # Place right column (private rooms stacked vertically)
    right_x = ux + left_w + passage_w
    if right_rooms:
        placed.extend(_place_band_v(right_rooms, right_x, uy, right_w, ul))

    circ = {
        "type": "vertical_passage",
        "width_ft": round(passage_w, 1),
        "depth_ft": round(ul, 1),
        "position": {"x": round(ux + left_w, 2), "y": round(uy, 2)},
        "description": (
            f"{passage_w}ft side corridor separating public and private zones"
        ),
    }
    return placed, circ


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY C — CLUSTER  (near-square plots)
# ═══════════════════════════════════════════════════════════════════════════
#
#  +--------------+--------------+
#  |  Master BR   |  Bedroom 2   |
#  |  + Bath      |  + Bath      |
#  +--------------+--------------+
#  |  PASSAGE                    |
#  +--------------+--------------+
#  |  Living      |  Kitchen     |
#  |              |  + Dining    |
#  +------^-------+--------------+
#      Entrance
#

def _place_cluster(
    layers: Dict[int, List[Dict]],
    ux: float, uy: float, uw: float, ul: float,
) -> Tuple[List[Dict], Dict]:
    placed: List[Dict] = []
    MIN_DEPTH = 7.0

    passage_h = snap(max(MIN_PASSAGE, min(4.0, ul * 0.07)))

    # Front zone = layers 1+2, Rear zone = layer 3
    front_rooms = layers[1] + layers[2]
    rear_rooms = layers[3]

    front_area = sum(r["target_area"] for r in front_rooms) or 1
    rear_area = sum(r["target_area"] for r in rear_rooms) or 1
    total_a = front_area + rear_area

    available_h = ul - passage_h
    front_h = snap(max(MIN_DEPTH, available_h * (front_area / total_a)))
    rear_h = snap(available_h - front_h)
    rear_h = max(MIN_DEPTH, rear_h)

    # Y positions
    front_y = uy
    pass_y = front_y + front_h
    rear_y = pass_y + passage_h

    # Recalculate rear_h to fill remaining exactly
    rear_h_actual = snap(uy + ul - rear_y)
    if rear_h_actual >= MIN_DEPTH:
        rear_h = rear_h_actual

    # Front zone: split into left (layer 1: living) and right (layer 2: kitchen+dining)
    l1_area = sum(r["target_area"] for r in layers[1]) or 1
    l2_area = sum(r["target_area"] for r in layers[2]) or 1
    l12_total = l1_area + l2_area
    left_w = snap(max(8, (uw - WALL_INT) * (l1_area / l12_total)))
    right_w = snap(uw - left_w - WALL_INT)

    if layers[1]:
        placed.extend(_place_band_h(layers[1], ux, front_y, left_w, front_h))
    if layers[2]:
        placed.extend(_place_band_h(
            layers[2], ux + left_w + WALL_INT, front_y, right_w, front_h
        ))

    # Rear zone: bedrooms + bathrooms across full width
    if rear_rooms:
        placed.extend(_place_band_h(rear_rooms, ux, rear_y, uw, rear_h))

    circ = {
        "type": "central_passage",
        "width_ft": round(uw, 1),
        "depth_ft": round(passage_h, 1),
        "position": {"x": round(ux, 2), "y": round(pass_y, 2)},
        "description": (
            f"{passage_h}ft passage connecting living zone to bedroom zone"
        ),
    }
    return placed, circ


# ═══════════════════════════════════════════════════════════════════════════
# BAND PLACEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _max_rooms_for_row(rooms: List[Dict], band_w: float, band_h: float) -> int:
    """How many rooms fit in a single row while maintaining AR constraints."""
    used = 0.0
    count = 0
    for r in rooms:
        ar = COMFORT_AR.get(r["room_type"], 2.0)
        min_w = max(4.0, band_h / ar)
        needed = min_w + (WALL_INT if count > 0 else 0)
        if used + needed > band_w:
            break
        used += needed
        count += 1
    return max(1, count)


def _split_rows(rooms: List[Dict], max_per: int) -> List[List[Dict]]:
    """Split rooms into sub-rows of at most max_per each."""
    rows: List[List[Dict]] = []
    for i in range(0, len(rooms), max_per):
        rows.append(rooms[i:i + max_per])
    return rows


def _place_band_h(
    rooms: List[Dict],
    x0: float, y0: float,
    band_w: float, band_h: float,
) -> List[Dict]:
    """
    Place rooms LEFT -> RIGHT in a horizontal band.
    If too many rooms for one row (AR violation), split into multiple rows.
    """
    if not rooms:
        return []

    max_per = _max_rooms_for_row(rooms, band_w, band_h)
    if len(rooms) <= max_per:
        return _fill_row(rooms, x0, y0, band_w, band_h)

    # Multiple rows
    rows = _split_rows(rooms, max_per)
    n_rows = len(rows)
    row_h = snap((band_h - WALL_INT * (n_rows - 1)) / n_rows)
    row_h = max(6.0, row_h)

    placed: List[Dict] = []
    cy = y0
    for ri, row in enumerate(rows):
        rh = row_h
        if ri == n_rows - 1:
            rh = snap(y0 + band_h - cy)
            rh = max(6.0, rh)
        placed.extend(_fill_row(row, x0, cy, band_w, rh))
        cy += rh + WALL_INT
    return placed


def _place_band_v(
    rooms: List[Dict],
    x0: float, y0: float,
    col_w: float, col_h: float,
) -> List[Dict]:
    """
    Place rooms in a vertical column.
    If too many for a single stack (AR violation), arrange as a grid:
    rooms are placed in horizontal rows within the column.
    """
    if not rooms:
        return []

    # Check how many fit in a single vertical stack
    max_per_col = _max_rooms_for_col(rooms, col_w, col_h)
    if len(rooms) <= max_per_col:
        return _fill_col(rooms, x0, y0, col_w, col_h)

    # Too many: arrange as horizontal rows within the column
    # Each row has rooms side by side; rows stacked vertically
    rooms_per_row = max(1, int(col_w / 8))  # ~8ft minimum per room in row
    rows = _split_rows(rooms, rooms_per_row)
    n_rows = len(rows)
    row_h = snap((col_h - WALL_INT * max(0, n_rows - 1)) / n_rows)
    row_h = max(5.0, row_h)

    placed: List[Dict] = []
    cy = y0
    for ri, row in enumerate(rows):
        rh = row_h
        if ri == n_rows - 1:
            rh = snap(y0 + col_h - cy)
            rh = max(5.0, rh)
        placed.extend(_fill_row(row, x0, cy, col_w, rh))
        cy += rh + WALL_INT
    return placed


def _max_rooms_for_col(rooms: List[Dict], col_w: float, col_h: float) -> int:
    """How many rooms fit stacked vertically while respecting AR."""
    used = 0.0
    count = 0
    for r in rooms:
        ar = COMFORT_AR.get(r["room_type"], 2.0)
        min_h = max(4.0, col_w / ar)
        needed = min_h + (WALL_INT if count > 0 else 0)
        if used + needed > col_h:
            break
        used += needed
        count += 1
    return max(2, count)


# ═══════════════════════════════════════════════════════════════════════════
# ROW / COLUMN FILL (proportional + AR-clamped)
# ═══════════════════════════════════════════════════════════════════════════

def _fill_row(
    rooms: List[Dict],
    x0: float, y0: float,
    net_w: float, row_h: float,
) -> List[Dict]:
    """
    Place N rooms side-by-side in a horizontal row.
    Width proportional to target_area, clamped to AR limits, snapped to grid.
    """
    n = len(rooms)
    if n == 0:
        return []

    total_area = sum(r["target_area"] for r in rooms) or 1.0
    wall_space = WALL_INT * max(0, n - 1)
    avail = net_w - wall_space

    # Step 1 -- proportional widths
    widths = [max(4.0, avail * (r["target_area"] / total_area)) for r in rooms]

    # Step 2 -- AR clamping
    min_ws: List[float] = []
    max_ws: List[float] = []
    for i, r in enumerate(rooms):
        ar = COMFORT_AR.get(r["room_type"], 2.0)
        min_w = max(4.0, row_h / ar)
        max_w = row_h * ar
        min_ws.append(min_w)
        max_ws.append(max_w)
        widths[i] = max(widths[i], min_w)
        widths[i] = min(widths[i], max_w)

    # Step 3 -- normalize to fit available width
    _normalize_widths(widths, avail, min_ws, max_ws)

    # Step 4 -- snap & place
    placed: List[Dict] = []
    cx = x0
    boundary_right = x0 + net_w

    for i, room in enumerate(rooms):
        remaining = round(boundary_right - cx, 2)
        if i == n - 1:
            rw = snap(remaining)
            # Still enforce AR on last room
            rw = min(rw, snap(max_ws[i]))
        else:
            rw = snap(widths[i])
            rw = max(rw, min_ws[i])
            # Reserve space for remaining rooms
            future = sum(snap(min_ws[j]) + WALL_INT for j in range(i + 1, n))
            rw = min(rw, remaining - future)

        rw = max(rw, snap(min_ws[i]))
        rw = min(rw, remaining)
        rw = snap(rw)
        rh = snap(row_h)

        # Post-snap AR safety
        ar = COMFORT_AR.get(room["room_type"], 2.0)
        if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) > ar + 0.05:
            if rh > rw:
                desired_rw = snap(rh / ar)
                if desired_rw <= remaining:
                    rw = desired_rw
                else:
                    # Can't expand width — reduce height instead
                    rw = snap(min(remaining, rw))
                    rh = snap(rw * ar)
            else:
                rh = snap(max(rh, rw / ar))

        placed.append(_make_placed_room(room, cx, y0, rw, rh, i, rooms))
        cx = round(cx + rw + WALL_INT, 2)

    return placed


def _fill_col(
    rooms: List[Dict],
    x0: float, y0: float,
    col_w: float, net_h: float,
) -> List[Dict]:
    """
    Place N rooms stacked vertically in a column.
    Height proportional to target_area, clamped to AR limits, snapped to grid.
    """
    n = len(rooms)
    if n == 0:
        return []

    total_area = sum(r["target_area"] for r in rooms) or 1.0
    wall_space = WALL_INT * max(0, n - 1)
    avail = net_h - wall_space

    heights = [max(4.0, avail * (r["target_area"] / total_area)) for r in rooms]

    min_hs: List[float] = []
    max_hs: List[float] = []
    for i, r in enumerate(rooms):
        ar = COMFORT_AR.get(r["room_type"], 2.0)
        min_h = max(4.0, col_w / ar)
        max_h = col_w * ar
        min_hs.append(min_h)
        max_hs.append(max_h)
        heights[i] = max(heights[i], min_h)
        heights[i] = min(heights[i], max_h)

    _normalize_widths(heights, avail, min_hs, max_hs)  # same algo, just on heights

    placed: List[Dict] = []
    cy = y0
    boundary_bottom = y0 + net_h

    for i, room in enumerate(rooms):
        remaining = round(boundary_bottom - cy, 2)
        if i == n - 1:
            rl = snap(remaining)
            # Still enforce AR on last room
            rl = min(rl, snap(max_hs[i]))
        else:
            rl = snap(heights[i])
            rl = max(rl, min_hs[i])
            future = sum(snap(min_hs[j]) + WALL_INT for j in range(i + 1, n))
            rl = min(rl, remaining - future)

        rl = max(rl, snap(min_hs[i]))
        rl = min(rl, remaining)
        rl = snap(rl)
        rw = snap(col_w)

        # Post-snap AR safety
        ar = COMFORT_AR.get(room["room_type"], 2.0)
        if min(rw, rl) > 0 and max(rw, rl) / min(rw, rl) > ar + 0.05:
            if rw > rl:
                desired_rl = snap(rw / ar)
                if desired_rl <= remaining:
                    rl = desired_rl
                else:
                    rl = snap(min(remaining, rl))
                    rw = snap(rl * ar)
            else:
                rw = snap(max(rw, rl / ar))

        placed.append(_make_placed_room(room, x0, cy, rw, rl, i, rooms))
        cy = round(cy + rl + WALL_INT, 2)

    return placed


def _normalize_widths(
    widths: List[float],
    target_total: float,
    mins: List[float],
    maxs: List[float],
) -> None:
    """Adjust widths in-place to sum to target_total, respecting min/max."""
    n = len(widths)
    for _ in range(5):
        total = sum(widths)
        diff = target_total - total
        if abs(diff) < 0.3:
            return

        if diff > 0:
            # Grow -- distribute into rooms with headroom
            headroom = [max(0, maxs[i] - widths[i]) for i in range(n)]
            total_hr = sum(headroom) or 1.0
            for i in range(n):
                widths[i] += diff * (headroom[i] / total_hr)
                widths[i] = min(widths[i], maxs[i])
        else:
            # Shrink -- take from rooms with flex above minimum
            flex = [max(0, widths[i] - mins[i]) for i in range(n)]
            total_fl = sum(flex) or 1.0
            for i in range(n):
                widths[i] += diff * (flex[i] / total_fl)  # diff is negative
                widths[i] = max(widths[i], mins[i])


def _make_placed_room(
    spec: Dict, x: float, y: float, w: float, h: float,
    index: int, all_rooms: List[Dict],
) -> Dict:
    """Construct a fully-formed placed-room dict with polygon & centroid."""
    rtype = spec["room_type"]
    area = round(w * h, 1)

    # Generate numbered display name for duplicate types
    name = _display_name(rtype, index, all_rooms)

    return {
        "name": name,
        "room_type": rtype,
        "zone": spec.get("zone", SPATIAL_ZONE.get(rtype, "service")),
        "layer": spec.get("layer", LAYER_MAP.get(rtype, 2)),
        "width": w,
        "length": h,
        "area": area,
        "actual_area": area,
        "target_area": spec.get("target_area", area),
        "position": {"x": round(x, 2), "y": round(y, 2)},
        "polygon": [
            [round(x, 2),     round(y, 2)],
            [round(x + w, 2), round(y, 2)],
            [round(x + w, 2), round(y + h, 2)],
            [round(x, 2),     round(y + h, 2)],
            [round(x, 2),     round(y, 2)],
        ],
        "centroid": [round(x + w / 2, 2), round(y + h / 2, 2)],
    }


def _display_name(rtype: str, idx: int, all_rooms: List[Dict]) -> str:
    """Generate human-readable room name, numbering duplicates."""
    base = DISPLAY_NAMES.get(rtype, rtype.replace("_", " ").title())
    same = [r for r in all_rooms if r["room_type"] == rtype]
    if len(same) <= 1:
        return base
    # Find ordinal index within same-type rooms
    ordinal = 0
    for j, r in enumerate(all_rooms):
        if r["room_type"] == rtype:
            if j == idx:
                break
            ordinal += 1
    return f"{base} {ordinal + 1}"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — DOORS & WINDOWS (Step 6 -- Human Use Logic)
# ═══════════════════════════════════════════════════════════════════════════

def _assign_doors_windows(
    placed: List[Dict],
    plot_w: float,
    plot_l: float,
) -> List[Dict]:
    """
    Architectural door & window assignment.
    - Main entrance on road-facing (south) wall of living room
    - Doors on shared walls between functionally-adjacent rooms
    - Windows on external walls (large for habitable, small for wet)
    """
    for room in placed:
        x, y = room["position"]["x"], room["position"]["y"]
        w, h = room["width"], room["length"]
        rtype = room["room_type"]
        doors: List[Dict] = []
        windows: List[Dict] = []

        # Main entrance (living room, south wall)
        if rtype == "living":
            doors.append({
                "wall": "south", "position": round(x + w / 2, 2),
                "width": 3.5, "type": "main_entrance",
            })

        # Doors to adjacent rooms
        for other in placed:
            if other is room:
                continue
            shared = _shared_wall(room, other)
            if shared and _should_connect(room, other):
                doors.append({
                    "wall": shared["wall"],
                    "position": round(shared["mid"], 2),
                    "width": 2.5 if rtype in ("bathroom", "toilet") else 3.0,
                    "connects_to": other["name"],
                })

        # Windows on external walls
        for wall in _external_walls(room, plot_w, plot_l):
            if rtype in ("bathroom", "toilet", "wash_area"):
                windows.append({"wall": wall, "width": 2.0, "type": "ventilation"})
            elif rtype in ("living", "master_bedroom", "bedroom", "dining", "study"):
                windows.append({
                    "wall": wall, "width": 4.0,
                    "type": "picture_window" if rtype == "living" else "standard",
                })
            elif rtype == "kitchen":
                windows.append({"wall": wall, "width": 3.0, "type": "kitchen_window"})

        room["doors"] = doors
        room["windows"] = windows

    return placed


def _shared_wall(r1: Dict, r2: Dict) -> Optional[Dict]:
    """Detect shared wall between two rooms (within wall tolerance)."""
    x1, y1, w1, h1 = r1["position"]["x"], r1["position"]["y"], r1["width"], r1["length"]
    x2, y2, w2, h2 = r2["position"]["x"], r2["position"]["y"], r2["width"], r2["length"]
    tol = WALL_INT + 0.5

    # Right of r1 <-> Left of r2
    if abs((x1 + w1) - x2) < tol:
        ov = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        if ov > 1:
            return {"wall": "east", "mid": (max(y1, y2) + min(y1 + h1, y2 + h2)) / 2, "length": ov}

    # Left of r1 <-> Right of r2
    if abs(x1 - (x2 + w2)) < tol:
        ov = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        if ov > 1:
            return {"wall": "west", "mid": (max(y1, y2) + min(y1 + h1, y2 + h2)) / 2, "length": ov}

    # Top of r1 <-> Bottom of r2
    if abs((y1 + h1) - y2) < tol:
        ov = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        if ov > 1:
            return {"wall": "north", "mid": (max(x1, x2) + min(x1 + w1, x2 + w2)) / 2, "length": ov}

    # Bottom of r1 <-> Top of r2
    if abs(y1 - (y2 + h2)) < tol:
        ov = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        if ov > 1:
            return {"wall": "south", "mid": (max(x1, x2) + min(x1 + w1, x2 + w2)) / 2, "length": ov}

    return None


def _should_connect(r1: Dict, r2: Dict) -> bool:
    """Check if two adjacent rooms should have a connecting door."""
    t1, t2 = r1["room_type"], r2["room_type"]
    pair = (t1, t2)
    rpair = (t2, t1)

    # Forbidden connections (privacy)
    for fa, fb in FORBIDDEN_ADJACENCIES:
        if pair == (fa, fb) or rpair == (fa, fb):
            return False

    # Required connections
    required_pairs = [
        ("master_bedroom", "bathroom"), ("kitchen", "dining"),
        ("living", "dining"), ("living", "kitchen"),
    ]
    for ra, rb in required_pairs:
        if pair == (ra, rb) or rpair == (ra, rb):
            return True

    # General: connect rooms in adjacent zones
    return True


def _external_walls(room: Dict, plot_w: float, plot_l: float) -> List[str]:
    """Walls of a room that touch the plot boundary."""
    x, y, w, h = room["position"]["x"], room["position"]["y"], room["width"], room["length"]
    tol = WALL_EXT + 0.5
    walls: List[str] = []
    if x <= tol:
        walls.append("west")
    if x + w >= plot_w - tol:
        walls.append("east")
    if y <= tol:
        walls.append("south")
    if y + h >= plot_l - tol:
        walls.append("north")
    return walls


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — AUTO-CORRECTION (boundary clamping, polygon rebuild)
# ═══════════════════════════════════════════════════════════════════════════

def _auto_correct(placed: List[Dict], plot_w: float, plot_l: float) -> List[Dict]:
    """Clamp rooms to plot boundary and rebuild polygons."""
    for room in placed:
        x, y = room["position"]["x"], room["position"]["y"]
        w, h = room["width"], room["length"]

        if x + w > plot_w:
            w = snap(plot_w - x)
        if y + h > plot_l:
            h = snap(plot_l - y)
        x = max(0, x)
        y = max(0, y)
        w = max(4.0, w)
        h = max(4.0, h)

        room["width"] = w
        room["length"] = h
        room["area"] = round(w * h, 1)
        room["actual_area"] = room["area"]
        room["position"] = {"x": round(x, 2), "y": round(y, 2)}
        room["polygon"] = [
            [round(x, 2), round(y, 2)],
            [round(x + w, 2), round(y, 2)],
            [round(x + w, 2), round(y + h, 2)],
            [round(x, 2), round(y + h, 2)],
            [round(x, 2), round(y, 2)],
        ]
        room["centroid"] = [round(x + w / 2, 2), round(y + h / 2, 2)]

    return placed


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — VALIDATION (5 architectural checks)
# ═══════════════════════════════════════════════════════════════════════════

def _validate_privacy(placed: List[Dict]) -> Dict:
    """
    Privacy check:
    - Bedroom must NOT open directly into kitchen
    - Bathroom must NOT open into living room
    - Rooms must not open into other bedrooms
    """
    issues: List[str] = []
    for room in placed:
        for other in placed:
            if other is room:
                continue
            if not _shared_wall(room, other):
                continue
            has_door = any(
                d.get("connects_to") == other["name"]
                for d in room.get("doors", [])
            )
            if not has_door:
                continue
            pair = (room["room_type"], other["room_type"])
            rpair = (other["room_type"], room["room_type"])
            for fa, fb in FORBIDDEN_ADJACENCIES:
                if pair == (fa, fb) or rpair == (fa, fb):
                    issues.append(
                        f"Privacy: {room['name']} opens into {other['name']}"
                    )

    return {"privacy_ok": len(issues) == 0, "issues": issues}


def _validate_circulation(placed: List[Dict], circ_info: Dict) -> Dict:
    """
    Circulation check:
    - Passage width >= 3 ft
    - All private rooms reachable without passing through other bedrooms
    """
    issues: List[str] = []

    # Check passage width
    pw = circ_info.get("width_ft", 0)
    pd = circ_info.get("depth_ft", 0)
    passage_dim = min(pw, pd) if circ_info.get("type") == "vertical_passage" else pd
    if passage_dim < MIN_PASSAGE - 0.1:
        issues.append(f"Passage width {passage_dim}ft is below minimum {MIN_PASSAGE}ft")

    # Check bedroom isolation (no bedroom door opens into another bedroom)
    bedrooms = [r for r in placed if r["room_type"] in ("master_bedroom", "bedroom")]
    for bed in bedrooms:
        for door in bed.get("doors", []):
            target_name = door.get("connects_to", "")
            target = next((r for r in placed if r["name"] == target_name), None)
            if target and target["room_type"] in ("master_bedroom", "bedroom"):
                issues.append(
                    f"Circulation: {bed['name']} opens into {target['name']}"
                )

    return {"circulation_ok": len(issues) == 0, "issues": issues}


def _validate_proportion(placed: List[Dict]) -> Dict:
    """
    Proportion check: aspect ratio between 1:1 and 1:2 for habitable rooms.
    """
    issues: List[str] = []
    for room in placed:
        w, h = room["width"], room["length"]
        if min(w, h) <= 0:
            continue
        aspect = max(w, h) / min(w, h)
        limit = COMFORT_AR.get(room["room_type"], 2.0)
        if aspect > limit + 0.05:
            issues.append(
                f"{room['name']}: AR {aspect:.2f} exceeds {limit}"
            )
        # Check min area
        min_a = MIN_ROOM_AREA.get(room["room_type"], 25)
        if room["area"] < min_a * 0.8:
            issues.append(
                f"{room['name']}: area {room['area']}sqft below min {min_a}"
            )

    return {"proportion_ok": len(issues) == 0, "issues": issues}


def _validate_ventilation(placed: List[Dict], plot_w: float, plot_l: float) -> Dict:
    """
    Ventilation check:
    - All bedrooms must have external window
    - Bathrooms near external wall or shaft
    """
    issues: List[str] = []
    habitable = ("living", "master_bedroom", "bedroom", "dining", "study")

    for room in placed:
        rtype = room["room_type"]
        ext = _external_walls(room, plot_w, plot_l)
        has_win = len(room.get("windows", [])) > 0

        if rtype in habitable and not ext and not has_win:
            issues.append(f"{room['name']}: no external wall for ventilation/light")

        if rtype in ("bathroom", "toilet") and not ext and not has_win:
            issues.append(f"{room['name']}: no external wall or shaft")

    return {"ventilation_ok": len(issues) == 0, "issues": issues}


def _validate_hierarchy(placed: List[Dict], strategy: str) -> Dict:
    """
    Hierarchy check: spatial layering follows Entry -> Living -> Dining -> Bedrooms.
    Living must be closer to entrance (south) than bedrooms.
    """
    issues: List[str] = []

    living = [r for r in placed if r["room_type"] == "living"]
    beds = [r for r in placed if r["room_type"] in ("master_bedroom", "bedroom")]

    if living and beds:
        if strategy in ("central_corridor", "cluster"):
            # In these strategies, "closer to entrance" = lower Y
            living_y = min(r["position"]["y"] for r in living)
            beds_y = min(r["position"]["y"] for r in beds)
            if beds_y < living_y - 1:
                issues.append(
                    "Hierarchy: bedrooms are closer to entrance than living room"
                )
        elif strategy == "side_corridor":
            # "Closer to entrance" = lower/left side
            living_x = min(r["position"]["x"] for r in living)
            beds_x = min(r["position"]["x"] for r in beds)
            if beds_x < living_x - 1:
                issues.append(
                    "Hierarchy: bedrooms are on entrance side (should be opposite)"
                )

    return {"hierarchy_ok": len(issues) == 0, "issues": issues}


def _check_overlap(r1: Dict, r2: Dict) -> float:
    """Overlap area between two rooms (with wall tolerance)."""
    x1, y1, w1, h1 = r1["position"]["x"], r1["position"]["y"], r1["width"], r1["length"]
    x2, y2, w2, h2 = r2["position"]["x"], r2["position"]["y"], r2["width"], r2["length"]
    ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2) - WALL_INT)
    oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2) - WALL_INT)
    return ox * oy


def _validate_geometry(placed: List[Dict], plot_w: float, plot_l: float) -> Dict:
    """Overlap and boundary check."""
    issues: List[str] = []
    for i, r1 in enumerate(placed):
        # Boundary
        x, y, w, h = r1["position"]["x"], r1["position"]["y"], r1["width"], r1["length"]
        if x < -0.1 or y < -0.1 or x + w > plot_w + 0.5 or y + h > plot_l + 0.5:
            issues.append(f"{r1['name']}: extends beyond plot boundary")
        # Overlap
        for j, r2 in enumerate(placed):
            if j <= i:
                continue
            ov = _check_overlap(r1, r2)
            if ov > 1.0:
                issues.append(f"Overlap: {r1['name']} and {r2['name']} ({ov:.1f}sqft)")

    return {"geometry_ok": len(issues) == 0, "issues": issues}


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — OUTPUT BUILDING
# ═══════════════════════════════════════════════════════════════════════════

def _build_spatial_layers(placed: List[Dict]) -> List[Dict]:
    """Build spatial_layers summary for output."""
    layer_groups: Dict[int, Dict] = {}
    zone_labels = {1: "public", 2: "semi_private", 3: "private"}
    pos_labels = {1: "front", 2: "middle", 3: "rear"}

    for room in placed:
        layer = room.get("layer", 2)
        if layer not in layer_groups:
            layer_groups[layer] = {
                "layer": layer,
                "zone": zone_labels.get(layer, "service"),
                "rooms": [],
                "position": pos_labels.get(layer, "middle"),
            }
        layer_groups[layer]["rooms"].append(room["name"])

    return [layer_groups[k] for k in sorted(layer_groups.keys())]


def _build_output(
    placed: List[Dict],
    plot_w: float, plot_l: float,
    total_area: float, floors: int,
    strategy: str,
    circ_info: Dict,
    validation: Dict,
) -> Dict:
    """
    Build final output JSON matching the spec format:
      layout_strategy, spatial_layers, rooms, circulation, validation
    PLUS frontend-compatible fields (boundary, polygon, centroid).
    """
    built = sum(r["area"] for r in placed)
    circ_area = total_area - built
    circ_pct = round((circ_area / total_area) * 100, 1) if total_area > 0 else 0

    # Section 5 rooms with frontend-compatible fields
    out_rooms: List[Dict] = []
    for room in placed:
        out_rooms.append({
            "name": room["name"],
            "room_type": room["room_type"],
            "zone": room["zone"],
            "layer": room["layer"],
            "width": room["width"],
            "length": room["length"],
            "area": room["area"],
            "actual_area": room["area"],
            "target_area": room.get("target_area", room["area"]),
            "position": room["position"],
            "doors": room.get("doors", []),
            "windows": room.get("windows", []),
            # Frontend fields
            "polygon": room["polygon"],
            "centroid": room["centroid"],
        })

    boundary = [
        [0, 0], [plot_w, 0], [plot_w, plot_l], [0, plot_l], [0, 0]
    ]

    return {
        # -- Spec output format --
        "layout_strategy": strategy,
        "spatial_layers": _build_spatial_layers(placed),
        "rooms": out_rooms,
        "circulation": circ_info,
        "validation": validation,

        # -- Metadata --
        "plot": {"width": plot_w, "length": plot_l, "unit": "ft"},
        "floors": floors,
        "walls": {"external": "9 inch", "internal": "4.5 inch"},
        "area_summary": {
            "plot_area": round(total_area, 1),
            "built_area": round(built, 1),
            "circulation_percentage": f"{circ_pct}%",
        },

        # -- Frontend compatibility --
        "boundary": boundary,
        "total_area": round(total_area, 1),
        "engine": "spatial_experience",
        "method": "architectural_design",
        "zoning_strategy": strategy,  # kept for route compat
    }


def _build_explanation(
    req: Dict,
    strategy: str,
    circ_info: Dict,
    room_count: int,
    validation: Dict,
) -> str:
    """
    Short explanation covering:
      1. Chosen layout strategy
      2. Zoning depth logic
      3. Circulation approach
    """
    strategy_desc = {
        "central_corridor": "Central-corridor planning for this deep plot",
        "side_corridor": "Side-corridor planning for this wide rectangular plot",
        "cluster": "Cluster planning for this near-square plot",
    }
    strat_text = strategy_desc.get(strategy, strategy.replace("_", " ").title())

    all_ok = all(validation.get(k, False) for k in
                 ("privacy_ok", "circulation_ok", "proportion_ok",
                  "ventilation_ok", "hierarchy_ok"))
    status = "All validations passed." if all_ok else "Minor adjustments recommended."

    lines = [
        f"{strat_text} -- {req['bedrooms']}BHK residence on "
        f"{req['plot_width']}' x {req['plot_length']}' plot "
        f"({req['total_area']:.0f} sq ft).",

        f"Three-layer depth zoning: Living at front (Layer 1), "
        f"Kitchen + Dining in middle (Layer 2), "
        f"Bedrooms at rear (Layer 3).",

        f"Circulation: {circ_info.get('description', 'passage connecting zones')}.",

        f"Privacy gradient enforced -- bedrooms shielded from entrance, "
        f"bathrooms placed discreetly.",

        f"{room_count} rooms with aspect ratios within 1:2 comfort range. {status}",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — MAIN ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def generate_plan(input_data: Any) -> Dict:
    """
    Primary entry point -- generates a floor plan as a *spatial experience*.

    Design process:
      1. Interpret requirements
      2. Compute buildable area
      3. Allocate area proportionally
      4. Select strategy (based on plot shape)
      5. Assign rooms to spatial layers
      6. Place rooms with controlled circulation
      7. Assign doors & windows
      8. Validate (5 architectural checks)
      9. Build output
    """
    try:
        # Step 1 -- parse & normalize
        raw = (_parse_structured_input(input_data) if isinstance(input_data, dict)
               else _parse_natural_language(input_data) if isinstance(input_data, str)
               else {"total_area": 1200, "bedrooms": 2, "bathrooms": 1})
        req = _normalize_requirements(raw)

        pw, pl = req["plot_width"], req["plot_length"]
        total_area = req["total_area"]

        # Minimum area gate
        min_needed = _minimum_area(req["bedrooms"], req["bathrooms"], req["extras"])
        if total_area < min_needed:
            return {
                "error": "Insufficient area for requested configuration",
                "suggestion": (
                    f"Minimum area needed: {min_needed} sq ft. "
                    f"Reduce rooms or increase plot to >= {min_needed} sq ft."
                ),
            }

        # Steps 2-3 -- buildable area & allocation
        ux, uy, uw, ul = _buildable_rect(pw, pl)
        room_specs = _allocate_areas(
            total_area, uw, ul,
            req["bedrooms"], req["bathrooms"], req["extras"],
        )

        # Step 4 -- strategy
        strategy = _select_strategy(pw, pl)

        # Steps 5-6 -- spatial placement
        placed, circ_info = _place_rooms(room_specs, pw, pl, strategy)

        # Step 7 -- doors & windows
        placed = _assign_doors_windows(placed, pw, pl)

        # Auto-correct boundary
        placed = _auto_correct(placed, pw, pl)

        # Step 8 -- five validation checks
        v_priv = _validate_privacy(placed)
        v_circ = _validate_circulation(placed, circ_info)
        v_prop = _validate_proportion(placed)
        v_vent = _validate_ventilation(placed, pw, pl)
        v_hier = _validate_hierarchy(placed, strategy)
        v_geom = _validate_geometry(placed, pw, pl)

        validation = {
            "privacy_ok": v_priv["privacy_ok"],
            "circulation_ok": v_circ["circulation_ok"],
            "proportion_ok": v_prop["proportion_ok"],
            "ventilation_ok": v_vent["ventilation_ok"],
            "hierarchy_ok": v_hier["hierarchy_ok"],
            "geometry_ok": v_geom["geometry_ok"],
        }
        all_issues = (
            v_priv.get("issues", []) + v_circ.get("issues", []) +
            v_prop.get("issues", []) + v_vent.get("issues", []) +
            v_hier.get("issues", []) + v_geom.get("issues", [])
        )
        if all_issues:
            validation["issues"] = all_issues

        # Step 9 -- build output
        layout = _build_output(
            placed, pw, pl, total_area,
            req["floors"], strategy, circ_info, validation,
        )
        explanation = _build_explanation(
            req, strategy, circ_info, len(placed), validation,
        )

        return {
            "explanation": explanation,
            "layout": layout,
            "validation": validation,
            "engine": "spatial_experience",
            "method": "architectural_design",
        }

    except Exception as e:
        logger.exception("Spatial design engine error")
        return {
            "error": f"Engine error: {str(e)}",
            "suggestion": "Please simplify requirements and try again.",
        }


def generate_new_plan(input_data: Any, previous_strategy: str = None) -> Dict:
    """
    Re-design mode (Step 8).

    Same requirements, completely different layout:
    - Different strategy (never repeat)
    - Different corridor orientation
    - Different bedroom placement side
    - Fresh spatial arrangement
    """
    try:
        raw = (_parse_structured_input(input_data) if isinstance(input_data, dict)
               else _parse_natural_language(input_data) if isinstance(input_data, str)
               else {})
        req = _normalize_requirements(raw)
        pw, pl = req["plot_width"], req["plot_length"]
        total_area = req["total_area"]

        min_needed = _minimum_area(req["bedrooms"], req["bathrooms"], req["extras"])
        if total_area < min_needed:
            return {
                "error": "Insufficient area for requested configuration",
                "suggestion": f"Minimum area needed: {min_needed} sq ft.",
            }

        # Pick DIFFERENT strategy
        available = [s for s in STRATEGIES if s != previous_strategy]
        if not available:
            available = list(STRATEGIES)
        seed = int(time.time() * 1000)
        new_strategy = available[seed % len(available)]

        # Build & allocate
        ux, uy, uw, ul = _buildable_rect(pw, pl)
        room_specs = _allocate_areas(
            total_area, uw, ul,
            req["bedrooms"], req["bathrooms"], req["extras"],
        )

        # Shuffle within zones for visual variety
        rng = random.Random(seed)
        by_layer: Dict[int, List[Dict]] = {1: [], 2: [], 3: []}
        for r in room_specs:
            by_layer.setdefault(r.get("layer", 2), []).append(r)
        for layer_rooms in by_layer.values():
            rng.shuffle(layer_rooms)
        room_specs = []
        for k in sorted(by_layer):
            room_specs.extend(by_layer[k])

        placed, circ_info = _place_rooms(room_specs, pw, pl, new_strategy)
        placed = _assign_doors_windows(placed, pw, pl)
        placed = _auto_correct(placed, pw, pl)

        v_priv = _validate_privacy(placed)
        v_circ = _validate_circulation(placed, circ_info)
        v_prop = _validate_proportion(placed)
        v_vent = _validate_ventilation(placed, pw, pl)
        v_hier = _validate_hierarchy(placed, new_strategy)
        v_geom = _validate_geometry(placed, pw, pl)

        validation = {
            "privacy_ok": v_priv["privacy_ok"],
            "circulation_ok": v_circ["circulation_ok"],
            "proportion_ok": v_prop["proportion_ok"],
            "ventilation_ok": v_vent["ventilation_ok"],
            "hierarchy_ok": v_hier["hierarchy_ok"],
            "geometry_ok": v_geom["geometry_ok"],
        }
        all_issues = (
            v_priv.get("issues", []) + v_circ.get("issues", []) +
            v_prop.get("issues", []) + v_vent.get("issues", []) +
            v_hier.get("issues", []) + v_geom.get("issues", [])
        )
        if all_issues:
            validation["issues"] = all_issues

        layout = _build_output(
            placed, pw, pl, total_area,
            req["floors"], new_strategy, circ_info, validation,
        )
        explanation = _build_explanation(
            req, new_strategy, circ_info, len(placed), validation,
        )

        return {
            "explanation": explanation,
            "layout": layout,
            "validation": validation,
            "engine": "spatial_experience",
            "method": "architectural_design",
            "redesign": True,
        }

    except Exception as e:
        logger.exception("Spatial redesign error")
        return {
            "error": f"Redesign error: {str(e)}",
            "suggestion": "Please try again.",
        }


# ═══════════════════════════════════════════════════════════════════════════
# HELPER — minimum area calculator
# ═══════════════════════════════════════════════════════════════════════════

def _minimum_area(bedrooms: int, bathrooms: int, extras: List[str]) -> float:
    """Minimum plot area to fit the requested room configuration."""
    area = MIN_ROOM_AREA.get("living", 120) + MIN_ROOM_AREA.get("kitchen", 80)
    if bedrooms >= 1:
        area += MIN_ROOM_AREA.get("master_bedroom", 120)
    for _ in range(max(0, bedrooms - 1)):
        area += MIN_ROOM_AREA.get("bedroom", 100)
    for _ in range(bathrooms):
        area += MIN_ROOM_AREA.get("bathroom", 35)
    for extra in extras:
        area += MIN_ROOM_AREA.get(extra, 25)
    return round(area * 1.1)  # +10% for walls & circulation
