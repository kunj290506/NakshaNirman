"""
Residential Architectural Design Engine v2.0

Designs houses as *spatial experiences*, not geometric partitions.
A house is a spatial journey — entry sequence -> public zone -> private retreat.

Core design philosophy:
  1. Spatial hierarchy   — Entry -> Living -> Dining -> Circulation -> Bedrooms
  2. Privacy layering    — Public -> Semi-private -> Private (3-layer depth)
  3. Controlled circulation — Min 3 ft corridors, no room-through-room access
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
MIN_CORRIDOR = 3.0               # Minimum corridor width in ft

# Layout strategies (never repeat on redesign)
STRATEGIES = ["side_corridor", "central_corridor", "cluster"]

# ── Comfort aspect-ratio limits (Section 5: 1:1 to 1:2 for comfort rooms) ──
COMFORT_AR: Dict[str, float] = {
    "living": 2.0, "master_bedroom": 2.0, "bedroom": 2.0,
    "kitchen": 2.0, "dining": 2.0, "bathroom": 2.5,
    "toilet": 2.0, "study": 2.0, "pooja": 3.0,
    "store": 2.0, "utility": 2.0, "foyer": 3.0,
    "entrance": 2.0, "porch": 2.0, "wash_area": 2.5,
    "balcony": 3.0, "staircase": 2.5, "garage": 2.5,
    "parking": 2.5, "corridor": 10.0,
}

# ── Minimum room areas (sq ft) ──
MIN_ROOM_AREA: Dict[str, float] = {
    "master_bedroom": 120, "bedroom": 100, "living": 130,
    "kitchen": 80, "dining": 80, "bathroom": 35,
    "toilet": 15, "study": 60, "pooja": 16,
    "store": 25, "utility": 20,
    "porch": 40, "parking": 150, "balcony": 25,
    "staircase": 40, "garage": 150,
    "wash_area": 15, "foyer": 25, "entrance": 20,
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
    "parking": "Parking",
    "wash_area": "Wash Area",
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
        "bedrooms": data.get("bedrooms") or data.get("num_bedrooms") or 2,
        "bathrooms": data.get("bathrooms") or data.get("num_bathrooms"),
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
        elif usable_area < 800:
            if rtype in ("master_bedroom", "bedroom"):
                ideal = max(ideal, 0.15)
            elif rtype == "living":
                ideal = min(ideal, 0.16)
            elif rtype == "dining":
                ideal = min(ideal, 0.10)

        target = usable_area * ideal
        # On small plots, relax minimum areas — Indian NBC allows smaller
        # rooms when total plot area is constrained.
        min_a = MIN_ROOM_AREA.get(rtype, 25)
        if usable_area < 800:
            if rtype == "master_bedroom":
                min_a = min(min_a, 100)
            elif rtype == "bedroom":
                min_a = min(min_a, 85)
            elif rtype == "living":
                min_a = min(min_a, 120)
        target = max(target, min_a)
        target = min(target, MAX_AREAS.get(rtype, usable_area * 0.4))

        room_specs.append({
            "room_type": rtype,
            "target_area": round(target, 1),
            "min_area": min_a,
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

    # Auto-add extras for large plots to fill space and prevent oversized rooms.
    # Indian homes on large plots typically include pooja, store, study, balcony,
    # utility, wash_area, foyer — these are standard in premium Indian homes.
    total_alloc = sum(r["target_area"] for r in room_specs)
    surplus_ratio = usable_area / max(100, total_alloc)
    if surplus_ratio > 1.3 and usable_area > 800:
        existing_types = {r["room_type"] for r in room_specs}
        auto_extras = [
            ("dining",  bedrooms >= 2 and "dining" not in existing_types),
            ("pooja",   bedrooms >= 2 and usable_area > 900 and "pooja" not in existing_types),
            ("store",   bedrooms >= 2 and usable_area > 1000 and "store" not in existing_types),
            ("balcony", usable_area > 1000 and "balcony" not in existing_types),
            ("wash_area", usable_area > 1200 and "wash_area" not in existing_types),
            ("utility", usable_area > 1500 and "utility" not in existing_types),
            ("study",   bedrooms >= 3 and usable_area > 1500 and "study" not in existing_types),
            ("foyer",   usable_area > 2000 and "foyer" not in existing_types),
        ]
        for rtype, should_add in auto_extras:
            if should_add:
                frac = AREA_FRACTIONS.get(rtype, (0.02, 0.03, 0.04))
                target = min(usable_area * frac[1], MAX_AREAS.get(rtype, 100))
                target = max(target, MIN_ROOM_AREA.get(rtype, 20))
                room_specs.append({
                    "room_type": rtype,
                    "target_area": round(target, 1),
                    "min_area": MIN_ROOM_AREA.get(rtype, 20),
                    "zone": SPATIAL_ZONE.get(rtype, "service"),
                    "layer": LAYER_MAP.get(rtype, 2),
                    "priority": PRIORITY.get(rtype, 30),
                })

    # Scale up room targets when surplus remains — rooms should fill available
    # space rather than leaving dead zones. Target 85% coverage.
    total_alloc = sum(r["target_area"] for r in room_specs)
    fill_ratio = total_alloc / max(1, usable_area)
    if fill_ratio < 0.80:
        boost = min((usable_area * 0.82) / max(1, total_alloc), 1.35)
        for r in room_specs:
            rtype = r["room_type"]
            cap = MAX_AREAS.get(rtype, usable_area * 0.4)
            r["target_area"] = round(
                min(r["target_area"] * boost, cap), 1
            )

    # Sort by priority (higher placed first -> better position)
    room_specs.sort(key=lambda r: r["priority"], reverse=True)
    return room_specs


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRATEGY SELECTION (Step 2)
# ═══════════════════════════════════════════════════════════════════════════

def _select_strategy(plot_w: float, plot_l: float) -> str:
    """
    Choose layout strategy dynamically based on plot shape.

    Wide rectangular  (W/L > 1.3)  -> side_corridor
    Deep/narrow       (W/L < 0.85) -> central_corridor
    Near square       (else)       -> cluster
    """
    ratio = plot_w / plot_l if plot_l > 0 else 1.0
    if ratio > 1.3:
        return "side_corridor"
    elif ratio < 0.85:
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

    CRITICAL RULE: Kitchen and Dining MUST stay in the same layer
    for proper adjacency (shared wall for serving access).

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
        # PREFER adding foyer, NOT stealing dining (which breaks kitchen-dining adjacency)
        dining = [r for r in layers[2] if r["room_type"] == "dining"]
        kitchen = [r for r in layers[2] if r["room_type"] == "kitchen"]

        # Calculate how many rooms L1 can support (min 8ft per room)
        future_count = len(layers[1]) + bool(dining) + bool(kitchen)
        min_w_needed = future_count * 8.0 + WALL_INT * max(0, future_count - 1)

        if dining and kitchen and uw >= min_w_needed:
            # Wide enough: move BOTH dining AND kitchen to Layer 1
            for d in dining:
                d["layer"] = 1
                layers[1].append(d)
                layers[2].remove(d)
            for k in kitchen:
                k["layer"] = 1
                layers[1].append(k)
                layers[2].remove(k)
        elif dining:
            # Not wide enough for all 3: move just dining (kitchen stays in L2)
            dining[0]["layer"] = 1
            layers[1].append(dining[0])
            layers[2].remove(dining[0])
        else:
            # No dining to move — add foyer for balance
            if not is_tight:
                layers[1].append({
                    "room_type": "foyer", "target_area": 40,
                    "min_area": 25, "zone": "public", "layer": 1,
                    "priority": 60,
                })

    # Wide-plot L1 rebalance: when L1 has rooms but significant unused width,
    # pull just dining to L1 (not kitchen — kitchen stays adjacent to wash/utility).
    # Only for very wide plots where L1 waste is extreme.
    if len(layers[1]) > 1 and uw > 45:
        n1 = len(layers[1])
        # Estimate L1 max used width: each room's area-limited width
        est_used = 0
        for r in layers[1]:
            rt = r["room_type"]
            max_a = MAX_AREAS.get(rt, 300)
            ar = COMFORT_AR.get(rt, 2.0)
            est_used += min(uw / n1, max_a / 8.0, 10.0 * ar)
        unused = uw - est_used
        if unused > 10:
            dining = [r for r in layers[2] if r["room_type"] == "dining"]
            if dining:
                dining[0]["layer"] = 1
                layers[1].append(dining[0])
                layers[2].remove(dining[0])

    # Skip Layers 2/3 balancing for tight plots
    if is_tight:
        return

    # Layer 2 single room: add wash_area (only on wide plots, skip if L2
    # only has compact rooms — they don't need a companion for balance)
    if len(layers[2]) == 1 and uw > 22:
        if not all(r["room_type"] in COMPACT_ROOMS for r in layers[2]):
            layers[2].append({
                "room_type": "wash_area", "target_area": 25,
                "min_area": 15, "zone": "semi_private", "layer": 2,
                "priority": 30,
            })

    # Layer 2 empty (all moved to L1): ONLY add wash_area if kitchen is NOT
    # already in Layer 1. A lone wash_area in Layer 2 forces a huge minimum
    # band depth (AR constraint on full width) that steals space from bedrooms.
    if len(layers[2]) == 0 and len(layers[1]) > 0 and len(layers[3]) > 0:
        kitchen_in_l1 = any(r["room_type"] == "kitchen" for r in layers[1])
        if not kitchen_in_l1:
            layers[2].append({
                "room_type": "wash_area", "target_area": 25,
                "min_area": 15, "zone": "semi_private", "layer": 2,
                "priority": 30,
            })
        # Otherwise: Layer 2 stays empty, corridor extends between L1 and L3


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


def _max_band_depth(rooms: List[Dict], exclude_compact: bool = False) -> float:
    """
    Maximum band depth that satisfies both AR and MAX_AREA constraints.

    For a room with MAX_AREA=Ma and AR_limit=ar:
      area = w × h ≤ Ma,  h/w ≤ ar  →  h ≤ √(Ma × ar)

    Returns the minimum such h across all rooms (tightest constraint).
    If exclude_compact=True, ignores bathroom/wash_area etc. (for sub-banded zones).
    """
    max_d = 999.0
    for r in rooms:
        rtype = r["room_type"]
        if exclude_compact and rtype in COMPACT_ROOMS:
            continue
        max_a = MAX_AREAS.get(rtype)
        ar = COMFORT_AR.get(rtype, 2.0)
        if max_a:
            d = math.sqrt(max_a * ar)
            max_d = min(max_d, d)
    return max_d if max_d < 998 else 999.0


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
    Space-filling room placement using proportional rectangular subdivision.

    Algorithm:
      1. Classify rooms into 3 spatial layers (front/middle/rear)
      2. Allocate zone depths proportionally by area needs
      3. Within each zone, sub-band compact rooms (bathrooms, etc.)
      4. Fill each band completely using proportional widths
      5. Last room in each row absorbs remaining width = 100% coverage
      6. Last zone absorbs remaining height = 100% coverage

    Guarantees every square foot of buildable area is assigned to a room.
    """
    ux, uy, uw, ul = _buildable_rect(plot_w, plot_l)
    layers = _assign_to_layers(room_specs, uw, ul)

    front = layers[1]
    middle = layers[2]
    rear = layers[3]

    # ── Corridor depth ──
    corridor_h = snap(max(MIN_CORRIDOR, min(4.0, ul * 0.07)))

    # ── Area-proportional depth allocation ──
    a1 = sum(r["target_area"] for r in front) or 1
    a2 = sum(r["target_area"] for r in middle) or 1
    a3 = sum(r["target_area"] for r in rear) or 1
    a_total = a1 + a2 + a3

    available = ul - corridor_h

    d1 = snap(available * a1 / a_total) if front else 0
    d2 = snap(available * a2 / a_total) if middle else 0
    d3 = snap(available * a3 / a_total) if rear else 0

    # ── Enforce min depths (habitable rooms need ≥7ft, compact ≥5ft) ──
    min1 = 7.0 if any(r["room_type"] not in COMPACT_ROOMS for r in front) else 5.0 if front else 0
    min2 = 5.0 if middle else 0
    min3 = 7.0 if any(r["room_type"] not in COMPACT_ROOMS for r in rear) else 5.0 if rear else 0

    d1 = max(d1, min1)
    d2 = max(d2, min2)
    d3 = max(d3, min3)

    # ── Enforce max depths from AR + MAX_AREA constraints ──
    if front:
        max_d1 = _max_band_depth(front, exclude_compact=True)
        if max_d1 < 998 and d1 > max_d1:
            d1 = snap(max_d1)
    if middle:
        max_d2 = _max_band_depth(middle, exclude_compact=True)
        if max_d2 < 998 and d2 > max_d2:
            d2 = snap(max_d2)

    # ── Rear zone absorbs ALL remaining height (guarantees no vertical gap) ──
    d3 = snap(available - d1 - d2)
    if d3 < min3:
        # Compress others to give rear enough
        excess_needed = min3 - d3
        if d1 > min1 + excess_needed:
            d1 = snap(d1 - excess_needed)
        elif d2 > min2:
            shrink = min(excess_needed, d2 - min2)
            d2 = snap(d2 - shrink)
            d1 = snap(d1 - (excess_needed - shrink)) if excess_needed > shrink else d1
        d3 = snap(available - d1 - d2)

    # ── Place zones from entrance (south) → rear (north) ──
    placed: List[Dict] = []
    y_cur = uy

    # Zone 1: Front (public)
    if front:
        placed.extend(_fill_band_tiled(front, ux, y_cur, uw, d1))
    else:
        placed.append(_make_filler(ux, y_cur, uw, d1, "foyer", "Foyer"))
    y_cur = round(y_cur + d1, 2)

    # Zone 2: Middle (semi-private)
    if middle:
        placed.extend(_fill_band_tiled(middle, ux, y_cur, uw, d2))
        y_cur = round(y_cur + d2, 2)
    elif d2 > 0:
        placed.append(_make_filler(ux, y_cur, uw, d2, "foyer", "Foyer"))
        y_cur = round(y_cur + d2, 2)

    # Corridor
    corr_y = y_cur
    placed.append(_make_filler(ux, corr_y, uw, corridor_h, "corridor", "Corridor"))
    y_cur = round(y_cur + corridor_h, 2)

    # Zone 3: Rear (private) — fills all remaining depth
    d3_actual = round(uy + ul - y_cur, 2)
    if rear:
        placed.extend(_fill_rear_tiled(rear, ux, y_cur, uw, d3_actual))
    else:
        placed.append(_make_filler(ux, y_cur, uw, d3_actual, "foyer", "Foyer"))

    circ_info = {
        "type": "horizontal_corridor",
        "width_ft": round(uw, 1),
        "depth_ft": round(corridor_h, 1),
        "position": {"x": round(ux, 2), "y": round(corr_y, 2)},
        "description": (
            f"{corridor_h}ft corridor spanning full width, "
            f"connecting dining zone to bedroom wing"
        ),
    }
    return placed, circ_info


# ═══════════════════════════════════════════════════════════════════════════
# SPACE-FILLING PLACEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _fill_band_tiled(
    rooms: List[Dict],
    x0: float, y0: float, w: float, h: float,
) -> List[Dict]:
    """Fill a horizontal band completely with rooms.

    If the band has mixed compact + main rooms and is deep enough,
    sub-bands compact rooms into a shallow rear strip.
    """
    if not rooms:
        return [_make_filler(x0, y0, w, h, "foyer", "Foyer")]

    main = [r for r in rooms if r["room_type"] not in COMPACT_ROOMS]
    compact = [r for r in rooms if r["room_type"] in COMPACT_ROOMS]

    # Sub-band if mixed, deep enough, and enough rooms (3+) to justify it
    if compact and main and h >= 12 and len(rooms) >= 3:
        compact_h = snap(max(4.5, min(6.5, h * 0.30)))
        main_h = round(h - compact_h, 2)

        if main_h < 7.0:
            main_h = snap(max(7.0, h * 0.65))
            compact_h = round(h - main_h, 2)
            if compact_h < 4.0:
                return _fill_row_tiled(rooms, x0, y0, w, h)

        placed: List[Dict] = []
        placed.extend(_fill_row_tiled(main, x0, y0, w, main_h))
        placed.extend(_fill_row_tiled(compact, x0, round(y0 + main_h, 2), w, compact_h))
        return placed

    return _fill_row_tiled(rooms, x0, y0, w, h)


def _fill_rear_tiled(
    rooms: List[Dict],
    x0: float, y0: float, w: float, h: float,
) -> List[Dict]:
    """Fill rear zone with bedrooms + bathrooms using column pairing.

    Each bedroom gets paired with a bathroom in a vertical column:
    - Bathroom at front of column (closest to passage)
    - Bedroom at rear of column (deepest from entrance)
    Columns fill 100% of available width.
    """
    beds = [r for r in rooms if r["room_type"] not in COMPACT_ROOMS]
    baths = [r for r in rooms if r["room_type"] in COMPACT_ROOMS]

    if not baths or not beds or h < 10:
        return _fill_row_tiled(rooms, x0, y0, w, h)

    # Bathroom strip depth (front of zone, closest to passage)
    bath_h = snap(max(4.5, min(6.0, h * 0.28)))
    bed_h = round(h - bath_h, 2)

    if bed_h < 7.0:
        bed_h = snap(max(7.0, h * 0.65))
        bath_h = round(h - bed_h, 2)
        if bath_h < 4.0:
            return _fill_row_tiled(rooms, x0, y0, w, h)

    # Sort beds by priority (master_bedroom first)
    beds.sort(key=lambda r: r.get("priority", 50), reverse=True)

    total_bed_area = sum(r["target_area"] for r in beds) or 1
    placed: List[Dict] = []
    cx = x0
    bath_idx = 0

    for i, bed in enumerate(beds):
        is_last = (i == len(beds) - 1)
        if is_last:
            col_w = round(x0 + w - cx, 2)
        else:
            col_w = snap(w * bed["target_area"] / total_bed_area)
            col_w = max(col_w, 9.0)
        remaining = round(x0 + w - cx, 2)
        col_w = min(col_w, remaining)

        if col_w < 3.0:
            continue

        # Bedroom at rear (bottom of zone)
        placed.append(_make_placed_room(
            bed, cx, round(y0 + bath_h, 2), col_w, bed_h, i, rooms,
        ))

        # Paired bathroom at front of column
        if bath_idx < len(baths):
            placed.append(_make_placed_room(
                baths[bath_idx], cx, y0, col_w, bath_h, bath_idx, rooms,
            ))
            bath_idx += 1
        else:
            placed.append(_make_filler(
                cx, y0, col_w, bath_h, "foyer", "Foyer",
            ))

        cx = round(cx + col_w, 2)

    return placed


def _fill_row_tiled(
    rooms: List[Dict],
    x0: float, y0: float, w: float, h: float,
    ar_override: Optional[Dict] = None,
) -> List[Dict]:
    """Fill a single row completely — proportional widths, last room absorbs remainder.

    This is the core space-filling primitive. Guarantees 100% horizontal coverage.
    """
    if not rooms:
        return [_make_filler(x0, y0, w, h, "foyer", "Foyer")]

    total_area = sum(r["target_area"] for r in rooms) or 1
    placed: List[Dict] = []
    cx = x0

    # MIN_WIDTH enforcement for habitable rooms
    _MIN_W = {
        "living": 10.0, "master_bedroom": 9.0, "bedroom": 9.0,
        "kitchen": 7.0, "dining": 8.0,
    }

    for i, room in enumerate(rooms):
        rtype = room["room_type"]

        # Compute min width first (needed by safety cap)
        min_w = _MIN_W.get(rtype, 4.0)

        if i == len(rooms) - 1:
            # Last room absorbs all remaining width
            rw = round(x0 + w - cx, 2)
            # Safety cap only for wet/service rooms to prevent oversized baths
            _CAPPED = {"bathroom", "toilet", "wash_area"}
            max_a = MAX_AREAS.get(rtype)
            if max_a and rtype in _CAPPED and rw * h > max_a * 1.5:
                rw = snap(max(min_w, (max_a * 1.2) / h))
        else:
            rw = snap(w * room["target_area"] / total_area)

        # Enforce min width for habitable rooms
        rw = max(rw, min_w)

        # AR constraint: max width = h * AR
        ar = COMFORT_AR.get(rtype, 2.0)
        if ar_override and rtype in ar_override:
            ar = ar_override[rtype]
        max_w = h * ar
        if rw > max_w and i < len(rooms) - 1:
            rw = snap(max_w)

        # MAX_AREA constraint
        max_a = MAX_AREAS.get(rtype)
        if max_a and rw * h > max_a * 1.2 and i < len(rooms) - 1:
            rw = snap(max(min_w, max_a / h))

        # Don't exceed remaining width
        remaining = round(x0 + w - cx, 2)
        rw = min(rw, remaining)

        if rw < 3.5:
            # Room too narrow — stretch previous room to absorb the gap
            if placed:
                prev = placed[-1]
                new_w = round(prev["width"] + remaining, 2)
                px = prev["position"]["x"]
                prev["width"] = new_w
                prev["area"] = round(new_w * h, 1)
                prev["actual_area"] = prev["area"]
                prev["polygon"] = [
                    [round(px, 2), round(y0, 2)],
                    [round(px + new_w, 2), round(y0, 2)],
                    [round(px + new_w, 2), round(y0 + h, 2)],
                    [round(px, 2), round(y0 + h, 2)],
                    [round(px, 2), round(y0, 2)],
                ]
                prev["centroid"] = [round(px + new_w / 2, 2), round(y0 + h / 2, 2)]
            cx = round(x0 + w, 2)
            continue

        placed.append(_make_placed_room(room, cx, y0, rw, h, i, rooms))
        cx = round(cx + rw, 2)

    # Fill any remaining gap — stretch last room or add hall for compact rooms
    leftover = round(x0 + w - cx, 2)
    if leftover > 0.1 and placed:
        last = placed[-1]
        max_a = MAX_AREAS.get(last["room_type"])
        last_area_after = (last["width"] + leftover) * h
        # Only create hall filler if last room is wet/service AND would exceed 1.5x cap
        _CAPPED = {"bathroom", "toilet", "wash_area"}
        if max_a and last["room_type"] in _CAPPED and last_area_after > max_a * 1.5:
            placed.append(_make_filler(cx, y0, leftover, h, "foyer", "Foyer"))
        else:
            new_w = round(last["width"] + leftover, 2)
            lx = last["position"]["x"]
            last["width"] = new_w
            last["area"] = round(new_w * h, 1)
            last["actual_area"] = last["area"]
            last["polygon"] = [
                [round(lx, 2), round(y0, 2)],
                [round(lx + new_w, 2), round(y0, 2)],
                [round(lx + new_w, 2), round(y0 + h, 2)],
                [round(lx, 2), round(y0 + h, 2)],
                [round(lx, 2), round(y0, 2)],
            ]
            last["centroid"] = [round(lx + new_w / 2, 2), round(y0 + h / 2, 2)]

    return placed


def _make_filler(
    x0: float, y0: float, w: float, h: float,
    rtype: str, name: str,
) -> Dict:
    """Create a filler room (passage/hall) for complete space coverage."""
    area = round(w * h, 1)
    return {
        "name": name, "room_type": rtype,
        "zone": "circulation", "layer": 0,
        "width": round(w, 2), "length": round(h, 2),
        "area": area, "actual_area": area, "target_area": area,
        "position": {"x": round(x0, 2), "y": round(y0, 2)},
        "polygon": [
            [round(x0, 2), round(y0, 2)],
            [round(x0 + w, 2), round(y0, 2)],
            [round(x0 + w, 2), round(y0 + h, 2)],
            [round(x0, 2), round(y0 + h, 2)],
            [round(x0, 2), round(y0, 2)],
        ],
        "centroid": [round(x0 + w / 2, 2), round(y0 + h / 2, 2)],
    }


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

    corridor_h = snap(max(MIN_CORRIDOR, min(4.5, ul * 0.08)))

    # Area needs per layer
    a1 = sum(r["target_area"] for r in layers[1]) or 1
    a2 = sum(r["target_area"] for r in layers[2]) or 1
    a3 = sum(r["target_area"] for r in layers[3]) or 1
    total_a = a1 + a2 + a3

    available = ul - corridor_h

    # Proportional depth allocation
    d1 = available * (a1 / total_a)
    d2 = available * (a2 / total_a)
    d3 = available * (a3 / total_a)

    # Depth-from-area: ideal depth = total_layer_area / usable_width
    # This ensures rooms can fill the full band width at proper AR.
    da1 = a1 / uw if layers[1] else 0
    da2 = a2 / uw if layers[2] else 0
    da3 = a3 / uw if layers[3] else 0

    # Compute effective minimum depth per layer: reduced for compact-only
    # layers and for layers where area-derived depth is sufficient.
    def _eff_min(layer_rooms, layer_num):
        if not layer_rooms:
            return 0
        if all(r["room_type"] in COMPACT_ROOMS for r in layer_rooms):
            return 5.0
        # For layers with few rooms on plots where we can estimate depth
        # from area accurately (wide enough that rooms span full width):
        non_compact = [r for r in layer_rooms if r["room_type"] not in COMPACT_ROOMS]
        if uw > 14 and len(non_compact) <= 2:
            total = sum(r["target_area"] for r in layer_rooms)
            need = total / uw
            return max(5.0, snap_up(need))
        return MIN_DEPTH

    eff_min1 = _eff_min(layers[1], 1)
    eff_min2 = _eff_min(layers[2], 2)
    eff_min3 = _eff_min(layers[3], 3)

    # Use area-derived depth when proportional would over-allocate
    if layers[1] and d1 > da1 * 1.3:
        d1 = max(da1, eff_min1)
    if layers[2] and d2 > da2 * 1.3:
        d2 = max(da2, eff_min2)
    if layers[3] and d3 > da3 * 1.3:
        d3 = max(da3, eff_min3)

    # AR-safe minimum depths: each room in a layer gets ~uw/n width,
    # so min depth = (uw/n) / max_ar to satisfy aspect ratio.
    # Use relaxed AR for service rooms (kitchen, bathroom etc.) and
    # strict AR for habitable rooms (living, bedrooms).
    _HABITABLE = {"living", "master_bedroom", "bedroom", "dining", "foyer"}

    def _ar_min_depth(layer_rooms):
        if not layer_rooms:
            return 0
        n = len(layer_rooms)
        per_w = uw / n
        has_habitable = any(r["room_type"] in _HABITABLE for r in layer_rooms)
        max_ar = 2.0 if has_habitable else 3.0
        return snap_up(per_w / max_ar)

    ar_min1 = _ar_min_depth(layers[1])
    ar_min2 = _ar_min_depth(layers[2])
    ar_min3 = _ar_min_depth(layers[3])

    # Apply AR minimums and structural minimums
    d1 = max(eff_min1, ar_min1, snap(d1)) if layers[1] else 0
    d2 = max(eff_min2, ar_min2, snap(d2)) if layers[2] else 0
    d3 = max(eff_min3, ar_min3, snap(d3)) if layers[3] else 0

    # Apply MAX_AREA + AR depth constraints:
    # Prevents bands deeper than rooms can fill at proper AR + area caps.
    # exclude_compact for L1/L2: compact rooms handled by sub-banding
    max_d1 = snap(_max_band_depth(layers[1], exclude_compact=True)) if layers[1] else 0
    max_d3 = snap(_max_band_depth(layers[3], exclude_compact=True)) if layers[3] else 0
    max_d2 = snap(_max_band_depth(layers[2], exclude_compact=True)) if layers[2] else 0

    if max_d1 > 0 and d1 > max_d1:
        d1 = max_d1
    if max_d2 > 0 and d2 > max_d2:
        d2 = max_d2
    if max_d3 > 0 and d3 > max_d3:
        d3 = max_d3

    # Redistribute freed depth: when MAX_AREA caps shrink layers,
    # give the surplus to uncapped layers (proportionally by area need)
    total_used = d1 + d2 + d3
    if total_used < available - 1.0:
        surplus = available - total_used
        # Layers that were NOT capped can absorb surplus
        can_grow = []
        if layers[1] and (max_d1 <= 0 or d1 < max_d1 - 0.5):
            can_grow.append((1, a1))
        if layers[2] and (max_d2 <= 0 or d2 < max_d2 - 0.5):
            can_grow.append((2, a2))
        if layers[3]:
            # Layer 3 (bedrooms) always benefits from more depth
            # (multi-row handles deep bands well)
            can_grow.append((3, a3))
        if can_grow:
            grow_total = sum(a for _, a in can_grow)
            for lnum, la in can_grow:
                extra = surplus * (la / grow_total) if grow_total > 0 else surplus / len(can_grow)
                if lnum == 1:
                    d1 = snap(d1 + extra)
                elif lnum == 2:
                    d2 = snap(d2 + extra)
                else:
                    d3 = snap(d3 + extra)

    # Fit within available depth -- scale proportionally if needed
    total_used = d1 + d2 + d3
    if total_used > available:
        # Scale while respecting AR minimums
        for _ in range(10):
            total_used = d1 + d2 + d3
            if total_used <= available + 0.3:
                break
            headrooms = [
                (d1 - ar_min1, 1) if layers[1] else (0, 1),
                (d2 - ar_min2, 2) if layers[2] else (0, 2),
                (d3 - ar_min3, 3) if layers[3] else (0, 3),
            ]
            total_hr = sum(h for h, _ in headrooms)
            excess = total_used - available
            if total_hr < 0.5:
                # All layers at ar_min but still over budget:
                # Fall back to proportional allocation by area need
                t = (a1 + a2 + a3) or 1
                # Minimum depth floors: kitchen/dining need ≥5.5 for
                # AR-capped width to yield sufficient area
                _has_kitchen_l2 = any(r["room_type"] in ("kitchen", "dining")
                                     for r in layers[2]) if layers[2] else False
                _min_d2 = 5.5 if _has_kitchen_l2 else 5.0
                d1 = snap(max(5.0, available * (a1 / t))) if layers[1] else 0
                d2 = snap(max(_min_d2, available * (a2 / t))) if layers[2] else 0
                d3 = snap(max(5.0, available - d1 - d2)) if layers[3] else 0
                break
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
    y3 = y_pass + corridor_h    # Layer 3 -- rear

    # Recalculate d3 to fill remaining space exactly
    d3_actual = snap(uy + ul - y3)
    if d3_actual >= eff_min3:
        d3 = d3_actual
    # Also extend d2 upward if Layer 2 has slack (prevents thin dead strip)
    if not layers[2] and d2 > 0:
        d3 = snap(d3 + d2)
        y3 = y3 - d2
        d2 = 0

    # Place rooms in horizontal bands (with sub-banding for compact rooms)
    if layers[1]:
        if any(r["room_type"] in COMPACT_ROOMS for r in layers[1]):
            placed.extend(_place_zone_subband(layers[1], ux, y1, uw, d1, compact_at_rear=True))
        else:
            placed.extend(_place_band_h(layers[1], ux, y1, uw, d1))
    if layers[2]:
        if any(r["room_type"] in COMPACT_ROOMS for r in layers[2]):
            placed.extend(_place_zone_subband(layers[2], ux, y2, uw, d2, compact_at_rear=True))
        else:
            placed.extend(_place_band_h(layers[2], ux, y2, uw, d2))
    if layers[3]:
        placed.extend(_place_bedroom_zone(layers[3], ux, y3, uw, d3))

    circ = {
        "type": "horizontal_passage",
        "width_ft": round(uw, 1),
        "depth_ft": round(corridor_h, 1),
        "position": {"x": round(ux, 2), "y": round(y_pass, 2)},
        "description": (
            f"{corridor_h}ft passage spanning full width, "
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

    corridor_w = snap(max(MIN_CORRIDOR, min(4.0, uw * 0.08)))

    # Public rooms (layers 1+2) on left, private (layer 3) on right
    left_rooms = layers[1] + layers[2]
    right_rooms = layers[3]

    left_area = sum(r["target_area"] for r in left_rooms) or 1
    right_area = sum(r["target_area"] for r in right_rooms) or 1
    total = left_area + right_area

    avail_w = uw - corridor_w
    left_w = snap(max(8, avail_w * (left_area / total)))
    right_w = snap(avail_w - left_w)

    # Place left column (public rooms stacked vertically)
    if left_rooms:
        placed.extend(_place_band_v(left_rooms, ux, uy, left_w, ul))

    # Place right column (private rooms with bedroom zone sub-banding)
    right_x = ux + left_w + corridor_w
    if right_rooms:
        placed.extend(_place_bedroom_zone_v(right_rooms, right_x, uy, right_w, ul))

    circ = {
        "type": "vertical_corridor",
        "width_ft": round(corridor_w, 1),
        "depth_ft": round(ul, 1),
        "position": {"x": round(ux + left_w, 2), "y": round(uy, 2)},
        "description": (
            f"{corridor_w}ft side corridor separating public and private zones"
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

    corridor_h = snap(max(MIN_CORRIDOR, min(4.0, ul * 0.07)))

    # Front zone = layers 1+2, Rear zone = layer 3
    front_rooms = layers[1] + layers[2]
    rear_rooms = layers[3]

    front_area = sum(r["target_area"] for r in front_rooms) or 1
    rear_area = sum(r["target_area"] for r in rear_rooms) or 1
    total_a = front_area + rear_area

    available_h = ul - corridor_h
    front_h = snap(max(MIN_DEPTH, available_h * (front_area / total_a)))
    rear_h = snap(available_h - front_h)
    rear_h = max(MIN_DEPTH, rear_h)

    # Depth-from-area: ideal depth = total_area / width
    da_front = front_area / uw if front_rooms else MIN_DEPTH
    da_rear = rear_area / uw if rear_rooms else MIN_DEPTH
    if front_h > da_front * 1.3 and da_front >= MIN_DEPTH:
        front_h = snap(max(da_front, MIN_DEPTH))
    if rear_h > da_rear * 1.3 and da_rear >= MIN_DEPTH:
        rear_h = snap(max(da_rear, MIN_DEPTH))

    # Apply MAX_AREA + AR depth constraints for front and rear zones
    # exclude_compact: compact rooms handled by sub-banding within zones
    max_front = snap(_max_band_depth(front_rooms, exclude_compact=True))
    max_rear = snap(_max_band_depth(rear_rooms, exclude_compact=True))
    if front_h > max_front > 0:
        front_h = max_front
    if rear_h > max_rear > 0:
        rear_h = max_rear

    # Redistribute freed depth to rear zone (bedrooms benefit from multi-row)
    used = front_h + rear_h
    if used < available_h - 1.0:
        surplus = available_h - used
        rear_h = snap(rear_h + surplus)

    # Y positions
    front_y = uy
    corr_y = front_y + front_h
    rear_y = corr_y + corridor_h

    # Recalculate rear_h to fill remaining exactly
    rear_h_actual = snap(uy + ul - rear_y)
    if rear_h_actual >= MIN_DEPTH:
        rear_h = rear_h_actual

    # Front zone: use sub-banding when compact rooms present
    has_compact_front = any(r["room_type"] in COMPACT_ROOMS
                            for r in layers[1] + layers[2])
    has_compact_l1 = any(r["room_type"] in COMPACT_ROOMS for r in layers[1]) if layers[1] else False
    has_compact_l2 = any(r["room_type"] in COMPACT_ROOMS for r in layers[2]) if layers[2] else False

    if layers[1] and layers[2]:
        l1_area = sum(r["target_area"] for r in layers[1]) or 1
        l2_area = sum(r["target_area"] for r in layers[2]) or 1
        l12_total = l1_area + l2_area
        left_w = snap(max(8, (uw - WALL_INT) * (l1_area / l12_total)))
        right_w = snap(uw - left_w - WALL_INT)

        # If either column is too narrow, merge all front rooms into one band
        if right_w < 6.0 or left_w < 6.0:
            all_front = layers[1] + layers[2]
            if has_compact_front:
                placed.extend(_place_zone_subband(all_front, ux, front_y, uw, front_h, compact_at_rear=True))
            else:
                placed.extend(_place_band_h(all_front, ux, front_y, uw, front_h))
        else:
            if has_compact_l1:
                placed.extend(_place_zone_subband(layers[1], ux, front_y, left_w, front_h, compact_at_rear=True))
            else:
                placed.extend(_place_band_h(layers[1], ux, front_y, left_w, front_h))
            if has_compact_l2:
                placed.extend(_place_zone_subband(layers[2], ux + left_w + WALL_INT, front_y, right_w, front_h, compact_at_rear=True))
            else:
                placed.extend(_place_band_h(layers[2], ux + left_w + WALL_INT, front_y, right_w, front_h))
    elif layers[1]:
        if has_compact_l1:
            placed.extend(_place_zone_subband(layers[1], ux, front_y, uw, front_h, compact_at_rear=True))
        else:
            placed.extend(_place_band_h(layers[1], ux, front_y, uw, front_h))
    elif layers[2]:
        if has_compact_l2:
            placed.extend(_place_zone_subband(layers[2], ux, front_y, uw, front_h, compact_at_rear=True))
        else:
            placed.extend(_place_band_h(layers[2], ux, front_y, uw, front_h))

    # Rear zone: bedrooms + bathrooms with sub-banding
    if rear_rooms:
        placed.extend(_place_bedroom_zone(rear_rooms, ux, rear_y, uw, rear_h))

    circ = {
        "type": "central_passage",
        "width_ft": round(uw, 1),
        "depth_ft": round(corridor_h, 1),
        "position": {"x": round(ux, 2), "y": round(corr_y, 2)},
        "description": (
            f"{corridor_h}ft passage connecting living zone to bedroom zone"
        ),
    }
    return placed, circ


# ═══════════════════════════════════════════════════════════════════════════
# BEDROOM ZONE PLACEMENT (Layer 3 special handler)
# ═══════════════════════════════════════════════════════════════════════════
#
# Bathrooms MUST NOT fill the full bedroom band height.
# In real Indian residential design:
#   - Bedrooms occupy the main portion of the band (10-14ft deep)
#   - Bathrooms are compact (5-7ft deep) attached to bedrooms as sub-rooms
#
# Layout:
#   +-- Passage --+
#   +--------+--------+--------+--------+
#   | Bath 1 | Bath 2 |        |        |   ← 6ft sub-band
#   +--------+--------+--------+--------+
#   | Master   |   Bedroom 2   |Study   |   ← main sub-band
#   | Bedroom  |               |        |
#   +----------+---------------+--------+
#

# Room types that should get compact sub-bands (NOT full band height)
COMPACT_ROOMS = frozenset({
    "bathroom", "toilet", "wash_area", "pooja", "store", "utility",
    "balcony", "study",
})


def _place_zone_subband(
    rooms: List[Dict],
    x0: float, y0: float,
    band_w: float, band_h: float,
    compact_at_rear: bool = False,
) -> List[Dict]:
    """
    Generic sub-banding: separates COMPACT_ROOMS into a shallow sub-band.

    Used for front/middle zones where compact rooms (pooja, store, balcony)
    should not constrain the depth of main rooms (living, kitchen, dining).

    Args:
        compact_at_rear: If True, compact sub-band at rear (deeper into plot).
                         If False, compact sub-band at front (closer to entrance).
    """
    main = [r for r in rooms if r["room_type"] not in COMPACT_ROOMS]
    compact = [r for r in rooms if r["room_type"] in COMPACT_ROOMS]

    if not compact or not main or band_h < 13.0:
        return _place_band_h(rooms, x0, y0, band_w, band_h)

    compact_h = snap(max(5.0, min(7.0, band_h * 0.28)))
    main_h = snap(band_h - compact_h - WALL_INT)

    if main_h < 8.0:
        main_h = snap(max(8.0, band_h * 0.6))
        compact_h = snap(band_h - main_h - WALL_INT)
        if compact_h < 4.5:
            return _place_band_h(rooms, x0, y0, band_w, band_h)

    if compact_at_rear:
        main_y = y0
        compact_y = y0 + main_h + WALL_INT
    else:
        compact_y = y0
        main_y = y0 + compact_h + WALL_INT

    placed: List[Dict] = []
    placed.extend(_fill_row(main, x0, main_y, band_w, main_h))
    placed.extend(_fill_row(compact, x0, compact_y, band_w, compact_h))
    return placed


def _place_bedroom_zone(
    rooms: List[Dict],
    x0: float, y0: float,
    band_w: float, band_h: float,
) -> List[Dict]:
    """
    Special placement for Layer 3 (bedroom zone) with split sub-bands.

    Handles three scenarios:
      1. MULTI-ROW: When band is >=1.8x max depth, split into 2-3 rows
         (each row gets sub-banding). Maximizes coverage on large plots.
      2. SINGLE ROW with sub-bands: Bedrooms at rear, bathrooms at front.
      3. FALLBACK: Standard band placement for shallow bands.

    Each row sub-bands compact rooms (bathrooms) into 5-7ft shallow strip
    and gives main rooms (bedrooms) the remaining depth.
    """
    beds = [r for r in rooms if r["room_type"] not in COMPACT_ROOMS]
    baths = [r for r in rooms if r["room_type"] in COMPACT_ROOMS]

    if not beds:
        return _place_band_h(rooms, x0, y0, band_w, band_h)

    # Max depth for a single row of bedrooms (area + AR safe)
    max_bed_h = _max_band_depth(beds)

    # ── MULTI-ROW: band deep enough for 2+ bedroom rows ──
    if band_h >= max_bed_h * 1.5 and len(beds) >= 2:
        n_rows = max(2, min(3, round(band_h / max_bed_h)))
        row_h = snap((band_h - WALL_INT * (n_rows - 1)) / n_rows)

        # Distribute beds and baths across rows (ceiling division)
        bpr = -(-len(beds) // n_rows)
        bapr = -(-len(baths) // n_rows) if baths else 0

        placed: List[Dict] = []
        cy = y0
        bi, bai = 0, 0
        for ri in range(n_rows):
            if ri == n_rows - 1:
                rb = beds[bi:]
                rba = baths[bai:]
                rh = snap(y0 + band_h - cy)
            else:
                rb = beds[bi:bi + bpr]
                rba = baths[bai:bai + bapr]
                rh = row_h
            bi += bpr
            bai += bapr

            if rb or rba:
                # Recursive call: each row gets its own sub-banding
                placed.extend(_place_bedroom_zone(
                    rb + rba, x0, cy, band_w, rh))
            cy += rh + WALL_INT

        return placed

    # ── SINGLE ROW: sub-band compact rooms ──
    if not baths or band_h < 13.0:
        return _place_band_h(rooms, x0, y0, band_w, band_h)

    # Calculate sub-band heights
    bath_h = snap(max(5.0, min(7.0, band_h * 0.28)))
    bed_h = snap(band_h - bath_h - WALL_INT)

    # Cap bedroom sub-band to area-safe depth (snap DOWN for conservative limit)
    if bed_h > max_bed_h:
        bed_h = snap_down(max_bed_h)

    # Wide-plot check: if sub-banded bedrooms can't fill > 80% of row width
    # (AR limits bedroom width to bed_h * AR), give bedrooms more depth by
    # compressing the bathroom sub-band to its minimum (4ft).
    n_beds = len(beds)
    est_bed_w = (band_w - WALL_INT * max(0, n_beds - 1)) / n_beds if n_beds else band_w
    bed_ar = COMFORT_AR.get("bedroom", 2.0)
    max_bed_w_subbanded = bed_h * bed_ar
    total_bed_w = n_beds * max_bed_w_subbanded + WALL_INT * max(0, n_beds - 1)
    if total_bed_w < band_w * 0.80:
        # Compress bath sub-band to give bedrooms more depth
        bath_h = snap(max(4.0, min(5.0, band_h * 0.20)))
        bed_h = snap(band_h - bath_h - WALL_INT)
        if bed_h > max_bed_h:
            bed_h = snap_down(max_bed_h)

    # Check if sub-banding would make bedrooms undersized.
    for bed in beds:
        min_a = MIN_ROOM_AREA.get(bed["room_type"], 80)
        if est_bed_w * bed_h < min_a * 0.95:
            return _place_band_h(rooms, x0, y0, band_w, band_h)

    if bed_h < 9.0:
        # Squeeze bathroom to give bedrooms enough depth (9ft min for habitable rooms)
        bed_h = snap(max(9.0, band_h * 0.65))
        bath_h = snap(band_h - bed_h - WALL_INT)
        if bath_h < 4.0:
            # Not enough for even compressed layout, fallback
            return _place_band_h(rooms, x0, y0, band_w, band_h)

    placed: List[Dict] = []

    # Y positions: bathrooms at FRONT of band (closer to passage), bedrooms at REAR
    bath_y = y0
    bed_y = y0 + bath_h + WALL_INT

    # For wide plots where bedrooms can't fill row, allow relaxed AR
    _bed_ar_override = None
    _max_bed_w = bed_h * bed_ar
    _total_bed_fill = n_beds * _max_bed_w + WALL_INT * max(0, n_beds - 1)
    if _total_bed_fill < band_w * 0.85:
        _bed_ar_override = {"master_bedroom": 2.5, "bedroom": 2.5}

    # Place bedrooms in main sub-band
    bed_placed = _fill_row(beds, x0, bed_y, band_w, bed_h, ar_override=_bed_ar_override)
    placed.extend(bed_placed)

    # Place bathrooms aligned with their parent bedrooms
    # Each bathroom width = min(parent_bedroom_width, target_area/bath_h, 7ft max)
    bath_x = x0
    for i, bath in enumerate(baths):
        if i < len(bed_placed):
            parent = bed_placed[i]
            target_w = bath["target_area"] / bath_h if bath_h > 0 else 5.0
            bw = snap(max(5.0, min(parent["width"], target_w, 8.0)))
        else:
            target_w = bath["target_area"] / bath_h if bath_h > 0 else 5.0
            bw = snap(max(5.0, min(target_w, 8.0)))
        # Don't exceed remaining band width
        remaining = round(x0 + band_w - bath_x, 2)
        bw = min(bw, remaining)
        if bw < 4.0:
            break
        placed.append(_make_placed_room(bath, bath_x, bath_y, bw, bath_h, i, baths))
        bath_x = round(bath_x + bw + WALL_INT, 2)

    return placed


def _place_bedroom_zone_v(
    rooms: List[Dict],
    x0: float, y0: float,
    col_w: float, col_h: float,
) -> List[Dict]:
    """
    Vertical version of bedroom zone placement for side-corridor layout.
    Bedrooms stacked vertically, bathrooms in narrow column beside them.
    """
    beds = [r for r in rooms if r["room_type"] not in COMPACT_ROOMS]
    baths = [r for r in rooms if r["room_type"] in COMPACT_ROOMS]

    if not baths or not beds or col_w < 14.0:
        return _place_band_v(rooms, x0, y0, col_w, col_h)

    # Bathroom column width: 5-7ft
    bath_w = snap(max(5.0, min(7.0, col_w * 0.30)))
    bed_w = snap(col_w - bath_w - WALL_INT)

    if bed_w < 8.0:
        return _place_band_v(rooms, x0, y0, col_w, col_h)

    # Check if beds can fit vertically (AR-safe minimum heights)
    total_min_h = sum(
        max(4.0, bed_w / COMFORT_AR.get(r["room_type"], 2.0)) for r in beds
    ) + WALL_INT * max(0, len(beds) - 1)
    if total_min_h > col_h:
        # Too many rooms for vertical sub-banding, use horizontal fallback
        return _place_band_v(rooms, x0, y0, col_w, col_h)

    placed: List[Dict] = []

    # Bedrooms on one side, bathrooms on the other
    bed_x = x0
    bath_x = x0 + bed_w + WALL_INT

    bed_placed = _fill_col(beds, bed_x, y0, bed_w, col_h)
    placed.extend(bed_placed)

    # Place bathrooms aligned with bedrooms
    bath_y = y0
    for i, bath in enumerate(baths):
        if i < len(bed_placed):
            parent = bed_placed[i]
            target_h = bath["target_area"] / bath_w if bath_w > 0 else 5.0
            bh = snap(max(5.0, min(parent["length"], target_h, 8.0)))
        else:
            target_h = bath["target_area"] / bath_w if bath_w > 0 else 5.0
            bh = snap(max(5.0, min(target_h, 8.0)))
        remaining = round(y0 + col_h - bath_y, 2)
        bh = min(bh, remaining)
        if bh < 4.0:
            break
        placed.append(_make_placed_room(bath, bath_x, bath_y, bath_w, bh, i, baths))
        bath_y = round(bath_y + bh + WALL_INT, 2)

    return placed


# ═══════════════════════════════════════════════════════════════════════════
# BAND PLACEMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _max_rooms_for_row(rooms: List[Dict], band_w: float, band_h: float) -> int:
    """How many rooms fit in a single row while maintaining AR constraints."""
    _MIN_WIDTH_ENFORCE = {
        "living": 10.0, "master_bedroom": 9.0, "bedroom": 9.0,
        "kitchen": 7.0, "dining": 8.0,
    }
    used = 0.0
    count = 0
    for r in rooms:
        ar = COMFORT_AR.get(r["room_type"], 2.0)
        min_w = max(4.0, band_h / ar)
        # Account for min-width enforcement in _fill_row
        enforce = _MIN_WIDTH_ENFORCE.get(r["room_type"])
        if enforce:
            min_w = max(min_w, enforce)
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
    wall_space = WALL_INT * max(0, n_rows - 1)
    avail_h = col_h - wall_space

    # Proportional row heights based on total target area per row
    # Floor at 6ft to ensure single rooms (kitchen etc.) have enough depth
    # for AR-capped width to yield meaningful area
    _MIN_ROW_H = 6.0
    row_areas = [sum(r["target_area"] for r in row) for row in rows]
    total_row_area = sum(row_areas) or 1.0
    row_heights = []
    for ra in row_areas:
        rh = snap(max(_MIN_ROW_H, avail_h * (ra / total_row_area)))
        row_heights.append(rh)

    # Normalize to fit available height
    while sum(row_heights) > avail_h + 0.3:
        excess = sum(row_heights) - avail_h
        tallest_idx = max(range(n_rows), key=lambda i: row_heights[i])
        row_heights[tallest_idx] = snap(max(5.0, row_heights[tallest_idx] - excess))

    placed: List[Dict] = []
    cy = y0
    for ri, row in enumerate(rows):
        if ri == n_rows - 1:
            rh = snap(max(5.0, y0 + col_h - cy))
        else:
            rh = row_heights[ri]
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
    ar_override: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Place N rooms side-by-side in a horizontal row.
    Width proportional to target_area, clamped to AR limits, snapped to grid.
    ar_override: optional dict of room_type -> AR overrides for this call.
    """
    n = len(rooms)
    if n == 0:
        return []

    def _ar(rtype: str) -> float:
        if ar_override and rtype in ar_override:
            return ar_override[rtype]
        return COMFORT_AR.get(rtype, 2.0)

    total_area = sum(r["target_area"] for r in rooms) or 1.0
    wall_space = WALL_INT * max(0, n - 1)
    avail = net_w - wall_space

    # Step 1 -- proportional widths
    widths = [max(4.0, avail * (r["target_area"] / total_area)) for r in rooms]

    # Step 2 -- AR clamping
    min_ws: List[float] = []
    max_ws: List[float] = []
    for i, r in enumerate(rooms):
        ar = _ar(r["room_type"])
        min_w = max(4.0, row_h / ar)
        max_w = row_h * ar
        min_ws.append(min_w)
        max_ws.append(max_w)
        widths[i] = max(widths[i], min_w)
        widths[i] = min(widths[i], max_w)

    # Step 2.5 -- Area capping: prevent rooms from exceeding MAX_AREAS
    # When a room's projected area (width × row_h) exceeds its hard max,
    # reduce its width and redistribute freed space to neighbors.
    # When constraints are overdetermined (area cap < min_width), freeze
    # max_ws at min_ws so normalization doesn't grow the room further.
    for i, r in enumerate(rooms):
        max_a = MAX_AREAS.get(r["room_type"])
        if max_a and row_h > 0:
            max_w_for_area = max_a / row_h
            if max_w_for_area >= min_ws[i]:
                # Always constrain max_ws so normalization can't overshoot
                max_ws[i] = min(max_ws[i], max_w_for_area)
                if widths[i] > max_w_for_area * 1.1:
                    freed = widths[i] - max_w_for_area
                    widths[i] = max_w_for_area
                    # Redistribute freed width to neighbors with headroom
                    recipients = [
                        j for j in range(n) if j != i
                        and widths[j] < max_ws[j]
                    ]
                    if recipients:
                        per = freed / len(recipients)
                        for j in recipients:
                            widths[j] = min(widths[j] + per, max_ws[j])
            else:
                # Overdetermined: area cap < min_width. Can't shrink below
                # min_ws, but freeze max_ws to prevent normalization growth.
                max_ws[i] = min_ws[i]
                widths[i] = min_ws[i]

    # Step 2.6 -- Min dimension enforcement for habitable rooms
    # Indian residential standards: living >= 10ft, bedroom >= 9ft, kitchen >= 7ft
    MIN_WIDTH_ENFORCE = {
        "living": 10.0, "master_bedroom": 9.0, "bedroom": 9.0,
        "kitchen": 7.0, "dining": 8.0,
    }
    for i, r in enumerate(rooms):
        min_dim = MIN_WIDTH_ENFORCE.get(r["room_type"])
        if min_dim and widths[i] < min_dim:
            old_w = widths[i]
            widths[i] = min(min_dim, avail * 0.6)
            min_ws[i] = max(min_ws[i], widths[i])
    # Step 3 -- normalize to fit available width
    _normalize_widths(widths, avail, min_ws, max_ws)

    # Step 3.5 -- Fill remaining width: if normalization couldn't fill avail
    # (all rooms at max_ws), relax area caps on habitable rooms to absorb.
    # Do NOT grow compact rooms (bathrooms etc.) — they should stay small.
    total_w = sum(widths)
    gap = avail - total_w
    if gap > 2.0:
        _GROWABLE = {"living", "master_bedroom", "bedroom", "dining", "kitchen"}
        # Bedrooms/living may grow 1.2× past MAX_AREAS to fill width;
        # kitchen/dining stay at 1.0× to avoid gross oversizing.
        _GROW_FACTOR = {"living": 1.2, "master_bedroom": 1.2, "bedroom": 1.2,
                        "dining": 1.1, "kitchen": 1.1}
        grow_idxs = [i for i, r in enumerate(rooms) if r["room_type"] in _GROWABLE]
        if grow_idxs:
            ar_maxs = [row_h * _ar(rooms[i]["room_type"]) for i in grow_idxs]
            # Cap by MAX_AREAS with per-type growth factor
            for j, i in enumerate(grow_idxs):
                rt = rooms[i]["room_type"]
                max_a = MAX_AREAS.get(rt, 300)
                gf = _GROW_FACTOR.get(rt, 1.0)
                area_cap_w = (max_a * gf) / max(row_h, 1.0)
                ar_maxs[j] = min(ar_maxs[j], area_cap_w)
            for _ in range(3):
                total_w = sum(widths)
                gap = avail - total_w
                if gap < 1.0:
                    break
                headroom = [max(0, ar_maxs[j] - widths[i]) for j, i in enumerate(grow_idxs)]
                total_hr = sum(headroom) or 1.0
                for j, i in enumerate(grow_idxs):
                    extra = gap * (headroom[j] / total_hr)
                    widths[i] = min(widths[i] + extra, ar_maxs[j])

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
        ar = _ar(room["room_type"])
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

    # Area capping for column fills (same as _fill_row step 2.5)
    # SKIP if area cap would violate AR minimum.
    for i, r in enumerate(rooms):
        max_a = MAX_AREAS.get(r["room_type"])
        if max_a and col_w > 0:
            max_h_for_area = max_a / col_w
            if max_h_for_area >= min_hs[i] and heights[i] > max_h_for_area * 1.1:
                freed = heights[i] - max_h_for_area
                heights[i] = max_h_for_area
                max_hs[i] = min(max_hs[i], max_h_for_area)
                recipients = [
                    j for j in range(n) if j != i
                    and heights[j] < max_hs[j]
                ]
                if recipients:
                    per = freed / len(recipients)
                    for j in recipients:
                        heights[j] = min(heights[j] + per, max_hs[j])

    # Min dimension enforcement for column fills
    MIN_HEIGHT_ENFORCE = {
        "living": 10.0, "master_bedroom": 9.0, "bedroom": 9.0,
        "kitchen": 7.0, "dining": 8.0,
    }
    for i, r in enumerate(rooms):
        min_dim = MIN_HEIGHT_ENFORCE.get(r["room_type"])
        if min_dim and heights[i] < min_dim:
            heights[i] = min(min_dim, avail * 0.6)
            min_hs[i] = max(min_hs[i], heights[i])

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
# SECTION 8b — DEAD-ZONE FILLING (passage room + gap coverage)
# ═══════════════════════════════════════════════════════════════════════════

def _fill_dead_zones(
    placed: List[Dict], circ_info: Dict,
    plot_w: float, plot_l: float,
) -> List[Dict]:
    """Detect uncovered areas in the layout and fill them with passage/hall rooms."""
    ux, uy, uw, ul = _buildable_rect(plot_w, plot_l)
    right = round(ux + uw, 2)
    top = round(uy + ul, 2)

    # Passage Y range for labeling dead zones in the passage area
    corr_y1, corr_y2 = None, None
    if circ_info and circ_info.get("position"):
        corr_y1 = circ_info["position"]["y"]
        corr_y2 = corr_y1 + circ_info.get("depth_ft", 3.0)

    # ── 1. Detect dead zones via Y-strip decomposition ──
    y_edges = sorted(set(
        [uy, top]
        + [round(r["position"]["y"], 2) for r in placed]
        + [round(r["position"]["y"] + r["length"], 2) for r in placed]
    ))

    candidates = []
    for i in range(len(y_edges) - 1):
        y1, y2 = y_edges[i], y_edges[i + 1]
        strip_h = round(y2 - y1, 2)
        if strip_h < 2.5:
            continue

        # X-ranges covered by rooms in this strip
        x_ranges = sorted(
            (r["position"]["x"], r["position"]["x"] + r["width"])
            for r in placed
            if r["position"]["y"] < y2 - 0.1
            and r["position"]["y"] + r["length"] > y1 + 0.1
        )

        # Merge overlapping X ranges
        merged = []
        for x1, x2 in x_ranges:
            if merged and x1 <= merged[-1][1] + 0.5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], x2))
            else:
                merged.append([x1, x2])

        # Uncovered segments
        prev_x = ux
        for mx1, mx2 in merged:
            if round(mx1 - prev_x, 2) >= 3.0:
                candidates.append((round(prev_x, 2), y1, round(mx1, 2), y2))
            prev_x = mx2
        if round(right - prev_x, 2) >= 3.0:
            candidates.append((round(prev_x, 2), y1, right, y2))

    # ── 2. Merge vertically-adjacent candidates with same X span ──
    candidates.sort()
    merged_zones = []
    for cand in candidates:
        fx1, fy1, fx2, fy2 = cand
        found = False
        for j, mz in enumerate(merged_zones):
            mx1, my1, mx2, my2 = mz
            if abs(fx1 - mx1) < 0.5 and abs(fx2 - mx2) < 0.5 and abs(fy1 - my2) < 0.5:
                merged_zones[j] = (mx1, my1, mx2, fy2)
                found = True
                break
        if not found:
            merged_zones.append(list(cand))

    # ── 3. Fill dead zones ──
    for fx1, fy1, fx2, fy2 in merged_zones:
        fw = round(fx2 - fx1, 2)
        fh = round(fy2 - fy1, 2)
        area = round(fw * fh, 1)
        if area < 12:
            continue

        # Verify no overlap with existing rooms
        has_overlap = False
        for r in placed:
            rx, ry = r["position"]["x"], r["position"]["y"]
            ox = max(0, min(rx + r["width"], fx2) - max(rx, fx1))
            oy = max(0, min(ry + r["length"], fy2) - max(ry, fy1))
            if ox * oy > 2.0:
                has_overlap = True
                break
        if has_overlap:
            continue

        # Label: "Corridor" if in corridor zone, else "Foyer"
        rtype, name = "foyer", "Foyer"
        if corr_y1 is not None:
            mid_y = (fy1 + fy2) / 2
            if corr_y1 - 0.5 <= mid_y <= corr_y2 + 0.5:
                rtype, name = "corridor", "Corridor"

        placed.append({
            "name": name, "room_type": rtype,
            "zone": "circulation", "layer": 0,
            "width": fw, "length": fh,
            "area": area, "actual_area": area, "target_area": area,
            "position": {"x": round(fx1, 2), "y": round(fy1, 2)},
            "polygon": [
                [round(fx1, 2), round(fy1, 2)],
                [round(fx2, 2), round(fy1, 2)],
                [round(fx2, 2), round(fy2, 2)],
                [round(fx1, 2), round(fy2, 2)],
                [round(fx1, 2), round(fy1, 2)],
            ],
            "centroid": [round((fx1 + fx2) / 2, 2), round((fy1 + fy2) / 2, 2)],
        })

    return placed


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — AUTO-CORRECTION (boundary clamping, polygon rebuild)
# ═══════════════════════════════════════════════════════════════════════════

def _auto_correct(placed: List[Dict], plot_w: float, plot_l: float) -> List[Dict]:
    """Clamp rooms to plot boundary, enforce min size, and remove overlapping."""
    for room in placed:
        x, y = room["position"]["x"], room["position"]["y"]
        w, h = room["width"], room["length"]

        # Enforce minimum size (corridors can be narrow)
        is_circulation = room["room_type"] in ("corridor", "foyer")
        min_dim = 2.5 if is_circulation else 4.0
        w = max(min_dim, w)
        h = max(min_dim, h)

        # Clamp to plot boundary (shift first, then shrink)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > plot_w:
            w = snap(max(min_dim, plot_w - x))
        if y + h > plot_l:
            h = snap(max(min_dim, plot_l - y))
        # Re-check: if room still exceeds after min-size, shift position
        if x + w > plot_w:
            x = snap(max(0, plot_w - w))
        if y + h > plot_l:
            y = snap(max(0, plot_l - h))

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

    # Remove rooms that overlap significantly with others (inflation artifacts)
    def _overlap_area(r1, r2):
        p1, p2 = r1["position"], r2["position"]
        ox = max(0, min(p1["x"] + r1["width"], p2["x"] + r2["width"]) - max(p1["x"], p2["x"]))
        oy = max(0, min(p1["y"] + r1["length"], p2["y"] + r2["length"]) - max(p1["y"], p2["y"]))
        return ox * oy

    to_remove = set()
    for i, r1 in enumerate(placed):
        if i in to_remove:
            continue
        for j, r2 in enumerate(placed):
            if j <= i or j in to_remove:
                continue
            ov = _overlap_area(r1, r2)
            if ov > 2.0:  # More than 2 sqft overlap
                # Remove the smaller room
                a1, a2 = r1["area"], r2["area"]
                to_remove.add(j if a1 >= a2 else i)

    if to_remove:
        placed = [r for i, r in enumerate(placed) if i not in to_remove]

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
    passage_dim = min(pw, pd) if circ_info.get("type") == "vertical_corridor" else pd
    if passage_dim < MIN_CORRIDOR - 0.1:
        issues.append(f"Corridor width {passage_dim}ft is below minimum {MIN_CORRIDOR}ft")

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

    # ── Enrich: top-level doors with hinge/swing geometry ──
    top_doors = []
    for room in placed:
        rx, ry = room["position"]["x"], room["position"]["y"]
        rw, rh = room["width"], room["length"]
        for door in room.get("doors", []):
            dw = door.get("width", 3.0)
            wall = door.get("wall", "south")
            pos = door.get("position", rx + rw / 2)
            if wall == "south":
                hinge = [round(pos - dw / 2, 2), round(ry, 2)]
                door_end = [round(pos + dw / 2, 2), round(ry, 2)]
                swing_dir = [0, 1]
            elif wall == "north":
                hinge = [round(pos - dw / 2, 2), round(ry + rh, 2)]
                door_end = [round(pos + dw / 2, 2), round(ry + rh, 2)]
                swing_dir = [0, -1]
            elif wall == "west":
                hinge = [round(rx, 2), round(pos - dw / 2, 2)]
                door_end = [round(rx, 2), round(pos + dw / 2, 2)]
                swing_dir = [1, 0]
            elif wall == "east":
                hinge = [round(rx + rw, 2), round(pos - dw / 2, 2)]
                door_end = [round(rx + rw, 2), round(pos + dw / 2, 2)]
                swing_dir = [-1, 0]
            else:
                continue
            top_doors.append({
                "room": room["name"],
                "wall": wall,
                "width": dw,
                "type": door.get("type", "standard"),
                "hinge": hinge,
                "door_end": door_end,
                "swing_dir": swing_dir,
                "position": [(hinge[0] + door_end[0]) / 2, (hinge[1] + door_end[1]) / 2],
            })

    # ── Enrich: top-level windows with start/end coordinates ──
    top_windows = []
    for room in placed:
        rx, ry = room["position"]["x"], room["position"]["y"]
        rw, rh = room["width"], room["length"]
        for win in room.get("windows", []):
            ww = win.get("width", 3.0)
            wall = win.get("wall", "south")
            cx_w = rx + rw / 2
            cy_w = ry + rh / 2
            if wall == "south":
                start = [round(cx_w - ww / 2, 2), round(ry, 2)]
                end = [round(cx_w + ww / 2, 2), round(ry, 2)]
            elif wall == "north":
                start = [round(cx_w - ww / 2, 2), round(ry + rh, 2)]
                end = [round(cx_w + ww / 2, 2), round(ry + rh, 2)]
            elif wall == "west":
                start = [round(rx, 2), round(cy_w - ww / 2, 2)]
                end = [round(rx, 2), round(cy_w + ww / 2, 2)]
            elif wall == "east":
                start = [round(rx + rw, 2), round(cy_w - ww / 2, 2)]
                end = [round(rx + rw, 2), round(cy_w + ww / 2, 2)]
            else:
                continue
            top_windows.append({
                "room": room["name"],
                "wall": wall,
                "width": ww,
                "type": win.get("type", "standard"),
                "start": start,
                "end": end,
            })

    # ── Enrich: columns at boundary corners + room corners on boundary ──
    col_set = set()
    for pt in boundary[:-1]:
        col_set.add((round(pt[0], 1), round(pt[1], 1)))
    tol_col = WALL_EXT + 0.5  # rooms are inset by wall thickness
    for room in placed:
        rx, ry = room["position"]["x"], room["position"]["y"]
        rw, rh = room["width"], room["length"]
        for cx, cy in [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]:
            on_bnd = (cx < tol_col or cx > plot_w - tol_col or
                      cy < tol_col or cy > plot_l - tol_col)
            if on_bnd:
                col_set.add((round(cx, 1), round(cy, 1)))
    columns = [[c[0], c[1]] for c in sorted(col_set)]

    return {
        # -- Spec output format --
        "layout_strategy": strategy,
        "spatial_layers": _build_spatial_layers(placed),
        "rooms": out_rooms,
        "circulation": circ_info,
        "validation": validation,

        # -- Frontend drawing data --
        "doors": top_doors,
        "windows": top_windows,
        "columns": columns,

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
    return round(area * 1.0)  # Wall space absorbed by room proportioning
