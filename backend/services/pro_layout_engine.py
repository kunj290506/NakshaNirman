"""
Professional Layout Engine — Architect-Grade Residential Floor Plan Generator.

Implements the 5-phase professional architectural workflow that mirrors how
real architects design homes:

  Phase 1: ROOM PROGRAMMING
    - Classify rooms into zones (public/semi-private/private/service)
    - Set target areas proportional to total plot area
    - Build adjacency requirements graph (required/preferred/forbidden pairs)
    - Pair attached bathrooms with their parent bedrooms

  Phase 2: ZONE-BAND PLANNING
    - Divide plot into horizontal bands: PUBLIC → CORRIDOR → PRIVATE
    - PUBLIC band (front/road-facing): Living, Dining, Kitchen
    - CORRIDOR band (3.5-4.5ft): separates public/private, ensures circulation
    - PRIVATE band (back): Bedrooms + attached bathrooms + service rooms
    - Adapt orientation for tall/narrow plots (vertical bands instead)

  Phase 3: CONSTRAINT-BASED PLACEMENT
    - Place rooms within each band using adjacency-aware ordering
    - Enforce minimum dimensions and max aspect ratios per room type
    - Cluster wet rooms (kitchen, bathrooms) on shared plumbing walls
    - Snap all dimensions to 0.5ft structural grid
    - Handle attached bathrooms as sub-cells within the private band

  Phase 4: MULTI-CANDIDATE SCORING
    - Generate 6-8 candidate layouts with different room orderings
    - Score each on: adjacency satisfaction, room proportions, Vastu compliance,
      natural light access, plumbing efficiency, circulation quality
    - Return the best-scoring candidate

  Phase 5: DOOR & WINDOW PLACEMENT
    - Doors placed on actual shared walls between adjacent rooms
    - Main entrance on road-facing wall of living room
    - Attached bathroom doors from bedroom side only
    - Windows on external walls (large for habitable, small for wet rooms)

Produces layouts that match real professional Indian residential plans
compliant with NBC 2016 and Vastu Shastra principles.

Author: CAD Professional Layout Engine v2.0
"""

import math
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from copy import deepcopy
from itertools import permutations

from services.layout_constants import (
    GRID_SNAP, WALL_EXTERNAL_FT as WALL_EXT, WALL_INTERNAL_FT as WALL_INT,
    MIN_DIMS, MAX_ASPECT, AREA_FRACTIONS, MIN_AREAS, MAX_AREAS,
    ZONE_MAP, PRIORITY, VASTU_PREFS,
    DESIRED_ADJACENCIES as DESIRED_ADJ,
    FORBIDDEN_ADJACENCIES as FORBIDDEN_ADJ,
)

logger = logging.getLogger(__name__)

# BHK-aware scaling: reduce per-room fractions when many rooms compete for space
def _get_scaled_fraction(rtype: str, total_area: float, n_rooms: int) -> float:
    """Get ideal area fraction scaled by total room count and plot size.
    
    Uses BHK-aware heuristics to ensure:
    - Living room always gets a generous share (the "hub" of the home)
    - Bedrooms maintain minimum usable sizes regardless of room count
    - Service rooms (bath/utility) don't steal from habitable rooms
    - Small plots prioritize living + kitchen over extras
    """
    frac = AREA_FRACTIONS.get(rtype, (0.04, 0.06, 0.08))
    lo, ideal, hi = frac
    
    # For larger homes (more rooms), reduce each room's share to avoid overflow
    if n_rooms > 6:
        # Use a gentler curve — sqrt-based scaling preserves room quality
        scale = max(0.72, math.sqrt(6.0 / n_rooms))
        ideal *= scale
    
    # BHK-aware bumps: ensure bedrooms are NEVER starved for area.
    # Architect's rule: bedrooms need MORE space than living (private vs shared).
    if total_area < 400:
        # Tiny homes (1BHK): cap living, maximize bedroom
        if rtype in ('master_bedroom', 'bedroom'):
            ideal = max(ideal, 0.22)   # bedroom is the main room
        elif rtype == 'living':
            ideal = min(ideal, 0.12)   # compact living, not a grand hall
        elif rtype == 'kitchen':
            ideal = max(ideal, 0.10)
    elif total_area < 600:
        if rtype in ('master_bedroom', 'bedroom'):
            ideal = max(ideal, 0.17)   # bedrooms get priority in small homes
        elif rtype == 'living':
            ideal = min(ideal, 0.14)   # living is adequate at 14%
        elif rtype == 'kitchen':
            ideal = max(ideal, 0.10)
    elif total_area < 900:
        if rtype in ('master_bedroom',):
            ideal = max(ideal, 0.17)   # master bedroom needs good size
        elif rtype in ('bedroom',):
            ideal = max(ideal, 0.14)   # regular bedrooms too
    
    # Cap service rooms so they don't eat into main rooms on large plots
    if total_area > 1500 and rtype in ('bathroom', 'toilet', 'utility', 'store'):
        ideal = min(ideal, hi)
    
    return ideal


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _snap(val: float) -> float:
    """Snap to nearest GRID_SNAP increment."""
    return round(val / GRID_SNAP) * GRID_SNAP


def _snap_down(val: float) -> float:
    """Snap down to nearest GRID_SNAP."""
    return math.floor(val / GRID_SNAP) * GRID_SNAP


def _snap_up(val: float) -> float:
    """Snap up to nearest GRID_SNAP."""
    return math.ceil(val / GRID_SNAP) * GRID_SNAP


def _aspect(w: float, h: float) -> float:
    """Compute aspect ratio (always ≥ 1.0)."""
    if w <= 0 or h <= 0:
        return float('inf')
    return max(w / h, h / w)


def _max_ar(rtype: str) -> float:
    """Get max allowed aspect ratio for a room type."""
    return MAX_ASPECT.get(rtype, 2.0)


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def _rooms_share_wall(a: Dict, b: Dict, min_overlap: float = 2.5) -> bool:
    """Check if two rooms share a wall segment of at least min_overlap ft."""
    ax, ay = a['x'], a['y']
    aw, ah = a['w'], a['h']
    bx, by = b['x'], b['y']
    bw, bh = b['w'], b['h']
    tol = 0.5

    # Check vertical shared wall (side-by-side)
    if abs((ax + aw) - bx) < tol or abs((bx + bw) - ax) < tol:
        overlap = min(ay + ah, by + bh) - max(ay, by)
        if overlap >= min_overlap:
            return True

    # Check horizontal shared wall (top-bottom)
    if abs((ay + ah) - by) < tol or abs((by + bh) - ay) < tol:
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap >= min_overlap:
            return True

    return False


def _get_quadrant(rx, ry, rw, rh, plot_cx, plot_cy) -> str:
    """Determine Vastu quadrant from room center position."""
    cx = rx + rw / 2
    cy = ry + rh / 2
    if cx <= plot_cx:
        return 'NW' if cy >= plot_cy else 'SW'
    else:
        return 'NE' if cy >= plot_cy else 'SE'


# =============================================================================
# PHASE 1: ROOM PROGRAMMING
# =============================================================================

def _build_room_program(
    rooms_config: Dict[str, int],
    total_area: float,
) -> Tuple[List[Dict], List[Tuple]]:
    """
    Build structured room program from user configuration.

    Returns:
        rooms: list of room dicts with type, name, zone, target_area, priority
        attached_pairs: list of (bedroom_idx, bathroom_idx) pairs
    """
    rooms = []

    # Count total rooms (living is always added separately, so only add +1
    # if 'living' is NOT already in the config to avoid double-counting)
    n_total_rooms = sum(rooms_config.values())
    if 'living' not in rooms_config:
        n_total_rooms += 1

    def _make_room(rtype, name, count_label=''):
        ideal = _get_scaled_fraction(rtype, total_area, n_total_rooms)
        target = ideal * total_area
        min_area = MIN_AREAS.get(rtype, 20)
        # For compact plots (< 600sqft), scale down minimum areas
        # proportionally — you can't have a 100sqft living room in
        # a 300sqft home. This prevents minima from dominating small layouts.
        if total_area < 600:
            area_scale = max(0.5, total_area / 600)
            min_area = round(min_area * area_scale, 1)
        target = max(target, min_area)
        mx = MAX_AREAS.get(rtype)
        if mx:
            target = min(target, mx)
        return {
            'room_type': rtype,
            'name': name,
            'zone': ZONE_MAP.get(rtype, 'private'),
            'target_area': round(target, 1),
            'priority': PRIORITY.get(rtype, 30),
            '_attached_to': None,
            '_is_attached_bath': False,
        }

    # --- Living Room (always exactly 1) ---
    rooms.append(_make_room('living', 'Drawing Room'))

    # --- Master Bedrooms ---
    n_master = rooms_config.get('master_bedroom', 0)
    master_indices = []
    for i in range(n_master):
        label = 'Master Bed Room' if n_master == 1 else f'Master Bed Room {i+1}'
        rooms.append(_make_room('master_bedroom', label))
        master_indices.append(len(rooms) - 1)

    # --- Regular Bedrooms ---
    n_bed = rooms_config.get('bedroom', 0)
    bed_indices = []
    for i in range(n_bed):
        label = f'Bed Room {i+1}'
        rooms.append(_make_room('bedroom', label))
        bed_indices.append(len(rooms) - 1)

    # --- Kitchen ---
    for i in range(rooms_config.get('kitchen', 1)):
        rooms.append(_make_room('kitchen', 'Kitchen'))

    # --- Dining ---
    for i in range(rooms_config.get('dining', 0)):
        rooms.append(_make_room('dining', 'Dining Area'))

    # --- Bathrooms ---
    # Auto-attach one bathroom per master bedroom
    n_bath = rooms_config.get('bathroom', 0)
    bath_indices = []
    for i in range(n_bath):
        if i == 0 and n_bath > 1:
            label = 'Wash Area'
        elif n_bath == 1:
            label = 'Bathroom'
        else:
            label = f'Bath {i+1}'
        rooms.append(_make_room('bathroom', label))
        bath_indices.append(len(rooms) - 1)

    # --- Pair attached bathrooms with master bedrooms ---
    attached_pairs = []
    remaining_bath = list(bath_indices)
    for mi in master_indices:
        if remaining_bath:
            bi = remaining_bath.pop(0)
            rooms[bi]['_attached_to'] = mi
            rooms[bi]['_is_attached_bath'] = True
            rooms[bi]['name'] = 'Attached Bath'
            attached_pairs.append((mi, bi))

    # --- Toilets ---
    for i in range(rooms_config.get('toilet', 0)):
        rooms.append(_make_room('toilet', 'Toilet'))

    # --- Extra rooms ---
    for rtype in ('study', 'pooja', 'store', 'utility', 'staircase', 'balcony', 'garage'):
        for i in range(rooms_config.get(rtype, 0)):
            name = rtype.replace('_', ' ').title()
            if rtype == 'pooja':
                name = 'Puja Room'
            elif rtype == 'store':
                name = 'Store Room'
            rooms.append(_make_room(rtype, name))

    return rooms, attached_pairs


# =============================================================================
# PHASE 2: ZONE-BAND PLANNING
# =============================================================================

def _determine_corridor(total_area: float, n_private_rooms: int) -> float:
    """Determine corridor width based on plot size and room count.
    
    Architect's rule: corridor should be MINIMAL — just enough for
    circulation (3ft = one person, 3.5ft = comfortable). Every extra
    foot of corridor steals from bedrooms. A 4ft+ corridor in a
    home under 1500sqft is wasteful.
    """
    if total_area < 700 or n_private_rooms <= 1:
        return 0.0       # Compact homes: no corridor, living connects directly
    elif total_area < 1000:
        return 3.0        # 2BHK: narrow passage
    elif total_area < 1500:
        return 2.5        # 3BHK: efficient passage (saves ~10sqft vs 3ft)
    else:
        return 3.5        # 4BHK+: comfortable passage


def _assign_to_bands(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    total_area: float,
) -> Tuple[List[Dict], List[Dict], List[Dict], float]:
    """
    Assign rooms to PUBLIC, PRIVATE, and SERVICE bands.

    Returns:
        public_rooms: rooms for the front/road-facing band
        private_rooms: rooms for the back/private band (bedrooms + attached baths)
        service_rooms: extra service rooms to integrate into private band
        corridor_h: corridor height (0 if not needed)
    """
    public_rooms = []
    private_rooms = []
    service_rooms = []

    attached_bath_indices = set()
    parent_bed_indices = set()
    for (pi, bi) in attached_pairs:
        attached_bath_indices.add(bi)
        parent_bed_indices.add(pi)

    for idx, room in enumerate(rooms):
        rtype = room['room_type']
        zone = room['zone']

        if rtype in ('living', 'kitchen', 'dining'):
            public_rooms.append(room)
        elif rtype == 'garage':
            # Garage goes to service band — NOT in the main public band
            # This prevents it from squeezing living/kitchen/dining
            service_rooms.append(room)
        elif rtype in ('master_bedroom', 'bedroom', 'study'):
            private_rooms.append(room)
        elif idx in attached_bath_indices:
            # Attached bathroom — stays with its parent bedroom in private band
            private_rooms.append(room)
        elif rtype in ('bathroom', 'toilet'):
            # Common bathroom — goes to service, placed in private band
            service_rooms.append(room)
        elif rtype == 'pooja':
            # Pooja room — goes in private band (sacred, quiet)
            private_rooms.append(room)
        elif rtype in ('balcony',):
            # Balcony — attached to public zone
            public_rooms.append(room)
        elif rtype in ('store', 'utility', 'staircase'):
            service_rooms.append(room)
        else:
            # Default: assign based on zone
            if zone in ('public', 'semi_private'):
                public_rooms.append(room)
            else:
                service_rooms.append(room)

    n_private = len([r for r in private_rooms
                     if r['room_type'] in ('master_bedroom', 'bedroom')])
    corridor_h = _determine_corridor(total_area, n_private)

    return public_rooms, private_rooms, service_rooms, corridor_h


# =============================================================================
# PHASE 3: CONSTRAINT-BASED PLACEMENT
# =============================================================================

def _allocate_band_heights(
    public_rooms: List[Dict],
    private_rooms: List[Dict],
    service_rooms: List[Dict],
    corridor_h: float,
    usable_h: float,
    usable_w: float,
    total_area: float,
) -> Tuple[float, float]:
    """
    Calculate optimal heights for public and private bands.

    Uses room target areas and minimum dimension requirements to determine
    how much vertical space each band needs.

    Returns:
        public_h: height of the public (front) band
        private_h: height of the private (back) band
    """
    available_h = usable_h - corridor_h

    # Calculate ideal heights from target areas
    pub_area = sum(r['target_area'] for r in public_rooms) or 100
    priv_area = sum(r['target_area'] for r in private_rooms) or 100
    serv_area = sum(r['target_area'] for r in service_rooms) or 0
    priv_area += serv_area  # Service rooms go into private band

    total_room_area = pub_area + priv_area
    pub_ratio = pub_area / total_room_area
    priv_ratio = priv_area / total_room_area

    public_h = available_h * pub_ratio
    private_h = available_h * priv_ratio

    # --- CAP public band height ---
    # Architect's rule: public band should NEVER be deeper than private band
    # in homes with 2+ bedrooms. Living rooms don't need 14ft depth — a
    # 10-12ft depth is generous for any living room.
    PUB_MAX_H = min(10.0 + max(0, (total_area - 1200) * 0.003), 14.0)
    if public_h > PUB_MAX_H:
        excess = public_h - PUB_MAX_H
        public_h = PUB_MAX_H
        private_h += excess

    # --- CAP private band height ---
    # Small plots: max 22ft.  Large plots: scale up to accommodate more rooms.
    PRIV_MAX_H = min(22.0 + max(0, (total_area - 1500) * 0.005), 32.0)

    # Also cap based on bedroom count to prevent AR violations.
    # Private band = bedroom_sub + service_sub.
    # Service sub uses adaptive height: min(6.0, max(5.0, band_h * 0.35))
    # Each bedroom gets width ≈ usable_w / n_beds (with 0.85 safety for unequal sizes).
    # Ensure bedroom AR ≤ 1.8 → bedroom_h ≤ min_bed_w * 1.8
    n_beds_priv = len([r for r in private_rooms
                       if r['room_type'] in ('master_bedroom', 'bedroom')])
    if n_beds_priv >= 2:
        svc_sub_h = 5.5  # adaptive service sub-band midpoint (range 5.0-6.0)
        min_bed_w = (usable_w / n_beds_priv) * 0.85  # safety margin
        max_bedroom_sub_h = min_bed_w * 1.8  # AR limit
        ar_cap = max_bedroom_sub_h + svc_sub_h
        PRIV_MAX_H = min(PRIV_MAX_H, ar_cap)

    if private_h > PRIV_MAX_H:
        private_h = PRIV_MAX_H

    # Enforce minimum heights based on room dimension requirements
    # For compact homes (< 500sqft), relax minimums to prevent public
    # band from stealing depth from private rooms.
    base_min_h = 8.0 if total_area < 500 else 10.0
    pub_min_h = base_min_h
    for r in public_rooms:
        min_d = MIN_DIMS.get(r['room_type'], (7, 7))
        pub_min_h = max(pub_min_h, min(min_d[1], PUB_MAX_H))
    # For tiny plots, don't let pub_min dominate
    if total_area < 500:
        pub_min_h = min(pub_min_h, available_h * 0.45)

    priv_min_h = base_min_h
    for r in private_rooms:
        if r['room_type'] in ('master_bedroom', 'bedroom'):
            min_d = MIN_DIMS.get(r['room_type'], (9, 9))
            priv_min_h = max(priv_min_h, min_d[1])

    # Check if we have bedrooms that need more depth
    n_private_major = len([r for r in private_rooms
                           if r['room_type'] in ('master_bedroom', 'bedroom', 'study')])
    n_attached_baths = len([r for r in private_rooms if r.get('_is_attached_bath')])

    # If attached bathrooms are in the private band, we may need sub-bands
    # Private band = bedroom sub-band + service sub-band
    if n_attached_baths > 0 and n_private_major > 0:
        # Need enough height for bedroom + bathroom side-by-side or stacked
        # Preferred: bathroom next to bedroom (horizontal), not stacked
        # So private_h just needs to accommodate bedrooms
        priv_min_h = max(priv_min_h, 10.0)

    public_h = max(public_h, pub_min_h)
    private_h = max(private_h, priv_min_h)

    # Scale to fit available height — but RESPECT the caps
    total_needed = public_h + private_h
    if total_needed > available_h:
        scale = available_h / total_needed
        public_h *= scale
        private_h *= scale
    elif total_needed < available_h:
        # Distribute extra space proportionally BUT respect caps
        extra = available_h - total_needed
        pub_extra = min(extra * pub_ratio, PUB_MAX_H - public_h)
        pub_extra = max(pub_extra, 0)
        priv_extra = min(extra * priv_ratio, PRIV_MAX_H - private_h)
        priv_extra = max(priv_extra, 0)
        public_h += pub_extra
        private_h += priv_extra
        # Any remaining space stays as unused back of plot (garden/setback)

    # Final snap
    public_h = _snap(public_h)
    private_h = _snap(private_h)

    return public_h, private_h


def _order_public_rooms(rooms: List[Dict], plot_w: float) -> List[Dict]:
    """
    Order public rooms for the front band.

    Professional ordering (left to right for Indian homes):
      Living Room → Kitchen → Dining → (Balcony if any)

    Living room is the HUB — placed FIRST (leftmost, near entrance).
    Kitchen is adjacent to Living (direct access from living room).
    Dining goes after Kitchen (cooking→serving flow, accessed from kitchen).
    This ensures: Living ↔ Kitchen ↔ Dining adjacency chain.
    """
    kitchen = [r for r in rooms if r['room_type'] == 'kitchen']
    dining = [r for r in rooms if r['room_type'] == 'dining']
    living = [r for r in rooms if r['room_type'] == 'living']
    balcony = [r for r in rooms if r['room_type'] == 'balcony']
    garage = [r for r in rooms if r['room_type'] == 'garage']
    other = [r for r in rooms if r['room_type'] not in
             ('kitchen', 'dining', 'living', 'balcony', 'garage')]

    # Check if total minimum widths exceed available band width — drop optional rooms
    all_rooms = living + kitchen + dining + other + balcony + garage
    total_min_w = sum(MIN_DIMS.get(r['room_type'], (5, 5))[0] for r in all_rooms)
    if total_min_w > plot_w and dining:
        # Not enough width for all rooms — merge dining into living
        # (Dining becomes part of the living/drawing room, common in small plans)
        dining = []

    # Living FIRST → then Kitchen → then Dining → extras
    # This guarantees Kitchen shares a wall with Living Room
    ordered = living + kitchen + dining + other + balcony + garage
    return ordered


def _order_private_rooms(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    all_rooms: List[Dict],
    plot_w: float,
) -> List[Dict]:
    """
    Order private rooms for the back band.

    Professional ordering (left to right):
      AttBath1 | MasterBR | BR2 | BR3 | CommonBath/Study/Pooja

    Attached bathroom goes NEXT TO its parent bedroom.
    Master bedroom at the LEFT (SW corner for Vastu).
    Common bathrooms and service rooms at the RIGHT edge.
    Study/pooja at the far end of the band.

    The key insight: bathroom next to bedroom (side-by-side in same band),
    NOT above/below in separate sub-bands. This gives better proportions
    and a more logical layout.
    """
    masters = [r for r in rooms if r['room_type'] == 'master_bedroom']
    bedrooms = [r for r in rooms if r['room_type'] == 'bedroom']
    attached_baths = [r for r in rooms if r.get('_is_attached_bath')]
    common_baths = [r for r in rooms
                    if r['room_type'] in ('bathroom', 'toilet')
                    and not r.get('_is_attached_bath')]
    studies = [r for r in rooms if r['room_type'] == 'study']
    poojas = [r for r in rooms if r['room_type'] == 'pooja']
    other = [r for r in rooms if r['room_type'] not in
             ('master_bedroom', 'bedroom', 'bathroom', 'toilet',
              'study', 'pooja')]

    # Build ordered list: Master + AttBath pairs, then regular beds, then service
    ordered = []

    # Pair each master with its attached bathroom
    attached_map = {}
    for pair in attached_pairs:
        parent_idx, bath_idx = pair
        parent_room = all_rooms[parent_idx]
        bath_room = all_rooms[bath_idx]
        attached_map[id(parent_room)] = bath_room

    for m in masters:
        att = attached_map.get(id(m))
        if att:
            ordered.append(att)   # Attached bath LEFT of master (plumbing side)
        ordered.append(m)

    for b in bedrooms:
        ordered.append(b)

    # Common bathrooms go next to regular bedrooms
    for cb in common_baths:
        ordered.append(cb)

    # Study and pooja at the end
    ordered.extend(studies)
    ordered.extend(poojas)
    ordered.extend(other)

    return ordered


def _place_rooms_in_band(
    rooms: List[Dict],
    band_x: float,
    band_y: float,
    band_w: float,
    band_h: float,
) -> List[Dict]:
    """
    Place rooms within a horizontal band using constraint-aware allocation.

    Algorithm:
    1. Calculate target widths proportional to target_area / band_h
    2. Enforce minimum widths per room type
    3. Cap bathroom/service room widths to prevent bloat
    4. Distribute remaining space to major rooms
    5. Snap all positions to structural grid
    6. Verify aspect ratios and adjust if needed

    Returns:
        List of room dicts with '_placed' positions set.
    """
    if not rooms:
        return []

    n = len(rooms)
    placed = []

    # Step 1: Calculate ideal widths from target areas
    ideal_widths = []
    for r in rooms:
        ideal_w = r['target_area'] / band_h if band_h > 0 else 7.0
        ideal_widths.append(ideal_w)

    # Step 2: Enforce minimum widths
    for i, r in enumerate(rooms):
        min_w = MIN_DIMS.get(r['room_type'], (5, 5))[0]
        ideal_widths[i] = max(ideal_widths[i], min_w)

    # Step 3: Cap service room widths to prevent them from eating bedroom space
    for i, r in enumerate(rooms):
        rtype = r['room_type']
        if rtype in ('bathroom', 'toilet'):
            ideal_widths[i] = min(ideal_widths[i], 8.0)
        elif rtype in ('pooja', 'store', 'utility'):
            ideal_widths[i] = min(ideal_widths[i], 7.0)
        elif rtype == 'living':
            # Architect's rule: living room should not exceed 50% of band
            # width on compact plots — prevents it from dominating the layout
            max_living_w = band_w * 0.50
            ideal_widths[i] = min(ideal_widths[i], max_living_w)

    # Step 4: Enforce maximum aspect ratios
    for i, r in enumerate(rooms):
        max_ar = _max_ar(r['room_type'])
        # If width/height ratio exceeds max, increase width
        if band_h / ideal_widths[i] > max_ar:
            ideal_widths[i] = band_h / max_ar
        # If height/width ratio exceeds max, decrease width (or accept)
        if ideal_widths[i] / band_h > max_ar:
            # Room is too wide for the band height — OK for corridors, cap others
            if r['room_type'] not in ('corridor', 'balcony'):
                ideal_widths[i] = band_h * max_ar

    # Step 5: Scale widths to fit available band width
    # KEY INSIGHT: Don't over-scale service rooms (bathrooms, toilets, etc.)
    # Major rooms (living, bedroom) should absorb extra space.
    total_ideal = sum(ideal_widths)
    if total_ideal <= 0:
        total_ideal = 1

    if abs(total_ideal - band_w) > 0.5:
        # Classify rooms as "scalable" (major) vs "fixed-max" (service)
        SERVICE_TYPES = ('bathroom', 'toilet', 'pooja', 'store', 'utility')
        MAX_SERVICE_W = {  # Maximum post-scaling widths for service rooms
            'bathroom': 8.0, 'toilet': 6.0, 'pooja': 7.0,
            'store': 7.0, 'utility': 7.0,
        }

        # First pass: scale all proportionally
        scale = band_w / total_ideal
        widths = [w * scale for w in ideal_widths]

        # Second pass: cap service rooms and redistribute excess to major rooms
        excess = 0
        major_indices = []
        max_living_w = band_w * 0.50  # Living should not exceed 50% of band
        for i, r in enumerate(rooms):
            rtype = r['room_type']
            if rtype in SERVICE_TYPES:
                max_w = MAX_SERVICE_W.get(rtype, 8.0)
                if widths[i] > max_w:
                    excess += widths[i] - max_w
                    widths[i] = max_w
            elif rtype == 'living' and widths[i] > max_living_w:
                excess += widths[i] - max_living_w
                widths[i] = max_living_w
                major_indices.append(i)
            else:
                major_indices.append(i)

        # Give excess width to major rooms (proportionally)
        if excess > 0 and major_indices:
            major_total = sum(widths[i] for i in major_indices)
            for i in major_indices:
                bonus = excess * (widths[i] / major_total) if major_total > 0 else 0
                widths[i] += bonus

        # Third pass: enforce minimum widths (best-effort, don't overflow band)
        deficit = 0
        flex_indices = []
        for i, r in enumerate(rooms):
            min_w = MIN_DIMS.get(r['room_type'], (5, 5))[0]
            if widths[i] < min_w:
                deficit += min_w - widths[i]
                widths[i] = min_w
            else:
                flex_indices.append(i)

        # Distribute deficit from flexible (larger) rooms
        if deficit > 0 and flex_indices:
            flex_total = sum(widths[i] for i in flex_indices)
            for i in flex_indices:
                reduction = deficit * (widths[i] / flex_total) if flex_total > 0 else 0
                widths[i] -= reduction
        
        # Safety: if total exceeds band_w (can happen when ALL rooms' minimums
        # > band_w), rescale proportionally. Some rooms will be under MIN_DIMS
        # but this is unavoidable on tight plots.
        total_after = sum(widths)
        if total_after > band_w + 0.5:
            rescale = band_w / total_after
            widths = [w * rescale for w in widths]
    else:
        widths = list(ideal_widths)

    # Step 5b: Targeted AR enforcement for severely stretched rooms.
    # Only fix rooms where AR > limit. Steal width from rooms that
    # have more than average width (they can afford to shrink slightly).
    avg_w = sum(widths) / n if n > 0 else 0
    for i, r in enumerate(rooms):
        rtype = r['room_type']
        max_ar = MAX_ASPECT.get(rtype, 2.0)
        current_ar = band_h / widths[i] if widths[i] > 0 else 999
        if current_ar <= max_ar:
            continue  # room is fine
        min_w_for_ar = _snap(band_h / max_ar)
        needed = min_w_for_ar - widths[i]
        if needed <= 0.5:
            continue
        # Find donors: rooms with width above average
        donors = []
        for j in range(n):
            if j == i:
                continue
            excess = widths[j] - avg_w
            if excess > 0.5:
                donors.append((j, excess))
        donors.sort(key=lambda x: x[1], reverse=True)
        for j, excess in donors:
            give = min(excess * 0.4, needed)
            widths[j] -= give
            widths[i] += give
            needed -= give
            if needed <= 0.3:
                break

    # Step 6: Snap to grid and place
    SERVICE_TYPES_SET = {'bathroom', 'toilet', 'pooja', 'store', 'utility'}
    cx = band_x
    for i, (r, w) in enumerate(zip(rooms, widths)):
        w = _snap(max(w, 3.0))
        if i == n - 1:
            remaining = _snap((band_x + band_w) - cx)
            # Only absorb remaining if it's minor rounding (< 3ft extra)
            # or the room is a major room type that benefits from extra width
            if remaining - w < 3.0 or r['room_type'] not in SERVICE_TYPES_SET:
                w = remaining
            else:
                # Service room shouldn't absorb large excess.
                # Place it at calculated width; a filler will be added after.
                pass

        room_entry = dict(r)
        room_entry['_placed'] = {
            'x': round(cx, 2),
            'y': round(band_y, 2),
            'w': round(w, 2),
            'h': round(band_h, 2),
        }
        placed.append(room_entry)
        cx += w

    # If there's leftover space (> 3ft), prefer EXPANDING existing rooms over
    # adding filler rooms. Only add a filler utility room if no room can absorb it.
    # This prevents the proliferation of tiny "Wash Area" rooms that clutter plans.
    leftover = _snap((band_x + band_w) - cx)
    MAX_FILLER_W = 5.0  # Max 5ft wide for filler room (tighter limit)
    MIN_FILLER_W = 3.5  # Minimum to be a meaningful room
    if leftover >= MIN_FILLER_W:
        # Strategy: try to expand the LAST major room first
        absorbed = False
        if placed:
            last_room = placed[-1]
            last_type = last_room['room_type']
            last_p = last_room['_placed']
            # Expand last major room if AR stays within limits
            new_w = last_p['w'] + leftover
            new_ar = _aspect(new_w, last_p['h'])
            max_ar = _max_ar(last_type)
            if new_ar <= max_ar and last_type not in ('bathroom', 'toilet'):
                last_p['w'] = round(new_w, 2)
                absorbed = True
            elif leftover <= 2.0:
                # Tiny gap — absorb into last room regardless
                last_p['w'] = round(new_w, 2)
                absorbed = True

        if not absorbed and leftover >= MIN_FILLER_W:
            # Try distributing to ALL expandable rooms proportionally
            if leftover <= MAX_FILLER_W:
                expand_indices = [j for j in range(len(placed))
                                  if placed[j]['room_type'] not in
                                  ('bathroom', 'toilet', 'pooja', 'store', 'utility')]
                if expand_indices and leftover <= len(expand_indices) * 2.0:
                    # Each room gets a small boost — better than a filler
                    share = leftover / len(expand_indices)
                    shift = 0
                    for j in range(len(placed)):
                        placed[j]['_placed']['x'] = round(
                            placed[j]['_placed']['x'] + shift, 2)
                        if j in expand_indices:
                            placed[j]['_placed']['w'] = round(
                                placed[j]['_placed']['w'] + share, 2)
                            shift += share
                    absorbed = True

        if not absorbed and leftover >= MIN_FILLER_W:
            filler_w = min(leftover, MAX_FILLER_W)
            # Redistribute excess beyond filler to major rooms
            if leftover > MAX_FILLER_W:
                excess = leftover - MAX_FILLER_W
                expand_indices = [j for j in range(len(placed))
                                  if placed[j]['room_type'] not in
                                  ('bathroom', 'toilet', 'pooja', 'store', 'utility')]
                if not expand_indices:
                    expand_indices = list(range(len(placed)))
                if expand_indices:
                    share = excess / len(expand_indices)
                    shift = 0
                    for j in range(len(placed)):
                        placed[j]['_placed']['x'] = round(
                            placed[j]['_placed']['x'] + shift, 2)
                        if j in expand_indices:
                            placed[j]['_placed']['w'] = round(
                                placed[j]['_placed']['w'] + share, 2)
                            shift += share

            placed.append({
                'room_type': 'utility',
                'name': 'Utility',
                'zone': 'service',
                'target_area': filler_w * band_h,
                '_placed': {
                    'x': round((band_x + band_w) - filler_w, 2),
                    'y': round(band_y, 2),
                    'w': round(filler_w, 2),
                    'h': round(band_h, 2),
                },
            })

    return placed


def _place_private_band_smart(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    all_rooms_ref: List[Dict],
    service_rooms: List[Dict],
    band_x: float,
    band_y: float,
    band_w: float,
    band_h: float,
    total_area: float,
) -> List[Dict]:
    """
    Place private band rooms with intelligent layout strategy.

    Uses three strategies in order of preference:
    
    Strategy 1 — COLUMN SPLIT (3+ bedrooms):
      Bedrooms in wide left section (full band height), service rooms
      stacked vertically in narrower right column. Master bedroom is
      placed rightmost in the bedroom section so it shares a wall with
      the attached bathroom in the service column.
      
      ┌───────┬───────┬───────┬─────────┐
      │       │       │       │ AttBath  │
      │  BR2  │  BR3  │ MBR   │─────────│
      │ 10×12 │ 9×12  │ 10×12 │ Bath2   │
      │       │       │       │─────────│
      │       │       │       │ Pooja   │
      └───────┴───────┴───────┴─────────┘
    
    Strategy 2 — SUB-BAND (enough depth):
      Service rooms above, bedrooms below with upward extension.
      
    Strategy 3 — SIDE-BY-SIDE (fallback):
      All rooms in a single row.
    """
    attached_baths = [r for r in rooms if r.get('_is_attached_bath')]
    common_baths = [r for r in rooms
                    if r['room_type'] in ('bathroom', 'toilet')
                    and not r.get('_is_attached_bath')]
    bedrooms = [r for r in rooms
                if r['room_type'] in ('master_bedroom', 'bedroom')]
    studies = [r for r in rooms if r['room_type'] == 'study']
    poojas = [r for r in rooms if r['room_type'] == 'pooja']
    other_private = [r for r in rooms
                     if r['room_type'] not in ('master_bedroom', 'bedroom',
                                               'bathroom', 'toilet',
                                               'study', 'pooja')]

    all_service = attached_baths + common_baths + service_rooms

    if not all_service:
        ordered = bedrooms + studies + poojas + other_private
        ordered.sort(key=lambda r: PRIORITY.get(r['room_type'], 30), reverse=True)
        return _place_rooms_in_band(ordered, band_x, band_y, band_w, band_h)

    if not bedrooms:
        ordered = all_service + studies + poojas + other_private
        return _place_rooms_in_band(ordered, band_x, band_y, band_w, band_h)

    # ------------------------------------------------------------------
    # Strategy 1: Column-split (best for 2 bedrooms with few service rooms)
    # ------------------------------------------------------------------
    large_rooms = bedrooms + studies
    small_rooms = all_service + poojas + other_private

    if len(bedrooms) >= 2 and len(small_rooms) >= 2 and len(small_rooms) <= 4:
        col_result = _try_private_column_split(
            large_rooms, small_rooms,
            band_x, band_y, band_w, band_h)
        if col_result is not None:
            return col_result

    # ------------------------------------------------------------------
    # Strategy 2: FLIPPED Sub-band layout
    # ------------------------------------------------------------------
    # KEY INSIGHT: Bedrooms go at the TOP of the band (touching the
    # back wall = exterior wall = guaranteed natural light). Service
    # rooms (bathrooms) go at the BOTTOM near the corridor for easy
    # access. This ensures every bedroom has at least one exterior wall.
    #
    #   ┌────────┬────────┬────────────────┐  ← Back wall (exterior)
    #   │  BR1   │  BR2   │   MBR          │  ← Bedrooms (top, natural light)
    #   ├────────┼────────┼───────┬────────┤
    #   │ ABath  │ Bath2  │ Study │ Pooja  │  ← Service (bottom, near corridor)
    #   └────────┴────────┴───────┴────────┘  ← Corridor side
    # Adaptive service depth: enough for bathrooms (min 5ft) but leave
    # enough for bedrooms below (need at least 7ft usable)
    service_h = min(6.0, max(5.0, band_h * 0.35))
    bedroom_h = band_h - service_h

    # Ensure bedroom AR stays within limits for all bedrooms
    n_beds = max(1, len(bedrooms))
    min_bed_w = (band_w / n_beds) * 0.85  # safety for unequal widths
    max_bed_ar = MAX_ASPECT.get('bedroom', 1.8)
    max_bedroom_h_for_ar = min_bed_w * max_bed_ar
    if bedroom_h > max_bedroom_h_for_ar and bedroom_h > 10.0:
        # Reduce bedroom_h; give extra to service sub-band (taller baths)
        bedroom_h = _snap(max(10.0, max_bedroom_h_for_ar))
        service_h = _snap(band_h - bedroom_h)

    # Ensure service sub-band has usable height (min 5ft for bathrooms)
    if service_h < 5.0 and band_h >= 14.0:
        service_h = 5.0
        bedroom_h = _snap(band_h - service_h)

    if bedroom_h < 7.0:
        # Not enough space for sub-bands — fall back to side-by-side placement
        ordered = _order_private_rooms(
            rooms + service_rooms, attached_pairs, all_rooms_ref, band_w)
        return _place_rooms_in_band(ordered, band_x, band_y, band_w, band_h)

    bedroom_h = _snap(bedroom_h)
    service_h = _snap(band_h - bedroom_h)

    # Classify rooms for each sub-band
    service_band_rooms = list(all_service)
    bedroom_band_rooms = list(bedrooms)

    for rm in studies + poojas:
        ideal_w = rm['target_area'] / bedroom_h
        ar = bedroom_h / ideal_w if ideal_w > 0 else 99
        max_ar = MAX_ASPECT.get(rm['room_type'], 2.0)
        # Move to service band if the room would be close to AR limit
        # Use 0.85 multiplier — better to have study in service band
        # at 7ft depth (good proportions) than in bedroom band at 15ft depth
        if ar > max_ar * 0.85:
            service_band_rooms.append(rm)
        else:
            bedroom_band_rooms.append(rm)
    bedroom_band_rooms.extend(other_private)

    placed = []

    # --- Step 1: Place bedroom sub-band (TOP — touching back wall for light) ---
    bedroom_y = band_y + service_h   # Bedrooms start above service band
    bedroom_band_rooms.sort(
        key=lambda r: PRIORITY.get(r['room_type'], 30), reverse=True)
    bed_placed = _place_rooms_in_band(
        bedroom_band_rooms, band_x, bedroom_y, band_w, bedroom_h)

    # --- Step 1b: Ensure master bedroom ≥ all regular bedrooms (hierarchy) ---
    masters_bp = [(i, r) for i, r in enumerate(bed_placed)
                  if r['room_type'] == 'master_bedroom']
    regulars_bp = [(i, r) for i, r in enumerate(bed_placed)
                   if r['room_type'] == 'bedroom']
    if masters_bp and regulars_bp:
        for mi, mr in masters_bp:
            mr_area = mr['_placed']['w'] * mr['_placed']['h']
            # Find the largest regular bedroom
            max_reg_i, max_reg = max(regulars_bp, key=lambda t: t[1]['_placed']['w'] * t[1]['_placed']['h'])
            max_reg_area = max_reg['_placed']['w'] * max_reg['_placed']['h']
            if mr_area < max_reg_area:
                # Swap widths between master and largest regular bedroom
                mr['_placed']['w'], max_reg['_placed']['w'] = max_reg['_placed']['w'], mr['_placed']['w']
                # Also swap x positions so they stay tiled
                mr['_placed']['x'], max_reg['_placed']['x'] = max_reg['_placed']['x'], mr['_placed']['x']

    # --- Step 2: Place service rooms (BOTTOM — near corridor) ---
    # Service rooms (bathrooms, etc.) get positioned to align with bedrooms
    # when possible (attached bath below its master bedroom).
    service_y = band_y   # Service at bottom of band

    # Sort service rooms: attached baths first, then common service
    attached_svc = [sr for sr in service_band_rooms if sr.get('_is_attached_bath')]
    common_svc = [sr for sr in service_band_rooms if not sr.get('_is_attached_bath')]

    svc_placed = []
    svc_occupied = []  # Track (x_start, x_end) for each placed service room
    matched_parents = set()  # Track which bedrooms have been matched

    # Place attached baths ABOVE their parent bedrooms
    for sr in attached_svc:
        parent_idx = sr.get('_attached_to')
        # Find an UNMATCHED parent bedroom (master_bedroom preferred)
        parent_bp = None
        for bi, br in enumerate(bed_placed):
            if bi in matched_parents:
                continue
            if br['room_type'] == 'master_bedroom':
                parent_bp = br['_placed']
                matched_parents.add(bi)
                break
        # Fallback: any unmatched bedroom
        if parent_bp is None:
            for bi, br in enumerate(bed_placed):
                if bi not in matched_parents and br['room_type'] in ('master_bedroom', 'bedroom'):
                    parent_bp = br['_placed']
                    matched_parents.add(bi)
                    break

        natural_w = sr['target_area'] / service_h if service_h > 0 else 7.0
        min_w = MIN_DIMS.get(sr['room_type'], (5, 5))[0]
        max_w = 8.0
        w = _snap(max(min_w, min(natural_w, max_w)))

        # Try to align with parent bedroom
        if parent_bp:
            sx = parent_bp['x']
            w = min(w, parent_bp['w'])  # Don't exceed parent width
        else:
            # Place at first available X
            sx = band_x
            for (occ_start, occ_end) in svc_occupied:
                if sx < occ_end:
                    sx = occ_end

        w = min(w, _snap((band_x + band_w) - sx))
        if w < 3.0:
            continue

        entry = dict(sr)
        entry['_placed'] = {
            'x': round(sx, 2),
            'y': round(service_y, 2),
            'w': round(w, 2),
            'h': round(service_h, 2),
        }
        svc_placed.append(entry)
        svc_occupied.append((sx, sx + w))

    # Place common service rooms after attached baths
    svc_x = band_x
    for (occ_start, occ_end) in sorted(svc_occupied):
        svc_x = max(svc_x, occ_end)

    for sr in common_svc:
        natural_w = sr['target_area'] / service_h if service_h > 0 else 7.0
        min_w = MIN_DIMS.get(sr['room_type'], (5, 5))[0]
        max_w = 8.0 if sr['room_type'] in ('bathroom', 'toilet') else 7.0
        w = _snap(max(min_w, min(natural_w, max_w)))
        w = min(w, _snap((band_x + band_w) - svc_x))
        if w < 3.0:
            break

        entry = dict(sr)
        entry['_placed'] = {
            'x': round(svc_x, 2),
            'y': round(service_y, 2),
            'w': round(w, 2),
            'h': round(service_h, 2),
        }
        svc_placed.append(entry)
        svc_occupied.append((svc_x, svc_x + w))
        svc_x += w

    # --- Step 3: Extend bedrooms DOWNWARD into service gaps ---
    # If no service room occupies the X range below a bedroom,
    # extend that bedroom downward to fill the gap (more area).
    svc_covered_end = band_x
    for sr in svc_placed:
        sp = sr['_placed']
        svc_covered_end = max(svc_covered_end, sp['x'] + sp['w'])

    for br in bed_placed:
        bp = br['_placed']
        bed_left = bp['x']
        bed_right = bp['x'] + bp['w']

        # Check if ANY service room overlaps this bedroom's X range
        has_overlap = False
        for sr in svc_placed:
            sp = sr['_placed']
            overlap = min(bed_right, sp['x'] + sp['w']) - max(bed_left, sp['x'])
            if overlap > 1.0:
                has_overlap = True
                break

        # Only extend downward if NO service room is below this bedroom
        # AND the extended height would not exceed the max aspect ratio
        if not has_overlap:
            new_h = bp['h'] + service_h
            new_ar = _aspect(bp['w'], new_h)
            max_ar = _max_ar(br['room_type'])
            if new_ar <= max_ar:
                bp['y'] = round(band_y, 2)  # Extend down to band bottom
                bp['h'] = round(new_h, 2)

    # --- Step 4: Fill ALL remaining gaps in service sub-band ---
    # Scan the entire service sub-band for uncovered horizontal stretches
    # and fill them with utility/wash rooms.
    covered_intervals = []
    for sr in svc_placed:
        sp = sr['_placed']
        covered_intervals.append((sp['x'], sp['x'] + sp['w']))
    for br in bed_placed:
        bp = br['_placed']
        if bp['y'] < bedroom_y - 0.1:
            # This bedroom was extended downward — it covers the service band
            covered_intervals.append((bp['x'], bp['x'] + bp['w']))

    # Merge overlapping intervals
    covered_intervals.sort()
    merged_coverage = []
    for start, end in covered_intervals:
        if merged_coverage and start <= merged_coverage[-1][1] + 0.3:
            merged_coverage[-1] = (merged_coverage[-1][0], max(merged_coverage[-1][1], end))
        else:
            merged_coverage.append((start, end))

    # Find gaps in coverage
    prev_end = band_x
    MAX_FILLER_W = 8.0
    for start, end in merged_coverage:
        gap_w = _snap(start - prev_end)
        if gap_w >= 3.5:
            filler_w = min(gap_w, MAX_FILLER_W)
            placed.append({
                'room_type': 'utility',
                'name': 'Wash Area',
                'zone': 'service',
                'target_area': filler_w * service_h,
                '_placed': {
                    'x': round(prev_end, 2),
                    'y': round(service_y, 2),
                    'w': round(filler_w, 2),
                    'h': round(service_h, 2),
                },
            })
        prev_end = max(prev_end, end)

    # Rightmost gap
    final_gap = _snap((band_x + band_w) - prev_end)
    if final_gap >= 3.5:
        filler_w = min(final_gap, MAX_FILLER_W)
        placed.append({
            'room_type': 'utility',
            'name': 'Utility',
            'zone': 'service',
            'target_area': filler_w * service_h,
            '_placed': {
                'x': round(prev_end, 2),
                'y': round(service_y, 2),
                'w': round(filler_w, 2),
                'h': round(service_h, 2),
            },
        })

    placed.extend(bed_placed)
    placed.extend(svc_placed)

    return placed


def _try_private_column_split(
    large_rooms: List[Dict],
    small_rooms: List[Dict],
    band_x: float,
    band_y: float,
    band_w: float,
    band_h: float,
) -> Optional[List[Dict]]:
    """
    Split private band into bedroom column (left) + service column (right).

    Bedrooms get full band height in a wide left section.
    Service rooms (bathrooms, pooja) are stacked vertically in a narrower
    right column. Master bedroom is placed RIGHTMOST in the bedroom section
    so it shares a wall with the attached bathroom in the service column.

    Returns placed rooms list, or None if column-split isn't viable.
    """
    if not large_rooms or not small_rooms:
        return None

    # Determine service column width.
    # Goal: wide enough that each stacked room meets its minimum area,
    # but not so wide that bedrooms lose their 7ft minimum width.
    small_total_area = sum(r['target_area'] for r in small_rooms)
    # Minimum area each stacked room needs (bathrooms ≥ 25sqft, others ≥ 20sqft)
    min_svc_area = sum(
        max(MIN_AREAS.get(r['room_type'], 20) * 0.7, 20)
        for r in small_rooms
    )
    # Width from minimum-area requirement (each room stacked at band_h/n_rooms)
    area_based_w = min_svc_area / band_h + 0.5 if band_h > 0 else 7.0
    # Cap at 30% of band width to leave room for bedrooms
    max_svc_w = band_w * 0.3
    service_col_w = _snap(max(5.0, min(max_svc_w, area_based_w)))

    bedroom_col_w = _snap(band_w - service_col_w)

    # Ensure each bedroom gets at least 7ft width
    min_bed_w_each = 7.0
    if bedroom_col_w < len(large_rooms) * min_bed_w_each:
        # Not enough width — try with a smaller service column
        service_col_w = _snap(max(4.5, band_w - len(large_rooms) * min_bed_w_each))
        bedroom_col_w = _snap(band_w - service_col_w)
        if bedroom_col_w < len(large_rooms) * min_bed_w_each:
            return None  # Still not enough

    # Order bedrooms: put MBR RIGHTMOST (adjacent to service column)
    # so attached bathroom shares a wall with it.
    masters = [r for r in large_rooms if r['room_type'] == 'master_bedroom']
    others = [r for r in large_rooms if r['room_type'] != 'master_bedroom']
    ordered_beds = others + masters  # MBR at right edge

    # Place bedrooms side-by-side in the left column (full band height)
    bed_placed = _place_rooms_in_band(
        ordered_beds, band_x, band_y, bedroom_col_w, band_h)

    # Stack service rooms vertically in the right column
    svc_x = band_x + bedroom_col_w
    svc_placed = _stack_rooms_vertically(
        small_rooms, svc_x, band_y, service_col_w, band_h)

    if not svc_placed:
        return None

    # Validate: all rooms must have acceptable aspect ratios.
    # Column-split is architecturally superior (every bedroom gets
    # natural light via exterior wall), so use generous tolerance —
    # a slightly tall bedroom is FAR better than an interior one.
    # Service-column rooms (stacked) need extra tolerance because
    # they share limited height among many rooms.
    svc_set = set(id(r) for r in svc_placed)
    all_placed = bed_placed + svc_placed
    for r in all_placed:
        p = r['_placed']
        ar = _aspect(p['w'], p['h'])
        max_ar = _max_ar(r['room_type'])
        # Bedrooms: tolerate up to max_ar + 1.0 (e.g. 1.8+1.0=2.8)
        # Service column rooms: +1.5 (compact proportions ok)
        # Small utility rooms (< 30 sqft): +1.5
        if r['room_type'] in ('bedroom', 'master_bedroom'):
            tolerance = 1.0
        elif id(r) in svc_set:
            tolerance = 1.5
        elif r.get('target_area', 99) < 30:
            tolerance = 1.5
        else:
            tolerance = 0.5
        if ar > max_ar + tolerance:
            return None  # AR too bad, reject this strategy

    return all_placed


def _stack_rooms_vertically(
    rooms: List[Dict],
    x: float,
    y: float,
    w: float,
    total_h: float,
) -> List[Dict]:
    """
    Stack rooms vertically (top to bottom) within a column.

    Heights are proportional to target area. Each room gets its fair
    share of the total column height.  Uses budget-aware allocation
    so the last room is never starved.
    """
    if not rooms:
        return []

    n = len(rooms)
    MIN_ROOM_H = 2.5  # absolute minimum per room

    # Feasibility: can we fit all rooms?
    if n * MIN_ROOM_H > total_h + 0.5:
        return []  # can't fit — caller should try another strategy

    total_target = sum(r['target_area'] for r in rooms) or 1

    # Proportional heights (clamped to min, then scaled to fit)
    raw = [max(MIN_ROOM_H, total_h * (r['target_area'] / total_target))
           for r in rooms]
    raw_sum = sum(raw)
    if raw_sum > total_h:
        # Scale down rooms above MIN_ROOM_H proportionally
        excess = raw_sum - total_h
        above = [(i, h - MIN_ROOM_H) for i, h in enumerate(raw) if h > MIN_ROOM_H]
        above_total = sum(a for _, a in above)
        if above_total > 0:
            for i, margin in above:
                raw[i] -= excess * (margin / above_total)

    # Snap and place
    placed = []
    cy = y

    for i, r in enumerate(rooms):
        if i == n - 1:
            # Last room gets remaining height (avoids rounding gaps)
            h = _snap((y + total_h) - cy)
        else:
            h = _snap(raw[i])
            h = max(h, MIN_ROOM_H)

        if h < MIN_ROOM_H:
            h = MIN_ROOM_H  # safety net

        entry = dict(r)
        entry['_placed'] = {
            'x': round(x, 2),
            'y': round(cy, 2),
            'w': round(w, 2),
            'h': round(h, 2),
        }
        placed.append(entry)
        cy += h

    return placed


def _order_service_for_plumbing(
    service_rooms: List[Dict],
    bedrooms: List[Dict],
    other_rooms: List[Dict],
    band_w: float,
) -> List[Dict]:
    """
    Order service rooms to maximize plumbing wall sharing.

    Rule: attached bathrooms should align ABOVE their parent bedrooms.
    Common bathrooms should be next to attached baths (shared plumbing stack).
    Kitchen (in public band below) is at the LEFT — so bathrooms at LEFT
    share the plumbing wall with kitchen.
    """
    attached = [r for r in service_rooms if r.get('_is_attached_bath')]
    common = [r for r in service_rooms if not r.get('_is_attached_bath')]

    # Put attached baths first (they'll be at the left, above master bedrooms)
    # Then common baths (next to attached for shared plumbing)
    # Then other service rooms
    ordered = attached + common
    return ordered


# =============================================================================
# PHASE 4: MULTI-CANDIDATE SCORING
# =============================================================================

def _score_layout(
    placed_rooms: List[Dict],
    all_rooms_ref: List[Dict],
    attached_pairs: List[Tuple],
    plot_w: float,
    plot_h: float,
    ux: float,
    uy: float,
) -> float:
    """
    Score a candidate layout on architectural quality (0.0 to 1.0).

    Metrics:
    1. Adjacency satisfaction (25%): required/preferred pairs touching
    2. Room proportions (20%): aspect ratios within limits
    3. Vastu compliance (10%): rooms in correct quadrants
    4. Natural light & ventilation (15%): habitable rooms on external walls
    5. Plumbing efficiency (10%): wet rooms sharing walls
    6. Coverage (10%): floor area utilization
    7. Area accuracy (10%): rooms near their target areas
    8. Cross-ventilation bonus: rooms with 2+ exterior walls
    """
    if not placed_rooms:
        return 0.0

    # Build position lookup
    rooms_by_type = defaultdict(list)
    room_rects = []
    for r in placed_rooms:
        p = r.get('_placed', {})
        rect = {
            'x': p.get('x', 0), 'y': p.get('y', 0),
            'w': p.get('w', 0), 'h': p.get('h', 0),
            'room_type': r['room_type'],
        }
        rooms_by_type[r['room_type']].append(rect)
        room_rects.append(rect)

    n_rooms = len(room_rects)

    # --- 1. Adjacency Score (25%) ---
    adj_score = 1.0
    adj_required = 0
    adj_required_met = 0
    adj_preferred = 0
    adj_preferred_met = 0

    for (rt_a, rt_b, strength) in DESIRED_ADJ:
        ra_list = rooms_by_type.get(rt_a, [])
        rb_list = rooms_by_type.get(rt_b, [])
        if not ra_list or not rb_list:
            continue
        found = False
        for ra in ra_list:
            for rb in rb_list:
                if _rooms_share_wall(ra, rb, min_overlap=2.0):
                    found = True
                    break
            if found:
                break
        if strength == 'required':
            adj_required += 1
            if found:
                adj_required_met += 1
        else:
            adj_preferred += 1
            if found:
                adj_preferred_met += 1

    # Required adjacencies weigh more heavily
    if adj_required > 0:
        req_ratio = adj_required_met / adj_required
    else:
        req_ratio = 1.0
    if adj_preferred > 0:
        pref_ratio = adj_preferred_met / adj_preferred
    else:
        pref_ratio = 1.0
    adj_score = req_ratio * 0.7 + pref_ratio * 0.3

    # Check forbidden adjacencies (penalty)
    forbidden_violations = 0
    for (rt_a, rt_b) in FORBIDDEN_ADJ:
        for ra in rooms_by_type.get(rt_a, []):
            for rb in rooms_by_type.get(rt_b, []):
                if _rooms_share_wall(ra, rb, min_overlap=1.0):
                    forbidden_violations += 1
    adj_score = max(0, adj_score - forbidden_violations * 0.2)

    # --- 2. Proportion Score (25%) ---
    prop_score = 0.0
    for rect in room_rects:
        ar = _aspect(rect['w'], rect['h'])
        max_ar = _max_ar(rect['room_type'])
        if ar <= max_ar:
            prop_score += 1.0
        elif ar <= max_ar * 1.3:
            prop_score += 0.5
        else:
            prop_score += 0.0
    prop_score = prop_score / n_rooms if n_rooms > 0 else 0

    # --- 3. Vastu Score (15%) ---
    plot_cx = ux + plot_w / 2
    plot_cy = uy + plot_h / 2
    vastu_score = 0.0
    vastu_checks = 0
    for rect in room_rects:
        prefs = VASTU_PREFS.get(rect['room_type'], [])
        if not prefs:
            continue
        vastu_checks += 1
        quad = _get_quadrant(rect['x'], rect['y'], rect['w'], rect['h'],
                             plot_cx, plot_cy)
        if quad in prefs:
            rank = prefs.index(quad)
            vastu_score += max(1.0 - rank * 0.25, 0.2)
        else:
            vastu_score += 0.0
    vastu_score = vastu_score / vastu_checks if vastu_checks > 0 else 0.5

    # --- 4. Natural Light Score (15%) ---
    # Now includes cross-ventilation bonus: rooms touching 2+ exterior walls
    # get better natural airflow (NBC 2016 recommends cross-ventilation)
    light_score = 0.0
    light_checks = 0
    ext_tolerance = WALL_EXT + 1.0
    cross_vent_bonus = 0.0
    cross_vent_count = 0
    for rect in room_rects:
        if rect['room_type'] in ('corridor', 'store', 'utility'):
            continue
        light_checks += 1
        # Count how many external walls this room touches
        ext_walls_count = 0
        if rect['x'] <= ux + ext_tolerance:
            ext_walls_count += 1
        if rect['x'] + rect['w'] >= ux + plot_w - ext_tolerance:
            ext_walls_count += 1
        if rect['y'] <= uy + ext_tolerance:
            ext_walls_count += 1
        if rect['y'] + rect['h'] >= uy + plot_h - ext_tolerance:
            ext_walls_count += 1

        if ext_walls_count >= 2:
            light_score += 1.0
            if rect['room_type'] not in ('bathroom', 'toilet', 'pooja'):
                cross_vent_count += 1  # Cross-ventilation for habitable rooms
        elif ext_walls_count == 1:
            light_score += 0.8
        elif rect['room_type'] in ('bathroom', 'toilet', 'pooja'):
            light_score += 0.7  # These can be interior
        else:
            light_score += 0.0   # Habitable interior room is bad
    light_score = light_score / light_checks if light_checks > 0 else 0.5
    # Cross-ventilation bonus: up to 0.05 extra
    habitable_count = sum(1 for r in room_rects
                          if r['room_type'] not in ('bathroom', 'toilet', 'pooja',
                                                    'corridor', 'store', 'utility'))
    if habitable_count > 0:
        cross_vent_bonus = min(0.05, 0.05 * cross_vent_count / habitable_count)

    # --- 5. Plumbing Efficiency Score (15%) ---
    plumb_score = 0.0
    wet_rooms = [r for r in room_rects
                 if r['room_type'] in ('bathroom', 'toilet', 'kitchen', 'utility')]
    if len(wet_rooms) >= 2:
        plumb_connections = 0
        plumb_possible = 0
        for i in range(len(wet_rooms)):
            for j in range(i + 1, len(wet_rooms)):
                plumb_possible += 1
                if _rooms_share_wall(wet_rooms[i], wet_rooms[j], min_overlap=2.0):
                    plumb_connections += 1
        plumb_score = plumb_connections / plumb_possible if plumb_possible > 0 else 0
    else:
        plumb_score = 0.5

    # --- 6. Coverage Score (bonus) ---
    # Penalize layouts that waste usable area (large gaps)
    total_room_area = sum(r['w'] * r['h'] for r in room_rects)
    usable_area = plot_w * plot_h
    coverage = min(total_room_area / usable_area, 1.0) if usable_area > 0 else 0
    coverage_score = coverage  # 1.0 = perfect coverage

    # --- 7. Area Accuracy Score (bonus) ---
    # Rooms should be close to their target areas
    area_acc = 0.0
    area_checks = 0
    for r in placed_rooms:
        target = r.get('target_area', 0)
        if target <= 0:
            continue
        p = r.get('_placed', {})
        actual = p.get('w', 0) * p.get('h', 0)
        ratio = actual / target if target > 0 else 0
        # Score: 1.0 for within ±20%, 0.5 for ±40%, 0 otherwise
        if 0.8 <= ratio <= 1.3:
            area_acc += 1.0
        elif 0.6 <= ratio <= 1.6:
            area_acc += 0.5
        area_checks += 1
    area_acc_score = area_acc / area_checks if area_checks > 0 else 0.5

    # --- Weighted total ---
    # Weights tuned for professional Indian residential architecture:
    # Adjacency (25%) is the most critical — rooms must relate properly
    # Proportions (20%) — rooms must be usable shapes
    # Light+vent (15%) — habitable rooms need natural light
    # Coverage (10%) — good space utilization
    # Area accuracy (10%) — rooms match target sizes
    # Vastu (10%) — directional compliance
    # Plumbing (10%) — cost efficiency
    total = (adj_score * 0.25 +
             prop_score * 0.20 +
             vastu_score * 0.10 +
             light_score * 0.15 +
             plumb_score * 0.10 +
             coverage_score * 0.10 +
             area_acc_score * 0.10 +
             cross_vent_bonus)

    return round(total, 4)


# =============================================================================
# POST-PLACEMENT OPTIMIZER
# =============================================================================

def _optimize_layout(placed: List[Dict], ux: float, uy: float,
                     uw: float, ul: float) -> List[Dict]:
    """
    Post-placement optimization pass that fixes common issues:
    1. Detect and fill gaps between rooms
    2. Fix overlapping rooms (nudge apart)
    3. Adjust rooms with extreme aspect ratios
    4. Align walls where nearly aligned (within 1ft)

    This runs AFTER candidate generation to polish the layout.
    """
    if not placed:
        return placed

    # --- Pass 1: Fix overlaps ---
    for i in range(len(placed)):
        pi = placed[i].get('_placed')
        if not pi:
            continue
        for j in range(i + 1, len(placed)):
            pj = placed[j].get('_placed')
            if not pj:
                continue
            # Check overlap
            x_overlap = min(pi['x'] + pi['w'], pj['x'] + pj['w']) - max(pi['x'], pj['x'])
            y_overlap = min(pi['y'] + pi['h'], pj['y'] + pj['h']) - max(pi['y'], pj['y'])
            if x_overlap > 0.5 and y_overlap > 0.5:
                # Rooms overlap — shrink the smaller one
                area_i = pi['w'] * pi['h']
                area_j = pj['w'] * pj['h']
                if area_i >= area_j:
                    # Shrink j
                    if x_overlap < y_overlap:
                        pj['x'] = round(pi['x'] + pi['w'], 2)
                        pj['w'] = round(max(3, pj['w'] - x_overlap), 2)
                    else:
                        pj['y'] = round(pi['y'] + pi['h'], 2)
                        pj['h'] = round(max(3, pj['h'] - y_overlap), 2)
                else:
                    if x_overlap < y_overlap:
                        pi['x'] = round(pj['x'] + pj['w'], 2)
                        pi['w'] = round(max(3, pi['w'] - x_overlap), 2)
                    else:
                        pi['y'] = round(pj['y'] + pj['h'], 2)
                        pi['h'] = round(max(3, pi['h'] - y_overlap), 2)

    # --- Pass 2: Wall alignment (snap nearly-aligned walls) ---
    ALIGN_THRESHOLD = 1.0  # If two walls are within 1ft, align them
    all_x_edges = []
    all_y_edges = []
    for r in placed:
        p = r.get('_placed')
        if not p:
            continue
        all_x_edges.append(p['x'])
        all_x_edges.append(p['x'] + p['w'])
        all_y_edges.append(p['y'])
        all_y_edges.append(p['y'] + p['h'])

    # Group nearby X edges
    all_x_edges.sort()
    x_groups = []
    for x in all_x_edges:
        merged = False
        for grp in x_groups:
            if abs(x - grp[0]) < ALIGN_THRESHOLD:
                grp.append(x)
                merged = True
                break
        if not merged:
            x_groups.append([x])

    # Snap X edges to group median
    x_snap_map = {}
    for grp in x_groups:
        if len(grp) > 1:
            median = _snap(sum(grp) / len(grp))
            for x in grp:
                x_snap_map[x] = median

    for r in placed:
        p = r.get('_placed')
        if not p:
            continue
        old_x = p['x']
        old_right = p['x'] + p['w']
        new_x = x_snap_map.get(old_x, old_x)
        new_right = x_snap_map.get(old_right, old_right)
        if new_x != old_x or new_right != old_right:
            p['x'] = round(new_x, 2)
            p['w'] = round(max(3.0, new_right - new_x), 2)

    # --- Pass 3: Fill gaps in usable area ---
    # Find horizontal gaps at each Y-band
    placed = _fill_coverage_gaps(placed, ux, uy, uw, ul)

    return placed


def _expand_adjacent_room(placed: List[Dict], gap: Dict, ux: float, ux_right: float):
    """
    Expand an adjacent room to fill a small gap instead of creating a filler.
    Finds the room that borders this gap and widens it.
    """
    gap_x = gap['x']
    gap_right = gap['x'] + gap['w']
    gap_y = gap['y']
    gap_top = gap['y'] + gap['h']
    tol = 0.5

    best_room = None
    best_overlap = 0

    for r in placed:
        p = r.get('_placed')
        if not p:
            continue
        # Check if this room shares a vertical edge with the gap
        ry_bot = p['y']
        ry_top = p['y'] + p['h']
        y_overlap = min(gap_top, ry_top) - max(gap_y, ry_bot)
        if y_overlap < 2.0:
            continue
        # Room's right edge touches gap's left edge?
        if abs((p['x'] + p['w']) - gap_x) < tol:
            if y_overlap > best_overlap:
                best_overlap = y_overlap
                best_room = r
        # Room's left edge touches gap's right edge?
        elif abs(p['x'] - gap_right) < tol:
            if y_overlap > best_overlap:
                best_overlap = y_overlap
                best_room = r

    if best_room:
        bp = best_room['_placed']
        new_w = bp['w'] + gap['w']
        max_ar = _max_ar(best_room['room_type'])
        if _aspect(new_w, bp['h']) <= max_ar * 1.1:
            # Expand room to fill gap
            if abs((bp['x'] + bp['w']) - gap_x) < tol:
                bp['w'] = round(new_w, 2)
            else:
                bp['x'] = round(bp['x'] - gap['w'], 2)
                bp['w'] = round(new_w, 2)


def _fill_coverage_gaps(placed: List[Dict], ux: float, uy: float,
                        uw: float, ul: float) -> List[Dict]:
    """
    Scan the placed rooms and fill any gaps > 3ft wide with utility rooms.
    Uses a sweep-line approach to find uncovered rectangles.
    Only fills within the Y-extent of existing rooms (not empty garden space).
    """
    # Find the actual Y-extent of placed rooms
    room_y_min = uy + ul
    room_y_max = uy
    for r in placed:
        p = r.get('_placed')
        if not p:
            continue
        room_y_min = min(room_y_min, p['y'])
        room_y_max = max(room_y_max, p['y'] + p['h'])

    if room_y_max <= room_y_min:
        return placed

    # Collect all Y-boundaries within the room extent
    y_bounds = set()
    y_bounds.add(round(room_y_min, 2))
    y_bounds.add(round(room_y_max, 2))
    for r in placed:
        p = r.get('_placed')
        if not p:
            continue
        y_bounds.add(round(p['y'], 2))
        y_bounds.add(round(p['y'] + p['h'], 2))

    y_sorted = sorted(y_bounds)
    new_fillers = []

    for yi in range(len(y_sorted) - 1):
        band_y = y_sorted[yi]
        band_top = y_sorted[yi + 1]
        band_h = round(band_top - band_y, 2)
        if band_h < 3.0:
            continue

        # Find all rooms that occupy this Y band
        x_occupied = []
        for r in placed:
            p = r.get('_placed')
            if not p:
                continue
            ry_bot = p['y']
            ry_top = p['y'] + p['h']
            # Room overlaps this band if it spans into it
            if ry_bot < band_top - 0.3 and ry_top > band_y + 0.3:
                x_occupied.append((p['x'], p['x'] + p['w']))

        # Merge overlapping X intervals
        x_occupied.sort()
        merged = []
        for start, end in x_occupied:
            if merged and start <= merged[-1][1] + 0.3:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Find gaps
        prev_end = ux
        for start, end in merged:
            gap_w = _snap(start - prev_end)
            if gap_w >= 3.0:
                new_fillers.append({
                    'room_type': 'utility',
                    'name': 'Utility',
                    'zone': 'service',
                    'target_area': gap_w * band_h,
                    '_placed': {
                        'x': round(prev_end, 2),
                        'y': round(band_y, 2),
                        'w': round(gap_w, 2),
                        'h': round(band_h, 2),
                    },
                })
            prev_end = max(prev_end, end)

        # Check gap at the right edge
        gap_w = _snap((ux + uw) - prev_end)
        if gap_w >= 3.0:
            new_fillers.append({
                'room_type': 'utility',
                'name': 'Utility',
                'zone': 'service',
                'target_area': gap_w * band_h,
                '_placed': {
                    'x': round(prev_end, 2),
                    'y': round(band_y, 2),
                    'w': round(gap_w, 2),
                    'h': round(band_h, 2),
                },
            })

    # --- Post-process fillers: skip very thin strips, prefer expanding
    # adjacent rooms over adding new filler rooms ---
    final_fillers = []
    MAX_FILLER_AR = 2.5
    MAX_CHUNK_W = 7.0
    MIN_FILLER_AREA = 20.0  # Don't create fillers smaller than 20 sqft
    for filler in new_fillers:
        fp = filler['_placed']
        filler_area = fp['w'] * fp['h']
        if filler_area < MIN_FILLER_AREA:
            # Tiny filler — try to expand an adjacent room instead
            _expand_adjacent_room(placed, fp, ux, ux + uw)
            continue
        ar = _aspect(fp['w'], fp['h'])
        if ar <= MAX_FILLER_AR:
            final_fillers.append(filler)
        elif fp['h'] < 3.0:
            # Very thin horizontal strip — skip entirely
            continue
        else:
            # Wide strip — split into chunks with acceptable AR
            chunk_w = min(MAX_CHUNK_W, fp['h'] * MAX_FILLER_AR)
            chunk_w = _snap(max(3.5, chunk_w))
            cx = fp['x']
            remaining = fp['w']
            while remaining >= 3.5:
                cw = min(chunk_w, remaining)
                if cw < 3.5:
                    break
                final_fillers.append({
                    'room_type': 'utility',
                    'name': 'Utility',
                    'zone': 'service',
                    'target_area': cw * fp['h'],
                    '_placed': {
                        'x': round(cx, 2),
                        'y': round(fp['y'], 2),
                        'w': round(cw, 2),
                        'h': round(fp['h'], 2),
                    },
                })
                cx += cw
                remaining -= cw

    # De-duplicate fillers (don't overlap with existing rooms)
    for filler in final_fillers:
        fp = filler['_placed']
        overlaps = False
        for r in placed:
            rp = r.get('_placed')
            if not rp:
                continue
            x_ov = min(fp['x'] + fp['w'], rp['x'] + rp['w']) - max(fp['x'], rp['x'])
            y_ov = min(fp['y'] + fp['h'], rp['y'] + rp['h']) - max(fp['y'], rp['y'])
            if x_ov > 1.0 and y_ov > 1.0:
                overlaps = True
                break
        if not overlaps:
            placed.append(filler)

    return placed


# =============================================================================
# PHASE 5: CANDIDATE GENERATION & OPTIMIZATION
# =============================================================================

def _generate_candidate(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    public_rooms: List[Dict],
    private_rooms: List[Dict],
    service_rooms: List[Dict],
    corridor_h: float,
    ux: float,
    uy: float,
    uw: float,
    ul: float,
    total_area: float,
    variant: int = 0,
) -> List[Dict]:
    """
    Generate one candidate layout with a specific room ordering variant.

    Variants:
      0: Standard order (Living→Kitchen→Dining, Master→BR→Bath)
      1: Private rooms reversed
      2: Wider corridor (5ft instead of 3.5-4ft)
      3: Swapped bedroom positions
      4: Compact corridor (3ft)
      5: Private rooms reversed with wider corridor
      6: Alternate Vastu-aware order (kitchen SE, pooja NE)
      7: Living room centered (not leftmost)
      8-11: Shuffled private rooms with varied corridor widths
      12-15: Additional randomized orderings for diversity
    """
    pub = list(public_rooms)
    priv = list(private_rooms)
    serv = list(service_rooms)

    # Corridor height variation for richer candidates
    corr_h_used = corridor_h

    # Apply variant transformations
    # RULE: Living Room ALWAYS stays first (leftmost) so it shares walls
    # with Kitchen AND (via extension through corridor) with bedrooms.
    # Kitchen ALWAYS stays next to Living (direct access required).
    if variant == 0:
        pass  # Standard order
    elif variant == 1:
        # Reverse private only
        priv = list(reversed(priv))
    elif variant == 2:
        # Wider corridor for better circulation
        if corridor_h > 0:
            corr_h_used = min(corridor_h + 1.0, 5.0)
    elif variant == 3:
        # Swap first two bedrooms/masters in private
        beds_in_priv = [r for r in priv if r['room_type'] in ('master_bedroom', 'bedroom')]
        if len(beds_in_priv) >= 2:
            priv = list(reversed(priv))
    elif variant == 4:
        # Compact corridor
        if corridor_h > 0:
            corr_h_used = max(corridor_h - 0.5, 3.0)
    elif variant >= 5:
        # Shuffle private rooms, keep public fixed
        # Use variant-specific seeds for reproducible diversity
        random.seed(variant * 37 + 13)
        priv_shuffled = list(priv)
        random.shuffle(priv_shuffled)
        priv = priv_shuffled
        # Alternate corridor widths for more diversity
        if variant % 3 == 0 and corridor_h > 0:
            corr_h_used = min(corridor_h + 1.0, 5.0)
        elif variant % 3 == 1 and corridor_h > 0:
            corr_h_used = max(corridor_h - 0.5, 3.0)

    # Order rooms within bands
    pub_ordered = _order_public_rooms(pub, uw) if variant < 5 else pub
    priv_ordered = _order_private_rooms(
        priv, attached_pairs, rooms, uw) if variant < 5 else priv

    # Calculate band heights
    public_h, private_h = _allocate_band_heights(
        pub_ordered, priv_ordered, serv,
        corr_h_used, ul, uw, total_area)

    placed = []

    # --- Place PUBLIC band (front/bottom of plot) ---
    pub_y = uy
    placed.extend(_place_rooms_in_band(
        pub_ordered, ux, pub_y, uw, public_h))

    # --- LIVING ROOM EXTENDS into corridor for direct access ---
    # Find the living room we just placed and extend it upward through
    # the corridor so it shares walls with BOTH public rooms AND the
    # private band. This makes bedrooms directly accessible from living.
    living_idx = None
    for i, r in enumerate(placed):
        if r.get('room_type') == 'living':
            living_idx = i
            break

    corridor_y = pub_y + public_h
    priv_y = corridor_y + corr_h_used
    raw_priv_h = _snap((uy + ul) - priv_y)
    # Use the capped private_h from band height allocation (prevents overly tall rooms)
    priv_h_actual = min(raw_priv_h, private_h) if private_h > 0 else raw_priv_h

    # Smart corridor placement — living room gets a SMALL extension (max 1.5ft)
    # into the corridor zone, making the corridor PARTIAL-width. This mirrors
    # real Indian architecture where the living room acts as the circulation
    # hub and the passage only serves the private zone.
    if corr_h_used > 0:
        # Find the living room and give it a small depth extension
        max_ext = min(1.5, corr_h_used * 0.5)

        living_found = False
        for r in placed:
            if r.get('room_type') == 'living':
                lp = r['_placed']
                lp['h'] = round(lp['h'] + max_ext, 2)

                # Corridor only spans from living's right edge to plot right edge
                corr_start_x = lp['x'] + lp['w']
                corr_w = round((ux + uw) - corr_start_x, 2)

                if corr_w > 2.0:
                    corridor_room = {
                        'room_type': 'corridor',
                        'name': 'Passage',
                        'zone': 'circulation',
                        'target_area': corr_w * corr_h_used,
                        'priority': 10,
                        '_placed': {
                            'x': round(corr_start_x, 2),
                            'y': round(corridor_y, 2),
                            'w': corr_w,
                            'h': round(corr_h_used, 2),
                        },
                    }
                    placed.append(corridor_room)
                living_found = True
                break

        if not living_found:
            # No living room — full-width corridor
            corridor_room = {
                'room_type': 'corridor',
                'name': 'Passage',
                'zone': 'circulation',
                'target_area': uw * corr_h_used,
                'priority': 10,
                '_placed': {
                    'x': round(ux, 2),
                    'y': round(corridor_y, 2),
                    'w': round(uw, 2),
                    'h': round(corr_h_used, 2),
                },
            }
            placed.append(corridor_room)

    # --- Place PRIVATE band (back/top of plot) ---
    placed.extend(_place_private_band_smart(
        priv_ordered, attached_pairs, rooms, serv,
        ux, priv_y, uw, priv_h_actual, total_area))

    # --- Post-placement optimization ---
    placed = _optimize_layout(placed, ux, uy, uw, ul)

    return placed


def _generate_and_score_candidates(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    public_rooms: List[Dict],
    private_rooms: List[Dict],
    service_rooms: List[Dict],
    corridor_h: float,
    ux: float,
    uy: float,
    uw: float,
    ul: float,
    total_area: float,
    n_candidates: int = 16,
) -> List[Dict]:
    """
    Generate multiple candidate layouts and return the best one.
    Generates 16 candidates with diverse ordering/corridor variations,
    scores each, and returns the highest-scoring layout.
    """
    best_score = -1
    best_layout = None
    all_scores = []

    for variant in range(n_candidates):
        try:
            candidate = _generate_candidate(
                rooms, attached_pairs,
                public_rooms, private_rooms, service_rooms,
                corridor_h, ux, uy, uw, ul, total_area, variant)

            score = _score_layout(
                candidate, rooms, attached_pairs,
                uw, ul, ux, uy)
            all_scores.append((variant, score))

            if score > best_score:
                best_score = score
                best_layout = candidate

        except Exception as e:
            logger.warning(f"Candidate {variant} failed: {e}")
            continue

    if best_layout is None:
        # Fallback: simple placement
        all_rooms = public_rooms + private_rooms + service_rooms
        best_layout = _place_rooms_in_band(all_rooms, ux, uy, uw, ul)

    logger.info(f"Best candidate: variant={all_scores[0][0] if all_scores else '?'}, "
                f"score={best_score:.3f} (of {len(all_scores)} candidates)")
    return best_layout


# =============================================================================
# TALL PLOT HANDLER (aspect < 0.75)
# =============================================================================

def _generate_tall_layout(
    rooms: List[Dict],
    attached_pairs: List[Tuple],
    public_rooms: List[Dict],
    private_rooms: List[Dict],
    service_rooms: List[Dict],
    corridor_h: float,
    ux: float,
    uy: float,
    uw: float,
    ul: float,
    total_area: float,
) -> List[Dict]:
    """
    Generate layout for tall/narrow plots using horizontal bands with
    max 2 rooms per band (side-by-side, each gets ~half the width).

    For narrow plots (width < 25ft), vertical columns would create
    unusable 3-4ft wide rooms. Instead, stack rooms in horizontal bands
    with at most 2 rooms per band:

      ┌──────┬──────┐
      │Bath1 │Bath2 │  ← Service band (7ft)
      ├──────┼──────┤
      │MBR   │ BR2  │  ← Private band (12ft)
      ├──────┴──────┤
      │ Corridor    │  ← 3.5ft
      ├──────┬──────┤
      │Kit   │Dining│  ← Public band 2 (8ft)
      ├──────┴──────┤
      │ Living Room │  ← Public band 1 (12ft)
      └─────────────┘
    """
    placed = []

    # Classify rooms
    living = [r for r in public_rooms if r['room_type'] == 'living']
    kitchen = [r for r in public_rooms if r['room_type'] == 'kitchen']
    dining = [r for r in public_rooms if r['room_type'] == 'dining']
    other_pub = [r for r in public_rooms
                 if r['room_type'] not in ('living', 'kitchen', 'dining')]

    bedrooms = [r for r in private_rooms
                if r['room_type'] in ('master_bedroom', 'bedroom')]
    attached_baths = [r for r in private_rooms if r.get('_is_attached_bath')]
    common_baths = [r for r in private_rooms
                    if r['room_type'] in ('bathroom', 'toilet')
                    and not r.get('_is_attached_bath')]
    other_priv = [r for r in private_rooms
                  if r['room_type'] not in ('master_bedroom', 'bedroom',
                                            'bathroom', 'toilet')]
    all_baths = attached_baths + common_baths + service_rooms

    # Build horizontal bands (each band = list of rooms, max 2 per band)
    bands = []

    # Band 1 (bottom): Living room gets full width
    if living:
        bands.append(living[:1])

    # Band 2: Kitchen + Dining (side by side)
    kit_din = kitchen + dining + other_pub
    if kit_din:
        if len(kit_din) <= 2:
            bands.append(kit_din)
        else:
            bands.append(kit_din[:2])
            bands.append(kit_din[2:])

    # Corridor band
    corr_h = min(corridor_h, 3.5)  # Narrower corridor for narrow plots

    # Band 3+: Bedrooms (pair them up, 2 per band)
    bed_bands = []
    all_beds = bedrooms + other_priv
    for i in range(0, len(all_beds), 2):
        chunk = all_beds[i:i+2]
        bed_bands.append(chunk)

    # Band 4: Bathrooms (pair them up)
    bath_bands = []
    for i in range(0, len(all_baths), 2):
        chunk = all_baths[i:i+2]
        bath_bands.append(chunk)

    # Calculate heights for each band
    all_bands = bands  # public bands
    corridor_idx = len(all_bands)  # where corridor goes
    all_bands = all_bands + bed_bands + bath_bands

    # Calculate proportional heights
    n_bands = len(all_bands)
    available_h = ul - corr_h
    band_targets = []
    for band in all_bands:
        target = sum(r['target_area'] for r in band)
        band_targets.append(max(target, 40))  # min 40 sqft per band

    total_target = sum(band_targets) or 1
    band_heights = []
    for i, (band, target) in enumerate(zip(all_bands, band_targets)):
        h = available_h * (target / total_target)
        # Enforce min heights based on room types
        has_bed = any(r['room_type'] in ('master_bedroom', 'bedroom', 'living') for r in band)
        has_bath = any(r['room_type'] in ('bathroom', 'toilet') for r in band)
        min_h = 10.0 if has_bed else (7.0 if has_bath else 7.0)
        # For single-room bands using full width, ensure AR is within limits
        if len(band) == 1:
            room_max_ar = _max_ar(band[0]['room_type'])
            # Room width = uw (full width) → min height = uw / max_ar
            ar_min_h = uw / room_max_ar
            min_h = max(min_h, ar_min_h)
        # Cap band height to prevent AR violations
        max_h = 30.0
        for r in band:
            room_max_ar = _max_ar(r['room_type'])
            # For multi-room bands, each room gets ~uw/n width
            room_w = uw / len(band)
            room_max_h = room_w * room_max_ar
            max_h = min(max_h, room_max_h)
        h = max(min(h, max_h), min_h)
        band_heights.append(h)

    # Scale to fit — preserving minimum heights
    total_h = sum(band_heights)
    if total_h > available_h:
        # Compute the minimum heights each band needs
        min_heights = []
        for band in all_bands:
            has_bed = any(r['room_type'] in ('master_bedroom', 'bedroom', 'living') for r in band)
            has_bath = any(r['room_type'] in ('bathroom', 'toilet') for r in band)
            mh = 10.0 if has_bed else (7.0 if has_bath else 7.0)
            if len(band) == 1:
                room_max_ar = _max_ar(band[0]['room_type'])
                ar_min = uw / room_max_ar
                mh = max(mh, ar_min)
            min_heights.append(mh)

        # First, set all bands to their minimums
        total_min = sum(min_heights)
        if total_min <= available_h:
            # Extra space beyond minimums — distribute proportionally
            extra = available_h - total_min
            above_min = [max(0, bh - mh) for bh, mh in zip(band_heights, min_heights)]
            total_above = sum(above_min) or 1
            band_heights = [
                mh + extra * (am / total_above)
                for mh, am in zip(min_heights, above_min)
            ]
        else:
            # Even minimums don't fit — scale minimums
            scale = available_h / total_min
            band_heights = [mh * scale for mh in min_heights]

    # Snap heights
    band_heights = [_snap(h) for h in band_heights]
    # Fix rounding
    used = sum(band_heights) + corr_h
    if abs(ul - used) > 0.3:
        band_heights[-1] += _snap(ul - used)

    # Place bands bottom to top
    cy = uy
    for band_idx, (band, bh) in enumerate(zip(all_bands, band_heights)):
        # Insert corridor after public bands
        if band_idx == corridor_idx and corr_h > 0:
            placed.append({
                'room_type': 'corridor',
                'name': 'Passage',
                'zone': 'circulation',
                'target_area': uw * corr_h,
                '_placed': {
                    'x': round(ux, 2), 'y': round(cy, 2),
                    'w': round(uw, 2), 'h': round(corr_h, 2),
                },
            })
            cy += corr_h

        # Place rooms in this band
        n = len(band)
        if n == 1:
            r = dict(band[0])
            rw = uw
            # Cap bathroom/service room widths in single-room bands
            if r['room_type'] in ('bathroom', 'toilet', 'utility', 'store'):
                max_w = 8.0
                rw = min(uw, max_w)
            r['_placed'] = {
                'x': round(ux, 2), 'y': round(cy, 2),
                'w': round(rw, 2), 'h': round(bh, 2),
            }
            placed.append(r)
            # If bathroom didn't fill the width, add a utility room
            if rw < uw - 3.0:
                placed.append({
                    'room_type': 'utility',
                    'name': 'Wash Area',
                    'zone': 'service',
                    'target_area': (uw - rw) * bh,
                    '_placed': {
                        'x': round(ux + rw, 2), 'y': round(cy, 2),
                        'w': round(uw - rw, 2), 'h': round(bh, 2),
                    },
                })
        elif n >= 2:
            # Split width proportionally to target areas
            areas = [r['target_area'] for r in band]
            total_a = sum(areas) or 1

            # Calculate initial proportional widths
            prop_widths = [uw * (a / total_a) for a in areas]

            # Apply minimum widths, but scale down if total mins > available
            min_widths = [MIN_DIMS.get(r['room_type'], (5, 5))[0] for r in band]
            total_mins = sum(min_widths)
            if total_mins > uw:
                # Scale down minimums proportionally to fit
                min_scale = uw / total_mins * 0.95  # 5% margin
                min_widths = [m * min_scale for m in min_widths]

            final_widths = [max(pw, mw) for pw, mw in zip(prop_widths, min_widths)]

            # Re-scale to fit exactly in uw
            total_fw = sum(final_widths)
            if total_fw > 0 and abs(total_fw - uw) > 0.5:
                scale = uw / total_fw
                final_widths = [w * scale for w in final_widths]

            cx = ux
            for i, r in enumerate(band):
                w = final_widths[i]
                if i == n - 1:
                    w = _snap((ux + uw) - cx)
                else:
                    w = _snap(w)
                rc = dict(r)
                rc['_placed'] = {
                    'x': round(cx, 2), 'y': round(cy, 2),
                    'w': round(w, 2), 'h': round(bh, 2),
                }
                placed.append(rc)
                cx += w

        cy += bh

    return placed


# =============================================================================
# POST-PROCESSING: FIX ROOM SIZING
# =============================================================================

def _fix_room_sizing(layout: List[Dict], total_area: float) -> List[Dict]:
    """
    Final post-processing to fix undersized bedrooms and oversized bathrooms.

    Architect's rules:
    1. No bedroom should be under MIN_AREAS (swap with oversized service room)
    2. No bathroom should exceed MAX_AREAS (shrink and redistribute)
    3. Bedrooms must have AR ≤ MAX_ASPECT
    """
    if not layout:
        return layout

    # Identify problems
    MAX_BATH_AREA = MAX_AREAS.get('bathroom', 60)
    undersized_beds = []
    oversized_baths = []

    for i, r in enumerate(layout):
        p = r.get('_placed')
        if not p:
            continue
        area = p['w'] * p['h']
        rtype = r.get('room_type', '')

        if rtype in ('master_bedroom', 'bedroom'):
            min_a = MIN_AREAS.get(rtype, 80)
            # Scale down for compact plots
            if total_area < 600:
                min_a = min_a * max(0.55, total_area / 600)
            if area < min_a * 0.85:  # Allow 15% tolerance
                undersized_beds.append(i)

        elif rtype == 'bathroom' and area > MAX_BATH_AREA:
            oversized_baths.append(i)

    # Swap undersized bedrooms with oversized bathrooms
    for bed_idx in undersized_beds:
        if not oversized_baths:
            break
        bath_idx = oversized_baths.pop(0)

        bed_r = layout[bed_idx]
        bath_r = layout[bath_idx]

        bed_p = bed_r['_placed']
        bath_p = bath_r['_placed']
        bed_area = bed_p['w'] * bed_p['h']
        bath_area = bath_p['w'] * bath_p['h']

        # Only swap if the bathroom's position would give the bedroom more area
        if bath_area > bed_area:
            # Swap positions and dimensions
            bed_p_copy = dict(bed_p)
            bath_p_copy = dict(bath_p)

            bed_r['_placed'] = bath_p_copy
            bath_r['_placed'] = bed_p_copy

    return layout


# =============================================================================
# MAIN API: GENERATE PROFESSIONAL PLAN
# =============================================================================

def generate_professional_plan(
    boundary_coords: List[Tuple[float, float]],
    rooms_config: Dict[str, int],
    total_area: float,
    front_door_pos: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Generate a professional-grade residential floor plan.

    This is the main entry point that implements the complete 5-phase
    architectural workflow.

    Args:
        boundary_coords: Plot boundary polygon coordinates
        rooms_config: Dict of room_type -> count (e.g., {'master_bedroom': 1, ...})
        total_area: Total plot area in sqft
        front_door_pos: Optional entrance position (defaults to center-south)

    Returns:
        centroids: Dict[str, List[Tuple[float, float]]] — room centroids
        sizes: Dict[str, List[Tuple[float, float]]] — room (width, height)
        room_specs: List[Dict] — room dicts with '_placed' positions
    """
    # --- Parse boundary ---
    try:
        from shapely.geometry import Polygon
        boundary_poly = Polygon(boundary_coords)
        minx, miny, maxx, maxy = boundary_poly.bounds
    except ImportError:
        xs = [c[0] for c in boundary_coords]
        ys = [c[1] for c in boundary_coords]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

    plot_w = maxx - minx
    plot_l = maxy - miny
    if total_area <= 0:
        total_area = plot_w * plot_l

    # --- Wide-plot transposition ---
    # If width > length × 1.5 AND depth < 15ft, horizontal bands create
    # narrow strip rooms with terrible AR. Transpose the boundary so we
    # generate as a tall plot (vertical bands), then un-transpose at the end.
    _transposed = False
    if plot_w > plot_l * 1.5 and plot_l < 15:
        _transposed = True
        minx, miny, maxx, maxy = miny, minx, maxy, maxx
        plot_w, plot_l = plot_l, plot_w

    # Usable area after external walls — SNAP to 0.5ft structural grid
    # so every room boundary falls on the grid. The wall inner face at
    # 0.75ft (9-inch wall) snaps inward to 1.0ft, giving us clean grid.
    ux = _snap_up(minx + WALL_EXT)
    uy = _snap_up(miny + WALL_EXT)
    uw = _snap_down(maxx - WALL_EXT) - ux
    ul = _snap_down(maxy - WALL_EXT) - uy
    aspect = uw / ul if ul > 0 else 1.0

    # --- Smart defaults: auto-add dining for 2BHK+ with enough area ---
    # Architect's rule: separate dining only in homes ≥ 1000 sqft.
    # Below 1000, dining merges with living ("Living cum Dining").
    total_beds = rooms_config.get('master_bedroom', 0) + rooms_config.get('bedroom', 0)
    if rooms_config.get('dining', 0) == 0 and total_beds >= 2 and total_area >= 1000:
        rooms_config['dining'] = 1
    # Auto-add utility for 3BHK+ if not specified
    if rooms_config.get('utility', 0) == 0 and total_beds >= 3 and total_area >= 1200:
        rooms_config['utility'] = 1

    # --- Phase 1: Room Programming ---
    rooms, attached_pairs = _build_room_program(rooms_config, total_area)

    # --- Phase 2: Zone-Band Planning ---
    public_rooms, private_rooms, service_rooms, corridor_h = _assign_to_bands(
        rooms, attached_pairs, total_area)

    # Ensure we have at least one public room
    if not public_rooms:
        public_rooms = [rooms[0]] if rooms else []

    # --- Phase 3-4: Generate candidates and pick best ---
    if aspect < 0.6:
        # Narrow plot (width/height < 0.6) — use stacked horizontal bands
        best_layout = _generate_tall_layout(
            rooms, attached_pairs,
            public_rooms, private_rooms, service_rooms, corridor_h,
            ux, uy, uw, ul, total_area)
    else:
        # Wide or square plot — use horizontal bands with 16 candidates
        best_layout = _generate_and_score_candidates(
            rooms, attached_pairs,
            public_rooms, private_rooms, service_rooms, corridor_h,
            ux, uy, uw, ul, total_area, n_candidates=16)

    # --- Post-processing: fix undersized bedrooms & oversized baths ---
    best_layout = _fix_room_sizing(best_layout, total_area)

    # --- Post-processing: ensure master bedroom hierarchy ---
    # Master bedroom should always be ≥ the largest regular bedroom.
    masters = [(i, r) for i, r in enumerate(best_layout)
               if r['room_type'] == 'master_bedroom' and r.get('_placed')]
    regulars = [(i, r) for i, r in enumerate(best_layout)
                if r['room_type'] == 'bedroom' and r.get('_placed')]
    if masters and regulars:
        for mi, mr in masters:
            mp = mr['_placed']
            mr_area = mp['w'] * mp['h']
            max_ri, max_rr = max(regulars, key=lambda t: t[1]['_placed']['w'] * t[1]['_placed']['h'])
            rp = max_rr['_placed']
            reg_area = rp['w'] * rp['h']
            if mr_area < reg_area and abs(mp['h'] - rp['h']) < 0.5:
                # Same band height — swap widths and x positions
                mp['w'], rp['w'] = rp['w'], mp['w']
                mp['x'], rp['x'] = rp['x'], mp['x']

    # --- Un-transpose if we transposed for wide-plot handling ---
    if _transposed:
        for r in best_layout:
            p = r.get('_placed')
            if p:
                p['x'], p['y'] = p['y'], p['x']
                p['w'], p['h'] = p['h'], p['w']

    # --- Build output format (with final grid snap) ---
    centroids = defaultdict(list)
    sizes = defaultdict(list)
    room_specs = []

    for r in best_layout:
        p = r.get('_placed', {})
        if not p:
            continue

        # Final grid snap — ensure all coordinates are on 0.5ft grid
        p['x'] = _snap(p['x'])
        p['y'] = _snap(p['y'])
        p['w'] = _snap(p['w'])
        p['h'] = _snap(p['h'])

        rtype = r['room_type']
        px, py = p['x'], p['y']
        pw, ph = p['w'], p['h']

        centroids[rtype].append((round(px + pw/2, 2), round(py + ph/2, 2)))
        sizes[rtype].append((round(pw, 2), round(ph, 2)))

        spec = dict(r)
        spec['zone_group'] = 'PROFESSIONAL'
        room_specs.append(spec)

    return dict(centroids), dict(sizes), room_specs
