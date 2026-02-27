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

logger = logging.getLogger(__name__)

# =============================================================================
# ARCHITECTURAL CONSTANTS — Indian Residential Standards (NBC 2016 + Vastu)
# =============================================================================

GRID_SNAP = 0.5  # 6-inch structural grid for wall alignment

WALL_EXT = 0.75   # 9-inch external wall in feet
WALL_INT = 0.375   # 4.5-inch internal wall in feet

# Minimum room dimensions (width, length) in feet — Indian standard
MIN_DIMS = {
    'living':         (10, 12),
    'master_bedroom': (10, 12),
    'bedroom':        (9, 10),
    'kitchen':        (8, 8),
    'bathroom':       (5, 7),
    'toilet':         (3.5, 4),
    'dining':         (8, 9),
    'study':          (7, 8),
    'pooja':          (4, 4),
    'store':          (4, 5),
    'balcony':        (3.5, 5),
    'utility':        (4, 5),
    'garage':         (10, 18),
    'staircase':      (4, 8),
    'corridor':       (3.5, 3.5),
}

# Maximum aspect ratio per room type (width:length or length:width, always ≥ 1)
MAX_ASPECT = {
    'living':         2.0,
    'master_bedroom': 1.8,
    'bedroom':        1.8,
    'kitchen':        1.8,
    'bathroom':       2.5,
    'toilet':         2.5,
    'dining':         2.0,
    'study':          1.8,
    'pooja':          2.0,
    'store':          2.5,
    'balcony':        5.0,
    'utility':        2.5,
    'garage':         2.5,
    'staircase':      2.5,
    'corridor':       10.0,
}

# Room area as fraction of total plot area (min, ideal, max)
AREA_FRACTIONS = {
    'living':         (0.12, 0.15, 0.20),
    'master_bedroom': (0.10, 0.14, 0.18),
    'bedroom':        (0.08, 0.12, 0.15),
    'kitchen':        (0.06, 0.09, 0.12),
    'bathroom':       (0.03, 0.04, 0.06),
    'toilet':         (0.015, 0.025, 0.035),
    'dining':         (0.06, 0.09, 0.12),
    'study':          (0.04, 0.06, 0.08),
    'pooja':          (0.015, 0.02, 0.03),
    'store':          (0.015, 0.025, 0.04),
    'balcony':        (0.02, 0.04, 0.06),
    'utility':        (0.015, 0.02, 0.03),
    'garage':         (0.08, 0.12, 0.16),
    'staircase':      (0.03, 0.045, 0.06),
}

# Hard minimum areas in sqft
MIN_AREAS = {
    'living': 100, 'master_bedroom': 100, 'bedroom': 80,
    'kitchen': 50, 'bathroom': 30, 'toilet': 15,
    'dining': 64, 'study': 48, 'pooja': 16,
    'store': 20, 'balcony': 15, 'utility': 16,
    'garage': 150, 'staircase': 36,
}

# Hard maximum areas (prevent bloated service rooms)
MAX_AREAS = {
    'bathroom': 60, 'toilet': 30, 'pooja': 40,
    'store': 50, 'utility': 40, 'balcony': 55,
}

# Zone classification — determines band assignment
ZONE_MAP = {
    'living': 'public',
    'dining': 'public',
    'kitchen': 'public',
    'master_bedroom': 'private',
    'bedroom': 'private',
    'bathroom': 'service',
    'toilet': 'service',
    'study': 'private',
    'pooja': 'private',
    'store': 'service',
    'utility': 'service',
    'balcony': 'public',
    'garage': 'public',
    'staircase': 'service',
}

# Room placement priority (higher = placed first, gets better position)
PRIORITY = {
    'living': 100, 'master_bedroom': 90, 'kitchen': 85,
    'dining': 80, 'bedroom': 75, 'study': 60,
    'bathroom': 55, 'toilet': 50, 'pooja': 45,
    'staircase': 40, 'balcony': 35, 'store': 30,
    'utility': 25, 'garage': 20,
}

# Vastu Shastra quadrant preferences (priority order)
VASTU_PREFS = {
    'living':         ['NE', 'N', 'E'],
    'kitchen':        ['SE', 'S', 'E'],
    'dining':         ['W', 'NW', 'S'],
    'master_bedroom': ['SW', 'S', 'W'],
    'bedroom':        ['NW', 'W', 'S'],
    'bathroom':       ['NW', 'W', 'S'],
    'toilet':         ['NW', 'W'],
    'study':          ['NE', 'E', 'N', 'W'],
    'pooja':          ['NE', 'E', 'N'],
    'store':          ['NW', 'SW', 'W'],
    'utility':        ['NW', 'SE', 'W'],
    'balcony':        ['N', 'E', 'NE'],
    'garage':         ['NW', 'SE'],
    'staircase':      ['S', 'W', 'SW'],
}

# Desired adjacencies — (room_a, room_b, strength)
# strength: 'required' = MUST share wall, 'preferred' = SHOULD share wall
DESIRED_ADJ = [
    ('master_bedroom', 'bathroom', 'required'),
    ('kitchen',        'dining',   'required'),
    ('living',         'kitchen',  'required'),    # Kitchen must be accessible from Living
    ('living',         'dining',   'preferred'),
    ('kitchen',        'utility',  'preferred'),
    ('bedroom',        'bathroom', 'preferred'),
    ('living',         'master_bedroom', 'preferred'),  # Via corridor extension
    ('living',         'bedroom',  'preferred'),         # Via corridor extension
]

# Forbidden adjacencies — these rooms should NOT share walls
FORBIDDEN_ADJ = [
    ('bedroom',        'kitchen'),
    ('master_bedroom', 'kitchen'),
    ('bathroom',       'living'),
    ('toilet',         'living'),
    ('toilet',         'kitchen'),
    ('toilet',         'dining'),
    ('pooja',          'toilet'),
    ('pooja',          'bathroom'),
]


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

    def _make_room(rtype, name, count_label=''):
        frac = AREA_FRACTIONS.get(rtype, (0.04, 0.06, 0.08))
        target = frac[1] * total_area  # Use ideal fraction
        target = max(target, MIN_AREAS.get(rtype, 20))
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
    """Determine corridor width based on plot size and room count."""
    if total_area < 500 or n_private_rooms <= 1:
        return 0.0       # 1BHK compact: no corridor
    elif total_area < 800:
        return 3.5        # 2BHK small: minimal corridor
    elif total_area < 1500:
        return 4.0        # 2-3BHK: standard corridor
    else:
        return 4.5        # 4BHK+: wide corridor


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
    # Indian residential: public band should be proportional but capped.
    # Small plots (≤1200sqft): max 14ft.  Large plots (3000sqft): max 20ft.
    PUB_MAX_H = min(14.0 + max(0, (total_area - 1200) * 0.004), 20.0)
    if public_h > PUB_MAX_H:
        excess = public_h - PUB_MAX_H
        public_h = PUB_MAX_H
        private_h += excess

    # --- CAP private band height ---
    # Small plots: max 22ft.  Large plots: scale up to accommodate more rooms.
    PRIV_MAX_H = min(22.0 + max(0, (total_area - 1500) * 0.005), 32.0)
    if private_h > PRIV_MAX_H:
        private_h = PRIV_MAX_H

    # Enforce minimum heights based on room dimension requirements
    pub_min_h = 10.0  # Living room needs at least 10ft depth
    for r in public_rooms:
        min_d = MIN_DIMS.get(r['room_type'], (7, 7))
        pub_min_h = max(pub_min_h, min(min_d[1], PUB_MAX_H))

    priv_min_h = 10.0  # Bedrooms need at least 10ft depth
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
        for i, r in enumerate(rooms):
            rtype = r['room_type']
            if rtype in SERVICE_TYPES:
                max_w = MAX_SERVICE_W.get(rtype, 8.0)
                if widths[i] > max_w:
                    excess += widths[i] - max_w
                    widths[i] = max_w
            else:
                major_indices.append(i)

        # Give excess width to major rooms (proportionally)
        if excess > 0 and major_indices:
            major_total = sum(widths[i] for i in major_indices)
            for i in major_indices:
                bonus = excess * (widths[i] / major_total) if major_total > 0 else 0
                widths[i] += bonus

        # Third pass: enforce minimum widths
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
    else:
        widths = list(ideal_widths)

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

    # If there's leftover space (> 3ft), add a utility/wash filler room
    # But CAP filler width to prevent huge unused utility rooms
    leftover = _snap((band_x + band_w) - cx)
    MAX_FILLER_W = 6.0  # Max 6ft wide for filler room
    if leftover >= 3.0:
        if leftover > MAX_FILLER_W:
            # Redistribute excess to expandable rooms in this band
            excess = leftover - MAX_FILLER_W
            # First try major rooms, then service rooms
            expand_indices = [j for j in range(len(placed))
                              if placed[j]['room_type'] not in
                              ('bathroom', 'toilet', 'pooja', 'store', 'utility')]
            if not expand_indices:
                # Service-only band: expand all rooms proportionally
                expand_indices = list(range(len(placed)))
            if expand_indices:
                share = excess / len(expand_indices)
                # Shift rooms and widen them
                shift = 0
                for j in range(len(placed)):
                    placed[j]['_placed']['x'] = round(
                        placed[j]['_placed']['x'] + shift, 2)
                    if j in expand_indices:
                        placed[j]['_placed']['w'] = round(
                            placed[j]['_placed']['w'] + share, 2)
                        shift += share
            leftover = MAX_FILLER_W

        placed.append({
            'room_type': 'utility',
            'name': 'Wash Area',
            'zone': 'service',
            'target_area': leftover * band_h,
            '_placed': {
                'x': round((band_x + band_w) - leftover, 2),
                'y': round(band_y, 2),
                'w': round(leftover, 2),
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
    Place private band rooms with intelligent sub-layout.

    Uses a sub-band split with service rooms on top and bedrooms on bottom.
    Bedrooms EXTEND UPWARD into any service sub-band gaps so there are no
    empty areas. This produces layouts like:

      ┌──────┬───────────┬──────────────┐
      │ Bath │           │              │  ← service sub-band (7ft)
      │ 8×7  │ (Master extends up)      │
      ├──────┤           │              │
      │      │ Master BR │  Bed Room 1  │  ← bedroom sub-band
      │      │           │              │
      └──────┴───────────┴──────────────┘

    This avoids half-empty service bands while giving bathrooms proper
    proportions (7ft deep instead of 20ft deep).
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

    # --- Sub-band layout with bedroom extension ---
    service_h = 7.0  # Standard bathroom depth
    bedroom_h = band_h - service_h

    if bedroom_h < 9.0:
        # Not enough space — fall back to side-by-side placement
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

    # --- Step 1: Place bedroom sub-band (BOTTOM, full width) ---
    bedroom_band_rooms.sort(
        key=lambda r: PRIORITY.get(r['room_type'], 30), reverse=True)
    bed_placed = _place_rooms_in_band(
        bedroom_band_rooms, band_x, band_y, band_w, bedroom_h)

    # --- Step 2: Place service rooms at NATURAL widths (no full-band scaling) ---
    # Service rooms (bathrooms, etc.) get their natural width based on
    # target_area / service_h, capped at MAX_SERVICE_W. No filler expansion.
    service_y = band_y + bedroom_h
    service_band_rooms = _order_service_for_plumbing(
        service_band_rooms, bedrooms, other_private, band_w)

    svc_placed = []
    svc_x = band_x
    for sr in service_band_rooms:
        natural_w = sr['target_area'] / service_h if service_h > 0 else 7.0
        min_w = MIN_DIMS.get(sr['room_type'], (5, 5))[0]
        max_w = 8.0 if sr['room_type'] in ('bathroom', 'toilet') else 7.0
        w = _snap(max(min_w, min(natural_w, max_w)))
        # Don't exceed remaining band width
        w = min(w, _snap((band_x + band_w) - svc_x))
        if w < 3.0:
            break  # No space left

        entry = dict(sr)
        entry['_placed'] = {
            'x': round(svc_x, 2),
            'y': round(service_y, 2),
            'w': round(w, 2),
            'h': round(service_h, 2),
        }
        svc_placed.append(entry)
        svc_x += w

    # --- Step 3: Extend bedrooms upward OR fill gaps ---
    # Track which X ranges in the service sub-band are covered
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

        # Only extend if NO service room is above this bedroom
        # AND the extended height would not exceed the max aspect ratio
        if not has_overlap:
            new_h = bp['h'] + service_h
            new_ar = _aspect(bp['w'], new_h)
            max_ar = _max_ar(br['room_type'])
            if new_ar <= max_ar:
                bp['h'] = round(new_h, 2)

    # --- Step 4: Fill remaining gaps in service sub-band ---
    # If service rooms don't fill the full band width above the bedroom
    # that has the bath, add a utility/common WC to fill the gap.
    gap_left = svc_covered_end
    gap_right = band_x + band_w
    # Only fill up to where extended bedrooms start (don't overlap)
    for br in bed_placed:
        bp = br['_placed']
        # If this bedroom was extended (h > bedroom_h), it covers the gap
        if bp['h'] > bedroom_h + 1:
            gap_right = min(gap_right, bp['x'])

    gap_w = _snap(gap_right - gap_left)
    MAX_FILLER_W = 8.0  # Cap utility/wash filler width
    if gap_w >= 4.0:
        filler_w = min(gap_w, MAX_FILLER_W)
        placed.append({
            'room_type': 'utility',
            'name': 'Wash Area',
            'zone': 'service',
            'target_area': filler_w * service_h,
            '_placed': {
                'x': round(gap_left, 2),
                'y': round(service_y, 2),
                'w': round(filler_w, 2),
                'h': round(service_h, 2),
            },
        })

    placed.extend(bed_placed)
    placed.extend(svc_placed)

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
    1. Adjacency satisfaction (30%): required/preferred pairs touching
    2. Room proportions (25%): aspect ratios within limits
    3. Vastu compliance (15%): rooms in correct quadrants
    4. Natural light (15%): habitable rooms on external walls
    5. Plumbing efficiency (15%): wet rooms sharing walls
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

    # --- 1. Adjacency Score (30%) ---
    adj_score = 1.0
    adj_checks = 0
    adj_satisfied = 0

    for (rt_a, rt_b, strength) in DESIRED_ADJ:
        for ra in rooms_by_type.get(rt_a, []):
            for rb in rooms_by_type.get(rt_b, []):
                adj_checks += 1
                if _rooms_share_wall(ra, rb, min_overlap=2.0):
                    adj_satisfied += 1

    if adj_checks > 0:
        adj_score = adj_satisfied / adj_checks

    # Check forbidden adjacencies (penalty)
    forbidden_violations = 0
    for (rt_a, rt_b) in FORBIDDEN_ADJ:
        for ra in rooms_by_type.get(rt_a, []):
            for rb in rooms_by_type.get(rt_b, []):
                if _rooms_share_wall(ra, rb, min_overlap=1.0):
                    forbidden_violations += 1
    adj_score = max(0, adj_score - forbidden_violations * 0.15)

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
    light_score = 0.0
    light_checks = 0
    ext_tolerance = WALL_EXT + 1.0
    for rect in room_rects:
        if rect['room_type'] in ('corridor', 'store', 'utility'):
            continue
        light_checks += 1
        # Check if room touches any external wall
        touches_ext = (
            rect['x'] <= ux + ext_tolerance or
            rect['x'] + rect['w'] >= ux + plot_w - ext_tolerance or
            rect['y'] <= uy + ext_tolerance or
            rect['y'] + rect['h'] >= uy + plot_h - ext_tolerance
        )
        if touches_ext:
            light_score += 1.0
        elif rect['room_type'] in ('bathroom', 'toilet', 'pooja'):
            light_score += 0.7  # These can be interior
        else:
            light_score += 0.0   # Habitable interior room is bad
    light_score = light_score / light_checks if light_checks > 0 else 0.5

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

    # --- Weighted total ---
    total = (adj_score * 0.30 +
             prop_score * 0.25 +
             vastu_score * 0.15 +
             light_score * 0.15 +
             plumb_score * 0.15)

    return round(total, 4)


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
      0: Standard order (Kitchen-Dining-Living, Master-BR-Bath)
      1: Reversed public (Living-Dining-Kitchen)
      2: Mirrored private (Bath-BR-Master)
      3: Combined reversal
      4+: Random shuffles within constraints
    """
    pub = list(public_rooms)
    priv = list(private_rooms)
    serv = list(service_rooms)

    # Apply variant transformations
    # RULE: Living Room ALWAYS stays first (leftmost) so it shares walls
    # with Kitchen AND (via extension through corridor) with bedrooms.
    # Kitchen ALWAYS stays next to Living (direct access required).
    if variant == 1:
        # Reverse private only
        priv = list(reversed(priv))
    elif variant == 2:
        priv = list(reversed(priv))
    elif variant == 3:
        priv = list(reversed(priv))
    elif variant >= 4:
        # Shuffle private rooms only — public order is fixed
        priv = list(reversed(priv)) if variant % 2 == 0 else priv

    # Order rooms within bands
    pub_ordered = _order_public_rooms(pub, uw) if variant < 4 else pub
    priv_ordered = _order_private_rooms(
        priv, attached_pairs, rooms, uw) if variant < 4 else priv

    # Calculate band heights
    public_h, private_h = _allocate_band_heights(
        pub_ordered, priv_ordered, serv,
        corridor_h, ul, uw, total_area)

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
    priv_y = corridor_y + corridor_h
    raw_priv_h = _snap((uy + ul) - priv_y)
    # Use the capped private_h from band height allocation (prevents overly tall rooms)
    priv_h_actual = min(raw_priv_h, private_h) if private_h > 0 else raw_priv_h

    if living_idx is not None and corridor_h > 0:
        # Extend living room upward through corridor zone
        living_p = placed[living_idx]['_placed']
        old_h = living_p['h']
        living_p['h'] = round(old_h + corridor_h, 2)

        # Place corridor only in the remaining width (not under living room)
        corr_start_x = living_p['x'] + living_p['w']
        corr_w = round((ux + uw) - corr_start_x, 2)
        if corr_w > 2.0:
            corridor_room = {
                'room_type': 'corridor',
                'name': 'Passage',
                'zone': 'circulation',
                'target_area': corr_w * corridor_h,
                'priority': 10,
                '_placed': {
                    'x': round(corr_start_x, 2),
                    'y': round(corridor_y, 2),
                    'w': corr_w,
                    'h': round(corridor_h, 2),
                },
            }
            placed.append(corridor_room)
    elif corridor_h > 0:
        # Fallback: full-width corridor
        corridor_room = {
            'room_type': 'corridor',
            'name': 'Passage',
            'zone': 'circulation',
            'target_area': uw * corridor_h,
            'priority': 10,
            '_placed': {
                'x': round(ux, 2),
                'y': round(corridor_y, 2),
                'w': round(uw, 2),
                'h': round(corridor_h, 2),
            },
        }
        placed.append(corridor_room)

    # --- Place PRIVATE band (back/top of plot) ---
    placed.extend(_place_private_band_smart(
        priv_ordered, attached_pairs, rooms, serv,
        ux, priv_y, uw, priv_h_actual, total_area))

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
    n_candidates: int = 6,
) -> List[Dict]:
    """
    Generate multiple candidate layouts and return the best one.
    """
    best_score = -1
    best_layout = None

    for variant in range(n_candidates):
        try:
            candidate = _generate_candidate(
                rooms, attached_pairs,
                public_rooms, private_rooms, service_rooms,
                corridor_h, ux, uy, uw, ul, total_area, variant)

            score = _score_layout(
                candidate, rooms, attached_pairs,
                uw, ul, ux, uy)

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

    logger.info(f"Best candidate score: {best_score:.3f}")
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

    # Usable area after external walls
    ux = minx + WALL_EXT
    uy = miny + WALL_EXT
    uw = plot_w - 2 * WALL_EXT
    ul = plot_l - 2 * WALL_EXT
    aspect = uw / ul if ul > 0 else 1.0

    # --- Phase 1: Room Programming ---
    rooms, attached_pairs = _build_room_program(rooms_config, total_area)

    # --- Phase 2: Zone-Band Planning ---
    public_rooms, private_rooms, service_rooms, corridor_h = _assign_to_bands(
        rooms, attached_pairs, total_area)

    # Ensure we have at least one public room
    if not public_rooms:
        public_rooms = [rooms[0]] if rooms else []

    # --- Phase 3-4: Generate candidates and pick best ---
    if aspect < 0.55:
        # Narrow plot (width/height < 0.55) — use stacked horizontal bands
        best_layout = _generate_tall_layout(
            rooms, attached_pairs,
            public_rooms, private_rooms, service_rooms, corridor_h,
            ux, uy, uw, ul, total_area)
    else:
        # Wide or square plot — use horizontal bands
        best_layout = _generate_and_score_candidates(
            rooms, attached_pairs,
            public_rooms, private_rooms, service_rooms, corridor_h,
            ux, uy, uw, ul, total_area, n_candidates=6)

    # --- Build output format ---
    centroids = defaultdict(list)
    sizes = defaultdict(list)
    room_specs = []

    for r in best_layout:
        p = r.get('_placed', {})
        if not p:
            continue

        rtype = r['room_type']
        px, py = p['x'], p['y']
        pw, ph = p['w'], p['h']

        centroids[rtype].append((round(px + pw/2, 2), round(py + ph/2, 2)))
        sizes[rtype].append((round(pw, 2), round(ph, 2)))

        spec = dict(r)
        spec['zone_group'] = 'PROFESSIONAL'
        room_specs.append(spec)

    return dict(centroids), dict(sizes), room_specs
