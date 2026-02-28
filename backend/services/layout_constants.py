"""
Centralized Layout Constants — Single source of truth for all layout engines.

Loads from region_rules.json and exposes consistent constants for:
  - Room dimensions (min, ideal, max)
  - Area fractions and targets
  - Aspect ratio limits
  - Adjacency rules (desired + forbidden)
  - Vastu Shastra placement preferences
  - Wall thicknesses and grid snap
  - BHK configuration templates

All layout engines (pro_layout_engine, perfect_layout, gnn_engine, arch_engine)
should import from this module instead of defining their own constants.
This eliminates contradictions between engines and makes rules maintainable.

Author: CAD Layout Constants v1.0
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ===========================================================================
# LOAD REGION RULES FROM JSON
# ===========================================================================

_RULES_PATH = os.path.join(os.path.dirname(__file__), '..', 'region_rules.json')
_region_rules = {}

try:
    with open(_RULES_PATH, 'r') as f:
        _region_rules = json.load(f)
    logger.info(f"Loaded region rules from {_RULES_PATH}")
except Exception as e:
    logger.warning(f"Could not load region_rules.json: {e}. Using built-in defaults.")


def get_vastu_rules() -> Dict:
    """Get Vastu Shastra rules from region_rules.json."""
    return _region_rules.get('india_vastu', {})


def get_mvp_rules() -> Dict:
    """Get simplified MVP setback rules."""
    return _region_rules.get('india_mvp', {})


def get_bhk_config(bhk: str) -> Dict:
    """Get BHK configuration template (1BHK, 2BHK, 3BHK, 4BHK)."""
    vastu = get_vastu_rules()
    configs = vastu.get('bhk_configs', {})
    return configs.get(bhk, {})


def get_standard_room_sizes() -> Dict:
    """Get standard room sizes from region_rules.json (min/ideal/max in feet)."""
    vastu = get_vastu_rules()
    return vastu.get('standard_room_sizes_ft', {})


def get_architectural_rules() -> Dict:
    """Get architectural rules (wall thickness, passage width, etc.)."""
    vastu = get_vastu_rules()
    return vastu.get('architectural_rules', {})


def get_setbacks() -> Dict:
    """Get setback rules (front, rear, left, right in meters)."""
    vastu = get_vastu_rules()
    setbacks = vastu.get('setbacks', {})
    if not setbacks:
        mvp = get_mvp_rules()
        setbacks = mvp.get('setbacks', {
            'front': 3.0, 'rear': 2.0, 'left': 1.5, 'right': 1.5
        })
    return setbacks


# ===========================================================================
# STRUCTURAL CONSTANTS
# ===========================================================================

GRID_SNAP = 0.5  # 6-inch structural grid for wall alignment

WALL_EXTERNAL_FT = 0.75   # 9 inches = 0.75 ft
WALL_INTERNAL_FT = 0.375  # 4.5 inches = 0.375 ft

# ===========================================================================
# ROOM DIMENSION STANDARDS (feet)
# ===========================================================================

# Minimum room dimensions (width, length) — Indian residential standards
MIN_DIMS = {
    'living':         (10, 12),
    'master_bedroom': (10, 12),
    'bedroom':        (9, 10),
    'kitchen':        (7, 8),
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
    'corridor':       (3.0, 3.0),
    'hallway':        (3.0, 3.0),
    'foyer':          (5, 5),
    'porch':          (6, 5),
}

# Maximum aspect ratio per room type
MAX_ASPECT = {
    'living':         2.0,
    'master_bedroom': 1.8,
    'bedroom':        1.8,
    'kitchen':        2.0,
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
    'hallway':        5.0,
    'foyer':          2.0,
    'porch':          2.5,
}

# Hard minimum areas in sqft
MIN_AREAS = {
    'living': 100, 'master_bedroom': 100, 'bedroom': 80,
    'kitchen': 50, 'bathroom': 35, 'toilet': 15,
    'dining': 64, 'study': 48, 'pooja': 16,
    'store': 20, 'balcony': 15, 'utility': 16,
    'garage': 150, 'staircase': 36, 'hallway': 21,
    'foyer': 25, 'porch': 30,
}

# Hard maximum areas (prevent bloated service rooms)
MAX_AREAS = {
    'bathroom': 60, 'toilet': 30, 'pooja': 42,
    'store': 50, 'utility': 40, 'balcony': 55,
}

# ===========================================================================
# AREA FRACTIONS (proportion of total plot area)
# ===========================================================================

# (min, ideal, max) fractions
# Rule: bedrooms should be EQUAL or LARGER than living room.
# Living is shared; bedrooms are private and need bed+wardrobe+walking space.
AREA_FRACTIONS = {
    'living':         (0.10, 0.14, 0.18),
    'master_bedroom': (0.13, 0.17, 0.22),
    'bedroom':        (0.10, 0.14, 0.17),
    'kitchen':        (0.07, 0.10, 0.13),
    'bathroom':       (0.03, 0.045, 0.06),
    'toilet':         (0.015, 0.025, 0.035),
    'dining':         (0.06, 0.08, 0.10),
    'study':          (0.05, 0.07, 0.09),
    'pooja':          (0.02, 0.025, 0.035),
    'store':          (0.015, 0.025, 0.04),
    'balcony':        (0.02, 0.04, 0.06),
    'utility':        (0.015, 0.02, 0.03),
    'garage':         (0.08, 0.12, 0.16),
    'staircase':      (0.03, 0.045, 0.06),
}

# ===========================================================================
# ZONE CLASSIFICATION
# ===========================================================================

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
    'corridor': 'circulation',
    'hallway': 'circulation',
    'foyer': 'public',
    'porch': 'public',
}

# Room placement priority (higher = placed first, gets better position)
PRIORITY = {
    'living': 100, 'master_bedroom': 90, 'kitchen': 85,
    'dining': 80, 'bedroom': 75, 'study': 60,
    'bathroom': 55, 'toilet': 50, 'pooja': 45,
    'staircase': 40, 'balcony': 35, 'store': 30,
    'utility': 25, 'garage': 20, 'foyer': 65,
    'porch': 18, 'hallway': 15,
}

# ===========================================================================
# ADJACENCY RULES (unified across all engines)
# ===========================================================================

# Desired adjacencies — (room_a, room_b, strength)
# strength: 'required' = MUST share wall, 'preferred' = SHOULD share wall
DESIRED_ADJACENCIES = [
    ('master_bedroom', 'bathroom', 'required'),
    ('kitchen',        'dining',   'required'),
    ('living',         'kitchen',  'required'),
    ('living',         'dining',   'preferred'),
    ('kitchen',        'utility',  'preferred'),
    ('bedroom',        'bathroom', 'preferred'),
    ('pooja',          'living',   'preferred'),
    ('study',          'bedroom',  'preferred'),
]

# Forbidden adjacencies — these rooms should NOT share walls
FORBIDDEN_ADJACENCIES = [
    ('bedroom',        'kitchen'),
    ('master_bedroom', 'kitchen'),
    ('bathroom',       'living'),
    ('toilet',         'living'),
    ('toilet',         'kitchen'),
    ('toilet',         'dining'),
    ('pooja',          'toilet'),
    ('pooja',          'bathroom'),
    ('bathroom',       'dining'),
    ('toilet',         'pooja'),
]

# ===========================================================================
# VASTU SHASTRA PREFERENCES
# ===========================================================================

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
    'foyer':          ['E', 'N', 'NE'],
    'porch':          ['E', 'N', 'NE'],
}

# ===========================================================================
# HELPER: Validate room configuration against BHK standards
# ===========================================================================

def validate_bhk_config(
    total_area: float,
    bedrooms: int,
    bathrooms: int,
    extras: Optional[List[str]] = None,
) -> Dict:
    """
    Validate a room configuration against standard BHK ranges.
    Returns warnings/suggestions if the config doesn't match typical ranges.
    """
    extras = extras or []
    total_beds = bedrooms
    bhk_label = f"{total_beds}BHK"
    
    config = get_bhk_config(bhk_label)
    if not config:
        return {"valid": True, "warnings": [], "bhk": bhk_label}
    
    area_range = config.get("area_range", [0, 99999])
    warnings = []
    
    if total_area < area_range[0]:
        warnings.append(
            f"Plot area {total_area} sqft is small for {bhk_label} "
            f"(recommended {area_range[0]}-{area_range[1]} sqft)"
        )
    elif total_area > area_range[1]:
        warnings.append(
            f"Plot area {total_area} sqft is large for {bhk_label} "
            f"(recommended {area_range[0]}-{area_range[1]} sqft)"
        )
    
    std_rooms = config.get("rooms", {})
    std_baths = std_rooms.get("bathroom", 1)
    if bathrooms < std_baths:
        warnings.append(
            f"{bhk_label} typically has {std_baths} bathroom(s), "
            f"you specified {bathrooms}"
        )
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "bhk": bhk_label,
        "recommended_area_range": area_range,
    }
