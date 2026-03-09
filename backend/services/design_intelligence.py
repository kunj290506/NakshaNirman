"""
Design Intelligence Module — 5 Layers of Architectural Intelligence

Layer 1: Family Profiler — detects family type, infers unstated needs
Layer 2: Climate Intelligence — India's 6 BIS Climate Zones
Layer 3: Vastu Shastra — complete 16-zone scoring
Layer 4: Architectural Rules — NBC 2016, acoustic, privacy, adjacency, light
Layer 5: Multi-Objective Optimizer — composite scoring + narrative generation
"""

import math
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# LAYER 1 — FAMILY PROFILER
# ════════════════════════════════════════════════════════════════

FAMILY_PROFILES = {
    "nuclear": {
        "keywords": ["nuclear", "small family", "couple with kids", "young family"],
        "needs": {
            "open_kitchen": True,
            "privacy_level": "medium",
            "storage": "standard",
            "dining_style": "attached",
        },
        "auto_rooms": [],
        "notes": [
            "Open-plan living and dining works great for nuclear families",
            "Kitchen can be semi-open to living area for supervision of children",
        ],
    },
    "joint_family": {
        "keywords": ["joint family", "joint", "parents", "in-laws", "large family",
                      "grandparents", "extended family", "multigenerational"],
        "needs": {
            "open_kitchen": False,
            "privacy_level": "high",
            "storage": "extra",
            "dining_style": "separate",
        },
        "auto_rooms": ["pooja", "dining"],
        "notes": [
            "Separate dining room recommended for joint family gatherings",
            "Dedicated pooja room is essential for daily family prayers",
            "Consider separate entry path to elder's bedroom for privacy",
        ],
    },
    "working_couple": {
        "keywords": ["working couple", "dink", "professionals", "work from home",
                      "wfh", "home office", "studio"],
        "needs": {
            "open_kitchen": True,
            "privacy_level": "low",
            "storage": "minimal",
            "dining_style": "breakfast_counter",
        },
        "auto_rooms": ["study"],
        "notes": [
            "Home office / study room included for WFH flexibility",
            "Open kitchen with breakfast counter suits modern lifestyle",
        ],
    },
    "elderly": {
        "keywords": ["elderly", "senior", "retired", "old age", "wheelchair",
                      "accessible", "barrier free"],
        "needs": {
            "open_kitchen": False,
            "privacy_level": "medium",
            "storage": "standard",
            "dining_style": "attached",
            "accessible": True,
        },
        "auto_rooms": ["pooja"],
        "notes": [
            "All rooms designed for ground floor accessibility",
            "Wider doorways (3ft minimum) for wheelchair access",
            "Attached bathroom to master bedroom is critical — no steps",
            "Anti-skid flooring recommended in bathrooms",
        ],
    },
    "rental": {
        "keywords": ["rental", "tenant", "investment", "rent", "income",
                      "paying guest", "pg"],
        "needs": {
            "open_kitchen": False,
            "privacy_level": "maximum",
            "storage": "standard",
            "dining_style": "none",
        },
        "auto_rooms": [],
        "notes": [
            "Each bedroom designed as independent unit with attached bath",
            "Separate entry possible for tenant privacy",
            "Minimal common areas to maximize rentable space",
        ],
    },
}

# Default if no match
_DEFAULT_PROFILE = {
    "type": "nuclear",
    "needs": FAMILY_PROFILES["nuclear"]["needs"],
    "auto_rooms": [],
    "notes": [],
}


def detect_family_profile(description: str, bedrooms: int = 2,
                           extras: List[str] = None) -> Dict:
    """Detect family type from description text and room count."""
    desc_lower = (description or "").lower()
    extras = extras or []

    # Try keyword matching
    for ftype, profile in FAMILY_PROFILES.items():
        for kw in profile["keywords"]:
            if kw in desc_lower:
                return {
                    "type": ftype,
                    "needs": profile["needs"],
                    "auto_rooms": profile["auto_rooms"],
                    "notes": profile["notes"],
                }

    # Heuristic inference
    if bedrooms >= 4:
        return {
            "type": "joint_family",
            "needs": FAMILY_PROFILES["joint_family"]["needs"],
            "auto_rooms": FAMILY_PROFILES["joint_family"]["auto_rooms"],
            "notes": ["4+ bedrooms suggests joint family — added pooja and separate dining"],
        }

    if "study" in extras or "office" in extras:
        return {
            "type": "working_couple",
            "needs": FAMILY_PROFILES["working_couple"]["needs"],
            "auto_rooms": [],
            "notes": ["Study/office room detected — optimizing for professional lifestyle"],
        }

    return dict(_DEFAULT_PROFILE)


# ════════════════════════════════════════════════════════════════
# LAYER 2 — CLIMATE INTELLIGENCE (India's 6 BIS Climate Zones)
# ════════════════════════════════════════════════════════════════

CLIMATE_ZONES = {
    "hot_dry": {
        "name": "Hot and Dry",
        "cities": [
            "ahmedabad", "jodhpur", "jaisalmer", "jaipur", "nagpur", "amritsar",
            "bikaner", "udaipur", "barmer", "ajmer", "rajkot", "bhuj",
            "hyderabad", "anantapur",
        ],
        "states": ["rajasthan", "gujarat"],
        "strategy": {
            "preferred_orientation": "north",
            "avoid_windows": "west",
            "maximize_windows": "north",
            "courtyard": True,
            "wall_thickness": "thick",
            "overhangs": "deep",
            "ventilation": "stack",
            "color_palette": "light",
        },
        "room_hints": {
            "living": {"orientation": "north", "reason": "Avoids harsh west sun"},
            "kitchen": {"orientation": "southeast", "reason": "Morning light, away from afternoon heat"},
            "master_bedroom": {"orientation": "south", "reason": "Cooler in evenings with thick south wall"},
            "bedroom": {"orientation": "north_or_east", "reason": "Minimal heat gain"},
        },
        "notes": [
            "Deep overhangs on west and south walls to block afternoon sun",
            "Consider internal courtyard for stack ventilation (hot air rises out)",
            "Light-colored external walls reflect heat",
            "Minimal west-facing windows — use high ventilators instead",
        ],
    },
    "warm_humid": {
        "name": "Warm and Humid",
        "cities": [
            "mumbai", "chennai", "kochi", "thiruvananthapuram", "mangalore",
            "goa", "panaji", "kozhikode", "calicut", "visakhapatnam",
            "bhubaneswar", "kolkata", "puducherry",
        ],
        "states": ["kerala", "goa", "karnataka_coast", "tamil_nadu_coast",
                    "west_bengal_coast"],
        "strategy": {
            "preferred_orientation": "any",
            "avoid_windows": None,
            "maximize_windows": "all_sides",
            "courtyard": False,
            "wall_thickness": "standard",
            "overhangs": "wide",
            "ventilation": "cross",
            "color_palette": "any",
        },
        "room_hints": {
            "living": {"orientation": "any", "reason": "Cross-ventilation is priority"},
            "kitchen": {"orientation": "any_external", "reason": "Must have exhaust + window for humidity"},
            "master_bedroom": {"orientation": "any", "reason": "Ensure two opposite windows for breeze"},
            "bedroom": {"orientation": "any", "reason": "Cross ventilation essential"},
        },
        "notes": [
            "Maximize cross-ventilation — every room should have windows on 2 walls",
            "Wide overhangs protect from monsoon rain while allowing air flow",
            "Verandah or sit-out recommended for tropical living",
            "Raised plinth (1.5ft+) protects from waterlogging",
        ],
    },
    "composite": {
        "name": "Composite",
        "cities": [
            "delhi", "new delhi", "lucknow", "kanpur", "allahabad", "prayagraj",
            "varanasi", "patna", "agra", "bhopal", "indore", "gwalior",
            "ranchi", "raipur",
        ],
        "states": ["uttar_pradesh", "madhya_pradesh", "bihar", "jharkhand",
                    "chhattisgarh", "haryana"],
        "strategy": {
            "preferred_orientation": "east_west_axis",
            "avoid_windows": "west",
            "maximize_windows": "south",
            "courtyard": True,
            "wall_thickness": "medium",
            "overhangs": "adjustable",
            "ventilation": "mixed",
            "color_palette": "medium",
        },
        "room_hints": {
            "living": {"orientation": "south", "reason": "Winter sun through south, shade in summer"},
            "kitchen": {"orientation": "southeast", "reason": "Morning warmth in winter"},
            "master_bedroom": {"orientation": "southwest", "reason": "Warm in winter, shaded in summer"},
            "bedroom": {"orientation": "east", "reason": "Morning sun, cool evenings"},
        },
        "notes": [
            "Building on E-W axis maximizes south exposure for harsh winters",
            "Operable windows: open for cross-vent in summer, closed in winter",
            "Internal courtyard helps with seasonal transitions",
        ],
    },
    "cold_cloudy": {
        "name": "Cold and Cloudy",
        "cities": [
            "shimla", "manali", "srinagar", "leh", "gangtok", "darjeeling",
            "dehradun", "mussoorie", "nainital", "shillong", "aizawl",
            "itanagar",
        ],
        "states": ["himachal_pradesh", "uttarakhand", "jammu_kashmir",
                    "sikkim", "arunachal_pradesh", "meghalaya"],
        "strategy": {
            "preferred_orientation": "south",
            "avoid_windows": "north",
            "maximize_windows": "south",
            "courtyard": False,
            "wall_thickness": "very_thick",
            "overhangs": "minimal",
            "ventilation": "controlled",
            "color_palette": "dark",
        },
        "room_hints": {
            "living": {"orientation": "south", "reason": "Maximum passive solar heat"},
            "kitchen": {"orientation": "south_or_east", "reason": "Cooking heat supplements room warmth"},
            "master_bedroom": {"orientation": "south", "reason": "Warm in winter sun"},
            "bedroom": {"orientation": "south_or_east", "reason": "Morning sun warmth"},
        },
        "notes": [
            "Large south-facing windows for passive solar heating",
            "Thick walls with insulation for thermal mass",
            "Minimize north-facing openings — heat loss",
            "Dark external colors absorb solar radiation",
            "Compact layout minimizes surface area for heat retention",
        ],
    },
    "cold_sunny": {
        "name": "Cold and Sunny",
        "cities": [
            "ladakh", "spiti", "lahaul",
        ],
        "states": ["ladakh"],
        "strategy": {
            "preferred_orientation": "south",
            "avoid_windows": "north",
            "maximize_windows": "south",
            "courtyard": False,
            "wall_thickness": "very_thick",
            "overhangs": "none",
            "ventilation": "minimal",
            "color_palette": "dark",
        },
        "room_hints": {
            "living": {"orientation": "south", "reason": "Direct solar gain"},
            "kitchen": {"orientation": "center", "reason": "Heat radiates to adjacent rooms"},
            "master_bedroom": {"orientation": "south", "reason": "Solar warmth"},
        },
        "notes": [
            "Trombe wall on south face recommended for passive solar heating",
            "No overhangs on south — maximize winter sun penetration",
            "Sunspace / greenhouse attached to south wall",
        ],
    },
    "moderate": {
        "name": "Moderate",
        "cities": [
            "bengaluru", "bangalore", "pune", "mysore", "mysuru", "coorg",
            "ooty", "kodaikanal", "mount_abu", "mahabaleshwar",
        ],
        "states": ["karnataka_plateau"],
        "strategy": {
            "preferred_orientation": "any",
            "avoid_windows": None,
            "maximize_windows": "east_south",
            "courtyard": False,
            "wall_thickness": "standard",
            "overhangs": "standard",
            "ventilation": "natural",
            "color_palette": "any",
        },
        "room_hints": {
            "living": {"orientation": "any", "reason": "Climate is balanced year-round"},
            "kitchen": {"orientation": "southeast", "reason": "Vastu-preferred, morning light"},
            "master_bedroom": {"orientation": "southwest", "reason": "Vastu-preferred"},
            "bedroom": {"orientation": "any", "reason": "No climate constraint"},
        },
        "notes": [
            "Moderate climate allows pure Vastu-driven layout without climate compromise",
            "Natural ventilation sufficient — no special orientation needed",
            "Year-round comfortable temperatures simplify design choices",
        ],
    },
}

# Default climate when no city/state detected
_DEFAULT_CLIMATE = CLIMATE_ZONES["composite"]


def detect_climate_zone(city: str = None, state: str = None,
                         description: str = None) -> Dict:
    """Detect India's BIS climate zone from city, state, or description text."""
    search_text = " ".join(filter(None, [city, state, description])).lower()
    search_text = re.sub(r'[^a-z\s]', '', search_text)

    if not search_text.strip():
        return dict(_DEFAULT_CLIMATE)

    # Search cities first (more specific)
    for zone_key, zone in CLIMATE_ZONES.items():
        for c in zone["cities"]:
            if c in search_text:
                return dict(zone)

    # Then search states
    for zone_key, zone in CLIMATE_ZONES.items():
        for s in zone["states"]:
            # Normalize state names
            s_clean = s.replace("_", " ")
            if s_clean in search_text:
                return dict(zone)

    return dict(_DEFAULT_CLIMATE)


# ════════════════════════════════════════════════════════════════
# LAYER 3 — VASTU SHASTRA (COMPLETE 16-ZONE SYSTEM)
# ════════════════════════════════════════════════════════════════

# 16 directional zones (8 cardinal + 8 intermediate + center)
VASTU_ZONES = {
    "N":   {"x_range": (0.33, 0.67), "y_range": (0.75, 1.00)},
    "NE":  {"x_range": (0.67, 1.00), "y_range": (0.75, 1.00)},
    "E":   {"x_range": (0.67, 1.00), "y_range": (0.33, 0.67)},
    "SE":  {"x_range": (0.67, 1.00), "y_range": (0.00, 0.33)},
    "S":   {"x_range": (0.33, 0.67), "y_range": (0.00, 0.25)},
    "SW":  {"x_range": (0.00, 0.33), "y_range": (0.00, 0.33)},
    "W":   {"x_range": (0.00, 0.33), "y_range": (0.33, 0.67)},
    "NW":  {"x_range": (0.00, 0.33), "y_range": (0.67, 1.00)},
    "CENTER": {"x_range": (0.33, 0.67), "y_range": (0.33, 0.67)},
}

# Room-to-Vastu zone mapping: ideal, good, acceptable, forbidden
VASTU_ROOM_RULES = {
    "pooja": {
        "ideal": ["NE"],
        "good": ["N", "E"],
        "acceptable": ["CENTER"],
        "forbidden": ["SW", "S", "SE", "W"],
        "door_facing": "east",
    },
    "kitchen": {
        "ideal": ["SE"],
        "good": ["S", "E", "NW"],
        "acceptable": ["W"],
        "forbidden": ["NE", "SW"],
        "platform_facing": "east",
    },
    "master_bedroom": {
        "ideal": ["SW"],
        "good": ["S", "W"],
        "acceptable": ["NW"],
        "forbidden": ["NE", "SE"],
        "headboard": "south",
    },
    "bedroom": {
        "ideal": ["S", "W", "NW"],
        "good": ["SW"],
        "acceptable": ["N"],
        "forbidden": ["NE", "SE"],
    },
    "living": {
        "ideal": ["N", "NE", "E"],
        "good": ["NW"],
        "acceptable": ["CENTER"],
        "forbidden": ["SW", "SE"],
    },
    "dining": {
        "ideal": ["W", "E"],
        "good": ["N", "NW"],
        "acceptable": ["S"],
        "forbidden": ["NE"],
    },
    "bathroom": {
        "ideal": ["NW", "W"],
        "good": ["S"],
        "acceptable": ["N"],
        "forbidden": ["NE", "SW", "CENTER"],
    },
    "toilet": {
        "ideal": ["NW", "W"],
        "good": ["S"],
        "acceptable": ["N"],
        "forbidden": ["NE", "SW", "CENTER", "SE"],
    },
    "study": {
        "ideal": ["N", "E", "NE"],
        "good": ["W"],
        "acceptable": ["NW"],
        "forbidden": ["SW", "SE"],
    },
    "store": {
        "ideal": ["NW", "SW"],
        "good": ["W", "S"],
        "acceptable": ["N"],
        "forbidden": ["NE"],
    },
    "staircase": {
        "ideal": ["S", "W", "SW"],
        "good": ["NW"],
        "acceptable": ["SE"],
        "forbidden": ["NE", "CENTER"],
    },
    "garage": {
        "ideal": ["NW", "SE"],
        "good": ["W", "S"],
        "acceptable": ["E"],
        "forbidden": ["NE", "SW"],
    },
    "balcony": {
        "ideal": ["N", "E", "NE"],
        "good": ["NW"],
        "acceptable": ["SE"],
        "forbidden": ["SW"],
    },
    "passage": {
        "ideal": ["CENTER"],
        "good": ["N", "E"],
        "acceptable": ["W", "S"],
        "forbidden": [],
    },
    "utility": {
        "ideal": ["NW"],
        "good": ["W"],
        "acceptable": ["SE"],
        "forbidden": ["NE"],
    },
}

# Special Vastu rules with point adjustments
VASTU_SPECIAL_RULES = [
    {
        "id": "pooja_ne",
        "description": "Pooja room in NE (ideal Vastu)",
        "check": lambda rooms: any(
            _room_in_zone(r, "NE") for r in rooms
            if _room_type_match(r, "pooja")
        ),
        "bonus": 10,
        "penalty": 0,
    },
    {
        "id": "toilet_ne",
        "description": "Toilet/Bathroom in NE (forbidden)",
        "check": lambda rooms: any(
            _room_in_zone(r, "NE") for r in rooms
            if _room_type_match(r, ("bathroom", "toilet"))
        ),
        "bonus": 0,
        "penalty": -15,
    },
    {
        "id": "kitchen_ne",
        "description": "Kitchen in NE (forbidden)",
        "check": lambda rooms: any(
            _room_in_zone(r, "NE") for r in rooms
            if _room_type_match(r, "kitchen")
        ),
        "bonus": 0,
        "penalty": -12,
    },
    {
        "id": "master_sw",
        "description": "Master Bedroom in SW (ideal)",
        "check": lambda rooms: any(
            _room_in_zone(r, "SW") for r in rooms
            if _room_type_match(r, "master_bedroom")
        ),
        "bonus": 8,
        "penalty": 0,
    },
    {
        "id": "kitchen_se",
        "description": "Kitchen in SE — Agni corner (ideal)",
        "check": lambda rooms: any(
            _room_in_zone(r, "SE") for r in rooms
            if _room_type_match(r, "kitchen")
        ),
        "bonus": 10,
        "penalty": 0,
    },
    {
        "id": "living_ne",
        "description": "Living Room in N/NE/E (ideal)",
        "check": lambda rooms: any(
            _room_in_zone(r, ("N", "NE", "E")) for r in rooms
            if _room_type_match(r, "living")
        ),
        "bonus": 6,
        "penalty": 0,
    },
    {
        "id": "brahmasthana",
        "description": "Center of house (Brahmasthana) should be open/passage",
        "check": lambda rooms: not any(
            _room_in_zone(r, "CENTER") for r in rooms
            if _room_type_match(r, ("bathroom", "toilet", "kitchen", "store"))
        ),
        "bonus": 5,
        "penalty": 0,
    },
    {
        "id": "staircase_center",
        "description": "Staircase in center (forbidden)",
        "check": lambda rooms: any(
            _room_in_zone(r, "CENTER") for r in rooms
            if _room_type_match(r, "staircase")
        ),
        "bonus": 0,
        "penalty": -10,
    },
]


def _room_type_match(room: Dict, types) -> bool:
    """Check if room matches given type(s)."""
    rtype = room.get("room_type", room.get("type", "")).lower().replace(" ", "_")
    if isinstance(types, str):
        types = (types,)
    return rtype in types


def _get_room_center(room: Dict, plot_width: float, plot_length: float) -> Tuple[float, float]:
    """Get normalized room center (0-1 range) within the plot."""
    x = room.get("x", 0)
    y = room.get("y", 0)
    w = room.get("width", room.get("w", 10))
    h = room.get("height", room.get("h", room.get("depth", 10)))

    cx = (x + w / 2) / plot_width if plot_width > 0 else 0.5
    cy = (y + h / 2) / plot_length if plot_length > 0 else 0.5

    return (max(0, min(1, cx)), max(0, min(1, cy)))


def _room_in_zone(room: Dict, zones, pw: float = 30, pl: float = 40) -> bool:
    """Check if room center falls within given Vastu zone(s)."""
    if isinstance(zones, str):
        zones = (zones,)

    cx, cy = _get_room_center(room, pw, pl)

    for zone_name in zones:
        zone = VASTU_ZONES.get(zone_name)
        if not zone:
            continue
        xr = zone["x_range"]
        yr = zone["y_range"]
        if xr[0] <= cx <= xr[1] and yr[0] <= cy <= yr[1]:
            return True
    return False


def _determine_room_zone(room: Dict, pw: float, pl: float) -> str:
    """Determine which Vastu zone a room is in."""
    cx, cy = _get_room_center(room, pw, pl)

    best_zone = "CENTER"
    best_dist = float("inf")

    for zone_name, zone in VASTU_ZONES.items():
        zx = (zone["x_range"][0] + zone["x_range"][1]) / 2
        zy = (zone["y_range"][0] + zone["y_range"][1]) / 2
        dist = math.sqrt((cx - zx) ** 2 + (cy - zy) ** 2)

        xr = zone["x_range"]
        yr = zone["y_range"]
        if xr[0] <= cx <= xr[1] and yr[0] <= cy <= yr[1]:
            if dist < best_dist:
                best_dist = dist
                best_zone = zone_name

    return best_zone


def score_vastu(rooms: List[Dict], plot_width: float,
                plot_length: float) -> Dict:
    """Score Vastu compliance of a layout. Returns 0-100 score with details."""
    if not rooms:
        return {"score": 50, "grade": "C", "bonuses": [], "violations": [], "details": []}

    total_score = 70  # Base score
    bonuses = []
    violations = []
    details = []

    # Score each room's zone placement
    for room in rooms:
        rtype = room.get("room_type", room.get("type", "")).lower().replace(" ", "_")
        rules = VASTU_ROOM_RULES.get(rtype)
        if not rules:
            continue

        zone = _determine_room_zone(room, plot_width, plot_length)
        room_name = room.get("room_type", room.get("type", rtype)).replace("_", " ").title()

        if zone in rules.get("ideal", []):
            total_score += 3
            bonuses.append({
                "room": rtype,
                "zone": zone,
                "message": f"{room_name} in {zone} — ideal Vastu placement",
            })
            details.append(f"✓ {room_name} in {zone} (ideal)")
        elif zone in rules.get("good", []):
            total_score += 1
            details.append(f"✓ {room_name} in {zone} (good)")
        elif zone in rules.get("forbidden", []):
            total_score -= 5
            violations.append({
                "room": rtype,
                "zone": zone,
                "severity": "critical",
                "message": f"{room_name} in {zone} — Vastu violation, move to {' or '.join(rules['ideal'])}",
            })
            details.append(f"✗ {room_name} in {zone} (forbidden — move to {'/'.join(rules['ideal'])})")
        else:
            details.append(f"○ {room_name} in {zone} (acceptable)")

    # Apply special rules
    for rule in VASTU_SPECIAL_RULES:
        try:
            triggered = rule["check"](rooms)
            if triggered and rule["bonus"] > 0:
                total_score += rule["bonus"]
                bonuses.append({
                    "room": rule["id"],
                    "zone": "",
                    "message": rule["description"] + " (+bonus)",
                })
            elif triggered and rule["penalty"] < 0:
                total_score += rule["penalty"]
                violations.append({
                    "room": rule["id"],
                    "zone": "",
                    "severity": "critical",
                    "message": rule["description"] + " (penalty)",
                })
        except Exception:
            pass

    # Clamp score
    total_score = max(0, min(100, total_score))

    # Grade
    if total_score >= 90:
        grade = "A+"
    elif total_score >= 80:
        grade = "A"
    elif total_score >= 70:
        grade = "B+"
    elif total_score >= 60:
        grade = "B"
    elif total_score >= 50:
        grade = "C"
    else:
        grade = "D"

    return {
        "score": total_score,
        "grade": grade,
        "bonuses": bonuses,
        "violations": violations,
        "details": details,
    }


# ════════════════════════════════════════════════════════════════
# LAYER 4 — ARCHITECTURAL RULES
# ════════════════════════════════════════════════════════════════

# NBC 2016 minimum room sizes (sq ft)
NBC_MINIMUMS = {
    "master_bedroom": {"min_area": 120, "min_width": 10, "min_height": 9.5},
    "bedroom":        {"min_area": 95,  "min_width": 9,  "min_height": 9.5},
    "kitchen":        {"min_area": 50,  "min_width": 6,  "min_height": 9.5},
    "bathroom":       {"min_area": 28,  "min_width": 4,  "min_height": 7},
    "toilet":         {"min_area": 15,  "min_width": 3,  "min_height": 7},
    "living":         {"min_area": 120, "min_width": 10, "min_height": 9.5},
    "dining":         {"min_area": 80,  "min_width": 9,  "min_height": 9.5},
    "pooja":          {"min_area": 16,  "min_width": 4,  "min_height": 8},
    "study":          {"min_area": 50,  "min_width": 7,  "min_height": 9.5},
    "store":          {"min_area": 20,  "min_width": 4,  "min_height": 8},
    "passage":        {"min_area": 0,   "min_width": 3.5, "min_height": 9.5},
    "staircase":      {"min_area": 0,   "min_width": 3,  "min_height": 0},
    "balcony":        {"min_area": 0,   "min_width": 4,  "min_height": 0},
    "garage":         {"min_area": 100, "min_width": 8,  "min_height": 0},
    "utility":        {"min_area": 20,  "min_width": 4,  "min_height": 8},
}

# Acoustic separation matrix: pairs that MUST NOT be adjacent
ACOUSTIC_CRITICAL = [
    ("bedroom", "kitchen"),
    ("master_bedroom", "kitchen"),
    ("study", "kitchen"),
    ("study", "living"),
    ("pooja", "bathroom"),
    ("pooja", "toilet"),
    ("bedroom", "garage"),
    ("master_bedroom", "garage"),
]

# Adjacency preferences (bonus for being adjacent)
ADJACENCY_PREFERRED = {
    ("kitchen", "dining"): 10,      # Critical — must be adjacent
    ("master_bedroom", "bathroom"): 8,  # Attached bath
    ("living", "dining"): 6,
    ("living", "entrance"): 5,
    ("kitchen", "utility"): 5,
    ("bedroom", "bathroom"): 5,
}

# Privacy gradient: 1 = most public, 10 = most private
PRIVACY_LEVELS = {
    "entrance": 1, "living": 2, "dining": 3, "kitchen": 4, "study": 5,
    "passage": 3, "pooja": 5, "store": 5, "utility": 5, "balcony": 2,
    "staircase": 3, "garage": 2,
    "bedroom": 8, "master_bedroom": 9, "bathroom": 10, "toilet": 10,
}


def _rooms_adjacent(r1: Dict, r2: Dict, threshold: float = 1.0) -> bool:
    """Check if two rooms share a wall (within threshold distance)."""
    x1, y1 = r1.get("x", 0), r1.get("y", 0)
    w1 = r1.get("width", r1.get("w", 10))
    h1 = r1.get("height", r1.get("h", r1.get("depth", 10)))

    x2, y2 = r2.get("x", 0), r2.get("y", 0)
    w2 = r2.get("width", r2.get("w", 10))
    h2 = r2.get("height", r2.get("h", r2.get("depth", 10)))

    # Check horizontal adjacency (share right-left wall)
    h_adj = (abs((x1 + w1) - x2) <= threshold or abs((x2 + w2) - x1) <= threshold)
    h_overlap = not (y1 + h1 <= y2 or y2 + h2 <= y1)

    # Check vertical adjacency (share top-bottom wall)
    v_adj = (abs((y1 + h1) - y2) <= threshold or abs((y2 + h2) - y1) <= threshold)
    v_overlap = not (x1 + w1 <= x2 or x2 + w2 <= x1)

    return (h_adj and h_overlap) or (v_adj and v_overlap)


def _room_touches_external_wall(room: Dict, plot_width: float,
                                 plot_length: float, threshold: float = 1.0) -> bool:
    """Check if room has at least one external wall (for natural light)."""
    x = room.get("x", 0)
    y = room.get("y", 0)
    w = room.get("width", room.get("w", 10))
    h = room.get("height", room.get("h", room.get("depth", 10)))

    touches_left = x <= threshold
    touches_right = (x + w) >= (plot_width - threshold)
    touches_bottom = y <= threshold
    touches_top = (y + h) >= (plot_length - threshold)

    return touches_left or touches_right or touches_bottom or touches_top


def score_nbc_compliance(rooms: List[Dict]) -> Dict:
    """Check NBC 2016 minimum room size compliance."""
    score = 100
    issues = []

    for room in rooms:
        rtype = room.get("room_type", room.get("type", "")).lower().replace(" ", "_")
        nbc = NBC_MINIMUMS.get(rtype)
        if not nbc:
            continue

        w = room.get("width", room.get("w", 10))
        h = room.get("height", room.get("h", room.get("depth", 10)))
        area = w * h

        if nbc["min_area"] > 0 and area < nbc["min_area"]:
            deficit = nbc["min_area"] - area
            score -= min(15, deficit / nbc["min_area"] * 30)
            room_name = rtype.replace("_", " ").title()
            issues.append({
                "category": "NBC",
                "severity": "critical",
                "message": f"{room_name}: {area:.0f} sqft below NBC minimum {nbc['min_area']} sqft",
            })

        if nbc["min_width"] > 0 and min(w, h) < nbc["min_width"]:
            score -= 5
            room_name = rtype.replace("_", " ").title()
            issues.append({
                "category": "NBC",
                "severity": "warning",
                "message": f"{room_name}: {min(w, h):.1f}ft width below minimum {nbc['min_width']}ft",
            })

    return {"score": max(0, min(100, score)), "issues": issues}


def score_acoustic(rooms: List[Dict]) -> Dict:
    """Score acoustic separation of the layout."""
    score = 100
    issues = []

    for r1 in rooms:
        for r2 in rooms:
            if r1 is r2:
                continue
            t1 = r1.get("room_type", r1.get("type", "")).lower().replace(" ", "_")
            t2 = r2.get("room_type", r2.get("type", "")).lower().replace(" ", "_")

            for pair in ACOUSTIC_CRITICAL:
                if (t1 in pair and t2 in pair and t1 != t2):
                    if _rooms_adjacent(r1, r2):
                        score -= 12
                        n1 = t1.replace("_", " ").title()
                        n2 = t2.replace("_", " ").title()
                        issues.append({
                            "category": "Acoustic",
                            "severity": "warning",
                            "message": f"{n1} adjacent to {n2} — acoustic conflict",
                        })

    return {"score": max(0, min(100, score)), "issues": issues}


def score_natural_light(rooms: List[Dict], plot_width: float,
                         plot_length: float) -> Dict:
    """Score natural light access — habitable rooms must touch external wall."""
    habitable = {"living", "dining", "bedroom", "master_bedroom", "kitchen", "study"}
    score = 100
    issues = []

    habitable_rooms = [
        r for r in rooms
        if r.get("room_type", r.get("type", "")).lower().replace(" ", "_") in habitable
    ]

    if not habitable_rooms:
        return {"score": 100, "issues": []}

    for room in habitable_rooms:
        rtype = room.get("room_type", room.get("type", "")).lower().replace(" ", "_")
        if not _room_touches_external_wall(room, plot_width, plot_length):
            score -= 15
            room_name = rtype.replace("_", " ").title()
            issues.append({
                "category": "Natural Light",
                "severity": "critical",
                "message": f"{room_name} has no external wall — no natural light or ventilation",
            })

    return {"score": max(0, min(100, score)), "issues": issues}


def score_circulation(rooms: List[Dict], plot_width: float,
                       plot_length: float) -> Dict:
    """Score circulation efficiency — privacy gradient from entrance to bedrooms."""
    score = 85  # Base
    issues = []

    # Find entrance zone rooms and private rooms
    entrance_rooms = [
        r for r in rooms
        if r.get("room_type", r.get("type", "")).lower().replace(" ", "_")
        in ("living", "entrance", "passage")
    ]
    private_rooms = [
        r for r in rooms
        if r.get("room_type", r.get("type", "")).lower().replace(" ", "_")
        in ("bedroom", "master_bedroom", "bathroom")
    ]

    # Check that private rooms are not directly at the entrance side
    for pr in private_rooms:
        py = pr.get("y", 0)
        ph = pr.get("height", pr.get("h", pr.get("depth", 10)))
        rtype = pr.get("room_type", pr.get("type", "")).lower().replace(" ", "_")

        # Entrance is typically at y=0 (front of plot)
        if py <= 1.0:
            score -= 8
            room_name = rtype.replace("_", " ").title()
            issues.append({
                "category": "Circulation",
                "severity": "warning",
                "message": f"{room_name} is at the entrance — poor privacy gradient",
            })

    # Bonus for adjacency preferences
    adjacency_score = 0
    adjacency_possible = 0
    for (t1, t2), bonus in ADJACENCY_PREFERRED.items():
        rooms_t1 = [r for r in rooms if r.get("room_type", r.get("type", "")).lower().replace(" ", "_") == t1]
        rooms_t2 = [r for r in rooms if r.get("room_type", r.get("type", "")).lower().replace(" ", "_") == t2]
        if rooms_t1 and rooms_t2:
            adjacency_possible += bonus
            for r1 in rooms_t1:
                for r2 in rooms_t2:
                    if _rooms_adjacent(r1, r2):
                        adjacency_score += bonus
                        break

    if adjacency_possible > 0:
        adj_ratio = adjacency_score / adjacency_possible
        score += int(adj_ratio * 15)

    return {"score": max(0, min(100, score)), "issues": issues}


def score_area_efficiency(rooms: List[Dict], plot_width: float,
                           plot_length: float) -> Dict:
    """Score how efficiently the plot area is used."""
    total_plot = plot_width * plot_length
    if total_plot <= 0:
        return {"score": 50, "issues": []}

    total_room_area = 0
    for room in rooms:
        w = room.get("width", room.get("w", 0))
        h = room.get("height", room.get("h", room.get("depth", 0)))
        total_room_area += w * h

    coverage = total_room_area / total_plot
    issues = []

    # Ideal coverage: 75-90%
    if coverage >= 0.75 and coverage <= 0.92:
        score = 90 + int((coverage - 0.75) / 0.17 * 10)
    elif coverage > 0.92:
        score = 85  # Too tight — no breathing room
        issues.append({
            "category": "Area",
            "severity": "info",
            "message": f"Coverage {coverage*100:.0f}% — very dense, consider reducing room sizes",
        })
    elif coverage >= 0.60:
        score = 60 + int((coverage - 0.60) / 0.15 * 30)
    else:
        score = max(30, int(coverage * 100))
        issues.append({
            "category": "Area",
            "severity": "warning",
            "message": f"Coverage only {coverage*100:.0f}% — significant wasted space",
        })

    return {"score": max(0, min(100, score)), "issues": issues}


# ════════════════════════════════════════════════════════════════
# LAYER 5 — MULTI-OBJECTIVE OPTIMIZER + NARRATIVE
# ════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "vastu": 0.25,
    "area_efficiency": 0.20,
    "circulation": 0.15,
    "natural_light": 0.15,
    "acoustic": 0.10,
    "nbc_compliance": 0.15,
}


def _adjust_weights(climate: Dict, family: Dict,
                     description: str = "") -> Dict[str, float]:
    """Auto-tune scoring weights by context."""
    w = dict(DEFAULT_WEIGHTS)
    desc_lower = (description or "").lower()

    # Climate adjustments
    climate_name = climate.get("name", "").lower()
    if "cold" in climate_name:
        w["natural_light"] += 0.05
        w["area_efficiency"] -= 0.05
    elif "hot" in climate_name:
        w["circulation"] += 0.05
        w["natural_light"] -= 0.05

    # Family adjustments
    ftype = family.get("type", "")
    if ftype == "elderly":
        w["circulation"] += 0.05
        w["vastu"] -= 0.05
    elif ftype == "joint_family":
        w["vastu"] += 0.05
        w["acoustic"] += 0.03
        w["area_efficiency"] -= 0.08

    # User priority
    if "vastu" in desc_lower:
        w["vastu"] += 0.10
        w["area_efficiency"] -= 0.05
        w["circulation"] -= 0.05

    # Normalize to sum = 1.0
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


def score_layout(rooms: List[Dict], plot_width: float, plot_length: float,
                  engine_input: Dict = None) -> Dict:
    """
    Compute composite design score across all 6 dimensions.

    Returns:
        {
            "composite": int (0-100),
            "grade": str (A+ to D),
            "breakdown": { dimension: {"score": int, "weight": float} },
            "issues": [ {"category": str, "severity": str, "message": str} ],
            "vastu_bonuses": [ {"room": str, "zone": str, "message": str} ],
            "climate_zone": str,
            "family_type": str,
        }
    """
    engine_input = engine_input or {}

    # Detect context
    climate = detect_climate_zone(
        engine_input.get("city"),
        engine_input.get("state"),
        engine_input.get("description"),
    )
    family = detect_family_profile(
        engine_input.get("description", ""),
        engine_input.get("bedrooms", 2),
        engine_input.get("extras"),
    )
    weights = _adjust_weights(climate, family, engine_input.get("description", ""))

    # Score each dimension
    vastu_result = score_vastu(rooms, plot_width, plot_length)
    nbc_result = score_nbc_compliance(rooms)
    acoustic_result = score_acoustic(rooms)
    light_result = score_natural_light(rooms, plot_width, plot_length)
    circ_result = score_circulation(rooms, plot_width, plot_length)
    area_result = score_area_efficiency(rooms, plot_width, plot_length)

    breakdown = {
        "vastu":           {"score": vastu_result["score"],   "weight": round(weights["vastu"], 2)},
        "area_efficiency": {"score": area_result["score"],    "weight": round(weights["area_efficiency"], 2)},
        "circulation":     {"score": circ_result["score"],    "weight": round(weights["circulation"], 2)},
        "natural_light":   {"score": light_result["score"],   "weight": round(weights["natural_light"], 2)},
        "acoustic":        {"score": acoustic_result["score"],"weight": round(weights["acoustic"], 2)},
        "nbc_compliance":  {"score": nbc_result["score"],     "weight": round(weights["nbc_compliance"], 2)},
    }

    # Composite score
    composite = sum(
        breakdown[dim]["score"] * breakdown[dim]["weight"]
        for dim in breakdown
    )
    composite = max(0, min(100, int(round(composite))))

    # Grade
    if composite >= 90:
        grade = "A+"
    elif composite >= 80:
        grade = "A"
    elif composite >= 70:
        grade = "B+"
    elif composite >= 60:
        grade = "B"
    elif composite >= 50:
        grade = "C"
    else:
        grade = "D"

    # Collect all issues
    all_issues = (
        vastu_result.get("violations", [])
        + nbc_result["issues"]
        + acoustic_result["issues"]
        + light_result["issues"]
        + circ_result["issues"]
        + area_result["issues"]
    )

    return {
        "composite": composite,
        "grade": grade,
        "breakdown": breakdown,
        "issues": all_issues,
        "vastu_bonuses": vastu_result.get("bonuses", []),
        "vastu_details": vastu_result.get("details", []),
        "climate_zone": climate.get("name", "Composite"),
        "family_type": family.get("type", "nuclear"),
    }


def generate_architect_narrative(engine_input: Dict, score_data: Dict,
                                  plot_width: float, plot_length: float) -> str:
    """Generate a professional architect narrative for the design."""
    total_area = engine_input.get("total_area", plot_width * plot_length)
    bedrooms = engine_input.get("bedrooms", 2)
    bathrooms = engine_input.get("bathrooms", 1)
    floors = engine_input.get("floors", 1)
    facing = engine_input.get("facing", "")
    extras = engine_input.get("extras", [])

    composite = score_data.get("composite", 0)
    grade = score_data.get("grade", "B")
    climate = score_data.get("climate_zone", "Composite")
    family = score_data.get("family_type", "nuclear").replace("_", " ")
    breakdown = score_data.get("breakdown", {})
    issues = score_data.get("issues", [])
    bonuses = score_data.get("vastu_bonuses", [])

    bhk = f"{bedrooms}BHK"
    parts = []

    # Opening
    facing_str = f" {facing}-facing" if facing else ""
    parts.append(
        f"Your {bhk} floor plan for a {plot_width:.0f}×{plot_length:.0f} ft "
        f"({total_area:.0f} sq ft){facing_str} plot is ready."
    )

    # Climate
    if climate and climate != "Composite":
        parts.append(
            f"Design is adapted for {climate} climate conditions."
        )

    # Family type
    if family != "nuclear":
        parts.append(
            f"Layout optimized for {family} living pattern."
        )

    # Vastu highlights
    vastu_score = breakdown.get("vastu", {}).get("score", 0)
    if vastu_score >= 85:
        parts.append(f"Vastu compliance is excellent ({vastu_score}/100).")
        if bonuses:
            bonus_strs = [b["message"] for b in bonuses[:3]]
            parts.append(" ".join(bonus_strs))
    elif vastu_score >= 70:
        parts.append(f"Vastu compliance is good ({vastu_score}/100).")
    elif vastu_score >= 50:
        parts.append(f"Vastu compliance is moderate ({vastu_score}/100) — some compromises for space efficiency.")

    # Critical issues
    critical = [i for i in issues if i.get("severity") == "critical"]
    if critical:
        parts.append(f"Note: {len(critical)} issue(s) flagged:")
        for ci in critical[:3]:
            parts.append(f"  • {ci['message']}")

    # Overall score
    parts.append(f"Overall design quality: {grade} ({composite}/100).")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════
# PUBLIC API — build_design_brief()
# ════════════════════════════════════════════════════════════════

def build_design_brief(input_data: Dict) -> Dict:
    """
    Build an enriched design brief from raw user input.

    Takes the raw engine_input dict and returns:
    {
        "engine_input": { ...enriched input for the layout engine... },
        "family_profile": { ...detected family info... },
        "climate": { ...detected climate zone... },
        "architect_notes": [ ...notes from profiling... ],
    }
    """
    description = input_data.get("description", "")
    city = input_data.get("city", "")
    state = input_data.get("state", "")
    bedrooms = input_data.get("bedrooms", 2)
    extras = input_data.get("extras", [])

    # Layer 1: Family profiling
    family = detect_family_profile(description, bedrooms, extras)

    # Layer 2: Climate detection
    climate = detect_climate_zone(city, state, description)

    # Build enriched engine input
    enriched = dict(input_data)

    # Add auto-rooms from family profile (don't duplicate)
    current_extras = list(extras)
    for auto_room in family.get("auto_rooms", []):
        if auto_room not in current_extras:
            current_extras.append(auto_room)
    enriched["extras"] = current_extras

    # Add climate hints
    enriched["climate_zone"] = climate.get("name", "Composite")
    enriched["climate_strategy"] = climate.get("strategy", {})

    # Add facing default from climate if not specified
    if not enriched.get("facing") and climate.get("strategy", {}).get("preferred_orientation"):
        pref = climate["strategy"]["preferred_orientation"]
        if pref not in ("any", "east_west_axis"):
            enriched["facing"] = pref

    # Collect architect notes
    notes = list(family.get("notes", []))
    notes.extend(climate.get("notes", []))

    return {
        "engine_input": enriched,
        "family_profile": family,
        "climate": climate,
        "architect_notes": notes,
    }


# Module-level singleton for convenience
class _DesignIntelligence:
    """Namespace wrapper so callers can use design_intelligence.method()."""

    build_design_brief = staticmethod(build_design_brief)
    score_layout = staticmethod(score_layout)
    generate_architect_narrative = staticmethod(generate_architect_narrative)
    detect_family_profile = staticmethod(detect_family_profile)
    detect_climate_zone = staticmethod(detect_climate_zone)
    score_vastu = staticmethod(score_vastu)
    score_nbc_compliance = staticmethod(score_nbc_compliance)
    score_acoustic = staticmethod(score_acoustic)
    score_natural_light = staticmethod(score_natural_light)
    score_circulation = staticmethod(score_circulation)
    score_area_efficiency = staticmethod(score_area_efficiency)


design_intelligence = _DesignIntelligence()
