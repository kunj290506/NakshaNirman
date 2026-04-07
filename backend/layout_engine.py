"""
Layout Engine for hybrid generation:
    1. OpenRouter attempt with strict time budget
    2. Claude coordinate-first attempt when API key is available
    3. Deterministic BSP fallback for guaranteed completion
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import httpx

def _rnd(val: float, d: int = 2) -> float:
    return int(val * (10**d)) / (10.0**d)

from models import (
    PlanRequest, PlanResponse, PlotInfo,
    RoomData, DoorData, WindowData, Point2D,
)
from llm import call_openrouter, call_openrouter_plan
from prompt_builder import build_master_prompt
from plan_validator import validate_draft, validate_final
from config import (
    ARCHITECT_REASONING_ENABLED,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    FAST_FALLBACK_MODE,
    FORCE_LOCAL_PLANNER,
    OPENROUTER_API_KEY,
    OPENROUTER_API_KEY_SECONDARY,
    PUBLIC_LLM_FALLBACK_ENABLED,
    PUBLIC_LLM_FALLBACK_MODEL,
    PUBLIC_LLM_FALLBACK_URL,
)

log = logging.getLogger("layout_engine")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SETBACKS = {"front": 6.5, "rear": 5.0, "left": 3.5, "right": 3.5}
OPENROUTER_ATTEMPT_TIMEOUT_SEC = 45
CLAUDE_API_TIMEOUT_SEC = 30
CLAUDE_ATTEMPT_DEADLINE_SEC = 35
ARCHITECT_REASONING_TIMEOUT_SEC = 14

LOGICAL_ADJ_MIN_SCORE = 72.0 if FAST_FALLBACK_MODE else 68.0

from typing import List, Dict, Any

# ─── Default room colors ────────────────────────────────────
ROOM_COLORS: Dict[str, str] = {
    "living": "#E8F5E9",
    "dining": "#FFF3E0",
    "kitchen": "#FFEBEE",
    "master_bedroom": "#E3F2FD",
    "bedroom": "#E3F2FD",
    "master_bath": "#E0F7FA",
    "bathroom": "#E0F7FA",
    "toilet": "#E0F7FA",
    "corridor": "#F5F5F5",
    "pooja": "#FFF8E1",
    "study": "#EDE7F6",
    "store": "#EFEBE9",
    "balcony": "#E8F5E9",
    "garage": "#ECEFF1",
    "utility": "#F3E5F5",
    "foyer": "#FAFAFA",
    "staircase": "#ECEFF1",
    "open_area": "#F8FAFC",
}

ROOM_TYPE_ALIASES: Dict[str, str] = {
    "bath": "bathroom",
    "washroom": "bathroom",
    "wc": "toilet",
    "toilet_room": "toilet",
    "master_bed": "master_bedroom",
    "masterbedroom": "master_bedroom",
    "bed_room": "bedroom",
    "living_room": "living",
    "hall": "living",
    "passage": "corridor",
    "lobby": "foyer",
    "mandir": "pooja",
    "puja": "pooja",
}

ZONE_NAME_BY_NUM = {1: "public", 2: "service", 3: "private"}

OPTIONAL_EXTRA_ROOM_TYPES = {
    "pooja", "study", "store", "balcony", "garage", "utility", "foyer", "staircase"
}

EXTRA_NAME_TO_ROOM_TYPE = {
    "pooja": "pooja",
    "puja": "pooja",
    "mandir": "pooja",
    "study": "study",
    "store": "store",
    "balcony": "balcony",
    "garage": "garage",
    "utility": "utility",
    "foyer": "foyer",
    "stair": "staircase",
    "stairs": "staircase",
    "staircase": "staircase",
}

ROOM_MIN_DIMS: Dict[str, tuple[float, float]] = {
    "living": (11.0, 11.0),
    "dining": (8.0, 8.0),
    "kitchen": (7.0, 8.0),
    "corridor": (3.0, 6.0),
    "master_bedroom": (10.0, 10.0),
    "bedroom": (9.0, 9.0),
    "master_bath": (4.5, 6.0),
    "bathroom": (4.0, 5.0),
    "toilet": (3.5, 4.5),
    "pooja": (4.0, 4.0),
    "study": (6.0, 7.0),
    "store": (4.0, 4.0),
    "balcony": (3.5, 6.0),
    "garage": (9.0, 15.0),
    "utility": (4.0, 5.0),
    "foyer": (4.0, 4.0),
    "staircase": (6.0, 8.0),
    "open_area": (6.0, 6.0),
}

ROOM_DEFAULT_ZONES: Dict[str, str] = {
    "living": "public",
    "dining": "public",
    "pooja": "public",
    "balcony": "public",
    "foyer": "public",
    "garage": "public",
    "open_area": "public",
    "kitchen": "service",
    "corridor": "service",
    "master_bath": "service",
    "bathroom": "service",
    "toilet": "service",
    "utility": "service",
    "store": "service",
    "staircase": "service",
    "master_bedroom": "private",
    "bedroom": "private",
    "study": "private",
}

ROOM_EXPECTED_ADJACENCY: Dict[str, set[str]] = {
    "living": {"dining", "foyer", "corridor"},
    "dining": {"living", "kitchen", "corridor"},
    "kitchen": {"dining", "utility", "corridor"},
    "master_bedroom": {"corridor", "master_bath"},
    "bedroom": {"corridor", "bathroom", "toilet"},
    "master_bath": {"master_bedroom"},
    "bathroom": {"corridor", "bedroom", "living"},
    "toilet": {"corridor", "living", "dining"},
    "study": {"corridor", "bedroom", "master_bedroom"},
    "pooja": {"living", "dining", "foyer"},
    "utility": {"kitchen", "corridor"},
    "store": {"kitchen", "corridor", "dining"},
    "foyer": {"living"},
    "garage": {"foyer", "living"},
}

DEFAULT_ROOM_LABELS: Dict[str, str] = {
    "living": "Living",
    "dining": "Dining",
    "kitchen": "Kitchen",
    "corridor": "Corridor",
    "master_bedroom": "Master Bedroom",
    "bedroom": "Bedroom",
    "master_bath": "Master Bath",
    "bathroom": "Bathroom",
    "toilet": "Toilet",
    "pooja": "Pooja",
    "study": "Study",
    "store": "Store",
    "balcony": "Balcony",
    "garage": "Garage",
    "utility": "Utility",
    "foyer": "Foyer",
    "staircase": "Staircase",
    "open_area": "Open Area",
}

def _requested_extra_room_types(extras: list[str]) -> set[str]:
    requested: set[str] = set()
    for extra in extras or []:
        key = str(extra or "").strip().lower().replace(" ", "_")
        key = ROOM_TYPE_ALIASES.get(key, key)
        key = EXTRA_NAME_TO_ROOM_TYPE.get(key, key)
        if key in OPTIONAL_EXTRA_ROOM_TYPES:
            requested.add(key)
    return requested


def _default_room_label(room_type: str, serial: int = 1) -> str:
    base = DEFAULT_ROOM_LABELS.get(room_type, room_type.replace("_", " ").title())
    if room_type == "bedroom" and serial > 1:
        return f"Bedroom {serial}"
    return base


def _push_reasoning(trace: list[str], message: str):
    msg = str(message or "").strip()
    if not msg:
        return
    if trace and trace[-1] == msg:
        return
    trace.append(msg)


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        f = float(value)
        if not math.isfinite(f):
            return fallback
        return f
    except Exception:
        return fallback


def _normalize_reasoning_lines(lines: list[str], limit: int = 16) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        msg = str(raw or "").strip()
        if not msg:
            continue
        if len(msg) > 170:
            msg = msg[:167].rstrip() + "..."
        key = msg.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(msg)
        if len(out) >= limit:
            break
    return out


def _merge_reasoning_trace(prefix: list[str], suffix: list[str], limit: int = 16) -> list[str]:
    return _normalize_reasoning_lines([*(prefix or []), *(suffix or [])], limit=limit)


def _coerce_reasoning_items(value: Any, max_items: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _build_local_architect_reasoning(req: PlanRequest, uw: float, ul: float) -> list[str]:
    area = max(1.0, uw * ul)
    aspect = ul / max(1.0, uw)
    feasible_bedrooms = _effective_bedroom_target(req.bedrooms, uw, ul)
    requested = int(getattr(req, "bedrooms", 1) or 1)

    if area < 650:
        footprint_class = "compact"
    elif area < 1100:
        footprint_class = "standard"
    else:
        footprint_class = "spacious"

    trace: list[str] = [
        f"Architect reasoning stage started for {uw:.1f}x{ul:.1f} ft usable plot ({footprint_class}).",
        f"Program fit check: requested {requested}BHK, feasible target {feasible_bedrooms}BHK.",
        f"Facing strategy: {str(req.facing).lower()}-side main entry with frontage-oriented public rooms.",
        "Zoning strategy: public front, service middle, private rear with one circulation spine.",
        "Geometry checks planned: overlap-free rectangles, practical minimum sizes, and adjacency continuity.",
        "Manual per-element adjustment is disabled; backend auto-optimizes each element before plotting.",
    ]

    if aspect >= 1.8:
        trace.append("Long-plot adaptation enabled: cap room depth to avoid bowling-alley proportions.")

    feasible_extras, deferred_extras = _partition_feasible_extras(
        req.extras,
        uw,
        ul,
        feasible_bedrooms,
    )
    if feasible_extras:
        trace.append("Feasible extras in program: " + ", ".join(sorted(feasible_extras)) + ".")
    if deferred_extras:
        trace.append("Deferred extras for this footprint: " + ", ".join(sorted(deferred_extras)) + ".")

    return _normalize_reasoning_lines(trace, limit=10)


def _default_element_reasoning_policy() -> dict[str, str]:
    return {
        "living": "Front/public placement near entry for guest circulation.",
        "dining": "Kept adjacent to living and near kitchen for flow.",
        "kitchen": "Service-zone placement with practical utility access.",
        "corridor": "Single circulation spine connecting all major rooms.",
        "master_bedroom": "Rear/private placement for privacy and noise isolation.",
        "bedroom": "Private-zone grouping for night-use separation.",
        "master_bath": "Wet-core clustering near master bedroom for plumbing efficiency.",
        "bathroom": "Shared wet-core placement to reduce service runs.",
    }


def _build_priority_weights(req: PlanRequest) -> dict[str, float]:
    raw = {
        "privacy": max(1.0, float(getattr(req, "privacy_priority", 3) or 3)),
        "natural_light": max(1.0, float(getattr(req, "natural_light_priority", 3) or 3)),
        "storage": max(1.0, float(getattr(req, "storage_priority", 3) or 3)),
        "vastu": max(1.0, float(getattr(req, "vastu_priority", 3) or 3)),
        "elder_friendly": 5.0 if bool(getattr(req, "elder_friendly", False)) else 2.0,
        "work_from_home": 5.0 if bool(getattr(req, "work_from_home", False)) else 2.0,
    }
    total = max(1.0, sum(raw.values()))
    return {k: _rnd((v / total) * 100.0, 1) for k, v in raw.items()}


def _build_program_area_budget(
    usable_area: float,
    feasible_bedrooms: int,
    feasible_extras: set[str],
) -> dict[str, Any]:
    public = 32.0
    service = 24.0
    private = 44.0

    if feasible_bedrooms >= 3:
        private += 4.0
        public -= 2.0
        service -= 2.0

    if "study" in feasible_extras:
        private += 3.0
        public -= 1.5
        service -= 1.5
    if "garage" in feasible_extras:
        public += 4.0
        private -= 2.0
        service -= 2.0
    if "utility" in feasible_extras or "store" in feasible_extras:
        service += 3.0
        public -= 1.5
        private -= 1.5

    public = max(20.0, public)
    service = max(16.0, service)
    private = max(26.0, private)
    total = max(1.0, public + service + private)
    public = (public / total) * 100.0
    service = (service / total) * 100.0
    private = (private / total) * 100.0

    return {
        "zone_budget_pct": {
            "public": _rnd(public, 1),
            "service": _rnd(service, 1),
            "private": _rnd(private, 1),
        },
        "zone_budget_ft2": {
            "public": _rnd((public * usable_area) / 100.0, 1),
            "service": _rnd((service * usable_area) / 100.0, 1),
            "private": _rnd((private * usable_area) / 100.0, 1),
        },
        "circulation_target_pct": _rnd(9.5 + max(0, feasible_bedrooms - 2) * 1.2, 1),
    }


def _build_constraint_matrix(
    req: PlanRequest,
    uw: float,
    ul: float,
    feasible_bedrooms: int,
    feasible_extras: set[str],
    deferred_extras: set[str],
) -> list[dict[str, Any]]:
    bathrooms_target = int(getattr(req, "bathrooms_target", 0) or 0)
    effective_bath_target = bathrooms_target if bathrooms_target > 0 else max(1, feasible_bedrooms)

    constraints = [
        {
            "id": "plot_setbacks",
            "target": f"usable plot {uw:.1f}x{ul:.1f} ft",
            "status": "applied",
            "detail": "Front/rear/side setbacks subtracted before room synthesis.",
        },
        {
            "id": "bedroom_fit",
            "target": f"{int(req.bedrooms)} requested",
            "status": "satisfied" if feasible_bedrooms == int(req.bedrooms) else "downscaled",
            "detail": f"Feasible bedroom program resolved to {feasible_bedrooms}.",
        },
        {
            "id": "bathroom_target",
            "target": f"{effective_bath_target} total",
            "status": "planned",
            "detail": "Wet-core clustering policy used for service efficiency.",
        },
        {
            "id": "orientation",
            "target": f"{str(req.facing).lower()}-facing entry logic",
            "status": "planned",
            "detail": "Public rooms biased toward frontage side.",
        },
        {
            "id": "extras_fit",
            "target": ", ".join(sorted(_requested_extra_room_types(req.extras))) or "none",
            "status": "satisfied" if not deferred_extras else "partially_satisfied",
            "detail": (
                "Feasible extras: " + (", ".join(sorted(feasible_extras)) if feasible_extras else "none")
                + "; deferred: " + (", ".join(sorted(deferred_extras)) if deferred_extras else "none")
            ),
        },
        {
            "id": "automation",
            "target": "full auto placement",
            "status": "locked",
            "detail": "Manual per-element adjustment disabled; backend auto-correct enabled.",
        },
    ]
    return constraints


def _normalize_architect_advisory(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "design_strategy": "",
            "priority_order": [],
            "critical_checks": [],
            "risks": [],
        }

    strategy = str(
        data.get("design_strategy")
        or data.get("layout_strategy")
        or data.get("summary")
        or ""
    ).strip()
    priorities = _coerce_reasoning_items(data.get("priority_order"), max_items=3)
    checks = _coerce_reasoning_items(data.get("critical_checks"), max_items=3)
    risks = _coerce_reasoning_items(data.get("risks"), max_items=2)

    return {
        "design_strategy": strategy,
        "priority_order": priorities,
        "critical_checks": checks,
        "risks": risks,
    }


def _extract_llm_architect_reasoning(data: dict[str, Any]) -> list[str]:
    advisory = _normalize_architect_advisory(data)

    trace: list[str] = []
    strategy = str(advisory.get("design_strategy", "")).strip()
    if strategy:
        trace.append(f"LLM architect strategy: {strategy}")

    for item in _coerce_reasoning_items(advisory.get("priority_order"), max_items=2):
        trace.append(f"LLM priority: {item}")

    for item in _coerce_reasoning_items(advisory.get("critical_checks"), max_items=2):
        trace.append(f"LLM check: {item}")

    for item in _coerce_reasoning_items(advisory.get("risks"), max_items=1):
        trace.append(f"LLM risk note: {item}")

    return _normalize_reasoning_lines(trace, limit=5)


def _build_architect_reasoning_object(
    req: PlanRequest,
    uw: float,
    ul: float,
    *,
    source: str,
    status: str,
    advisory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    area = max(1.0, uw * ul)
    feasible_bedrooms = _effective_bedroom_target(req.bedrooms, uw, ul)
    requested_extras = sorted(_requested_extra_room_types(req.extras))
    feasible_extras, deferred_extras = _partition_feasible_extras(req.extras, uw, ul, feasible_bedrooms)
    priority_weights = _build_priority_weights(req)
    program_area_budget = _build_program_area_budget(area, feasible_bedrooms, feasible_extras)
    constraint_matrix = _build_constraint_matrix(
        req,
        uw,
        ul,
        feasible_bedrooms,
        feasible_extras,
        deferred_extras,
    )
    element_policy = _default_element_reasoning_policy()
    for extra in sorted(feasible_extras):
        if extra == "study":
            element_policy[extra] = "Private-zone placement for focused work with acoustic separation."
        elif extra == "pooja":
            element_policy[extra] = "Placed in calm front/public quadrant aligned with cultural practice."
        elif extra == "utility":
            element_policy[extra] = "Service-zone placement supporting kitchen and wet-core workflows."
        elif extra == "balcony":
            element_policy[extra] = "Boundary placement for daylight and ventilation extension."
        elif extra == "store":
            element_policy[extra] = "Service-zone storage near daily-use circulation."
        elif extra == "garage":
            element_policy[extra] = "Front-edge vehicle access aligned to road-facing side."
        elif extra == "foyer":
            element_policy[extra] = "Entry transition space for privacy buffering."
        elif extra == "staircase":
            element_policy[extra] = "Service-core stair placement for vertical circulation continuity."

    return {
        "stage": "preplot_architect_reasoning",
        "status": status,
        "source": source,
        "manual_adjustment_required": False,
        "automation_mode": "full_auto_element_placement",
        "usable_plot_ft": {
            "width": _rnd(uw, 2),
            "length": _rnd(ul, 2),
            "area": _rnd(area, 1),
        },
        "request": {
            "bedrooms": int(req.bedrooms),
            "bathrooms_target": int(getattr(req, "bathrooms_target", 0) or 0),
            "facing": str(req.facing).lower(),
            "extras": requested_extras,
            "family_type": str(getattr(req, "family_type", "") or "").lower() or "nuclear",
            "design_style": str(getattr(req, "design_style", "") or "modern").lower(),
            "kitchen_preference": str(getattr(req, "kitchen_preference", "") or "semi_open").lower(),
            "city": str(getattr(req, "city", "") or "").strip(),
            "work_from_home": bool(getattr(req, "work_from_home", False)),
            "elder_friendly": bool(getattr(req, "elder_friendly", False)),
        },
        "program_fit": {
            "bedrooms_requested": int(req.bedrooms),
            "bedrooms_feasible": int(feasible_bedrooms),
            "extras_feasible": sorted(feasible_extras),
            "extras_deferred": sorted(deferred_extras),
        },
        "priority_weights": priority_weights,
        "program_area_budget": program_area_budget,
        "constraint_matrix": constraint_matrix,
        "auto_adjustments_applied": [
            "grid_snap_and_rounding",
            "zone_alignment",
            "overlap_resolution",
            "minimum_dimension_enforcement",
            "opening_regeneration",
        ],
        "zoning_plan": ["public_front", "service_middle", "private_rear", "single_corridor_spine"],
        "element_reasoning_policy": element_policy,
        "advisory": _normalize_architect_advisory(advisory or {}),
    }


def _room_exterior_walls(room: RoomData, usable_w: float, usable_l: float, eps: float = 0.35) -> list[str]:
    walls: list[str] = []
    if room.x <= eps:
        walls.append("west")
    if room.x + room.width >= usable_w - eps:
        walls.append("east")
    if room.y <= eps:
        walls.append("south")
    if room.y + room.height >= usable_l - eps:
        walls.append("north")
    return walls


def _reason_for_room(room: RoomData, facing: str, exterior_walls: list[str]) -> str:
    room_type = str(room.type or "").strip().lower()
    facing_norm = str(facing or "south").strip().lower()

    base_map: dict[str, str] = {
        "living": f"Placed in public zone near {facing_norm}-facing entry for guest circulation.",
        "dining": "Placed between living and kitchen to reduce movement friction.",
        "kitchen": "Placed in service zone for utility and wet-core efficiency.",
        "corridor": "Acts as the central circulation spine connecting public and private zones.",
        "master_bedroom": "Placed in private rear zone for privacy and acoustic comfort.",
        "bedroom": "Grouped in private zone for night-use privacy.",
        "master_bath": "Clustered with wet core and adjacent to master bedroom.",
        "bathroom": "Clustered in service band to optimize plumbing lines.",
        "study": "Placed in quiet private zone for focused work.",
        "pooja": "Placed in calm front-side zone per cultural preference.",
        "utility": "Placed close to kitchen/service band for daily chores.",
        "balcony": "Placed on boundary for daylight and ventilation extension.",
        "store": "Placed in service zone near operational spaces.",
        "garage": "Placed near frontage for convenient vehicle entry.",
        "foyer": "Placed at transition from entrance to interior privacy.",
        "staircase": "Placed in service core to maintain vertical circulation.",
        "open_area": "Kept open on boundary for light, air, and future flexibility.",
    }

    reason = base_map.get(room_type, "Placed automatically for non-overlapping program fit.")
    if exterior_walls:
        reason += " Exterior access/light on " + ", ".join(exterior_walls) + " side(s)."
    return reason


def _room_adjacency_refs(room: RoomData, rooms: list[RoomData], min_shared: float = 2.0) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for other in rooms:
        if other.id == room.id:
            continue
        shared = _shared_wall_length(room, other)
        if shared < min_shared:
            continue
        refs.append(
            {
                "id": str(other.id),
                "type": str(other.type),
                "shared_wall_ft": _rnd(float(shared), 2),
            }
        )
    refs.sort(key=lambda item: (-float(item.get("shared_wall_ft", 0.0)), str(item.get("id", ""))))
    return refs[:8]


def _room_adjacency_names(room: RoomData, rooms: list[RoomData], min_shared: float = 2.0) -> list[str]:
    refs = _room_adjacency_refs(room, rooms, min_shared=min_shared)
    return sorted(str(ref.get("id", "")) for ref in refs if str(ref.get("id", "")))[:6]


def _room_checks(
    room: RoomData,
    exterior_walls: list[str],
    adjacent_to: list[str],
    adjacent_types: list[str],
) -> dict[str, bool]:
    room_type = str(room.type or "").strip().lower()
    expected_zone = ROOM_DEFAULT_ZONES.get(room_type, "public")
    zone_match = str(room.zone or "").strip().lower() == expected_zone

    min_dims = ROOM_MIN_DIMS.get(room_type, (4.0, 4.0))
    dimension_ok = float(room.width) >= (min_dims[0] * 0.8) and float(room.height) >= (min_dims[1] * 0.8)

    daylight_access = bool(exterior_walls)
    corridor_linked = any("corridor" in n for n in adjacent_to)
    if room_type in ("corridor", "living"):
        corridor_linked = True

    expected_adj = ROOM_EXPECTED_ADJACENCY.get(room_type, set())
    adjacency_alignment = True
    if expected_adj:
        adjacency_alignment = any(t in expected_adj for t in adjacent_types)

    privacy_buffer = True
    if room_type in ("master_bedroom", "bedroom", "study"):
        privacy_buffer = not any(t in ("living", "kitchen") for t in adjacent_types)
    elif room_type == "master_bath":
        privacy_buffer = "master_bedroom" in adjacent_types

    return {
        "zone_match": zone_match,
        "dimension_ok": dimension_ok,
        "daylight_access": daylight_access,
        "circulation_link": corridor_linked,
        "adjacency_alignment": adjacency_alignment,
        "privacy_buffer": privacy_buffer,
    }


def _rect_gap_distance(a: RoomData, b: RoomData) -> float:
    ax0, ay0 = float(a.x), float(a.y)
    ax1, ay1 = float(a.x) + float(a.width), float(a.y) + float(a.height)
    bx0, by0 = float(b.x), float(b.y)
    bx1, by1 = float(b.x) + float(b.width), float(b.y) + float(b.height)

    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return math.sqrt(dx * dx + dy * dy)


def _reasoning_confidence(checks: dict[str, bool]) -> float:
    if not checks:
        return 0.0
    passed = sum(1 for v in checks.values() if bool(v))
    return _rnd((passed / max(1, len(checks))) * 100.0, 1)


def _room_metrics(
    room: RoomData,
    usable_w: float,
    usable_l: float,
    corridor_rooms: list[RoomData],
) -> dict[str, float]:
    width = float(room.width)
    height = float(room.height)
    area = max(0.1, width * height)
    usable_area = max(1.0, usable_w * usable_l)
    short_edge = max(0.1, min(width, height))
    long_edge = max(width, height)
    min_edge_buffer = min(
        float(room.x),
        float(room.y),
        max(0.0, usable_w - (float(room.x) + width)),
        max(0.0, usable_l - (float(room.y) + height)),
    )

    corridor_gap = 0.0
    room_type = str(getattr(room, "type", "") or "").strip().lower()
    if corridor_rooms and room_type != "corridor":
        corridor_gap = min(_rect_gap_distance(room, c) for c in corridor_rooms)

    return {
        "area_ft2": _rnd(area, 2),
        "area_share_pct": _rnd((area / usable_area) * 100.0, 2),
        "aspect_ratio": _rnd(long_edge / short_edge, 2),
        "min_edge_buffer_ft": _rnd(min_edge_buffer, 2),
        "corridor_gap_ft": _rnd(corridor_gap, 2),
    }


def _failed_check_keys(checks: dict[str, bool]) -> list[str]:
    return [str(k) for k, v in (checks or {}).items() if not bool(v)]


def _compute_reasoning_quality_scores(
    plan: PlanResponse,
    element_items: list[dict[str, Any]],
) -> dict[str, float]:
    if not element_items:
        return {
            "privacy": 0.0,
            "circulation": 0.0,
            "daylight": 0.0,
            "adjacency_quality": 0.0,
            "program_integrity": 0.0,
            "check_pass_rate": 0.0,
            "space_efficiency": 0.0,
            "confidence_mean": 0.0,
            "overall": 0.0,
        }

    bedroom_items = [e for e in element_items if str(e.get("type", "")) in ("master_bedroom", "bedroom")]
    wet_items = [e for e in element_items if str(e.get("type", "")) in ("master_bath", "bathroom", "toilet")]

    privacy = 100.0
    if bedroom_items:
        in_private = sum(1 for e in bedroom_items if str(e.get("zone", "")).lower() == "private")
        privacy = (in_private / len(bedroom_items)) * 100.0

    circulation = (
        sum(1 for e in element_items if bool((e.get("checks", {}) or {}).get("circulation_link")))
        / len(element_items)
    ) * 100.0

    daylight = (
        sum(1 for e in element_items if bool((e.get("checks", {}) or {}).get("daylight_access")))
        / len(element_items)
    ) * 100.0

    wet_service = 100.0
    if wet_items:
        wet_service = (
            sum(1 for e in wet_items if str(e.get("zone", "")).lower() == "service")
            / len(wet_items)
        ) * 100.0

    integrity = (
        sum(1 for e in element_items if bool((e.get("checks", {}) or {}).get("zone_match")) and bool((e.get("checks", {}) or {}).get("dimension_ok")))
        / len(element_items)
    ) * 100.0

    adjacency_quality = (
        sum(1 for e in element_items if bool((e.get("checks", {}) or {}).get("adjacency_alignment")))
        / len(element_items)
    ) * 100.0

    check_values: list[bool] = []
    for e in element_items:
        checks = e.get("checks", {}) or {}
        check_values.extend(bool(v) for v in checks.values())
    check_pass_rate = (sum(1 for v in check_values if v) / max(1, len(check_values))) * 100.0

    confidence_mean = (
        sum(float(e.get("confidence", 0.0) or 0.0) for e in element_items)
        / len(element_items)
    )

    usable_area = max(1.0, float(getattr(plan.plot, "usable_width", 0.0) or 0.0) * float(getattr(plan.plot, "usable_length", 0.0) or 0.0))
    placed_area = sum(max(0.0, float(getattr(r, "width", 0.0) or 0.0) * float(getattr(r, "height", 0.0) or 0.0)) for r in (plan.rooms or []))
    space_efficiency = max(0.0, min(100.0, (placed_area / usable_area) * 100.0))

    overall = (
        (privacy * 0.20)
        + (circulation * 0.18)
        + (daylight * 0.15)
        + (wet_service * 0.12)
        + (integrity * 0.17)
        + (adjacency_quality * 0.10)
        + (check_pass_rate * 0.08)
    )
    return {
        "privacy": _rnd(privacy, 1),
        "circulation": _rnd(circulation, 1),
        "daylight": _rnd(daylight, 1),
        "wet_core_efficiency": _rnd(wet_service, 1),
        "adjacency_quality": _rnd(adjacency_quality, 1),
        "program_integrity": _rnd(integrity, 1),
        "check_pass_rate": _rnd(check_pass_rate, 1),
        "space_efficiency": _rnd(space_efficiency, 1),
        "confidence_mean": _rnd(confidence_mean, 1),
        "overall": _rnd(overall, 1),
    }


def _build_reasoning_diagnostics(element_items: list[dict[str, Any]]) -> dict[str, Any]:
    total_checks = 0
    passed_checks = 0
    failed_elements: list[dict[str, Any]] = []

    for item in element_items:
        checks = item.get("checks", {}) or {}
        failures = _failed_check_keys(checks)
        total_checks += len(checks)
        passed_checks += sum(1 for v in checks.values() if bool(v))
        if failures:
            failed_elements.append(
                {
                    "id": str(item.get("id", "")),
                    "type": str(item.get("type", "")),
                    "failed_checks": failures,
                    "confidence": _rnd(float(item.get("confidence", 0.0) or 0.0), 1),
                }
            )

    pass_rate = (passed_checks / max(1, total_checks)) * 100.0
    high_risk_count = sum(1 for item in element_items if float(item.get("confidence", 0.0) or 0.0) < 70.0)

    return {
        "checks": {
            "total": int(total_checks),
            "passed": int(passed_checks),
            "failed": int(max(0, total_checks - passed_checks)),
            "pass_rate": _rnd(pass_rate, 1),
        },
        "high_risk_element_count": int(high_risk_count),
        "failed_elements": failed_elements[:12],
    }


def _build_tradeoff_notes(
    base_reasoning: dict[str, Any],
    quality_scores: dict[str, float],
    diagnostics: dict[str, Any],
) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    program_fit = (base_reasoning or {}).get("program_fit", {}) if isinstance(base_reasoning, dict) else {}
    deferred_extras = list(program_fit.get("extras_deferred", []) or []) if isinstance(program_fit, dict) else []

    if deferred_extras:
        notes.append(
            {
                "theme": "program_scope",
                "decision": "Deferred low-priority extras to preserve core-room quality and clear circulation.",
                "impact": "Deferred extras: " + ", ".join(deferred_extras),
            }
        )

    circulation = float(quality_scores.get("circulation", 0.0) or 0.0)
    daylight = float(quality_scores.get("daylight", 0.0) or 0.0)
    privacy = float(quality_scores.get("privacy", 0.0) or 0.0)

    if circulation < 72.0:
        notes.append(
            {
                "theme": "circulation_vs_compactness",
                "decision": "Maintained compact footprint while accepting tighter movement links in a few spaces.",
                "impact": f"Circulation score {circulation:.1f} indicates some near-corridor links instead of direct shared walls.",
            }
        )
    if daylight < 70.0:
        notes.append(
            {
                "theme": "daylight_vs_privacy",
                "decision": "Protected private-zone arrangement even where full exterior frontage was not possible.",
                "impact": f"Daylight score {daylight:.1f} reflects interior rooms buffered by service spaces.",
            }
        )
    if privacy < 75.0:
        notes.append(
            {
                "theme": "privacy_vs_adjacency",
                "decision": "Prioritized functional adjacency for key rooms despite reduced bedroom isolation in compact zones.",
                "impact": f"Privacy score {privacy:.1f} indicates at least one bedroom has active-zone adjacency.",
            }
        )

    high_risk = int((diagnostics.get("high_risk_element_count", 0) or 0))
    if high_risk > 0:
        notes.append(
            {
                "theme": "micro_adjustment_residual",
                "decision": "Applied automatic micro-adjustments; residual low-confidence elements are flagged for future optimization cycles.",
                "impact": f"Low-confidence elements: {high_risk}.",
            }
        )

    if not notes:
        notes.append(
            {
                "theme": "balanced_outcome",
                "decision": "Program, circulation, and privacy stayed balanced without requiring manual edits.",
                "impact": "No major trade-off triggered in post-plot diagnostics.",
            }
        )

    return notes[:6]


def _build_reasoning_passes(
    base_reasoning: dict[str, Any],
    element_items: list[dict[str, Any]],
    quality_scores: dict[str, float],
    diagnostics: dict[str, Any],
    tradeoff_notes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    program_fit = (base_reasoning or {}).get("program_fit", {}) if isinstance(base_reasoning, dict) else {}
    deferred_extras = list(program_fit.get("extras_deferred", []) or []) if isinstance(program_fit, dict) else []
    constraints_count = len((base_reasoning or {}).get("constraint_matrix", []) or []) if isinstance(base_reasoning, dict) else 0

    pass_1 = {
        "name": "input_decomposition",
        "summary": "Parsed priorities, hard constraints, and program feasibility before plotting.",
        "highlights": [
            f"Constraint rules loaded: {constraints_count}.",
            f"Deferred extras: {', '.join(deferred_extras) if deferred_extras else 'none'}.",
            "Target zones fixed before plotting to reduce late-stage geometry drift.",
        ],
    }

    pass_2 = {
        "name": "element_placement_logic",
        "summary": "Placed each element with zone-first logic, adjacency checks, and geometric guardrails.",
        "highlights": [
            f"Element count evaluated: {len(element_items)}.",
            "Each element scored for zone, dimensions, daylight, circulation, adjacency, and privacy buffer.",
        ],
    }

    pass_3 = {
        "name": "self_review",
        "summary": "Post-plot quality review executed with weighted scores and trade-off diagnostics.",
        "highlights": [
            f"Overall quality score: {quality_scores.get('overall', 0.0)}.",
            f"Checks pass rate: {(diagnostics.get('checks', {}) or {}).get('pass_rate', 0.0)} with {len(tradeoff_notes)} trade-off note(s).",
            f"Privacy {quality_scores.get('privacy', 0.0)}, circulation {quality_scores.get('circulation', 0.0)}, daylight {quality_scores.get('daylight', 0.0)}.",
        ],
    }

    return [pass_1, pass_2, pass_3]


def _build_element_reasoning(
    plan: PlanResponse,
    req: PlanRequest,
    usable_w: float,
    usable_l: float,
) -> list[dict[str, Any]]:
    zone_order = {"public": 1, "service": 2, "private": 3}
    ordered_rooms = sorted(
        list(plan.rooms or []),
        key=lambda r: (zone_order.get(str(getattr(r, "zone", "")).lower(), 9), float(r.y), float(r.x)),
    )
    corridor_rooms = [r for r in ordered_rooms if str(getattr(r, "type", "")).lower() == "corridor"]

    items: list[dict[str, Any]] = []
    for room in ordered_rooms:
        exterior_walls = _room_exterior_walls(room, usable_w, usable_l)
        adjacency_refs = _room_adjacency_refs(room, ordered_rooms)
        adjacent_to = [str(ref.get("id", "")) for ref in adjacency_refs if str(ref.get("id", ""))]
        adjacent_types = [str(ref.get("type", "") or "").strip().lower() for ref in adjacency_refs]
        checks = _room_checks(room, exterior_walls, adjacent_to, adjacent_types)

        corridor_gap = 0.0
        if corridor_rooms and str(getattr(room, "type", "") or "").strip().lower() != "corridor":
            corridor_gap = min(_rect_gap_distance(room, c) for c in corridor_rooms)

        if not checks.get("circulation_link", False) and corridor_rooms:
            if corridor_gap <= 1.25:
                checks["circulation_link"] = True

        metrics = _room_metrics(room, usable_w, usable_l, corridor_rooms)
        failed_checks = _failed_check_keys(checks)

        items.append(
            {
                "id": room.id,
                "type": room.type,
                "zone": room.zone,
                "position_ft": {
                    "x": _rnd(float(room.x), 2),
                    "y": _rnd(float(room.y), 2),
                    "width": _rnd(float(room.width), 2),
                    "height": _rnd(float(room.height), 2),
                },
                "exterior_walls": exterior_walls,
                "adjacent_to": adjacent_to,
                "adjacent_types": sorted(set(adjacent_types)),
                "adjacency_refs": adjacency_refs,
                "checks": checks,
                "failed_checks": failed_checks,
                "metrics": metrics,
                "confidence": _reasoning_confidence(checks),
                "reason": _reason_for_room(room, req.facing, exterior_walls),
                "auto_adjusted": True,
            }
        )

    return items[:28]


def _attach_plan_reasoning(
    plan: PlanResponse,
    base_reasoning: dict[str, Any],
    req: PlanRequest,
    usable_w: float,
    usable_l: float,
    generation_path: str,
) -> PlanResponse:
    reasoning = dict(base_reasoning or {})
    reasoning["generation_path"] = generation_path
    element_items = _build_element_reasoning(plan, req, usable_w, usable_l)
    quality_scores = _compute_reasoning_quality_scores(plan, element_items)
    diagnostics = _build_reasoning_diagnostics(element_items)
    tradeoff_notes = _build_tradeoff_notes(base_reasoning, quality_scores, diagnostics)
    reasoning["element_reasoning"] = element_items
    reasoning["quality_scores"] = quality_scores
    reasoning["diagnostics"] = diagnostics
    reasoning["tradeoff_notes"] = tradeoff_notes
    reasoning["explainability_mode"] = "detailed_backend_trace"
    reasoning["reasoning_depth"] = "high"
    reasoning["reasoning_passes"] = _build_reasoning_passes(
        base_reasoning,
        element_items,
        quality_scores,
        diagnostics,
        tradeoff_notes,
    )
    reasoning["execution_summary"] = {
        "rooms": len(plan.rooms or []),
        "doors": len(plan.doors or []),
        "windows": len(plan.windows or []),
        "generation_method": str(plan.generation_method or generation_path),
        "checks_pass_rate": (diagnostics.get("checks", {}) or {}).get("pass_rate", 0.0),
        "high_risk_elements": diagnostics.get("high_risk_element_count", 0),
    }
    reasoning["manual_adjustment_required"] = False
    plan.architect_reasoning = reasoning
    plan.reasoning_trace = _merge_reasoning_trace(
        list(plan.reasoning_trace or []),
        [
            "All elements were auto-adjusted and optimized by backend reasoning.",
            f"Post-plot quality score: {quality_scores.get('overall', 0.0)}.",
            f"Check pass rate: {(diagnostics.get('checks', {}) or {}).get('pass_rate', 0.0)}.",
        ],
        limit=24,
    )
    return plan


def _extract_openai_like_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message", {}) if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        text = str(part.get("text", "")).strip()
                        if text:
                            parts.append(text)
                merged = "\n".join(parts).strip()
                if merged:
                    return merged
    output_text = data.get("output_text") if isinstance(data, dict) else None
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    return ""


async def _call_public_architect_advisory(system_prompt: str, user_message: str) -> dict[str, Any] | None:
    if not PUBLIC_LLM_FALLBACK_ENABLED:
        return None

    payload = {
        "model": PUBLIC_LLM_FALLBACK_MODEL,
        "temperature": 0.1,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=12, verify=True) as client:
            resp = await client.post(PUBLIC_LLM_FALLBACK_URL, json=payload)
        if resp.status_code >= 400:
            return None

        content = _extract_openai_like_text(resp.json())
        if not content:
            return None

        parsed = _extract_json_object(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        log.info("Public architect advisory skipped: %s", str(e)[:160])

    return None


async def _build_preplot_reasoning_trace(req: PlanRequest, uw: float, ul: float) -> tuple[list[str], dict[str, Any]]:
    trace = _build_local_architect_reasoning(req, uw, ul)

    system_prompt = (
        "You are a senior residential architect. "
        "Reason about a floor-plan request before plotting. "
        "Return JSON only."
    )
    user_message = (
        "Analyze this request and return only JSON with keys: "
        "design_strategy (string), priority_order (array), critical_checks (array), risks (array).\n"
        f"plot_usable_ft: {uw} x {ul}\n"
        f"road_facing: {str(req.facing).lower()}\n"
        f"requested_bedrooms: {int(req.bedrooms)}\n"
        f"bathrooms_target: {int(getattr(req, 'bathrooms_target', 0) or 0)}\n"
        f"extras: {', '.join(sorted(_requested_extra_room_types(req.extras))) or 'none'}\n"
        f"family_type: {str(getattr(req, 'family_type', '') or '').lower() or 'nuclear'}\n"
        "Keep each item concise and practical."
    )

    if not ARCHITECT_REASONING_ENABLED:
        trace.append("LLM architect advisory disabled by config; continuing with deterministic reasoning.")
        return (
            _normalize_reasoning_lines(trace, limit=14),
            _build_architect_reasoning_object(req, uw, ul, source="local", status="disabled"),
        )

    if not (OPENROUTER_API_KEY.strip() or OPENROUTER_API_KEY_SECONDARY.strip()):
        public_data = await _call_public_architect_advisory(system_prompt, user_message)
        public_trace = _extract_llm_architect_reasoning(public_data or {})
        if public_trace:
            trace.append("Public LLM architect advisory completed before plotting.")
            return (
                _merge_reasoning_trace(trace, public_trace, limit=14),
                _build_architect_reasoning_object(
                    req,
                    uw,
                    ul,
                    source="public_fallback",
                    status="advisory_applied",
                    advisory=public_data,
                ),
            )

        trace.append("LLM architect advisory unavailable (no OpenRouter key); using local architect reasoning.")
        return (
            _normalize_reasoning_lines(trace, limit=14),
            _build_architect_reasoning_object(req, uw, ul, source="local", status="no_llm_key"),
        )

    try:
        llm_data = await asyncio.wait_for(
            call_openrouter(system_prompt, user_message, temperature=0.1, max_tokens=800),
            timeout=ARCHITECT_REASONING_TIMEOUT_SEC,
        )
        llm_trace = _extract_llm_architect_reasoning(llm_data)
        if llm_trace:
            trace.append("LLM architect advisory completed before plotting.")
            trace = _merge_reasoning_trace(trace, llm_trace, limit=14)
            return (
                _normalize_reasoning_lines(trace, limit=14),
                _build_architect_reasoning_object(
                    req,
                    uw,
                    ul,
                    source="openrouter",
                    status="advisory_applied",
                    advisory=llm_data,
                ),
            )
        else:
            trace.append("LLM architect advisory returned no structured guidance; local reasoning used.")
            return (
                _normalize_reasoning_lines(trace, limit=14),
                _build_architect_reasoning_object(req, uw, ul, source="openrouter", status="advisory_empty"),
            )
    except Exception as e:
        log.info("Architect advisory skipped: %s", str(e)[:160])
        # Second chance advisory via public fallback when OpenRouter advisory fails.
        public_data = await _call_public_architect_advisory(system_prompt, user_message)
        public_trace = _extract_llm_architect_reasoning(public_data or {})
        if public_trace:
            trace.append("OpenRouter advisory failed; public LLM advisory applied as fallback.")
            trace = _merge_reasoning_trace(trace, public_trace, limit=16)
            return (
                _normalize_reasoning_lines(trace, limit=16),
                _build_architect_reasoning_object(
                    req,
                    uw,
                    ul,
                    source="public_fallback",
                    status="advisory_applied_after_openrouter_failure",
                    advisory=public_data,
                ),
            )

        trace.append("LLM architect advisory timed out/unavailable; local reasoning used.")

    return (
        _normalize_reasoning_lines(trace, limit=16),
        _build_architect_reasoning_object(req, uw, ul, source="local", status="advisory_timeout"),
    )


def _sanitize_room_type(raw_type: str, raw_label: str = "") -> str:
    t = (raw_type or "").strip().lower().replace(" ", "_")
    if t in ROOM_COLORS:
        return t

    t = ROOM_TYPE_ALIASES.get(t, t)
    if t in ROOM_COLORS:
        return t

    label = (raw_label or "").strip().lower()
    if "master" in label and "bed" in label:
        return "master_bedroom"
    if "bed" in label:
        return "bedroom"
    if "bath" in label or "toilet" in label or "wash" in label:
        return "bathroom"
    if "living" in label or "hall" in label:
        return "living"
    if "dining" in label:
        return "dining"
    if "kitchen" in label:
        return "kitchen"
    if "corridor" in label or "passage" in label:
        return "corridor"
    if "study" in label:
        return "study"
    if "pooja" in label or "puja" in label or "mandir" in label:
        return "pooja"
    if "balcony" in label:
        return "balcony"
    if "store" in label:
        return "store"
    if "utility" in label:
        return "utility"
    if "open" in label or "sit-out" in label or "terrace" in label:
        return "open_area"

    return t


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _room_overlap(a: RoomData, b: RoomData, eps: float = 0.05) -> bool:
    return (
        a.x < b.x + b.width - eps
        and a.x + a.width > b.x + eps
        and a.y < b.y + b.height - eps
        and a.y + a.height > b.y + eps
    )


def _shared_wall_length(a: RoomData, b: RoomData, eps: float = 0.35) -> float:
    # Vertical shared wall
    if abs((a.x + a.width) - b.x) <= eps or abs((b.x + b.width) - a.x) <= eps:
        return max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))

    # Horizontal shared wall
    if abs((a.y + a.height) - b.y) <= eps or abs((b.y + b.height) - a.y) <= eps:
        return max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))

    return 0.0


def _normalize_polygon_points(
    raw_points: Any,
    x_max: float,
    y_max: float,
) -> list[dict[str, float]]:
    if not isinstance(raw_points, list):
        return []

    pts: list[dict[str, float]] = []
    for p in raw_points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = _to_float(p[0], float("nan"))
            y = _to_float(p[1], float("nan"))
        else:
            continue

        if not math.isfinite(x) or not math.isfinite(y):
            continue
        x = _clamp(x, 0.0, max(0.0, x_max))
        y = _clamp(y, 0.0, max(0.0, y_max))
        pts.append({"x": _rnd(x, 2), "y": _rnd(y, 2)})

    # Remove consecutive duplicates
    cleaned: list[dict[str, float]] = []
    for p in pts:
        if not cleaned:
            cleaned.append(p)
            continue
        if abs(cleaned[-1]["x"] - p["x"]) < 0.01 and abs(cleaned[-1]["y"] - p["y"]) < 0.01:
            continue
        cleaned.append(p)

    if len(cleaned) >= 2:
        first = cleaned[0]
        last = cleaned[-1]
        if abs(first["x"] - last["x"]) < 0.01 and abs(first["y"] - last["y"]) < 0.01:
            cleaned.pop()

    return cleaned if len(cleaned) >= 3 else []


def _polygon_bbox(points: list[dict[str, float]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        else:
            x = _to_float(getattr(p, "x", float("nan")), float("nan"))
            y = _to_float(getattr(p, "y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return 0.0, 0.0, 0.01, 0.01

    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return x0, y0, max(0.01, x1 - x0), max(0.01, y1 - y0)


def _translate_polygon(
    points: list[dict[str, float]],
    dx: float,
    dy: float,
    x_max: float,
    y_max: float,
) -> list[dict[str, float]]:
    moved = []
    for p in points:
        if isinstance(p, dict):
            px = _to_float(p.get("x"), 0.0)
            py = _to_float(p.get("y"), 0.0)
        else:
            px = _to_float(getattr(p, "x", 0.0), 0.0)
            py = _to_float(getattr(p, "y", 0.0), 0.0)
        moved.append(
            {
                "x": _rnd(_clamp(px + dx, 0.0, max(0.0, x_max)), 2),
                "y": _rnd(_clamp(py + dy, 0.0, max(0.0, y_max)), 2),
            }
        )
    return moved


def _polygon_area(points: list[dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0

    clean: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        else:
            x = _to_float(getattr(p, "x", float("nan")), float("nan"))
            y = _to_float(getattr(p, "y", float("nan")), float("nan"))
        if math.isfinite(x) and math.isfinite(y):
            clean.append((x, y))

    if len(clean) < 3:
        return 0.0

    area2 = 0.0
    for i, (x1, y1) in enumerate(clean):
        x2, y2 = clean[(i + 1) % len(clean)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


# ─────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────
async def generate_plan(req: PlanRequest) -> PlanResponse:
    """Generate a floor plan with OpenRouter -> Claude -> BSP fallback."""
    uw = _rnd(req.plot_width - SETBACKS["left"] - SETBACKS["right"], 2)
    ul = _rnd(req.plot_length - SETBACKS["front"] - SETBACKS["rear"], 2)
    preplot_trace, preplot_reasoning = await _build_preplot_reasoning_trace(req, uw, ul)

    has_llm_keys = bool(OPENROUTER_API_KEY.strip() or OPENROUTER_API_KEY_SECONDARY.strip())
    force_local_only = FORCE_LOCAL_PLANNER and not has_llm_keys
    if FORCE_LOCAL_PLANNER and has_llm_keys:
        log.info(
            "FORCE_LOCAL_PLANNER is enabled but API keys are present; "
            "using reasoning+LLM pipeline instead of deterministic lock.",
        )

    if force_local_only:
        local_plan = await _generate_via_bsp(req, uw, ul)
        local_plan.generation_method = "deterministic_demo"
        existing_trace = list(local_plan.reasoning_trace or [])
        local_plan.reasoning_trace = _merge_reasoning_trace(
            preplot_trace + ["Presentation-safe mode enabled: deterministic planner selected."],
            existing_trace + ["Returned stable adjacency-first layout without external model variance."],
            limit=18,
        )
        if not str(local_plan.architect_note or "").strip():
            local_plan.architect_note = (
                "Deterministic demo mode is enabled for consistent architectural output."
            )
        return _attach_plan_reasoning(
            local_plan,
            preplot_reasoning,
            req,
            uw,
            ul,
            "deterministic_demo",
        )

    # Attempt 1: OpenRouter with hard 45s cap.
    openrouter_issues: list[str] = []
    try:
        openrouter_plan, openrouter_issues, _ = await asyncio.wait_for(
            _generate_via_llm(req, uw, ul),
            timeout=OPENROUTER_ATTEMPT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        openrouter_plan = None
        openrouter_issues = ["OpenRouter attempt timed out"]
        log.warning("OpenRouter attempt timed out after %ss", OPENROUTER_ATTEMPT_TIMEOUT_SEC)

    if openrouter_plan is not None:
        openrouter_plan.reasoning_trace = _merge_reasoning_trace(
            preplot_trace,
            list(openrouter_plan.reasoning_trace or [])
            + ["Plotting completed via OpenRouter architectural planner."],
            limit=18,
        )
        return _attach_plan_reasoning(
            openrouter_plan,
            preplot_reasoning,
            req,
            uw,
            ul,
            "openrouter",
        )

    if openrouter_issues:
        log.warning("OpenRouter draft rejected: %s", openrouter_issues[:4])

    # Attempt 2: Claude (only if key is configured). No OpenRouter retries.
    if ANTHROPIC_API_KEY.strip():
        try:
            claude_plan = await asyncio.wait_for(
                _generate_via_claude(req, uw, ul),
                timeout=CLAUDE_ATTEMPT_DEADLINE_SEC,
            )
            claude_plan.reasoning_trace = _merge_reasoning_trace(
                preplot_trace,
                list(claude_plan.reasoning_trace or [])
                + ["Plotting completed via Claude after architect reasoning stage."],
                limit=18,
            )
            return _attach_plan_reasoning(
                claude_plan,
                preplot_reasoning,
                req,
                uw,
                ul,
                "claude",
            )
        except asyncio.TimeoutError:
            log.warning("Claude attempt exceeded %ss end-to-end deadline", CLAUDE_ATTEMPT_DEADLINE_SEC)
        except Exception as e:
            log.warning("Claude attempt failed: %s", str(e)[:200])
    else:
        log.info("ANTHROPIC_API_KEY not set; skipping Claude and falling back to BSP")

    # Final fallback: deterministic BSP immediately.
    fallback = await _generate_via_bsp(req, uw, ul)
    fallback.generation_method = "bsp"
    fallback.reasoning_trace = _merge_reasoning_trace(
        preplot_trace,
        [
            "External plotting attempts were unavailable or invalid.",
            *list(fallback.reasoning_trace or []),
            "Returned deterministic BSP fallback for guaranteed completion.",
        ],
        limit=18,
    )
    return _attach_plan_reasoning(
        fallback,
        preplot_reasoning,
        req,
        uw,
        ul,
        "bsp_fallback",
    )


def _zone_bounds_for_room_spec(
    room_spec: dict[str, Any],
    uw: float,
    ul: float,
) -> tuple[float, float, float, float]:
    zone = int(_to_float(room_spec.get("zone"), 2))
    room_type = str(room_spec.get("type", "")).strip().lower()

    front_max = _rnd(max(8.0, ul * 0.32), 2)
    service_min = _rnd(max(0.0, ul * 0.22), 2)
    service_max = _rnd(max(service_min + 2.0, ul * 0.78), 2)
    private_min = _rnd(max(0.0, ul * 0.45), 2)

    x0, y0, x1, y1 = 0.0, 0.0, uw, ul
    if zone == 1:
        y1 = front_max
    elif zone == 2:
        y0, y1 = service_min, min(ul, service_max)
    elif zone == 3:
        y0 = private_min

    if room_type == "kitchen":
        x0 = max(x0, uw * 0.5)
    if room_type == "master_bedroom":
        x1 = min(x1, uw * 0.55)
    if room_type == "pooja":
        x0 = max(x0, uw * 0.5)
        y1 = min(y1, front_max)
    if room_type == "corridor":
        y0 = 0.0
        y1 = ul

    x0 = _rnd(_clamp(x0, 0.0, uw), 2)
    y0 = _rnd(_clamp(y0, 0.0, ul), 2)
    x1 = _rnd(_clamp(x1, x0, uw), 2)
    y1 = _rnd(_clamp(y1, y0, ul), 2)
    return x0, y0, x1, y1


def _build_room_coordinate_constraints(
    req: PlanRequest,
    uw: float,
    ul: float,
) -> list[dict[str, Any]]:
    room_specs = build_room_list(req.bedrooms, req.extras, uw, ul, req.facing, req.floors)

    requested_baths = max(0, int(getattr(req, "bathrooms_target", 0) or 0))
    existing_baths = sum(
        1 for spec in room_specs if spec.get("type") in ("master_bath", "bathroom", "toilet")
    )
    extra_baths = max(0, min(8, requested_baths) - existing_baths)
    for idx in range(extra_baths):
        room_specs.append({
            "type": "bathroom",
            "label": f"Bathroom {existing_baths + idx + 1}",
            "min_w": 5,
            "min_h": 6,
            "zone": 2,
            "priority": 20 + idx,
        })

    type_counts: dict[str, int] = {}
    constraints: list[dict[str, Any]] = []
    for spec in room_specs:
        room_type = str(spec.get("type", "room")).strip().lower()
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        room_id = f"{room_type}_{type_counts[room_type]:02d}"
        label = str(spec.get("label", DEFAULT_ROOM_LABELS.get(room_type, room_type.title())))

        min_w = _rnd(max(3.5, _to_float(spec.get("min_w"), 5.0)), 2)
        min_h = _rnd(max(3.5, _to_float(spec.get("min_h"), 5.0)), 2)

        zx0, zy0, zx1, zy1 = _zone_bounds_for_room_spec(spec, uw, ul)
        x_min = zx0
        y_min = zy0
        x_max = _rnd(max(x_min, zx1 - min_w), 2)
        y_max = _rnd(max(y_min, zy1 - min_h), 2)

        constraints.append(
            {
                "id": room_id,
                "type": room_type,
                "label": label,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "min_w": min_w,
                "min_h": min_h,
            }
        )

    return constraints


def _build_claude_coordinate_messages(
    req: PlanRequest,
    uw: float,
    ul: float,
) -> tuple[str, str]:
    constraints = _build_room_coordinate_constraints(req, uw, ul)
    room_lines = [
        (
            f"{c['label']}: x {c['x_min']} to {c['x_max']}, y {c['y_min']} to {c['y_max']}, "
            f"min width {c['min_w']}, min height {c['min_h']}."
        )
        for c in constraints
    ]

    system_prompt = (
        "You are a floor-plan coordinate generator. "
        "Use only numeric ranges provided by the user and return valid JSON only."
    )

    user_message = (
        f"Plot usable area is {uw} x {ul} feet. Place rooms in these exact coordinate ranges.\n"
        + "\n".join(room_lines)
        + "\nReturn only JSON matching this schema exactly: "
        "rooms array where each room has id, type, label, x, y, width, height. "
        "No markdown, no explanation."
    )
    return system_prompt, user_message


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    candidates: list[str] = [text]
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break

    for candidate in candidates:
        payload = candidate.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON object from model output")


def _plan_to_dict(plan: PlanResponse) -> dict[str, Any]:
    return {
        "rooms": [
            {
                "id": r.id,
                "type": r.type,
                "label": r.label,
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "height": r.height,
                "area": r.area,
                "polygon": [{"x": p.x, "y": p.y} for p in (r.polygon or [])],
            }
            for r in plan.rooms
        ]
    }


async def _generate_via_claude(req: PlanRequest, uw: float, ul: float) -> PlanResponse:
    """Generate a plan using Anthropic Claude with coordinate-constrained prompting."""
    if not ANTHROPIC_API_KEY.strip():
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    reasoning_trace: list[str] = [
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
        "Built coordinate-first room ranges from deterministic zoning rules.",
    ]

    system_prompt, user_message = _build_claude_coordinate_messages(req, uw, ul)
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 2200,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    timeout = httpx.Timeout(CLAUDE_API_TIMEOUT_SEC)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)

    if response.status_code >= 400:
        raise RuntimeError(f"Claude API error {response.status_code}: {response.text[:220]}")

    data = response.json()
    content = data.get("content", [])
    text_blocks = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    raw_text = "\n".join(text_blocks).strip()
    plan_dict = _extract_json_object(raw_text)

    draft_valid, draft_issues = validate_draft(plan_dict, uw, ul)
    if not draft_valid:
        raise ValueError("Claude draft failed fatal checks: " + "; ".join(draft_issues[:6]))

    reasoning_trace.append("Claude draft passed draft validation; applying deterministic normalization.")
    plan = _parse_llm_plan(plan_dict, req, uw, ul, reasoning_trace=reasoning_trace)

    final_valid, final_issues = validate_final(_plan_to_dict(plan), uw, ul, req)
    if not final_valid:
        raise ValueError("Claude plan failed final validation: " + "; ".join(final_issues[:6]))

    plan.generation_method = "claude"
    plan.reasoning_trace = [
        *plan.reasoning_trace,
        "Claude layout passed full final validation and was accepted.",
    ][:12]
    return plan


async def _generate_via_llm(
    req: PlanRequest,
    uw: float,
    ul: float,
    prev_issues: list[str] | None = None,
) -> tuple[PlanResponse | None, list[str], list[str] | None]:
    """
    Try to generate a plan via the LLM.
    Returns (PlanResponse, [], None) if successful.
    Returns (None, validator_issues, correction_feedback) if failed.
    """
    reasoning_trace: list[str] = []
    _push_reasoning(
        reasoning_trace,
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
    )

    system_prompt, user_message = build_master_prompt(req)
    _push_reasoning(
        reasoning_trace,
        "Built coordinate-first OpenRouter prompt with hard room ranges.",
        )

    # Append correction feedback if retrying
    if prev_issues:
        # Earlier retries sent vague issue lists; now we provide concrete geometry
        # feedback so the model repairs coordinates instead of regenerating randomly.
        correction = (
            "\n\nYour previous plan had these specific geometric errors:\n"
            + "\n".join(f"- {issue}" for issue in prev_issues[:14])
            + "\n\nKeep all already-valid rooms stable. "
            + "Only change coordinates that are explicitly called out above."
        )
        user_message += correction
        log.info("Retrying LLM with %d correction items", len(prev_issues))
        _push_reasoning(
            reasoning_trace,
            f"Retrying with {len(prev_issues)} validator corrections from previous draft.",
        )

    _push_reasoning(reasoning_trace, "Requesting floor plan draft from OpenRouter.")

    try:
        plan_dict = await call_openrouter_plan(system_prompt, user_message)
    except Exception as e:
        log.error("LLM call failed across all models: %s", e)
        return None, [str(e)], None

    draft_rooms = plan_dict.get("rooms", []) if isinstance(plan_dict, dict) else []
    if isinstance(draft_rooms, list):
        _push_reasoning(
            reasoning_trace,
            f"OpenRouter returned draft with {len(draft_rooms)} rooms; running draft validation.",
        )

    draft_valid, draft_issues = validate_draft(plan_dict, uw, ul)
    if not draft_valid:
        correction_feedback = _build_geometric_correction_feedback(plan_dict, uw, ul, draft_issues)
        _push_reasoning(
            reasoning_trace,
            f"OpenRouter draft failed fatal validation with {len(draft_issues)} issues.",
        )
        return None, draft_issues, correction_feedback

    # Build production-grade plan with geometry/opening repair first.
    try:
        plan = _parse_llm_plan(plan_dict, req, uw, ul, reasoning_trace=reasoning_trace)
    except Exception as e:
        parse_error = f"LLM draft parsing failed: {str(e)[:120]}"
        issues = [parse_error]
        if parse_error not in issues:
            issues.append(parse_error)
        correction_feedback = _build_geometric_correction_feedback(plan_dict, uw, ul, issues)
        log.warning("LLM plan parse failed: %s", parse_error)
        return None, issues, correction_feedback

    repaired_dict = _plan_to_dict(plan)
    final_valid, final_issues = validate_final(repaired_dict, uw, ul, req)
    if not final_valid:
        correction_feedback = _build_geometric_correction_feedback(repaired_dict, uw, ul, final_issues)
        log.warning("Post-repair plan invalid (%d issues): %s", len(final_issues), final_issues[:3])
        _push_reasoning(
            reasoning_trace,
            f"Post-repair validation failed with {len(final_issues)} issues.",
        )
        return None, [f"post-repair: {issue}" for issue in final_issues], correction_feedback

    _push_reasoning(reasoning_trace, "Final repaired plan passed validation and is ready for preview.")

    plan.generation_method = "llm"
    return plan, [], None


def _build_geometric_correction_feedback(
    plan_dict: dict[str, Any],
    uw: float,
    ul: float,
    issues: list[str],
) -> list[str]:
    """
    Build explicit coordinate-level correction guidance for retry prompts.
    This replaces vague issue summaries with room-wise geometric actions.
    """
    raw_rooms = plan_dict.get("rooms", []) if isinstance(plan_dict, dict) else []
    if not isinstance(raw_rooms, list):
        raw_rooms = []

    front_max = ul * 0.30
    private_min = ul * 0.45
    middle_min = ul * 0.30
    middle_max = ul * 0.75

    rooms: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rooms):
        if not isinstance(raw, dict):
            continue
        room_type = _sanitize_room_type(str(raw.get("type", "")), str(raw.get("label", "")))
        label = str(raw.get("label", raw.get("id", f"room_{idx+1}"))).strip() or f"room_{idx+1}"
        w = max(0.1, _to_float(raw.get("width"), 0.0))
        h = max(0.1, _to_float(raw.get("height"), 0.0))
        x = _to_float(raw.get("x"), 0.0)
        y = _to_float(raw.get("y"), 0.0)
        rooms.append({
            "type": room_type,
            "label": label,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    feedback: list[str] = []

    for room in rooms:
        x = room["x"]
        y = room["y"]
        w = room["w"]
        h = room["h"]
        room_type = room["type"]
        label = room["label"]

        room_issues: list[str] = []
        x_lo = 0.0
        x_hi = max(0.0, uw - w)
        y_lo = 0.0
        y_hi = max(0.0, ul - h)

        if x < -1.0 or y < -1.0 or x + w > uw + 1.0 or y + h > ul + 1.0:
            room_issues.append("It is outside usable bounds")
            x_lo = max(x_lo, 0.0)
            x_hi = min(x_hi, max(0.0, uw - w))
            y_lo = max(y_lo, 0.0)
            y_hi = min(y_hi, max(0.0, ul - h))

        if room_type in ("master_bedroom", "bedroom"):
            if y < private_min:
                room_issues.append("Bedroom is below rear privacy band")
            y_lo = max(y_lo, private_min)

        if room_type == "living":
            if y + h > front_max + 0.1:
                room_issues.append("Living room is not fully in the front public band")
            y_hi = min(y_hi, max(0.0, front_max - h))

        if room_type == "kitchen":
            if y < middle_min or y + h > middle_max:
                room_issues.append("Kitchen is not fully in the middle band")
            if x + w < uw * 0.5:
                room_issues.append("Kitchen is not in southeast side")
            y_lo = max(y_lo, middle_min)
            y_hi = min(y_hi, max(y_lo, middle_max - h))
            x_lo = max(x_lo, uw * 0.5 - w)

        if room_type == "master_bedroom":
            if x + w > uw * 0.55:
                room_issues.append("Master bedroom is not in southwest quadrant")
            x_hi = min(x_hi, max(0.0, uw * 0.55 - w))

        if room_type == "pooja":
            if x < uw * 0.5 or y + h > front_max + 0.1:
                room_issues.append("Pooja room is not in northeast front quadrant")
            x_lo = max(x_lo, uw * 0.5)
            y_hi = min(y_hi, max(0.0, front_max - h))

        if room_type == "corridor" and w < 3.5:
            room_issues.append("Corridor is narrower than 3.5 ft")

        if room_issues:
            feedback.append(
                f"{label} is at x={x:.1f}, y={y:.1f}, width={w:.1f}, height={h:.1f}. "
                + "; ".join(room_issues)
                + f". Move it to x in [{max(0.0, x_lo):.1f}, {max(x_lo, x_hi):.1f}] "
                + f"and y in [{max(0.0, y_lo):.1f}, {max(y_lo, y_hi):.1f}]."
            )

    # Overlap feedback with explicit target direction.
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            overlap_x = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
            overlap_y = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
            if overlap_x <= 0.2 or overlap_y <= 0.2:
                continue

            target_x = b["x"]
            target_y = b["y"]
            if overlap_x >= overlap_y:
                target_y = min(max(0.0, a["y"] + a["h"] + 0.5), max(0.0, ul - b["h"]))
            else:
                target_x = min(max(0.0, a["x"] + a["w"] + 0.5), max(0.0, uw - b["w"]))

            feedback.append(
                f"{b['label']} is at x={b['x']:.1f}, y={b['y']:.1f}, width={b['w']:.1f}, height={b['h']:.1f}. "
                + f"It overlaps {a['label']} by {overlap_x:.1f} ft x {overlap_y:.1f} ft. "
                + f"Move it near x={target_x:.1f}, y={target_y:.1f} to remove overlap."
            )

    # Preserve direct validator messages as backstop guidance.
    if not feedback:
        for issue in issues[:10]:
            feedback.append(f"Validator issue to fix: {issue}")

    return feedback[:18]


def _parse_llm_plan(
    plan_dict: dict,
    req: PlanRequest,
    uw: float,
    ul: float,
    reasoning_trace: list[str] | None = None,
) -> PlanResponse:
    """Convert LLM JSON into a production-ready PlanResponse."""
    trace = reasoning_trace if reasoning_trace is not None else []
    meta = plan_dict.get("metadata", {}) if isinstance(plan_dict, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    synthetic_layout = bool(meta.get("synthetic_layout", False))

    # 1) Normalize room program to requested BHK + extras
    rooms = _normalize_rooms_from_program(plan_dict.get("rooms", []), req, uw, ul)
    _push_reasoning(
        trace,
        f"Normalized and filtered rooms to {len(rooms)} valid spaces within requested program.",
    )

    # 2) Repair geometry into buildable layout
    if synthetic_layout:
        _push_reasoning(trace, "Using synthesized LLM-assisted geometry without aggressive re-snapping.")
    else:
        _snap_and_fix_layout(rooms, uw, ul)
        _push_reasoning(trace, "Applied geometric repair: snap, zone anchoring, and overlap resolution.")

        # 2b) Deterministic post-processing after validator pass.
        # This cleans sub-foot slivers and enforces buildable corridor width.
        _postprocess_llm_layout(rooms, uw, ul)
        _push_reasoning(trace, "Post-processed layout: 0.5ft snap, sliver-gap fill, corridor width normalization.")

    # 3) Compute area/exterior walls after repair
    _infer_exterior_walls(rooms, uw, ul)
    for r in rooms:
        if r.polygon:
            p_area = _polygon_area(r.polygon)
            r.area = _rnd(p_area if p_area > 1.0 else (r.width * r.height), 1)
        else:
            r.area = _rnd(r.width * r.height, 1)

    # 4) Openings are regenerated from repaired layout for consistency
    doors, windows = _build_openings_from_rooms(rooms, req.facing, uw, ul)
    _push_reasoning(
        trace,
        f"Regenerated openings from repaired geometry ({len(doors)} doors, {len(windows)} windows).",
    )

    # 5) Metadata
    vastu_score = _clamp(_to_float(meta.get("vastu_score"), 75.0), 40.0, 100.0)
    architect_note = meta.get(
        "architect_note",
        "LLM-generated plan refined with architectural constraints for practical use."
    )
    vastu_issues = meta.get("vastu_issues", [])
    if not isinstance(vastu_issues, list):
        vastu_issues = []
    adjacency_score = _to_float(meta.get("adjacency_score"), -1.0)
    if adjacency_score < 0:
        adjacency_score = _compute_adjacency_score(rooms)
    adjacency_score = _clamp(adjacency_score, 0.0, 100.0)

    raw_boundary = plan_dict.get("plot_boundary", meta.get("plot_boundary", []))
    boundary_points = _normalize_polygon_points(raw_boundary, uw, ul)
    if boundary_points:
        _push_reasoning(
            trace,
            f"Detected free-form plot boundary with {len(boundary_points)} points.",
        )
    else:
        _push_reasoning(trace, "Using rectangular usable boundary (no plot polygon in draft).")

    plot = PlotInfo(
        width=req.plot_width,
        length=req.plot_length,
        usable_width=uw,
        usable_length=ul,
        road_side=req.facing,
        setbacks=SETBACKS,
        boundary=[Point2D(x=p["x"], y=p["y"]) for p in boundary_points],
    )

    return PlanResponse(
        plot=plot,
        rooms=rooms,
        doors=doors,
        windows=windows,
        vastu_score=vastu_score,
        architect_note=architect_note,
        generation_method="llm",
        vastu_issues=vastu_issues,
        adjacency_score=adjacency_score,
        reasoning_trace=trace[:12],
    )


def _normalize_rooms_from_program(
    raw_rooms: list[dict[str, Any]],
    req: PlanRequest,
    uw: float,
    ul: float,
) -> list[RoomData]:
    """
    Normalize raw LLM rooms while preserving LLM creativity.
    This path intentionally avoids deterministic 1/2/3/4 BHK template remapping.
    """
    requested_extra_types = _requested_extra_room_types(req.extras)
    allowed_types = set(ROOM_COLORS.keys()) | {"toilet"}

    if isinstance(raw_rooms, dict):
        mapped_rooms: list[dict[str, Any]] = []
        for room_name, room_data in raw_rooms.items():
            if not isinstance(room_data, dict):
                continue
            merged = dict(room_data)
            merged.setdefault("name", str(room_name))
            merged.setdefault("type", str(room_name))
            mapped_rooms.append(merged)
        raw_rooms = mapped_rooms
    elif not isinstance(raw_rooms, list):
        raw_rooms = []

    rooms: list[RoomData] = []
    type_serial: dict[str, int] = {}
    used_ids: set[str] = set()

    for raw in raw_rooms or []:
        if not isinstance(raw, dict):
            continue

        type_or_name = raw.get("type", raw.get("name", ""))
        label_or_name = raw.get("label", raw.get("name", ""))
        room_type = _sanitize_room_type(
            str(type_or_name),
            str(label_or_name),
        )
        if room_type not in allowed_types:
            continue
        if room_type in OPTIONAL_EXTRA_ROOM_TYPES and room_type not in requested_extra_types:
            continue

        min_w, min_h = ROOM_MIN_DIMS.get(room_type, (4.0, 4.0))
        min_w = min(min_w, max(4.0, uw))
        min_h = min(min_h, max(4.0, ul))

        raw_polygon = _normalize_polygon_points(
            raw.get("polygon", raw.get("vertices", [])),
            uw,
            ul,
        )

        polygon_bbox: tuple[float, float, float, float] | None = None
        if raw_polygon:
            bx, by, bw, bh = _polygon_bbox(raw_polygon)
            if bw >= max(2.8, min_w * 0.6) and bh >= max(2.8, min_h * 0.6):
                polygon_bbox = (bx, by, bw, bh)
            else:
                raw_polygon = []

        if polygon_bbox:
            x, y, width, height = polygon_bbox
        else:
            width = _clamp(_to_float(raw.get("width"), min_w), min_w, max(min_w, uw))
            height = _clamp(_to_float(raw.get("height"), min_h), min_h, max(min_h, ul))
            x = _to_float(raw.get("x"), 0.0)
            y = _to_float(raw.get("y"), 0.0)

        x = _clamp(x, 0.0, max(0.0, uw - width))
        y = _clamp(y, 0.0, max(0.0, ul - height))

        zone_name = str(raw.get("zone", "")).strip().lower()
        if zone_name not in ("public", "service", "private"):
            zone_name = ROOM_DEFAULT_ZONES.get(room_type, "service")

        band = int(_to_float(raw.get("band"), 0))
        if band not in (1, 2, 3):
            band = 1 if zone_name == "public" else 2 if zone_name == "service" else 3

        type_serial[room_type] = type_serial.get(room_type, 0) + 1
        default_id = f"{room_type}_{type_serial[room_type]:02d}"
        room_id = str(raw.get("id", "")).strip().lower().replace(" ", "_") or default_id
        while room_id in used_ids:
            room_id = f"{room_id}_{type_serial[room_type]}"
        used_ids.add(room_id)

        label = str(raw.get("label", raw.get("name", ""))).strip()
        if not label:
            label = _default_room_label(room_type, type_serial[room_type])

        rooms.append(
            RoomData(
                id=room_id,
                type=room_type,
                label=label,
                x=x,
                y=y,
                width=width,
                height=height,
                area=_rnd(width * height, 1),
                zone=zone_name,
                band=band,
                exterior_walls=[],
                color=ROOM_COLORS.get(room_type, "#F5F5F5"),
                polygon=[Point2D(x=p["x"], y=p["y"]) for p in raw_polygon],
            )
        )

    # Keep requested bedroom count bounded while preserving one master bedroom.
    if req.bedrooms > 0:
        bedroom_rooms = [r for r in rooms if r.type in ("master_bedroom", "bedroom")]
        if bedroom_rooms:
            master_idx = next((i for i, r in enumerate(bedroom_rooms) if r.type == "master_bedroom"), None)
            if master_idx is None:
                bedroom_rooms[0].type = "master_bedroom"
                bedroom_rooms[0].label = _default_room_label("master_bedroom")
                bedroom_rooms[0].zone = "private"
                bedroom_rooms[0].band = 3
                bedroom_rooms[0].color = ROOM_COLORS.get("master_bedroom", bedroom_rooms[0].color)
                master_idx = 0

            for i, room in enumerate(bedroom_rooms):
                if i == master_idx:
                    continue
                if room.type == "master_bedroom":
                    room.type = "bedroom"
                    room.color = ROOM_COLORS.get("bedroom", room.color)
                    if "master" in room.label.lower():
                        room.label = _default_room_label("bedroom", i + 1)

            keep_ids: set[str] = {bedroom_rooms[master_idx].id}
            for room in bedroom_rooms:
                if len(keep_ids) >= req.bedrooms:
                    break
                keep_ids.add(room.id)

            if len(bedroom_rooms) > req.bedrooms:
                rooms = [
                    room
                    for room in rooms
                    if room.type not in ("master_bedroom", "bedroom") or room.id in keep_ids
                ]

    return rooms


def _infer_exterior_walls(rooms: list[RoomData], uw: float, ul: float):
    eps = 0.35
    for room in rooms:
        walls: list[str] = []
        if room.x <= eps:
            walls.append("west")
        if room.y <= eps:
            walls.append("south")
        if abs((room.x + room.width) - uw) <= eps:
            walls.append("east")
        if abs((room.y + room.height) - ul) <= eps:
            walls.append("north")
        room.exterior_walls = walls


def _compute_adjacency_score(rooms: list[RoomData]) -> float:
    weighted_pairs = [
        ("living", "dining", 16.0),
        ("dining", "kitchen", 18.0),
        ("master_bedroom", "master_bath", 20.0),
        ("corridor", "master_bedroom", 14.0),
        ("corridor", "bedroom", 16.0),
        ("living", "foyer", 8.0),
        ("kitchen", "utility", 8.0),
    ]

    possible = 0.0
    achieved = 0.0

    for type_a, type_b, weight in weighted_pairs:
        a_rooms = [r for r in rooms if r.type == type_a]
        b_rooms = [r for r in rooms if r.type == type_b]
        if not a_rooms or not b_rooms:
            continue

        possible += weight
        if any(_shared_wall_length(a, b) >= 2.0 for a in a_rooms for b in b_rooms):
            achieved += weight

    if possible <= 0:
        return 75.0
    return _rnd((achieved / possible) * 100.0, 1)


def _logical_layout_issues(rooms: list[RoomData], uw: float, ul: float) -> list[str]:
    """Logical quality gate for accepting LLM plans."""
    issues: list[str] = []
    if not rooms:
        return ["No rooms found in candidate plan."]

    by_type: dict[str, list[RoomData]] = {}
    for room in rooms:
        by_type.setdefault(room.type, []).append(room)

    def _adjacent(type_a: str, type_b: str, min_len: float) -> bool:
        a_rooms = by_type.get(type_a, [])
        b_rooms = by_type.get(type_b, [])
        if not a_rooms or not b_rooms:
            return True
        return any(_shared_wall_length(a, b) >= min_len for a in a_rooms for b in b_rooms)

    def _in_front(room: RoomData) -> bool:
        return room.y + room.height <= ul * 0.40 + 0.5

    def _in_middle(room: RoomData) -> bool:
        return room.y >= ul * 0.18 and room.y + room.height <= ul * 0.84

    def _in_private(room: RoomData) -> bool:
        return room.y >= ul * 0.40

    living = by_type.get("living", [])
    kitchens = by_type.get("kitchen", [])
    bedrooms = by_type.get("master_bedroom", []) + by_type.get("bedroom", [])
    corridors = by_type.get("corridor", [])

    for room in living:
        if not _in_front(room):
            issues.append(f"{room.label} is not in front public zone.")

    for room in kitchens:
        if not _in_middle(room):
            issues.append(f"{room.label} is not in middle service zone.")

    for room in bedrooms:
        if not _in_private(room):
            issues.append(f"{room.label} is not in rear private zone.")

    if not _adjacent("living", "dining", 2.0):
        issues.append("Living and dining are not physically connected.")
    if not _adjacent("dining", "kitchen", 2.0):
        issues.append("Dining and kitchen are not physically connected.")
    if not _adjacent("master_bedroom", "master_bath", 1.5):
        issues.append("Master bath is not attached to master bedroom.")

    if bedrooms and not corridors:
        issues.append("Corridor is missing; circulation spine must connect entry to bedrooms.")

    if corridors and living:
        if not any(_shared_wall_length(liv, cor) >= 1.5 for liv in living for cor in corridors):
            issues.append("Living room is not connected to corridor spine.")

    wet_rooms = by_type.get("bathroom", []) + by_type.get("master_bath", []) + by_type.get("toilet", [])
    if corridors and wet_rooms:
        if not any(_shared_wall_length(wet, cor) >= 1.0 for wet in wet_rooms for cor in corridors):
            issues.append("Bathrooms are not connected to corridor spine.")

    if corridors and bedrooms:
        for bed in bedrooms:
            if not any(_shared_wall_length(bed, c) >= 1.5 for c in corridors):
                issues.append(f"{bed.label} is not connected to corridor.")

    # Reject visibly skinny key rooms even if they pass tolerance-based validator.
    for room in bedrooms + kitchens + living:
        min_w, min_h = ROOM_MIN_DIMS.get(room.type, (4.0, 4.0))
        if room.width < min_w * 0.9 or room.height < min_h * 0.9:
            issues.append(f"{room.label} is undersized for practical use.")

    adj = _compute_adjacency_score(rooms)
    if adj < LOGICAL_ADJ_MIN_SCORE:
        issues.append(f"Adjacency score too low ({adj:.1f}).")

    # Keep feedback concise for retry prompting.
    return issues[:12]


def _build_openings_from_rooms(
    rooms: list[RoomData],
    facing: str,
    uw: float,
    ul: float,
) -> tuple[list[DoorData], list[WindowData]]:
    placed_rooms = [
        {
            "id": r.id,
            "type": r.type,
            "label": r.label,
            "x": r.x,
            "y": r.y,
            "width": r.width,
            "height": r.height,
        }
        for r in rooms
    ]

    doors, windows = add_doors_and_windows(placed_rooms, uw, ul, facing)

    # Deduplicate by geometry signature to avoid stacked duplicate symbols.
    deduped_doors: list[DoorData] = []
    seen_doors: set[tuple[str, float, float, float]] = set()
    for d in doors:
        key = (d.wall, _rnd(d.x, 1), _rnd(d.y, 1), _rnd(d.width, 1))
        if key in seen_doors:
            continue
        seen_doors.add(key)
        d.width = _clamp(_to_float(d.width, 3.0), 2.5, 4.5)
        deduped_doors.append(d)

    deduped_windows: list[WindowData] = []
    seen_windows: set[tuple[str, float, float, float]] = set()
    for w in windows:
        key = (w.wall, _rnd(w.x, 1), _rnd(w.y, 1), _rnd(w.width, 1))
        if key in seen_windows:
            continue
        seen_windows.add(key)
        w.width = _clamp(_to_float(w.width, 4.0), 3.0, 6.0)
        deduped_windows.append(w)

    return deduped_doors, deduped_windows


def _snap_and_fix_layout(rooms: list[RoomData], uw: float, ul: float):
    """
    Greedy non-overlapping placement on 0.5ft grid.
    1) Sort rooms by area descending.
    2) Place largest room first inside its zone.
    3) Place each next room in the first available non-overlapping cell in zone.
    """
    if not rooms:
        return

    grid = 0.5
    eps = 0.05

    def _snap(val: float) -> float:
        return round(val / grid) * grid

    def _normalize_geometry(room: RoomData):
        if room.polygon:
            bx, by, bw, bh = _polygon_bbox(room.polygon)
            room.width = _clamp(_snap(max(3.0, bw)), 3.0, max(3.0, uw))
            room.height = _clamp(_snap(max(3.0, bh)), 3.0, max(3.0, ul))
            target_x = _clamp(_snap(bx), 0.0, max(0.0, uw - room.width))
            target_y = _clamp(_snap(by), 0.0, max(0.0, ul - room.height))
            dx = target_x - bx
            dy = target_y - by
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                moved = _translate_polygon(room.polygon, dx, dy, uw, ul)
                room.polygon = [Point2D(x=p["x"], y=p["y"]) for p in moved]
            room.x = target_x
            room.y = target_y
            return

        room.width = _clamp(_snap(room.width), 3.0, max(3.0, uw))
        room.height = _clamp(_snap(room.height), 3.0, max(3.0, ul))
        room.x = _clamp(_snap(room.x), 0.0, max(0.0, uw - room.width))
        room.y = _clamp(_snap(room.y), 0.0, max(0.0, ul - room.height))

    def _zone_origin_bounds(room: RoomData) -> tuple[float, float, float, float]:
        front_max = ul * 0.32
        service_min = ul * 0.22
        service_max = ul * 0.78
        private_min = ul * 0.45

        x_left, y_bottom, x_right, y_top = 0.0, 0.0, uw, ul
        if room.zone == "public":
            y_top = front_max
        elif room.zone == "service":
            y_bottom, y_top = service_min, service_max
        else:
            y_bottom = private_min

        if room.type == "kitchen":
            x_left = max(x_left, uw * 0.5)
        if room.type == "master_bedroom":
            x_right = min(x_right, uw * 0.55)
        if room.type == "pooja":
            x_left = max(x_left, uw * 0.5)
            y_top = min(y_top, front_max)
        if room.type == "corridor":
            y_bottom = 0.0
            y_top = ul

        x_lo = _clamp(_snap(x_left), 0.0, max(0.0, uw - room.width))
        y_lo = _clamp(_snap(y_bottom), 0.0, max(0.0, ul - room.height))
        x_hi = _clamp(_snap(max(x_left, x_right - room.width)), x_lo, max(0.0, uw - room.width))
        y_hi = _clamp(_snap(max(y_bottom, y_top - room.height)), y_lo, max(0.0, ul - room.height))
        return x_lo, x_hi, y_lo, y_hi

    def _candidate_overlaps(
        room: RoomData,
        x: float,
        y: float,
        other: RoomData,
    ) -> bool:
        return (
            x < other.x + other.width - eps
            and x + room.width > other.x + eps
            and y < other.y + other.height - eps
            and y + room.height > other.y + eps
        )

    def _first_available_slot(
        room: RoomData,
        placed: list[RoomData],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
    ) -> tuple[float, float] | None:
        y = _snap(y_lo)
        while y <= y_hi + 1e-6:
            x = _snap(x_lo)
            while x <= x_hi + 1e-6:
                if all(not _candidate_overlaps(room, x, y, other) for other in placed):
                    return _rnd(x, 2), _rnd(y, 2)
                x += grid
            y += grid
        return None

    def _apply_position(room: RoomData, x: float, y: float):
        x = _clamp(_snap(x), 0.0, max(0.0, uw - room.width))
        y = _clamp(_snap(y), 0.0, max(0.0, ul - room.height))
        if room.polygon:
            bx, by, _, _ = _polygon_bbox(room.polygon)
            moved = _translate_polygon(room.polygon, x - bx, y - by, uw, ul)
            room.polygon = [Point2D(x=p["x"], y=p["y"]) for p in moved]
        room.x = _rnd(x, 2)
        room.y = _rnd(y, 2)

    for room in rooms:
        _normalize_geometry(room)

    ordered = sorted(rooms, key=lambda r: r.width * r.height, reverse=True)
    placed: list[RoomData] = []
    for idx, room in enumerate(ordered):
        x_lo, x_hi, y_lo, y_hi = _zone_origin_bounds(room)

        preferred_x = _clamp(_snap(room.x), x_lo, x_hi)
        preferred_y = _clamp(_snap(room.y), y_lo, y_hi)

        slot: tuple[float, float] | None = None
        if idx == 0:
            if all(not _candidate_overlaps(room, preferred_x, preferred_y, other) for other in placed):
                slot = (_rnd(preferred_x, 2), _rnd(preferred_y, 2))
            else:
                slot = _first_available_slot(room, placed, x_lo, x_hi, y_lo, y_hi)
        else:
            slot = _first_available_slot(room, placed, x_lo, x_hi, y_lo, y_hi)

        if slot is None:
            fallback_slot = _first_available_slot(
                room,
                placed,
                0.0,
                max(0.0, uw - room.width),
                0.0,
                max(0.0, ul - room.height),
            )
            slot = fallback_slot

        if slot is None:
            raise ValueError(f"Unable to place {room.label} without overlap.")

        _apply_position(room, slot[0], slot[1])
        placed.append(room)

    hard_overlaps: list[str] = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
            overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
            if overlap_x > 0.1 and overlap_y > 0.1:
                hard_overlaps.append(
                    f"{a.label} overlaps {b.label} by {overlap_x:.2f}ft x {overlap_y:.2f}ft"
                )

    if hard_overlaps:
        for msg in hard_overlaps[:12]:
            log.error("layout hard-overlap: %s", msg)
        raise ValueError(f"Residual overlaps remain after repair ({len(hard_overlaps)} pairs > 0.1ft)")


def _postprocess_llm_layout(rooms: list[RoomData], uw: float, ul: float):
    """
    Deterministic cleanups after a draft passes validation.
    Steps:
    1) snap coordinates to 0.5ft grid,
    2) expand rooms to remove slivers below 2ft,
    3) normalize narrow corridors to 3.5ft minimum width.
    """
    if not rooms:
        return

    grid = 0.5

    def _snap(value: float) -> float:
        return round(value / grid) * grid

    def _fit(room: RoomData):
        room.width = _clamp(_snap(room.width), 3.0, max(3.0, uw))
        room.height = _clamp(_snap(room.height), 3.0, max(3.0, ul))
        room.x = _clamp(_snap(room.x), 0.0, max(0.0, uw - room.width))
        room.y = _clamp(_snap(room.y), 0.0, max(0.0, ul - room.height))

    for room in rooms:
        _fit(room)

    # If LLM returned corridor narrower than 3.5ft, widen it deterministically.
    for room in rooms:
        if room.type != "corridor":
            continue
        if room.width >= 3.5:
            continue
        room.width = 3.5
        room.x = _clamp(room.x, 0.0, max(0.0, uw - room.width))
        _fit(room)

    # Fill tiny boundary slivers that are difficult to build onsite.
    for room in rooms:
        left_gap = room.x
        right_gap = uw - (room.x + room.width)
        bottom_gap = room.y
        top_gap = ul - (room.y + room.height)

        if 0.0 < left_gap < 2.0:
            room.x = 0.0
            room.width = room.width + left_gap
        if 0.0 < right_gap < 2.0:
            room.width = room.width + right_gap
        if 0.0 < bottom_gap < 2.0:
            room.y = 0.0
            room.height = room.height + bottom_gap
        if 0.0 < top_gap < 2.0:
            room.height = room.height + top_gap
        _fit(room)

    # Fill tiny interior slivers by extending to nearest neighboring room.
    for _ in range(2):
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]

                y_overlap = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if y_overlap >= 2.0:
                    if a.x + a.width <= b.x:
                        gap = b.x - (a.x + a.width)
                        if 0.0 < gap < 2.0:
                            a.width = a.width + gap
                    elif b.x + b.width <= a.x:
                        gap = a.x - (b.x + b.width)
                        if 0.0 < gap < 2.0:
                            b.width = b.width + gap

                x_overlap = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                if x_overlap >= 2.0:
                    if a.y + a.height <= b.y:
                        gap = b.y - (a.y + a.height)
                        if 0.0 < gap < 2.0:
                            a.height = a.height + gap
                    elif b.y + b.height <= a.y:
                        gap = a.y - (b.y + b.height)
                        if 0.0 < gap < 2.0:
                            b.height = b.height + gap

                _fit(a)
                _fit(b)

    for room in rooms:
        _fit(room)



# ─────────────────────────────────────────────────────────────
# BSP FALLBACK — original deterministic engine (unchanged)
# ─────────────────────────────────────────────────────────────
async def _generate_via_bsp(req: PlanRequest, uw: float, ul: float) -> PlanResponse:
    """Generate a floor plan using deterministic BSP packing (fallback)."""
    # Keep fallback fully deterministic and fast (no external LLM calls).
    target_bedrooms = _effective_bedroom_target(req.bedrooms, uw, ul)
    downgraded_bedrooms = target_bedrooms < int(req.bedrooms)

    architect_note = (
        "Logical deterministic layout generated with strict zoning and practical adjacency."
    )
    if downgraded_bedrooms:
        architect_note = (
            f"Compact usable footprint supports about {target_bedrooms} bedrooms; "
            f"generated best-fit {target_bedrooms}BHK from requested {req.bedrooms}BHK."
        )

    # Build room specs
    room_specs = build_room_list(target_bedrooms, req.extras, uw, ul, req.facing, req.floors)
    requested_baths = max(0, int(getattr(req, "bathrooms_target", 0) or 0))
    existing_baths = sum(
        1 for spec in room_specs if spec.get("type") in ("master_bath", "bathroom", "toilet")
    )
    extra_baths = max(0, min(8, requested_baths) - existing_baths)
    for idx in range(extra_baths):
        room_specs.append({
            "type": "bathroom", "label": f"Bathroom {existing_baths + idx + 1}",
            "min_w": 5, "min_h": 6, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 20 + idx,
        })

    # Pack rooms deterministically
    placed = pack_rooms_bsp(room_specs, uw, ul, facing=req.facing)
    vastu_score = _compute_fallback_vastu_score(placed, uw, ul)

    # Ensure all BSP rooms carry deterministic unique IDs before opening generation.
    type_counts: Dict[str, int] = {}
    for room in placed:
        room_type = str(room.get("type", "room"))
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        room["id"] = f"{room_type}_{type_counts[room_type]:02d}"

    # Add doors and windows
    doors, windows = add_doors_and_windows(placed, uw, ul, req.facing)

    # Build response
    rooms = []
    for i, p in enumerate(placed):
        rooms.append(RoomData(
            id=str(p.get("id", f"{p['type']}_{i+1:02d}")),
            type=p["type"],
            label=p["label"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            area=_rnd(p["width"] * p["height"], 1),
            zone=("public" if p["zone"] == 1
                  else "service" if p["zone"] == 2
                  else "private"),
            band=p["band"],
            color=ROOM_COLORS.get(p["type"], "#F5F5F5"),
        ))

    adjacency_score = _compute_adjacency_score(rooms)

    plot = PlotInfo(
        width=req.plot_width,
        length=req.plot_length,
        usable_width=uw,
        usable_length=ul,
        road_side=req.facing,
        setbacks=SETBACKS,
    )

    return PlanResponse(
        plot=plot,
        rooms=rooms,
        doors=doors,
        windows=windows,
        vastu_score=vastu_score,
        architect_note=architect_note,
        generation_method="bsp",
        adjacency_score=adjacency_score,
        reasoning_trace=[
            "Deterministic planner used strict 3-band zoning.",
            "Living-dining-kitchen and bedroom-corridor adjacency were enforced.",
            (
                f"Bedroom program compacted to {target_bedrooms} due usable footprint limits."
                if downgraded_bedrooms
                else "Requested bedroom count was feasible within usable footprint."
            ),
            "Returned stable non-overlapping layout for reliable preview/export.",
        ],
    )


# ─────────────────────────────────────────────────────────────
# Room spec builder
# ─────────────────────────────────────────────────────────────
def _max_feasible_bedrooms(usable_w: float, usable_l: float) -> int:
    """
    Conservative bedroom feasibility for compact plots.
    Keeps deterministic output valid instead of forcing impossible programs.
    """
    area = max(1.0, usable_w * usable_l)
    max_beds = 1

    if area >= 470 and usable_w >= 16.5 and usable_l >= 22.0:
        max_beds = 2
    if area >= 700 and usable_w >= 20.5 and usable_l >= 31.5:
        max_beds = 3
    if area >= 930 and usable_w >= 23.0 and usable_l >= 34.0:
        max_beds = 4

    return max_beds


def _effective_bedroom_target(requested_bedrooms: int, usable_w: float, usable_l: float) -> int:
    requested = max(1, min(4, int(requested_bedrooms or 1)))
    return min(requested, _max_feasible_bedrooms(usable_w, usable_l))


def _is_extra_feasible(
    extra_type: str,
    usable_w: float,
    usable_l: float,
    bedrooms: int,
) -> bool:
    area = max(1.0, usable_w * usable_l)
    extra = str(extra_type or "").strip().lower()

    if extra == "study":
        return area >= 760 and usable_l >= 30.0 and bedrooms >= 2
    if extra == "garage":
        return area >= 1150 and usable_w >= 24.0
    if extra == "balcony":
        return area >= 700 and usable_w >= 20.0
    if extra == "utility":
        min_area = 900 + max(0, bedrooms) * 220
        return area >= min_area and usable_w >= 20.0
    if extra == "staircase":
        return usable_w >= 19.0 and usable_l >= 28.0
    if extra == "open_area":
        return area >= 900
    return True


def _partition_feasible_extras(
    extras: list[str],
    usable_w: float,
    usable_l: float,
    bedrooms: int,
) -> tuple[set[str], set[str]]:
    requested = _requested_extra_room_types(extras)
    feasible: set[str] = set()
    deferred: set[str] = set()
    for extra in requested:
        if _is_extra_feasible(extra, usable_w, usable_l, bedrooms):
            feasible.add(extra)
        else:
            deferred.add(extra)
    return feasible, deferred


def build_room_list(
    bedrooms: int,
    extras: List[str],
    usable_w: float,
    usable_l: float,
    facing: str = "south",
    floors: int = 1,
) -> List[Dict[str, Any]]:
    """Build room specifications based on BHK count and extras."""
    bedrooms = _effective_bedroom_target(bedrooms, usable_w, usable_l)
    rooms = []

    feasible_extras, _ = _partition_feasible_extras(extras, usable_w, usable_l, bedrooms)
    extras_norm = set(feasible_extras)
    usable_area = max(1.0, usable_w * usable_l)

    if floors >= 2 and "staircase" not in extras_norm:
        extras_norm.add("staircase")

    # --- Always present ---
    rooms.append({
        "type": "living", "label": "Living Room",
        "min_w": 12, "min_h": 11, "pref_w": 14, "pref_h": 13,
        "zone": 1, "priority": 1,
    })
    rooms.append({
        "type": "dining", "label": "Dining Room",
        "min_w": 9, "min_h": 9, "pref_w": 10, "pref_h": 10,
        "zone": 1, "priority": 2,
    })
    rooms.append({
        "type": "kitchen", "label": "Kitchen",
        "min_w": 8, "min_h": 9, "pref_w": 9, "pref_h": 10,
        "zone": 2, "priority": 3,
    })
    rooms.append({
        "type": "corridor", "label": "Corridor",
        "min_w": 3.5, "min_h": usable_l, "pref_w": 4, "pref_h": usable_l,
        "zone": 2, "priority": 4,
    })
    rooms.append({
        "type": "master_bedroom", "label": "Master Bedroom",
        "min_w": 11, "min_h": 11, "pref_w": 13, "pref_h": 13,
        "zone": 3, "priority": 5,
    })
    rooms.append({
        "type": "master_bath", "label": "Master Bath",
        "min_w": 5, "min_h": 7, "pref_w": 5, "pref_h": 8,
        # Wet rooms are intentionally clustered in service core for plumbing efficiency.
        "zone": 2, "priority": 6,
    })

    # --- Additional bedrooms/bathrooms ---
    if bedrooms >= 2:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 2",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 12,
            "zone": 3, "priority": 7,
        })
        rooms.append({
            "type": "bathroom", "label": "Bathroom 2",
            "min_w": 5, "min_h": 7, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 8,
        })

    if bedrooms >= 3:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 3",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 12,
            "zone": 3, "priority": 9,
        })
        # Keep wet-room program proportional to usable area to avoid tiny strips.
        if usable_area >= 1400:
            rooms.append({
                "type": "bathroom", "label": "Bathroom 3",
                "min_w": 5, "min_h": 6, "pref_w": 5, "pref_h": 7,
                "zone": 2, "priority": 10,
            })
        if usable_area >= 1700:
            rooms.append({
                "type": "bathroom", "label": "Common Bath",
                "min_w": 5, "min_h": 5, "pref_w": 5, "pref_h": 6,
                "zone": 2, "priority": 11,
            })

    if bedrooms >= 4:
        rooms.append({
            "type": "bedroom", "label": "Bedroom 4",
            "min_w": 10, "min_h": 10, "pref_w": 11, "pref_h": 11,
            "zone": 3, "priority": 12,
        })
        if usable_area >= 1900:
            rooms.append({
                "type": "bathroom", "label": "Bathroom 4",
                "min_w": 5, "min_h": 5, "pref_w": 5, "pref_h": 6,
                "zone": 2, "priority": 13,
            })

    # --- Extras ---
    if "pooja" in extras_norm:
        rooms.append({
            "type": "pooja", "label": "Pooja Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 1, "priority": 2,
        })
    elif (
        str(facing).strip().lower() == "south"
        and usable_area >= 680
        and usable_w >= 20.5
        and usable_l >= 28.0
    ):
        # South-facing plots are compensated with a small pooja/energy buffer near entry.
        rooms.append({
            "type": "pooja", "label": "Pooja Buffer",
            "min_w": 4.5, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 1, "priority": 2,
        })
    if "study" in extras_norm:
        rooms.append({
            "type": "study", "label": "Study Room",
            "min_w": 8, "min_h": 9, "pref_w": 9, "pref_h": 10,
            "zone": 3, "priority": 8,
        })
    if "store" in extras_norm:
        rooms.append({
            "type": "store", "label": "Store Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 2, "priority": 9,
        })
    if "balcony" in extras_norm:
        rooms.append({
            "type": "balcony", "label": "Balcony",
            "min_w": 4, "min_h": 8, "pref_w": 5, "pref_h": 10,
            "zone": 1, "priority": 3,
        })
    if "garage" in extras_norm:
        rooms.append({
            "type": "garage", "label": "Garage",
            "min_w": 10, "min_h": 18, "pref_w": 11, "pref_h": 20,
            "zone": 1, "priority": 10,
        })
    if "utility" in extras_norm:
        rooms.append({
            "type": "utility", "label": "Utility",
            "min_w": 4, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 2, "priority": 9,
        })
    if "foyer" in extras_norm:
        rooms.append({
            "type": "foyer", "label": "Foyer",
            "min_w": 4, "min_h": 4, "pref_w": 5, "pref_h": 5,
            "zone": 1, "priority": 2,
        })
    if "staircase" in extras_norm:
        rooms.append({
            "type": "staircase", "label": "Staircase",
            "min_w": 6, "min_h": 8, "pref_w": 6.5, "pref_h": 9,
            "zone": 2, "priority": 9,
        })

    planned_area = sum(
        max(0.0, _to_float(r.get("pref_w"), 0.0) * _to_float(r.get("pref_h"), 0.0))
        for r in rooms
    )
    residual_area = max(0.0, usable_area - planned_area)
    if (
        residual_area >= usable_area * 0.24
        and usable_w >= 22.0
        and usable_l >= 30.0
        and bedrooms <= 2
    ):
        open_side = _clamp(usable_w * 0.30, 6.0, max(6.0, usable_w * 0.45))
        open_depth = _clamp(
            residual_area / max(1.0, open_side),
            6.0,
            min(12.0, max(6.0, usable_l * 0.30)),
        )
        rooms.append({
            "type": "open_area", "label": "Open Area",
            "min_w": 6, "min_h": 6, "pref_w": _rnd(open_side, 2), "pref_h": _rnd(open_depth, 2),
            "zone": 1, "priority": 12,
        })

    return rooms


# ─────────────────────────────────────────────────────────────
# BSP Packing v2 — smart grid, zero-overlap, proper proportions
# ─────────────────────────────────────────────────────────────
def _spec_min_h(spec: Dict[str, Any], fallback: float = 6.0) -> float:
    return max(3.0, _to_float(spec.get("min_h"), fallback))


def _spec_soft_min_h(spec: Dict[str, Any], fallback: float = 6.0) -> float:
    # Use validator-compatible lower bound for early band budgeting.
    return max(4.0, _spec_min_h(spec, fallback) * 0.8)


def _compute_program_band_heights(
    zone_rooms: Dict[int, List[Dict[str, Any]]],
    usable_l: float,
) -> tuple[float, float, float]:
    public_specs = list(zone_rooms.get(1, []))
    living_specs = [r for r in zone_rooms.get(1, []) if r.get("type") == "living"]
    living_min_h = _spec_soft_min_h(living_specs[0], 12.0) if living_specs else 9.0

    private_specs = list(zone_rooms.get(3, []))
    service_specs = [
        r for r in zone_rooms.get(2, [])
        if r.get("type") not in ("corridor", "master_bath")
    ]

    private_sum_h = sum(_spec_soft_min_h(r, 8.0) for r in private_specs)
    service_sum_h = sum(_spec_soft_min_h(r, 7.0) for r in service_specs)

    bedroom_count = sum(1 for r in private_specs if r.get("type") == "bedroom")
    private_band_min = max(8.0, bedroom_count * 7.2)
    wet_count = sum(1 for r in service_specs if r.get("type") in ("bathroom", "toilet"))
    service_other_count = sum(
        1
        for r in service_specs
        if r.get("type") not in ("bathroom", "toilet", "kitchen")
    )
    service_band_min = max(
        7.0,
        7.2 if any(r.get("type") == "kitchen" for r in service_specs) else 0.0,
        wet_count * 4.0 + service_other_count * 3.2,
    )
    front_band_floor = max(8.8, living_min_h)
    max_private_feasible = max(8.0, usable_l - front_band_floor - service_band_min)
    private_band_min = min(private_band_min, max_private_feasible)

    public_extra = max(0.0, (len(public_specs) - 2) * 0.9)
    front_target_h = max(
        living_min_h,
        min(living_min_h + 3.4 + public_extra, usable_l * 0.36),
    )
    total_h = front_target_h + service_sum_h + private_sum_h

    if total_h <= usable_l + 0.05:
        band1_h = front_target_h
        band2_h = max(7.0, service_sum_h)
        band3_h = max(8.0, private_sum_h)
    else:
        log.warning(
            "Program-derived band heights exceed usable length (front=%.2f, service=%.2f, private=%.2f, usable=%.2f). "
            "Compressing service/private bands while preserving living minimum.",
            front_target_h,
            service_sum_h,
            private_sum_h,
            usable_l,
        )
        min_front_h = front_band_floor
        alloc_h = max(15.0, usable_l - min_front_h)
        ratio_den = max(1.0, service_sum_h + private_sum_h)
        band2_h = max(7.0, alloc_h * (service_sum_h / ratio_den))
        band3_h = max(8.0, alloc_h * (private_sum_h / ratio_den))
        scale = alloc_h / max(1.0, band2_h + band3_h)
        band2_h *= scale
        band3_h *= scale
        band1_h = min_front_h

    # Final validation: enforce total <= usable length before packing bands.
    band1_h = max(front_band_floor, band1_h)
    band2_h = max(service_band_min, band2_h)
    max_band12 = usable_l - private_band_min
    if band1_h + band2_h > max_band12:
        overflow = (band1_h + band2_h) - max_band12
        reduce_band2 = min(overflow, max(0.0, band2_h - service_band_min))
        band2_h -= reduce_band2
        overflow -= reduce_band2
        if overflow > 0:
            reduce_band1 = min(overflow, max(0.0, band1_h - front_band_floor))
            band1_h -= reduce_band1

    band3_h = max(private_band_min, usable_l - band1_h - band2_h)
    if band1_h + band2_h + band3_h > usable_l:
        band3_h = max(private_band_min, band3_h - ((band1_h + band2_h + band3_h) - usable_l))

    band1_h = _rnd(band1_h, 2)
    band2_h = _rnd(band2_h, 2)
    band3_h = _rnd(max(private_band_min, usable_l - band1_h - band2_h), 2)
    drift = _rnd(usable_l - (band1_h + band2_h + band3_h), 2)
    if abs(drift) > 0.01:
        band1_h = _rnd(max(front_band_floor, band1_h + drift), 2)
        band3_h = _rnd(max(private_band_min, usable_l - band1_h - band2_h), 2)

    if band1_h + band2_h + band3_h > usable_l + 0.05:
        raise ValueError("Program-derived band heights exceed usable plot length")

    return band1_h, band2_h, band3_h


def _assign_bsp_room_ids(placed: List[Dict[str, Any]]):
    counts: Dict[str, int] = {}
    for room in placed:
        room_type = str(room.get("type", "room"))
        counts[room_type] = counts.get(room_type, 0) + 1
        room["id"] = f"{room_type}_{counts[room_type]:02d}"


def _enforce_vastu_clamps(placed: List[Dict[str, Any]], usable_w: float, band1_h: float):
    half_w = usable_w * 0.5
    for room in placed:
        room_type = str(room.get("type", ""))
        width = _to_float(room.get("width"), 0.0)
        height = _to_float(room.get("height"), 0.0)

        if room_type == "kitchen":
            min_x = max(0.0, half_w)
            max_x = max(0.0, usable_w - width)
            room["x"] = _rnd(_clamp(_to_float(room.get("x"), 0.0), min_x, max_x), 2)

        elif room_type == "master_bedroom":
            if width > half_w:
                width = _rnd(max(7.0, half_w), 2)
                room["width"] = width
            max_x = max(0.0, half_w - width)
            room["x"] = _rnd(_clamp(_to_float(room.get("x"), 0.0), 0.0, max_x), 2)

            # Keep post-clamp proportions practical; width adjustments can
            # otherwise create bowling-alley bedrooms on long plots.
            height = _to_float(room.get("height"), 0.0)
            max_h = _rnd(max(10.0, width * 2.95), 2)
            if height > max_h:
                room["height"] = max_h

        elif room_type == "pooja":
            min_x = max(0.0, half_w)
            max_x = max(0.0, usable_w - width)
            max_y = max(0.0, band1_h - height)
            room["x"] = _rnd(_clamp(_to_float(room.get("x"), 0.0), min_x, max_x), 2)
            room["y"] = _rnd(_clamp(_to_float(room.get("y"), 0.0), 0.0, max_y), 2)


def pack_rooms_bsp(room_specs: List[Dict[str, Any]],
                   usable_w: float, usable_l: float,
                   facing: str = "south") -> List[Dict[str, Any]]:
    """
    Pack rooms into three horizontal bands with zero overlaps.
    Uses smart grid layout that pairs bedrooms with baths.
    Band 1 (public):  ~28% depth — living, dining, pooja, balcony
    Band 2 (service): ~25% depth — corridor spine, kitchen, bathrooms
    Band 3 (private): ~47% depth — bedrooms with attached baths, study
    """
    # Separate rooms by zone
    zone_rooms: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for spec in room_specs:
        z = spec["zone"]
        if z in zone_rooms:
            zone_rooms[z].append(spec)

    # Sort each zone by priority
    for z in zone_rooms:
        zone_rooms[z].sort(key=lambda r: r["priority"])

    band1_h, band2_h, band3_h = _compute_program_band_heights(zone_rooms, usable_l)
    # Keep front band continuous so dining and kitchen can stay physically connected.
    entry_stub_h = 0.0
    front_room_h = _rnd(max(8.0, band1_h), 2)

    master_bath_spec = None
    service_zone_rooms = list(zone_rooms[2])

    placed: List[Dict[str, Any]] = []

    # ── Band 1: Public (living, dining, pooja, balcony) ──────
    placed.extend(_pack_band1(zone_rooms[1], 0, 0, usable_w, front_room_h, facing=facing))

    # ── Band 2: Service (corridor + kitchen + bathrooms) ─────
    band2_rooms = _pack_band2(
        service_zone_rooms,
        0,
        band1_h,
        usable_w,
        band2_h,
        corridor_extend_h=0.0,
        entry_stub_h=entry_stub_h,
    )
    placed.extend(band2_rooms)

    corridor_hint: tuple[float, float] | None = None
    for r in band2_rooms:
        if r.get("type") == "corridor":
            corridor_hint = (_to_float(r.get("x"), 0.0), _to_float(r.get("width"), 0.0))
            break

    # ── Band 3: Private (bedrooms + attached baths + study) ──
    placed.extend(
        _pack_band3(
            zone_rooms[3],
            0,
            band1_h + band2_h,
            usable_w,
            band3_h,
            corridor_hint,
            master_bath_spec=master_bath_spec,
        )
    )

    _enforce_vastu_clamps(placed, usable_w, band1_h)

    _assign_bsp_room_ids(placed)

    # Final overlap check
    _verify_no_overlaps(placed)
    return placed


def _pack_band1(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float, facing: str = "south") -> List[Dict[str, Any]]:
    """
    Band 1: Public zone.
    Layout: Living room (60% width) | Dining + small rooms stacked (40% width)
    """
    if not rooms:
        return []

    placed = []

    # Separate large rooms from small utility rooms
    living = None
    dining = None
    small_rooms = []

    for r in rooms:
        if r["type"] == "living":
            living = r
        elif r["type"] == "dining":
            dining = r
        else:
            small_rooms.append(r)

    # Calculate proportions
    has_small = len(small_rooms) > 0

    if living and dining:
        if has_small:
            # 3-column: Living (50%) | Dining (30%) | Small rooms stacked (20%)
            living_min_w = 8.8
            dining_min_w = 6.4
            small_min_w = 3.8
            living_w = _rnd(max(living_min_w, bw * 0.50), 2)
            dining_w = _rnd(max(dining_min_w, bw * 0.30), 2)
            small_w = _rnd(bw - living_w - dining_w, 2)

            if small_w < small_min_w:
                deficit = _rnd(small_min_w - small_w, 2)
                reduce_living = min(deficit, max(0.0, living_w - living_min_w))
                living_w = _rnd(living_w - reduce_living, 2)
                deficit = _rnd(deficit - reduce_living, 2)
                if deficit > 0:
                    reduce_dining = min(deficit, max(0.0, dining_w - dining_min_w))
                    dining_w = _rnd(dining_w - reduce_dining, 2)
                small_w = _rnd(max(0.0, bw - living_w - dining_w), 2)
        else:
            # 2-column: Living (58%) | Dining (42%)
            living_w = _rnd(bw * 0.58, 2)
            dining_w = _rnd(bw - living_w, 2)
            small_w = 0

        # Keep public-room frontage aligned with road side when possible.
        # For east-facing plots, living is placed on the east edge.
        place_living_east = str(facing or "").strip().lower() == "east"
        if place_living_east:
            dining_x = _rnd(bx, 2)
            living_x = _rnd(bx + dining_w, 2)
        else:
            living_x = _rnd(bx, 2)
            dining_x = _rnd(bx + living_w, 2)

        placed.append({
            "type": living["type"], "label": living["label"],
            "x": living_x, "y": _rnd(by, 2),
            "width": _rnd(living_w, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })
        placed.append({
            "type": dining["type"], "label": dining["label"],
            "x": dining_x, "y": _rnd(by, 2),
            "width": _rnd(dining_w, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })

        if has_small and small_w >= 3.8:
            # Stack small rooms vertically in the remaining column
            current_y = by
            per_h = _rnd(bh / len(small_rooms), 2)
            for i, sr in enumerate(small_rooms):
                rh = per_h if i < len(small_rooms) - 1 else _rnd(by + bh - current_y, 2)
                min_rw = max(3.2, _to_float(sr.get("min_w"), 4.0) * 0.8)
                min_rh = max(3.2, _to_float(sr.get("min_h"), 4.0) * 0.8)
                if small_w < min_rw or rh < min_rh:
                    current_y = _rnd(current_y + rh, 2)
                    continue
                placed.append({
                    "type": sr["type"], "label": sr["label"],
                    "x": _rnd(bx + living_w + dining_w, 2),
                    "y": _rnd(current_y, 2),
                    "width": _rnd(small_w, 2), "height": _rnd(rh, 2),
                    "zone": 1, "band": 1,
                })
                current_y = _rnd(current_y + rh, 2)
        elif has_small:
            # Not enough width — tuck small rooms inside dining area (bottom portion)
            dining_room = next((r for r in placed if r.get("type") == "dining"), placed[-1])
            min_small_h = max([
                max(3.8, _spec_soft_min_h(sr, 5.0))
                for sr in small_rooms
            ], default=3.8)
            tuck_h = _rnd(max(min_small_h, bh * 0.45), 2)
            tuck_h = _rnd(min(tuck_h, max(3.8, bh - 4.0)), 2)
            dining_room["height"] = _rnd(bh - tuck_h, 2)

            current_x = dining_room["x"]
            tuck_w = _rnd(dining_w / len(small_rooms), 2)
            for i, sr in enumerate(small_rooms):
                rw = tuck_w if i < len(small_rooms) - 1 else _rnd(dining_room["x"] + dining_w - current_x, 2)
                min_rw = max(3.2, _to_float(sr.get("min_w"), 4.0) * 0.8)
                min_rh = max(3.2, _to_float(sr.get("min_h"), 4.0) * 0.8)
                if rw < min_rw or tuck_h < min_rh:
                    current_x = _rnd(current_x + rw, 2)
                    continue
                placed.append({
                    "type": sr["type"], "label": sr["label"],
                    "x": _rnd(current_x, 2),
                    "y": _rnd(by + bh - tuck_h, 2),
                    "width": _rnd(rw, 2), "height": _rnd(tuck_h, 2),
                    "zone": 1, "band": 1,
                })
                current_x = _rnd(current_x + rw, 2)

    elif living:
        placed.append({
            "type": living["type"], "label": living["label"],
            "x": _rnd(bx, 2), "y": _rnd(by, 2),
            "width": _rnd(bw, 2), "height": _rnd(bh, 2),
            "zone": 1, "band": 1,
        })
    else:
        # Fallback: strip pack everything
        placed.extend(_pack_strip_safe(rooms, bx, by, bw, bh, 1))

    return placed


def _pack_band2(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float,
                corridor_extend_h: float = 0.0,
                entry_stub_h: float = 0.0) -> List[Dict[str, Any]]:
    """
    Band 2: Service zone.
    Layout: corridor spine + east kitchen + boundary-flush common bathrooms.
    """
    if not rooms:
        return []

    placed: List[Dict[str, Any]] = []
    kitchen = None
    corridor_spec = None
    wet_rooms: List[Dict[str, Any]] = []
    service_other: List[Dict[str, Any]] = []

    for r in rooms:
        room_type = r.get("type")
        if room_type == "corridor":
            corridor_spec = r
        elif room_type == "kitchen":
            kitchen = r
        elif room_type in ("bathroom", "toilet"):
            wet_rooms.append(r)
        elif room_type == "master_bath":
            # Master bath is placed in private band next to master bedroom.
            continue
        else:
            service_other.append(r)

    corridor_w = 3.5 if corridor_spec else 0.0
    corridor_w = _rnd(min(corridor_w, max(0.0, bw - 12.0)), 2)

    if corridor_spec and corridor_w >= 3.0:
        left_w = _rnd(max(5.0, (bw - corridor_w) * 0.42), 2)
        right_w = _rnd(max(6.0, bw - corridor_w - left_w), 2)
        corridor_x = _rnd(bx + left_w, 2)
        wet_x = _rnd(bx, 2)
        wet_w = _rnd(max(5.0, corridor_x - wet_x), 2)
        if wet_w > 10.5:
            wet_w = 10.5
            corridor_x = _rnd(wet_x + wet_w, 2)
        kitchen_w = _rnd(max(6.0, right_w), 2)
        kitchen_x = _rnd(bx + bw - kitchen_w, 2)
    else:
        corridor_w = 0.0
        wet_x = _rnd(bx, 2)
        wet_w = _rnd(min(max(5.0, bw * 0.36), max(5.0, bw - 8.0)), 2)
        kitchen_w = _rnd(max(6.0, bw - wet_w), 2)
        kitchen_x = _rnd(bx + bw - kitchen_w, 2)
        corridor_x = _rnd(kitchen_x - 3.5, 2)

    if corridor_spec and corridor_w >= 3.0:
        front_extra = max(0.0, entry_stub_h)
        rear_extra = max(0.0, corridor_extend_h)
        corridor_y = _rnd(max(0.0, by - front_extra), 2)
        corridor_h = _rnd(max(bh, bh + front_extra + rear_extra), 2)
        placed.append({
            "type": "corridor", "label": "Corridor",
            "x": corridor_x, "y": corridor_y,
            "width": _rnd(corridor_w, 2), "height": corridor_h,
            "zone": 2, "band": 2,
        })

    east_stack: List[Dict[str, Any]] = []
    central_stack: List[Dict[str, Any]] = []
    for r in service_other:
        if r["type"] in ("utility", "store"):
            east_stack.append(r)
        else:
            central_stack.append(r)

    if kitchen:
        reserve_h = 0.0
        if east_stack:
            reserve_h = _rnd(min(max(3.5, bh * 0.28), max(3.5, bh - 4.0)), 2)
        kitchen_h = _rnd(max(4.0, bh - reserve_h), 2)

        placed.append({
            "type": kitchen["type"], "label": kitchen["label"],
            "x": kitchen_x, "y": _rnd(by, 2),
            "width": _rnd(kitchen_w, 2), "height": kitchen_h,
            "zone": 2, "band": 2,
        })

        if reserve_h > 0 and east_stack:
            y = _rnd(by + kitchen_h, 2)
            per_h = _rnd(reserve_h / len(east_stack), 2)
            for idx, room in enumerate(east_stack):
                rh = per_h if idx < len(east_stack) - 1 else _rnd(by + bh - y, 2)
                placed.append({
                    "type": room["type"], "label": room["label"],
                    "x": kitchen_x, "y": _rnd(y, 2),
                    "width": _rnd(kitchen_w, 2), "height": _rnd(max(3.5, rh), 2),
                    "zone": 2, "band": 2,
                })
                y = _rnd(y + rh, 2)

    # Common bathrooms are anchored at the band2/band3 boundary for bedroom access.
    boundary_baths = [r for r in wet_rooms if r.get("type") in ("bathroom", "toilet")]
    stack_rooms = boundary_baths + central_stack
    if stack_rooms and wet_w > 0:
        def _band2_stack_min_h(spec: Dict[str, Any]) -> float:
            room_type = str(spec.get("type", ""))
            if room_type in ("bathroom", "toilet", "master_bath"):
                return 4.0
            return max(3.2, _spec_soft_min_h(spec, 5.0) * 0.7)

        current_top = _rnd(by + bh, 2)
        per_h = _rnd(bh / len(stack_rooms), 2)
        for idx, room in enumerate(stack_rooms):
            available_h = _rnd(max(0.0, current_top - by), 2)
            if available_h <= 0.1:
                break

            min_h = _band2_stack_min_h(room)

            if idx < len(stack_rooms) - 1:
                rh = _rnd(max(min_h, per_h), 2)
                reserve_h = _rnd(
                    sum(_band2_stack_min_h(rem) for rem in stack_rooms[idx + 1:]),
                    2,
                )
                max_h = _rnd(max(3.0, available_h - reserve_h), 2)
                rh = _rnd(min(rh, max_h), 2)
            else:
                rh = _rnd(max(min_h, available_h), 2)

            if rh > available_h:
                rh = available_h

            y = _rnd(max(by, current_top - rh), 2)
            placed.append({
                "type": room["type"], "label": room["label"],
                "x": wet_x, "y": y,
                "width": _rnd(wet_w, 2), "height": _rnd(rh, 2),
                "zone": 2, "band": 2,
            })
            current_top = y

    return placed


def _pack_private_single_column(
    rooms: List[Dict[str, Any]],
    bx: float,
    by: float,
    bw: float,
    bh: float,
    master_bath_spec: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    stack = list(rooms)
    if master_bath_spec is not None:
        stack = [master_bath_spec] + stack
    if not stack:
        return []

    placed: List[Dict[str, Any]] = []
    current_y = by
    per_h = _rnd(bh / len(stack), 2)
    for idx, room in enumerate(stack):
        if idx < len(stack) - 1:
            rh = _rnd(max(4.0, min(per_h, by + bh - current_y - (len(stack) - idx - 1) * 4.0)), 2)
        else:
            rh = _rnd(max(4.0, by + bh - current_y), 2)

        placed.append({
            "type": room["type"], "label": room["label"],
            "x": _rnd(bx, 2), "y": _rnd(current_y, 2),
            "width": _rnd(bw, 2), "height": _rnd(rh, 2),
            "zone": 3, "band": 3,
        })
        current_y = _rnd(current_y + rh, 2)

    return placed


def _pack_band3(rooms: List[Dict[str, Any]], bx: float, by: float,
                bw: float, bh: float,
                corridor_hint: tuple[float, float] | None = None,
                master_bath_spec: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Band 3: Private zone.
    Master bedroom stays southwest with attached master bath on its south wall.
    """
    if not rooms and master_bath_spec is None:
        return []

    placed: List[Dict[str, Any]] = []

    master = None
    other_beds: List[Dict[str, Any]] = []
    studies: List[Dict[str, Any]] = []
    misc_private: List[Dict[str, Any]] = []

    for r in rooms:
        if r["type"] == "master_bedroom" and master is None:
            master = r
        elif r["type"] == "bedroom":
            other_beds.append(r)
        elif r["type"] == "study":
            studies.append(r)
        else:
            misc_private.append(r)

    if master is None:
        for r in rooms:
            if r["type"] == "bedroom":
                master = r
                break
        if master is not None:
            other_beds = [r for r in other_beds if r is not master]

    if master is None:
        return _pack_private_single_column(rooms, bx, by, bw, bh, master_bath_spec)

    use_corridor_channel = False
    corridor_x = 0.0
    corridor_w = 0.0
    if corridor_hint is not None:
        corridor_x = _to_float(corridor_hint[0], 0.0)
        corridor_w = _to_float(corridor_hint[1], 0.0)
        if corridor_w >= 3.0 and corridor_x > bx + 6.0 and corridor_x + corridor_w < bx + bw - 6.0:
            use_corridor_channel = True

    master_min_w = max(8.0, min(_to_float(master.get("min_w"), 10.0), bw * 0.62))
    other_bed_min_w = max([
        max(7.0, min(_to_float(r.get("min_w"), 9.0), bw * 0.48))
        for r in other_beds
    ], default=0.0)

    compact_split = False

    if use_corridor_channel:
        target_master_w = corridor_x - bx
        max_master_w = bw - other_bed_min_w - corridor_w
    else:
        target_master_w = bw * 0.55
        max_master_w = bw - other_bed_min_w

    if other_beds and max_master_w < master_min_w:
        if use_corridor_channel:
            # On narrow plots, relax bedroom width targets before dropping the
            # corridor channel. This keeps private rooms corridor-connected.
            relaxed_master_min = max(8.5, min(master_min_w, bw * 0.42))
            relaxed_other_min = max(8.0, min(other_bed_min_w or 8.0, bw * 0.36))
            relaxed_max_master = bw - relaxed_other_min - corridor_w
            if relaxed_max_master >= relaxed_master_min:
                master_min_w = _rnd(relaxed_master_min, 2)
                other_bed_min_w = _rnd(relaxed_other_min, 2)
                max_master_w = _rnd(relaxed_max_master, 2)

    if other_beds and max_master_w < master_min_w:
        if bw >= 14.0:
            log.warning(
                "Private-band width is tight for standard bedroom mins. "
                "Using compact two-column private split.",
            )
            compact_split = True
            use_corridor_channel = False
            target_master_w = bw * 0.5
            master_min_w = max(7.5, min(master_min_w, bw * 0.5))
            max_master_w = bw - max(6.5, bw * 0.35)
        else:
            log.warning(
                "Private-band width is too narrow for master+secondary bedrooms. "
                "Switching to full-width single-column stacking.",
            )
            fallback_rooms = [master] + other_beds + studies + misc_private
            return _pack_private_single_column(fallback_rooms, bx, by, bw, bh, master_bath_spec)

    if other_beds:
        master_w = _rnd(_clamp(target_master_w, master_min_w, max_master_w), 2)
    else:
        master_w = _rnd(_clamp(target_master_w, master_min_w, bw), 2)

    if use_corridor_channel:
        # Keep private-band channel aligned with service-band corridor so the
        # corridor is one continuous logical spine.
        hinted_x = _to_float(corridor_hint[0], bx + master_w) if corridor_hint is not None else (bx + master_w)
        min_corridor_x = _rnd(bx + max(6.0, master_min_w), 2)
        max_corridor_x = _rnd(max(min_corridor_x, bx + bw - corridor_w - max(6.0, other_bed_min_w)), 2)

        if max_corridor_x <= min_corridor_x + 0.05:
            use_corridor_channel = False
        else:
            corridor_x = _rnd(_clamp(hinted_x, min_corridor_x, max_corridor_x), 2)
            master_w = _rnd(max(master_min_w, corridor_x - bx), 2)
            east_x = _rnd(corridor_x + corridor_w, 2)
            east_w = _rnd(max(0.0, bx + bw - east_x), 2)
            if other_beds and east_w < other_bed_min_w:
                use_corridor_channel = False

    if not use_corridor_channel:
        east_x = _rnd(bx + master_w, 2)
        east_w = _rnd(max(0.0, bw - master_w), 2)

    if compact_split:
        east_w = _rnd(max(6.5, east_w), 2)
        master_w = _rnd(max(7.5, bw - east_w), 2)
        east_x = _rnd(bx + master_w, 2)

    master_min_h = max(10.0, _to_float(master.get("min_h"), 11.0))
    study_h = 0.0
    if studies:
        study_h = _rnd(max(6.0, min(bh * 0.30, max(6.0, bh - master_min_h - 4.0))), 2)

    bath_h = 0.0
    if master_bath_spec is not None:
        bath_h = _rnd(max(5.0, min(_spec_min_h(master_bath_spec, 7.0), max(5.0, bh * 0.32))), 2)

    master_h = _rnd(max(6.5, bh - study_h - bath_h), 2)
    if master_h < master_min_h:
        deficit = _rnd(master_min_h - master_h, 2)
        if study_h > 0:
            cut = min(deficit, max(0.0, study_h - 4.0))
            study_h = _rnd(study_h - cut, 2)
            deficit = _rnd(deficit - cut, 2)
        if deficit > 0 and bath_h > 0:
            cut = min(deficit, max(0.0, bath_h - 5.0))
            bath_h = _rnd(bath_h - cut, 2)
            deficit = _rnd(deficit - cut, 2)
        # On compact plots, do not force impossible bedroom heights that spill
        # outside the private band; fit to remaining height deterministically.
        master_h = _rnd(max(6.5, bh - study_h - bath_h), 2)

    # For 1BHK elongated plots, cap master height to avoid bowling-alley shape.
    if not other_beds and not studies and not misc_private:
        max_master_h = _rnd(max(10.0, master_w * 2.85), 2)
        if master_h > max_master_h:
            master_h = max_master_h

    if master_bath_spec is not None and bath_h > 0:
        placed.append({
            "type": master_bath_spec["type"], "label": master_bath_spec["label"],
            "x": _rnd(bx, 2), "y": _rnd(by, 2),
            "width": _rnd(master_w, 2), "height": _rnd(bath_h, 2),
            "zone": 3, "band": 3,
        })

    master_y = _rnd(by + bath_h, 2)
    placed.append({
        "type": master["type"], "label": master["label"],
        "x": _rnd(bx, 2), "y": master_y,
        "width": _rnd(master_w, 2), "height": _rnd(master_h, 2),
        "zone": 3, "band": 3,
    })

    if studies and study_h > 0:
        placed.append({
            "type": studies[0]["type"], "label": studies[0]["label"],
            "x": _rnd(bx, 2), "y": _rnd(master_y + master_h, 2),
            "width": _rnd(master_w, 2), "height": _rnd(study_h, 2),
            "zone": 3, "band": 3,
        })

    east_rooms = other_beds + misc_private
    if east_w > 0 and east_rooms:
        current_y = _rnd(by, 2)
        east_h = _rnd(max(8.0, bh), 2)
        per_h = _rnd(east_h / len(east_rooms), 2)
        for idx, room in enumerate(east_rooms):
            rh = per_h if idx < len(east_rooms) - 1 else _rnd(by + bh - current_y, 2)
            placed.append({
                "type": room["type"], "label": room["label"],
                "x": east_x, "y": _rnd(current_y, 2),
                "width": _rnd(max(4.0, east_w), 2), "height": _rnd(max(4.0, rh), 2),
                "zone": 3, "band": 3,
            })
            current_y = _rnd(current_y + rh, 2)

    return placed


def _pack_strip_safe(rooms: List[Dict[str, Any]], bx: float, by: float,
                     bw: float, bh: float, band_num: int) -> List[Dict[str, Any]]:
    """Fallback strip packing that respects min widths and never goes negative."""
    if not rooms:
        return []

    total_min = sum(r["min_w"] for r in rooms)
    placed = []
    current_x = bx

    for i, room in enumerate(rooms):
        if i == len(rooms) - 1:
            rw = _rnd(bx + bw - current_x, 2)
        else:
            if total_min <= bw:
                # Enough space — use proportional
                ratio = room["pref_w"] / sum(r["pref_w"] for r in rooms)
                rw = _rnd(bw * ratio, 2)
                rw = max(rw, room["min_w"])
            else:
                # Not enough — give min widths
                rw = room["min_w"]

        rw = max(rw, 3.0)  # Never below 3ft
        if current_x + rw > bx + bw:
            rw = _rnd(bx + bw - current_x, 2)
        if rw < 2.0:
            continue  # Skip rooms that can't fit

        placed.append({
            "type": room["type"], "label": room["label"],
            "x": _rnd(current_x, 2), "y": _rnd(by, 2),
            "width": _rnd(rw, 2), "height": _rnd(bh, 2),
            "zone": room.get("zone", 1), "band": band_num,
        })
        current_x = _rnd(current_x + rw, 2)

    return placed


def _verify_no_overlaps(placed: List[Dict[str, Any]]):
    """Check all placed rooms for overlaps. Log warnings if found."""
    for i, a in enumerate(placed):
        for j in range(i + 1, len(placed)):
            b = placed[j]
            if _rects_overlap(a, b):
                log.warning(
                    "Overlap detected: %s (%.1f,%.1f,%.1f,%.1f) vs %s (%.1f,%.1f,%.1f,%.1f)",
                    a["label"], a["x"], a["y"], a["width"], a["height"],
                    b["label"], b["x"], b["y"], b["width"], b["height"],
                )
                b["width"] = _rnd(b["width"] - 0.2, 2)


def _compute_fallback_vastu_score(
    placed: List[Dict[str, Any]],
    usable_w: float,
    usable_l: float,
) -> float:
    """Score deterministic fallback layouts using key vastu placements."""
    base = 68.0

    def _first(room_type: str) -> Dict[str, Any] | None:
        for room in placed:
            if room.get("type") == room_type:
                return room
        return None

    living = _first("living")
    kitchen = _first("kitchen")
    master = _first("master_bedroom")
    pooja = _first("pooja")
    corridor = _first("corridor")

    rear_min = usable_l * 0.45
    front_max = usable_l * 0.35

    if living and living["y"] + living["height"] <= front_max + 0.5:
        base += 3.0
    if kitchen and kitchen["x"] + kitchen["width"] >= usable_w * 0.5 and kitchen["y"] >= usable_l * 0.22:
        base += 4.0
    if master and master["x"] + master["width"] <= usable_w * 0.6 and master["y"] >= rear_min:
        base += 4.0
    if pooja and pooja["x"] >= usable_w * 0.5 and pooja["y"] + pooja["height"] <= front_max + 0.5:
        base += 3.0
    if corridor and corridor["width"] >= 3.5:
        base += 2.0

    return _rnd(_clamp(base, 60.0, 88.0), 1)


def _rects_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check if two rectangles overlap (with 0.1ft tolerance)."""
    eps = 0.1
    return (
        a["x"] < b["x"] + b["width"] - eps and
        a["x"] + a["width"] > b["x"] + eps and
        a["y"] < b["y"] + b["height"] - eps and
        a["y"] + a["height"] > b["y"] + eps
    )


# ─────────────────────────────────────────────────────────────
# Door & Window placement
# ─────────────────────────────────────────────────────────────
def _door_limit_for_room(room_type: str) -> int:
    if room_type in ("corridor", "living"):
        return 12 if room_type == "corridor" else 7
    if room_type in ("dining", "kitchen"):
        return 4
    if room_type in ("master_bedroom", "bedroom", "study", "staircase"):
        return 3
    if room_type in ("master_bath", "bathroom", "toilet"):
        return 2
    return 2


def _door_priority_for_pair(type_a: str, type_b: str) -> int:
    pair = tuple(sorted((str(type_a), str(type_b))))

    # Skip low-value or architecturally incorrect direct links.
    wet = {"master_bath", "bathroom", "toilet"}
    if pair[0] in wet and pair[1] in wet:
        return -1

    weights: Dict[tuple[str, str], int] = {
        ("corridor", "living"): 140,
        ("corridor", "dining"): 132,
        ("corridor", "kitchen"): 130,
        ("corridor", "master_bedroom"): 128,
        ("corridor", "bedroom"): 126,
        ("corridor", "study"): 124,
        ("corridor", "staircase"): 122,
        ("corridor", "bathroom"): 120,
        ("corridor", "toilet"): 118,
        ("corridor", "master_bath"): 118,
        ("living", "dining"): 116,
        ("living", "kitchen"): 112,
        ("living", "foyer"): 108,
        ("living", "balcony"): 106,
        ("living", "open_area"): 104,
        ("living", "garage"): 102,
        ("kitchen", "utility"): 100,
        ("kitchen", "store"): 96,
        ("master_bath", "master_bedroom"): 112,
        ("bathroom", "bedroom"): 102,
        ("bedroom", "toilet"): 98,
    }
    if pair in weights:
        return weights[pair]

    # Keep generic pairs possible but lower priority.
    if "open_area" in pair and pair != ("living", "open_area"):
        return -1
    return 72


def _room_touches_wall(
    room: Dict[str, Any],
    wall: str,
    usable_w: float,
    usable_l: float,
    eps: float = 0.25,
) -> bool:
    if wall == "north":
        return abs(_to_float(room.get("y"), 0.0) + _to_float(room.get("height"), 0.0) - usable_l) <= eps
    if wall == "south":
        return _to_float(room.get("y"), 0.0) <= eps
    if wall == "east":
        return abs(_to_float(room.get("x"), 0.0) + _to_float(room.get("width"), 0.0) - usable_w) <= eps
    return _to_float(room.get("x"), 0.0) <= eps


def _choose_entry_room(
    placed_rooms: List[Dict[str, Any]],
    facing: str,
    usable_w: float,
    usable_l: float,
) -> Dict[str, Any] | None:
    if not placed_rooms:
        return None

    preferred = str(facing or "south").strip().lower()
    if preferred not in ("north", "south", "east", "west"):
        preferred = "south"

    def _room_area(room: Dict[str, Any]) -> float:
        return max(0.0, _to_float(room.get("width"), 0.0) * _to_float(room.get("height"), 0.0))

    for room_type in ("living", "foyer", "dining"):
        candidates = [
            r for r in placed_rooms
            if str(r.get("type", "")) == room_type and _room_touches_wall(r, preferred, usable_w, usable_l)
        ]
        if candidates:
            return sorted(candidates, key=_room_area, reverse=True)[0]

    boundary_candidates = [
        r for r in placed_rooms
        if _room_touches_wall(r, preferred, usable_w, usable_l)
    ]
    if boundary_candidates:
        return sorted(boundary_candidates, key=_room_area, reverse=True)[0]

    living = next((r for r in placed_rooms if str(r.get("type", "")) == "living"), None)
    return living if living is not None else placed_rooms[0]


def _choose_entry_wall(
    room: Dict[str, Any],
    facing: str,
    usable_w: float,
    usable_l: float,
) -> tuple[str, float, float]:
    preferred = str(facing or "south").lower()
    wall_order = [preferred, "east", "north", "south", "west"]
    seen: set[str] = set()
    ordered_walls: list[str] = []
    for wall in wall_order:
        if wall in ("north", "south", "east", "west") and wall not in seen:
            ordered_walls.append(wall)
            seen.add(wall)

    selected = next((wall for wall in ordered_walls if _room_touches_wall(room, wall, usable_w, usable_l)), "east")

    if selected in ("north", "south"):
        x = _to_float(room.get("x"), 0.0) + _to_float(room.get("width"), 0.0) * 0.5
        y = _to_float(room.get("y"), 0.0) + (_to_float(room.get("height"), 0.0) if selected == "north" else 0.0)
    else:
        x = _to_float(room.get("x"), 0.0) + (_to_float(room.get("width"), 0.0) if selected == "east" else 0.0)
        y = _to_float(room.get("y"), 0.0) + _to_float(room.get("height"), 0.0) * 0.5

    return selected, _rnd(x, 2), _rnd(y, 2)


def add_doors_and_windows(placed_rooms: List[Dict[str, Any]],
                          usable_w: float, usable_l: float,
                          facing: str) -> tuple[list[DoorData], list[WindowData]]:
    """Add doors between adjacent rooms and windows on exterior walls."""
    doors: List[DoorData] = []
    windows: List[WindowData] = []
    door_count: int = 0
    win_count: int = 0
    room_door_counts: Dict[str, int] = {}

    # Main entrance is placed on a road-facing boundary room.
    entry_room = _choose_entry_room(placed_rooms, facing, usable_w, usable_l)
    if entry_room:
        door_count = door_count + 1
        entry_room_id = str(entry_room.get("id", "entry_room"))
        main_wall, main_x, main_y = _choose_entry_wall(entry_room, facing, usable_w, usable_l)
        entry_span = _to_float(entry_room.get("height"), 10.0) if main_wall in ("east", "west") else _to_float(entry_room.get("width"), 10.0)
        main_width = _rnd(_clamp(min(4.0, entry_span * 0.35), 3.0, 4.0), 2)
        doors.append(DoorData(
            id=f"door_{door_count:02d}", type="main",
            room_id=entry_room_id, wall=main_wall,
            x=main_x,
            y=main_y,
            width=main_width,
        ))
        room_door_counts[entry_room_id] = room_door_counts.get(entry_room_id, 0) + 1

    # Interior doors between adjacent rooms.
    # Candidate selection is deterministic and scored by architectural connectivity.
    candidates: List[Dict[str, Any]] = []
    for i, a in enumerate(placed_rooms):
        for j in range(i + 1, len(placed_rooms)):
            b = placed_rooms[j]
            shared = _shared_wall(a, b)
            if not shared:
                continue

            room_a = str(a.get("id", f"{a.get('type', 'room')}_{i+1:02d}"))
            room_b = str(b.get("id", f"{b.get('type', 'room')}_{j+1:02d}"))
            type_a = str(a.get("type", "room"))
            type_b = str(b.get("type", "room"))
            priority = _door_priority_for_pair(type_a, type_b)
            if priority < 0:
                continue

            wall_side, sx, sy, shared_len = shared
            if shared_len < 2.7:
                continue

            candidates.append({
                "room_a": room_a,
                "room_b": room_b,
                "type_a": type_a,
                "type_b": type_b,
                "wall": wall_side,
                "x": _rnd(sx, 2),
                "y": _rnd(sy, 2),
                "shared_len": _rnd(shared_len, 2),
                "priority": priority,
            })

    candidates.sort(
        key=lambda c: (
            -c["priority"],
            -c["shared_len"],
            c["room_a"],
            c["room_b"],
        )
    )

    added_pairs: set[tuple[str, str]] = set()

    def _try_add_candidate(cand: Dict[str, Any], force: bool = False) -> bool:
        nonlocal door_count

        room_a = str(cand["room_a"])
        room_b = str(cand["room_b"])
        pair_key = tuple(sorted((room_a, room_b)))
        if pair_key in added_pairs:
            return False

        limit_a = _door_limit_for_room(str(cand["type_a"]))
        limit_b = _door_limit_for_room(str(cand["type_b"]))
        count_a = room_door_counts.get(room_a, 0)
        count_b = room_door_counts.get(room_b, 0)
        if force:
            # Force pass is only for disconnected rooms; allow mild overrun.
            if count_a >= limit_a + 2:
                return False
            if count_b >= limit_b + 2:
                return False
        else:
            if count_a >= limit_a:
                return False
            if count_b >= limit_b:
                return False

        door_width = _rnd(_clamp(min(3.4, cand["shared_len"] - 0.6), 2.4, 3.2), 2)
        if door_width < 2.4:
            return False

        door_count = door_count + 1
        doors.append(DoorData(
            id=f"door_{door_count:02d}", type="interior",
            room_id=room_a,
            wall=str(cand["wall"]),
            x=_rnd(_to_float(cand["x"], 0.0), 2),
            y=_rnd(_to_float(cand["y"], 0.0), 2),
            width=door_width,
        ))

        room_door_counts[room_a] = room_door_counts.get(room_a, 0) + 1
        room_door_counts[room_b] = room_door_counts.get(room_b, 0) + 1
        added_pairs.add(pair_key)
        return True

    # First pass: add best architectural connections.
    for cand in candidates:
        _try_add_candidate(cand)

    # Second pass: guarantee each important room gets at least one access door.
    must_connect = {
        str(r.get("id", f"room_{idx+1:02d}"))
        for idx, r in enumerate(placed_rooms)
        if str(r.get("type", "")) not in ("open_area",)
    }
    for _ in range(2):
        unconnected_set = {
            rid for rid in must_connect
            if room_door_counts.get(rid, 0) <= 0
        }
        if not unconnected_set:
            break

        progress = False
        for cand in candidates:
            room_a = str(cand["room_a"])
            room_b = str(cand["room_b"])
            if room_a not in unconnected_set and room_b not in unconnected_set:
                continue
            if _try_add_candidate(cand, force=True):
                progress = True

        if not progress:
            break

    unconnected = [rid for rid in must_connect if room_door_counts.get(rid, 0) <= 0]
    if unconnected:
        log.warning("rooms without interior access door: %s", ", ".join(sorted(unconnected)))

    # Windows on exterior walls
    for idx, r in enumerate(placed_rooms):
        if r["type"] in ("corridor", "store"):
            continue  # No windows for corridor or store

        room_id = str(r.get("id", f"{r.get('type', 'room')}_{idx+1:02d}"))

        eps = 0.3
        # South wall (y == 0 means touching front setback boundary)
        if r["y"] <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=room_id,
                wall="south",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # North wall (touching rear boundary)
        if abs(r["y"] + r["height"] - usable_l) <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=room_id,
                wall="north",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"] + r["height"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # West wall (x == 0 means touching left setback boundary)
        if r["x"] <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=room_id,
                wall="west",
                x=r["x"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))
        # East wall (touching right boundary)
        if abs(r["x"] + r["width"] - usable_w) <= eps:
            win_count = win_count + 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=room_id,
                wall="east",
                x=r["x"] + r["width"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))

    # Kitchen must have east-wall ventilation.
    for idx, r in enumerate(placed_rooms):
        if r.get("type") != "kitchen":
            continue
        room_id = str(r.get("id", f"kitchen_{idx+1:02d}"))
        has_east = any(w.room_id == room_id and w.wall == "east" for w in windows)
        if has_east:
            continue
        win_count = win_count + 1
        windows.append(WindowData(
            id=f"win_{win_count:02d}", room_id=room_id,
            wall="east",
            x=r["x"] + r["width"],
            y=r["y"] + r["height"] * 0.5,
            width=min(4.0, r["height"] * 0.35),
        ))

    # Garage requires front ventilation window when missing.
    for idx, r in enumerate(placed_rooms):
        if r.get("type") != "garage":
            continue
        room_id = str(r.get("id", f"garage_{idx+1:02d}"))
        has_south = any(w.room_id == room_id and w.wall == "south" for w in windows)
        if has_south:
            continue
        win_count = win_count + 1
        windows.append(WindowData(
            id=f"win_{win_count:02d}", room_id=room_id,
            wall="south",
            x=r["x"] + r["width"] * 0.5,
            y=r["y"],
            width=min(4.0, r["width"] * 0.4),
        ))

    return doors, windows


def _shared_wall(a: Dict[str, Any], b: Dict[str, Any]) -> tuple[str, float, float, float] | None:
    """
    Check if rooms a and b share a wall segment.
    Returns (wall_side, door_x, door_y) or None.
    """
    eps = 0.3
    # a's right edge == b's left edge (vertical wall between them)
    if abs(a["x"] + a["width"] - b["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy >= 2.6:
            span = ey - oy
            mid_y = oy + span * 0.5
            return "east", a["x"] + a["width"], mid_y, span

    # b's right edge == a's left edge
    if abs(b["x"] + b["width"] - a["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy >= 2.6:
            span = ey - oy
            mid_y = oy + span * 0.5
            return "west", a["x"], mid_y, span

    # a's top edge == b's bottom edge (horizontal wall)
    if abs(a["y"] + a["height"] - b["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox >= 2.6:
            span = ex - ox
            mid_x = ox + span * 0.5
            return "north", mid_x, a["y"] + a["height"], span

    # b's top edge == a's bottom edge
    if abs(b["y"] + b["height"] - a["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox >= 2.6:
            span = ex - ox
            mid_x = ox + span * 0.5
            return "south", mid_x, a["y"], span

    return None


# ─────────────────────────────────────────────────────────────
# Emergency local-only fallback
# ─────────────────────────────────────────────────────────────
async def generate_architect_reasoning(req: PlanRequest) -> dict[str, Any]:
    """Return structured pre-plot architect reasoning without generating a full plan."""
    uw = _rnd(req.plot_width - SETBACKS["left"] - SETBACKS["right"], 2)
    ul = _rnd(req.plot_length - SETBACKS["front"] - SETBACKS["rear"], 2)
    trace, reasoning = await _build_preplot_reasoning_trace(req, uw, ul)
    payload = dict(reasoning)
    payload["reasoning_trace"] = trace
    return payload


async def generate_plan_emergency_local(req: PlanRequest) -> PlanResponse:
    """Always generate a local deterministic plan without LLM calls."""
    uw = _rnd(req.plot_width - SETBACKS["left"] - SETBACKS["right"], 2)
    ul = _rnd(req.plot_length - SETBACKS["front"] - SETBACKS["rear"], 2)

    plan = await _generate_via_bsp(req, uw, ul)
    plan.generation_method = "emergency_local"
    plan.reasoning_trace = [
        f"Computed usable plot after setbacks: {uw:.1f} ft x {ul:.1f} ft.",
        "Emergency reliability mode used local deterministic planner.",
        "Returned a stable layout to avoid runtime dependency failures.",
    ]
    plan.architect_note = (
        "Emergency local planner was used to ensure uninterrupted generation."
    )
    base_reasoning = _build_architect_reasoning_object(
        req,
        uw,
        ul,
        source="local",
        status="emergency_local",
    )
    return _attach_plan_reasoning(
        plan,
        base_reasoning,
        req,
        uw,
        ul,
        "emergency_local",
    )


# ─────────────────────────────────────────────────────────────
# Legacy alias for backward compatibility
# ─────────────────────────────────────────────────────────────
async def generate_plan_deterministic(req: PlanRequest) -> PlanResponse:
    """Backward-compatible alias — now routes through LLM-first pipeline."""
    return await generate_plan(req)
