"""Multi-agent floor plan orchestration for NakshaNirman.

This module implements a deterministic multi-agent pipeline:
1) Plot Intelligence Agent
2) Program Architect Agent
3) Geometry Engine Agent
4) Connectivity and Doors Agent
5) Design Intelligence and Scoring Agent

The orchestrator retries with strategy changes when validation rejects a plan.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


HABITABLE_TYPES = {"living", "master_bedroom", "bedroom", "kitchen", "dining", "study"}


ABS_MIN_AREAS = {
    "living": 100.0,
    "master_bedroom": 100.0,
    "bedroom": 80.0,
    "kitchen": 50.0,
    "bathroom": 35.0,
    "toilet": 15.0,
    "dining": 70.0,
    "pooja": 20.0,
    "study": 60.0,
    "store": 20.0,
    "balcony": 20.0,
}


ROOM_LABELS = {
    "living": "Living Room",
    "dining": "Dining",
    "kitchen": "Kitchen",
    "master_bedroom": "Master Bedroom",
    "bedroom": "Bedroom",
    "bathroom": "Bathroom",
    "toilet": "Toilet",
    "pooja": "Pooja Room",
    "study": "Study",
    "store": "Store Room",
    "balcony": "Balcony",
    "garage": "Garage",
    "staircase": "Staircase",
}


def _r2(v: float) -> float:
    return round(float(v), 2)


def _snap_half(v: float) -> float:
    return round(v * 2.0) / 2.0


def _norm_facing(value: Any) -> str:
    facing = str(value or "east").strip().lower()
    return facing if facing in {"east", "west", "north", "south"} else "east"


def _norm_family(value: Any) -> str:
    family = str(value or "nuclear").strip().lower()
    allowed = {"nuclear", "joint-family", "working-couple", "elderly", "rental"}
    return family if family in allowed else "nuclear"


def _norm_extras(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, dict):
        value = list(value.keys())
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for token in value:
        k = str(token).strip().lower()
        if k in {"pooja", "study", "store", "balcony", "garage", "staircase"} and k not in out:
            out.append(k)
    return out


def _norm_constraints(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    items: List[Any]
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        room = str(item.get("room") or "").strip().lower()
        intent = str(item.get("intent") or "").strip().lower()
        if not room or not intent:
            continue
        normalized = {
            "room": room,
            "intent": intent,
            "band": item.get("band"),
            "forbid_adjacent": list(item.get("forbid_adjacent") or []),
            "prefer_walls": list(item.get("prefer_walls") or []),
            "note": str(item.get("note") or "")[:200],
        }
        out.append(normalized)
    return out


def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _pick_alias(payload: Dict[str, Any], aliases: List[str]) -> Any:
    table = {_norm_key(k): v for k, v in payload.items()}
    for key in aliases:
        nk = _norm_key(key)
        if nk in table:
            return table[nk]
    return None


def _extract_plot_dims(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if value is None:
        return None, None
    text = str(value).strip().lower().replace("feet", "ft").replace("foot", "ft")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|\*)\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None, None
    try:
        a = float(m.group(1))
        b = float(m.group(2))
        if not math.isfinite(a) or not math.isfinite(b):
            return None, None
        return a, b
    except Exception:
        return None, None


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str):
            cleaned = value.strip().lower().replace("sqft", "").replace("ft", "")
            cleaned = cleaned.replace(",", "")
            if cleaned in {"", "none", "null", "nan", "inf", "-inf"}:
                return float(default)
            out = float(cleaned)
        else:
            out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _to_int(value: Any, default: int, min_v: int, max_v: int) -> int:
    try:
        if value is None:
            out = int(default)
        elif isinstance(value, bool):
            out = int(value)
        elif isinstance(value, str):
            cleaned = value.strip().lower().replace("bhk", "").replace("rooms", "")
            cleaned = cleaned.replace(",", "")
            if cleaned in {"", "none", "null", "nan", "inf", "-inf"}:
                out = int(default)
            else:
                out = int(float(cleaned))
        else:
            out = int(float(value))
    except Exception:
        out = int(default)
    return max(min_v, min(max_v, out))


def _to_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(payload or {})

    plot_size = _pick_alias(raw, ["plot_size", "plotsize", "plot_dimensions", "plotsizedimensions", "site_size", "sitesize"])
    parsed_w, parsed_l = _extract_plot_dims(plot_size)

    width_raw = _pick_alias(raw, ["plot_width", "plotwidth", "width", "width_ft", "site_width", "plot_w"])
    length_raw = _pick_alias(raw, ["plot_length", "plotlength", "length", "length_ft", "site_length", "plot_l"])
    area_raw = _pick_alias(raw, ["total_area", "totalarea", "area", "area_sqft", "sqft", "plot_area", "site_area"])
    bhk_raw = _pick_alias(raw, ["bedrooms", "bedroom_count", "bedroomcount", "bhk", "rooms"])
    bath_raw = _pick_alias(raw, ["bathrooms", "bathroom_count", "bathroomcount", "bath_count", "baths"])
    floors_raw = _pick_alias(raw, ["floors", "floor_count", "floorcount", "storeys", "stories"])
    facing_raw = _pick_alias(raw, ["facing", "direction", "entry_facing", "front_direction"])
    family_raw = _pick_alias(raw, ["family_type", "familytype", "household", "family"])
    vastu_raw = _pick_alias(raw, ["vastu", "vastu_enabled", "needs_vastu", "is_vastu_required"])
    extras_raw = _pick_alias(raw, ["extras", "extra_rooms", "extrarooms", "optional_rooms", "add_ons", "addons"])
    city_raw = _pick_alias(raw, ["city", "location_city", "town"])
    state_raw = _pick_alias(raw, ["state", "province", "region", "location_state"])
    prev_strategy_raw = _pick_alias(raw, ["previous_strategy", "previousstrategy", "last_strategy", "retry_strategy"])
    constraints_raw = _pick_alias(
        raw,
        [
            "placement_constraints",
            "placementconstraints",
            "chat_constraints",
            "design_constraints",
            "preferences",
        ],
    )

    out = dict(raw)
    out["plot_width"] = _to_float(width_raw if width_raw is not None else parsed_w, 0.0)
    out["plot_length"] = _to_float(length_raw if length_raw is not None else parsed_l, 0.0)
    out["total_area"] = _to_float(area_raw, 1200.0)
    out["bedrooms"] = _to_int(bhk_raw, 2, 1, 4)
    out["bathrooms"] = _to_int(bath_raw, out["bedrooms"], 1, 6)
    out["floors"] = _to_int(floors_raw, 1, 1, 1)
    out["facing"] = _norm_facing(facing_raw)
    out["family_type"] = _norm_family(family_raw)
    out["vastu"] = _to_bool(vastu_raw, True)
    out["extras"] = _norm_extras(extras_raw)
    out["city"] = str(city_raw or "").strip()[:64]
    out["state"] = str(state_raw or "").strip()[:64]
    prev_strategy = prev_strategy_raw
    out["previous_strategy"] = str(prev_strategy).strip()[:64] if prev_strategy is not None else None
    out["placement_constraints"] = _norm_constraints(constraints_raw)

    # If width/length are not meaningful, rely on total area path.
    if out["plot_width"] <= 0 or out["plot_length"] <= 0:
        out["plot_width"] = None
        out["plot_length"] = None
    return out


def _normalize_payload_for_feasibility(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    out = dict(payload)
    notes: List[str] = []

    width = payload.get("plot_width")
    length = payload.get("plot_length")
    total_area = payload.get("total_area")
    if width and length:
        gross_area = float(width) * float(length)
    else:
        gross_area = float(total_area or 1200.0)

    bedrooms = max(1, min(4, int(payload.get("bedrooms") or 2)))
    bathrooms = max(1, min(6, int(payload.get("bathrooms") or bedrooms)))
    extras = _norm_extras(payload.get("extras"))

    # Area-aware guardrails to avoid impossible programs.
    if gross_area < 650 and bedrooms > 2:
        bedrooms = 2
        notes.append("Bedrooms reduced to 2 for plot feasibility")
    if gross_area < 900 and bathrooms > bedrooms:
        bathrooms = bedrooms
        notes.append("Bathrooms capped to bedroom count for compact plan")
    if gross_area < 750 and bathrooms > 2:
        bathrooms = 2
        notes.append("Bathrooms reduced to 2 due to limited area")

    if gross_area < 950:
        drop_order = ["garage", "balcony", "store", "study", "pooja"]
        dropped: List[str] = []
        for key in drop_order:
            if key in extras and len(extras) > 1:
                extras.remove(key)
                dropped.append(key)
            if gross_area >= 800 and len(dropped) >= 1:
                break
            if gross_area < 800 and len(dropped) >= 2:
                break
        if dropped:
            notes.append("Dropped extras for feasibility: " + ", ".join(dropped))

    out["bedrooms"] = bedrooms
    out["bathrooms"] = bathrooms
    out["extras"] = extras
    return out, notes


def _derive_plot(plot_width: Optional[float], plot_length: Optional[float], total_area: Optional[float]) -> Tuple[float, float, float]:
    if plot_width and plot_length:
        w = float(plot_width)
        l = float(plot_length)
        return _r2(w), _r2(l), _r2(w * l)
    area = float(total_area or 1200.0)
    w = math.sqrt(area * 0.75)
    l = area / max(w, 1.0)
    return _r2(w), _r2(l), _r2(area)


def _setbacks(area_sqft: float) -> Dict[str, float]:
    if area_sqft < 750:
        return {"front": 4.0, "rear": 3.0, "left": 2.0, "right": 2.0}
    if area_sqft <= 1800:
        return {"front": 6.5, "rear": 5.0, "left": 3.5, "right": 3.5}
    return {"front": 10.0, "rear": 5.0, "left": 5.0, "right": 5.0}


def _climate_zone(city: str, state: str) -> Tuple[str, str, str]:
    c = (city or "").strip().lower()
    s = (state or "").strip().lower()
    moderate = {"bangalore", "bengaluru", "pune", "mysore", "ooty"}
    warm_humid = {"mumbai", "chennai", "kochi", "goa"}
    hot_dry = {"delhi", "lucknow", "jaipur", "ahmedabad"}
    cold = {"shimla", "dehradun", "srinagar", "manali"}

    token = c if c else s
    if token in warm_humid:
        return "warm_humid", "both", "cross_ventilation"
    if token in hot_dry:
        return "composite_hot_dry", "east-west", "solar_passive"
    if token in cold:
        return "cold", "north-south", "solar_passive"
    if token in moderate:
        return "moderate", "both", "cross_ventilation"
    return "moderate", "both", "cross_ventilation"


def _corner_analysis() -> List[Dict[str, Any]]:
    return [
        {
            "corner": "north_east",
            "suitable_for": ["pooja", "living_extension", "balcony"],
            "notes": "Sacred corner; avoid toilet and heavy structure",
        },
        {
            "corner": "south_east",
            "suitable_for": ["kitchen"],
            "notes": "Fire element zone; ideal for kitchen",
        },
        {
            "corner": "south_west",
            "suitable_for": ["master_bedroom"],
            "notes": "Earth element zone; ideal for master bedroom",
        },
        {
            "corner": "north_west",
            "suitable_for": ["guest_bedroom", "garage", "store"],
            "notes": "Air element zone",
        },
    ]


def agent_plot_intelligence(
    payload: Dict[str, Any],
    band_profile: Optional[Dict[str, float]] = None,
    spine_override: Optional[str] = None,
) -> Dict[str, Any]:
    width, length, total_area = _derive_plot(payload.get("plot_width"), payload.get("plot_length"), payload.get("total_area"))
    facing = _norm_facing(payload.get("facing"))
    setbacks = _setbacks(total_area)

    usable_width = _r2(width - setbacks["left"] - setbacks["right"])
    usable_length = _r2(length - setbacks["front"] - setbacks["rear"])
    usable_area = _r2(max(usable_width, 0.0) * max(usable_length, 0.0))

    constraints: List[str] = []
    if usable_width < 12.0:
        constraints.append("Usable width below 12 ft; plot too narrow for standard planning")
    if usable_width <= 0 or usable_length <= 0:
        constraints.append("Setbacks leave no usable envelope")

    longer = max(usable_width, usable_length)
    shorter = min(usable_width, usable_length)
    if shorter <= 0:
        raise ValueError("Invalid usable dimensions after setbacks")

    aspect_ratio = _r2(usable_width / max(usable_length, 0.01))
    if aspect_ratio < 1.0:
        aspect_ratio = _r2(1.0 / max(aspect_ratio, 0.01))

    ratio_w_l = usable_width / max(usable_length, 0.01)
    if ratio_w_l < 0.65:
        plot_class = "narrow-deep"
        recommended_spine = "horizontal"
    elif ratio_w_l > 1.5:
        plot_class = "wide-shallow"
        recommended_spine = "vertical"
    elif 0.8 <= ratio_w_l <= 1.2:
        plot_class = "near-square"
        recommended_spine = "horizontal"
    else:
        plot_class = "elongated"
        recommended_spine = "horizontal" if usable_length >= usable_width else "vertical"

    primary_spine = spine_override if spine_override in {"horizontal", "vertical"} else recommended_spine

    seed = int(round(width + length))
    rng = random.Random(seed)
    if band_profile:
        public_pct = float(band_profile.get("public", 0.32))
        service_pct = float(band_profile.get("service", 0.20))
    else:
        public_pct = 0.28 + (rng.random() * 0.07)
        service_pct = 0.18 + (rng.random() * 0.04)
    private_pct = max(0.40, 1.0 - public_pct - service_pct)

    public_depth = _r2(usable_length * public_pct)
    service_depth = _r2(usable_length * service_pct)
    private_depth = _r2(max(usable_length - public_depth - service_depth, usable_length * 0.40))
    depth_total = public_depth + service_depth + private_depth
    if depth_total > usable_length:
        private_depth = _r2(private_depth - (depth_total - usable_length))

    climate, vent_axis, climate_strategy = _climate_zone(payload.get("city", ""), payload.get("state", ""))
    dominant_axis = primary_spine
    road_side = facing
    entry_wall = road_side
    plot_shape = plot_class
    if plot_class == "narrow-deep":
        layout_hint = "rear_private_spine"
    elif plot_class == "wide-shallow":
        layout_hint = "side_private_spine"
    elif plot_class == "near-square":
        layout_hint = "courtyard_hub"
    else:
        layout_hint = "elongated_spine"

    parking_feasible = (usable_width if road_side in {"north", "south"} else usable_length) >= 18.0
    grid = _r2(min(15.0, max(10.0, longer / max(round(longer / 12.0), 1))))

    constraints.append("At least two opposite-wall windows required for habitable rooms")
    constraints.append("Brahmasthan must remain open/light; no toilet, kitchen, or heavy loads")

    corners = {
        "south_west": {"x": 0.0, "y": 0.0},
        "south_east": {"x": usable_width, "y": 0.0},
        "north_west": {"x": 0.0, "y": usable_length},
        "north_east": {"x": usable_width, "y": usable_length},
    }

    strategy_name = f"{plot_class}_{primary_spine}_spine"

    return {
        "plot_shape": plot_shape,
        "road_side": road_side,
        "entry_wall": entry_wall,
        "dominant_axis": dominant_axis,
        "aspect_ratio": aspect_ratio,
        "layout_hint": layout_hint,
        "strategy_name": strategy_name,
        "spatial_contract": {
            "plot_class": plot_class,
            "primary_spine": primary_spine,
            "band_ratios": {
                "public": _r2(public_pct),
                "service": _r2(service_pct),
                "private": _r2(private_pct),
            },
            "corner_coordinates": corners,
            "usable_dimensions": {
                "width": usable_width,
                "length": usable_length,
            },
        },
        "plot": {
            "width": width,
            "length": length,
            "area": total_area,
            "setbacks": setbacks,
            "usable_width": usable_width,
            "usable_length": usable_length,
            "usable_area": usable_area,
        },
        "structural_grid": {"column_spacing_ft": grid},
        "zone_bands": {
            "public": public_depth,
            "service": service_depth,
            "private": private_depth,
        },
        "parking_feasibility": bool(parking_feasible),
        "corner_analysis": _corner_analysis(),
        "ventilation_axis": vent_axis,
        "climate_zone": climate,
        "climate_strategy": climate_strategy,
        "special_constraints": constraints,
        "brahmasthan": {
            "rule": "center_open",
            "note": "Keep central zone open and light; avoid toilet/kitchen/heavy structure",
        },
    }


def _strategy_for(aspect_ratio: float, bhk: int, seed: int) -> str:
    near_square = aspect_ratio < 1.4
    if bhk >= 4:
        return "winged"
    if near_square and bhk == 1:
        return "compact_open_plan"
    if near_square and bhk == 2:
        return "hub" if seed % 2 == 0 else "butterfly"
    if near_square and bhk == 3:
        return "cluster" if seed % 2 == 0 else "distributed"
    if aspect_ratio >= 1.4 and bhk >= 3:
        return "central_corridor"
    return "side_corridor"


def _next_strategy(current: str) -> str:
    order = ["cluster", "central_spine", "l_plan", "pinwheel"]
    if current not in order:
        return order[0]
    idx = order.index(current)
    return order[(idx + 1) % len(order)]


def _room_aspect_ratio(room_type: str, seed: int) -> float:
    rng = random.Random(seed + hash(room_type) % 10000)
    if room_type == "living":
        return 1.2 + rng.random() * 0.6
    if room_type == "master_bedroom":
        return 1.0 + rng.random() * 0.4
    if room_type == "kitchen":
        return 1.2 + rng.random() * 0.8
    if room_type in {"bathroom", "toilet"}:
        return 1.2 + rng.random() * 1.0
    return 1.0 + rng.random() * 0.5


def _build_room(
    rid: str,
    room_type: str,
    area: float,
    seed: int,
    zone: str,
    band: int,
    vastu_zone: str,
    must_touch_exterior: bool,
    adjacent_to: List[str],
    priority: int,
) -> Dict[str, Any]:
    ratio = _room_aspect_ratio(room_type, seed)
    width = max(4.0, math.sqrt(max(area, 1.0) * ratio))
    height = max(4.0, area / width)
    return {
        "id": rid,
        "type": room_type,
        "label": ROOM_LABELS.get(room_type, room_type.replace("_", " ").title()),
        "target_width": _r2(width),
        "target_height": _r2(height),
        "target_area": _r2(area),
        "zone": zone,
        "band": band,
        "vastu_zone": vastu_zone,
        "must_touch_exterior": must_touch_exterior,
        "adjacent_to": adjacent_to,
        "priority": int(priority),
    }


def _ensure_program_rules(rooms: List[Dict[str, Any]], bhk: int, bathrooms: int, family: str, seed: int, usable_area: float) -> List[Dict[str, Any]]:
    out = [dict(r) for r in rooms]
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for room in out:
        by_type.setdefault(room["type"], []).append(room)

    # Critical expectations for realistic residential planning.
    if bhk >= 2 and not by_type.get("dining"):
        out.append(
            _build_room(
                "dining",
                "dining",
                max(70.0, usable_area * 0.07),
                seed + 90,
                "public",
                1,
                "east",
                False,
                ["living", "kitchen"],
                9,
            )
        )

    if not by_type.get("kitchen"):
        out.append(
            _build_room(
                "kitchen",
                "kitchen",
                max(55.0, usable_area * 0.08),
                seed + 91,
                "service",
                2,
                "south_east",
                True,
                ["dining"],
                10,
            )
        )

    # Ensure at least one bathroom per bedroom count recommendation.
    current_baths = len([r for r in out if r["type"] in {"bathroom", "toilet"}])
    needed_baths = max(1, bathrooms)
    for idx in range(current_baths, needed_baths):
        out.append(
            _build_room(
                f"bath_auto_{idx + 1}",
                "bathroom",
                36.0,
                seed + 120 + idx,
                "service",
                2 if idx == 0 else 3,
                "west",
                True,
                [],
                7,
            )
        )

    # Elderly households should avoid deep-private only bedroom placement.
    if family == "elderly":
        for room in out:
            if room["type"] in {"master_bedroom", "bedroom"} and room.get("band", 3) == 3:
                room["band"] = 2
                room["zone"] = "service"

    # Keep core Vastu/functional intentions explicit.
    for room in out:
        if room["type"] == "kitchen":
            room["must_touch_exterior"] = True
            room["vastu_zone"] = str(room.get("vastu_zone") or "south_east")
        if room["type"] == "master_bedroom":
            room["vastu_zone"] = str(room.get("vastu_zone") or "south_west")
        if room["type"] == "pooja":
            room["vastu_zone"] = "north_east"
            room["must_touch_exterior"] = True

    return out


def _apply_placement_constraints(
    rooms: List[Dict[str, Any]],
    constraints: List[Dict[str, Any]],
) -> List[str]:
    notes: List[str] = []
    if not constraints:
        return notes

    def _find_room(room_type: str) -> Optional[Dict[str, Any]]:
        for room in rooms:
            if room.get("type") == room_type:
                return room
        return None

    for c in constraints:
        room = _find_room(str(c.get("room") or ""))
        if not room:
            continue
        intent = str(c.get("intent") or "")
        band = c.get("band")
        if isinstance(band, int) and 1 <= band <= 3:
            room["band"] = int(band)
            room["zone"] = "public" if band == 1 else "service" if band == 2 else "private"
            notes.append(f"Constraint applied: {room['id']} moved to band {band}")

        if intent == "rear_garden_preference":
            room["band"] = 3
            room["zone"] = "private"
            walls = set(c.get("prefer_walls") or [])
            room["preferred_exposure"] = sorted(walls) if walls else ["north", "rear"]
            notes.append(f"Constraint applied: {room['id']} prefers rear/north exposure")

        forbidden = {str(v).strip().lower() for v in (c.get("forbid_adjacent") or [])}
        if forbidden:
            room["forbid_adjacent"] = sorted(forbidden)
            notes.append(f"Constraint applied: {room['id']} forbids adjacency with {', '.join(sorted(forbidden))}")

    return notes


def _dynamic_kitchen_preferences(
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    constraints: List[Dict[str, Any]],
    seed: int,
) -> Tuple[int, str, List[str]]:
    facing = str(spatial_brief.get("road_side") or "east")
    climate = str(spatial_brief.get("climate_zone") or "moderate")
    family = _norm_family(payload.get("family_type"))
    vastu = bool(payload.get("vastu", True))

    rear_wall = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(facing, "north")

    for c in constraints:
        if str(c.get("room") or "") == "kitchen" and str(c.get("intent") or "") == "rear_garden_preference":
            pref_walls = [str(w).lower() for w in (c.get("prefer_walls") or [])]
            if not pref_walls:
                pref_walls = [rear_wall]
            return 3, "rear_service", pref_walls

    if vastu:
        band = 2 if family != "joint-family" else 3
        return band, "south_east", ["south", "east"]

    # Relaxed mode: adapt by facing + climate instead of fixed south-east.
    variant = seed % 2
    if climate in {"warm_humid", "moderate"}:
        zone_map = {
            "east": ["north", "south"],
            "west": ["north_west", "south_west"],
            "north": ["west", "south_west"],
            "south": ["east", "north_east"],
        }
    else:
        zone_map = {
            "east": ["south", "south_west"],
            "west": ["south_east", "south"],
            "north": ["west", "south"],
            "south": ["east", "north"],
        }

    zone_opts = zone_map.get(facing, ["south", "east"])
    kitchen_zone = zone_opts[variant % len(zone_opts)]
    band = 2 if family in {"nuclear", "working-couple", "elderly"} else (3 if variant else 2)

    wall_hint = {
        "north": ["north"],
        "south": ["south"],
        "east": ["east"],
        "west": ["west"],
        "north_east": ["north", "east"],
        "north_west": ["north", "west"],
        "south_east": ["south", "east"],
        "south_west": ["south", "west"],
    }.get(kitchen_zone, [rear_wall])
    return band, kitchen_zone, wall_hint


def _dynamic_master_preferences(spatial_brief: Dict[str, Any], payload: Dict[str, Any], seed: int) -> Tuple[int, str]:
    family = _norm_family(payload.get("family_type"))
    vastu = bool(payload.get("vastu", True))
    climate = str(spatial_brief.get("climate_zone") or "moderate")

    band = 2 if family == "elderly" else 3
    if vastu:
        return band, "south_west"

    options = ["west", "south_west", "north_west"] if climate in {"warm_humid", "moderate"} else ["south_west", "west"]
    return band, options[seed % len(options)]


def _dynamic_living_zone(spatial_brief: Dict[str, Any], payload: Dict[str, Any], seed: int) -> str:
    facing = str(spatial_brief.get("road_side") or "east")
    climate = str(spatial_brief.get("climate_zone") or "moderate")
    family = _norm_family(payload.get("family_type"))
    if climate == "warm_humid":
        living_opts = {
            "east": ["east", "north_east"],
            "west": ["north_west", "west"],
            "north": ["north", "north_east"],
            "south": ["east", "south_east"],
        }
    else:
        living_opts = {
            "east": ["north", "east"],
            "west": ["south", "west"],
            "north": ["east", "north"],
            "south": ["east", "south"],
        }
    picks = living_opts.get(facing, ["east", "north"])
    if family == "joint-family":
        return picks[0]
    return picks[seed % len(picks)]


def _dynamic_dining_preferences(
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    strategy: str,
    seed: int,
) -> Tuple[int, str, List[str]]:
    family = _norm_family(payload.get("family_type"))
    facing = str(spatial_brief.get("road_side") or "east")
    climate = str(spatial_brief.get("climate_zone") or "moderate")

    if family == "joint-family":
        return 1, "center", ["living", "kitchen"]

    if strategy in {"linear_spine", "elongated_spine", "central_corridor"}:
        band = 2
    else:
        band = 1 if (seed % 2 == 0) else 2

    zone_map = {
        "east": ["east", "center"],
        "west": ["west", "center"],
        "north": ["north", "center"],
        "south": ["south", "center"],
    }
    zone_opts = zone_map.get(facing, ["center"])
    if climate in {"warm_humid", "moderate"}:
        zone = zone_opts[0]
    else:
        zone = zone_opts[seed % len(zone_opts)]
    return band, zone, ["living", "kitchen"]


def _dynamic_bedroom_preferences(
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    idx: int,
    seed: int,
) -> Tuple[int, str]:
    family = _norm_family(payload.get("family_type"))
    facing = str(spatial_brief.get("road_side") or "east")
    climate = str(spatial_brief.get("climate_zone") or "moderate")
    vastu = bool(payload.get("vastu", True))

    band = 2 if family == "elderly" else 3
    if vastu:
        opts = ["north_west", "west", "north"]
        return band, opts[idx % len(opts)]

    if climate in {"warm_humid", "moderate"}:
        by_facing = {
            "east": ["north", "north_west", "west"],
            "west": ["north_west", "north", "south_west"],
            "north": ["north", "west", "east"],
            "south": ["east", "north_east", "west"],
        }
    else:
        by_facing = {
            "east": ["west", "north_west", "south_west"],
            "west": ["west", "south_west", "north_west"],
            "north": ["west", "south", "north_west"],
            "south": ["east", "north", "south_east"],
        }
    opts = by_facing.get(facing, ["north_west", "west"])
    return band, opts[(seed + idx) % len(opts)]


def _dynamic_bathroom_preferences(
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    idx: int,
    seed: int,
) -> Tuple[int, str]:
    family = _norm_family(payload.get("family_type"))
    facing = str(spatial_brief.get("road_side") or "east")
    vastu = bool(payload.get("vastu", True))

    if family == "elderly":
        band = 2
    else:
        band = 2 if idx == 0 else (2 if (seed + idx) % 2 == 0 else 3)

    if vastu:
        opts = ["north_west", "west", "south"]
        return band, opts[idx % len(opts)]

    by_facing = {
        "east": ["west", "south_west", "north_west"],
        "west": ["east", "south_east", "north_east"],
        "north": ["west", "south_west", "south"],
        "south": ["east", "north_east", "north"],
    }
    opts = by_facing.get(facing, ["west", "south"])
    # Still avoid north-east in relaxed mode for practical hygiene preference.
    opts = [z for z in opts if z != "north_east"] or ["west", "south"]
    return band, opts[(seed + idx) % len(opts)]


def _dynamic_extra_preferences(
    extra_type: str,
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    seed: int,
) -> Tuple[int, str, bool]:
    family = _norm_family(payload.get("family_type"))
    facing = str(spatial_brief.get("road_side") or "east")
    climate = str(spatial_brief.get("climate_zone") or "moderate")
    vastu = bool(payload.get("vastu", True))

    if extra_type == "pooja":
        return 1 if family != "elderly" else 2, "north_east", True
    if extra_type == "study":
        zone = "north" if climate in {"warm_humid", "moderate"} else "west"
        return 2 if family == "working-couple" else 3, zone, True
    if extra_type == "store":
        return 2, "north_west" if vastu else "west", False
    if extra_type == "balcony":
        zone_by_facing = {"east": "east", "west": "west", "north": "north", "south": "south"}
        return 1, zone_by_facing.get(facing, "north"), True
    if extra_type == "garage":
        if vastu:
            return 1, "north_west" if (seed % 2 == 0) else "south_east", True
        zone_by_facing = {"east": "east", "west": "west", "north": "north_west", "south": "south_east"}
        return 1, zone_by_facing.get(facing, "west"), True
    return 2, "center", False


def agent_program_architect(spatial_brief: Dict[str, Any], payload: Dict[str, Any], forced_strategy: Optional[str] = None) -> Dict[str, Any]:
    bhk = max(1, min(4, int(payload.get("bedrooms") or 2)))
    bathrooms = max(1, int(payload.get("bathrooms") or bhk))
    extras = _norm_extras(payload.get("extras"))
    family = _norm_family(payload.get("family_type"))
    vastu = bool(payload.get("vastu", True))
    usable_area = float(spatial_brief["plot"]["usable_area"])
    aspect_ratio = float(spatial_brief["aspect_ratio"])
    seed = int(round(spatial_brief["plot"]["width"] + spatial_brief["plot"]["length"]))

    strategy = forced_strategy or spatial_brief.get("strategy_name") or _strategy_for(aspect_ratio, bhk, seed)
    circulation_type = "hub" if strategy in {"hub", "butterfly", "cluster"} else "linear"

    allocations: Dict[str, float] = {}
    allocations["living"] = usable_area * (0.14 + (seed % 5) * 0.01)
    allocations["master_bedroom"] = usable_area * (0.15 + (seed % 3) * 0.01)
    other_bedrooms = max(0, bhk - 1)
    allocations["bedroom"] = usable_area * (0.10 + (seed % 4) * 0.01)
    allocations["kitchen"] = usable_area * (0.08 + (seed % 3) * 0.01)
    allocations["dining"] = usable_area * (0.07 + (seed % 3) * 0.01)
    allocations["bathroom"] = usable_area * (0.04 + (seed % 2) * 0.01)

    if "pooja" in extras:
        allocations["pooja"] = usable_area * 0.03
    if "study" in extras or family == "working-couple":
        allocations["study"] = usable_area * 0.06
    if "store" in extras or family == "joint-family":
        allocations["store"] = usable_area * 0.03
    if "balcony" in extras:
        allocations["balcony"] = usable_area * 0.03
    if "garage" in extras and spatial_brief.get("parking_feasibility"):
        allocations["garage"] = max(150.0, usable_area * 0.12)

    allocations["master_bedroom"] = max(
        allocations["master_bedroom"],
        100.0 if usable_area > 600 else allocations["master_bedroom"],
    )
    allocations["living"] = max(allocations["living"], 100.0)
    allocations["kitchen"] = max(allocations["kitchen"], 50.0)
    allocations["bathroom"] = max(allocations["bathroom"], 35.0)
    if bhk >= 2:
        allocations["dining"] = max(allocations["dining"], 70.0)

    constraints = _norm_constraints(payload.get("placement_constraints"))
    living_zone = _dynamic_living_zone(spatial_brief, payload, seed)
    dining_band, dining_zone, dining_adj = _dynamic_dining_preferences(spatial_brief, payload, strategy, seed)
    kitchen_band, kitchen_zone, kitchen_wall_pref = _dynamic_kitchen_preferences(
        spatial_brief,
        payload,
        constraints,
        seed,
    )
    master_band, master_zone = _dynamic_master_preferences(spatial_brief, payload, seed)

    rooms: List[Dict[str, Any]] = []
    rooms.append(_build_room("living", "living", allocations["living"], seed, "public", 1, living_zone, True, ["dining", "kitchen"], 10))
    rooms.append(
        _build_room(
            "dining",
            "dining",
            allocations["dining"],
            seed + 1,
            "public" if dining_band == 1 else "service",
            dining_band,
            dining_zone,
            True,
            dining_adj,
            9,
        )
    )
    rooms.append(
        _build_room(
            "kitchen",
            "kitchen",
            allocations["kitchen"],
            seed + 2,
            "service" if kitchen_band == 2 else "private",
            kitchen_band,
            kitchen_zone,
            True,
            ["dining"],
            10,
        )
    )
    rooms.append(
        _build_room(
            "master",
            "master_bedroom",
            allocations["master_bedroom"],
            seed + 3,
            "service" if master_band == 2 else "private",
            master_band,
            master_zone,
            True,
            ["bath_1"],
            10,
        )
    )
    rooms[-2]["preferred_exposure"] = list(kitchen_wall_pref)

    for idx in range(other_bedrooms):
        rid = f"bed_{idx + 1}"
        bed_band, bed_zone = _dynamic_bedroom_preferences(spatial_brief, payload, idx, seed)
        rooms.append(
            _build_room(
                rid,
                "bedroom",
                allocations["bedroom"],
                seed + 10 + idx,
                "service" if bed_band == 2 else "private",
                bed_band,
                bed_zone,
                True,
                [f"bath_{min(idx + 2, bathrooms)}"],
                8,
            )
        )

    for idx in range(bathrooms):
        rid = f"bath_{idx + 1}"
        bath_band, bathroom_zone = _dynamic_bathroom_preferences(spatial_brief, payload, idx, seed)
        rooms.append(
            _build_room(
                rid,
                "bathroom",
                allocations["bathroom"],
                seed + 20 + idx,
                "service" if bath_band == 2 else "private",
                bath_band,
                bathroom_zone,
                True,
                [],
                7,
            )
        )

    if "pooja" in allocations:
        pb, pz, pe = _dynamic_extra_preferences("pooja", spatial_brief, payload, seed)
        rooms.append(_build_room("pooja", "pooja", allocations["pooja"], seed + 30, "public" if pb == 1 else "service", pb, pz, pe, ["living"], 8))
    if "study" in allocations:
        sb, sz, se = _dynamic_extra_preferences("study", spatial_brief, payload, seed)
        rooms.append(_build_room("study", "study", allocations["study"], seed + 31, "service" if sb == 2 else "private", sb, sz, se, ["master"], 7))
    if "store" in allocations:
        stb, stz, ste = _dynamic_extra_preferences("store", spatial_brief, payload, seed)
        rooms.append(_build_room("store", "store", allocations["store"], seed + 32, "service" if stb == 2 else "private", stb, stz, ste, ["kitchen"], 4))
    if "balcony" in allocations:
        bb, bz, be = _dynamic_extra_preferences("balcony", spatial_brief, payload, seed)
        rooms.append(_build_room("balcony", "balcony", allocations["balcony"], seed + 33, "public", bb, bz, be, ["living"], 3))
    if "garage" in allocations:
        gb, gz, ge = _dynamic_extra_preferences("garage", spatial_brief, payload, seed)
        rooms.append(_build_room("garage", "garage", allocations["garage"], seed + 34, "service" if gb == 2 else "public", gb, gz, ge, ["living"], 6))

    if family == "joint-family":
        has_formal_dining = any(r["type"] == "dining" for r in rooms)
        if not has_formal_dining:
            rooms.append(_build_room("dining", "dining", max(90.0, usable_area * 0.08), seed + 35, "public", 1, "east", False, ["living", "kitchen"], 9))
        if not any(r["type"] == "pooja" for r in rooms):
            rooms.append(_build_room("pooja_jf", "pooja", max(20.0, usable_area * 0.025), seed + 36, "public", 1, "north_east", True, ["living"], 8))
    if family == "elderly":
        circulation_type = "linear"

    if vastu:
        for r in rooms:
            if r["type"] == "kitchen":
                r["vastu_zone"] = "south_east"
            elif r["type"] == "master_bedroom":
                r["vastu_zone"] = "south_west"
            elif r["type"] == "pooja":
                r["vastu_zone"] = "north_east"

    rooms = _ensure_program_rules(
        rooms=rooms,
        bhk=bhk,
        bathrooms=bathrooms,
        family=family,
        seed=seed,
        usable_area=usable_area,
    )

    constraint_notes = _apply_placement_constraints(rooms, constraints)

    explanation = (
        f"Strategy '{strategy}' fits a {aspect_ratio:.2f} aspect ratio plot with {bhk}BHK requirements. "
        f"The program prioritizes living-kitchen-dining flow, Vastu alignment, and climate-aware exterior exposure."
    )

    return {
        "layout_strategy": strategy,
        "circulation_type": circulation_type,
        "rooms": rooms,
        "constraint_notes": constraint_notes,
        "explanation": explanation,
    }


def _adjacent(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.25) -> bool:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    touch_x = abs(ax2 - bx1) <= tol or abs(bx2 - ax1) <= tol
    touch_y = abs(ay2 - by1) <= tol or abs(by2 - ay1) <= tol
    return (touch_x and y_overlap > 0.5) or (touch_y and x_overlap > 0.5)


def _overlap_area(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ox = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return ox * oy


def _room_min_area(room_type: str) -> float:
    return ABS_MIN_AREAS.get(room_type, 25.0)


def agent_geometry_engine(
    room_program: Dict[str, Any],
    spatial_brief: Dict[str, Any],
    service_first: bool = False,
) -> Dict[str, Any]:
    usable_width = float(spatial_brief["plot"]["usable_width"])
    usable_length = float(spatial_brief["plot"]["usable_length"])
    bands = spatial_brief["zone_bands"]
    road_side = str(spatial_brief.get("road_side") or "east")

    # Facing-aware bands: public must sit near road-facing edge, private farthest away.
    if road_side == "south":
        band_rects = {
            1: {"x": 0.0, "y": 0.0, "w": usable_width, "h": bands["public"]},
            2: {"x": 0.0, "y": bands["public"], "w": usable_width, "h": bands["service"]},
            3: {"x": 0.0, "y": bands["public"] + bands["service"], "w": usable_width, "h": max(0.5, usable_length - bands["public"] - bands["service"])},
        }
    elif road_side == "north":
        p1 = max(0.0, usable_length - bands["public"])
        p2 = max(0.0, p1 - bands["service"])
        band_rects = {
            1: {"x": 0.0, "y": p1, "w": usable_width, "h": bands["public"]},
            2: {"x": 0.0, "y": p2, "w": usable_width, "h": bands["service"]},
            3: {"x": 0.0, "y": 0.0, "w": usable_width, "h": max(0.5, p2)},
        }
    elif road_side == "west":
        band_rects = {
            1: {"x": 0.0, "y": 0.0, "w": bands["public"], "h": usable_length},
            2: {"x": bands["public"], "y": 0.0, "w": bands["service"], "h": usable_length},
            3: {"x": bands["public"] + bands["service"], "y": 0.0, "w": max(0.5, usable_width - bands["public"] - bands["service"]), "h": usable_length},
        }
    else:  # east
        p1 = max(0.0, usable_width - bands["public"])
        p2 = max(0.0, p1 - bands["service"])
        band_rects = {
            1: {"x": p1, "y": 0.0, "w": bands["public"], "h": usable_length},
            2: {"x": p2, "y": 0.0, "w": bands["service"], "h": usable_length},
            3: {"x": 0.0, "y": 0.0, "w": max(0.5, p2), "h": usable_length},
        }

    grouped: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
    for room in room_program["rooms"]:
        grouped[max(1, min(3, int(room.get("band", 2))))].append(dict(room))

    if service_first:
        for band in grouped:
            grouped[band].sort(key=lambda r: (0 if r["type"] in {"bathroom", "toilet", "kitchen", "store"} else 1, -int(r.get("priority", 1))))
    else:
        for band in grouped:
            grouped[band].sort(key=lambda r: -int(r.get("priority", 1)))

    placed: List[Dict[str, Any]] = []
    adjustments: List[str] = []
    primary_spine = str(spatial_brief.get("spatial_contract", {}).get("primary_spine") or "horizontal")

    def _clamp_room(room: Dict[str, Any]) -> None:
        room["x"] = _r2(max(0.0, min(usable_width - room["width"], _snap_half(room["x"]))))
        room["y"] = _r2(max(0.0, min(usable_length - room["height"], _snap_half(room["y"]))))
        room["area"] = _r2(room["width"] * room["height"])

    def _sum_overlap(candidate: Dict[str, Any], others: List[Dict[str, Any]]) -> float:
        return sum(_overlap_area(candidate, other) for other in others if other["id"] != candidate["id"])

    def _zone_target(zone: str, y1: float, y2: float) -> Tuple[float, float]:
        z = str(zone or "center").lower()
        xm = usable_width * 0.5
        ym = (y1 + y2) * 0.5
        x_map = {
            "west": usable_width * 0.18,
            "north_west": usable_width * 0.18,
            "south_west": usable_width * 0.18,
            "east": usable_width * 0.82,
            "north_east": usable_width * 0.82,
            "south_east": usable_width * 0.82,
            "center": usable_width * 0.5,
            "north": usable_width * 0.5,
            "south": usable_width * 0.5,
            "rear_service": usable_width * 0.5,
        }
        y_map = {
            "south": y1 + (y2 - y1) * 0.18,
            "south_west": y1 + (y2 - y1) * 0.18,
            "south_east": y1 + (y2 - y1) * 0.18,
            "center": y1 + (y2 - y1) * 0.5,
            "east": y1 + (y2 - y1) * 0.5,
            "west": y1 + (y2 - y1) * 0.5,
            "north": y1 + (y2 - y1) * 0.82,
            "north_west": y1 + (y2 - y1) * 0.82,
            "north_east": y1 + (y2 - y1) * 0.82,
            "rear_service": y1 + (y2 - y1) * 0.82,
        }
        return x_map.get(z, xm), y_map.get(z, ym)

    def _touches_exterior_rect(rect: Dict[str, float]) -> bool:
        rx1 = rect["x"]
        ry1 = rect["y"]
        rx2 = rect["x"] + rect["w"]
        ry2 = rect["y"] + rect["h"]
        return (
            abs(rx1 - 0.0) < 0.01
            or abs(rx2 - usable_width) < 0.01
            or abs(ry1 - 0.0) < 0.01
            or abs(ry2 - usable_length) < 0.01
        )

    def _rect_exterior_walls(rect: Dict[str, float]) -> List[str]:
        walls: List[str] = []
        if abs(rect["x"] - 0.0) < 0.01:
            walls.append("west")
        if abs((rect["x"] + rect["w"]) - usable_width) < 0.01:
            walls.append("east")
        if abs(rect["y"] - 0.0) < 0.01:
            walls.append("south")
        if abs((rect["y"] + rect["h"]) - usable_length) < 0.01:
            walls.append("north")
        return walls

    for band in [1, 2, 3]:
        rooms = grouped[band]
        if not rooms:
            continue
        band_rect = band_rects[band]
        y1 = band_rect["y"]
        y2 = band_rect["y"] + band_rect["h"]

        free_rects: List[Dict[str, float]] = [dict(band_rect)]

        for idx, room in enumerate(rooms):
            target_area = max(_room_min_area(room["type"]), float(room.get("target_area", 40.0)))
            zone_x, zone_y = _zone_target(str(room.get("vastu_zone") or "center"), y1, y2)
            if not free_rects:
                container = dict(band_rect)
            else:
                scored_rects: List[Tuple[float, Dict[str, float]]] = []
                for rect in free_rects:
                    cx = rect["x"] + rect["w"] / 2.0
                    cy = rect["y"] + rect["h"] / 2.0
                    area_score = rect["w"] * rect["h"]
                    dist_penalty = abs(cx - zone_x) + abs(cy - zone_y)
                    ext_bonus = 8.0 if bool(room.get("must_touch_exterior")) and _touches_exterior_rect(rect) else 0.0
                    pref_walls = {str(w).strip().lower() for w in (room.get("preferred_exposure") or [])}
                    rect_walls = set(_rect_exterior_walls(rect))
                    pref_bonus = 5.0 if pref_walls and (pref_walls & rect_walls) else 0.0
                    score = area_score - dist_penalty + ext_bonus + pref_bonus
                    scored_rects.append((score, rect))
                scored_rects.sort(key=lambda t: t[0], reverse=True)
                container = scored_rects[0][1]
                free_rects.remove(container)

            min_w = 4.0
            min_h = 4.0
            if idx == len(rooms) - 1:
                rw = container["w"]
                rh = container["h"]
            else:
                if primary_spine == "vertical" and container["h"] > min_h:
                    rh = _snap_half(max(min_h, min(container["h"], target_area / max(container["w"], 0.5))))
                    rw = _snap_half(max(min_w, container["w"]))
                else:
                    rw = _snap_half(max(min_w, min(container["w"], target_area / max(container["h"], 0.5))))
                    rh = _snap_half(max(min_h, container["h"]))

            rw = _snap_half(max(3.0, min(container["w"], rw)))
            rh = _snap_half(max(3.0, min(container["h"], rh)))

            px = _snap_half(container["x"])
            py = _snap_half(container["y"])

            placed_room = {
                "id": room["id"],
                "type": room["type"],
                "label": room["label"],
                "zone": room["zone"],
                "x": _r2(px),
                "y": _r2(py),
                "width": _r2(rw),
                "height": _r2(rh),
                "area": _r2(rw * rh),
                "must_touch_exterior": bool(room.get("must_touch_exterior")),
                "adjacent_to": list(room.get("adjacent_to") or []),
                "forbid_adjacent": list(room.get("forbid_adjacent") or []),
                "preferred_exposure": list(room.get("preferred_exposure") or []),
                "vastu_zone": str(room.get("vastu_zone") or "center"),
                "priority": int(room.get("priority", 1)),
                "band": band,
            }
            placed.append(placed_room)

            right_w = container["w"] - rw
            top_h = container["h"] - rh
            if right_w >= 2.5:
                free_rects.append({"x": container["x"] + rw, "y": container["y"], "w": right_w, "h": rh})
            if top_h >= 2.5:
                free_rects.append({"x": container["x"], "y": container["y"] + rh, "w": container["w"], "h": top_h})

    room_map = {r["id"]: r for r in placed}
    for _ in range(8):
        moved = False
        for room in placed:
            for target_id in room.get("adjacent_to", []):
                other = room_map.get(target_id)
                if not other or _adjacent(room, other):
                    continue
                dx = (other["x"] + other["width"] / 2.0) - (room["x"] + room["width"] / 2.0)
                dy = (other["y"] + other["height"] / 2.0) - (room["y"] + room["height"] / 2.0)
                step_x = 0.5 if dx > 0 else -0.5 if dx < 0 else 0.0
                step_y = 0.5 if dy > 0 else -0.5 if dy < 0 else 0.0
                old_x = room["x"]
                old_y = room["y"]
                current_overlap = _sum_overlap(room, placed)

                if abs(dx) >= abs(dy):
                    room["x"] += step_x
                else:
                    room["y"] += step_y
                _clamp_room(room)

                new_overlap = _sum_overlap(room, placed)
                if new_overlap <= current_overlap + 0.05:
                    moved = moved or (old_x != room["x"] or old_y != room["y"])
                    if old_x != room["x"] or old_y != room["y"]:
                        adjustments.append(
                            f"Adjacency slide: {room['id']} moved ({old_x},{old_y}) -> ({room['x']},{room['y']}) toward {other['id']}"
                        )
                else:
                    room["x"] = old_x
                    room["y"] = old_y

        if not moved:
            break

    for _ in range(6):
        had_overlap = False
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a = placed[i]
                b = placed[j]
                ov = _overlap_area(a, b)
                if ov <= 0.05:
                    continue
                had_overlap = True
                low = b if b["priority"] <= a["priority"] else a

                candidates = []
                candidates.append((low["x"] + 0.5, low["y"]))
                candidates.append((low["x"] - 0.5, low["y"]))
                candidates.append((low["x"], low["y"] + 0.5))
                candidates.append((low["x"], low["y"] - 0.5))

                best_xy = (low["x"], low["y"])
                best_ov = _sum_overlap(low, placed)
                best_disp = 0.0
                for cand_x, cand_y in candidates:
                    old_x = low["x"]
                    old_y = low["y"]
                    low["x"] = cand_x
                    low["y"] = cand_y
                    _clamp_room(low)
                    cand_ov = _sum_overlap(low, placed)
                    disp = abs(low["x"] - old_x) + abs(low["y"] - old_y)
                    if cand_ov < best_ov - 0.01 or (abs(cand_ov - best_ov) <= 0.01 and disp < best_disp):
                        best_ov = cand_ov
                        best_xy = (low["x"], low["y"])
                        best_disp = disp
                    low["x"] = old_x
                    low["y"] = old_y

                if best_xy != (low["x"], low["y"]):
                    old_x = low["x"]
                    old_y = low["y"]
                    low["x"], low["y"] = best_xy
                    _clamp_room(low)
                    adjustments.append(
                        f"Overlap resolve: moved {low['id']} ({old_x},{old_y}) -> ({low['x']},{low['y']})"
                    )

        if not had_overlap:
            break

    room_map = {r["id"]: r for r in placed}
    for room in placed:
        actual_shared: List[str] = []
        for other in placed:
            if other["id"] == room["id"]:
                continue
            if _adjacent(room, other):
                actual_shared.append(other["id"])
        room["actual_shared_walls"] = sorted(actual_shared)
        room["polygon"] = [
            [_r2(room["x"]), _r2(room["y"])],
            [_r2(room["x"] + room["width"]), _r2(room["y"])],
            [_r2(room["x"] + room["width"]), _r2(room["y"] + room["height"])],
            [_r2(room["x"]), _r2(room["y"] + room["height"])],
        ]

    checks: Dict[str, bool] = {
        "all_within_bbox": True,
        "no_overlap": True,
        "coverage_ge_85": True,
        "aspect_ratio_ok": True,
        "must_touch_exterior_ok": True,
        "adjacency_ok": True,
    }

    failures: List[str] = []

    for room in placed:
        if room["x"] < 0 or room["y"] < 0 or room["x"] + room["width"] > usable_width + 0.01 or room["y"] + room["height"] > usable_length + 0.01:
            checks["all_within_bbox"] = False
            failures.append(f"{room['id']} extends outside usable box")
        ratio = max(room["width"], room["height"]) / max(min(room["width"], room["height"]), 0.5)
        if ratio > 2.5:
            checks["aspect_ratio_ok"] = False
            failures.append(f"{room['id']} aspect ratio > 2.5")
        if room["must_touch_exterior"]:
            touches = (
                abs(room["x"] - 0.0) < 0.01
                or abs(room["y"] - 0.0) < 0.01
                or abs((room["x"] + room["width"]) - usable_width) < 0.01
                or abs((room["y"] + room["height"]) - usable_length) < 0.01
            )
            if not touches:
                checks["must_touch_exterior_ok"] = False
                failures.append(f"{room['id']} missing exterior contact")

    total_overlap = 0.0
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            ov = _overlap_area(placed[i], placed[j])
            total_overlap += ov
            if ov > 0.1:
                checks["no_overlap"] = False
                failures.append(f"Overlap {placed[i]['id']} vs {placed[j]['id']}: {ov:.2f} sqft")

    gross_usable = max(usable_width * usable_length, 1.0)
    coverage = sum(r["area"] for r in placed) / gross_usable
    if gross_usable < 750:
        min_coverage = 0.78
    elif gross_usable < 1000:
        min_coverage = 0.82
    else:
        min_coverage = 0.85
    if coverage < min_coverage:
        checks["coverage_ge_85"] = False
        failures.append(f"Coverage below {min_coverage * 100:.0f}% ({coverage * 100:.1f}%)")

    by_id = {r["id"]: r for r in placed}
    for room in placed:
        for need in room.get("adjacent_to", []):
            if need in by_id and not _adjacent(room, by_id[need]):
                checks["adjacency_ok"] = False
                failures.append(f"Adjacency missing: {room['id']} -> {need}")

        forbidden = {str(v).strip().lower() for v in room.get("forbid_adjacent", [])}
        if forbidden:
            for other in placed:
                if other["id"] == room["id"]:
                    continue
                if other["type"] in forbidden and _adjacent(room, other):
                    checks["adjacency_ok"] = False
                    failures.append(
                        f"Forbidden adjacency present: {room['id']} next to {other['id']}"
                    )

    return {
        "rooms": placed,
        "validation_report": {
            "checks": checks,
            "pass": all(checks.values()),
            "total_overlap": _r2(total_overlap),
            "coverage": _r2(coverage * 100.0),
            "failures": failures,
            "adjustments": adjustments,
        },
    }


def _shared_wall(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.25) -> Optional[Tuple[str, float, float, float]]:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    y_ov1 = max(ay1, by1)
    y_ov2 = min(ay2, by2)
    if abs(ax2 - bx1) <= tol and (y_ov2 - y_ov1) > 3.0:
        return ("east", ax2, y_ov1, y_ov2)
    if abs(bx2 - ax1) <= tol and (y_ov2 - y_ov1) > 3.0:
        return ("west", ax1, y_ov1, y_ov2)

    x_ov1 = max(ax1, bx1)
    x_ov2 = min(ax2, bx2)
    if abs(ay2 - by1) <= tol and (x_ov2 - x_ov1) > 3.0:
        return ("north", ay2, x_ov1, x_ov2)
    if abs(by2 - ay1) <= tol and (x_ov2 - x_ov1) > 3.0:
        return ("south", ay1, x_ov1, x_ov2)

    return None


def _room_on_edge(room: Dict[str, Any], side: str, usable_width: float, usable_length: float) -> bool:
    if side == "east":
        return abs((room["x"] + room["width"]) - usable_width) < 0.01
    if side == "west":
        return abs(room["x"] - 0.0) < 0.01
    if side == "north":
        return abs((room["y"] + room["height"]) - usable_length) < 0.01
    return abs(room["y"] - 0.0) < 0.01


def agent_connectivity_doors(
    placed_layout: Dict[str, Any],
    spatial_brief: Dict[str, Any],
    room_program: Dict[str, Any],
) -> Dict[str, Any]:
    rooms = [dict(r) for r in placed_layout["rooms"]]
    by_id = {r["id"]: r for r in rooms}
    usable_width = float(spatial_brief["plot"]["usable_width"])
    usable_length = float(spatial_brief["plot"]["usable_length"])
    road_side = spatial_brief.get("road_side", "east")
    climate_zone = str(spatial_brief.get("climate_zone") or "moderate")

    doors: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    issues: List[str] = []
    suggested_adjustments: List[str] = []
    adjacency_edges: Dict[str, List[str]] = {r["id"]: [] for r in rooms}

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]
            shared = _shared_wall(a, b)
            if not shared:
                continue

            wall, anchor, ov1, ov2 = shared
            span = ov2 - ov1
            if span < 3.0:
                continue

            if (
                (a["type"] in {"bathroom", "toilet"} and b["type"] in {"kitchen", "pooja"})
                or (b["type"] in {"bathroom", "toilet"} and a["type"] in {"kitchen", "pooja"})
            ):
                continue

            door_width = 3.0
            if a["type"] in {"bathroom", "toilet"} or b["type"] in {"bathroom", "toilet"}:
                door_width = 2.5

            center = (ov1 + ov2) / 2.0
            if center - ov1 < 1.0:
                center = ov1 + 1.0
            if ov2 - center < 1.0:
                center = ov2 - 1.0

            if wall in {"east", "west"}:
                x = anchor
                y = center
            else:
                x = center
                y = anchor

            did = f"door_{a['id']}_{b['id']}"
            doors.append(
                {
                    "id": did,
                    "room_id": a["id"],
                    "to_room_id": b["id"],
                    "wall": wall,
                    "wall_direction": wall,
                    "x": _r2(_snap_half(x)),
                    "y": _r2(_snap_half(y)),
                    "width": door_width,
                    "type": "bathroom" if door_width < 3.0 else "internal",
                }
            )
            adjacency_edges[a["id"]].append(b["id"])
            adjacency_edges[b["id"]].append(a["id"])

    living = next((r for r in rooms if r["type"] == "living"), None)
    if living:
        side = road_side
        if not _room_on_edge(living, side, usable_width, usable_length):
            side = "south"

        if side == "east":
            mx, my = living["x"] + living["width"], living["y"] + living["height"] / 2.0
        elif side == "west":
            mx, my = living["x"], living["y"] + living["height"] / 2.0
        elif side == "north":
            mx, my = living["x"] + living["width"] / 2.0, living["y"] + living["height"]
        else:
            mx, my = living["x"] + living["width"] / 2.0, living["y"]

        doors.append(
            {
                "id": "door_main",
                "room_id": living["id"],
                "to_room_id": "outside",
                "wall": side,
                "wall_direction": side,
                "x": _r2(_snap_half(mx)),
                "y": _r2(_snap_half(my)),
                "width": 3.5,
                "type": "main",
            }
        )
    else:
        issues.append("No living room found for main entrance placement")

    # Ensure each room has at least one traversable connection where possible.
    for room in rooms:
        if room["id"] == (living["id"] if living else ""):
            continue
        if adjacency_edges.get(room["id"]):
            continue
        if not living:
            continue
        shared = _shared_wall(room, living)
        if not shared:
            continue
        wall, anchor, ov1, ov2 = shared
        center = (ov1 + ov2) / 2.0
        if wall in {"east", "west"}:
            x, y = anchor, center
        else:
            x, y = center, anchor
        did = f"door_link_{room['id']}_{living['id']}"
        doors.append(
            {
                "id": did,
                "room_id": room["id"],
                "to_room_id": living["id"],
                "wall": wall,
                "wall_direction": wall,
                "x": _r2(_snap_half(x)),
                "y": _r2(_snap_half(y)),
                "width": 3.0,
                "type": "internal",
            }
        )
        adjacency_edges[room["id"]].append(living["id"])
        adjacency_edges[living["id"]].append(room["id"])

    # Windows
    for room in rooms:
        room_windows = 0
        edges: List[Tuple[str, float, float, float]] = []
        if abs(room["x"] - 0.0) < 0.01:
            edges.append(("west", room["x"], room["y"], room["y"] + room["height"]))
        if abs((room["x"] + room["width"]) - usable_width) < 0.01:
            edges.append(("east", room["x"] + room["width"], room["y"], room["y"] + room["height"]))
        if abs(room["y"] - 0.0) < 0.01:
            edges.append(("south", room["y"], room["x"], room["x"] + room["width"]))
        if abs((room["y"] + room["height"]) - usable_length) < 0.01:
            edges.append(("north", room["y"] + room["height"], room["x"], room["x"] + room["width"]))

        if room.get("must_touch_exterior") and edges:
            preferred = {str(w).strip().lower() for w in room.get("preferred_exposure", [])}
            hot_dry = climate_zone in {"composite_hot_dry", "hot_dry"}
            candidate_edges = list(edges)
            non_west = [e for e in candidate_edges if e[0] != "west"]
            # In hot-dry climates, avoid west-facing primary openings when alternatives exist.
            if hot_dry and non_west:
                candidate_edges = non_west

            def _edge_score(edge: Tuple[str, float, float, float]) -> float:
                wall, _, p1, p2 = edge
                length = p2 - p1
                pref_bonus = 5.0 if preferred and wall in preferred else 0.0
                climate_penalty = 0.0
                if hot_dry and wall == "west":
                    climate_penalty = 2.5
                return length + pref_bonus - climate_penalty

            # Place one centered window on the best-scoring exterior edge.
            chosen = max(candidate_edges, key=_edge_score)
            wall, anchor, p1, p2 = chosen
            center = (p1 + p2) / 2.0
            if wall in {"east", "west"}:
                wx, wy = anchor, center
                base_dim = room["height"]
            else:
                wx, wy = center, anchor
                base_dim = room["width"]

            if room["type"] in {"living", "bedroom", "master_bedroom"}:
                ww = _r2(max(2.5, base_dim * 0.40))
            elif room["type"] == "kitchen":
                ww = _r2(max(2.0, base_dim * 0.25))
            elif room["type"] in {"bathroom", "toilet"}:
                ww = _r2(max(1.5, base_dim * 0.25))
            else:
                ww = _r2(max(1.5, base_dim * 0.25))

            windows.append(
                {
                    "id": f"win_{room['id']}_1",
                    "room_id": room["id"],
                    "wall": wall,
                    "wall_direction": wall,
                    "x": _r2(_snap_half(wx)),
                    "y": _r2(_snap_half(wy)),
                    "width": ww,
                    "type": "vent" if room["type"] in {"bathroom", "toilet"} else "standard",
                    "sill_height": 6.0 if room["type"] in {"bathroom", "toilet"} else 3.0,
                    "opening_height": 1.5 if room["type"] in {"bathroom", "toilet"} else 4.0,
                }
            )
            room_windows += 1

        # Cross ventilation request: for wider rooms, add opposite-side window if possible.
        if room["width"] > 12.0 and len(edges) >= 2:
            walls = {e[0] for e in edges}
            opposite = None
            primary_wall = next((w.get("wall") for w in windows if w.get("room_id") == room["id"]), None)
            if primary_wall == "east" and "west" in walls:
                opposite = "west"
            elif primary_wall == "west" and "east" in walls:
                opposite = "east"
            elif primary_wall == "north" and "south" in walls:
                opposite = "south"
            elif primary_wall == "south" and "north" in walls:
                opposite = "north"
            elif "north" in walls and "south" in walls:
                opposite = "south"
            elif "east" in walls and "west" in walls:
                opposite = "east" if climate_zone in {"composite_hot_dry", "hot_dry"} else "west"
            if opposite:
                alt = next((e for e in edges if e[0] == opposite), None)
                if alt:
                    wall, anchor, p1, p2 = alt
                    center = (p1 + p2) / 2.0
                    if wall in {"east", "west"}:
                        wx, wy = anchor, center
                    else:
                        wx, wy = center, anchor
                    windows.append(
                        {
                            "id": f"win_{room['id']}_{room_windows + 1}",
                            "room_id": room["id"],
                            "wall": wall,
                            "wall_direction": wall,
                            "x": _r2(_snap_half(wx)),
                            "y": _r2(_snap_half(wy)),
                            "width": _r2(max(2.0, (room["width"] if wall in {"north", "south"} else room["height"]) * 0.35)),
                            "type": "standard",
                        }
                    )

    # Path-finding tests over room adjacency graph.
    def shortest_doors(src: str, dst: str) -> Optional[int]:
        if src == dst:
            return 0
        seen = {src}
        q: List[Tuple[str, int]] = [(src, 0)]
        while q:
            node, dist = q.pop(0)
            for nxt in adjacency_edges.get(node, []):
                if nxt in seen:
                    continue
                if nxt == dst:
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return None

    living_id = living["id"] if living else ""
    kitchen = next((r for r in rooms if r["type"] == "kitchen"), None)
    master = next((r for r in rooms if r["type"] == "master_bedroom"), None)
    dining = next((r for r in rooms if r["type"] == "dining"), None)
    bedrooms = [r for r in rooms if r["type"] in {"master_bedroom", "bedroom"}]
    baths = [r for r in rooms if r["type"] in {"bathroom", "toilet"}]

    path_1 = bool(living)
    path_2 = kitchen is not None and shortest_doors(living_id, kitchen["id"]) is not None and shortest_doors(living_id, kitchen["id"]) <= 1
    path_3 = master is not None and shortest_doors(living_id, master["id"]) is not None and shortest_doors(living_id, master["id"]) <= 2
    path_4 = False
    if kitchen and dining:
        path_4 = _adjacent(kitchen, dining) and any(
            {d.get("room_id"), d.get("to_room_id")} == {kitchen["id"], dining["id"]}
            for d in doors
        )
    path_5 = True
    for bed in bedrooms:
        if not baths:
            path_5 = False
            break
        best = min((shortest_doors(bed["id"], b["id"]) for b in baths if shortest_doors(bed["id"], b["id"]) is not None), default=None)
        if best is None or best > 1:
            path_5 = False
            break

    paths = {
        "entrance_to_living": path_1,
        "living_to_kitchen": path_2,
        "living_to_master": path_3,
        "kitchen_dining_direct": path_4,
        "bedroom_to_bathroom": path_5,
        "all_bedrooms_reachable_within_2_doors": all(
            shortest_doors(living_id, bed["id"]) is not None and shortest_doors(living_id, bed["id"]) <= 2
            for bed in bedrooms
        ) if living else False,
    }
    failed = sum(1 for ok in paths.values() if not ok)
    circulation_score = max(0, 100 - failed * 20)

    if not all(paths.values()):
        issues.append("One or more circulation tests failed; consider regeneration")
        if not paths["living_to_kitchen"]:
            suggested_adjustments.append("Move kitchen to share wall with living/dining or add direct connector")
        if not paths["all_bedrooms_reachable_within_2_doors"]:
            suggested_adjustments.append("Reposition bedrooms closer to transition corridor from living")
        if not paths["bedroom_to_bathroom"]:
            suggested_adjustments.append("Swap bathroom with nearest service room in same band")

    return {
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "circulation_score": circulation_score,
        "paths": paths,
        "issues": issues,
        "suggested_adjustments": suggested_adjustments,
    }


def _room_zone_9(room: Dict[str, Any], usable_width: float, usable_length: float) -> str:
    cx = room["x"] + room["width"] / 2.0
    cy = room["y"] + room["height"] / 2.0
    col = 0 if cx < usable_width / 3 else 1 if cx < 2 * usable_width / 3 else 2
    row = 0 if cy < usable_length / 3 else 1 if cy < 2 * usable_length / 3 else 2
    grid = {
        (2, 2): "north_east",
        (2, 1): "north",
        (2, 0): "north_west",
        (1, 2): "east",
        (1, 1): "center",
        (1, 0): "west",
        (0, 2): "south_east",
        (0, 1): "south",
        (0, 0): "south_west",
    }
    return grid[(row, col)]


def _grade(score: float) -> str:
    if score >= 92:
        return "A-plus"
    if score >= 85:
        return "A"
    if score >= 75:
        return "B-plus"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def agent_design_intelligence(
    layout_with_openings: Dict[str, Any],
    spatial_brief: Dict[str, Any],
    payload: Dict[str, Any],
    geometry_report: Dict[str, Any],
) -> Dict[str, Any]:
    rooms = layout_with_openings["rooms"]
    doors = layout_with_openings["doors"]
    windows = layout_with_openings["windows"]
    paths = layout_with_openings["paths"]
    circulation_score = int(layout_with_openings["circulation_score"])

    usable_width = float(spatial_brief["plot"]["usable_width"])
    usable_length = float(spatial_brief["plot"]["usable_length"])
    usable_area = max(1.0, usable_width * usable_length)

    by_id = {r["id"]: r for r in rooms}
    vastu_enabled = bool(payload.get("vastu", True))
    vastu_score = 70 if vastu_enabled else 85
    vastu_bonuses: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    if vastu_enabled:
        for room in rooms:
            zone = _room_zone_9(room, usable_width, usable_length)
            rtype = room["type"]
            if rtype == "kitchen" and zone == "south_east":
                vastu_score += 10
                vastu_bonuses.append({"room": room["id"], "zone": zone, "message": "Kitchen in south-east"})
            if rtype == "master_bedroom" and zone == "south_west":
                vastu_score += 8
                vastu_bonuses.append({"room": room["id"], "zone": zone, "message": "Master bedroom in south-west"})
            if rtype == "pooja" and zone == "north_east":
                vastu_score += 10
                vastu_bonuses.append({"room": room["id"], "zone": zone, "message": "Pooja in north-east"})
            if rtype == "living" and zone in {"north", "east", "north_east"}:
                vastu_score += 6
                vastu_bonuses.append({"room": room["id"], "zone": zone, "message": "Living in north/east sector"})
            if rtype in {"bathroom", "toilet"} and zone == "north_east":
                vastu_score -= 15
                issues.append({"severity": "critical", "message": "Bathroom/toilet in north-east"})
            if rtype == "kitchen" and zone == "north_east":
                vastu_score -= 12
                issues.append({"severity": "critical", "message": "Kitchen in north-east"})
            if rtype == "staircase" and zone == "center":
                vastu_score -= 10
                issues.append({"severity": "warning", "message": "Staircase in center"})

    vastu_score = max(0, min(100, vastu_score))

    nbc_score = 100
    for room in rooms:
        area = room["area"]
        if room["type"] == "living" and area < 120:
            nbc_score -= 15
            issues.append({"severity": "critical", "message": "Living below 120 sqft NBC minimum"})
        if room["type"] == "master_bedroom" and area < 120:
            nbc_score -= 15
            issues.append({"severity": "critical", "message": "Master bedroom below 120 sqft NBC minimum"})
        if room["type"] == "bedroom" and area < 100:
            nbc_score -= 10
            issues.append({"severity": "warning", "message": "Bedroom below 100 sqft NBC recommendation"})
        if room["type"] == "kitchen" and area < 50:
            nbc_score -= 10
            issues.append({"severity": "critical", "message": "Kitchen below 50 sqft minimum"})
        if room["type"] == "bathroom" and area < 30:
            nbc_score -= 8
            issues.append({"severity": "warning", "message": "Bathroom below 30 sqft"})
    nbc_score = max(0, nbc_score)

    natural_light = 100
    room_windows = {}
    for w in windows:
        rid = w.get("room_id")
        room_windows[rid] = room_windows.get(rid, 0) + 1
    for room in rooms:
        if room["type"] not in HABITABLE_TYPES:
            continue
        if room_windows.get(room["id"], 0) <= 0:
            natural_light -= 15
            issues.append({"severity": "warning", "message": f"No window for habitable room {room['label']}"})
    natural_light = max(0, natural_light)

    privacy = 100
    main_door = next((d for d in doors if d.get("type") == "main"), None)
    living = next((r for r in rooms if r["type"] == "living"), None)
    if main_door and living:
        for room in rooms:
            if room["type"] not in {"master_bedroom", "bedroom", "study", "bathroom", "toilet"}:
                continue
            if _adjacent(living, room):
                privacy -= 8
                issues.append({"severity": "warning", "message": f"Privacy gradient weak near entrance for {room['label']}"})
    privacy = max(0, privacy)

    named_area = sum(r["area"] for r in rooms)
    coverage = named_area / usable_area
    if 0.85 <= coverage <= 0.92:
        space_eff = 90 + min(10, (coverage - 0.85) * 100)
    elif coverage < 0.85:
        space_eff = max(0.0, 90 - (0.85 - coverage) * 200)
    else:
        space_eff = max(70.0, 95 - (coverage - 0.92) * 120)
    space_eff = _r2(space_eff)

    circulation = max(0, min(100, circulation_score))

    # Weighted composite requested by product spec.
    composite = (
        vastu_score * 0.25
        + space_eff * 0.20
        + circulation * 0.15
        + natural_light * 0.15
        + nbc_score * 0.15
        + privacy * 0.10
    )
    composite = _r2(composite)

    breakdown = {
        "vastu": {"score": int(round(vastu_score)), "weight": 0.25},
        "nbc": {"score": int(round(nbc_score)), "weight": 0.15},
        "circulation": {"score": int(round(circulation)), "weight": 0.15},
        "natural_light": {"score": int(round(natural_light)), "weight": 0.15},
        "privacy": {"score": int(round(privacy)), "weight": 0.10},
        "space_efficiency": {"score": int(round(space_eff)), "weight": 0.20},
    }

    total_overlap = float(geometry_report["validation_report"].get("total_overlap", 0.0))
    living_area = max((r["area"] for r in rooms if r["type"] == "living"), default=0.0)
    master_area = max((r["area"] for r in rooms if r["type"] == "master_bedroom"), default=0.0)
    kitchen_has_exterior = any(
        r["type"] == "kitchen"
        and (
            abs(r["x"] - 0.0) < 0.01
            or abs(r["y"] - 0.0) < 0.01
            or abs((r["x"] + r["width"]) - usable_width) < 0.01
            or abs((r["y"] + r["height"]) - usable_length) < 0.01
        )
        for r in rooms
    )

    bedroom_landlocked = False
    for r in rooms:
        if r["type"] not in {"master_bedroom", "bedroom"}:
            continue
        touches = (
            abs(r["x"] - 0.0) < 0.01
            or abs(r["y"] - 0.0) < 0.01
            or abs((r["x"] + r["width"]) - usable_width) < 0.01
            or abs((r["y"] + r["height"]) - usable_length) < 0.01
        )
        if not touches:
            bedroom_landlocked = True
            break

    compact_plot = usable_area < 900.0
    living_min = 90.0 if compact_plot else 100.0
    master_min = 90.0 if compact_plot else 100.0
    overlap_limit = 8.0 if compact_plot else 5.0
    circulation_min = 45 if compact_plot else 50
    nbc_min = 55 if compact_plot else 60

    reject_reason = None
    if vastu_enabled and vastu_score < 40:
        reject_reason = "Vastu score below 40"
    elif not kitchen_has_exterior:
        reject_reason = "Kitchen has no exterior wall"
    elif bedroom_landlocked and not compact_plot:
        reject_reason = "Bedroom is landlocked"
    elif master_area < master_min:
        reject_reason = f"Master bedroom below {int(master_min)} sqft"
    elif living_area < living_min:
        reject_reason = f"Living room below {int(living_min)} sqft"
    elif total_overlap > overlap_limit:
        reject_reason = f"Total overlap exceeds {int(overlap_limit)} sqft"
    elif circulation < circulation_min:
        reject_reason = f"Circulation score below {circulation_min}"
    elif nbc_score < nbc_min:
        reject_reason = f"NBC score below {nbc_min}"
    elif natural_light < 55:
        reject_reason = "Natural light score below 55"

    family = _norm_family(payload.get("family_type"))
    narrative = (
        f"This layout balances social and private zones for a {family.replace('-', ' ')} household on a "
        f"{spatial_brief['plot']['usable_width']:.1f} x {spatial_brief['plot']['usable_length']:.1f} ft usable footprint. "
        f"The strategy improves circulation between living, dining, and kitchen while preserving exterior access for light and ventilation."
    )

    return {
        "accept": reject_reason is None,
        "reject_reason": reject_reason,
        "scores": {
            "vastu": int(round(vastu_score)),
            "nbc": int(round(nbc_score)),
            "circulation": int(round(circulation)),
            "natural_light": int(round(natural_light)),
            "privacy": int(round(privacy)),
            "space_efficiency": int(round(space_eff)),
            "composite": composite,
        },
        "breakdown": breakdown,
        "grade": _grade(composite),
        "vastu_bonuses": vastu_bonuses,
        "issues": issues,
        "architect_narrative": narrative,
        "paths": paths,
    }


def _layout_signature(rooms: List[Dict[str, Any]], bhk: int, strategy: str) -> str:
    tokens = []
    for r in sorted(rooms, key=lambda x: x["id"]):
        tokens.append(
            f"{r['id']}:{r['x']:.2f},{r['y']:.2f},{r['width']:.2f},{r['height']:.2f}"
        )
    payload = f"{bhk}|{strategy}|" + "|".join(tokens)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{bhk}bhk-{digest}"


@dataclass
class _AttemptResult:
    layout: Dict[str, Any]
    score: float
    accept: bool
    reject_reason: Optional[str]


def generate_dynamic_layout(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = _sanitize_payload(payload)
    payload, feasibility_notes = _normalize_payload_for_feasibility(payload)

    band_profile: Optional[Dict[str, float]] = None
    spine_override: Optional[str] = None
    try:
        spatial = agent_plot_intelligence(payload, band_profile=band_profile, spine_override=spine_override)
    except ValueError as exc:
        return {
            "error": str(exc),
            "input_normalized": payload,
        }
    if spatial["plot"]["usable_width"] < 12.0:
        return {
            "error": "Plot too narrow after setbacks (usable width < 12 ft)",
            "spatial_brief": spatial,
        }

    bhk = max(1, min(4, int(payload.get("bedrooms") or 2)))
    forced_strategy: Optional[str] = payload.get("previous_strategy")

    best: Optional[_AttemptResult] = None
    rejection_count = 0
    strategy_used = None
    service_first = False
    drop_lowest_extra = False
    enable_swap_non_critical = False
    attempt_scores: List[float] = []
    failure_history: List[Dict[str, Any]] = []

    def _swap_non_critical_same_band(rooms: List[Dict[str, Any]]) -> bool:
        candidates = [
            r for r in rooms if r.get("type") not in {"living", "master_bedroom", "kitchen", "pooja"}
        ]
        by_band: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}
        for room in candidates:
            by_band[int(room.get("band", 2))].append(room)
        for band in [2, 1, 3]:
            bucket = sorted(by_band.get(band, []), key=lambda r: int(r.get("priority", 1)))
            if len(bucket) < 2:
                continue
            a = bucket[0]
            b = bucket[-1]
            a["x"], b["x"] = b["x"], a["x"]
            a["y"], b["y"] = b["y"], a["y"]
            a["width"], b["width"] = b["width"], a["width"]
            a["height"], b["height"] = b["height"], a["height"]
            a["area"] = _r2(a["width"] * a["height"])
            b["area"] = _r2(b["width"] * b["height"])
            a["polygon"] = [
                [_r2(a["x"]), _r2(a["y"])],
                [_r2(a["x"] + a["width"]), _r2(a["y"])],
                [_r2(a["x"] + a["width"]), _r2(a["y"] + a["height"])],
                [_r2(a["x"]), _r2(a["y"] + a["height"])],
            ]
            b["polygon"] = [
                [_r2(b["x"]), _r2(b["y"])],
                [_r2(b["x"] + b["width"]), _r2(b["y"])],
                [_r2(b["x"] + b["width"]), _r2(b["y"] + b["height"])],
                [_r2(b["x"]), _r2(b["y"] + b["height"])],
            ]
            return True
        return False

    def _drop_lowest_priority_extra(program: Dict[str, Any]) -> Optional[str]:
        core = {"living", "dining", "kitchen", "master_bedroom", "bedroom", "bathroom", "toilet", "pooja"}
        extras = [r for r in program.get("rooms", []) if r.get("type") not in core]
        if not extras:
            return None
        victim = sorted(extras, key=lambda r: int(r.get("priority", 1)))[0]
        victim_area = float(victim.get("target_area", 0.0))
        neighbors = [r for r in program.get("rooms", []) if r.get("id") in (victim.get("adjacent_to") or [])]
        if neighbors and victim_area > 0:
            bump = victim_area / len(neighbors)
            for n in neighbors:
                n["target_area"] = _r2(float(n.get("target_area", 0.0)) + bump)
        program["rooms"] = [r for r in program.get("rooms", []) if r.get("id") != victim.get("id")]
        return str(victim.get("id"))

    for attempt in range(1, 6):
        program_input = dict(payload)
        if drop_lowest_extra:
            extras = _norm_extras(program_input.get("extras"))
            if extras:
                program_input["extras"] = extras[:-1]

        program = agent_program_architect(spatial, program_input, forced_strategy=forced_strategy)
        dropped_room = None
        if drop_lowest_extra:
            dropped_room = _drop_lowest_priority_extra(program)
        strategy_used = program["layout_strategy"]
        geometry = agent_geometry_engine(program, spatial, service_first=service_first)
        if enable_swap_non_critical:
            _swap_non_critical_same_band(geometry["rooms"])
        connectivity = agent_connectivity_doors(geometry, spatial, program)
        scoring = agent_design_intelligence(connectivity, spatial, payload, geometry)

        signature = _layout_signature(connectivity["rooms"], bhk, strategy_used)
        design_score = {
            "composite": scoring["scores"]["composite"],
            "composite_score": scoring["scores"]["composite"],
            "grade": scoring["grade"],
            "breakdown": scoring["breakdown"],
            "per_dimension": scoring["breakdown"],
            "issues": scoring["issues"],
            "vastu_bonuses": scoring["vastu_bonuses"],
            "layout_signature": signature,
            "family_type": _norm_family(payload.get("family_type")),
            "climate_zone": spatial.get("climate_zone"),
        }

        layout = {
            "plot": {
                "width": spatial["plot"]["width"],
                "length": spatial["plot"]["length"],
                "usable_width": spatial["plot"]["usable_width"],
                "usable_length": spatial["plot"]["usable_length"],
                "setbacks": spatial["plot"]["setbacks"],
                "road_side": spatial["road_side"],
            },
            "rooms": connectivity["rooms"],
            "doors": connectivity["doors"],
            "windows": connectivity["windows"],
            "design_score": design_score,
            "grade": scoring["grade"],
            "architect_narrative": scoring["architect_narrative"],
            "vastu_bonuses": scoring["vastu_bonuses"],
            "issues": [i.get("message", "") for i in scoring["issues"]],
            "connectivity_checks": connectivity["paths"],
            "layout_signature": signature,
            "bhk": bhk,
            "layout_strategy": strategy_used,
            "layout_type": strategy_used,
            "zoning_strategy": strategy_used,
            "spatial_brief": spatial,
            "room_program": program,
            "validation": geometry["validation_report"],
            "attempt": attempt,
            "constraint_notes": program.get("constraint_notes", []),
            "circulation": {
                "score": connectivity["circulation_score"],
                "paths": connectivity["paths"],
            },
            "connectivity_suggestions": connectivity.get("suggested_adjustments", []),
            "dropped_room": dropped_room,
        }

        result = _AttemptResult(
            layout=layout,
            score=float(scoring["scores"]["composite"]),
            accept=bool(scoring["accept"]),
            reject_reason=scoring.get("reject_reason"),
        )
        attempt_scores.append(result.score)

        if best is None or result.score > best.score:
            best = result

        if result.accept:
            break

        rejection_count += 1
        failed_checks = [
            key for key, ok in geometry.get("validation_report", {}).get("checks", {}).items() if not ok
        ]
        failed_paths = [
            key for key, ok in connectivity.get("paths", {}).items() if not ok
        ]
        reason = scoring.get("reject_reason") or "unknown"
        failure_history.append(
            {
                "attempt": attempt,
                "reason": reason,
                "failed_checks": failed_checks,
                "failed_paths": failed_paths,
            }
        )

        if rejection_count == 1:
            band_profile = {"public": 0.35, "service": 0.18}
            spatial = agent_plot_intelligence(payload, band_profile=band_profile, spine_override=spine_override)
        elif rejection_count == 2:
            enable_swap_non_critical = True
            service_first = True
        elif rejection_count == 3:
            drop_lowest_extra = True
        elif rejection_count == 4:
            current = str(spatial.get("spatial_contract", {}).get("primary_spine") or "horizontal")
            spine_override = "vertical" if current == "horizontal" else "horizontal"
            spatial = agent_plot_intelligence(payload, band_profile=band_profile, spine_override=spine_override)

    assert best is not None

    out = dict(best.layout)
    if feasibility_notes:
        out.setdefault("warnings", [])
        out["warnings"].extend(feasibility_notes)
    if not best.accept:
        out.setdefault("issues", [])
        out["issues"].append(
            f"Accepted best attempt after retries. Last reject reason: {best.reject_reason or 'unspecified'}"
        )
        out.setdefault("warnings", [])
        out["warnings"].append("Manual adjustment recommended after retry budget exhausted")
        out["partial_accept"] = True

    operational_accept = bool(best.accept)
    if not operational_accept and float(best.score) >= 70.0 and bool(out.get("partial_accept")):
        operational_accept = True

    out["retry_count"] = rejection_count
    out["strict_accepted"] = bool(best.accept)
    out["accepted"] = operational_accept
    out["attempt_scores"] = attempt_scores
    out["failure_history"] = failure_history
    out["design_score"] = out.get("design_score") or {
        "composite": best.score,
        "grade": out.get("grade", _grade(best.score)),
    }
    return out
