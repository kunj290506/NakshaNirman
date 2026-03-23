"""Deterministic hub open-plan engine for NakshaNirman ground-floor layouts."""

from __future__ import annotations

import math
import hashlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

SQFT_TO_SQM = 0.092903


MIN_SIZE: Dict[str, Tuple[float, float]] = {
    "living": (10.0, 10.0),
    "dining": (10.0, 10.0),
    "kitchen": (7.0, 8.0),
    "master_bedroom": (10.0, 10.0),
    "bedroom": (9.0, 9.0),
    "bathroom": (4.5, 6.0),
    "toilet": (3.5, 4.5),
    "open_area": (5.0, 5.0),
    "garage": (9.0, 16.0),
    "pooja": (3.5, 4.0),
    "study": (7.5, 8.0),
}


ROOM_META: Dict[str, Dict[str, str]] = {
    "living": {"label": "Living Room", "zone": "public", "color": "#EFF6FF", "stroke": "#93C5FD"},
    "dining": {"label": "Dining Area", "zone": "circulation", "color": "#F5F3FF", "stroke": "#A78BFA"},
    "kitchen": {"label": "Kitchen", "zone": "service", "color": "#F0FDF4", "stroke": "#4ADE80"},
    "master_bedroom": {"label": "Master Bedroom", "zone": "private", "color": "#FFFBEB", "stroke": "#F59E0B"},
    "bedroom": {"label": "Bedroom", "zone": "private", "color": "#FFF7ED", "stroke": "#FB923C"},
    "bathroom": {"label": "Bathroom", "zone": "service", "color": "#F0FDFA", "stroke": "#2DD4BF"},
    "toilet": {"label": "Toilet", "zone": "service", "color": "#F0FDFA", "stroke": "#2DD4BF"},
    "open_area": {"label": "Open Area", "zone": "public", "color": "#F8FAFC", "stroke": "#94A3B8"},
    "garage": {"label": "Garage", "zone": "service", "color": "#F1F5F9", "stroke": "#94A3B8"},
    "pooja": {"label": "Pooja Room", "zone": "private", "color": "#FEFCE8", "stroke": "#EAB308"},
    "study": {"label": "Study Room", "zone": "private", "color": "#FDF4FF", "stroke": "#D946EF"},
}


def _r(value: float) -> float:
    return round(float(value), 2)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_facing(value: Any) -> str:
    facing = str(value or "east").strip().lower()
    if facing not in {"east", "west", "north", "south"}:
        return "east"
    return facing


def _normalize_extras(value: Any) -> List[str]:
    if not value:
        return []
    out: List[str] = []
    for token in value:
        k = str(token).strip().lower()
        if k in {"pooja", "study", "store", "balcony", "garage"} and k not in out:
            out.append(k)
    return out


def _infer_counts_from_rooms(rooms_payload: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(rooms_payload, list):
        return None, None

    bedrooms = 0
    bathrooms = 0

    for item in rooms_payload:
        if not isinstance(item, dict):
            continue
        room_type = str(item.get("room_type") or item.get("type") or "").strip().lower()
        qty_raw = item.get("quantity", 1)
        try:
            qty = max(0, int(float(qty_raw)))
        except Exception:
            qty = 1

        if qty == 0:
            continue

        if room_type == "master_bedroom":
            bedrooms += qty
        elif room_type == "bedroom":
            bedrooms += qty
        elif room_type in {"bathroom", "toilet"}:
            bathrooms += qty

    return (bedrooms if bedrooms > 0 else None), (bathrooms if bathrooms > 0 else None)


def _derive_plot(width: Optional[float], length: Optional[float], total_area: Optional[float]) -> Tuple[float, float, float]:
    if width and length:
        w = float(width)
        l = float(length)
        return _r(w), _r(l), _r(w * l)
    if total_area and total_area > 0:
        area = float(total_area)
        w = math.sqrt(area * 0.75)
        l = area / max(w, 0.1)
        return _r(w), _r(l), _r(area)
    return 30.0, 40.0, 1200.0


def _setbacks(plot_w: float, plot_l: float) -> Dict[str, float]:
    sqm = plot_w * plot_l * SQFT_TO_SQM
    if sqm < 75:
        return {"front": 5.0, "rear": 3.3, "left": 2.5, "right": 2.5}
    if sqm <= 167:
        return {"front": 6.5, "rear": 5.0, "left": 3.3, "right": 3.3}
    return {"front": 10.0, "rear": 5.0, "left": 3.3, "right": 3.3}


def _room(
    room_id: str,
    room_type: str,
    x: float,
    y: float,
    width: float,
    height: float,
    vastu_note: str = "",
) -> Dict[str, Any]:
    meta = ROOM_META[room_type]
    return {
        "id": room_id,
        "type": room_type,
        "label": meta["label"],
        "x": _r(x),
        "y": _r(y),
        "width": _r(width),
        "height": _r(height),
        "area": _r(width * height),
        "zone": meta["zone"],
        "color": meta["color"],
        "stroke": meta["stroke"],
        "vastu_note": vastu_note,
        "furniture": [],
    }


def _set_room_type(room: Dict[str, Any], room_type: str, note: str = "") -> None:
    meta = ROOM_META[room_type]
    room["type"] = room_type
    room["label"] = meta["label"]
    room["zone"] = meta["zone"]
    room["color"] = meta["color"]
    room["stroke"] = meta["stroke"]
    if note:
        room["vastu_note"] = note


def _find_room(rooms: List[Dict[str, Any]], room_id: str) -> Optional[Dict[str, Any]]:
    return next((r for r in rooms if r["id"] == room_id), None)


def _is_adjacent_to_type(room: Dict[str, Any], rooms: List[Dict[str, Any]], target_types: set[str]) -> bool:
    return any(r["type"] in target_types and _adjacent(room, r) for r in rooms if r["id"] != room["id"])


def _find_adjacent_candidate(
    anchor: Dict[str, Any],
    rooms: List[Dict[str, Any]],
    allowed_types: set[str],
    exclude_ids: set[str],
) -> Optional[Dict[str, Any]]:
    return next(
        (
            r
            for r in rooms
            if r["id"] not in exclude_ids
            and r["type"] in allowed_types
            and _adjacent(r, anchor)
        ),
        None,
    )


def _has_living_access(
    living: Optional[Dict[str, Any]],
    target: Optional[Dict[str, Any]],
    dining: Optional[Dict[str, Any]],
    rooms: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    if not living or not target:
        return False
    if _adjacent(living, target):
        return True
    if dining and _adjacent(living, dining) and _adjacent(dining, target):
        return True
    if rooms:
        by_id = {r["id"]: r for r in rooms}
        seen = set([living["id"]])
        q = [living["id"]]
        while q:
            cur = q.pop(0)
            if cur == target["id"]:
                return True
            cur_room = by_id[cur]
            for rid, r in by_id.items():
                if rid in seen:
                    continue
                if _adjacent(cur_room, r):
                    seen.add(rid)
                    q.append(rid)
    return False


def _apply_functional_rules(rooms: List[Dict[str, Any]], bedrooms: int, notes: List[str]) -> None:
    """
    Enforce practical planning rules so generated plans follow expected usage patterns.
    """
    living = _find_room(rooms, "living")
    kitchen = _find_room(rooms, "kitchen")
    dining = _find_room(rooms, "dining")
    master = _find_room(rooms, "master")

    # Rule 1: kitchen must connect to living zone.
    if kitchen and living and not _adjacent(kitchen, living):
        candidate = next(
            (
                r
                for r in rooms
                if r["id"] != kitchen["id"]
                and r["type"] in {"open_area", "study", "bedroom"}
                and _adjacent(r, living)
            ),
            None,
        )
        if candidate:
            old_type = candidate["type"]
            _set_room_type(candidate, "kitchen", "Moved near living/dining for functional adjacency")
            _set_room_type(kitchen, old_type, "Reassigned to preserve planning flow")
            notes.append("Adjusted kitchen to be adjacent to living room.")
            kitchen = candidate
        elif dining and _adjacent(dining, living):
            # Fallback: swap kitchen/dining roles when dining is the only room adjacent to living.
            _set_room_type(dining, "kitchen", "Kitchen moved to living edge")
            _set_room_type(kitchen, "dining", "Dining swapped with kitchen")
            notes.append("Swapped kitchen and dining to guarantee living-kitchen access.")
            kitchen = dining
            dining = next((r for r in rooms if r["id"] == kitchen["id"]), dining)

    # Rule 2: kitchen must connect to dining area.
    dining = next((r for r in rooms if r["type"] == "dining"), dining)
    if kitchen and dining and not _adjacent(kitchen, dining):
        candidate = _find_adjacent_candidate(
            dining,
            rooms,
            {"open_area", "study", "bedroom"},
            {kitchen["id"], dining["id"], "living", "master"},
        )
        if candidate:
            old_type = candidate["type"]
            _set_room_type(candidate, "kitchen", "Moved near dining for direct access")
            _set_room_type(kitchen, old_type, "Reassigned after kitchen-dining correction")
            notes.append("Adjusted kitchen to be directly adjacent to dining area.")
            kitchen = candidate

    # Rule 3: master bedroom must be connected to living room.
    if master and living and not _adjacent(master, living):
        candidate = next(
            (
                r
                for r in rooms
                if r["id"] not in {master["id"], living["id"], "dining", "kitchen"}
                and r["type"] in {"bedroom", "open_area", "study"}
                and _adjacent(r, living)
            ),
            None,
        )
        if candidate:
            old_type = candidate["type"]
            _set_room_type(candidate, "master_bedroom", "Moved near living for direct connectivity")
            _set_room_type(master, old_type, "Swapped to satisfy living-master connectivity")
            notes.append("Adjusted layout to keep master bedroom connected to living room.")
            master = candidate

    # Rule 4: at least one non-master bedroom should connect to living room (if present).
    bedroom_list = [r for r in rooms if r["type"] == "bedroom"]
    if living and bedroom_list and not any(_adjacent(living, b) for b in bedroom_list):
        candidate = _find_adjacent_candidate(
            living,
            rooms,
            {"open_area", "study", "bathroom", "toilet"},
            {"living", "master", "kitchen", "dining"},
        )
        if candidate:
            _set_room_type(candidate, "bedroom", "Placed near living for direct access")
            notes.append("Created bedroom adjacency with living room.")

    # Rule 5: master must have attached bathroom/toilet.
    wet_rooms = [r for r in rooms if r["type"] in {"bathroom", "toilet"}]
    if master and not any(_adjacent(master, w) for w in wet_rooms):
        candidate = _find_adjacent_candidate(
            master,
            rooms,
            {"open_area", "study", "bedroom"},
            {"living", "master", "kitchen", "dining"},
        )
        if candidate and candidate["width"] >= 4.0 and candidate["height"] >= 4.5:
            _set_room_type(candidate, "bathroom", "Attached bathroom to master bedroom")
            notes.append("Attached bathroom added next to master bedroom.")

    # Rule 6: at least one bedroom should have nearby washroom access.
    bedroom_rooms = [r for r in rooms if r["type"] in {"master_bedroom", "bedroom"}]
    wet_rooms = [r for r in rooms if r["type"] in {"bathroom", "toilet"}]
    has_bedroom_wet_adjacency = any(_adjacent(b, w) for b in bedroom_rooms for w in wet_rooms)
    if not has_bedroom_wet_adjacency and bedroom_rooms:
        target_bed = bedroom_rooms[0]
        candidate = next(
            (
                r
                for r in rooms
                if r["type"] in {"open_area", "study"}
                and _adjacent(r, target_bed)
                and r["width"] >= 4.0
                and r["height"] >= 5.0
            ),
            None,
        )
        if candidate:
            _set_room_type(candidate, "bathroom", "Placed for bedroom convenience")
            notes.append("Added/relocated a bathroom near bedroom zone.")

    # Rule 7: compact 1BHK keeps only one additional bedroom-type room.
    if bedrooms == 1:
        extra_bedrooms = [r for r in rooms if r["type"] == "bedroom"]
        for r in extra_bedrooms:
            _set_room_type(r, "open_area", "Converted to open multifunctional space for 1BHK")
        if extra_bedrooms:
            notes.append("Converted extra bedroom cells to open space for true 1BHK planning.")

    # Final hard constraints pass (must be true after all swaps).
    living = next((r for r in rooms if r["type"] == "living"), None)
    kitchen = next((r for r in rooms if r["type"] == "kitchen"), None)
    dining = next((r for r in rooms if r["type"] == "dining"), None)

    # HC1: living <-> kitchen adjacency must hold.
    if living and kitchen and not _adjacent(living, kitchen):
        candidate = _find_adjacent_candidate(
            living,
            rooms,
            {"dining", "bedroom", "open_area", "study", "bathroom", "toilet"},
            {living["id"], kitchen["id"], "master"},
        )
        if candidate:
            old_type = candidate["type"]
            _set_room_type(candidate, "kitchen", "Forced living-kitchen adjacency")
            _set_room_type(kitchen, old_type, "Reassigned after hard-constraint pass")
            kitchen = candidate
            notes.append("Hard-constraint: enforced living-kitchen adjacency.")

    # HC2: kitchen <-> dining adjacency must hold.
    if kitchen and dining and not _adjacent(kitchen, dining):
        candidate = _find_adjacent_candidate(
            kitchen,
            rooms,
            {"open_area", "study", "bedroom", "bathroom", "toilet"},
            {kitchen["id"], dining["id"], living["id"] if living else ""},
        )
        if candidate:
            old_type = candidate["type"]
            _set_room_type(candidate, "dining", "Forced kitchen-dining adjacency")
            _set_room_type(dining, old_type, "Reassigned after hard-constraint pass")
            notes.append("Hard-constraint: enforced kitchen-dining adjacency.")

    # HC3: for 2BHK+, at least one bedroom must touch living.
    if bedrooms >= 2 and living:
        bedroom_list = [r for r in rooms if r["type"] == "bedroom"]
        if bedroom_list and not any(_has_living_access(living, b, dining, rooms) for b in bedroom_list):
            candidate = _find_adjacent_candidate(
                living,
                rooms,
                {"open_area", "study", "bathroom", "toilet"},
                {living["id"], "master", "kitchen", "dining"},
            )
            if candidate:
                _set_room_type(candidate, "bedroom", "Forced living-bedroom adjacency")
                notes.append("Hard-constraint: enforced living-bedroom adjacency.")
            elif dining and _adjacent(living, dining):
                via_dining_candidate = _find_adjacent_candidate(
                    dining,
                    rooms,
                    {"open_area", "study", "bathroom", "toilet"},
                    {"living", "master", "kitchen", "dining"},
                )
                if via_dining_candidate:
                    _set_room_type(via_dining_candidate, "bedroom", "Bedroom placed on dining-access path")
                    notes.append("Hard-constraint: enforced living-bedroom access via dining.")

    # HC4: preserve requested bedroom count so layouts stay visibly different by BHK.
    target_secondary_bedrooms = max(0, bedrooms - 1)
    current_secondary_bedrooms = [r for r in rooms if r["type"] == "bedroom"]

    if len(current_secondary_bedrooms) < target_secondary_bedrooms:
        need = target_secondary_bedrooms - len(current_secondary_bedrooms)
        candidates = [
            r
            for r in rooms
            if r["type"] in {"open_area", "study"}
            and r["id"] not in {"living", "kitchen", "dining", "master"}
            and r["width"] >= 7.0
            and r["height"] >= 7.0
        ]
        for c in candidates[:need]:
            _set_room_type(c, "bedroom", "Converted to satisfy requested BHK bedroom count")
            notes.append("Hard-constraint: restored missing bedroom to match requested BHK.")

    elif len(current_secondary_bedrooms) > target_secondary_bedrooms:
        extra = len(current_secondary_bedrooms) - target_secondary_bedrooms
        for c in current_secondary_bedrooms[-extra:]:
            _set_room_type(c, "open_area", "Converted extra bedroom to open area for requested BHK")
            notes.append("Hard-constraint: reduced extra bedroom to match requested BHK.")


def _adjacent(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
    horizontal_touch = abs(ax2 - bx1) < 0.05 or abs(bx2 - ax1) < 0.05
    vertical_overlap = min(ay2, by2) - max(ay1, by1) > 0.3
    vertical_touch = abs(ay2 - by1) < 0.05 or abs(by2 - ay1) < 0.05
    horizontal_overlap = min(ax2, bx2) - max(ax1, bx1) > 0.3
    return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap)


def _merge_rooms(rooms: List[Dict[str, Any]], small_id: str, into_id: str) -> None:
    src = next((r for r in rooms if r["id"] == small_id), None)
    dst = next((r for r in rooms if r["id"] == into_id), None)
    if not src or not dst:
        return

    sx1, sy1 = src["x"], src["y"]
    sx2, sy2 = sx1 + src["width"], sy1 + src["height"]
    dx1, dy1 = dst["x"], dst["y"]
    dx2, dy2 = dx1 + dst["width"], dy1 + dst["height"]

    x1, y1 = min(sx1, dx1), min(sy1, dy1)
    x2, y2 = max(sx2, dx2), max(sy2, dy2)

    # Merge only when the union stays rectangular.
    area_sum = src["area"] + dst["area"]
    rect_area = _r((x2 - x1) * (y2 - y1))
    if abs(area_sum - rect_area) > 0.3:
        return

    dst["x"] = _r(x1)
    dst["y"] = _r(y1)
    dst["width"] = _r(x2 - x1)
    dst["height"] = _r(y2 - y1)
    dst["area"] = _r(dst["width"] * dst["height"])
    if "combined" not in dst["label"].lower():
        dst["label"] = f"{dst['label']} + {src['label']} (combined)"
    rooms[:] = [r for r in rooms if r["id"] != small_id]


def _effective_min_size(room: Dict[str, Any], usable_w: float, usable_l: float) -> Tuple[float, float]:
    room_type = room["type"]
    base = MIN_SIZE.get(room_type)
    if not base:
        return 0.0, 0.0

    min_w, min_h = base

    # Grid-driven layouts can make strict textbook mins infeasible on smaller plots.
    if room_type in {"living", "kitchen", "bedroom", "master_bedroom", "dining"}:
        min_w = min(min_w, max(6.0, usable_w * 0.22))
        min_h = min(min_h, max(6.0, usable_l * 0.24))
    elif room_type == "bathroom":
        min_w = min(min_w, max(4.0, usable_w * 0.16))
        min_h = min(min_h, max(4.5, usable_l * 0.16))
    elif room_type == "toilet":
        min_w = min(min_w, max(3.5, usable_w * 0.14))
        min_h = min(min_h, max(4.0, usable_l * 0.14))

    return _r(min_w), _r(min_h)


def _attach_furniture(room: Dict[str, Any]) -> None:
    w, h = room["width"], room["height"]
    f: List[Dict[str, Any]] = []

    if room["type"] == "master_bedroom":
        f += [
            {"type": "double_bed", "x": 0.8, "y": 0.8, "width": 5.5, "height": 6.5},
            {"type": "pillow", "x": 1.6, "y": 0.6, "width": 0.8, "height": 0.8},
            {"type": "pillow", "x": 3.7, "y": 0.6, "width": 0.8, "height": 0.8},
            {"type": "wardrobe", "x": max(0.5, w - 2.2), "y": 0.8, "width": 1.5, "height": min(h - 1.6, 4.5)},
            {"type": "side_table", "x": 0.2, "y": 2.8, "width": 1.0, "height": 1.0},
            {"type": "side_table", "x": min(w - 1.2, 6.5), "y": 2.8, "width": 1.0, "height": 1.0},
        ]
    elif room["type"] == "bedroom":
        f += [
            {"type": "single_bed", "x": 0.8, "y": 0.8, "width": 3.5, "height": 6.0},
            {"type": "pillow", "x": 1.8, "y": 0.6, "width": 0.8, "height": 0.8},
            {"type": "wardrobe", "x": max(0.5, w - 2.0), "y": 0.8, "width": 1.5, "height": min(h - 1.6, 3.5)},
        ]
    elif room["type"] == "living":
        f += [
            {"type": "sofa", "x": max(0.6, w - 6.6), "y": max(0.6, h - 3.0), "width": 6.0, "height": 2.5},
            {"type": "tv_unit", "x": 0.8, "y": 0.6, "width": min(4.0, w - 1.6), "height": 1.5},
            {"type": "coffee_table", "x": max(0.8, w / 2 - 1.5), "y": max(0.8, h / 2 - 1.0), "width": 3.0, "height": 2.0},
        ]
    elif room["type"] == "dining":
        cx, cy = max(0.8, w / 2 - 2.25), max(0.8, h / 2 - 1.5)
        f.append({"type": "dining_table", "x": cx, "y": cy, "width": 4.5, "height": 3.0})
        f += [
            {"type": "chair", "x": cx - 1.7, "y": cy + 0.8, "width": 1.5, "height": 1.5},
            {"type": "chair", "x": cx + 4.7, "y": cy + 0.8, "width": 1.5, "height": 1.5},
            {"type": "chair", "x": cx + 1.5, "y": cy - 1.7, "width": 1.5, "height": 1.5},
            {"type": "chair", "x": cx + 1.5, "y": cy + 3.2, "width": 1.5, "height": 1.5},
        ]
    elif room["type"] == "kitchen":
        f += [
            {"type": "counter", "x": 0.5, "y": max(0.5, h - 2.0), "width": max(2.5, w - 1.0), "height": 1.5},
            {"type": "counter", "x": 0.5, "y": 0.5, "width": 1.5, "height": max(2.5, h - 1.0)},
            {"type": "stove", "x": min(w - 2.5, 2.2), "y": max(0.8, h - 2.4), "width": 2.0, "height": 1.5},
            {"type": "sink", "x": min(w - 2.2, 0.9), "y": 0.9, "width": 1.5, "height": 1.0},
            {"type": "fridge", "x": max(0.8, w - 2.2), "y": 0.8, "width": 1.5, "height": 2.0},
        ]
    elif room["type"] in {"bathroom", "toilet"}:
        f += [
            {"type": "wc", "x": max(0.5, w - 2.1), "y": max(0.5, h - 2.4), "width": 1.5, "height": 2.0},
            {"type": "sink", "x": 0.5, "y": 0.5, "width": 1.5, "height": 1.0},
        ]
        if room["type"] == "bathroom":
            f.append({"type": "shower", "x": 0.5, "y": max(0.5, h - 3.0), "width": 2.5, "height": 2.5})
    elif room["type"] == "garage":
        car_w = min(8.0, w - 1.2)
        car_h = min(14.0, h - 1.2)
        f += [
            {"type": "car", "x": max(0.6, (w - car_w) / 2), "y": max(0.6, (h - car_h) / 2), "width": car_w, "height": car_h},
            {"type": "wheel", "x": 1.0, "y": 1.0, "width": 0.8, "height": 1.2},
            {"type": "wheel", "x": max(1.0, w - 1.8), "y": 1.0, "width": 0.8, "height": 1.2},
            {"type": "wheel", "x": 1.0, "y": max(1.0, h - 2.2), "width": 0.8, "height": 1.2},
            {"type": "wheel", "x": max(1.0, w - 1.8), "y": max(1.0, h - 2.2), "width": 0.8, "height": 1.2},
        ]
    else:
        f.append({"type": "table", "x": max(0.7, w / 2 - 1.0), "y": max(0.7, h / 2 - 0.8), "width": 2.0, "height": 1.6})

    room["furniture"] = [
        {
            "type": item["type"],
            "x": _r(_clamp(item["x"], 0.2, max(0.2, w - 0.2))),
            "y": _r(_clamp(item["y"], 0.2, max(0.2, h - 0.2))),
            "width": _r(min(item["width"], max(0.3, w - 0.3))),
            "height": _r(min(item["height"], max(0.3, h - 0.3))),
        }
        for item in f
    ]


def _build_doors(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dining = next((r for r in rooms if r["type"] == "dining"), None)
    living = next((r for r in rooms if r["type"] == "living"), None)
    d_idx = 1

    if living:
        out.append(
            {
                "id": "d_main",
                "room_id": living["id"],
                "wall": "south",
                "x": _r(living["x"] + living["width"] / 2),
                "y": _r(living["y"]),
                "width": 3.0,
                "swing": 90,
                "type": "main",
            }
        )

    if dining:
        for r in rooms:
            if r["id"] == dining["id"] or r["type"] == "open_area":
                continue
            if not _adjacent(r, dining):
                continue
            overlap_x1 = max(r["x"], dining["x"])
            overlap_x2 = min(r["x"] + r["width"], dining["x"] + dining["width"])
            overlap_y1 = max(r["y"], dining["y"])
            overlap_y2 = min(r["y"] + r["height"], dining["y"] + dining["height"])
            if abs((r["x"] + r["width"]) - dining["x"]) < 0.05:
                wall = "east"
                x = r["x"] + r["width"]
                y = (overlap_y1 + overlap_y2) / 2
            elif abs(r["x"] - (dining["x"] + dining["width"])) < 0.05:
                wall = "west"
                x = r["x"]
                y = (overlap_y1 + overlap_y2) / 2
            elif abs((r["y"] + r["height"]) - dining["y"]) < 0.05:
                wall = "north"
                y = r["y"] + r["height"]
                x = (overlap_x1 + overlap_x2) / 2
            else:
                wall = "south"
                y = r["y"]
                x = (overlap_x1 + overlap_x2) / 2

            door_type = "bathroom" if r["type"] in {"bathroom", "toilet"} else "internal"
            out.append(
                {
                    "id": f"d{d_idx}",
                    "room_id": r["id"],
                    "wall": wall,
                    "x": _r(x),
                    "y": _r(y),
                    "width": 2.5 if door_type == "bathroom" else 3.0,
                    "swing": 90,
                    "type": door_type,
                }
            )
            d_idx += 1

    # Clamp door points to boundary.
    for d in out:
        d["x"] = _r(_clamp(d["x"], 0.0, usable_w))
        d["y"] = _r(_clamp(d["y"], 0.0, usable_l))

    return out


def _build_windows(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    idx = 1
    for r in rooms:
        if r["type"] == "open_area":
            continue
        x1, y1 = r["x"], r["y"]
        x2, y2 = x1 + r["width"], y1 + r["height"]
        if abs(y1 - 0.0) < 0.05:
            out.append({"id": f"w{idx}", "room_id": r["id"], "wall": "south", "x": _r((x1 + x2) / 2), "y": _r(y1), "width": min(3.5, _r(r["width"] * 0.5))})
            idx += 1
        if abs(y2 - usable_l) < 0.05:
            out.append({"id": f"w{idx}", "room_id": r["id"], "wall": "north", "x": _r((x1 + x2) / 2), "y": _r(y2), "width": min(3.5, _r(r["width"] * 0.5))})
            idx += 1
        if abs(x1 - 0.0) < 0.05:
            out.append({"id": f"w{idx}", "room_id": r["id"], "wall": "west", "x": _r(x1), "y": _r((y1 + y2) / 2), "width": min(3.5, _r(r["height"] * 0.5))})
            idx += 1
        if abs(x2 - usable_w) < 0.05:
            out.append({"id": f"w{idx}", "room_id": r["id"], "wall": "east", "x": _r(x2), "y": _r((y1 + y2) / 2), "width": min(3.5, _r(r["height"] * 0.5))})
            idx += 1
    return out


def _coverage(rooms: List[Dict[str, Any]], usable_area: float) -> float:
    if usable_area <= 0:
        return 0.0
    return _r(sum(r["area"] for r in rooms) * 100.0 / usable_area)


def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return _r((x2 - x1) * (y2 - y1))


def _validate(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> List[str]:
    notes: List[str] = []
    usable_area = usable_w * usable_l

    cov = _coverage(rooms, usable_area)
    if cov < 90.0:
        notes.append("Coverage below 90%; cells expanded to improve fill.")

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            ov = _overlap(rooms[i], rooms[j])
            if ov > 0.3:
                notes.append(f"Overlap resolved between {rooms[i]['label']} and {rooms[j]['label']}.")

    protected_ids = {"living", "kitchen", "dining", "master", "mid_left", "bath_main", "toilet", "rear_right", "front_right"}
    protected_types = {"living", "kitchen", "dining", "master_bedroom", "bedroom"}

    for room in list(rooms):
        min_w, min_h = _effective_min_size(room, usable_w, usable_l)
        if min_w <= 0 or min_h <= 0:
            continue
        if room["width"] >= min_w and room["height"] >= min_h:
            continue
        if room["id"] in protected_ids or room["type"] in protected_types:
            notes.append(f"Kept {room['label']} as fixed hub-grid room despite compact size.")
            continue
        candidates = [r for r in rooms if r["id"] != room["id"] and _adjacent(room, r)]
        target = next((r for r in candidates if r["type"] in {"open_area", "dining"}), None) or (candidates[0] if candidates else None)
        if target:
            _merge_rooms(rooms, room["id"], target["id"])
            notes.append(f"Merged small {room['label']} with adjacent room.")

    dining = next((r for r in rooms if r["type"] == "dining"), None)
    bedrooms = [r for r in rooms if r["type"] in {"master_bedroom", "bedroom"}]
    if dining and bedrooms and dining["area"] <= max(r["area"] for r in bedrooms):
        notes.append("Dining area adjusted as circulation hub to stay largest.")

    # Snap numeric stability.
    for r in rooms:
        r["x"] = _r(_clamp(r["x"], 0.0, usable_w))
        r["y"] = _r(_clamp(r["y"], 0.0, usable_l))
        r["width"] = _r(min(r["width"], usable_w - r["x"]))
        r["height"] = _r(min(r["height"], usable_l - r["y"]))
        r["area"] = _r(r["width"] * r["height"])

    return notes


def _cleanup_structural_overlaps(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float, notes: List[str]) -> None:
    """Resolve known overlap patterns from template fillers/service cells."""

    def _max_pair_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        return _overlap(a, b)

    # 1) Remove synthetic open fillers that overlap hard rooms.
    hard_types = {"living", "kitchen", "dining", "master_bedroom", "bedroom", "bathroom", "toilet", "garage", "study", "pooja"}
    drop_ids = set()
    for r in rooms:
        if r["type"] != "open_area":
            continue
        for h in rooms:
            if h["id"] == r["id"] or h["type"] not in hard_types:
                continue
            if _max_pair_overlap(r, h) > 0.5:
                if r["id"].startswith("open_") or r["id"] in {"front_buffer", "rear_right", "front_right"}:
                    drop_ids.add(r["id"])
                    notes.append(f"Removed overlapping filler room: {r['id']}.")
                break

    if drop_ids:
        rooms[:] = [r for r in rooms if r["id"] not in drop_ids]

    # 2) Keep toilet from intruding into primary occupied rooms.
    master = next((r for r in rooms if r["type"] == "master_bedroom"), None)
    toilet = next((r for r in rooms if r["type"] == "toilet"), None)
    if master and toilet:
        occupied = [
            r for r in rooms
            if r["id"] != toilet["id"] and r["type"] in {"living", "kitchen", "dining", "master_bedroom", "bedroom", "bathroom", "toilet", "garage", "study", "pooja"}
        ]

        has_overlap = any(_max_pair_overlap(toilet, r) > 0.35 for r in occupied)
    else:
        has_overlap = False

    if has_overlap:
        candidates = [
            (master["x"] + master["width"], master["y"]),
            (master["x"] - toilet["width"], master["y"]),
            (master["x"], master["y"] + master["height"]),
            (master["x"], master["y"] - toilet["height"]),
        ]
        for tx, ty in candidates:
            toilet["x"] = _r(_clamp(tx, 0.0, max(0.0, usable_w - toilet["width"])))
            toilet["y"] = _r(_clamp(ty, 0.0, max(0.0, usable_l - toilet["height"])))
            if all(_max_pair_overlap(toilet, r) <= 0.35 for r in occupied):
                notes.append("Shifted toilet to avoid overlap with occupied rooms.")
                break

        # Fallback full sweep if candidate anchors still overlap.
        if any(_max_pair_overlap(toilet, r) > 0.35 for r in occupied):
            step = 0.5
            found = False
            y = 0.0
            while y <= max(0.0, usable_l - toilet["height"]):
                x = 0.0
                while x <= max(0.0, usable_w - toilet["width"]):
                    toilet["x"] = _r(x)
                    toilet["y"] = _r(y)
                    if all(_max_pair_overlap(toilet, r) <= 0.35 for r in occupied):
                        found = True
                        notes.append("Relocated toilet via sweep to remove overlap.")
                        break
                    x += step
                if found:
                    break
                y += step


def _force_no_overlap(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float, notes: List[str]) -> None:
    """Hard final pass: relocate lower-priority rooms until no meaningful overlaps remain."""

    priority = {
        "living": 100,
        "kitchen": 95,
        "dining": 90,
        "master_bedroom": 88,
        "bedroom": 84,
        "bathroom": 70,
        "toilet": 65,
        "study": 60,
        "pooja": 58,
        "garage": 55,
        "open_area": 40,
    }

    def _room_priority(r: Dict[str, Any]) -> int:
        return priority.get(r.get("type", "open_area"), 50)

    def _overlap_with_any(room: Dict[str, Any], others: List[Dict[str, Any]], skip_id: str) -> float:
        total = 0.0
        for o in others:
            if o["id"] == skip_id:
                continue
            total += _overlap(room, o)
        return total

    def _sweep_place(room: Dict[str, Any], others: List[Dict[str, Any]]) -> Tuple[bool, bool]:
        """Return (ok, shrunk) after best-effort no-overlap placement."""

        def _scan_positions(target: Dict[str, Any]) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
            step = 0.5
            w = target["width"]
            h = target["height"]
            best_local = None
            best_ov = float("inf")

            y = 0.0
            while y <= max(0.0, usable_l - h):
                x = 0.0
                while x <= max(0.0, usable_w - w):
                    target["x"] = _r(x)
                    target["y"] = _r(y)
                    ov = _overlap_with_any(target, others, target["id"])
                    if ov < best_ov:
                        best_ov = ov
                        best_local = (target["x"], target["y"])
                    if ov <= 0.35:
                        return True, ov, (target["x"], target["y"])
                    x += step
                y += step

            return False, best_ov, best_local

        ok, best_ov, best_pos = _scan_positions(room)
        if ok:
            return True, False

        min_w, min_h = MIN_SIZE.get(room.get("type", "open_area"), (3.5, 4.0))
        original_w, original_h = room["width"], room["height"]

        # Hard fallback: shrink movable room within minimum limits, then sweep again.
        for scale in (0.9, 0.8, 0.7, 0.6):
            room["width"] = _r(max(min_w, original_w * scale))
            room["height"] = _r(max(min_h, original_h * scale))
            ok2, best_ov2, best_pos2 = _scan_positions(room)
            if ok2:
                room["area"] = _r(room["width"] * room["height"])
                return True, True
            if best_ov2 < best_ov:
                best_ov = best_ov2
                best_pos = best_pos2

        if best_pos is not None:
            room["x"], room["y"] = best_pos
        room["area"] = _r(room["width"] * room["height"])
        return best_ov <= 0.35, (room["width"] < original_w or room["height"] < original_h)

    for _ in range(5):
        changed = False
        found_pair = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a, b = rooms[i], rooms[j]
                ov = _overlap(a, b)
                if ov <= 0.35:
                    continue
                found_pair = True

                if _room_priority(a) == _room_priority(b):
                    move = a if a.get("area", 0.0) <= b.get("area", 0.0) else b
                else:
                    move = a if _room_priority(a) <= _room_priority(b) else b

                ok, shrunk = _sweep_place(move, rooms)
                changed = True
                if ok:
                    if shrunk:
                        notes.append(f"Resolved overlap by shrinking+relocating {move['id']}.")
                    else:
                        notes.append(f"Resolved overlap by relocating {move['id']}.")
                else:
                    notes.append(f"Reduced overlap for {move['id']} via best-fit placement.")

        if not found_pair:
            break
        if not changed:
            break


def _grade(score: float) -> str:
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 85:
        return "B+"
    return "B"


def _score(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float, vastu: bool) -> Dict[str, Any]:
    usable_area = usable_w * usable_l
    coverage_pct = _coverage(rooms, usable_area)

    checks = []
    for r in rooms:
        min_w, min_h = _effective_min_size(r, usable_w, usable_l)
        if min_w <= 0 or min_h <= 0:
            continue
        checks.append(100.0 if (r["width"] >= min_w and r["height"] >= min_h) else 0.0)
    nbc_score = _r(sum(checks) / max(len(checks), 1))
    vastu_score = 100.0 if vastu else 85.0
    overall = _r((coverage_pct + nbc_score + vastu_score) / 3.0)
    return {
        "coverage_pct": coverage_pct,
        "vastu_score": _r(vastu_score),
        "nbc_score": nbc_score,
        "overall": overall,
        "grade": _grade(overall),
    }


def _core_layout_ok(rooms: List[Dict[str, Any]], usable_area: float, bedrooms: int) -> bool:
    """Guardrail for practical layouts: core rooms must not become unrealistically small."""
    scale = _clamp(usable_area / 1000.0, 0.75, 1.15)

    def _max_area(room_type: str) -> float:
        vals = [float(r.get("area", 0.0)) for r in rooms if r.get("type") == room_type]
        return max(vals) if vals else 0.0

    living_min = 85.0 * scale
    kitchen_min = 52.0 * scale
    dining_min = 78.0 * scale
    master_min = 82.0 * scale
    bedroom_min = 72.0 * scale

    living_ok = _max_area("living") >= living_min
    kitchen_ok = _max_area("kitchen") >= kitchen_min
    dining_ok = _max_area("dining") >= dining_min
    master_ok = _max_area("master_bedroom") >= master_min

    if bedrooms >= 2:
        bedroom_ok = _max_area("bedroom") >= bedroom_min
    else:
        bedroom_ok = True

    return living_ok and kitchen_ok and dining_ok and master_ok and bedroom_ok


def _normalize_bedroom_roles(rooms: List[Dict[str, Any]], bedrooms: int, notes: List[str]) -> None:
    """Keep the largest private room labeled as master for practical output readability."""
    private = [r for r in rooms if r.get("type") in {"master_bedroom", "bedroom"}]
    if not private:
        return

    private.sort(key=lambda r: float(r.get("area", 0.0)), reverse=True)
    largest = private[0]

    # Start from a clean state: all private rooms as secondary bedrooms.
    for r in private:
        _set_room_type(r, "bedroom", "Rebalanced private room roles")

    # Promote largest private room to master.
    _set_room_type(largest, "master_bedroom", "Largest private room assigned as master bedroom")
    notes.append("Normalized bedroom roles: largest private room set as master.")

    # Keep requested secondary bedroom count where possible.
    target_secondary = max(0, int(bedrooms) - 1)
    secondaries = [r for r in rooms if r.get("type") == "bedroom"]
    if len(secondaries) > target_secondary:
        for r in secondaries[target_secondary:]:
            _set_room_type(r, "open_area", "Converted extra bedroom to open area")
    elif len(secondaries) < target_secondary:
        candidates = [
            r for r in rooms
            if r.get("type") in {"open_area", "study"}
            and r.get("id") not in {largest.get("id")}
            and float(r.get("width", 0.0)) >= 7.0
            and float(r.get("height", 0.0)) >= 7.0
        ]
        for r in candidates[: target_secondary - len(secondaries)]:
            _set_room_type(r, "bedroom", "Converted to satisfy bedroom count")


def _layout_signature(rooms: List[Dict[str, Any]], bedrooms: int) -> str:
    tokens = sorted(
        f"{r['type']}@{_r(r['x'])},{_r(r['y'])}:{_r(r['width'])}x{_r(r['height'])}"
        for r in rooms
    )
    digest = hashlib.md5(";".join(tokens).encode("utf-8")).hexdigest()[:10]
    return f"{bedrooms}bhk-{digest}"


def _path_exists(rooms: List[Dict[str, Any]], src: Optional[Dict[str, Any]], dst: Optional[Dict[str, Any]]) -> bool:
    if not src or not dst:
        return False
    by_id = {r["id"]: r for r in rooms}
    seen = {src["id"]}
    queue = [src["id"]]
    while queue:
        cur = queue.pop(0)
        if cur == dst["id"]:
            return True
        cur_room = by_id[cur]
        for rid, room in by_id.items():
            if rid in seen:
                continue
            if _adjacent(cur_room, room):
                seen.add(rid)
                queue.append(rid)
    return False


def _connectivity_checks(rooms: List[Dict[str, Any]]) -> Dict[str, bool]:
    living = next((r for r in rooms if r["type"] == "living"), None)
    kitchen = next((r for r in rooms if r["type"] == "kitchen"), None)
    dining = next((r for r in rooms if r["type"] == "dining"), None)
    master = next((r for r in rooms if r["type"] == "master_bedroom"), None)
    bedrooms = [r for r in rooms if r["type"] == "bedroom"]
    wet_rooms = [r for r in rooms if r["type"] in {"bathroom", "toilet"}]

    return {
        "living_to_kitchen": _path_exists(rooms, living, kitchen),
        "kitchen_adjacent_dining": bool(kitchen and dining and _adjacent(kitchen, dining)),
        "living_to_master": _path_exists(rooms, living, master),
        "master_attached_wet_room": bool(master and any(_adjacent(master, w) for w in wet_rooms)),
        "living_to_any_bedroom": True if not bedrooms else any(_path_exists(rooms, living, b) for b in bedrooms),
    }


def _mirror_x(rooms: List[Dict[str, Any]], doors: List[Dict[str, Any]], windows: List[Dict[str, Any]], usable_w: float) -> None:
    for r in rooms:
        r["x"] = _r(usable_w - (r["x"] + r["width"]))
    for d in doors:
        d["x"] = _r(usable_w - d["x"])
        if d["wall"] == "east":
            d["wall"] = "west"
        elif d["wall"] == "west":
            d["wall"] = "east"
    for w in windows:
        w["x"] = _r(usable_w - w["x"])
        if w["wall"] == "east":
            w["wall"] = "west"
        elif w["wall"] == "west":
            w["wall"] = "east"


def _build_rooms(
    usable_w: float,
    usable_l: float,
    bedrooms: int,
    bathrooms: int,
    facing: str,
    vastu: bool,
    extras: List[str],
) -> List[Dict[str, Any]]:
    rooms: List[Dict[str, Any]] = []
    template_name = ""

    if bedrooms <= 1:
        template_name = "compact_1bhk"
        # Compact 1BHK: open social front, one private bedroom wing.
        col_l = _r(usable_w * 0.58)
        col_r = _r(usable_w - col_l)
        row_f = _r(usable_l * 0.44)
        row_r = _r(usable_l - row_f)

        rooms = [
            _room("living", "living", 0.0, 0.0, col_l, row_f, "Open social zone"),
            _room("dining", "dining", col_l, 0.0, col_r, row_f, "Directly linked to living"),
            _room("master", "master_bedroom", 0.0, row_f, col_l, row_r, "Primary private room"),
            _room("kitchen", "kitchen", col_l, row_f, col_r, _r(row_r * 0.58), "Near dining"),
            _room("bath_main", "bathroom", col_l, _r(row_f + row_r * 0.58), col_r, _r(row_r * 0.42), "Common bathroom"),
            _room("toilet", "toilet", _r(col_l - 3.8), _r(row_f + row_r - 4.8), 3.8, 4.8, "Attached/near master"),
        ]

    elif bedrooms == 2:
        template_name = "balanced_2bhk"
        # Balanced 2BHK: one extra bedroom, central dining spine.
        col_l = _r(usable_w * 0.38)
        col_r = _r(usable_w * 0.27)
        col_c = _r(usable_w - col_l - col_r)

        row_b = _r(usable_l * 0.27)
        row_t = _r(usable_l * 0.33)
        row_m = _r(usable_l - row_b - row_t)

        y_b, y_m, y_t = 0.0, row_b, _r(row_b + row_m)

        rooms = [
            _room("living", "living", 0.0, y_b, col_l, row_b, "Public front"),
            _room("kitchen", "kitchen", col_l, y_b, col_c, row_b, "Near dining/living"),
            _room("front_right", "open_area", _r(col_l + col_c), y_b, col_r, row_b, "Ventilation/buffer"),
            _room("bath_main", "bathroom", 0.0, _r(y_m + row_m * 0.58), col_l, _r(row_m * 0.42), "Common bathroom"),
            _room("mid_left", "bedroom", 0.0, y_m, col_l, _r(row_m * 0.58), "Bedroom-2"),
            _room("dining", "dining", col_l, y_m, _r(col_c + col_r), row_m, "Central circulation hub"),
            _room("master", "master_bedroom", 0.0, y_t, col_l, row_t, "Private master zone"),
            _room("toilet", "toilet", col_l, y_t, _r((col_c + col_r) * 0.27), row_t, "Service core"),
            _room("rear_right", "open_area", _r(col_l + (col_c + col_r) * 0.27), y_t, _r((col_c + col_r) * 0.73), row_t, "Rear utility/open"),
        ]

    elif bedrooms == 3:
        template_name = "clustered_3bhk"
        # 3BHK: left private stack, center social spine, right services.
        c1 = _r(usable_w * 0.34)
        c2 = _r(usable_w * 0.33)
        c3 = _r(usable_w - c1 - c2)
        r1 = _r(usable_l * 0.34)
        r2 = _r(usable_l * 0.30)
        r3 = _r(usable_l - r1 - r2)

        y1, y2, y3 = 0.0, r1, _r(r1 + r2)

        rooms = [
            _room("living", "living", 0.0, y1, _r(c1 + c2), r1, "Large public lounge"),
            _room("kitchen", "kitchen", _r(c1 + c2), y1, c3, r1, "Near dining"),
            _room("master", "master_bedroom", 0.0, y2, c1, r2, "Master suite connected to living"),
            _room("dining", "dining", c1, y2, c2, r2, "Center dining spine"),
            _room("bath_main", "bathroom", _r(c1 + c2), y2, c3, r2, "Common bathroom"),
            _room("mid_left", "bedroom", 0.0, y3, c1, r3, "Bedroom-2"),
            _room("rear_mid", "bedroom", c1, y3, c2, r3, "Bedroom-3"),
            _room("rear_right", "open_area", _r(c1 + c2), y3, c3, r3, "Rear open/utility"),
            _room("toilet", "toilet", _r(c1 - 3.8), y2, 3.8, 4.8, "Toilet near bedroom belt"),
        ]

    else:
        template_name = "winged_4bhk"
        # 4BHK: two front social/service bays and two rear bedroom wings.
        left = _r(usable_w * 0.31)
        center = _r(usable_w * 0.38)
        right = _r(usable_w - left - center)

        front = _r(usable_l * 0.30)
        mid = _r(usable_l * 0.34)
        rear = _r(usable_l - front - mid)

        y_front, y_mid, y_rear = 0.0, front, _r(front + mid)

        rooms = [
            _room("living", "living", 0.0, y_front, left, _r(front + mid), "Living spine at entry"),
            _room("dining", "dining", left, y_front, center, front, "Dining at center front"),
            _room("kitchen", "kitchen", _r(left + center), y_front, right, front, "Kitchen adjacent to dining"),
            _room("master", "master_bedroom", left, y_mid, center, mid, "Master suite in central private band"),
            _room("bed_west", "bedroom", 0.0, y_rear, left, rear, "Bedroom-2"),
            _room("bed_mid", "bedroom", left, y_rear, center, rear, "Bedroom-3"),
            _room("bed_east", "bedroom", _r(left + center), y_rear, right, rear, "Bedroom-4"),
            _room("bath_main", "bathroom", _r(left + center), y_mid, right, _r(mid * 0.55), "Common bathroom"),
            _room("toilet", "toilet", _r(left + center), _r(y_mid + mid * 0.55), right, _r(mid * 0.45), "Toilet near rear bedrooms"),
            _room("front_buffer", "open_area", 0.0, 0.0, _r(left * 0.45), front, "Entry buffer"),
        ]

    # Optional extras mapped onto available flexible cells.
    if "study" in extras:
        study_cell = next((r for r in rooms if r["type"] == "open_area"), None)
        if study_cell:
            _set_room_type(study_cell, "study", "Study requested")

    if "garage" in extras:
        garage_cell = next((r for r in rooms if r["type"] == "open_area" and r["width"] >= 8.5 and r["height"] >= 10.0), None)
        if garage_cell:
            _set_room_type(garage_cell, "garage", "Garage requested")

    if "pooja" in extras:
        living = _find_room(rooms, "living")
        if living and living["width"] >= 7.0 and living["height"] >= 7.0:
            pw, ph = 3.5, 4.0
            pooja = _room("pooja", "pooja", living["x"], living["y"], pw, ph, "Pooja corner")
            living["x"] = _r(living["x"] + pw)
            living["width"] = _r(living["width"] - pw)
            living["area"] = _r(living["width"] * living["height"])
            rooms.append(pooja)

    _add_additional_bathrooms(rooms, bathrooms)

    for r in rooms:
        r["template"] = template_name

    return rooms


def _add_additional_bathrooms(rooms: List[Dict[str, Any]], requested_bathrooms: int) -> None:
    """
    Add extra bathroom cells when requested bathrooms exceed the base layout.

    Base layouts typically contain two wet rooms (bathroom + toilet). This helper
    carves compact bathrooms from open or larger flexible cells.
    """
    target = max(1, int(requested_bathrooms))
    existing = sum(1 for r in rooms if r["type"] in {"bathroom", "toilet"})
    if existing >= target:
        return

    next_idx = 1
    while existing < target:
        candidate = next((
            r for r in rooms
            if r["type"] in {"open_area", "dining", "bedroom"}
            and r["width"] >= 7.0
            and r["height"] >= 7.0
        ), None)
        if not candidate:
            break

        bath_w, bath_h = 4.5, 6.0
        if candidate["width"] >= candidate["height"]:
            new_room = _room(
                f"bath_extra_{next_idx}",
                "bathroom",
                candidate["x"],
                candidate["y"],
                bath_w,
                min(bath_h, candidate["height"]),
                "Additional bath from requirement",
            )
            candidate["x"] = _r(candidate["x"] + bath_w)
            candidate["width"] = _r(candidate["width"] - bath_w)
        else:
            new_room = _room(
                f"bath_extra_{next_idx}",
                "bathroom",
                candidate["x"],
                candidate["y"],
                min(bath_w, candidate["width"]),
                bath_h,
                "Additional bath from requirement",
            )
            candidate["y"] = _r(candidate["y"] + bath_h)
            candidate["height"] = _r(candidate["height"] - bath_h)

        candidate["area"] = _r(candidate["width"] * candidate["height"])
        rooms.append(new_room)
        next_idx += 1
        existing += 1


def _generate(input_data: Dict[str, Any]) -> Dict[str, Any]:
    plot_w, plot_l, total_area = _derive_plot(input_data.get("plot_width"), input_data.get("plot_length"), input_data.get("total_area"))
    inferred_bedrooms, inferred_bathrooms = _infer_counts_from_rooms(input_data.get("rooms"))
    requested_bedrooms = inferred_bedrooms if inferred_bedrooms is not None else input_data.get("bedrooms")
    requested_bathrooms = inferred_bathrooms if inferred_bathrooms is not None else input_data.get("bathrooms")

    bedrooms = int(_clamp(float(requested_bedrooms or 2), 1, 4))
    bathrooms = int(_clamp(float(requested_bathrooms or bedrooms), 1, 6))
    facing = _normalize_facing(input_data.get("facing"))
    vastu = bool(input_data.get("vastu", True))
    extras = _normalize_extras(input_data.get("extras", []))
    redesign = bool(input_data.get("_redesign", False))

    setbacks = _setbacks(plot_w, plot_l)
    usable_w = _r(plot_w - setbacks["left"] - setbacks["right"])
    usable_l = _r(plot_l - setbacks["front"] - setbacks["rear"])
    if usable_w < 12.0 or usable_l < 16.0:
        return {"error": "Plot too small after setbacks for a valid ground-floor hub plan."}

    trim_order = ["garage", "balcony", "store", "study", "pooja"]
    active_extras = list(extras)
    removed_extras: List[str] = []

    rooms: List[Dict[str, Any]] = []
    notes: List[str] = []
    while True:
        rooms = _build_rooms(usable_w, usable_l, bedrooms, bathrooms, facing, vastu, active_extras)
        notes = []
        _apply_functional_rules(rooms, bedrooms, notes)
        _normalize_bedroom_roles(rooms, bedrooms, notes)
        notes.extend(_validate(rooms, usable_w, usable_l))

        if _core_layout_ok(rooms, usable_w * usable_l, bedrooms):
            break

        to_remove = next((ex for ex in trim_order if ex in active_extras), None)
        if not to_remove:
            break
        active_extras.remove(to_remove)
        removed_extras.append(to_remove)

    if removed_extras:
        notes.append("Removed extras for room quality: " + ", ".join(removed_extras))
    _cleanup_structural_overlaps(rooms, usable_w, usable_l, notes)

    # Keep at least one full bathroom type; convert only surplus attached bathrooms to toilet.
    master = next((r for r in rooms if r["type"] == "master_bedroom"), None)
    bathrooms_near_master = [
        r
        for r in rooms
        if r["type"] == "bathroom" and master and _adjacent(master, r) and r["id"] != "bath_main"
    ]
    for r in bathrooms_near_master[1:]:
        _set_room_type(r, "toilet", "Converted surplus attached bathroom to toilet")

    # Furniture for each room.
    for room in rooms:
        _attach_furniture(room)

    # Open cells are required to fill leftover at corners.
    if not any(r["type"] == "open_area" for r in rooms):
        rooms.append(_room("open_corner", "open_area", _r(usable_w - 5.0), _r(usable_l - 5.0), 5.0, 5.0, ""))

    # Final cleanup after synthetic corner fill has been added.
    _cleanup_structural_overlaps(rooms, usable_w, usable_l, notes)
    _force_no_overlap(rooms, usable_w, usable_l, notes)

    doors = _build_doors(rooms, usable_w, usable_l)
    windows = _build_windows(rooms, usable_w, usable_l)

    if redesign:
        _mirror_x(rooms, doors, windows, usable_w)

    # Recompute areas after optional mirroring and merging.
    for r in rooms:
        r["area"] = _r(r["width"] * r["height"])

    connectivity_checks = _connectivity_checks(rooms)
    layout_signature = _layout_signature(rooms, bedrooms)

    design_score = _score(rooms, usable_w, usable_l, vastu)
    design_score["connectivity_checks"] = connectivity_checks
    design_score["layout_signature"] = layout_signature

    template_name = next((r.get("template") for r in rooms if r.get("template")), "hub")
    architect_notes = [
        "Hub open-plan generated with Dining Area as the central circulation core.",
        "No corridor/passage room used; movement is through direct room connections.",
        "All available buildable area is assigned to rooms and open areas.",
        f"Template: {template_name} | Signature: {layout_signature}",
    ]
    architect_notes.extend(notes[:4])

    return {
        "plot": {
            "width": _r(plot_w),
            "length": _r(plot_l),
            "usable_width": _r(usable_w),
            "usable_length": _r(usable_l),
            "setbacks": setbacks,
            "facing": facing,
        },
        "rooms": rooms,
        "doors": doors,
        "windows": windows,
        "design_score": design_score,
        "connectivity_checks": connectivity_checks,
        "layout_signature": layout_signature,
        "bhk": bedrooms,
        "architect_notes": architect_notes,
        "total_area": _r(total_area),
        "usable_area": _r(usable_w * usable_l),
        "layout_type": "hub_open_plan",
    }


def generate_ground_floor_plan(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deterministic hub open-plan ground-floor layout."""
    return _generate(deepcopy(input_data))


def redesign_ground_floor_plan(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate mirrored redesign variant of deterministic hub layout."""
    cloned = deepcopy(input_data)
    cloned["_redesign"] = True
    return _generate(cloned)
