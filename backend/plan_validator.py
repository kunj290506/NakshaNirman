"""
Plan Validator — production-grade checks for LLM generated plans.
Rejects impractical plans early so repair/retry can produce buildable outputs.
"""
from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("plan_validator")

ALLOWED_ROOM_TYPES = {
    "living", "dining", "kitchen", "corridor", "master_bedroom", "bedroom",
    "master_bath", "bathroom", "toilet", "pooja", "study", "store", "balcony",
    "garage", "utility", "foyer", "staircase", "open_area",
}

ROOM_TYPE_ALIASES = {
    "living_room": "living",
    "master bedroom": "master_bedroom",
    "masterbedroom": "master_bedroom",
    "bath": "bathroom",
    "washroom": "bathroom",
    "puja": "pooja",
    "mandir": "pooja",
}

MIN_DIMS = {
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

# Previous tolerances were too strict for first-pass LLM geometry and caused
# valid plans to be rejected. These values reflect practical rounding tolerance.
OOB_TOLERANCE_FT = 1.0
MAJOR_OVERLAP_AREA_FT2 = 8.0
MAX_BOWLING_ALLEY_RATIO = 3.0

EXTRA_TYPE_MAP = {
    "pooja": "pooja",
    "study": "study",
    "store": "store",
    "balcony": "balcony",
    "garage": "garage",
    "utility": "utility",
    "foyer": "foyer",
    "staircase": "staircase",
}


def _to_float(v: Any, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        if not math.isfinite(f):
            return fallback
        return f
    except Exception:
        return fallback


def _rect_overlap_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap_x = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    overlap_y = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    return overlap_x * overlap_y


def _touching(a: dict[str, Any], b: dict[str, Any], tol: float = 0.35) -> bool:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]

    vertical_contact = (
        abs(ax2 - bx1) <= tol or abs(bx2 - ax1) <= tol
    ) and (min(ay2, by2) - max(ay1, by1) >= 2.0)

    horizontal_contact = (
        abs(ay2 - by1) <= tol or abs(by2 - ay1) <= tol
    ) and (min(ax2, bx2) - max(ax1, bx1) >= 2.0)

    return vertical_contact or horizontal_contact


def _polygon_bbox(points: list[Any]) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        if isinstance(p, dict):
            x = _to_float(p.get("x"), float("nan"))
            y = _to_float(p.get("y"), float("nan"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = _to_float(p[0], float("nan"))
            y = _to_float(p[1], float("nan"))
        else:
            x = float("nan")
            y = float("nan")

        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)

    if len(xs) < 3:
        return None

    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)


def validate_llm_plan(
    plan_dict: dict[str, Any],
    usable_w: float,
    usable_l: float,
    req: Any | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate an LLM-generated plan dict.
    Returns (is_valid, issues). Issues are actionable and used for retry prompts.
    """
    issues: list[str] = []
    warnings: list[str] = []

    rooms_raw = plan_dict.get("rooms", [])
    if not isinstance(rooms_raw, list) or not rooms_raw:
        return False, ["No rooms found in plan"]

    rooms: list[dict[str, Any]] = []
    for idx, room in enumerate(rooms_raw):
        if not isinstance(room, dict):
            continue

        room_type = str(room.get("type", "")).strip().lower()
        room_type = ROOM_TYPE_ALIASES.get(room_type, room_type)
        label = str(room.get("label", room.get("id", f"room_{idx+1}"))).strip() or f"room_{idx+1}"
        x = _to_float(room.get("x"), 0.0)
        y = _to_float(room.get("y"), 0.0)
        w = _to_float(room.get("width"), 0.0)
        h = _to_float(room.get("height"), 0.0)

        poly = room.get("polygon", room.get("vertices", []))
        if isinstance(poly, list) and poly:
            bbox = _polygon_bbox(poly)
            if bbox is not None:
                x, y, w, h = bbox
            else:
                issues.append(f"{label} has invalid polygon geometry")

        if room_type not in ALLOWED_ROOM_TYPES:
            warnings.append(f"Unknown room type: {room_type or 'missing'} ({label})")

        rooms.append({"type": room_type, "label": label, "x": x, "y": y, "w": w, "h": h})

    if len(rooms) < 4:
        issues.append(f"Only {len(rooms)} rooms generated (need at least 4)")

    # Core program checks
    type_counts: dict[str, int] = {}
    for r in rooms:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

    for core in ("living", "kitchen", "master_bedroom"):
        if type_counts.get(core, 0) == 0:
            issues.append(f"Missing mandatory room: {core}")

    # Circulation spine is mandatory; do not fail for short corridors, only absence.
    if type_counts.get("corridor", 0) == 0:
        issues.append("No circulation spine found.")

    if req is not None:
        requested_bedrooms = int(getattr(req, "bedrooms", 1) or 1)
        actual_bedrooms = type_counts.get("master_bedroom", 0) + type_counts.get("bedroom", 0)
        if actual_bedrooms < requested_bedrooms:
            issues.append(
                f"Only {actual_bedrooms} bedrooms generated for {requested_bedrooms}BHK request"
            )

        requested_bathrooms = int(getattr(req, "bathrooms_target", 0) or 0)
        if requested_bathrooms > 0:
            actual_bathrooms = (
                type_counts.get("master_bath", 0)
                + type_counts.get("bathroom", 0)
                + type_counts.get("toilet", 0)
            )
            if actual_bathrooms < requested_bathrooms:
                issues.append(
                    f"Only {actual_bathrooms} bathrooms generated for target {requested_bathrooms}"
                )

        requested_extras = set(getattr(req, "extras", []) or [])
        for extra in requested_extras:
            expected_type = EXTRA_TYPE_MAP.get(str(extra).lower())
            if expected_type and type_counts.get(expected_type, 0) == 0:
                issues.append(f"Requested extra missing: {expected_type}")

    # Bounds and room-dimension checks
    severe_oob = 0
    for r in rooms:
        min_w, min_h = MIN_DIMS.get(r["type"], (4.0, 4.0))
        if r["w"] < min_w * 0.8 or r["h"] < min_h * 0.8:
            issues.append(
                f"{r['label']} too small ({r['w']:.1f}x{r['h']:.1f} ft) for type {r['type'] or 'unknown'}"
            )

        # Reject unusable long and narrow rooms that the LLM occasionally emits.
        # Corridor is intentionally linear circulation space and should not be
        # flagged by the general bowling-alley room proportion rule.
        if r["type"] != "corridor":
            shorter = max(0.01, min(r["w"], r["h"]))
            longer = max(r["w"], r["h"])
            if longer / shorter > MAX_BOWLING_ALLEY_RATIO:
                issues.append(f"Room {r['label']} has bowling-alley proportions.")

        if r["x"] < -OOB_TOLERANCE_FT or r["y"] < -OOB_TOLERANCE_FT:
            severe_oob += 1
        if r["x"] + r["w"] > usable_w + OOB_TOLERANCE_FT:
            severe_oob += 1
        if r["y"] + r["h"] > usable_l + OOB_TOLERANCE_FT:
            severe_oob += 1

    if severe_oob >= max(2, len(rooms) // 4):
        issues.append(f"{severe_oob} major out-of-bounds violations in room geometry")
    elif severe_oob > 0:
        warnings.append(f"{severe_oob} mild out-of-bounds violations detected")

    # Overlap checks
    major_overlaps = 0
    overlap_area_total = 0.0
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            area = _rect_overlap_area(rooms[i], rooms[j])
            if area <= 0:
                continue
            overlap_area_total += area
            if area >= MAJOR_OVERLAP_AREA_FT2:
                major_overlaps += 1

    if major_overlaps > 0:
        issues.append(
            f"{major_overlaps} major room overlaps found (total overlap {overlap_area_total:.1f} sq.ft)"
        )
    elif overlap_area_total > 0.0:
        warnings.append(f"Minor room overlaps found ({overlap_area_total:.1f} sq.ft total)")

    # Area utilization checks
    usable_area = max(1.0, usable_w * usable_l)
    gross_room_area = sum(max(0.0, r["w"] * r["h"]) for r in rooms)
    utilization = gross_room_area / usable_area
    # Removed the old low-utilization hard fail; architecturally valid plans can
    # intentionally keep more breathing space after setbacks and circulation.
    if utilization > 1.08:
        issues.append(f"Room area exceeds usable footprint ({utilization*100:.1f}%)")

    # Soft adjacency quality checks (warnings only)
    def has_adj(type_a: str, type_b: str) -> bool:
        a_rooms = [r for r in rooms if r["type"] == type_a]
        b_rooms = [r for r in rooms if r["type"] == type_b]
        if not a_rooms or not b_rooms:
            return True
        return any(_touching(a, b) for a in a_rooms for b in b_rooms)

    if not has_adj("kitchen", "dining"):
        warnings.append("Kitchen is not adjacent to dining")
    if not has_adj("master_bedroom", "master_bath"):
        warnings.append("Master bath is not adjacent to master bedroom")
    if not has_adj("living", "dining"):
        warnings.append("Living is not adjacent to dining")

    is_valid = len(issues) == 0
    if warnings:
        log.info("Plan validation warnings (%d): %s", len(warnings), warnings[:4])

    if is_valid:
        log.info(
            "Plan validation PASSED | rooms=%d utilization=%.2f",
            len(rooms),
            utilization,
        )
    else:
        log.warning("Plan validation FAILED (%d issues): %s", len(issues), issues[:5])

    return is_valid, issues
