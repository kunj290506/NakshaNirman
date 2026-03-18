"""Deterministic hub open-plan engine for NakshaNirman ground-floor layouts."""

from __future__ import annotations

import math
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

    for room in list(rooms):
        mn = MIN_SIZE.get(room["type"])
        if not mn:
            continue
        if room["width"] >= mn[0] and room["height"] >= mn[1]:
            continue
        candidates = [r for r in rooms if r["id"] != room["id"] and _adjacent(room, r)]
        target = next((r for r in candidates if r["type"] == "dining"), None) or (candidates[0] if candidates else None)
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
        mn = MIN_SIZE.get(r["type"])
        if not mn:
            continue
        checks.append(100.0 if (r["width"] >= mn[0] and r["height"] >= mn[1]) else 0.0)
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
    col_l = _r(usable_w * 0.38)
    col_r = _r(usable_w * 0.27)
    col_c = _r(usable_w - col_l - col_r)

    row_b = _r(usable_l * 0.27)
    row_t = _r(usable_l * 0.33)
    row_m = _r(usable_l - row_b - row_t)

    y_b, y_m, y_t = 0.0, row_b, _r(row_b + row_m)

    # Front row: living + kitchen + garage/open.
    bot_left = _room("living", "living", 0.0, y_b, col_l, row_b, "North-East zone preferred")
    pooja_room = None
    if "pooja" in extras and bot_left["width"] >= 7.2:
        pw, ph = 3.5, min(4.0, row_b)
        pooja_room = _room("pooja", "pooja", 0.0, y_b, pw, ph, "North-East alcove")
        bot_left["x"] = _r(bot_left["x"] + pw)
        bot_left["width"] = _r(bot_left["width"] - pw)
        bot_left["area"] = _r(bot_left["width"] * bot_left["height"])
    bot_ctr_type = "kitchen"
    bot_right_type = "garage" if "garage" in extras else "open_area"
    bot_ctr = _room("kitchen", bot_ctr_type, col_l, y_b, col_c, row_b, "South-East guidance applied" if vastu else "")
    bot_right = _room("front_right", bot_right_type, _r(col_l + col_c), y_b, col_r, row_b, "")

    # Middle row: left stack + central dining hub.
    mid_left_top_h = _r(max(4.5, row_m * 0.42))
    mid_left_bot_h = _r(row_m - mid_left_top_h)
    mid_left_top = _room("bath_main", "bathroom", 0.0, _r(y_m + mid_left_bot_h), col_l, mid_left_top_h, "Away from North-East")

    if bedrooms >= 2:
        mid_left_bot_type = "bedroom"
        mid_left_bot_label_note = ""
    else:
        mid_left_bot_type = "bathroom"
        mid_left_bot_label_note = "Compact service core"
    mid_left_bot = _room("mid_left", mid_left_bot_type, 0.0, y_m, col_l, mid_left_bot_h, mid_left_bot_label_note)

    dining = _room("dining", "dining", col_l, y_m, _r(col_c + col_r), row_m, "Central hub circulation")

    # Rear row: master + toilet + right cell.
    top_left = _room("master", "master_bedroom", 0.0, y_t, col_l, row_t, "South-West master zone")
    toilet_w = _r(_clamp((col_c + col_r) * 0.27, 3.5, 5.0))
    top_ctr = _room("toilet", "toilet", col_l, y_t, toilet_w, row_t, "")

    top_right_type = "open_area"
    if bedrooms >= 3:
        top_right_type = "bedroom"
    elif bedrooms == 2 and "study" in extras:
        top_right_type = "study"
    top_right = _room("rear_right", top_right_type, _r(col_l + toilet_w), y_t, _r(col_c + col_r - toilet_w), row_t, "")

    rooms: List[Dict[str, Any]] = [
        bot_left,
        bot_ctr,
        bot_right,
        mid_left_top,
        mid_left_bot,
        dining,
        top_left,
        top_ctr,
        top_right,
    ]
    if pooja_room:
        rooms.append(pooja_room)

    # 4BHK adds a split in the largest bedroom/open cell.
    if bedrooms >= 4:
        split_target = next((r for r in rooms if r["id"] == "rear_right"), None)
        if split_target and split_target["width"] >= 9.0:
            h1 = _r(split_target["height"] * 0.52)
            h2 = _r(split_target["height"] - h1)
            split_target["type"] = "bedroom"
            split_target["label"] = ROOM_META["bedroom"]["label"]
            split_target["height"] = h1
            split_target["area"] = _r(split_target["width"] * split_target["height"])
            rooms.append(_room("bed4", "bedroom", split_target["x"], _r(split_target["y"] + h1), split_target["width"], h2, ""))

    return rooms


def _generate(input_data: Dict[str, Any]) -> Dict[str, Any]:
    plot_w, plot_l, total_area = _derive_plot(input_data.get("plot_width"), input_data.get("plot_length"), input_data.get("total_area"))
    bedrooms = int(_clamp(float(input_data.get("bedrooms") or 2), 1, 4))
    bathrooms = int(_clamp(float(input_data.get("bathrooms") or bedrooms), 1, 6))
    facing = _normalize_facing(input_data.get("facing"))
    vastu = bool(input_data.get("vastu", True))
    extras = _normalize_extras(input_data.get("extras", []))
    redesign = bool(input_data.get("_redesign", False))

    setbacks = _setbacks(plot_w, plot_l)
    usable_w = _r(plot_w - setbacks["left"] - setbacks["right"])
    usable_l = _r(plot_l - setbacks["front"] - setbacks["rear"])
    if usable_w < 12.0 or usable_l < 16.0:
        return {"error": "Plot too small after setbacks for a valid ground-floor hub plan."}

    rooms = _build_rooms(usable_w, usable_l, bedrooms, bathrooms, facing, vastu, extras)
    notes = _validate(rooms, usable_w, usable_l)

    # Ensure only one toilet style room near rear for this hub style.
    master = next((r for r in rooms if r["type"] == "master_bedroom"), None)
    for r in rooms:
        if r["type"] == "bathroom" and master and r["y"] >= master["y"] - 0.1 and r["id"] != "toilet":
            r["type"] = "toilet"
            r["label"] = ROOM_META["toilet"]["label"]

    # Furniture for each room.
    for room in rooms:
        _attach_furniture(room)

    # Open cells are required to fill leftover at corners.
    if not any(r["type"] == "open_area" for r in rooms):
        rooms.append(_room("open_corner", "open_area", _r(usable_w - 5.0), _r(usable_l - 5.0), 5.0, 5.0, ""))

    doors = _build_doors(rooms, usable_w, usable_l)
    windows = _build_windows(rooms, usable_w, usable_l)

    if redesign:
        _mirror_x(rooms, doors, windows, usable_w)

    # Recompute areas after optional mirroring and merging.
    for r in rooms:
        r["area"] = _r(r["width"] * r["height"])

    design_score = _score(rooms, usable_w, usable_l, vastu)
    architect_notes = [
        "Hub open-plan generated with Dining Area as the central circulation core.",
        "No corridor/passage room used; movement is through direct room connections.",
        "All available buildable area is assigned to rooms and open areas.",
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
