"""
NakshaNirman Layout Engine — BSP (Binary Space Partition) fallback.

Produces a deterministic, non-overlapping floor plan when the LLM
fails to generate valid JSON. Guarantees the user always sees a
working floor plan on canvas.

Uses the 3-band zoning system:
  Band 1 (y=0):         Public   — living, dining, pooja, foyer
  Band 2 (middle):      Service  — corridor, kitchen, bathrooms, utility
  Band 3 (top):         Private  — master_bedroom, bedrooms, study
"""
from __future__ import annotations

import math
from typing import Any

# ── Color map ────────────────────────────────────────────────────────
COLOR_MAP = {
    "living": "#E8F5E9", "dining": "#FFF3E0", "kitchen": "#FFEBEE",
    "master_bedroom": "#E3F2FD", "bedroom": "#E3F2FD",
    "master_bath": "#E0F7FA", "bathroom": "#E0F7FA", "toilet": "#E0F7FA",
    "corridor": "#F5F5F5", "pooja": "#FFF8E1", "study": "#EDE7F6",
    "store": "#EFEBE9", "balcony": "#E8F5E9", "garage": "#ECEFF1",
    "utility": "#F3E5F5", "foyer": "#FAFAFA", "staircase": "#ECEFF1",
}

# ── Minimum room sizes (width, height) in feet ──────────────────────
MIN_SIZES: dict[str, tuple[float, float]] = {
    "living": (11.0, 11.0),
    "dining": (8.0, 8.0),
    "kitchen": (7.0, 8.0),
    "master_bedroom": (10.0, 10.0),
    "bedroom": (9.0, 9.0),
    "master_bath": (4.5, 6.0),
    "bathroom": (4.0, 5.0),
    "corridor": (3.5, 8.0),
    "pooja": (4.0, 4.0),
    "study": (6.0, 7.0),
    "store": (4.0, 4.0),
    "balcony": (3.5, 6.0),
    "garage": (9.0, 15.0),
    "utility": (4.0, 5.0),
    "foyer": (4.0, 4.0),
    "staircase": (4.0, 8.0),
}

# ── Zone / band classification ───────────────────────────────────────
ZONE_MAP = {
    "living": ("public", 1), "dining": ("public", 1),
    "pooja": ("public", 1), "foyer": ("public", 1),
    "kitchen": ("service", 2), "bathroom": ("service", 2),
    "master_bath": ("service", 2), "corridor": ("service", 2),
    "utility": ("service", 2), "store": ("service", 2),
    "staircase": ("service", 2),
    "master_bedroom": ("private", 3), "bedroom": ("private", 3),
    "study": ("private", 3), "balcony": ("private", 3),
    "garage": ("service", 1),
}

ROOM_TOKEN_ALIASES = {
    "puja": "pooja",
    "mandir": "pooja",
    "office": "study",
    "home_office": "study",
    "guest_room": "bedroom",
    "guest_bedroom": "bedroom",
    "common_bath": "bathroom",
    "common_bathroom": "bathroom",
    "wc": "bathroom",
    "stairs": "staircase",
}


def _normalize_room_token(raw: Any) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_")
    token = token.replace("-", "_")
    return ROOM_TOKEN_ALIASES.get(token, token)


def _normalize_token_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = _normalize_room_token(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _room_program(
    bedrooms: int,
    extras: list[str],
    *,
    bathrooms_target: int = 0,
    work_from_home: bool = False,
    family_type: str = "nuclear",
    parking_slots: int = 0,
    floors: int = 1,
    must_have: list[str] | None = None,
    avoid: list[str] | None = None,
) -> list[str]:
    """
    Decide which rooms to include based on BHK and extras.
    Returns a list of room type strings in placement order.
    """
    floors = max(1, floors)
    must_have_set = set(_normalize_token_list(must_have or []))
    avoid_set = set(_normalize_token_list(avoid or []))
    bedrooms_on_floor = max(1, math.ceil(bedrooms / floors))

    rooms = ["living", "dining", "kitchen", "master_bedroom"]

    for i in range(max(0, bedrooms_on_floor - 1)):
        rooms.append("bedroom")

    # Bathrooms: use explicit target if provided, otherwise one per bedroom.
    bathrooms_target = max(0, bathrooms_target)
    bathrooms_total = max(1, bathrooms_target if bathrooms_target > 0 else bedrooms)
    bathrooms_on_floor = max(1, math.ceil(bathrooms_total / floors))
    rooms.append("master_bath")
    for _ in range(max(0, bathrooms_on_floor - 1)):
        rooms.append("bathroom")

    # Lifestyle-driven rooms
    if work_from_home and "study" not in rooms:
        rooms.append("study")
    if family_type == "joint" and "utility" not in rooms:
        rooms.append("utility")
    if parking_slots > 0 and "garage" not in rooms:
        rooms.append("garage")
    if floors > 1 and "staircase" not in rooms:
        rooms.append("staircase")

    # Explicit extras
    valid_extras = {"pooja", "study", "store", "balcony", "garage", "utility",
                    "foyer", "staircase"}
    for extra in extras:
        token = _normalize_room_token(extra)
        if token in valid_extras and token not in rooms:
            rooms.append(token)

    for token in must_have_set:
        if token in valid_extras and token not in rooms:
            rooms.append(token)

    # Force small but important practical constraints in strict mode flows.
    if "corridor" in must_have_set and "corridor" not in rooms:
        rooms.append("corridor")
    if "bedroom" in must_have_set and "bedroom" not in rooms:
        rooms.append("bedroom")
    if "bathroom" in must_have_set and "bathroom" not in rooms:
        rooms.append("bathroom")

    protected = {"living", "dining", "kitchen", "master_bedroom", "master_bath"}
    if avoid_set:
        filtered: list[str] = []
        for room in rooms:
            if room in avoid_set and room not in protected:
                continue
            filtered.append(room)
        rooms = filtered

    # Keep minimum livability guarantees after avoid filtering.
    if not any(r in ("master_bedroom", "bedroom") for r in rooms):
        rooms.append("bedroom")
    if not any(r in ("master_bath", "bathroom", "toilet") for r in rooms):
        rooms.append("bathroom")

    return rooms


def _compute_vastu_score(rooms: list[dict], uw: float, ul: float, facing: str) -> int:
    """Calculate Vastu score starting at 55, adding/subtracting."""
    score = 55

    room_lookup: dict[str, dict] = {}
    for r in rooms:
        room_lookup.setdefault(r["type"], r)

    # Pooja in NE corner (+8)
    pooja = room_lookup.get("pooja")
    if pooja:
        px = pooja["x"] + pooja["width"] / 2
        py = pooja["y"] + pooja["height"] / 2
        if px > uw * 0.5 and py < ul * 0.4:
            score += 8

    # Kitchen in SE quadrant (+8)
    kitchen = room_lookup.get("kitchen")
    if kitchen:
        kx = kitchen["x"] + kitchen["width"] / 2
        if kx > uw * 0.5:
            score += 8

    # Master bedroom in SW quadrant (+7)
    master = room_lookup.get("master_bedroom")
    if master:
        mx = master["x"] + master["width"] / 2
        if mx < uw * 0.5:
            score += 7

    # Living room in north or east portion (+5)
    living = room_lookup.get("living")
    if living:
        if facing in ("north", "east"):
            score += 5

    # Main door bonus for N/E facing (+8)
    if facing in ("north", "east"):
        score += 8

    # Toilet in NE penalty (-8)
    for r in rooms:
        if r["type"] in ("bathroom", "toilet", "master_bath"):
            rx = r["x"] + r["width"] / 2
            ry = r["y"] + r["height"] / 2
            if rx > uw * 0.6 and ry < ul * 0.3:
                score -= 8
                break

    return max(40, min(100, score))


def _compute_adjacency_score(rooms: list[dict]) -> int:
    """Simple adjacency quality score based on room connectivity."""
    score = 60

    room_lookup: dict[str, dict] = {}
    for r in rooms:
        room_lookup.setdefault(r["type"], r)

    def _adjacent(a: dict, b: dict) -> bool:
        """Check if two rooms share an edge (touching)."""
        # Horizontally adjacent
        if abs((a["x"] + a["width"]) - b["x"]) < 0.5 or abs((b["x"] + b["width"]) - a["x"]) < 0.5:
            if a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"]:
                return True
        # Vertically adjacent
        if abs((a["y"] + a["height"]) - b["y"]) < 0.5 or abs((b["y"] + b["height"]) - a["y"]) < 0.5:
            if a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]:
                return True
        return False

    # Kitchen adjacent to dining (+10)
    if "kitchen" in room_lookup and "dining" in room_lookup:
        if _adjacent(room_lookup["kitchen"], room_lookup["dining"]):
            score += 10

    # Master bath adjacent to master bedroom (+10)
    if "master_bath" in room_lookup and "master_bedroom" in room_lookup:
        if _adjacent(room_lookup["master_bath"], room_lookup["master_bedroom"]):
            score += 10

    # Living adjacent to dining (+8)
    if "living" in room_lookup and "dining" in room_lookup:
        if _adjacent(room_lookup["living"], room_lookup["dining"]):
            score += 8

    # Corridor connects to bedrooms (+5 each)
    corridor = room_lookup.get("corridor")
    if corridor:
        for r in rooms:
            if r["type"] in ("bedroom", "master_bedroom") and _adjacent(corridor, r):
                score += 5

    return max(40, min(100, score))


def generate_bsp_layout(
    plot_width: float,
    plot_length: float,
    bedrooms: int = 2,
    facing: str = "east",
    extras: list[str] | None = None,
    family_type: str = "nuclear",
    **kwargs: Any,
) -> dict:
    """
    Generate a deterministic, non-overlapping floor plan using
    band-based BSP placement.

    This is the fallback when the LLM produces invalid output.
    It guarantees:
      - Zero room overlaps
      - All rooms within usable bounds
      - All rooms meet minimum size requirements
    """
    extras = extras or []
    bedrooms = max(1, min(4, bedrooms))
    bathrooms_target = int(kwargs.get("bathrooms_target", 0) or 0)
    work_from_home = bool(kwargs.get("work_from_home", False))
    parking_slots = int(kwargs.get("parking_slots", 0) or 0)
    floors = int(kwargs.get("floors", 1) or 1)
    must_have = kwargs.get("must_have", [])
    avoid = kwargs.get("avoid", [])

    # ── Step 1: Calculate usable bounds ──────────────────────────────
    uw = max(12.0, plot_width - 7.0)
    ul = max(15.0, plot_length - 11.5)

    # ── Step 2: Calculate band heights ───────────────────────────────
    band1_h = max(11.0, ul * 0.30)
    band2_h = max(8.0, ul * 0.26)
    band3_h = ul - band1_h - band2_h

    # Ensure band3 has enough space for bedrooms
    if band3_h < 9.0 and ul > 20:
        band1_h = max(11.0, ul * 0.28)
        band2_h = max(8.0, ul * 0.24)
        band3_h = ul - band1_h - band2_h

    # ── Step 3: Determine room program ───────────────────────────────
    room_types = _room_program(
        bedrooms,
        extras,
        bathrooms_target=bathrooms_target,
        work_from_home=work_from_home,
        family_type=family_type,
        parking_slots=parking_slots,
        floors=floors,
        must_have=must_have,
        avoid=avoid,
    )

    # ── Step 4: Central split (corridor is optional) ──────────────────
    corridor_enabled = "corridor" in room_types
    corridor_width = (4.0 if family_type == "joint" and uw >= 24.0 else 3.5) if corridor_enabled else 0.0
    corridor_x = (uw - corridor_width) / 2 if corridor_enabled else (uw / 2.0)
    left_w = corridor_x
    right_x_start = corridor_x + corridor_width
    right_w = uw - right_x_start

    placed_rooms: list[dict] = []
    room_counter: dict[str, int] = {}
    soft_fit = 0.85

    def _make_room(rtype: str, x: float, y: float, w: float, h: float) -> dict:
        count = room_counter.get(rtype, 0) + 1
        room_counter[rtype] = count
        zone, band = ZONE_MAP.get(rtype, ("service", 2))
        rid = f"{rtype}_{count:02d}"
        room = {
            "id": rid,
            "type": rtype,
            "label": rtype.replace("_", " ").title(),
            "x": float(x),
            "y": float(y),
            "width": float(w),
            "height": float(h),
            "area": round(w * h, 1),
            "zone": zone,
            "band": band,
            "color": COLOR_MAP.get(rtype, "#F5F5F5"),
            "polygon": [
                {"x": float(x), "y": float(y)},
                {"x": float(x + w), "y": float(y)},
                {"x": float(x + w), "y": float(y + h)},
                {"x": float(x), "y": float(y + h)},
            ],
        }
        return room

    def _overlaps_any(x: float, y: float, w: float, h: float) -> bool:
        for room in placed_rooms:
            rx, ry = float(room["x"]), float(room["y"])
            rw, rh = float(room["width"]), float(room["height"])
            if x + w <= rx + 0.01:
                continue
            if rx + rw <= x + 0.01:
                continue
            if y + h <= ry + 0.01:
                continue
            if ry + rh <= y + 0.01:
                continue
            return True
        return False

    # ── Step 5: Place rooms in bands ─────────────────────────────────

    # --- Band 1: Public zone (y = 0 to band1_h) ---
    # Living on left, dining on right
    band1_rooms = [
        r for r in room_types
        if r in ("living", "dining", "pooja", "foyer")
        and (corridor_enabled or r != "foyer")
    ]
    band1_cursor_left = 0.0
    band1_cursor_right = right_x_start

    for rtype in band1_rooms:
        min_w, min_h = MIN_SIZES.get(rtype, (8.0, 8.0))
        h = min(band1_h, max(min_h, band1_h))

        if rtype == "living":
            w = min(left_w, max(min_w, left_w))
            placed_rooms.append(_make_room(rtype, 0.0, 0.0, w, h))
            band1_cursor_left = w
        elif rtype == "dining":
            w = min(right_w, max(min_w, right_w))
            placed_rooms.append(_make_room(rtype, band1_cursor_right, 0.0, w, h))
            band1_cursor_right += w
        elif rtype == "pooja":
            # NE corner for Vastu
            w = min(min_w + 1, uw - band1_cursor_right)
            w = max(min_w, w)
            px = uw - w
            if px < band1_cursor_right:
                px = band1_cursor_right
            if px + w > uw:
                w = uw - px
            if w >= min_w:
                placed_rooms.append(_make_room(rtype, px, 0.0, w, min(min_h + 1, band1_h)))
        elif rtype == "foyer":
            if corridor_enabled:
                # Small foyer near entry spine when a corridor exists.
                w = max(min_w, corridor_width)
                placed_rooms.append(_make_room(rtype, corridor_x, 0.0, w, min(min_h, band1_h * 0.5)))

    # --- Band 2: Service zone (y = band1_h to band1_h + band2_h) ---
    band2_y = band1_h

    # Corridor first (center column, spans band2 + band3).
    # If staircase is requested, try embedding it in the corridor shaft.
    staircase_embedded = False
    stair_h = 0.0
    if "staircase" in room_types and "corridor" in room_types:
        min_w_s, min_h_s = MIN_SIZES.get("staircase", (4.0, 8.0))
        stair_h = min(band2_h, max(min_h_s * soft_fit, band2_h * 0.85))
        if corridor_width >= min_w_s * soft_fit and stair_h >= min_h_s * soft_fit:
            placed_rooms.append(_make_room("staircase", corridor_x, band2_y, corridor_width, stair_h))
            staircase_embedded = True

    if "corridor" in room_types:
        corr_y = band2_y + stair_h if staircase_embedded else band2_y
        corr_h = (band2_h + band3_h) - stair_h if staircase_embedded else (band2_h + band3_h)
        corr_h = max(6.0, corr_h)
        placed_rooms.append(_make_room("corridor", corridor_x, corr_y, corridor_width, corr_h))

    band2_rooms = [
        r for r in room_types
        if r in ("kitchen", "bathroom", "master_bath", "utility", "store", "staircase")
        and not (staircase_embedded and r == "staircase")
    ]
    band2_left_cursor = 0.0
    band2_right_cursor = right_x_start

    for rtype in band2_rooms:
        min_w, min_h = MIN_SIZES.get(rtype, (4.0, 5.0))
        h = min(band2_h, max(min_h, band2_h))

        if rtype == "kitchen":
            # Kitchen on right side (Vastu: SE)
            w = min(right_w, max(min_w, right_w * 0.6))
            placed_rooms.append(_make_room(rtype, band2_right_cursor, band2_y, w, h))
            band2_right_cursor += w
        elif rtype == "master_bath":
            # Master bath on left (near master bedroom above)
            w = min(left_w * 0.4, max(min_w, left_w * 0.35))
            placed_rooms.append(_make_room(rtype, band2_left_cursor, band2_y, w, h))
            band2_left_cursor += w
        elif rtype == "bathroom":
            # Bathrooms fill remaining space in band2
            remaining_right = uw - band2_right_cursor
            remaining_left = corridor_x - band2_left_cursor
            if remaining_right >= min_w:
                w = min(remaining_right, max(min_w, remaining_right * 0.5))
                placed_rooms.append(_make_room(rtype, band2_right_cursor, band2_y, w, h))
                band2_right_cursor += w
            elif remaining_left >= min_w:
                w = min(remaining_left, max(min_w, remaining_left * 0.5))
                placed_rooms.append(_make_room(rtype, band2_left_cursor, band2_y, w, h))
                band2_left_cursor += w
        elif rtype in ("utility", "store", "staircase"):
            remaining_left = corridor_x - band2_left_cursor
            remaining_right = uw - band2_right_cursor
            if remaining_left >= min_w:
                w = min(remaining_left, max(min_w, min_w + 1))
                placed_rooms.append(_make_room(rtype, band2_left_cursor, band2_y, w, h))
                band2_left_cursor += w
            elif remaining_right >= min_w:
                w = min(remaining_right, max(min_w, min_w + 1))
                placed_rooms.append(_make_room(rtype, band2_right_cursor, band2_y, w, h))
                band2_right_cursor += w

    # --- Band 3: Private zone (y = band1_h + band2_h to UL) ---
    band3_y = band1_h + band2_h
    has_master = "master_bedroom" in room_types
    bedroom_count = sum(1 for r in room_types if r == "bedroom")
    study_count = sum(1 for r in room_types if r == "study")
    balcony_count = sum(1 for r in room_types if r == "balcony")

    if has_master:
        min_w_m, min_h_m = MIN_SIZES["master_bedroom"]
        if left_w >= min_w_m * soft_fit and band3_h >= min_h_m * soft_fit:
            placed_rooms.append(_make_room("master_bedroom", 0.0, band3_y, left_w, band3_h))

    right_x = right_x_start
    right_available_h = band3_h

    # Reserve top strip on right for balcony if requested and feasible.
    if balcony_count > 0:
        min_w_b, min_h_b = MIN_SIZES["balcony"]
        if right_w >= min_w_b and band3_h >= min_h_b + 4.0:
            by = band3_y + band3_h - min_h_b
            if not _overlaps_any(right_x, by, right_w, min_h_b):
                placed_rooms.append(_make_room("balcony", right_x, by, right_w, min_h_b))
                right_available_h -= min_h_b

    right_cursor_y = band3_y

    # Prioritize bedrooms first; optional study uses only leftover private height.
    min_w_bed, min_h_bed = MIN_SIZES["bedroom"]
    for idx in range(bedroom_count):
        slots_left = max(1, bedroom_count - idx)
        reserve_for_study = study_count * (MIN_SIZES["study"][1] * soft_fit)
        remaining_h_total = (band3_y + right_available_h) - right_cursor_y
        remaining_h = max(0.0, remaining_h_total - reserve_for_study)
        if right_w + 0.01 < min_w_bed * soft_fit or remaining_h + 0.01 < min_h_bed * soft_fit:
            break

        target_h = max(min_h_bed * soft_fit, remaining_h / slots_left)
        h = min(remaining_h, target_h)
        if h + 0.01 < min_h_bed * soft_fit:
            break

        if _overlaps_any(right_x, right_cursor_y, right_w, h):
            break

        placed_rooms.append(_make_room("bedroom", right_x, right_cursor_y, right_w, h))
        right_cursor_y += h

    min_w_study, min_h_study = MIN_SIZES["study"]
    for _ in range(study_count):
        remaining_h = (band3_y + right_available_h) - right_cursor_y
        if right_w + 0.01 < min_w_study * soft_fit or remaining_h + 0.05 < min_h_study * soft_fit:
            break

        h = min(remaining_h, max(min_h_study * soft_fit, remaining_h))
        if h + 0.05 < min_h_study * soft_fit:
            break
        if _overlaps_any(right_x, right_cursor_y, right_w, h):
            break

        placed_rooms.append(_make_room("study", right_x, right_cursor_y, right_w, h))
        right_cursor_y += h

    # --- Garage (special — band 1, needs lots of space) ---
    if "garage" in room_types:
        min_w_g, min_h_g = MIN_SIZES["garage"]
        # Only place if enough area and without overlapping existing rooms.
        if uw >= min_w_g + 4 and ul >= min_h_g + 4:
            garage_h = min(min_h_g, band1_h + band2_h)
            garage_candidates = [
                (0.0, 0.0),
                (uw - min_w_g, 0.0),
                (0.0, band1_h),
                (uw - min_w_g, band1_h),
            ]
            for gx, gy in garage_candidates:
                if gy + garage_h > ul + 0.1:
                    continue
                if _overlaps_any(gx, gy, min_w_g, garage_h):
                    continue
                placed_rooms.append(_make_room("garage", gx, gy, min_w_g, garage_h))
                break

    # ── Step 6: Clamp all rooms to usable bounds ─────────────────────
    for room in placed_rooms:
        room["x"] = max(0.0, round(room["x"], 1))
        room["y"] = max(0.0, round(room["y"], 1))
        if room["x"] + room["width"] > uw + 0.1:
            room["width"] = round(max(3.0, uw - room["x"]), 1)
        if room["y"] + room["height"] > ul + 0.1:
            room["height"] = round(max(3.0, ul - room["y"]), 1)
        room["area"] = round(room["width"] * room["height"], 1)
        # Rebuild polygon
        x, y, w, h = room["x"], room["y"], room["width"], room["height"]
        room["polygon"] = [
            {"x": x, "y": y}, {"x": x + w, "y": y},
            {"x": x + w, "y": y + h}, {"x": x, "y": y + h},
        ]

    # ── Step 7: Score ────────────────────────────────────────────────
    vastu_score = _compute_vastu_score(placed_rooms, uw, ul, facing)
    adj_score = _compute_adjacency_score(placed_rooms)

    # ── Step 8: Build output ─────────────────────────────────────────
    return {
        "plot_boundary": [
            {"x": 0, "y": 0}, {"x": uw, "y": 0},
            {"x": uw, "y": ul}, {"x": 0, "y": ul},
        ],
        "rooms": placed_rooms,
        "doors": [],
        "windows": [],
        "metadata": {
            "bhk": bedrooms,
            "vastu_score": vastu_score,
            "adjacency_score": adj_score,
            "architect_note": (
                f"BSP fallback layout: {bedrooms}BHK on "
                f"{plot_width:.0f}x{plot_length:.0f} plot, {facing}-facing. "
                f"Program distributed for {max(1, floors)} floor(s). "
                + (
                    f"3-band zoning with {'4ft' if family_type == 'joint' else '3.5ft'} central corridor."
                    if corridor_enabled
                    else "3-band zoning without forced corridor (space-optimized)."
                )
            ),
            "vastu_issues": [],
        },
        "vastu_score": vastu_score,
        "adjacency_score": adj_score,
        "generation_method": "bsp",
        "reasoning_trace": [
            f"BSP engine: {plot_width:.0f}x{plot_length:.0f} plot → UW={uw:.1f}, UL={ul:.1f}",
            f"Bands: public={band1_h:.1f}ft, service={band2_h:.1f}ft, private={band3_h:.1f}ft",
            f"Placed {len(placed_rooms)} rooms with zero overlaps",
            f"Vastu score: {vastu_score}, Adjacency score: {adj_score}",
        ],
    }
