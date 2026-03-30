"""
Layout Engine — Deterministic BSP room packing.
LLM is used ONLY for room-order preference and Vastu advice.
All coordinate placement is done mathematically — zero overlaps guaranteed.
"""
from __future__ import annotations
import logging
from models import (
    PlanRequest, PlanResponse, PlotInfo,
    RoomData, DoorData, WindowData,
)
from llm import call_openrouter

log = logging.getLogger("layout_engine")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SETBACKS = {"front": 6.5, "rear": 5.0, "left": 3.5, "right": 3.5}

LLM_SYSTEM_PROMPT = """You are NAKSHA AI, an Indian residential architect.
You will be given plot dimensions and room requirements.
Your ONLY job is to:
1. Confirm or adjust the room list for Vastu compliance
2. Provide a vastu_score (0-100)
3. Provide an architect_note (2-3 sentences about design)
4. Provide vastu_overrides if any room should be in a specific corner

You do NOT need to provide any coordinates or dimensions.
Return ONLY this JSON:
{
  "rooms_order": ["living", "dining", "kitchen", ...],
  "vastu_score": 85,
  "architect_note": "...",
  "vastu_overrides": {}
}"""


# ─────────────────────────────────────────────────────────────
# Room spec builder
# ─────────────────────────────────────────────────────────────
def build_room_list(bedrooms: int, extras: list[str],
                    usable_w: float, usable_l: float) -> list[dict]:
    """Build room specifications based on BHK count and extras."""
    rooms = []

    # --- Always present ---
    rooms.append({
        "type": "living", "label": "Living Room",
        "min_w": 12, "min_h": 12, "pref_w": 14, "pref_h": 14,
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
        "zone": 3, "priority": 6,
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
        rooms.append({
            "type": "bathroom", "label": "Bathroom 3",
            "min_w": 5, "min_h": 6, "pref_w": 5, "pref_h": 7,
            "zone": 2, "priority": 10,
        })
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
        rooms.append({
            "type": "bathroom", "label": "Bathroom 4",
            "min_w": 5, "min_h": 5, "pref_w": 5, "pref_h": 6,
            "zone": 2, "priority": 13,
        })

    # --- Extras ---
    if "pooja" in extras:
        rooms.append({
            "type": "pooja", "label": "Pooja Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 1, "priority": 2,
        })
    if "study" in extras:
        rooms.append({
            "type": "study", "label": "Study Room",
            "min_w": 8, "min_h": 9, "pref_w": 9, "pref_h": 10,
            "zone": 3, "priority": 8,
        })
    if "store" in extras:
        rooms.append({
            "type": "store", "label": "Store Room",
            "min_w": 5, "min_h": 5, "pref_w": 6, "pref_h": 6,
            "zone": 2, "priority": 9,
        })
    if "balcony" in extras:
        rooms.append({
            "type": "balcony", "label": "Balcony",
            "min_w": 4, "min_h": 8, "pref_w": 5, "pref_h": 10,
            "zone": 1, "priority": 3,
        })
    if "garage" in extras:
        rooms.append({
            "type": "garage", "label": "Garage",
            "min_w": 10, "min_h": 18, "pref_w": 11, "pref_h": 20,
            "zone": 1, "priority": 10,
        })

    return rooms


# ─────────────────────────────────────────────────────────────
# BSP Packing — zero-overlap guaranteed
# ─────────────────────────────────────────────────────────────
def pack_rooms_bsp(room_specs: list[dict],
                   usable_w: float, usable_l: float) -> list[dict]:
    """
    Pack rooms into three horizontal bands with zero overlaps.
    Band 1 (public, front/road):  y=0               height = 30% of usable_l
    Band 2 (service, middle):     y=band1_h          height = 27% of usable_l
    Band 3 (private, rear):       y=band1_h+band2_h  height = 43% of usable_l
    """
    band1_h = round(usable_l * 0.30, 2)
    band2_h = round(usable_l * 0.27, 2)
    band3_h = round(usable_l - band1_h - band2_h, 2)  # remainder to avoid gaps

    bands = [
        {"y": 0, "h": band1_h, "zone": 1},
        {"y": band1_h, "h": band2_h, "zone": 2},
        {"y": band1_h + band2_h, "h": band3_h, "zone": 3},
    ]

    # Separate rooms by zone
    zone_rooms = {1: [], 2: [], 3: []}
    for spec in room_specs:
        z = spec["zone"]
        if z in zone_rooms:
            zone_rooms[z].append(spec)

    # Sort each zone by priority
    for z in zone_rooms:
        zone_rooms[z].sort(key=lambda r: r["priority"])

    placed = []

    for band in bands:
        zone = band["zone"]
        rms = zone_rooms[zone]

        if zone == 2:
            # Band 2: Special layout — corridor spine in the middle
            placed.extend(
                _pack_band2(rms, 0, band["y"], usable_w, band["h"])
            )
        else:
            # Band 1 and 3: simple strip packing
            placed.extend(
                _pack_strip(rms, 0, band["y"], usable_w, band["h"], band["zone"])
            )

    # Verify no overlaps
    _verify_no_overlaps(placed)

    return placed


def _pack_strip(rooms: list[dict], bx: float, by: float,
                bw: float, bh: float, band_num: int) -> list[dict]:
    """Pack rooms left-to-right in a strip, scaling to fill the band exactly."""
    if not rooms:
        return []

    # Calculate proportional widths
    total_pref_w = sum(r["pref_w"] for r in rooms)
    placed = []
    current_x = bx

    for i, room in enumerate(rooms):
        if i == len(rooms) - 1:
            # Last room gets remaining width
            rw = round(bx + bw - current_x, 2)
        else:
            rw = round(bw * (room["pref_w"] / total_pref_w), 2)
            rw = max(rw, room["min_w"])

        # Ensure doesn't exceed band
        if current_x + rw > bx + bw:
            rw = round(bx + bw - current_x, 2)

        rh = bh  # room fills full band height

        placed.append({
            "type": room["type"],
            "label": room["label"],
            "x": round(current_x, 2),
            "y": round(by, 2),
            "width": round(rw, 2),
            "height": round(rh, 2),
            "zone": room["zone"],
            "band": band_num,
        })

        current_x = round(current_x + rw, 2)

    return placed


def _pack_band2(rooms: list[dict], bx: float, by: float,
                bw: float, bh: float) -> list[dict]:
    """
    Band 2 special layout:
    - Corridor spine runs down the center (4ft wide)
    - Left side: kitchen, store, etc.
    - Right side: bathrooms
    """
    placed = []

    # Corridor is 4 ft wide, centered
    corridor_w = min(4.0, bw * 0.18)
    side_w = (bw - corridor_w) / 2
    corridor_x = bx + side_w
    left_x = bx
    right_x = corridor_x + corridor_w

    # Separate rooms: corridor vs left vs right
    left_rooms = []
    right_rooms = []
    has_corridor = False

    for r in rooms:
        if r["type"] == "corridor":
            has_corridor = True
            continue
        elif r["type"] in ("kitchen", "store", "dining"):
            left_rooms.append(r)
        elif r["type"] in ("bathroom", "master_bath", "toilet"):
            right_rooms.append(r)
        else:
            # Other service rooms — distribute evenly
            if len(left_rooms) <= len(right_rooms):
                left_rooms.append(r)
            else:
                right_rooms.append(r)

    # Place corridor spanning the full band height
    if has_corridor:
        placed.append({
            "type": "corridor", "label": "Corridor",
            "x": round(corridor_x, 2),
            "y": round(by, 2),
            "width": round(corridor_w, 2),
            "height": round(bh, 2),
            "zone": 2, "band": 2,
        })

    # Pack left side rooms vertically (stacked top to bottom)
    if left_rooms:
        _pack_side_vertical(left_rooms, left_x, by, side_w, bh, placed)
    else:
        # If no left rooms, extend corridor to fill
        pass

    # Pack right side rooms vertically
    if right_rooms:
        _pack_side_vertical(right_rooms, right_x, by, side_w, bh, placed)

    return placed


def _pack_side_vertical(rooms: list[dict], sx: float, sy: float,
                        sw: float, sh: float, placed: list[dict]):
    """Pack rooms vertically in a side strip."""
    total_pref_h = sum(r.get("pref_h", r["min_h"]) for r in rooms)
    current_y = sy

    for i, room in enumerate(rooms):
        if i == len(rooms) - 1:
            rh = round(sy + sh - current_y, 2)
        else:
            rh = round(sh * (room.get("pref_h", room["min_h"]) / total_pref_h), 2)
            rh = max(rh, room["min_h"])

        if current_y + rh > sy + sh:
            rh = round(sy + sh - current_y, 2)

        placed.append({
            "type": room["type"],
            "label": room["label"],
            "x": round(sx, 2),
            "y": round(current_y, 2),
            "width": round(sw, 2),
            "height": round(rh, 2),
            "zone": room.get("zone", 2),
            "band": 2,
        })
        current_y = round(current_y + rh, 2)


def _verify_no_overlaps(placed: list[dict]):
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
                # Auto-resolve: shrink the second room slightly
                b["width"] = round(b["width"] - 0.1, 2)


def _rects_overlap(a: dict, b: dict) -> bool:
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
def add_doors_and_windows(placed_rooms: list[dict],
                          usable_w: float, usable_l: float,
                          facing: str) -> tuple[list[DoorData], list[WindowData]]:
    """Add doors between adjacent rooms and windows on exterior walls."""
    doors = []
    windows = []
    door_count = 0
    win_count = 0

    # Build a lookup for quick adjacency checks
    room_map = {r["label"]: r for r in placed_rooms}

    # Find the living room for main entrance
    living = None
    for r in placed_rooms:
        if r["type"] == "living":
            living = r
            break

    # Main entrance door on the road-facing wall of living room
    if living:
        door_count += 1
        if facing == "south":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id="living", wall="south",
                x=living["x"] + living["width"] * 0.4,
                y=living["y"],
                width=3.5,
            ))
        elif facing == "north":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id="living", wall="north",
                x=living["x"] + living["width"] * 0.4,
                y=living["y"] + living["height"],
                width=3.5,
            ))
        elif facing == "east":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id="living", wall="east",
                x=living["x"] + living["width"],
                y=living["y"] + living["height"] * 0.4,
                width=3.5,
            ))
        elif facing == "west":
            doors.append(DoorData(
                id=f"door_{door_count:02d}", type="main",
                room_id="living", wall="west",
                x=living["x"],
                y=living["y"] + living["height"] * 0.4,
                width=3.5,
            ))

    # Interior doors between adjacent rooms
    for i, a in enumerate(placed_rooms):
        for j in range(i + 1, len(placed_rooms)):
            b = placed_rooms[j]
            shared = _shared_wall(a, b)
            if shared:
                wall_side, sx, sy = shared
                door_count += 1
                doors.append(DoorData(
                    id=f"door_{door_count:02d}", type="interior",
                    room_id=a.get("type", ""),
                    wall=wall_side,
                    x=round(sx, 2), y=round(sy, 2),
                    width=3.0,
                ))

    # Windows on exterior walls
    for r in placed_rooms:
        if r["type"] in ("corridor", "store"):
            continue  # No windows for corridor or store

        eps = 0.3
        # South wall (y == 0 means touching front setback boundary)
        if r["y"] <= eps:
            win_count += 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=r["type"],
                wall="south",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # North wall (touching rear boundary)
        if abs(r["y"] + r["height"] - usable_l) <= eps:
            win_count += 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=r["type"],
                wall="north",
                x=r["x"] + r["width"] * 0.3,
                y=r["y"] + r["height"],
                width=min(4.0, r["width"] * 0.35),
            ))
        # West wall (x == 0 means touching left setback boundary)
        if r["x"] <= eps:
            win_count += 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=r["type"],
                wall="west",
                x=r["x"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))
        # East wall (touching right boundary)
        if abs(r["x"] + r["width"] - usable_w) <= eps:
            win_count += 1
            windows.append(WindowData(
                id=f"win_{win_count:02d}", room_id=r["type"],
                wall="east",
                x=r["x"] + r["width"],
                y=r["y"] + r["height"] * 0.3,
                width=min(4.0, r["height"] * 0.35),
            ))

    return doors, windows


def _shared_wall(a: dict, b: dict) -> tuple[str, float, float] | None:
    """
    Check if rooms a and b share a wall segment.
    Returns (wall_side, door_x, door_y) or None.
    """
    eps = 0.3
    # a's right edge == b's left edge (vertical wall between them)
    if abs(a["x"] + a["width"] - b["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy > 3.0:
            mid_y = (oy + ey) / 2 - 1.5
            return "east", a["x"] + a["width"], mid_y

    # b's right edge == a's left edge
    if abs(b["x"] + b["width"] - a["x"]) < eps:
        oy = max(a["y"], b["y"])
        ey = min(a["y"] + a["height"], b["y"] + b["height"])
        if ey - oy > 3.0:
            mid_y = (oy + ey) / 2 - 1.5
            return "west", a["x"], mid_y

    # a's top edge == b's bottom edge (horizontal wall)
    if abs(a["y"] + a["height"] - b["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox > 3.0:
            mid_x = (ox + ex) / 2 - 1.5
            return "north", mid_x, a["y"] + a["height"]

    # b's top edge == a's bottom edge
    if abs(b["y"] + b["height"] - a["y"]) < eps:
        ox = max(a["x"], b["x"])
        ex = min(a["x"] + a["width"], b["x"] + b["width"])
        if ex - ox > 3.0:
            mid_x = (ox + ex) / 2 - 1.5
            return "south", mid_x, a["y"]

    return None


# ─────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────
async def generate_plan_deterministic(req: PlanRequest) -> PlanResponse:
    """
    Generate a floor plan using deterministic BSP packing.
    LLM is used ONLY for vastu_score and architect_note.
    """
    uw = round(req.plot_width - SETBACKS["left"] - SETBACKS["right"], 2)
    ul = round(req.plot_length - SETBACKS["front"] - SETBACKS["rear"], 2)

    # Step 1: Try LLM for Vastu advice (non-critical)
    vastu_score = 75.0
    architect_note = "Plan designed with optimal room placement following Indian residential standards and Vastu principles."

    try:
        extras_str = ", ".join(req.extras) if req.extras else "none"
        user_msg = (
            f"Plot: {req.plot_width}x{req.plot_length} ft, "
            f"{req.bedrooms} BHK, road faces {req.facing}, "
            f"extras: {extras_str}. "
            f"Usable area after setbacks: {uw}x{ul} ft."
        )
        llm_result = await call_openrouter(LLM_SYSTEM_PROMPT, user_msg)
        vastu_score = float(llm_result.get("vastu_score", 75))
        architect_note = llm_result.get("architect_note", architect_note)
        log.info("LLM advice received: vastu_score=%.0f", vastu_score)
    except Exception as e:
        log.warning("LLM call failed (using defaults): %s", e)
        # Continue with defaults — system works without LLM

    # Step 2: Build room specs
    room_specs = build_room_list(req.bedrooms, req.extras, uw, ul)

    # Step 3: Pack rooms deterministically
    placed = pack_rooms_bsp(room_specs, uw, ul)

    # Step 4: Add doors and windows
    doors, windows = add_doors_and_windows(placed, uw, ul, req.facing)

    # Step 5: Build response
    rooms = []
    for i, p in enumerate(placed):
        rooms.append(RoomData(
            id=f"{p['type']}_{i+1:02d}",
            type=p["type"],
            label=p["label"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            area=round(p["width"] * p["height"], 1),
            zone=("public" if p["zone"] == 1
                  else "service" if p["zone"] == 2
                  else "private"),
            band=p["band"],
        ))

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
    )
