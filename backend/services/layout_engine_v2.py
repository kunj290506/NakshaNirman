"""
Deterministic ground-floor layout engine for NakshaNirman.

Implements a single, geometry-first 3-band strategy:
- Public band (front)
- Corridor band (middle)
- Private band (rear)

All units are feet / square feet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


FT_PER_M = 3.28084
SQFT_TO_SQM = 0.092903


ROOM_RULES: Dict[str, Dict[str, float]] = {
    "living": {"min_w": 11.0, "min_h": 13.0, "max_aspect": 1.8},
    "master_bedroom": {"min_w": 11.0, "min_h": 12.0, "max_aspect": 1.6},
    "bedroom": {"min_w": 10.0, "min_h": 10.0, "max_aspect": 1.6},
    "kitchen": {"min_w": 8.0, "min_h": 9.0, "max_aspect": 2.0},
    "dining": {"min_w": 9.0, "min_h": 10.0, "max_aspect": 1.8},
    "attached_bathroom": {"min_w": 5.0, "min_h": 7.0, "max_aspect": 2.0},
    "common_bathroom": {"min_w": 5.0, "min_h": 6.0, "max_aspect": 2.0},
    "pooja": {"min_w": 4.0, "min_h": 4.0, "max_aspect": 1.5},
    "study": {"min_w": 8.0, "min_h": 9.0, "max_aspect": 1.5},
    "store": {"min_w": 5.0, "min_h": 5.0, "max_aspect": 1.5},
    "balcony": {"min_w": 5.0, "min_h": 7.0, "max_aspect": 2.5},
    "corridor": {"min_w": 3.5, "min_h": 3.5, "max_aspect": 99.0},
}


@dataclass
class Rect:
    id: str
    type: str
    label: str
    x: float
    y: float
    width: float
    height: float
    zone: str
    attached_to: Optional[str] = None
    vastu_note: str = ""
    architect_note: str = ""

    @property
    def area(self) -> float:
        return round_half(self.width * self.height)


def round_half(value: float) -> float:
    return round(value * 2.0) / 2.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _room_min_width(room_type: str) -> float:
    key = "attached_bathroom" if room_type == "attached_bathroom" else room_type
    return ROOM_RULES.get(key, {}).get("min_w", 4.0)


def _room_min_height(room_type: str) -> float:
    key = "attached_bathroom" if room_type == "attached_bathroom" else room_type
    return ROOM_RULES.get(key, {}).get("min_h", 4.0)


def _aspect_limit(room_type: str) -> float:
    key = "attached_bathroom" if room_type == "attached_bathroom" else room_type
    return ROOM_RULES.get(key, {}).get("max_aspect", 2.0)


def _normalize_facing(value: Any) -> str:
    facing = str(value or "east").strip().lower()
    if facing not in {"east", "west", "north", "south"}:
        return "east"
    return facing


def _normalize_extras(extras: Any) -> List[str]:
    if not extras:
        return []
    out: List[str] = []
    for item in extras:
        token = str(item).strip().lower()
        if token in {"pooja", "study", "store", "balcony", "garage"} and token not in out:
            out.append(token)
    return out


def _setbacks(plot_w: float, plot_l: float) -> Dict[str, float]:
    area_sqm = (plot_w * plot_l) * SQFT_TO_SQM
    if area_sqm < 75:
        front = 5.0
    elif area_sqm <= 150:
        front = 6.5
    else:
        front = 10.0
    return {"front": front, "rear": 5.0, "left": 3.3, "right": 3.3}


def _derive_plot(width: Optional[float], length: Optional[float], total_area: Optional[float]) -> Tuple[float, float, float]:
    if width and length:
        return float(width), float(length), float(width * length)
    if total_area and total_area > 0:
        # Deterministic default proportion close to common Indian plots.
        w = (float(total_area) * 0.75) ** 0.5
        l = float(total_area) / w
        return round_half(w), round_half(l), float(total_area)
    # Last deterministic fallback.
    return 30.0, 40.0, 1200.0


def _rect_overlap(a: Rect, b: Rect) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _to_room_dict(room: Rect) -> Dict[str, Any]:
    return {
        "id": room.id,
        "type": room.type,
        "label": room.label,
        "x": room.x,
        "y": room.y,
        "width": room.width,
        "height": room.height,
        "area": round_half(room.width * room.height),
        "zone": room.zone,
        "vastu_note": room.vastu_note,
        "architect_note": room.architect_note,
        "attached_to": room.attached_to,
    }


def _make_public_rooms(
    usable_w: float,
    public_depth: float,
    extras: List[str],
    vastu: bool,
    facing: str,
    small_plot: bool,
    variant: str,
) -> List[Rect]:
    rooms: List[Rect] = []

    include_dining = not small_plot
    include_pooja_public = vastu and "pooja" in extras and facing == "east" and usable_w >= 24.0

    living_w = round_half(usable_w * 0.45)
    dining_w = round_half(usable_w * 0.28) if include_dining else 0.0
    kitchen_w = round_half(usable_w - living_w - dining_w)

    # Keep kitchen valid by shrinking dining first.
    while kitchen_w < 8.0 and dining_w > 0.0:
        dining_w = round_half(max(0.0, dining_w - 1.0))
        kitchen_w = round_half(usable_w - living_w - dining_w)

    if include_pooja_public:
        pooja_w = 4.0
        living_w = round_half(max(8.0, living_w - pooja_w))
    else:
        pooja_w = 0.0

    order = ["living", "dining", "kitchen"] if include_dining else ["living", "kitchen"]
    if variant == "redesign":
        if include_dining:
            order = ["dining", "living", "kitchen"]
        else:
            order = ["kitchen", "living"]

    x = 0.0
    if include_pooja_public:
        rooms.append(
            Rect(
                id="room_pooja_public",
                type="pooja",
                label="Pooja Room",
                x=x,
                y=0.0,
                width=pooja_w,
                height=max(4.0, round_half(public_depth * 0.40)),
                zone="public",
                vastu_note="North-East placement — Vastu compliant",
                architect_note="Pooja kept away from bathrooms",
            )
        )
        x += pooja_w

    for token in order:
        width = living_w if token == "living" else dining_w if token == "dining" else kitchen_w
        if width <= 0.0:
            continue
        label = {
            "living": "Living Room",
            "dining": "Dining Room" if include_dining else "Living + Dining",
            "kitchen": "Kitchen",
        }[token]
        vastu_note = ""
        if token == "living" and vastu:
            vastu_note = "North-East/North preferred living placement"
        if token == "kitchen" and vastu:
            vastu_note = "South-East — Vastu compliant"

        rooms.append(
            Rect(
                id=f"room_public_{token}",
                type=token,
                label=label,
                x=round_half(x),
                y=0.0,
                width=round_half(width),
                height=public_depth,
                zone="public",
                vastu_note=vastu_note,
            )
        )
        x += width

    # Force exact band fit.
    if rooms:
        delta = round_half(usable_w - (rooms[-1].x + rooms[-1].width))
        rooms[-1].width = round_half(max(rooms[-1].width + delta, 3.0))

    if "balcony" in extras:
        living = next((r for r in rooms if r.type == "living"), None)
        if living is not None:
            rooms.append(
                Rect(
                    id="room_balcony",
                    type="balcony",
                    label="Balcony",
                    x=living.x,
                    y=0.0,
                    width=round_half(max(5.0, min(living.width, 8.0))),
                    height=round_half(max(5.0, min(public_depth * 0.45, 7.0))),
                    zone="public",
                    architect_note="Attached to living room front edge",
                )
            )

    return rooms


def _make_private_rooms(
    usable_w: float,
    public_depth: float,
    corridor_depth: float,
    private_depth: float,
    bedrooms: int,
    bathrooms: int,
    extras: List[str],
    vastu: bool,
    variant: str,
) -> List[Rect]:
    rooms: List[Rect] = []
    y0 = public_depth + corridor_depth

    master_h = round_half(max(12.0, private_depth * 0.75))
    master_w = 12.0
    att_w = 6.0

    regular_bedrooms = max(0, bedrooms - 1)
    common_bathrooms = max(0, bathrooms - 1)

    tokens: List[Tuple[str, float]] = [("master_bedroom", master_w), ("attached_bathroom", att_w)]
    for _ in range(regular_bedrooms):
        tokens.append(("bedroom", 11.0))
    for _ in range(common_bathrooms):
        tokens.append(("common_bathroom", 6.0))

    if "study" in extras:
        tokens.append(("study", 8.0))
    if "store" in extras:
        tokens.append(("store", 5.5))
    if "pooja" in extras and not (vastu and "pooja" in extras):
        tokens.append(("pooja", 4.5))

    if variant == "redesign":
        # Keep zoning same but swap within private order.
        body = [t for t in tokens[2:] if t[0] != "common_bathroom"]
        baths = [t for t in tokens[2:] if t[0] == "common_bathroom"]
        body.reverse()
        tokens = tokens[:2] + body + baths

    total_pref = sum(w for _, w in tokens)
    widths: List[float] = []

    if total_pref <= usable_w:
        widths = [w for _, w in tokens]
        # Fill exact width with last regular bedroom, otherwise master.
        gap = round_half(usable_w - sum(widths))
        idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i][0] == "bedroom":
                idx = i
                break
        if idx == -1:
            idx = 0
        widths[idx] = round_half(widths[idx] + gap)
    else:
        # Over-constrained narrow plots: proportional shrink (deterministic).
        scale = usable_w / total_pref
        for typ, pref in tokens:
            min_soft = 5.0 if typ in {"bedroom", "master_bedroom"} else max(3.5, _room_min_width(typ) * 0.7)
            widths.append(round_half(max(min_soft, pref * scale)))

        # Force exact sum to usable width.
        drift = round_half(usable_w - sum(widths))
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i][0] in {"bedroom", "master_bedroom", "study", "store"}:
                widths[i] = round_half(max(4.0, widths[i] + drift))
                drift = 0.0
                break
        if drift != 0.0:
            widths[-1] = round_half(max(3.5, widths[-1] + drift))

    x = 0.0
    room_idx = 1
    master_room_id = ""

    for (typ, _), w in zip(tokens, widths):
        if typ == "master_bedroom":
            rid = f"room_private_{room_idx}"
            master_room_id = rid
            rooms.append(
                Rect(
                    id=rid,
                    type="master_bedroom",
                    label="Master Bedroom",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=master_h,
                    zone="private",
                    vastu_note="South-West placement preferred" if vastu else "",
                )
            )
        elif typ == "attached_bathroom":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="bathroom",
                    label="Attached Bathroom",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=master_h,
                    zone="service",
                    attached_to=master_room_id or None,
                )
            )
        elif typ == "bedroom":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="bedroom",
                    label="Bedroom",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=private_depth,
                    zone="private",
                )
            )
        elif typ == "common_bathroom":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="bathroom",
                    label="Common Bathroom",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=round_half(private_depth * 0.5),
                    zone="service",
                )
            )
        elif typ == "study":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="study",
                    label="Study Room",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=round_half(max(9.0, private_depth * 0.6)),
                    zone="private",
                )
            )
        elif typ == "store":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="store",
                    label="Store Room",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=round_half(max(5.0, private_depth * 0.4)),
                    zone="service",
                )
            )
        elif typ == "pooja":
            rooms.append(
                Rect(
                    id=f"room_private_{room_idx}",
                    type="pooja",
                    label="Pooja Room",
                    x=round_half(x),
                    y=y0,
                    width=round_half(w),
                    height=round_half(max(4.0, private_depth * 0.4)),
                    zone="private",
                )
            )

        x += w
        room_idx += 1

    # Exact private-width closure.
    if rooms:
        private_rooms = [r for r in rooms if r.y >= y0]
        if private_rooms:
            right = max(r.x + r.width for r in private_rooms)
            drift = round_half(usable_w - right)
            private_rooms[-1].width = round_half(max(3.5, private_rooms[-1].width + drift))

    return rooms


def _snap_rooms(rooms: List[Rect]) -> None:
    for room in rooms:
        room.x = round_half(room.x)
        room.y = round_half(room.y)
        room.width = round_half(room.width)
        room.height = round_half(room.height)


def _inside_bounds(room: Rect, usable_w: float, usable_l: float) -> bool:
    return room.x >= -1e-6 and room.y >= -1e-6 and (room.x + room.width) <= usable_w + 1e-6 and (room.y + room.height) <= usable_l + 1e-6


def _validate_and_fix(
    rooms: List[Rect],
    corridor: Rect,
    usable_w: float,
    usable_l: float,
    warnings: List[str],
) -> None:
    # Check 1: inside bounds
    for room in rooms:
        if room.x + room.width > usable_w:
            room.width = round_half(max(3.0, usable_w - room.x))
            warnings.append(f"Clipped {room.label} to right boundary")
        if room.y + room.height > usable_l:
            room.height = round_half(max(3.0, usable_l - room.y))
            warnings.append(f"Clipped {room.label} to top boundary")

    # Check 2: overlaps
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            overlap = _rect_overlap(rooms[i], rooms[j])
            if overlap > 0.5:
                shift = round_half((overlap / max(rooms[j].height, 1.0)) + 0.5)
                rooms[j].x = round_half(min(usable_w - rooms[j].width, rooms[j].x + shift))
                warnings.append(f"Resolved overlap by shifting {rooms[j].label}")

    # Check 3: corridor
    if corridor.height < 3.5:
        corridor.height = 3.5
        warnings.append("Corridor widened to minimum 3.5 ft")
    corridor.width = usable_w

    # Check 6: master has attached bath
    master = next((r for r in rooms if r.type == "master_bedroom"), None)
    attached = next((r for r in rooms if r.type == "bathroom" and r.attached_to == (master.id if master else None)), None)
    if master and attached is None:
        w = min(6.0, max(4.0, usable_w - (master.x + master.width)))
        bath = Rect(
            id="room_auto_attached_bath",
            type="bathroom",
            label="Attached Bathroom",
            x=round_half(master.x + master.width),
            y=master.y,
            width=round_half(max(4.0, w)),
            height=round_half(max(7.0, min(master.height, usable_l - master.y))),
            zone="service",
            attached_to=master.id,
            architect_note="Auto-added to satisfy attached bath requirement",
        )
        if _inside_bounds(bath, usable_w, usable_l):
            rooms.append(bath)
            warnings.append("Added missing attached bathroom for master bedroom")

    # Check 7: aspect ratio
    for room in rooms:
        aspect = max(room.width, room.height) / max(min(room.width, room.height), 0.1)
        lim = _aspect_limit(room.type if room.type != "bathroom" else ("attached_bathroom" if room.attached_to else "common_bathroom"))
        if aspect > lim:
            if room.width > room.height:
                room.width = round_half(max(room.height * lim, 3.0))
            else:
                room.height = round_half(max(room.width * lim, 3.0))
            warnings.append(f"Adjusted aspect ratio for {room.label}")


def _room_for_id(rooms: List[Rect], room_id: str) -> Optional[Rect]:
    return next((r for r in rooms if r.id == room_id), None)


def _build_doors(rooms: List[Rect], corridor: Rect) -> List[Dict[str, Any]]:
    doors: List[Dict[str, Any]] = []

    living = next((r for r in rooms if r.type == "living"), None)
    kitchen = next((r for r in rooms if r.type == "kitchen"), None)
    if living:
        doors.append(
            {
                "id": "door_main",
                "room_id": living.id,
                "wall": "south",
                "position": round_half(living.width / 2),
                "width": 3.0,
                "type": "main",
            }
        )

    if kitchen:
        doors.append(
            {
                "id": "door_kitchen",
                "room_id": kitchen.id,
                "wall": "east",
                "position": round_half(kitchen.height / 2),
                "width": 2.5,
                "type": "internal",
            }
        )

    d_idx = 1
    for room in rooms:
        if room.type in {"master_bedroom", "bedroom"}:
            doors.append(
                {
                    "id": f"door_bed_{d_idx}",
                    "room_id": room.id,
                    "wall": "south",
                    "position": round_half(room.width / 2),
                    "width": 3.0,
                    "type": "internal",
                }
            )
            d_idx += 1

        if room.type == "bathroom" and room.attached_to:
            doors.append(
                {
                    "id": f"door_att_bath_{d_idx}",
                    "room_id": room.id,
                    "wall": "west",
                    "position": round_half(room.height / 2),
                    "width": 2.5,
                    "type": "bathroom",
                }
            )
            d_idx += 1

        if room.type == "bathroom" and not room.attached_to:
            doors.append(
                {
                    "id": f"door_com_bath_{d_idx}",
                    "room_id": room.id,
                    "wall": "south",
                    "position": round_half(room.width / 2),
                    "width": 2.5,
                    "type": "bathroom",
                }
            )
            d_idx += 1

    _ = corridor
    return doors


def _build_windows(rooms: List[Rect]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    idx = 1

    for room in rooms:
        if room.type == "living":
            windows.append({"id": f"window_{idx}", "room_id": room.id, "wall": "south", "position": round_half(room.width * 0.25), "width": 4.0, "height": 4.0})
            idx += 1
            windows.append({"id": f"window_{idx}", "room_id": room.id, "wall": "south", "position": round_half(room.width * 0.70), "width": 4.0, "height": 4.0})
            idx += 1
        elif room.type in {"master_bedroom", "bedroom"}:
            windows.append({"id": f"window_{idx}", "room_id": room.id, "wall": "north", "position": round_half(room.width * 0.5), "width": 3.5, "height": 4.0})
            idx += 1
        elif room.type == "kitchen":
            windows.append({"id": f"window_{idx}", "room_id": room.id, "wall": "east", "position": round_half(room.height * 0.5), "width": 2.5, "height": 3.0})
            idx += 1
        elif room.type == "bathroom":
            windows.append({"id": f"window_{idx}", "room_id": room.id, "wall": "north", "position": round_half(room.width * 0.5), "width": 1.5, "height": 2.0})
            idx += 1

    return windows


def _score(
    rooms: List[Rect],
    usable_area: float,
    vastu: bool,
    facing: str,
    warnings: List[str],
) -> Dict[str, float]:
    coverage = 0.0
    if usable_area > 0:
        coverage = sum(r.area for r in rooms if r.type != "balcony") / usable_area

    nbc_hits = 0
    nbc_total = 0
    for room in rooms:
        if room.type == "corridor":
            continue
        key = "common_bathroom" if room.type == "bathroom" and not room.attached_to else "attached_bathroom" if room.type == "bathroom" else room.type
        if key not in ROOM_RULES:
            continue
        nbc_total += 1
        min_w = ROOM_RULES[key]["min_w"]
        min_h = ROOM_RULES[key]["min_h"]
        if room.width >= min_w and room.height >= min_h:
            nbc_hits += 1

    nbc_score = 100.0 if nbc_total == 0 else round((nbc_hits / nbc_total) * 100.0, 1)

    if not vastu:
        vastu_score = 100.0
    else:
        vastu_score = 95.0
        if facing in {"south", "west"}:
            vastu_score -= 10.0
        if any("Vastu" in w for w in warnings):
            vastu_score -= 10.0
        vastu_score = clamp(vastu_score, 60.0, 100.0)

    space_eff = round(clamp(coverage * 100.0, 0.0, 100.0), 1)
    overall = round((vastu_score + nbc_score + space_eff) / 3.0, 1)

    return {
        "vastu_compliance": round(vastu_score, 1),
        "nbc_compliance": round(nbc_score, 1),
        "space_efficiency": space_eff,
        "overall": overall,
    }


def _architect_notes(vastu: bool, facing: str, warnings: List[str]) -> List[str]:
    notes = [
        "Living Room placed in front public band for guest access and daylight",
        "Kitchen placed in the front zone with external wall access for ventilation",
        "Private rear band isolates bedrooms from the entry-facing public area",
        "Corridor provides clear circulation between public and private zones",
    ]
    if vastu:
        notes[1] = "Kitchen in South-East per Vastu Shastra"
        notes.append("Master Bedroom in South-West owner zone per Vastu intent")
        notes.append("Living Room kept towards North/East side where feasible")
        if facing in {"south", "west"}:
            notes.append("Main entrance is not East/North facing; flagged as Vastu advisory")
    if warnings:
        notes.extend([f"Validation warning: {w}" for w in warnings[:3]])
    return notes


def _generate_once(input_data: Dict[str, Any], variant: str = "base") -> Dict[str, Any]:
    width, length, total_area = _derive_plot(input_data.get("plot_width"), input_data.get("plot_length"), input_data.get("total_area"))
    bedrooms = int(input_data.get("bedrooms") or 2)
    bathrooms = int(input_data.get("bathrooms") or bedrooms)
    facing = _normalize_facing(input_data.get("facing"))
    vastu = bool(input_data.get("vastu", True))
    extras = _normalize_extras(input_data.get("extras", []))

    setbacks = _setbacks(width, length)
    usable_w = round_half(width - setbacks["left"] - setbacks["right"])
    usable_l = round_half(length - setbacks["front"] - setbacks["rear"])
    usable_area = round_half(usable_w * usable_l)

    if usable_w <= 8.0 or usable_l <= 12.0:
        return {"error": "Plot too small after setbacks for a valid ground-floor layout"}

    public_depth = round_half(usable_l * 0.40)
    corridor_depth = 4.0 if usable_w > 25.0 else 3.5
    private_depth = round_half(usable_l - public_depth - corridor_depth)

    if private_depth < 12.0:
        public_depth = round_half(max(8.0, public_depth - 2.0))
        private_depth = round_half(usable_l - public_depth - corridor_depth)

    if private_depth < 10.0:
        return {"error": "Plot depth is insufficient for required private-zone sizing"}

    small_plot = usable_area < 600.0

    public_rooms = _make_public_rooms(usable_w, public_depth, extras, vastu, facing, small_plot, variant)
    corridor = Rect(
        id="room_corridor",
        type="corridor",
        label="Corridor",
        x=0.0,
        y=public_depth,
        width=usable_w,
        height=corridor_depth,
        zone="circulation",
    )
    private_rooms = _make_private_rooms(usable_w, public_depth, corridor_depth, private_depth, bedrooms, bathrooms, extras, vastu, variant)

    all_rooms = public_rooms + [corridor] + private_rooms

    # Add garage as service room at front edge inside usable bounds for deterministic rendering.
    if "garage" in extras:
        garage_w = round_half(min(usable_w * 0.45, 18.0))
        garage_h = round_half(min(public_depth, 9.0))
        all_rooms.append(
            Rect(
                id="room_garage",
                type="garage",
                label="Garage",
                x=round_half(usable_w - garage_w),
                y=0.0,
                width=garage_w,
                height=garage_h,
                zone="service",
                architect_note="Garage requested; represented near front edge",
            )
        )

    _snap_rooms(all_rooms)

    warnings: List[str] = []
    for _ in range(3):
        prev_warn_count = len(warnings)
        _validate_and_fix(all_rooms, corridor, usable_w, usable_l, warnings)
        if len(warnings) == prev_warn_count:
            break

    # Final strict inside clamp.
    for room in all_rooms:
        room.x = round_half(clamp(room.x, 0.0, usable_w))
        room.y = round_half(clamp(room.y, 0.0, usable_l))
        room.width = round_half(max(0.5, min(room.width, usable_w - room.x)))
        room.height = round_half(max(0.5, min(room.height, usable_l - room.y)))

    doors = _build_doors(all_rooms, corridor)
    windows = _build_windows(all_rooms)

    # Vastu warning for south/west entries.
    if vastu and facing in {"south", "west"}:
        warnings.append("Vastu advisory: entrance facing is not East/North")

    result = {
        "plot": {
            "width": usable_w,
            "length": usable_l,
            "original_width": width,
            "original_length": length,
            "setbacks": setbacks,
        },
        "rooms": [_to_room_dict(r) for r in all_rooms],
        "doors": doors,
        "windows": windows,
        "corridor": {
            "x": corridor.x,
            "y": corridor.y,
            "width": corridor.width,
            "height": corridor.height,
        },
        "design_score": _score(all_rooms, usable_area, vastu, facing, warnings),
        "architect_notes": _architect_notes(vastu, facing, warnings),
        "warnings": warnings,
        "meta": {
            "deterministic": True,
            "engine": "layout_engine_v2",
            "variant": variant,
            "total_area": total_area,
        },
    }

    return result


def generate_ground_floor_plan(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the deterministic ground-floor layout."""
    return _generate_once(input_data, variant="base")


def redesign_ground_floor_plan(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an alternate layout by swapping intra-band ordering only."""
    return _generate_once(input_data, variant="redesign")
