"""
Prompt Builder — coordinate-first compact prompt for OpenRouter planning.
"""
from __future__ import annotations
from models import PlanRequest

SYSTEM_PROMPT = (
    "You are NAKSHA-AI. Generate strict JSON only for a residential floor plan. "
    "Use only the numeric coordinate ranges provided. "
    "Do not output markdown or explanations."
)


def _room_program(req: PlanRequest) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = [
        {"type": "living", "label": "Living Room", "min_w": 12.0, "min_h": 12.0, "zone": 1},
        {"type": "dining", "label": "Dining", "min_w": 9.0, "min_h": 9.0, "zone": 1},
        {"type": "kitchen", "label": "Kitchen", "min_w": 8.0, "min_h": 9.0, "zone": 2},
        {"type": "corridor", "label": "Corridor", "min_w": 3.5, "min_h": 8.0, "zone": 2},
        {"type": "master_bedroom", "label": "Master Bedroom", "min_w": 11.0, "min_h": 11.0, "zone": 3},
        {"type": "master_bath", "label": "Master Bath", "min_w": 5.0, "min_h": 7.0, "zone": 2},
    ]

    if req.bedrooms >= 2:
        rooms.append({"type": "bedroom", "label": "Bedroom 2", "min_w": 10.0, "min_h": 10.0, "zone": 3})
        rooms.append({"type": "bathroom", "label": "Bathroom 2", "min_w": 5.0, "min_h": 6.0, "zone": 2})
    if req.bedrooms >= 3:
        rooms.append({"type": "bedroom", "label": "Bedroom 3", "min_w": 10.0, "min_h": 10.0, "zone": 3})
        rooms.append({"type": "bathroom", "label": "Bathroom 3", "min_w": 5.0, "min_h": 6.0, "zone": 2})
    if req.bedrooms >= 4:
        rooms.append({"type": "bedroom", "label": "Bedroom 4", "min_w": 10.0, "min_h": 10.0, "zone": 3})
        rooms.append({"type": "bathroom", "label": "Bathroom 4", "min_w": 5.0, "min_h": 6.0, "zone": 2})

    extras = {str(x).strip().lower() for x in (req.extras or [])}
    if "pooja" in extras:
        rooms.append({"type": "pooja", "label": "Pooja Room", "min_w": 4.0, "min_h": 5.0, "zone": 1})
    if "study" in extras:
        rooms.append({"type": "study", "label": "Study", "min_w": 8.0, "min_h": 9.0, "zone": 3})
    if "store" in extras:
        rooms.append({"type": "store", "label": "Store", "min_w": 4.0, "min_h": 5.0, "zone": 2})
    if "balcony" in extras:
        rooms.append({"type": "balcony", "label": "Balcony", "min_w": 4.0, "min_h": 7.0, "zone": 1})
    if "garage" in extras:
        rooms.append({"type": "garage", "label": "Garage", "min_w": 10.0, "min_h": 18.0, "zone": 1})
    if "utility" in extras:
        rooms.append({"type": "utility", "label": "Utility", "min_w": 4.0, "min_h": 5.0, "zone": 2})
    if "foyer" in extras:
        rooms.append({"type": "foyer", "label": "Foyer", "min_w": 4.0, "min_h": 4.0, "zone": 1})
    if "staircase" in extras or req.floors >= 2:
        rooms.append({"type": "staircase", "label": "Staircase", "min_w": 6.0, "min_h": 8.0, "zone": 2})

    return rooms


def _zone_bounds(room_type: str, zone: int, uw: float, ul: float) -> tuple[float, float, float, float]:
    front_max = ul * 0.32
    service_min = ul * 0.22
    service_max = ul * 0.78
    private_min = ul * 0.45

    x0, y0, x1, y1 = 0.0, 0.0, uw, ul
    if zone == 1:
        y1 = front_max
    elif zone == 2:
        y0, y1 = service_min, service_max
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
        y0, y1 = 0.0, ul

    return round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)


def build_master_prompt(req: PlanRequest) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the LLM call."""
    uw = round(req.plot_width - 7.0, 2)
    ul = round(req.plot_length - 11.5, 2)
    rooms = _room_program(req)

    type_counts: dict[str, int] = {}
    lines: list[str] = []
    for room in rooms:
        room_type = str(room["type"])
        type_counts[room_type] = type_counts.get(room_type, 0) + 1
        room_id = f"{room_type}_{type_counts[room_type]:02d}"
        zone = int(room["zone"])
        min_w = float(room["min_w"])
        min_h = float(room["min_h"])
        x0, y0, x1, y1 = _zone_bounds(room_type, zone, uw, ul)
        x_max = round(max(x0, x1 - min_w), 2)
        y_max = round(max(y0, y1 - min_h), 2)
        lines.append(
            f"{room_id} ({room_type}): x {x0} to {x_max}, y {y0} to {y_max}, min width {min_w}, min height {min_h}."
        )

    msg = (
        f"Plot usable area is {uw} x {ul} feet. "
        f"Plot size is {req.plot_width} x {req.plot_length} feet, facing {req.facing}, {req.bedrooms}BHK.\n"
        "Place rooms only in these exact coordinate ranges:\n"
        + "\n".join(lines)
        + "\nReturn only JSON matching this schema exactly:\n"
        "{\n"
        "  \"plot_boundary\": [{\"x\":0,\"y\":0},{\"x\":0,\"y\":0},{\"x\":0,\"y\":0},{\"x\":0,\"y\":0}],\n"
        "  \"rooms\": [{\"id\":\"living_01\",\"type\":\"living\",\"label\":\"Living Room\",\"x\":0,\"y\":0,\"width\":12,\"height\":12,\"area\":144,\"polygon\":[{\"x\":0,\"y\":0},{\"x\":12,\"y\":0},{\"x\":12,\"y\":12},{\"x\":0,\"y\":12}],\"zone\":\"public\",\"band\":1,\"color\":\"#E8F5E9\"}],\n"
        "  \"doors\": [],\n"
        "  \"windows\": [],\n"
        "  \"metadata\": {\"bhk\":2,\"vastu_score\":75,\"adjacency_score\":80,\"architect_note\":\"\",\"vastu_issues\":[]}\n"
        "}\n"
        "Return only JSON."
    )

    return SYSTEM_PROMPT, msg
