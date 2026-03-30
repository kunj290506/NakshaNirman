"""
Layout engine — builds prompts, calls LLM, validates response, handles retries.
"""
from __future__ import annotations
import logging
from models import PlanRequest, PlanResponse, PlotInfo, RoomData, DoorData, WindowData
from llm import call_openrouter

log = logging.getLogger("layout")

# ─────────────────────────────────────────────────────────────
# System prompt (static)
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are NAKSHA AI, a senior Indian residential architect with 35 years of experience.
You design ground-floor house plans that follow Indian building codes, NBC 2016 standards,
and Vastu Shastra principles. You think step by step before placing any room.

YOUR JOB: Receive plot dimensions and requirements. Return a valid JSON floor plan.

THINKING PROCESS (do this silently before outputting JSON):
1. Calculate usable area after setbacks (front 6.5ft, rear 5ft, left 3.5ft, right 3.5ft)
2. Divide usable area into three bands: public front 30%, service middle 27%, private rear 43%
3. Place public rooms (living, dining, pooja) in front band near road
4. Place service rooms (kitchen, bathrooms) in middle band
5. Place bedrooms in rear band for privacy
6. Assign minimum sizes: living 12x14ft, master 12x12ft, bedroom 11x11ft, kitchen 9x10ft, bath 5x7ft
7. Ensure kitchen touches exterior wall
8. Ensure all bedrooms touch exterior wall for windows
9. Ensure kitchen is adjacent to dining
10. Validate no overlaps, no Vastu violations, minimum 85% area coverage

VASTU RULES TO ENFORCE:
- Never place toilet or kitchen in north-east corner
- Place kitchen toward south-east if possible
- Place master bedroom toward south-west
- Place pooja in north-east if requested
- Keep center area lighter (corridor preferred there)

OUTPUT FORMAT: Return ONLY this JSON structure, no markdown, no explanation:
{
  "plot": {
    "width": number,
    "length": number,
    "usable_width": number,
    "usable_length": number,
    "road_side": "south",
    "setbacks": {"front": 6.5, "rear": 5, "left": 3.5, "right": 3.5}
  },
  "rooms": [
    {
      "id": "living_01",
      "type": "living",
      "label": "Drawing Room",
      "x": number,
      "y": number,
      "width": number,
      "height": number,
      "area": number,
      "zone": "public",
      "band": 1
    }
  ],
  "doors": [
    {
      "id": "door_main",
      "type": "main",
      "room_id": "living_01",
      "wall": "south",
      "x": number,
      "y": number,
      "width": 3.5
    }
  ],
  "windows": [
    {
      "id": "win_01",
      "room_id": "living_01",
      "wall": "south",
      "x": number,
      "y": number,
      "width": 4.0
    }
  ],
  "vastu_score": number,
  "architect_note": "brief professional explanation of design decisions"
}

ROOM TYPE VALUES TO USE:
living, dining, kitchen, master_bedroom, bedroom, bathroom, toilet,
pooja, study, store, balcony, garage, corridor"""


# ─────────────────────────────────────────────────────────────
# User message builder (dynamic)
# ─────────────────────────────────────────────────────────────
def _build_user_message(req: PlanRequest, *, suffix: str = "") -> str:
    extras_str = ", ".join(req.extras) if req.extras else "none"
    msg = f"""Design a complete ground floor plan with these specifications:
Plot size: {req.plot_width} feet wide by {req.plot_length} feet long
Bedrooms: {req.bedrooms} BHK (so {req.bedrooms} bedrooms total, {req.bedrooms} bathrooms minimum)
Road faces: {req.facing} side
Vastu compliance: required
Extra rooms requested: {extras_str}
Family type: nuclear family

Requirements:
- Create one master bedroom with attached bathroom
- Create {req.bedrooms - 1} additional bedrooms each with bathroom access
- Create living room, dining room, and kitchen as mandatory rooms
- All habitable rooms must have exterior wall access for windows
- Kitchen must be adjacent to dining room
- Main entrance must open directly into living room
- Include interior corridor connecting all zones

Return the complete JSON floor plan now."""
    if suffix:
        msg += f"\n\n{suffix}"
    return msg


# ─────────────────────────────────────────────────────────────
# Validate LLM response
# ─────────────────────────────────────────────────────────────
def _validate_plan(raw: dict, req: PlanRequest) -> PlanResponse:
    """Parse and lightly validate the LLM output."""
    plot_raw = raw.get("plot", {})
    usable_w = plot_raw.get("usable_width", req.plot_width - 7)
    usable_l = plot_raw.get("usable_length", req.plot_length - 11.5)

    plot = PlotInfo(
        width=req.plot_width,
        length=req.plot_length,
        usable_width=usable_w,
        usable_length=usable_l,
        road_side=plot_raw.get("road_side", req.facing),
        setbacks=plot_raw.get(
            "setbacks",
            {"front": 6.5, "rear": 5, "left": 3.5, "right": 3.5},
        ),
    )

    rooms = []
    for r in raw.get("rooms", []):
        rooms.append(
            RoomData(
                id=r.get("id", "room"),
                type=r.get("type", "room"),
                label=r.get("label", r.get("type", "Room")),
                x=float(r.get("x", 0)),
                y=float(r.get("y", 0)),
                width=float(r.get("width", 10)),
                height=float(r.get("height", 10)),
                area=float(r.get("area", r.get("width", 10) * r.get("height", 10))),
                zone=r.get("zone", "public"),
                band=int(r.get("band", 1)),
            )
        )

    doors = []
    for d in raw.get("doors", []):
        doors.append(
            DoorData(
                id=d.get("id", "door"),
                type=d.get("type", "interior"),
                room_id=d.get("room_id", ""),
                wall=d.get("wall", "south"),
                x=float(d.get("x", 0)),
                y=float(d.get("y", 0)),
                width=float(d.get("width", 3.5)),
            )
        )

    windows = []
    for w in raw.get("windows", []):
        windows.append(
            WindowData(
                id=w.get("id", "win"),
                room_id=w.get("room_id", ""),
                wall=w.get("wall", "south"),
                x=float(w.get("x", 0)),
                y=float(w.get("y", 0)),
                width=float(w.get("width", 4.0)),
            )
        )

    if len(rooms) < 3:
        raise ValueError(f"Plan has only {len(rooms)} rooms, need at least 3")

    return PlanResponse(
        plot=plot,
        rooms=rooms,
        doors=doors,
        windows=windows,
        vastu_score=raw.get("vastu_score", 0),
        architect_note=raw.get("architect_note", ""),
    )


# ─────────────────────────────────────────────────────────────
# Hardcoded fallback plan (2 BHK)
# ─────────────────────────────────────────────────────────────
def _fallback_plan(req: PlanRequest) -> PlanResponse:
    """Minimal fallback when all LLM retries fail."""
    uw = req.plot_width - 7
    ul = req.plot_length - 11.5
    return PlanResponse(
        plot=PlotInfo(
            width=req.plot_width,
            length=req.plot_length,
            usable_width=uw,
            usable_length=ul,
            road_side=req.facing,
        ),
        rooms=[
            RoomData(id="living_01", type="living", label="Living Room",
                     x=0, y=0, width=uw * 0.55, height=ul * 0.30, area=0,
                     zone="public", band=1),
            RoomData(id="dining_01", type="dining", label="Dining Room",
                     x=uw * 0.55, y=0, width=uw * 0.45, height=ul * 0.30, area=0,
                     zone="public", band=1),
            RoomData(id="kitchen_01", type="kitchen", label="Kitchen",
                     x=uw * 0.55, y=ul * 0.30, width=uw * 0.45, height=ul * 0.27, area=0,
                     zone="service", band=2),
            RoomData(id="bath_common", type="bathroom", label="Bathroom",
                     x=0, y=ul * 0.30, width=uw * 0.25, height=ul * 0.27, area=0,
                     zone="service", band=2),
            RoomData(id="corridor_01", type="corridor", label="Corridor",
                     x=uw * 0.25, y=ul * 0.30, width=uw * 0.30, height=ul * 0.27, area=0,
                     zone="service", band=2),
            RoomData(id="master_01", type="master_bedroom", label="Master Bedroom",
                     x=0, y=ul * 0.57, width=uw * 0.55, height=ul * 0.43, area=0,
                     zone="private", band=3),
            RoomData(id="bed_02", type="bedroom", label="Bedroom 2",
                     x=uw * 0.55, y=ul * 0.57, width=uw * 0.45, height=ul * 0.43, area=0,
                     zone="private", band=3),
        ],
        doors=[
            DoorData(id="door_main", type="main", room_id="living_01", wall="south",
                     x=uw * 0.25, y=0, width=3.5),
        ],
        vastu_score=60,
        architect_note="Fallback plan generated due to AI service issues.",
    )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
MAX_RETRIES = 2


async def generate_plan(req: PlanRequest) -> PlanResponse:
    """Generate a floor plan. Retries up to MAX_RETRIES, then falls back."""
    suffix = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt == 1:
                suffix = (
                    "IMPORTANT: Return ONLY the JSON object. "
                    "No text before or after. Start with { and end with }"
                )
            elif attempt == 2:
                suffix = (
                    "CRITICAL: In your previous attempt, the response was invalid. "
                    "Return ONLY valid JSON matching the exact schema specified. "
                    f"All rooms must fit within usable area of "
                    f"{req.plot_width - 7}x{req.plot_length - 11.5} feet. "
                    "No overlapping rooms."
                )

            user_msg = _build_user_message(req, suffix=suffix)
            raw = await call_openrouter(SYSTEM_PROMPT, user_msg)
            plan = _validate_plan(raw, req)

            # Recalculate areas
            for room in plan.rooms:
                room.area = round(room.width * room.height, 1)

            log.info("Plan generated on attempt %d with %d rooms", attempt + 1, len(plan.rooms))
            return plan

        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt == MAX_RETRIES:
                log.error("All retries failed, using fallback plan")
                fb = _fallback_plan(req)
                for room in fb.rooms:
                    room.area = round(room.width * room.height, 1)
                return fb

    # Should never reach here
    return _fallback_plan(req)
