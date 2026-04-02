"""
Prompt Builder — compact NAKSHA-AI prompt for fast LLM floor plan generation.
Stripped to essentials: zone rules, room sizes, JSON schema. ~60% smaller than v1.
"""
from __future__ import annotations
import uuid
from models import PlanRequest

CLIMATE_MAP: dict[str, str] = {
    "ahmedabad": "hot-dry", "surat": "warm-humid", "vadodara": "hot-dry",
    "rajkot": "hot-dry", "gandhinagar": "hot-dry",
    "mumbai": "warm-humid", "pune": "composite", "nagpur": "hot-dry",
    "nashik": "composite",
    "jaipur": "hot-dry", "jodhpur": "hot-dry", "udaipur": "hot-dry",
    "bengaluru": "warm-humid", "mysuru": "warm-humid", "mangaluru": "warm-humid",
    "chennai": "warm-humid", "coimbatore": "warm-humid",
    "delhi": "composite", "hyderabad": "composite", "lucknow": "composite",
    "kolkata": "warm-humid", "bhopal": "composite", "indore": "composite",
    "chandigarh": "composite", "kochi": "hot-humid", "trivandrum": "hot-humid",
}

SYSTEM_PROMPT = r"""You are NAKSHA-AI, an expert Indian residential architect. Generate a practical floor plan as strict JSON.

RULES:
- Usable area = plot minus setbacks (front:6.5, rear:5, left:3.5, right:3.5 ft)
- All coordinates in feet from bottom-left of usable area
- NO room may exceed usable area bounds
- NO two rooms may overlap
- Keep dimensions realistic and round to nearest 0.5 ft
- Keep circulation practical: avoid dead-end micro spaces
- Every response must be a fresh layout concept; do not repeat prior arrangements
- DO NOT add unrequested optional rooms
- Return exactly the requested bedroom count (master_bedroom + bedroom)
- Rooms may be non-rectangular; for angled/free-form walls include polygon points

ZONE LAYOUT (3 horizontal bands):
- FRONT (y=0, ~30% depth): living, dining, pooja, balcony, foyer
- MIDDLE (~25% depth): kitchen, corridor, bathrooms, utility, store
- REAR (~45% depth): bedrooms, master bath, study

ROOM SIZES (min width×height):
- living:12×12, dining:10×10, kitchen:8×9, master_bedroom:12×12
- bedroom:10×10, master_bath:5×7, bathroom:5×6, corridor:3.5×full
- pooja:5×5, study:8×9, store:5×5, balcony:4×8, garage:10×18
- utility:4×5, foyer:4×4, staircase:6×8

VASTU:
- Kitchen in SE, Master bedroom in SW, Pooja in NE
- Main entrance faces east or north

ADJACENCY:
- Kitchen adjacent to dining
- Master bath attached to master bedroom
- All bedrooms connect to corridor
- Corridor connects bedrooms to living area

OUTPUT: Return ONLY this JSON (no markdown, no explanation):
{
  "plot_boundary": [
    {"x":0,"y":0}, {"x":23,"y":0}, {"x":23,"y":38.5}, {"x":0,"y":38.5}
  ],
  "rooms": [
    {
      "id":"type_01","type":"room_type","label":"Name",
      "x":0,"y":0,"width":12,"height":12,"area":144,
      "polygon":[{"x":0,"y":0},{"x":12,"y":0},{"x":11,"y":8},{"x":0,"y":10}],
      "zone":"public|service|private","band":1,"color":"#hex"
    }
  ],
  "doors": [
    {"id":"door_01","type":"main|interior","room_id":"living_01","wall":"south","x":6,"y":0,"width":3}
  ],
  "windows": [
    {"id":"win_01","room_id":"living_01","wall":"east","x":0,"y":6,"width":4}
  ],
  "metadata": {
    "bhk":3,"vastu_score":85,"adjacency_score":80,
    "architect_note":"Brief design rationale",
    "vastu_issues":[]
  }
}

Room types: living,dining,kitchen,corridor,master_bedroom,bedroom,master_bath,bathroom,pooja,study,store,balcony,garage,utility,foyer,staircase
Colors: living=#E8F5E9,dining=#FFF3E0,kitchen=#FFEBEE,master_bedroom=#E3F2FD,bedroom=#E3F2FD,master_bath=#E0F7FA,bathroom=#E0F7FA,corridor=#F5F5F5,pooja=#FFF8E1,study=#EDE7F6,store=#EFEBE9,balcony=#E8F5E9,garage=#ECEFF1"""


def build_master_prompt(req: PlanRequest) -> tuple[str, str]:
    """Build (system_prompt, user_message) for the LLM call."""
    design_nonce = uuid.uuid4().hex[:10]
    uw = round(req.plot_width - 7.0, 2)
    ul = round(req.plot_length - 11.5, 2)
    area = round(uw * ul, 1)

    city = (req.city or "").lower().strip()
    climate = CLIMATE_MAP.get(city, "composite")

    extras = ", ".join(req.extras) if req.extras else "none"

    auto_baths = 2 if req.bedrooms <= 2 else 3 if req.bedrooms == 3 else 4
    requested_baths = req.bathrooms_target if req.bathrooms_target > 0 else auto_baths

    required_rooms = ["living", "dining", "kitchen", "corridor", "master_bedroom", "master_bath"]
    if req.bedrooms >= 2:
        required_rooms.extend(["bedroom", "bathroom"])
    if req.bedrooms >= 3:
        required_rooms.extend(["bedroom", "bathroom", "bathroom"])
    if req.bedrooms >= 4:
        required_rooms.extend(["bedroom", "bathroom"])

    required_rooms_text = ", ".join(required_rooms)

    msg = f"""REQUEST_NONCE: {design_nonce}
BRIEF: {req.plot_width}×{req.plot_length}ft plot, {req.facing}-facing, {req.bedrooms}BHK
Usable: {uw}×{ul}ft = {area}sqft
Extras: {extras}
Family: {req.family_type or 'nuclear'}
City: {req.city or 'general'} ({climate} climate)
  Freshness rules:
  - Treat REQUEST_NONCE {design_nonce} as a new independent design run
  - Do not reuse room placements from earlier responses
  - Keep the output practical, clean, and distinct
  Program rules:
  - Exact bedrooms required: {req.bedrooms}
  - Target bathrooms: {requested_baths}
  - Mandatory rooms must include: {required_rooms_text}
  - Optional extras allowed ONLY from: {extras}
  Customization details:
  - Floors: {req.floors}
  - Style: {req.design_style}
  - Kitchen preference: {req.kitchen_preference}
  - Parking slots requested: {req.parking_slots}
  - Priority weights (1-5): vastu={req.vastu_priority}, daylight={req.natural_light_priority}, privacy={req.privacy_priority}, storage={req.storage_priority}
  - Elder friendly circulation: {'yes' if req.elder_friendly else 'no'}
  - Work-from-home utility: {'yes' if req.work_from_home else 'no'}
"""

    if req.family_notes:
        msg += f"Family notes: {req.family_notes}\n"
    if req.notes:
        msg += f"Additional custom notes: {req.notes}\n"

    msg += "\nGenerate the complete floor plan JSON now."

    return SYSTEM_PROMPT, msg
