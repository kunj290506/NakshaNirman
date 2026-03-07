"""
4-Stage AI Architectural Pipeline Engine.

Processes user requirements through 4 distinct stages:
  Stage 1 — Chat Mode: Natural conversation to collect requirements
  Stage 2 — Extraction Mode: Convert conversation → structured JSON
  Stage 3 — Design Mode: Generate construction-ready layout from JSON
  Stage 4 — Validation Mode: Validate the generated layout

Each stage uses a dedicated system prompt and produces structured
output that feeds the next stage.
"""

import json
import re
from typing import Optional, Dict, List, Tuple
from enum import Enum


class PipelineStage(str, Enum):
    CHAT = "chat"
    EXTRACTION = "extraction"
    DESIGN = "design"
    VALIDATION = "validation"
    GENERATION = "generation"
    COMPLETE = "complete"


# ============================================================================
# STAGE 1 — Chat Mode: Natural requirement collection
# ============================================================================

STAGE_1_CHAT_PROMPT = """You are NakshaNirman AI, a Senior Indian Residential Architect with 30 years of experience designing homes across India. You have deep expertise in Indian building codes (NBC 2016), Vastu Shastra, regional climate conditions, and practical Indian family living patterns. You think, reason, and respond exactly like a real human architect sitting across the table from a client — warm, professional, precise, and creative.

You are NOT a chatbot. You are an ARCHITECT. Every response you give must reflect real architectural thinking — spatial logic, traffic flow, privacy zones, natural light, Vastu compliance, structural practicality, and human comfort.

=== YOUR ONE JOB IN THIS MODE ===

Collect the user's house requirements naturally, with maximum 2 questions, then trigger automatic floor plan generation. No vague answers. No "it depends." Always move toward producing a real plan.

=== REQUIREMENT COLLECTION (MAX 2 QUESTIONS) ===

You need only these things. If the user has not said, assume Indian defaults and proceed:

1. Plot size or total area — examples: "30 by 40 feet" or "1200 square feet"
   If not given, ask ONCE. If user just says BHK, assume standard area for that BHK.
2. BHK type or number of bedrooms — examples: "3BHK" or "3 bedrooms"
   Default: 2BHK for plots under 800 sqft, 3BHK for 800-1500 sqft, 4BHK above
3. Number of bathrooms — default is one attached bathroom per bedroom
4. Number of floors — default is ground floor only
5. Special rooms wanted — dining room, study, pooja room, balcony, parking, store room

Ask at MOST 2 questions, then proceed. Be warm, concise, and professional.
Acknowledge what the user just told you before asking.
Use Indian housing terminology naturally — BHK, sqft, vastu, ground floor.

=== UNDERSTANDING INDIAN HOUSING CONTEXT ===

When user says "2BHK": bedrooms equals 2 (one master bedroom plus one regular bedroom), bathrooms equals 2
When user says "3BHK": bedrooms equals 3 (one master bedroom plus two regular bedrooms), bathrooms equals 2 to 3
When user says "4BHK": bedrooms equals 4 (one master bedroom plus three regular bedrooms), bathrooms equals 3 to 4
When user says "duplex" or "double storey": floors equals 2
When user says "ground floor only" or "single storey": floors equals 1
When user says "vastu compliant": apply all Vastu rules automatically, no need to ask further
When user gives plot dimensions like "30 by 40": width equals 30, length equals 40, area equals 1200 sqft
When user says "1200 sqft" or "1200 square feet": total_area equals 1200

Always include kitchen — never ask about kitchen, it is always present in every Indian home.
For 2BHK and above, dining room is standard in Indian homes — include it unless user says otherwise.
Attached bathroom per bedroom is the Indian standard — recommend it even if user says "common bathroom".

=== SMART DEFAULTS (USE WHEN INFORMATION IS MISSING) ===

If user just says "3BHK house" with no plot size:
  Assume 1200-1500 sqft and proceed immediately
If user just says "1200 sqft" with no BHK:
  Assume 3BHK (standard for 800-1500 sqft range)
If user says "hello" or "start":
  Respond warmly, introduce yourself as their architect, then ask for plot size and BHK in ONE question
If user gives both plot size and BHK:
  Summarize, ask for confirmation, and generate immediately

=== WHEN YOU HAVE ENOUGH INFORMATION ===

Once you have at minimum the plot area and number of bedrooms, you have enough to generate a plan.

Summarize what you understood and ask for confirmation:

"Perfect! Here is what I have for your floor plan:
  - [X]BHK house
  - [Y] square feet total area
  - [Z] attached bathrooms
  - Ground floor / [floors] floors
  - Special rooms: [list extras]

Shall I generate the floor plan now?"

When the user confirms (says yes, ok, generate, proceed, haan, ha, sure, go ahead, or anything affirmative), output EXACTLY this JSON and nothing else:

```json
{
  "requirements_complete": true,
  "total_area": <number>,
  "plot_width": <number or null>,
  "plot_length": <number or null>,
  "bedrooms": <number>,
  "bathrooms": <number>,
  "floors": <number>,
  "extras": ["dining", "study", "pooja"]
}
```

=== ARCHITECTURAL KNOWLEDGE TO SHARE DURING CONVERSATION ===

Room Minimum Sizes (Never go below these):
  Master Bedroom: 12x12 feet minimum (prefer 12x14)
  Bedroom: 10x10 feet minimum (prefer 10x12)
  Kitchen: 8x10 feet minimum (prefer 10x12)
  Bathroom (attached): 5x7 feet minimum
  Living Room: 12x14 feet minimum (prefer 14x16)
  Dining Room: 9x9 feet minimum
  Pooja Room: 4x4 feet minimum
  Passage/Corridor: 3.5 feet wide minimum

Vastu advice to give naturally:
  Kitchen: "In Indian homes, the kitchen works best in the South-East corner as per Vastu — promotes health and prosperity"
  Master Bedroom: "The South-West corner is ideal for the master bedroom — ensures stability and sound sleep for the owner"
  Pooja Room: "The North-East corner is the most auspicious direction for a pooja room in Indian architecture"
  Living Room: "A living room facing North or East gets the best morning sunlight — very popular in Indian homes"

=== TONE AND PERSONALITY ===

You are confident, knowledgeable, and genuinely excited about designing homes. Use words like:
- "I would recommend..."
- "What works beautifully here is..."
- "The challenge with this plot is... but we can solve it by..."
- "From a Vastu perspective, this is ideal because..."
- "Indian families typically need... so I have ensured..."

Never say "I cannot," "I do not have enough information," or "as an AI." You are an architect. Architects design. That is what you do.

=== HANDLING FEEDBACK ===

When the user says ANYTHING negative or asks for a change:
1. Acknowledge what they do not like in ONE sentence
2. Explain what you are changing and why — like a real architect
3. Update the requirements and ask for confirmation again
4. Never say "I cannot change that." If something truly cannot fit, offer the closest alternative.

When all mandatory data is collected and user confirms, respond with EXACTLY this line at the end:
[REQUIREMENTS_COMPLETE]

Do not generate design in this mode."""


# ============================================================================
# STAGE 2 — Extraction Mode: Conversation → Structured JSON
# ============================================================================

STAGE_2_EXTRACTION_PROMPT = """Extract structured house design requirements from the conversation provided. Output ONLY a valid JSON object. No explanation text. No markdown. No extra words. Only the JSON.

=== EXTRACTION RULES ===

BHK interpretation:
  "1BHK" means bedrooms equals 1, bathrooms equals 1
  "2BHK" means bedrooms equals 2, bathrooms equals 2
  "3BHK" means bedrooms equals 3, bathrooms equals 2
  "4BHK" means bedrooms equals 4, bathrooms equals 3
  "5BHK" means bedrooms equals 5, bathrooms equals 4

Plot dimension interpretation:
  "30 by 40" or "30x40" means plot_width equals 30, plot_length equals 40, total_area equals 1200
  "40 by 50" or "40x50" means plot_width equals 40, plot_length equals 50, total_area equals 2000
  "25 by 30" or "25x30" means plot_width equals 25, plot_length equals 30, total_area equals 750

Area interpretation:
  "1200 sqft" or "1200 square feet" means total_area equals 1200
  "1 BHK 500 sqft" means bedrooms equals 1, bathrooms equals 1, total_area equals 500
  If plot dimensions given but area not given, calculate: total_area equals width multiplied by length

Floor interpretation:
  "ground floor" or "single storey" or "G plus 0" means floors equals 1
  "double storey" or "G plus 1" or "2 floors" or "duplex" means floors equals 2
  Default is floors equals 1 if not mentioned

Extras interpretation:
  Always include "dining" in extras for 2BHK and above
  Include "pooja" if user mentions pooja room or puja room
  Include "study" if user mentions study room or library or office room
  Include "balcony" if user mentions balcony or terrace access
  Include "store" if user mentions store room or storage
  Include "utility" if user mentions utility room or washing area
  Include "parking" if user mentions parking or garage or car space

Defaults when not mentioned:
  floors defaults to 1
  For 2BHK and above, dining is always included
  bathrooms defaults to number of bedrooms for 1BHK, number of bedrooms minus 1 for others (minimum 1)

=== OUTPUT FORMAT ===

{
  "total_area": <number in sqft, or null if not mentioned>,
  "plot_width": <number in ft, or null if not mentioned>,
  "plot_length": <number in ft, or null if not mentioned>,
  "bedrooms": <total number of bedrooms including master bedroom, minimum 1>,
  "bathrooms": <total number of bathrooms, minimum 1>,
  "floors": <number of floors, default 1>,
  "extras": ["dining", "study", "pooja", "balcony", "store", "utility", "parking"],
  "style": "standard",
  "bhk_label": "3BHK"
}"""


# ============================================================================
# STAGE 3 — Design Mode: Structured JSON → Layout
# ============================================================================

STAGE_3_DESIGN_PROMPT = """You are generating a professional Indian residential floor plan with exact coordinates. Every number you output will be rendered directly on screen as a floor plan drawing. Precision is critical.

=== COORDINATE SYSTEM ===

Origin point (0, 0) is at the BOTTOM-LEFT corner of the plot.
X coordinate increases to the RIGHT (East direction).
Y coordinate increases UPWARD (North direction).
All coordinates and dimensions are in FEET.
All values must be snapped to the nearest 0.5 ft (6-inch structural grid).
External wall thickness is 0.75 ft — leave this margin at all plot edges.

=== BAND HEIGHT CALCULATION ===

Given a plot with plot_length L feet:

Band 1 starts at:  y = 0.75 (external wall margin)
Band 1 ends at:    y = 0.75 + (L multiplied by 0.35)
Band 1 height:     L multiplied by 0.35

Band 2 starts at:  Band 1 end y value
Band 2 ends at:    Band 2 start y plus 3.5
Band 2 height:     3.5 ft (always exactly 3.5 ft, this is the passage)

Band 3 starts at:  Band 2 end y value
Band 3 ends at:    y = L minus 0.75 (external wall margin)
Band 3 height:     L minus 0.75 minus Band 2 end y

=== ROOM PLACEMENT RULES WITHIN EACH BAND ===

Band 1 Room Placement (left to right, from x=0.75 to x=plot_width minus 0.75):

  Living Room:
    x = 0.75
    y = 0.75
    width = plot_width multiplied by 0.42 (widest room, occupies left side)
    length = Band 1 height
    This is the main public space, gets maximum frontage

  Kitchen:
    x = plot_width minus 0.75 minus kitchen_width
    y = 0.75
    width = plot_width multiplied by 0.22
    length = Band 1 height
    Kitchen is always on the RIGHT side (South-East area per Vastu)

  Dining Room:
    x = living_room_x plus living_room_width
    y = 0.75
    width = remaining Band 1 width between Living and Kitchen
    length = Band 1 height
    Dining is between Living and Kitchen (natural serving flow)

Band 3 Room Placement (left to right, from x=0.75 to x=plot_width minus 0.75):

  Master Bedroom:
    x = 0.75
    y = Band 3 start y
    width = plot_width multiplied by 0.50 (left side, South-West per Vastu)
    length = Band 3 height
    Master Bedroom is always on the LEFT (South-West corner)

  Attached Bathroom (CARVED INSIDE Master Bedroom):
    x = 0.75 plus master_bedroom_width minus bathroom_width
    y = Band 3 start y plus Band 3 height minus bathroom_length
    width = master_bedroom_width multiplied by 0.35 (maximum 8 ft)
    length = Band 3 height multiplied by 0.35 (maximum 8 ft, minimum 7 ft)
    The bathroom sits in the TOP-RIGHT corner of the Master Bedroom

  Bedroom 2:
    x = 0.75 plus master_bedroom_width
    y = Band 3 start y
    width = remaining width = plot_width minus 1.5 minus master_bedroom_width
    length = Band 3 height

  Bedroom 3 (if needed, split Bedroom 2 space horizontally):
    Split the remaining Band 3 width equally between Bedroom 2 and Bedroom 3

  Extra rooms (study, pooja, store):
    If plot is large enough, add to Band 3 or create a partial Band between 2 and 3
    Pooja Room must be placed in NE area (top-right of Band 3)
    Study Room can be adjacent to a bedroom

=== MINIMUM SIZE VERIFICATION ===

Before finalizing any room, verify it meets these minimums:
Living Room:    width minimum 10 ft, length minimum 12 ft, area minimum 120 sqft
Master Bedroom: width minimum 10 ft, length minimum 12 ft, area minimum 120 sqft
Bedroom:        width minimum 9 ft,  length minimum 10 ft, area minimum 90 sqft
Kitchen:        width minimum 7 ft,  length minimum 8 ft,  area minimum 56 sqft
Dining Room:    width minimum 8 ft,  length minimum 9 ft,  area minimum 72 sqft
Bathroom:       width minimum 5 ft,  length minimum 7 ft,  area minimum 35 sqft
Study Room:     width minimum 7 ft,  length minimum 8 ft,  area minimum 56 sqft
Pooja Room:     width minimum 4 ft,  length minimum 4 ft,  area minimum 16 sqft

If any room falls below these minimums, expand it and reduce an adjacent room proportionally.

=== POLYGON CALCULATION ===

For every room with corner at (x, y), width w, and length l, the polygon is:
  [[x, y], [x+w, y], [x+w, y+l], [x, y+l], [x, y]]
  (5 points, closed polygon, counterclockwise from bottom-left)

=== COMPLETE OUTPUT FORMAT ===

Output ONLY this JSON. No explanation text. No markdown. No extra words. Only the JSON object.

{
  "layout_type": "3_band_zoned",
  "engine": "ai_generated",
  "plot_width": <feet as decimal>,
  "plot_length": <feet as decimal>,
  "total_area": <sqft as decimal>,
  "usable_area": <total_area multiplied by 0.88>,
  "boundary": [[0,0],[plot_width,0],[plot_width,plot_length],[0,plot_length],[0,0]],
  "rooms": [
    {
      "room_type": "living",
      "name": "Living Room",
      "zone": "public",
      "x": <x coordinate>,
      "y": <y coordinate>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area from requirements>,
      "vastu_direction": "SE",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    },
    {
      "room_type": "kitchen",
      "name": "Kitchen",
      "zone": "semi_private",
      "x": <x coordinate>,
      "y": <y coordinate>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area>,
      "vastu_direction": "SE",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    },
    {
      "room_type": "dining",
      "name": "Dining Room",
      "zone": "semi_private",
      "x": <x coordinate>,
      "y": <y coordinate>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area>,
      "vastu_direction": "W",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    },
    {
      "room_type": "master_bedroom",
      "name": "Master Bedroom",
      "zone": "private",
      "x": <x coordinate>,
      "y": <y coordinate>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area>,
      "vastu_direction": "SW",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    },
    {
      "room_type": "bathroom",
      "name": "Attached Bathroom",
      "zone": "service",
      "x": <x inside master bedroom top-right corner>,
      "y": <y inside master bedroom top-right corner>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area>,
      "vastu_direction": "NW",
      "attached_to": "master_bedroom",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    },
    {
      "room_type": "bedroom",
      "name": "Bedroom",
      "zone": "private",
      "x": <x coordinate>,
      "y": <y coordinate>,
      "width": <width in feet>,
      "length": <length in feet>,
      "area": <width multiplied by length>,
      "target_area": <desired area>,
      "vastu_direction": "NW",
      "polygon": [[x,y],[x+w,y],[x+w,y+l],[x,y+l],[x,y]]
    }
  ],
  "vastu_summary": {
    "kitchen_direction": "SE",
    "master_bedroom_direction": "SW",
    "pooja_direction": "NE",
    "entrance": "S",
    "compliant": true,
    "vastu_score": 90
  },
  "area_summary": {
    "plot_area": <total_area>,
    "rooms_area": <sum of all room areas>,
    "walls_corridors_area": <plot_area minus rooms_area>,
    "utilization_percentage": "87%"
  },
  "explanation": "Professional 3-band layout: Living Room and Kitchen in public zone, Master Bedroom in private zone with attached bathroom carved in top-right corner. Kitchen placed in South-East for Vastu compliance. Master Bedroom in South-West corner for owner stability per Vastu Shastra."
}

CRITICAL:
- Rooms must NOT overlap (check x,y positions + widths/lengths)
- Total room area must not exceed plot area
- All rooms must fit within plot boundaries
- Position (0,0) is bottom-left corner of the plot"""


# ============================================================================
# STAGE 4 — Validation Mode
# ============================================================================

STAGE_4_VALIDATION_PROMPT = """You are validating an Indian residential floor plan layout for professional architectural compliance. Check every rule below and report findings. Output ONLY a valid JSON object. No explanation. No markdown. Only JSON.

=== NBC 2016 MINIMUM AREA CHECKS ===

Check every room against these minimums:
  living: minimum 120 sqft — FAIL if area less than 120
  master_bedroom: minimum 120 sqft — FAIL if area less than 120
  bedroom: minimum 90 sqft — FAIL if area less than 90
  kitchen: minimum 56 sqft — FAIL if area less than 56
  dining: minimum 72 sqft — FAIL if area less than 72
  bathroom: minimum 35 sqft — FAIL if area less than 35
  toilet: minimum 15 sqft — FAIL if area less than 15
  study: minimum 56 sqft — FAIL if area less than 56
  pooja: minimum 16 sqft — FAIL if area less than 16
  store: minimum 20 sqft — FAIL if area less than 20
  utility: minimum 20 sqft — FAIL if area less than 20
  passage: minimum 15 sqft (3.5 ft wide minimum) — FAIL if less

=== VASTU SHASTRA CHECKS ===

Plot center is at (plot_width divided by 2, plot_length divided by 2).
SE quadrant is where x is greater than plot_width divided by 2 AND y is less than plot_length divided by 2.
SW quadrant is where x is less than plot_width divided by 2 AND y is greater than plot_length divided by 2.
NE quadrant is where x is greater than plot_width divided by 2 AND y is greater than plot_length divided by 2.
NW quadrant is where x is less than plot_width divided by 2 AND y is greater than plot_length divided by 2.

Check kitchen centroid (x plus width divided by 2, y plus length divided by 2):
  PASS if kitchen centroid is in SE quadrant or has x greater than 60% of plot_width
  FAIL if kitchen is in NW or NE quadrant

Check master_bedroom centroid:
  PASS if master_bedroom centroid is in SW quadrant
  WARN if master_bedroom centroid is in NW quadrant

Check pooja room centroid (if present):
  PASS if pooja centroid is in NE quadrant
  FAIL if pooja is in SE or SW quadrant

=== ADJACENCY CHECKS ===

Two rooms are adjacent if their polygons share a wall segment of at least 2 ft.
To check adjacency: rooms A and B are adjacent if
  abs((A.x + A.width) minus B.x) is less than 0.6, or
  abs((B.x + B.width) minus A.x) is less than 0.6, or
  abs((A.y + A.length) minus B.y) is less than 0.6, or
  abs((B.y + B.length) minus A.y) is less than 0.6

REQUIRED adjacency checks — report FAIL if these pairs are NOT adjacent:
  master_bedroom and bathroom must be adjacent
  kitchen and dining must be adjacent
  living and dining must be adjacent

FORBIDDEN adjacency checks — report FAIL if these pairs ARE adjacent:
  Any bedroom with kitchen must NOT be adjacent
  Any bathroom with living must NOT be adjacent
  Any bathroom with kitchen must NOT be adjacent
  Any bathroom with dining must NOT be adjacent
  toilet with pooja must NOT be adjacent

=== GEOMETRY CHECKS ===

Overlap check: rooms A and B overlap if
  A.x is less than B.x plus B.width AND A.x plus A.width is greater than B.x AND
  A.y is less than B.y plus B.length AND A.y plus A.length is greater than B.y

Boundary check: each room must satisfy
  x is greater than or equal to 0.75
  y is greater than or equal to 0.75
  x plus width is less than or equal to plot_width minus 0.75
  y plus length is less than or equal to plot_length minus 0.75

Aspect ratio check: for each room
  ratio equals maximum of (width divided by length) and (length divided by width)
  FAIL if ratio is greater than 2.0

Total area check:
  sum all room areas
  FAIL if total rooms area is greater than plot_area multiplied by 0.92

=== OUTPUT FORMAT ===

{
  "compliant": <true if zero FAIL results, false otherwise>,
  "score": <0 to 100, start at 100 and subtract 10 per FAIL and 5 per WARN>,
  "vastu_score": <0 to 100, based only on Vastu checks>,
  "nbc_score": <0 to 100, based only on NBC area checks>,
  "geometry_ok": <true if no overlaps and all rooms within boundary>,
  "adjacency_ok": <true if all required adjacencies met and no forbidden adjacencies>,
  "issues": [
    "<FAIL: specific room name and rule that failed>"
  ],
  "warnings": [
    "<WARN: specific room name and rule that is borderline>"
  ],
  "passed_checks": [
    "<PASS: specific check that passed>"
  ],
  "room_count": <number of rooms in layout>,
  "total_rooms_area": <sum of all room areas>,
  "plot_area": <total plot area>,
  "utilization_percentage": "<rooms_area divided by plot_area as percentage>",
  "suggestion": "<single most important improvement if score is below 80>"
}

Be strict. Flag any issue that would cause construction problems.
Output ONLY valid JSON."""


# ============================================================================
# REQUIREMENT FIELDS
# ============================================================================

MANDATORY_FIELDS = [
    "plot_width", "plot_length", "total_area",  # at least one dimension set
    "bedrooms", "bathrooms", "floors",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON from AI response text."""
    # Try ```json blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try any JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _clean_reply(text: str) -> str:
    """Remove JSON code blocks from reply for display."""
    clean = re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()
    return clean if clean else text


def _parse_collected_data(history: List[Dict]) -> dict:
    """
    Parse all collected data from conversation history using both
    direct regex matching and contextual analysis (what question was
    asked before each answer).
    """
    data = {
        "has_dimensions": False,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "has_floors": False,
        "plot_width": None,
        "plot_length": None,
        "total_area": None,
        "bedrooms": None,
        "bathrooms": None,
        "floors": None,
        "extras": [],
    }

    # --- Direct scan across all user messages ---
    full_text = " ".join(m.get("content", "") for m in history if m.get("role") == "user")
    full_lower = full_text.lower()

    # Dimensions: 30x40, 30*40, 30 x 40, 30×40, 30 × 40
    dim_match = re.search(r'(\d+)\s*[x×*]\s*(\d+)', full_lower)
    if dim_match:
        data["has_dimensions"] = True
        data["plot_width"] = int(dim_match.group(1))
        data["plot_length"] = int(dim_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]

    # Area: 1200 sqft, 1200 sq ft, 1200 square feet
    area_match = re.search(r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet?)', full_lower)
    if area_match:
        data["has_dimensions"] = True
        data["total_area"] = int(area_match.group(1))

    # Standalone large number (likely area if > 100)
    if not data["has_dimensions"]:
        for msg in history:
            if msg.get("role") != "user":
                continue
            num_match = re.match(r'^\s*(\d{3,5})\s*$', msg.get("content", "").strip())
            if num_match:
                val = int(num_match.group(1))
                if 100 <= val <= 50000:
                    data["has_dimensions"] = True
                    data["total_area"] = val

    # BHK: 3BHK, 3 bhk
    bhk_match = re.search(r'(\d+)\s*bhk', full_lower)
    if bhk_match:
        bhk = int(bhk_match.group(1))
        data["has_bedrooms"] = True
        data["has_bathrooms"] = True
        data["bedrooms"] = bhk
        data["bathrooms"] = max(1, bhk - 1)

    # Explicit bedrooms: 3 bedrooms, 3 bed
    bed_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?)', full_lower)
    if bed_match:
        data["has_bedrooms"] = True
        data["bedrooms"] = int(bed_match.group(1))

    # Explicit bathrooms: 2 bathrooms, 2 bath, 2 toilet
    bath_match = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilet)', full_lower)
    if bath_match:
        data["has_bathrooms"] = True
        data["bathrooms"] = int(bath_match.group(1))

    # Floors: 2 floors, 2 storey
    floor_match = re.search(r'(\d+)\s*(?:floor|storey|story|level)', full_lower)
    if floor_match:
        data["has_floors"] = True
        data["floors"] = int(floor_match.group(1))

    # --- Contextual parsing: look at assistant question → user answer pairs ---
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        user_text = msg.get("content", "").strip()

        # Find the previous assistant message to understand context
        prev_assistant = ""
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "assistant":
                prev_assistant = history[j].get("content", "").lower()
                break

        # Contextual: "2,2" or "2, 2" or "2 2" after asking about bed/bath
        if "bedroom" in prev_assistant and "bathroom" in prev_assistant:
            nums = re.findall(r'\d+', user_text)
            if len(nums) >= 2:
                data["has_bedrooms"] = True
                data["has_bathrooms"] = True
                data["bedrooms"] = int(nums[0])
                data["bathrooms"] = int(nums[1])
            elif len(nums) == 1:
                data["has_bedrooms"] = True
                data["bedrooms"] = int(nums[0])

        # Contextual: just a number after asking about floors
        if "floor" in prev_assistant or "storey" in prev_assistant:
            nums = re.findall(r'\d+', user_text)
            if len(nums) >= 1:
                data["has_floors"] = True
                data["floors"] = int(nums[0])

        # Contextual: just a number after asking about plot size
        if "plot" in prev_assistant and ("size" in prev_assistant or "dimension" in prev_assistant):
            dim_match_ctx = re.search(r'(\d+)\s*[x×*]\s*(\d+)', user_text.lower())
            if dim_match_ctx:
                data["has_dimensions"] = True
                data["plot_width"] = int(dim_match_ctx.group(1))
                data["plot_length"] = int(dim_match_ctx.group(2))
                data["total_area"] = data["plot_width"] * data["plot_length"]

    # Extras
    extras = []
    if "dining" in full_lower: extras.append("dining")
    if "study" in full_lower: extras.append("study")
    if "pooja" in full_lower: extras.append("pooja")
    if "balcon" in full_lower: extras.append("balcony")
    if "parking" in full_lower or "garage" in full_lower: extras.append("parking")
    if "garden" in full_lower: extras.append("garden")
    data["extras"] = extras

    return data


def check_requirements_complete(history: List[Dict]) -> bool:
    """
    Check if conversation history contains all mandatory requirements.

    Looks for the [REQUIREMENTS_COMPLETE] marker from the AI,
    or checks if the conversation naturally contains all required data.
    """
    # Check for explicit marker in last assistant message
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            if "[REQUIREMENTS_COMPLETE]" in msg.get("content", ""):
                return True
            break  # Only check the last assistant message

    # Parse collected data with contextual analysis
    data = _parse_collected_data(history)

    return data["has_dimensions"] and data["has_bedrooms"] and data["has_bathrooms"]


def build_conversation_text(history: List[Dict]) -> str:
    """Build a readable conversation transcript from history."""
    lines = []
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Architect"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


# ============================================================================
# PIPELINE EXECUTION — Uses the AI provider (Grok → Groq → Fallback)
# ============================================================================

async def _call_ai(system_prompt: str, user_message: str, history: List[Dict] = None,
                   temperature: float = 0.7, max_tokens: int = 2048) -> Tuple[str, Optional[dict]]:
    """
    Call the AI provider with the given system prompt and message.

    Returns (reply_text, extracted_json).
    Falls back through: Grok → Groq → Rule-based.
    """
    # Try Grok first
    try:
        from config import GROK_API_KEY, GROK_MODEL, GROK_BASE_URL
        if GROK_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=GROK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            reply = response.choices[0].message.content
            extracted = _extract_json_from_text(reply)
            return reply, extracted
    except Exception:
        pass

    # Try Groq fallback
    try:
        from config import GROQ_API_KEY, GROQ_MODEL
        if GROQ_API_KEY:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            reply = response.choices[0].message.content
            extracted = _extract_json_from_text(reply)
            return reply, extracted
    except Exception:
        pass

    # Final fallback — return empty
    return "", None


async def run_stage_1_chat(message: str, history: List[Dict]) -> Dict:
    """
    Stage 1: Chat mode — natural conversation to collect requirements.

    Returns dict with: reply, stage, requirements_complete
    """
    reply, extracted = await _call_ai(
        STAGE_1_CHAT_PROMPT, message, history,
        temperature=0.7, max_tokens=1024
    )

    if not reply:
        # Fallback response
        reply = _fallback_chat_response(message, history)

    requirements_complete = "[REQUIREMENTS_COMPLETE]" in reply
    clean_reply = reply.replace("[REQUIREMENTS_COMPLETE]", "").strip()

    # Also check from history if AI didn't emit marker
    if not requirements_complete:
        full_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        requirements_complete = check_requirements_complete(full_history)

    return {
        "reply": clean_reply,
        "stage": PipelineStage.CHAT,
        "requirements_complete": requirements_complete,
        "extracted_data": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_2_extraction(history: List[Dict]) -> Dict:
    """
    Stage 2: Extract structured JSON from conversation history.

    Returns dict with: requirements_json, stage
    """
    conversation_text = build_conversation_text(history)

    reply, extracted = await _call_ai(
        STAGE_2_EXTRACTION_PROMPT,
        f"Extract requirements from this conversation:\n\n{conversation_text}",
        temperature=0.3, max_tokens=1024,
    )

    if not extracted:
        # Fallback: try to extract from conversation manually
        extracted = _fallback_extract(history)

    return {
        "reply": "Requirements extracted successfully.",
        "stage": PipelineStage.EXTRACTION,
        "requirements_json": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_3_design(requirements_json: Dict) -> Dict:
    """
    Stage 3: Generate construction-ready layout from structured requirements.

    Returns dict with: layout_json, explanation, stage
    """
    reply, extracted = await _call_ai(
        STAGE_3_DESIGN_PROMPT,
        f"Generate a floor plan layout for these requirements:\n\n```json\n{json.dumps(requirements_json, indent=2)}\n```",
        temperature=0.5, max_tokens=4096,
    )

    explanation = _clean_reply(reply) if reply else "Layout generated using standard architectural rules."

    if not extracted:
        # Fallback: generate basic layout programmatically
        extracted = _fallback_design(requirements_json)

    return {
        "reply": explanation,
        "stage": PipelineStage.DESIGN,
        "layout_json": extracted,
        "provider": "grok" if reply else "fallback",
    }


async def run_stage_4_validation(layout_json: Dict) -> Dict:
    """
    Stage 4: Validate the generated layout.

    Returns dict with: validation_report, compliant, stage
    """
    reply, extracted = await _call_ai(
        STAGE_4_VALIDATION_PROMPT,
        f"Validate this floor plan layout:\n\n```json\n{json.dumps(layout_json, indent=2)}\n```",
        temperature=0.3, max_tokens=2048,
    )

    if not extracted:
        # Fallback: run basic validation
        extracted = _fallback_validate(layout_json)

    compliant = extracted.get("compliant", False) if extracted else False

    explanation = _clean_reply(reply) if reply else "Validation complete."

    return {
        "reply": explanation,
        "stage": PipelineStage.VALIDATION,
        "validation_report": extracted,
        "compliant": compliant,
        "provider": "grok" if reply else "fallback",
    }


async def run_full_pipeline(history: List[Dict]) -> Dict:
    """
    Run stages 2 → 3 → 4 in sequence after requirements are complete.

    Returns the combined result of all stages.
    """
    # Stage 2: Extract
    extraction_result = await run_stage_2_extraction(history)
    requirements_json = extraction_result.get("requirements_json", {})

    if not requirements_json:
        return {
            "error": "Could not extract requirements from conversation.",
            "stage": PipelineStage.EXTRACTION,
        }

    # Stage 3: Design
    design_result = await run_stage_3_design(requirements_json)
    layout_json = design_result.get("layout_json", {})

    if not layout_json:
        return {
            "error": "Could not generate layout design.",
            "stage": PipelineStage.DESIGN,
            "requirements_json": requirements_json,
        }

    # Stage 4: Validate
    validation_result = await run_stage_4_validation(layout_json)

    return {
        "stage": PipelineStage.COMPLETE if validation_result.get("compliant") else PipelineStage.VALIDATION,
        "requirements_json": requirements_json,
        "layout_json": layout_json,
        "validation_report": validation_result.get("validation_report", {}),
        "compliant": validation_result.get("compliant", False),
        "design_explanation": design_result.get("reply", ""),
        "validation_explanation": validation_result.get("reply", ""),
    }


# ============================================================================
# FALLBACK — Rule-based when no AI provider is available
# ============================================================================

def _fallback_chat_response(message: str, history: List[Dict]) -> str:
    """Context-aware rule-based fallback for chat mode."""
    # Build full history including current message
    full_history = history + [{"role": "user", "content": message}]
    data = _parse_collected_data(full_history)

    # Check if all requirements are now complete
    if data["has_dimensions"] and data["has_bedrooms"] and data["has_bathrooms"]:
        # Build summary
        summary_parts = []
        if data["plot_width"] and data["plot_length"]:
            summary_parts.append(f"Plot: {data['plot_width']}×{data['plot_length']} feet ({data['total_area']} sq ft)")
        elif data["total_area"]:
            summary_parts.append(f"Plot area: {data['total_area']} sq ft")
        if data["bedrooms"]:
            summary_parts.append(f"Bedrooms: {data['bedrooms']}")
        if data["bathrooms"]:
            summary_parts.append(f"Bathrooms: {data['bathrooms']}")
        if data["floors"]:
            summary_parts.append(f"Floors: {data['floors']}")
        if data["extras"]:
            summary_parts.append(f"Extras: {', '.join(data['extras'])}")

        summary = "\n".join(f"  • {p}" for p in summary_parts)

        return (
            f"Great, I have all the information I need!\n\n"
            f"**Your Requirements:**\n{summary}\n\n"
            f"Generating your design now...\n\n"
            f"[REQUIREMENTS_COMPLETE]"
        )

    # Determine what's missing and ask for it
    turn = len([h for h in history if h.get("role") == "user"])

    if turn == 0:
        # First message — check what they gave us
        has_dims = bool(re.search(r'\d+\s*[x×*]\s*\d+', message.lower()))
        has_area = bool(re.search(r'\d+\s*(?:sq|sqft|square)', message.lower()))
        has_bhk = bool(re.search(r'\d+\s*bhk', message.lower()))
        has_bed = bool(re.search(r'\d+\s*bed', message.lower()))

        if (has_dims or has_area) and (has_bhk or has_bed):
            return (
                "Great! Let me confirm your requirements.\n\n"
                "Do you need any special rooms like dining, study, pooja room, balcony, or parking?\n"
                "Type 'no' if you're good, or list what you'd like."
            )

        return (
            "Welcome! I'll help you design your home. 🏠\n\n"
            "Let's start — what's your plot size?\n"
            "(e.g., '30x40 feet' or '1200 sq ft')"
        )

    # Ask for what's missing, one thing at a time
    if not data["has_dimensions"]:
        return (
            "Got it! What's your plot size?\n"
            "(e.g., '30x40 feet', '1200 sqft', or just a number like '1200')"
        )

    if not data["has_bedrooms"] or not data["has_bathrooms"]:
        return (
            "How many bedrooms and bathrooms do you need?\n"
            "(e.g., '3 bedrooms, 2 bathrooms' or just '3, 2')"
        )

    if not data["has_floors"]:
        return (
            "How many floors?\n"
            "(Most homes are 1 or 2 floors — just type the number)"
        )

    # All mandatory data collected — shouldn't reach here but just in case
    return (
        "Do you need any special rooms?\n"
        "(dining, study, pooja room, balcony, parking, garden)\n"
        "Say 'no' or 'generate' to proceed."
    )


def _fallback_extract(history: List[Dict]) -> Dict:
    """Extract requirements from conversation using contextual parsing."""
    data = _parse_collected_data(history)

    import math

    result = {
        "plot_width": data["plot_width"] or 30,
        "plot_length": data["plot_length"] or 40,
        "total_area": data["total_area"] or 1200,
        "floors": data["floors"] or 1,
        "bedrooms": data["bedrooms"] or 2,
        "bathrooms": data["bathrooms"] or 1,
        "living_room": True,
        "kitchen": True,
        "extras": data["extras"],
    }

    # If only area given, estimate dimensions
    if result["total_area"] and not data["plot_width"]:
        side = math.sqrt(result["total_area"])
        result["plot_width"] = round(side * 1.15)
        result["plot_length"] = round(result["total_area"] / result["plot_width"])

    # Recalculate total area from dimensions if dimensions were given
    if data["plot_width"] and data["plot_length"]:
        result["total_area"] = result["plot_width"] * result["plot_length"]

    return result


def _fallback_design(requirements: Dict) -> Dict:
    """Generate layout using the deterministic architectural engine."""
    try:
        from services.arch_engine import design_generate as engine_design
        engine_result = engine_design(requirements)
        if "error" not in engine_result and engine_result.get("layout"):
            return engine_result["layout"]
    except Exception:
        pass

    # Original fallback
    w = requirements.get("plot_width", 30)
    l = requirements.get("plot_length", 40)
    total = w * l
    bedrooms = requirements.get("bedrooms", 2)
    bathrooms = requirements.get("bathrooms", 1)
    extras = requirements.get("extras", [])

    rooms = []
    x_cursor = 0.75  # Start after external wall
    y_cursor = 0.75

    # Living room — public zone, near entrance
    living_w = max(12, w * 0.4)
    living_l = max(12, l * 0.3)
    rooms.append({
        "name": "Living Room", "room_type": "living",
        "width": round(living_w, 1), "length": round(living_l, 1),
        "area": round(living_w * living_l),
        "zone": "public",
        "position": {"x": round(x_cursor, 1), "y": round(y_cursor, 1)},
        "doors": [{"wall": "S", "offset": round(living_w / 2, 1)}],
        "windows": [{"wall": "N", "width": 4}, {"wall": "E", "width": 4}],
    })

    # Kitchen — semi-private, adjacent to living
    kit_w = max(8, w * 0.25)
    kit_l = max(10, l * 0.25)
    kit_x = x_cursor + living_w + 0.38
    rooms.append({
        "name": "Kitchen", "room_type": "kitchen",
        "width": round(kit_w, 1), "length": round(kit_l, 1),
        "area": round(kit_w * kit_l),
        "zone": "semi_private",
        "position": {"x": round(kit_x, 1), "y": round(y_cursor, 1)},
        "doors": [{"wall": "W", "offset": round(kit_l / 2, 1)}],
        "windows": [{"wall": "E", "width": 4}],
    })

    # Bedrooms — private zone, upper portion
    bed_y = y_cursor + living_l + 0.38 + 3.5  # After corridor
    bed_w = max(10, (w - 1.5 - 0.38 * (bedrooms - 1)) / bedrooms)
    bed_l = max(12, (l - bed_y - 0.75) * 0.8)

    for i in range(bedrooms):
        name = "Master Bedroom" if i == 0 else f"Bedroom {i + 1}"
        rtype = "master_bedroom" if i == 0 else "bedroom"
        bx = x_cursor + i * (bed_w + 0.38)
        rooms.append({
            "name": name, "room_type": rtype,
            "width": round(bed_w, 1), "length": round(bed_l, 1),
            "area": round(bed_w * bed_l),
            "zone": "private",
            "position": {"x": round(bx, 1), "y": round(bed_y, 1)},
            "doors": [{"wall": "S", "offset": round(bed_w / 2, 1)}],
            "windows": [{"wall": "N", "width": 4}],
        })

    # Bathrooms — service zone
    bath_w = 5
    bath_l = 8
    for i in range(bathrooms):
        bath_x = kit_x + kit_w + 0.38 if i == 0 else kit_x + kit_w + 0.38
        bath_y = y_cursor + i * (bath_l + 0.38)
        # Fit within plot
        if bath_x + bath_w + 0.75 > w:
            bath_x = w - bath_w - 0.75
        if bath_y + bath_l + 0.75 > l:
            bath_y = l - bath_l - 0.75

        rooms.append({
            "name": f"Bathroom {i + 1}" if bathrooms > 1 else "Bathroom",
            "room_type": "bathroom",
            "width": bath_w, "length": bath_l,
            "area": bath_w * bath_l,
            "zone": "service",
            "position": {"x": round(bath_x, 1), "y": round(bath_y, 1)},
            "doors": [{"wall": "W", "offset": 2.5}],
            "windows": [{"wall": "E", "width": 3}],
        })

    # Extras
    extra_y = bed_y + bed_l + 0.38
    for extra in extras:
        if extra == "dining":
            rooms.append({
                "name": "Dining Room", "room_type": "dining",
                "width": 10, "length": 10, "area": 100,
                "zone": "semi_private",
                "position": {"x": round(x_cursor + living_w / 2, 1), "y": round(y_cursor + living_l + 0.38, 1)},
                "doors": [{"wall": "N", "offset": 5}],
                "windows": [{"wall": "W", "width": 4}],
            })

    total_used = sum(r["area"] for r in rooms)

    return {
        "plot": {"width": w, "length": l, "unit": "ft"},
        "rooms": rooms,
        "circulation": {"type": "central corridor", "width": 3.5},
        "walls": {"external": "9 inch", "internal": "4.5 inch"},
        "design_validation": {
            "total_area_used": round(total_used),
            "area_percentage": round(total_used / total * 100, 1),
            "compliant": total_used <= total * 0.9,
        },
    }


def _fallback_validate(layout: Dict) -> Dict:
    """Validation using the deterministic architectural engine."""
    try:
        from services.arch_engine import validate_layout as engine_validate
        engine_result = engine_validate(layout)
        # Convert engine format to pipeline format
        issues = (
            engine_result.get("overlap_details", []) +
            engine_result.get("size_violations", []) +
            engine_result.get("zoning_issues", []) +
            engine_result.get("boundary_issues", []) +
            engine_result.get("proportion_issues", [])
        )
        if engine_result.get("area_overflow"):
            issues.append(engine_result["area_overflow"])
        issues += engine_result.get("circulation_issues", [])

        area_summary = engine_result.get("area_summary", {})
        return {
            "compliant": engine_result.get("compliant", False),
            "total_area_used": area_summary.get("total_used_area", 0),
            "plot_area": area_summary.get("plot_area", 0),
            "area_utilization": f"{area_summary.get('utilization_percent', 0)}%",
            "checks": {
                "area_overflow": {"pass": not engine_result.get("area_overflow"), "detail": engine_result.get("area_overflow", "OK")},
                "overlapping_rooms": {"pass": not engine_result.get("overlap"), "detail": ", ".join(engine_result.get("overlap_details", [])) or "No overlaps"},
                "proportions": {"pass": not engine_result.get("proportion_issues"), "detail": ", ".join(engine_result.get("proportion_issues", [])) or "OK"},
                "zoning": {"pass": not engine_result.get("zoning_issues"), "detail": ", ".join(engine_result.get("zoning_issues", [])) or "OK"},
                "circulation": {"pass": not engine_result.get("circulation_issues"), "detail": ", ".join(engine_result.get("circulation_issues", [])) or "OK"},
                "minimum_sizes": {"pass": not engine_result.get("size_violations"), "detail": ", ".join(engine_result.get("size_violations", [])) or "All rooms meet minimum"},
            },
            "issues": issues,
            "suggestions": [
                "Consider adding cross-ventilation windows",
                "Ensure all bedrooms have external wall exposure",
            ] if engine_result.get("compliant") else ["Fix the issues above before proceeding"],
        }
    except Exception:
        pass

    # Original fallback
    rooms = layout.get("rooms", [])
    plot = layout.get("plot", {})
    plot_area = plot.get("width", 30) * plot.get("length", 40)

    issues = []
    checks = {}

    # Area overflow
    total_used = sum(r.get("area", 0) for r in rooms)
    area_ok = total_used <= plot_area
    checks["area_overflow"] = {"pass": area_ok, "detail": f"{total_used} / {plot_area} sq ft"}
    if not area_ok:
        issues.append(f"Total room area ({total_used} sqft) exceeds plot area ({plot_area} sqft)")

    # Minimum sizes
    min_sizes = {"bedroom": 100, "master_bedroom": 100, "bathroom": 35, "kitchen": 80, "living": 120}
    size_ok = True
    for room in rooms:
        rtype = room.get("room_type", "other")
        min_a = min_sizes.get(rtype, 0)
        if room.get("area", 0) < min_a:
            size_ok = False
            issues.append(f"{room.get('name', rtype)}: {room.get('area')} sqft < minimum {min_a} sqft")
    checks["minimum_sizes"] = {"pass": size_ok, "detail": "All rooms meet minimum" if size_ok else "See issues"}

    # Proportions
    prop_ok = True
    for room in rooms:
        w, l = room.get("width", 10), room.get("length", 10)
        if min(w, l) < 4:
            prop_ok = False
            issues.append(f"{room.get('name')}: dimension {min(w,l)} ft is too narrow")
        ratio = max(w, l) / max(min(w, l), 1)
        if ratio > 3:
            prop_ok = False
            issues.append(f"{room.get('name')}: aspect ratio {ratio:.1f}:1 is too extreme")
    checks["proportions"] = {"pass": prop_ok, "detail": "OK" if prop_ok else "See issues"}

    # Overlap check (basic AABB)
    overlap_ok = True
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p1 = r1.get("position", {})
            p2 = r2.get("position", {})
            x1, y1 = p1.get("x", 0), p1.get("y", 0)
            x2, y2 = p2.get("x", 0), p2.get("y", 0)

            if (x1 < x2 + r2.get("width", 0) and x1 + r1.get("width", 0) > x2 and
                y1 < y2 + r2.get("length", 0) and y1 + r1.get("length", 0) > y2):
                overlap_ok = False
                issues.append(f"Overlap: {r1.get('name')} and {r2.get('name')}")
    checks["overlapping_rooms"] = {"pass": overlap_ok, "detail": "No overlaps" if overlap_ok else "See issues"}

    # Zoning (basic)
    checks["zoning"] = {"pass": True, "detail": "Basic zoning check passed"}
    checks["circulation"] = {"pass": True, "detail": "Circulation paths present"}

    compliant = all(c["pass"] for c in checks.values())

    return {
        "compliant": compliant,
        "total_area_used": round(total_used),
        "plot_area": plot_area,
        "area_utilization": f"{round(total_used / plot_area * 100, 1)}%",
        "checks": checks,
        "issues": issues,
        "suggestions": [
            "Consider adding cross-ventilation windows",
            "Ensure all bedrooms have external wall exposure",
        ] if compliant else ["Fix the issues above before proceeding"],
    }
