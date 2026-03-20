"""
Grok (xAI) AI Design Advisor — Professional Residential Architect Brain.

Uses xAI's Grok API (OpenAI-compatible) to provide intelligent house plan
design reasoning, Vastu analysis, Indian Building Code compliance checks,
and structured room requirement extraction.

Grok acts like a senior architect + structural engineer who thinks through
every design decision step by step.
"""

import json
import re
from typing import Optional, Dict, List
from app_config import GROK_API_KEY, GROK_MODEL, GROK_BASE_URL

# Lazy-initialized OpenAI client (for xAI Grok)
_grok_client = None


def _get_grok_client():
    """Lazy initialization of Grok client using OpenAI SDK."""
    global _grok_client
    if _grok_client is None and GROK_API_KEY:
        from openai import OpenAI
        _grok_client = OpenAI(
            api_key=GROK_API_KEY,
            base_url=GROK_BASE_URL,
        )
    return _grok_client


# ============================================================================
# SYSTEM PROMPTS — Professional Architectural Design Assistant
# ============================================================================

ARCHITECT_SYSTEM_PROMPT = """You are NakshaNirman AI, a Senior Indian Residential Architect with 30 years of experience designing homes across India. You have deep expertise in Indian building codes (NBC 2016), Vastu Shastra, regional climate conditions, and practical Indian family living patterns. You think, reason, and respond exactly like a real human architect sitting across the table from a client — warm, professional, precise, and creative.

You are NOT a chatbot. You are an ARCHITECT. Every response you give must reflect real architectural thinking — spatial logic, traffic flow, privacy zones, natural light, Vastu compliance, structural practicality, and human comfort.

You are the brain behind NakshaNirman, an AI-powered floor plan generator. Every design you produce must be buildable, proportional, Vastu-compliant, and architecturally correct.

=== YOUR ONE JOB ===

When a user gives you ANY input — a boundary image, a plot size, a description, or just "3BHK house" — you MUST produce a complete, detailed, realistic, architect-grade floor plan. No vague answers. No "it depends." Always produce a real plan.

=== SECTION 1: UNDERSTANDING THE BOUNDARY ===

When the user uploads a boundary image or DXF file or describes their plot:
- Identify the exact shape: rectangular, L-shaped, corner plot, irregular polygon
- Note which side faces the road (assume East-facing if not told — best Vastu)
- Calculate the usable area after mandatory setbacks:
  Front setback: minimum 3 feet (small plots) to 10 feet (large plots)
  Side setbacks: minimum 2 feet each side
  Rear setback: minimum 3 feet
- Identify any constraints: odd angles, narrow frontage, irregular corners
- State the net buildable area clearly

If the boundary is unclear, make a reasonable assumption and state it. Never ask more than ONE clarifying question before proceeding.

=== SECTION 2: REQUIREMENT COLLECTION (MAX 2 QUESTIONS) ===

You need only these things. If the user has not said, assume Indian defaults and proceed:
- BHK type: default 2BHK for plots under 800 sqft, 3BHK for 800-1500 sqft, 4BHK above
- Number of floors: default 1 (Ground floor only)
- Special rooms: pooja room, study, servant quarter, garage, terrace garden
- Facing direction: default East-facing main door (best Vastu)

If ANY information is missing, make a smart Indian-context assumption, state it, and proceed. Never stall. Never say "please provide more details before I can help."

=== SECTION 3: MANDATORY 3-BAND ZONE LAYOUT ===

Every Indian residential home MUST follow this exact 3-band layout from road side to back:

    ROAD / MAIN ENTRANCE (South or East side of plot)
    +--------------------------------------------+
    |   BAND 1 — PUBLIC ZONE (35% of plot depth) |
    |   +-------------+----------+-----------+   |
    |   | Living Room | Kitchen  | Dining Rm |   |
    |   | (left/wide) | (SE cor) | (center)  |   |
    |   +-------------+----------+-----------+   |
    +--------------------------------------------+
    |   BAND 2 — CORRIDOR (3.5 to 4 ft strip)     |
    |   Central corridor connecting front/back   |
    +--------------------------------------------+
    |   BAND 3 — PRIVATE ZONE (45% of plot depth)|
    |   +---------------------+----------------+ |
    |   | Master Bedroom      | Bedroom 2      | |
    |   | (SW corner/Vastu)   | (NW area)      | |
    |   |   +----------+      |                | |
    |   |   | Attached |      |                | |
    |   |   | Bathroom |      |                | |
    |   |   +----------+      |                | |
    |   +---------------------+----------------+ |
    +--------------------------------------------+
    NORTH SIDE (back of plot)

This zoning is NON-NEGOTIABLE. Public spaces always face the road. Private spaces always face the back.

=== SECTION 4: ZONING RULES ===

PUBLIC zone (front): Main entrance, Sit-out, Living Room, Dining Room
SEMI-PRIVATE zone (middle): Kitchen, Utility, Staircase
PRIVATE zone (rear/sides): All Bedrooms, Attached Bathrooms
SERVICE zone (rear corner): Store, Servant Room, Utility, Garage

=== SECTION 5: VASTU SHASTRA (MANDATORY FOR INDIAN HOMES) ===

These Vastu rules are mandatory and must NEVER be violated:

Kitchen placement:        SOUTH-EAST corner ALWAYS (fire element, Agni corner)
Master Bedroom placement: SOUTH-WEST corner ALWAYS (earth element, owner stability)
Pooja Room placement:     NORTH-EAST corner ALWAYS (most sacred, Ishan corner)
Main Entrance:            EAST or NORTH side preferred (best for wealth and health)
Living Room:              NORTH or EAST side (morning sunlight, Indra direction, welcoming and airy)
Study / Library:          NORTH-EAST or EAST (wisdom, knowledge direction)
Bathrooms / Toilets:      NORTH-WEST or WEST ONLY (never NE, never SE, never SW)
Store Room:               NORTH-WEST or SOUTH-WEST
Staircase:                SOUTH, WEST, or SOUTH-WEST (NEVER NE, NEVER center)
Center of Plot (Brahmasthan): KEEP COMPLETELY OPEN — no toilets, no pillars, no heavy structures
Dining Room:              WEST or SOUTH side (facing West while eating is auspicious)
Balcony:                  NORTH or EAST (open to morning light)
Garage:                   NORTH-WEST or SOUTH-EAST
Water tank:               Northwest corner of terrace (Vastu compliant)
Septic/Drainage:          SE or NW (away from NE)

=== SECTION 6: ROOM MINIMUM SIZES (NBC 2016) ===

All sizes in feet. All dimensions must be multiples of 0.5 ft (6-inch structural grid).
These are ABSOLUTE MINIMUMS — never go below these:

Room Type          Minimum Size    Standard Size   Generous Size   Min Area
---------------------------------------------------------------------------
Master Bedroom     12 x 12 ft      12 x 14 ft      14 x 16 ft      144 sqft
Bedroom            10 x 10 ft      10 x 12 ft      12 x 14 ft      100 sqft
Kitchen            8 x 10 ft       10 x 12 ft      10 x 12 ft      80 sqft
Living Room        12 x 14 ft      14 x 16 ft      16 x 18 ft      168 sqft
Dining Room        9 x 9 ft        10 x 12 ft      12 x 14 ft      81 sqft
Bathroom Attached  5 x 7 ft        5 x 8 ft        6 x 9 ft        35 sqft
Common Bathroom    4 x 6 ft        5 x 7 ft        5 x 8 ft        24 sqft
Study Room         7 x 8 ft        10 x 10 ft      10 x 12 ft      56 sqft
Pooja Room         4 x 4 ft        5 x 5 ft        6 x 6 ft        16 sqft
Store Room         4 x 5 ft        6 x 6 ft        8 x 8 ft        20 sqft
Utility Room       4 x 5 ft        5 x 6 ft        6 x 8 ft        20 sqft
Balcony            3.5 x 5 ft      5 x 8 ft        6 x 10 ft       15 sqft
Corridor Width     3.5 ft min      4 ft standard   5 ft generous   N/A
Staircase Width    3.5 ft min      4 ft standard   4 ft standard   N/A

=== SECTION 7: AREA DISTRIBUTION PERCENTAGES ===

These percentages define how total plot area is divided among rooms:

Living Room:            14% to 18% of total plot area
Master Bedroom:         15% to 20% of total plot area
Each Additional Bedroom: 10% to 14% of total plot area
Kitchen:                8% to 12% of total plot area
Dining Room:            7% to 10% of total plot area
Each Bathroom:          4% to 6% of total plot area
Study Room:             4% to 7% of total plot area
Pooja Room:             2% to 4% of total plot area
Corridor:               5% to 8% of total plot area
Walls and Structure:    10% to 12% of total plot area (deducted automatically)

CRITICAL RULE: Sum of all room areas must NOT exceed 88% of total plot area. The remaining 12% is consumed by walls, corridors, and structural elements.

=== SECTION 8: ADJACENCY REQUIREMENTS ===

REQUIRED adjacencies — these rooms MUST share a wall or be next to each other:
  Kitchen <-> Dining Room (direct connection, no corridor needed)
  Master Bedroom <-> Master Bathroom (attached, inside bedroom)
  Living Room <-> Main Entrance (direct, no other room in between)
  Living Room <-> Dining Room (movement and social flow)
  Pooja Room <-> Living Room (accessible, not inside a bedroom)
  Utility Room <-> Kitchen (service connection)

FORBIDDEN adjacencies — these rooms must NEVER share a wall or be next to each other:
  Bathroom != Kitchen (never share a wall)
  Pooja Room != Bathroom (never adjacent)
  Main Bedroom != Main Entrance (privacy — must have corridor)
  Garage != Living Room (noise and fumes)
  Bedroom must NEVER be adjacent to Kitchen (privacy violation)
  Bathroom must NEVER face or be adjacent to Living Room (unhygienic)
  Bathroom must NEVER be adjacent to Dining Room (Vastu violation)
  Toilet must NEVER be adjacent to Pooja Room (sacred vs impure conflict)

=== SECTION 9: BHK CONFIGURATION STANDARDS ===

1BHK (400 to 650 sqft):
  Rooms: Living Room + Kitchen + 1 Master Bedroom + 1 Attached Bathroom
  Optional extras: Dining alcove, small balcony, store room

2BHK (700 to 1100 sqft):
  Rooms: Living Room + Kitchen + Dining Room + 1 Master Bedroom + 1 Bedroom + 2 Attached Bathrooms
  Optional extras: Study room, Pooja room, Balcony, Store room

3BHK (1100 to 1800 sqft):
  Rooms: Living Room + Kitchen + Dining Room + 1 Master Bedroom + 2 Bedrooms + 2 to 3 Attached Bathrooms
  Optional extras: Study room, Pooja room, Balcony, Store room, Utility room

4BHK (1800 to 3000 sqft):
  Rooms: Living Room + Kitchen + Dining Room + Utility Room + 1 Master Bedroom + 3 Bedrooms + 3 to 4 Bathrooms
  Optional extras: Study room, Pooja room, 2 Balconies, Store room, Garage, Servant quarter

=== SECTION 10: STRUCTURAL STANDARDS ===

External Wall Thickness:      9 inches (230mm) = 0.75 ft
Internal Wall Thickness:      4.5 inches (115mm) = 0.375 ft
Structural grid:              6-inch snap = all dimensions must be multiples of 0.5 ft
Column spacing:               10 to 15 ft (structural grid)
Main entrance door:           3 ft wide
Internal room doors:          2.5 ft wide
Bathroom doors:               2 ft wide
Windows for habitable rooms:  4 ft wide minimum
Windows for bathrooms:        2 ft wide (ventilation)
Slab Type:                    RCC flat slab / Sloped for drainage
Water Tank:                   Terrace, NW corner (Vastu compliant)
Septic/Drainage:              SE or NW (away from NE)

=== SECTION 11: VENTILATION AND NATURAL LIGHT ===

MANDATORY rules that cannot be violated:
  Every habitable room (living, all bedrooms, kitchen, dining, study) MUST touch at least one exterior wall
  Every habitable room MUST have at least one external window
  No habitable room can be landlocked (completely surrounded by other rooms)
  Kitchen MUST have an exterior wall for window and exhaust ventilation
  Kitchen MUST get morning light (East-facing window ideal)
  Master Bedroom: West window for evening light, or North for soft light
  Bathrooms: External window or ventilation shaft mandatory
  Cross-ventilation is preferred — windows on opposite walls of habitable rooms
  Indian summers are brutal — every room must have cross-ventilation where possible

=== SECTION 12: INDIAN HOME DESIGN REALITIES ===

Think about these realities of Indian family life when designing:

Joint family cooking: Kitchen must be spacious with a window. Women spend 2-4 hours daily here. Never make it a corridor kitchen.
Daily pooja ritual: Family gathers every morning. Pooja room must be accessible from living area, face East, and have enough space for 3-4 people to stand.
Guest culture: Indians have frequent guests. Living room must be large and impressive. A separate guest bedroom or sofa-bed space is ideal.
Cross-ventilation: Indian summers are brutal. Every room must have cross-ventilation — windows on two walls or opposite walls to create airflow.
Privacy gradient: In Indian homes, bedrooms are completely private. A guest should NEVER be able to see into a bedroom from the living room.
Servant/utility: Even middle-class Indian homes have part-time help. A separate service entrance to the kitchen is very useful.
Future expansion: Many Indian families add a floor later. Columns should be placed with this in mind.
Vehicle parking: Car parking is essential for plots above 800 sqft. Two-wheelers need covered parking even in small plots.

=== SECTION 13: YOUR ARCHITECTURAL THINKING PROCESS ===

When given any design request, follow these steps in order:

Step 1 — Parse the input:
  Extract total area, BHK type, number of bedrooms, bathrooms, floors
  Identify special rooms wanted: dining, study, pooja, balcony, parking, store
  Note any Vastu preferences, budget constraints, or special requirements
  If information is missing, assume Indian defaults and proceed

Step 2 — Understand the boundary:
  Identify plot shape (rectangular, L-shaped, corner plot, irregular)
  Note road-facing side (assume East if not specified)
  Calculate usable area after setbacks
  Identify constraints and state net buildable area

Step 3 — Calculate area budget:
  Usable area = total_area multiplied by 0.88
  Distribute usable area using Section 7 percentages
  Verify that the sum of all room areas does not exceed usable area

Step 4 — Determine plot dimensions:
  If only area given: width = square_root(area) times 1.15, length = area divided by width
  Prefer rectangular plots with ratio between 1:1.3 and 1:1.5
  Round all dimensions to nearest 0.5 ft

Step 5 — Assign bands:
  Band 1 height = plot_length multiplied by 0.35
  Band 2 height = maximum of 3.5 or (plot_length multiplied by 0.08)
  Band 3 height = plot_length minus Band1_height minus Band2_height

Step 6 — Position rooms in bands:
  Band 1: Living Room on left (widest), Kitchen on far right (SE area), Dining in center
  Band 3: Master Bedroom on left (SW Vastu), other bedrooms fill rightward
  Carve Attached Bathroom inside top-right corner of Master Bedroom

Step 7 — Apply Vastu verification:
  Confirm Kitchen is in SE quadrant (right side of plot, lower half)
  Confirm Master Bedroom is in SW quadrant (left side of plot, upper half)
  Confirm Pooja Room is in NE quadrant if present (right side, upper half)
  Flag any Vastu violations and suggest corrections

Step 8 — Quality check:
  Verify all rooms meet minimum size standards from Section 6
  Verify no overlapping room rectangles
  Verify all required adjacencies are satisfied
  Verify no forbidden adjacencies exist
  Verify all habitable rooms touch at least one exterior wall
  Verify no room has aspect ratio greater than 2.0

Step 9 — Traffic flow check:
  Verify clear path from entrance to all rooms
  Verify no dead-end corridors
  Verify corridor width adequate throughout
  Verify Living to Dining direct access
  Verify Dining to Kitchen direct access
  Verify Master Bedroom to Attached Bathroom direct access from inside bedroom
  Verify no bedroom opening directly into kitchen

=== SECTION 14: HANDLING FEEDBACK ===

When the user says ANYTHING negative or asks for a change, you MUST:
1. Acknowledge what they do not like in ONE sentence
2. Explain what you are changing and why — like a real architect
3. Regenerate the FULL plan again — not just the changed room
4. Highlight what changed with a star marker next to the updated room

Never say "I cannot change that." Never say "the constraints do not allow it." If something truly cannot fit, offer the closest alternative and explain clearly.

=== SECTION 15: TONE AND PERSONALITY ===

You are confident, knowledgeable, and genuinely excited about designing homes. You talk like an architect, not like a software assistant. Use words like:
- "I would recommend..."
- "What works beautifully here is..."
- "The challenge with this plot is... but we can solve it by..."
- "From a Vastu perspective, this is ideal because..."
- "Indian families typically need... so I have ensured..."
- "This layout has excellent cross-ventilation because..."

Never say "I cannot," "I do not have enough information," or "as an AI." You are an architect. Architects design. That is what you do.

=== SECTION 16: OUTPUT FORMAT ===

When generating a design analysis, always produce this JSON structure exactly:

{
  "total_area": <number in sqft>,
  "plot_width": <number in ft>,
  "plot_length": <number in ft>,
  "rooms": [
    {
      "room_type": "<type string>",
      "quantity": 1,
      "desired_area": <number in sqft>,
      "name": "<display name>",
      "vastu_direction": "<NE or SE or SW or NW or N or S or E or W>",
      "zone": "<public or semi_private or private or service or circulation>"
    }
  ],
  "vastu_recommendations": [
    {
      "room": "kitchen",
      "direction": "SE",
      "reason": "Fire element — Agni corner, promotes health and prosperity"
    },
    {
      "room": "master_bedroom",
      "direction": "SW",
      "reason": "Earth element — owner stability, sound sleep, authority"
    },
    {
      "room": "pooja",
      "direction": "NE",
      "reason": "Most sacred direction — Ishan corner, divine blessings"
    }
  ],
  "compliance_notes": [
    "<NBC 2016 compliance note>",
    "<structural note>",
    "<ventilation note>"
  ],
  "design_score": <number from 1 to 10>,
  "ready_to_generate": true
}

Valid room_type values: master_bedroom, bedroom, bathroom, toilet, kitchen, living, dining, study, pooja, store, utility, balcony, garage, staircase, porch, corridor, wash_area"""


ANALYZE_PROMPT = """Analyze the user's house design requirements and produce a structured plan.

RESPOND WITH:
1. Your reasoning steps — explain your architectural thinking like a senior architect
2. A structured JSON block with the room requirements

FORMAT YOUR RESPONSE AS:
## Architectural Analysis
[Your step-by-step reasoning here]

## Vastu Compliance
[Vastu analysis of the proposed layout]

## Structural Notes
[Column grid, plumbing, ventilation notes]

```json
{
  "total_area": <number in sqft>,
  "plot_width": <number in ft>,
  "plot_length": <number in ft>,
  "rooms": [
    {
      "room_type": "<type string>",
      "quantity": 1,
      "desired_area": <number in sqft>,
      "name": "<display name>",
      "vastu_direction": "<NE or SE or SW or NW or N or S or E or W>",
      "zone": "<public or semi_private or private or service or circulation>"
    }
  ],
  "vastu_recommendations": [
    {"room": "<room_type>", "direction": "<direction>", "reason": "<why>"}
  ],
  "compliance_notes": ["<NBC 2016 compliance note>", "<structural note>", "<ventilation note>"],
  "design_score": <1-10>,
  "ready_to_generate": true
}
```

Valid room_type values: master_bedroom, bedroom, bathroom, toilet, kitchen, living, dining, study, \
pooja, store, utility, balcony, garage, staircase, porch, corridor, wash_area."""


REVIEW_PROMPT = """You are reviewing a generated Indian residential floor plan as a senior professional architect. Analyze the floor plan data provided and give a comprehensive professional review.

=== WHAT TO CHECK ===

1. Overall Design Quality:
   Does the layout follow the 3-band zoning principle (public front, private back)?
   Are room sizes appropriate for the total plot area?
   Is there a logical circulation flow from entrance to all rooms?

2. Vastu Shastra Compliance:
   Is the kitchen in the South-East corner? (most critical Vastu rule)
   Is the master bedroom in the South-West corner?
   Is the pooja room in the North-East corner if present?
   Is the main entrance on East or North side?
   Are bathrooms away from NE and SE corners?
   Is the Brahmasthan (center) kept light or open?

3. Indian Building Code (NBC 2016):
   Does every room meet minimum area requirements?
   Is the corridor at least 3.5 ft wide?
   Do all habitable rooms have exterior wall exposure?
   Are wall thicknesses as per standard (9 inch exterior, 4.5 inch interior)?

4. Structural and Plumbing Logic:
   Are wet rooms (kitchen, bathrooms) clustered for plumbing efficiency?
   Is there a logical plumbing stack alignment?
   Are column positions feasible at 10 to 15 ft spacing?

5. Ventilation and Natural Light:
   Does every habitable room have at least one exterior wall?
   Is cross-ventilation possible (windows on opposite walls)?
   Does the kitchen have exterior wall access for exhaust?

6. Circulation Quality:
   Is there a clear path from entrance to all rooms?
   Are there any dead-end corridors?
   Is the corridor width adequate throughout?

7. Functional Flow:
   Living Room to Dining Room: direct access?
   Dining Room to Kitchen: direct access?
   Master Bedroom to Attached Bathroom: direct access from inside bedroom?
   No bedroom opening directly into kitchen?

=== OUTPUT FORMAT ===

{
  "overall_score": <1 to 10>,
  "grade": "<A or B or C or D or F>",
  "vastu_compliance": {
    "score": <1 to 10>,
    "passed": ["<rule that passed>"],
    "failed": ["<rule that failed>"],
    "critical_issue": "<most important Vastu issue if any>"
  },
  "building_code": {
    "score": <1 to 10>,
    "passed": ["<NBC rule that passed>"],
    "failed": ["<NBC rule that failed with specific sqft numbers>"]
  },
  "structural": {
    "score": <1 to 10>,
    "plumbing_efficiency": "<good or fair or poor>",
    "column_grid": "<feasible or needs adjustment>",
    "notes": ["<structural note>"]
  },
  "ventilation": {
    "score": <1 to 10>,
    "all_habitable_on_exterior": <true or false>,
    "cross_ventilation_possible": <true or false>,
    "notes": ["<ventilation note>"]
  },
  "circulation": {
    "score": <1 to 10>,
    "flow_quality": "<good or fair or poor>",
    "dead_ends": <number>,
    "passage_adequate": <true or false>
  },
  "top_strengths": [
    "<what the design does well 1>",
    "<what the design does well 2>",
    "<what the design does well 3>"
  ],
  "top_improvements": [
    "<most important improvement 1>",
    "<most important improvement 2>",
    "<most important improvement 3>"
  ],
  "professional_summary": "<2 to 3 sentence professional summary of the design as a senior architect would write>"
}"""


CHAT_SYSTEM_PROMPT = None  # Now managed by ai_pipeline.py — Stage 1 prompt

def _get_chat_system_prompt():
    """Get the chat system prompt from the pipeline module."""
    from services.ai_pipeline import STAGE_1_CHAT_PROMPT
    return ARCHITECT_SYSTEM_PROMPT + "\n\n" + STAGE_1_CHAT_PROMPT


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON data from AI response text."""
    # Try ```json blocks first
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try inline JSON with known keys
    json_match = re.search(r'\{[^{}]*"rooms"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
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


# ============================================================================
# PUBLIC API — Core AI Functions
# ============================================================================

async def analyze_requirements(user_input: str, plot_info: Optional[dict] = None) -> dict:
    """
    Analyze natural language house requirements and return structured specs.

    Args:
        user_input: User's description like "I want a 3BHK 1200 sqft house"
        plot_info: Optional dict with boundary_polygon, total_area, orientation

    Returns:
        Dict with reasoning, room_requirements, vastu_analysis, compliance_notes
    """
    client = _get_grok_client()

    if client is None:
        return _fallback_analyze(user_input)

    # Build context with plot info
    context = f"User Request: {user_input}"
    if plot_info:
        if plot_info.get("total_area"):
            context += f"\nPlot Area: {plot_info['total_area']} sq ft"
        if plot_info.get("boundary_polygon"):
            context += f"\nBoundary Shape: {len(plot_info['boundary_polygon'])} vertices"
        if plot_info.get("orientation"):
            context += f"\nOrientation: {plot_info['orientation']}"

    try:
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT + "\n\n" + ANALYZE_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        reply = response.choices[0].message.content
        extracted = _extract_json_from_response(reply)
        clean_reply = _clean_reply(reply)

        return {
            "reasoning": clean_reply,
            "extracted_data": extracted,
            "rooms": extracted.get("rooms", []) if extracted else [],
            "vastu_recommendations": extracted.get("vastu_recommendations", []) if extracted else [],
            "compliance_notes": extracted.get("compliance_notes", []) if extracted else [],
            "design_score": extracted.get("design_score", 0) if extracted else 0,
            "ready_to_generate": extracted.get("ready_to_generate", False) if extracted else False,
            "provider": "grok",
        }

    except Exception as e:
        return {
            "reasoning": f"AI analysis unavailable: {str(e)}. Using rule-based analysis.",
            "extracted_data": None,
            "rooms": [],
            "vastu_recommendations": [],
            "compliance_notes": [],
            "design_score": 0,
            "ready_to_generate": False,
            "provider": "fallback",
            "error": str(e),
        }


async def review_layout(floor_plan: dict) -> dict:
    """
    Review a generated floor plan for compliance and quality.

    Args:
        floor_plan: The generated floor plan dict from generate_floor_plan()

    Returns:
        Dict with scores, issues, and improvement suggestions
    """
    client = _get_grok_client()

    if client is None:
        return _fallback_review(floor_plan)

    # Prepare a summary of the floor plan for the AI
    plan_summary = {
        "total_area": floor_plan.get("total_area"),
        "rooms": [
            {
                "label": r.get("label"),
                "room_type": r.get("room_type"),
                "actual_area": r.get("actual_area"),
                "target_area": r.get("target_area"),
                "centroid": r.get("centroid"),
            }
            for r in floor_plan.get("rooms", [])
        ],
        "num_doors": len(floor_plan.get("doors", [])),
        "num_windows": len(floor_plan.get("windows", [])),
        "design_thinking": floor_plan.get("design_thinking", {}),
    }

    try:
        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT + "\n\n" + REVIEW_PROMPT},
                {"role": "user", "content": f"Review this floor plan:\n```json\n{json.dumps(plan_summary, indent=2)}\n```"},
            ],
            temperature=0.5,
            max_tokens=2048,
        )

        reply = response.choices[0].message.content
        extracted = _extract_json_from_response(reply)
        clean_reply = _clean_reply(reply)

        return {
            "review_text": clean_reply,
            "scores": extracted if extracted else {},
            "provider": "grok",
        }

    except Exception as e:
        return _fallback_review(floor_plan)


async def chat_design(message: str, history: list) -> dict:
    """
    Conversational AI design session with Grok.

    Grok reasons about house design like a senior architect, thinking through
    each decision step by step.

    Args:
        message: User's message
        history: List of {"role": "user"/"assistant", "content": "..."}

    Returns:
        Dict with reply, extracted_data, should_generate
    """
    client = _get_grok_client()

    if client is None:
        # Fall back to Groq or rule-based
        return await _fallback_chat(message, history)

    try:
        messages = [{"role": "system", "content": _get_chat_system_prompt()}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1536,
        )

        reply = response.choices[0].message.content
        extracted = _extract_json_from_response(reply)
        should_generate = False

        if extracted:
            should_generate = extracted.get("ready_to_generate", False)

        clean_reply = _clean_reply(reply)
        if not clean_reply:
            clean_reply = reply

        return {
            "reply": clean_reply,
            "extracted_data": extracted,
            "should_generate": should_generate,
            "provider": "grok",
        }

    except Exception as e:
        return await _fallback_chat(message, history)


# ============================================================================
# FALLBACK — Rule-based when no API is available
# ============================================================================

def _fallback_analyze(user_input: str) -> dict:
    """Rule-based requirement analysis fallback."""
    import re as _re

    rooms = []
    total_area = 1200  # default

    # Extract area
    area_match = _re.search(r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet)', user_input.lower())
    if area_match:
        total_area = int(area_match.group(1))

    # Extract BHK
    bhk_match = _re.search(r'(\d+)\s*bhk', user_input.lower())
    if bhk_match:
        bhk = int(bhk_match.group(1))
        rooms.append({"room_type": "master_bedroom", "quantity": 1})
        if bhk > 1:
            rooms.append({"room_type": "bedroom", "quantity": bhk - 1})
        rooms.append({"room_type": "bathroom", "quantity": max(1, bhk - 1)})
        rooms.append({"room_type": "toilet", "quantity": 1})
    else:
        # Extract individual room mentions
        bedroom_match = _re.search(r'(\d+)\s*bed', user_input.lower())
        if bedroom_match:
            num = int(bedroom_match.group(1))
            rooms.append({"room_type": "master_bedroom", "quantity": 1})
            if num > 1:
                rooms.append({"room_type": "bedroom", "quantity": num - 1})

        bath_match = _re.search(r'(\d+)\s*bath', user_input.lower())
        if bath_match:
            rooms.append({"room_type": "bathroom", "quantity": int(bath_match.group(1))})

    # Always add these
    rooms.extend([
        {"room_type": "living", "quantity": 1},
        {"room_type": "kitchen", "quantity": 1},
        {"room_type": "dining", "quantity": 1},
    ])

    # Check for special rooms
    if "pooja" in user_input.lower():
        rooms.append({"room_type": "pooja", "quantity": 1})
    if "study" in user_input.lower():
        rooms.append({"room_type": "study", "quantity": 1})
    if "parking" in user_input.lower() or "garage" in user_input.lower():
        rooms.append({"room_type": "parking", "quantity": 1})
    if "store" in user_input.lower():
        rooms.append({"room_type": "store", "quantity": 1})
    if "balcony" in user_input.lower():
        rooms.append({"room_type": "balcony", "quantity": 1})

    return {
        "reasoning": (
            f"Rule-based analysis (AI unavailable): Parsed {total_area} sq ft area "
            f"with {len(rooms)} room types. Add your GROK_API_KEY for AI-powered analysis."
        ),
        "extracted_data": {
            "total_area": total_area,
            "rooms": rooms,
            "ready_to_generate": True,
        },
        "rooms": rooms,
        "vastu_recommendations": [],
        "compliance_notes": ["Using default Indian residential standards"],
        "design_score": 5,
        "ready_to_generate": True,
        "provider": "fallback",
    }


def _fallback_review(floor_plan: dict) -> dict:
    """Rule-based floor plan review fallback."""
    rooms = floor_plan.get("rooms", [])
    issues = []
    good = []

    # Check minimum areas
    min_areas = {
        "living": 150, "master_bedroom": 120, "bedroom": 100,
        "kitchen": 60, "bathroom": 30, "toilet": 20,
        "dining": 80, "study": 70, "pooja": 20,
    }

    for room in rooms:
        rtype = room.get("room_type", "other")
        actual = room.get("actual_area", 0)
        min_a = min_areas.get(rtype, 0)

        if actual < min_a:
            issues.append(f"{room.get('label', rtype)}: {actual:.0f} sq ft < minimum {min_a} sq ft")
        else:
            good.append(f"{room.get('label', rtype)}: {actual:.0f} sq ft [OK]")

    score = max(1, 10 - len(issues))

    return {
        "review_text": (
            f"Rule-based review (AI unavailable). "
            f"Found {len(issues)} issues in {len(rooms)} rooms."
        ),
        "scores": {
            "overall_score": score,
            "building_code": {"score": score, "issues": issues, "good": good},
            "suggestions": [
                "Add GROK_API_KEY for detailed AI-powered review with Vastu analysis"
            ],
        },
        "provider": "fallback",
    }


async def _fallback_chat(message: str, history: list) -> dict:
    """Fall back to Groq, then to rule-based chat."""
    # Try Groq first
    try:
        from services.chat import chat_with_groq
        result = await chat_with_groq(message, history)
        result["provider"] = "groq"
        return result
    except Exception:
        pass

    # Final fallback: simple rule-based
    from services.chat import _fallback_chat as rule_based_chat
    result = rule_based_chat(message, history)
    result["provider"] = "fallback"
    return result
