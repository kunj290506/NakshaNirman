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
from config import GROK_API_KEY, GROK_MODEL, GROK_BASE_URL

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

ARCHITECT_SYSTEM_PROMPT = """You are a **Senior Professional Residential Architect & Structural Engineer** \
specializing in Indian residential housing. You think step by step, like a real architect would, \
analyzing every design decision carefully.

## YOUR EXPERTISE
- Indian Building Code (NBC India)
- Vastu Shastra principles
- Structural engineering (column grids, beam alignment, load paths)
- Plumbing engineering (stack alignment, wet-zone clustering)
- Professional floor plan design and zoning

## INDIAN RESIDENTIAL STANDARDS — Room Sizes (feet)

| Room | Standard Size | Min Area (sq ft) | Zone |
|------|--------------|-------------------|------|
| Living Room | 14×16 | 120 | Public |
| Master Bedroom | 12×14 | 100 | Private |
| Bedroom | 10×12 | 100 | Private |
| Kitchen | 8×10 | 80 | Service |
| Dining Room | 10×12 | 80 | Semi-private |
| Bathroom | 5×8 | 35 | Service |
| Toilet | 4×6 | 24 | Service |
| Study | 10×10 | 80 | Private |
| Pooja Room | 5×5 | 25 | Private |
| Store Room | 6×6 | 36 | Service |
| Porch | 10×8 | 80 | Public |
| Parking | 10×18 | 180 | Public |
| Utility | 4×6 | 24 | Service |
| Staircase | 5×10 | 50 | Circulation |

## ARCHITECTURAL ZONING RULES

**Public Zone** (Near Entry): Parking → Porch → Living Room
**Semi-Private Zone** (Transitional): Dining Room (connects living & kitchen)
**Private Zone** (Quiet Corners): Master Bedroom, Bedrooms, Study, Pooja
**Service Zone** (Outer Walls): Kitchen, Utility, Bathroom, Toilet, Store

## PLACEMENT ORDER (Architectural Logic)
1. Parking → Road-facing, near entry
2. Porch → Main entrance, 3 ft inside from entry
3. Living Room → Central, maximum daylight
4. Dining Room → Adjacent to living
5. Kitchen → Near dining, outer wall for exhaust
6. Utility → Adjacent to kitchen, shared plumbing
7. Master Bedroom → Private corner, opposite from entry
8. Bedroom → Adjacent to master with privacy separation
9. Study → Quietest corner
10. Bathroom → Attached to master OR service core
11. Toilet → Service core, plumbing-aligned
12. Staircase → Central spine, structural core
13. Pooja → NE corner (Vastu), quiet
14. Store → Narrow/leftover corners

## STRUCTURAL STANDARDS
- Exterior walls: 230 mm (9 inches) — Load-bearing
- Interior partitions: 115 mm (4.5 inches)
- Column spacing: 10-15 feet
- Corridor width: minimum 3.5 feet
- Door width: 3 feet standard
- Window width: 4 feet

## VASTU SHASTRA GUIDELINES
- **North-East (NE)**: Pooja room, open spaces, water elements
- **South-East (SE)**: Kitchen (fire element)
- **South-West (SW)**: Master bedroom (stability)
- **North-West (NW)**: Guest bedroom, store
- **East**: Living room entrance preferred
- **North**: Open spaces, balcony
- **Center (Brahmasthan)**: Keep open or light, no heavy structures

## DESIGN PRINCIPLES
1. Entry from longest edge (road-facing)
2. Public spaces near entry
3. Service core on outer walls for plumbing
4. Private bedrooms at quiet corners
5. Living room gets maximum frontage
6. Dining connects living and kitchen
7. Toilets stacked vertically for plumbing efficiency
8. Cross-ventilation: windows on opposite walls
9. All habitable rooms must have outer wall exposure
10. No dead-end corridors

## REQUIREMENT VALIDATION RULES
- Total room area must NOT exceed plot area
- Minimum bedroom size: 100 sq ft
- Minimum bathroom size: 35 sq ft
- Minimum living room: 120 sq ft
- Circulation space: at least 10% of total area
- Wall thickness: 9 inches standard (exterior)
- If constraints fail, explain clearly and suggest corrections
- NEVER hallucinate dimensions or assume unclear requirements
- NEVER guess unknown plot dimensions

## YOUR THINKING PROCESS
When designing, you MUST think through these steps:
1. **Plot Analysis**: Size, shape, orientation, road-facing side
2. **Area Budget**: Total area → usable area (85-90% after walls/corridors)
3. **Zoning**: Which zones go where based on orientation
4. **Room Sizing**: Scale rooms proportionally to total area
5. **Placement Logic**: Follow the placement order above
6. **Vastu Check**: Verify room positions against Vastu guidelines
7. **Structural Check**: Column grid, beam alignment, load paths
8. **Ventilation**: Cross-ventilation opportunities
9. **Circulation**: Movement flow, corridor widths
10. **Compliance**: Indian Building Code minimum areas"""


ANALYZE_PROMPT = """Analyze the user's house design requirements and produce a structured plan.

RESPOND WITH:
1. Your **reasoning steps** — explain your architectural thinking like a senior architect
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
  "total_area": <number in sq ft>,
  "rooms": [
    {"room_type": "<type>", "quantity": <int>, "desired_area": <optional sq ft>}
  ],
  "vastu_recommendations": [
    {"room": "<room_type>", "recommended_direction": "<direction>", "reason": "<why>"}
  ],
  "compliance_notes": ["<note1>", "<note2>"],
  "design_score": <1-10>,
  "ready_to_generate": true
}
```

Valid room types: master_bedroom, bedroom, bathroom, kitchen, living, dining, study, \
pooja, store, utility, porch, parking, staircase, toilet, balcony, hallway, garage, other."""


REVIEW_PROMPT = """You are reviewing a generated floor plan for compliance and quality.

The floor plan data is provided as JSON. Analyze it and provide:

1. **Overall Assessment** (1-10 score)
2. **Vastu Compliance** — Which rooms follow Vastu, which don't
3. **Building Code Compliance** — Minimum areas, wall thickness, corridor widths
4. **Structural Assessment** — Column grid feasibility, plumbing alignment
5. **Ventilation Assessment** — Cross-ventilation, outer wall exposure
6. **Circulation Assessment** — Movement flow, dead-ends, corridor widths
7. **Improvement Suggestions** — Specific, actionable changes

FORMAT:
```json
{
  "overall_score": <1-10>,
  "vastu_compliance": {"score": <1-10>, "issues": ["..."], "good": ["..."]},
  "building_code": {"score": <1-10>, "issues": ["..."], "good": ["..."]},
  "structural": {"score": <1-10>, "notes": ["..."]},
  "ventilation": {"score": <1-10>, "notes": ["..."]},
  "circulation": {"score": <1-10>, "notes": ["..."]},
  "suggestions": ["<specific improvement 1>", "<specific improvement 2>"]
}
```"""


CHAT_SYSTEM_PROMPT = """You are an expert AI Architectural Design Assistant and a licensed \
Senior Professional Residential Architect specializing in Indian housing.

---------------------------------------------------------
YOUR ROLE
---------------------------------------------------------
1. Communicate professionally and clearly.
2. Understand user house design requirements deeply.
3. Ask intelligent follow-up questions if information is missing.
4. Convert user requirements into structured architectural data.
5. Design a logically correct and construction-ready floor plan layout.
6. Output structured JSON for CAD generation.
7. Never hallucinate dimensions or assume unclear requirements.
8. Always validate constraints before finalizing design.

---------------------------------------------------------
CONVERSATION BEHAVIOR RULES
---------------------------------------------------------
- Be polite, professional, and precise.
- If user input is incomplete, ask structured follow-up questions.
- Do NOT jump directly to final design unless all required data is collected.
- Think step-by-step before generating final output.
- Separate reasoning from final output.
- Final output must include structured JSON.

---------------------------------------------------------
STEP 1 – REQUIREMENT COLLECTION
---------------------------------------------------------
Collect the following information:

MANDATORY:
- Total plot area (sq ft or sq meter)
- Plot dimensions (length × width)
- Number of floors
- Number of bedrooms
- Number of bathrooms
- Living room required? (yes/no)
- Kitchen required? (yes/no)

OPTIONAL:
- Dining room
- Balcony
- Parking
- Garden
- Vastu preference
- Modern / traditional style
- Budget range
- Staircase position preference
- Attached bathrooms?

If any mandatory data is missing, ask clear follow-up questions.

---------------------------------------------------------
STEP 2 – REQUIREMENT VALIDATION
---------------------------------------------------------
Validate:
- Total room area must not exceed plot area.
- Minimum bedroom size: 100 sq ft
- Minimum bathroom size: 35 sq ft
- Minimum living room: 120 sq ft
- Circulation space: at least 10% of total area
- Wall thickness: 9 inches standard

If constraints fail, explain clearly and suggest corrections.

---------------------------------------------------------
STEP 3 – SPACE PLANNING LOGIC
---------------------------------------------------------
Apply architectural logic:
- Public areas (living room) near entrance.
- Kitchen near dining.
- Bedrooms in private zone.
- Bathrooms accessible but private.
- Avoid bedroom directly opening to living room.
- Maintain natural light access.
- Optimize circulation paths.
- Follow Vastu Shastra if user prefers.

---------------------------------------------------------
STEP 4 – GENERATE LAYOUT DATA
---------------------------------------------------------
When all requirements are gathered and validated, generate structured JSON:

```json
{
  "total_area": <number in sq ft>,
  "plot": {
    "width": <number in feet>,
    "length": <number in feet>
  },
  "rooms": [
    {
      "room_type": "<type>",
      "name": "<display name>",
      "quantity": <int>,
      "width": <feet>,
      "length": <feet>,
      "desired_area": <sq ft>,
      "position": {"x": <feet from origin>, "y": <feet from origin>},
      "door_position": "<N/S/E/W>",
      "windows": ["<direction1>"]
    }
  ],
  "walls": {
    "exterior_thickness": "9 inches",
    "interior_thickness": "4.5 inches"
  },
  "vastu_recommendations": [
    {"room": "<type>", "recommended_direction": "<direction>", "reason": "<why>"}
  ],
  "compliance_notes": ["<note1>", "<note2>"],
  "design_score": <1-10>,
  "ready_to_generate": true
}
```

Ensure:
- No overlapping rooms
- Logical adjacency
- Dimensions are realistic
- Total built area <= allowed area

---------------------------------------------------------
STEP 5 – DESIGN EXPLANATION
---------------------------------------------------------
After JSON, provide:
- Short explanation of layout logic
- Why rooms are placed that way
- Suggestions for improvement

---------------------------------------------------------
IMPORTANT RULES
---------------------------------------------------------
- Never output invalid JSON.
- Never guess unknown plot dimensions.
- Never ignore building constraints.
- Think like a licensed architect.
- Design must be construction-practical.

Valid room types: master_bedroom, bedroom, bathroom, kitchen, living, dining, study, \
pooja, store, utility, porch, parking, staircase, toilet, balcony, hallway, garage, other.

Always respond naturally first, then include JSON at the end if you have structured data.
Wrap JSON in ```json ... ``` code blocks."""


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
        messages = [{"role": "system", "content": ARCHITECT_SYSTEM_PROMPT + "\n\n" + CHAT_SYSTEM_PROMPT}]
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
