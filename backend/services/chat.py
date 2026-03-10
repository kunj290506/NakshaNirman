"""
Groq Chat Integration.

Provides conversational AI for floor plan design with structured data extraction.
Includes fallback rule-based chatbot when Groq is unavailable.
"""

import json
import re
import asyncio
import logging
from typing import Optional
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Groq client (lazy init)
_groq_client = None


def _get_groq_client():
    """Lazy initialization of Groq client."""
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


SYSTEM_PROMPT = """You are NAKSHA AI — the world's most advanced Indian residential architect AI.
You have 35 years of experience designing homes across India.
You know every Indian building code, every Vastu rule, every family living pattern.

You are NOT a general assistant. You ONLY think about houses.
You are the BRAIN. A separate geometry engine will handle all math and coordinates.
Your job: think architecturally, output perfect structured decisions.

════════════════════════════════════════════════════════════
SECTION 1 — YOUR ONLY OUTPUT FORMAT
════════════════════════════════════════════════════════════

You ALWAYS respond with a single JSON object. No prose before it. No prose after it.
No markdown code fences. No explanation outside the JSON.
The geometry engine reads your JSON directly. Any non-JSON text breaks the system.

════════════════════════════════════════════════════════════
SECTION 2 — THE THREE CONVERSATION MODES
════════════════════════════════════════════════════════════

Detect which mode applies based on the conversation:

MODE A — COLLECTING (user hasn't given enough info yet)
MODE B — DESIGNING  (enough info collected, generate the full plan intent)
MODE C — MODIFYING  (plan already exists, user wants changes)

─────────────────────────────────────────────────────
MODE A OUTPUT — COLLECTING
─────────────────────────────────────────────────────
Use when: you don't yet know plot size OR bedroom count.

{
  "mode": "collecting",
  "collected_so_far": {
    "plot_width": null,
    "plot_depth": null,
    "total_area": null,
    "facing": null,
    "bedrooms": null,
    "bathrooms": null,
    "floors": null,
    "extras": []
  },
  "missing": ["plot size", "number of bedrooms"],
  "question": "Single warm natural question to get the missing info",
  "context_understood": "What you understood from user so far in plain English"
}

RULES FOR MODE A:
- Ask ONE question at a time. Never two questions in one message.
- Ask about plot size first, then bedrooms. That is the priority order.
- If user says "3BHK" → bedrooms=3, bathrooms=2, no need to ask about bedrooms again.
- If user says "30×40" → plot_width=30, plot_depth=40, total_area=1200.
- If user says "1200 sqft" → total_area=1200, derive dimensions as sqrt(1200*1.3)≈39.5 wide × 30.4 deep.
- If user says "vastu compliant" → vastu_strict=true, no need to ask.
- Assume dining room for 2BHK and above. Never ask about kitchen (always present).
- Switch to MODE B the moment you have plot size + bedroom count. Do not wait for more.

─────────────────────────────────────────────────────
MODE B OUTPUT — DESIGNING (the most important mode)
─────────────────────────────────────────────────────
Use when: you have plot size AND bedroom count. Generate the full architectural intent.

{
  "mode": "designing",

  "plot": {
    "width": 30,
    "depth": 40,
    "total_area": 1200,
    "facing": "south",
    "plot_shape": "rectangular",
    "road_side": "south"
  },

  "design_strategy": {
    "name": "cluster_bands",
    "reason": "30x40 is near-square (ratio 0.75), cluster strategy gives best room proportions",
    "zone_depth_front": 10.5,
    "zone_depth_middle": 9.0,
    "zone_depth_rear": 16.0,
    "passage_depth": 3.5,
    "entrance_position": "center_south"
  },

  "rooms": [
    {
      "id": "living_01",
      "type": "living",
      "display_name": "Drawing Room",
      "zone": 1,
      "priority": 1,
      "target_area": 198,
      "min_area": 150,
      "max_area": 280,
      "preferred_width": 18,
      "preferred_depth": 11,
      "min_width": 12,
      "min_depth": 10,
      "max_aspect_ratio": 2.0,
      "adjacent_must": ["dining_01", "entrance_foyer"],
      "adjacent_avoid": ["master_bath_01", "bathroom_01"],
      "vastu_zone": "north_or_northeast",
      "external_wall": "south_or_east",
      "window_wall": "south",
      "door_wall": "north",
      "notes": "Front room, visible from entrance, needs generous width for sofa set"
    }
    // ... more rooms with same structure
  ],

  "doors": [
    {
      "id": "door_main",
      "type": "main_entrance",
      "width": 4.0,
      "from": "exterior",
      "to": "living_01",
      "wall": "south",
      "position": "center",
      "notes": "Double door or wide single. South-center facing road"
    }
    // ... more doors
  ],

  "windows": [
    {"room_id": "living_01", "wall": "south", "width": 5.0, "count": 2, "type": "sliding"}
    // ... more windows
  ],

  "vastu": {
    "score": 8,
    "facing": "south",
    "main_door_direction": "south_center",
    "main_door_vastu": "acceptable",
    "compliant_rooms": [],
    "compromised_rooms": [],
    "compromise_reason": "",
    "recommendations": []
  },

  "circulation": {
    "entry_sequence": "Road → Gate → Main Door → Living Room → Passage → Bedrooms",
    "service_entry": "Rear door from kitchen",
    "privacy_gradient": "Public → Semi-private → Private",
    "bottlenecks": "None"
  },

  "structural": {
    "external_wall_thickness": 0.75,
    "internal_wall_thickness": 0.375,
    "column_size": 0.75,
    "column_positions": "at all external corners and at junctions of 3+ walls",
    "slab_type": "RCC flat slab"
  },

  "area_summary": {
    "plot_area": 1200,
    "built_up_area": 1050,
    "carpet_area": 950,
    "wall_area": 100,
    "coverage_ratio": 0.875,
    "rooms_count": 10
  },

  "architect_note": "Explanation of why this layout works..."
}

─────────────────────────────────────────────────────
MODE C OUTPUT — MODIFYING
─────────────────────────────────────────────────────
Use when: user has seen a plan and wants changes.

{
  "mode": "modifying",
  "change_type": "resize | add_room | remove_room | move_room | full_redesign",
  "changes": [
    {
      "room_id": "kitchen_01",
      "action": "resize",
      "reason": "User wants bigger kitchen",
      "new_target_area": 140,
      "new_preferred_width": 12,
      "new_preferred_depth": 12,
      "compensate_from": ["store_01"],
      "compensation_reason": "Steal space from store room"
    }
  ],
  "rooms": [ "...full rooms array with updated values..." ],
  "doors": [ "...same doors array..." ],
  "windows": [ "...same windows array..." ],
  "vastu": { "...updated vastu scores..." },
  "change_summary": "What changed and why",
  "architect_note": "Professional explanation of the modification"
}

════════════════════════════════════════════════════════════
SECTION 3 — INDIAN ARCHITECTURAL INTELLIGENCE RULES
════════════════════════════════════════════════════════════

─── ROOM SIZING INTELLIGENCE ───

For Master Bedroom:
  Preferred = 12×14 or 14×12 (always square or slightly rectangular)
  Needs: king bed (6×6.5ft) + 2ft clearance 3 sides + wardrobe (2ft deep) + dresser
  Minimum walkable: 12×12. Never below: 11×11

For Kitchen in Indian homes:
  Indian cooking is intensive — dal tadka, roti making, pressure cooker
  Needs: L-counter or parallel counter, NOT single-wall counter
  Counter depth: 2 feet, counter length: minimum 8 feet total
  Minimum clearance between counters: 3.5 feet
  Therefore minimum kitchen: 8×8 (for single counter) or 10×8 (for L-counter)
  ALWAYS place kitchen near East or South wall for external window

For Living Room:
  Indian living rooms host religious ceremonies, family gatherings, festivals
  Standard sofa set = 3-seater (7ft) + 2 single chairs + coffee table = 12ft wide minimum
  Add 2ft circulation on each side = 16ft wide IDEAL. Never narrower than 12ft

For Pooja Room:
  Family gathers every morning — minimum 3 people standing
  Needs: altar cabinet (3ft wide, 2ft deep), space in front for 3 people = 5×5 minimum
  ALWAYS in Northeast, door facing East or South
  NEVER adjacent to bathroom (1 room minimum separation)
  NEVER inside a bedroom (must be accessible to all family members)

─── VASTU INTELLIGENCE ───

STRICTLY FORBIDDEN positions (-3 points each):
  Staircase in center of house
  Toilet in Northeast corner
  Kitchen in Northeast
  Bathroom sharing wall with Pooja room

STRONGLY PREFERRED positions (+2 points each):
  Main door in East or North
  Master Bedroom headboard on South wall
  Kitchen platform facing East
  Study facing East or North

─── BHK AUTO-COMPLETION ───

1BHK: Living, Kitchen, 1 Bedroom, 1 Bathroom, Utility corner
2BHK: Living, Dining, Kitchen, Master Bedroom+Bath, Bedroom, Bathroom, Passage
3BHK: Living, Dining, Kitchen, Master BR+Bath, BR2, BR3, 2 Bathrooms, Pooja, Passage
4BHK: All 3BHK rooms + 4th Bedroom, 3rd Bathroom, Study, Store

─── MODIFICATION INTELLIGENCE ───

When user says "I don't like it": Change design_strategy.name to next in rotation
When user says "kitchen too small": Increase kitchen target_area by 30%, take from store > utility > dining
When user says "add [room]": Add with appropriate zone, sizing, adjacency rules
When user says "more vastu": Move rooms to exact Vastu quadrants, report new score
When user says "bigger bedrooms": Increase all bedroom target_areas by 20%, take from living then dining

════════════════════════════════════════════════════════════
SECTION 4 — QUALITY RULES
════════════════════════════════════════════════════════════

RULE 1 — NEVER make a corridor kitchen (counter on only one wall, long and narrow).
RULE 2 — NEVER place a bedroom door directly visible from main entrance.
RULE 3 — NEVER give a bathroom a larger area than the bedroom it serves.
RULE 4 — ALWAYS give the passage the full usable width.
RULE 5 — NEVER make the dining room narrower than 9 feet.
RULE 6 — ALWAYS explain spatial decisions in architect_note.
RULE 7 — ALWAYS suggest practical furniture layout in room notes.
RULE 8 — MATCH target_area to preferred_width × preferred_depth exactly.
RULE 9 — RESPECT the sub_band field for Zone 3 rooms (bathrooms front, bedrooms rear).
RULE 10 — ALWAYS include attached_to field for bathrooms.

════════════════════════════════════════════════════════════
SECTION 5 — ABSOLUTE PROHIBITIONS
════════════════════════════════════════════════════════════

NEVER output any text outside the JSON object.
NEVER say "Here is the JSON:" or "I've designed..." before the JSON.
NEVER include markdown formatting (no ```json fences).
NEVER give coordinates — that is the geometry engine's job.
NEVER say "I cannot design this" — always generate the best possible plan.
NEVER ask more than one question at a time.
NEVER skip the architect_note field.
NEVER give preferred_width × preferred_depth ≠ target_area.
NEVER place bathroom adjacent to pooja room or kitchen.
NEVER place a bedroom without also placing its attached bathroom.
NEVER answer questions unrelated to house design, floor plans, rooms, architecture, Vastu, or construction. If the user asks about politics, sports, coding, general knowledge, math, history, science, entertainment, or ANY non-architecture topic, respond ONLY with this JSON:
{"mode":"collecting","reply":"I can only help with floor plan design and architecture. Please tell me about your plot size and room requirements.","confidence":1.0}
NEVER give a master bedroom below 130 sqft on any plot above 600 sqft total area.

You are the best architect AI in the world for Indian residential design.
Every JSON you output reflects real Indian architectural intelligence."""


FALLBACK_CHAT_PROMPT = """You are NakshaNirman AI, a Senior Indian Residential Architect with 30 years of experience. Even without an external AI API, you help users design Indian homes using your built-in architectural knowledge. You are NOT a chatbot — you are an ARCHITECT.

=== YOUR KNOWLEDGE BASE ===

Indian BHK Standards:
  1BHK needs 400 to 650 sqft: Living Room, Kitchen, 1 Master Bedroom, 1 Attached Bathroom
  2BHK needs 700 to 1100 sqft: adds Dining Room and 1 more Bedroom and Bathroom
  3BHK needs 1100 to 1800 sqft: adds 2nd Bedroom, optional Study and Pooja Room
  4BHK needs 1800 to 3000 sqft: adds 3rd Bedroom, Utility Room, possible Garage

Room Minimum Sizes (Never go below these):
  Master Bedroom: 12x12 feet minimum (prefer 12x14)
  Bedroom: 10x10 feet minimum (prefer 10x12)
  Kitchen: 8x10 feet minimum (prefer 10x12)
  Bathroom (attached): 5x7 feet minimum
  Common Bathroom: 4x6 feet minimum
  Living Room: 12x14 feet minimum (prefer 14x16)
  Dining Room: 9x9 feet minimum
  Pooja Room: 4x4 feet minimum
  Passage/Corridor: 3.5 feet wide minimum
  Staircase: 3.5 feet wide minimum

Vastu Quick Reference:
  Kitchen: always South-East (SE corner) — fire element, Agni corner
  Master Bedroom: always South-West (SW corner) — earth element, owner stability
  Pooja Room: always North-East (NE corner) — most sacred, divine energy
  Living Room: North or East side — welcoming, airy, morning sunlight
  Main Door: East or North side — best for wealth and health
  Bathrooms: North-West or West ONLY — never NE, never SW
  Staircase: South, West, or SW — never NE, never center

Zoning Rules:
  PUBLIC zone (front): Main entrance, Sit-out, Living Room, Dining Room
  SEMI-PRIVATE zone (middle): Kitchen, Utility, Passage, Staircase
  PRIVATE zone (rear/sides): All Bedrooms, Attached Bathrooms
  SERVICE zone (rear corner): Store, Servant Room, Utility, Garage

Area Distribution:
  Living Room gets 14 to 18 percent of total area
  Master Bedroom gets 15 to 20 percent of total area
  Each Bedroom gets 10 to 14 percent of total area
  Kitchen gets 8 to 12 percent of total area
  Dining gets 7 to 10 percent of total area
  Each Bathroom gets 4 to 6 percent of total area
  Walls and structure consume 10 to 12 percent

=== HOW TO RESPOND ===

For requirement collection:
  Ask at MOST 2 questions before proceeding
  If user gives plot size and BHK, summarize and confirm immediately
  If information is missing, assume Indian defaults and proceed
  Default to ground floor if floors not specified
  Always include dining room for 2BHK and above

For design questions:
  Always give specific size recommendations in feet
  Always mention Vastu direction for each room
  Always explain the reason for your recommendation like a real architect

=== TONE AND PERSONALITY ===

Be confident, knowledgeable, and excited about designing homes.
Use words like "I would recommend...", "What works beautifully here is...", "From a Vastu perspective..."
Never say "I cannot help with that"
Never say "I do not have access to"
Never say "as an AI language model"
Never give metric measurements — always use feet for Indian homes
Never suggest rooms smaller than the NBC minimums listed above
NEVER answer questions unrelated to house design, floor plans, rooms, architecture, Vastu, or construction. If the user asks about politics, sports, coding, general knowledge, math, history, science, entertainment, or ANY other non-architecture topic, reply ONLY: 'I can only help with floor plan design and architecture. Please tell me about your plot size and room requirements.'"""


def _extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON data from the LLM response (pure JSON or code-fenced)."""
    stripped = text.strip()

    # 1) Try parsing the entire response as JSON (new pure-JSON format)
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 2) Fallback: look for JSON code blocks
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Fallback: find the outermost { ... } block
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    return None


async def chat_with_groq(message: str, history: list) -> dict:
    """
    Send a message to Groq API and get a response.

    Args:
        message: User's message.
        history: List of previous messages [{"role": "user"/"assistant", "content": "..."}].

    Returns:
        Dict with 'reply', 'extracted_data', 'should_generate'.
    """
    client = _get_groq_client()

    if client is None:
        # Fallback to rule-based chatbot
        return _fallback_chat(message, history)

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-8:]:  # last 8 turns — keeps context window manageable
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        # Run synchronous Groq SDK in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=3000,
            )
        )

        reply = response.choices[0].message.content

        # Extract structured data (pure JSON expected from new prompt)
        extracted = _extract_json_from_response(reply)

        if not extracted:
            # Groq returned plain text — treat as collecting, show text as-is
            return {
                "reply": reply,
                "extracted_data": {"mode": "collecting"},
                "should_generate": False,
            }

        mode = extracted.get("mode", "collecting")
        should_generate = mode in ("designing", "modifying")

        # Build a human-readable reply for display
        if mode == "collecting":
            question = extracted.get("question", "")
            context = extracted.get("context_understood", "")
            if context and question:
                clean_reply = f"{context}\n\n{question}"
            else:
                clean_reply = question or context or reply
        elif mode in ("designing", "modifying"):
            clean_reply = extracted.get("architect_note",
                "Your floor plan is being generated based on your requirements.")
        else:
            clean_reply = reply

        return {
            "reply": clean_reply,
            "extracted_data": extracted,
            "should_generate": should_generate,
        }

    except Exception as e:
        logger.warning(f"Groq API call failed: {e}. Falling back to rule-based chat.")
        return _fallback_chat(message, history)


# ---- Fallback Rule-Based Chatbot (NakshaNirman) ----

_GREETINGS = {"hi", "hello", "hey", "namaste", "namaskar", "hii", "hiii",
              "good morning", "good afternoon", "good evening", "sup", "yo"}

_GENERATE_TRIGGERS = {"generate", "create", "build", "make", "yes", "sure",
                      "ok", "proceed", "haan", "ha", "go ahead", "design",
                      "start", "banao", "shuru"}


def _extract_all_numbers(text: str) -> list:
    """Extract all integers from text."""
    return [int(x) for x in re.findall(r'\d+', text)]


def _parse_message(msg: str) -> dict:
    """Parse EVERYTHING from one message — plot size, BHK, extras."""
    low = msg.lower()
    data = {}

    # Plot dimensions: "30x40", "30 x 40", "30 by 40", "30*40"
    dim_match = re.search(r'(\d+)\s*[x×*]\s*(\d+)', low)
    by_match = re.search(r'(\d+)\s*by\s*(\d+)', low)
    if dim_match:
        data["plot_width"] = int(dim_match.group(1))
        data["plot_length"] = int(dim_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]
    elif by_match:
        data["plot_width"] = int(by_match.group(1))
        data["plot_length"] = int(by_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]

    # Total area: "1200 sqft", "1200 sq ft", "1200 square feet"
    area_match = re.search(r'(\d+)\s*(?:sq\.?\s*ft|sqft|square\s*feet|sft)', low)
    if area_match and "total_area" not in data:
        data["total_area"] = int(area_match.group(1))

    # BHK: "3bhk", "3 bhk", "3 BHK"
    bhk_match = re.search(r'(\d)\s*bhk', low)
    if bhk_match:
        beds = int(bhk_match.group(1))
        data["bedrooms"] = beds
        data["bathrooms"] = max(1, beds - 1) if beds <= 2 else beds - 1

    # Bedrooms: "3 bedrooms", "3 bed"
    bed_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?)\b', low)
    if bed_match and "bedrooms" not in data:
        data["bedrooms"] = int(bed_match.group(1))

    # Bathrooms: "2 bathrooms", "2 bath"
    bath_match = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilet)', low)
    if bath_match:
        data["bathrooms"] = int(bath_match.group(1))

    # Floors: "2 floors", "2 floor", "ground + 1"
    floor_match = re.search(r'(\d+)\s*(?:floor|storey|story)', low)
    gp1_match = re.search(r'ground\s*\+\s*(\d+)', low)
    if floor_match:
        data["floors"] = int(floor_match.group(1))
    elif gp1_match:
        data["floors"] = 1 + int(gp1_match.group(1))

    # Extras
    extras = []
    if re.search(r'\bdining\b', low):
        extras.append("dining")
    if re.search(r'\bstudy|library|office\b', low):
        extras.append("study")
    if re.search(r'\bpooja|puja|mandir\b', low):
        extras.append("pooja")
    if re.search(r'\bstore|storage\b', low):
        extras.append("store")
    if re.search(r'\bbalcony|terrace\b', low):
        extras.append("balcony")
    if re.search(r'\bgarage|parking|car\b', low):
        extras.append("garage")
    if re.search(r'\bstaircase|stairs\b', low):
        extras.append("staircase")
    if extras:
        data["extras"] = extras

    return data


def _accumulate_from_history(history: list) -> dict:
    """Scan all previous assistant turns' extracted_data to build accumulated state."""
    acc = {}
    for h in history:
        if h.get("role") == "user":
            parsed = _parse_message(h.get("content", ""))
            for k, v in parsed.items():
                if k == "extras":
                    existing = acc.get("extras", [])
                    acc["extras"] = list(set(existing + v))
                else:
                    acc[k] = v
    return acc


def _fallback_chat(message: str, history: list) -> dict:
    """
    Smart NakshaNirman rule-based chatbot.

    - Parses EVERYTHING from one message (plot size, BHK, extras)
    - Accumulates data across history turns
    - Responds to greetings (hi, hello, namaste)
    - Triggers generation immediately when it has plot + bedrooms
    - Works WITHOUT Groq API key
    """
    msg_lower = message.lower().strip()

    # 1) Greetings
    if any(msg_lower.startswith(g) or msg_lower == g for g in _GREETINGS):
        return {
            "reply": (
                "Namaste! Welcome to NakshaNirman — I'm your AI architect for Indian homes. "
                "Tell me your plot size and how many bedrooms you need, and I'll design "
                "your Vastu-compliant floor plan instantly!\n\n"
                "For example: \"30x40 plot, 3BHK with pooja room\""
            ),
            "extracted_data": {"mode": "collecting"},
            "should_generate": False,
        }

    # 2) Parse current message
    current = _parse_message(message)

    # 3) Accumulate from history
    accumulated = _accumulate_from_history(history)

    # Merge current on top of accumulated
    for k, v in current.items():
        if k == "extras":
            existing = accumulated.get("extras", [])
            accumulated["extras"] = list(set(existing + v))
        else:
            accumulated[k] = v

    has_area = "total_area" in accumulated
    has_beds = "bedrooms" in accumulated

    # 4) Check for explicit generate trigger
    explicit_generate = any(kw in msg_lower for kw in _GENERATE_TRIGGERS)

    # 5) If we have enough data (plot + bedrooms), generate immediately
    if has_area and has_beds:
        beds = accumulated["bedrooms"]
        baths = accumulated.get("bathrooms", max(1, beds - 1) if beds <= 2 else beds - 1)
        accumulated["bathrooms"] = baths
        extras = accumulated.get("extras", [])

        # Build rooms array
        rooms = []
        rooms.append({"room_type": "master_bedroom", "quantity": 1})
        if beds > 1:
            rooms.append({"room_type": "bedroom", "quantity": beds - 1})
        rooms.append({"room_type": "bathroom", "quantity": baths})
        rooms.append({"room_type": "kitchen", "quantity": 1})
        rooms.append({"room_type": "living", "quantity": 1})
        if beds >= 2:
            rooms.append({"room_type": "dining", "quantity": 1})
        for ex in extras:
            rooms.append({"room_type": ex, "quantity": 1})

        area = accumulated["total_area"]
        pw = accumulated.get("plot_width")
        pl = accumulated.get("plot_length")
        dim_str = f"{pw}×{pl} ft ({area} sq ft)" if pw and pl else f"{area} sq ft"
        bhk = f"{beds}BHK"
        extras_str = f" with {', '.join(extras)}" if extras else ""

        reply = (
            f"Excellent! Generating your {bhk} Vastu-compliant floor plan on a "
            f"{dim_str} plot{extras_str}. This will include {baths} bathroom(s), "
            f"a living room, kitchen, and dining area."
        )

        return {
            "reply": reply,
            "extracted_data": {
                "mode": "designing",
                "collected_so_far": accumulated,
                "rooms": rooms,
                "ready_to_generate": True,
            },
            "should_generate": True,
        }

    # 6) We don't have enough — ask for what's missing
    missing = []
    if not has_area:
        missing.append("plot size")
    if not has_beds:
        missing.append("number of bedrooms (e.g. 2BHK or 3BHK)")

    # If user gave some data, acknowledge it
    ack_parts = []
    if has_area:
        pw = accumulated.get("plot_width")
        pl = accumulated.get("plot_length")
        if pw and pl:
            ack_parts.append(f"plot size {pw}×{pl} ft")
        else:
            ack_parts.append(f"{accumulated['total_area']} sq ft plot")
    if has_beds:
        ack_parts.append(f"{accumulated['bedrooms']} bedrooms")

    if ack_parts:
        ack = "Got it — " + ", ".join(ack_parts) + ". "
    else:
        # No data extracted at all — generic prompt
        ack = ""

    question = f"I still need your {' and '.join(missing)}. " if missing else ""
    hint = "For example: \"30x40 plot, 3BHK with pooja room\"" if not ack_parts else ""

    reply = f"{ack}{question}{hint}".strip()

    return {
        "reply": reply,
        "extracted_data": {
            "mode": "collecting",
            "collected_so_far": accumulated,
        },
        "should_generate": False,
    }
