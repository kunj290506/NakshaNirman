"""
Groq Chat Integration.

Provides conversational AI for floor plan design with structured data extraction.
Includes fallback rule-based chatbot when Groq is unavailable.
"""

import json
import re
import asyncio
from typing import Optional
from config import GROQ_API_KEY, GROQ_MODEL

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
        should_generate = False

        if extracted:
            mode = extracted.get("mode", "")
            # Designing or modifying mode → trigger plan generation
            should_generate = mode in ("designing", "modifying")
            # Legacy fallback
            if not should_generate:
                should_generate = extracted.get("ready_to_generate", False) or extracted.get("requirements_complete", False)

        # Build a human-readable reply for display
        if extracted:
            mode = extracted.get("mode", "")
            if mode == "collecting":
                # Show the question to the user
                clean_reply = extracted.get("question", "")
                context = extracted.get("context_understood", "")
                if context and clean_reply:
                    clean_reply = f"{context}\n\n{clean_reply}"
                elif not clean_reply:
                    clean_reply = reply
            elif mode in ("designing", "modifying"):
                # Show the architect note
                clean_reply = extracted.get("architect_note", "Your floor plan is being generated!")
            else:
                # Legacy format: strip JSON/code fences for display
                clean_reply = re.sub(r'```(?:json)?\s*.*?\s*```', '', reply, flags=re.DOTALL).strip()
                if not clean_reply:
                    clean_reply = reply
        else:
            clean_reply = reply

        return {
            "reply": clean_reply,
            "extracted_data": extracted,
            "should_generate": should_generate,
        }

    except Exception as e:
        # Fallback on any Groq error
        return _fallback_chat(message, history)


# ---- Fallback Rule-Based Chatbot (NakshaNirman) ----

_FALLBACK_STATES = {
    "greeting": {
        "keywords": [],
        "response": "Welcome to NakshaNirman! I will help you design your dream home. First, what is your plot size? For example, 30 by 40 feet, or tell me the total area in square feet.",
        "next": "area",
    },
    "area": {
        "keywords": ["sq ft", "square feet", "sqft", "area"],
        "response": "Great! How many bedrooms are you planning — would you like a 2BHK or 3BHK?",
        "next": "bedrooms",
    },
    "bedrooms": {
        "keywords": ["bedroom", "bed", "room"],
        "response": "How many bathrooms?",
        "next": "bathrooms",
    },
    "bathrooms": {
        "keywords": ["bathroom", "bath", "toilet"],
        "response": "Do you need any special rooms? (study, pooja room, store room, garage)",
        "next": "special",
    },
    "special": {
        "keywords": ["study", "pooja", "store", "garage", "no", "none"],
        "response": "Wonderful! Would you like any special rooms — for example, a separate dining room, a study room, or a pooja room?",
        "next": "generate",
    },
    "generate": {
        "keywords": ["generate", "create", "build", "yes", "make", "haan", "ha", "sure", "proceed"],
        "response": "Generating your Vastu-compliant floor plan now!",
        "next": "done",
    },
}


def _extract_number(text: str) -> Optional[int]:
    """Extract the first number from text."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def _fallback_chat(message: str, history: list) -> dict:
    """NakshaNirman rule-based fallback chatbot with Indian housing knowledge."""
    msg_lower = message.lower()
    extracted = None
    should_generate = False

    # Determine current state from history length
    turn = len([h for h in history if h.get("role") == "user"])

    # Off-topic detection — only allow architecture/housing related messages
    architecture_keywords = [
        "plot", "sqft", "sq ft", "square", "feet", "foot", "meter",
        "bhk", "bedroom", "bathroom", "kitchen", "living", "dining",
        "room", "house", "home", "floor", "plan", "design", "build",
        "vastu", "pooja", "puja", "balcony", "terrace", "garage",
        "parking", "store", "storage", "study", "office", "library",
        "staircase", "stairs", "passage", "corridor", "entrance",
        "door", "window", "wall", "area", "size", "width", "length",
        "generate", "create", "make", "yes", "ok", "sure", "proceed",
        "haan", "ha", "go ahead", "no", "nahi",
        "east", "west", "north", "south",
        "architect", "construction", "setback", "boundary",
        "utility", "servant", "sitout", "sit-out", "foyer",
        "naksha", "nirman", "ghar", "kamra", "rasoi",
        "hi", "hello", "hey", "help", "start", "thanks", "thank",
    ]
    if turn > 0 and not any(kw in msg_lower for kw in architecture_keywords):
        return {
            "reply": "I can only help with floor plan design and architecture. Please tell me about your plot size and room requirements.",
            "extracted_data": None,
            "should_generate": False,
        }

    if turn == 0:
        reply = _FALLBACK_STATES["greeting"]["response"]
    elif turn == 1:
        num = _extract_number(message)
        area = num if num else 1200
        reply = f"Great, {area} square feet gives a comfortable home! {_FALLBACK_STATES['area']['response']}"
        extracted = {"total_area": area, "rooms": [], "ready_to_generate": False}
    elif turn == 2:
        num = _extract_number(message)
        bedrooms = num if num else 2
        # Indian standard: attached bathroom per bedroom
        bathrooms = max(1, bedrooms - 1) if bedrooms <= 2 else bedrooms - 1
        bhk_label = f"{bedrooms}BHK"
        reply = (
            f"Excellent choice — a {bhk_label}! For Indian homes, I recommend {bathrooms} attached "
            f"bathrooms as the standard. How many bathrooms would you like?"
        )
        extracted = {"rooms": [{"room_type": "bedroom", "quantity": bedrooms}], "ready_to_generate": False}
    elif turn == 3:
        num = _extract_number(message)
        bathrooms = num if num else 2
        reply = f"Perfect, {bathrooms} attached bathroom(s). {_FALLBACK_STATES['bathrooms']['response']}"
        extracted = {"rooms": [{"room_type": "bathroom", "quantity": bathrooms}], "ready_to_generate": False}
    elif turn == 4:
        rooms = []
        extras_list = []
        if "dining" in msg_lower:
            rooms.append({"room_type": "dining", "quantity": 1})
            extras_list.append("Dining Room")
        if "study" in msg_lower or "library" in msg_lower or "office" in msg_lower:
            rooms.append({"room_type": "study", "quantity": 1})
            extras_list.append("Study Room")
        if "pooja" in msg_lower or "puja" in msg_lower:
            rooms.append({"room_type": "pooja", "quantity": 1})
            extras_list.append("Pooja Room (NE corner as per Vastu)")
        if "store" in msg_lower or "storage" in msg_lower:
            rooms.append({"room_type": "store", "quantity": 1})
            extras_list.append("Store Room")
        if "balcony" in msg_lower or "terrace" in msg_lower:
            rooms.append({"room_type": "balcony", "quantity": 1})
            extras_list.append("Balcony")
        if "garage" in msg_lower or "parking" in msg_lower:
            rooms.append({"room_type": "garage", "quantity": 1})
            extras_list.append("Parking")

        if extras_list:
            reply = (
                f"Noted: {', '.join(extras_list)}. "
                f"Shall I generate your Vastu-compliant floor plan now? "
                f"Type 'generate' or 'yes' to proceed."
            )
        else:
            reply = (
                "No problem! Shall I generate your floor plan now? "
                "Type 'generate' or 'yes' to proceed."
            )
        if rooms:
            extracted = {"rooms": rooms, "ready_to_generate": False}
    else:
        if any(kw in msg_lower for kw in ["generate", "create", "build", "make", "yes",
                                           "sure", "ok", "proceed", "haan", "ha", "go ahead"]):
            reply = "Generating your Vastu-compliant NakshaNirman floor plan now!"
            should_generate = True
            extracted = {"ready_to_generate": True}
        else:
            reply = (
                "Would you like me to generate the floor plan? "
                "Say 'yes', 'generate', or 'haan' when ready."
            )

    return {
        "reply": reply,
        "extracted_data": extracted,
        "should_generate": should_generate,
    }
