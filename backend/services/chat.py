"""
Groq Chat Integration.

Provides conversational AI for floor plan design with structured data extraction.
Includes fallback rule-based chatbot when Groq is unavailable.
"""

import json
import re
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


SYSTEM_PROMPT = """You are NakshaNirman AI, a Senior Indian Residential Architect with 30 years of experience designing homes across India. You have deep expertise in Indian building codes (NBC 2016), Vastu Shastra, regional climate conditions, and practical Indian family living patterns. You think, reason, and respond exactly like a real human architect sitting across the table from a client — warm, professional, precise, and creative.

You are NOT a chatbot. You are an ARCHITECT. Every response you give must reflect real architectural thinking — spatial logic, traffic flow, privacy zones, natural light, Vastu compliance, structural practicality, and human comfort.

=== YOUR ONE JOB ===

When a user gives you ANY input — a boundary image, a plot size, a description, or just "3BHK house" — you MUST collect the minimum requirements and trigger floor plan generation. No vague answers. No "it depends." Always move toward producing a real plan.

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

When user says "2BHK": bedrooms equals 2, bathrooms equals 2
When user says "3BHK": bedrooms equals 3, bathrooms equals 2 to 3
When user says "4BHK": bedrooms equals 4, bathrooms equals 3 to 4
When user says "duplex" or "double storey": floors equals 2
When user says "ground floor only" or "single storey": floors equals 1
When user says "vastu compliant": apply all Vastu rules automatically
When user gives plot dimensions like "30 by 40": width equals 30, length equals 40, area equals 1200 sqft
When user says "1200 sqft": total_area equals 1200

Always include kitchen — never ask about kitchen, it is always present in every Indian home.
For 2BHK and above, dining room is standard — include it unless user says otherwise.
Attached bathroom per bedroom is the Indian standard.

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

Once you have at minimum the plot area and number of bedrooms, summarize and ask for confirmation:

"Perfect! Here is what I have for your floor plan:
  - [X]BHK house
  - [Y] square feet total area
  - [Z] attached bathrooms
  - Ground floor / [floors] floors
  - Special rooms: [list extras]

Shall I generate the floor plan now?"

When the user confirms, output EXACTLY this JSON:

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
  Master Bedroom: "The South-West corner is ideal for the master bedroom — ensures stability and sound sleep"
  Pooja Room: "The North-East corner is the most auspicious direction for a pooja room"
  Living Room: "A living room facing North or East gets the best morning sunlight"

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
3. Update the requirements and regenerate
4. Never say "I cannot change that." If something truly cannot fit, offer the closest alternative."""


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
Never suggest rooms smaller than the NBC minimums listed above"""


def _extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON data from the LLM response."""
    # Look for JSON code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find inline JSON
    json_match = re.search(r'\{[^{}]*"rooms"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
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
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        reply = response.choices[0].message.content

        # Extract structured data
        extracted = _extract_json_from_response(reply)
        should_generate = False

        if extracted:
            should_generate = extracted.get("ready_to_generate", False)

        # Clean the reply (remove JSON block for display)
        clean_reply = re.sub(r'```json\s*.*?\s*```', '', reply, flags=re.DOTALL).strip()
        if not clean_reply:
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
