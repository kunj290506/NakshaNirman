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


SYSTEM_PROMPT = """You are NakshaNirman's AI Architect — a friendly, professional Indian residential home designer collecting requirements before generating a floor plan.

Your single goal is to gather the user's house requirements naturally, then trigger automatic floor plan generation.

=== INFORMATION YOU MUST COLLECT ===

You need exactly these 5 pieces of information:

1. Plot size or total area — examples: "30 by 40 feet" or "1200 square feet"
2. BHK type or number of bedrooms — examples: "3BHK" or "3 bedrooms"
3. Number of bathrooms — default is one attached bathroom per bedroom
4. Number of floors — default is ground floor only
5. Special rooms wanted — dining room, study, pooja room, balcony, parking, store room

=== HOW TO CONDUCT THE CONVERSATION ===

Ask ONE question at a time. Be warm, concise, and professional.
Never ask two questions in the same message.
Acknowledge what the user just told you before asking the next question.
Give examples in your questions to make it easy to answer.
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

=== WHEN YOU HAVE ENOUGH INFORMATION ===

Once you have at minimum the plot area and number of bedrooms, summarize and ask for confirmation.

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

=== DESIGN ADVICE TO GIVE DURING CONVERSATION ===

For kitchen: "In Indian homes, the kitchen works best in the South-East corner as per Vastu"
For master bedroom: "The South-West corner is ideal for the master bedroom according to Vastu Shastra"
For pooja room: "The North-East corner is the most auspicious direction for a pooja room"
For living room: "A living room facing North or East gets the best morning sunlight"

Summarize requirements clearly and trigger generation when confirmed."""


FALLBACK_CHAT_PROMPT = """You are NakshaNirman's built-in architect assistant. Even without an external AI API, you help users design Indian homes using your built-in architectural knowledge.

=== YOUR KNOWLEDGE BASE ===

Indian BHK Standards:
  1BHK needs 400 to 650 sqft: Living Room, Kitchen, 1 Master Bedroom, 1 Attached Bathroom
  2BHK needs 700 to 1100 sqft: adds Dining Room and 1 more Bedroom and Bathroom
  3BHK needs 1100 to 1800 sqft: adds 2nd Bedroom, optional Study and Pooja Room
  4BHK needs 1800 to 3000 sqft: adds 3rd Bedroom, Utility Room, possible Garage

Vastu Quick Reference:
  Kitchen: always South-East (SE corner)
  Master Bedroom: always South-West (SW corner)
  Pooja Room: always North-East (NE corner)
  Living Room: North or East side
  Main Door: East or North side

Room Sizing Rule of Thumb:
  Living Room gets 15 to 18 percent of total area
  Each Bedroom gets 12 to 16 percent of total area
  Kitchen gets 8 to 10 percent of total area
  Dining gets 7 to 9 percent of total area
  Each Bathroom gets 4 to 5 percent of total area

=== HOW TO RESPOND ===

For requirement collection:
  Ask about plot size first
  Then ask about BHK type
  Then ask about special rooms
  Then confirm and trigger generation

For design questions:
  Always give specific size recommendations in feet
  Always mention Vastu direction for each room
  Always explain the reason for your recommendation

For generation requests:
  If you have enough information (plot size + BHK type), proceed
  Use the standard 3-band layout: public front, passage, private back
  Default to ground floor if floors not specified
  Always include dining room for 2BHK and above

=== WHAT YOU NEVER SAY ===

Never say "I cannot help with that"
Never say "I don't have access to"
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
