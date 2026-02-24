"""
Production-Grade Residential Architectural Planning Engine.

Deterministic, rule-based architectural layout engine that:
  - Auto-detects input mode (CHAT / FORM / DESIGN / VALIDATION)
  - Enforces global architectural constraints
  - Generates CAD-ready JSON layouts
  - Validates layouts for compliance
  - Never crashes, never returns broken JSON

All outputs are machine-readable and backend-safe.
"""

import json
import math
import re
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum


# ===========================================================================
# CONSTANTS — Architectural Rules (NEVER BREAK)
# ===========================================================================

class EngineMode(str, Enum):
    CHAT = "chat"
    FORM = "form"
    DESIGN = "design"
    VALIDATION = "validation"


# Minimum room sizes (sq ft)
MIN_ROOM_SIZES = {
    "master_bedroom": 100,
    "bedroom": 100,
    "bathroom": 35,
    "toilet": 24,
    "kitchen": 80,
    "living": 120,
    "dining": 80,
    "study": 60,
    "pooja": 25,
    "store": 25,
    "utility": 24,
    "hallway": 30,
    "porch": 40,
    "parking": 150,
    "balcony": 30,
    "staircase": 50,
    "garage": 150,
}

# Standard room sizes (width x length in feet)
STANDARD_ROOM_SIZES = {
    "master_bedroom": (12, 14),
    "bedroom": (10, 12),
    "bathroom": (5, 8),
    "toilet": (4, 6),
    "kitchen": (8, 10),
    "living": (14, 16),
    "dining": (10, 12),
    "study": (10, 10),
    "pooja": (5, 5),
    "store": (6, 6),
    "utility": (4, 6),
    "hallway": (3.5, 10),
    "porch": (10, 8),
    "parking": (10, 18),
    "balcony": (4, 10),
    "staircase": (5, 10),
    "garage": (10, 18),
}

# Wall thickness
WALL_EXTERNAL_INCHES = 9
WALL_INTERNAL_INCHES = 4.5
WALL_EXTERNAL_FT = WALL_EXTERNAL_INCHES / 12  # 0.75 ft
WALL_INTERNAL_FT = WALL_INTERNAL_INCHES / 12  # 0.375 ft

# Minimum passage width
MIN_PASSAGE_WIDTH_FT = 3

# Area distribution guidelines (percentage of total)
AREA_DISTRIBUTION = {
    "living": (18, 22),
    "bedrooms_total": (30, 35),
    "kitchen": (8, 12),
    "bathrooms_total": (8, 12),
    "circulation": (10, 15),
    "walls_structure": (8, 10),
}

# Zone classification
ZONE_MAP = {
    "living": "public",
    "porch": "public",
    "parking": "public",
    "dining": "semi_private",
    "kitchen": "semi_private",
    "master_bedroom": "private",
    "bedroom": "private",
    "study": "private",
    "pooja": "private",
    "bathroom": "service",
    "toilet": "service",
    "utility": "service",
    "store": "service",
    "hallway": "circulation",
    "staircase": "circulation",
    "balcony": "public",
    "garage": "public",
}

# Display names
DISPLAY_NAMES = {
    "master_bedroom": "Master Bedroom",
    "bedroom": "Bedroom",
    "bathroom": "Bathroom",
    "toilet": "Toilet",
    "kitchen": "Kitchen",
    "living": "Living Room",
    "dining": "Dining Room",
    "study": "Study Room",
    "pooja": "Pooja Room",
    "store": "Store Room",
    "utility": "Utility Room",
    "hallway": "Hallway",
    "porch": "Porch",
    "parking": "Parking",
    "balcony": "Balcony",
    "staircase": "Staircase",
    "garage": "Garage",
}

# Zoning adjacency rules (forbidden direct openings)
FORBIDDEN_ADJACENCY = [
    ("bedroom", "kitchen"),
    ("master_bedroom", "kitchen"),
    ("bathroom", "living"),
    ("toilet", "living"),
    ("bathroom", "kitchen"),
    ("toilet", "kitchen"),
]

# Mandatory rooms for any plan
MANDATORY_ROOMS = ["living", "kitchen"]

# Required input fields for FORM mode
REQUIRED_FORM_FIELDS = ["bedrooms", "bathrooms", "floors"]


# ===========================================================================
# MODE DETECTION — Auto-detect intelligently
# ===========================================================================

def detect_mode(input_data: Any) -> EngineMode:
    """
    Auto-detect the engine mode from input data.

    Rules:
      - Dict with "rooms" containing position data → VALIDATION
      - Dict with generatePlan=True or text "generate plan" → DESIGN
      - Dict with structured fields (bedrooms, bathrooms, etc.) → FORM
      - String (natural language) → CHAT
      - Dict with "message" field only → CHAT
    """
    if isinstance(input_data, str):
        text_lower = input_data.lower().strip()
        if "generate plan" in text_lower:
            return EngineMode.DESIGN
        return EngineMode.CHAT

    if isinstance(input_data, dict):
        # Check for layout JSON → VALIDATION
        if "rooms" in input_data:
            rooms = input_data["rooms"]
            if isinstance(rooms, list) and len(rooms) > 0:
                first_room = rooms[0] if rooms else {}
                if isinstance(first_room, dict) and "position" in first_room:
                    return EngineMode.VALIDATION

        # Check for generatePlan flag → DESIGN
        if input_data.get("generatePlan") is True or input_data.get("generate_plan") is True:
            return EngineMode.DESIGN

        # Check message text for "generate plan"
        msg = input_data.get("message", "")
        if isinstance(msg, str) and "generate plan" in msg.lower():
            return EngineMode.DESIGN

        # Check for structured form fields (bedrooms, bathrooms, etc.) → FORM
        form_keys = {"bedrooms", "bathrooms", "floors", "plot_width", "plot_length",
                      "total_area", "kitchen", "max_area"}
        if form_keys.intersection(set(input_data.keys())):
            return EngineMode.FORM

        # If only message field → CHAT
        if "message" in input_data:
            return EngineMode.CHAT

    return EngineMode.CHAT


# ===========================================================================
# CHAT MODE — Collect requirements conversationally
# ===========================================================================

def chat_response(message: str, history: List[Dict]) -> Dict:
    """
    Handle CHAT MODE input.

    Collect requirements naturally. Ask follow-up questions.
    Never outputs JSON. Never generates layouts.

    Returns:
      { "reply": str, "mode": "chat", "collected": dict, "ready": bool }
    """
    collected = _parse_requirements_from_history(message, history)

    # Check if user said "generate plan"
    if "generate plan" in message.lower():
        if collected["complete"]:
            return {
                "reply": "Switching to design mode. Generating your architectural plan now.",
                "mode": EngineMode.CHAT,
                "collected": collected,
                "ready": True,
                "switch_to": EngineMode.DESIGN,
            }
        else:
            missing = _get_missing_fields(collected)
            return {
                "reply": f"I need a few more details before generating: {', '.join(missing)}. Could you provide those?",
                "mode": EngineMode.CHAT,
                "collected": collected,
                "ready": False,
            }

    turn = len([h for h in history if h.get("role") == "user"])

    # First turn — greeting and initial parsing
    if turn == 0:
        if collected["has_dimensions"] and collected["has_bedrooms"]:
            # User gave a lot upfront
            summary = _build_summary(collected)
            missing = _get_missing_fields(collected)
            if not missing:
                return {
                    "reply": (
                        f"Excellent! Here's what I've gathered:\n\n{summary}\n\n"
                        f"Everything looks good. Say \"Generate Plan\" when you're ready, "
                        f"or tell me if you'd like to make any changes."
                    ),
                    "mode": EngineMode.CHAT,
                    "collected": collected,
                    "ready": False,
                }
            else:
                return {
                    "reply": (
                        f"Great start! I've noted:\n\n{summary}\n\n"
                        f"I still need: {', '.join(missing)}. Could you provide those?"
                    ),
                    "mode": EngineMode.CHAT,
                    "collected": collected,
                    "ready": False,
                }
        else:
            return {
                "reply": (
                    "Welcome! I'll help you design your residential floor plan.\n\n"
                    "To get started, I need:\n"
                    "  • Plot size (e.g., 30×40 feet or 1200 sq ft)\n"
                    "  • Number of bedrooms\n"
                    "  • Number of bathrooms\n"
                    "  • Number of floors\n\n"
                    "You can provide all at once (e.g., '30x40, 3 bedrooms, 2 bathrooms, 1 floor') "
                    "or one at a time."
                ),
                "mode": EngineMode.CHAT,
                "collected": collected,
                "ready": False,
            }

    # Follow-up turns — ask for what's missing
    missing = _get_missing_fields(collected)

    if not missing:
        summary = _build_summary(collected)
        return {
            "reply": (
                f"All requirements collected:\n\n{summary}\n\n"
                f"Say \"Generate Plan\" to create your architectural layout, "
                f"or tell me about any changes you'd like."
            ),
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    # Ask for next missing field
    if not collected["has_dimensions"]:
        return {
            "reply": "What is your plot size? (e.g., '30×40 feet' or '1200 sq ft')",
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_bedrooms"]:
        return {
            "reply": "How many bedrooms do you need?",
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_bathrooms"]:
        return {
            "reply": "How many bathrooms do you need?",
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_floors"]:
        return {
            "reply": "How many floors? (Most residential homes are 1 or 2 floors.)",
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    # Shouldn't reach here
    return {
        "reply": "I have all mandatory data. Say \"Generate Plan\" when ready.",
        "mode": EngineMode.CHAT,
        "collected": collected,
        "ready": False,
    }


# ===========================================================================
# FORM MODE — Validate structured input
# ===========================================================================

def form_validate(data: Dict) -> Dict:
    """
    Handle FORM MODE input.

    Validates structured JSON/checkbox data. Does NOT ask questions.

    Returns:
      { "valid": bool, "error": str|None, "missing_fields": [], "normalized": dict }
      OR if generatePlan is true, returns design result.
    """
    missing = []

    # Check required fields
    if "bedrooms" not in data or data.get("bedrooms") is None:
        missing.append("bedrooms")
    if "bathrooms" not in data or data.get("bathrooms") is None:
        missing.append("bathrooms")
    if "floors" not in data or data.get("floors") is None:
        missing.append("floors")

    # Need either dimensions or total_area or max_area
    has_area = (
        data.get("total_area") or data.get("max_area") or
        (data.get("plot_width") and data.get("plot_length"))
    )
    if not has_area:
        missing.append("total_area or plot dimensions")

    if missing:
        return {
            "error": "Missing required field",
            "missing_fields": missing,
            "mode": EngineMode.FORM,
        }

    # Normalize data
    normalized = _normalize_form_data(data)

    # Validate values
    errors = _validate_form_values(normalized)
    if errors:
        return {
            "error": "Invalid field values",
            "validation_errors": errors,
            "mode": EngineMode.FORM,
        }

    # If generatePlan flag → switch to DESIGN
    if data.get("generatePlan") is True or data.get("generate_plan") is True:
        return design_generate(normalized)

    return {
        "valid": True,
        "normalized": normalized,
        "mode": EngineMode.FORM,
    }


# ===========================================================================
# DESIGN MODE — Generate full architectural layout
# ===========================================================================

def design_generate(requirements: Dict) -> Dict:
    """
    Handle DESIGN MODE.

    Generates a full architectural layout. No questions asked.
    Applies area distribution, zoning, geometry constraints, circulation,
    and validates all constraints.

    Returns professional explanation (max 8 lines) + strict JSON.
    """
    # Normalize requirements
    req = _normalize_form_data(requirements)

    plot_w = req["plot_width"]
    plot_l = req["plot_length"]
    total_area = req["total_area"]
    floors = req.get("floors", 1)
    bedrooms = req.get("bedrooms", 2)
    bathrooms = req.get("bathrooms", 1)
    extras = req.get("extras", [])

    # Check if area is sufficient
    min_required = _calculate_minimum_area(bedrooms, bathrooms, extras)
    if total_area < min_required:
        return {
            "error": "Insufficient area for requested configuration",
            "suggestion": (
                f"Minimum area needed: {min_required} sq ft. "
                f"Reduce bedroom count or increase plot size to at least {min_required} sq ft."
            ),
            "mode": EngineMode.DESIGN,
        }

    # Generate the layout
    layout = _generate_layout(
        plot_w=plot_w, plot_l=plot_l,
        total_area=total_area, floors=floors,
        bedrooms=bedrooms, bathrooms=bathrooms,
        extras=extras,
    )

    # Validate the layout
    validation = validate_layout(layout)

    # Build explanation (max 8 lines)
    explanation = _build_design_explanation(req, layout, validation)

    return {
        "explanation": explanation,
        "layout": layout,
        "validation": validation,
        "mode": EngineMode.DESIGN,
    }


# ===========================================================================
# VALIDATION MODE — Validate existing layout JSON
# ===========================================================================

def validate_layout(layout: Dict) -> Dict:
    """
    Handle VALIDATION MODE.

    Validates a layout JSON for architectural compliance.
    Returns structured validation report. No conversation text.
    """
    rooms = layout.get("rooms", [])
    plot = layout.get("plot", {})
    plot_w = plot.get("width", 30)
    plot_l = plot.get("length", 40)
    plot_area = plot_w * plot_l

    overlap_issues = _check_overlaps(rooms)
    size_violations = _check_minimum_sizes(rooms)
    zoning_issues = _check_zoning(rooms)
    area_overflow = _check_area_overflow(rooms, plot_area)
    circulation_issues = _check_circulation(rooms, plot_w, plot_l)
    boundary_issues = _check_boundary_fit(rooms, plot_w, plot_l)
    proportion_issues = _check_proportions(rooms)

    total_used = sum(r.get("area", 0) for r in rooms)

    all_ok = (
        not overlap_issues and not size_violations and
        not zoning_issues and not area_overflow and
        not circulation_issues and not boundary_issues and
        not proportion_issues
    )

    return {
        "overlap": bool(overlap_issues),
        "overlap_details": overlap_issues,
        "size_violations": size_violations,
        "zoning_issues": zoning_issues,
        "area_overflow": area_overflow,
        "circulation_issues": circulation_issues,
        "boundary_issues": boundary_issues,
        "proportion_issues": proportion_issues,
        "area_summary": {
            "plot_area": plot_area,
            "total_used_area": round(total_used, 1),
            "utilization_percent": round(total_used / max(plot_area, 1) * 100, 1),
        },
        "compliant": all_ok,
        "mode": EngineMode.VALIDATION,
    }


# ===========================================================================
# UNIFIED PROCESSOR — Single entry point
# ===========================================================================

def process(input_data: Any, history: List[Dict] = None) -> Dict:
    """
    Unified entry point for the architectural planning engine.

    Auto-detects mode and routes accordingly.
    Never crashes. Always returns valid JSON-serializable dict.
    """
    if history is None:
        history = []

    try:
        mode = detect_mode(input_data)

        if mode == EngineMode.CHAT:
            message = input_data if isinstance(input_data, str) else input_data.get("message", "")
            result = chat_response(message, history)

            # If chat says switch to design
            if result.get("switch_to") == EngineMode.DESIGN:
                collected = result.get("collected", {})
                design_input = _collected_to_requirements(collected)
                design_result = design_generate(design_input)
                design_result["chat_summary"] = result["reply"]
                return design_result

            return result

        elif mode == EngineMode.FORM:
            return form_validate(input_data)

        elif mode == EngineMode.DESIGN:
            if isinstance(input_data, str):
                return {
                    "error": "Design mode requires structured requirements data.",
                    "suggestion": "Provide plot dimensions, bedrooms, bathrooms, and floors.",
                    "mode": EngineMode.DESIGN,
                }
            return design_generate(input_data)

        elif mode == EngineMode.VALIDATION:
            return validate_layout(input_data)

        else:
            return {"error": "Unknown mode", "mode": "error"}

    except Exception as e:
        return {
            "error": f"Engine error: {str(e)}",
            "mode": "error",
        }


# ===========================================================================
# INTERNAL — Requirement Parsing
# ===========================================================================

def _parse_requirements_from_history(current_msg: str, history: List[Dict]) -> Dict:
    """Parse all collected data from conversation history + current message."""
    all_user_text = " ".join(
        m.get("content", "") for m in history if m.get("role") == "user"
    )
    all_user_text += " " + current_msg
    text_lower = all_user_text.lower()

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
        "complete": False,
    }

    # Dimensions: 30x40, 30*40, 30 x 40, 30×40
    dim_match = re.search(r'(\d+)\s*[x×*]\s*(\d+)', text_lower)
    if dim_match:
        data["has_dimensions"] = True
        data["plot_width"] = int(dim_match.group(1))
        data["plot_length"] = int(dim_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]

    # Area: 1200 sqft, 1200 sq ft, 1200 square feet
    area_match = re.search(r'(\d+)\s*(?:sq\s*ft|sqft|square\s*feet?)', text_lower)
    if area_match:
        data["has_dimensions"] = True
        data["total_area"] = int(area_match.group(1))

    # Standalone number > 100 (likely area) when no other dimensions known
    if not data["has_dimensions"]:
        all_messages = [m.get("content", "") for m in history if m.get("role") == "user"]
        all_messages.append(current_msg)
        for msg_text in all_messages:
            num_match = re.match(r'^\s*(\d{3,5})\s*$', msg_text.strip())
            if num_match:
                val = int(num_match.group(1))
                if 100 <= val <= 50000:
                    data["has_dimensions"] = True
                    data["total_area"] = val

    # BHK: 3BHK, 3 bhk → bedrooms and bathrooms
    bhk_match = re.search(r'(\d+)\s*bhk', text_lower)
    if bhk_match:
        bhk = int(bhk_match.group(1))
        data["has_bedrooms"] = True
        data["has_bathrooms"] = True
        data["bedrooms"] = bhk
        data["bathrooms"] = max(1, bhk - 1)

    # Explicit bedrooms
    bed_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?)', text_lower)
    if bed_match:
        data["has_bedrooms"] = True
        data["bedrooms"] = int(bed_match.group(1))

    # Explicit bathrooms
    bath_match = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilet)', text_lower)
    if bath_match:
        data["has_bathrooms"] = True
        data["bathrooms"] = int(bath_match.group(1))

    # Floors
    floor_match = re.search(r'(\d+)\s*(?:floor|storey|story|level)', text_lower)
    if floor_match:
        data["has_floors"] = True
        data["floors"] = int(floor_match.group(1))

    # Contextual parsing — look at assistant question → user answer pairs
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        user_text = msg.get("content", "").strip()
        prev_assistant = ""
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "assistant":
                prev_assistant = history[j].get("content", "").lower()
                break

        # Number after bedrooms question
        if "bedroom" in prev_assistant and not data["has_bedrooms"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_bedrooms"] = True
                data["bedrooms"] = int(nums[0])

        # Number after bathrooms question
        if "bathroom" in prev_assistant and not data["has_bathrooms"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_bathrooms"] = True
                data["bathrooms"] = int(nums[0])

        # Number after floors question
        if "floor" in prev_assistant and not data["has_floors"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_floors"] = True
                data["floors"] = int(nums[0])

        # Dimensions after plot size question
        if "plot" in prev_assistant and not data["has_dimensions"]:
            dim_ctx = re.search(r'(\d+)\s*[x×*]\s*(\d+)', user_text.lower())
            if dim_ctx:
                data["has_dimensions"] = True
                data["plot_width"] = int(dim_ctx.group(1))
                data["plot_length"] = int(dim_ctx.group(2))
                data["total_area"] = data["plot_width"] * data["plot_length"]
            else:
                nums = re.findall(r'\d+', user_text)
                if nums:
                    val = int(nums[0])
                    if val >= 100:
                        data["has_dimensions"] = True
                        data["total_area"] = val

    # Also parse current message contextually
    if history:
        last_assistant = ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "").lower()
                break

        msg_nums = re.findall(r'\d+', current_msg)
        if msg_nums:
            if "bedroom" in last_assistant and not data["has_bedrooms"]:
                data["has_bedrooms"] = True
                data["bedrooms"] = int(msg_nums[0])
            elif "bathroom" in last_assistant and not data["has_bathrooms"]:
                data["has_bathrooms"] = True
                data["bathrooms"] = int(msg_nums[0])
            elif "floor" in last_assistant and not data["has_floors"]:
                data["has_floors"] = True
                data["floors"] = int(msg_nums[0])
            elif ("plot" in last_assistant or "area" in last_assistant) and not data["has_dimensions"]:
                val = int(msg_nums[0])
                if val >= 100:
                    data["has_dimensions"] = True
                    data["total_area"] = val

    # Extras
    extras = []
    if "dining" in text_lower:
        extras.append("dining")
    if "study" in text_lower:
        extras.append("study")
    if "pooja" in text_lower:
        extras.append("pooja")
    if "balcon" in text_lower:
        extras.append("balcony")
    if "parking" in text_lower or "garage" in text_lower:
        extras.append("parking")
    if "garden" in text_lower:
        extras.append("garden")
    if "store" in text_lower:
        extras.append("store")
    data["extras"] = extras

    # Default floors to 1 if not provided (common assumption)
    if not data["has_floors"]:
        data["floors"] = 1
        data["has_floors"] = True

    # Check completeness
    data["complete"] = (
        data["has_dimensions"] and data["has_bedrooms"] and
        data["has_bathrooms"] and data["has_floors"]
    )

    return data


def _get_missing_fields(collected: Dict) -> List[str]:
    """Return list of missing required fields."""
    missing = []
    if not collected.get("has_dimensions"):
        missing.append("plot size (width × length or total area)")
    if not collected.get("has_bedrooms"):
        missing.append("number of bedrooms")
    if not collected.get("has_bathrooms"):
        missing.append("number of bathrooms")
    if not collected.get("has_floors"):
        missing.append("number of floors")
    return missing


def _build_summary(collected: Dict) -> str:
    """Build a human-readable summary of collected requirements."""
    parts = []
    if collected.get("plot_width") and collected.get("plot_length"):
        parts.append(f"  Plot: {collected['plot_width']} × {collected['plot_length']} ft ({collected.get('total_area', '?')} sq ft)")
    elif collected.get("total_area"):
        parts.append(f"  Plot area: {collected['total_area']} sq ft")
    if collected.get("bedrooms"):
        parts.append(f"  Bedrooms: {collected['bedrooms']}")
    if collected.get("bathrooms"):
        parts.append(f"  Bathrooms: {collected['bathrooms']}")
    if collected.get("floors"):
        parts.append(f"  Floors: {collected['floors']}")
    if collected.get("extras"):
        parts.append(f"  Additional: {', '.join(collected['extras'])}")
    return "\n".join(parts)


def _collected_to_requirements(collected: Dict) -> Dict:
    """Convert parsed collected data to normalized requirements dict."""
    return {
        "plot_width": collected.get("plot_width"),
        "plot_length": collected.get("plot_length"),
        "total_area": collected.get("total_area"),
        "floors": collected.get("floors", 1),
        "bedrooms": collected.get("bedrooms", 2),
        "bathrooms": collected.get("bathrooms", 1),
        "extras": collected.get("extras", []),
    }


# ===========================================================================
# INTERNAL — Form Normalization & Validation
# ===========================================================================

def _normalize_form_data(data: Dict) -> Dict:
    """Normalize form input into a consistent requirements dict."""
    result = {
        "floors": int(data.get("floors", 1)),
        "bedrooms": int(data.get("bedrooms", 2)),
        "bathrooms": int(data.get("bathrooms", 1)),
        "extras": data.get("extras", []),
    }

    # Plot dimensions
    pw = data.get("plot_width")
    pl = data.get("plot_length")
    ta = data.get("total_area") or data.get("max_area")

    if pw and pl:
        result["plot_width"] = float(pw)
        result["plot_length"] = float(pl)
        result["total_area"] = float(pw) * float(pl)
    elif ta:
        ta = float(ta)
        result["total_area"] = ta
        # Assume 1:1.3 ratio
        side = math.sqrt(ta)
        result["plot_width"] = round(side * 1.14, 1)
        result["plot_length"] = round(ta / (side * 1.14), 1)
        result["assumed_dimensions"] = True
    else:
        result["plot_width"] = 30
        result["plot_length"] = 40
        result["total_area"] = 1200
        result["assumed_dimensions"] = True

    # Handle extras from soft constraints
    extras = list(result["extras"])
    if data.get("balcony"):
        extras.append("balcony")
    if data.get("parking"):
        extras.append("parking")
    if data.get("pooja_room") or data.get("pooja"):
        extras.append("pooja")
    if data.get("dining"):
        extras.append("dining")
    if data.get("study"):
        extras.append("study")
    if data.get("store"):
        extras.append("store")
    result["extras"] = list(set(extras))

    return result


def _validate_form_values(data: Dict) -> List[str]:
    """Validate form values for sanity."""
    errors = []
    if data.get("bedrooms", 0) < 1:
        errors.append("At least 1 bedroom is required")
    if data.get("bedrooms", 0) > 10:
        errors.append("Maximum 10 bedrooms supported")
    if data.get("bathrooms", 0) < 1:
        errors.append("At least 1 bathroom is required")
    if data.get("floors", 0) < 1:
        errors.append("At least 1 floor is required")
    if data.get("floors", 0) > 4:
        errors.append("Maximum 4 floors supported")
    ta = data.get("total_area", 0)
    if ta < 200:
        errors.append("Minimum plot area is 200 sq ft")
    if ta > 50000:
        errors.append("Maximum plot area is 50,000 sq ft")
    return errors


# ===========================================================================
# INTERNAL — Minimum Area Calculator
# ===========================================================================

def _calculate_minimum_area(bedrooms: int, bathrooms: int, extras: List[str]) -> float:
    """Calculate minimum total area required for the given room configuration."""
    area = 0
    # Living room
    area += MIN_ROOM_SIZES["living"]
    # Kitchen
    area += MIN_ROOM_SIZES["kitchen"]
    # Bedrooms (first one is master)
    if bedrooms >= 1:
        area += MIN_ROOM_SIZES["master_bedroom"]
    for _ in range(max(0, bedrooms - 1)):
        area += MIN_ROOM_SIZES["bedroom"]
    # Bathrooms
    for _ in range(bathrooms):
        area += MIN_ROOM_SIZES["bathroom"]
    # Extras
    for extra in extras:
        area += MIN_ROOM_SIZES.get(extra, 30)
    # Circulation (10%)
    area *= 1.15
    # Walls (8%)
    area *= 1.08
    return round(area)


# ===========================================================================
# INTERNAL — Layout Generation Engine
# ===========================================================================

def _generate_layout(
    plot_w: float, plot_l: float, total_area: float,
    floors: int, bedrooms: int, bathrooms: int,
    extras: List[str],
) -> Dict:
    """
    Generate a complete architectural layout.

    Uses a deterministic placement algorithm:
      1. Calculate usable area (after external walls)
      2. Allocate area proportionally
      3. Place rooms by zone priority
      4. Add circulation corridor
      5. Assign doors and windows
    """
    # Usable interior dimensions (after external walls)
    usable_w = plot_w - 2 * WALL_EXTERNAL_FT
    usable_l = plot_l - 2 * WALL_EXTERNAL_FT
    usable_area = usable_w * usable_l

    # Build room list with target areas
    room_specs = _allocate_room_areas(
        usable_area=usable_area,
        bedrooms=bedrooms, bathrooms=bathrooms,
        extras=extras,
    )

    # Place rooms using grid-based placement
    placed_rooms = _place_rooms(
        room_specs=room_specs,
        plot_w=plot_w, plot_l=plot_l,
        usable_w=usable_w, usable_l=usable_l,
    )

    # Assign doors and windows
    placed_rooms = _assign_doors_windows(placed_rooms, plot_w, plot_l)

    # Calculate totals
    total_used = sum(r["area"] for r in placed_rooms)
    circulation_area = usable_area - total_used
    circulation_pct = round(circulation_area / max(total_area, 1) * 100, 1)

    # ── Build PlanPreview-compatible boundary + room polygons ──
    boundary = [
        [0, 0], [plot_w, 0], [plot_w, plot_l], [0, plot_l], [0, 0]
    ]

    doors_list = []
    for room in placed_rooms:
        rx = room["position"]["x"]
        ry = room["position"]["y"]
        rw = room["width"]
        rl = room["length"]
        # Polygon: 5-point closed rectangle
        room["polygon"] = [
            [rx, ry], [rx + rw, ry], [rx + rw, ry + rl], [rx, ry + rl], [rx, ry]
        ]
        room["centroid"] = [round(rx + rw / 2, 1), round(ry + rl / 2, 1)]
        # Room label for preview
        room["label"] = room.get("name", room.get("room_type", "Room"))
        # PlanPreview expects actual_area
        room["actual_area"] = round(room.get("area", rw * rl), 1)

        # Build doors for PlanPreview (quarter-arc door icons)
        for door in room.get("doors", []):
            wall = door.get("wall", "S")
            dw = door.get("width", 2.5)
            if wall in ("S", "bottom"):
                hx, hy = round(rx + rw * 0.3, 1), ry
                doors_list.append({
                    "position": [hx, hy],
                    "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [round(hx + dw, 1), hy],
                    "swing_dir": [0, 1],
                })
            elif wall in ("N", "top"):
                hx, hy = round(rx + rw * 0.3, 1), round(ry + rl, 1)
                doors_list.append({
                    "position": [hx, hy],
                    "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [round(hx + dw, 1), hy],
                    "swing_dir": [0, -1],
                })
            elif wall in ("W", "left"):
                hx, hy = rx, round(ry + rl * 0.3, 1)
                doors_list.append({
                    "position": [hx, hy],
                    "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [hx, round(hy + dw, 1)],
                    "swing_dir": [1, 0],
                })
            elif wall in ("E", "right"):
                hx, hy = round(rx + rw, 1), round(ry + rl * 0.3, 1)
                doors_list.append({
                    "position": [hx, hy],
                    "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [hx, round(hy + dw, 1)],
                    "swing_dir": [-1, 0],
                })

    return {
        "boundary": boundary,
        "rooms": placed_rooms,
        "doors": doors_list,
        "total_area": round(total_area, 1),
        "plot": {
            "width": plot_w,
            "length": plot_l,
            "unit": "ft",
        },
        "floors": floors,
        "circulation": {
            "type": "central" if plot_w >= 25 else "side" if plot_w >= 15 else "minimal",
            "width": max(MIN_PASSAGE_WIDTH_FT, 3.5),
        },
        "walls": {
            "external": "9 inch",
            "internal": "4.5 inch",
        },
        "area_summary": {
            "plot_area": round(total_area, 1),
            "total_used_area": round(total_used, 1),
            "circulation_area": round(max(0, circulation_area), 1),
            "circulation_percentage": f"{max(0, circulation_pct)}%",
            "utilization_percentage": f"{round(total_used / max(total_area, 1) * 100, 1)}%",
        },
        "validation": {
            "overlap": False,
            "zoning_ok": True,
            "min_size_ok": True,
            "area_ok": total_used <= total_area,
        },
    }


def _allocate_room_areas(
    usable_area: float, bedrooms: int, bathrooms: int, extras: List[str]
) -> List[Dict]:
    """Allocate target areas proportionally based on architectural guidelines."""
    # Reserve 12% for circulation and 9% for walls
    available = usable_area * 0.79  # 100% - 12% circ - 9% walls

    rooms = []

    # Living room: 18-22% of plot
    living_target = usable_area * 0.20
    living_w, living_l = _scale_room("living", living_target)
    rooms.append({
        "name": "Living Room", "room_type": "living",
        "target_area": round(living_w * living_l),
        "width": living_w, "length": living_l,
    })

    # Kitchen: 8-12%
    kitchen_target = usable_area * 0.10
    kit_w, kit_l = _scale_room("kitchen", kitchen_target)
    rooms.append({
        "name": "Kitchen", "room_type": "kitchen",
        "target_area": round(kit_w * kit_l),
        "width": kit_w, "length": kit_l,
    })

    # Bedrooms: 30-35% total
    bedroom_share = usable_area * 0.32 / max(bedrooms, 1)
    for i in range(bedrooms):
        rtype = "master_bedroom" if i == 0 else "bedroom"
        name = "Master Bedroom" if i == 0 else f"Bedroom {i + 1}"
        # Master gets 10% more
        target = bedroom_share * (1.1 if i == 0 else 1.0)
        bw, bl = _scale_room(rtype, target)
        rooms.append({
            "name": name, "room_type": rtype,
            "target_area": round(bw * bl),
            "width": bw, "length": bl,
        })

    # Bathrooms: 8-12% total
    bathroom_share = usable_area * 0.10 / max(bathrooms, 1)
    for i in range(bathrooms):
        name = f"Bathroom {i + 1}" if bathrooms > 1 else "Bathroom"
        bw, bl = _scale_room("bathroom", bathroom_share)
        rooms.append({
            "name": name, "room_type": "bathroom",
            "target_area": round(bw * bl),
            "width": bw, "length": bl,
        })

    # Extras
    for extra in extras:
        if extra in STANDARD_ROOM_SIZES:
            extra_target = usable_area * 0.06
            ew, el = _scale_room(extra, extra_target)
            rooms.append({
                "name": DISPLAY_NAMES.get(extra, extra.title()),
                "room_type": extra,
                "target_area": round(ew * el),
                "width": ew, "length": el,
            })

    return rooms


def _scale_room(room_type: str, target_area: float) -> Tuple[float, float]:
    """Scale a room to approximate target area while maintaining reasonable proportions."""
    std_w, std_l = STANDARD_ROOM_SIZES.get(room_type, (10, 10))
    std_area = std_w * std_l
    min_area = MIN_ROOM_SIZES.get(room_type, 25)

    # Ensure minimum
    target_area = max(target_area, min_area)

    # Scale factor
    scale = math.sqrt(target_area / max(std_area, 1))
    new_w = round(std_w * scale, 1)
    new_l = round(std_l * scale, 1)

    # Enforce minimum dimensions
    new_w = max(new_w, 4.0)
    new_l = max(new_l, 4.0)

    # Enforce max aspect ratio 3:1
    if max(new_w, new_l) / max(min(new_w, new_l), 1) > 3:
        if new_w > new_l:
            new_w = new_l * 2.5
        else:
            new_l = new_w * 2.5

    return round(new_w, 1), round(new_l, 1)


def _place_rooms(
    room_specs: List[Dict],
    plot_w: float, plot_l: float,
    usable_w: float, usable_l: float,
) -> List[Dict]:
    """
    Wall-to-wall architectural room placement.

    Rooms tile the ENTIRE usable area with zero gaps. Follows Indian
    residential conventions:

        ┌──────────┬──────────┬──────────┐
        │ Living   │ Kitchen  │ Dining   │  ← public / semi-private
        ├──────────┴──────────┴──────────┤
        │           CORRIDOR             │  ← 3.5 ft passage
        ├────────┬─────┬─────────┬───────┤
        │ Master │Bath │ Bedroom │ Study  │  ← private / service
        └────────┴─────┴─────────┴───────┘

    Rules applied:
      • Every row fills the FULL usable width — no side gaps.
      • Room widths within a row are proportional to target area.
      • The last room in each row absorbs rounding residual.
      • The last row extends to the plot's back boundary.
      • Kitchen stays adjacent to Dining (cooking-serving flow).
      • Bedrooms interleave with Bathrooms (attached-bath pattern).
      • Corridor inserted between public and private zone groups.
    """
    placed = []
    WALL = WALL_EXTERNAL_FT
    IWALL = WALL_INTERNAL_FT
    CORRIDOR_H = 3.5

    def zone_of(r):
        return ZONE_MAP.get(r["room_type"], "private")

    # ── 1. Classify rooms into zone groups ──────────────────────────────
    public_rooms = []   # public + semi_private (Living, Kitchen, Dining …)
    private_rooms = []  # private + service (Bedrooms, Bathrooms, Study …)

    for r in room_specs:
        z = zone_of(r)
        if z in ("public", "semi_private"):
            public_rooms.append(r)
        else:
            private_rooms.append(r)

    # ── 2. Smart ordering within each group ─────────────────────────────
    # Public: Living first (entrance), then Kitchen → Dining (adjacent!)
    pub_order = {
        "living": 0, "foyer": 1, "kitchen": 2, "dining": 3,
        "balcony": 4, "garage": 5, "porch": 6,
    }
    public_rooms.sort(key=lambda r: pub_order.get(r["room_type"], 99))

    # Private: interleave Bed-Bath pairs, extras at end
    beds = sorted(
        [r for r in private_rooms if r["room_type"] in ("master_bedroom", "bedroom")],
        key=lambda r: r["target_area"], reverse=True,
    )
    baths = sorted(
        [r for r in private_rooms if r["room_type"] in ("bathroom", "toilet")],
        key=lambda r: r["target_area"], reverse=True,
    )
    extras = [r for r in private_rooms if r not in beds and r not in baths]

    interleaved: List[Dict] = []
    for i in range(max(len(beds), len(baths))):
        if i < len(beds):
            interleaved.append(beds[i])
        if i < len(baths):
            interleaved.append(baths[i])
    interleaved.extend(extras)
    private_rooms = interleaved

    # ── 3. Build rows (split groups with >4 rooms) ─────────────────────
    MAX_PER_ROW = 4

    def _split_rows(rooms: List[Dict]) -> List[List[Dict]]:
        if not rooms:
            return []
        if len(rooms) <= MAX_PER_ROW:
            return [rooms]
        n_rows = math.ceil(len(rooms) / MAX_PER_ROW)
        per_row = math.ceil(len(rooms) / n_rows)
        return [rooms[i:i + per_row] for i in range(0, len(rooms), per_row)]

    public_rows = _split_rows(public_rooms)
    private_rows = _split_rows(private_rooms)

    zone_groups = []  # list of (zone_label, [rows])
    if public_rows:
        zone_groups.append(("public", public_rows))
    if private_rows:
        zone_groups.append(("private", private_rows))
    if not zone_groups:
        return placed

    # ── 4. Compute row heights ──────────────────────────────────────────
    n_corridors = max(0, len(zone_groups) - 1)
    avail_h = usable_l - n_corridors * CORRIDOR_H

    all_rows: List[List[Dict]] = []
    for _, rows in zone_groups:
        all_rows.extend(rows)

    row_areas = [sum(r["target_area"] for r in row) for row in all_rows]
    total_area = sum(row_areas) or 1.0

    # Proportional heights, minimum 8 ft per row
    MIN_ROW_H = 8.0
    heights = [max(MIN_ROW_H, (a / total_area) * avail_h) for a in row_areas]

    # Re-scale so total == avail_h
    ht_sum = sum(heights)
    if abs(ht_sum - avail_h) > 0.05:
        factor = avail_h / max(ht_sum, 1.0)
        heights = [h * factor for h in heights]

    # ── 5. Place rooms row by row (wall-to-wall) ───────────────────────
    current_y = WALL
    row_idx = 0

    for gi, (zone_label, rows) in enumerate(zone_groups):
        for ri, row in enumerate(rows):
            is_last_row = (gi == len(zone_groups) - 1 and ri == len(rows) - 1)
            row_h = round(WALL + usable_l - current_y, 1) if is_last_row \
                else round(heights[row_idx], 1)
            row_h = max(row_h, MIN_ROW_H)
            row_idx += 1

            n = len(row)
            areas = [r["target_area"] for r in row]
            total_row_area = sum(areas) or 1.0
            total_gaps = IWALL * max(0, n - 1)
            net_w = usable_w - total_gaps

            current_x = WALL
            for i, room in enumerate(row):
                # Last room in row absorbs rounding residual
                if i == n - 1:
                    rw = round(WALL + usable_w - current_x, 1)
                else:
                    rw = round(net_w * (areas[i] / total_row_area), 1)
                rw = max(rw, 4.0)
                rl = row_h

                placed.append({
                    "name": room["name"],
                    "room_type": room["room_type"],
                    "zone": zone_of(room),
                    "width": round(rw, 1),
                    "length": round(rl, 1),
                    "area": round(rw * rl, 1),
                    "position": {"x": round(current_x, 1), "y": round(current_y, 1)},
                    "doors": [],
                    "windows": [],
                })
                current_x += rw + IWALL

            current_y += row_h

        # Corridor gap between zone groups
        if gi < len(zone_groups) - 1:
            current_y += CORRIDOR_H

    return placed


def _assign_doors_windows(rooms: List[Dict], plot_w: float, plot_l: float) -> List[Dict]:
    """
    Assign doors and windows based on room position & zoning.

    Door logic:
      • Public-zone rooms  → door on N wall (toward corridor / interior)
      • Private-zone rooms → door on S wall (toward corridor / interior)
      • Bathrooms prefer door toward adjacent bedroom (shared E/W wall)
      • Living room gets an additional entrance on S wall (front boundary)

    Window logic:
      • Windows on every external wall (touching plot boundary)
      • Habitable rooms guaranteed at least one window
      • Bathrooms get small (2 ft) windows only
    """
    # Detect corridor Y-midpoint to decide which wall faces "inward"
    public_bottom = 0.0
    private_top = plot_l
    for room in rooms:
        z = room.get("zone", "private")
        ry = room["position"]["y"]
        rl = room["length"]
        if z in ("public", "semi_private"):
            public_bottom = max(public_bottom, ry + rl)
        elif z in ("private", "service"):
            private_top = min(private_top, ry)
    corridor_mid = (public_bottom + private_top) / 2

    for room in rooms:
        pos = room["position"]
        rw = room["width"]
        rl = room["length"]
        zone = room.get("zone", "private")
        rtype = room["room_type"]
        rx, ry = pos["x"], pos["y"]

        doors: List[Dict] = []
        windows: List[Dict] = []

        # ── Door placement ──────────────────────────────────────────────
        room_center_y = ry + rl / 2.0
        faces_corridor_on_bottom = room_center_y < corridor_mid  # public row
        faces_corridor_on_top = room_center_y > corridor_mid     # private row

        if rtype in ("bathroom", "toilet"):
            # Bathroom door toward adjacent bedroom (shared wall)
            door_placed = False
            for other in rooms:
                if other["room_type"] not in ("master_bedroom", "bedroom"):
                    continue
                ox = other["position"]["x"]
                ow = other["width"]
                oy = other["position"]["y"]
                ol = other["length"]
                # Bedroom to the left?
                if abs((ox + ow) - rx) < 1.0 and abs(oy - ry) < 1.0:
                    doors.append({"wall": "W", "width": 2.5})
                    door_placed = True
                    break
                # Bedroom to the right?
                if abs((rx + rw) - ox) < 1.0 and abs(oy - ry) < 1.0:
                    doors.append({"wall": "E", "width": 2.5})
                    door_placed = True
                    break
            if not door_placed:
                # No adjacent bedroom — door toward corridor
                if faces_corridor_on_bottom:
                    doors.append({"wall": "N", "width": 2.5})
                else:
                    doors.append({"wall": "S", "width": 2.5})
        elif zone in ("public", "semi_private"):
            # Door toward corridor (N wall = bottom edge of room)
            doors.append({"wall": "N", "width": 3})
            # Living room also gets the main entrance on S wall (front)
            if rtype == "living":
                doors.append({"wall": "S", "width": 3})
        else:
            # Private rooms — door toward corridor (S wall = top edge)
            if faces_corridor_on_top:
                doors.append({"wall": "S", "width": 3})
            else:
                doors.append({"wall": "N", "width": 3})

        # ── Window placement — external walls only ──────────────────────
        is_south = ry <= WALL_EXTERNAL_FT + 1
        is_north = ry + rl >= plot_l - WALL_EXTERNAL_FT - 1
        is_west = rx <= WALL_EXTERNAL_FT + 1
        is_east = rx + rw >= plot_w - WALL_EXTERNAL_FT - 1

        if rtype in ("bathroom", "toilet"):
            # Small window on one external wall only
            if is_east:
                windows.append({"wall": "E", "width": 2})
            elif is_north:
                windows.append({"wall": "N", "width": 2})
            elif is_west:
                windows.append({"wall": "W", "width": 2})
            elif is_south:
                windows.append({"wall": "S", "width": 2})
        else:
            if is_north:
                windows.append({"wall": "N", "width": 4})
            if is_south:
                windows.append({"wall": "S", "width": 4})
            if is_east:
                windows.append({"wall": "E", "width": 4})
            if is_west:
                windows.append({"wall": "W", "width": 4})

        # Habitable rooms must have at least one window
        if not windows and rtype not in (
            "bathroom", "toilet", "store", "utility", "hallway", "staircase",
        ):
            # Pick the external-most wall
            if is_north:
                windows.append({"wall": "N", "width": 4})
            elif is_south:
                windows.append({"wall": "S", "width": 4})
            else:
                windows.append({"wall": "N", "width": 4})

        room["doors"] = doors
        room["windows"] = windows

    return rooms


# ===========================================================================
# INTERNAL — Validation Checks
# ===========================================================================

def _check_overlaps(rooms: List[Dict]) -> List[str]:
    """Check for room overlaps using AABB collision detection."""
    issues = []
    for i, r1 in enumerate(rooms):
        p1 = r1.get("position", {})
        x1, y1 = p1.get("x", 0), p1.get("y", 0)
        w1, l1 = r1.get("width", 0), r1.get("length", 0)

        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p2 = r2.get("position", {})
            x2, y2 = p2.get("x", 0), p2.get("y", 0)
            w2, l2 = r2.get("width", 0), r2.get("length", 0)

            # AABB overlap with small tolerance for shared walls
            tolerance = WALL_INTERNAL_FT * 0.5
            if (x1 < x2 + w2 - tolerance and x1 + w1 > x2 + tolerance and
                y1 < y2 + l2 - tolerance and y1 + l1 > y2 + tolerance):
                issues.append(
                    f"Overlap: {r1.get('name', '?')} and {r2.get('name', '?')}"
                )
    return issues


def _check_minimum_sizes(rooms: List[Dict]) -> List[str]:
    """Check rooms meet minimum size requirements."""
    issues = []
    for room in rooms:
        rtype = room.get("room_type", "other")
        area = room.get("area", 0)
        min_a = MIN_ROOM_SIZES.get(rtype, 0)
        if area < min_a:
            issues.append(
                f"{room.get('name', rtype)}: {area} sq ft < minimum {min_a} sq ft"
            )
    return issues


def _check_zoning(rooms: List[Dict]) -> List[str]:
    """Check zoning rules — forbidden direct openings."""
    issues = []
    # Build adjacency map (rooms that share a wall)
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            if _are_adjacent(r1, r2):
                t1 = r1.get("room_type", "other")
                t2 = r2.get("room_type", "other")
                for forbidden_a, forbidden_b in FORBIDDEN_ADJACENCY:
                    if (t1 == forbidden_a and t2 == forbidden_b) or \
                       (t1 == forbidden_b and t2 == forbidden_a):
                        # Check if they have doors facing each other
                        issues.append(
                            f"Zoning: {r1.get('name')} should not open directly into {r2.get('name')}"
                        )
    return issues


def _are_adjacent(r1: Dict, r2: Dict) -> bool:
    """Check if two rooms share a wall (are adjacent)."""
    p1 = r1.get("position", {})
    p2 = r2.get("position", {})
    x1, y1 = p1.get("x", 0), p1.get("y", 0)
    x2, y2 = p2.get("x", 0), p2.get("y", 0)
    w1, l1 = r1.get("width", 0), r1.get("length", 0)
    w2, l2 = r2.get("width", 0), r2.get("length", 0)

    tolerance = WALL_INTERNAL_FT + 0.5

    # Check shared vertical wall (right side of r1 ≈ left side of r2)
    if abs((x1 + w1) - x2) < tolerance or abs((x2 + w2) - x1) < tolerance:
        # Check vertical overlap
        if y1 < y2 + l2 and y1 + l1 > y2:
            return True

    # Check shared horizontal wall
    if abs((y1 + l1) - y2) < tolerance or abs((y2 + l2) - y1) < tolerance:
        # Check horizontal overlap
        if x1 < x2 + w2 and x1 + w1 > x2:
            return True

    return False


def _check_area_overflow(rooms: List[Dict], plot_area: float) -> str:
    """Check if total room area exceeds plot area."""
    total = sum(r.get("area", 0) for r in rooms)
    if total > plot_area:
        return f"Total room area ({round(total)} sq ft) exceeds plot area ({round(plot_area)} sq ft)"
    return ""


def _check_circulation(rooms: List[Dict], plot_w: float, plot_l: float) -> List[str]:
    """Check circulation and accessibility."""
    issues = []
    total_room_area = sum(r.get("area", 0) for r in rooms)
    plot_area = plot_w * plot_l
    circulation_area = plot_area - total_room_area

    if circulation_area < plot_area * 0.08:
        issues.append(
            f"Insufficient circulation space: {round(circulation_area)} sq ft "
            f"({round(circulation_area / max(plot_area, 1) * 100, 1)}%) — minimum 10% recommended"
        )

    # Check passage widths between adjacent rooms
    # Only flag gaps between rooms that actually share an edge (overlap in perpendicular axis)
    wall_thickness_tolerance = WALL_INTERNAL_FT + 0.1  # ~0.475 ft — anything this small is a wall, not a passage

    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p1, p2 = r1.get("position", {}), r2.get("position", {})
            x1, y1 = p1.get("x", 0), p1.get("y", 0)
            x2, y2 = p2.get("x", 0), p2.get("y", 0)
            w1, l1 = r1.get("width", 0), r1.get("length", 0)
            w2, l2 = r2.get("width", 0), r2.get("length", 0)

            # Check horizontal gap (rooms side by side) — must overlap in Y
            gap_x = x2 - (x1 + w1)
            y_overlap = min(y1 + l1, y2 + l2) - max(y1, y2)
            if y_overlap > 0.5 and wall_thickness_tolerance < gap_x < MIN_PASSAGE_WIDTH_FT:
                issues.append(
                    f"Narrow passage ({round(gap_x, 1)} ft) between "
                    f"{r1.get('name')} and {r2.get('name')} — minimum {MIN_PASSAGE_WIDTH_FT} ft required"
                )

            # Check vertical gap (rooms above/below) — must overlap in X
            gap_y = y2 - (y1 + l1)
            x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            if x_overlap > 0.5 and wall_thickness_tolerance < gap_y < MIN_PASSAGE_WIDTH_FT:
                issues.append(
                    f"Narrow passage ({round(gap_y, 1)} ft) between "
                    f"{r1.get('name')} and {r2.get('name')} — minimum {MIN_PASSAGE_WIDTH_FT} ft required"
                )

    return issues


def _check_boundary_fit(rooms: List[Dict], plot_w: float, plot_l: float) -> List[str]:
    """Check that all rooms fit within plot boundaries."""
    issues = []
    for room in rooms:
        pos = room.get("position", {})
        rx = pos.get("x", 0)
        ry = pos.get("y", 0)
        rw = room.get("width", 0)
        rl = room.get("length", 0)

        if rx < 0 or ry < 0:
            issues.append(f"{room.get('name')}: position ({rx}, {ry}) is outside plot boundary")
        if rx + rw > plot_w + 0.1:  # small tolerance
            issues.append(
                f"{room.get('name')}: extends beyond plot width "
                f"(room ends at {round(rx + rw, 1)} ft, plot width is {plot_w} ft)"
            )
        if ry + rl > plot_l + 0.1:
            issues.append(
                f"{room.get('name')}: extends beyond plot length "
                f"(room ends at {round(ry + rl, 1)} ft, plot length is {plot_l} ft)"
            )
    return issues


def _check_proportions(rooms: List[Dict]) -> List[str]:
    """Check room proportions are realistic."""
    issues = []
    for room in rooms:
        w = room.get("width", 10)
        l = room.get("length", 10)
        min_dim = min(w, l)
        max_dim = max(w, l)

        if min_dim < 4 and room.get("room_type") not in ("balcony", "utility", "toilet", "hallway"):
            issues.append(
                f"{room.get('name')}: minimum dimension {min_dim} ft is too narrow (min 4 ft)"
            )
        if max_dim / max(min_dim, 0.1) > 3:
            issues.append(
                f"{room.get('name')}: aspect ratio {round(max_dim / max(min_dim, 0.1), 1)}:1 "
                f"exceeds maximum 3:1"
            )
    return issues


# ===========================================================================
# INTERNAL — Design Explanation Builder
# ===========================================================================

def _build_design_explanation(req: Dict, layout: Dict, validation: Dict) -> str:
    """Build a short professional explanation (max 8 lines)."""
    plot = layout.get("plot", {})
    rooms = layout.get("rooms", [])
    area_summary = layout.get("area_summary", {})
    floors = req.get("floors", 1)

    lines = []

    # Line 1: Plot info
    assumed = " (dimensions estimated from area)" if req.get("assumed_dimensions") else ""
    lines.append(
        f"Layout designed for {plot.get('width')}×{plot.get('length')} ft plot "
        f"({area_summary.get('plot_area', '?')} sq ft){assumed}."
    )

    # Line 2: Room count
    bedroom_count = sum(1 for r in rooms if r["room_type"] in ("master_bedroom", "bedroom"))
    bathroom_count = sum(1 for r in rooms if r["room_type"] in ("bathroom", "toilet"))
    lines.append(
        f"Configuration: {bedroom_count} bedroom(s), {bathroom_count} bathroom(s), "
        f"{floors} floor(s), {len(rooms)} total rooms."
    )

    # Line 3: Area utilization
    lines.append(
        f"Area utilization: {area_summary.get('utilization_percentage', '?')} of plot area. "
        f"Circulation: {area_summary.get('circulation_percentage', '?')}."
    )

    # Line 4: Wall specification
    lines.append("Walls: 9-inch external (load-bearing), 4.5-inch internal partitions.")

    # Line 5: Zoning
    zones = {}
    for r in rooms:
        z = r.get("zone", "other")
        zones.setdefault(z, []).append(r.get("name", ""))
    zone_desc = "; ".join(f"{k}: {', '.join(v)}" for k, v in zones.items())
    lines.append(f"Zoning: {zone_desc}.")

    # Line 6: Compliance
    compliant = validation.get("compliant", True)
    if compliant:
        lines.append("All architectural constraints validated. Layout is CAD-ready.")
    else:
        issue_count = (
            len(validation.get("overlap_details", [])) +
            len(validation.get("size_violations", [])) +
            len(validation.get("zoning_issues", []))
        )
        lines.append(f"Validation found {issue_count} issue(s). Review recommended before construction.")

    return "\n".join(lines[:8])
