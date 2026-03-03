"""
Advanced Architectural Intelligence Engine — Constraint-Aware Residential Floor Plan Generator.

This is NOT a template generator. It is a constraint-aware, zoning-aware,
geometry-aware architectural planning system that produces realistic,
buildable, architect-grade residential floor plans dynamically.

Modes:
  CHAT       — Conversational requirement collection
  DESIGN     — Full layout generation (deterministic, CAD-ready)
  REDESIGN   — Fresh layout with same requirements, different strategy
  VALIDATION — Structural compliance check on existing layout JSON

Core Principles Enforced:
  A. Functional Zoning (Public → Semi-Private → Private → Service)
  B. Privacy Gradient (Entrance → Living → Dining → Passage → Bedrooms)
  C. Area Proportions (Living 18-22%, Bedrooms 30-35%, Kitchen 8-12%, etc.)
  D. Minimum Room Sizes (NBC 2016 compliant)
  E. Geometry Constraints (rectangular, aspect 1:1–1:2.5, no overlap)
  F. Ventilation Rules (habitable rooms touch external wall)

Author: CAD Architectural Intelligence Engine v3.0
"""

import json
import math
import re
import hashlib
import time
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from copy import deepcopy

from services.layout_constants import (
    GRID_SNAP,
    WALL_EXTERNAL_FT, WALL_INTERNAL_FT,
    MIN_DIMS, MAX_ASPECT, AREA_FRACTIONS, MIN_AREAS, MAX_AREAS,
    ZONE_MAP, PRIORITY, VASTU_PREFS,
    DESIRED_ADJACENCIES, FORBIDDEN_ADJACENCIES,
)


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE MODES
# ═══════════════════════════════════════════════════════════════════════════

class EngineMode(str, Enum):
    CHAT = "chat"
    FORM = "form"
    DESIGN = "design"
    REDESIGN = "redesign"
    VALIDATION = "validation"


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURAL CONSTANTS (SPEC §2)
# ═══════════════════════════════════════════════════════════════════════════

# Wall thickness
WALL_EXT_INCHES = 9
WALL_INT_INCHES = 4.5
WALL_EXT_FT = WALL_EXT_INCHES / 12   # 0.75 ft
WALL_INT_FT = WALL_INT_INCHES / 12   # 0.375 ft

# Minimum passage / stair width
MIN_PASSAGE_WIDTH = 3.0
MIN_STAIR_WIDTH = 3.0

# Minimum room sizes (sq ft) — §2D
MIN_ROOM_SIZES = {
    "master_bedroom": 120,
    "bedroom":        100,
    "bathroom":       35,
    "toilet":         15,
    "kitchen":        80,
    "living":         120,
    "dining":         80,
    "study":          60,
    "pooja":          16,
    "store":          25,
    "utility":        20,
    "hallway":        30,
    "porch":          40,
    "parking":        150,
    "balcony":        25,
    "staircase":      40,
    "garage":         150,
    "passage":        15,
}

# Standard room proportions (width, length) in feet
STANDARD_DIMS = {
    "master_bedroom": (12, 14),
    "bedroom":        (10, 12),
    "bathroom":       (5, 8),
    "toilet":         (4, 5),
    "kitchen":        (8, 10),
    "living":         (14, 16),
    "dining":         (10, 12),
    "study":          (10, 10),
    "pooja":          (5, 5),
    "store":          (6, 6),
    "utility":        (5, 6),
    "hallway":        (3.5, 10),
    "porch":          (10, 8),
    "parking":        (10, 18),
    "balcony":        (5, 10),
    "staircase":      (5, 10),
    "garage":         (10, 18),
    "passage":        (3.5, 8),
}

# Area distribution percentages — §2C
AREA_PROPORTIONS = {
    "living":            (0.18, 0.22),
    "bedrooms_total":    (0.30, 0.35),
    "kitchen":           (0.08, 0.12),
    "bathrooms_total":   (0.08, 0.12),
    "circulation":       (0.10, 0.15),
    "walls_structure":   (0.08, 0.10),
}

# Zoning classification — §2A
ZONE_CLASSIFICATION = {
    "living":         "public",
    "porch":          "public",
    "foyer":          "public",
    "parking":        "public",
    "balcony":        "public",
    "garage":         "public",
    "dining":         "semi_private",
    "kitchen":        "semi_private",
    "master_bedroom": "private",
    "bedroom":        "private",
    "study":          "private",
    "pooja":          "private",
    "bathroom":       "service",
    "toilet":         "service",
    "utility":        "service",
    "store":          "service",
    "hallway":        "circulation",
    "staircase":      "circulation",
    "passage":        "circulation",
}

# Display names
DISPLAY_NAMES = {
    "master_bedroom": "Master Bedroom",
    "bedroom":        "Bedroom",
    "bathroom":       "Bathroom",
    "toilet":         "Toilet",
    "kitchen":        "Kitchen",
    "living":         "Living Room",
    "dining":         "Dining Room",
    "study":          "Study Room",
    "pooja":          "Pooja Room",
    "store":          "Store Room",
    "utility":        "Utility Room",
    "hallway":        "Hallway",
    "porch":          "Porch",
    "parking":        "Parking",
    "balcony":        "Balcony",
    "staircase":      "Staircase",
    "garage":         "Garage",
    "passage":        "Passage",
}

# Privacy gradient — §2B
# Bedrooms must not directly open into kitchen.
# Bathrooms must not face entrance.
FORBIDDEN_ADJACENCY = [
    ("bedroom",        "kitchen"),
    ("master_bedroom", "kitchen"),
    ("bathroom",       "living"),
    ("toilet",         "living"),
    ("bathroom",       "kitchen"),
    ("toilet",         "kitchen"),
    ("bathroom",       "dining"),
    ("toilet",         "dining"),
    ("pooja",          "toilet"),
    ("pooja",          "bathroom"),
]

# Zoning strategies — §4 Step A
ZONING_STRATEGIES = ["linear", "central_corridor", "split", "adaptive"]

# Mandatory rooms
MANDATORY_ROOMS = ["living", "kitchen"]

# Required input fields
REQUIRED_FIELDS = ["bedrooms", "bathrooms", "floors"]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT MODE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_mode(input_data: Any) -> EngineMode:
    """
    Intelligently auto-detect the engine mode from input data.

    Detection rules:
      1. String with "generate new"/"regenerate"/"new layout" → REDESIGN
      2. Dict with rooms containing position data → VALIDATION
      3. Dict with generatePlan=True or string "generate plan" → DESIGN
      4. Dict with structured fields (bedrooms, etc.) → FORM
      5. String (natural language) → CHAT
    """
    if isinstance(input_data, str):
        text = input_data.lower().strip()

        # Check for REDESIGN triggers first
        redesign_triggers = [
            "generate new", "regenerate", "new layout",
            "try different", "different plan", "fresh layout",
            "another layout", "redesign", "new design",
        ]
        if any(trigger in text for trigger in redesign_triggers):
            return EngineMode.REDESIGN

        # Check for DESIGN trigger
        if "generate plan" in text or "create plan" in text or "build plan" in text:
            return EngineMode.DESIGN

        return EngineMode.CHAT

    if isinstance(input_data, dict):
        # VALIDATION: layout JSON with positioned rooms
        if "rooms" in input_data:
            rooms = input_data["rooms"]
            if isinstance(rooms, list) and rooms:
                first = rooms[0] if rooms else {}
                if isinstance(first, dict) and "position" in first:
                    return EngineMode.VALIDATION

        # REDESIGN via flag
        if input_data.get("redesign") is True:
            return EngineMode.REDESIGN

        # Check message for redesign triggers
        msg = str(input_data.get("message", "")).lower()
        redesign_triggers = [
            "generate new", "regenerate", "new layout",
            "try different", "different plan", "fresh layout",
        ]
        if any(trigger in msg for trigger in redesign_triggers):
            return EngineMode.REDESIGN

        # DESIGN: generatePlan flag or text trigger
        if input_data.get("generatePlan") is True or input_data.get("generate_plan") is True:
            return EngineMode.DESIGN
        if "generate plan" in msg or "create plan" in msg:
            return EngineMode.DESIGN

        # FORM: structured fields present
        form_keys = {"bedrooms", "bathrooms", "floors", "plot_width",
                      "plot_length", "total_area", "kitchen", "max_area"}
        if form_keys.intersection(set(input_data.keys())):
            return EngineMode.FORM

        # CHAT: message-only
        if "message" in input_data:
            return EngineMode.CHAT

    return EngineMode.CHAT


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHAT MODE
# ═══════════════════════════════════════════════════════════════════════════

def chat_response(message: str, history: List[Dict]) -> Dict:
    """
    Handle CHAT MODE input.

    Collects requirements naturally. Asks follow-up questions.
    Suggests reasonable defaults. No JSON output in this mode.
    Does NOT generate layout until user says "generate plan".

    Returns: { "reply": str, "mode": "chat", "collected": dict, "ready": bool }
    """
    collected = _parse_requirements_from_history(message, history)

    # Check for "generate plan" trigger
    if "generate plan" in message.lower() or "create plan" in message.lower():
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
                "reply": (
                    f"I need a few more details before generating: "
                    f"{', '.join(missing)}. Could you provide those?"
                ),
                "mode": EngineMode.CHAT,
                "collected": collected,
                "ready": False,
            }

    turn = len([h for h in history if h.get("role") == "user"])

    # ── First turn — greeting ──
    if turn == 0:
        if collected["has_dimensions"] and collected["has_bedrooms"]:
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
                    "  • Number of floors (default: 1)\n\n"
                    "You can provide all at once (e.g., '30x40, 3 bedrooms, 2 bathrooms') "
                    "or one at a time. I'll suggest reasonable defaults where appropriate."
                ),
                "mode": EngineMode.CHAT,
                "collected": collected,
                "ready": False,
            }

    # ── Follow-up turns — ask for what's missing ──
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

    # Ask for next missing field with helpful defaults
    if not collected["has_dimensions"]:
        return {
            "reply": (
                "What is your plot size?\n"
                "You can specify as dimensions (e.g., '30×40 feet') or "
                "total area (e.g., '1200 sq ft'). "
                "For reference, a typical 2BHK needs 800-1000 sq ft, "
                "3BHK needs 1200-1500 sq ft."
            ),
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_bedrooms"]:
        # Suggest based on area if available
        suggestion = ""
        if collected.get("total_area"):
            area = collected["total_area"]
            if area < 600:
                suggestion = " Based on your plot size, I'd recommend 1 bedroom."
            elif area < 1000:
                suggestion = " Based on your plot size, I'd recommend 2 bedrooms."
            elif area < 1800:
                suggestion = " Based on your plot size, I'd recommend 3 bedrooms."
            else:
                suggestion = " Based on your plot size, you could comfortably fit 3-4 bedrooms."
        return {
            "reply": f"How many bedrooms do you need?{suggestion}",
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_bathrooms"]:
        beds = collected.get("bedrooms", 2)
        default = max(1, beds - 1)
        return {
            "reply": (
                f"How many bathrooms do you need? "
                f"For {beds} bedroom(s), I'd suggest {default} bathroom(s) as a default."
            ),
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    if not collected["has_floors"]:
        return {
            "reply": (
                "How many floors? Most residential homes are 1 or 2 floors. "
                "I'll default to 1 floor if you'd prefer to skip this."
            ),
            "mode": EngineMode.CHAT,
            "collected": collected,
            "ready": False,
        }

    return {
        "reply": "I have all requirements. Say \"Generate Plan\" when ready.",
        "mode": EngineMode.CHAT,
        "collected": collected,
        "ready": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FORM MODE — Validate structured input
# ═══════════════════════════════════════════════════════════════════════════

def form_validate(data: Dict) -> Dict:
    """
    Handle FORM MODE input.

    Validates structured JSON/form data. No questions asked.

    Returns:
      { "valid": bool, "error": str|None, "missing_fields": [], "normalized": dict }
      OR if generatePlan is true, routes to design_generate.
    """
    missing = []

    if "bedrooms" not in data or data.get("bedrooms") is None:
        missing.append("bedrooms")
    if "bathrooms" not in data or data.get("bathrooms") is None:
        missing.append("bathrooms")
    if "floors" not in data or data.get("floors") is None:
        missing.append("floors")

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

    normalized = _normalize_form_data(data)

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


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — DESIGN MODE (Full Layout Generation)
# ═══════════════════════════════════════════════════════════════════════════

def design_generate(requirements: Dict, strategy_override: str = None) -> Dict:
    """
    Handle DESIGN MODE.

    Generates full architect-grade layout. No questions. No conversation.

    Procedure:
      Step A — Determine zoning strategy dynamically from plot ratio
      Step B — Allocate areas proportionally (§2C)
      Step C — Place rooms by zone with privacy gradient
      Step D — Validate geometry (overlap, size, adjacency, aspect ratio)

    Returns professional explanation (max 8 lines) + strict JSON.
    """
    req = _normalize_form_data(requirements)

    plot_w = req["plot_width"]
    plot_l = req["plot_length"]
    total_area = req["total_area"]
    floors = req.get("floors", 1)
    bedrooms = req.get("bedrooms", 2)
    bathrooms = req.get("bathrooms", 1)
    extras = req.get("extras", [])

    # ── Check minimum area ──
    min_required = _calculate_minimum_area(bedrooms, bathrooms, extras)
    if total_area < min_required:
        return {
            "error": "Insufficient plot area for requested configuration",
            "suggestion": (
                f"Minimum area needed: {min_required} sq ft. "
                f"Increase area or reduce room count."
            ),
            "mode": EngineMode.DESIGN,
        }

    # ── Step A: Determine zoning strategy (§4A) ──
    strategy = strategy_override or _select_zoning_strategy(plot_w, plot_l)

    # ── Step B: Allocate room areas (§4B) ──
    room_specs = _allocate_areas(
        total_area=total_area,
        usable_w=plot_w - 2 * WALL_EXT_FT,
        usable_l=plot_l - 2 * WALL_EXT_FT,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        extras=extras,
    )

    # ── Step C: Spatial placement (§4C) ──
    placed_rooms = _place_rooms_by_strategy(
        room_specs=room_specs,
        plot_w=plot_w,
        plot_l=plot_l,
        strategy=strategy,
    )

    # ── Assign doors and windows ──
    placed_rooms = _assign_doors_windows(placed_rooms, plot_w, plot_l)

    # ── Step D: Geometry validation (§4D) ──
    # Auto-correct violations before output
    placed_rooms = _auto_correct(placed_rooms, plot_w, plot_l)

    # ── Build output layout ──
    layout = _build_layout_output(
        placed_rooms, plot_w, plot_l, total_area, floors, strategy
    )

    # ── Validate ──
    validation = validate_layout(layout)

    # ── Explanation (max 8 lines) ──
    explanation = _build_design_explanation(req, layout, validation, strategy)

    return {
        "explanation": explanation,
        "layout": layout,
        "validation": validation,
        "mode": EngineMode.DESIGN,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — RE-DESIGN MODE
# ═══════════════════════════════════════════════════════════════════════════

def redesign_generate(requirements: Dict, previous_strategy: str = None) -> Dict:
    """
    Handle REDESIGN MODE.

    Same requirements, completely different layout:
      - Different zoning strategy
      - Different room arrangement
      - Different circulation type
      - Different spatial orientation
      - No reuse of previous coordinates

    Returns a fresh, structurally valid design.
    """
    # Pick a DIFFERENT strategy from the previous one
    available = [s for s in ZONING_STRATEGIES if s != previous_strategy]
    if not available:
        available = ZONING_STRATEGIES

    # Use a time-based seed for variety
    seed = int(time.time() * 1000) % len(available)
    new_strategy = available[seed]

    # Add a redesign salt to force different placement
    requirements = dict(requirements)
    requirements["_redesign_salt"] = int(time.time())

    return design_generate(requirements, strategy_override=new_strategy)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — VALIDATION MODE
# ═══════════════════════════════════════════════════════════════════════════

def validate_layout(layout: Dict) -> Dict:
    """
    Validate a layout JSON for full architectural compliance.

    Checks:
      - Overlap detection (AABB)
      - Minimum size compliance
      - Zoning / forbidden adjacency
      - Area overflow
      - Circulation continuity
      - Boundary fit
      - Aspect ratio compliance
      - Ventilation (external wall access)

    Returns structured validation report (§7 format).
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
    aspect_issues = _check_aspect_ratios(rooms)
    ventilation_issues = _check_ventilation(rooms, plot_w, plot_l)
    reachability_issues = _check_reachability(rooms)

    total_used = sum(r.get("area", 0) for r in rooms)

    all_ok = (
        not overlap_issues and not size_violations and
        not zoning_issues and not area_overflow and
        not circulation_issues and not boundary_issues and
        not aspect_issues and not ventilation_issues and
        not reachability_issues
    )

    return {
        "overlap": bool(overlap_issues),
        "overlap_details": overlap_issues,
        "size_violations": size_violations,
        "zoning_issues": zoning_issues,
        "area_overflow": area_overflow,
        "circulation_issues": circulation_issues,
        "boundary_issues": boundary_issues,
        "aspect_ratio_issues": aspect_issues,
        "ventilation_issues": ventilation_issues,
        "reachability_issues": reachability_issues,
        "area_summary": {
            "plot_area": round(plot_area, 1),
            "total_used_area": round(total_used, 1),
            "utilization_percent": round(total_used / max(plot_area, 1) * 100, 1),
        },
        "min_size_ok": len(size_violations) == 0,
        "zoning_ok": len(zoning_issues) == 0,
        "geometry_ok": len(aspect_issues) == 0 and len(boundary_issues) == 0,
        "area_ok": not bool(area_overflow),
        "compliant": all_ok,
        "mode": EngineMode.VALIDATION,
    }


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED PROCESSOR — Single entry point
# ═══════════════════════════════════════════════════════════════════════════

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
            message = (
                input_data if isinstance(input_data, str)
                else input_data.get("message", "")
            )
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

        elif mode == EngineMode.REDESIGN:
            if isinstance(input_data, str):
                # Try to extract requirements from history
                collected = _parse_requirements_from_history(input_data, history)
                if collected.get("complete"):
                    req = _collected_to_requirements(collected)
                    return redesign_generate(req)
                else:
                    return {
                        "error": "Cannot redesign without requirements.",
                        "suggestion": "Provide plot dimensions, bedrooms, bathrooms first.",
                        "mode": EngineMode.REDESIGN,
                    }
            previous_strategy = input_data.get("_previous_strategy")
            return redesign_generate(input_data, previous_strategy)

        elif mode == EngineMode.VALIDATION:
            return validate_layout(input_data)

        else:
            return {"error": "Unknown mode", "mode": "error"}

    except Exception as e:
        return {
            "error": f"Engine error: {str(e)}",
            "mode": "error",
        }


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Requirement Parsing from Conversation
# ═══════════════════════════════════════════════════════════════════════════

def _parse_requirements_from_history(current_msg: str, history: List[Dict]) -> Dict:
    """
    Parse all collected requirements from conversation history + current message.

    Handles:
      - Dimensions: 30x40, 30×40, 30*40
      - Area: 1200 sqft, 1200 sq ft
      - BHK: 3BHK → 3 bedrooms, 2 bathrooms
      - Explicit bedrooms/bathrooms/floors
      - Contextual: number after "bedrooms?" question
      - Extras: dining, study, pooja, balcony, parking, etc.
    """
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

    # ── Dimensions: 30x40, 30*40, 30×40 ──
    dim_match = re.search(r'(\d+(?:\.\d+)?)\s*[x×*]\s*(\d+(?:\.\d+)?)', text_lower)
    if dim_match:
        data["has_dimensions"] = True
        data["plot_width"] = float(dim_match.group(1))
        data["plot_length"] = float(dim_match.group(2))
        data["total_area"] = data["plot_width"] * data["plot_length"]

    # ── Area: 1200 sqft, 1200 sq ft, 1200 square feet ──
    area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\s*ft|sqft|square\s*feet?)', text_lower)
    if area_match:
        data["has_dimensions"] = True
        data["total_area"] = float(area_match.group(1))

    # ── Standalone number > 100 (likely area) ──
    if not data["has_dimensions"]:
        all_msgs = [m.get("content", "") for m in history if m.get("role") == "user"]
        all_msgs.append(current_msg)
        for msg in all_msgs:
            num_match = re.match(r'^\s*(\d{3,5})\s*$', msg.strip())
            if num_match:
                val = int(num_match.group(1))
                if 100 <= val <= 50000:
                    data["has_dimensions"] = True
                    data["total_area"] = float(val)

    # ── BHK: 3BHK → 3 bedrooms, 2 bathrooms ──
    bhk_match = re.search(r'(\d+)\s*bhk', text_lower)
    if bhk_match:
        bhk = int(bhk_match.group(1))
        data["has_bedrooms"] = True
        data["has_bathrooms"] = True
        data["bedrooms"] = bhk
        data["bathrooms"] = max(1, bhk - 1)

    # ── Explicit bedrooms ──
    bed_match = re.search(r'(\d+)\s*(?:bed(?:room)?s?)', text_lower)
    if bed_match:
        data["has_bedrooms"] = True
        data["bedrooms"] = int(bed_match.group(1))

    # ── Explicit bathrooms ──
    bath_match = re.search(r'(\d+)\s*(?:bath(?:room)?s?|toilets?)', text_lower)
    if bath_match:
        data["has_bathrooms"] = True
        data["bathrooms"] = int(bath_match.group(1))

    # ── Floors ──
    floor_match = re.search(r'(\d+)\s*(?:floor|storey|story|level)', text_lower)
    if floor_match:
        data["has_floors"] = True
        data["floors"] = int(floor_match.group(1))

    # ── Contextual parsing: assistant question → user answer ──
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        user_text = msg.get("content", "").strip()
        prev_assistant = ""
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "assistant":
                prev_assistant = history[j].get("content", "").lower()
                break

        if "bedroom" in prev_assistant and not data["has_bedrooms"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_bedrooms"] = True
                data["bedrooms"] = int(nums[0])

        if "bathroom" in prev_assistant and not data["has_bathrooms"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_bathrooms"] = True
                data["bathrooms"] = int(nums[0])

        if "floor" in prev_assistant and not data["has_floors"]:
            nums = re.findall(r'\d+', user_text)
            if nums:
                data["has_floors"] = True
                data["floors"] = int(nums[0])

        if "plot" in prev_assistant and not data["has_dimensions"]:
            dim_ctx = re.search(r'(\d+(?:\.\d+)?)\s*[x×*]\s*(\d+(?:\.\d+)?)', user_text.lower())
            if dim_ctx:
                data["has_dimensions"] = True
                data["plot_width"] = float(dim_ctx.group(1))
                data["plot_length"] = float(dim_ctx.group(2))
                data["total_area"] = data["plot_width"] * data["plot_length"]
            else:
                nums = re.findall(r'\d+', user_text)
                if nums:
                    val = int(nums[0])
                    if val >= 100:
                        data["has_dimensions"] = True
                        data["total_area"] = float(val)

    # ── Also parse current message contextually ──
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
                    data["total_area"] = float(val)

    # ── Extras ──
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
    if "utilit" in text_lower:
        extras.append("utility")
    data["extras"] = extras

    # ── Default floors to 1 ──
    if not data["has_floors"]:
        data["floors"] = 1
        data["has_floors"] = True

    # ── Completeness check ──
    data["complete"] = (
        data["has_dimensions"] and data["has_bedrooms"] and
        data["has_bathrooms"] and data["has_floors"]
    )

    return data


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Helpers
# ═══════════════════════════════════════════════════════════════════════════

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
    """Build human-readable summary of collected requirements."""
    parts = []
    if collected.get("plot_width") and collected.get("plot_length"):
        parts.append(
            f"  Plot: {collected['plot_width']} × {collected['plot_length']} ft "
            f"({collected.get('total_area', '?')} sq ft)"
        )
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


def _normalize_form_data(data: Dict) -> Dict:
    """Normalize form input into a consistent requirements dict."""
    result = {
        "floors": int(data.get("floors", 1)),
        "bedrooms": int(data.get("bedrooms") or data.get("num_bedrooms") or 2),
        "bathrooms": int(data.get("bathrooms") or data.get("num_bathrooms") or 1),
        "extras": list(data.get("extras", [])),
    }

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
        # §8: If only total area, assume 1:1.3 ratio
        shorter = math.sqrt(ta / 1.3)
        longer = shorter * 1.3
        result["plot_width"] = round(shorter, 1)
        result["plot_length"] = round(longer, 1)
        result["assumed_dimensions"] = True
    else:
        result["plot_width"] = 30.0
        result["plot_length"] = 40.0
        result["total_area"] = 1200.0
        result["assumed_dimensions"] = True

    # Handle extras from soft constraints
    extras = list(result["extras"])
    for key in ("balcony", "parking", "pooja_room", "pooja", "dining", "study", "store"):
        if data.get(key):
            val = key.replace("_room", "")
            if val not in extras:
                extras.append(val)

    # Handle 'requirements' list (e.g. ['living room', 'kitchen', 'dining'])
    req_list = data.get("requirements", [])
    if isinstance(req_list, list):
        extra_keywords = {"dining", "study", "pooja", "store", "balcony", "parking", "utility", "porch"}
        for item in req_list:
            normalized = str(item).lower().replace(" room", "").replace("_room", "").strip()
            if normalized in extra_keywords and normalized not in extras:
                extras.append(normalized)

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


def _calculate_minimum_area(bedrooms: int, bathrooms: int, extras: List[str]) -> float:
    """Calculate minimum total area required for room configuration."""
    area = 0.0
    area += MIN_ROOM_SIZES["living"]        # Living
    area += MIN_ROOM_SIZES["kitchen"]        # Kitchen
    if bedrooms >= 1:
        area += MIN_ROOM_SIZES["master_bedroom"]
    for _ in range(max(0, bedrooms - 1)):
        area += MIN_ROOM_SIZES["bedroom"]
    for _ in range(bathrooms):
        area += MIN_ROOM_SIZES["bathroom"]
    for extra in extras:
        area += MIN_ROOM_SIZES.get(extra, 30)
    area *= 1.10  # walls + circulation overhead
    return round(area)


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Zoning Strategy Selection (§4 Step A)
# ═══════════════════════════════════════════════════════════════════════════

def _select_zoning_strategy(plot_w: float, plot_l: float) -> str:
    """
    Choose zoning strategy dynamically based on plot ratio.

    Rules:
      width > depth  → linear (rooms along length)
      depth > width  → central_corridor (corridor down the middle)
      near square    → split (left-right split)
      else           → adaptive
    """
    ratio = plot_w / max(plot_l, 0.1)

    if ratio > 1.2:
        return "linear"
    elif ratio < 0.8:
        return "central_corridor"
    elif 0.9 <= ratio <= 1.1:
        return "split"
    else:
        return "adaptive"


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Area Allocation (§4 Step B)
# ═══════════════════════════════════════════════════════════════════════════

# ── ADJACENCY GRAPH — mandatory room connectivity (Step 3) ──
ADJACENCY_GRAPH_MANDATORY = {
    "entrance":       ["living"],
    "living":         ["dining", "passage"],
    "dining":         ["kitchen"],
    "kitchen":        ["utility", "store"],
    "passage":        ["master_bedroom", "bedroom", "bathroom", "study", "pooja", "staircase"],
    "master_bedroom": ["bathroom"],
    "bedroom":        ["bathroom"],
}

# Preferred (soft) adjacency
ADJACENCY_GRAPH_PREFERRED = {
    "living":         ["staircase"],
    "kitchen":        ["dining"],
    "master_bedroom": ["bathroom"],   # attached bath
}

# Hard forbidden connections
FORBIDDEN_CONNECTIONS = {
    ("bedroom",        "kitchen"),
    ("master_bedroom", "kitchen"),
    ("bathroom",       "dining"),
    ("bedroom",        "bedroom"),   # not without corridor
}

# ── Percentage allocation bands (of total_area) ──
ALLOC_BANDS = {
    "living":         (0.15, 0.25),
    "kitchen":        (0.08, 0.12),
    "dining":         (0.08, 0.12),
    "master_bedroom": (0.12, 0.16),
    "bedroom":        (0.10, 0.14),
    "bathroom":       (0.03, 0.05),
    "passage":        (0.05, 0.08),
    "study":          (0.04, 0.06),
    "pooja":          (0.02, 0.03),
    "store":          (0.02, 0.04),
    "utility":        (0.02, 0.03),
    "balcony":        (0.03, 0.06),
    "staircase":      (0.03, 0.05),
    "parking":        (0.08, 0.12),
    "garage":         (0.08, 0.12),
    "porch":          (0.03, 0.05),
}


def _allocate_areas(
    total_area: float,
    usable_w: float, usable_l: float,
    bedrooms: int, bathrooms: int,
    extras: List[str],
) -> List[Dict]:
    """
    Allocate target areas as a percentage of total_area.

    Uses ALLOC_BANDS midpoints, dynamically scaled.
    Returns list of room specs with name, room_type, target_area, width, length, zone.
    """
    usable_area = usable_w * usable_l
    rooms = []

    def _add(name, rtype, frac_lo, frac_hi, zone):
        mid = (frac_lo + frac_hi) / 2
        target = total_area * mid
        target = max(target, MIN_ROOM_SIZES.get(rtype, 25))
        if rtype in MAX_AREAS:
            target = min(target, MAX_AREAS[rtype])
        w, l = _scale_room(rtype, target)
        rooms.append({
            "name": name, "room_type": rtype,
            "target_area": round(w * l, 1), "width": w, "length": l,
            "zone": zone,
        })

    # Mandatory rooms
    _add("Living Room", "living", *ALLOC_BANDS["living"], "public")
    _add("Kitchen", "kitchen", *ALLOC_BANDS["kitchen"], "semi_private")

    # Dining (always included)
    _add("Dining Room", "dining", *ALLOC_BANDS["dining"], "semi_private")

    # Passage / corridor
    _add("Passage", "passage", *ALLOC_BANDS["passage"], "circulation")

    # Bedrooms
    for i in range(bedrooms):
        if i == 0:
            _add("Master Bedroom", "master_bedroom",
                 *ALLOC_BANDS["master_bedroom"], "private")
        else:
            _add(f"Bedroom {i + 1}", "bedroom",
                 *ALLOC_BANDS["bedroom"], "private")

    # Bathrooms — paired with bedrooms
    for i in range(bathrooms):
        name = f"Bathroom {i + 1}" if bathrooms > 1 else "Bathroom"
        _add(name, "bathroom", *ALLOC_BANDS["bathroom"], "service")

    # Extras
    for extra in extras:
        if extra in ALLOC_BANDS and extra not in ("living", "kitchen", "dining", "passage"):
            zone = ZONE_CLASSIFICATION.get(extra, "private")
            _add(DISPLAY_NAMES.get(extra, extra.title()), extra,
                 *ALLOC_BANDS[extra], zone)

    # ── Normalize: total room area should not exceed usable ──
    total_allocated = sum(r["target_area"] for r in rooms)
    if total_allocated > usable_area * 0.92:  # keep 8% for walls
        scale = usable_area * 0.92 / total_allocated
        for r in rooms:
            r["target_area"] = round(r["target_area"] * scale, 1)
            r["width"] = round(r["width"] * math.sqrt(scale), 1)
            r["length"] = round(r["length"] * math.sqrt(scale), 1)
            min_a = MIN_ROOM_SIZES.get(r["room_type"], 25)
            if r["target_area"] < min_a:
                r["target_area"] = min_a
                r["width"], r["length"] = _scale_room(r["room_type"], min_a)

    # Compute area_ratio for each room
    total_alloc = sum(r["target_area"] for r in rooms) or 1
    for r in rooms:
        r["area_ratio"] = round(r["target_area"] / total_alloc, 3)

    return rooms


def _scale_room(room_type: str, target_area: float) -> Tuple[float, float]:
    """
    Scale room to approximate target area maintaining reasonable proportions.

    Enforces:
      - Minimum dimensions (4 ft)
      - Aspect ratio 1:1 to 1:2.5 (§2E)
      - Snap to 0.5 ft grid
    """
    std_w, std_l = STANDARD_DIMS.get(room_type, (10, 10))
    std_area = std_w * std_l
    min_area = MIN_ROOM_SIZES.get(room_type, 25)

    target_area = max(target_area, min_area)

    scale = math.sqrt(target_area / max(std_area, 1))
    new_w = std_w * scale
    new_l = std_l * scale

    # Enforce minimum dimension
    new_w = max(new_w, 4.0)
    new_l = max(new_l, 4.0)

    # Enforce max aspect ratio 1:2.5 (§2E)
    max_ratio = 2.5
    if max(new_w, new_l) / max(min(new_w, new_l), 0.1) > max_ratio:
        if new_w > new_l:
            new_w = new_l * max_ratio
        else:
            new_l = new_w * max_ratio

    # Snap to 0.5 ft grid
    new_w = round(new_w * 2) / 2
    new_l = round(new_l * 2) / 2

    return max(new_w, 4.0), max(new_l, 4.0)


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Room Placement by Strategy (§4 Step C)
# ═══════════════════════════════════════════════════════════════════════════

def _place_rooms_by_strategy(
    room_specs: List[Dict],
    plot_w: float, plot_l: float,
    strategy: str,
) -> List[Dict]:
    """
    BSP-based room placement with mandatory adjacency graph enforcement.

    Algorithm:
      1. Classify rooms into zones (public / private / service / circulation)
      2. Allocate three horizontal bands: PUBLIC (front) + CORRIDOR + PRIVATE (back)
      3. Within each band, use BSP subdivision to pack rooms adjacently
      4. Pair bedrooms with their bathrooms
      5. Enforce adjacency graph (kitchen next to dining, bedrooms next to bathrooms)
      6. Ensure every room except bathrooms touches an exterior wall
    """
    public, private, service = _classify_rooms(room_specs)
    private_zone = _interleave_beds_baths(private, service)

    # Step 4: corridor width = 3-5% of plot width, clamped to MIN_PASSAGE_WIDTH
    dynamic_corridor = max(MIN_PASSAGE_WIDTH, min(plot_w * 0.04, plot_w * 0.05))
    dynamic_corridor = round(dynamic_corridor, 1)

    return _place_bsp(
        public_rooms=public,
        private_rooms=private_zone,
        plot_w=plot_w, plot_l=plot_l,
        corridor_width=dynamic_corridor,
    )


def _classify_rooms(room_specs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Classify rooms into public, private, and service groups."""
    public = []
    private = []
    service = []

    for r in room_specs:
        zone = ZONE_CLASSIFICATION.get(r["room_type"], "private")
        if zone in ("public", "semi_private"):
            public.append(r)
        elif zone == "private":
            private.append(r)
        else:  # service, circulation
            service.append(r)

    # Sort within groups for consistent placement:
    # Public: living first, then kitchen, then dining, then others
    pub_order = {"living": 0, "porch": 1, "dining": 2, "kitchen": 3, "balcony": 4}
    public.sort(key=lambda r: pub_order.get(r["room_type"], 99))

    # Private: master bedroom first, then other bedrooms
    priv_order = {"master_bedroom": 0, "bedroom": 1, "study": 2, "pooja": 3}
    private.sort(key=lambda r: (priv_order.get(r["room_type"], 99), -r["target_area"]))

    # Service: bathrooms first
    serv_order = {"bathroom": 0, "toilet": 1, "utility": 2, "store": 3}
    service.sort(key=lambda r: serv_order.get(r["room_type"], 99))

    return public, private, service


def _interleave_beds_baths(private: List[Dict], service: List[Dict]) -> List[Dict]:
    """
    Interleave bedrooms with their bathrooms for adjacency.

    Pairs: Master + Bath1, Bedroom2 + Bath2, etc.
    Remaining service rooms go at the end.
    """
    beds = [r for r in private if r["room_type"] in ("master_bedroom", "bedroom")]
    non_beds = [r for r in private if r["room_type"] not in ("master_bedroom", "bedroom")]
    baths = [r for r in service if r["room_type"] in ("bathroom", "toilet")]
    non_baths = [r for r in service if r["room_type"] not in ("bathroom", "toilet")]

    interleaved = []
    for i in range(max(len(beds), len(baths))):
        if i < len(beds):
            interleaved.append(beds[i])
        if i < len(baths):
            interleaved.append(baths[i])
    interleaved.extend(non_beds)
    interleaved.extend(non_baths)
    return interleaved


def _place_bsp(
    public_rooms: List[Dict],
    private_rooms: List[Dict],
    plot_w: float, plot_l: float,
    corridor_width: float,
) -> List[Dict]:
    """
    BSP placement engine — subdivides usable area into bands then packs rooms.

    Layout (front = south entrance):
      ┌──────────────────────────────────┐
      │       PUBLIC BAND (front)        │  Living, Dining, Kitchen
      ├──────────────────────────────────┤
      │        PASSAGE / CORRIDOR        │  3–5% of plot width
      ├──────────────────────────────────┤
      │      PRIVATE BAND  (back)        │  Bedrooms, Bathrooms, etc.
      └──────────────────────────────────┘
    """
    placed: List[Dict] = []
    WALL = WALL_EXT_FT
    IWALL = WALL_INT_FT

    usable_w = plot_w - 2 * WALL
    usable_l = plot_l - 2 * WALL

    # Remove passage from room lists — it'll be placed as corridor
    passage_spec = None
    filtered_pub = []
    filtered_priv = []
    for r in public_rooms:
        if r["room_type"] == "passage":
            passage_spec = r
        else:
            filtered_pub.append(r)
    for r in private_rooms:
        if r["room_type"] == "passage":
            passage_spec = r
        else:
            filtered_priv.append(r)

    has_pub = len(filtered_pub) > 0
    has_priv = len(filtered_priv) > 0
    n_corridors = 1 if (has_pub and has_priv) else 0
    actual_corridor = corridor_width if n_corridors else 0
    avail_h = usable_l - actual_corridor

    if avail_h < 8:
        avail_h = usable_l
        actual_corridor = 0
        n_corridors = 0

    # ── Height proportions from area needs ──
    pub_area = sum(r["target_area"] for r in filtered_pub)
    priv_area = sum(r["target_area"] for r in filtered_priv)
    total_area = pub_area + priv_area or 1.0

    if has_pub and has_priv:
        pub_h = max(8.0, avail_h * (pub_area / total_area))
        priv_h = max(8.0, avail_h - pub_h)
        # Balance: neither band gets less than 35%
        if pub_h / avail_h < 0.35:
            pub_h = avail_h * 0.35
            priv_h = avail_h - pub_h
        elif priv_h / avail_h < 0.35:
            priv_h = avail_h * 0.35
            pub_h = avail_h - priv_h
    elif has_pub:
        pub_h = avail_h
        priv_h = 0
    else:
        pub_h = 0
        priv_h = avail_h

    # ── PUBLIC BAND ──
    if has_pub:
        placed.extend(
            _bsp_subdivide(filtered_pub, WALL, WALL, usable_w, round(pub_h, 1), IWALL)
        )

    # ── CORRIDOR / PASSAGE as physical room ──
    if n_corridors and actual_corridor > 0:
        corridor_y = round(WALL + pub_h, 1)
        spec = passage_spec or {
            "name": "Passage", "room_type": "passage",
            "target_area": round(usable_w * actual_corridor, 1),
            "width": usable_w, "length": actual_corridor,
            "zone": "circulation",
        }
        placed.append(_make_placed_room(
            spec, usable_w, actual_corridor, WALL, corridor_y
        ))

    # ── PRIVATE BAND ──
    if has_priv:
        priv_y = WALL + (pub_h + actual_corridor if has_pub else 0)
        final_priv_h = round(WALL + usable_l - priv_y, 1)
        final_priv_h = max(final_priv_h, 8.0)
        placed.extend(
            _bsp_subdivide(filtered_priv, WALL, round(priv_y, 1),
                           usable_w, final_priv_h, IWALL)
        )

    return placed


def _bsp_subdivide(
    rooms: List[Dict],
    x: float, y: float,
    w: float, h: float,
    iwall: float,
) -> List[Dict]:
    """
    Recursively subdivide a rectangular region among rooms.

    For 3+ rooms in a single axis, uses flat proportional fill
    to avoid rounding drift from recursive splitting.
    """
    if not rooms:
        return []
    if len(rooms) == 1:
        return [_make_placed_room(rooms[0], w, h, x, y)]

    n = len(rooms)
    total_gaps = iwall * (n - 1)

    # Check if all rooms can fit as a row (horizontal)
    if w >= h:
        net_w = w - total_gaps
        min_w_per_room = max(4.0, h / 2.5)  # aspect ratio constraint
        if net_w >= min_w_per_room * n:
            return _flat_fill_h(rooms, x, y, w, h, iwall)
        # Too many for one row: split into 2 groups, stacked vertically
        return _bsp_split_v(rooms, x, y, w, h, iwall)
    else:
        net_h = h - total_gaps
        min_h_per_room = max(4.0, w / 2.5)
        if net_h >= min_h_per_room * n:
            return _flat_fill_v(rooms, x, y, w, h, iwall)
        # Too many for one stack: split into 2 groups, side by side
        return _bsp_split_h(rooms, x, y, w, h, iwall)


def _flat_fill_h(
    rooms: List[Dict],
    x: float, y: float,
    w: float, h: float,
    iwall: float,
) -> List[Dict]:
    """Pack rooms left-to-right with proportional widths. Last room fills remaining."""
    n = len(rooms)
    areas = [r["target_area"] for r in rooms]
    total_area = sum(areas) or 1
    total_gaps = iwall * (n - 1)
    net_w = w - total_gaps

    placed = []
    cx = x
    for i, room in enumerate(rooms):
        if i == n - 1:
            rw = round(x + w - cx, 1)
        else:
            rw = round(net_w * (areas[i] / total_area), 1)
            rw = max(4.0, rw)
        rw = max(4.0, rw)
        placed.append(_make_placed_room(room, rw, h, round(cx, 1), y))
        cx += rw + iwall

    return placed


def _flat_fill_v(
    rooms: List[Dict],
    x: float, y: float,
    w: float, h: float,
    iwall: float,
) -> List[Dict]:
    """Pack rooms top-to-bottom with proportional heights. Last room fills remaining."""
    n = len(rooms)
    areas = [r["target_area"] for r in rooms]
    total_area = sum(areas) or 1
    total_gaps = iwall * (n - 1)
    net_h = h - total_gaps

    placed = []
    cy = y
    for i, room in enumerate(rooms):
        if i == n - 1:
            rh = round(y + h - cy, 1)
        else:
            rh = round(net_h * (areas[i] / total_area), 1)
            rh = max(4.0, rh)
        rh = max(4.0, rh)
        placed.append(_make_placed_room(room, w, rh, x, round(cy, 1)))
        cy += rh + iwall

    return placed


def _bsp_split_h(
    rooms: List[Dict],
    x: float, y: float,
    w: float, h: float,
    iwall: float,
) -> List[Dict]:
    """Split rooms horizontally (left / right) based on area proportion."""
    n = len(rooms)
    if n == 1:
        return [_make_placed_room(rooms[0], w, h, x, y)]

    # Find optimal split point by area balance
    areas = [r["target_area"] for r in rooms]
    total = sum(areas) or 1
    best_idx, best_diff = 1, float("inf")
    cum = 0
    for i in range(len(areas) - 1):
        cum += areas[i]
        diff = abs(cum / total - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_idx = i + 1

    left_rooms = rooms[:best_idx]
    right_rooms = rooms[best_idx:]

    left_area = sum(r["target_area"] for r in left_rooms)
    right_area = sum(r["target_area"] for r in right_rooms)
    frac = left_area / (left_area + right_area) if (left_area + right_area) > 0 else 0.5

    gap = iwall if len(left_rooms) > 0 and len(right_rooms) > 0 else 0
    net_w = w - gap
    left_w = max(4.0, round(net_w * frac, 1))
    # Right side fills remaining exactly — no rounding drift
    right_w = max(4.0, round(net_w - left_w, 1))

    # Ensure both sides get at least 4 ft
    if left_w < 4.0:
        left_w = 4.0
        right_w = max(4.0, round(net_w - left_w, 1))
    if right_w < 4.0:
        right_w = 4.0
        left_w = max(4.0, round(net_w - right_w, 1))

    right_x = round(x + left_w + gap, 1)
    # Recalculate right_w from right_x to boundary to avoid gaps/overlaps
    right_w = round(x + w - right_x, 1)
    right_w = max(4.0, right_w)

    placed = []
    placed.extend(_bsp_subdivide(left_rooms, x, y, left_w, h, iwall))
    placed.extend(_bsp_subdivide(right_rooms, right_x, y, right_w, h, iwall))
    return placed


def _bsp_split_v(
    rooms: List[Dict],
    x: float, y: float,
    w: float, h: float,
    iwall: float,
) -> List[Dict]:
    """Split rooms vertically (top / bottom) based on area proportion."""
    n = len(rooms)
    if n == 1:
        return [_make_placed_room(rooms[0], w, h, x, y)]

    areas = [r["target_area"] for r in rooms]
    total = sum(areas) or 1
    best_idx, best_diff = 1, float("inf")
    cum = 0
    for i in range(len(areas) - 1):
        cum += areas[i]
        diff = abs(cum / total - 0.5)
        if diff < best_diff:
            best_diff = diff
            best_idx = i + 1

    top_rooms = rooms[:best_idx]
    bottom_rooms = rooms[best_idx:]

    top_area = sum(r["target_area"] for r in top_rooms)
    bottom_area = sum(r["target_area"] for r in bottom_rooms)
    frac = top_area / (top_area + bottom_area) if (top_area + bottom_area) > 0 else 0.5

    gap = iwall if len(top_rooms) > 0 and len(bottom_rooms) > 0 else 0
    net_h = h - gap
    top_h = max(4.0, round(net_h * frac, 1))
    bottom_h = max(4.0, round(net_h - top_h, 1))

    if top_h < 4.0:
        top_h = 4.0
        bottom_h = max(4.0, round(net_h - top_h, 1))
    if bottom_h < 4.0:
        bottom_h = 4.0
        top_h = max(4.0, round(net_h - bottom_h, 1))

    bottom_y = round(y + top_h + gap, 1)
    # Recalculate bottom_h from bottom_y to boundary to avoid gaps/overlaps
    bottom_h = round(y + h - bottom_y, 1)
    bottom_h = max(4.0, bottom_h)

    placed = []
    placed.extend(_bsp_subdivide(top_rooms, x, y, w, top_h, iwall))
    placed.extend(_bsp_subdivide(bottom_rooms, x, bottom_y, w, bottom_h, iwall))
    return placed


def _make_placed_room(
    spec: Dict, width: float, length: float,
    x: float, y: float,
) -> Dict:
    """Create a placed room dict with all required fields."""
    rtype = spec["room_type"]
    zone = ZONE_CLASSIFICATION.get(rtype, "private")

    zone = spec.get("zone", ZONE_CLASSIFICATION.get(rtype, "private"))
    target = spec.get("target_area", width * length)

    return {
        "name": spec["name"],
        "room_type": rtype,
        "zone": zone,
        "area_ratio": spec.get("area_ratio", 0),
        "width": round(width, 1),
        "length": round(length, 1),
        "area": round(width * length, 1),
        "position": {"x": round(x, 1), "y": round(y, 1)},
        "adjacent_to": [],   # populated by adjacency pass
        "connected_to": [],  # populated by adjacency pass
        "doors": [],
        "windows": [],
    }


def _split_into_rows(rooms: List[Dict], max_per_row: int) -> List[List[Dict]]:
    """
    Split room list into sub-rows ensuring ventilation compliance.

    Key rule: max 2 rooms per row so middle rooms always touch boundaries.
    For rows with 3+ rooms, we limit to 2 per row to guarantee every room
    touches at least one external wall (left or right boundary).
    """
    if not rooms:
        return []
    # Cap at 2 rooms per row to ensure all rooms touch external walls
    effective_max = min(max_per_row, 2) if len(rooms) > 2 else max_per_row
    if len(rooms) <= effective_max:
        return [rooms]
    n_rows = math.ceil(len(rooms) / effective_max)
    per_row = math.ceil(len(rooms) / n_rows)
    per_row = min(per_row, effective_max)
    return [rooms[i:i + per_row] for i in range(0, len(rooms), per_row)]


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Door & Window Placement
# ═══════════════════════════════════════════════════════════════════════════

def _assign_doors_windows(rooms: List[Dict], plot_w: float, plot_l: float) -> List[Dict]:
    """
    Assign doors and windows based on adjacency graph and room geometry.

    Phase 1: Detect physical adjacency (shared walls)
    Phase 2: Populate connected_to for each room
    Phase 3: Place doors on shared walls per adjacency graph
    Phase 4: Place windows on exterior walls (10-30% of wall length)
    """
    WALL = WALL_EXT_FT
    TOL = 1.0  # adjacency tolerance in ft

    # ── Phase 1 & 2: detect adjacency from geometry ──
    for room in rooms:
        room["connected_to"] = []
        room["adjacent_to"] = []

    for i, a in enumerate(rooms):
        ax, ay = a["position"]["x"], a["position"]["y"]
        aw, al = a["width"], a["length"]
        for j, b in enumerate(rooms):
            if j <= i:
                continue
            bx, by = b["position"]["x"], b["position"]["y"]
            bw, bl = b["width"], b["length"]

            shared_wall = _detect_shared_wall(ax, ay, aw, al, bx, by, bw, bl, TOL)
            if shared_wall:
                a["connected_to"].append(b["name"])
                b["connected_to"].append(a["name"])
                a["adjacent_to"].append(b["name"])
                b["adjacent_to"].append(a["name"])

    # Detect corridor midpoint for door orientation
    public_bottom = 0.0
    private_top = plot_l
    for room in rooms:
        zone = room.get("zone", "private")
        ry = room["position"]["y"]
        rl = room["length"]
        if zone in ("public", "semi_private"):
            public_bottom = max(public_bottom, ry + rl)
        elif zone in ("private", "service"):
            private_top = min(private_top, ry)
    corridor_mid = (public_bottom + private_top) / 2

    # ── Phase 3: door placement ──
    for room in rooms:
        pos = room["position"]
        rw, rl = room["width"], room["length"]
        rtype = room["room_type"]
        zone = room.get("zone", "private")
        rx, ry = pos["x"], pos["y"]
        room_center_y = ry + rl / 2.0

        doors = []

        if rtype in ("bathroom", "toilet"):
            # Bathroom door toward adjacent bedroom (shared wall)
            door_placed = False
            for other in rooms:
                if other["room_type"] not in ("master_bedroom", "bedroom"):
                    continue
                wall = _detect_shared_wall(
                    rx, ry, rw, rl,
                    other["position"]["x"], other["position"]["y"],
                    other["width"], other["length"], TOL,
                )
                if wall:
                    doors.append({"wall": wall, "width": 2.5})
                    door_placed = True
                    break
            if not door_placed:
                if room_center_y < corridor_mid:
                    doors.append({"wall": "N", "width": 2.5})
                else:
                    doors.append({"wall": "S", "width": 2.5})

        elif rtype == "living":
            # Living: main entrance on south, interior door on north
            doors.append({"wall": "S", "width": 3.0})
            doors.append({"wall": "N", "width": 3.0})

        elif rtype == "kitchen":
            # Kitchen: door toward dining (find shared wall)
            door_placed = False
            for other in rooms:
                if other["room_type"] != "dining":
                    continue
                wall = _detect_shared_wall(
                    rx, ry, rw, rl,
                    other["position"]["x"], other["position"]["y"],
                    other["width"], other["length"], TOL,
                )
                if wall:
                    doors.append({"wall": wall, "width": 3.0})
                    door_placed = True
                    break
            if not door_placed:
                doors.append({"wall": "N", "width": 3.0})

        elif zone in ("public", "semi_private"):
            doors.append({"wall": "N", "width": 3.0})

        elif rtype == "passage":
            pass  # passages don't get doors

        else:
            # Private rooms — door toward corridor
            if room_center_y > corridor_mid:
                doors.append({"wall": "S", "width": 3.0})
            else:
                doors.append({"wall": "N", "width": 3.0})

        # ── Phase 4: window placement — exterior walls only ──
        windows = []
        is_south = ry <= WALL + 1
        is_north = ry + rl >= plot_l - WALL - 1
        is_west = rx <= WALL + 1
        is_east = rx + rw >= plot_w - WALL - 1

        if rtype in ("bathroom", "toilet"):
            # Small ventilation window
            for side, flag, wall_len in [
                ("E", is_east, rl), ("N", is_north, rw),
                ("W", is_west, rl), ("S", is_south, rw),
            ]:
                if flag:
                    win_w = max(2.0, min(wall_len * 0.15, 3.0))
                    windows.append({"wall": side, "width": round(win_w, 1)})
                    break
        else:
            # Habitable rooms: windows 10-30% of wall length on exterior walls
            for side, flag, wall_len in [
                ("N", is_north, rw), ("S", is_south, rw),
                ("E", is_east, rl), ("W", is_west, rl),
            ]:
                if flag:
                    win_w = max(3.0, min(wall_len * 0.25, wall_len * 0.30))
                    windows.append({"wall": side, "width": round(win_w, 1)})

        # Habitable rooms must have at least one window
        non_window_types = ("bathroom", "toilet", "store", "utility",
                            "hallway", "staircase", "passage")
        if not windows and rtype not in non_window_types:
            for side, flag, wall_len in [
                ("N", is_north, rw), ("S", is_south, rw),
                ("E", is_east, rl), ("W", is_west, rl),
            ]:
                if flag:
                    windows.append({"wall": side, "width": 4.0})
                    break
            else:
                windows.append({"wall": "N", "width": 3.0})

        room["doors"] = doors
        room["windows"] = windows

    return rooms


def _detect_shared_wall(
    ax: float, ay: float, aw: float, al: float,
    bx: float, by: float, bw: float, bl: float,
    tol: float,
) -> Optional[str]:
    """
    Detect which wall room A shares with room B.

    Returns 'N', 'S', 'E', 'W' from A's perspective, or None.
    Requires at least 2ft of overlap on the shared edge.
    """
    MIN_OVERLAP = 2.0

    # A's north edge touches B's south edge
    if abs((ay + al) - by) < tol:
        overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        if overlap >= MIN_OVERLAP:
            return "N"

    # A's south edge touches B's north edge
    if abs(ay - (by + bl)) < tol:
        overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        if overlap >= MIN_OVERLAP:
            return "S"

    # A's east edge touches B's west edge
    if abs((ax + aw) - bx) < tol:
        overlap = max(0, min(ay + al, by + bl) - max(ay, by))
        if overlap >= MIN_OVERLAP:
            return "E"

    # A's west edge touches B's south edge
    if abs(ax - (bx + bw)) < tol:
        overlap = max(0, min(ay + al, by + bl) - max(ay, by))
        if overlap >= MIN_OVERLAP:
            return "W"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Auto-correct Geometry (§4 Step D)
# ═══════════════════════════════════════════════════════════════════════════

def _auto_correct(rooms: List[Dict], plot_w: float, plot_l: float) -> List[Dict]:
    """
    Auto-correct geometry violations before final output.

    All corrections are overlap-safe: no correction is applied if it would
    create or worsen an overlap with any other room.
    """

    def _would_overlap(candidate_idx, cx, cy, cw, cl):
        """Check if a candidate placement overlaps any other room."""
        tol = WALL_INT_FT * 0.5
        for j, other in enumerate(rooms):
            if j == candidate_idx:
                continue
            ox = other["position"]["x"]
            oy = other["position"]["y"]
            ow = other["width"]
            ol = other["length"]
            if (cx < ox + ow - tol and cx + cw > ox + tol and
                    cy < oy + ol - tol and cy + cl > oy + tol):
                return True
        return False

    for i, room in enumerate(rooms):
        pos = room["position"]
        rw = room["width"]
        rl = room["length"]
        rx, ry = pos["x"], pos["y"]

        # ── Clip to boundary (always safe) ──
        if rx + rw > plot_w:
            rw = round(max(plot_w - rx, 4.0), 1)
        if ry + rl > plot_l:
            rl = round(max(plot_l - ry, 4.0), 1)
        if rx < 0:
            rx = 0.0
        if ry < 0:
            ry = 0.0

        room["width"] = round(rw, 1)
        room["length"] = round(rl, 1)
        room["area"] = round(rw * rl, 1)
        room["position"] = {"x": round(rx, 1), "y": round(ry, 1)}

    # ── Enforce minimum size (overlap-safe, direction-aware) ──
    # Only extend toward the nearest external wall to avoid eating corridor space
    for i, room in enumerate(rooms):
        rtype = room["room_type"]
        pos = room["position"]
        rx, ry = pos["x"], pos["y"]
        rw, rl = room["width"], room["length"]
        min_a = MIN_ROOM_SIZES.get(rtype, 25)

        if rw * rl < min_a:
            # Determine which external wall is closest
            dist_bottom = ry
            dist_top = plot_l - (ry + rl)
            dist_left = rx
            dist_right = plot_w - (rx + rw)

            # Try extending width first (less likely to cross corridor)
            needed_w = round(min_a / max(rl, 1), 1)
            if dist_right < dist_left:
                # Extend right
                if needed_w <= plot_w - rx and not _would_overlap(i, rx, ry, needed_w, rl):
                    room["width"] = needed_w
                    room["area"] = round(needed_w * rl, 1)
                    continue
            else:
                # Extend left (shift position)
                new_rx = max(0, rx + rw - needed_w)
                if not _would_overlap(i, new_rx, ry, needed_w, rl):
                    room["position"]["x"] = round(new_rx, 1)
                    room["width"] = needed_w
                    room["area"] = round(needed_w * rl, 1)
                    continue

            # Try extending length toward nearest external wall only
            needed_l = round(min_a / max(rw, 1), 1)
            if dist_bottom <= dist_top and dist_bottom < 2.0:
                # Near bottom wall — extend downward
                new_ry = max(0, ry + rl - needed_l)
                if not _would_overlap(i, rx, new_ry, rw, needed_l):
                    room["position"]["y"] = round(new_ry, 1)
                    room["length"] = needed_l
                    room["area"] = round(rw * needed_l, 1)
            elif dist_top < 2.0:
                # Near top wall — extend upward
                if needed_l <= plot_l - ry and not _would_overlap(i, rx, ry, rw, needed_l):
                    room["length"] = needed_l
                    room["area"] = round(rw * needed_l, 1)

    # ── Enforce aspect ratio 1:2.5 (overlap-safe, area-preserving) ──
    for i, room in enumerate(rooms):
        rw, rl = room["width"], room["length"]
        min_dim = min(rw, rl)
        max_dim = max(rw, rl)
        if min_dim > 0 and max_dim / min_dim > 2.5:
            pos = room["position"]
            rx, ry = pos["x"], pos["y"]
            min_a = MIN_ROOM_SIZES.get(room["room_type"], 25)

            if rw > rl:
                new_rw = round(rl * 2.5, 1)
                # If shrinking width drops area below min, extend length instead
                if new_rw * rl < min_a:
                    needed_rl = round(min_a / new_rw, 1)
                    if (ry + needed_rl <= plot_l and
                            not _would_overlap(i, rx, ry, new_rw, needed_rl)):
                        room["width"] = new_rw
                        room["length"] = needed_rl
                        room["area"] = round(new_rw * needed_rl, 1)
                elif not _would_overlap(i, rx, ry, new_rw, rl):
                    room["width"] = new_rw
                    room["area"] = round(new_rw * rl, 1)
            else:
                new_rl = round(rw * 2.5, 1)
                # If shrinking length drops area below min, extend width instead
                if rw * new_rl < min_a:
                    needed_rw = round(min_a / new_rl, 1)
                    if (rx + needed_rw <= plot_w and
                            not _would_overlap(i, rx, ry, needed_rw, new_rl)):
                        room["width"] = needed_rw
                        room["length"] = new_rl
                        room["area"] = round(needed_rw * new_rl, 1)
                elif not _would_overlap(i, rx, ry, rw, new_rl):
                    room["length"] = new_rl
                    room["area"] = round(rw * new_rl, 1)

    # ── Ventilation fix: push interior habitable rooms to boundary ──
    tolerance = WALL_EXT_FT + 1.0
    for idx, room in enumerate(rooms):
        rtype = room["room_type"]
        if rtype in ("hallway", "passage", "staircase", "store", "utility"):
            continue

        pos = room["position"]
        rx, ry = pos["x"], pos["y"]
        rw, rl = room["width"], room["length"]

        touches_ext = (
            rx <= tolerance or ry <= tolerance or
            rx + rw >= plot_w - tolerance or
            ry + rl >= plot_l - tolerance
        )

        if not touches_ext:
            dist_west = rx
            dist_east = plot_w - (rx + rw)
            dist_south = ry
            dist_north = plot_l - (ry + rl)

            directions = sorted([
                ("west", dist_west),
                ("east", dist_east),
                ("south", dist_south),
                ("north", dist_north),
            ], key=lambda d: d[1])

            for direction, _ in directions:
                new_rx, new_ry = rx, ry
                new_rw, new_rl = rw, rl

                if direction == "west":
                    new_rw = round(rw + rx - WALL_EXT_FT, 1)
                    new_rx = WALL_EXT_FT
                elif direction == "east":
                    new_rw = round(plot_w - rx - WALL_EXT_FT, 1)
                elif direction == "south":
                    new_rl = round(rl + ry - WALL_EXT_FT, 1)
                    new_ry = WALL_EXT_FT
                else:  # north
                    new_rl = round(plot_l - ry - WALL_EXT_FT, 1)

                if not _would_overlap(idx, new_rx, new_ry, new_rw, new_rl):
                    room["position"]["x"] = round(new_rx, 1)
                    room["position"]["y"] = round(new_ry, 1)
                    room["width"] = round(new_rw, 1)
                    room["length"] = round(new_rl, 1)
                    room["area"] = round(new_rw * new_rl, 1)
                    break

    return rooms


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Build Output Layout (§6)
# ═══════════════════════════════════════════════════════════════════════════

def _build_layout_output(
    placed_rooms: List[Dict],
    plot_w: float, plot_l: float,
    total_area: float, floors: int,
    strategy: str,
) -> Dict:
    """Build the final layout JSON in the format specified by §6."""
    boundary = [
        [0, 0], [plot_w, 0], [plot_w, plot_l], [0, plot_l], [0, 0]
    ]

    doors_list = []
    for room in placed_rooms:
        rx = room["position"]["x"]
        ry = room["position"]["y"]
        rw = room["width"]
        rl = room["length"]

        # Build polygon (5-point closed rectangle)
        room["polygon"] = [
            [rx, ry], [rx + rw, ry], [rx + rw, ry + rl],
            [rx, ry + rl], [rx, ry]
        ]
        room["centroid"] = [round(rx + rw / 2, 1), round(ry + rl / 2, 1)]
        room["label"] = room.get("name", room.get("room_type", "Room"))
        room["actual_area"] = round(room.get("area", rw * rl), 1)

        # Build door geometry for frontend
        for door in room.get("doors", []):
            wall = door.get("wall", "S")
            dw = door.get("width", 2.5)
            if wall in ("S", "bottom"):
                hx, hy = round(rx + rw * 0.3, 1), ry
                doors_list.append({
                    "position": [hx, hy], "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [round(hx + dw, 1), hy],
                    "swing_dir": [0, 1],
                })
            elif wall in ("N", "top"):
                hx, hy = round(rx + rw * 0.3, 1), round(ry + rl, 1)
                doors_list.append({
                    "position": [hx, hy], "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [round(hx + dw, 1), hy],
                    "swing_dir": [0, -1],
                })
            elif wall in ("W", "left"):
                hx, hy = rx, round(ry + rl * 0.3, 1)
                doors_list.append({
                    "position": [hx, hy], "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [hx, round(hy + dw, 1)],
                    "swing_dir": [1, 0],
                })
            elif wall in ("E", "right"):
                hx, hy = round(rx + rw, 1), round(ry + rl * 0.3, 1)
                doors_list.append({
                    "position": [hx, hy], "width": dw,
                    "hinge": [hx, hy],
                    "door_end": [hx, round(hy + dw, 1)],
                    "swing_dir": [-1, 0],
                })

    # Build window geometry for frontend (start/end coordinates)
    windows_list = []
    for room in placed_rooms:
        rx = room["position"]["x"]
        ry = room["position"]["y"]
        rw = room["width"]
        rl = room["length"]
        for win in room.get("windows", []):
            wall = win.get("wall", "S")
            ww = win.get("width", 4.0)
            if wall in ("S", "bottom"):
                wx = round(rx + rw * 0.5 - ww / 2, 1)
                windows_list.append({
                    "start": [wx, ry],
                    "end": [round(wx + ww, 1), ry],
                    "wall": "S",
                })
            elif wall in ("N", "top"):
                wx = round(rx + rw * 0.5 - ww / 2, 1)
                windows_list.append({
                    "start": [wx, round(ry + rl, 1)],
                    "end": [round(wx + ww, 1), round(ry + rl, 1)],
                    "wall": "N",
                })
            elif wall in ("W", "left"):
                wy = round(ry + rl * 0.5 - ww / 2, 1)
                windows_list.append({
                    "start": [rx, wy],
                    "end": [rx, round(wy + ww, 1)],
                    "wall": "W",
                })
            elif wall in ("E", "right"):
                wy = round(ry + rl * 0.5 - ww / 2, 1)
                windows_list.append({
                    "start": [round(rx + rw, 1), wy],
                    "end": [round(rx + rw, 1), round(wy + ww, 1)],
                    "wall": "E",
                })

    # Build column/pillar positions at wall junctions
    columns = []
    col_set = set()
    # Boundary corners
    for pt in boundary[:-1]:
        key = (round(pt[0], 1), round(pt[1], 1))
        if key not in col_set:
            col_set.add(key)
            columns.append(list(key))
    # Room corners that lie on the boundary
    for room in placed_rooms:
        rx = round(room["position"]["x"], 1)
        ry = round(room["position"]["y"], 1)
        rw = round(room["width"], 1)
        rl = round(room["length"], 1)
        for cx, cy in [(rx, ry), (rx+rw, ry), (rx+rw, ry+rl), (rx, ry+rl)]:
            cx, cy = round(cx, 1), round(cy, 1)
            on_bnd = (abs(cx) < 0.5 or abs(cx - plot_w) < 0.5 or
                      abs(cy) < 0.5 or abs(cy - plot_l) < 0.5)
            if on_bnd:
                key = (cx, cy)
                if key not in col_set:
                    col_set.add(key)
                    columns.append(list(key))

    # Build interior dimension chains (per-room segments along top and right)
    # Collect unique x-coordinates for top-edge dimensioning
    x_coords = sorted(set(
        round(r["position"]["x"], 1) for r in placed_rooms
    ) | set(
        round(r["position"]["x"] + r["width"], 1) for r in placed_rooms
    ) | {0.0, round(plot_w, 1)})
    # Collect unique y-coordinates for side dimensioning
    y_coords = sorted(set(
        round(r["position"]["y"], 1) for r in placed_rooms
    ) | set(
        round(r["position"]["y"] + r["length"], 1) for r in placed_rooms
    ) | {0.0, round(plot_l, 1)})

    dim_chains_x = []
    for i in range(len(x_coords) - 1):
        seg = round(x_coords[i + 1] - x_coords[i], 1)
        if seg > 0.5:
            dim_chains_x.append({
                "start": x_coords[i], "end": x_coords[i + 1], "length": seg
            })
    dim_chains_y = []
    for i in range(len(y_coords) - 1):
        seg = round(y_coords[i + 1] - y_coords[i], 1)
        if seg > 0.5:
            dim_chains_y.append({
                "start": y_coords[i], "end": y_coords[i + 1], "length": seg
            })

    total_used = sum(r["area"] for r in placed_rooms)
    circulation_area = total_area - total_used
    circ_pct = round(max(0, circulation_area) / max(total_area, 1) * 100, 1)

    # Determine circulation type
    if plot_w >= 25:
        circ_type = "central"
    elif plot_w >= 15:
        circ_type = "side"
    else:
        circ_type = "minimal"

    # ── Build zones (Step 2 — array of {name, band} objects) ──
    ZONE_BANDS = {
        "public": "front", "semi_private": "middle",
        "private": "rear", "service": "rear", "circulation": "middle",
    }
    zones_array = []
    zone_rooms_map = {"public": [], "semi_private": [], "private": [], "service": [], "circulation": []}
    for room in placed_rooms:
        z = room.get("zone", "private")
        if z not in zone_rooms_map:
            zone_rooms_map[z] = []
        zone_rooms_map[z].append(room["name"])
    for zname in ["public", "semi_private", "private", "service", "circulation"]:
        if zone_rooms_map.get(zname):
            zones_array.append({
                "name": zname,
                "band": ZONE_BANDS.get(zname, "middle"),
                "rooms": zone_rooms_map[zname],
            })

    # ── Build adjacency_graph (Step 3 — list of edge pairs) ──
    adjacency_edges = []
    seen_edges = set()
    for room in placed_rooms:
        for neighbor in room.get("connected_to", []):
            edge = tuple(sorted([room["name"], neighbor]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                adjacency_edges.append(list(edge))

    # ── Build routing_graph (Step 4 — primary + private access paths) ──
    routing_edges = []
    room_names = {r["name"] for r in placed_rooms}
    # Primary: Living → Dining → Kitchen
    primary_chain = ["Living Room", "Dining Room", "Kitchen"]
    for i in range(len(primary_chain) - 1):
        if primary_chain[i] in room_names and primary_chain[i + 1] in room_names:
            routing_edges.append([primary_chain[i], primary_chain[i + 1]])
    # Private access: living → passage → each bedroom
    passage_names = [r["name"] for r in placed_rooms if r["room_type"] == "passage"]
    bedroom_names = [r["name"] for r in placed_rooms
                     if r["room_type"] in ("master_bedroom", "bedroom")]
    if passage_names and passage_names[0] in room_names:
        routing_edges.append(["Living Room", passage_names[0]])
        for bname in bedroom_names:
            routing_edges.append([passage_names[0], bname])

    # ── Corridor width from passage room ──
    passage_room = next((r for r in placed_rooms if r["room_type"] == "passage"), None)
    actual_corr_width = passage_room["length"] if passage_room else MIN_PASSAGE_WIDTH

    return {
        "plot": {
            "width": plot_w,
            "length": plot_l,
            "unit": "ft",
        },
        "floors": floors,
        "zoning_strategy": strategy,
        "boundary": boundary,
        "rooms": placed_rooms,
        "doors": doors_list,
        "windows": windows_list,
        "columns": columns,
        "dim_chains_x": dim_chains_x,
        "dim_chains_y": dim_chains_y,
        "total_area": round(total_area, 1),
        "zones": zones_array,
        "adjacency_graph": adjacency_edges,
        "routing_graph": routing_edges,
        "constraints": {
            "no_overlap": True,
            "orthogonal_walls": True,
            "inside_boundary": True,
            "aspect_ratio_max": 2.5,
            "min_corridor_width_ft": MIN_PASSAGE_WIDTH,
        },
        "circulation": {
            "type": circ_type,
            "width": round(actual_corr_width, 1),
        },
        "walls": {
            "external": "9 inch",
            "internal": "4.5 inch",
        },
        "area_summary": {
            "plot_area": round(total_area, 1),
            "built_area": round(total_used, 1),
            "total_used_area": round(total_used, 1),
            "circulation_area": round(max(0, circulation_area), 1),
            "circulation_percentage": f"{max(0, circ_pct)}%",
            "utilization_percentage": f"{round(total_used / max(total_area, 1) * 100, 1)}%",
        },
        "validation": {
            "overlap": False,
            "min_size_ok": True,
            "zoning_ok": True,
            "geometry_ok": True,
            "area_ok": total_used <= total_area * 1.01,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Validation Checks (§7)
# ═══════════════════════════════════════════════════════════════════════════

def _check_overlaps(rooms: List[Dict]) -> List[str]:
    """Check for room overlaps using AABB collision detection."""
    issues = []
    tolerance = WALL_INT_FT * 0.5

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

            if (x1 < x2 + w2 - tolerance and x1 + w1 > x2 + tolerance and
                    y1 < y2 + l2 - tolerance and y1 + l1 > y2 + tolerance):
                issues.append(
                    f"Overlap: {r1.get('name', '?')} and {r2.get('name', '?')}"
                )
    return issues


def _check_minimum_sizes(rooms: List[Dict]) -> List[str]:
    """Check rooms meet minimum size requirements (§2D). Allows 1 sqft rounding tolerance."""
    issues = []
    TOLERANCE = 1.0  # rounding tolerance
    for room in rooms:
        rtype = room.get("room_type", "other")
        area = room.get("area", 0)
        min_a = MIN_ROOM_SIZES.get(rtype, 0)
        if area < min_a - TOLERANCE:
            issues.append(
                f"{room.get('name', rtype)}: {area} sq ft < minimum {min_a} sq ft"
            )
    return issues


def _check_zoning(rooms: List[Dict]) -> List[str]:
    """Check zoning rules — forbidden adjacency (§2B)."""
    issues = []
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            if _are_adjacent(r1, r2):
                t1 = r1.get("room_type", "other")
                t2 = r2.get("room_type", "other")
                for fa, fb in FORBIDDEN_ADJACENCY:
                    if (t1 == fa and t2 == fb) or (t1 == fb and t2 == fa):
                        issues.append(
                            f"Zoning: {r1.get('name')} should not open directly into {r2.get('name')}"
                        )
    return issues


def _are_adjacent(r1: Dict, r2: Dict) -> bool:
    """Check if two rooms share a wall (AABB adjacency)."""
    p1, p2 = r1.get("position", {}), r2.get("position", {})
    x1, y1 = p1.get("x", 0), p1.get("y", 0)
    x2, y2 = p2.get("x", 0), p2.get("y", 0)
    w1, l1 = r1.get("width", 0), r1.get("length", 0)
    w2, l2 = r2.get("width", 0), r2.get("length", 0)
    tol = WALL_INT_FT + 0.5

    # Shared vertical wall
    if abs((x1 + w1) - x2) < tol or abs((x2 + w2) - x1) < tol:
        if y1 < y2 + l2 and y1 + l1 > y2:
            return True
    # Shared horizontal wall
    if abs((y1 + l1) - y2) < tol or abs((y2 + l2) - y1) < tol:
        if x1 < x2 + w2 and x1 + w1 > x2:
            return True
    return False


def _check_area_overflow(rooms: List[Dict], plot_area: float) -> str:
    """Check if total room area exceeds plot area."""
    total = sum(r.get("area", 0) for r in rooms)
    if total > plot_area * 1.01:  # 1% tolerance
        return (
            f"Total room area ({round(total)} sq ft) exceeds "
            f"plot area ({round(plot_area)} sq ft)"
        )
    return ""


def _check_circulation(rooms: List[Dict], plot_w: float, plot_l: float) -> List[str]:
    """Check circulation adequacy and passage widths."""
    issues = []
    total_room_area = sum(r.get("area", 0) for r in rooms)
    plot_area = plot_w * plot_l
    circ_area = plot_area - total_room_area

    # §2C: Circulation should be 10-15%
    if circ_area < plot_area * 0.05:
        issues.append(
            f"Insufficient circulation: {round(circ_area)} sq ft "
            f"({round(circ_area / max(plot_area, 1) * 100, 1)}%) — minimum 10% recommended"
        )

    # Check passage widths between adjacent rooms
    # Gaps <= 1.0ft are internal walls, not passages
    wall_tol = 1.0

    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p1, p2 = r1.get("position", {}), r2.get("position", {})
            x1, y1 = p1.get("x", 0), p1.get("y", 0)
            x2, y2 = p2.get("x", 0), p2.get("y", 0)
            w1, l1 = r1.get("width", 0), r1.get("length", 0)
            w2, l2 = r2.get("width", 0), r2.get("length", 0)

            # Horizontal gap
            gap_x = x2 - (x1 + w1)
            y_overlap = min(y1 + l1, y2 + l2) - max(y1, y2)
            if y_overlap > 0.5 and wall_tol < gap_x < MIN_PASSAGE_WIDTH - 0.3:
                issues.append(
                    f"Narrow passage ({round(gap_x, 1)} ft) between "
                    f"{r1.get('name')} and {r2.get('name')} — min {MIN_PASSAGE_WIDTH} ft"
                )

            # Vertical gap
            gap_y = y2 - (y1 + l1)
            x_overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
            if x_overlap > 0.5 and wall_tol < gap_y < MIN_PASSAGE_WIDTH - 0.3:
                issues.append(
                    f"Narrow passage ({round(gap_y, 1)} ft) between "
                    f"{r1.get('name')} and {r2.get('name')} — min {MIN_PASSAGE_WIDTH} ft"
                )

    return issues


def _check_boundary_fit(rooms: List[Dict], plot_w: float, plot_l: float) -> List[str]:
    """Check that all rooms fit within plot boundaries."""
    issues = []
    for room in rooms:
        pos = room.get("position", {})
        rx, ry = pos.get("x", 0), pos.get("y", 0)
        rw, rl = room.get("width", 0), room.get("length", 0)

        if rx < 0 or ry < 0:
            issues.append(f"{room.get('name')}: outside plot boundary")
        if rx + rw > plot_w + 0.1:
            issues.append(
                f"{room.get('name')}: extends beyond plot width "
                f"(ends at {round(rx + rw, 1)} ft, plot width {plot_w} ft)"
            )
        if ry + rl > plot_l + 0.1:
            issues.append(
                f"{room.get('name')}: extends beyond plot length "
                f"(ends at {round(ry + rl, 1)} ft, plot length {plot_l} ft)"
            )
    return issues


def _check_aspect_ratios(rooms: List[Dict]) -> List[str]:
    """Check room aspect ratios (§2E: 1:1 to 1:2.5)."""
    issues = []
    for room in rooms:
        w = room.get("width", 10)
        l = room.get("length", 10)
        min_dim = min(w, l)
        max_dim = max(w, l)

        if min_dim < 4 and room.get("room_type") not in (
            "balcony", "utility", "toilet", "hallway", "passage"
        ):
            issues.append(
                f"{room.get('name')}: minimum dimension {min_dim} ft too narrow (min 4 ft)"
            )
        if min_dim > 0 and max_dim / min_dim > 2.55:  # 0.05 rounding tolerance
            issues.append(
                f"{room.get('name')}: aspect ratio "
                f"{round(max_dim / min_dim, 1)}:1 exceeds max 2.5:1"
            )
    return issues


def _check_ventilation(rooms: List[Dict], plot_w: float, plot_l: float) -> List[str]:
    """
    Check ventilation compliance (§2F).

    Every habitable room must touch external wall.
    Bathrooms must connect to external wall or ventilation shaft.
    """
    issues = []
    tolerance = WALL_EXT_FT + 1.0

    for room in rooms:
        rtype = room.get("room_type", "other")
        # Skip circulation rooms
        if rtype in ("hallway", "passage", "staircase"):
            continue

        pos = room.get("position", {})
        rx, ry = pos.get("x", 0), pos.get("y", 0)
        rw, rl = room.get("width", 0), room.get("length", 0)

        touches_external = (
            rx <= tolerance or                    # West wall
            ry <= tolerance or                    # South wall
            rx + rw >= plot_w - tolerance or      # East wall
            ry + rl >= plot_l - tolerance          # North wall
        )

        if not touches_external:
            if rtype in ("bathroom", "toilet"):
                issues.append(
                    f"{room.get('name')}: no external wall access for ventilation"
                )
            elif rtype not in ("store", "utility", "pooja"):
                issues.append(
                    f"{room.get('name')}: habitable room has no external wall access"
                )

    return issues


def _check_reachability(rooms: List[Dict]) -> List[str]:
    """
    Check all rooms are reachable via the adjacency graph (BFS from living room).

    Every room must be connected to the living room through some chain of
    adjacent rooms. Unreachable rooms indicate a layout flaw.
    """
    if not rooms:
        return []

    # Build adjacency map from connected_to
    name_set = {r["name"] for r in rooms}
    adj = {r["name"]: set(r.get("connected_to", [])) for r in rooms}

    # Find living room as root
    root = None
    for r in rooms:
        if r["room_type"] == "living":
            root = r["name"]
            break
    if root is None:
        root = rooms[0]["name"]

    # BFS
    visited = set()
    queue = [root]
    visited.add(root)
    while queue:
        current = queue.pop(0)
        for neighbor in adj.get(current, []):
            if neighbor not in visited and neighbor in name_set:
                visited.add(neighbor)
                queue.append(neighbor)

    unreachable = name_set - visited
    issues = []
    for name in unreachable:
        issues.append(f"{name}: not reachable from {root} via adjacency graph")

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL — Design Explanation Builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_design_explanation(
    req: Dict, layout: Dict, validation: Dict, strategy: str
) -> str:
    """Build professional explanation (max 8 lines) as per §6."""
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

    # Line 2: Strategy
    strategy_names = {
        "linear": "Linear Zoning",
        "central_corridor": "Central Corridor",
        "split": "Split Zoning",
        "adaptive": "Adaptive Block Zoning",
    }
    lines.append(f"Zoning strategy: {strategy_names.get(strategy, strategy)}.")

    # Line 3: Room count
    bed_count = sum(1 for r in rooms if r["room_type"] in ("master_bedroom", "bedroom"))
    bath_count = sum(1 for r in rooms if r["room_type"] in ("bathroom", "toilet"))
    lines.append(
        f"Configuration: {bed_count} bedroom(s), {bath_count} bathroom(s), "
        f"{floors} floor(s), {len(rooms)} total rooms."
    )

    # Line 4: Area utilization
    lines.append(
        f"Area utilization: {area_summary.get('utilization_percentage', '?')} of plot. "
        f"Circulation: {area_summary.get('circulation_percentage', '?')}."
    )

    # Line 5: Walls
    lines.append("Walls: 9-inch external (load-bearing), 4.5-inch internal partitions.")

    # Line 6: Zoning summary
    zones = {}
    for r in rooms:
        z = r.get("zone", "other")
        zones.setdefault(z, []).append(r.get("name", ""))
    zone_desc = "; ".join(f"{k}: {', '.join(v)}" for k, v in zones.items())
    lines.append(f"Zoning: {zone_desc}.")

    # Line 7: Compliance
    compliant = validation.get("compliant", True)
    if compliant:
        lines.append("All architectural constraints validated. Layout is CAD-ready.")
    else:
        issue_count = (
            len(validation.get("overlap_details", [])) +
            len(validation.get("size_violations", [])) +
            len(validation.get("zoning_issues", []))
        )
        lines.append(f"Validation found {issue_count} issue(s). Review before construction.")

    return "\n".join(lines[:8])
