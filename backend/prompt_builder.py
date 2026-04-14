"""
NakshaNirman Prompt Builder — Converts form data into a concise
user message for the local LLM.
"""
from __future__ import annotations

# Minimum usable area needed per BHK
MIN_AREA = {1: 280, 2: 480, 3: 680, 4: 900}


def build_user_prompt(data: dict) -> str:
    """
    Build a user-facing prompt from the frontend form payload.

    Expected keys in *data*:
        plot_width, plot_length, bedrooms, facing, extras,
        family_type, city, state, vastu, elder_friendly,
        work_from_home, notes, bathrooms_target, kitchen_preference
    """
    pw = float(data.get("plot_width", 30))
    pl = float(data.get("plot_length", 40))
    bedrooms = int(data.get("bedrooms", 2))
    facing = str(data.get("facing", "east")).lower()
    extras = data.get("extras", [])
    family_type = str(data.get("family_type", "nuclear")).lower()
    city = str(data.get("city", "") or "").strip()
    state = str(data.get("state", "") or "").strip()
    vastu = data.get("vastu", True)
    elder_friendly = data.get("elder_friendly", False)
    work_from_home = data.get("work_from_home", False)
    notes = str(data.get("notes", "") or "").strip()
    bathrooms_target = int(data.get("bathrooms_target", 0) or 0)
    kitchen_preference = str(data.get("kitchen_preference", "semi_open"))
    floors = int(data.get("floors", 1) or 1)
    parking_slots = int(data.get("parking_slots", 0) or 0)
    vastu_priority = int(data.get("vastu_priority", 3) or 3)
    natural_light_priority = int(data.get("natural_light_priority", 3) or 3)
    privacy_priority = int(data.get("privacy_priority", 3) or 3)
    storage_priority = int(data.get("storage_priority", 3) or 3)
    strict_real_life = bool(data.get("strict_real_life", False))
    must_have = [str(x).strip().lower().replace(" ", "_") for x in data.get("must_have", []) if str(x).strip()]
    avoid = [str(x).strip().lower().replace(" ", "_") for x in data.get("avoid", []) if str(x).strip()]

    # Usable area check
    uw = pw - 7.0
    ul = pl - 11.5
    usable_area = uw * ul
    min_needed = MIN_AREA.get(bedrooms, 900)

    parts = []

    # Core specification
    parts.append(f"{pw:.0f}x{pl:.0f} plot, {facing}-facing, {bedrooms}BHK")

    # Family type
    if family_type == "joint":
        parts.append("joint family (corridor 4ft wide, elder room ground floor)")
    elif family_type == "couple":
        parts.append("working couple")
    else:
        parts.append("nuclear family")

    # Location
    location_parts = []
    if city:
        location_parts.append(city)
    if state:
        location_parts.append(state)
    if location_parts:
        parts.append(", ".join(location_parts))

    # Extras
    if isinstance(extras, list) and extras:
        parts.append(f"Extra rooms: {', '.join(extras)}")
    else:
        parts.append("Extra rooms: none")

    # Bathrooms
    if bathrooms_target and bathrooms_target > 0:
        parts.append(f"Target bathrooms: {bathrooms_target}")
    else:
        parts.append(f"Target bathrooms: {max(1, bedrooms)}")

    # Kitchen preference
    if kitchen_preference and kitchen_preference != "semi_open":
        parts.append(f"Kitchen: {kitchen_preference.replace('_', ' ')}")

    # Vastu
    if vastu is False or vastu == "false":
        parts.append("Vastu compliance: not required")

    # Special requirements
    if elder_friendly:
        parts.append("Elder-friendly: all rooms ground floor, no steps, wide corridors")
    if work_from_home:
        parts.append("Work-from-home: include study/office room")
    if floors > 1:
        parts.append("Multi-floor home: include staircase access")
    if parking_slots > 0:
        parts.append(f"Parking requirement: {parking_slots} slot(s)")

    # Priority profile
    priority_tags = []
    if vastu_priority >= 4:
        priority_tags.append("strong_vastu")
    if natural_light_priority >= 4:
        priority_tags.append("daylight")
    if privacy_priority >= 4:
        priority_tags.append("privacy")
    if storage_priority >= 4:
        priority_tags.append("storage")
    if priority_tags:
        parts.append(f"High priorities: {', '.join(priority_tags)}")

    if must_have:
        parts.append(f"Must-have program constraints: {', '.join(sorted(set(must_have)))}")
    if avoid:
        parts.append(f"Avoid constraints: {', '.join(sorted(set(avoid)))}")
    if strict_real_life:
        parts.append("Strict practical mode: do not randomize room program; follow all constraints before aesthetics")

    # Real-world flow constraints used by stronger planning models.
    parts.append(
        "Real-life checks: maintain privacy gradient (public->service->private), "
        "kitchen near dining, and at least one common bathroom reachable without crossing bedrooms"
    )

    # Notes
    if notes:
        parts.append(f"Design notes: {notes}")

    # Area feasibility warning
    if usable_area < min_needed:
        max_feasible = max(
            (bhk for bhk, area in MIN_AREA.items() if usable_area >= area),
            default=1,
        )
        parts.append(
            f"WARNING: Usable area {usable_area:.0f} sqft is below minimum "
            f"{min_needed} sqft for {bedrooms}BHK. Maximum feasible: {max_feasible}BHK. "
            f"Generate {max_feasible}BHK instead."
        )

    parts.append("Output JSON only.")

    return ". ".join(parts)


def build_system_prompt(data: dict) -> str:
    """
    Build a system prompt.  For local Ollama, the llm.py module overrides
    this with NAKSHA_SYSTEM_PROMPT, but this function is kept for
    interface compatibility.
    """
    return (
        "You are NAKSHA-MASTER, an expert Indian residential floor plan "
        "architect. Generate geometrically correct, Vastu-compliant JSON "
        "floor plans. Output only valid JSON."
    )
