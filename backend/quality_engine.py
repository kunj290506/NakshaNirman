"""
NakshaNirman Quality Engine.

Evaluates whether a generated layout is practical for real-life use,
not just geometrically valid.
"""
from __future__ import annotations

import math
from typing import Any


CORE_ROOM_TYPES = ("living", "dining", "kitchen")
BEDROOM_TYPES = ("master_bedroom", "bedroom")
BATHROOM_TYPES = ("master_bath", "bathroom", "toilet")
ROOM_TOKEN_ALIASES = {
    "puja": "pooja",
    "mandir": "pooja",
    "office": "study",
    "home_office": "study",
    "guest_room": "bedroom",
    "guest_bedroom": "bedroom",
    "common_bath": "bathroom",
    "common_bathroom": "bathroom",
    "wc": "bathroom",
    "stairs": "staircase",
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_room_list(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rooms = plan.get("rooms", [])
    if not isinstance(rooms, list):
        return []
    return [r for r in rooms if isinstance(r, dict)]


def _type_counts(rooms: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for room in rooms:
        rtype = str(room.get("type", "")).strip().lower()
        if not rtype:
            continue
        counts[rtype] = counts.get(rtype, 0) + 1
    return counts


def _coverage_ratio(have: int, need: int) -> float:
    if need <= 0:
        return 1.0
    if have <= 0:
        return 0.0
    return min(1.0, have / need)


def _normalize_room_token(raw: Any) -> str:
    token = str(raw or "").strip().lower().replace(" ", "_")
    token = token.replace("-", "_")
    return ROOM_TOKEN_ALIASES.get(token, token)


def _normalize_token_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        items = str(raw or "").split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = _normalize_room_token(item)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _band_zone_alignment(rooms: list[dict[str, Any]]) -> float:
    """
    Approximate privacy/circulation quality by checking whether key room types
    are in expected zones or bands.
    """
    checks = 0
    ok = 0

    def room_ok(room: dict[str, Any], expected_zone: str, expected_band: int) -> bool:
        zone = str(room.get("zone", "")).strip().lower()
        band = _to_int(room.get("band"), 0)
        return zone == expected_zone or band == expected_band

    for room in rooms:
        rtype = str(room.get("type", "")).strip().lower()
        if rtype in ("living", "dining", "pooja", "foyer"):
            checks += 1
            ok += 1 if room_ok(room, "public", 1) else 0
        elif rtype in ("kitchen", "corridor", "bathroom", "master_bath", "utility", "store"):
            checks += 1
            ok += 1 if room_ok(room, "service", 2) else 0
        elif rtype in ("master_bedroom", "bedroom", "study", "balcony"):
            checks += 1
            ok += 1 if room_ok(room, "private", 3) else 0

    if checks == 0:
        return 0.5
    return ok / checks


def _utilization_ratio(usable_area: float, built_area: float) -> float:
    if usable_area <= 0:
        return 0.0
    util = built_area / usable_area
    if 0.55 <= util <= 0.85:
        return 1.0
    if 0.45 <= util < 0.55 or 0.85 < util <= 0.92:
        return 0.7
    if 0.35 <= util < 0.45 or 0.92 < util <= 1.0:
        return 0.45
    return 0.2


def evaluate_real_life_fit(plan: dict[str, Any], request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Score plan quality for practical residential use.

    Score is 0-100 and combines requirement coverage, circulation,
    zoning quality, and real-world usability signals.
    """
    rooms = _safe_room_list(plan)
    counts = _type_counts(rooms)

    bedrooms_total = max(1, _to_int(request_data.get("bedrooms"), 2))
    baths_target = max(0, _to_int(request_data.get("bathrooms_target"), 0))
    bathrooms_total = max(1, baths_target if baths_target > 0 else bedrooms_total)
    floors = max(1, _to_int(request_data.get("floors"), 1))

    # Score against likely per-floor program when multi-floor homes are requested.
    bedrooms_req = max(1, math.ceil(bedrooms_total / floors))
    bathrooms_req = max(1, math.ceil(bathrooms_total / floors))

    extras = request_data.get("extras", [])
    extras = [_normalize_room_token(x) for x in extras if str(x).strip()]
    must_have = _normalize_token_list(request_data.get("must_have", []))
    avoid = _normalize_token_list(request_data.get("avoid", []))

    # Convert high-level lifestyle requirements to expected program features.
    if bool(request_data.get("work_from_home")) and "study" not in extras:
        extras.append("study")
    if _to_int(request_data.get("parking_slots"), 0) > 0 and "garage" not in extras:
        extras.append("garage")
    if _to_int(request_data.get("floors"), 1) > 1 and "staircase" not in extras:
        extras.append("staircase")
    if str(request_data.get("family_type", "")).strip().lower() == "joint" and "utility" not in extras:
        extras.append("utility")

    bedroom_count = sum(counts.get(t, 0) for t in BEDROOM_TYPES)
    bathroom_count = sum(counts.get(t, 0) for t in BATHROOM_TYPES)

    core_present = sum(1 for t in CORE_ROOM_TYPES if counts.get(t, 0) > 0)
    core_ratio = core_present / len(CORE_ROOM_TYPES)

    extras_matched = sum(1 for extra in extras if counts.get(extra, 0) > 0)
    extras_ratio = 1.0 if not extras else extras_matched / len(extras)

    must_have_matched = sum(1 for token in must_have if counts.get(token, 0) > 0)
    must_have_required = 0 if not must_have else max(1, math.ceil(len(must_have) / floors))
    must_have_ratio = 1.0 if must_have_required == 0 else min(1.0, must_have_matched / must_have_required)
    avoid_violations = sum(1 for token in avoid if counts.get(token, 0) > 0)
    avoid_ratio = 1.0 if not avoid else max(0.0, 1.0 - (avoid_violations / len(avoid)))

    corridor_required = False
    corridor_present = counts.get("corridor", 0) > 0
    corridor_ratio = 1.0

    zoning_ratio = _band_zone_alignment(rooms)

    plot_width = _to_float(request_data.get("plot_width"), 30.0)
    plot_length = _to_float(request_data.get("plot_length"), 40.0)
    usable_width = max(0.0, plot_width - 7.0)
    usable_length = max(0.0, plot_length - 11.5)
    usable_area = usable_width * usable_length
    built_area = sum(max(0.0, _to_float(r.get("area"), _to_float(r.get("width")) * _to_float(r.get("height")))) for r in rooms)
    util_ratio = _utilization_ratio(usable_area, built_area)

    vastu_score = max(0.0, min(100.0, _to_float(plan.get("vastu_score"), 0.0)))

    weighted_score = (
        22.0 * _coverage_ratio(bedroom_count, bedrooms_req)
        + 16.0 * _coverage_ratio(bathroom_count, bathrooms_req)
        + 16.0 * core_ratio
        + 12.0 * extras_ratio
        + 12.0 * zoning_ratio
        + 8.0 * (vastu_score / 100.0)
        + 7.0 * util_ratio
        + 5.0 * must_have_ratio
        + 2.0 * avoid_ratio
    )
    score = int(round(max(0.0, min(100.0, weighted_score))))

    findings: list[str] = []
    opportunities: list[str] = []

    if bedroom_count < bedrooms_req:
        opportunities.append(f"Add {bedrooms_req - bedroom_count} more bedroom(s) to meet BHK target.")
    else:
        findings.append(f"Bedroom target met ({bedroom_count}/{bedrooms_req}).")

    if bathroom_count < bathrooms_req:
        opportunities.append(f"Bathroom count below target ({bathroom_count}/{bathrooms_req}).")
    else:
        findings.append(f"Bathroom target met ({bathroom_count}/{bathrooms_req}).")

    missing_core = [t for t in CORE_ROOM_TYPES if counts.get(t, 0) == 0]
    if missing_core:
        opportunities.append(f"Core program missing: {', '.join(missing_core)}.")
    else:
        findings.append("Core daily-use spaces (living/dining/kitchen) are present.")

    if corridor_present:
        findings.append("Corridor present for bedroom access and circulation.")

    if extras:
        missing_extras = [e for e in extras if counts.get(e, 0) == 0]
        if missing_extras:
            opportunities.append(f"Requested extras not fully covered: {', '.join(missing_extras)}.")
        else:
            findings.append("Requested lifestyle extras are covered.")

    if must_have:
        missing_must_have = [token for token in must_have if counts.get(token, 0) == 0]
        if must_have_ratio < 1.0:
            opportunities.append(
                "Must-have constraints are under-covered on this floor "
                f"({must_have_matched}/{must_have_required}); missing now: {', '.join(missing_must_have)}."
            )
        else:
            findings.append("All must-have constraints are satisfied.")

    if avoid and avoid_violations > 0:
        violated = [token for token in avoid if counts.get(token, 0) > 0]
        opportunities.append(f"Avoid constraints violated: {', '.join(violated)}.")

    utilization = 0.0 if usable_area <= 0 else built_area / usable_area
    if utilization < 0.4:
        opportunities.append("Low space utilization; rooms may feel sparse for the plot size.")
    elif utilization > 0.95:
        opportunities.append("High space utilization; add breathing room for circulation/daylight.")
    else:
        findings.append("Space utilization is in practical range for residential comfort.")

    if zoning_ratio < 0.6:
        opportunities.append("Improve public/service/private zoning separation for privacy.")

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"

    # Heuristic "alignment" scores inspired by strong model behavior patterns.
    chatgpt_like = int(round(
        100.0 * (
            0.30 * _coverage_ratio(bedroom_count, bedrooms_req)
            + 0.25 * core_ratio
            + 0.25 * (zoning_ratio)
            + 0.20 * extras_ratio
        )
    ))
    gemini_like = int(round(
        100.0 * (
            0.25 * _coverage_ratio(bedroom_count, bedrooms_req)
            + 0.20 * _coverage_ratio(bathroom_count, bathrooms_req)
            + 0.25 * zoning_ratio
            + 0.20 * util_ratio
            + 0.10 * (vastu_score / 100.0)
        )
    ))

    return {
        "score": score,
        "grade": grade,
        "required": {
            "bedrooms": bedrooms_req,
            "bathrooms": bathrooms_req,
            "bedrooms_total": bedrooms_total,
            "bathrooms_total": bathrooms_total,
            "floors": floors,
            "extras": extras,
            "must_have": must_have,
            "avoid": avoid,
            "must_have_required_on_floor": must_have_required,
            "corridor_required": corridor_required,
        },
        "actual": {
            "bedrooms": bedroom_count,
            "bathrooms": bathroom_count,
            "core_present": core_present,
            "extras_matched": extras_matched,
            "must_have_matched": must_have_matched,
            "must_have_required": must_have_required,
            "avoid_violations": avoid_violations,
            "corridor_present": corridor_present,
            "usable_area": round(usable_area, 1),
            "built_area": round(built_area, 1),
            "utilization": round(utilization, 3),
        },
        "coverage": {
            "bedroom": round(_coverage_ratio(bedroom_count, bedrooms_req), 3),
            "bathroom": round(_coverage_ratio(bathroom_count, bathrooms_req), 3),
            "core": round(core_ratio, 3),
            "extras": round(extras_ratio, 3),
            "must_have": round(must_have_ratio, 3),
            "avoid": round(avoid_ratio, 3),
            "zoning": round(zoning_ratio, 3),
        },
        "model_alignment": {
            "chatgpt_like": max(0, min(100, chatgpt_like)),
            "gemini_like": max(0, min(100, gemini_like)),
        },
        "findings": findings[:6],
        "opportunities": opportunities[:6],
    }


def build_real_life_architect_note(
    request_data: dict[str, Any],
    quality_report: dict[str, Any],
    base_note: str = "",
    advisory_strategy: str = "",
) -> str:
    """Build a concise, practical architect note for frontend display."""
    bedrooms = max(1, _to_int(request_data.get("bedrooms"), 2))
    facing = str(request_data.get("facing", "east")).strip().lower() or "east"
    family_type = str(request_data.get("family_type", "nuclear")).strip().lower() or "nuclear"
    plot_width = _to_float(request_data.get("plot_width"), 30.0)
    plot_length = _to_float(request_data.get("plot_length"), 40.0)

    summary_bits = [
        f"{bedrooms}BHK {facing}-facing layout for a {family_type} family on {plot_width:.0f}x{plot_length:.0f} plot",
        f"real-life fit score {quality_report.get('score', 0)}/100 ({quality_report.get('grade', 'C')})",
    ]

    if advisory_strategy:
        summary_bits.append(f"strategy: {advisory_strategy[:110]}")

    if base_note:
        summary_bits.append(base_note[:140])

    opportunities = quality_report.get("opportunities", [])
    if opportunities:
        summary_bits.append(f"next refinement: {str(opportunities[0])[:130]}")

    return ". ".join(summary_bits) + "."
