"""Chat agent service for collecting requirements and emitting GENERATE_PLAN token."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app_config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_ENABLED,
    OPENROUTER_TEXT_MAX_TOKENS,
    OPENROUTER_TEXT_MODEL,
    OPENROUTER_TEXT_TEMPERATURE,
    OPENROUTER_VERIFY_SSL,
)
from services.naksha_prompt import get_naksha_master_system_prompt

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = get_naksha_master_system_prompt()


YES_TOKENS = {"yes", "y", "generate", "go ahead", "proceed", "ok", "okay", "sure", "haan", "han", "done"}
EXTRA_TOKENS = ["pooja", "study", "store", "balcony", "garage"]
_DEEPSEEK_CHAT_FALLBACK_MODEL = "deepseek/deepseek-chat"


def _round_half(value: float) -> float:
    return round(float(value) * 2.0) / 2.0


def _plot_class(width: float, length: float) -> str:
    ratio = float(width) / max(float(length), 0.01)
    if ratio < 0.65:
        return "narrow-deep"
    if ratio > 1.5:
        return "wide-shallow"
    if 0.75 <= ratio <= 1.25:
        return "near-square"
    return "elongated"


def _setbacks(total_area: float) -> Dict[str, float]:
    if total_area < 750:
        front, rear, left, right = 4.0, 3.0, 2.0, 2.0
    elif total_area <= 1800:
        front, rear, left, right = 6.5, 5.0, 3.5, 3.5
    else:
        front, rear, left, right = 10.0, 5.0, 5.0, 5.0

    return {
        "front": front,
        "rear": rear,
        "left": left,
        "right": right,
    }


def _build_collecting_mode_reply(collected: Dict[str, Any]) -> Dict[str, Any]:
    width = collected.get("plot_width")
    length = collected.get("plot_length")
    total_area = collected.get("total_area")

    if width and length and not total_area:
        total_area = round(float(width) * float(length), 1)

    missing: List[str] = []
    if not (width and length):
        missing.append("plot_size")
    if not collected.get("bedrooms"):
        missing.append("bedrooms")

    if "plot_size" in missing:
        question = "Please share your plot size in feet, for example 30x40."
    else:
        question = "How many bedrooms do you want: 1BHK, 2BHK, 3BHK, or 4BHK?"

    understood_parts: List[str] = []
    if width and length:
        understood_parts.append(f"Plot appears to be {width}x{length} ft")
    if collected.get("bedrooms"):
        understood_parts.append(f"You prefer {int(collected['bedrooms'])}BHK")
    if collected.get("facing"):
        understood_parts.append(f"Facing is {str(collected['facing']).lower()}")

    context_understood = "; ".join(understood_parts) if understood_parts else "I am collecting your plot and BHK requirement."

    return {
        "mode": "collecting",
        "collected_so_far": {
            "plot_width": float(width) if width else None,
            "plot_length": float(length) if length else None,
            "total_area": float(total_area) if total_area else None,
            "facing": str(collected.get("facing") or "").lower() or None,
            "bedrooms": int(collected["bedrooms"]) if collected.get("bedrooms") else None,
            "bathrooms": int(collected["bathrooms"]) if collected.get("bathrooms") else None,
            "floors": 1,
            "extras": list(collected.get("extras") or []),
        },
        "missing": missing,
        "question": question,
        "context_understood": context_understood,
    }


def _room_spec(
    room_id: str,
    room_type: str,
    display_name: str,
    zone: str,
    band: int,
    priority: int,
    width: float,
    depth: float,
    min_area: float,
    max_area: float,
    vastu_zone: str,
    must_touch_exterior: bool,
    adjacent_must: List[str],
    adjacent_avoid: List[str],
    preferred_walls: List[str],
    window_walls: List[str],
    door_walls: List[str],
    notes: str,
) -> Dict[str, Any]:
    preferred_width = _round_half(width)
    preferred_depth = _round_half(depth)
    target_area = int(round(preferred_width * preferred_depth))
    return {
        "id": room_id,
        "type": room_type,
        "display_name": display_name,
        "zone": zone,
        "band": int(band),
        "priority": int(priority),
        "target_area": target_area,
        "min_area": int(round(min_area)),
        "max_area": int(round(max_area)),
        "preferred_width": preferred_width,
        "preferred_depth": preferred_depth,
        "min_width": max(8.0, _round_half(preferred_width - 2.0)),
        "min_depth": max(7.0, _round_half(preferred_depth - 2.0)),
        "max_aspect_ratio": 2.0,
        "vastu_zone": vastu_zone,
        "must_touch_exterior": bool(must_touch_exterior),
        "adjacent_must": adjacent_must,
        "adjacent_avoid": adjacent_avoid,
        "preferred_walls": preferred_walls,
        "window_walls": window_walls,
        "door_walls": door_walls,
        "notes": notes,
    }


def _build_designing_mode_reply(payload: Dict[str, Any]) -> Dict[str, Any]:
    width = float(payload.get("plot_width") or 30.0)
    length = float(payload.get("plot_length") or 40.0)
    total_area = float(payload.get("total_area") or (width * length))
    facing = str(payload.get("facing") or "east").lower()
    bedrooms = int(payload.get("bedrooms") or 2)
    bathrooms = int(payload.get("bathrooms") or max(1, bedrooms))
    extras = list(payload.get("extras") or [])
    vastu_enabled = bool(payload.get("vastu", True))

    aspect_ratio = round(width / max(length, 0.01), 2)
    plot_class = _plot_class(width, length)
    strategy_map = {
        "narrow-deep": ("linear_spine", "horizontal"),
        "wide-shallow": ("lateral_band", "vertical"),
        "near-square": ("cluster_hub", "horizontal"),
        "elongated": ("central_corridor", "horizontal"),
    }
    strategy_name, primary_spine = strategy_map.get(plot_class, ("cluster_hub", "horizontal"))

    sb = _setbacks(total_area)
    usable_width = _round_half(max(10.0, width - sb["left"] - sb["right"]))
    usable_length = _round_half(max(10.0, length - sb["front"] - sb["rear"]))
    usable_area = round(usable_width * usable_length, 1)

    band_public = _round_half(max(8.0, usable_length * 0.34))
    band_service = _round_half(max(7.0, usable_length * 0.27))
    band_private = _round_half(max(7.0, usable_length - band_public - band_service))

    rooms: List[Dict[str, Any]] = [
        _room_spec(
            "living_01",
            "living",
            "Drawing Room",
            "public",
            1,
            10,
            16,
            12,
            100,
            280,
            "east_or_north",
            True,
            ["dining_01", "entrance"],
            ["master_bathroom_01", "bathroom_01"],
            ["east", "north"],
            ["east", "north"],
            ["west", "south"],
            "Sofa seating faces the TV wall with clear guest circulation from entrance to dining.",
        ),
        _room_spec(
            "dining_01",
            "dining",
            "Dining",
            "public",
            1,
            9,
            10,
            10,
            90,
            140,
            "center_or_east",
            True,
            ["living_01", "kitchen_01"],
            ["bedroom_02", "bedroom_03"],
            ["east", "north"],
            ["east"],
            ["west", "south"],
            "Six-seater dining is positioned between living and kitchen for daily family flow.",
        ),
        _room_spec(
            "kitchen_01",
            "kitchen",
            "Kitchen",
            "service",
            2,
            10,
            10,
            9,
            50,
            140,
            "south_east_preferred",
            True,
            ["dining_01"],
            ["pooja_01", "bathroom_01"],
            ["east", "south"],
            ["east", "south"],
            ["north", "west"],
            "L-counter kitchen with hob on south wall and prep near east window for daylight.",
        ),
        _room_spec(
            "master_bedroom_01",
            "master_bedroom",
            "Master Bedroom",
            "private",
            3,
            10,
            12,
            12,
            130 if total_area > 600 else 120,
            220,
            "south_west_preferred",
            True,
            ["master_bathroom_01"],
            ["entrance", "living_01"],
            ["south", "west"],
            ["south", "west"],
            ["north", "east"],
            "King bed on south wall with wardrobe on west side and private access to attached bathroom.",
        ),
        _room_spec(
            "master_bathroom_01",
            "bathroom",
            "Master Bathroom",
            "service",
            2,
            8,
            5,
            8,
            35,
            60,
            "west_or_south",
            True,
            ["master_bedroom_01"],
            ["kitchen_01", "pooja_01"],
            ["west", "south"],
            ["west"],
            ["east"],
            "Attached bath includes shower, WC and basin with external ventilation.",
        ),
    ]

    for index in range(2, bedrooms + 1):
        rooms.append(
            _room_spec(
                f"bedroom_{index:02d}",
                "bedroom",
                f"Bedroom {index - 1}",
                "private",
                3,
                8,
                10,
                11,
                100,
                170,
                "north_west_or_west",
                True,
                ["bathroom_01"],
                ["entrance"],
                ["north", "west"],
                ["north", "west"],
                ["south"],
                "Bed is aligned to maintain privacy, with study table near the window wall.",
            )
        )

    common_needed = max(0, bathrooms - 1)
    for index in range(1, common_needed + 1):
        rooms.append(
            _room_spec(
                f"bathroom_{index:02d}",
                "bathroom",
                f"Common Bathroom {index}",
                "service",
                2,
                7,
                5,
                7,
                30,
                55,
                "west_or_north_west",
                True,
                ["bedroom_02" if bedrooms >= 2 else "living_01"],
                ["kitchen_01", "pooja_01"],
                ["west", "north"],
                ["west"],
                ["east"],
                "Common bathroom is centrally reachable while avoiding direct visibility from living.",
            )
        )

    for extra in extras:
        token = str(extra).strip().lower()
        if token == "pooja":
            rooms.append(
                _room_spec(
                    "pooja_01",
                    "pooja",
                    "Pooja Room",
                    "public",
                    1,
                    8,
                    6,
                    5,
                    25,
                    45,
                    "north_east",
                    True,
                    ["living_01"],
                    ["bathroom_01", "master_bathroom_01", "kitchen_01"],
                    ["north", "east"],
                    ["east"],
                    ["south"],
                    "Altar wall stays in north-east with standing prayer space for family use.",
                )
            )
        elif token == "study":
            rooms.append(
                _room_spec(
                    "study_01",
                    "study",
                    "Study",
                    "private",
                    3,
                    7,
                    10,
                    10,
                    64,
                    120,
                    "north_or_east",
                    True,
                    ["bedroom_02" if bedrooms >= 2 else "master_bedroom_01"],
                    ["kitchen_01"],
                    ["north", "east"],
                    ["north", "east"],
                    ["south"],
                    "Desk faces north light with bookshelf on interior wall for low glare work zone.",
                )
            )
        elif token == "store":
            rooms.append(
                _room_spec(
                    "store_01",
                    "store",
                    "Store Room",
                    "service",
                    2,
                    5,
                    6,
                    5,
                    25,
                    50,
                    "north_west",
                    False,
                    ["kitchen_01"],
                    ["living_01"],
                    ["west"],
                    ["none"],
                    ["east"],
                    "Dry storage close to kitchen for groceries and utility items.",
                )
            )
        elif token == "balcony":
            rooms.append(
                _room_spec(
                    "balcony_01",
                    "balcony",
                    "Balcony",
                    "public",
                    1,
                    4,
                    8,
                    5,
                    32,
                    60,
                    "front_facing",
                    True,
                    ["living_01"],
                    ["bathroom_01"],
                    ["front"],
                    ["open"],
                    ["rear"],
                    "Balcony connects to living for morning use and natural daylight.",
                )
            )
        elif token == "garage":
            rooms.append(
                _room_spec(
                    "garage_01",
                    "garage",
                    "Garage",
                    "public",
                    1,
                    6,
                    18,
                    10,
                    160,
                    220,
                    "north_west_or_south_east",
                    True,
                    ["entrance"],
                    ["pooja_01"],
                    ["front"],
                    ["front"],
                    ["rear"],
                    "Single-car bay with direct road-facing shutter and safe pedestrian side access.",
                )
            )

    room_area_total = int(round(sum(float(r.get("target_area") or 0.0) for r in rooms)))
    built_up_area = min(room_area_total, int(round(usable_area * 0.92)))
    carpet_area = int(round(built_up_area * 0.9))
    coverage_ratio = round((built_up_area / max(usable_area, 1.0)), 2)

    vastu_score = 8 if vastu_enabled else 6
    compromised_rooms: List[str] = []
    compromise_reason = ""
    recommendations = [
        "Keep north-east zone light and uncluttered.",
        "Avoid toilet placement in north-east corner.",
    ]
    if facing == "south":
        compromised_rooms.append("main_entrance")
        compromise_reason = "South-facing road requires careful entrance balancing."
        recommendations.append("Use a Vastu-friendly threshold treatment and keep entrance well-lit.")
        vastu_score = max(5, vastu_score - 1)

    return {
        "mode": "designing",
        "plot": {
            "width": _round_half(width),
            "depth": _round_half(length),
            "total_area": int(round(total_area)),
            "facing": facing,
            "plot_shape": plot_class,
            "road_side": facing,
            "aspect_ratio": aspect_ratio,
            "plot_class": plot_class,
        },
        "setbacks": {
            "front": sb["front"],
            "rear": sb["rear"],
            "left": sb["left"],
            "right": sb["right"],
            "usable_width": usable_width,
            "usable_length": usable_length,
            "usable_area": usable_area,
        },
        "design_strategy": {
            "name": strategy_name,
            "reason": f"{plot_class} plot profile supports {strategy_name} for balanced public, service, and private zoning.",
            "band_public_depth": band_public,
            "band_service_depth": band_service,
            "band_private_depth": band_private,
            "primary_spine": primary_spine,
            "entrance_position": f"center_{facing}",
            "circulation_type": "hub" if strategy_name == "cluster_hub" else "linear",
        },
        "rooms": rooms,
        "doors": [
            {
                "id": "door_main",
                "type": "main_entrance",
                "width": 4.0,
                "from": "exterior",
                "to": "living_01",
                "wall": facing,
                "position": "center",
                "notes": "Primary entry connects road-facing side directly to living room.",
            }
        ],
        "windows": [
            {
                "room_id": "living_01",
                "wall": "east" if facing != "east" else "north",
                "width": 5.0,
                "count": 2,
                "type": "sliding",
                "sill_height": 3.0,
            },
            {
                "room_id": "kitchen_01",
                "wall": "east",
                "width": 4.0,
                "count": 1,
                "type": "sliding",
                "sill_height": 3.0,
            },
        ],
        "vastu": {
            "score": vastu_score,
            "facing": facing,
            "main_door_direction": f"{facing}_center",
            "main_door_vastu": "acceptable" if facing != "south" else "compromise",
            "compliant_rooms": [
                "Kitchen aligned toward south-east preference",
                "Master bedroom aligned toward south-west preference",
            ],
            "compromised_rooms": compromised_rooms,
            "compromise_reason": compromise_reason,
            "recommendations": recommendations,
        },
        "structural": {
            "external_wall_thickness_ft": 0.75,
            "internal_wall_thickness_ft": 0.375,
            "column_grid_ft": 12.0,
            "column_positions": "External corners and major wall junctions",
        },
        "area_summary": {
            "plot_area": int(round(total_area)),
            "usable_area": int(round(usable_area)),
            "built_up_area": int(round(built_up_area)),
            "carpet_area": int(round(carpet_area)),
            "coverage_ratio": coverage_ratio,
            "rooms_count": len(rooms),
        },
        "circulation": {
            "entry_sequence": "Road -> Main entrance -> Living -> Dining -> Corridor -> Bedrooms",
            "service_entry": "Kitchen can receive a rear-side service access",
            "privacy_gradient": "Public front to service middle to private rear",
            "bottlenecks": "None identified in this conceptual program stage",
        },
        "architect_note": (
            f"This {bedrooms}BHK concept uses a {strategy_name} layout for a {plot_class} plot. "
            "Public spaces are placed near entry, service functions are centralized for plumbing efficiency, "
            "and private bedrooms are buffered deeper in the layout for privacy."
        ),
    }


def _extract_collected(text: str, seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = dict(seed or {})
    t = text.lower()

    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×*by]\s*(\d+(?:\.\d+)?)", t)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        data["plot_width"] = min(a, b)
        data["plot_length"] = max(a, b)

    sqm = re.search(r"(\d+(?:\.\d+)?)\s*(sq\.?\s*ft|sqft|square\s*feet)", t)
    if sqm and "plot_width" not in data:
        area = float(sqm.group(1))
        w = (area * 0.75) ** 0.5
        l = area / w
        data["plot_width"] = round(w, 1)
        data["plot_length"] = round(l, 1)

    bhk = re.search(r"([1-4])\s*bhk|([1-4])\s*bed", t)
    if bhk:
        beds = int(bhk.group(1) or bhk.group(2))
        data["bedrooms"] = beds

    baths = re.search(r"([1-6])\s*bath", t)
    if baths:
        data["bathrooms"] = int(baths.group(1))

    face = re.search(r"\b(east|west|north|south)\b", t)
    if face:
        data["facing"] = face.group(1)

    if "vastu" in t:
        if any(tok in t for tok in ["no vastu", "without vastu", "vastu no", "vastu off"]):
            data["vastu"] = False
        else:
            data["vastu"] = True

    extras = set(data.get("extras", []))
    for token in EXTRA_TOKENS:
        if token in t:
            extras.add(token)
    if extras:
        data["extras"] = sorted(extras)

    constraints = list(data.get("placement_constraints", []))
    if any(tok in t for tok in ["kitchen near back", "kitchen near rear", "kitchen near garden", "kitchen at back garden"]):
        constraints.append(
            {
                "room": "kitchen",
                "intent": "rear_garden_preference",
                "band": 3,
                "prefer_walls": ["north", "rear"],
                "note": "Kitchen preferred near rear garden side",
            }
        )
    if any(tok in t for tok in ["privacy for master", "more privacy for master", "master more private"]):
        constraints.append(
            {
                "room": "master_bedroom",
                "intent": "privacy_buffer",
                "forbid_adjacent": ["living"],
                "note": "Master should not directly open/abut living room",
            }
        )
    if constraints:
        dedup = []
        seen = set()
        for c in constraints:
            key = (c.get("room"), c.get("intent"), tuple(c.get("forbid_adjacent", [])), tuple(c.get("prefer_walls", [])))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(c)
        data["placement_constraints"] = dedup

    return data


def _confirmed_to_generate(text: str) -> bool:
    t = text.strip().lower()
    return any(tok in t for tok in YES_TOKENS)


def _build_generate_payload(collected: Dict[str, Any]) -> Dict[str, Any]:
    bedrooms = int(collected.get("bedrooms") or 2)
    bathrooms = int(collected.get("bathrooms") or bedrooms)
    payload = {
        "plot_width": round(float(collected["plot_width"]), 1),
        "plot_length": round(float(collected["plot_length"]), 1),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "facing": str(collected.get("facing") or "east").lower(),
        "vastu": bool(collected.get("vastu", True)),
        "extras": list(collected.get("extras", [])),
        "placement_constraints": list(collected.get("placement_constraints", [])),
    }
    return payload


def _from_history(history: List[Dict[str, str]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for turn in history[-10:]:
        merged = _extract_collected(turn.get("content", ""), merged)
    return merged


def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _infer_payload_from_design_json(design_obj: Dict[str, Any], fallback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plot = design_obj.get("plot") or {}
    rooms = design_obj.get("rooms") or []

    width = plot.get("width")
    length = plot.get("depth") if "depth" in plot else plot.get("length")
    area = plot.get("total_area")

    if width is None or length is None:
        width = fallback.get("plot_width")
        length = fallback.get("plot_length")

    if area is None and width and length:
        area = float(width) * float(length)

    bedroom_types = {"bedroom", "master_bedroom"}
    bathroom_types = {"bathroom", "toilet", "master_bathroom", "common_bathroom"}
    extra_types = {"pooja", "study", "store", "balcony", "garage"}

    room_types = [str((r or {}).get("type") or "").strip().lower() for r in rooms if isinstance(r, dict)]
    bedrooms = sum(1 for t in room_types if t in bedroom_types)
    bathrooms = sum(1 for t in room_types if t in bathroom_types)
    extras = sorted({t for t in room_types if t in extra_types})

    if bedrooms <= 0:
        bedrooms = int(fallback.get("bedrooms") or 0)
    if bathrooms <= 0:
        bathrooms = int(fallback.get("bathrooms") or bedrooms or 0)

    if not width or not length or bedrooms <= 0:
        return None

    facing = str(plot.get("facing") or plot.get("road_side") or fallback.get("facing") or "east").lower()
    vastu_enabled = bool(fallback.get("vastu", True))
    vastu_obj = design_obj.get("vastu")
    if isinstance(vastu_obj, dict) and "score" in vastu_obj:
        try:
            vastu_enabled = float(vastu_obj.get("score") or 0) >= 5.0
        except Exception:
            pass

    return {
        "plot_width": round(float(width), 1),
        "plot_length": round(float(length), 1),
        "total_area": round(float(area), 1) if area else round(float(width) * float(length), 1),
        "bedrooms": int(bedrooms),
        "bathrooms": max(1, int(bathrooms)),
        "facing": facing,
        "vastu": bool(vastu_enabled),
        "extras": extras,
        "placement_constraints": list(fallback.get("placement_constraints") or []),
    }


def _deepseek_chat_model_name() -> str:
    token = str(OPENROUTER_TEXT_MODEL or "").strip().lower()
    if token.endswith(":free"):
        token = token[:-5]
    if token.startswith("deepseek/"):
        return token
    return _DEEPSEEK_CHAT_FALLBACK_MODEL


async def _call_openrouter(prompt: str, history: List[Dict[str, str]]) -> Optional[str]:
    if not OPENROUTER_ENABLED or not OPENROUTER_API_KEY:
        return None

    model_name = _deepseek_chat_model_name()

    def _request() -> Optional[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": model_name,
            "messages": messages,
            "temperature": max(0.0, min(1.2, float(OPENROUTER_TEXT_TEMPERATURE))),
            "max_tokens": max(400, min(int(OPENROUTER_TEXT_MAX_TOKENS), 3500)),
        }
        req = urllib.request.Request(
            url=f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "NakshaNirman",
            },
            method="POST",
        )

        ssl_context = None
        if not OPENROUTER_VERIFY_SSL:
            ssl_context = ssl._create_unverified_context()

        with urllib.request.urlopen(req, timeout=30, context=ssl_context) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))

        choices = result.get("choices") or []
        if not choices:
            return None
        text = ((choices[0] or {}).get("message") or {}).get("content")
        return text.strip() if isinstance(text, str) and text.strip() else None

    try:
        return await asyncio.to_thread(_request)
    except urllib.error.HTTPError as exc:
        logger.warning("OpenRouter chat HTTP failure: %s", exc)
        return None
    except Exception as exc:
        logger.warning("OpenRouter chat failed: %s", exc)
        return None


async def chat_reply(user_message: str, history: List[Dict[str, str]]) -> str:
    """
    Produce assistant reply.

    Priority:
    1) Deterministic local flow for guaranteed token behavior.
    2) Optional LLM rephrase when API key is available.
    """
    collected = _from_history(history)
    collected = _extract_collected(user_message, collected)

    has_plot = collected.get("plot_width") and collected.get("plot_length")
    has_bhk = collected.get("bedrooms")

    if has_plot and has_bhk and _confirmed_to_generate(user_message):
        payload = _build_generate_payload(collected)
        return "GENERATE_PLAN: " + json.dumps(payload, separators=(",", ":"))

    if has_plot and has_bhk:
        llm = await _call_openrouter(user_message, history)
        if llm:
            cleaned = llm.strip()
            if "GENERATE_PLAN:" in cleaned:
                token = cleaned.split("GENERATE_PLAN:", 1)[1].strip().splitlines()[0].strip()
                return f"GENERATE_PLAN: {token}"
            llm_json = _extract_json_dict(llm)
            if llm_json:
                return json.dumps(llm_json, indent=2)

        payload = _build_generate_payload(collected)
        return json.dumps(_build_designing_mode_reply(payload), indent=2)

    collecting_fallback = json.dumps(_build_collecting_mode_reply(collected), indent=2)

    # Try DeepSeek polish, but keep deterministic fallback for reliability.
    llm = await _call_openrouter(user_message, history)

    if llm:
        cleaned = llm.strip()
        if "GENERATE_PLAN:" in cleaned:
            token = cleaned.split("GENERATE_PLAN:", 1)[1].strip().splitlines()[0].strip()
            # Guardrail: emit token only when we truly have enough data.
            if has_plot and has_bhk:
                return f"GENERATE_PLAN: {token}"
            return collecting_fallback
        llm_json = _extract_json_dict(cleaned)
        if llm_json:
            return json.dumps(llm_json, indent=2)
        return collecting_fallback

    return collecting_fallback
