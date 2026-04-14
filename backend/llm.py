"""
NakshaNirman LLM — Local Ollama only.
GTX 1650 + 24GB RAM optimized.
No external API. No internet required after model download.
"""
from __future__ import annotations
import json
import logging
import re
import httpx
from typing import Any
from config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_PLAN_MODEL,
    LOCAL_LLM_BACKUP_MODEL,
    LOCAL_LLM_ADVISORY_MODEL,
)

log = logging.getLogger("llm")

# Timeouts — GTX 1650 with 4GB VRAM needs extra time for complex JSON.
# 7B model can take 2-4 minutes for floor plans (lots of structured tokens).
# Advisory is simpler (~30s).
PRIMARY_TIMEOUT = 300.0
BACKUP_TIMEOUT = 360.0
ADVISORY_TIMEOUT = 90.0

# ── The System Prompt ────────────────────────────────────────────────
NAKSHA_SYSTEM_PROMPT = (
    "You are NAKSHA-MASTER, a specialized AI architect for Indian residential "
    "floor plans. Output only JSON floor plans. No prose. No explanation. Only "
    "valid JSON.\n\n"
    "THREE LAWS — NEVER BREAK:\n\n"
    "LAW 1 ZERO OVERLAP: For every pair of rooms A,B: (A.x+A.width<=B.x) OR "
    "(B.x+B.width<=A.x) OR (A.y+A.height<=B.y) OR (B.y+B.height<=A.y). "
    "Verify before output.\n\n"
    "LAW 2 BOUNDARY: UW=plot_width-7.0, UL=plot_length-11.5. Every room: "
    "x>=0, y>=0, x+width<=UW, y+height<=UL.\n\n"
    "LAW 3 MINIMUMS (width x height feet): living 11x11, dining 8x8, kitchen "
    "7x8, master_bedroom 10x10, bedroom 9x9, master_bath 4.5x6, bathroom 4x5, "
    "optional corridor 3.5 wide, pooja 4x4, study 6x7, store 4x4.\n\n"
    "BAND STRUCTURE:\n"
    "Band1 height = max(11.0, UL*0.30) at y=0 [public: living,dining,pooja]\n"
    "Band2 height = max(8.0, UL*0.26) [service: kitchen,bathrooms,utility]\n"
    "Band3 height = UL-Band1-Band2 [private: bedrooms]\n\n"
    "CIRCULATION RULE: Prefer direct room adjacency and compact movement paths. "
    "Do NOT force a dedicated corridor unless explicitly requested by constraints.\n\n"
    "VASTU: Kitchen in right half Band2 (+8). Master_bedroom in left half "
    "Band3 (+7). Pooja in NE corner (+8). Start score at 55.\n\n"
    "OUTPUT SCHEMA:\n"
    '{"plot_boundary":[{"x":0,"y":0},{"x":UW,"y":0},{"x":UW,"y":UL},'
    '{"x":0,"y":UL}],"rooms":[{"id":"living_01","type":"living",'
    '"label":"Living Room","x":0.0,"y":0.0,"width":13.0,"height":9.5,'
    '"area":123.5,"zone":"public","band":1,"color":"#E8F5E9",'
    '"polygon":[{"x":0,"y":0},{"x":13,"y":0},{"x":13,"y":9.5},'
    '{"x":0,"y":9.5}]}],"doors":[],"windows":[],"metadata":{"bhk":2,'
    '"vastu_score":76,"adjacency_score":82,"architect_note":'
    '"Layout description.","vastu_issues":[]}}\n\n'
    "COLORS: living=#E8F5E9 dining=#FFF3E0 kitchen=#FFEBEE "
    "master_bedroom=#E3F2FD bedroom=#E3F2FD master_bath=#E0F7FA "
    "bathroom=#E0F7FA corridor=#F5F5F5 pooja=#FFF8E1 study=#EDE7F6 "
    "store=#EFEBE9 balcony=#E8F5E9 garage=#ECEFF1 utility=#F3E5F5 "
    "foyer=#FAFAFA staircase=#ECEFF1\n\n"
    "Output ONLY the JSON object. Nothing before it. Nothing after it."
)


def _extract_json(text: str) -> dict:
    """Pull valid JSON out of model output, handling common issues."""
    # Strip thinking blocks (some models output <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find outermost { }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"Model did not return valid JSON. Got: {text[:300]}")


def _build_generation_user_message(
    user_message: str,
    advisory: dict[str, Any] | None = None,
    request_data: dict[str, Any] | None = None,
) -> str:
    """Build a richer instruction payload for more realistic plans."""
    parts = [str(user_message).strip()]

    if isinstance(request_data, dict):
        bedrooms = int(request_data.get("bedrooms", 2) or 2)
        baths_target = int(request_data.get("bathrooms_target", 0) or 0)
        bathrooms = baths_target if baths_target > 0 else bedrooms
        family = str(request_data.get("family_type", "nuclear") or "nuclear")
        facing = str(request_data.get("facing", "east") or "east")
        extras = request_data.get("extras", [])
        extras = [str(x).strip() for x in extras if str(x).strip()]
        must_have = request_data.get("must_have", [])
        must_have = [str(x).strip() for x in must_have if str(x).strip()]
        avoid = request_data.get("avoid", [])
        avoid = [str(x).strip() for x in avoid if str(x).strip()]
        strict_real_life = bool(request_data.get("strict_real_life", False))

        priorities: list[str] = []
        if int(request_data.get("vastu_priority", 3) or 3) >= 4:
            priorities.append("strong_vastu")
        if int(request_data.get("natural_light_priority", 3) or 3) >= 4:
            priorities.append("daylight")
        if int(request_data.get("privacy_priority", 3) or 3) >= 4:
            priorities.append("privacy")
        if int(request_data.get("storage_priority", 3) or 3) >= 4:
            priorities.append("storage")

        lifestyle: list[str] = []
        if bool(request_data.get("work_from_home")):
            lifestyle.append("work_from_home_requires_study")
        if bool(request_data.get("elder_friendly")):
            lifestyle.append("elder_friendly_no_tight_corners")
        if int(request_data.get("parking_slots", 0) or 0) > 0:
            lifestyle.append("vehicle_parking_space_required")
        if int(request_data.get("floors", 1) or 1) > 1:
            lifestyle.append("staircase_required")

        parts.append("PRACTICAL REQUIREMENTS:")
        parts.append(f"- Target: {bedrooms}BHK, bathrooms={bathrooms}, family={family}, facing={facing}")
        parts.append("- Daily flow: living+dining near entry; kitchen adjacent to dining; private bedrooms separated from public zone")
        parts.append("- Ensure at least one bathroom is accessible from common circulation")
        if extras:
            parts.append(f"- Requested extras: {', '.join(extras)}")
        if must_have:
            parts.append(f"- Must-have constraints: {', '.join(must_have)}")
        if avoid:
            parts.append(f"- Avoid constraints: {', '.join(avoid)}")
        if priorities:
            parts.append(f"- High priorities: {', '.join(priorities)}")
        if lifestyle:
            parts.append(f"- Lifestyle constraints: {', '.join(lifestyle)}")
        if strict_real_life:
            parts.append("- Strict mode: prioritize practical compliance over stylistic variation")

    if isinstance(advisory, dict) and advisory:
        strategy = str(advisory.get("design_strategy", "") or "").strip()
        priority_order = advisory.get("priority_order", [])
        critical_checks = advisory.get("critical_checks", [])
        risks = advisory.get("risks", [])
        lifestyle_moves = advisory.get("lifestyle_moves", [])

        parts.append("ARCHITECT ADVISORY:")
        if strategy:
            parts.append(f"- Strategy: {strategy}")
        if isinstance(priority_order, list) and priority_order:
            parts.append(f"- Priority order: {', '.join(str(x) for x in priority_order[:6])}")
        if isinstance(critical_checks, list) and critical_checks:
            parts.append(f"- Critical checks: {', '.join(str(x) for x in critical_checks[:6])}")
        if isinstance(lifestyle_moves, list) and lifestyle_moves:
            parts.append(f"- Lifestyle moves: {', '.join(str(x) for x in lifestyle_moves[:6])}")
        if isinstance(risks, list) and risks:
            parts.append(f"- Risks to avoid: {', '.join(str(x) for x in risks[:4])}")

    parts.append(
        "Output JSON only. In metadata.architect_note, explain the real-life rationale in one concise sentence."
    )
    return "\n".join(part for part in parts if part)


def _normalize_plan(plan: dict, user_message: str) -> dict:
    """Ensure plan has all required fields with correct types."""
    if not isinstance(plan, dict):
        plan = {}

    # Extract dimensions from user message for fallback boundary
    dims = re.search(
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)", str(user_message)
    )
    uw = float(dims.group(1)) - 7.0 if dims else 23.0
    ul = float(dims.group(2)) - 11.5 if dims else 28.5

    # Ensure plot_boundary
    if (
        not isinstance(plan.get("plot_boundary"), list)
        or len(plan.get("plot_boundary", [])) < 3
    ):
        plan["plot_boundary"] = [
            {"x": 0, "y": 0},
            {"x": uw, "y": 0},
            {"x": uw, "y": ul},
            {"x": 0, "y": ul},
        ]

    # Ensure rooms is a list
    rooms = plan.get("rooms", [])
    if isinstance(rooms, dict):
        # Some models return rooms as a dict — convert
        rooms = [
            {"id": k, "type": k, "label": k.replace("_", " ").title(), **v}
            for k, v in rooms.items()
            if isinstance(v, dict)
        ]
    if not isinstance(rooms, list):
        rooms = []

    # Normalize each room
    COLOR_MAP = {
        "living": "#E8F5E9", "dining": "#FFF3E0", "kitchen": "#FFEBEE",
        "master_bedroom": "#E3F2FD", "bedroom": "#E3F2FD",
        "master_bath": "#E0F7FA", "bathroom": "#E0F7FA", "toilet": "#E0F7FA",
        "corridor": "#F5F5F5", "pooja": "#FFF8E1", "study": "#EDE7F6",
        "store": "#EFEBE9", "balcony": "#E8F5E9", "garage": "#ECEFF1",
        "utility": "#F3E5F5", "foyer": "#FAFAFA", "staircase": "#ECEFF1",
    }
    ZONE_DEFAULTS = {
        "living": ("public", 1),
        "dining": ("public", 1),
        "pooja": ("public", 1),
        "foyer": ("public", 1),
        "kitchen": ("service", 2),
        "bathroom": ("service", 2),
        "master_bath": ("service", 2),
        "toilet": ("service", 2),
        "corridor": ("service", 2),
        "utility": ("service", 2),
        "store": ("service", 2),
        "staircase": ("service", 2),
        "master_bedroom": ("private", 3),
        "bedroom": ("private", 3),
        "study": ("private", 3),
        "balcony": ("private", 3),
        "garage": ("service", 1),
    }

    def _to_num(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return default

    for i, room in enumerate(rooms):
        if not isinstance(room, dict):
            continue
        room.setdefault("id", f"room_{i+1:02d}")
        room.setdefault("type", "room")
        room["type"] = str(room.get("type", "room")).strip().lower() or "room"
        room.setdefault("label", room["type"].replace("_", " ").title())
        room["x"] = _to_num(room.get("x", 0.0), 0.0)
        room["y"] = _to_num(room.get("y", 0.0), 0.0)
        room["width"] = max(0.5, _to_num(room.get("width", 10.0), 10.0))
        room["height"] = max(0.5, _to_num(room.get("height", 10.0), 10.0))

        # Calculate area if missing
        w = _to_num(room.get("width", 0), 0.0)
        h = _to_num(room.get("height", 0), 0.0)
        room["area"] = round(_to_num(room.get("area", w * h), w * h), 1)

        # Fill color from type
        rtype = str(room.get("type", "room"))
        room.setdefault("color", COLOR_MAP.get(rtype, "#F5F5F5"))

        # Zone/band defaults
        default_zone, default_band = ZONE_DEFAULTS.get(rtype, ("service", 2))
        room.setdefault("zone", default_zone)
        room.setdefault("band", default_band)

        # Build polygon if missing
        if not isinstance(room.get("polygon"), list) or len(room.get("polygon", [])) < 3:
            x, y = float(room["x"]), float(room["y"])
            room["polygon"] = [
                {"x": x, "y": y},
                {"x": x + w, "y": y},
                {"x": x + w, "y": y + h},
                {"x": x, "y": y + h},
            ]

    plan["rooms"] = [r for r in rooms if isinstance(r, dict)]

    plan.setdefault("doors", [])
    plan.setdefault("windows", [])

    meta = plan.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("bhk", 2)
    meta.setdefault("vastu_score", 68)
    meta.setdefault("adjacency_score", 72)
    meta.setdefault(
        "architect_note",
        "Floor plan generated by NAKSHA-MASTER local model.",
    )
    meta.setdefault("vastu_issues", [])
    plan["metadata"] = meta

    # Flatten metadata fields to top level for frontend compatibility
    plan.setdefault("vastu_score", meta.get("vastu_score", 68))
    plan.setdefault("adjacency_score", meta.get("adjacency_score", 72))
    plan.setdefault("architect_note", meta.get("architect_note", ""))
    plan.setdefault("generation_method", "llm")

    return plan


async def _call_local(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: float,
    label: str = "local",
) -> str:
    """Make one HTTP call to local Ollama API. Returns raw text content."""
    url = f"{LOCAL_LLM_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.05,  # Very low temp for consistent JSON
        "max_tokens": 3072,  # Enough for floor plans, faster than 4096
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    log.info("%s: calling %s (timeout=%.0fs)", label, model, timeout)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        log.info("%s: got %d chars", label, len(content))
        return content

    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running?\n"
            "Check: Open browser -> http://localhost:11434\n"
            "Fix: Click the Ollama icon in system tray or restart Ollama."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Model '{model}' took too long (>{timeout}s).\n"
            "Try: Use the 7B model instead of 14B, or increase timeout."
        )
    except KeyError:
        raise RuntimeError(
            f"Unexpected Ollama response format: {resp.text[:300] if resp else 'no response'}"
        )


async def call_openrouter(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> dict:
    """
    Advisory call (used for architect reasoning stage).
    Returns a dict — doesn't need to be a floor plan JSON.
    """
    advisory_system = (
        "You are an expert Indian residential architect. "
        "Analyze the request and return a JSON object with keys: "
        "design_strategy (string), priority_order (array of strings), "
        "critical_checks (array of strings), risks (array of strings), "
        "lifestyle_moves (array of strings), circulation_plan (string). "
        "Return only JSON."
    )
    try:
        content = await _call_local(
            advisory_system,
            user_message,
            LOCAL_LLM_ADVISORY_MODEL,
            ADVISORY_TIMEOUT,
            label="advisory",
        )
        return _extract_json(content)
    except Exception as e:
        log.warning("Advisory call failed: %s. Using defaults.", str(e)[:100])
        return {
            "design_strategy": "Standard 3-band zoning with Vastu compliance.",
            "priority_order": ["program_fit", "vastu", "circulation"],
            "critical_checks": [
                "no_overlap",
                "minimum_sizes",
                "corridor_connectivity",
            ],
            "lifestyle_moves": ["separate_public_private_zones"],
            "circulation_plan": "Use central corridor spine for bedroom access.",
            "risks": [],
        }


async def call_openrouter_plan(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 4000,
    advisory: dict[str, Any] | None = None,
    request_data: dict[str, Any] | None = None,
) -> dict:
    """
    Main floor plan generation. Tries 7B first, falls back to 14B.
    The system_prompt from prompt_builder.py is overridden with
    NAKSHA_SYSTEM_PROMPT because the local model needs a more compact,
    precise version.
    """
    # Use our optimized local system prompt
    effective_system = NAKSHA_SYSTEM_PROMPT
    combined_user = _build_generation_user_message(
        user_message=user_message,
        advisory=advisory,
        request_data=request_data,
    )

    # Try primary model (7B — fast on GTX 1650)
    try:
        content = await _call_local(
            effective_system,
            combined_user,
            LOCAL_LLM_PLAN_MODEL,
            PRIMARY_TIMEOUT,
            label="plan-primary",
        )
        plan = _extract_json(content)
        return _normalize_plan(plan, user_message)

    except Exception as e:
        log.warning("Primary model failed: %s. Trying backup.", str(e)[:120])

    # Try backup model (14B — slower but smarter)
    try:
        content = await _call_local(
            effective_system,
            combined_user,
            LOCAL_LLM_BACKUP_MODEL,
            BACKUP_TIMEOUT,
            label="plan-backup",
        )
        plan = _extract_json(content)
        return _normalize_plan(plan, user_message)

    except Exception as e:
        log.error("Backup model also failed: %s", str(e)[:120])
        raise RuntimeError(
            f"Both local models failed to generate a floor plan.\n"
            f"Primary: {LOCAL_LLM_PLAN_MODEL}\n"
            f"Backup: {LOCAL_LLM_BACKUP_MODEL}\n"
            f"Error: {e}\n\n"
            f"Troubleshooting:\n"
            f"1. Open browser -> http://localhost:11434 (should show 'Ollama is running')\n"
            f"2. Open Command Prompt -> ollama list (should show your models)\n"
            f"3. Try: ollama run {LOCAL_LLM_PLAN_MODEL} 'say hello'\n"
        )


async def call_openrouter_plan_backup(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 4000,
) -> dict:
    """Backup planner — same as primary for local setup."""
    return await call_openrouter_plan(system_prompt, user_message)
