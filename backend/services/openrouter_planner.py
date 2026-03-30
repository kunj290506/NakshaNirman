"""OpenRouter-assisted payload shaping for floor plan generation.

This module is intentionally optional:
- It enriches incoming requests when OpenRouter is configured.
- It never blocks generation; deterministic planning remains the fallback.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from app_config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_ENABLED,
    OPENROUTER_TEXT_MODEL,
    OPENROUTER_VERIFY_SSL,
)

logger = logging.getLogger(__name__)

_ALLOWED_EXTRAS = {"pooja", "study", "store", "balcony", "garage", "staircase"}
_ALLOWED_FACING = {"east", "west", "north", "south"}
_ALLOWED_FAMILY = {"nuclear", "joint-family", "working-couple", "elderly", "rental"}


def _to_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        out = int(float(value))
    except Exception:
        out = default
    return max(low, min(high, out))


def _to_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    return out if out > 0 else default


def _to_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _norm_mode(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"perfcat", "perfcat_plan", "perfect", "performance", "pro"}:
        return "perfcat"
    return "standard"


def _norm_facing(value: Any) -> str:
    token = str(value or "east").strip().lower()
    return token if token in _ALLOWED_FACING else "east"


def _norm_family(value: Any) -> str:
    token = str(value or "nuclear").strip().lower()
    return token if token in _ALLOWED_FAMILY else "nuclear"


def _norm_extras(value: Any) -> List[str]:
    if isinstance(value, str):
        source = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, list):
        source = value
    else:
        source = []

    out: List[str] = []
    for item in source:
        token = str(item).strip().lower()
        if token in _ALLOWED_EXTRAS and token not in out:
            out.append(token)
    return out


def _norm_constraints(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        room = str(item.get("room") or "").strip().lower()
        intent = str(item.get("intent") or "").strip().lower()
        if not room or not intent:
            continue

        normalized = {
            "room": room,
            "intent": intent,
            "band": item.get("band"),
            "forbid_adjacent": list(item.get("forbid_adjacent") or []),
            "prefer_walls": list(item.get("prefer_walls") or []),
            "note": str(item.get("note") or "")[:200],
        }

        key = (
            normalized["room"],
            normalized["intent"],
            tuple(normalized["forbid_adjacent"]),
            tuple(normalized["prefer_walls"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)

    return out


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return {}

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _sanitize_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if "plot_width" in data:
        out["plot_width"] = _to_float(data.get("plot_width"), 0.0)
    if "plot_length" in data:
        out["plot_length"] = _to_float(data.get("plot_length"), 0.0)
    if "total_area" in data:
        out["total_area"] = _to_float(data.get("total_area"), 1200.0)
    if "bedrooms" in data:
        out["bedrooms"] = _to_int(data.get("bedrooms"), 2, 1, 4)
    if "bathrooms" in data:
        out["bathrooms"] = _to_int(data.get("bathrooms"), 2, 1, 6)
    if "facing" in data:
        out["facing"] = _norm_facing(data.get("facing"))
    if "vastu" in data:
        out["vastu"] = _to_bool(data.get("vastu"), True)
    if "extras" in data:
        out["extras"] = _norm_extras(data.get("extras"))
    if "family_type" in data:
        out["family_type"] = _norm_family(data.get("family_type"))
    if "placement_constraints" in data:
        out["placement_constraints"] = _norm_constraints(data.get("placement_constraints"))

    if out.get("plot_width", 1.0) <= 0:
        out.pop("plot_width", None)
    if out.get("plot_length", 1.0) <= 0:
        out.pop("plot_length", None)

    return out


def _merge_payload(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)

    for key, value in (overrides or {}).items():
        if key == "extras":
            merged = _norm_extras(out.get("extras"))
            for item in _norm_extras(value):
                if item not in merged:
                    merged.append(item)
            out["extras"] = merged
            continue

        if key == "placement_constraints":
            existing = _norm_constraints(out.get("placement_constraints") or [])
            incoming = _norm_constraints(value)
            out["placement_constraints"] = _norm_constraints(existing + incoming)
            continue

        out[key] = value

    return out


def _perfcat_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    bedrooms = _to_int(payload.get("bedrooms"), 2, 1, 4)
    bathrooms = _to_int(payload.get("bathrooms"), max(1, bedrooms), 1, 6)

    extras = _norm_extras(payload.get("extras"))
    if bedrooms >= 3 and "store" not in extras:
        extras.append("store")

    constraints = _norm_constraints(payload.get("placement_constraints") or [])
    constraints.append(
        {
            "room": "kitchen",
            "intent": "rear_garden_preference",
            "band": 3,
            "prefer_walls": ["north", "rear"],
            "note": "Perfcat profile prefers the kitchen near rear side for service flow",
        }
    )
    constraints.append(
        {
            "room": "master_bedroom",
            "intent": "privacy_buffer",
            "forbid_adjacent": ["living"],
            "note": "Perfcat profile keeps master bedroom privacy from public living zone",
        }
    )

    return {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "facing": _norm_facing(payload.get("facing") or "east"),
        "vastu": _to_bool(payload.get("vastu"), True),
        "family_type": _norm_family(payload.get("family_type") or "nuclear"),
        "extras": extras,
        "placement_constraints": constraints,
    }


def _openrouter_messages(payload: Dict[str, Any], plan_mode: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an Indian residential planning assistant. Return one compact JSON object only. "
        "No markdown and no explanation. Suggest only valid fields used by a floor-plan generator."
    )

    user_prompt = (
        "Optimize this payload for a realistic single-floor house plan. "
        f"Plan profile: {plan_mode}. "
        "Allowed keys: bedrooms, bathrooms, facing, vastu, extras, family_type, placement_constraints, "
        "plot_width, plot_length, total_area. "
        "Only include keys that should be changed.\n"
        f"Payload: {json.dumps(payload, separators=(',', ':'))}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_openrouter_overrides(payload: Dict[str, Any], plan_mode: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not OPENROUTER_ENABLED or not OPENROUTER_API_KEY or not OPENROUTER_TEXT_MODEL:
        return {}, warnings

    try:
        body = {
            "model": OPENROUTER_TEXT_MODEL,
            "messages": _openrouter_messages(payload, plan_mode),
            "temperature": 0.15,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
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

        with urllib.request.urlopen(req, timeout=20, context=ssl_context) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))

        choices = result.get("choices") or []
        raw_text = ""
        if choices:
            raw_text = str(((choices[0] or {}).get("message") or {}).get("content") or "")

        parsed = _extract_json_object(raw_text)
        return _sanitize_overrides(parsed), warnings

    except urllib.error.HTTPError as exc:
        warnings.append("OpenRouter planner unavailable; used deterministic fallback.")
        logger.warning("OpenRouter planner HTTP failure: %s", exc)
        return {}, warnings
    except Exception as exc:
        warnings.append("OpenRouter planner unavailable; used deterministic fallback.")
        logger.warning("OpenRouter planner call failed: %s", exc)
        return {}, warnings


def prepare_floorplan_payload(payload: Dict[str, Any], plan_mode: Any = "perfcat") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare a robust generation payload with optional OpenRouter overrides.

    Returns:
        Tuple[prepared_payload, planner_meta]
    """
    mode = _norm_mode(plan_mode)
    out = dict(payload or {})

    meta: Dict[str, Any] = {
        "provider": "deterministic",
        "model": None,
        "used": False,
        "plan_mode": mode,
        "warnings": [],
    }

    if mode == "perfcat":
        out = _merge_payload(out, _perfcat_defaults(out))

    overrides, warnings = _call_openrouter_overrides(out, mode)
    if warnings:
        meta["warnings"].extend(warnings)

    if overrides:
        out = _merge_payload(out, overrides)
        meta["provider"] = "openrouter"
        meta["model"] = OPENROUTER_TEXT_MODEL
        meta["used"] = True

    out["plan_mode"] = mode
    return out, meta
