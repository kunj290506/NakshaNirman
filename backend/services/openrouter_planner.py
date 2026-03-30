"""DeepSeek (via OpenRouter) payload shaping for Perfcat floor plan generation."""

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
    OPENROUTER_PLANNER_MODEL,
    OPENROUTER_TEXT_MAX_TOKENS,
    OPENROUTER_TEXT_MODEL,
    OPENROUTER_TEXT_TEMPERATURE,
    OPENROUTER_VERIFY_SSL,
)

logger = logging.getLogger(__name__)

_ALLOWED_EXTRAS = {"pooja", "study", "store", "balcony", "garage", "staircase"}
_ALLOWED_FACING = {"east", "west", "north", "south"}
_ALLOWED_FAMILY = {"nuclear", "joint-family", "working-couple", "elderly", "rental"}
_PLANNER_ALLOWED_KEYS = (
    "plot_width",
    "plot_length",
    "total_area",
    "bedrooms",
    "bathrooms",
    "facing",
    "vastu",
    "extras",
    "family_type",
    "placement_constraints",
)
_PERFCAT_MODE = "perfcat"
_DEEPSEEK_PLANNER_MODEL = "deepseek/deepseek-chat"


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
    # Perfcat is mandatory for this planner path.
    return _PERFCAT_MODE


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


def _normalize_model_slug(value: str) -> str:
    token = str(value or "").strip().lower()
    if token.endswith(":free"):
        token = token[:-5]
    return token


def _deepseek_model(value: str, fallback: str) -> str:
    token = _normalize_model_slug(value)
    if token.startswith("deepseek/"):
        return token
    return fallback


def _planner_model_name() -> str:
    configured = str(OPENROUTER_PLANNER_MODEL or OPENROUTER_TEXT_MODEL or "").strip()
    return _deepseek_model(configured, _DEEPSEEK_PLANNER_MODEL)


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


def _planner_seed_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    seed: Dict[str, Any] = {}

    for key in _PLANNER_ALLOWED_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue

        if key in {"plot_width", "plot_length"}:
            normalized = _to_float(value, 0.0)
            if normalized <= 0:
                continue
            seed[key] = round(normalized, 1)
            continue

        if key == "total_area":
            normalized = _to_float(value, 1200.0)
            if normalized <= 0:
                continue
            seed[key] = round(normalized, 1)
            continue

        if key == "bedrooms":
            seed[key] = _to_int(value, 2, 1, 4)
            continue

        if key == "bathrooms":
            default_baths = int(seed.get("bedrooms") or 2)
            seed[key] = _to_int(value, default_baths, 1, 6)
            continue

        if key == "facing":
            seed[key] = _norm_facing(value)
            continue

        if key == "vastu":
            seed[key] = _to_bool(value, True)
            continue

        if key == "extras":
            seed[key] = _norm_extras(value)
            continue

        if key == "family_type":
            seed[key] = _norm_family(value)
            continue

        if key == "placement_constraints":
            constraints = _norm_constraints(value)
            if constraints:
                seed[key] = constraints
            continue

        seed[key] = value

    return seed


def _openrouter_messages(payload: Dict[str, Any], plan_mode: str) -> List[Dict[str, str]]:
    seed_payload = _planner_seed_payload(payload)
    system_prompt = (
        "You are Naksha Planner Optimizer for Indian residential floor plans. "
        "Return one compact JSON object only with minimal override keys that improve feasibility, "
        "adjacency quality, and practical Vastu alignment while preserving user intent. "
        "Do not include markdown or explanations. Allowed keys only: "
        "bedrooms, bathrooms, facing, vastu, extras, family_type, placement_constraints, "
        "plot_width, plot_length, total_area."
    )

    user_prompt = (
        "Optimize this seed payload for a realistic single-floor house plan. "
        f"Plan profile: {plan_mode}. "
        "Allowed keys: bedrooms, bathrooms, facing, vastu, extras, family_type, placement_constraints, "
        "plot_width, plot_length, total_area. "
        "Only include keys that should be changed.\n"
        f"Seed payload: {json.dumps(seed_payload, separators=(',', ':'))}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _call_openrouter_overrides(payload: Dict[str, Any], plan_mode: str) -> Dict[str, Any]:
    planner_model = _planner_model_name()

    if not OPENROUTER_ENABLED:
        raise RuntimeError("DeepSeek planner is disabled. Set OPENROUTER_ENABLED=true.")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("DeepSeek planner is not configured. Set OPENROUTER_API_KEY.")
    if not planner_model:
        raise RuntimeError("DeepSeek planner model is not configured.")

    try:
        max_tokens = max(220, min(int(OPENROUTER_TEXT_MAX_TOKENS), 520))
        body = {
            "model": planner_model,
            "messages": _openrouter_messages(payload, plan_mode),
            "temperature": max(0.0, min(0.6, float(OPENROUTER_TEXT_TEMPERATURE))),
            "max_tokens": max_tokens,
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

        with urllib.request.urlopen(req, timeout=45, context=ssl_context) as resp:
            result = json.loads(resp.read().decode("utf-8", errors="replace"))

        choices = result.get("choices") or []
        raw_text = ""
        if choices:
            raw_text = str(((choices[0] or {}).get("message") or {}).get("content") or "")

        parsed = _extract_json_object(raw_text)
        return _sanitize_overrides(parsed)

    except urllib.error.HTTPError as exc:
        logger.warning("DeepSeek planner HTTP failure: %s", exc)
        raise RuntimeError(f"DeepSeek planner request failed ({exc.code}).") from exc
    except Exception as exc:
        logger.warning("DeepSeek planner call failed: %s", exc)
        raise RuntimeError("DeepSeek planner call failed.") from exc


def prepare_floorplan_payload(payload: Dict[str, Any], plan_mode: Any = "perfcat") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare a robust Perfcat payload with required DeepSeek overrides.

    Returns:
        Tuple[prepared_payload, planner_meta]
    """
    mode = _norm_mode(plan_mode)
    out = dict(payload or {})

    meta: Dict[str, Any] = {
        "provider": "openrouter",
        "model": _planner_model_name(),
        "used": False,
        "plan_mode": mode,
        "warnings": [],
    }

    out = _merge_payload(out, _perfcat_defaults(out))

    overrides = _call_openrouter_overrides(out, mode)

    if overrides:
        out = _merge_payload(out, overrides)
    meta["used"] = True

    out["plan_mode"] = mode
    return out, meta
