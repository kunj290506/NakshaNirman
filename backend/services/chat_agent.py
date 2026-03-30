"""Chat agent service for collecting requirements and emitting GENERATE_PLAN token."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import importlib
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from app_config import (
    GROQ_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_ENABLED,
    OPENROUTER_TEXT_MODEL,
    OPENROUTER_VERIFY_SSL,
)

logger = logging.getLogger(__name__)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


SYSTEM_PROMPT = """
You are Naksha, a Senior Indian Residential Architect with 20 years experience.
You help clients design ground floor house plans.

YOUR ONLY JOB: Collect 5 things, then generate.

WHAT YOU NEED:
    1. Plot size: width × length in feet (for example "30 by 40 feet") OR total sqft
    2. BHK type: how many bedrooms (1BHK to 4BHK)
    3. Extra rooms: pooja room, study, garage, store room, balcony
    4. Facing direction: which side faces the road (default: East)
    5. Vastu preference: yes or no (default: Yes)

RULES:
    - Ask MAX 2 questions per response
    - After getting plot size + BHK, you have enough to generate
    - Always acknowledge what user said before asking
    - Use Indian terms: BHK, sqft, Vastu, ground floor, drawing room
    - For missing items, assume defaults: East-facing, Vastu yes, no extras
    - Never discuss anything outside house design and architecture

SUMMARY BEFORE GENERATING:
When you have enough info, show:
"Here is your floor plan summary:
    Plot: [size]
    [N]BHK | [N] Bathrooms
    Facing: [direction]
    Vastu: [Yes/No]
    Extra rooms: [list or None]
Shall I generate your floor plan?"

ON USER CONFIRMATION:
Output EXACTLY this — no other text — just this JSON line:
GENERATE_PLAN: {"plot_width":30,"plot_length":40,"bedrooms":2,"bathrooms":2,"facing":"east","vastu":true,"extras":[]}
""".strip()


YES_TOKENS = {"yes", "y", "generate", "go ahead", "proceed", "ok", "okay", "sure", "haan", "han", "done"}
EXTRA_TOKENS = ["pooja", "study", "store", "balcony", "garage"]


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


async def _call_groq(prompt: str, history: List[Dict[str, str]]) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=GROQ_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": prompt})
        completion = await client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        text = completion.choices[0].message.content if completion.choices else None
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("Groq chat failed: %s", exc)
        return None


async def _call_openrouter(prompt: str, history: List[Dict[str, str]]) -> Optional[str]:
    if not OPENROUTER_ENABLED or not OPENROUTER_API_KEY or not OPENROUTER_TEXT_MODEL:
        return None

    def _request() -> Optional[str]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": OPENROUTER_TEXT_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 400,
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


async def _call_claude(prompt: str, history: List[Dict[str, str]]) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        anthropic_mod = importlib.import_module("anthropic")
        async_anthropic_cls = getattr(anthropic_mod, "AsyncAnthropic", None)
        if async_anthropic_cls is None:
            return None

        client = async_anthropic_cls(api_key=ANTHROPIC_API_KEY)
        messages = []
        for item in history[-8:]:
            role = "assistant" if item.get("role") == "assistant" else "user"
            messages.append({"role": role, "content": item.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        resp = await client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=0.2,
        )
        if not resp.content:
            return None
        text = "".join(part.text for part in resp.content if hasattr(part, "text"))
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("Claude chat failed: %s", exc)
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
        payload = _build_generate_payload(collected)
        summary = (
            "Here is your floor plan summary:\n"
            f"  Plot: {payload['plot_width']} x {payload['plot_length']} ft\n"
            f"  {payload['bedrooms']}BHK | {payload['bathrooms']} Bathrooms\n"
            f"  Facing: {payload['facing'].capitalize()}\n"
            f"  Vastu: {'Yes' if payload['vastu'] else 'No'}\n"
            f"  Extra rooms: {', '.join(payload['extras']) if payload['extras'] else 'None'}\n"
            "Shall I generate your floor plan?"
        )
        return summary

    if has_plot and not has_bhk:
        fallback = "Noted your plot size. Please confirm BHK type: 1BHK, 2BHK, 3BHK, or 4BHK."
    elif has_bhk and not has_plot:
        fallback = "Understood your BHK requirement. Please share plot size in feet (for example 30x40)."
    else:
        fallback = (
            "Please share your plot size and BHK type. "
            "Example: 30x40, 2BHK. I will assume East-facing with Vastu and no extras unless you specify."
        )

    # Try LLM polish, but keep strict fallback for reliability.
    llm = await _call_openrouter(user_message, history)
    if not llm:
        llm = await _call_groq(user_message, history)
    if not llm:
        llm = await _call_claude(user_message, history)

    if llm:
        cleaned = llm.strip()
        if "GENERATE_PLAN:" in cleaned:
            token = cleaned.split("GENERATE_PLAN:", 1)[1].strip().splitlines()[0].strip()
            # Guardrail: emit token only when we truly have enough data.
            if has_plot and has_bhk:
                return f"GENERATE_PLAN: {token}"
            return fallback
        return cleaned

    return fallback
