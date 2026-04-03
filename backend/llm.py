"""
LLM caller — async functions that hit OpenRouter and return parsed JSON.
Uses a chain of free models: if one is rate-limited (429), tries the next.
Optimized for architectural correctness — reasoning models first with longer timeouts.
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
import time
import uuid
import httpx
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_KEY_SECONDARY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_PLAN_MODEL,
    PUBLIC_LLM_FALLBACK_ENABLED,
    PUBLIC_LLM_FALLBACK_MODEL,
    PUBLIC_LLM_FALLBACK_URL,
)

log = logging.getLogger("llm")

# ── Fallback model chains — ordered by speed/reliability ─────
FALLBACK_PLAN_MODELS = [
    # Previously the chain preferred fast coders and truncated reasoning.
    # Reordered for spatial reasoning strength in floor planning tasks.
    "deepseek/deepseek-r1",
    "google/gemma-3-27b-it",
    "qwen/qwen3-coder",
]

FALLBACK_ADVICE_MODELS = [
    "qwen/qwen3-coder:free",
    "qwen/qwen3.6-plus:free",
    "google/gemma-3-27b-it:free",
]

BACKUP_PLAN_MODELS = [
    "google/gemma-3-27b-it",
    "qwen/qwen3-coder",
    "meta-llama/llama-3.3-70b-instruct",
]

# Cache temporarily unavailable models so repeated retries don't waste
# generation windows on known 429/402 responses.
_MODEL_COOLDOWN_SEC = 90
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

REASONING_GUARD = (
    "Think step by step before writing any JSON. "
    "Verify your coordinate math explicitly. "
    "Do not output JSON until you are certain no two rooms overlap "
    "and all adjacency requirements are met."
)


def _build_openrouter_headers(api_key: str, request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nakshanirman.app",
        "X-Title": "NakshaNirman",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-Request-Id": request_id,
    }


def _build_json_repair_messages(raw_text: str) -> list[dict[str, str]]:
    snippet = (raw_text or "").strip()
    if len(snippet) > 12000:
        snippet = snippet[:12000]

    system = (
        "You are a strict JSON repair formatter for architectural floor plans. "
        "Return only valid JSON object with these top-level keys: "
        "plot_boundary, rooms, doors, windows, metadata. "
        "Do not include markdown or explanation. "
        "Each room object must include: id,type,label,x,y,width,height,area,zone,band,color,polygon."
    )
    user = (
        "Convert the following model output into strict JSON only. "
        "If geometry is missing, infer practical rectangular values within bounds and fill required fields. "
        "Never return partial room objects with only names.\n\n"
        "MODEL OUTPUT:\n"
        f"{snippet}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_compact_plan_messages(user_message: str) -> list[dict[str, str]]:
    """Build a shorter, schema-focused prompt for public fallback providers."""
    msg = str(user_message or "")

    m_dims = re.search(r"Usable:\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)ft", msg, flags=re.IGNORECASE)
    usable_w = m_dims.group(1) if m_dims else "33"
    usable_l = m_dims.group(2) if m_dims else "28.5"

    m_bhk = re.search(r"BRIEF:.*?(\d+)BHK", msg, flags=re.IGNORECASE | re.DOTALL)
    bhk = m_bhk.group(1) if m_bhk else "2"

    m_facing = re.search(r"BRIEF:.*?(north|south|east|west)-facing", msg, flags=re.IGNORECASE | re.DOTALL)
    facing = (m_facing.group(1).lower() if m_facing else "south")

    m_extras = re.search(r"Extras:\s*(.+)", msg, flags=re.IGNORECASE)
    extras = (m_extras.group(1).strip() if m_extras else "none")

    system = (
        "Return ONLY a valid JSON object. "
        "Do not include analysis, markdown, or extra text."
    )

    user = (
        f"Create a {bhk}BHK floor plan for usable {usable_w}x{usable_l} ft ({facing}-facing). Extras: {extras}. "
        "Output schema exactly: "
        "{"
        "\"plot_boundary\":[{\"x\":0,\"y\":0},{\"x\":0,\"y\":0},{\"x\":0,\"y\":0},{\"x\":0,\"y\":0}],"
        "\"rooms\":{\"living\":{\"x\":0,\"y\":0,\"width\":0,\"height\":0}},"
        "\"metadata\":{\"bhk\":2,\"adjacency_score\":80},"
        "\"doors\":[],\"windows\":[]"
        "}. "
        "rooms must include: living,dining,kitchen,corridor,master_bedroom,master_bath,bedroom,bathroom. "
        "Use numeric x,y,width,height. Keep all rectangles in bounds and non-overlapping."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_outer_json_block(text: str) -> str:
    raw = str(text or "").strip()
    start = raw.find("{")
    if start < 0:
        return raw

    depth = 0
    in_string = False
    escape = False
    quote = '"'

    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start: idx + 1]

    return raw[start:]


def _python_repair_json(raw_text: str) -> dict | None:
    snippet = _extract_outer_json_block(raw_text)
    if not snippet:
        return None

    candidates: list[str] = [snippet]

    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", snippet)
    candidates.append(no_trailing_commas)

    single_quoted = re.sub(r"([\{,]\s*)'([^'\n\r]+?)'\s*:", r'\1"\2":', no_trailing_commas)
    single_quoted = re.sub(r":\s*'([^'\n\r]*?)'(\s*[,}\]])", r': "\1"\2', single_quoted)
    single_quoted = re.sub(r"\bTrue\b", "true", single_quoted)
    single_quoted = re.sub(r"\bFalse\b", "false", single_quoted)
    single_quoted = re.sub(r"\bNone\b", "null", single_quoted)
    candidates.append(single_quoted)

    broad_quote_swap = single_quoted.replace("'", '"')
    candidates.append(broad_quote_swap)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return None


async def _repair_json_via_openai_compatible(
    *,
    url: str,
    headers: dict[str, str] | None,
    model: str,
    raw_text: str,
    timeout: int,
    label: str,
    verify_ssl: bool,
) -> dict | None:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2400,
        "messages": _build_json_repair_messages(raw_text),
    }

    try:
        call_timeout = max(12, min(timeout, 35))
        async with httpx.AsyncClient(timeout=call_timeout, verify=verify_ssl) as client:
            if headers:
                resp = await client.post(url, json=payload, headers=headers)
            else:
                resp = await client.post(url, json=payload)

        if resp.status_code >= 400:
            return None

        data = resp.json()
        repaired_content = _extract_openai_compatible_content(data)
        repaired = _extract_json(repaired_content)
        log.info("%s: repaired non-JSON model output", label)
        return repaired
    except Exception as e:
        log.info("%s: json-repair failed: %s", label, str(e)[:120])
        return None


def _build_model_chain(primary: str | None, fallbacks: list[str]) -> list[str]:
    chain: list[str] = []
    if primary and primary.strip():
        chain.append(primary.strip())

    for model in fallbacks:
        if model and model not in chain:
            chain.append(model)
    return chain


def _extract_openai_compatible_content(data: dict) -> str:
    """Extract assistant content across common OpenAI-compatible response shapes."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}

        message = first.get("message", {}) if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        txt = part.get("text")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt)
                merged = "\n".join(parts).strip()
                if merged:
                    return merged

            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning

        direct_text = first.get("text") if isinstance(first, dict) else None
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

    output_text = data.get("output_text") if isinstance(data, dict) else None
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    raise ValueError(f"Public fallback response missing content fields: keys={list(data.keys()) if isinstance(data, dict) else type(data)}")


def _normalize_plan_shape(plan: dict, user_message: str) -> dict:
    """Normalize compact or irregular JSON into the plan schema expected downstream."""
    if not isinstance(plan, dict):
        return {}

    out = dict(plan)

    dims = re.search(
        r"Usable:\s*([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)ft",
        str(user_message or ""),
        flags=re.IGNORECASE,
    )
    uw = float(dims.group(1)) if dims else 33.0
    ul = float(dims.group(2)) if dims else 28.5

    plot_boundary = out.get("plot_boundary")
    if not isinstance(plot_boundary, list) or len(plot_boundary) < 4:
        out["plot_boundary"] = [
            {"x": 0.0, "y": 0.0},
            {"x": uw, "y": 0.0},
            {"x": uw, "y": ul},
            {"x": 0.0, "y": ul},
        ]

    rooms = out.get("rooms", [])
    if isinstance(rooms, dict):
        converted: list[dict] = []
        for room_name, room_data in rooms.items():
            if not isinstance(room_data, dict):
                continue
            room = dict(room_data)
            room_type = str(room.get("type") or room_name or "room")
            room["type"] = room_type
            room.setdefault("label", room_type.replace("_", " ").title())
            converted.append(room)
        rooms = converted
    elif not isinstance(rooms, list):
        rooms = []

    out["rooms"] = rooms
    if not isinstance(out.get("doors"), list):
        out["doors"] = []
    if not isinstance(out.get("windows"), list):
        out["windows"] = []

    metadata = out.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if "bhk" not in metadata:
        m_bhk = re.search(r"BRIEF:.*?(\d+)BHK", str(user_message or ""), flags=re.IGNORECASE | re.DOTALL)
        metadata["bhk"] = int(m_bhk.group(1)) if m_bhk else 2
    out["metadata"] = metadata
    return out


async def _call_public_fallback_openai_compatible(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    label: str,
) -> dict:
    """
    Use a free public OpenAI-compatible endpoint as last-resort fallback.
    This path intentionally does not require API keys.
    """
    compact_messages = _build_compact_plan_messages(user_message)

    payload = {
        "model": PUBLIC_LLM_FALLBACK_MODEL,
        "temperature": min(temperature, 0.15),
        "max_tokens": min(max_tokens, 4500),
        "messages": compact_messages,
        "response_format": {"type": "json_object"},
    }

    log.warning(
        "%s: OpenRouter unavailable, switching to public fallback model %s",
        label,
        PUBLIC_LLM_FALLBACK_MODEL,
    )

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=timeout, verify=True) as client:
                resp = await client.post(PUBLIC_LLM_FALLBACK_URL, json=payload)

            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"public fallback HTTP {resp.status_code}")

            resp.raise_for_status()
            data = resp.json()
            content = _extract_openai_compatible_content(data)
            try:
                parsed = _extract_json(content)
                log.info("%s: public fallback ✓ (%d chars) on attempt %d", label, len(content), attempt)
                return parsed
            except Exception:
                python_repaired = _python_repair_json(content)
                if python_repaired is not None:
                    log.info("%s: public fallback repaired via python parser", label)
                    return python_repaired
                repaired = await _repair_json_via_openai_compatible(
                    url=PUBLIC_LLM_FALLBACK_URL,
                    headers=None,
                    model=PUBLIC_LLM_FALLBACK_MODEL,
                    raw_text=content,
                    timeout=timeout,
                    label=f"{label}:public",
                    verify_ssl=True,
                )
                if repaired is not None:
                    log.info("%s: public fallback repaired on attempt %d", label, attempt)
                    return repaired

                # If repair still fails, re-ask with a compact JSON-focused prompt.
                compact_payload = {
                    "model": PUBLIC_LLM_FALLBACK_MODEL,
                    "temperature": 0,
                    "max_tokens": min(max_tokens, 2600),
                    "messages": _build_compact_plan_messages(user_message),
                    "response_format": {"type": "json_object"},
                }
                try:
                    async with httpx.AsyncClient(timeout=timeout, verify=True) as client:
                        compact_resp = await client.post(PUBLIC_LLM_FALLBACK_URL, json=compact_payload)
                    if compact_resp.status_code < 400:
                        compact_data = compact_resp.json()
                        compact_content = _extract_openai_compatible_content(compact_data)
                        try:
                            compact_parsed = _extract_json(compact_content)
                            log.info("%s: compact public fallback ✓ on attempt %d", label, attempt)
                            return compact_parsed
                        except Exception:
                            compact_python = _python_repair_json(compact_content)
                            if compact_python is not None:
                                log.info("%s: compact public fallback repaired via python parser", label)
                                return compact_python
                except Exception:
                    pass
                raise
        except Exception as e:
            last_error = e
            if attempt < 3:
                await asyncio.sleep(1.5 * attempt)
                continue

    raise RuntimeError(f"public fallback failed after retries: {last_error}")


async def _call_with_fallback(
    models: list[str],
    system_prompt: str,
    user_message: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: int,
    label: str = "LLM",
) -> dict:
    """
    Try each model in order. Skip models that return 429/402/404.
    Timeout is controlled by caller; plan generation intentionally allows
    longer reasoning windows before fallback.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    request_id = uuid.uuid4().hex
    api_keys = [
        k.strip()
        for k in (OPENROUTER_API_KEY, OPENROUTER_API_KEY_SECONDARY)
        if isinstance(k, str) and k.strip()
    ]
    if not api_keys:
        raise RuntimeError("OpenRouter API key is not configured")

    last_error = None

    attempted_models = 0
    skipped_models: list[str] = []

    for model in models:
        now = time.time()
        cooldown_until = _MODEL_COOLDOWN_UNTIL.get(model, 0.0)
        if cooldown_until > now:
            remain = int(cooldown_until - now)
            log.info("%s: skipping %s (cooldown %ss)", label, model.split("/")[-1], remain)
            skipped_models.append(model)
            continue

        attempted_models += 1
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        log.info("%s: trying %s", label, model.split("/")[-1])

        for key_idx, api_key in enumerate(api_keys, start=1):
            headers = _build_openrouter_headers(api_key, f"{request_id}-{key_idx}")
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                    resp = await client.post(url, json=payload, headers=headers)

                if resp.status_code in (429, 402, 404):
                    log.warning(
                        "%s: %s key#%d → %d, next...",
                        label, model.split("/")[-1], key_idx, resp.status_code,
                    )
                    if resp.status_code == 429:
                        _MODEL_COOLDOWN_UNTIL[model] = time.time() + _MODEL_COOLDOWN_SEC
                    last_error = f"{model} (key#{key_idx}) → {resp.status_code}"
                    if resp.status_code == 404:
                        break
                    continue

                resp.raise_for_status()

                data = resp.json()
                content = _extract_openai_compatible_content(data)
                try:
                    parsed = _extract_json(content)
                    log.info("%s: ✓ %s (%d chars)", label, model.split("/")[-1], len(content))
                    return parsed
                except Exception as parse_error:
                    python_repaired = _python_repair_json(content)
                    if python_repaired is not None:
                        log.info("%s: python repaired %s", label, model.split("/")[-1])
                        return python_repaired
                    repaired = await _repair_json_via_openai_compatible(
                        url=url,
                        headers=headers,
                        model=model,
                        raw_text=content,
                        timeout=timeout,
                        label=f"{label}:{model.split('/')[-1]}:key#{key_idx}",
                        verify_ssl=False,
                    )
                    if repaired is not None:
                        return repaired
                    last_error = f"{model} (key#{key_idx}) non-json: {str(parse_error)[:80]}"
                    continue

            except httpx.TimeoutException:
                log.warning("%s: %s key#%d timed out, next...", label, model.split("/")[-1], key_idx)
                last_error = f"{model} (key#{key_idx}) timed out"
                continue
            except Exception as e:
                log.warning("%s: %s key#%d failed: %s", label, model.split("/")[-1], key_idx, str(e)[:100])
                last_error = f"{model} (key#{key_idx}): {e}"
                continue

    # If every model was skipped due cooldown, force one probe call instead of
    # immediately falling through to local fallback on this request.
    if attempted_models == 0 and skipped_models:
        probe_model = skipped_models[0]
        payload = {
            "model": probe_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        log.info("%s: probing cooled model %s", label, probe_model.split("/")[-1])
        for key_idx, api_key in enumerate(api_keys, start=1):
            headers = _build_openrouter_headers(api_key, f"{request_id}-probe-{key_idx}")
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                    resp = await client.post(url, json=payload, headers=headers)

                if resp.status_code not in (429, 402, 404):
                    resp.raise_for_status()
                    data = resp.json()
                    content = _extract_openai_compatible_content(data)
                    try:
                        parsed = _extract_json(content)
                        log.info("%s: probe ✓ %s (%d chars)", label, probe_model.split("/")[-1], len(content))
                        return parsed
                    except Exception:
                        repaired = await _repair_json_via_openai_compatible(
                            url=url,
                            headers=headers,
                            model=probe_model,
                            raw_text=content,
                            timeout=timeout,
                            label=f"{label}:{probe_model.split('/')[-1]}:probe:key#{key_idx}",
                            verify_ssl=False,
                        )
                        if repaired is not None:
                            return repaired

                last_error = f"{probe_model} (key#{key_idx}) → {resp.status_code}"
            except Exception as e:
                last_error = f"{probe_model} (key#{key_idx}): {e}"

    raise RuntimeError(
        f"All {len(models)} models failed. Last: {last_error}"
    )


async def call_openrouter(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> dict:
    """Lightweight call for Vastu advice with configurable primary model."""
    advice_models = _build_model_chain(OPENROUTER_MODEL, FALLBACK_ADVICE_MODELS)
    return await _call_with_fallback(
        advice_models,
        system_prompt,
        user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=12,
        label="advice",
    )


async def call_openrouter_plan(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 5000,
) -> dict:
    """
    Call LLM for full floor plan generation.
    Uses configured primary model first, then fallback chain.
    """
    if REASONING_GUARD not in user_message:
        user_message = f"{user_message.strip()}\n\n{REASONING_GUARD}"

    primary = (OPENROUTER_PLAN_MODEL or OPENROUTER_MODEL or "").strip()
    # Keep deepseek first unless caller explicitly sets a different planner.
    if primary.lower() in ("", "qwen/qwen3-coder", "qwen/qwen3-coder:free"):
        primary = FALLBACK_PLAN_MODELS[0]
    plan_models = _build_model_chain(primary, FALLBACK_PLAN_MODELS)
    try:
        return await _call_with_fallback(
            plan_models,
            system_prompt,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            # Planner calls need time for full architectural reasoning.
            timeout=55,
            label="plan",
        )
    except Exception:
        if not PUBLIC_LLM_FALLBACK_ENABLED:
            raise
        return await _call_public_fallback_openai_compatible(
            system_prompt,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=75,
            label="plan",
        )


async def call_openrouter_plan_backup(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 5000,
) -> dict:
    """
    Backup planner call using an alternate free-model pool.
    Used when primary fast model chain is unavailable.
    """
    primary = (OPENROUTER_PLAN_MODEL or OPENROUTER_MODEL or "").strip()
    backup_models = [m for m in BACKUP_PLAN_MODELS if m != primary]
    if not backup_models:
        backup_models = BACKUP_PLAN_MODELS[:]

    try:
        return await _call_with_fallback(
            backup_models,
            system_prompt,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=55,
            label="plan-backup",
        )
    except Exception:
        if not PUBLIC_LLM_FALLBACK_ENABLED:
            raise
        return await _call_public_fallback_openai_compatible(
            system_prompt,
            user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=75,
            label="plan-backup",
        )


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM text output."""
    # Strip <think>...</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    python_repaired = _python_repair_json(text)
    if python_repaired is not None:
        return python_repaired

    raise ValueError(f"No valid JSON in LLM response:\n{text[:300]}")
