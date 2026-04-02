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
        "Do not include markdown or explanation."
    )
    user = (
        "Convert the following model output into strict JSON only. "
        "If some optional fields are missing, keep arrays empty but preserve structure.\n\n"
        "MODEL OUTPUT:\n"
        f"{snippet}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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
    payload = {
        "model": PUBLIC_LLM_FALLBACK_MODEL,
        "temperature": min(temperature, 0.15),
        "max_tokens": min(max_tokens, 4500),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
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
    # Previous requests relied on implicit expectations and produced random geometry.
    # Add explicit reasoning and geometry-verification instruction for planner stability.
    reasoning_guard = (
        "Think step by step before writing any JSON. "
        "Verify your coordinate math explicitly. "
        "Do not output JSON until you are certain no two rooms overlap "
        "and all adjacency requirements are met."
    )
    if reasoning_guard not in user_message:
        user_message = f"{user_message.strip()}\n\n{reasoning_guard}"

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

    raise ValueError(f"No valid JSON in LLM response:\n{text[:300]}")
