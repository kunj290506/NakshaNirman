"""
LLM caller — async functions that hit OpenRouter and return parsed JSON.
Uses a chain of free models: if one is rate-limited (429), tries the next.
Optimized for speed — fast models first, short timeouts.
"""
from __future__ import annotations
import json
import logging
import re
import uuid
import httpx
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_PLAN_MODEL,
)

log = logging.getLogger("llm")

# ── Fallback model chains — ordered by speed/reliability ─────
FALLBACK_PLAN_MODELS = [
    "qwen/qwen3-coder:free",                          # fast, great at JSON
    "qwen/qwen3.6-plus:free",                         # strong + still fast
    "google/gemma-3-27b-it:free",                     # backup
]

FALLBACK_ADVICE_MODELS = [
    "qwen/qwen3-coder:free",
    "qwen/qwen3.6-plus:free",
    "google/gemma-3-27b-it:free",
]

BACKUP_PLAN_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3.6-plus:free",
]


def _build_model_chain(primary: str | None, fallbacks: list[str]) -> list[str]:
    chain: list[str] = []
    if primary and primary.strip():
        chain.append(primary.strip())

    for model in fallbacks:
        if model and model not in chain:
            chain.append(model)
    return chain


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
    Short timeout per model to fail fast.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    request_id = uuid.uuid4().hex
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://nakshanirman.app",
        "X-Title": "NakshaNirman",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "X-Request-Id": request_id,
    }

    last_error = None

    for model in models:
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

        try:
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code in (429, 402, 404):
                log.warning(
                    "%s: %s → %d, next...",
                    label, model.split("/")[-1], resp.status_code,
                )
                last_error = f"{model} → {resp.status_code}"
                continue

            resp.raise_for_status()

            data = resp.json()
            content: str = data["choices"][0]["message"]["content"]
            log.info("%s: ✓ %s (%d chars)", label, model.split("/")[-1], len(content))
            return _extract_json(content)

        except httpx.TimeoutException:
            log.warning("%s: %s timed out, next...", label, model.split("/")[-1])
            last_error = f"{model} timed out"
            continue
        except Exception as e:
            log.warning("%s: %s failed: %s", label, model.split("/")[-1], str(e)[:100])
            last_error = f"{model}: {e}"
            continue

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
    temperature: float = 0.2,
    max_tokens: int = 2400,
) -> dict:
    """
    Call LLM for full floor plan generation.
    Uses configured primary model first, then fallback chain.
    """
    primary = OPENROUTER_PLAN_MODEL or OPENROUTER_MODEL
    plan_models = _build_model_chain(primary, FALLBACK_PLAN_MODELS)[:2]
    return await _call_with_fallback(
        plan_models,
        system_prompt,
        user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=14,
        label="plan",
    )


async def call_openrouter_plan_backup(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2400,
) -> dict:
    """
    Backup planner call using an alternate free-model pool.
    Used when primary fast model chain is unavailable.
    """
    primary = (OPENROUTER_PLAN_MODEL or OPENROUTER_MODEL or "").strip()
    backup_models = [m for m in BACKUP_PLAN_MODELS if m != primary]
    if not backup_models:
        backup_models = BACKUP_PLAN_MODELS[:]

    return await _call_with_fallback(
        backup_models,
        system_prompt,
        user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=20,
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
