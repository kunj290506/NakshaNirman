"""OpenRouter image generation helpers for floor plan concepts."""

from __future__ import annotations

import json
import logging
import math
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from app_config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_ENABLED,
    OPENROUTER_IMAGE_MODEL,
    OPENROUTER_VERIFY_SSL,
)

logger = logging.getLogger(__name__)
_ALLOWED_RESPONSE_FORMATS = {"b64_json", "url"}
_SIZE_PATTERN = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_SUPPORTED_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
_MODEL_ALIASES = {
    "flux.2 klein 4b": "black-forest-labs/flux.2-klein-4b",
    "flux.2 max": "black-forest-labs/flux.2-max",
    "flux.2 flex": "black-forest-labs/flux.2-flex",
    "flux.2 pro": "black-forest-labs/flux.2-pro",
}


def _normalize_image_model(model_name: str) -> str:
    token = str(model_name or "").strip()
    if not token:
        return ""
    return _MODEL_ALIASES.get(token.lower(), token)


def _parse_size(size: str) -> Tuple[int, int]:
    token = str(size or "").strip().lower()
    match = _SIZE_PATTERN.match(token)
    if not match:
        return 1024, 1024
    return int(match.group(1)), int(match.group(2))


def _build_image_config(size: str) -> Dict[str, str]:
    width, height = _parse_size(size)

    gcd = math.gcd(width, height)
    aspect_ratio = f"{width // gcd}:{height // gcd}" if gcd else "1:1"

    image_size = "2K"
    if max(width, height) <= 1024:
        image_size = "1K"

    config: Dict[str, str] = {"image_size": image_size}
    if aspect_ratio in _SUPPORTED_RATIOS:
        config["aspect_ratio"] = aspect_ratio
    return config


def _clean_image_items(choices: Any, response_format: str) -> List[Dict[str, str]]:
    if not isinstance(choices, list) or not choices:
        return []

    message = ((choices[0] or {}).get("message") or {}) if isinstance(choices[0], dict) else {}
    images = message.get("images") or []
    if not isinstance(images, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for item in images:
        if not isinstance(item, dict):
            continue

        entry: Dict[str, str] = {}
        image_url = item.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else item.get("url")

        if isinstance(url, str) and url.strip():
            normalized_url = url.strip()
            if "base64," in normalized_url:
                b64_value = normalized_url.split("base64,", 1)[1]
                if response_format == "b64_json":
                    entry["b64_json"] = b64_value
                else:
                    entry["url"] = normalized_url
            else:
                entry["url"] = normalized_url

        if entry:
            cleaned.append(entry)

    return cleaned


def generate_floorplan_image(
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
    response_format: str = "b64_json",
) -> Dict[str, Any]:
    """Generate floor plan concept image(s) using OpenRouter chat completions."""
    if not OPENROUTER_ENABLED:
        raise ValueError("OpenRouter is disabled. Set OPENROUTER_ENABLED=true.")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured.")
    if not OPENROUTER_IMAGE_MODEL:
        raise ValueError("OPENROUTER_IMAGE_MODEL is not configured.")

    model_name = _normalize_image_model(OPENROUTER_IMAGE_MODEL)
    if not model_name:
        raise ValueError("OPENROUTER_IMAGE_MODEL is not configured.")

    fmt = str(response_format or "b64_json").strip().lower()
    if fmt not in _ALLOWED_RESPONSE_FORMATS:
        fmt = "b64_json"

    target_count = max(1, min(int(n), 4))

    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image"],
        "stream": False,
        "image_config": _build_image_config(size),
    }

    request = urllib.request.Request(
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

    try:
        with urllib.request.urlopen(request, timeout=45, context=ssl_context) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.warning("OpenRouter image HTTP failure: %s", exc)
        raise RuntimeError(f"OpenRouter image request failed ({exc.code}). {detail[:400]}") from exc
    except Exception as exc:
        logger.warning("OpenRouter image request failed: %s", exc)
        raise RuntimeError("OpenRouter image request failed.") from exc

    images = _clean_image_items(payload.get("choices"), fmt)
    if not images:
        raise RuntimeError("OpenRouter returned no image data.")

    images = images[:target_count]

    return {
        "provider": "openrouter",
        "model": model_name,
        "count": len(images),
        "data": images,
    }
