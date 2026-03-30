"""Prompt loader for NAKSHA AI PERFCAT architecture system prompt."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "naksha_perfcat_system_prompt.txt"
_FALLBACK_PROMPT = (
    "You are NAKSHA AI, an Indian residential architecture specialist. "
    "Focus only on floor plan decisions, room programs, Vastu-aware placement, and practical family circulation. "
    "Return strict JSON without markdown."
)


@lru_cache(maxsize=1)
def get_naksha_master_system_prompt() -> str:
    """Return the configured NAKSHA master system prompt from disk."""
    try:
        text = _PROMPT_PATH.read_text(encoding="utf-8").strip()
        return text or _FALLBACK_PROMPT
    except Exception:
        return _FALLBACK_PROMPT
