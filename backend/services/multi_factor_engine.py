"""Legacy-compatible multi-factor engine wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional

from services._legacy_adapters import (
    COMFORT_AR,
    MIN_ROOM_AREA,
    WALL_EXT,
    generate_legacy,
    normalize_requirements,
    parse_input_text,
    select_strategy,
)


def _select_strategy(plot_width: float, plot_length: float) -> str:
    return select_strategy(float(plot_width), float(plot_length))


def _normalize_requirements(req: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_requirements(req)


def parse_input(text: str) -> Dict[str, Any]:
    parsed = parse_input_text(text)
    normalized = normalize_requirements(parsed)
    normalized["is_redesign"] = bool(parsed.get("is_redesign"))
    return normalized


def generate_plan(requirements: Dict[str, Any]) -> Dict[str, Any]:
    req = normalize_requirements(requirements)
    return generate_legacy(req, redesign=False)


def generate_new_plan(requirements: Dict[str, Any], previous_strategy: Optional[str] = None) -> Dict[str, Any]:
    req = normalize_requirements(requirements)
    _ = previous_strategy
    return generate_legacy(req, redesign=True)
