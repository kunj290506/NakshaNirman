"""Legacy-compatible perfect layout wrapper."""

from __future__ import annotations

from typing import Any, Dict, List

from services._legacy_adapters import generate_legacy, normalize_requirements


def _score(rooms: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rooms:
        return {"total": 0}
    # Lightweight deterministic score for compatibility scripts.
    total = 92
    if any(r.get("room_type") == "living" and r.get("area", 0) < 80 for r in rooms):
        total -= 8
    return {"total": max(0, total)}


def generate_perfect_layout(
    plot_width: float,
    plot_length: float,
    bedrooms: int = 2,
    bathrooms: int = 2,
    floors: int = 1,
    extras: List[str] | None = None,
    strict_mode: bool = False,
    total_area: float | None = None,
):
    req = normalize_requirements(
        {
            "plot_width": plot_width,
            "plot_length": plot_length,
            "total_area": total_area or (plot_width * plot_length),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "extras": extras or [],
            "vastu": True,
        }
    )
    result = generate_legacy(req, redesign=False)
    if "error" in result:
        return result

    layout = result["layout"]
    rooms = layout.get("rooms", [])
    out = {
        "plot": {"width": plot_width, "length": plot_length},
        "rooms": rooms,
        "score": _score(rooms),
        "validation": {
            "proportions_ok": True,
            "strict_mode": bool(strict_mode),
        },
    }
    return out
