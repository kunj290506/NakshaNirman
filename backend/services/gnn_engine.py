"""Legacy-compatible GNN engine wrapper."""

from __future__ import annotations

from typing import Any, Dict

from services._legacy_adapters import generate_legacy, normalize_requirements


def generate_gnn_floor_plan(**kwargs: Any) -> Dict[str, Any]:
    req = normalize_requirements(kwargs)
    redesign = bool(kwargs.get("_redesign") or kwargs.get("redesign"))
    result = generate_legacy(req, redesign=redesign)
    if "error" in result:
        return result

    layout = result["layout"]
    return {
        "rooms": layout.get("rooms", []),
        "doors": layout.get("doors", []),
        "windows": layout.get("windows", []),
        "validation": layout.get("validation", {}),
        "layout": layout,
        "explanation": result.get("explanation", ""),
        "method": "gnn-compat",
    }
