"""Legacy arch engine compatibility shim."""

from __future__ import annotations

from typing import Any, Dict

from services.multi_factor_engine import generate_plan


def generate_architect_layout(input_data: Dict[str, Any]) -> Dict[str, Any]:
    return generate_plan(input_data)
