"""Pytest compatibility fixtures for legacy script-style test files."""

from __future__ import annotations

from typing import Any, List

import pytest

from services.gnn_engine import generate_gnn_floor_plan
from services.perfect_layout import generate_perfect_layout
from services.pro_layout_engine import generate_professional_plan


@pytest.fixture(params=["PRO", "GNN", "PERFECT"])
def engine_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def configs(request: pytest.FixtureRequest) -> Any:
    return getattr(request.module, "CONFIGS", [])


@pytest.fixture
def generate_fn(engine_name: str):
    if engine_name == "PRO":
        return generate_professional_plan
    if engine_name == "GNN":
        return generate_gnn_floor_plan
    return generate_perfect_layout


@pytest.fixture
def label() -> str:
    return "Compatibility Fixture Run"


@pytest.fixture
def boundary() -> List[List[float]]:
    return [[0, 0], [30, 0], [30, 20], [0, 20]]


@pytest.fixture
def rooms_config() -> dict:
    return {
        "living": 1,
        "master_bedroom": 1,
        "bedroom": 1,
        "kitchen": 1,
        "bathroom": 2,
        "dining": 1,
    }


@pytest.fixture
def total_area() -> float:
    return 600.0


@pytest.fixture
def pw() -> float:
    return 30.0


@pytest.fixture
def pl() -> float:
    return 20.0


@pytest.fixture
def beds() -> int:
    return 2


@pytest.fixture
def baths() -> int:
    return 2


@pytest.fixture
def extras() -> List[str]:
    return ["dining"]
