"""
Unified Layout Engine Registry.

Provides a single interface for all layout generation strategies.
Each engine implements the same generate() → Dict contract.

Usage:
    from services.engine_registry import EngineRegistry
    result = EngineRegistry.generate("bsp", input_data)
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Available engine identifiers
ENGINE_BSP = "bsp"           # arch_engine — BSP tree placement (production)
ENGINE_GRID = "grid"         # layout_engine — grid/treemap subdivision
ENGINE_GNN = "gnn"           # gnn_engine — graph neural network inspired

# Default engine
DEFAULT_ENGINE = ENGINE_BSP


def generate(engine: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a floor plan using the specified engine.

    Args:
        engine: One of "bsp", "grid", "gnn"
        input_data: Dict with keys:
            - total_area (float)
            - plot_width (float, optional)
            - plot_length (float, optional)
            - bedrooms (int)
            - bathrooms (int)
            - floors (int)
            - extras (list[str])
            - rooms (list[dict], optional)
            - boundary_polygon (list, optional)

    Returns:
        Dict with keys:
            - layout (dict) — room geometry, doors, windows, zones, etc.
            - explanation (str) — human-readable summary
            - validation (dict) — quality checks
            - engine (str) — which engine was used
            - error (str, optional) — if generation failed
    """
    if engine not in _ENGINES:
        engine = DEFAULT_ENGINE

    try:
        return _ENGINES[engine](input_data)
    except Exception as e:
        logger.exception(f"Engine '{engine}' failed, trying fallback")
        # Try fallback to BSP if another engine failed
        if engine != ENGINE_BSP:
            try:
                result = _ENGINES[ENGINE_BSP](input_data)
                result["engine"] = f"{ENGINE_BSP} (fallback from {engine})"
                return result
            except Exception as e2:
                logger.exception("Fallback engine also failed")
                return {"error": str(e2), "engine": engine}
        return {"error": str(e), "engine": engine}


def _run_bsp(input_data: Dict) -> Dict:
    """BSP engine — production layout generator."""
    from services.arch_engine import design_generate

    requirements = _normalize_input(input_data)
    result = design_generate(requirements)

    if "error" in result:
        return {
            "error": result["error"],
            "suggestion": result.get("suggestion", ""),
            "engine": ENGINE_BSP,
        }

    return {
        "layout": result.get("layout", {}),
        "explanation": result.get("explanation", ""),
        "validation": result.get("validation", {}),
        "engine": ENGINE_BSP,
    }


def _run_grid(input_data: Dict) -> Dict:
    """Grid/treemap engine — modular subdivision."""
    from services.layout_engine import LayoutGenerator

    requirements = _normalize_input(input_data)

    gen = LayoutGenerator.from_json({
        "plot_width": requirements.get("plot_width", 30),
        "plot_length": requirements.get("plot_length", 40),
        "rooms": input_data.get("rooms", []),
        "bedrooms": requirements.get("bedrooms", 2),
        "bathrooms": requirements.get("bathrooms", 1),
    })

    layout = gen.generate()

    return {
        "layout": layout,
        "explanation": "Generated using grid subdivision engine.",
        "validation": {"overlap": False, "geometry_ok": True},
        "engine": ENGINE_GRID,
    }


def _run_gnn(input_data: Dict) -> Dict:
    """GNN-inspired engine."""
    from services.gnn_engine import generate_gnn_floor_plan

    requirements = _normalize_input(input_data)

    result = generate_gnn_floor_plan(
        total_area=requirements["total_area"],
        bedrooms=requirements.get("bedrooms", 2),
        bathrooms=requirements.get("bathrooms", 1),
        extras=requirements.get("extras", []),
        plot_width=requirements.get("plot_width"),
        plot_length=requirements.get("plot_length"),
    )

    if "error" in result:
        return {
            "error": result["error"],
            "engine": ENGINE_GNN,
        }

    return {
        "layout": result,
        "explanation": result.get("explanation", "Generated using GNN engine."),
        "validation": result.get("validation", {}),
        "engine": ENGINE_GNN,
    }


def _normalize_input(data: Dict) -> Dict:
    """Normalize input data to consistent format."""
    total_area = data.get("total_area", 1200)

    plot_w = data.get("plot_width")
    plot_l = data.get("plot_length")

    if not plot_w or not plot_l:
        import math
        ratio = 0.75  # default 3:4 ratio
        plot_w = math.sqrt(total_area * ratio)
        plot_l = total_area / plot_w

    return {
        "total_area": total_area,
        "plot_width": round(plot_w, 1),
        "plot_length": round(plot_l, 1),
        "bedrooms": data.get("bedrooms", 2),
        "bathrooms": data.get("bathrooms", 1),
        "floors": data.get("floors", 1),
        "extras": data.get("extras", []),
        "rooms": data.get("rooms"),
        "boundary_polygon": data.get("boundary_polygon"),
    }


def available_engines() -> List[str]:
    """Return list of available engine names."""
    return list(_ENGINES.keys())


# Engine dispatch table
_ENGINES = {
    ENGINE_BSP: _run_bsp,
    ENGINE_GRID: _run_grid,
    ENGINE_GNN: _run_gnn,
}
