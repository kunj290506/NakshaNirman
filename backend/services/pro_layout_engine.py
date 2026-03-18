"""Legacy-compatible pro layout engine wrapper."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from services._legacy_adapters import generate_legacy, normalize_requirements


def _boundary_to_dims(boundary: Iterable[Iterable[float]]) -> Tuple[float, float]:
    xs = [float(p[0]) for p in boundary]
    ys = [float(p[1]) for p in boundary]
    return max(xs) - min(xs), max(ys) - min(ys)


def generate_professional_plan(
    boundary_coords: Iterable[Iterable[float]],
    rooms_config: Dict[str, Any],
    total_area: float,
):
    width, length = _boundary_to_dims(boundary_coords)
    bedrooms = int(rooms_config.get("master_bedroom", 0)) + int(rooms_config.get("bedroom", 0))
    bathrooms = int(rooms_config.get("bathroom", 0))
    extras = [k for k, v in rooms_config.items() if k in {"dining", "study", "pooja", "garage", "store", "balcony"} and int(v) > 0]

    req = normalize_requirements(
        {
            "plot_width": width,
            "plot_length": length,
            "total_area": total_area,
            "bedrooms": max(1, bedrooms),
            "bathrooms": max(1, bathrooms),
            "extras": extras,
        }
    )

    result = generate_legacy(req, redesign=False)
    layout = result.get("layout", {})
    rooms = layout.get("rooms", [])

    specs: List[Dict[str, Any]] = []
    centroids: List[Tuple[float, float]] = []
    sizes: List[Tuple[float, float]] = []
    for room in rooms:
        x = float(room["position"]["x"])
        y = float(room["position"]["y"])
        w = float(room["width"])
        h = float(room["length"])
        spec = {
            "name": room["name"],
            "room_type": room["room_type"],
            "_placed": {"x": x, "y": y, "w": w, "h": h},
        }
        specs.append(spec)
        centroids.append((x + w / 2, y + h / 2))
        sizes.append((w, h))

    return centroids, sizes, specs
