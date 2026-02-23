"""
Polygon clipping and overlap validation utilities.

Ensures all generated rooms fit within the usable boundary polygon
and do not overlap each other.
"""

from typing import List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.validation import make_valid


def clip_to_boundary(room_polygon: Polygon, boundary: Polygon) -> Optional[Polygon]:
    """
    Clip *room_polygon* to lie within *boundary*.

    Returns the clipped polygon, or ``None`` if the intersection
    has zero area (the room fell entirely outside).
    """
    if not room_polygon.is_valid:
        room_polygon = make_valid(room_polygon)
    if not boundary.is_valid:
        boundary = make_valid(boundary)

    clipped = room_polygon.intersection(boundary)
    if clipped.is_empty or clipped.area < 1e-6:
        return None

    # intersection may return a collection; keep only polygonal parts
    if clipped.geom_type == "Polygon":
        return clipped
    if clipped.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in clipped.geoms if g.geom_type == "Polygon" and g.area > 1e-6]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def detect_overlaps(rooms: List[Polygon],
                     tolerance: float = 0.01) -> List[Tuple[int, int]]:
    """
    Return a list of (i, j) index pairs for rooms that overlap.

    Rooms sharing only an edge (zero-area intersection) are **not**
    considered overlapping.

    Parameters
    ----------
    rooms : list[Polygon]
        Room polygons to check.
    tolerance : float
        Minimum intersection area to count as an overlap (sq m).
    """
    overlaps = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            inter = rooms[i].intersection(rooms[j])
            if inter.area > tolerance:
                overlaps.append((i, j))
    return overlaps


def has_overlaps(rooms: List[Polygon], tolerance: float = 0.01) -> bool:
    """Quick check — are there *any* overlapping room pairs?"""
    return len(detect_overlaps(rooms, tolerance)) > 0


def total_coverage(rooms: List[Polygon]) -> float:
    """Sum of individual room areas (not union — may double-count overlaps)."""
    return sum(r.area for r in rooms)


def rooms_within_boundary(rooms: List[Polygon],
                           boundary: Polygon,
                           min_coverage: float = 0.95) -> bool:
    """
    Check that rooms collectively cover at least *min_coverage* fraction
    of the boundary area.
    """
    from shapely.ops import unary_union

    merged = unary_union(rooms)
    covered = merged.intersection(boundary).area
    return covered >= boundary.area * min_coverage
