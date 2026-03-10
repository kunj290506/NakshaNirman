"""
Entrance placement on the outer boundary.

Places an entrance rectangle on the exterior wall of the usable polygon.
The entrance is connected to the nearest interior room.
"""

import random
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import nearest_points

from .room_model import Room


# Default entrance dimensions (meters)
DEFAULT_ENTRANCE_WIDTH = 1.2
DEFAULT_ENTRANCE_DEPTH = 1.5


def find_entrance_wall_segment(
    boundary: Polygon,
    preferred_side: str = "south",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Pick a wall segment on the boundary for the entrance.
    Always uses south wall (road-facing). preferred_side parameter is kept
    for API compatibility but overridden to "south".
    """
    preferred_side = "south"   # ALWAYS south — road is always south
    coords = list(boundary.exterior.coords)[:-1]  # drop closing duplicate
    segments = [(coords[i], coords[(i + 1) % len(coords)]) for i in range(len(coords))]

    minx, miny, maxx, maxy = boundary.bounds
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # Direction target point (far off in the preferred direction)
    targets = {
        "south": (cx, miny - 100),
        "north": (cx, maxy + 100),
        "west":  (minx - 100, cy),
        "east":  (maxx + 100, cy),
    }
    target = targets.get(preferred_side, targets["south"])

    # Pick the segment whose midpoint is closest to the target
    best_seg = segments[0]
    best_dist = float("inf")
    for seg in segments:
        mid_x = (seg[0][0] + seg[1][0]) / 2
        mid_y = (seg[0][1] + seg[1][1]) / 2
        dist = (mid_x - target[0]) ** 2 + (mid_y - target[1]) ** 2
        # Also prefer longer segments (more room for a door)
        seg_len = ((seg[1][0] - seg[0][0]) ** 2 + (seg[1][1] - seg[0][1]) ** 2) ** 0.5
        if seg_len < DEFAULT_ENTRANCE_WIDTH * 0.8:
            continue  # skip segments too short for an entrance
        if dist < best_dist:
            best_dist = dist
            best_seg = seg

    return best_seg


def place_entrance(
    boundary: Polygon,
    rooms: List[Room],
    entrance_width: float = DEFAULT_ENTRANCE_WIDTH,
    entrance_depth: float = DEFAULT_ENTRANCE_DEPTH,
    preferred_side: str = "south",
) -> Optional[Room]:
    """
    Place an entrance rectangle on the south wall of the boundary.
    preferred_side is kept for API compatibility but always overridden to south.
    """
    preferred_side = "south"   # ALWAYS south — road is always south
    seg_start, seg_end = find_entrance_wall_segment(boundary, preferred_side)

    # Midpoint of the segment
    mx = (seg_start[0] + seg_end[0]) / 2
    my = (seg_start[1] + seg_end[1]) / 2

    # Segment direction vector
    dx = seg_end[0] - seg_start[0]
    dy = seg_end[1] - seg_start[1]
    seg_len = (dx ** 2 + dy ** 2) ** 0.5

    if seg_len < 1e-6:
        return None

    # Unit vectors: along wall and inward normal
    ux, uy = dx / seg_len, dy / seg_len  # along the wall
    # Inward normal — perpendicular, pointing inside the polygon
    nx, ny = -uy, ux  # candidate normal

    # Check which direction is inward by testing a point
    test_point = Point(mx + nx * 0.1, my + ny * 0.1)
    if not boundary.contains(test_point):
        nx, ny = -nx, -ny

    # Build entrance rectangle:
    # Along the wall: half-width each side of midpoint
    hw = entrance_width / 2
    # Four corners
    corners = [
        (mx - ux * hw, my - uy * hw),                              # wall left
        (mx + ux * hw, my + uy * hw),                              # wall right
        (mx + ux * hw + nx * entrance_depth,
         my + uy * hw + ny * entrance_depth),                      # inner right
        (mx - ux * hw + nx * entrance_depth,
         my - uy * hw + ny * entrance_depth),                      # inner left
    ]

    entrance_poly = Polygon(corners)

    # Clip to boundary to ensure it stays inside
    clipped = entrance_poly.intersection(boundary)
    if clipped.is_empty or clipped.area < 0.01:
        return None

    if clipped.geom_type != "Polygon":
        polys = [g for g in clipped.geoms if g.geom_type == "Polygon"]
        if not polys:
            return None
        clipped = max(polys, key=lambda p: p.area)

    # Verify connection to at least one interior room
    connected = False
    for room in rooms:
        shared = clipped.intersection(room.polygon)
        if shared.length > 0.01:  # shares a boundary
            connected = True
            break

    if not connected:
        # Try touching/overlapping check
        for room in rooms:
            if clipped.touches(room.polygon) or clipped.intersects(room.polygon):
                connected = True
                break

    if not connected:
        return None

    entrance_room = Room(
        room_type="entrance",
        polygon=clipped,
        target_area=entrance_width * entrance_depth,
        floor=0,
    )
    return entrance_room
