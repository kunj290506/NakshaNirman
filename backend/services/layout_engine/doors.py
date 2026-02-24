"""
Door placement on shared walls between adjacent rooms.

For each pair of rooms that share a wall, a door is placed at the
midpoint of the shared boundary segment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon


# Default door dimensions (meters)
DEFAULT_DOOR_WIDTH = 0.9


@dataclass
class Door:
    """A single door placed on a shared wall between two rooms."""

    room_a_id: int
    room_b_id: int
    position: Tuple[float, float]       # midpoint (x, y)
    width: float = DEFAULT_DOOR_WIDTH
    door_id: Optional[int] = None

    # Class-level ID counter
    _next_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.door_id is None:
            self.door_id = Door._next_id
            Door._next_id += 1

    @property
    def geometry(self) -> Point:
        """Door location as a Shapely Point."""
        return Point(self.position)

    def to_dict(self) -> dict:
        """Serialize door to a dictionary."""
        return {
            "door_id": self.door_id,
            "room_a_id": self.room_a_id,
            "room_b_id": self.room_b_id,
            "position": {"x": round(self.position[0], 4),
                         "y": round(self.position[1], 4)},
            "width": self.width,
        }

    @staticmethod
    def reset_counter():
        """Reset the auto-increment ID counter."""
        Door._next_id = 0

    def __repr__(self) -> str:
        return (
            f"Door(id={self.door_id}, rooms=({self.room_a_id},{self.room_b_id}), "
            f"pos=({self.position[0]:.2f},{self.position[1]:.2f}))"
        )


def _shared_wall_midpoint(
    poly_a: Polygon, poly_b: Polygon
) -> Optional[Tuple[float, float]]:
    """
    Compute the midpoint of the shared boundary between two polygons.

    Returns None if there is no shared linear boundary.
    """
    shared = poly_a.intersection(poly_b)
    if shared.is_empty or shared.length < 0.01:
        return None
    mid = shared.centroid
    return (mid.x, mid.y)


def place_doors(
    rooms: list,
    tolerance: float = 0.05,
    door_width: float = DEFAULT_DOOR_WIDTH,
) -> List[Door]:
    """
    Place doors at the midpoint of every shared wall.

    Parameters
    ----------
    rooms : list
        List of Room objects (must have ``room_id`` and ``polygon``).
    tolerance : float
        Minimum shared boundary length (m) to consider as a wall.
    door_width : float
        Width of each door (meters).

    Returns
    -------
    list[Door]
        One door per shared wall.
    """
    Door.reset_counter()
    doors: List[Door] = []

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            poly_i = rooms[i].polygon
            poly_j = rooms[j].polygon

            shared = poly_i.intersection(poly_j)
            if shared.is_empty or shared.length < tolerance:
                continue

            midpoint = _shared_wall_midpoint(poly_i, poly_j)
            if midpoint is None:
                continue

            doors.append(
                Door(
                    room_a_id=rooms[i].room_id,
                    room_b_id=rooms[j].room_id,
                    position=midpoint,
                    width=door_width,
                )
            )

    return doors
