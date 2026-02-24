"""
Layout Engine for Floor Plan Generation.

Provides grid-based and treemap-based subdivision of arbitrary polygons
into room layouts. All geometry uses Shapely polygons.
"""

from .generator import LayoutGenerator
from .loaders import load_usable_polygon, save_usable_polygon, load_min_areas, load_region_rules
from .entrance import place_entrance
from .room_model import Room
from .doors import Door, place_doors

__all__ = [
    "LayoutGenerator",
    "load_usable_polygon",
    "save_usable_polygon",
    "load_min_areas",
    "load_region_rules",
    "place_entrance",
    "Room",
    "Door",
    "place_doors",
]
