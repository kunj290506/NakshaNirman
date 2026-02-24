"""
Main layout generator — public API for Phase 3.

Coordinates grid subdivision, room placement, geometry conversion,
clipping, validation, scoring, and candidate selection.
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .adjacency import build_adjacency_graph, is_connected
from .doors import Door, place_doors
from .entrance import place_entrance
from .geometry_utils import clip_to_boundary, has_overlaps
from .loaders import load_usable_polygon, load_min_areas
from .placement import compute_room_specs, place_all_rooms
from .room_model import Room
from .scoring import score_layout, corridor_penalty
from .subdivision import SubdivisionGrid
from .treemap import treemap_subdivide


class LayoutGenerator:
    """
    Generate and rank single-floor room layouts inside a boundary polygon.

    Typical workflow::

        gen = LayoutGenerator(boundary_polygon, room_requirements)
        best = gen.generate(n_candidates=200)
    """

    def __init__(
        self,
        boundary: Polygon,
        room_requirements: List[dict],
        min_areas: Optional[Dict[str, float]] = None,
        desired_adjacencies: Optional[List[tuple]] = None,
    ):
        """
        Parameters
        ----------
        boundary : Polygon
            The usable floor polygon (Shapely).
        room_requirements : list[dict]
            Each dict: ``{"room_type": str, "size": int}``.
            ``size`` is a relative weight (e.g. 7 for living, 3 for bath).
        min_areas : dict, optional
            ``{room_type: min_area_sqm}``.  Candidates violating these
            are rejected.
        desired_adjacencies : list[tuple], optional
            Pairs ``(type_a, type_b)`` the scorer rewards.
        """
        self.boundary = boundary
        self.room_requirements = room_requirements
        self.min_areas = min_areas or {}
        self.desired_adjacencies = desired_adjacencies or []

    # ------------------------------------------------------------------
    # Convenience: build from file paths
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        polygon_path: str,
        room_requirements: List[dict],
        rules_path: Optional[str] = None,
        region: str = "india_mvp",
        desired_adjacencies: Optional[List[tuple]] = None,
    ) -> "LayoutGenerator":
        """
        Build a LayoutGenerator by loading the usable polygon (and
        optionally min-area rules) from JSON files.

        Parameters
        ----------
        polygon_path : str
            Path to usable_polygon.json (Step 4).
        room_requirements : list[dict]
            Room specs: ``[{"room_type": str, "size": int}, ...]``.
        rules_path : str, optional
            Path to region_rules.json (Step 7).
        region : str
            Region key inside region_rules.json.
        desired_adjacencies : list[tuple], optional
            Pairs ``(type_a, type_b)`` to reward adjacency.
        """
        boundary = load_usable_polygon(polygon_path)
        min_areas = {}
        if rules_path:
            min_areas = load_min_areas(rules_path, region)
        return cls(
            boundary=boundary,
            room_requirements=room_requirements,
            min_areas=min_areas,
            desired_adjacencies=desired_adjacencies,
        )

    # ------------------------------------------------------------------
    # Internal: single candidate via grid method
    # ------------------------------------------------------------------

    def _generate_grid_candidate(
        self, seed: Optional[int] = None
    ) -> Optional[List[Room]]:
        """Generate one layout candidate using grid subdivision."""
        minx, miny, maxx, maxy = self.boundary.bounds
        width = maxx - minx
        height = maxy - miny

        # Choose cell size so grid has reasonable resolution
        cell_size = min(width, height) / max(len(self.room_requirements) * 2, 8)
        cell_size = max(cell_size, 0.25)  # never smaller than 25 cm

        cols = max(int(width / cell_size), 4)
        rows = max(int(height / cell_size), 4)

        grid = SubdivisionGrid(cols, rows, cell_size)
        grooms = compute_room_specs(self.room_requirements, grid.interior_area)
        placed = place_all_rooms(grid, grooms, seed=seed)

        Room.reset_counter()
        rooms: List[Room] = []
        for gr in placed:
            poly = grid.cells_to_polygon(gr.cells, origin_x=minx, origin_y=miny)
            clipped = clip_to_boundary(poly, self.boundary)
            if clipped is None or clipped.area < 0.1:
                return None  # reject this candidate
            target = gr.area_wanted * (cell_size ** 2)
            rooms.append(
                Room(
                    room_type=gr.room_type,
                    polygon=clipped,
                    target_area=target,
                    floor=0,
                )
            )
        return rooms

    # ------------------------------------------------------------------
    # Internal: single candidate via treemap method
    # ------------------------------------------------------------------

    def _generate_treemap_candidate(
        self, seed: Optional[int] = None
    ) -> Optional[List[Room]]:
        """Generate one layout candidate using squarified treemaps."""
        if seed is not None:
            random.seed(seed)

        minx, miny, maxx, maxy = self.boundary.bounds
        width = maxx - minx
        height = maxy - miny

        # Randomly shuffle order for variety
        reqs = list(self.room_requirements)
        random.shuffle(reqs)

        areas = [r["size"] ** 2 for r in reqs]
        total_wanted = sum(areas) if areas else 1.0
        boundary_area = width * height
        area_scale = boundary_area / total_wanted
        polys = treemap_subdivide(width, height, areas, origin_x=minx, origin_y=miny)

        Room.reset_counter()
        rooms: List[Room] = []
        for req, poly in zip(reqs, polys):
            clipped = clip_to_boundary(poly, self.boundary)
            if clipped is None or clipped.area < 0.1:
                return None
            target = req["size"] ** 2 * area_scale
            rooms.append(
                Room(
                    room_type=req["room_type"],
                    polygon=clipped,
                    target_area=target,
                    floor=0,
                )
            )
        return rooms

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, rooms: List[Room]) -> bool:
        """Run rejection checks on a candidate layout."""
        if not rooms:
            return False

        # 1. Minimum area enforcement
        for r in rooms:
            min_a = self.min_areas.get(r.room_type, 0)
            if r.area < min_a:
                return False

        # 2. Overlap detection (entrance may overlap adjacent rooms — exclude it)
        non_entrance = [r.polygon for r in rooms if r.room_type != "entrance"]
        if has_overlaps(non_entrance, tolerance=0.05):
            return False

        # 3. Connectivity check
        room_dicts = [
            {"room_id": r.room_id, "room_type": r.room_type, "polygon": r.polygon}
            for r in rooms
        ]
        graph = build_adjacency_graph(room_dicts)
        if not is_connected(graph):
            return False

        return True

    # ------------------------------------------------------------------
    # Corridor metric (Step 12)
    # ------------------------------------------------------------------

    def _compute_corridor(self, rooms: List[Room]) -> Dict[str, float]:
        """
        Compute corridor / wasted space as:
            usable_area − sum(room_areas)

        Returns a dict with 'area' and 'fraction'.
        """
        boundary_area = self.boundary.area
        room_area_sum = sum(r.area for r in rooms)
        corridor_area = max(0.0, boundary_area - room_area_sum)
        fraction = corridor_area / boundary_area if boundary_area > 0 else 0.0
        return {
            "area": round(corridor_area, 4),
            "fraction": round(fraction, 4),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_candidates: int = 200,
        method: str = "mixed",
    ) -> Dict[str, Any]:
        """
        Generate *n_candidates* layouts and return the best one.

        Parameters
        ----------
        n_candidates : int
            How many candidates to attempt (invalid ones are discarded).
        method : str
            ``"grid"``, ``"treemap"``, or ``"mixed"`` (default).

        Returns
        -------
        dict
            ``best_layout`` (list of Room dicts), ``score`` (dict),
            ``candidates_generated``, ``candidates_valid``.
        """
        candidates: List[Tuple[List[Room], Dict, List[Door], Dict]] = []

        for i in range(n_candidates):
            seed = i * 17 + 42  # deterministic but varied

            if method == "grid":
                rooms = self._generate_grid_candidate(seed)
            elif method == "treemap":
                rooms = self._generate_treemap_candidate(seed)
            else:
                # mixed: alternate between methods
                if i % 2 == 0:
                    rooms = self._generate_grid_candidate(seed)
                else:
                    rooms = self._generate_treemap_candidate(seed)

            if rooms is None:
                continue

            # Step 9 — Entrance placement
            entrance = place_entrance(self.boundary, rooms)
            if entrance is not None:
                rooms.append(entrance)

            if not self._validate(rooms):
                continue

            # Step 11 — Door placement
            doors = place_doors(rooms)

            # Step 12 — Corridor detection
            corridor_info = self._compute_corridor(rooms)

            scores = score_layout(
                rooms,
                self.boundary,
                self.desired_adjacencies,
            )
            candidates.append((rooms, scores, doors, corridor_info))

        if not candidates:
            return {
                "best_layout": [],
                "doors": [],
                "corridor": {"area": 0, "fraction": 0},
                "score": {"total": 0, "area": 0, "adjacency": 0, "corridor": 0},
                "candidates_generated": n_candidates,
                "candidates_valid": 0,
            }

        # Step 15 — Select highest scoring candidate
        best_rooms, best_score, best_doors, best_corridor = max(
            candidates, key=lambda c: c[1]["total"]
        )

        return {
            "best_layout": [r.to_dict() for r in best_rooms],
            "doors": [d.to_dict() for d in best_doors],
            "corridor": best_corridor,
            "score": best_score,
            "candidates_generated": n_candidates,
            "candidates_valid": len(candidates),
        }

    def generate_all_valid(
        self,
        n_candidates: int = 200,
        method: str = "mixed",
    ) -> List[Dict[str, Any]]:
        """
        Like ``generate()`` but return *all* valid candidates ranked by score.
        """
        candidates: List[Dict[str, Any]] = []

        for i in range(n_candidates):
            seed = i * 17 + 42
            if method == "grid":
                rooms = self._generate_grid_candidate(seed)
            elif method == "treemap":
                rooms = self._generate_treemap_candidate(seed)
            else:
                rooms = (
                    self._generate_grid_candidate(seed)
                    if i % 2 == 0
                    else self._generate_treemap_candidate(seed)
                )

            if rooms is None:
                continue

            # Step 9 — Entrance placement
            entrance = place_entrance(self.boundary, rooms)
            if entrance is not None:
                rooms.append(entrance)

            if not self._validate(rooms):
                continue

            # Step 11 — Door placement
            doors = place_doors(rooms)

            # Step 12 — Corridor detection
            corridor_info = self._compute_corridor(rooms)

            scores = score_layout(rooms, self.boundary, self.desired_adjacencies)
            candidates.append({
                "layout": [r.to_dict() for r in rooms],
                "doors": [d.to_dict() for d in doors],
                "corridor": corridor_info,
                "score": scores,
            })

        candidates.sort(key=lambda c: c["score"]["total"], reverse=True)
        return candidates
