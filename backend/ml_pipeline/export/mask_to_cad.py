"""
Convert predicted layout masks into CAD-style drawings.

Pipeline:
  1. Predicted mask (H×W int) → vectorise room regions (shapely polygons)
  2. Simplify & snap to grid → clean rectangular rooms
  3. Compute wall centre-lines → double-line walls with proper thickness
  4. Place door symbols (hinged arcs) on shared internal walls
  5. Place window symbols on exterior walls
  6. Add room labels with area text
  7. Add dimension lines (millimetres — Indian standard)
  8. Export to DXF via ezdxf

Ensures output is a realistic, construction-ready concept plan.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY = True
except ImportError:
    SHAPELY = False

try:
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    EZDXF = True
except ImportError:
    EZDXF = False

from ml_pipeline.config import (
    PipelineConfig,
    ROOM_TYPES,
    ROOM_TO_IDX,
    NUM_ROOM_CLASSES,
    EXPORT_DIR,
)

logger = logging.getLogger(__name__)

# Indian Building Code constants
WALL_EXT_MM = 230   # 9 inches
WALL_INT_MM = 115   # 4.5 inches
WALL_EXT_FT = 0.75
WALL_INT_FT = 0.375
FT_TO_MM = 304.8
DOOR_WIDTH_FT = 2.5
WINDOW_WIDTH_FT = 4.0

# Room display names (Indian market)
DISPLAY_NAMES = {
    "living_room": "Drawing Room",
    "master_bedroom": "Master Bed Room",
    "bedroom": "Bed Room",
    "kitchen": "Kitchen",
    "bathroom": "Bath",
    "toilet": "Toilet",
    "dining": "Dining Area",
    "study": "Study",
    "pooja": "Puja Room",
    "store": "Store Room",
    "utility": "Wash Area",
    "balcony": "Balcony",
    "staircase": "Staircase",
    "hallway": "Passage",
    "parking": "Parking",
    "porch": "Porch",
    "garden": "Garden",
}


class MaskToCAD:
    """
    Convert a predicted room mask into a professional DXF drawing.

    Parameters
    ----------
    cfg : PipelineConfig
    plot_width_ft, plot_length_ft : real-world plot dimensions
    """

    def __init__(
        self,
        cfg: Optional[PipelineConfig] = None,
        plot_width_ft: float = 30.0,
        plot_length_ft: float = 40.0,
    ):
        self.cfg = cfg or PipelineConfig()
        self.plot_w = plot_width_ft
        self.plot_l = plot_length_ft
        self.scale_x = plot_width_ft / self.cfg.img_size
        self.scale_y = plot_length_ft / self.cfg.img_size

    # ==================================================================
    # Main entry point
    # ==================================================================

    def convert(
        self,
        mask: np.ndarray,             # (H, W) int   — class indices
        output_path: Optional[str] = None,
        boundary_coords: Optional[List[List[float]]] = None,
    ) -> Dict:
        """
        Convert predicted mask → vectorised rooms → DXF file.

        Returns dict with:
          rooms: list of room dicts (polygon, area, label, doors, windows)
          dxf_path: path to generated DXF (or None if ezdxf not available)
          boundary: plot boundary polygon
        """
        # Step 1: Vectorise rooms
        rooms = self._vectorise_rooms(mask)

        # Step 2: Snap to grid & simplify
        rooms = self._snap_to_grid(rooms)

        # Step 3: Detect shared walls → place doors & windows
        rooms = self._place_doors_windows(rooms)

        # Step 4: Build boundary
        if boundary_coords is None:
            boundary_coords = [
                [0, 0],
                [self.plot_w, 0],
                [self.plot_w, self.plot_l],
                [0, self.plot_l],
                [0, 0],
            ]

        result = {
            "rooms": rooms,
            "boundary": boundary_coords,
        }

        # Step 5: Export DXF
        if output_path and EZDXF:
            dxf_path = self._export_dxf(rooms, boundary_coords, output_path)
            result["dxf_path"] = dxf_path
        else:
            result["dxf_path"] = None

        return result

    # ==================================================================
    # Step 1: Vectorise
    # ==================================================================

    def _vectorise_rooms(self, mask: np.ndarray) -> List[Dict]:
        """Extract room bounding boxes from class mask."""
        ext_idx = ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1)
        wall_idx = ROOM_TO_IDX.get("wall", -1)
        door_idx = ROOM_TO_IDX.get("door", -1)
        window_idx = ROOM_TO_IDX.get("window", -1)
        skip = {ext_idx, wall_idx, door_idx, window_idx}

        rooms = []
        seen_types = {}

        for cidx in range(NUM_ROOM_CLASSES):
            if cidx in skip:
                continue
            region = (mask == cidx)
            if not region.any():
                continue

            ys, xs = np.where(region)
            px1, py1, px2, py2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

            # Convert pixel → feet
            x1 = px1 * self.scale_x
            y1 = py1 * self.scale_y
            x2 = px2 * self.scale_x
            y2 = py2 * self.scale_y

            rtype = ROOM_TYPES[cidx] if cidx < len(ROOM_TYPES) else f"room_{cidx}"
            seen_types[rtype] = seen_types.get(rtype, 0) + 1
            count = seen_types[rtype]

            base_name = DISPLAY_NAMES.get(rtype, rtype.replace("_", " ").title())
            name = base_name if count == 1 else f"{base_name} {count}"

            w = round(x2 - x1, 1)
            h = round(y2 - y1, 1)

            rooms.append({
                "room_type": rtype,
                "name": name,
                "position": {"x": round(x1, 1), "y": round(y1, 1)},
                "width": w,
                "length": h,
                "area": round(w * h, 1),
                "polygon": [
                    [round(x1, 1), round(y1, 1)],
                    [round(x2, 1), round(y1, 1)],
                    [round(x2, 1), round(y2, 1)],
                    [round(x1, 1), round(y2, 1)],
                    [round(x1, 1), round(y1, 1)],
                ],
                "centroid": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                "doors": [],
                "windows": [],
            })

        return rooms

    # ==================================================================
    # Step 2: Snap to grid
    # ==================================================================

    def _snap_to_grid(self, rooms: List[Dict], grid_ft: float = 0.5) -> List[Dict]:
        """Snap room coordinates to a 6-inch (0.5 ft) grid."""
        for r in rooms:
            r["position"]["x"] = round(r["position"]["x"] / grid_ft) * grid_ft
            r["position"]["y"] = round(r["position"]["y"] / grid_ft) * grid_ft
            r["width"] = max(round(r["width"] / grid_ft) * grid_ft, 3.0)
            r["length"] = max(round(r["length"] / grid_ft) * grid_ft, 3.0)
            r["area"] = round(r["width"] * r["length"], 1)

            x, y, w, h = (r["position"]["x"], r["position"]["y"],
                           r["width"], r["length"])
            r["polygon"] = [
                [x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]
            ]
            r["centroid"] = [round(x + w / 2, 1), round(y + h / 2, 1)]
        return rooms

    # ==================================================================
    # Step 3: Doors & windows
    # ==================================================================

    def _place_doors_windows(self, rooms: List[Dict]) -> List[Dict]:
        """
        Detect shared walls between adjacent rooms and place:
          - Doors on internal shared walls
          - Windows on exterior walls (bedrooms, living)
        """
        service_types = {"kitchen", "bathroom", "toilet", "utility", "store"}
        habitable_types = {"living_room", "master_bedroom", "bedroom",
                           "study", "dining"}

        for i, room in enumerate(rooms):
            rx, ry = room["position"]["x"], room["position"]["y"]
            rw, rl = room["width"], room["length"]
            rtype = room["room_type"]

            # Find nearest neighbour on each wall
            walls = {
                "S": (rx, ry, rx + rw, ry),
                "N": (rx, ry + rl, rx + rw, ry + rl),
                "W": (rx, ry, rx, ry + rl),
                "E": (rx + rw, ry, rx + rw, ry + rl),
            }

            has_door = False
            for wall_name, (wx1, wy1, wx2, wy2) in walls.items():
                # Check if wall is shared with another room
                for j, other in enumerate(rooms):
                    if i == j:
                        continue
                    ox, oy = other["position"]["x"], other["position"]["y"]
                    ow, ol = other["width"], other["length"]

                    shared = self._wall_overlap(
                        wall_name, rx, ry, rw, rl, ox, oy, ow, ol
                    )
                    if shared > 2.0 and not has_door:
                        room["doors"].append({
                            "wall": wall_name,
                            "width": DOOR_WIDTH_FT,
                            "to_room": other["name"],
                        })
                        has_door = True
                        break

            # Windows on exterior walls (touching plot boundary)
            if rtype in habitable_types:
                if abs(ry) < 1.0:  # near south boundary
                    room["windows"].append({"wall": "S", "width": WINDOW_WIDTH_FT})
                if abs(ry + rl - self.plot_l) < 1.0:  # near north
                    room["windows"].append({"wall": "N", "width": WINDOW_WIDTH_FT})
                if abs(rx) < 1.0:  # near west
                    room["windows"].append({"wall": "W", "width": WINDOW_WIDTH_FT})
                if abs(rx + rw - self.plot_w) < 1.0:  # near east
                    room["windows"].append({"wall": "E", "width": WINDOW_WIDTH_FT})

        return rooms

    @staticmethod
    def _wall_overlap(
        wall: str, rx: float, ry: float, rw: float, rl: float,
        ox: float, oy: float, ow: float, ol: float,
    ) -> float:
        """Compute overlap length between a wall of room R and room O."""
        tol = 0.5
        if wall == "S":  # bottom edge of R
            if abs(oy + ol - ry) < tol:  # O's top edge matches R's bottom
                overlap_start = max(rx, ox)
                overlap_end = min(rx + rw, ox + ow)
                return max(overlap_end - overlap_start, 0)
        elif wall == "N":
            if abs(ry + rl - oy) < tol:
                overlap_start = max(rx, ox)
                overlap_end = min(rx + rw, ox + ow)
                return max(overlap_end - overlap_start, 0)
        elif wall == "W":
            if abs(ox + ow - rx) < tol:
                overlap_start = max(ry, oy)
                overlap_end = min(ry + rl, oy + ol)
                return max(overlap_end - overlap_start, 0)
        elif wall == "E":
            if abs(rx + rw - ox) < tol:
                overlap_start = max(ry, oy)
                overlap_end = min(ry + rl, oy + ol)
                return max(overlap_end - overlap_start, 0)
        return 0.0

    # ==================================================================
    # Step 5: DXF export
    # ==================================================================

    def _export_dxf(
        self,
        rooms: List[Dict],
        boundary: List[List[float]],
        output_path: str,
    ) -> str:
        """Generate a professional DXF drawing."""
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # --- Layers ---
        doc.layers.add("BOUNDARY", color=7)
        doc.layers.add("WALLS_EXT", color=0)
        doc.layers.add("WALLS_INT", color=8)
        doc.layers.add("DOORS", color=3)
        doc.layers.add("WINDOWS", color=5)
        doc.layers.add("LABELS", color=10)
        doc.layers.add("DIMENSIONS", color=6)
        doc.layers.add("HATCHING", color=252)

        # --- Boundary ---
        if boundary and len(boundary) >= 3:
            pts = [(p[0] * FT_TO_MM, p[1] * FT_TO_MM) for p in boundary]
            msp.add_lwpolyline(pts, dxfattribs={
                "layer": "BOUNDARY",
                "lineweight": 50,
            })

        # --- Rooms: walls, labels, dims ---
        for room in rooms:
            x = room["position"]["x"] * FT_TO_MM
            y = room["position"]["y"] * FT_TO_MM
            w = room["width"] * FT_TO_MM
            h = room["length"] * FT_TO_MM

            # External wall (thick) if room touches boundary
            is_ext_s = room["position"]["y"] < 1.0
            is_ext_n = abs(room["position"]["y"] + room["length"] - self.plot_l) < 1.0
            is_ext_w = room["position"]["x"] < 1.0
            is_ext_e = abs(room["position"]["x"] + room["width"] - self.plot_w) < 1.0

            # Double-line walls
            ext_off = WALL_EXT_MM / 2
            int_off = WALL_INT_MM / 2

            # Room outline (centre-line)
            msp.add_lwpolyline([
                (x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)
            ], dxfattribs={"layer": "WALLS_INT"})

            # External wall thickening (draw offset rectangle)
            sides = [
                ("S", is_ext_s, (x, y), (x + w, y), "H"),
                ("N", is_ext_n, (x, y + h), (x + w, y + h), "H"),
                ("W", is_ext_w, (x, y), (x, y + h), "V"),
                ("E", is_ext_e, (x + w, y), (x + w, y + h), "V"),
            ]
            for side_name, is_ext, p1, p2, orient in sides:
                off = ext_off if is_ext else int_off
                layer = "WALLS_EXT" if is_ext else "WALLS_INT"
                if orient == "H":
                    msp.add_line(
                        (p1[0], p1[1] - off), (p2[0], p2[1] - off),
                        dxfattribs={"layer": layer}
                    )
                    msp.add_line(
                        (p1[0], p1[1] + off), (p2[0], p2[1] + off),
                        dxfattribs={"layer": layer}
                    )
                else:
                    msp.add_line(
                        (p1[0] - off, p1[1]), (p2[0] - off, p2[1]),
                        dxfattribs={"layer": layer}
                    )
                    msp.add_line(
                        (p1[0] + off, p1[1]), (p2[0] + off, p2[1]),
                        dxfattribs={"layer": layer}
                    )

            # --- Room label ---
            cx = x + w / 2
            cy = y + h / 2
            area_sqft = room["area"]
            label_text = room["name"]

            msp.add_text(
                label_text,
                dxfattribs={
                    "layer": "LABELS",
                    "height": max(w, h) * 0.04,
                    "insert": (cx, cy + max(w, h) * 0.03),
                },
            ).set_placement(
                (cx, cy + max(w, h) * 0.03),
                align=TextEntityAlignment.MIDDLE_CENTER,
            )
            msp.add_text(
                f"{area_sqft:.0f} sq ft",
                dxfattribs={
                    "layer": "LABELS",
                    "height": max(w, h) * 0.025,
                    "insert": (cx, cy - max(w, h) * 0.03),
                },
            ).set_placement(
                (cx, cy - max(w, h) * 0.03),
                align=TextEntityAlignment.MIDDLE_CENTER,
            )

            # --- Dimension lines ---
            # Bottom dimension (width)
            dim_y = y - WALL_EXT_MM * 2
            msp.add_aligned_dim(
                p1=(x, dim_y), p2=(x + w, dim_y),
                distance=WALL_EXT_MM,
                dxfattribs={"layer": "DIMENSIONS"},
            ).render()

            # Left dimension (height)
            dim_x = x - WALL_EXT_MM * 2
            msp.add_aligned_dim(
                p1=(dim_x, y), p2=(dim_x, y + h),
                distance=WALL_EXT_MM,
                dxfattribs={"layer": "DIMENSIONS"},
            ).render()

            # --- Doors ---
            for door in room.get("doors", []):
                self._draw_door(msp, room, door)

            # --- Windows ---
            for win in room.get("windows", []):
                self._draw_window(msp, room, win)

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(output_path)
        logger.info("DXF exported → %s", output_path)
        return output_path

    def _draw_door(self, msp, room: Dict, door: Dict):
        """Draw a door symbol with swing arc."""
        x = room["position"]["x"] * FT_TO_MM
        y = room["position"]["y"] * FT_TO_MM
        w = room["width"] * FT_TO_MM
        h = room["length"] * FT_TO_MM
        dw = door["width"] * FT_TO_MM
        wall = door["wall"]

        if wall == "S":
            hx, hy = x + w * 0.3, y
            msp.add_line((hx, hy), (hx + dw, hy), dxfattribs={"layer": "DOORS"})
            msp.add_arc(
                center=(hx, hy), radius=dw,
                start_angle=0, end_angle=90,
                dxfattribs={"layer": "DOORS"},
            )
        elif wall == "N":
            hx, hy = x + w * 0.3, y + h
            msp.add_line((hx, hy), (hx + dw, hy), dxfattribs={"layer": "DOORS"})
            msp.add_arc(
                center=(hx, hy), radius=dw,
                start_angle=270, end_angle=360,
                dxfattribs={"layer": "DOORS"},
            )
        elif wall == "W":
            hx, hy = x, y + h * 0.3
            msp.add_line((hx, hy), (hx, hy + dw), dxfattribs={"layer": "DOORS"})
            msp.add_arc(
                center=(hx, hy), radius=dw,
                start_angle=0, end_angle=90,
                dxfattribs={"layer": "DOORS"},
            )
        elif wall == "E":
            hx, hy = x + w, y + h * 0.3
            msp.add_line((hx, hy), (hx, hy + dw), dxfattribs={"layer": "DOORS"})
            msp.add_arc(
                center=(hx, hy), radius=dw,
                start_angle=90, end_angle=180,
                dxfattribs={"layer": "DOORS"},
            )

    def _draw_window(self, msp, room: Dict, win: Dict):
        """Draw a window symbol (double line break in wall)."""
        x = room["position"]["x"] * FT_TO_MM
        y = room["position"]["y"] * FT_TO_MM
        w = room["width"] * FT_TO_MM
        h = room["length"] * FT_TO_MM
        ww = win["width"] * FT_TO_MM
        wall = win["wall"]
        off = WALL_EXT_MM / 2

        if wall == "S":
            wx = x + w * 0.4
            msp.add_line((wx, y - off), (wx, y + off), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx + ww, y - off), (wx + ww, y + off), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx, y), (wx + ww, y), dxfattribs={"layer": "WINDOWS"})
        elif wall == "N":
            wx = x + w * 0.4
            wy = y + h
            msp.add_line((wx, wy - off), (wx, wy + off), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx + ww, wy - off), (wx + ww, wy + off), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx, wy), (wx + ww, wy), dxfattribs={"layer": "WINDOWS"})
        elif wall == "W":
            wy = y + h * 0.4
            msp.add_line((x - off, wy), (x + off, wy), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((x - off, wy + ww), (x + off, wy + ww), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((x, wy), (x, wy + ww), dxfattribs={"layer": "WINDOWS"})
        elif wall == "E":
            wx = x + w
            wy = y + h * 0.4
            msp.add_line((wx - off, wy), (wx + off, wy), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx - off, wy + ww), (wx + off, wy + ww), dxfattribs={"layer": "WINDOWS"})
            msp.add_line((wx, wy), (wx, wy + ww), dxfattribs={"layer": "WINDOWS"})
