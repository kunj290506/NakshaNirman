"""
Professional 3D Architectural Model Generator.

Creates photorealistic-style 3D floor plan models with:
  - Proper wall geometry with door/window openings (boolean CSG)
  - Room-specific floor materials & colors
  - 3D furniture placement per room type
  - Glass window panes & door panels
  - Outdoor landscaping (grass, pathways)
  - Section-cut walls for interior visibility
  - PBR-compatible material assignments

Exports as glTF/GLB for Three.js rendering.

References:
  - Architectural visualization best practices
  - Standard furniture dimensions (Neufert's Architects' Data)
  - Building section/cutaway conventions
"""

import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box as shapely_box
from pathlib import Path
import logging
import math

logger = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================

WALL_HEIGHT = 10.0        # ft (~3m standard ceiling)
WALL_HEIGHT_SECTION = 7.0 # ft — section cut height for interior visibility
EXT_WALL_THICK = 0.75     # ft (9 inch brick)
INT_WALL_THICK = 0.375    # ft (4.5 inch brick)
FLOOR_THICK = 0.4         # ft (slab)
SILL_HEIGHT = 3.0         # ft — window sill height
WINDOW_HEIGHT = 4.0       # ft — window opening height
DOOR_HEIGHT = 7.0         # ft — door opening height

# Material colors (RGBA) — designed for professional arch-viz look
COLORS = {
    # Walls
    'ext_wall':       [235, 225, 215, 255],   # Warm off-white plaster
    'int_wall':       [245, 240, 235, 255],   # Light cream interior

    # Floors by room type
    'floor_living':   [180, 140, 100, 255],   # Warm wood / vitrified tile
    'floor_bedroom':  [190, 155, 120, 255],   # Light wood laminate
    'floor_kitchen':  [200, 195, 185, 255],   # Light ceramic tile
    'floor_bathroom': [170, 185, 195, 255],   # Blue-grey anti-skid tile
    'floor_dining':   [185, 150, 110, 255],   # Wood tone
    'floor_study':    [175, 145, 105, 255],   # Warm wood
    'floor_corridor': [210, 205, 195, 255],   # Neutral tile
    'floor_default':  [200, 195, 185, 255],   # Default neutral

    # Outdoor
    'grass':          [120, 165, 90, 255],    # Lawn green
    'pathway':        [190, 180, 165, 255],   # Concrete path
    'boundary_wall':  [180, 170, 155, 255],   # Compound wall

    # Furniture
    'wood_dark':      [120, 80, 50, 255],     # Dark wood furniture
    'wood_light':     [185, 150, 110, 255],   # Light wood
    'wood_medium':    [155, 115, 75, 255],    # Medium wood
    'fabric_sofa':    [160, 145, 130, 255],   # Beige sofa fabric
    'fabric_bed':     [220, 215, 205, 255],   # White bedsheet
    'fabric_pillow':  [200, 195, 185, 255],   # Pillow
    'ceramic_white':  [240, 240, 240, 255],   # Bathroom fixtures
    'counter_top':    [160, 155, 145, 255],   # Kitchen counter
    'glass':          [180, 210, 230, 120],   # Window glass (semi-transparent)

    # Roof
    'roof':           [165, 140, 115, 255],   # Terracotta/concrete roof
    'roof_edge':      [140, 120, 100, 255],   # Roof lip
}

# Standard furniture dimensions (L x W x H) in feet — Neufert reference
FURNITURE = {
    'living': [
        {'name': 'sofa_3seat', 'size': (7.0, 2.8, 2.5), 'color': 'fabric_sofa',
         'pos': 'back_center', 'offset': (0, 0.5)},
        {'name': 'coffee_table', 'size': (3.5, 2.0, 1.3), 'color': 'wood_dark',
         'pos': 'center', 'offset': (0, 0)},
        {'name': 'tv_unit', 'size': (5.0, 1.5, 2.0), 'color': 'wood_medium',
         'pos': 'front_center', 'offset': (0, 0.3)},
        {'name': 'side_table', 'size': (1.5, 1.5, 1.8), 'color': 'wood_dark',
         'pos': 'back_right', 'offset': (-0.3, 0.5)},
    ],
    'master_bedroom': [
        {'name': 'double_bed', 'size': (6.5, 5.5, 2.0), 'color': 'wood_medium',
         'pos': 'back_center', 'offset': (0, 0.3)},
        {'name': 'mattress', 'size': (6.0, 5.0, 0.8), 'color': 'fabric_bed',
         'pos': 'back_center', 'offset': (0, 0.3), 'z_base': 2.0},
        {'name': 'pillow_l', 'size': (1.8, 1.2, 0.5), 'color': 'fabric_pillow',
         'pos': 'back_center', 'offset': (-1.2, 0.5), 'z_base': 2.8},
        {'name': 'pillow_r', 'size': (1.8, 1.2, 0.5), 'color': 'fabric_pillow',
         'pos': 'back_center', 'offset': (1.2, 0.5), 'z_base': 2.8},
        {'name': 'nightstand_l', 'size': (1.5, 1.3, 2.0), 'color': 'wood_dark',
         'pos': 'back_left', 'offset': (0.3, 0.3)},
        {'name': 'nightstand_r', 'size': (1.5, 1.3, 2.0), 'color': 'wood_dark',
         'pos': 'back_right', 'offset': (-0.3, 0.3)},
        {'name': 'wardrobe', 'size': (6.0, 2.0, 6.5), 'color': 'wood_medium',
         'pos': 'right_center', 'offset': (-0.2, 0)},
    ],
    'bedroom': [
        {'name': 'single_bed', 'size': (6.5, 3.5, 2.0), 'color': 'wood_light',
         'pos': 'back_left', 'offset': (0.5, 0.3)},
        {'name': 'mattress', 'size': (6.0, 3.0, 0.8), 'color': 'fabric_bed',
         'pos': 'back_left', 'offset': (0.5, 0.3), 'z_base': 2.0},
        {'name': 'pillow', 'size': (1.8, 1.2, 0.5), 'color': 'fabric_pillow',
         'pos': 'back_left', 'offset': (0.5, 0.5), 'z_base': 2.8},
        {'name': 'study_desk', 'size': (3.5, 1.8, 2.5), 'color': 'wood_medium',
         'pos': 'front_right', 'offset': (-0.3, 0.3)},
        {'name': 'wardrobe', 'size': (4.0, 1.8, 6.5), 'color': 'wood_medium',
         'pos': 'right_center', 'offset': (-0.2, 0)},
    ],
    'kitchen': [
        {'name': 'counter_L', 'size': (0, 2.0, 3.0), 'color': 'counter_top',
         'pos': 'L_counter', 'offset': (0, 0)},
        {'name': 'upper_cabinet', 'size': (0, 1.2, 2.5), 'color': 'wood_medium',
         'pos': 'L_upper', 'offset': (0, 0), 'z_base': 4.5},
    ],
    'dining': [
        {'name': 'dining_table', 'size': (5.0, 3.0, 2.5), 'color': 'wood_dark',
         'pos': 'center', 'offset': (0, 0)},
        {'name': 'chair_1', 'size': (1.5, 1.5, 3.0), 'color': 'wood_medium',
         'pos': 'center', 'offset': (-2.0, -1.2)},
        {'name': 'chair_2', 'size': (1.5, 1.5, 3.0), 'color': 'wood_medium',
         'pos': 'center', 'offset': (2.0, -1.2)},
        {'name': 'chair_3', 'size': (1.5, 1.5, 3.0), 'color': 'wood_medium',
         'pos': 'center', 'offset': (-2.0, 1.2)},
        {'name': 'chair_4', 'size': (1.5, 1.5, 3.0), 'color': 'wood_medium',
         'pos': 'center', 'offset': (2.0, 1.2)},
    ],
    'bathroom': [
        {'name': 'wc', 'size': (1.8, 1.5, 1.5), 'color': 'ceramic_white',
         'pos': 'back_right', 'offset': (-0.3, 0.3)},
        {'name': 'basin', 'size': (1.8, 1.2, 2.8), 'color': 'ceramic_white',
         'pos': 'front_right', 'offset': (-0.3, 0.3)},
        {'name': 'shower_tray', 'size': (3.0, 3.0, 0.3), 'color': 'ceramic_white',
         'pos': 'back_left', 'offset': (0.3, 0.3)},
    ],
    'toilet': [
        {'name': 'wc', 'size': (1.8, 1.5, 1.5), 'color': 'ceramic_white',
         'pos': 'back_center', 'offset': (0, 0.3)},
        {'name': 'basin', 'size': (1.5, 1.0, 2.8), 'color': 'ceramic_white',
         'pos': 'front_center', 'offset': (0, 0.3)},
    ],
    'study': [
        {'name': 'desk', 'size': (5.0, 2.0, 2.5), 'color': 'wood_dark',
         'pos': 'back_center', 'offset': (0, 0.3)},
        {'name': 'chair', 'size': (2.0, 2.0, 3.5), 'color': 'fabric_sofa',
         'pos': 'back_center', 'offset': (0, 2.5)},
        {'name': 'bookshelf', 'size': (4.0, 1.0, 6.0), 'color': 'wood_medium',
         'pos': 'right_center', 'offset': (-0.2, 0)},
    ],
    'pooja': [
        {'name': 'temple', 'size': (2.5, 1.5, 4.0), 'color': 'wood_dark',
         'pos': 'back_center', 'offset': (0, 0.2)},
    ],
}


# ===========================================================================
# GEOMETRY HELPERS
# ===========================================================================

def _safe_polygon(coords):
    """Create a valid Shapely polygon from coords."""
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if poly.is_empty or poly.area < 0.01:
            return None
        return poly
    except Exception:
        return None


def _make_box(x, y, z, w, d, h):
    """Create a 3D box mesh at position (x,y,z) with size (w,d,h)."""
    if w < 0.01 or d < 0.01 or h < 0.01:
        return trimesh.Trimesh()
    mesh = trimesh.creation.box(extents=[w, d, h])
    mesh.apply_translation([x + w / 2, y + d / 2, z + h / 2])
    return mesh


def _color_mesh(mesh, color_key):
    """Apply a solid color to a mesh."""
    if mesh is None or not hasattr(mesh, 'vertices') or mesh.vertices.shape[0] == 0:
        return mesh
    c = COLORS.get(color_key, [200, 200, 200, 255])
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=c)
    return mesh


def _extrude_poly(poly, height, z_base=0.0, color_key=None):
    """Extrude a Shapely polygon to 3D and optionally color it."""
    try:
        if poly is None or poly.is_empty or poly.area < 0.01:
            return None
        mesh = trimesh.creation.extrude_polygon(poly, height)
        if z_base != 0.0:
            mesh.apply_translation([0, 0, z_base])
        if color_key:
            _color_mesh(mesh, color_key)
        return mesh
    except Exception:
        return None


def _is_valid_mesh(mesh):
    """Check if a mesh has valid geometry."""
    return (mesh is not None and hasattr(mesh, 'vertices')
            and mesh.vertices.shape[0] > 0)


# ===========================================================================
# WALL GENERATION WITH DOOR/WINDOW OPENINGS
# ===========================================================================

def _create_wall_segment(x1, y1, x2, y2, thickness, height,
                         openings=None, section_cut=False):
    """
    Create a single wall segment between two points with door/window openings.

    openings: list of dicts with:
      - 'type': 'door' or 'window'
      - 'offset': distance along wall from start (in feet)
      - 'width': opening width
      - 'height': opening height
      - 'sill': sill height (0 for doors, 3ft for windows)
    """
    dx = x2 - x1
    dy = y2 - y1
    wall_len = math.sqrt(dx * dx + dy * dy)
    if wall_len < 0.1:
        return None

    h = WALL_HEIGHT_SECTION if section_cut else height

    meshes = []

    if not openings:
        # Solid wall — no openings
        wall = trimesh.creation.box(extents=[wall_len, thickness, h])
        wall.apply_translation([wall_len / 2, 0, h / 2])
        meshes.append(wall)
    else:
        # Sort openings by offset
        openings = sorted(openings, key=lambda o: o.get('offset', 0))

        # Deduplicate overlapping openings
        filtered = []
        for op in openings:
            ofs = op.get('offset', 0)
            w = op.get('width', 3.0)
            if not filtered:
                filtered.append(op)
            else:
                last = filtered[-1]
                last_end = last.get('offset', 0) + last.get('width', 3.0)
                if ofs < last_end:
                    # Overlap — keep the larger one
                    if w > last.get('width', 3.0):
                        filtered[-1] = op
                else:
                    filtered.append(op)
        openings = filtered

        cursor = 0
        for opening in openings:
            o_offset = max(0, min(opening.get('offset', 0), wall_len - opening.get('width', 3.0)))
            o_width = opening.get('width', 3.0)
            o_height = min(opening.get('height', 7.0), h - 0.3)
            o_sill = opening.get('sill', 0)

            # Clamp
            if o_offset + o_width > wall_len:
                o_width = wall_len - o_offset
            if o_width < 0.5:
                continue

            # Wall segment before opening
            seg_len = o_offset - cursor
            if seg_len > 0.1:
                seg = trimesh.creation.box(extents=[seg_len, thickness, h])
                seg.apply_translation([cursor + seg_len / 2, 0, h / 2])
                meshes.append(seg)

            # Wall above opening (lintel)
            lintel_h = h - (o_sill + o_height)
            if lintel_h > 0.1:
                lintel = trimesh.creation.box(extents=[o_width, thickness, lintel_h])
                lintel.apply_translation([
                    o_offset + o_width / 2, 0,
                    o_sill + o_height + lintel_h / 2
                ])
                meshes.append(lintel)

            # Wall below opening (sill wall) — for windows
            if o_sill > 0.1:
                sill_wall = trimesh.creation.box(extents=[o_width, thickness, o_sill])
                sill_wall.apply_translation([
                    o_offset + o_width / 2, 0, o_sill / 2
                ])
                meshes.append(sill_wall)

            cursor = o_offset + o_width

        # Wall segment after last opening
        remaining = wall_len - cursor
        if remaining > 0.1:
            seg = trimesh.creation.box(extents=[remaining, thickness, h])
            seg.apply_translation([cursor + remaining / 2, 0, h / 2])
            meshes.append(seg)

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    # Rotate to actual wall orientation and translate
    angle = math.atan2(dy, dx)
    rot = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    combined.apply_transform(rot)
    combined.apply_translation([x1, y1, 0])

    return combined


def _point_offset_on_segment(px, py, x1, y1, x2, y2, tolerance=1.5):
    """
    Project point (px, py) onto line segment.
    Returns offset distance along segment, or None if too far.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.sqrt(dx * dx + dy * dy)
    if seg_len < 0.1:
        return None

    t = ((px - x1) * dx + (py - y1) * dy) / (seg_len * seg_len)
    if t < -0.1 or t > 1.1:
        return None

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    if dist > tolerance:
        return None

    return t * seg_len


def _build_walls_with_openings(rooms, boundary, doors_list):
    """
    Build all walls (external + internal) with door/window openings.
    Returns list of colored trimesh objects.
    """
    meshes = []
    boundary_poly = _safe_polygon(boundary)
    if boundary_poly is None:
        return meshes

    bx0, by0, bx1, by1 = boundary_poly.bounds

    # ── External walls ───────────────────────────────────────────
    boundary_coords = list(boundary_poly.exterior.coords)
    for i in range(len(boundary_coords) - 1):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[i + 1]

        wall_openings = []

        # Collect window openings from rooms on this wall
        for room in rooms:
            rx = room.get('position', {}).get('x', 0)
            ry = room.get('position', {}).get('y', 0)
            rw = room.get('width', 0)
            rl = room.get('length', 0)

            for win in room.get('windows', []):
                w_wall = win.get('wall', '')
                w_width = win.get('width', 4.0)
                w_type = win.get('type', 'standard')
                sill = SILL_HEIGHT if w_type != 'ventilation' else 5.0
                w_height = WINDOW_HEIGHT if w_type != 'ventilation' else 2.0

                # Get window center position on boundary
                wcx, wcy = _window_center_on_boundary(
                    w_wall, rx, ry, rw, rl, bx0, by0, bx1, by1)
                if wcx is None:
                    continue

                offset = _point_offset_on_segment(
                    wcx, wcy, x1, y1, x2, y2, tolerance=1.5)
                if offset is not None:
                    wall_openings.append({
                        'type': 'window',
                        'offset': max(0.5, offset - w_width / 2),
                        'width': w_width, 'height': w_height, 'sill': sill
                    })

        # Collect door openings from main doors list
        for door_info in doors_list:
            dpos = door_info.get('position', [0, 0])
            d_width = door_info.get('width', 3.0)
            offset = _point_offset_on_segment(
                dpos[0], dpos[1], x1, y1, x2, y2, tolerance=1.5)
            if offset is not None:
                wall_openings.append({
                    'type': 'door',
                    'offset': max(0.3, offset - d_width / 2),
                    'width': d_width, 'height': DOOR_HEIGHT, 'sill': 0
                })

        wall = _create_wall_segment(
            x1, y1, x2, y2, EXT_WALL_THICK, WALL_HEIGHT,
            openings=wall_openings if wall_openings else None
        )
        if _is_valid_mesh(wall):
            _color_mesh(wall, 'ext_wall')
            meshes.append(wall)

    # ── Internal walls (section-cut for visibility) ──────────────
    processed = set()
    for i, r1 in enumerate(rooms):
        p1 = r1.get('position', {})
        x1, y1 = p1.get('x', 0), p1.get('y', 0)
        w1, l1 = r1.get('width', 0), r1.get('length', 0)

        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            key = (i, j)
            if key in processed:
                continue

            p2 = r2.get('position', {})
            x2, y2 = p2.get('x', 0), p2.get('y', 0)
            w2, l2 = r2.get('width', 0), r2.get('length', 0)

            tol = 1.0
            shared = None

            # Detect shared wall between r1 and r2
            if abs((x1 + w1) - x2) < tol:
                os_ = max(y1, y2)
                oe = min(y1 + l1, y2 + l2)
                if oe - os_ > 1:
                    shared = (x1 + w1, os_, x1 + w1, oe)
            elif abs(x1 - (x2 + w2)) < tol:
                os_ = max(y1, y2)
                oe = min(y1 + l1, y2 + l2)
                if oe - os_ > 1:
                    shared = (x1, os_, x1, oe)
            elif abs((y1 + l1) - y2) < tol:
                os_ = max(x1, x2)
                oe = min(x1 + w1, x2 + w2)
                if oe - os_ > 1:
                    shared = (os_, y1 + l1, oe, y1 + l1)
            elif abs(y1 - (y2 + l2)) < tol:
                os_ = max(x1, x2)
                oe = min(x1 + w1, x2 + w2)
                if oe - os_ > 1:
                    shared = (os_, y1, oe, y1)

            if shared:
                processed.add(key)
                sx1, sy1, sx2, sy2 = shared

                # Collect door openings on this internal wall
                int_openings = []
                for room in [r1, r2]:
                    for door in room.get('doors', []):
                        d_wall = door.get('wall', '')
                        d_width = door.get('width', 3.0)
                        r_pos = room.get('position', {})
                        rrx, rry = r_pos.get('x', 0), r_pos.get('y', 0)

                        if d_wall in ('E', 'W'):
                            dy = rry + 0.5
                            offset = _point_offset_on_segment(
                                sx1, dy, sx1, sy1, sx2, sy2, tolerance=2.0)
                        elif d_wall in ('N', 'S'):
                            dx_ = rrx + 0.5
                            offset = _point_offset_on_segment(
                                dx_, sy1, sx1, sy1, sx2, sy2, tolerance=2.0)
                        else:
                            offset = None

                        if offset is not None:
                            int_openings.append({
                                'type': 'door',
                                'offset': max(0.3, offset),
                                'width': d_width,
                                'height': DOOR_HEIGHT, 'sill': 0
                            })

                wall = _create_wall_segment(
                    sx1, sy1, sx2, sy2, INT_WALL_THICK,
                    WALL_HEIGHT_SECTION,
                    openings=int_openings if int_openings else None,
                    section_cut=True
                )
                if _is_valid_mesh(wall):
                    _color_mesh(wall, 'int_wall')
                    meshes.append(wall)

    return meshes


def _window_center_on_boundary(wall_dir, rx, ry, rw, rl, bx0, by0, bx1, by1):
    """Get the (x,y) center position of a window on the boundary."""
    tol = 1.5
    if wall_dir == 'S' and abs(ry - by0) < tol:
        return (rx + rw / 2, by0)
    elif wall_dir == 'N' and abs(ry + rl - by1) < tol:
        return (rx + rw / 2, by1)
    elif wall_dir == 'W' and abs(rx - bx0) < tol:
        return (bx0, ry + rl / 2)
    elif wall_dir == 'E' and abs(rx + rw - bx1) < tol:
        return (bx1, ry + rl / 2)
    return (None, None)


# ===========================================================================
# FLOOR GENERATION
# ===========================================================================

def _create_room_floors(rooms):
    """Create individual floor slabs per room with appropriate colors."""
    meshes = []
    floor_color_map = {
        'living': 'floor_living',
        'master_bedroom': 'floor_bedroom',
        'bedroom': 'floor_bedroom',
        'kitchen': 'floor_kitchen',
        'bathroom': 'floor_bathroom',
        'toilet': 'floor_bathroom',
        'dining': 'floor_dining',
        'study': 'floor_study',
        'corridor': 'floor_corridor',
        'foyer': 'floor_corridor',
        'pooja': 'floor_default',
        'store': 'floor_default',
        'utility': 'floor_default',
        'balcony': 'floor_corridor',
    }

    for room in rooms:
        pos = room.get('position', {})
        rx = pos.get('x', 0)
        ry = pos.get('y', 0)
        rw = room.get('width', 0)
        rl = room.get('length', 0)
        rtype = room.get('room_type', 'default')

        if rw < 1 or rl < 1:
            continue

        floor = _make_box(rx, ry, 0.01, rw, rl, 0.1)
        color_key = floor_color_map.get(rtype, 'floor_default')
        _color_mesh(floor, color_key)
        meshes.append(floor)

    return meshes


# ===========================================================================
# FURNITURE GENERATION
# ===========================================================================

def _place_furniture_in_room(room):
    """
    Place 3D furniture meshes inside a room based on room type.
    Uses relative positioning with auto-scaling for small rooms.
    """
    meshes = []
    rtype = room.get('room_type', '')
    pos = room.get('position', {})
    rx = pos.get('x', 0)
    ry = pos.get('y', 0)
    rw = room.get('width', 0)
    rl = room.get('length', 0)

    furniture_list = FURNITURE.get(rtype, [])
    if not furniture_list:
        return meshes

    pad = 0.3

    for item in furniture_list:
        fw, fd, fh = item['size']
        color_key = item.get('color', 'wood_medium')
        placement = item.get('pos', 'center')
        ox, oy = item.get('offset', (0, 0))
        z_base = item.get('z_base', 0.15)

        # Kitchen L-counter
        if placement == 'L_counter':
            counter_d = fd
            cw = rw - 2 * pad
            if cw > 1:
                c1 = _make_box(rx + pad, ry + rl - counter_d - pad, z_base,
                               cw, counter_d, fh)
                _color_mesh(c1, color_key)
                meshes.append(c1)
            cl = rl * 0.6
            if cl > 1:
                c2 = _make_box(rx + pad, ry + rl - cl - pad, z_base,
                               counter_d, cl, fh)
                _color_mesh(c2, color_key)
                meshes.append(c2)
            continue

        if placement == 'L_upper':
            cw = rw - 2 * pad
            if cw > 1:
                c1 = _make_box(rx + pad, ry + rl - fd - pad, z_base,
                               cw, fd, fh)
                _color_mesh(c1, color_key)
                meshes.append(c1)
            continue

        # Auto-scale if too large for room
        if fw > rw - 2 * pad or fd > rl - 2 * pad:
            scale = min((rw - 2 * pad) / max(fw, 0.1),
                        (rl - 2 * pad) / max(fd, 0.1), 1.0)
            if scale < 0.4:
                continue
            fw *= scale
            fd *= scale
            fh *= min(scale, 1.0)

        # Calculate position
        if placement == 'center':
            fx = rx + (rw - fw) / 2 + ox
            fy = ry + (rl - fd) / 2 + oy
        elif placement == 'back_center':
            fx = rx + (rw - fw) / 2 + ox
            fy = ry + rl - fd - pad + oy
        elif placement == 'front_center':
            fx = rx + (rw - fw) / 2 + ox
            fy = ry + pad + oy
        elif placement == 'back_left':
            fx = rx + pad + ox
            fy = ry + rl - fd - pad + oy
        elif placement == 'back_right':
            fx = rx + rw - fw - pad + ox
            fy = ry + rl - fd - pad + oy
        elif placement == 'front_left':
            fx = rx + pad + ox
            fy = ry + pad + oy
        elif placement == 'front_right':
            fx = rx + rw - fw - pad + ox
            fy = ry + pad + oy
        elif placement == 'left_center':
            fx = rx + pad + ox
            fy = ry + (rl - fd) / 2 + oy
        elif placement == 'right_center':
            fx = rx + rw - fw - pad + ox
            fy = ry + (rl - fd) / 2 + oy
        else:
            fx = rx + (rw - fw) / 2 + ox
            fy = ry + (rl - fd) / 2 + oy

        # Clamp inside room
        fx = max(rx + pad, min(fx, rx + rw - fw - pad))
        fy = max(ry + pad, min(fy, ry + rl - fd - pad))

        furn = _make_box(fx, fy, z_base, fw, fd, fh)
        if _is_valid_mesh(furn):
            _color_mesh(furn, color_key)
            meshes.append(furn)

    return meshes


# ===========================================================================
# GLASS WINDOWS
# ===========================================================================

def _create_window_glass(rooms, bx0, by0, bx1, by1):
    """Create semi-transparent glass panes for all windows."""
    meshes = []

    for room in rooms:
        pos = room.get('position', {})
        rx = pos.get('x', 0)
        ry = pos.get('y', 0)
        rw = room.get('width', 0)
        rl = room.get('length', 0)

        for win in room.get('windows', []):
            w_wall = win.get('wall', '')
            w_width = win.get('width', 4.0)
            w_type = win.get('type', 'standard')
            sill = SILL_HEIGHT if w_type != 'ventilation' else 5.0
            win_h = WINDOW_HEIGHT if w_type != 'ventilation' else 2.0

            gt = 0.08  # Glass thickness

            glass = None
            if w_wall == 'S' and abs(ry - by0) < 1.5:
                gx = rx + (rw - w_width) / 2
                glass = _make_box(gx, by0 - gt / 2, sill, w_width, gt, win_h)

            elif w_wall == 'N' and abs(ry + rl - by1) < 1.5:
                gx = rx + (rw - w_width) / 2
                glass = _make_box(gx, by1 - gt / 2, sill, w_width, gt, win_h)

            elif w_wall == 'W' and abs(rx - bx0) < 1.5:
                gy = ry + (rl - w_width) / 2
                glass = _make_box(bx0 - gt / 2, gy, sill, gt, w_width, win_h)

            elif w_wall == 'E' and abs(rx + rw - bx1) < 1.5:
                gy = ry + (rl - w_width) / 2
                glass = _make_box(bx1 - gt / 2, gy, sill, gt, w_width, win_h)

            if glass and _is_valid_mesh(glass):
                _color_mesh(glass, 'glass')
                meshes.append(glass)

    return meshes


# ===========================================================================
# DOOR PANELS
# ===========================================================================

def _create_door_panels(doors_list):
    """Create door panel meshes shown slightly open for visual effect."""
    meshes = []
    for door_info in doors_list:
        dpos = door_info.get('position', [0, 0])
        d_width = door_info.get('width', 3.0)
        swing = door_info.get('swing_dir', [0, 1])
        dx, dy = dpos[0], dpos[1]

        panel_thick = 0.12
        angle_rad = math.radians(25)  # 25 degree open
        pw = d_width * math.cos(angle_rad)
        pd = d_width * math.sin(angle_rad) + panel_thick

        if abs(swing[0]) > abs(swing[1]):
            if swing[0] > 0:
                panel = _make_box(dx, dy, 0.1, pw, pd, DOOR_HEIGHT - 0.3)
            else:
                panel = _make_box(dx - pw, dy, 0.1, pw, pd, DOOR_HEIGHT - 0.3)
        else:
            if swing[1] > 0:
                panel = _make_box(dx, dy, 0.1, pw, pd, DOOR_HEIGHT - 0.3)
            else:
                panel = _make_box(dx, dy - pd, 0.1, pw, pd, DOOR_HEIGHT - 0.3)

        if _is_valid_mesh(panel):
            _color_mesh(panel, 'wood_light')
            meshes.append(panel)

    return meshes


# ===========================================================================
# OUTDOOR / LANDSCAPING
# ===========================================================================

def _create_outdoor(boundary, plot_margin=8.0):
    """
    Create outdoor elements: grass, pathway, boundary wall.
    Matches the reference image style with green lawn and compound wall.
    """
    meshes = []
    boundary_poly = _safe_polygon(boundary)
    if boundary_poly is None:
        return meshes

    bx0, by0, bx1, by1 = boundary_poly.bounds
    plot_w = bx1 - bx0
    plot_h = by1 - by0

    gx0 = bx0 - plot_margin
    gy0 = by0 - plot_margin
    gw = plot_w + 2 * plot_margin
    gh = plot_h + 2 * plot_margin

    # Grass ground plane
    ground = _make_box(gx0, gy0, -0.6, gw, gh, 0.5)
    _color_mesh(ground, 'grass')
    meshes.append(ground)

    # Concrete pathway from front
    path_w = min(5.0, plot_w * 0.15)
    path_x = bx0 + plot_w / 2 - path_w / 2
    pathway = _make_box(path_x, gy0, -0.08, path_w, plot_margin + 1, 0.15)
    _color_mesh(pathway, 'pathway')
    meshes.append(pathway)

    # Side patio/pathway along building
    side_path = _make_box(bx0, by0 - 2, -0.08, plot_w, 2, 0.12)
    _color_mesh(side_path, 'pathway')
    meshes.append(side_path)

    # Compound boundary wall (low wall around property)
    bw_h = 3.5
    bw_t = 0.5

    # South
    sw = _make_box(gx0 + 1, gy0 + 1, 0, gw - 2, bw_t, bw_h)
    _color_mesh(sw, 'boundary_wall')
    meshes.append(sw)
    # North
    nw = _make_box(gx0 + 1, gy0 + gh - 1 - bw_t, 0, gw - 2, bw_t, bw_h)
    _color_mesh(nw, 'boundary_wall')
    meshes.append(nw)
    # West
    ww = _make_box(gx0 + 1, gy0 + 1, 0, bw_t, gh - 2, bw_h)
    _color_mesh(ww, 'boundary_wall')
    meshes.append(ww)
    # East
    ew_m = _make_box(gx0 + gw - 1 - bw_t, gy0 + 1, 0, bw_t, gh - 2, bw_h)
    _color_mesh(ew_m, 'boundary_wall')
    meshes.append(ew_m)

    return meshes


# ===========================================================================
# ROOF GENERATION
# ===========================================================================

def _create_roof(boundary):
    """Create flat slab roof with overhang and parapet (Indian style)."""
    meshes = []
    boundary_poly = _safe_polygon(boundary)
    if boundary_poly is None:
        return meshes

    # Roof slab with slight overhang
    overhang = 0.8
    roof_poly = boundary_poly.buffer(overhang)
    roof_mesh = _extrude_poly(roof_poly, 0.5, z_base=WALL_HEIGHT)
    if _is_valid_mesh(roof_mesh):
        _color_mesh(roof_mesh, 'roof')
        meshes.append(roof_mesh)

    # Parapet wall around roof edge
    try:
        inner = roof_poly.buffer(-0.01)
        parapet_ring = roof_poly.buffer(0.3).difference(inner)
        parapet = _extrude_poly(parapet_ring, 3.0, z_base=WALL_HEIGHT + 0.5)
        if _is_valid_mesh(parapet):
            _color_mesh(parapet, 'roof_edge')
            meshes.append(parapet)
    except Exception:
        pass

    return meshes


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def generate_3d_model(plan: dict, output_path: str) -> str:
    """
    Generate a professional 3D architectural model (glTF/GLB) from floor plan.

    Architecture:
      1. Outdoor landscaping (grass, pathways, compound wall)
      2. Room-specific floor slabs with material colors
      3. External walls with window/door openings
      4. Internal walls (section-cut height for visibility)
      5. Glass window panes (semi-transparent)
      6. Door panels (shown partially open)
      7. Furniture per room type (Neufert dimensions)
      8. Roof with overhang and parapet

    Args:
        plan: Floor plan dict from GNN/layout engine.
        output_path: Path to save .glb file.

    Returns:
        Path to the generated file.
    """
    all_meshes = []

    boundary = plan.get("boundary", [])
    rooms = plan.get("rooms", [])
    doors_list = plan.get("doors", [])

    if not boundary or len(boundary) < 3:
        raise ValueError("Invalid boundary coordinates for 3D generation.")

    boundary_poly = _safe_polygon(boundary)
    if boundary_poly is None:
        raise ValueError("Could not create valid boundary polygon.")

    bx0, by0, bx1, by1 = boundary_poly.bounds

    logger.info(f"Generating 3D model: {len(rooms)} rooms, "
                f"boundary {bx1 - bx0:.1f}x{by1 - by0:.1f}ft")

    # 1. Outdoor
    try:
        outdoor = _create_outdoor(boundary)
        all_meshes.extend(outdoor)
        logger.info(f"  Outdoor: {len(outdoor)} meshes")
    except Exception as e:
        logger.warning(f"  Outdoor failed: {e}")

    # 2. Room floors
    try:
        floors = _create_room_floors(rooms)
        all_meshes.extend(floors)
        logger.info(f"  Floors: {len(floors)} meshes")
    except Exception as e:
        logger.warning(f"  Floors failed: {e}")

    # 3. Walls with openings
    try:
        walls = _build_walls_with_openings(rooms, boundary, doors_list)
        all_meshes.extend(walls)
        logger.info(f"  Walls: {len(walls)} meshes")
    except Exception as e:
        logger.warning(f"  Walls failed: {e}")
        # Fallback: simple wall ring extrusion
        try:
            wall_ring = boundary_poly.buffer(EXT_WALL_THICK / 2).difference(
                boundary_poly.buffer(-EXT_WALL_THICK / 2))
            fallback = _extrude_poly(wall_ring, WALL_HEIGHT, color_key='ext_wall')
            if _is_valid_mesh(fallback):
                all_meshes.append(fallback)
        except Exception:
            pass

    # 4. Glass windows
    try:
        glass = _create_window_glass(rooms, bx0, by0, bx1, by1)
        all_meshes.extend(glass)
        logger.info(f"  Windows: {len(glass)} panes")
    except Exception as e:
        logger.warning(f"  Glass failed: {e}")

    # 5. Door panels
    try:
        doors = _create_door_panels(doors_list)
        all_meshes.extend(doors)
        logger.info(f"  Doors: {len(doors)} panels")
    except Exception as e:
        logger.warning(f"  Door panels failed: {e}")

    # 6. Furniture
    try:
        furn_count = 0
        for room in rooms:
            furn = _place_furniture_in_room(room)
            all_meshes.extend(furn)
            furn_count += len(furn)
        logger.info(f"  Furniture: {furn_count} pieces")
    except Exception as e:
        logger.warning(f"  Furniture failed: {e}")

    # 7. Roof
    try:
        roof = _create_roof(boundary)
        all_meshes.extend(roof)
        logger.info(f"  Roof: {len(roof)} meshes")
    except Exception as e:
        logger.warning(f"  Roof failed: {e}")

    # ── Filter & export ──────────────────────────────────────────
    valid = [m for m in all_meshes if _is_valid_mesh(m)]

    if not valid:
        raise ValueError("No valid geometry generated for 3D model.")

    logger.info(f"  Total valid meshes: {len(valid)}")

    scene = trimesh.Scene(valid)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith((".glb", ".gltf")):
        scene.export(output_path, file_type="glb")
    elif output_path.endswith(".obj"):
        scene.export(output_path, file_type="obj")
    else:
        output_path = output_path + ".glb"
        scene.export(output_path, file_type="glb")

    logger.info(f"3D model exported: {output_path}")
    return output_path
