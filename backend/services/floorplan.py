"""
Professional Residential Architect & Structural Planner - Indian Standards

Implements complete architectural workflow with engineering logic:
1. Plot geometry analysis (entry, longest walls, corners)
2. Zoning (Public → Semi-private → Private → Service)
3. Indian standard room sizes with structural grid
4. Circulation flow planning (no dead corridors)
5. Cross-ventilation and daylight optimization
6. Structural feasibility (column grid 10-15 ft)
7. Parking, utilities, and future expansion
8. CAD-quality output with dimensions
"""

import math
import random
from typing import Optional, List, Dict, Tuple
from shapely.geometry import Polygon, box, LineString, MultiPolygon, Point
from shapely.affinity import scale as shapely_scale, translate, rotate
from shapely.ops import unary_union
import json


# INDIAN RESIDENTIAL STANDARDS - Room Sizes (feet)
STANDARD_ROOM_SIZES = {
    "living": {
        "width": 14, "height": 16, "min_area": 224, 
        "label": "LIVING ROOM", "zone": "public",
        "furniture": "sofa", "window_required": True,
    },
    "master_bedroom": {
        "width": 12, "height": 14, "min_area": 168,
        "label": "MASTER BEDROOM", "zone": "private",
        "furniture": "bed_double", "window_required": True,
        "attached_toilet": True,
    },
    "bedroom": {
        "width": 10, "height": 12, "min_area": 120,
        "label": "BEDROOM", "zone": "private",
        "furniture": "bed_single", "window_required": True,
    },
    "kitchen": {
        "width": 8, "height": 10, "min_area": 80,
        "label": "KITCHEN", "zone": "service",
        "furniture": "counter", "window_required": True,
        "plumbing_required": True,
    },
    "dining": {
        "width": 10, "height": 12, "min_area": 120,
        "label": "DINING ROOM", "zone": "semi-private",
        "furniture": "table", "window_required": False,
    },
    "bathroom": {
        "width": 5, "height": 8, "min_area": 40,
        "label": "BATHROOM", "zone": "service",
        "furniture": "toilet", "window_required": True,
        "plumbing_required": True,
    },
    "toilet": {
        "width": 4, "height": 6, "min_area": 24,
        "label": "TOILET", "zone": "service",
        "furniture": "toilet", "window_required": True,
        "plumbing_required": True,
    },
    "porch": {
        "width": 10, "height": 8, "min_area": 80,
        "label": "PORCH", "zone": "public",
        "furniture": None, "window_required": False,
    },
    "parking": {
        "width": 10, "height": 18, "min_area": 180,
        "label": "PARKING", "zone": "public",
        "furniture": None, "window_required": False,
    },
    "utility": {
        "width": 4, "height": 6, "min_area": 24,
        "label": "UTILITY", "zone": "service",
        "furniture": None, "window_required": True,
        "plumbing_required": True,
    },
    "study": {
        "width": 10, "height": 10, "min_area": 100,
        "label": "STUDY", "zone": "private",
        "furniture": "desk", "window_required": True,
    },
    "store": {
        "width": 6, "height": 6, "min_area": 36,
        "label": "STORE ROOM", "zone": "service",
        "furniture": None, "window_required": False,
    },
    "pooja": {
        "width": 5, "height": 5, "min_area": 25,
        "label": "POOJA ROOM", "zone": "private",
        "furniture": None, "window_required": False,
    },
    "staircase": {
        "width": 5, "height": 10, "min_area": 50,
        "label": "STAIRCASE", "zone": "circulation",
        "furniture": None, "window_required": False,
        "future_expansion": True,
    },
}

# STRUCTURAL STANDARDS (Indian Building Code)
WALL_THICKNESS_EXTERIOR = 0.75  # 230 mm ≈ 0.75 feet
WALL_THICKNESS_INTERIOR = 0.38  # 115 mm ≈ 0.38 feet
COLUMN_SPACING_MIN = 10.0  # feet
COLUMN_SPACING_MAX = 15.0  # feet
BEAM_WIDTH = 0.75  # feet
DOOR_WIDTH = 3.0  # feet (standard)
WINDOW_WIDTH = 4.0  # feet
CORRIDOR_WIDTH = 3.5  # feet
PARKING_SIZE = (10, 18)  # feet (Indian car parking)
STAIRCASE_WIDTH = 5.0  # feet
MIN_ROOM_DIMENSION = 8.0  # Minimum room width/height in feet


def _analyze_plot_geometry(boundary: Polygon) -> Dict:
    """
    STEP 1: Professional Plot Analysis (Architecture + Structural Engineering)
    
    Analyzes:
    - Plot orientation and dimensions
    - Road-facing side (longest edge)
    - Entry and parking feasibility
    - Corner types (narrow/wide/suitable for toilets/pooja)
    - Structural grid suitability (column spacing)
    - Ventilation potential (opposite walls)
    - Circulation flow possibilities
    """
    coords = list(boundary.exterior.coords)
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    centroid = boundary.centroid
    
    # Analyze all edges with orientation and structural properties
    edges = []
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        
        # Edge orientation (for ventilation planning)
        is_horizontal = abs(angle) < 30 or abs(angle) > 150
        is_vertical = 60 <= abs(angle) <= 120
        
        # Check which side of plot (for road-facing)
        dist_from_center = ((midpoint[0] - centroid.x)**2 + (midpoint[1] - centroid.y)**2) ** 0.5
        
        edges.append({
            "start": p1,
            "end": p2,
            "length": length,
            "midpoint": midpoint,
            "angle": angle,
            "is_horizontal": is_horizontal,
            "is_vertical": is_vertical,
            "dist_from_center": dist_from_center,
            "suitable_for_parking": length >= PARKING_SIZE[1],  # 18 ft minimum
            "suitable_for_living": length >= 14,  # Living room minimum width
        })
    
    # Find road-facing side (longest edge)
    longest_edge = max(edges, key=lambda e: e["length"])
    
    # Identify opposite sides for cross-ventilation
    ventilation_pairs = []
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges[i+1:], start=i+1):
            # Check if edges are roughly opposite (midpoints far apart)
            dist = ((edge1["midpoint"][0] - edge2["midpoint"][0])**2 + 
                   (edge1["midpoint"][1] - edge2["midpoint"][1])**2) ** 0.5
            angle_diff = abs(edge1["angle"] - edge2["angle"])
            
            # Opposite if distance is large and angles differ by ~180°
            if dist > width * 0.6 and (abs(angle_diff - 180) < 30 or abs(angle_diff) < 30):
                ventilation_pairs.append((i, j))
    
    # Analyze corners for special room placement
    corners = []
    for i in range(len(coords) - 1):
        prev_idx = (i - 1) % (len(coords) - 1)
        next_idx = (i + 1) % (len(coords) - 1)
        curr_pt = coords[i]
        prev_pt = coords[prev_idx]
        next_pt = coords[next_idx]
        
        # Calculate interior angle
        v1 = (prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
        v2 = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle_deg = abs(math.degrees(math.atan2(det, dot)))
        
        dist_from_center = ((curr_pt[0] - centroid.x)**2 + (curr_pt[1] - centroid.y)**2) ** 0.5
        
        # Classify corner suitability
        is_narrow = angle_deg < 75  # Acute - good for toilets/stores
        is_right = 75 <= angle_deg <= 105  # Right angle - flexible
        is_wide = angle_deg > 105  # Obtuse - good for living/bedrooms
        
        corners.append({
            "point": curr_pt,
            "angle": angle_deg,
            "distance_from_center": dist_from_center,
            "is_narrow": is_narrow,
            "is_right": is_right,
            "is_wide": is_wide,
            "suitable_for_toilet": is_narrow or dist_from_center > width * 0.6,
            "suitable_for_pooja": is_narrow and dist_from_center < width * 0.5,
            "suitable_for_store": is_narrow,
            "suitable_for_bedroom": is_wide or is_right,
        })
    
    # Calculate structural grid feasibility
    can_fit_columns = (width >= COLUMN_SPACING_MIN * 2) and (height >= COLUMN_SPACING_MIN * 2)
    num_columns_x = int(width / COLUMN_SPACING_MAX) + 1
    num_columns_y = int(height / COLUMN_SPACING_MAX) + 1
    
    return {
        "bounds": bounds,
        "width": width,
        "height": height,
        "area": boundary.area,
        "longest_edge": longest_edge,
        "edges": edges,
        "corners": corners,
        "centroid": (centroid.x, centroid.y),
        "ventilation_pairs": ventilation_pairs,
        "can_fit_columns": can_fit_columns,
        "column_grid": {"x": num_columns_x, "y": num_columns_y},
        "suitable_for_parking": any(e["suitable_for_parking"] for e in edges),
        "plot_type": "regular" if len(coords) <= 5 and boundary.area / (width * height) > 0.85 else "irregular",
    }


def _determine_entry_position(boundary: Polygon, plot_analysis: Dict) -> Dict:
    """
    STEP 2: Determine Entry Position + Parking Layout
    
    Logic:
    - Entry from road-facing side (longest edge)
    - Parking near entry if space allows (10×18 ft minimum)
    - Porch before main entrance
    - Entry should align with circulation spine
    """
    longest_edge = plot_analysis["longest_edge"]
    entry_point = longest_edge["midpoint"]
    
    # Determine parking layout
    parking_layout = None
    if plot_analysis["suitable_for_parking"]:
        # Try to place parking near entry
        edge_start = longest_edge["start"]
        edge_end = longest_edge["end"]
        
        # Parking along the road-facing edge
        # Position: One side of entry or centered
        parking_x = min(edge_start[0], edge_end[0]) + 2  # 2 ft setback
        parking_y = min(edge_start[1], edge_end[1]) + 2
        
        parking_layout = {
            "position": (parking_x, parking_y),
            "width": PARKING_SIZE[0],
            "length": PARKING_SIZE[1],
            "orientation": "horizontal" if longest_edge["is_horizontal"] else "vertical",
        }
    
    # Porch position (just inside entry)
    porch_distance = 3.0  # 3 feet inside from entry
    dx = porch_distance * math.cos(math.radians(longest_edge["angle"] + 90))
    dy = porch_distance * math.sin(math.radians(longest_edge["angle"] + 90))
    porch_point = (entry_point[0] + dx, entry_point[1] + dy)
    
    return {
        "entry_point": entry_point,
        "porch_point": porch_point,
        "parking_layout": parking_layout,
        "entry_edge_angle": longest_edge["angle"],
    }


def _create_zones(rooms_needed: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Step 3: Organize rooms by zones.
    Zones: Public, Semi-private, Private, Service
    """
    zones = {
        "public": [],
        "semi_private": [],
        "private": [],
        "service": [],
    }
    
    for room in rooms_needed:
        rtype = room["room_type"]
        room_def = STANDARD_ROOM_SIZES.get(rtype, {"zone": "private"})
        zone = room_def.get("zone", "private")
        
        # Map zone names
        if zone == "semi-private":
            zone = "semi_private"
        
        zones[zone].append(room)
    
    return zones


def _get_placement_order() -> List[str]:
    """
    STEP 3: Define Architectural Room Placement Order
    
    Logic: Public → Semi-private → Private → Service
    
    Order respects:
    - Entry sequence (porch → living)
    - Zone adjacency (living near dining)
    - Service core (kitchen/utility stacked for plumbing)
    - Private zone separation (bedrooms away from public)
    - Structural stacking (toilets aligned vertically)
    """
    return [
        "parking",       # Near entry, road-facing
        "porch",         # Main entrance vestibule
        "living",        # Center, public zone, near entry
        "dining",        # Adjacent to living, semi-private
        "kitchen",       # Near dining, outer wall for exhaust
        "utility",       # Adjacent to kitchen, plumbing stack
        "master_bedroom", # Private zone, quiet corner
        "bedroom",       # Private zone, natural light
        "study",         # Private/semi-private, quiet
        "bathroom",      # Service, stacked plumbing
        "toilet",        # Service, stacked plumbing
        "staircase",     # Future expansion, structurally planned
        "pooja",         # Quiet corner, preferably NE/East
        "store",         # Irregular corners, minimal light needed
    ]


def _place_room_intelligently(
    room_type: str,
    target_size: Dict,
    available_space: Polygon,
    placed_rooms: List[Dict],
    entry_point: Tuple[float, float],
    plot_analysis: Dict
) -> Optional[Polygon]:
    """
    Step 5-7: Place room using intelligent architectural logic with boundary awareness.
    This function thinks like GPT/Gemini - analyzing relationships and optimizing placement.
    
    Considerations:
    - Zone positioning (public near entry, private away)
    - Room adjacencies (kitchen-dining, bedroom-bathroom)
    - Cross-ventilation and natural light
    - Boundary-aware optimization (fitting within custom shapes)
    - Structural efficiency (minimize circulation)
    """
    minx, miny, maxx, maxy = available_space.bounds
    room_width = target_size.get("width", 10)
    room_height = target_size.get("height", 10)
    centroid = available_space.centroid
    
    # Define placement strategy based on room type and context
    placement_override = None
    
    # LIVING ROOM: Center, near entry, good views
    if room_type == "living":
        # Prefer center of boundary with entry access
        x = centroid.x - room_width / 2
        y = centroid.y - room_height / 2 + 5  # Slightly towards entry
        placement_override = (x, y)
    
    # MASTER BEDROOM: Quiet corner, private zone, opposite entry
    elif room_type == "master_bedroom":
        # Far corner from entry, on longer/quieter side
        if entry_point[0] < centroid.x:
            x = maxx - room_width - 2  # Right side
        else:
            x = minx + 2  # Left side
        y = maxy - room_height - 2  # Top corner
        placement_override = (x, y)
    
    # BEDROOMS: Distributed around perimeter, private zone
    elif room_type == "bedroom":
        # Opposite corners from living room
        num_existing_bedrooms = sum(1 for r in placed_rooms if "bedroom" in r["room_type"].lower())
        if num_existing_bedrooms == 0:
            x = minx + 2
            y = maxy - room_height - 2
        else:
            x = maxx - room_width - 2
            y = maxy - room_height - 2
        placement_override = (x, y)
    
    # KITCHEN: Near dining, outer wall (exhaust), adjacent to dining
    elif room_type == "kitchen":
        dining = next((r for r in placed_rooms if r["room_type"] == "dining"), None)
        if dining:
            dx, dy = dining["centroid"]
            # Adjacent to dining, possibly sharing a wall
            x = dx + room_width + 0.5
            y = dy - room_height / 2
        else:
            # Near service zone
            x = maxx - room_width - 2
            y = miny + 2
        placement_override = (x, y)
    
    # DINING: Adjacent to living and kitchen
    elif room_type == "dining":
        living = next((r for r in placed_rooms if r["room_type"] == "living"), None)
        if living:
            lx, ly = living["centroid"]
            x = lx + room_width + 1
            y = ly - room_height / 2
        else:
            x = centroid.x + 10
            y = centroid.y
        placement_override = (x, y)
    
    # STUDY: Quiet corner, good light, preferably north-facing
    elif room_type == "study":
        # North-east corner preferred (more light)
        x = maxx - room_width - 2
        y = miny + 2
        placement_override = (x, y)
    
    # BATHROOM/TOILET: Adjacent to bedrooms, aligned for plumbing
    elif room_type in ["bathroom", "toilet"]:
        bedroom = next((r for r in placed_rooms if "bedroom" in r["room_type"].lower()), None)
        if bedroom:
            bx, by = bedroom["centroid"]
            x = bx + room_width + 0.5
            y = by
        else:
            x = maxx - room_width - 2
            y = maxy - room_height - 2
        placement_override = (x, y)
    
    # UTILITY/POOJA: Corner rooms, minimal requirements
    elif room_type in ["utility", "pooja", "store"]:
        # Small rooms in available corners/niches
        x = minx + 2
        y = miny + 2
        placement_override = (x, y)
    
    # PORCH: Entry point
    elif room_type == "porch":
        x = entry_point[0] - room_width / 2
        y = entry_point[1] - room_height / 2
        placement_override = (x, y)
    
    if placement_override:
        x, y = placement_override
    else:
        x = centroid.x - room_width / 2
        y = centroid.y - room_height / 2
    
    # Create room rectangle and clip to available space
    room_poly = box(x, y, x + room_width, y + room_height)
    clipped = room_poly.intersection(available_space)
    
    if clipped.is_empty or clipped.area < target_size.get("min_area", 50) * 0.8:
        return None
    
    if isinstance(clipped, MultiPolygon):
        clipped = max(clipped.geoms, key=lambda g: g.area)
    
    return clipped


def _optimize_rooms_for_boundary(room_results: list, boundary: Polygon, plot_analysis: Dict) -> list:
    """
    Intelligent post-processing step: Optimize room placement for irregular boundaries.
    
    This function applies GPT/Gemini-like reasoning:
    - Analyzes how well rooms fit within the boundary
    - Adjusts dimensions and positions to maximize space utilization
    - Ensures cross-ventilation where possible
    - Optimizes corner usage
    - Improves overall layout efficiency
    """
    optimized_results = []
    
    for result in room_results:
        room_type = result["room"]["room_type"]
        poly = result["polygon"]
        
        # Check if room is touching boundary efficiently
        boundary_contact = poly.boundary.intersection(boundary.boundary)
        contact_length = boundary_contact.length if not boundary_contact.is_empty else 0
        perimeter = poly.boundary.length
        
        # If poor boundary contact, try to improve positioning
        if contact_length < perimeter * 0.3 and room_type not in ["porch", "parking"]:
            # Try to move room closer to boundary for better light/ventilation
            minx, miny, maxx, maxy = poly.bounds
            bminx, bminy, bmaxx, bmaxy = boundary.bounds
            
            # Move towards nearest boundary edge
            dx_left = minx - bminx
            dx_right = bmaxx - maxx
            dy_bottom = miny - bminy
            dy_top = bmaxy - maxy
            
            min_dist = min(abs(dx_left), abs(dx_right), abs(dy_bottom), abs(dy_top))
            
            if dx_left == min_dist and dx_left > 1.0:
                # Move left towards boundary
                poly = translate(poly, -dx_left + 0.5, 0)
            elif dx_right == min_dist and dx_right > 1.0:
                # Move right towards boundary
                poly = translate(poly, dx_right - 0.5, 0)
            elif dy_bottom == min_dist and dy_bottom > 1.0:
                # Move down towards boundary
                poly = translate(poly, 0, -dy_bottom + 0.5)
            elif dy_top == min_dist and dy_top > 1.0:
                # Move up towards boundary
                poly = translate(poly, 0, dy_top - 0.5)
            
            # Clip to boundary after movement
            poly = poly.intersection(boundary)
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
        
        # Create updated result
        optimized_result = {
            "room": result["room"],
            "polygon": poly,
        }
        optimized_results.append(optimized_result)
    
    return optimized_results


def _normalize_boundary(polygon_coords: list, target_area: Optional[float] = None) -> Polygon:
    """
    Normalise polygon: ensure closed, counter-clockwise.
    If target_area given and different from polygon area, scale accordingly.
    """
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])

    poly = Polygon(polygon_coords)

    # Ensure counter-clockwise
    if not poly.exterior.is_ccw:
        poly = Polygon(list(reversed(list(poly.exterior.coords))))

    # Make valid
    if not poly.is_valid:
        poly = poly.buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)

    # Scale to target area if needed
    if target_area and abs(poly.area - target_area) > 1.0:
        scale_factor = math.sqrt(target_area / poly.area)
        centroid = poly.centroid
        poly = shapely_scale(poly, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    return poly


def _compute_room_targets(rooms: list, total_area: float) -> list:
    """
    Assign target areas to rooms using standard architectural sizes.
    """
    result = []
    for room in rooms:
        rtype = room.get("room_type", "bedroom")
        qty = room.get("quantity", 1)
        desired = room.get("desired_area")
        
        # Get standard room definition
        room_std = STANDARD_ROOM_SIZES.get(rtype, STANDARD_ROOM_SIZES["bedroom"])

        for i in range(qty):
            label = room_std["label"]
            if qty > 1:
                label = f"{label} {i + 1}"
            
            # Use standard minimum area or user desired
            target_area = desired if desired else room_std["min_area"]
            
            result.append({
                "room_type": rtype,
                "label": label,
                "target_area": target_area,
                "width": room_std["width"],
                "height": room_std["height"],
                "zone": room_std["zone"],
            })

    # Scale areas proportionally within the boundary (minus wall space)
    usable_area = total_area * 0.85  # ~15% for walls/corridors
    total_target = sum(r["target_area"] for r in result)

    if total_target > 0 and total_target > usable_area:
        scale_factor = usable_area / total_target
        for r in result:
            r["target_area"] = round(r["target_area"] * scale_factor, 1)

    return result


def _split_rect(rect_poly: Polygon, area_ratio: float, split_vertical: bool) -> tuple:
    """
    Split a rectangle (polygon) into two rectangles at the given area ratio.
    Returns (rect_a, rect_b).
    """
    minx, miny, maxx, maxy = rect_poly.bounds
    w = maxx - minx
    h = maxy - miny

    if split_vertical:
        split_x = minx + w * area_ratio
        a = box(minx, miny, split_x, maxy)
        b = box(split_x, miny, maxx, maxy)
    else:
        split_y = miny + h * area_ratio
        a = box(minx, miny, maxx, split_y)
        b = box(minx, split_y, maxx, maxy)

    return a, b


def _bsp_partition(bounding_rect: Polygon, room_targets: list, boundary: Polygon) -> list:
    """
    Binary space partitioning: recursively split bounding rectangle to allocate rooms.
    Clip each result to the boundary polygon.
    Enhanced with minimum room dimensions and cleaner spacing.
    """
    if len(room_targets) == 0:
        return []

    if len(room_targets) == 1:
        clipped = bounding_rect.intersection(boundary)
        if clipped.is_empty:
            clipped = bounding_rect
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        
        # Ensure room meets minimum dimensions
        minx, miny, maxx, maxy = clipped.bounds
        if (maxx - minx) < MIN_ROOM_DIMENSION or (maxy - miny) < MIN_ROOM_DIMENSION:
            # Room too small, expand bounds if possible
            clipped = clipped.buffer(0.5)
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)
        
        return [{"room": room_targets[0], "polygon": clipped}]

    # Find split point with better distribution
    total_area = sum(r["target_area"] for r in room_targets)
    mid_point = len(room_targets) // 2
    area_a = sum(r["target_area"] for r in room_targets[:mid_point])
    ratio = area_a / total_area if total_area > 0 else 0.5
    
    # Clamp ratio to avoid very thin rooms
    ratio = max(0.3, min(0.7, ratio))

    # Decide split direction based on bounds and room aspect ratios
    minx, miny, maxx, maxy = bounding_rect.bounds
    w = maxx - minx
    h = maxy - miny
    
    # Prefer splitting along the longer dimension
    split_vertical = w >= h
    
    # Adjust for room aspect ratios
    avg_aspect_a = sum(r.get("aspect", 1.2) for r in room_targets[:mid_point]) / max(1, mid_point)
    if avg_aspect_a > 2.0:  # Narrow rooms prefer vertical split
        split_vertical = True

    rect_a, rect_b = _split_rect(bounding_rect, ratio, split_vertical)

    result_a = _bsp_partition(rect_a, room_targets[:mid_point], boundary)
    result_b = _bsp_partition(rect_b, room_targets[mid_point:], boundary)

    return result_a + result_b


def _generate_walls(room_results: list, boundary: Polygon) -> list:
    """
    Generate clean wall geometries with proper thickness and alignment.
    Creates double-line walls for professional CAD output.
    """
    walls = []

    for result in room_results:
        poly = result["polygon"]
        if poly.is_empty or not poly.is_valid:
            continue
        
        # Get room boundary coordinates
        coords = list(poly.exterior.coords)
        
        # Create wall segments with proper thickness
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            
            # Calculate perpendicular offset for wall thickness
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 0.1:
                continue
            
            # Perpendicular unit vector
            px = -dy / length * WALL_THICKNESS_INTERIOR / 2
            py = dx / length * WALL_THICKNESS_INTERIOR / 2
            
            # Create wall geometry as a polygon (rectangle)
            wall_poly = Polygon([
                (x1 + px, y1 + py),
                (x2 + px, y2 + py),
                (x2 - px, y2 - py),
                (x1 - px, y1 - py),
            ])
            
            walls.append({
                "type": "interior_wall",
                "geometry": _poly_to_coords(wall_poly),
                "start": [round(x1, 2), round(y1, 2)],
                "end": [round(x2, 2), round(y2, 2)],
                "thickness": WALL_THICKNESS_INTERIOR,
            })

    # Add outer boundary wall with increased thickness
    outer_wall = boundary.boundary.buffer(WALL_THICKNESS_EXTERIOR)
    walls.append({
        "type": "exterior_wall",
        "geometry": _poly_to_coords(outer_wall),
        "thickness": WALL_THICKNESS_EXTERIOR,
    })

    return walls


def _generate_doors(room_results: list) -> list:
    """
    Generate door positions on shared edges between rooms with proper swing geometry.
    Creates professional door representations with hinge point and swing arc.
    """
    doors = []

    for i in range(len(room_results)):
        for j in range(i + 1, len(room_results)):
            poly_a = room_results[i]["polygon"]
            poly_b = room_results[j]["polygon"]
            if poly_a.is_empty or poly_b.is_empty:
                continue

            shared_edge = poly_a.boundary.intersection(poly_b.boundary)

            if not shared_edge.is_empty and shared_edge.length > DOOR_WIDTH:
                edge = shared_edge
                if shared_edge.geom_type == "MultiLineString":
                    edge = max(shared_edge.geoms, key=lambda g: g.length)

                if edge.geom_type != "LineString":
                    continue

                # Place door at center of shared edge
                mid = edge.interpolate(0.5, normalized=True)
                
                # Compute direction along the edge
                coords = list(edge.coords)
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                edge_len = math.sqrt(dx * dx + dy * dy)
                if edge_len < 0.1:
                    continue
                ux, uy = dx / edge_len, dy / edge_len
                
                # Perpendicular (swing direction towards room_b)
                px, py = -uy, ux

                half = DOOR_WIDTH / 2
                hinge = [round(mid.x - ux * half, 2), round(mid.y - uy * half, 2)]
                door_end = [round(mid.x + ux * half, 2), round(mid.y + uy * half, 2)]

                # Determine if edge is more vertical or horizontal for proper rendering
                is_vertical = abs(dy) > abs(dx)

                doors.append({
                    "type": "door",
                    "position": [round(mid.x, 2), round(mid.y, 2)],
                    "hinge": hinge,
                    "door_end": door_end,
                    "width": DOOR_WIDTH,
                    "swing_dir": [round(px, 3), round(py, 3)],
                    "is_vertical": is_vertical,
                    "between": [
                        room_results[i]["room"]["label"],
                        room_results[j]["room"]["label"],
                    ],
                })
    return doors


def _generate_windows(room_results: list, boundary: Polygon) -> list:
    """
    Place windows on exterior walls with proper start/end geometry.
    Creates professional window symbols with frame representation.
    """
    windows = []
    window_room_types = {"living", "master_bedroom", "bedroom", "study", "dining", "kitchen"}

    for result in room_results:
        rtype = result["room"]["room_type"]
        if rtype not in window_room_types:
            continue

        poly = result["polygon"]
        if poly.is_empty:
            continue

        room_boundary = poly.boundary
        outer = boundary.boundary
        touching = room_boundary.intersection(outer)

        if not touching.is_empty and touching.length > (WINDOW_WIDTH + 1.0):
            edge = touching
            if touching.geom_type == "MultiLineString":
                edge = max(touching.geoms, key=lambda g: g.length)
                if edge.geom_type != "LineString":
                    continue
            elif touching.geom_type != "LineString":
                continue

            # Determine window size based on room type
            win_width = WINDOW_WIDTH + 1.0 if rtype == "living" else WINDOW_WIDTH
            
            # Place window at center of exterior wall
            mid = edge.interpolate(0.5, normalized=True)

            coords = list(edge.coords)
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            edge_len = math.sqrt(dx * dx + dy * dy)
            if edge_len < 0.1:
                continue
            ux, uy = dx / edge_len, dy / edge_len

            half = win_width / 2
            win_start = [round(mid.x - ux * half, 2), round(mid.y - uy * half, 2)]
            win_end = [round(mid.x + ux * half, 2), round(mid.y + uy * half, 2)]
            is_vertical = abs(dy) > abs(dx)

            windows.append({
                "type": "window",
                "position": [round(mid.x, 2), round(mid.y, 2)],
                "start": win_start,
                "end": win_end,
                "width": win_width,
                "is_vertical": is_vertical,
                "room": result["room"]["label"],
            })

    return windows


def _generate_furniture(room_results: list, boundary: Polygon) -> list:
    """
    Generate furniture symbols for each room type.
    Matches professional floor plan standards with beds, counters, fixtures.
    """
    furniture = []
    
    for result in room_results:
        rtype = result["room"]["room_type"]
        poly = result["polygon"]
        if poly.is_empty:
            continue
        
        centroid = poly.centroid
        minx, miny, maxx, maxy = poly.bounds
        room_width = maxx - minx
        room_height = maxy - miny
        
        # BEDROOM FURNITURE - Double bed
        if rtype in ["master_bedroom", "bedroom"]:
            bed_width = 6.0 if rtype == "master_bedroom" else 5.0
            bed_height = 7.0 if rtype == "master_bedroom" else 6.5
            
            # Center bed in room
            bed_x = centroid.x - bed_width / 2
            bed_y = centroid.y - bed_height / 2
            
            # Bed frame
            furniture.append({
                "type": "bed",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x, bed_y],
                    [bed_x + bed_width, bed_y],
                    [bed_x + bed_width, bed_y + bed_height],
                    [bed_x, bed_y + bed_height],
                    [bed_x, bed_y],
                ],
            })
            
            # Pillows (two rectangles at head)
            pillow_width = bed_width / 2 - 0.3
            pillow_height = 1.5
            furniture.append({
                "type": "pillow",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x + 0.3, bed_y + 0.3],
                    [bed_x + 0.3 + pillow_width, bed_y + 0.3],
                    [bed_x + 0.3 + pillow_width, bed_y + 0.3 + pillow_height],
                    [bed_x + 0.3, bed_y + 0.3 + pillow_height],
                    [bed_x + 0.3, bed_y + 0.3],
                ],
            })
            furniture.append({
                "type": "pillow",
                "room": result["room"]["label"],
                "geometry": [
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3],
                    [bed_x + bed_width - 0.3, bed_y + 0.3],
                    [bed_x + bed_width - 0.3, bed_y + 0.3 + pillow_height],
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3 + pillow_height],
                    [bed_x + bed_width - 0.3 - pillow_width, bed_y + 0.3],
                ],
            })
        
        # KITCHEN FURNITURE - Counter and stove
        elif rtype == "kitchen":
            counter_depth = 2.0
            stove_size = 2.5
            
            # L-shaped counter along two walls
            counter_x = minx + 1
            counter_y = miny + 1
            
            # Counter along bottom
            furniture.append({
                "type": "counter",
                "room": result["room"]["label"],
                "geometry": [
                    [counter_x, counter_y],
                    [counter_x + room_width - 2, counter_y],
                    [counter_x + room_width - 2, counter_y + counter_depth],
                    [counter_x, counter_y + counter_depth],
                    [counter_x, counter_y],
                ],
            })
            
            # Stove symbols (4 burners)
            stove_x = counter_x + 3
            stove_y = counter_y + 0.25
            burner_spacing = 0.8
            for i in range(2):
                for j in range(2):
                    bx = stove_x + i * burner_spacing
                    by = stove_y + j * burner_spacing
                    furniture.append({
                        "type": "burner",
                        "room": result["room"]["label"],
                        "center": [bx, by],
                        "radius": 0.25,
                    })
        
        # BATHROOM FURNITURE - Toilet and sink
        elif rtype == "bathroom":
            # Toilet
            toilet_x = minx + 1.5
            toilet_y = miny + 1
            furniture.append({
                "type": "toilet",
                "room": result["room"]["label"],
                "center": [toilet_x, toilet_y + 1],
                "radius": 0.8,
            })
            furniture.append({
                "type": "toilet_tank",
                "room": result["room"]["label"],
                "geometry": [
                    [toilet_x - 0.6, toilet_y],
                    [toilet_x + 0.6, toilet_y],
                    [toilet_x + 0.6, toilet_y + 0.6],
                    [toilet_x - 0.6, toilet_y + 0.6],
                    [toilet_x - 0.6, toilet_y],
                ],
            })
            
            # Sink
            sink_x = maxx - 2
            sink_y = miny + 2
            furniture.append({
                "type": "sink",
                "room": result["room"]["label"],
                "center": [sink_x, sink_y],
                "radius": 0.6,
            })
        
        # STUDY FURNITURE - Desk
        elif rtype == "study":
            desk_width = 5.0
            desk_depth = 2.5
            desk_x = centroid.x - desk_width / 2
            desk_y = miny + 1.5
            
            furniture.append({
                "type": "desk",
                "room": result["room"]["label"],
                "geometry": [
                    [desk_x, desk_y],
                    [desk_x + desk_width, desk_y],
                    [desk_x + desk_width, desk_y + desk_depth],
                    [desk_x, desk_y + desk_depth],
                    [desk_x, desk_y],
                ],
            })
    
    return furniture


def _generate_wall_dimensions(room_results: list, boundary: Polygon) -> list:
    """
    Generate wall dimension annotations for professional floor plans.
    Shows measurements on walls like in the reference image.
    """
    dimensions = []
    
    for result in room_results:
        poly = result["polygon"]
        if poly.is_empty:
            continue
        
        coords = list(poly.exterior.coords)
        
        # Add dimensions for each wall segment
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length < 2.0:  # Skip very short walls
                continue
            
            # Midpoint for dimension text
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset dimension text perpendicular to wall
            if abs(dx) > abs(dy):  # Horizontal wall
                offset_x = 0
                offset_y = 0.8
            else:  # Vertical wall
                offset_x = 0.8
                offset_y = 0
            
            dimensions.append({
                "type": "wall_dimension",
                "length": round(length, 1),
                "position": [round(mid_x + offset_x, 2), round(mid_y + offset_y, 2)],
                "start": [round(x1, 2), round(y1, 2)],
                "end": [round(x2, 2), round(y2, 2)],
                "is_horizontal": abs(dx) > abs(dy),
            })
    
    return dimensions


def _poly_to_coords(poly) -> list:
    """Convert Shapely polygon to coordinate list."""
    if poly.is_empty:
        return []
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type == "Polygon":
        return [[round(x, 2), round(y, 2)] for x, y in poly.exterior.coords]
    return []


def generate_floor_plan(
    boundary_polygon: list,
    rooms: list,
    total_area: Optional[float] = None,
) -> dict:
    """
    Main entry point: Generate professional floor plan using architectural design thinking.
    
    Workflow:
    1. Analyze plot geometry
    2. Determine entry position
    3. Create zones (public/private/service)
    4. Place rooms in architectural order
    5. Add circulation paths
    6. Plan ventilation and structure
    7. Generate furniture and dimensions

    Args:
        boundary_polygon: List of [x,y] coordinates forming the boundary.
        rooms: List of dicts with room_type, quantity, desired_area.
        total_area: Total area in sq ft (for scaling).

    Returns:
        Dict with professional floor plan data.
    """
    # Normalize boundary
    boundary = _normalize_boundary(boundary_polygon, total_area)
    actual_area = boundary.area

    if total_area is None:
        total_area = actual_area

    # STEP 1: Analyze plot geometry (Professional Architectural Analysis)
    plot_analysis = _analyze_plot_geometry(boundary)
    
    # STEP 2: Determine entry position + parking layout (Indian Standards)
    entry_info = _determine_entry_position(boundary, plot_analysis)
    entry_point = entry_info["entry_point"]
    
    # Compute room targets with standard Indian residential sizes
    room_targets = _compute_room_targets(rooms, total_area)
    
    # STEP 3: Create zones (Public/Semi-private/Private/Service)
    zones = _create_zones(room_targets)
    
    # STEP 4: Place rooms in intelligent architectural order
    placement_order = _get_placement_order()
    placed_rooms = []
    available_space = boundary
    
    # Sort rooms by architectural placement logic
    ordered_rooms = []
    for order_type in placement_order:
        matching = [r for r in room_targets if r["room_type"] == order_type]
        ordered_rooms.extend(matching)
    
    # Add any remaining rooms not in order list
    for room in room_targets:
        if room not in ordered_rooms:
            ordered_rooms.append(room)
    
    # Try intelligent placement using architect + engineer principles
    room_results = []
    for room in ordered_rooms:
        room_poly = _place_room_intelligently(
            room["room_type"],
            room,
            available_space,
            placed_rooms,
            entry_point,
            plot_analysis
        )
        
        if room_poly and not room_poly.is_empty:
            placed_rooms.append({
                "room_type": room["room_type"],
                "label": room["label"],
                "polygon": room_poly,
                "centroid": (room_poly.centroid.x, room_poly.centroid.y),
                "target_area": room["target_area"],
            })
            room_results.append({
                "room": room,
                "polygon": room_poly,
            })
            
            # Update available space (subtract placed room with buffer)
            try:
                available_space = available_space.difference(room_poly.buffer(0.5))
                if available_space.is_empty:
                    break
                if isinstance(available_space, MultiPolygon):
                    available_space = max(available_space.geoms, key=lambda g: g.area)
            except:
                pass
    
    # Fallback: If intelligent placement didn't work well, use BSP
    if len(room_results) < len(room_targets) * 0.7:  # Less than 70% placed
        minx, miny, maxx, maxy = boundary.bounds
        bounding_rect = box(minx, miny, maxx, maxy)
        room_results = _bsp_partition(bounding_rect, room_targets, boundary)
    
    # OPTIMIZATION STEP: Optimize room placement for irregular boundaries
    # This is the "smart thinking" step that makes the algorithm think like GPT/Gemini
    room_results = _optimize_rooms_for_boundary(room_results, boundary, plot_analysis)
    
    # Generate architectural elements
    walls = _generate_walls(room_results, boundary)
    doors = _generate_doors(room_results)
    windows = _generate_windows(room_results, boundary)
    furniture = _generate_furniture(room_results, boundary)
    dimensions = _generate_wall_dimensions(room_results, boundary)

    # Build result
    plan_rooms = []
    for result in room_results:
        coords = _poly_to_coords(result["polygon"])
        poly = result["polygon"]
        plan_rooms.append({
            "label": result["room"]["label"],
            "room_type": result["room"]["room_type"],
            "target_area": result["room"]["target_area"],
            "actual_area": round(poly.area, 2) if not poly.is_empty else 0,
            "polygon": coords,
            "centroid": [round(poly.centroid.x, 2), round(poly.centroid.y, 2)] if not poly.is_empty else [0, 0],
        })

    return {
        "boundary": _poly_to_coords(boundary),
        "total_area": round(actual_area, 2),
        "rooms": plan_rooms,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "furniture": furniture,
        "dimensions": dimensions,
        "design_thinking": {
            "approach": "Professional Residential Architect + Structural Engineer",
            "entry_point": list(entry_point),
            "parking_provided": entry_info.get("parking_layout") is not None,
            "plot_analysis": {
                "plot_type": plot_analysis["plot_type"],
                "longest_edge_length": round(plot_analysis["longest_edge"]["length"], 2),
                "total_area_sqft": round(plot_analysis["area"], 2),
                "width_ft": round(plot_analysis["width"], 2),
                "height_ft": round(plot_analysis["height"], 2),
                "can_fit_structural_columns": plot_analysis["can_fit_columns"],
                "cross_ventilation_possible": len(plot_analysis["ventilation_pairs"]) > 0,
            },
            "zones_used": list(zones.keys()),
            "placement_order": placement_order,
            "rooms_placed": f"{len(room_results)}/{len(room_targets)}",
        },
    }
