"""
Smart Space-Filling Floor Plan Generator
==========================================
Uses adaptive room placement to fill irregular boundary shapes.
Scans the polygon to find largest placeable rectangles.
"""

import json
import os
import math
from typing import Dict, List, Tuple, Optional
import structlog

from app.core.config import settings
from app.api.routes.jobs import update_job_status, JobStage

logger = structlog.get_logger()


def generate_design(job_id: str, requirements: Dict):
    """Main design generation."""
    logger.info("Starting design generation", job_id=job_id)
    
    try:
        update_job_status(job_id, JobStage.GENERATING_DESIGN, 40, "Loading plot data...")
        boundary_data = load_boundary_data(job_id)
        
        update_job_status(job_id, JobStage.GENERATING_DESIGN, 50, "Designing room layout...")
        design = create_smart_layout(boundary_data, requirements)
        
        output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "design.json"), 'w') as f:
            json.dump(design, f, indent=2)
        
        update_job_status(job_id, JobStage.CREATING_CAD, 70, "Rendering floor plan...")
        render_floorplan(design, output_dir)
        
        update_job_status(job_id, JobStage.CREATING_CAD, 80, "Generating CAD files (DXF)...")
        generate_dxf_output(design, output_dir, job_id)
        
        update_job_status(job_id, JobStage.COMPLETED, 100, "Design complete!")
        return design
        
    except Exception as e:
        logger.error("Design failed", error=str(e))
        update_job_status(job_id, JobStage.FAILED, 0, f"Error: {str(e)}")
        raise


def load_boundary_data(job_id: str) -> Dict:
    """Load boundary data."""
    try:
        path = os.path.join(settings.OUTPUT_DIR, job_id, "boundary.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Could not load boundary", error=str(e))
    
    return {
        "points": [(0, 0), (15, 0), (15, 10), (0, 10)],
        "area_sqm": 150,
        "dimensions": [15, 10]
    }


def point_in_polygon(x: float, y: float, polygon: List[Tuple]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def find_max_rect_at_point(x: float, y: float, polygon: List[Tuple], 
                           max_w: float, max_h: float, step: float = 0.2) -> Tuple[float, float]:
    """Find maximum rectangle that fits at given point inside polygon."""
    if not point_in_polygon(x, y, polygon):
        return 0, 0
    
    best_w, best_h = 0, 0
    
    # Try different sizes
    for w in [i * step for i in range(1, int(max_w / step) + 1)]:
        for h in [i * step for i in range(1, int(max_h / step) + 1)]:
            # Check all 4 corners
            corners = [
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ]
            if all(point_in_polygon(cx, cy, polygon) for cx, cy in corners):
                if w * h > best_w * best_h:
                    best_w, best_h = w, h
    
    return best_w, best_h


def rect_fits_in_polygon(x: float, y: float, w: float, h: float, polygon: List[Tuple], margin: float = 0.1) -> bool:
    """Check if a rectangle fits inside the polygon."""
    # Check corners and edges
    check_points = [
        (x + margin, y + margin),
        (x + w - margin, y + margin),
        (x + w - margin, y + h - margin),
        (x + margin, y + h - margin),
        (x + w/2, y + margin),
        (x + w/2, y + h - margin),
        (x + margin, y + h/2),
        (x + w - margin, y + h/2),
        (x + w/2, y + h/2)
    ]
    return all(point_in_polygon(px, py, polygon) for px, py in check_points)


def find_best_room_position(polygon: List[Tuple], min_x: float, min_y: float, 
                            max_x: float, max_y: float, target_w: float, target_h: float,
                            placed_rooms: List[Dict], step: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
    """Find best position for a room - optimized for speed."""
    
    best = None
    best_score = -1
    
    # Use larger step for speed
    x_steps = max(1, int((max_x - min_x - target_w) / step))
    y_steps = max(1, int((max_y - min_y - target_h) / step))
    
    # Limit iterations for performance
    max_iterations = 200
    iteration = 0
    
    for xi in range(x_steps + 1):
        x = min_x + xi * step
        for yi in range(y_steps + 1):
            iteration += 1
            if iteration > max_iterations:
                break
                
            y = min_y + yi * step
            
            # Quick bounds check
            if x + target_w > max_x or y + target_h > max_y:
                continue
            
            # Check if fits in polygon (simplified - just check corners)
            corners = [
                (x + 0.1, y + 0.1),
                (x + target_w - 0.1, y + 0.1),
                (x + target_w - 0.1, y + target_h - 0.1),
                (x + 0.1, y + target_h - 0.1)
            ]
            if not all(point_in_polygon(px, py, polygon) for px, py in corners):
                continue
            
            # Check overlap with placed rooms
            overlaps = False
            for room in placed_rooms:
                rx, ry = room["x"], room["y"]
                rw, rh = room["width"], room["height"]
                if not (x + target_w <= rx or x >= rx + rw or y + target_h <= ry or y >= ry + rh):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # Found a valid position - score it
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            dist_to_center = math.sqrt((x + target_w/2 - cx)**2 + (y + target_h/2 - cy)**2)
            score = 1000 - dist_to_center
            
            if score > best_score:
                best_score = score
                best = (x, y, target_w, target_h)
        
        if iteration > max_iterations:
            break
    
    return best


def create_smart_layout(boundary: Dict, requirements: Dict) -> Dict:
    """Create layout with rooms placed inside irregular polygon boundaries."""
    
    points = boundary.get("points", [(0, 0), (15, 0), (15, 10), (0, 10)])
    polygon = [(p[0], p[1]) for p in points]
    
    # Get bounding box
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate actual polygon area using shoelace formula
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    total_area = abs(area) / 2
    
    bedrooms = requirements.get("bedrooms", 3)
    bathrooms = requirements.get("bathrooms", 2)
    style = requirements.get("style", "modern")
    
    rooms = []
    placed = []
    
    # Room size targets based on total area
    living_area = total_area * 0.25
    kitchen_area = total_area * 0.12
    dining_area = total_area * 0.10
    bedroom_area = total_area * 0.12
    bathroom_area = total_area * 0.05
    
    # Minimum dimensions
    min_room_dim = 2.5
    wall_gap = 0.15
    
    def add_room(name: str, room_type: str, target_area: float, aspect_ratio: float = 1.2):
        """Try to place a room with target area."""
        # Calculate dimensions
        h = math.sqrt(target_area / aspect_ratio)
        w = target_area / h
        w = max(min_room_dim, min(w, width * 0.5))
        h = max(min_room_dim, min(h, height * 0.5))
        
        # Try to find position with larger step for speed
        pos = find_best_room_position(polygon, min_x + wall_gap, min_y + wall_gap,
                                       max_x - wall_gap, max_y - wall_gap, 
                                       w, h, placed, step=1.0)
        
        # Try smaller if not found (only one retry)
        if pos is None:
            sw, sh = w * 0.7, h * 0.7
            pos = find_best_room_position(polygon, min_x + wall_gap, min_y + wall_gap,
                                           max_x - wall_gap, max_y - wall_gap,
                                           sw, sh, placed, step=1.0)
        
        if pos:
            x, y, rw, rh = pos
            room = create_room(name, room_type, x, y, rw - wall_gap, rh - wall_gap)
            rooms.append(room)
            placed.append({"x": x, "y": y, "width": rw, "height": rh})
            return True
        return False
    
    # Place rooms in priority order
    add_room("Living Room", "living", living_area, 1.5)
    add_room("Kitchen", "kitchen", kitchen_area, 1.3)
    add_room("Dining", "dining", dining_area, 1.0)
    
    # Bedrooms
    add_room("Master\\nBedroom", "bedroom", bedroom_area * 1.3, 1.2)
    for i in range(bedrooms - 1):
        add_room(f"Bedroom {i+2}", "bedroom", bedroom_area, 1.1)
    
    # Bathrooms
    add_room("Master\\nBath", "bathroom", bathroom_area * 1.2, 0.8)
    for i in range(bathrooms - 1):
        add_room(f"Bath {i+2}", "bathroom", bathroom_area, 0.7)
    
    logger.info(f"Created layout with {len(rooms)} rooms for irregular polygon")
    
    return {
        "boundary": points,
        "rooms": rooms,
        "total_area": total_area,
        "style": style,
        "dimensions": [round(width, 1), round(height, 1)]
    }


def _legacy_create_smart_layout(boundary: Dict, requirements: Dict) -> Dict:
    """Legacy grid-based layout for simple rectangles."""
    
    points = boundary.get("points", [(0, 0), (15, 0), (15, 10), (0, 10)])
    
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    
    width = max_x - min_x
    height = max_y - min_y
    total_area = width * height
    
    bedrooms = requirements.get("bedrooms", 3)
    bathrooms = requirements.get("bathrooms", 2)
    style = requirements.get("style", "modern")
    
    rooms = []
    wall_thickness = 0.15
    padding = wall_thickness
    
    usable_width = width - 2 * padding
    usable_height = height - 2 * padding
    
    top_row_height = usable_height * 0.55
    bottom_row_height = usable_height * 0.45
    
    living_width = usable_width * 0.5
    kitchen_width = usable_width * 0.3
    dining_width = usable_width * 0.2
    
    # Living Room
    rooms.append(create_room(
        "Living Room", "living",
        min_x + padding, min_y + padding + bottom_row_height,
        living_width - wall_thickness, top_row_height - wall_thickness
    ))
    
    # Kitchen
    rooms.append(create_room(
        "Kitchen", "kitchen",
        min_x + padding + living_width, min_y + padding + bottom_row_height,
        kitchen_width - wall_thickness, top_row_height - wall_thickness
    ))
    
    # Dining
    rooms.append(create_room(
        "Dining", "dining",
        min_x + padding + living_width + kitchen_width, min_y + padding + bottom_row_height,
        dining_width - wall_thickness, top_row_height - wall_thickness
    ))
    
    # Bottom row: Bedrooms and Bathrooms
    num_bottom_rooms = bedrooms + bathrooms
    room_width = usable_width / num_bottom_rooms
    
    current_x = min_x + padding
    
    # Master Bedroom (larger)
    master_width = room_width * 1.3
    rooms.append(create_room(
        "Master\\nBedroom", "bedroom",
        current_x, min_y + padding,
        master_width - wall_thickness, bottom_row_height - wall_thickness
    ))
    current_x += master_width
    
    # Master Bathroom (attached)
    bath_width = room_width * 0.7
    rooms.append(create_room(
        "Master\\nBath", "bathroom",
        current_x, min_y + padding,
        bath_width - wall_thickness, bottom_row_height - wall_thickness
    ))
    current_x += bath_width
    
    # Remaining bedrooms
    remaining_width = (min_x + padding + usable_width) - current_x
    remaining_rooms = bedrooms - 1 + bathrooms - 1
    if remaining_rooms > 0:
        each_width = remaining_width / remaining_rooms
        
        # Additional bedrooms
        for i in range(bedrooms - 1):
            rooms.append(create_room(
                f"Bedroom {i+2}", "bedroom",
                current_x, min_y + padding,
                each_width - wall_thickness, bottom_row_height - wall_thickness
            ))
            current_x += each_width
        
        # Additional bathrooms
        for i in range(bathrooms - 1):
            rooms.append(create_room(
                f"Bath {i+2}", "bathroom",
                current_x, min_y + padding,
                each_width - wall_thickness, bottom_row_height - wall_thickness
            ))
            current_x += each_width
    
    logger.info(f"Created layout with {len(rooms)} rooms")
    
    return {
        "boundary": points,
        "rooms": rooms,
        "total_area": total_area,
        "style": style,
        "dimensions": [round(width, 1), round(height, 1)]
    }


def create_room(name: str, room_type: str, x: float, y: float, w: float, h: float) -> Dict:
    """Create room with furniture."""
    
    furniture_templates = {
        "living": [
            {"type": "sofa", "rx": 0.1, "ry": 0.6, "rw": 0.35, "rh": 0.12},
            {"type": "table", "rx": 0.2, "ry": 0.35, "rw": 0.18, "rh": 0.15},
            {"type": "tv", "rx": 0.55, "ry": 0.85, "rw": 0.25, "rh": 0.05},
            {"type": "chair", "rx": 0.55, "ry": 0.45, "rw": 0.1, "rh": 0.1},
            {"type": "chair", "rx": 0.7, "ry": 0.3, "rw": 0.1, "rh": 0.1},
        ],
        "bedroom": [
            {"type": "bed", "rx": 0.15, "ry": 0.25, "rw": 0.5, "rh": 0.55},
            {"type": "nightstand", "rx": 0.7, "ry": 0.35, "rw": 0.12, "rh": 0.12},
            {"type": "wardrobe", "rx": 0.15, "ry": 0.85, "rw": 0.4, "rh": 0.1},
        ],
        "kitchen": [
            {"type": "counter", "rx": 0.05, "ry": 0.7, "rw": 0.9, "rh": 0.12},
            {"type": "counter", "rx": 0.05, "ry": 0.05, "rw": 0.12, "rh": 0.6},
            {"type": "island", "rx": 0.35, "ry": 0.35, "rw": 0.35, "rh": 0.2},
            {"type": "sink", "rx": 0.3, "ry": 0.72, "rw": 0.12, "rh": 0.08},
            {"type": "stove", "rx": 0.5, "ry": 0.72, "rw": 0.12, "rh": 0.08},
        ],
        "dining": [
            {"type": "table", "rx": 0.2, "ry": 0.25, "rw": 0.6, "rh": 0.5},
            {"type": "chair", "rx": 0.1, "ry": 0.4, "rw": 0.08, "rh": 0.08},
            {"type": "chair", "rx": 0.82, "ry": 0.4, "rw": 0.08, "rh": 0.08},
            {"type": "chair", "rx": 0.35, "ry": 0.12, "rw": 0.08, "rh": 0.08},
            {"type": "chair", "rx": 0.55, "ry": 0.12, "rw": 0.08, "rh": 0.08},
            {"type": "chair", "rx": 0.35, "ry": 0.78, "rw": 0.08, "rh": 0.08},
            {"type": "chair", "rx": 0.55, "ry": 0.78, "rw": 0.08, "rh": 0.08},
        ],
        "bathroom": [
            {"type": "toilet", "rx": 0.1, "ry": 0.1, "rw": 0.25, "rh": 0.3},
            {"type": "sink", "rx": 0.45, "ry": 0.1, "rw": 0.35, "rh": 0.22},
            {"type": "shower", "rx": 0.1, "ry": 0.55, "rw": 0.4, "rh": 0.4},
        ],
    }
    
    furniture = []
    for tmpl in furniture_templates.get(room_type, []):
        furniture.append({
            "type": tmpl["type"],
            "x": round(x + tmpl["rx"] * w, 2),
            "y": round(y + tmpl["ry"] * h, 2),
            "w": round(tmpl["rw"] * w, 2),
            "h": round(tmpl["rh"] * h, 2)
        })
    
    return {
        "name": name,
        "type": room_type,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "furniture": furniture
    }


def render_floorplan(design: Dict, output_dir: str):
    """Render professional floor plan."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse, Arc
    from matplotlib.path import Path
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor='white', dpi=150)
    ax.set_facecolor('white')
    
    boundary = design.get("boundary", [])
    rooms = design.get("rooms", [])
    
    min_x = min(p[0] for p in boundary)
    min_y = min(p[1] for p in boundary)
    max_x = max(p[0] for p in boundary)
    max_y = max(p[1] for p in boundary)
    
    # Fill boundary
    poly = Polygon(boundary, closed=True, 
                   facecolor='#FAFAFA', edgecolor='black', linewidth=5, zorder=1)
    ax.add_patch(poly)
    
    # Draw rooms
    room_colors = {
        "bedroom": "#FFFFFF",
        "bathroom": "#E8F4F8",
        "living": "#FFFFFF",
        "dining": "#FFFAF0",
        "kitchen": "#FFF8F0",
    }
    
    for room in rooms:
        x, y = room["x"], room["y"]
        w, h = room["width"], room["height"]
        rtype = room.get("type", "room")
        name = room.get("name", "")
        
        # Room box
        rect = Rectangle((x, y), w, h, linewidth=2, 
                         edgecolor='black', facecolor=room_colors.get(rtype, "#FFFFFF"),
                         zorder=5)
        ax.add_patch(rect)
        
        # Furniture
        for furn in room.get("furniture", []):
            draw_furniture(ax, furn)
        
        # Door
        draw_door(ax, x + w * 0.35, y, min(0.9, w * 0.2))
        
        # Label
        for i, line in enumerate(name.split('\n')):
            offset = (len(name.split('\n')) - 1) / 2 - i
            ax.text(x + w/2, y + h/2 + offset * h * 0.1, line.upper(),
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='#333', zorder=15)
    
    # Boundary on top
    bx = [p[0] for p in boundary] + [boundary[0][0]]
    by = [p[1] for p in boundary] + [boundary[0][1]]
    ax.plot(bx, by, color='black', linewidth=5, solid_capstyle='round', zorder=20)
    
    # Dimensions
    dims = design.get("dimensions", [max_x - min_x, max_y - min_y])
    ax.annotate('', xy=(max_x, min_y - 1.5), xytext=(min_x, min_y - 1.5),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1))
    ax.text((min_x + max_x)/2, min_y - 2.2, f'{dims[0]:.1f}m', ha='center', fontsize=9, color='#555')
    
    ax.annotate('', xy=(max_x + 1.5, max_y), xytext=(max_x + 1.5, min_y),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1))
    ax.text(max_x + 2.2, (min_y + max_y)/2, f'{dims[1]:.1f}m', ha='left', va='center', 
            fontsize=9, color='#555', rotation=90)
    
    # Limits
    padding = 3
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.02, f'{design.get("style", "Modern").title()} Style | Total Area: {design.get("total_area", 0):.0f} sqm',
             ha='center', fontsize=11, color='#555')
    
    # North arrow
    ax.annotate('', xy=(0.93, 0.95), xycoords='axes fraction',
                xytext=(0.93, 0.88), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.93, 0.85, 'N', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "floorplan.png"), 
                dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.savefig(os.path.join(output_dir, "floorplan.jpg"), 
                dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.2, format='jpeg')
    plt.close()
    
    logger.info("Floor plan rendered (PNG and JPG)")


def draw_furniture(ax, furn: Dict):
    """Draw furniture."""
    from matplotlib.patches import Rectangle, Circle, Ellipse
    
    x, y = furn["x"], furn["y"]
    w, h = furn["w"], furn["h"]
    ftype = furn["type"]
    
    if ftype == "bed":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#888', fc='#E8E8E8', zorder=8))
        # Pillows
        pw, ph = w * 0.4, h * 0.1
        ax.add_patch(Rectangle((x + w*0.05, y + h - ph - h*0.05), pw, ph, lw=0.3, ec='#888', fc='#DDD', zorder=9))
        ax.add_patch(Rectangle((x + w*0.55, y + h - ph - h*0.05), pw, ph, lw=0.3, ec='#888', fc='#DDD', zorder=9))
    
    elif ftype == "sofa":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#888', fc='#E0E0E0', zorder=8))
        ax.add_patch(Rectangle((x, y + h*0.75), w, h*0.25, lw=0.3, ec='#888', fc='#D0D0D0', zorder=9))
    
    elif ftype == "table":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#888', fc='#E8E8E8', zorder=8))
    
    elif ftype == "chair":
        ax.add_patch(Circle((x + w/2, y + h/2), min(w, h)/2, lw=0.4, ec='#888', fc='#E0E0E0', zorder=8))
    
    elif ftype == "tv":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#333', fc='#333', zorder=8))
    
    elif ftype in ["counter", "island"]:
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#888', fc='#D8D8D8', zorder=8))
    
    elif ftype == "nightstand":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.3, ec='#888', fc='#E0E0E0', zorder=8))
    
    elif ftype == "wardrobe":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#666', fc='#D8D8D8', zorder=8))
        ax.plot([x + w/2, x + w/2], [y, y + h], color='#888', lw=0.3, zorder=9)
    
    elif ftype == "toilet":
        ax.add_patch(Ellipse((x + w/2, y + h*0.4), w*0.7, h*0.6, lw=0.4, ec='#888', fc='white', zorder=8))
        ax.add_patch(Rectangle((x + w*0.15, y + h*0.7), w*0.7, h*0.25, lw=0.3, ec='#888', fc='#F5F5F5', zorder=8))
    
    elif ftype == "sink":
        ax.add_patch(Ellipse((x + w/2, y + h/2), w*0.8, h*0.7, lw=0.4, ec='#888', fc='#F0F0F0', zorder=8))
    
    elif ftype == "stove":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.4, ec='#888', fc='#E0E0E0', zorder=8))
        # Burners
        r = min(w, h) * 0.15
        for i in range(4):
            bx = x + w * (0.25 + (i % 2) * 0.5)
            by = y + h * (0.3 + (i // 2) * 0.4)
            ax.add_patch(Circle((bx, by), r, lw=0.2, ec='#666', fc='#CCC', zorder=9))
    
    elif ftype == "shower":
        ax.add_patch(Rectangle((x, y), w, h, lw=0.5, ec='#888', fc='#E8F4F8', ls='--', zorder=8))


def draw_door(ax, x: float, y: float, width: float):
    """Draw door symbol."""
    from matplotlib.patches import Rectangle, Arc
    
    ax.add_patch(Rectangle((x, y - 0.02), width, 0.04, fc='white', ec='white', lw=0, zorder=12))
    arc = Arc((x, y), width * 1.5, width * 1.5, angle=0, theta1=0, theta2=90,
             lw=0.5, ec='#666', zorder=10)
    ax.add_patch(arc)
    ax.plot([x, x + width * 0.75], [y, y + width * 0.45], color='#666', lw=0.5, zorder=10)


def generate_dxf_output(design: Dict, output_dir: str, job_id: str = "sample"):
    """
    Generate DXF CAD file from design data.
    Creates a professional 2D floor plan in DXF format.
    """
    import ezdxf
    from ezdxf import units
    
    logger.info("Generating DXF output", job_id=job_id)
    
    # Create DXF document
    doc = ezdxf.new('R2013')
    doc.header['$INSUNITS'] = units.M
    doc.header['$MEASUREMENT'] = 1  # Metric
    
    # Create layers
    layers = {
        "A-WALL": {"color": 7},      # White - Walls
        "A-DOOR": {"color": 4},      # Cyan - Doors
        "A-GLAZ": {"color": 5},      # Blue - Windows
        "A-DIM": {"color": 3},       # Green - Dimensions
        "A-ANNO": {"color": 2},      # Yellow - Annotations
        "A-FURN": {"color": 6},      # Magenta - Furniture
        "A-BOUNDARY": {"color": 1},  # Red - Boundary
    }
    
    for layer_name, config in layers.items():
        doc.layers.add(layer_name, color=config["color"])
    
    msp = doc.modelspace()
    
    boundary = design.get("boundary", [])
    rooms = design.get("rooms", [])
    
    # Draw boundary
    if boundary:
        points = [(p[0], p[1]) for p in boundary]
        msp.add_lwpolyline(points, dxfattribs={'layer': 'A-BOUNDARY', 'closed': True})
    
    # Draw rooms
    for room in rooms:
        x = room.get('x', 0)
        y = room.get('y', 0)
        w = room.get('width', 3)
        h = room.get('height', 3)
        name = room.get('name', 'Room')
        area = room.get('width', 3) * room.get('height', 3)
        
        # Room walls (outer rectangle)
        room_points = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        msp.add_lwpolyline(room_points, dxfattribs={'layer': 'A-WALL', 'closed': True})
        
        # Room label
        center_x = x + w / 2
        center_y = y + h / 2
        label_text = name.replace('\n', ' ') + f"\\P{area:.1f} sqm"
        msp.add_mtext(
            label_text,
            dxfattribs={
                'layer': 'A-ANNO',
                'insert': (center_x, center_y),
                'char_height': 0.2,
                'attachment_point': 5  # Center
            }
        )
        
        # Door (simplified representation)
        door_x = x + w * 0.35
        door_y = y
        door_width = min(0.9, w * 0.2)
        msp.add_line((door_x, door_y), (door_x + door_width, door_y), 
                     dxfattribs={'layer': 'A-DOOR'})
        msp.add_arc(
            center=(door_x, door_y),
            radius=door_width,
            start_angle=0,
            end_angle=90,
            dxfattribs={'layer': 'A-DOOR'}
        )
        
        # Draw furniture
        for furn in room.get('furniture', []):
            fx, fy = furn.get('x', 0), furn.get('y', 0)
            fw, fh = furn.get('w', 0.5), furn.get('h', 0.5)
            furn_points = [
                (fx, fy),
                (fx + fw, fy),
                (fx + fw, fy + fh),
                (fx, fy + fh)
            ]
            msp.add_lwpolyline(furn_points, dxfattribs={'layer': 'A-FURN', 'closed': True})
    
    # Add dimensions
    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Width dimension (horizontal)
        msp.add_linear_dim(
            base=(min_x, min_y - 1),
            p1=(min_x, min_y),
            p2=(max_x, min_y),
            override={'layer': 'A-DIM'}
        )
        
        # Height dimension (vertical)
        msp.add_linear_dim(
            base=(max_x + 1, min_y),
            p1=(max_x, min_y),
            p2=(max_x, max_y),
            angle=90,
            override={'layer': 'A-DIM'}
        )
    
    # Title block
    total_area = design.get('total_area', 0)
    style = design.get('style', 'Modern')
    title_text = f"AutoArchitect AI - Floor Plan\\PJob: {job_id[:8]}...\\PArea: {total_area:.1f} sqm\\PStyle: {style}"
    
    if boundary:
        title_x = max(xs) + 2
        title_y = min(ys) - 3
    else:
        title_x = 15
        title_y = -3
    
    msp.add_mtext(
        title_text,
        dxfattribs={
            'layer': 'A-ANNO',
            'insert': (title_x, title_y),
            'char_height': 0.15
        }
    )
    
    # Save DXF file
    dxf_path = os.path.join(output_dir, "floor_plan.dxf")
    doc.saveas(dxf_path)
    
    logger.info("DXF file generated", path=dxf_path)
    return dxf_path
