"""
Sample Processing Script for AutoArchitect AI
==============================================
Processes the sample input.png and generates:
- floorplan.jpg (2D floor plan image)
- floor_plan.dxf (CAD file for AutoCAD/etc)

Run: python process_sample.py
"""

import os
import sys
import json
import math
import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE = os.path.join(SAMPLE_DIR, "input.png")


def process_image(file_path: str) -> dict:
    """
    Process image to extract plot boundary using OpenCV.
    """
    print(f"[1/5] Processing input image: {file_path}")
    
    img = cv2.imread(file_path)
    if img is None:
        print("    Warning: Could not read image, using default boundary")
        return create_default_boundary()
    
    original_height, original_width = img.shape[:2]
    print(f"    Image size: {original_width}x{original_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine
    combined = cv2.bitwise_or(thresh, edges)
    combined = cv2.dilate(combined, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("    Warning: No contours found, using default boundary")
        return create_default_boundary()
    
    # Largest contour
    main_contour = max(contours, key=cv2.contourArea)
    
    # Simplify
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
    
    if len(points) < 3:
        print("    Warning: Too few points, using default boundary")
        return create_default_boundary()
    
    print(f"    Detected {len(points)} vertices")
    
    # Calculate area
    area_pixels = cv2.contourArea(approx)
    
    # Scale to ~150 sqm
    target_area_sqm = 150
    scale = math.sqrt(target_area_sqm / area_pixels) if area_pixels > 0 else 0.01
    
    # Normalize and scale points
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    
    scaled_points = []
    for px, py in points:
        x = (px - min_x) * scale
        y = (py - min_y) * scale
        scaled_points.append((round(x, 2), round(y, 2)))
    
    # Calculate dimensions
    max_x = max(p[0] for p in scaled_points)
    max_y = max(p[1] for p in scaled_points)
    
    # Calculate area using shoelace formula
    n = len(scaled_points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += scaled_points[i][0] * scaled_points[j][1]
        area -= scaled_points[j][0] * scaled_points[i][1]
    area = abs(area) / 2
    
    boundary_data = {
        "points": scaled_points,
        "area_sqm": round(area, 1),
        "dimensions": [round(max_x, 1), round(max_y, 1)],
    }
    
    print(f"    Area: {boundary_data['area_sqm']} sqm")
    print(f"    Dimensions: {boundary_data['dimensions'][0]}m x {boundary_data['dimensions'][1]}m")
    
    return boundary_data


def create_default_boundary() -> dict:
    """Create a default rectangular boundary."""
    return {
        "points": [(0, 0), (15, 0), (15, 10), (0, 10)],
        "area_sqm": 150,
        "dimensions": [15, 10],
    }


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray casting point-in-polygon test."""
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


def find_max_rect_at_point(x: float, y: float, polygon: list, 
                           max_w: float, max_h: float, step: float = 0.2):
    """Find maximum rectangle at point inside polygon."""
    if not point_in_polygon(x, y, polygon):
        return 0, 0
    
    best_w, best_h = 0, 0
    
    for w in [i * step for i in range(1, int(max_w / step) + 1)]:
        for h in [i * step for i in range(1, int(max_h / step) + 1)]:
            corners = [(x + w, y), (x + w, y + h), (x, y + h)]
            if all(point_in_polygon(cx, cy, polygon) for cx, cy in corners):
                if w * h > best_w * best_h:
                    best_w, best_h = w, h
    
    return best_w, best_h


def create_design(boundary: dict, requirements: dict) -> dict:
    """Create floor plan design from boundary."""
    print("\n[2/5] Generating floor plan design...")
    
    points = boundary.get("points", [(0, 0), (15, 0), (15, 10), (0, 10)])
    polygon = [(p[0], p[1]) for p in points]
    
    min_x = min(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_x = max(p[0] for p in polygon)
    max_y = max(p[1] for p in polygon)
    
    width = max_x - min_x
    height = max_y - min_y
    
    bedrooms = requirements.get("bedrooms", 3)
    bathrooms = requirements.get("bathrooms", 2)
    style = requirements.get("style", "modern")
    
    rooms = []
    occupied = []
    
    # Room specifications
    room_specs = [
        ("Living Room", "living", 20, width * 0.35, height * 0.4),
        ("Kitchen", "kitchen", 10, width * 0.25, height * 0.3),
        ("Dining", "dining", 12, width * 0.2, height * 0.25),
        ("Master\nBedroom", "bedroom", 15, width * 0.25, height * 0.35),
    ]
    
    for i in range(1, bedrooms):
        room_specs.append((f"Bedroom {i+1}", "bedroom", 10, width * 0.2, height * 0.25))
    
    for i in range(bathrooms):
        name = "Master\nBath" if i == 0 else f"Bath {i+1}"
        room_specs.append((name, "bathroom", 4, width * 0.12, height * 0.15))
    
    # Grid scan for placement
    grid_step = 0.5
    placement_points = []
    
    for gx in range(int((max_x - min_x) / grid_step)):
        for gy in range(int((max_y - min_y) / grid_step)):
            px = min_x + gx * grid_step + grid_step
            py = min_y + gy * grid_step + grid_step
            if point_in_polygon(px, py, polygon):
                placement_points.append((px, py))
    
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    placement_points.sort(key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
    
    # Place rooms
    for name, room_type, min_area, pref_w, pref_h in room_specs:
        placed = False
        
        for px, py in placement_points:
            if placed:
                break
            
            overlaps = False
            for ox, oy, ow, oh in occupied:
                if not (px + pref_w < ox or px > ox + ow or py + pref_h < oy or py > oy + oh):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            rw, rh = find_max_rect_at_point(px, py, polygon, pref_w * 1.5, pref_h * 1.5)
            
            if rw * rh < min_area:
                continue
            
            final_w = min(pref_w, rw * 0.95)
            final_h = min(pref_h, rh * 0.95)
            
            if final_w * final_h >= min_area * 0.7:
                room = {
                    "name": name,
                    "type": room_type,
                    "x": round(px, 2),
                    "y": round(py, 2),
                    "width": round(final_w, 2),
                    "height": round(final_h, 2),
                    "furniture": []
                }
                rooms.append(room)
                occupied.append((px, py, final_w, final_h))
                placed = True
                print(f"    Placed: {name.replace(chr(10), ' ')} ({final_w:.1f}m x {final_h:.1f}m)")
        
        if not placed:
            print(f"    Warning: Could not place {name.replace(chr(10), ' ')}")
    
    design = {
        "boundary": points,
        "rooms": rooms,
        "total_area": boundary.get("area_sqm", 150),
        "style": style,
        "dimensions": [width, height]
    }
    
    print(f"    Total rooms placed: {len(rooms)}")
    
    return design


def render_floorplan_jpg(design: dict, output_path: str):
    """Render floor plan to JPG image."""
    print(f"\n[3/5] Rendering floor plan to JPG...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse, Arc
    
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
    
    # Room colors
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
        
        # Room rectangle
        rect = Rectangle((x, y), w, h, linewidth=2, 
                         edgecolor='black', facecolor=room_colors.get(rtype, "#FFFFFF"),
                         zorder=5)
        ax.add_patch(rect)
        
        # Door arc
        door_x = x + w * 0.35
        door_y = y
        door_width = min(0.9, w * 0.2)
        ax.add_patch(Rectangle((door_x, door_y - 0.02), door_width, 0.04, 
                               fc='white', ec='white', lw=0, zorder=12))
        arc = Arc((door_x, door_y), door_width * 1.5, door_width * 1.5, 
                  angle=0, theta1=0, theta2=90, lw=0.5, ec='#666', zorder=10)
        ax.add_patch(arc)
        ax.plot([door_x, door_x + door_width * 0.75], [door_y, door_y + door_width * 0.45], 
                color='#666', lw=0.5, zorder=10)
        
        # Room label
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', 
                pad_inches=0.2, format='jpeg')
    plt.close()
    
    print(f"    Saved: {output_path}")


def generate_dxf(design: dict, output_path: str):
    """Generate DXF CAD file from design."""
    print(f"\n[4/5] Generating DXF CAD file...")
    
    import ezdxf
    from ezdxf import units
    
    # Create DXF document
    doc = ezdxf.new('R2013')
    doc.header['$INSUNITS'] = units.M
    doc.header['$MEASUREMENT'] = 1
    
    # Layers
    layers = {
        "A-WALL": {"color": 7},
        "A-DOOR": {"color": 4},
        "A-GLAZ": {"color": 5},
        "A-DIM": {"color": 3},
        "A-ANNO": {"color": 2},
        "A-FURN": {"color": 6},
        "A-BOUNDARY": {"color": 1},
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
        area = w * h
        
        # Room walls
        room_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
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
                'attachment_point': 5
            }
        )
        
        # Door
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
    
    # Dimensions
    if boundary:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        msp.add_linear_dim(
            base=(min_x, min_y - 1),
            p1=(min_x, min_y),
            p2=(max_x, min_y),
            override={'layer': 'A-DIM'}
        )
        
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
    title_text = f"AutoArchitect AI - Floor Plan\\PArea: {total_area:.1f} sqm\\PStyle: {style}"
    
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
    
    doc.saveas(output_path)
    print(f"    Saved: {output_path}")


def main():
    """Main processing function."""
    print("=" * 60)
    print("AutoArchitect AI - Sample Processing")
    print("=" * 60)
    
    # Check input file
    if not os.path.exists(INPUT_IMAGE):
        print(f"\nError: Input file not found: {INPUT_IMAGE}")
        print("Please ensure input.png exists in the sample folder.")
        return
    
    # Process input image
    boundary = process_image(INPUT_IMAGE)
    
    # Save boundary data
    boundary_path = os.path.join(SAMPLE_DIR, "boundary.json")
    with open(boundary_path, 'w') as f:
        json.dump(boundary, f, indent=2)
    print(f"\n    Saved: {boundary_path}")
    
    # Generate design
    requirements = {
        "bedrooms": 3,
        "bathrooms": 2,
        "style": "modern"
    }
    design = create_design(boundary, requirements)
    
    # Save design data
    design_path = os.path.join(SAMPLE_DIR, "design.json")
    with open(design_path, 'w') as f:
        json.dump(design, f, indent=2)
    print(f"    Saved: {design_path}")
    
    # Render JPG output
    jpg_output = os.path.join(SAMPLE_DIR, "output.jpg")
    render_floorplan_jpg(design, jpg_output)
    
    # Generate DXF output
    dxf_output = os.path.join(SAMPLE_DIR, "floor_plan.dxf")
    generate_dxf(design, dxf_output)
    
    # Summary
    print("\n[5/5] Processing complete!")
    print("=" * 60)
    print("Generated files:")
    print(f"  - {os.path.basename(jpg_output)} (2D floor plan image)")
    print(f"  - {os.path.basename(dxf_output)} (CAD file for AutoCAD)")
    print(f"  - boundary.json (extracted boundary data)")
    print(f"  - design.json (floor plan design data)")
    print("=" * 60)


if __name__ == "__main__":
    main()
