"""
CAD Generation Service
======================
Generates DXF floor plans from design data.
"""

import ezdxf
from ezdxf import units
from ezdxf.addons import text2mtext
import os
import json
from typing import Dict, List, Tuple
import structlog

from app.core.config import settings
from app.api.routes.jobs import update_job_status, JobStage

logger = structlog.get_logger()


# Standard CAD layer configuration
LAYERS = {
    "A-WALL": {"color": 7, "linetype": "CONTINUOUS"},      # White - Walls
    "A-WALL-FIRE": {"color": 1, "linetype": "CONTINUOUS"}, # Red - Fire walls
    "A-DOOR": {"color": 4, "linetype": "CONTINUOUS"},      # Cyan - Doors
    "A-GLAZ": {"color": 5, "linetype": "CONTINUOUS"},      # Blue - Windows
    "A-DIM": {"color": 3, "linetype": "CONTINUOUS"},       # Green - Dimensions
    "A-ANNO": {"color": 2, "linetype": "CONTINUOUS"},      # Yellow - Annotations
    "A-FLOR": {"color": 8, "linetype": "CONTINUOUS"},      # Gray - Floor patterns
    "A-FURN": {"color": 6, "linetype": "CONTINUOUS"},      # Magenta - Furniture
}

# Wall thickness in meters
WALL_THICKNESS = 0.2

# Standard door dimensions
DOOR_SINGLE_WIDTH = 0.9
DOOR_DOUBLE_WIDTH = 1.2
DOOR_HEIGHT = 2.1

# Standard window dimensions
WINDOW_SILL_HEIGHT = 0.9
WINDOW_HEIGHT = 1.2


def generate_cad_files(job_id: str):
    """
    Main CAD generation function.
    Creates DXF, PDF, and PNG outputs.
    """
    logger.info("Starting CAD generation", job_id=job_id)
    
    try:
        update_job_status(job_id, JobStage.CREATING_CAD, 75, "Creating CAD drawings...")
        
        # Load design data
        design_file = os.path.join(settings.OUTPUT_DIR, job_id, "design.json")
        with open(design_file, 'r') as f:
            design = json.load(f)
        
        # Create DXF document
        doc = create_dxf_document()
        msp = doc.modelspace()
        
        # Draw boundary
        draw_boundary(msp, design.get("boundary", []))
        
        # Draw rooms
        for room in design.get("rooms", []):
            draw_room(msp, room)
        
        # Add dimensions
        add_dimensions(msp, design)
        
        # Add title block
        add_title_block(msp, job_id, design)
        
        # Save DXF
        output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
        dxf_path = os.path.join(output_dir, "floor_plan.dxf")
        doc.saveas(dxf_path)
        
        # Generate PNG preview
        create_png_preview(design, output_dir)
        
        update_job_status(job_id, JobStage.CREATING_CAD, 85, "CAD files generated")
        
        logger.info("CAD generation complete", job_id=job_id)
        
        return dxf_path
        
    except Exception as e:
        logger.error("CAD generation failed", job_id=job_id, error=str(e))
        update_job_status(job_id, JobStage.FAILED, 0, f"CAD failed: {str(e)}")
        raise


def create_dxf_document() -> ezdxf.document.Drawing:
    """Create a new DXF document with standard setup."""
    doc = ezdxf.new('R2013')
    
    # Set units to meters
    doc.header['$INSUNITS'] = units.M
    doc.header['$MEASUREMENT'] = 1  # Metric
    
    # Create standard layers
    for layer_name, config in LAYERS.items():
        doc.layers.add(
            layer_name,
            color=config["color"],
            linetype=config["linetype"]
        )
    
    # Setup dimension style
    doc.dimstyles.new('ARCH', dxfattribs={
        'dimtxt': 0.18,      # Text height
        'dimasz': 0.18,      # Arrow size
        'dimexe': 0.1,       # Extension line extension
        'dimexo': 0.05,      # Extension line offset
        'dimtad': 1,         # Text above dimension line
        'dimgap': 0.05,      # Gap around dimension text
    })
    
    return doc


def draw_boundary(msp, boundary: List[List[float]]):
    """Draw plot boundary as closed polyline."""
    if not boundary:
        return
    
    points = [(p[0], p[1]) for p in boundary]
    msp.add_lwpolyline(
        points,
        dxfattribs={
            'layer': 'A-WALL',
            'closed': True
        }
    )


def draw_room(msp, room: Dict):
    """Draw a single room with walls, doors, and windows."""
    x = room.get('x', 0)
    y = room.get('y', 0)
    w = room.get('width', 3)
    h = room.get('height', 3)
    
    # Draw walls (double lines for thickness)
    draw_walls(msp, x, y, w, h)
    
    # Draw doors
    for door in room.get('doors', []):
        draw_door(msp, x, y, w, h, door)
    
    # Draw windows
    for window in room.get('windows', []):
        draw_window(msp, x, y, w, h, window)
    
    # Add room label
    center_x = x + w / 2
    center_y = y + h / 2
    area = room.get('area', w * h)
    
    msp.add_mtext(
        f"{room.get('name', 'Room')}\n{area:.1f} sqm",
        dxfattribs={
            'layer': 'A-ANNO',
            'insert': (center_x, center_y),
            'char_height': 0.2,
            'attachment_point': 5  # Center
        }
    )


def draw_walls(msp, x: float, y: float, w: float, h: float):
    """Draw room walls with thickness."""
    # Outer wall
    outer = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]
    msp.add_lwpolyline(outer, dxfattribs={'layer': 'A-WALL', 'closed': True})
    
    # Inner wall (offset by wall thickness)
    t = WALL_THICKNESS / 2
    inner = [
        (x + t, y + t),
        (x + w - t, y + t),
        (x + w - t, y + h - t),
        (x + t, y + h - t)
    ]
    msp.add_lwpolyline(inner, dxfattribs={'layer': 'A-WALL', 'closed': True})


def draw_door(msp, room_x: float, room_y: float, room_w: float, room_h: float, door: Dict):
    """Draw door with swing arc."""
    wall = door.get('wall', 'south')
    offset = door.get('offset', 1.0)
    width = door.get('width', DOOR_SINGLE_WIDTH)
    
    # Calculate door position based on wall
    if wall == 'south':
        x1 = room_x + offset
        y1 = room_y
        x2 = x1 + width
        y2 = y1
    elif wall == 'north':
        x1 = room_x + offset
        y1 = room_y + room_h
        x2 = x1 + width
        y2 = y1
    elif wall == 'west':
        x1 = room_x
        y1 = room_y + offset
        x2 = x1
        y2 = y1 + width
    else:  # east
        x1 = room_x + room_w
        y1 = room_y + offset
        x2 = x1
        y2 = y1 + width
    
    # Draw door opening (gap in wall)
    msp.add_line((x1, y1), (x2, y2), dxfattribs={'layer': 'A-DOOR'})
    
    # Draw door swing arc
    if wall in ['south', 'north']:
        msp.add_arc(
            center=(x1, y1),
            radius=width,
            start_angle=0 if wall == 'south' else 180,
            end_angle=90 if wall == 'south' else 270,
            dxfattribs={'layer': 'A-DOOR'}
        )


def draw_window(msp, room_x: float, room_y: float, room_w: float, room_h: float, window: Dict):
    """Draw window symbol."""
    wall = window.get('wall', 'south')
    offset = window.get('offset', 1.0)
    width = window.get('width', 1.2)
    
    # Calculate window position
    if wall == 'south':
        x1, y1 = room_x + offset, room_y
        x2, y2 = x1 + width, y1
    elif wall == 'north':
        x1, y1 = room_x + offset, room_y + room_h
        x2, y2 = x1 + width, y1
    elif wall == 'west':
        x1, y1 = room_x, room_y + offset
        x2, y2 = x1, y1 + width
    else:
        x1, y1 = room_x + room_w, room_y + offset
        x2, y2 = x1, y1 + width
    
    # Draw window symbol (three parallel lines)
    msp.add_line((x1, y1), (x2, y2), dxfattribs={'layer': 'A-GLAZ'})


def add_dimensions(msp, design: Dict):
    """Add dimension annotations to the drawing."""
    # Overall dimensions
    boundary = design.get('boundary', [])
    if len(boundary) >= 4:
        xs = [p[0] for p in boundary]
        ys = [p[1] for p in boundary]
        
        # Width dimension
        msp.add_linear_dim(
            base=(min(xs), min(ys) - 1),
            p1=(min(xs), min(ys)),
            p2=(max(xs), min(ys)),
            dimstyle='ARCH',
            override={'layer': 'A-DIM'}
        )
        
        # Height dimension
        msp.add_linear_dim(
            base=(min(xs) - 1, min(ys)),
            p1=(min(xs), min(ys)),
            p2=(min(xs), max(ys)),
            angle=90,
            dimstyle='ARCH',
            override={'layer': 'A-DIM'}
        )


def add_title_block(msp, job_id: str, design: Dict):
    """Add title block with project information."""
    # Title block position (bottom-right of drawing)
    boundary = design.get('boundary', [(0, 0), (20, 0)])
    max_x = max(p[0] for p in boundary) + 2
    min_y = min(p[1] for p in boundary) - 5
    
    # Title text
    msp.add_mtext(
        f"AutoArchitect AI - Floor Plan\nJob ID: {job_id[:8]}...\nTotal Area: {design.get('total_area', 0):.1f} sqm\nStyle: {design.get('style', 'Modern')}",
        dxfattribs={
            'layer': 'A-ANNO',
            'insert': (max_x, min_y),
            'char_height': 0.15
        }
    )


def create_png_preview(design: Dict, output_dir: str):
    """Create PNG preview of the floor plan."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Arc
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw rooms
    for room in design.get('rooms', []):
        x = room.get('x', 0)
        y = room.get('y', 0)
        w = room.get('width', 3)
        h = room.get('height', 3)
        
        rect = Rectangle((x, y), w, h, fill=True, facecolor='#f0f0f0', 
                         edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Room label
        ax.text(x + w/2, y + h/2, f"{room.get('name', 'Room')}\n{room.get('area', 0):.1f} sqm",
               ha='center', va='center', fontsize=8)
    
    # Draw boundary
    boundary = design.get('boundary', [])
    if boundary:
        pts = boundary + [boundary[0]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, 'b-', linewidth=3)
    
    ax.set_aspect('equal')
    ax.set_title(f"Floor Plan - {design.get('total_area', 0):.1f} sqm")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'floor_plan.png'), dpi=150)
    plt.close()
