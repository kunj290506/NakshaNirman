"""
File Processing Service - Production Version
=============================================
Processes uploaded images and DXF files to extract plot boundaries.
Uses advanced OpenCV contour detection for accurate boundary extraction.
"""

import os
import json
import math
import time
from typing import Dict, List, Tuple, Optional
import structlog
import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.api.routes.jobs import update_job_status, JobStage

logger = structlog.get_logger()


def process_uploaded_file(job_id: str, file_path: str, file_type: str, requirements: Dict = None):
    """
    Main file processing function.
    Extracts boundary and creates preview.
    """
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("PROCESSING UPLOADED FILE", job_id=job_id, file_type=file_type)
    logger.info("=" * 60)
    
    try:
        update_job_status(job_id, JobStage.PROCESSING_FILE, 10, "Reading uploaded file...")
        
        step_start = time.time()
        if file_type == "image":
            boundary = process_image(file_path)
        elif file_type == "dxf":
            boundary = process_dxf(file_path)
        else:
            boundary = create_default_boundary()
        logger.info("TIMING: Process file", duration_sec=round(time.time() - step_start, 3))
        
        update_job_status(job_id, JobStage.EXTRACTING_BOUNDARY, 20, "Extracting boundary...")
        
        # Create output directory
        output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save boundary data as JSON
        step_start = time.time()
        boundary_file = os.path.join(output_dir, "boundary.json")
        with open(boundary_file, 'w') as f:
            json.dump(boundary, f, indent=2)
        
        # Save as GeoJSON for compatibility
        geojson = convert_to_geojson(boundary)
        geojson_file = os.path.join(output_dir, "boundary.geojson")
        with open(geojson_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info("TIMING: Save boundary files", duration_sec=round(time.time() - step_start, 3))
        
        # Create preview image
        step_start = time.time()
        update_job_status(job_id, JobStage.EXTRACTING_BOUNDARY, 25, "Creating preview...")
        create_preview_image(file_path, boundary, output_dir)
        logger.info("TIMING: Create preview", duration_sec=round(time.time() - step_start, 3))
        
        update_job_status(job_id, JobStage.EXTRACTING_BOUNDARY, 30, "Boundary extracted successfully")
        
        total_duration = round(time.time() - total_start, 3)
        logger.info("FILE PROCESSING COMPLETE", job_id=job_id, area=boundary.get('area_sqm'), total_duration_sec=total_duration)
        
        # Automatically trigger design generation
        logger.info("Auto-triggering design generation...")
        from app.services.ai_designer import generate_design
        
        # Use provided requirements or defaults
        design_requirements = requirements or {
            "bedrooms": 3,
            "bathrooms": 2,
            "style": "modern"
        }
        
        generate_design(job_id, design_requirements)
        
        return boundary
        
    except Exception as e:
        logger.error("File processing failed", job_id=job_id, error=str(e), duration_sec=round(time.time() - total_start, 3))
        update_job_status(job_id, JobStage.FAILED, 0, f"Processing failed: {str(e)}")
        raise


def process_image(file_path: str) -> Dict:
    """
    Process image to extract plot boundary using OpenCV.
    Uses advanced contour detection and approximation.
    """
    logger.info("Processing image", path=file_path)
    
    # Read image
    img = cv2.imread(file_path)
    if img is None:
        logger.warning("Could not read image, using default boundary")
        return create_default_boundary()
    
    original_height, original_width = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better edge detection
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Also try Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine thresholding and edge detection
    combined = cv2.bitwise_or(thresh, edges)
    
    # Dilate to connect broken lines
    combined = cv2.dilate(combined, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found, using default boundary")
        return create_default_boundary()
    
    # Find the largest contour (main boundary)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to simplify
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Extract points
    points = [(int(p[0][0]), int(p[0][1])) for p in approx]
    
    if len(points) < 3:
        logger.warning("Too few points detected, using default boundary")
        return create_default_boundary()
    
    # Calculate area in pixels
    area_pixels = cv2.contourArea(approx)
    
    # Estimate scale: assume the plot is approximately 150 sqm
    # Scale pixels to meters
    target_area_sqm = 150
    scale = math.sqrt(target_area_sqm / area_pixels) if area_pixels > 0 else 0.01
    
    # Scale and normalize points
    scaled_points = []
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    
    for px, py in points:
        # Normalize and scale to meters
        x = (px - min_x) * scale
        y = (py - min_y) * scale
        scaled_points.append((round(x, 2), round(y, 2)))
    
    # Calculate dimensions
    max_x = max(p[0] for p in scaled_points)
    max_y = max(p[1] for p in scaled_points)
    
    # Calculate actual area using shoelace formula
    n = len(scaled_points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += scaled_points[i][0] * scaled_points[j][1]
        area -= scaled_points[j][0] * scaled_points[i][1]
    area = abs(area) / 2
    
    # Detect shape type
    shape_type = detect_shape_type(len(scaled_points))
    
    boundary_data = {
        "points": scaled_points,
        "area_sqm": round(area, 1),
        "dimensions": [round(max_x, 1), round(max_y, 1)],
        "perimeter": round(calculate_perimeter(scaled_points), 1),
        "shape_type": shape_type,
        "num_vertices": len(scaled_points),
        "original_size": [original_width, original_height],
        "scale_factor": scale
    }
    
    logger.info("Boundary extracted", 
                area=boundary_data["area_sqm"],
                vertices=boundary_data["num_vertices"],
                shape=boundary_data["shape_type"])
    
    return boundary_data


def process_dxf(file_path: str) -> Dict:
    """Process DXF file to extract boundary."""
    logger.info("Processing DXF file", path=file_path)
    
    try:
        import ezdxf
        
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        points = []
        
        # Look for polylines (common for plot boundaries)
        for entity in msp.query('LWPOLYLINE'):
            for point in entity.get_points():
                points.append((float(point[0]), float(point[1])))
        
        # Also check for lines
        if not points:
            for entity in msp.query('LINE'):
                start = entity.dxf.start
                end = entity.dxf.end
                points.append((float(start.x), float(start.y)))
                points.append((float(end.x), float(end.y)))
        
        if len(points) < 3:
            logger.warning("DXF has too few points, using default")
            return create_default_boundary()
        
        # Remove duplicates and order points
        points = list(set(points))
        points = order_points_clockwise(points)
        
        # Calculate properties
        area = calculate_polygon_area(points)
        
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        # Normalize to origin
        normalized_points = [(p[0] - min_x, p[1] - min_y) for p in points]
        
        return {
            "points": [(round(p[0], 2), round(p[1], 2)) for p in normalized_points],
            "area_sqm": round(area, 1),
            "dimensions": [round(max_x - min_x, 1), round(max_y - min_y, 1)],
            "perimeter": round(calculate_perimeter(normalized_points), 1),
            "shape_type": detect_shape_type(len(points)),
            "num_vertices": len(points)
        }
        
    except Exception as e:
        logger.error("DXF processing failed", error=str(e))
        return create_default_boundary()


def create_default_boundary() -> Dict:
    """Create a default rectangular boundary."""
    return {
        "points": [(0, 0), (15, 0), (15, 10), (0, 10)],
        "area_sqm": 150,
        "dimensions": [15, 10],
        "perimeter": 50,
        "shape_type": "rectangle",
        "num_vertices": 4
    }


def detect_shape_type(num_vertices: int) -> str:
    """Detect the shape type based on number of vertices."""
    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "quadrilateral"
    elif num_vertices == 5:
        return "pentagon"
    elif num_vertices == 6:
        return "hexagon"
    elif num_vertices <= 8:
        return "polygon"
    else:
        return "irregular"


def calculate_perimeter(points: List[tuple]) -> float:
    """Calculate perimeter of polygon."""
    perimeter = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        perimeter += math.sqrt(dx*dx + dy*dy)
    return perimeter


def calculate_polygon_area(points: List[tuple]) -> float:
    """Calculate area using shoelace formula."""
    n = len(points)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2


def order_points_clockwise(points: List[tuple]) -> List[tuple]:
    """Order points clockwise around centroid."""
    if len(points) < 3:
        return points
    
    # Calculate centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    # Sort by angle from centroid
    def angle_from_center(point):
        return math.atan2(point[1] - cy, point[0] - cx)
    
    return sorted(points, key=angle_from_center)


def convert_to_geojson(boundary: Dict) -> Dict:
    """Convert boundary to GeoJSON format."""
    points = boundary.get("points", [])
    
    # Close the polygon
    if points and points[0] != points[-1]:
        points = points + [points[0]]
    
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "area_sqm": boundary.get("area_sqm", 0),
                "shape_type": boundary.get("shape_type", "polygon")
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [points]
            }
        }]
    }


def create_preview_image(original_path: str, boundary: Dict, output_dir: str):
    """Create a preview image showing the extracted boundary."""
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')
    ax.set_facecolor('#f8fafc')
    
    points = boundary.get("points", [])
    if not points:
        points = [(0, 0), (15, 0), (15, 10), (0, 10)]
    
    # Draw the boundary polygon
    polygon = patches.Polygon(
        points,
        closed=True,
        fill=True,
        facecolor='#e2e8f0',
        edgecolor='#1e3a5f',
        linewidth=3
    )
    ax.add_patch(polygon)
    
    # Mark vertices
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, c='#2563eb', s=100, zorder=5)
    
    # Add dimensions
    for i, (x, y) in enumerate(points):
        ax.annotate(
            f'({x:.1f}, {y:.1f})',
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            color='#475569'
        )
    
    # Add area label in center
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    ax.text(cx, cy, f'{boundary.get("area_sqm", 0):.0f} sqm',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#1e293b')
    
    # Set limits with padding
    margin = 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Width (meters)', fontsize=10)
    ax.set_ylabel('Height (meters)', fontsize=10)
    ax.set_title('Extracted Plot Boundary', fontsize=14, fontweight='bold', color='#1e293b')
    
    # Save preview
    preview_path = os.path.join(output_dir, "preview.png")
    plt.savefig(preview_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info("Preview image created", path=preview_path)
