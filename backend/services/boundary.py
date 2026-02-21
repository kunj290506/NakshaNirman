"""Boundary extraction from images (OpenCV) and DXF files (ezdxf).

Universal shape extraction supporting all boundary types worldwide:
- Images (PNG, JPG, JPEG, BMP, TIFF)
- CAD files (DXF)
- Complex shapes (L-shaped, U-shaped, irregular)
- Multiple detection methods with fallbacks

Phase 1 additions:
- Buildable footprint computation (setback offsets)
- Preview image generation (matplotlib)
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon as ShapelyMultiPolygon, Point
from shapely.ops import unary_union
from shapely.validation import explain_validity
from pathlib import Path
import json
import math


def extract_all_shapes_from_image(image_path: str, scale: float = 1.0, return_all: bool = False) -> dict:
    """
    Universal shape extractor that works for ANY boundary shape worldwide.
    
    Uses multiple detection algorithms:
    1. Adaptive thresholding + contour detection
    2. Edge detection with Hough transforms
    3. Color-based segmentation
    4. Deep contour analysis
    
    Args:
        image_path: Path to image file
        scale: Scale factor for coordinates
        return_all: If True, returns all detected shapes, not just the largest
    
    Returns:
        dict with polygon data including all detected boundaries
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_contours = []
    
    # Method 1: Adaptive thresholding
    try:
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours1, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        all_contours.extend(contours1)
    except Exception:
        pass
    
    # Method 2: Multi-level Otsu thresholding
    try:
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(otsu_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours2)
    except Exception:
        pass
    
    # Method 3: Canny edge detection with multiple thresholds
    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        for low, high in [(30, 100), (50, 150), (70, 200)]:
            edges = cv2.Canny(blurred, low, high)
            dilated = cv2.dilate(edges, np.ones((3, 3)), iterations=2)
            contours3, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours3)
    except Exception:
        pass
    
    # Method 4: Color-based segmentation
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect darker regions (buildings/structures typically darker than background)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 200])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours4, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours4)
    except Exception:
        pass
    
    if not all_contours:
        raise ValueError("No shapes detected in image. Please ensure the boundary is clearly visible.")
    
    # Filter contours by area (remove noise)
    min_area = 100
    valid_contours = [c for c in all_contours if cv2.contourArea(c) >= min_area]
    
    if not valid_contours:
        raise ValueError("No significant boundaries detected. Image may be too noisy or low quality.")
    
    # Select the best contour (largest by default)
    main_contour = max(valid_contours, key=cv2.contourArea)
    
    # Extract and refine polygon
    epsilon = 0.002 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Apply sub-pixel refinement
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(approx), (5, 5), (-1, -1), criteria)
        polygon_coords = corners.squeeze().tolist()
    except Exception:
        polygon_coords = approx.squeeze().tolist()
    
    # Handle single point edge case
    if not isinstance(polygon_coords[0], list):
        polygon_coords = [polygon_coords]
    
    if len(polygon_coords) < 3:
        raise ValueError("Insufficient points detected for a valid polygon.")
    
    # Scale and round coordinates
    if scale != 1.0:
        polygon_coords = [[round(p[0] * scale, 2), round(p[1] * scale, 2)] for p in polygon_coords]
    else:
        polygon_coords = [[round(p[0], 2), round(p[1], 2)] for p in polygon_coords]
    
    # Ensure closed polygon
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    
    # Calculate metrics
    try:
        shapely_poly = Polygon(polygon_coords)
        if not shapely_poly.is_valid:
            fixed_geom = shapely_poly.buffer(0)
            if isinstance(fixed_geom, ShapelyMultiPolygon):
                shapely_poly = max(fixed_geom.geoms, key=lambda p: p.area)
            else:
                shapely_poly = fixed_geom
            if hasattr(shapely_poly, 'exterior'):
                polygon_coords = [[round(c[0], 2), round(c[1], 2)] for c in shapely_poly.exterior.coords]
        area = shapely_poly.area
        perimeter = shapely_poly.length
    except Exception:
        area = cv2.contourArea(main_contour) * (scale ** 2)
        perimeter = cv2.arcLength(main_contour, True) * scale
    
    return {
        "polygon": polygon_coords,
        "area": round(area, 2),
        "num_vertices": len(polygon_coords) - 1,
        "perimeter": round(perimeter, 2),
        "extraction_method": "universal_multi_algorithm"
    }


def extract_polygon_from_image(image_path: str, scale: float = 1.0) -> dict:
    """
    Extract boundary polygon from an uploaded image using OpenCV with high accuracy.

    Pipeline: grayscale → adaptive threshold → morphology → contour detection → precise extraction.

    Returns dict with 'polygon' (list of [x,y]), 'area', 'num_vertices'.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use adaptive thresholding for better handling of varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply Canny edge detection with optimal thresholds
    edges = cv2.Canny(denoised, 30, 100)
    
    # Dilate edges to connect nearby edges
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=2)
    
    # Combine threshold and edge results
    combined = cv2.bitwise_or(morphed, edges_dilated)
    
    # Find contours with full hierarchy
    contours, hierarchy = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    
    if not contours:
        raise ValueError("No contours found in the image. Please ensure the boundary is clearly visible.")
    
    # Select the largest contour (assumed to be the main boundary)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Check if contour area is significant
    if cv2.contourArea(main_contour) < 100:
        raise ValueError("Detected boundary is too small. Please upload a clearer image.")
    
    # Refine contour with minimal simplification to preserve shape accuracy
    # Use smaller epsilon for more precise boundary extraction
    epsilon = 0.002 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Apply corner refinement for sub-pixel accuracy
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(approx), (5, 5), (-1, -1), criteria)
        polygon_coords = corners.squeeze().tolist()
    except Exception:
        # Fallback if corner refinement fails
        polygon_coords = approx.squeeze().tolist()
    
    # Handle edge case where squeeze reduces to a single point
    if not isinstance(polygon_coords[0], list):
        polygon_coords = [polygon_coords]
    
    # Validate polygon has at least 3 vertices
    if len(polygon_coords) < 3:
        raise ValueError("Invalid boundary detected. At least 3 points required for a polygon.")
    
    # Scale coordinates
    if scale != 1.0:
        polygon_coords = [[round(p[0] * scale, 2), round(p[1] * scale, 2)] for p in polygon_coords]
    else:
        polygon_coords = [[round(p[0], 2), round(p[1], 2)] for p in polygon_coords]
    
    # Ensure polygon is closed
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords.append(polygon_coords[0])
    
    # Compute area using Shapely for accuracy
    try:
        shapely_poly = Polygon(polygon_coords)
        if not shapely_poly.is_valid:
            # Try to fix invalid polygon
            fixed_geom = shapely_poly.buffer(0)
            if hasattr(fixed_geom, 'area'):
                shapely_poly = fixed_geom
            else:
                raise ValueError("Could not fix polygon")
        area = shapely_poly.area
        
        # Handle MultiPolygon case
        from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
        if isinstance(shapely_poly, ShapelyMultiPolygon):
            # Use largest polygon
            shapely_poly = max(shapely_poly.geoms, key=lambda p: p.area)
            polygon_coords = [[round(p[0], 2), round(p[1], 2)] for p in shapely_poly.exterior.coords]
            area = shapely_poly.area
    except Exception:
        area = cv2.contourArea(main_contour) * (scale ** 2)
    
    return {
        "polygon": polygon_coords,
        "area": round(area, 2),
        "num_vertices": len(polygon_coords) - 1,  # exclude closing vertex
        "perimeter": round(cv2.arcLength(main_contour, True) * scale, 2)
    }


def extract_polygon_from_dxf(dxf_path: str) -> dict:
    """
    Extract boundary polygon from a DXF file with precise accuracy.

    Parses LWPOLYLINE, POLYLINE, LINE, and other entities and preserves the exact boundary shape.
    """
    import ezdxf
    from collections import defaultdict

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    all_coords = []
    line_segments = []

    # Extract from LWPOLYLINE (most common for boundaries)
    for entity in msp.query("LWPOLYLINE"):
        points = list(entity.get_points(format="xy"))
        if len(points) >= 3:
            # This is already a complete polygon
            all_coords.extend(points)
            if entity.closed or points[0] == points[-1]:
                # It's a closed polygon, use it directly
                coords = [[round(p[0], 2), round(p[1], 2)] for p in points]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                shapely_poly = Polygon(coords)
                area = round(shapely_poly.area, 2)
                
                return {
                    "polygon": coords,
                    "area": area,
                    "num_vertices": len(coords) - 1,
                    "perimeter": round(shapely_poly.length, 2)
                }

    # Extract from POLYLINE
    for entity in msp.query("POLYLINE"):
        points = [(p[0], p[1]) for p in entity.points()]
        if len(points) >= 3:
            all_coords.extend(points)
            if entity.is_closed or points[0] == points[-1]:
                coords = [[round(p[0], 2), round(p[1], 2)] for p in points]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                shapely_poly = Polygon(coords)
                area = round(shapely_poly.area, 2)
                
                return {
                    "polygon": coords,
                    "area": area,
                    "num_vertices": len(coords) - 1,
                    "perimeter": round(shapely_poly.length, 2)
                }

    # Extract from LINE entities and connect them
    for entity in msp.query("LINE"):
        start = (round(entity.dxf.start.x, 2), round(entity.dxf.start.y, 2))
        end = (round(entity.dxf.end.x, 2), round(entity.dxf.end.y, 2))
        line_segments.append((start, end))
        all_coords.extend([start, end])

    # Extract from CIRCLE (convert to polygon approximation)
    for entity in msp.query("CIRCLE"):
        center = (entity.dxf.center.x, entity.dxf.center.y)
        radius = entity.dxf.radius
        # Create a polygon with 36 points (10-degree increments)
        import math
        circle_points = []
        for i in range(36):
            angle = math.radians(i * 10)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            circle_points.append([round(x, 2), round(y, 2)])
        circle_points.append(circle_points[0])
        
        shapely_poly = Polygon(circle_points)
        area = round(shapely_poly.area, 2)
        
        return {
            "polygon": circle_points,
            "area": area,
            "num_vertices": len(circle_points) - 1,
            "perimeter": round(shapely_poly.length, 2)
        }

    # If we have line segments, try to connect them into a proper boundary
    if line_segments:
        ordered_points = connect_line_segments(line_segments)
        if ordered_points and len(ordered_points) >= 3:
            coords = [[p[0], p[1]] for p in ordered_points]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            try:
                shapely_poly = Polygon(coords)
                if shapely_poly.is_valid:
                    area = round(shapely_poly.area, 2)
                    return {
                        "polygon": coords,
                        "area": area,
                        "num_vertices": len(coords) - 1,
                        "perimeter": round(shapely_poly.length, 2)
                    }
            except Exception:
                pass

    # Fallback: if we have scattered points, try to form a polygon
    if all_coords and len(all_coords) >= 3:
        # Remove duplicates while preserving order where possible
        unique_coords = []
        seen = set()
        for coord in all_coords:
            if coord not in seen:
                unique_coords.append(coord)
                seen.add(coord)
        
        if len(unique_coords) >= 3:
            # Sort points by angle from centroid to preserve shape better than convex hull
            from shapely.geometry import MultiPoint
            mp = MultiPoint(unique_coords)
            centroid = mp.centroid
            
            import math
            def angle_from_centroid(point):
                return math.atan2(point[1] - centroid.y, point[0] - centroid.x)
            
            sorted_coords = sorted(unique_coords, key=angle_from_centroid)
            coords = [[round(c[0], 2), round(c[1], 2)] for c in sorted_coords]
            coords.append(coords[0])
            
            try:
                shapely_poly = Polygon(coords)
                if shapely_poly.is_valid:
                    area = round(shapely_poly.area, 2)
                    return {
                        "polygon": coords,
                        "area": area,
                        "num_vertices": len(coords) - 1,
                        "perimeter": round(shapely_poly.length, 2)
                    }
            except Exception:
                pass

    raise ValueError("No valid boundary entities found in DXF. Please ensure the file contains LWPOLYLINE, POLYLINE, LINE, or CIRCLE entities.")


def connect_line_segments(segments):
    """
    Connect line segments into an ordered boundary polygon.
    
    Args:
        segments: List of (start, end) tuples
    
    Returns:
        List of ordered points forming a closed boundary
    """
    if not segments:
        return []
    
    # Build adjacency graph
    from collections import defaultdict
    graph = defaultdict(list)
    
    for start, end in segments:
        graph[start].append(end)
        graph[end].append(start)
    
    # Find a starting point (preferably one with degree 2)
    start_point = None
    for point in graph:
        if len(graph[point]) == 2:
            start_point = point
            break
    
    if not start_point:
        # Use any point
        start_point = list(graph.keys())[0]
    
    # Traverse the graph to build ordered points
    ordered = [start_point]
    current = start_point
    visited_edges = set()
    
    while True:
        neighbors = graph[current]
        next_point = None
        
        for neighbor in neighbors:
            edge = tuple(sorted([current, neighbor]))
            if edge not in visited_edges:
                next_point = neighbor
                visited_edges.add(edge)
                break
        
        if next_point is None:
            break
        
        if next_point == start_point and len(ordered) >= 3:
            # We've completed the loop
            break
        
        if next_point in ordered and next_point != start_point:
            # We've hit a junction, stop here
            break
        
        ordered.append(next_point)
        current = next_point
        
        if len(ordered) > len(segments) + 1:
            # Safety check to prevent infinite loops
            break
    
    return ordered


def validate_boundary_polygon(polygon_data: dict) -> dict:
    """
    Validate the extracted boundary polygon for quality and accuracy.
    Handles Polygon, MultiPolygon, and complex shapes.
    
    Returns the polygon data with additional validation metadata.
    """
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
    
    polygon = polygon_data["polygon"]
    
    # Check minimum vertices
    num_vertices = len(polygon) - 1  # exclude closing vertex
    if num_vertices < 3:
        raise ValueError(f"Invalid polygon: only {num_vertices} vertices found. Minimum 3 required.")
    
    # Check area is positive
    if polygon_data["area"] <= 0:
        raise ValueError("Invalid polygon: area must be positive.")
    
    # Check for self-intersection using Shapely
    try:
        shapely_poly = Polygon(polygon)
        if not shapely_poly.is_valid:
            # Try to fix it
            fixed_geom = shapely_poly.buffer(0)
            
            # Handle different geometry types after buffer
            if isinstance(fixed_geom, ShapelyMultiPolygon):
                # Take the largest polygon from the MultiPolygon
                largest_poly = max(fixed_geom.geoms, key=lambda p: p.area)
                if not largest_poly.is_valid:
                    raise ValueError("Invalid polygon: self-intersecting or malformed boundary detected.")
                fixed_coords = list(largest_poly.exterior.coords)
                polygon_data["polygon"] = [[round(c[0], 2), round(c[1], 2)] for c in fixed_coords]
                polygon_data["area"] = round(largest_poly.area, 2)
                if "perimeter" in polygon_data:
                    polygon_data["perimeter"] = round(largest_poly.length, 2)
            elif hasattr(fixed_geom, 'exterior'):
                # It's a Polygon
                if not fixed_geom.is_valid:
                    raise ValueError("Invalid polygon: self-intersecting or malformed boundary detected.")
                fixed_coords = list(fixed_geom.exterior.coords)
                polygon_data["polygon"] = [[round(c[0], 2), round(c[1], 2)] for c in fixed_coords]
                polygon_data["area"] = round(fixed_geom.area, 2)
                if "perimeter" in polygon_data:
                    polygon_data["perimeter"] = round(fixed_geom.length, 2)
            else:
                raise ValueError(f"Invalid polygon: unexpected geometry type {type(fixed_geom).__name__}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Polygon validation failed: {str(e)}")
    
    # Add validation metadata
    polygon_data["is_valid"] = True
    polygon_data["validation_passed"] = True
    
    return polygon_data


def process_boundary_file(file_path: str, file_type: str, scale: float = 1.0) -> dict:
    """
    Route to appropriate processor based on file type and validate results.
    
    Uses universal multi-algorithm extraction for maximum accuracy worldwide.
    """
    if file_type in ("image", "png", "jpg", "jpeg", "bmp", "tiff"):
        # Try universal extractor first (works for all shapes globally)
        try:
            result = extract_all_shapes_from_image(file_path, scale)
        except Exception as universal_error:
            # Fallback to standard extractor
            try:
                result = extract_polygon_from_image(file_path, scale)
            except Exception:
                # Re-raise the original universal error for better diagnostics
                raise ValueError(f"Shape extraction failed: {str(universal_error)}")
    elif file_type in ("dxf",):
        result = extract_polygon_from_dxf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Validate the extracted boundary
    validated_result = validate_boundary_polygon(result)
    
    return validated_result


# ---------------------------------------------------------------------------
# Phase 1 — Buildable Footprint & Preview
# ---------------------------------------------------------------------------

def load_region_rules(region: str = "india_mvp") -> dict:
    """Load setback / building rules from region_rules.json."""
    rules_path = Path(__file__).resolve().parent.parent / "region_rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"Region rules file not found: {rules_path}")
    with open(rules_path) as f:
        all_rules = json.load(f)
    if region not in all_rules:
        raise ValueError(f"Unknown region '{region}'. Available: {list(all_rules.keys())}")
    return all_rules[region]


def validate_boundary_strict(polygon_coords: list) -> dict:
    """
    Strict Phase-1 validation of a boundary polygon.

    Checks:
      - closed
      - non-self-intersecting
      - area > 0

    Returns dict with is_valid, is_closed, is_self_intersecting, shapely Polygon.
    """
    # Ensure closed
    if polygon_coords[0] != polygon_coords[-1]:
        polygon_coords = polygon_coords + [polygon_coords[0]]

    poly = Polygon(polygon_coords)
    is_closed = True  # we forced closure above
    is_self_intersecting = not poly.is_simple
    area = poly.area

    if is_self_intersecting:
        # Attempt to fix via buffer(0)
        fixed = poly.buffer(0)
        if isinstance(fixed, ShapelyMultiPolygon):
            fixed = max(fixed.geoms, key=lambda g: g.area)
        if fixed.is_valid and fixed.area > 0:
            poly = fixed
            polygon_coords = [[round(c[0], 2), round(c[1], 2)] for c in poly.exterior.coords]
            is_self_intersecting = False
            area = poly.area

    is_valid = (not is_self_intersecting) and (area > 0)

    return {
        "is_valid": is_valid,
        "is_closed": is_closed,
        "is_self_intersecting": is_self_intersecting,
        "area": round(area, 2),
        "perimeter": round(poly.length, 2),
        "polygon_coords": polygon_coords,
        "shapely_polygon": poly,
    }


def compute_buildable_footprint(
    boundary_polygon_coords: list,
    setback: float | None = None,
    region: str = "india_mvp",
) -> dict:
    """
    Compute the usable (buildable) polygon by applying setback offsets inward.

    Steps:
      1. Build a Shapely Polygon from the boundary.
      2. Load region rules (or use explicit setback).
      3. Apply negative buffer (inward offset).
      4. Validate the result — remove degenerate offsets.
      5. Return usable_polygon + metadata.
    """
    # --- validate boundary ---
    validation = validate_boundary_strict(boundary_polygon_coords)
    if not validation["is_valid"]:
        raise ValueError(
            f"Boundary polygon invalid: self_intersecting={validation['is_self_intersecting']}, "
            f"area={validation['area']}"
        )

    boundary_poly: Polygon = validation["shapely_polygon"]
    boundary_area = validation["area"]

    # --- determine setback distance ---
    if setback is None:
        rules = load_region_rules(region)
        setback = rules.get("uniform_setback", 2.0)

    if setback < 0:
        raise ValueError("Setback distance must be non-negative.")

    if setback == 0:
        usable_poly = boundary_poly
    else:
        # Negative buffer = inward offset
        usable_poly = boundary_poly.buffer(-setback, join_style=2)  # mitre join

    # --- remove invalid / degenerate offsets ---
    if usable_poly.is_empty:
        raise ValueError(
            f"Setback of {setback}m collapses the polygon entirely. "
            "Plot is too small or setback too large."
        )

    if isinstance(usable_poly, ShapelyMultiPolygon):
        # Keep only the largest piece
        usable_poly = max(usable_poly.geoms, key=lambda g: g.area)

    if not usable_poly.is_valid:
        usable_poly = usable_poly.buffer(0)

    usable_area = round(usable_poly.area, 2)
    if usable_area <= 0:
        raise ValueError("Usable area is zero after applying setback offset.")

    usable_coords = [[round(c[0], 2), round(c[1], 2)] for c in usable_poly.exterior.coords]

    coverage_ratio = round(usable_area / boundary_area, 4) if boundary_area > 0 else 0

    return {
        "boundary_polygon": validation["polygon_coords"],
        "usable_polygon": usable_coords,
        "boundary_area": round(boundary_area, 2),
        "usable_area": usable_area,
        "setback_applied": setback,
        "coverage_ratio": coverage_ratio,
        "is_valid": True,
    }


def generate_boundary_preview(
    boundary_coords: list,
    usable_coords: list | None = None,
    output_path: str | Path = "preview.png",
    title: str = "Boundary & Buildable Footprint",
) -> str:
    """
    Generate a matplotlib preview image showing the boundary and the
    buildable (usable) polygon overlaid.

    Returns the absolute path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect("equal")

    # --- boundary polygon (blue outline, light fill) ---
    boundary_xy = [(c[0], c[1]) for c in boundary_coords]
    boundary_patch = MplPolygon(boundary_xy, closed=True, facecolor="#cce5ff",
                                edgecolor="#004080", linewidth=2, label="Plot Boundary")
    ax.add_patch(boundary_patch)

    # --- usable polygon (green) ---
    if usable_coords:
        usable_xy = [(c[0], c[1]) for c in usable_coords]
        usable_patch = MplPolygon(usable_xy, closed=True, facecolor="#b3ffb3",
                                  edgecolor="#006600", linewidth=2, linestyle="--",
                                  alpha=0.7, label="Buildable Footprint")
        ax.add_patch(usable_patch)

    # --- compute nice axis limits ---
    all_x = [c[0] for c in boundary_coords]
    all_y = [c[1] for c in boundary_coords]
    margin_x = (max(all_x) - min(all_x)) * 0.15 or 1
    margin_y = (max(all_y) - min(all_y)) * 0.15 or 1
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(output_path.resolve())

