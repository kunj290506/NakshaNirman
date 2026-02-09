"""
Gemini AI Service for Intelligent Floor Plan Generation
=========================================================
Uses Google Gemini AI to analyze boundaries and generate optimal room layouts.
"""

import json
import os
import structlog
from typing import Dict, List, Tuple, Optional

from app.core.config import settings

logger = structlog.get_logger()

# Gemini client
_gemini_model = None


def get_gemini_model():
    """Initialize Gemini model lazily."""
    global _gemini_model
    if _gemini_model is None:
        try:
            import google.generativeai as genai
            
            api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set, falling back to algorithmic placement")
                return None
            
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini AI model initialized")
        except Exception as e:
            logger.error("Failed to initialize Gemini", error=str(e))
            return None
    return _gemini_model


def analyze_boundary_with_gemini(boundary_points: List[Tuple], total_area: float, requirements: Dict) -> Dict:
    """
    Use Gemini AI to analyze boundary and suggest optimal room layout.
    
    Returns a structured layout plan with room positions.
    """
    model = get_gemini_model()
    
    if model is None:
        logger.info("Gemini not available, using fallback algorithm")
        return None
    
    # Calculate bounding box
    min_x = min(p[0] for p in boundary_points)
    max_x = max(p[0] for p in boundary_points)
    min_y = min(p[1] for p in boundary_points)
    max_y = max(p[1] for p in boundary_points)
    
    width = max_x - min_x
    height = max_y - min_y
    
    bedrooms = requirements.get("bedrooms", 3)
    bathrooms = requirements.get("bathrooms", 2)
    style = requirements.get("style", "modern")
    
    prompt = f"""You are an expert architect AI. Analyze this irregular plot boundary and generate an optimal floor plan layout.

PLOT BOUNDARY POLYGON (coordinates in meters):
{json.dumps(boundary_points)}

PLOT DETAILS:
- Bounding box: {width:.1f}m x {height:.1f}m
- Total usable area: {total_area:.1f} sqm
- Min X: {min_x:.1f}, Max X: {max_x:.1f}, Min Y: {min_y:.1f}, Max Y: {max_y:.1f}

REQUIREMENTS:
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Style: {style}
- Must include: Living Room, Kitchen, Dining

ARCHITECTURAL PRINCIPLES TO FOLLOW:
1. All rooms MUST fit entirely within the polygon boundary
2. Living room should be central and largest
3. Kitchen should be accessible from dining
4. Bedrooms should be in quieter areas (away from living)
5. Master bedroom should have attached bathroom
6. Leave some space for hallways/circulation (don't fill 100%)
7. No rooms should overlap

OUTPUT FORMAT (JSON only, no markdown):
{{
  "rooms": [
    {{
      "name": "Living Room",
      "type": "living",
      "x": <left edge x coordinate>,
      "y": <bottom edge y coordinate>,
      "width": <room width in meters>,
      "height": <room height in meters>,
      "reason": "<brief reason for placement>"
    }},
    ... more rooms
  ],
  "layout_notes": "<overall layout strategy>"
}}

Generate room positions that FIT WITHIN the irregular polygon boundary. Be conservative with sizes to ensure rooms don't extend outside the plot."""

    try:
        logger.info("Requesting room layout from Gemini AI...")
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        if text.startswith("json"):
            text = text[4:].strip()
        
        layout = json.loads(text)
        logger.info("Gemini AI generated layout", rooms=len(layout.get("rooms", [])))
        return layout
        
    except Exception as e:
        logger.error("Gemini layout generation failed", error=str(e))
        return None


def validate_room_in_polygon(room: Dict, polygon: List[Tuple]) -> bool:
    """Check if room fits within polygon."""
    from app.services.ai_designer import point_in_polygon
    
    x, y = room["x"], room["y"]
    w, h = room["width"], room["height"]
    
    # Check all corners
    corners = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]
    
    return all(point_in_polygon(cx, cy, polygon) for cx, cy in corners)


def generate_layout_with_gemini(boundary: Dict, requirements: Dict) -> Optional[List[Dict]]:
    """
    Generate room layout using Gemini AI.
    Returns list of room definitions or None if failed.
    """
    points = boundary.get("points", [])
    if not points or len(points) < 3:
        return None
    
    polygon = [(p[0], p[1]) for p in points]
    
    # Calculate area
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    total_area = abs(area) / 2
    
    # Get Gemini suggestions
    layout = analyze_boundary_with_gemini(polygon, total_area, requirements)
    
    if layout is None or "rooms" not in layout:
        return None
    
    # Validate and adjust rooms
    valid_rooms = []
    for room_data in layout["rooms"]:
        room = {
            "name": room_data.get("name", "Room"),
            "type": room_data.get("type", "other"),
            "x": float(room_data.get("x", 0)),
            "y": float(room_data.get("y", 0)),
            "width": float(room_data.get("width", 3)),
            "height": float(room_data.get("height", 3))
        }
        
        # Validate room is inside polygon
        if validate_room_in_polygon(room, polygon):
            valid_rooms.append(room)
            logger.info(f"Valid room: {room['name']}", x=room['x'], y=room['y'])
        else:
            logger.warning(f"Room outside boundary, skipping: {room['name']}")
    
    if len(valid_rooms) < 3:
        logger.warning("Too few valid rooms from Gemini, falling back")
        return None
    
    return valid_rooms
