"""
PerfectCAD Floor Plan Generation API Route.

The "PerfCat Agent" — generates architecturally-perfect house floor plans
with proper proportions, zone-based layout, wall-aligned grid, and
guaranteed adjacency compliance.

Endpoints:
  POST /api/perfect/design   — Generate a perfect floor plan
  POST /api/perfect/validate — Validate an existing layout
  GET  /api/perfect/status   — Check engine status
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.perfect_layout import (
    generate_perfect_layout,
    validate_perfect_layout,
)
from services.cad_export import generate_dxf
from config import EXPORT_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/perfect", tags=["perfectcad"])


# ---------- Request / Response Models ----------

class PerfectDesignRequest(BaseModel):
    """Request body for PerfectCAD floor plan generation."""
    # Boundary (optional — uses plot_width/plot_length if missing)
    boundary_polygon: Optional[List[List[float]]] = None
    front_door: Optional[List[float]] = None

    # Room preferences
    total_area: float = Field(default=1200, ge=200, le=10000,
                              description="Total plot area in sq ft")
    bedrooms: int = Field(default=2, ge=1, le=8)
    bathrooms: int = Field(default=1, ge=1, le=6)
    floors: int = Field(default=1, ge=1, le=4)
    extras: Optional[List[str]] = Field(
        default=[],
        description="Extra rooms: dining, study, pooja, balcony, store, utility, garage"
    )

    # Plot dimensions (alternative to boundary_polygon)
    plot_width: Optional[float] = Field(
        default=None, ge=10, le=200,
        description="Plot width in feet"
    )
    plot_length: Optional[float] = Field(
        default=None, ge=10, le=200,
        description="Plot length in feet"
    )

    # Room configuration from frontend (alternative format)
    rooms: Optional[List[Dict]] = Field(
        default=None,
        description="Room list: [{room_type, quantity}, ...]"
    )

    # Generation settings
    num_candidates: int = Field(
        default=80, ge=10, le=500,
        description="Number of layout candidates to evaluate"
    )
    project_id: Optional[str] = None


class PerfectDesignResponse(BaseModel):
    """Response for PerfectCAD floor plan generation."""
    mode: str = "design"
    engine: str = "perfectcad"
    method: str = "constraint_strip_packing"
    explanation: str = ""
    layout: Optional[Dict] = None
    validation: Optional[Dict] = None
    dxf_url: Optional[str] = None
    score: Optional[Dict] = None
    error: Optional[str] = None


class PerfectValidateRequest(BaseModel):
    """Request body for layout validation."""
    layout: Dict


class PerfectStatusResponse(BaseModel):
    """Engine status."""
    engine: str = "perfectcad"
    version: str = "1.0.0"
    features: List[str] = [
        "zone_based_placement",
        "constraint_strip_packing",
        "grid_aligned_walls",
        "aspect_ratio_enforcement",
        "adjacency_optimization",
        "multi_candidate_scoring",
    ]
    status: str = "ready"


# ---------- Endpoints ----------

@router.get("/status", response_model=PerfectStatusResponse)
async def perfect_status():
    """Check PerfectCAD engine status."""
    return PerfectStatusResponse()


@router.post("/design", response_model=PerfectDesignResponse)
async def perfect_design(
    req: PerfectDesignRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a perfect floor plan using the PerfectCAD agent.

    This engine produces architecturally-correct layouts with:
      - Proper room proportions (≤ 2:1 aspect ratio)
      - Wall-aligned grid snapping (6-inch grid)
      - Architectural zoning (public front → corridor → private back)
      - Correct bathroom-bedroom adjacency
      - Zero overlaps, maximum coverage
      - Doors & windows on correct walls
    """
    try:
        # Parse boundary coordinates
        boundary_coords = None
        if req.boundary_polygon and len(req.boundary_polygon) >= 3:
            boundary_coords = [tuple(p) for p in req.boundary_polygon]

        # Parse front door position
        front_door_pos = None
        if req.front_door and len(req.front_door) >= 2:
            front_door_pos = (req.front_door[0], req.front_door[1])

        # Extract room counts from rooms list if provided
        bedrooms = req.bedrooms
        bathrooms = req.bathrooms
        extras = list(req.extras or [])

        if req.rooms:
            bed_count = 0
            bath_count = 0
            extra_list = []
            for r in req.rooms:
                rtype = r.get("room_type", "")
                qty = r.get("quantity", 1)
                if rtype in ("master_bedroom", "bedroom"):
                    bed_count += qty
                elif rtype == "bathroom":
                    bath_count += qty
                elif rtype == "kitchen":
                    pass  # always included
                elif rtype == "living":
                    pass  # always included
                else:
                    for _ in range(qty):
                        extra_list.append(rtype)
            if bed_count > 0:
                bedrooms = bed_count
            if bath_count > 0:
                bathrooms = bath_count
            if extra_list:
                extras = extra_list

        # Determine plot dimensions
        plot_w = req.plot_width
        plot_l = req.plot_length

        if not plot_w or not plot_l:
            if boundary_coords:
                xs = [c[0] for c in boundary_coords]
                ys = [c[1] for c in boundary_coords]
                plot_w = max(xs) - min(xs)
                plot_l = max(ys) - min(ys)
            else:
                # Derive from total_area with ~1.3:1 ratio
                import math
                side = math.sqrt(req.total_area)
                plot_w = round(side * 1.14, 1)
                plot_l = round(req.total_area / plot_w, 1)

        # Generate the layout
        layout = generate_perfect_layout(
            plot_width=plot_w,
            plot_length=plot_l,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floors=req.floors,
            extras=extras,
            boundary_coords=boundary_coords,
            front_door_pos=front_door_pos,
            total_area=req.total_area,
            num_candidates=req.num_candidates,
        )

        # Validate the generated layout
        validation = validate_perfect_layout(layout)

        # Generate DXF export if project_id provided
        dxf_url = None
        if req.project_id and layout.get("rooms"):
            try:
                dxf_url = await _generate_dxf(req.project_id, layout)
            except Exception as e:
                logger.warning(f"DXF generation failed: {e}")

        return PerfectDesignResponse(
            engine="perfectcad",
            method=layout.get("method", "constraint_strip_packing"),
            explanation=layout.get("explanation", ""),
            layout=layout,
            validation=validation,
            score=layout.get("score"),
            dxf_url=dxf_url,
        )

    except Exception as e:
        logger.error(f"PerfectCAD design failed: {e}", exc_info=True)
        return PerfectDesignResponse(
            error=str(e),
            engine="perfectcad",
        )


@router.post("/validate")
async def perfect_validate(req: PerfectValidateRequest):
    """Validate an existing floor plan against PerfectCAD standards."""
    try:
        result = validate_perfect_layout(req.layout)
        return result
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Internal Helpers ----------

async def _generate_dxf(project_id: str, layout: Dict) -> Optional[str]:
    """Generate DXF file from layout and return URL."""
    rooms_for_dxf = []
    boundary = layout.get("boundary", [])

    for room in layout.get("rooms", []):
        polygon = room.get("polygon", [])
        if not polygon:
            pos = room.get("position", {})
            w = room.get("width", 0)
            h = room.get("length", 0)
            x, y = pos.get("x", 0), pos.get("y", 0)
            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]

        rooms_for_dxf.append({
            "room_type": room.get("room_type", "room"),
            "polygon": polygon,
            "label": room.get("label", room.get("name", "")),
        })

    doors_for_dxf = layout.get("doors", [])
    file_path = generate_dxf(
        project_id=project_id,
        boundary=boundary,
        rooms=rooms_for_dxf,
        doors=doors_for_dxf,
    )

    if file_path and os.path.exists(file_path):
        filename = os.path.basename(file_path)
        return f"/exports/{filename}"

    return None
