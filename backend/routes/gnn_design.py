"""
GNN-Inspired Floor Plan Generation API Route.

Endpoints:
  POST /api/gnn/design — Generate floor plan using GNN pipeline
  POST /api/gnn/validate — Validate an existing layout
  GET  /api/gnn/status — Check engine status (model availability)
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.gnn_engine import (
    generate_gnn_floor_plan,
    validate_gnn_layout,
    TORCH_AVAILABLE,
    TORCH_GEO_AVAILABLE,
    SHAPELY_AVAILABLE,
    NX_AVAILABLE,
)
from services.cad_export import generate_dxf
from config import EXPORT_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/gnn", tags=["gnn"])


# ---------- Request / Response Models ----------

class GNNDesignRequest(BaseModel):
    """Request body for GNN-based floor plan generation."""
    # Boundary (optional — will auto-generate rectangle if missing)
    boundary_polygon: Optional[List[List[float]]] = None
    front_door: Optional[List[float]] = None  # [x, y]

    # Room preferences
    total_area: float = Field(default=1200, ge=200, le=10000)
    bedrooms: int = Field(default=2, ge=0, le=8)
    bathrooms: int = Field(default=0, ge=0, le=6)
    kitchens: int = Field(default=1, ge=0, le=3)
    floors: int = Field(default=1, ge=1, le=4)
    extras: Optional[List[str]] = []

    # Plot dimensions (alternative to boundary_polygon)
    plot_width: Optional[float] = None
    plot_length: Optional[float] = None

    # Room configuration from frontend (alternative format)
    rooms: Optional[List[Dict]] = None

    # Model settings
    model_path: Optional[str] = None
    project_id: Optional[str] = None


class GNNDesignResponse(BaseModel):
    """Response for GNN-based floor plan generation."""
    mode: str = "design"
    engine: str = "gnn"
    method: str = "heuristic"
    explanation: str = ""
    layout: Optional[Dict] = None
    validation: Optional[Dict] = None
    dxf_url: Optional[str] = None
    error: Optional[str] = None


class GNNStatusResponse(BaseModel):
    """Engine status information."""
    engine: str = "gnn"
    torch_available: bool = False
    torch_geometric_available: bool = False
    shapely_available: bool = False
    networkx_available: bool = False
    model_loaded: bool = False
    mode: str = "heuristic"


# ---------- Endpoints ----------

@router.get("/status", response_model=GNNStatusResponse)
async def gnn_status():
    """Check GNN engine status and available libraries."""
    from services.gnn_engine import load_gat_model
    model = load_gat_model()
    return GNNStatusResponse(
        torch_available=TORCH_AVAILABLE,
        torch_geometric_available=TORCH_GEO_AVAILABLE,
        shapely_available=SHAPELY_AVAILABLE,
        networkx_available=NX_AVAILABLE,
        model_loaded=model is not None,
        mode="model" if model else "heuristic",
    )


@router.post("/design", response_model=GNNDesignResponse)
async def gnn_design(req: GNNDesignRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate a floor plan using the GNN-inspired pipeline.

    Accepts boundary polygon + room preferences and returns a complete layout.
    Falls back to heuristic mode if GATNet model weights are not available.
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
        kitchens = req.kitchens
        master_bedrooms = None  # will be passed to engine
        extras = list(req.extras or [])

        if req.rooms:
            # Parse frontend room format [{room_type, quantity}, ...]
            master_bed_count = 0
            regular_bed_count = 0
            bath_count = 0
            kitchen_count = 0
            extra_list = []
            for r in req.rooms:
                rtype = r.get('room_type', '')
                qty = r.get('quantity', 1)
                if rtype == 'master_bedroom':
                    master_bed_count += qty
                elif rtype == 'bedroom':
                    regular_bed_count += qty
                elif rtype == 'bathroom':
                    bath_count += qty
                elif rtype == 'kitchen':
                    kitchen_count += qty
                elif rtype == 'living':
                    pass  # always included
                else:
                    for _ in range(qty):
                        extra_list.append(rtype)
            total_beds = master_bed_count + regular_bed_count
            if total_beds > 0:
                bedrooms = total_beds
                master_bedrooms = master_bed_count  # how many are master
            if bath_count > 0:
                bathrooms = bath_count  # EXTRA common bathrooms only
            if kitchen_count > 0:
                kitchens = kitchen_count
            if extra_list:
                extras = extra_list

        # Generate floor plan
        logger.info(
            f"GNN design request: total_area={req.total_area}, bedrooms={bedrooms}, "
            f"master_bedrooms={master_bedrooms}, bathrooms={bathrooms}, "
            f"kitchens={kitchens}, floors={req.floors}, "
            f"extras={extras}, rooms_from_frontend={bool(req.rooms)}"
        )
        layout = generate_gnn_floor_plan(
            boundary_coords=boundary_coords,
            front_door_pos=front_door_pos,
            total_area=req.total_area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            kitchens=kitchens,
            floors=req.floors,
            extras=extras,
            model_path=req.model_path,
            plot_width=req.plot_width,
            plot_length=req.plot_length,
            master_bedrooms=master_bedrooms,
        )

        # Generate DXF export if project_id provided
        dxf_url = None
        if req.project_id and layout.get('rooms'):
            try:
                dxf_url = await _generate_dxf(req.project_id, layout)
            except Exception as e:
                logger.warning(f"DXF generation failed: {e}")

        return GNNDesignResponse(
            engine="gnn",
            method=layout.get('method', 'heuristic'),
            explanation=layout.get('explanation', ''),
            layout=layout,
            validation=layout.get('validation'),
            dxf_url=dxf_url,
        )

    except Exception as e:
        logger.error(f"GNN design failed: {e}", exc_info=True)
        return GNNDesignResponse(
            error=str(e),
            engine="gnn",
        )


@router.post("/validate")
async def gnn_validate(layout: Dict):
    """Validate an existing floor plan layout."""
    result = validate_gnn_layout(layout)
    return result


# Also serve at /api/engine/design for backward compatibility
@router.post("/engine-design")
async def gnn_engine_design_compat(req: GNNDesignRequest, db: AsyncSession = Depends(get_db)):
    """Backward-compatible endpoint that maps to GNN design."""
    return await gnn_design(req, db)


# ---------- Helpers ----------

async def _generate_dxf(project_id: str, layout: Dict) -> Optional[str]:
    """Generate DXF file for the layout."""
    try:
        rooms = layout.get('rooms', [])
        plot = layout.get('plot', {})

        # Convert to format expected by DXF exporter
        boundary_coords = layout.get('boundary', [[0, 0], [30, 0], [30, 40], [0, 40], [0, 0]])
        plan_data = {
            'boundary': boundary_coords,
            'rooms': rooms,
            'total_area': layout.get('total_area', 0),
        }

        dxf_path = generate_dxf(project_id, plan_data, str(EXPORT_DIR))
        if dxf_path and os.path.exists(dxf_path):
            filename = os.path.basename(dxf_path)
            return f"/exports/{filename}"
    except Exception as e:
        logger.error(f"DXF export error: {e}")

    return None
