"""Floor plan generation and DXF download routes."""

import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Project, Room, ProjectStatus, RoomType
from schemas import GenerateRequest, GenerateResponse
from services.multi_agent_orchestrator import generate_dynamic_layout
from services.cad_export import generate_dxf
from app_config import EXPORT_DIR
import json

router = APIRouter(prefix="/api", tags=["floorplan"])


def _default_boundary(total_area: float) -> list:
    """Generate a default rectangular boundary for a given area."""
    import math
    side = math.sqrt(total_area)
    w = side * 1.3
    h = total_area / w
    return [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]


def _rect_polygon(x: float, y: float, w: float, h: float) -> list:
    return [
        [round(x, 2), round(y, 2)],
        [round(x + w, 2), round(y, 2)],
        [round(x + w, 2), round(y + h, 2)],
        [round(x, 2), round(y + h, 2)],
        [round(x, 2), round(y, 2)],
    ]


@router.post("/generate-floorplan", response_model=GenerateResponse)
async def generate_floorplan(data: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Generate a floor plan and DXF file."""
    # Get project
    result = await db.execute(select(Project).where(Project.id == data.project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get boundary
    boundary = data.boundary_polygon
    if not boundary and project.boundary_polygon:
        boundary = json.loads(project.boundary_polygon)
    
    total_area = data.total_area or project.total_area or 1200

    if not boundary:
        boundary = _default_boundary(total_area)

    # Prepare room list
    rooms = [{"room_type": r.room_type, "quantity": r.quantity, "desired_area": r.desired_area} for r in data.rooms]

    if not rooms:
        # Default rooms
        rooms = [
            {"room_type": "living", "quantity": 1},
            {"room_type": "master_bedroom", "quantity": 1},
            {"room_type": "bedroom", "quantity": 1},
            {"room_type": "kitchen", "quantity": 1},
            {"room_type": "bathroom", "quantity": 1},
            {"room_type": "dining", "quantity": 1},
        ]

    # Update project status
    project.status = ProjectStatus.PROCESSING
    project.total_area = total_area
    await db.flush()

    try:
        # Infer coarse requirements and generate via multi-agent orchestrator
        bedroom_count = 0
        bathroom_count = 0
        extras = []
        for r in rooms:
            rt = str(r.get("room_type", "")).lower()
            qty = int(r.get("quantity") or 1)
            if rt in {"bedroom", "master_bedroom"}:
                bedroom_count += qty
            elif rt in {"bathroom", "toilet"}:
                bathroom_count += qty
            elif rt not in {"living", "kitchen", "dining"}:
                extras.extend([rt] * qty)

        payload = {
            "total_area": total_area,
            "plot_width": None,
            "plot_length": None,
            "bedrooms": max(1, bedroom_count or 2),
            "bathrooms": max(1, bathroom_count or max(1, bedroom_count or 2)),
            "extras": sorted(list(set(extras))),
            "facing": "east",
            "vastu": True,
            "boundary_polygon": boundary,
            "family_type": "nuclear",
        }
        plan = generate_dynamic_layout(payload)
        if "error" in plan:
            raise ValueError(plan["error"])

        # Save rooms to DB
        for room_data in plan.get("rooms", []):
            rtype_str = room_data.get("type", room_data.get("room_type", "other"))
            try:
                rtype = RoomType(rtype_str)
            except ValueError:
                rtype = RoomType.OTHER

            poly = room_data.get("polygon")
            if not poly and all(k in room_data for k in ["x", "y", "width", "height"]):
                poly = _rect_polygon(
                    float(room_data.get("x", 0)),
                    float(room_data.get("y", 0)),
                    float(room_data.get("width", 0)),
                    float(room_data.get("height", 0)),
                )

            room = Room(
                project_id=project.id,
                room_type=rtype,
                quantity=1,
                desired_area=room_data.get("target_area", room_data.get("area")),
                generated_polygon=json.dumps(poly or []),
            )
            db.add(room)

        # Generate DXF
        dxf_filename = f"{project.id}.dxf"
        dxf_path = os.path.join(str(EXPORT_DIR), dxf_filename)
        generate_dxf(plan, dxf_path)

        # Update project
        project.generated_plan = json.dumps(plan)
        project.dxf_path = dxf_path
        project.boundary_polygon = json.dumps(boundary)
        project.status = ProjectStatus.COMPLETED
        await db.flush()

        return GenerateResponse(
            project_id=project.id,
            status="completed",
            plan=plan,
            dxf_url=f"/api/download-dxf/{project.id}",
        )

    except Exception as e:
        project.status = ProjectStatus.DRAFTING
        await db.flush()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/download-dxf/{project_id}")
async def download_dxf(project_id: str, db: AsyncSession = Depends(get_db)):
    """Download the generated DXF file."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.dxf_path or not os.path.exists(project.dxf_path):
        raise HTTPException(status_code=404, detail="DXF file not found. Generate floor plan first.")

    return FileResponse(
        project.dxf_path,
        media_type="application/dxf",
        filename=f"floorplan_{project_id}.dxf",
    )
