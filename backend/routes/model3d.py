"""3D model generation and download routes."""

import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Project
from services.model3d import generate_3d_model
from app_config import EXPORT_DIR
import json

router = APIRouter(prefix="/api", tags=["3d"])


@router.post("/generate-3d/{project_id}")
async def generate_3d(project_id: str, db: AsyncSession = Depends(get_db)):
    """Generate 3D model from an existing floor plan."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.generated_plan:
        raise HTTPException(status_code=400, detail="No floor plan generated yet. Generate floor plan first.")

    plan = json.loads(project.generated_plan)

    try:
        model_filename = f"{project_id}.glb"
        model_path = os.path.join(str(EXPORT_DIR), model_filename)

        generate_3d_model(plan, model_path)

        project.model3d_path = model_path
        await db.flush()

        return {
            "status": "success",
            "model_url": f"/api/3d-model/{project_id}",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D generation failed: {str(e)}")


@router.get("/3d-model/{project_id}")
async def get_3d_model(project_id: str, db: AsyncSession = Depends(get_db)):
    """Download the 3D model file."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.model3d_path or not os.path.exists(project.model3d_path):
        raise HTTPException(status_code=404, detail="3D model not found. Generate it first.")

    return FileResponse(
        project.model3d_path,
        media_type="model/gltf-binary",
        filename=f"model_{project_id}.glb",
    )
