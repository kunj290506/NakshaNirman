"""
Results retrieval API endpoints.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os

from app.core.config import settings

router = APIRouter()


class RoomSpecification(BaseModel):
    """Specification for a single room."""
    name: str
    area_sqm: float
    dimensions: str
    features: List[str] = []


class DesignSpecifications(BaseModel):
    """Complete design specifications."""
    total_area_sqm: float
    rooms: List[RoomSpecification]
    materials: Dict[str, str]
    estimated_cost: Optional[str] = None


class ResultFiles(BaseModel):
    """Generated file URLs."""
    dxf_url: Optional[str] = None
    pdf_url: Optional[str] = None
    png_url: Optional[str] = None
    jpg_url: Optional[str] = None
    gltf_url: Optional[str] = None
    video_url: Optional[str] = None


class JobResults(BaseModel):
    """Complete job results."""
    job_id: str
    status: str
    files: ResultFiles
    specifications: Optional[DesignSpecifications] = None
    alternatives_count: int = 0
    share_url: Optional[str] = None


@router.get("/results/{job_id}", response_model=JobResults)
async def get_results(job_id: str):
    """
    Retrieve all generated files and specifications for a completed job.
    """
    output_dir = os.path.join(settings.OUTPUT_DIR, job_id)
    
    if not os.path.exists(output_dir):
        raise HTTPException(
            status_code=404,
            detail="Results not found. Job may still be processing."
        )
    
    # Check for generated files
    files = ResultFiles()
    
    dxf_path = os.path.join(output_dir, "floor_plan.dxf")
    if os.path.exists(dxf_path):
        files.dxf_url = f"/outputs/{job_id}/floor_plan.dxf"
    
    pdf_path = os.path.join(output_dir, "floor_plan.pdf")
    if os.path.exists(pdf_path):
        files.pdf_url = f"/outputs/{job_id}/floor_plan.pdf"
    
    png_path = os.path.join(output_dir, "floorplan.png")
    if os.path.exists(png_path):
        files.png_url = f"/outputs/{job_id}/floorplan.png"
    
    jpg_path = os.path.join(output_dir, "floorplan.jpg")
    if os.path.exists(jpg_path):
        files.jpg_url = f"/outputs/{job_id}/floorplan.jpg"
    
    gltf_path = os.path.join(output_dir, "model.gltf")
    if os.path.exists(gltf_path):
        files.gltf_url = f"/outputs/{job_id}/model.gltf"
    
    video_path = os.path.join(output_dir, "animation.mp4")
    if os.path.exists(video_path):
        files.video_url = f"/outputs/{job_id}/animation.mp4"
    
    # Load specifications if available
    specs = None
    # TODO: Load from JSON file
    
    return JobResults(
        job_id=job_id,
        status="completed",
        files=files,
        specifications=specs,
        alternatives_count=0,
        share_url=f"/share/{job_id}"
    )


@router.get("/results/{job_id}/download/{file_type}")
async def download_file(job_id: str, file_type: str):
    """
    Download a specific generated file.
    
    file_type: dxf, pdf, png, gltf, video
    """
    file_map = {
        "dxf": "floor_plan.dxf",
        "pdf": "floor_plan.pdf",
        "png": "floorplan.png",
        "jpg": "floorplan.jpg",
        "gltf": "model.gltf",
        "video": "animation.mp4"
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(settings.OUTPUT_DIR, job_id, file_map[file_type])
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        filename=file_map[file_type],
        media_type="application/octet-stream"
    )


@router.get("/results/{job_id}/specifications")
async def get_specifications(job_id: str):
    """
    Get detailed room-by-room specifications.
    """
    # TODO: Load from database/file
    return {
        "job_id": job_id,
        "specifications": {
            "total_area": 150,
            "rooms": []
        }
    }
