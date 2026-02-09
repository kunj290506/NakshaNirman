"""
Upload API endpoints for file processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import aiofiles

from app.core.config import settings
from app.services.file_processor import process_uploaded_file

router = APIRouter()


class UploadResponse(BaseModel):
    """Response model for file upload."""
    job_id: str
    status: str
    message: str
    filename: str
    file_type: str
    estimated_time: int


class UploadMetadata(BaseModel):
    """Metadata for uploaded file."""
    name: Optional[str] = None
    scale: Optional[float] = None
    north_angle: Optional[float] = 0


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a boundary file (image or DXF) for processing.
    
    Supported formats:
    - Images: JPG, PNG
    - CAD: DXF, DWG
    
    Returns a job ID for tracking processing status.
    """
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Check file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(job_dir, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Determine file type
    file_type = "image" if file_ext in [".jpg", ".jpeg", ".png"] else "cad"
    
    # Queue background processing
    background_tasks.add_task(
        process_uploaded_file,
        job_id=job_id,
        file_path=file_path,
        file_type=file_type
    )
    
    return UploadResponse(
        job_id=job_id,
        status="processing",
        message="File uploaded successfully. Processing started.",
        filename=file.filename,
        file_type=file_type,
        estimated_time=30  # Initial estimate in seconds
    )


@router.get("/upload/{job_id}/preview")
async def get_upload_preview(job_id: str):
    """Get preview thumbnail of uploaded file."""
    preview_path = os.path.join(settings.UPLOAD_DIR, job_id, "preview.png")
    
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview not available yet")
    
    return JSONResponse({
        "preview_url": f"/outputs/{job_id}/preview.png"
    })
