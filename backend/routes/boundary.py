"""Boundary upload and processing routes — Phase 1.

Endpoints:
  POST /api/upload-boundary        — Upload DXF/image, get file_id
  GET  /api/extract-boundary/{id}  — Parse file → boundary_polygon.json
  POST /api/buildable-footprint/{id} — Apply setback → usable_polygon.json + preview
  GET  /api/boundary-preview/{id}  — Serve preview PNG
"""

import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import Project, BoundaryUpload
from services.boundary import (
    process_boundary_file,
    compute_buildable_footprint,
    generate_boundary_preview,
)
from app_config import UPLOAD_DIR, EXPORT_DIR
import json

router = APIRouter(prefix="/api", tags=["boundary"])


# ---------- 1. Upload DXF / Image ----------

@router.post("/upload-boundary")
async def upload_boundary(
    file: UploadFile = File(...),
    project_id: str = Form(None),
    scale: float = Form(1.0),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a boundary file (.dxf or image).

    - If *project_id* is provided the upload is linked to that project.
    - Returns a **file_id** that identifies this upload for subsequent calls
      (`/extract-boundary`, `/buildable-footprint`).
    """
    # Determine file type
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    image_extensions = ("png", "jpg", "jpeg", "bmp", "tiff", "tif", "gif", "webp")

    if ext in image_extensions:
        file_type = "image"
    elif ext == "dxf":
        file_type = "dxf"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join([*image_extensions, 'dxf'])}",
        )

    # Validate project if provided
    if project_id:
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    # Save file to disk
    file_id = str(uuid.uuid4())
    save_path = os.path.join(str(UPLOAD_DIR), f"{file_id}.{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Persist record (polygon extraction is deferred to /extract-boundary)
    upload = BoundaryUpload(
        id=file_id,
        project_id=project_id,
        file_path=save_path,
        file_type=file_type,
    )
    db.add(upload)
    await db.flush()

    return {
        "status": "uploaded",
        "file_id": file_id,
        "filename": filename,
        "file_type": file_type,
    }


# ---------- 2. Extract Boundary Polygon ----------

@router.get("/extract-boundary/{file_id}")
async def extract_boundary(
    file_id: str,
    scale: float = Query(1.0, description="Scale factor for coordinates"),
    db: AsyncSession = Depends(get_db),
):
    """
    Parse a previously-uploaded DXF/image and return the boundary polygon.

    Output: boundary_polygon (coordinate list), area, perimeter, validation flags.
    """
    result = await db.execute(select(BoundaryUpload).where(BoundaryUpload.id == file_id))
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if not os.path.exists(upload.file_path):
        raise HTTPException(status_code=404, detail="Uploaded file missing from disk")

    try:
        boundary_data = process_boundary_file(upload.file_path, upload.file_type, scale)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # Persist extracted polygon
    upload.processed_polygon = json.dumps(boundary_data["polygon"])
    upload.boundary_area = boundary_data["area"]

    # Update project boundary if linked
    if upload.project_id:
        proj_result = await db.execute(select(Project).where(Project.id == upload.project_id))
        project = proj_result.scalar_one_or_none()
        if project:
            project.boundary_polygon = json.dumps(boundary_data["polygon"])
            project.total_area = boundary_data["area"]

    await db.flush()

    return {
        "file_id": file_id,
        "boundary_polygon": boundary_data["polygon"],
        "area": boundary_data["area"],
        "num_vertices": boundary_data["num_vertices"],
        "perimeter": boundary_data.get("perimeter", 0),
        "is_valid": boundary_data.get("is_valid", True),
        "is_closed": True,
        "is_self_intersecting": False,
    }


# ---------- 3. Compute Buildable Footprint ----------

@router.post("/buildable-footprint/{file_id}")
async def buildable_footprint(
    file_id: str,
    setback: float | None = Query(None, description="Override setback distance in meters"),
    region: str = Query("india_mvp", description="Region rule set to apply"),
    db: AsyncSession = Depends(get_db),
):
    """
    Apply setback offset inward to produce the usable (buildable) polygon.

    If a boundary has not yet been extracted for *file_id*, extraction is
    performed automatically first.

    Returns usable_polygon, areas, coverage ratio, and a preview image URL.
    """
    result = await db.execute(select(BoundaryUpload).where(BoundaryUpload.id == file_id))
    upload = result.scalar_one_or_none()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Auto-extract if not yet done
    if not upload.processed_polygon:
        if not os.path.exists(upload.file_path):
            raise HTTPException(status_code=404, detail="Uploaded file missing from disk")
        try:
            boundary_data = process_boundary_file(upload.file_path, upload.file_type)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        upload.processed_polygon = json.dumps(boundary_data["polygon"])
        upload.boundary_area = boundary_data["area"]
        await db.flush()

    boundary_polygon = json.loads(upload.processed_polygon)

    # Compute buildable footprint
    try:
        footprint = compute_buildable_footprint(
            boundary_polygon_coords=boundary_polygon,
            setback=setback,
            region=region,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Generate preview image
    preview_dir = EXPORT_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{file_id}_preview.png"

    try:
        generate_boundary_preview(
            boundary_coords=footprint["boundary_polygon"],
            usable_coords=footprint["usable_polygon"],
            output_path=preview_path,
            title=f"Plot Boundary — setback {footprint['setback_applied']}m",
        )
        preview_url = f"/api/boundary-preview/{file_id}"
    except Exception:
        preview_url = None

    # Persist results
    upload.usable_polygon = json.dumps(footprint["usable_polygon"])
    upload.usable_area = footprint["usable_area"]
    upload.setback_applied = footprint["setback_applied"]
    upload.preview_path = str(preview_path)
    await db.flush()

    return {
        "file_id": file_id,
        "boundary_polygon": footprint["boundary_polygon"],
        "usable_polygon": footprint["usable_polygon"],
        "boundary_area": footprint["boundary_area"],
        "usable_area": footprint["usable_area"],
        "setback_applied": footprint["setback_applied"],
        "coverage_ratio": footprint["coverage_ratio"],
        "preview_url": preview_url,
        "is_valid": footprint["is_valid"],
    }


# ---------- 4. Preview Image ----------

@router.get("/boundary-preview/{file_id}")
async def boundary_preview(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the preview PNG for a boundary upload."""
    result = await db.execute(select(BoundaryUpload).where(BoundaryUpload.id == file_id))
    upload = result.scalar_one_or_none()
    if not upload or not upload.preview_path:
        raise HTTPException(status_code=404, detail="Preview not found. Run /buildable-footprint first.")

    if not os.path.exists(upload.preview_path):
        raise HTTPException(status_code=404, detail="Preview file missing from disk")

    return FileResponse(upload.preview_path, media_type="image/png")
