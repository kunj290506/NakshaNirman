"""
Job status tracking API endpoints.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import asyncio

router = APIRouter()


class JobStage(str, Enum):
    """Processing stages for a job."""
    UPLOADED = "uploaded"
    PROCESSING_FILE = "processing_file"
    EXTRACTING_BOUNDARY = "extracting_boundary"
    GENERATING_DESIGN = "generating_design"
    CREATING_CAD = "creating_cad"
    RENDERING_3D = "rendering_3d"
    CREATING_ANIMATION = "creating_animation"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    status: str
    stage: JobStage
    progress: int  # 0-100
    message: str
    estimated_remaining: Optional[int] = None  # seconds
    error: Optional[str] = None


# In-memory job tracking (replace with Redis in production)
job_states = {}


@router.get("/job/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get current processing status for a job.
    
    Returns stage, progress percentage, and estimated time remaining.
    """
    if job_id not in job_states:
        # Return initial status for new jobs
        return JobStatus(
            job_id=job_id,
            status="processing",
            stage=JobStage.UPLOADED,
            progress=0,
            message="Job queued for processing",
            estimated_remaining=300
        )
    
    return job_states[job_id]


@router.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel a running job.
    
    Only jobs in processing state can be cancelled.
    """
    if job_id in job_states:
        job_states[job_id].status = "cancelled"
        job_states[job_id].message = "Job cancelled by user"
    
    return {"job_id": job_id, "status": "cancelled"}


@router.websocket("/job/{job_id}/live")
async def job_status_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job status updates.
    
    Sends status updates every 2 seconds while job is processing.
    """
    await websocket.accept()
    
    try:
        while True:
            # Get current status
            status = job_states.get(job_id, {
                "job_id": job_id,
                "status": "processing",
                "stage": "uploaded",
                "progress": 0,
                "message": "Waiting for updates..."
            })
            
            # Send status update
            await websocket.send_json(status if isinstance(status, dict) else status.model_dump())
            
            # Check if job completed
            if isinstance(status, dict):
                if status.get("status") in ["completed", "failed", "cancelled"]:
                    break
            else:
                if status.status in ["completed", "failed", "cancelled"]:
                    break
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        pass


def update_job_status(job_id: str, stage: JobStage, progress: int, message: str):
    """Helper function to update job status (called by background tasks)."""
    job_states[job_id] = JobStatus(
        job_id=job_id,
        status="processing" if progress < 100 else "completed",
        stage=stage,
        progress=progress,
        message=message,
        estimated_remaining=max(0, int((100 - progress) * 3))  # Rough estimate
    )
