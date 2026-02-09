"""
Design generation API endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from app.services.ai_designer import generate_design

router = APIRouter()


class ArchitecturalStyle(str, Enum):
    """Available architectural styles."""
    MODERN = "modern"
    TRADITIONAL = "traditional"
    CONTEMPORARY = "contemporary"
    MINIMALIST = "minimalist"
    MEDITERRANEAN = "mediterranean"


class BudgetLevel(str, Enum):
    """Budget levels for design."""
    ECONOMY = "economy"
    STANDARD = "standard"
    PREMIUM = "premium"


class DesignRequirements(BaseModel):
    """User requirements for design generation."""
    bedrooms: int = Field(ge=1, le=6, default=3)
    bathrooms: int = Field(ge=1, le=4, default=2)
    style: ArchitecturalStyle = ArchitecturalStyle.MODERN
    features: List[str] = []
    budget: BudgetLevel = BudgetLevel.STANDARD
    additional_notes: Optional[str] = None


class DesignRequest(BaseModel):
    """Request model for design generation."""
    job_id: str
    requirements: DesignRequirements


class DesignResponse(BaseModel):
    """Response model for design generation."""
    job_id: str
    status: str
    message: str
    estimated_time: int


@router.post("/design/generate", response_model=DesignResponse)
async def trigger_design_generation(
    request: DesignRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger AI design generation for an uploaded boundary.
    
    Requires a valid job_id from a previous upload.
    """
    # Validate job exists
    # TODO: Check job exists in database
    
    # Queue design generation
    background_tasks.add_task(
        generate_design,
        job_id=request.job_id,
        requirements=request.requirements.model_dump()
    )
    
    return DesignResponse(
        job_id=request.job_id,
        status="processing",
        message="Design generation started. This may take 1-2 minutes.",
        estimated_time=90
    )


@router.post("/design/modify")
async def modify_design(job_id: str, modifications: dict):
    """
    Request modifications to an existing design.
    
    Allows users to adjust room sizes, positions, or features.
    """
    # TODO: Implement design modification
    return {
        "job_id": job_id,
        "status": "modification_queued",
        "message": "Design modification request received"
    }


@router.get("/design/{job_id}/alternatives")
async def get_design_alternatives(job_id: str):
    """
    Get alternative design variations for a job.
    
    Returns up to 3 different layout options.
    """
    # TODO: Fetch alternatives from database
    return {
        "job_id": job_id,
        "alternatives": [],
        "count": 0
    }
