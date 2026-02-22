from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from services.requirements import (
    create_requirements,
    get_requirements_by_id,
    get_requirements_by_project,
    requirements_to_json,
)
from schemas import RequirementsIn, RequirementsOut

router = APIRouter(prefix="/api", tags=["requirements"])


@router.post('/requirements', response_model=RequirementsOut)
async def post_requirements(req: RequirementsIn, db: AsyncSession = Depends(get_db)):
    """Store structured requirements (hard + soft constraints)."""
    try:
        row = await create_requirements(db, req.dict())
        return row
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/requirements/{req_id}', response_model=RequirementsOut)
async def get_requirements(req_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve requirements by ID."""
    row = await get_requirements_by_id(db, req_id)
    if not row:
        raise HTTPException(status_code=404, detail="Requirements not found")
    return row


@router.get('/requirements/project/{project_id}', response_model=RequirementsOut)
async def get_project_requirements(project_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve the latest requirements for a project."""
    row = await get_requirements_by_project(db, project_id)
    if not row:
        raise HTTPException(status_code=404, detail="No requirements found for this project")
    return row


@router.get('/requirements/{req_id}/json')
async def get_requirements_json(req_id: str, db: AsyncSession = Depends(get_db)):
    """Export requirements as the strict JSON format (requirements.json)."""
    row = await get_requirements_by_id(db, req_id)
    if not row:
        raise HTTPException(status_code=404, detail="Requirements not found")
    return JSONResponse(content=requirements_to_json(row))
