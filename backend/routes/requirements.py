from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from services.requirements import create_requirements
from schemas import RequirementsIn, RequirementsOut

router = APIRouter(prefix="/api", tags=["requirements"])


@router.post('/requirements', response_model=RequirementsOut)
async def post_requirements(req: RequirementsIn, db: AsyncSession = Depends(get_db)):
    try:
        row = await create_requirements(db, req.dict())
        return row
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
