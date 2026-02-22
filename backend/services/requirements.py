import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Requirements, Project


async def create_requirements(db: AsyncSession, data: dict):
    """Create a Requirements row from dict and link to project if provided."""
    row = Requirements(
        project_id=data.get('project_id'),
        floors=int(data['floors']),
        bedrooms=int(data['bedrooms']),
        bathrooms=int(data['bathrooms']),
        kitchen=int(data['kitchen']),
        max_area=float(data['max_area']),
        balcony=1 if data.get('balcony') else 0,
        parking=1 if data.get('parking') else 0,
        pooja_room=1 if data.get('pooja_room') else 0,
    )
    db.add(row)
    await db.flush()
    await db.commit()
    await db.refresh(row)
    return row


async def get_requirements_by_id(db: AsyncSession, req_id: str):
    """Retrieve a single requirements row by its primary key."""
    result = await db.execute(select(Requirements).where(Requirements.id == req_id))
    return result.scalars().first()


async def get_requirements_by_project(db: AsyncSession, project_id: str):
    """Retrieve the latest requirements for a project."""
    result = await db.execute(
        select(Requirements)
        .where(Requirements.project_id == project_id)
        .order_by(Requirements.created_at.desc())
    )
    return result.scalars().first()


def requirements_to_json(row: Requirements) -> dict:
    """Convert a Requirements ORM row to the strict JSON output format."""
    return {
        "hard_constraints": {
            "floors": row.floors,
            "bedrooms": row.bedrooms,
            "bathrooms": row.bathrooms,
            "kitchen": row.kitchen,
            "max_area": row.max_area,
        },
        "soft_constraints": {
            "balcony": bool(row.balcony),
            "parking": bool(row.parking),
            "pooja_room": bool(row.pooja_room),
        },
    }
