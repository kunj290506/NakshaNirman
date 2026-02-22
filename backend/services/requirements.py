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
