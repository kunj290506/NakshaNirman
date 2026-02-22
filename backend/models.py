"""SQLAlchemy ORM models matching the data model in final_doc.md."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from database import Base
import enum


def generate_uuid():
    return str(uuid.uuid4())


class ProjectStatus(enum.Enum):
    DRAFTING = "drafting"
    PROCESSING = "processing"
    COMPLETED = "completed"


class RoomType(enum.Enum):
    MASTER_BEDROOM = "master_bedroom"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    LIVING = "living"
    DINING = "dining"
    STUDY = "study"
    GARAGE = "garage"
    GARDEN = "garden"
    HALLWAY = "hallway"
    BALCONY = "balcony"
    POOJA = "pooja"
    STORE = "store"
    OTHER = "other"


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    total_area = Column(Float, nullable=True)
    boundary_polygon = Column(Text, nullable=True)  # JSON string
    status = Column(SAEnum(ProjectStatus), default=ProjectStatus.DRAFTING)
    chat_history = Column(Text, nullable=True)  # JSON string
    generated_plan = Column(Text, nullable=True)  # JSON string of generated layout
    dxf_path = Column(String, nullable=True)
    model3d_path = Column(String, nullable=True)

    rooms = relationship("Room", back_populates="project", cascade="all, delete-orphan")
    boundary_uploads = relationship("BoundaryUpload", back_populates="project", cascade="all, delete-orphan")
    requirements = relationship("Requirements", back_populates="project", cascade="all, delete-orphan")


class Room(Base):
    __tablename__ = "rooms"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    room_type = Column(SAEnum(RoomType), nullable=False)
    quantity = Column(Integer, default=1)
    desired_area = Column(Float, nullable=True)
    generated_polygon = Column(Text, nullable=True)  # JSON string

    project = relationship("Project", back_populates="rooms")


class BoundaryUpload(Base):
    __tablename__ = "boundary_uploads"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "image" or "dxf"
    processed_polygon = Column(Text, nullable=True)  # JSON string of boundary polygon
    usable_polygon = Column(Text, nullable=True)  # JSON string of buildable footprint
    setback_applied = Column(Float, nullable=True)  # setback distance in meters
    boundary_area = Column(Float, nullable=True)  # total plot area
    usable_area = Column(Float, nullable=True)  # buildable area after setback
    preview_path = Column(String, nullable=True)  # path to preview image

    project = relationship("Project", back_populates="boundary_uploads")


class Requirements(Base):
    __tablename__ = "requirements"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)

    # Hard constraints
    floors = Column(Integer, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Integer, nullable=False)
    kitchen = Column(Integer, nullable=False)
    max_area = Column(Float, nullable=False)

    # Soft constraints
    balcony = Column(Integer, nullable=False, default=0)
    parking = Column(Integer, nullable=False, default=0)
    pooja_room = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    project = relationship("Project", back_populates="requirements")
