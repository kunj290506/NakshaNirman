"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ---------- Room ----------
class RoomCreate(BaseModel):
    room_type: str
    quantity: int = 1
    desired_area: Optional[float] = None


class RoomOut(BaseModel):
    id: str
    room_type: str
    quantity: int
    desired_area: Optional[float]
    generated_polygon: Optional[list] = None

    class Config:
        from_attributes = True


# ---------- Project ----------
class ProjectCreate(BaseModel):
    session_id: str
    total_area: Optional[float] = None


class ProjectOut(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    total_area: Optional[float]
    status: str
    boundary_polygon: Optional[list] = None
    rooms: list[RoomOut] = []

    class Config:
        from_attributes = True


# ---------- Floor Plan Generation ----------
class GenerateRequest(BaseModel):
    project_id: str
    rooms: list[RoomCreate]
    total_area: Optional[float] = None
    boundary_polygon: Optional[list] = None


class GenerateResponse(BaseModel):
    project_id: str
    status: str
    plan: Optional[dict] = None
    dxf_url: Optional[str] = None


# ---------- Chat ----------
class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    project_id: str
    message: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    reply: str
    extracted_data: Optional[dict] = None
    should_generate: bool = False


# ---------- Boundary ----------
class BoundaryResponse(BaseModel):
    polygon: list
    area: float
    num_vertices: int


class DXFUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str = "uploaded"


class BoundaryExtractionResponse(BaseModel):
    file_id: str
    boundary_polygon: list
    area: float
    num_vertices: int
    perimeter: float
    is_valid: bool
    is_closed: bool
    is_self_intersecting: bool


class BuildableFootprintResponse(BaseModel):
    file_id: str
    boundary_polygon: list
    usable_polygon: list
    boundary_area: float
    usable_area: float
    setback_applied: float
    coverage_ratio: float
    preview_url: Optional[str] = None
    is_valid: bool


class SetbackRequest(BaseModel):
    setback: Optional[float] = None
    region: str = "india_mvp"


# ---------- Requirements ----------
class RequirementsIn(BaseModel):
    # Hard constraints
    floors: int
    bedrooms: int
    bathrooms: int
    kitchen: int
    max_area: float

    # Soft constraints (optional)
    balcony: bool = False
    parking: bool = False
    pooja_room: bool = False
    project_id: Optional[str] = None


class RequirementsOut(BaseModel):
    id: str
    project_id: Optional[str]
    floors: int
    bedrooms: int
    bathrooms: int
    kitchen: int
    max_area: float
    balcony: bool = False
    parking: bool = False
    pooja_room: bool = False

    class Config:
        from_attributes = True


# ---------- AI Design (Grok) ----------
class AIDesignRequest(BaseModel):
    message: str = Field(..., description="Natural language design requirements")
    project_id: Optional[str] = None
    total_area: Optional[float] = None


class AIDesignResponse(BaseModel):
    reasoning: str = ""
    rooms: list = []
    vastu_recommendations: list = []
    compliance_notes: list = []
    design_score: int = 0
    ready_to_generate: bool = False
    provider: str = "unknown"
    extracted_data: Optional[dict] = None


class AIReviewRequest(BaseModel):
    floor_plan: Optional[dict] = None
    project_id: Optional[str] = None


class AIReviewResponse(BaseModel):
    review_text: str = ""
    scores: dict = {}
    provider: str = "unknown"
