"""
Pydantic models for request/response validation.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


# ── Request ──────────────────────────────────────────────────
class PlanRequest(BaseModel):
    plot_width: float = Field(..., ge=20, le=200, description="Plot width in feet")
    plot_length: float = Field(..., ge=20, le=200, description="Plot length in feet")
    bedrooms: int = Field(..., ge=1, le=4, description="Number of bedrooms (BHK)")
    facing: str = Field(
        default="south",
        description="Road-facing direction",
        pattern="^(north|south|east|west)$",
    )
    extras: List[str] = Field(
        default_factory=list,
        description="Optional rooms: pooja, study, garage, balcony, store, utility, foyer, staircase",
    )
    bathrooms_target: int = Field(default=0, ge=0, le=8, description="Preferred total bathrooms (0 = auto)")
    floors: int = Field(default=1, ge=1, le=4, description="Target number of floors")
    design_style: str = Field(
        default="modern",
        description="Design style preference",
        pattern="^(modern|contemporary|traditional|minimal)$",
    )
    kitchen_preference: str = Field(
        default="semi_open",
        description="Kitchen layout preference",
        pattern="^(open|semi_open|closed)$",
    )
    parking_slots: int = Field(default=0, ge=0, le=4, description="Preferred parking slots")
    vastu_priority: int = Field(default=3, ge=1, le=5, description="Vastu strictness priority")
    natural_light_priority: int = Field(default=3, ge=1, le=5, description="Natural light priority")
    privacy_priority: int = Field(default=3, ge=1, le=5, description="Privacy priority")
    storage_priority: int = Field(default=3, ge=1, le=5, description="Storage priority")
    elder_friendly: bool = Field(default=False, description="Prioritize elder-friendly movement")
    work_from_home: bool = Field(default=False, description="Include work-from-home usability")
    notes: str = Field(default="", description="Additional custom design notes")
    city: str = Field(default="", description="City name for climate adaptation")
    state: str = Field(default="", description="State for regional rules")
    family_type: str = Field(
        default="nuclear",
        description="Family type: nuclear, joint, or couple",
        pattern="^(nuclear|joint|couple)$",
    )
    family_notes: str = Field(
        default="",
        description="Optional family description for richer context",
    )


# ── Room / Door / Window ────────────────────────────────────
class RoomData(BaseModel):
    id: str
    type: str
    label: str
    x: float
    y: float
    width: float
    height: float
    area: float
    zone: str = "public"
    band: int = 1
    exterior_walls: List[str] = Field(default_factory=list)
    color: str = "#F5F5F5"
    polygon: List[Point2D] = Field(default_factory=list)


class DoorData(BaseModel):
    id: str
    type: str = "interior"
    room_id: str = ""
    wall: str = "south"
    x: float = 0
    y: float = 0
    width: float = 3.5


class WindowData(BaseModel):
    id: str
    room_id: str = ""
    wall: str = "south"
    x: float = 0
    y: float = 0
    width: float = 4.0


# ── Plot info ────────────────────────────────────────────────
class PlotInfo(BaseModel):
    width: float
    length: float
    usable_width: float
    usable_length: float
    road_side: str = "south"
    setbacks: Dict[str, float] = Field(
        default_factory=lambda: {
            "front": 6.5,
            "rear": 5,
            "left": 3.5,
            "right": 3.5,
        }
    )
    boundary: List[Point2D] = Field(default_factory=list)


# ── Full plan response ──────────────────────────────────────
class PlanResponse(BaseModel):
    plot: PlotInfo
    rooms: List[RoomData]
    doors: List[DoorData] = []
    windows: List[WindowData] = []
    vastu_score: float = 0
    architect_note: str = ""
    dxf_url: Optional[str] = None
    generation_method: str = "bsp"
    vastu_issues: List[str] = Field(default_factory=list)
    adjacency_score: float = 0
    reasoning_trace: List[str] = Field(default_factory=list)
    architect_reasoning: Dict[str, Any] = Field(default_factory=dict)
