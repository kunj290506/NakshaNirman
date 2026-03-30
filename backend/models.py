"""
Pydantic models for request/response validation.
"""
from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


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
        description="Optional rooms: pooja, study, garage, balcony, store",
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


# ── Full plan response ──────────────────────────────────────
class PlanResponse(BaseModel):
    plot: PlotInfo
    rooms: List[RoomData]
    doors: List[DoorData] = []
    windows: List[WindowData] = []
    vastu_score: float = 0
    architect_note: str = ""
    dxf_url: Optional[str] = None
