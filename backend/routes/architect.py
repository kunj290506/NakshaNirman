"""Architect routes backed by Perfcat + DeepSeek preprocessing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app_config import EXPORT_DIR
from services.cad_export import generate_dxf
from services.chat_agent import chat_reply
from services.multi_agent_orchestrator import generate_dynamic_layout
from services.openrouter_planner import prepare_floorplan_payload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/architect", tags=["architect"])

_PLANNER_PREPROCESS_TIMEOUT_SECONDS = 60.0
_DESIGN_GENERATION_TIMEOUT_SECONDS = 45.0


class ArchitectDesignRequest(BaseModel):
    plot_width: Optional[float] = Field(None, gt=0)
    plot_length: Optional[float] = Field(None, gt=0)
    total_area: Optional[float] = Field(None, gt=0)
    bedrooms: int = Field(2, ge=1, le=4)
    bathrooms: Optional[int] = Field(None, ge=1, le=6)
    floors: int = Field(1, ge=1, le=1)
    facing: str = Field("east")
    vastu: bool = Field(True)
    extras: List[str] = Field(default_factory=list)
    engine_mode: str = Field("gnn_advanced")
    rooms: List[Dict[str, Any]] = Field(default_factory=list)
    project_id: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    family_type: Optional[str] = "nuclear"
    plan_mode: str = Field("perfcat")
    previous_strategy: Optional[str] = None
    placement_constraints: List[Dict[str, Any]] = Field(default_factory=list)


class ArchitectDesignResponse(BaseModel):
    layout: Optional[Dict[str, Any]] = None
    dxf_url: Optional[str] = None
    design_score: Optional[Dict[str, Any]] = None
    architect_notes: List[str] = Field(default_factory=list)
    architect_narrative: Optional[str] = None
    layout_type: Optional[str] = None
    zoning_strategy: Optional[str] = None
    project_id: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class ArchitectChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = Field(default_factory=list)


async def _generate_dxf(project_id: str, layout: Dict[str, Any]) -> Optional[str]:
    try:
        dxf_filename = f"{project_id}_architect.dxf"
        dxf_path = os.path.join(str(EXPORT_DIR), dxf_filename)
        plan_data = {
            "rooms": layout.get("rooms", []),
            "boundary": layout.get("boundary", []),
            "plot": layout.get("plot", {}),
        }
        generate_dxf(plan_data, dxf_path)
        return f"/exports/{dxf_filename}"
    except Exception as exc:
        logger.warning("DXF generation failed: %s", exc)
        return None


@router.post("/design", response_model=ArchitectDesignResponse)
async def architect_design(req: ArchitectDesignRequest):
    if req.plot_width is not None and req.plot_width <= 15:
        raise HTTPException(status_code=400, detail="plot_width must be greater than 15 ft")
    if req.plot_length is not None and req.plot_length <= 15:
        raise HTTPException(status_code=400, detail="plot_length must be greater than 15 ft")
    if req.plot_width is None and req.plot_length is None and req.total_area is None:
        raise HTTPException(status_code=400, detail="Provide plot_width/plot_length or total_area")

    input_data = req.dict() if hasattr(req, "dict") else req.model_dump()
    input_data["bathrooms"] = req.bathrooms or req.bedrooms
    input_data["plan_mode"] = "perfcat"

    planner_meta = {
        "provider": "openrouter",
        "model": None,
        "used": False,
        "plan_mode": "perfcat",
        "warnings": [],
    }

    try:
        input_data, planner_meta = await asyncio.wait_for(
            asyncio.to_thread(
                prepare_floorplan_payload,
                input_data,
                "perfcat",
            ),
            timeout=_PLANNER_PREPROCESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("DeepSeek planner timed out after %.1fs", _PLANNER_PREPROCESS_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="DeepSeek planner timed out. Please retry.")
    except Exception as exc:
        logger.warning("DeepSeek planner failed during pre-processing", exc_info=True)
        raise HTTPException(status_code=503, detail=f"DeepSeek planner unavailable: {exc}")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_dynamic_layout, input_data),
            timeout=_DESIGN_GENERATION_TIMEOUT_SECONDS,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        planner_warnings = list(planner_meta.get("warnings") or [])
        if planner_meta.get("used") or planner_warnings:
            result["planner"] = {
                "mode": planner_meta.get("plan_mode"),
                "provider": planner_meta.get("provider"),
                "model": planner_meta.get("model"),
            }
        if planner_warnings:
            existing = list(result.get("warnings") or [])
            for warning in planner_warnings:
                if warning not in existing:
                    existing.append(warning)
            result["warnings"] = existing

        dxf_url = None
        if req.project_id:
            dxf_url = await _generate_dxf(req.project_id, result)

        return ArchitectDesignResponse(
            layout=result,
            dxf_url=dxf_url,
            design_score=result.get("design_score") or result.get("scores"),
            architect_notes=result.get("architect_notes", []),
            architect_narrative=result.get("architect_narrative"),
            layout_type=result.get("layout_type") or result.get("layout_strategy"),
            zoning_strategy=result.get("zoning_strategy") or result.get("layout_strategy"),
            project_id=req.project_id,
            warnings=result.get("warnings", []),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Design generation timed out. Please retry.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Architect design failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/redesign", response_model=ArchitectDesignResponse)
async def architect_redesign(req: ArchitectDesignRequest):
    if req.plot_width is not None and req.plot_width <= 15:
        raise HTTPException(status_code=400, detail="plot_width must be greater than 15 ft")
    if req.plot_length is not None and req.plot_length <= 15:
        raise HTTPException(status_code=400, detail="plot_length must be greater than 15 ft")
    if req.plot_width is None and req.plot_length is None and req.total_area is None:
        raise HTTPException(status_code=400, detail="Provide plot_width/plot_length or total_area")

    input_data = req.dict() if hasattr(req, "dict") else req.model_dump()
    input_data["bathrooms"] = req.bathrooms or req.bedrooms
    input_data["previous_strategy"] = req.previous_strategy
    input_data["plan_mode"] = "perfcat"

    planner_meta = {
        "provider": "openrouter",
        "model": None,
        "used": False,
        "plan_mode": "perfcat",
        "warnings": [],
    }

    try:
        input_data, planner_meta = await asyncio.wait_for(
            asyncio.to_thread(
                prepare_floorplan_payload,
                input_data,
                "perfcat",
            ),
            timeout=_PLANNER_PREPROCESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("DeepSeek planner timed out after %.1fs", _PLANNER_PREPROCESS_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="DeepSeek planner timed out. Please retry.")
    except Exception as exc:
        logger.warning("DeepSeek planner failed during pre-processing", exc_info=True)
        raise HTTPException(status_code=503, detail=f"DeepSeek planner unavailable: {exc}")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(generate_dynamic_layout, input_data),
            timeout=_DESIGN_GENERATION_TIMEOUT_SECONDS,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        planner_warnings = list(planner_meta.get("warnings") or [])
        if planner_meta.get("used") or planner_warnings:
            result["planner"] = {
                "mode": planner_meta.get("plan_mode"),
                "provider": planner_meta.get("provider"),
                "model": planner_meta.get("model"),
            }
        if planner_warnings:
            existing = list(result.get("warnings") or [])
            for warning in planner_warnings:
                if warning not in existing:
                    existing.append(warning)
            result["warnings"] = existing

        dxf_url = None
        if req.project_id:
            dxf_url = await _generate_dxf(req.project_id, result)

        return ArchitectDesignResponse(
            layout=result,
            dxf_url=dxf_url,
            design_score=result.get("design_score") or result.get("scores"),
            architect_notes=result.get("architect_notes", []),
            architect_narrative=result.get("architect_narrative"),
            layout_type=result.get("layout_type") or result.get("layout_strategy"),
            zoning_strategy=result.get("zoning_strategy") or result.get("layout_strategy"),
            project_id=req.project_id,
            warnings=result.get("warnings", []),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Design generation timed out. Please retry.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Architect redesign failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat")
async def architect_chat(req: ArchitectChatRequest):
    """
    Chat endpoint used by frontend chat mode.

    Returns assistant text. When requirements are complete and user confirms,
    assistant text contains a line starting with: GENERATE_PLAN: { ... }
    """
    try:
        reply = await chat_reply(req.message, req.history)

        generate_payload = None
        if "GENERATE_PLAN:" in reply:
            token = reply.split("GENERATE_PLAN:", 1)[1].strip()
            try:
                generate_payload = json.loads(token)
            except json.JSONDecodeError:
                generate_payload = None

        return {
            "reply": reply,
            "generate_payload": generate_payload,
        }
    except Exception as exc:
        logger.exception("Architect chat failed")
        raise HTTPException(status_code=500, detail=str(exc))
