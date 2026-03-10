"""
Architect Engine route.

Provides REST and WebSocket endpoints for floor plan generation
using the unified engine registry (BSP production engine by default).

Endpoints:
  POST /api/architect/design     — Generate floor plan
  POST /api/architect/redesign   — Generate new layout (same requirements)
  WS   /api/architect/ws         — Interactive chat + design pipeline
"""

import json
import os
import logging
import random
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from services.engine_registry import generate as engine_generate, DEFAULT_ENGINE
from services.cad_export import generate_dxf
from services.chat import chat_with_groq
from services.design_intelligence import (
    build_design_brief, score_layout, generate_architect_narrative,
)
from config import EXPORT_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/architect", tags=["architect"])


# ──────────── Request/Response Models ────────────

class ArchitectDesignRequest(BaseModel):
    """Request body for multi-factor architectural design."""
    total_area: float = Field(1200, description="Total plot area in sq ft")
    plot_width: Optional[float] = None
    plot_length: Optional[float] = None
    bedrooms: int = Field(2, ge=1, le=10)
    bathrooms: int = Field(1, ge=1, le=10)
    floors: int = Field(1, ge=1, le=4)
    extras: Optional[List[str]] = []
    rooms: Optional[List[Dict]] = None
    boundary_polygon: Optional[List[List[float]]] = None
    front_door: Optional[List[float]] = None
    project_id: Optional[str] = None
    previous_strategy: Optional[str] = None


class ArchitectRedesignRequest(ArchitectDesignRequest):
    """Request body for redesign (new layout, same requirements)."""
    previous_strategy: Optional[str] = None


class ArchitectDesignResponse(BaseModel):
    """Response for architectural design."""
    engine: str = "bsp"
    method: str = "architectural_reasoning"
    explanation: str = ""
    layout: Optional[Dict] = None
    validation: Optional[Dict] = None
    dxf_url: Optional[str] = None
    error: Optional[str] = None
    suggestion: Optional[str] = None
    design_score: Optional[Dict] = None
    architect_narrative: Optional[str] = None
    zoning_strategy: Optional[str] = None


# ──────────── DXF Generation Helper ────────────

async def _generate_dxf(project_id: str, layout: Dict) -> Optional[str]:
    """Generate DXF file and return URL."""
    try:
        dxf_filename = f"{project_id}_architect.dxf"
        dxf_path = os.path.join(str(EXPORT_DIR), dxf_filename)

        # Convert layout to format expected by cad_export
        plan_data = {
            "rooms": layout.get("rooms", []),
            "boundary": layout.get("boundary", []),
            "plot": layout.get("plot", {}),
        }
        generate_dxf(plan_data, dxf_path)

        return f"/exports/{dxf_filename}"
    except Exception as e:
        logger.warning(f"DXF generation failed: {e}")
        return None


# ──────────── REST Endpoints ────────────

@router.post("/design", response_model=ArchitectDesignResponse)
async def architect_design(req: ArchitectDesignRequest):
    """
    Generate a floor plan using the Multi-Factor Architectural Reasoning Engine.

    This engine uses 8 architectural design factors:
    1. Functional Zoning
    2. Privacy Gradient
    3. Area Distribution
    4. Circulation Strategy
    5. Geometry Rules
    6. Ventilation & Light
    7. Structural Logic
    8. Human Comfort
    """
    try:
        input_data = {
            "total_area": req.total_area,
            "plot_width": req.plot_width,
            "plot_length": req.plot_length,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "floors": req.floors,
            "extras": req.extras or [],
            "rooms": req.rooms,
            "boundary_polygon": req.boundary_polygon,
        }

        # Enrich input with design intelligence
        brief = build_design_brief(input_data)
        engine_input = brief["engine_input"]

        result = engine_generate(DEFAULT_ENGINE, engine_input)

        if "error" in result:
            return ArchitectDesignResponse(
                error=result["error"],
                suggestion=result.get("suggestion"),
            )

        dxf_url = None
        layout = result.get("layout", {})
        if req.project_id and layout.get("rooms"):
            dxf_url = await _generate_dxf(req.project_id, layout)

        # Score the layout
        pw = engine_input.get("plot_width") or 30
        pl = engine_input.get("plot_length") or 40
        score_data = score_layout(layout.get("rooms", []), pw, pl, engine_input)
        narrative = generate_architect_narrative(engine_input, score_data, pw, pl)

        used_strategy = layout.get("zoning_strategy") or random.choice(
            ["linear", "L_shape", "cross_vent", "compact"]
        )

        return ArchitectDesignResponse(
            engine=result.get("engine", DEFAULT_ENGINE),
            method="architectural_reasoning",
            explanation=result.get("explanation", ""),
            layout=layout,
            validation=result.get("validation"),
            dxf_url=dxf_url,
            design_score=score_data,
            architect_narrative=narrative,
            zoning_strategy=used_strategy,
        )

    except Exception as e:
        logger.exception("Architect design endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redesign", response_model=ArchitectDesignResponse)
async def architect_redesign(req: ArchitectRedesignRequest):
    """
    Generate a NEW floor plan with the same requirements but different strategy.

    Changes zoning strategy, circulation type, and spatial arrangement
    to produce a completely different layout.
    """
    try:
        input_data = {
            "total_area": req.total_area,
            "plot_width": req.plot_width,
            "plot_length": req.plot_length,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "floors": req.floors,
            "extras": req.extras or [],
            "rooms": req.rooms,
            "boundary_polygon": req.boundary_polygon,
        }

        # Enrich input with design intelligence
        brief = build_design_brief(input_data)
        engine_input = brief["engine_input"]

        # Pass previous strategy so engine picks a different one
        if req.previous_strategy:
            engine_input["_previous_strategy"] = req.previous_strategy

        result = engine_generate(DEFAULT_ENGINE, engine_input)

        if "error" in result:
            return ArchitectDesignResponse(
                error=result["error"],
                suggestion=result.get("suggestion"),
            )

        dxf_url = None
        layout = result.get("layout", {})
        if req.project_id and layout.get("rooms"):
            dxf_url = await _generate_dxf(req.project_id, layout)

        # Score the layout
        pw = engine_input.get("plot_width") or 30
        pl = engine_input.get("plot_length") or 40
        score_data = score_layout(layout.get("rooms", []), pw, pl, engine_input)
        narrative = generate_architect_narrative(engine_input, score_data, pw, pl)

        used_strategy = layout.get("zoning_strategy") or random.choice(
            ["linear", "L_shape", "cross_vent", "compact"]
        )

        return ArchitectDesignResponse(
            engine=result.get("engine", DEFAULT_ENGINE),
            method="architectural_reasoning",
            explanation=result.get("explanation", ""),
            layout=layout,
            validation=result.get("validation"),
            dxf_url=dxf_url,
            design_score=score_data,
            architect_narrative=narrative,
            zoning_strategy=used_strategy,
        )

    except Exception as e:
        logger.exception("Architect redesign endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────── REST Chat Endpoint ────────────

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict]] = []
    project_id: Optional[str] = None

@router.post("/chat")
async def architect_chat_rest(req: ChatRequest):
    """REST fallback for chat when WebSocket is not available."""
    try:
        result = await chat_with_groq(req.message, (req.history or [])[-8:])
        return {
            "reply": result.get("reply", ""),
            "extracted_data": result.get("extracted_data"),
            "should_generate": result.get("should_generate", False),
            "mode": (result.get("extracted_data") or {}).get("mode", "collecting"),
        }
    except Exception as e:
        logger.error(f"REST chat error: {e}")
        return {
            "reply": "I'm having trouble connecting to the AI service. Please try again.",
            "mode": "chat",
        }


# ──────────── Groq→Engine Translator ────────────


# ──────────── WebSocket Endpoint ────────────

@router.websocket("/ws")
async def architect_ws(websocket: WebSocket):
    """
    WebSocket endpoint for interactive architectural design.
    Uses Groq AI for intelligent conversation. Falls back to rule-based if no key.
    """
    await websocket.accept()

    history: List[Dict] = []
    collected: Dict = {}

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg_data = json.loads(raw)
            except json.JSONDecodeError:
                msg_data = {"message": raw}

            user_text = msg_data.get("message", "").strip()
            project_id = msg_data.get("project_id")

            if not user_text:
                continue

            # Always run regex extraction as numeric fallback
            _extract_from_text(user_text, collected)

            # Merge any structured data sent from frontend
            if msg_data.get("data"):
                for key in ("plot_width", "plot_length", "total_area",
                            "bedrooms", "bathrooms", "floors", "extras", "rooms"):
                    if msg_data["data"].get(key) is not None:
                        collected[key] = msg_data["data"][key]

            # Add user turn to history
            history.append({"role": "user", "content": user_text})

            # Call Groq AI with conversation history
            try:
                groq_result = await chat_with_groq(user_text, history[:-1][-8:])
            except Exception as groq_err:
                logger.warning(f"Groq error: {groq_err}")
                groq_result = {
                    "reply": "I'm having trouble reaching the AI. Please try again.",
                    "extracted_data": None,
                    "should_generate": False,
                }

            extracted = groq_result.get("extracted_data") or {}
            groq_mode = extracted.get("mode", "collecting")
            reply_text = groq_result.get("reply", "")

            # Record assistant turn
            history.append({"role": "assistant", "content": reply_text})

            # Update collected from Groq's collected_so_far
            if extracted.get("collected_so_far"):
                csf = extracted["collected_so_far"]
                for key in ("plot_width", "plot_depth", "total_area", "bedrooms", "bathrooms", "floors"):
                    if csf.get(key) is not None:
                        mapped_key = "plot_length" if key == "plot_depth" else key
                        collected[mapped_key] = csf[key]
                if csf.get("extras"):
                    collected["extras"] = csf["extras"]

            if groq_mode in ("designing", "modifying"):
                # Groq has enough info — translate and generate
                engine_input = _groq_to_engine_input(extracted, collected)

                if not engine_input.get("total_area"):
                    await websocket.send_text(json.dumps({
                        "mode": "chat",
                        "reply": "I need your plot dimensions to generate the plan. What is the width and length in feet?",
                        "collected": collected,
                    }))
                    continue

                # Enrich with design intelligence
                brief = build_design_brief(engine_input)
                engine_input = brief["engine_input"]

                try:
                    result = engine_generate(DEFAULT_ENGINE, engine_input)
                except Exception as eng_err:
                    logger.error(f"Engine error: {eng_err}")
                    await websocket.send_text(json.dumps({
                        "mode": "error",
                        "reply": "Plan generation failed. Please check your plot dimensions and try again.",
                    }))
                    continue

                if "error" in result:
                    await websocket.send_text(json.dumps({
                        "mode": "error",
                        "reply": result["error"] + (" " + result.get("suggestion", "") if result.get("suggestion") else ""),
                    }))
                    continue

                layout = result.get("layout", {})

                # Score the layout
                pw = engine_input.get("plot_width") or 30
                pl = engine_input.get("plot_length") or 40
                score_data = score_layout(layout.get("rooms", []), pw, pl, engine_input)
                narrative = generate_architect_narrative(engine_input, score_data, pw, pl)

                architect_note = narrative or extracted.get("architect_note", "") or result.get("explanation", "Your floor plan is ready.")

                dxf_url = None
                if project_id and layout.get("rooms"):
                    try:
                        dxf_url = await _generate_dxf(project_id, layout)
                    except Exception:
                        pass

                rooms_list = [
                    {"room_type": r["room_type"], "quantity": 1}
                    for r in layout.get("rooms", [])
                ]

                await websocket.send_text(json.dumps({
                    "mode": "design",
                    "reply": architect_note,
                    "layout": layout,
                    "validation": result.get("validation"),
                    "dxf_url": dxf_url,
                    "should_generate": True,
                    "design_score": score_data,
                    "architect_notes": brief.get("architect_notes", []),
                    "extracted_data": {
                        "rooms": rooms_list,
                        "total_area": engine_input.get("total_area"),
                        "bedrooms": engine_input.get("bedrooms"),
                        "bathrooms": engine_input.get("bathrooms"),
                        "extras": engine_input.get("extras", []),
                        "floors": engine_input.get("floors", 1),
                    },
                }))

            else:
                # Still collecting — send Groq's question to user
                if not reply_text:
                    reply_text = "Could you tell me your plot size and how many bedrooms you need?"

                await websocket.send_text(json.dumps({
                    "mode": "chat",
                    "reply": reply_text,
                    "collected": collected,
                    "ready": False,
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket handler error")
        try:
            await websocket.send_text(json.dumps({
                "mode": "error",
                "reply": "An unexpected error occurred. Please refresh the page.",
            }))
        except Exception:
            pass


def _groq_to_engine_input(groq_json: dict, fallback: dict) -> dict:
    """
    Translate Groq's designing-mode JSON into the flat dict engine_generate() needs.
    Merges Groq's rich output with regex-collected fallback values.
    """
    result = dict(fallback)

    plot = groq_json.get("plot", {})
    if plot.get("width"): result["plot_width"] = float(plot["width"])
    if plot.get("depth"): result["plot_length"] = float(plot["depth"])
    if plot.get("total_area"): result["total_area"] = float(plot["total_area"])

    # Derive total_area from dimensions if only dimensions provided
    if not result.get("total_area") and result.get("plot_width") and result.get("plot_length"):
        result["total_area"] = result["plot_width"] * result["plot_length"]

    rooms = groq_json.get("rooms", [])
    if rooms:
        bedroom_types = {"master_bedroom", "bedroom"}
        bath_types = {"bathroom", "toilet", "wc"}
        core_types = bedroom_types | bath_types | {"living", "kitchen", "dining", "entrance"}

        bedrooms = sum(1 for r in rooms if r.get("type") in bedroom_types)
        bathrooms = sum(1 for r in rooms if r.get("type") in bath_types)
        extras = [r["type"] for r in rooms if r.get("type") and r["type"] not in core_types]

        if bedrooms > 0: result["bedrooms"] = bedrooms
        if bathrooms > 0: result["bathrooms"] = bathrooms
        if extras: result["extras"] = list(set(extras))

    result.setdefault("bedrooms", 2)
    result.setdefault("bathrooms", 1)
    result.setdefault("floors", 1)
    result.setdefault("extras", [])

    return result


import re as _re

def _extract_from_text(text: str, reqs: Dict) -> None:
    """Extract plot dimensions and room counts from natural language."""
    text_lower = text.lower()

    m = _re.search(r'(\d+)\s*[xX×by]+\s*(\d+)', text_lower)
    if m:
        w, l = float(m.group(1)), float(m.group(2))
        reqs["plot_width"] = min(w, l)
        reqs["plot_length"] = max(w, l)
        reqs["total_area"] = w * l

    m = _re.search(r'(\d+)\s*(?:sq\.?\s*ft|sqft|square\s*feet)', text_lower)
    if m:
        reqs["total_area"] = float(m.group(1))

    m = _re.search(r'(\d+)\s*(?:bhk|bed(?:room)?s?)', text_lower)
    if m:
        reqs["bedrooms"] = int(m.group(1))

    m = _re.search(r'(\d+)\s*bath(?:room)?s?', text_lower)
    if m:
        reqs["bathrooms"] = int(m.group(1))

    m = _re.search(r'(\d+)\s*(?:floor|stor(?:ey|y|ies))', text_lower)
    if m:
        reqs["floors"] = int(m.group(1))
