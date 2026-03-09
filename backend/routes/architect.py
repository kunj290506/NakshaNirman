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
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from services.engine_registry import generate as engine_generate, DEFAULT_ENGINE
from services.cad_export import generate_dxf
from services.chat import chat_with_groq
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

        result = engine_generate(DEFAULT_ENGINE, input_data)

        if "error" in result:
            return ArchitectDesignResponse(
                error=result["error"],
                suggestion=result.get("suggestion"),
            )

        dxf_url = None
        layout = result.get("layout", {})
        if req.project_id and layout.get("rooms"):
            dxf_url = await _generate_dxf(req.project_id, layout)

        return ArchitectDesignResponse(
            engine=result.get("engine", DEFAULT_ENGINE),
            method="architectural_reasoning",
            explanation=result.get("explanation", ""),
            layout=layout,
            validation=result.get("validation"),
            dxf_url=dxf_url,
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

        result = engine_generate(DEFAULT_ENGINE, input_data)

        if "error" in result:
            return ArchitectDesignResponse(
                error=result["error"],
                suggestion=result.get("suggestion"),
            )

        dxf_url = None
        layout = result.get("layout", {})
        if req.project_id and layout.get("rooms"):
            dxf_url = await _generate_dxf(req.project_id, layout)

        return ArchitectDesignResponse(
            engine=result.get("engine", DEFAULT_ENGINE),
            method="architectural_reasoning",
            explanation=result.get("explanation", ""),
            layout=layout,
            validation=result.get("validation"),
            dxf_url=dxf_url,
        )

    except Exception as e:
        logger.exception("Architect redesign endpoint error")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────── WebSocket Endpoint ────────────

@router.websocket("/ws")
async def architect_ws(websocket: WebSocket):
    """
    WebSocket endpoint for interactive architectural design.

    Supports:
    - Chat mode: Collect requirements conversationally
    - Design mode: Generate plan when ready
    - Redesign mode: Generate fresh layout with same requirements
    """
    await websocket.accept()

    history: List[Dict] = []
    collected_requirements: Dict = {}
    last_strategy: Optional[str] = None

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg_data = json.loads(raw)
            except json.JSONDecodeError:
                msg_data = {"message": raw}

            user_text = msg_data.get("message", "")
            project_id = msg_data.get("project_id")

            # Parse structured data from message
            is_generate = msg_data.get("generatePlan", False)

            # Merge structured data into collected requirements
            if msg_data.get("data"):
                for key in ("plot_width", "plot_length", "total_area",
                            "bedrooms", "bathrooms", "floors", "extras", "rooms"):
                    if msg_data["data"].get(key) is not None:
                        collected_requirements[key] = msg_data["data"][key]

            # Simple NL parsing for requirements
            _extract_from_text(user_text, collected_requirements)

            # Detect redesign intent
            is_redesign = any(w in user_text.lower() for w in
                            ["redesign", "different", "another", "new layout", "regenerate"])
            if not is_generate:
                is_generate = any(w in user_text.lower() for w in
                                ["generate", "create", "make", "design", "build"])

            history.append({"role": "user", "content": user_text})

            # Check completeness
            has_area = (
                collected_requirements.get("total_area") or
                (collected_requirements.get("plot_width") and
                 collected_requirements.get("plot_length"))
            )
            has_beds = collected_requirements.get("bedrooms") is not None
            has_baths = collected_requirements.get("bathrooms") is not None
            is_complete = has_area and has_beds and has_baths

            if is_redesign and is_complete:
                # REDESIGN MODE
                result = engine_generate(DEFAULT_ENGINE, collected_requirements)
                if "error" in result:
                    await websocket.send_text(json.dumps({
                        "mode": "error",
                        "reply": result["error"],
                        "suggestion": result.get("suggestion"),
                    }))
                else:
                    layout = result.get("layout", {})
                    last_strategy = layout.get("zoning_strategy")

                    dxf_url = None
                    if project_id and layout.get("rooms"):
                        try:
                            dxf_url = await _generate_dxf(project_id, layout)
                        except Exception:
                            pass

                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": result.get("explanation", ""),
                        "layout": layout,
                        "validation": result.get("validation"),
                        "dxf_url": dxf_url,
                        "should_generate": True,
                        "extracted_data": {
                            "rooms": [
                                {"room_type": r["room_type"], "quantity": 1}
                                for r in layout.get("rooms", [])
                            ],
                            "total_area": collected_requirements.get("total_area"),
                            "ready_to_generate": True,
                        },
                    }))

                history.append({
                    "role": "assistant",
                    "content": result.get("explanation", "New plan generated."),
                })

            elif (is_generate or is_complete) and is_complete:
                # DESIGN MODE
                result = engine_generate(DEFAULT_ENGINE, collected_requirements)

                if "error" in result:
                    await websocket.send_text(json.dumps({
                        "mode": "error",
                        "reply": result["error"],
                        "suggestion": result.get("suggestion"),
                    }))
                else:
                    layout = result.get("layout", {})
                    last_strategy = layout.get("zoning_strategy")

                    dxf_url = None
                    if project_id and layout.get("rooms"):
                        try:
                            dxf_url = await _generate_dxf(project_id, layout)
                        except Exception:
                            pass

                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": result.get("explanation", ""),
                        "layout": layout,
                        "validation": result.get("validation"),
                        "dxf_url": dxf_url,
                        "should_generate": True,
                        "extracted_data": {
                            "rooms": [
                                {"room_type": r["room_type"], "quantity": 1}
                                for r in layout.get("rooms", [])
                            ],
                            "total_area": collected_requirements.get("total_area"),
                            "ready_to_generate": True,
                        },
                    }))

                history.append({
                    "role": "assistant",
                    "content": result.get("explanation", "Plan generated."),
                })

            else:
                # CHAT MODE — use Groq AI for intelligent conversation
                groq_result = await chat_with_groq(user_text, history)

                extracted = groq_result.get("extracted_data")
                should_gen = groq_result.get("should_generate", False)

                if should_gen and extracted and extracted.get("mode") in ("designing", "modifying"):
                    # Groq says it has enough info — translate and generate
                    engine_input = _groq_to_engine(extracted, collected_requirements)
                    collected_requirements.update(engine_input)

                    result = engine_generate(DEFAULT_ENGINE, collected_requirements)

                    if "error" in result:
                        await websocket.send_text(json.dumps({
                            "mode": "error",
                            "reply": result["error"],
                            "suggestion": result.get("suggestion"),
                        }))
                    else:
                        layout = result.get("layout", {})
                        last_strategy = layout.get("zoning_strategy")

                        dxf_url = None
                        if project_id and layout.get("rooms"):
                            try:
                                dxf_url = await _generate_dxf(project_id, layout)
                            except Exception:
                                pass

                        architect_note = extracted.get("architect_note", result.get("explanation", ""))

                        await websocket.send_text(json.dumps({
                            "mode": "design",
                            "reply": architect_note,
                            "layout": layout,
                            "validation": result.get("validation"),
                            "dxf_url": dxf_url,
                            "should_generate": True,
                            "extracted_data": {
                                "rooms": [
                                    {"room_type": r["room_type"], "quantity": 1}
                                    for r in layout.get("rooms", [])
                                ],
                                "total_area": collected_requirements.get("total_area"),
                                "ready_to_generate": True,
                            },
                        }))

                    history.append({
                        "role": "assistant",
                        "content": extracted.get("architect_note", "Plan generated."),
                    })
                else:
                    # Groq is still collecting — send its question to user
                    if extracted and extracted.get("mode") == "collecting":
                        reply = extracted.get("question", groq_result.get("reply", "Could you tell me more?"))
                        context = extracted.get("context_understood", "")
                        if context and reply:
                            reply = f"{context}\n\n{reply}"
                    else:
                        reply = groq_result.get("reply", "Please tell me about your plot size and room requirements.")

                    # Update collected_requirements from Groq's collected_so_far
                    if extracted and extracted.get("collected_so_far"):
                        csf = extracted["collected_so_far"]
                        for key in ("plot_width", "plot_depth", "total_area", "bedrooms", "bathrooms", "floors"):
                            if csf.get(key) is not None:
                                mapped_key = "plot_length" if key == "plot_depth" else key
                                collected_requirements[mapped_key] = csf[key]
                        if csf.get("extras"):
                            collected_requirements["extras"] = csf["extras"]

                    await websocket.send_text(json.dumps({
                        "mode": "chat",
                        "reply": reply,
                        "collected": collected_requirements,
                        "ready": is_complete,
                    }))

                    history.append({"role": "assistant", "content": reply})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "mode": "error",
                "reply": f"Engine error: {str(e)}",
                "error": str(e),
            }))
        except Exception:
            pass


def _groq_to_engine(groq_json: dict, existing: dict) -> dict:
    """Translate Groq's designing-mode JSON into engine input format."""
    result = dict(existing)

    # Extract from Groq's plot object
    plot = groq_json.get("plot", {})
    if plot.get("width"): result["plot_width"] = plot["width"]
    if plot.get("depth"): result["plot_length"] = plot["depth"]
    if plot.get("total_area"): result["total_area"] = plot["total_area"]

    # Count rooms from Groq's rooms array
    rooms = groq_json.get("rooms", [])
    if rooms:
        bedrooms = sum(1 for r in rooms if r.get("type") in ("master_bedroom", "bedroom"))
        bathrooms = sum(1 for r in rooms if r.get("type") in ("bathroom", "toilet"))
        extras = [r["type"] for r in rooms
                  if r.get("type") not in ("master_bedroom", "bedroom", "bathroom",
                                           "toilet", "living", "kitchen", "dining", "passage")]

        if bedrooms > 0: result["bedrooms"] = bedrooms
        if bathrooms > 0: result["bathrooms"] = bathrooms
        if extras: result["extras"] = extras

        # Pass Groq's room specs as constraints for the engine
        result["groq_room_specs"] = [
            {
                "room_type": r.get("type"),
                "target_area": r.get("target_area"),
                "min_area": r.get("min_area"),
                "preferred_width": r.get("preferred_width"),
                "preferred_depth": r.get("preferred_depth"),
                "zone": r.get("zone"),
                "vastu_zone": r.get("vastu_zone"),
            }
            for r in rooms if r.get("type")
        ]

    # Facing/orientation from Groq
    if plot.get("facing"): result["facing"] = plot["facing"]
    if groq_json.get("design_strategy", {}).get("entrance_position"):
        result["entrance_position"] = groq_json["design_strategy"]["entrance_position"]

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
