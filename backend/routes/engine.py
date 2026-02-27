"""
Architectural Planning Engine route.

Single unified endpoint that auto-detects mode and routes to the
appropriate engine function (CHAT / FORM / DESIGN / VALIDATION).
"""

import json
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict
from database import get_db, async_session
from models import Project, ProjectStatus
from services.arch_engine import (
    process, detect_mode, EngineMode,
    chat_response, form_validate, design_generate, validate_layout,
)
from services.floorplan import generate_floor_plan
from services.cad_export import generate_dxf
from config import EXPORT_DIR
import os

router = APIRouter(prefix="/api/engine", tags=["engine"])


# ---------- Request/Response Models ----------

class EngineRequest(BaseModel):
    """Unified engine request — accepts any input type."""
    message: Optional[str] = None
    data: Optional[dict] = None
    history: Optional[List[Dict]] = []
    project_id: Optional[str] = None
    generatePlan: Optional[bool] = False


class EngineResponse(BaseModel):
    """Unified engine response."""
    mode: str
    reply: Optional[str] = None
    layout: Optional[dict] = None
    validation: Optional[dict] = None
    error: Optional[str] = None
    suggestion: Optional[str] = None
    collected: Optional[dict] = None
    ready: Optional[bool] = False
    dxf_url: Optional[str] = None
    missing_fields: Optional[list] = None


# ---------- REST Endpoint ----------

@router.post("/process", response_model=EngineResponse)
async def engine_process(req: EngineRequest, db: AsyncSession = Depends(get_db)):
    """
    Unified architectural engine endpoint.

    Auto-detects input mode and processes accordingly:
      - Chat text → CHAT MODE (collects requirements)
      - Structured JSON → FORM MODE (validates data)
      - generatePlan=true → DESIGN MODE (generates layout)
      - Layout with room positions → VALIDATION MODE

    Returns deterministic, architecturally valid, CAD-ready JSON.
    """
    # Build input for the engine
    if req.data:
        input_data = dict(req.data)
        if req.generatePlan:
            input_data["generatePlan"] = True
        if req.message:
            input_data["message"] = req.message
    elif req.message:
        # Check if message says "generate plan" and we have history data
        if req.generatePlan or "generate plan" in (req.message or "").lower():
            # Try to extract requirements from history
            collected = _extract_from_history(req.history or [])
            if collected.get("complete"):
                input_data = {
                    "generatePlan": True,
                    "plot_width": collected.get("plot_width"),
                    "plot_length": collected.get("plot_length"),
                    "total_area": collected.get("total_area"),
                    "bedrooms": collected.get("bedrooms"),
                    "bathrooms": collected.get("bathrooms"),
                    "floors": collected.get("floors", 1),
                    "extras": collected.get("extras", []),
                }
            else:
                input_data = req.message
        else:
            input_data = req.message
    else:
        raise HTTPException(status_code=400, detail="Either 'message' or 'data' is required")

    # Process through engine
    result = process(input_data, req.history or [])

    # Build response
    mode = result.get("mode", "error")
    if isinstance(mode, EngineMode):
        mode = mode.value

    response = EngineResponse(mode=mode)

    if "error" in result:
        response.error = result["error"]
        response.suggestion = result.get("suggestion")
        response.missing_fields = result.get("missing_fields")
        return response

    if mode == "chat":
        response.reply = result.get("reply", "")
        response.collected = result.get("collected")
        response.ready = result.get("ready", False)

    elif mode == "form":
        if result.get("valid"):
            response.reply = "Form data validated successfully."
            response.collected = result.get("normalized")
        else:
            response.error = result.get("error")
            response.missing_fields = result.get("missing_fields")

    elif mode == "design":
        response.reply = result.get("explanation", "")
        response.layout = result.get("layout")
        response.validation = result.get("validation")
        response.ready = True

        # Generate DXF if project_id provided
        if req.project_id and result.get("layout"):
            dxf_url = await _generate_dxf_for_project(
                db, req.project_id, result["layout"], req.history or []
            )
            response.dxf_url = dxf_url

    elif mode == "validation":
        response.validation = result
        response.reply = "Validation complete." if result.get("compliant") else "Validation found issues."

    return response


class DesignRequest(BaseModel):
    """Design-specific request that accepts requirements directly."""
    data: Optional[dict] = None
    project_id: Optional[str] = None
    # Direct fields from frontend form
    total_area: Optional[float] = None
    rooms: Optional[list] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    floors: Optional[int] = None
    extras: Optional[list] = None
    boundary_polygon: Optional[Any] = None
    plot_width: Optional[float] = None
    plot_length: Optional[float] = None


@router.post("/design")
async def engine_design_direct(req: DesignRequest, db: AsyncSession = Depends(get_db)):
    """
    Direct design endpoint — skips mode detection, goes straight to DESIGN MODE.

    Accepts structured data via the 'data' field or directly in request body.
    """
    # Build input from either 'data' field or top-level fields
    input_data = dict(req.data) if req.data else {}
    for field in ("total_area", "bedrooms", "bathrooms", "floors", "extras",
                  "plot_width", "plot_length", "boundary_polygon"):
        val = getattr(req, field, None)
        if val is not None and field not in input_data:
            input_data[field] = val

    # Parse room counts from rooms list if provided (frontend format)
    if req.rooms and not input_data.get("bedrooms"):
        bed_count = 0
        bath_count = 0
        extra_list = list(input_data.get("extras", []))
        for r in req.rooms:
            rtype = r.get('room_type', '') if isinstance(r, dict) else ''
            qty = r.get('quantity', 1) if isinstance(r, dict) else 1
            if rtype in ('master_bedroom', 'bedroom'):
                bed_count += qty
            elif rtype == 'bathroom':
                bath_count += qty
            elif rtype not in ('kitchen', 'living', ''):
                for _ in range(qty):
                    extra_list.append(rtype)
        if bed_count > 0:
            input_data["bedrooms"] = bed_count
        if bath_count > 0:
            input_data["bathrooms"] = bath_count
        if extra_list:
            input_data["extras"] = list(set(extra_list))

    input_data["generatePlan"] = True

    result = design_generate(input_data)

    if "error" in result:
        return result

    # Generate DXF if project_id provided
    dxf_url = None
    if req.project_id and result.get("layout"):
        dxf_url = await _generate_dxf_for_project(
            db, req.project_id, result["layout"], []
        )

    return {
        "mode": "design",
        "explanation": result.get("explanation", ""),
        "layout": result.get("layout"),
        "validation": result.get("validation"),
        "dxf_url": dxf_url,
    }


@router.post("/validate")
async def engine_validate_direct(layout: dict):
    """
    Direct validation endpoint — validates a layout JSON.

    Accepts layout JSON and returns structured validation report.
    """
    result = validate_layout(layout)
    return result


# ---------- WebSocket for Real-Time Chat + Auto-Pipeline ----------

@router.websocket("/chat")
async def engine_chat_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time architectural planning.

    Auto-detects mode transitions:
      CHAT → collects requirements
      "Generate Plan" → DESIGN → VALIDATION → DXF generation

    Frontend sends: { "message": "...", "project_id": "..." }
    Backend responds: { "mode": "...", "reply": "...", ... }
    """
    await websocket.accept()
    history = []
    project_id = None

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            user_text = data.get("message", "")
            project_id = data.get("project_id", project_id)

            if not user_text.strip():
                continue

            # Auto-detect mode
            mode = detect_mode(user_text)

            if mode == EngineMode.CHAT:
                result = chat_response(user_text, history)

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": result["reply"]})

                await websocket.send_text(json.dumps({
                    "mode": "chat",
                    "reply": result["reply"],
                    "collected": result.get("collected"),
                    "ready": result.get("ready", False),
                }))

                # Save to DB
                await _save_history(project_id, history)

                # If switching to design mode
                if result.get("switch_to") == EngineMode.DESIGN:
                    collected = result.get("collected", {})

                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": "Generating architectural layout...",
                        "stage_transition": True,
                    }))

                    # Run design
                    req_data = {
                        "plot_width": collected.get("plot_width"),
                        "plot_length": collected.get("plot_length"),
                        "total_area": collected.get("total_area"),
                        "bedrooms": collected.get("bedrooms"),
                        "bathrooms": collected.get("bathrooms"),
                        "floors": collected.get("floors", 1),
                        "extras": collected.get("extras", []),
                    }
                    design_result = design_generate(req_data)

                    if "error" in design_result:
                        await websocket.send_text(json.dumps({
                            "mode": "design",
                            "reply": design_result["error"],
                            "error": design_result["error"],
                            "suggestion": design_result.get("suggestion"),
                        }))
                        continue

                    # Send design result
                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": design_result.get("explanation", "Layout generated."),
                        "layout": design_result.get("layout"),
                        "validation": design_result.get("validation"),
                    }))

                    # Generate DXF
                    dxf_url = None
                    if project_id and design_result.get("layout"):
                        try:
                            async with async_session() as db:
                                dxf_url = await _do_generate_dxf(
                                    db, project_id,
                                    design_result["layout"], history,
                                )
                        except Exception:
                            pass

                    await websocket.send_text(json.dumps({
                        "mode": "complete",
                        "reply": (
                            "Floor plan generated successfully. "
                            + ("DXF file ready for download." if dxf_url else "")
                            + "\n\nSay anything to start a new design."
                        ),
                        "layout": design_result.get("layout"),
                        "validation": design_result.get("validation"),
                        "dxf_url": dxf_url,
                        "should_generate": True,
                        "extracted_data": {
                            "rooms": _layout_to_rooms(design_result.get("layout", {})),
                            "total_area": collected.get("total_area"),
                            "ready_to_generate": True,
                        },
                    }))

                    # Reset for new design
                    history = []

            elif mode == EngineMode.DESIGN:
                # Direct "generate plan" message
                # Parse requirements from history
                from services.arch_engine import _parse_requirements_from_history
                collected = _parse_requirements_from_history(user_text, history)

                if not collected.get("complete"):
                    await websocket.send_text(json.dumps({
                        "mode": "chat",
                        "reply": "I need more information before generating. What's your plot size?",
                        "collected": collected,
                        "ready": False,
                    }))
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": "I need more information."})
                    continue

                # Generate design
                req_data = {
                    "plot_width": collected.get("plot_width"),
                    "plot_length": collected.get("plot_length"),
                    "total_area": collected.get("total_area"),
                    "bedrooms": collected.get("bedrooms"),
                    "bathrooms": collected.get("bathrooms"),
                    "floors": collected.get("floors", 1),
                    "extras": collected.get("extras", []),
                }

                design_result = design_generate(req_data)

                if "error" in design_result:
                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": design_result["error"],
                        "error": design_result["error"],
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "mode": "design",
                        "reply": design_result.get("explanation", ""),
                        "layout": design_result.get("layout"),
                        "validation": design_result.get("validation"),
                    }))

                    # Generate DXF
                    dxf_url = None
                    if project_id:
                        try:
                            async with async_session() as db:
                                dxf_url = await _do_generate_dxf(
                                    db, project_id,
                                    design_result["layout"], history,
                                )
                        except Exception:
                            pass

                    await websocket.send_text(json.dumps({
                        "mode": "complete",
                        "reply": "Floor plan generated successfully.",
                        "dxf_url": dxf_url,
                        "should_generate": True,
                        "extracted_data": {
                            "rooms": _layout_to_rooms(design_result.get("layout", {})),
                            "total_area": collected.get("total_area"),
                            "ready_to_generate": True,
                        },
                        "layout": design_result.get("layout"),
                    }))

                    history = []

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


# ---------- Helper Functions ----------

def _extract_from_history(history: list) -> dict:
    """Extract collected requirements from chat history."""
    from services.arch_engine import _parse_requirements_from_history
    all_user = " ".join(m.get("content", "") for m in history if m.get("role") == "user")
    return _parse_requirements_from_history(all_user, history)


def _layout_to_rooms(layout: dict) -> list:
    """Convert layout rooms to the format expected by generate_floor_plan."""
    rooms = []
    for room in layout.get("rooms", []):
        rooms.append({
            "room_type": room.get("room_type", "other"),
            "quantity": 1,
            "desired_area": room.get("area"),
        })
    return rooms


async def _generate_dxf_for_project(
    db: AsyncSession, project_id: str, layout: dict, history: list
) -> str:
    """Generate DXF and save to project. Returns DXF URL or None."""
    try:
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return None

        rooms = _layout_to_rooms(layout)
        total_area = layout.get("area_summary", {}).get("plot_area", 1200)

        boundary = None
        if project.boundary_polygon:
            try:
                boundary = json.loads(project.boundary_polygon)
            except (json.JSONDecodeError, TypeError):
                pass

        if not boundary:
            pw = layout.get("plot", {}).get("width", 30)
            pl = layout.get("plot", {}).get("length", 40)
            boundary = [[0, 0], [pw, 0], [pw, pl], [0, pl], [0, 0]]

        plan = generate_floor_plan(boundary, rooms, total_area)

        dxf_filename = f"{project_id}.dxf"
        dxf_path = os.path.join(str(EXPORT_DIR), dxf_filename)
        generate_dxf(plan, dxf_path)

        project.generated_plan = json.dumps(layout)
        project.dxf_path = dxf_path
        project.total_area = total_area
        project.chat_history = json.dumps(history)
        project.status = ProjectStatus.COMPLETED
        await db.commit()

        return f"/api/download-dxf/{project_id}"
    except Exception:
        return None


async def _do_generate_dxf(
    db: AsyncSession, project_id: str, layout: dict, history: list
) -> str:
    """Generate DXF within an existing session."""
    return await _generate_dxf_for_project(db, project_id, layout, history)


async def _save_history(project_id: str, history: list):
    """Save chat history to project."""
    if not project_id:
        return
    try:
        async with async_session() as db:
            result = await db.execute(
                select(Project).where(Project.id == project_id)
            )
            project = result.scalar_one_or_none()
            if project:
                project.chat_history = json.dumps(history)
                await db.commit()
    except Exception:
        pass
