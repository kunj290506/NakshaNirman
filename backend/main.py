"""
NakshaNirman — FastAPI backend.
Uses LLM-first layout engine with BSP fallback.
"""
from __future__ import annotations
import logging
import os
import re
import uuid
import time
from typing import Any, Dict, List
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import EXPORTS_DIR
from models import PlanRequest, PlanResponse
from layout_engine import generate_plan
from dxf_export import plan_to_dxf
from plan_validator import validate_llm_plan

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="NakshaNirman API",
    version="3.0.0",
    description="Indian residential floor plan generator with deterministic BSP layout",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Security headers middleware ──────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ── Simple rate limiter (per IP) ─────────────────────────────
_rate_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT = 20
RATE_WINDOW = 60


@app.middleware("http")
async def rate_limit_generate(request: Request, call_next):
    if request.url.path == "/api/generate" and request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < RATE_WINDOW]
        if len(_rate_store[ip]) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a minute."},
            )
        _rate_store[ip].append(now)
    return await call_next(request)


# ── Health ───────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "NakshaNirman", "version": "3.0.0"}


# ── Generate ─────────────────────────────────────────────────
@app.post("/api/generate", response_model=PlanResponse)
async def generate(req: PlanRequest | dict[str, Any]):
    """Generate a floor plan using LLM-first engine with BSP fallback."""
    # Support a known frontend/client alias without changing PlanResponse shape.
    if isinstance(req, dict):
        payload = dict(req)
        if str(payload.get("family_type", "")).strip().lower() == "working_couple":
            payload["family_type"] = "couple"
        req = PlanRequest(**payload)

    log.info(
        "Generate request: %sx%s %dBHK %s extras=%s",
        req.plot_width, req.plot_length, req.bedrooms, req.facing, req.extras,
    )
    try:
        plan = await generate_plan(req)
    except Exception as e:
        log.exception("Plan generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Generate DXF
    try:
        filename = f"plan_{uuid.uuid4().hex[:8]}.dxf"
        filepath = os.path.join(EXPORTS_DIR, filename)
        plan_to_dxf(plan, filepath)
        plan.dxf_url = f"/api/download/{filename}"
        log.info("DXF saved: %s", filename)
    except Exception as e:
        log.warning("DXF generation failed: %s", e)
        plan.dxf_url = None

    return plan


# ── Download DXF ─────────────────────────────────────────────
@app.get("/api/download/{filename}")
async def download(filename: str):
    """Download a generated DXF file."""
    if not re.match(r"^plan_[a-f0-9]{8}\.dxf$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename format")
    filepath = os.path.join(EXPORTS_DIR, filename)
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(os.path.realpath(EXPORTS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        real_path,
        media_type="application/dxf",
        filename=filename,
    )


# ── Validate ─────────────────────────────────────────────────
@app.post("/api/validate")
async def validate(plan: PlanResponse):
    """Validate an existing plan using the same rules as the generation pipeline."""
    # Earlier this endpoint used stricter ad-hoc checks than plan generation,
    # which created contradictory pass/fail results. Reuse the shared validator.
    plan_dict = {
        "rooms": [
            {
                "id": room.id,
                "type": room.type,
                "label": room.label,
                "x": room.x,
                "y": room.y,
                "width": room.width,
                "height": room.height,
                "area": room.area,
                "polygon": [{"x": p.x, "y": p.y} for p in (room.polygon or [])],
            }
            for room in plan.rooms
        ]
    }

    is_valid, issues = validate_llm_plan(
        plan_dict,
        plan.plot.usable_width,
        plan.plot.usable_length,
        None,
    )

    return {
        "valid": is_valid,
        "issues": issues,
        "room_count": len(plan.rooms),
    }


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
