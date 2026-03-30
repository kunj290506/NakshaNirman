"""
NakshaNirman — FastAPI backend.
Uses deterministic BSP layout engine. LLM only for Vastu advice.
"""
from __future__ import annotations
import logging
import os
import re
import uuid
import time
from typing import Dict, List
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import EXPORTS_DIR
from models import PlanRequest, PlanResponse
from layout_engine import generate_plan_deterministic
from dxf_export import plan_to_dxf

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
async def generate(req: PlanRequest):
    """Generate a floor plan using deterministic BSP layout engine."""
    log.info(
        "Generate request: %sx%s %dBHK %s extras=%s",
        req.plot_width, req.plot_length, req.bedrooms, req.facing, req.extras,
    )
    try:
        plan = await generate_plan_deterministic(req)
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
    """Validate an existing plan for overlaps and bounds."""
    issues = []
    uw = plan.plot.usable_width
    ul = plan.plot.usable_length

    for room in plan.rooms:
        if room.x + room.width > uw + 0.5:
            issues.append(f"{room.label} exceeds usable width")
        if room.y + room.height > ul + 0.5:
            issues.append(f"{room.label} exceeds usable length")
        if room.x < -0.5 or room.y < -0.5:
            issues.append(f"{room.label} has negative coordinates")

    for i, a in enumerate(plan.rooms):
        for b in plan.rooms[i + 1:]:
            if (
                a.x < b.x + b.width
                and a.x + a.width > b.x
                and a.y < b.y + b.height
                and a.y + a.height > b.y
            ):
                overlap_w = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
                overlap_h = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
                if overlap_w > 0.5 and overlap_h > 0.5:
                    issues.append(f"{a.label} overlaps with {b.label}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "room_count": len(plan.rooms),
    }


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
