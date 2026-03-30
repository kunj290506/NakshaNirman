"""
AutoCAD Floor Plan Generator – FastAPI Backend

Main entry point. Sets up CORS, includes all routes, initializes DB.
"""

from contextlib import asynccontextmanager
import importlib
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from app_config import CORS_ORIGINS, EXPORT_DIR, UPLOAD_DIR
from database import init_db
from middleware.security import SecurityMiddleware, check_rate_limit, log_security_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="NakshaNirman Floor Plan Generator",
    description="Generate 2D floor plans and 3D models from simple inputs",
    version="2.0.0",
    lifespan=lifespan,
)

# Security middleware (must be first)
app.add_middleware(SecurityMiddleware)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    client_ip = request.client.host
    
    # Skip rate limiting for health check
    if request.url.path == "/api/health":
        return await call_next(request)
    
    if not check_rate_limit(client_ip):
        log_security_event(
            "rate_limit_exceeded",
            {"ip": client_ip, "path": request.url.path},
            severity="WARNING"
        )
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    return await call_next(request)

# CORS (after security middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for exports
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")

def _include_router(module_path: str, label: str) -> None:
    """Load and include route modules without crashing startup on optional deps."""
    try:
        module = importlib.import_module(module_path)
        router = getattr(module, "router", None)
        if router is None:
            logger.warning("Route module %s has no router object; skipped", module_path)
            return
        app.include_router(router)
        logger.info("Loaded router: %s", label)
    except Exception as exc:
        logger.warning("Skipping router %s due to import error: %s", label, exc)


# Include routers
_include_router("routes.project", "project")
_include_router("routes.boundary", "boundary")
_include_router("routes.floorplan", "floorplan")
_include_router("routes.model3d", "model3d")
_include_router("routes.chat", "chat")
_include_router("routes.requirements", "requirements")
_include_router("routes.architect", "architect")
_include_router("routes.image", "image")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "security": "enabled"
    }


if __name__ == "__main__":
    import uvicorn
    from app_config import HOST, PORT
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
