"""
AutoArchitect AI - Backend API Server
=====================================
FastAPI application for AI-driven architectural design generation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import structlog

from app.api.routes import upload, design, jobs, results, templates
from app.core.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Create FastAPI application
app = FastAPI(
    title="AutoArchitect AI",
    description="Transform plot boundaries into complete home designs with 3D animations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Include API routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(design.router, prefix="/api", tags=["Design"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(results.router, prefix="/api", tags=["Results"])
app.include_router(templates.router, prefix="/api", tags=["Templates"])


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "name": "AutoArchitect AI",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    logger.info("AutoArchitect AI starting up...")
    # Initialize database connection pool
    # Initialize Redis connection
    # Load AI models


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown."""
    logger.info("AutoArchitect AI shutting down...")
    # Close database connections
    # Close Redis connections
