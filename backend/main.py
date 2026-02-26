"""
AutoCAD Floor Plan Generator – FastAPI Backend

Main entry point. Sets up CORS, includes all routes, initializes DB.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import CORS_ORIGINS, EXPORT_DIR, UPLOAD_DIR
from database import init_db

# Import route modules
from routes.project import router as project_router
from routes.boundary import router as boundary_router
from routes.floorplan import router as floorplan_router
from routes.model3d import router as model3d_router
from routes.chat import router as chat_router
from routes.requirements import router as requirements_router
from routes.ai_design import router as ai_design_router
from routes.engine import router as engine_router
from routes.gnn_design import router as gnn_router
from routes.perfect_design import router as perfect_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="AutoCAD Floor Plan Generator",
    description="Generate 2D floor plans and 3D models from simple inputs",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for exports
app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR)), name="exports")

# Include routers
app.include_router(project_router)
app.include_router(boundary_router)
app.include_router(floorplan_router)
app.include_router(model3d_router)
app.include_router(chat_router)
app.include_router(requirements_router)
app.include_router(ai_design_router)
app.include_router(engine_router)
app.include_router(gnn_router)
app.include_router(perfect_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
