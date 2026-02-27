"""
ML-Based Floor Plan Generation API Route.

Endpoints:
  POST /api/ml/generate   — Generate floor plan using trained cGAN model
  GET  /api/ml/status      — Check ML pipeline status (model availability)
  POST /api/ml/batch       — Generate multiple variants for comparison
"""

import os
import uuid
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from config import EXPORT_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ml", tags=["ml-pipeline"])

# Lazy imports for heavy dependencies
_generator = None
_mask_to_cad = None


def _load_generator():
    """Lazy-load the trained FloorPlanGenerator from latest checkpoint."""
    global _generator
    if _generator is not None:
        return _generator

    try:
        import torch
        from ml_pipeline.config import PipelineConfig, CHECKPOINT_DIR
        from ml_pipeline.models.generator import FloorPlanGenerator

        cfg = PipelineConfig()

        # Look for latest checkpoint
        ckpt_dir = Path(CHECKPOINT_DIR)
        if not ckpt_dir.exists():
            logger.info("No checkpoint directory found at %s", ckpt_dir)
            return None

        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if not ckpts:
            # Also try best checkpoint
            best = ckpt_dir / "best_model.pt"
            if not best.exists():
                logger.info("No checkpoint files found")
                return None
            ckpt_path = best
        else:
            ckpt_path = ckpts[-1]  # latest epoch

        logger.info("Loading ML generator from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        gen = FloorPlanGenerator(cfg)
        if "generator_ema" in state:
            gen.load_state_dict(state["generator_ema"])
        elif "generator" in state:
            gen.load_state_dict(state["generator"])
        else:
            logger.warning("Checkpoint missing generator weights")
            return None

        gen.eval()
        _generator = gen
        logger.info("ML generator loaded successfully")
        return _generator

    except Exception as e:
        logger.warning("Failed to load ML generator: %s", e)
        return None


# ---------- Request / Response Models ----------

class MLGenerateRequest(BaseModel):
    """Request body for ML-based floor plan generation."""
    boundary_polygon: Optional[List[List[float]]] = None
    plot_width: float = Field(default=30.0, ge=10, le=200)
    plot_length: float = Field(default=40.0, ge=10, le=200)
    entry_side: str = Field(default="south")
    north_direction: float = Field(default=0.0, ge=0, le=360)

    bedrooms: int = Field(default=2, ge=1, le=6)
    bathrooms: int = Field(default=2, ge=1, le=4)
    kitchen_type: str = Field(default="open")
    parking: bool = Field(default=True)
    budget: str = Field(default="medium")

    num_variants: int = Field(default=1, ge=1, le=8)
    project_id: Optional[str] = None


class MLRoomResult(BaseModel):
    name: str
    room_type: str
    position: Dict[str, float]
    width: float
    length: float
    area: float
    doors: List[Dict] = []
    windows: List[Dict] = []


class MLGenerateResponse(BaseModel):
    engine: str = "ml-cgan"
    method: str = "learned"
    status: str = "ok"
    variants: List[Dict] = []
    error: Optional[str] = None


class MLStatusResponse(BaseModel):
    engine: str = "ml-cgan"
    model_loaded: bool = False
    checkpoint: Optional[str] = None
    torch_available: bool = False


# ---------- Endpoints ----------

@router.get("/status", response_model=MLStatusResponse)
async def ml_status():
    """Check ML pipeline status and model availability."""
    torch_avail = False
    try:
        import torch
        torch_avail = True
    except ImportError:
        pass

    gen = _load_generator()
    ckpt_name = None
    if gen is not None:
        from ml_pipeline.config import CHECKPOINT_DIR
        ckpt_dir = Path(CHECKPOINT_DIR)
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        best = ckpt_dir / "best_model.pt"
        if best.exists():
            ckpt_name = "best_model.pt"
        elif ckpts:
            ckpt_name = ckpts[-1].name

    return MLStatusResponse(
        model_loaded=gen is not None,
        checkpoint=ckpt_name,
        torch_available=torch_avail,
    )


@router.post("/generate", response_model=MLGenerateResponse)
async def ml_generate(req: MLGenerateRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate floor plan(s) using the trained conditional GAN.

    Returns one or more layout variants. Falls back to heuristic engine
    if no trained model is available.
    """
    try:
        gen = _load_generator()

        if gen is None:
            # Fall back to heuristic engine
            return await _fallback_heuristic(req)

        import torch
        from ml_pipeline.config import PipelineConfig, ROOM_TYPES, NUM_ROOM_CLASSES
        from ml_pipeline.data.preprocessing import (
            normalise_polygon, polygon_to_occupancy_grid, encode_condition,
        )
        from ml_pipeline.models.encoders import PolygonEncoder, ConditionEncoder, OccupancyGridEncoder
        from ml_pipeline.export.mask_to_cad import MaskToCAD

        cfg = PipelineConfig()

        # Build boundary polygon
        if req.boundary_polygon and len(req.boundary_polygon) >= 3:
            boundary = req.boundary_polygon
        else:
            boundary = [
                [0, 0], [req.plot_width, 0],
                [req.plot_width, req.plot_length],
                [0, req.plot_length], [0, 0],
            ]

        # Normalise polygon
        norm_poly = normalise_polygon(np.array(boundary), cfg.max_poly_pts)

        # Occupancy grid
        occ_grid = polygon_to_occupancy_grid(
            np.array(boundary), cfg.img_size
        )

        # Condition vector
        entry_map = {"south": 0, "north": 1, "east": 2, "west": 3}
        budget_map = {"low": 0, "medium": 1, "high": 2, "premium": 3}

        cond = encode_condition(
            num_bedrooms=req.bedrooms,
            num_bathrooms=req.bathrooms,
            num_kitchens=1,
            kitchen_type={"open": 0, "closed": 1, "semi": 2}.get(
                req.kitchen_type.lower(), 0),
            parking=1 if req.parking else 0,
            entry_side=entry_map.get(req.entry_side.lower(), 0),
            budget_level=budget_map.get(req.budget.lower(), 1),
            north_dir=req.north_direction,
            plot_area=req.plot_width * req.plot_length,
            plot_aspect=req.plot_width / max(req.plot_length, 1),
            cfg=cfg,
        )

        # Convert to tensors
        poly_t = torch.FloatTensor(norm_poly).unsqueeze(0)  # (1, P, 2)
        occ_t = torch.FloatTensor(occ_grid).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cond_t = torch.FloatTensor(cond).unsqueeze(0)  # (1, C)

        # Generate variants
        converter = MaskToCAD(
            cfg=cfg,
            plot_width_ft=req.plot_width,
            plot_length_ft=req.plot_length,
        )

        variants = []
        # Build boundary mask (True for valid polygon points)
        poly_mask = (poly_t.abs().sum(dim=-1) > 1e-6)  # (1, P)

        with torch.no_grad():
            for v in range(req.num_variants):
                out = gen.sample(
                    boundary=poly_t,
                    boundary_mask=poly_mask,
                    condition=cond_t,
                    occ_grid=occ_t,
                )
                mask_pred = out['mask_logits'].argmax(dim=1)[0].cpu().numpy()  # (H, W)

                # Convert mask to CAD
                pid = req.project_id or str(uuid.uuid4())
                dxf_path = str(EXPORT_DIR / f"{pid}_ml_v{v}.dxf")
                result = converter.convert(
                    mask_pred,
                    output_path=dxf_path,
                    boundary_coords=boundary,
                )

                dxf_url = None
                if result.get("dxf_path"):
                    dxf_url = f"/exports/{Path(result['dxf_path']).name}"

                variants.append({
                    "variant_id": v,
                    "rooms": result["rooms"],
                    "boundary": result["boundary"],
                    "dxf_url": dxf_url,
                    "total_area": sum(r["area"] for r in result["rooms"]),
                    "room_count": len(result["rooms"]),
                })

        return MLGenerateResponse(
            engine="ml-cgan",
            method="learned",
            status="ok",
            variants=variants,
        )

    except Exception as e:
        logger.error("ML generation failed: %s", e, exc_info=True)
        return MLGenerateResponse(
            error=str(e),
            status="error",
        )


@router.post("/batch")
async def ml_batch(req: MLGenerateRequest, db: AsyncSession = Depends(get_db)):
    """Generate multiple variants for comparison (convenience alias)."""
    req.num_variants = max(req.num_variants, 3)
    return await ml_generate(req, db)


# ---------- Heuristic fallback ----------

async def _fallback_heuristic(req: MLGenerateRequest) -> MLGenerateResponse:
    """Use the existing heuristic engine when no ML model is available."""
    try:
        from services.gnn_engine import generate_gnn_floor_plan

        boundary_coords = None
        if req.boundary_polygon and len(req.boundary_polygon) >= 3:
            boundary_coords = [tuple(p) for p in req.boundary_polygon]

        layout = generate_gnn_floor_plan(
            boundary_coords=boundary_coords,
            total_area=req.plot_width * req.plot_length,
            bedrooms=req.bedrooms,
            bathrooms=req.bathrooms,
            kitchens=1,
            floors=1,
            extras=["parking"] if req.parking else [],
            plot_width=req.plot_width,
            plot_length=req.plot_length,
        )

        rooms = layout.get("rooms", [])
        variant = {
            "variant_id": 0,
            "rooms": rooms,
            "boundary": layout.get("boundary", []),
            "dxf_url": None,
            "total_area": layout.get("total_area", 0),
            "room_count": len(rooms),
        }

        return MLGenerateResponse(
            engine="ml-cgan",
            method="heuristic-fallback",
            status="ok",
            variants=[variant],
        )
    except Exception as e:
        logger.error("Heuristic fallback failed: %s", e)
        return MLGenerateResponse(error=str(e), status="error")
