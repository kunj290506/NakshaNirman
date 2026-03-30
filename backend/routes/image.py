"""Image generation routes for floor plan concepts."""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.openrouter_image import generate_floorplan_image

router = APIRouter(prefix="/api", tags=["image"])

_SIZE_PATTERN = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_MIN_DIMENSION = 256
_MAX_DIMENSION = 2048
_ALLOWED_RESPONSE_FORMATS = {"b64_json", "url"}


class FloorplanImageRequest(BaseModel):
    prompt: str = Field(..., min_length=8, max_length=2000)
    size: str = Field("1024x1024")
    n: int = Field(1, ge=1, le=4)
    response_format: str = Field("b64_json")


def _validate_size(value: str) -> str:
    token = str(value or "").strip().lower()
    match = _SIZE_PATTERN.match(token)
    if not match:
        raise ValueError("size must be WIDTHxHEIGHT, for example 1024x1024")

    width = int(match.group(1))
    height = int(match.group(2))
    if width < _MIN_DIMENSION or width > _MAX_DIMENSION:
        raise ValueError(f"width must be between {_MIN_DIMENSION} and {_MAX_DIMENSION}")
    if height < _MIN_DIMENSION or height > _MAX_DIMENSION:
        raise ValueError(f"height must be between {_MIN_DIMENSION} and {_MAX_DIMENSION}")

    return f"{width}x{height}"


def _validate_response_format(value: str) -> str:
    token = str(value or "b64_json").strip().lower()
    if token not in _ALLOWED_RESPONSE_FORMATS:
        raise ValueError("response_format must be 'b64_json' or 'url'")
    return token


@router.post("/generate-floorplan-image")
async def generate_floorplan_concept_image(data: FloorplanImageRequest):
    """Generate one or more floor plan concept images from a prompt."""
    try:
        size = _validate_size(data.size)
        response_format = _validate_response_format(data.response_format)

        result = await asyncio.to_thread(
            generate_floorplan_image,
            data.prompt.strip(),
            size,
            data.n,
            response_format,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc
