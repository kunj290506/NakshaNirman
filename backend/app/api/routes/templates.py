"""
Templates API - architectural styles and presets.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()


class StyleTemplate(BaseModel):
    """Architectural style template."""
    id: str
    name: str
    description: str
    preview_image: str
    features: List[str]
    materials: dict


class FeatureOption(BaseModel):
    """Available feature option."""
    id: str
    name: str
    description: str
    icon: str


# Predefined architectural styles
STYLES = [
    StyleTemplate(
        id="modern",
        name="Modern",
        description="Clean lines, open spaces, large windows, minimalist design",
        preview_image="/assets/styles/modern.jpg",
        features=["Open floor plan", "Floor-to-ceiling windows", "Flat roof"],
        materials={"walls": "white plaster", "floor": "polished concrete", "windows": "black aluminum"}
    ),
    StyleTemplate(
        id="traditional",
        name="Traditional",
        description="Classic proportions, formal layout, detailed moldings",
        preview_image="/assets/styles/traditional.jpg",
        features=["Formal living room", "Crown moldings", "Pitched roof"],
        materials={"walls": "painted drywall", "floor": "hardwood", "windows": "white wood"}
    ),
    StyleTemplate(
        id="contemporary",
        name="Contemporary",
        description="Blend of modern and classic, warm materials, flowing spaces",
        preview_image="/assets/styles/contemporary.jpg",
        features=["Mixed materials", "Natural light focus", "Indoor-outdoor flow"],
        materials={"walls": "textured plaster", "floor": "engineered wood", "windows": "dark metal"}
    ),
    StyleTemplate(
        id="minimalist",
        name="Minimalist",
        description="Extreme simplicity, essential furniture only, neutral colors",
        preview_image="/assets/styles/minimalist.jpg",
        features=["Hidden storage", "Built-in furniture", "Seamless surfaces"],
        materials={"walls": "pure white", "floor": "light wood", "windows": "frameless glass"}
    ),
    StyleTemplate(
        id="mediterranean",
        name="Mediterranean",
        description="Warm terracotta, arched doorways, outdoor living spaces",
        preview_image="/assets/styles/mediterranean.jpg",
        features=["Courtyard", "Tile accents", "Covered patio"],
        materials={"walls": "stucco", "floor": "terracotta tile", "windows": "wrought iron"}
    )
]

# Available feature options
FEATURES = [
    FeatureOption(id="home_office", name="Home Office", description="Dedicated work space", icon=""),
    FeatureOption(id="garage", name="Garage", description="Covered vehicle parking", icon=""),
    FeatureOption(id="garden", name="Garden", description="Outdoor green space", icon=""),
    FeatureOption(id="pool", name="Swimming Pool", description="Outdoor pool area", icon=""),
    FeatureOption(id="solar", name="Solar Panels", description="Renewable energy system", icon=""),
    FeatureOption(id="gym", name="Home Gym", description="Exercise room", icon=""),
    FeatureOption(id="theater", name="Home Theater", description="Media room", icon=""),
    FeatureOption(id="wine_cellar", name="Wine Cellar", description="Temperature controlled storage", icon=""),
]


@router.get("/templates/styles", response_model=List[StyleTemplate])
async def get_architectural_styles():
    """Get all available architectural styles."""
    return STYLES


@router.get("/templates/styles/{style_id}", response_model=StyleTemplate)
async def get_style_details(style_id: str):
    """Get details for a specific architectural style."""
    for style in STYLES:
        if style.id == style_id:
            return style
    return {"error": "Style not found"}


@router.get("/templates/features", response_model=List[FeatureOption])
async def get_available_features():
    """Get all available feature options."""
    return FEATURES
