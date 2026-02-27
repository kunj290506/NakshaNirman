"""Data loading, preprocessing, and augmentation for floor plan datasets."""

from ml_pipeline.data.preprocessing import (
    FloorPlanSample,
    normalise_polygon,
    polygon_to_occupancy_grid,
    extract_room_masks,
    build_adjacency_matrix,
)
from ml_pipeline.data.cubicasa import CubiCasa5KDataset
from ml_pipeline.data.rplan import RPLANDataset
from ml_pipeline.data.combined import CombinedFloorPlanDataset, build_dataloaders

__all__ = [
    "FloorPlanSample",
    "normalise_polygon",
    "polygon_to_occupancy_grid",
    "extract_room_masks",
    "build_adjacency_matrix",
    "CubiCasa5KDataset",
    "RPLANDataset",
    "CombinedFloorPlanDataset",
    "build_dataloaders",
]
