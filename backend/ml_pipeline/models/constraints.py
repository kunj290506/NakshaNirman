"""
Architectural constraint network — differentiable rule checker.

Provides differentiable penalties for:
  1. Zoning logic          — rooms in correct zone (public/private/service)
  2. Circulation flow      — hallway connectivity, dead-end avoidance
  3. Ventilation           — rooms touching exterior walls
  4. Plumbing alignment    — wet rooms (kitchen, bath) adjacent/aligned
  5. Structural column grid — rooms align to column spacing multiples
  6. Min room area         — no room smaller than threshold
  7. Boundary fitting      — rooms must not leak outside plot

These are used as differentiable loss terms during training and as
hard constraints during post-processing / inference.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.config import PipelineConfig, ROOM_TO_IDX, NUM_ROOM_CLASSES


# Zone classification (same as arch_engine.py)
ZONE_PUBLIC = {ROOM_TO_IDX.get(r, -1) for r in
               ["living_room", "porch", "parking", "hallway"]}
ZONE_PRIVATE = {ROOM_TO_IDX.get(r, -1) for r in
                ["master_bedroom", "bedroom", "study", "pooja"]}
ZONE_SERVICE = {ROOM_TO_IDX.get(r, -1) for r in
                ["kitchen", "bathroom", "toilet", "utility", "store"]}
# Remove -1 sentinel
ZONE_PUBLIC.discard(-1)
ZONE_PRIVATE.discard(-1)
ZONE_SERVICE.discard(-1)

# Wet rooms — must be plumbing-aligned
WET_ROOMS = {ROOM_TO_IDX.get(r, -1) for r in
             ["kitchen", "bathroom", "toilet", "utility"]}
WET_ROOMS.discard(-1)

EXTERIOR_IDX = ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1)


class ArchitecturalConstraintNet(nn.Module):
    """
    Compute differentiable constraint violations from predicted layout masks.

    All losses return scalar tensors that can be back-propagated.
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        super().__init__()
        self.cfg = cfg or PipelineConfig()

        # Learnable zone preference map  (C,)  → encourage correct placement
        self.zone_logits = nn.Parameter(torch.zeros(NUM_ROOM_CLASSES, 3))
        # 3 zones: 0=public (front/south), 1=private (back/north), 2=service

        # Column grid spacing in pixels (at 256px image size)
        col_ft = self.cfg.column_grid_ft
        img_ft = 60.0  # assume ~60 ft max dimension for normalisation
        self.col_spacing_px = col_ft / img_ft * self.cfg.img_size

    # ----- Combined loss ---------------------------------------------------

    def forward(
        self,
        mask_logits: torch.Tensor,        # (B, C, H, W)
        boundary_grid: torch.Tensor,      # (B, 1, H, W)  — plot occupancy
        adjacency_pred: torch.Tensor,     # (B, C, C)
    ) -> Dict[str, torch.Tensor]:
        """Return dict of named constraint loss scalars."""
        mask_prob = F.softmax(mask_logits, dim=1)
        losses = {}
        losses["boundary"]    = self.boundary_loss(mask_prob, boundary_grid)
        losses["min_area"]    = self.min_area_loss(mask_prob)
        losses["ventilation"] = self.ventilation_loss(mask_prob, boundary_grid)
        losses["plumbing"]    = self.plumbing_alignment_loss(mask_prob)
        losses["column_grid"] = self.column_grid_loss(mask_prob)
        losses["zoning"]      = self.zoning_loss(mask_prob)
        losses["circulation"] = self.circulation_loss(mask_prob, adjacency_pred)
        return losses

    # ----- Individual constraints ------------------------------------------

    def boundary_loss(
        self,
        mask_prob: torch.Tensor,
        boundary_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalise rooms that extend outside the plot boundary.

        Any predicted room probability outside the boundary → L2 penalty.
        """
        # Everything except exterior
        room_prob = 1.0 - mask_prob[:, EXTERIOR_IDX:EXTERIOR_IDX + 1]  # (B,1,H,W)
        outside = (1.0 - boundary_grid)  # 1 where outside plot
        violation = (room_prob * outside).pow(2)
        return violation.mean()

    def min_area_loss(self, mask_prob: torch.Tensor) -> torch.Tensor:
        """
        Penalise rooms whose total pixel count is below threshold.

        Soft penalty: max(0, min_pixels - area_pixels)² for each class.
        """
        B, C, H, W = mask_prob.shape
        # Minimum fraction of image each room should cover
        min_frac = self.cfg.min_room_area_sqft / 3600.0  # rough normalisation
        room_areas = mask_prob.sum(dim=(2, 3))  # (B, C)
        total_pixels = H * W
        min_pixels = min_frac * total_pixels

        # Only penalise rooms that ARE present (area > some threshold)
        present = (room_areas > total_pixels * 0.005).float()
        deficit = F.relu(min_pixels - room_areas) * present
        return deficit.pow(2).mean()

    def ventilation_loss(
        self,
        mask_prob: torch.Tensor,
        boundary_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bedrooms and living room should touch exterior walls (ventilation).

        Exterior wall = edge of boundary_grid  (pixels where boundary == 1
        and at least one 4-neighbour == 0).
        """
        # Find exterior wall pixels
        pad = F.pad(boundary_grid, (1, 1, 1, 1), value=0)
        eroded = F.max_pool2d(-pad, 3, 1, 0)  # min-pool via negation
        eroded = -eroded[:, :, :boundary_grid.size(2), :boundary_grid.size(3)]
        edge_mask = boundary_grid * (1.0 - eroded)  # 1 on boundary edge

        # Habitat rooms that need ventilation
        vent_classes = list(ZONE_PRIVATE) | [ROOM_TO_IDX.get("living_room", 0)]
        penalty = torch.tensor(0.0, device=mask_prob.device)
        for cidx in vent_classes:
            if cidx < 0 or cidx >= mask_prob.size(1):
                continue
            room = mask_prob[:, cidx:cidx + 1]
            touching = (room * edge_mask).sum(dim=(2, 3))
            room_area = room.sum(dim=(2, 3)).clamp(min=1)
            # Fraction of room touching exterior edge should be > 0
            frac = touching / room_area
            penalty = penalty + F.relu(0.05 - frac).mean()

        return penalty

    def plumbing_alignment_loss(self, mask_prob: torch.Tensor) -> torch.Tensor:
        """
        Wet rooms (kitchen, bath, toilet) should be adjacent to each other
        so plumbing stacks align.  Penalise isolated wet rooms.
        """
        B, C, H, W = mask_prob.shape
        wet_total = torch.zeros(B, 1, H, W, device=mask_prob.device)
        for cidx in WET_ROOMS:
            if cidx < C:
                wet_total = wet_total + mask_prob[:, cidx:cidx + 1]

        # Dilate wet_total → check if each wet pixel has wet neighbours
        wet_dilated = F.max_pool2d(wet_total, 5, 1, 2)
        # Pixels that are wet but have no wet neighbours → penalty
        isolated = wet_total * (1.0 - (wet_dilated - wet_total).clamp(0, 1))
        return isolated.pow(2).mean()

    def column_grid_loss(self, mask_prob: torch.Tensor) -> torch.Tensor:
        """
        Encourage room boundaries to align with structural column grid.

        Compute room edge map (gradient of class probabilities) and reward
        edges that fall on column-grid lines.
        """
        # Sobel-like edge detection
        dx = mask_prob[:, :, :, 1:] - mask_prob[:, :, :, :-1]
        dy = mask_prob[:, :, 1:, :] - mask_prob[:, :, :-1, :]
        edge_x = dx.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W-1)
        edge_y = dy.abs().sum(dim=1, keepdim=True)  # (B, 1, H-1, W)

        spacing = max(self.col_spacing_px, 4)
        W = edge_x.size(3)
        H = edge_y.size(2)

        # Column grid lines (vertical)
        col_lines = torch.zeros(1, 1, 1, W, device=mask_prob.device)
        for c in range(0, W, int(spacing)):
            lo = max(c - 1, 0)
            hi = min(c + 2, W)
            col_lines[0, 0, 0, lo:hi] = 1.0

        # Row grid lines (horizontal)
        row_lines = torch.zeros(1, 1, H, 1, device=mask_prob.device)
        for r in range(0, H, int(spacing)):
            lo = max(r - 1, 0)
            hi = min(r + 2, H)
            row_lines[0, 0, lo:hi, 0] = 1.0

        # Reward edges on grid, penalise off-grid
        off_grid_x = edge_x * (1.0 - col_lines)
        off_grid_y = edge_y * (1.0 - row_lines)

        return (off_grid_x.mean() + off_grid_y.mean()) * 0.5

    def zoning_loss(self, mask_prob: torch.Tensor) -> torch.Tensor:
        """
        Public rooms should be in the front half (lower Y),
        private rooms in the back half (upper Y).
        """
        B, C, H, W = mask_prob.shape
        # Spatial weight: 0 at bottom, 1 at top
        y_weight = torch.linspace(0, 1, H, device=mask_prob.device)
        y_weight = y_weight.view(1, 1, H, 1).expand(B, 1, H, W)

        penalty = torch.tensor(0.0, device=mask_prob.device)

        # Public rooms penalised for being in upper half
        for cidx in ZONE_PUBLIC:
            if cidx < C:
                room = mask_prob[:, cidx:cidx + 1]
                penalty = penalty + (room * y_weight).mean()

        # Private rooms penalised for being in lower half
        for cidx in ZONE_PRIVATE:
            if cidx < C:
                room = mask_prob[:, cidx:cidx + 1]
                penalty = penalty + (room * (1.0 - y_weight)).mean()

        return penalty

    def circulation_loss(
        self,
        mask_prob: torch.Tensor,
        adjacency_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hallway should connect to most rooms (circulation hub).
        Penalise if hallway adjacency row sums are low.
        """
        hall_idx = ROOM_TO_IDX.get("hallway", -1)
        if hall_idx < 0 or hall_idx >= adjacency_pred.size(1):
            return torch.tensor(0.0, device=mask_prob.device)

        # adjacency_pred: (B, C, C) — values in [0,1]
        hall_connections = adjacency_pred[:, hall_idx, :]      # (B, C)
        # How many rooms does hallway connect to?
        n_connected = hall_connections.sum(dim=1)              # (B,)
        # Should connect to at least 3 rooms
        deficit = F.relu(3.0 - n_connected)
        return deficit.mean()
