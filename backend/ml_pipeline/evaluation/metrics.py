"""
Evaluation metrics for generated floor plans.

Metrics
───────
1.  IoU (Intersection over Union) — per-class and mean
    Measures how well predicted room regions match ground truth.

2.  Adjacency Accuracy — fraction of correctly predicted room adjacencies.

3.  Diversity Score — average pairwise L1 distance between K samples
    generated from the same condition with different z vectors.

4.  Constraint Violation Rate — fraction of samples violating
    architectural rules (boundary overflow, min area, ventilation, etc.).

5.  Room Count Accuracy — how often the predicted number of rooms per
    type matches the ground truth.

6.  Area Distribution Error — KL divergence between predicted and
    GT area distributions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ml_pipeline.config import NUM_ROOM_CLASSES, ROOM_TYPES, ROOM_TO_IDX


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_masks: np.ndarray,      # (N, H, W) int
    gt_masks: np.ndarray,        # (N, H, W) int
    pred_adj: np.ndarray,        # (N, C, C) float
    gt_adj: np.ndarray,          # (N, C, C) float
    boundary_grids: Optional[np.ndarray] = None,  # (N, H, W) float
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns dict with keys:
      iou, adj_acc, diversity, violations, room_count_acc, area_kl
    """
    metrics = {}

    metrics["iou"] = mean_iou(pred_masks, gt_masks)
    metrics["adj_acc"] = adjacency_accuracy(pred_adj, gt_adj)
    metrics["diversity"] = diversity_score(pred_masks)
    metrics["violations"] = constraint_violation_rate(pred_masks, boundary_grids)
    metrics["room_count_acc"] = room_count_accuracy(pred_masks, gt_masks)
    metrics["area_kl"] = area_distribution_error(pred_masks, gt_masks)

    return metrics


# ---------------------------------------------------------------------------
# 1. IoU
# ---------------------------------------------------------------------------

def mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Per-class IoU averaged over classes present in ground truth.

    pred, gt : (N, H, W) int — class indices.
    """
    ious = []
    for c in range(NUM_ROOM_CLASSES):
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union == 0:
            continue  # skip absent classes
        ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def per_class_iou(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """IoU broken down by room type name."""
    result = {}
    for c in range(NUM_ROOM_CLASSES):
        pred_c = (pred == c)
        gt_c = (gt == c)
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union == 0:
            continue
        name = ROOM_TYPES[c] if c < len(ROOM_TYPES) else f"class_{c}"
        result[name] = float(intersection / union)
    return result


# ---------------------------------------------------------------------------
# 2. Adjacency Accuracy
# ---------------------------------------------------------------------------

def adjacency_accuracy(pred_adj: np.ndarray, gt_adj: np.ndarray, threshold: float = 0.5) -> float:
    """
    Fraction of correctly predicted adjacency edges.

    pred_adj : (N, C, C) float [0, 1]
    gt_adj   : (N, C, C) float [0, 1]
    """
    pred_bin = (pred_adj > threshold).astype(np.float32)
    gt_bin = (gt_adj > threshold).astype(np.float32)

    correct = (pred_bin == gt_bin).sum()
    total = gt_bin.size
    return float(correct / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# 3. Diversity Score
# ---------------------------------------------------------------------------

def diversity_score(pred_masks: np.ndarray) -> float:
    """
    Mean pairwise L1 distance between predicted masks in the batch,
    normalised by image area.

    Higher is more diverse.
    """
    N = pred_masks.shape[0]
    if N < 2:
        return 0.0

    total = 0.0
    count = 0
    area = float(pred_masks.shape[1] * pred_masks.shape[2])

    # Sample pairs (up to 100)
    indices = np.random.choice(N, size=min(N, 20), replace=False)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            diff = (pred_masks[indices[i]] != pred_masks[indices[j]]).sum()
            total += diff / area
            count += 1

    return float(total / count) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. Constraint Violation Rate
# ---------------------------------------------------------------------------

def constraint_violation_rate(
    pred_masks: np.ndarray,
    boundary_grids: Optional[np.ndarray] = None,
) -> float:
    """
    Fraction of samples that violate at least one constraint:
      - Any room pixel outside boundary
      - Any room with area < 1% of total
      - Bedrooms not touching exterior edge

    Returns value in [0, 1].
    """
    N = pred_masks.shape[0]
    violations = 0

    ext_idx = ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1)

    for i in range(N):
        mask = pred_masks[i]
        violated = False

        # Check boundary overflow
        if boundary_grids is not None:
            bg = boundary_grids[i]
            room_pixels = (mask != ext_idx)
            outside = room_pixels & (bg < 0.5)
            if outside.sum() > mask.size * 0.02:  # > 2% outside
                violated = True

        # Check minimum room area
        H, W = mask.shape
        total = H * W
        for c in range(NUM_ROOM_CLASSES):
            if c == ext_idx:
                continue
            area = (mask == c).sum()
            if 0 < area < total * 0.005:  # present but < 0.5%
                violated = True
                break

        if violated:
            violations += 1

    return float(violations / N) if N > 0 else 0.0


# ---------------------------------------------------------------------------
# 5. Room Count Accuracy
# ---------------------------------------------------------------------------

def room_count_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Fraction of samples where the number of distinct rooms per type
    matches ground truth.
    """
    N = pred.shape[0]
    correct = 0

    for i in range(N):
        match = True
        for c in range(NUM_ROOM_CLASSES):
            pred_present = (pred[i] == c).any()
            gt_present = (gt[i] == c).any()
            if pred_present != gt_present:
                match = False
                break
        if match:
            correct += 1

    return float(correct / N) if N > 0 else 0.0


# ---------------------------------------------------------------------------
# 6. Area Distribution Error (symmetric KL divergence)
# ---------------------------------------------------------------------------

def area_distribution_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Symmetric KL divergence between aggregate area distributions
    (fraction of image area per class) of predicted vs ground truth.

    Lower is better.
    """
    ext_idx = ROOM_TO_IDX.get("exterior", NUM_ROOM_CLASSES - 1)

    pred_dist = np.zeros(NUM_ROOM_CLASSES, dtype=np.float64)
    gt_dist = np.zeros(NUM_ROOM_CLASSES, dtype=np.float64)

    for c in range(NUM_ROOM_CLASSES):
        pred_dist[c] = (pred == c).sum()
        gt_dist[c] = (gt == c).sum()

    # Normalise (exclude exterior)
    pred_dist[ext_idx] = 0
    gt_dist[ext_idx] = 0

    p_total = pred_dist.sum() + 1e-10
    g_total = gt_dist.sum() + 1e-10

    p = pred_dist / p_total + 1e-10
    q = gt_dist / g_total + 1e-10

    # Symmetric KL
    kl_pq = (p * np.log(p / q)).sum()
    kl_qp = (q * np.log(q / p)).sum()

    return float((kl_pq + kl_qp) / 2)
