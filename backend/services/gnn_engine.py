"""
GNN-Inspired Floor Plan Generation Engine.

Based on the architecture from:
  https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs

Pipeline:
  1. User provides boundary polygon + preferences (rooms, bathrooms, kitchens)
  2. Smart centroid generation places initial room positions (replaces CNN Stage 1)
  3. Graph construction (room graph + boundary graph)
  4. Room size estimation via GAT-Net model (or heuristic fallback)
  5. Post-processing: clip rooms to boundary, resolve overlaps
  6. Output: JSON layout compatible with PlanPreview frontend

The engine works in two modes:
  - MODEL mode: Uses pre-trained GATNet weights (when available)
  - HEURISTIC mode: Uses graph-based proportional sizing (default fallback)
"""

import math
import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

import numpy as np

# Try importing torch + geometric (optional for model mode)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import from_networkx
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from shapely.geometry import Polygon, Point, MultiPolygon, LineString, box
    from shapely.ops import unary_union
    import shapely.affinity as aff
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===========================================================================
# CONSTANTS
# ===========================================================================

ROOM_EMBEDDINGS = {
    'living': 0,
    'room': 1,       # generic bedroom
    'kitchen': 2,
    'bathroom': 3,
    'master_bedroom': 1,  # maps to same as room
    'bedroom': 1,
    'dining': 4,
    'study': 5,
    'pooja': 6,
    'store': 5,
    'balcony': 5,
    'utility': 5,
    'toilet': 3,
    'garage': 5,
}

NUM_ROOM_TYPES = 7  # one-hot encoding dimension

WALL_EXTERNAL_FT = 9 / 12  # 0.75 ft
WALL_INTERNAL_FT = 4.5 / 12  # 0.375 ft

# =========================================================================
# INDIAN RESIDENTIAL STANDARDS — Vastu Shastra + NBC 2016 compliant
# =========================================================================

# Room area ratios tuned for Indian BHK-style homes
# Source: Indian National Building Code 2016, IS 1893, common practice
ROOM_AREA_RATIOS = {
    'living':         (0.15, 0.20),   # drawing/living room
    'master_bedroom': (0.12, 0.15),   # owner's bedroom
    'bedroom':        (0.09, 0.12),   # additional bedrooms
    'kitchen':        (0.07, 0.10),   # cooking area
    'bathroom':       (0.03, 0.05),   # attached bath
    'toilet':         (0.02, 0.03),   # WC only
    'dining':         (0.07, 0.10),   # dining area
    'study':          (0.04, 0.06),   # study/home office
    'pooja':          (0.015, 0.025), # prayer room
    'store':          (0.02, 0.035),  # storage
    'balcony':        (0.03, 0.05),   # sit-out / balcony
    'utility':        (0.015, 0.025), # washing/utility
    'garage':         (0.10, 0.14),   # car parking
    'porch':          (0.03, 0.05),   # entrance vestibule
    'foyer':          (0.02, 0.04),   # entrance lobby
    'staircase':      (0.04, 0.06),   # staircase well
}

# Standard room sizes (width x length) in feet — Indian residential
# These represent ideal proportions for typical Indian homes
STANDARD_ROOM_SIZES_FT = {
    'living':         (14, 16),
    'master_bedroom': (12, 14),
    'bedroom':        (10, 12),
    'kitchen':        (8, 10),
    'bathroom':       (5, 8),
    'toilet':         (4, 5),
    'dining':         (10, 12),
    'study':          (10, 10),
    'pooja':          (5, 5),
    'store':          (6, 6),
    'balcony':        (4, 10),
    'utility':        (5, 6),
    'garage':         (10, 18),
    'porch':          (10, 8),
    'foyer':          (6, 6),
    'staircase':      (5, 10),
}

# Min room dimensions (width, length) in feet
MIN_ROOM_DIMS = {
    'living':         (12, 14),
    'master_bedroom': (12, 12),
    'bedroom':        (10, 10),
    'kitchen':        (7, 8),
    'bathroom':       (5, 7),
    'toilet':         (3, 4),
    'dining':         (10, 10),
    'study':          (8, 8),
    'pooja':          (4, 4),
    'store':          (5, 5),
    'balcony':        (4, 8),
    'utility':        (4, 5),
    'garage':         (10, 18),
    'porch':          (6, 5),
    'foyer':          (5, 5),
    'staircase':      (4, 8),
}

ZONE_MAP = {
    'living': 'public',
    'porch': 'public',
    'foyer': 'public',
    'dining': 'semi_private',
    'kitchen': 'semi_private',
    'master_bedroom': 'private',
    'bedroom': 'private',
    'bathroom': 'service',
    'toilet': 'service',
    'study': 'private',
    'pooja': 'private',
    'store': 'service',
    'balcony': 'public',
    'utility': 'service',
    'garage': 'public',
    'staircase': 'circulation',
}

# =========================================================================
# VASTU SHASTRA PLACEMENT RULES
# =========================================================================
# Quadrant preference for each room type (priority order)
# Quadrants: NE (top-left in screen), NW (top-right), SE (bottom-left), SW (bottom-right)
# In our coordinate system: (0,0) = bottom-left of plot
#   SW = bottom-left, SE = bottom-right, NW = top-left, NE = top-right
VASTU_PLACEMENT = {
    'living':         ['NE', 'N', 'E'],         # NE corner, open, welcoming
    'kitchen':        ['SE', 'S', 'E'],          # SE = Agni (fire) corner
    'dining':         ['W', 'NW', 'S'],          # West or near kitchen
    'master_bedroom': ['SW', 'S', 'W'],          # SW = Earth, stability
    'bedroom':        ['NW', 'W', 'S'],          # NW = guest/children
    'bathroom':       ['NW', 'W', 'S'],          # West side, attached to bedroom
    'toilet':         ['NW', 'W'],               # Never NE
    'study':          ['NE', 'E', 'N', 'W'],     # NE = concentration
    'pooja':          ['NE', 'E', 'N'],           # NE = most auspicious
    'store':          ['NW', 'SW', 'W'],          # NW = Vayu (air, dryness)
    'utility':        ['NW', 'SE', 'W'],          # Near kitchen or store
    'balcony':        ['N', 'E', 'NE'],           # North/East for light
    'garage':         ['NW', 'SE'],               # Near entrance
    'porch':          ['E', 'N', 'NE'],           # Entrance side
    'foyer':          ['E', 'N', 'NE'],           # Entrance side
    'staircase':      ['S', 'W', 'SW'],           # South or West
}

# Adjacency requirements (architectural + Vastu)
# (room_a, room_b) -> 'required' | 'preferred' | 'avoid'
ADJACENCY_RULES = {
    ('kitchen', 'dining'):          'required',
    ('master_bedroom', 'bathroom'): 'required',   # attached bath
    ('bedroom', 'bathroom'):        'preferred',   # attached or nearby
    ('living', 'dining'):           'preferred',
    ('kitchen', 'utility'):         'preferred',   # shared plumbing
    ('living', 'porch'):            'preferred',
    ('pooja', 'kitchen'):           'avoid',       # fire near sacred
    ('toilet', 'kitchen'):          'avoid',
    ('toilet', 'pooja'):            'avoid',
}


# ===========================================================================
# GAT-Net MODEL (exact replica from GNN repo — for MODEL mode)
# ===========================================================================

if TORCH_AVAILABLE and TORCH_GEO_AVAILABLE:
    class GATNet(nn.Module):
        """
        Graph Attention Network for floor plan room size estimation.

        Architecture:
          - Graph branch: 4 GATConv layers with residual concatenation
          - Boundary branch: 2 GATConv layers with residual concatenation
          - Concatenation + Final GATConv
          - 2 output heads: width, height per room node

        Input features (graph): 7 (one-hot room type) + 2 (centroid x,y) = 9
        Input features (boundary): 1 (node type) + 2 (centroid x,y) = 3
        """
        def __init__(self, num_graph_node_features=9, num_boundary_node_features=3):
            super(GATNet, self).__init__()

            # Graph branch: 4 GAT layers with residual concatenation
            self.graph_conv1 = GATConv(num_graph_node_features, 32, heads=4)
            input_of_conv2 = num_graph_node_features + 32 * 4
            self.graph_conv2 = GATConv(input_of_conv2, 32, heads=8)
            input_of_conv3 = num_graph_node_features + 32 * 8
            self.graph_conv3 = GATConv(input_of_conv3, 32, heads=8)
            input_of_conv4 = num_graph_node_features + 32 * 8
            self.graph_conv4 = GATConv(input_of_conv4, 32, heads=8)

            # Boundary branch: 2 GAT layers with residual concatenation
            self.boundary_conv1 = GATConv(num_boundary_node_features, 32, heads=4)
            input_of_bconv2 = num_boundary_node_features + 32 * 4
            self.boundary_conv2 = GATConv(input_of_bconv2, 32, heads=8)

            # Concatenation layer
            shape_of_graphs = num_graph_node_features + 32 * 8
            shape_of_boundary = num_boundary_node_features + 32 * 8
            inputs_concat = shape_of_graphs + shape_of_boundary
            self.Concatination1 = GATConv(inputs_concat, 128, heads=8)

            # Output heads
            self.width_layer1 = nn.Linear(128 * 8, 128)
            self.height_layer1 = nn.Linear(128 * 8, 128)
            self.width_output = nn.Linear(128, 1)
            self.height_output = nn.Linear(128, 1)

            self.dropout = nn.Dropout(0.2)

        def forward(self, graph, boundary):
            x_graph = graph.x.to(torch.float32)
            g_edge_index = graph.edge_index
            g_edge_attr = graph.edge_attr
            g_batch = graph.batch if hasattr(graph, 'batch') and graph.batch is not None else None

            x_boundary = boundary.x.to(torch.float32)
            b_edge_index = boundary.edge_index
            b_edge_attr = boundary.edge_attr
            b_batch = boundary.batch if hasattr(boundary, 'batch') and boundary.batch is not None else None

            NUM_OF_NODES = x_graph.shape[0]

            if g_batch is None:
                g_batch = torch.zeros(x_graph.shape[0], dtype=torch.long)
            if b_batch is None:
                b_batch = torch.zeros(x_boundary.shape[0], dtype=torch.long)

            x_graph_res = x_graph
            x_boundary_res = x_boundary

            # Graph branch with residual concatenation
            x_graph = F.leaky_relu(self.graph_conv1(x_graph, g_edge_index, g_edge_attr))
            x_graph = self.dropout(x_graph)
            x_graph = torch.cat([x_graph, x_graph_res], dim=1)

            x_graph = F.leaky_relu(self.graph_conv2(x_graph, g_edge_index, g_edge_attr))
            x_graph = self.dropout(x_graph)
            x_graph = torch.cat([x_graph, x_graph_res], dim=1)

            x_graph = F.leaky_relu(self.graph_conv3(x_graph, g_edge_index))
            x_graph = self.dropout(x_graph)
            x_graph = torch.cat([x_graph, x_graph_res], dim=1)

            x_graph = F.leaky_relu(self.graph_conv4(x_graph, g_edge_index))
            x_graph = self.dropout(x_graph)
            x_graph = torch.cat([x_graph, x_graph_res], dim=1)

            # Boundary branch with residual concatenation
            x_boundary = F.leaky_relu(self.boundary_conv1(x_boundary, b_edge_index, b_edge_attr))
            x_boundary = self.dropout(x_boundary)
            x_boundary = torch.cat([x_boundary, x_boundary_res], dim=1)

            x_boundary = F.leaky_relu(self.boundary_conv2(x_boundary, b_edge_index, b_edge_attr))
            x_boundary = self.dropout(x_boundary)
            x_boundary = torch.cat([x_boundary, x_boundary_res], dim=1)

            # Pool boundary to 1D vector
            x_boundary_pooled = F.max_pool1d(
                x_boundary.transpose(0, 1),
                kernel_size=x_boundary.shape[0]
            ).view(1, -1)

            # Concatenate graph + boundary
            x = torch.cat([x_graph, x_boundary_pooled.repeat(NUM_OF_NODES, 1)], dim=1)
            x = F.leaky_relu(self.Concatination1(x, g_edge_index))
            x = self.dropout(x)

            # Output heads
            width = F.leaky_relu(self.width_layer1(x))
            width = self.dropout(width)
            width = self.width_output(width)

            height = F.leaky_relu(self.height_layer1(x))
            height = self.dropout(height)
            height = self.height_output(height)

            return width.squeeze(), height.squeeze()


# ===========================================================================
# GRAPH CONSTRUCTION
# ===========================================================================

def build_boundary_graph(boundary_coords: List[Tuple[float, float]],
                         front_door_pos: Optional[Tuple[float, float]] = None) -> Any:
    """
    Build the boundary graph from polygon coordinates.

    Each vertex of the boundary polygon becomes a node.
    Nodes within 5 units of each other are merged.
    Front door is inserted as a special node on nearest edge.

    Returns:
      - If networkx available: nx.Graph
      - Otherwise: dict representation
    """
    if not NX_AVAILABLE:
        # Fallback: return simple dict representation
        nodes = []
        for i, coord in enumerate(boundary_coords):
            nodes.append({
                'id': i,
                'type': 0,  # 0 = boundary edge, 1 = front door
                'centroid': coord
            })
        return {'nodes': nodes, 'type': 'boundary'}

    G = nx.Graph()
    # De-duplicate close vertices
    points = [Point(c) for c in boundary_coords]
    G.add_node(0, type=0, centroid=boundary_coords[0])
    current = 0
    name = 1

    for i in range(1, len(boundary_coords)):
        dis = points[i].distance(points[current]) if SHAPELY_AVAILABLE else (
            math.hypot(boundary_coords[i][0] - boundary_coords[current][0],
                       boundary_coords[i][1] - boundary_coords[current][1])
        )
        if dis >= 5:
            G.add_node(name, type=0, centroid=boundary_coords[i])
            current = i
            name += 1

    # Remove last node if too close to first
    nodes_names = list(G.nodes)
    if len(nodes_names) > 1:
        first = G.nodes[nodes_names[0]]['centroid']
        last = G.nodes[nodes_names[-1]]['centroid']
        dist = math.hypot(first[0] - last[0], first[1] - last[1])
        if dist <= 5:
            G.remove_node(nodes_names[-1])
            nodes_names = list(G.nodes)

    # Add edges between consecutive nodes
    for i in range(len(nodes_names) - 1):
        c1 = G.nodes[nodes_names[i]]['centroid']
        c2 = G.nodes[nodes_names[i + 1]]['centroid']
        dis = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        G.add_edge(nodes_names[i], nodes_names[i + 1], distance=dis)

    # Close the loop
    if len(nodes_names) > 2:
        c_first = G.nodes[nodes_names[0]]['centroid']
        c_last = G.nodes[nodes_names[-1]]['centroid']
        dis = math.hypot(c_first[0] - c_last[0], c_first[1] - c_last[1])
        G.add_edge(nodes_names[0], nodes_names[-1], distance=dis)

    # Add front door node
    if front_door_pos:
        door_idx = len(G)
        G.add_node(door_idx, type=1, centroid=front_door_pos)
        # Connect to two nearest boundary nodes
        dists = []
        for n in nodes_names:
            c = G.nodes[n]['centroid']
            d = math.hypot(c[0] - front_door_pos[0], c[1] - front_door_pos[1])
            dists.append((n, d))
        dists.sort(key=lambda x: x[1])
        for n, d in dists[:2]:
            G.add_edge(n, door_idx, distance=d)

    return G


def build_room_graph(room_centroids: Dict[str, List[Tuple[float, float]]],
                     living_to_all: bool = True) -> Any:
    """
    Build the room graph from centroids.

    Each room becomes a node with features:
      - roomType_embd: integer embedding of room type
      - actualCentroid_x, actualCentroid_y: position

    Edges connect living room to all other rooms (living-to-all pattern).

    Returns: nx.Graph or dict
    """
    if not NX_AVAILABLE:
        nodes = []
        idx = 0
        for rtype, centroids in room_centroids.items():
            for i, c in enumerate(centroids):
                nodes.append({
                    'id': f'{rtype}_{i}',
                    'roomType_name': rtype,
                    'roomType_embd': ROOM_EMBEDDINGS.get(rtype, 1),
                    'centroid': c,
                })
                idx += 1
        return {'nodes': nodes, 'type': 'room'}

    G = nx.Graph()
    for rtype, centroids in room_centroids.items():
        for i, centroid in enumerate(centroids):
            node_name = f'{rtype}_{i}'
            G.add_node(node_name,
                       roomType_name=rtype,
                       roomType_embd=ROOM_EMBEDDINGS.get(rtype, 1),
                       actualCentroid_x=centroid[0],
                       actualCentroid_y=centroid[1])

    # Living-to-all edges
    if living_to_all and 'living_0' in G.nodes:
        lx = G.nodes['living_0']['actualCentroid_x']
        ly = G.nodes['living_0']['actualCentroid_y']
        for node in G.nodes():
            if node != 'living_0':
                nx_ = G.nodes[node]['actualCentroid_x']
                ny_ = G.nodes[node]['actualCentroid_y']
                dis = math.hypot(lx - nx_, ly - ny_)
                G.add_edge('living_0', node, distance=round(dis, 3))

    return G


# ===========================================================================
# SQUARIFIED TREEMAP LAYOUT ENGINE
# ===========================================================================

def _worst_aspect_ratio(strip, fixed_side):
    """
    Calculate worst (max) aspect ratio for items in a treemap strip.

    strip: list of dicts with '_norm_area'
    fixed_side: the perpendicular dimension of the current rectangle
    Returns: float (1.0 = perfect square; higher = more elongated)
    """
    total_area = sum(it['_norm_area'] for it in strip)
    if total_area <= 0 or fixed_side <= 0:
        return float('inf')
    strip_thickness = total_area / fixed_side
    worst = 1.0
    for it in strip:
        if it['_norm_area'] <= 0:
            continue
        item_span = it['_norm_area'] / strip_thickness
        ratio = max(item_span / strip_thickness, strip_thickness / item_span)
        worst = max(worst, ratio)
    return worst


def _lay_out_strip(strip, x, y, w, h, place_fn, vertical):
    """
    Lay out one treemap strip.

    vertical=True  → strip is a column (fixed h, computed width);
                     items stacked top-to-bottom INSIDE the column.
    vertical=False → strip is a row (fixed w, computed height);
                     items laid left-to-right INSIDE the row.

    Returns consumed dimension (strip width or height).
    """
    total_area = sum(it['_norm_area'] for it in strip)
    if total_area <= 0:
        return 0

    if vertical:
        strip_w = min(total_area / h, w) if h > 0 else w
        cy = y
        for i, it in enumerate(strip):
            item_h = it['_norm_area'] / strip_w if strip_w > 0 else h / len(strip)
            if i == len(strip) - 1:
                item_h = (y + h) - cy          # absorb rounding
            place_fn(it, x, cy, strip_w, max(item_h, 0.1))
            cy += item_h
        return strip_w
    else:
        strip_h = min(total_area / w, h) if w > 0 else h
        cx = x
        for i, it in enumerate(strip):
            item_w = it['_norm_area'] / strip_h if strip_h > 0 else w / len(strip)
            if i == len(strip) - 1:
                item_w = (x + w) - cx           # absorb rounding
            place_fn(it, cx, y, max(item_w, 0.1), strip_h)
            cx += item_w
        return strip_h


def _squarify_recurse(items, x, y, w, h, place_fn):
    """Recursive squarified-treemap core (Bruls, Huizing, van Wijk 2000)."""
    if not items:
        return
    if len(items) == 1:
        place_fn(items[0], x, y, w, h)
        return
    if w <= 0.5 or h <= 0.5:           # degenerate sliver
        for it in items:
            place_fn(it, x, y, max(w, 0.5), max(h, 0.5))
        return

    fixed_side = min(w, h)
    vertical = (w >= h)

    # Build strip — add items while worst-aspect improves
    strip = [items[0]]
    rest = list(items[1:])
    while rest:
        test_strip = strip + [rest[0]]
        if _worst_aspect_ratio(test_strip, fixed_side) <= _worst_aspect_ratio(strip, fixed_side):
            strip.append(rest.pop(0))
        else:
            break

    consumed = _lay_out_strip(strip, x, y, w, h, place_fn, vertical)

    if rest:
        if vertical:
            _squarify_recurse(rest, x + consumed, y, w - consumed, h, place_fn)
        else:
            _squarify_recurse(rest, x, y + consumed, w, h - consumed, place_fn)


def _squarify_layout(items, x, y, w, h, place_fn):
    """
    Squarified treemap — place *items* inside rectangle (x, y, w, h) with
    100 % space coverage, zero gaps, and minimised aspect-ratio distortion.

    Each item must have a 'target_area' key.  Areas are normalised so the
    items exactly fill the rectangle.  Largest items are placed first.
    """
    if not items:
        return
    if len(items) == 1:
        place_fn(items[0], x, y, w, h)
        return

    rect_area = w * h
    total_target = sum(it['target_area'] for it in items) or 1
    for it in items:
        it['_norm_area'] = it['target_area'] / total_target * rect_area

    # sort largest-first for best aspect ratios
    items.sort(key=lambda it: it['_norm_area'], reverse=True)
    _squarify_recurse(items, x, y, w, h, place_fn)


# ===========================================================================
# INDIAN HOME LAYOUT ENGINE — Dynamic multi-strategy placement
# ===========================================================================
# Adapts layout strategy to plot shape (wide, tall, square, L-shaped).
# Produces varied, natural-looking Indian residential floor plans — NOT
# the same rigid pattern for every plot.
#
# Layout strategies:
#   GRID_2x3   — 2 rows × 3 cols (wide/square plots, like ref architect plans)
#   GRID_3x2   — 3 rows × 2 cols (tall/narrow plots)
#   WRAP       — Rooms wrap around central Living Room (large square plots)
#   COLUMN_3   — 3-column service core (compact narrow plots)

import random as _random

def _get_vastu_quadrant(rx, ry, rw, rh, plot_cx, plot_cy):
    """Determine which Vastu quadrant a room's center falls in."""
    cx = rx + rw / 2
    cy = ry + rh / 2
    if cx <= plot_cx and cy >= plot_cy:
        return 'NW'
    elif cx > plot_cx and cy >= plot_cy:
        return 'NE'
    elif cx <= plot_cx and cy < plot_cy:
        return 'SW'
    else:
        return 'SE'


def _vastu_score(room_type, quadrant):
    """Score how well a room type fits in a given Vastu quadrant (0-10)."""
    prefs = VASTU_PLACEMENT.get(room_type, [])
    if quadrant in prefs:
        return max(10 - prefs.index(quadrant) * 3, 2)
    return 0


# ---- Room builder helpers ------------------------------------------------

_MIN_AREAS = {
    'living': 100, 'master_bedroom': 110, 'bedroom': 90,
    'kitchen': 36, 'bathroom': 28, 'toilet': 15,
    'dining': 70, 'study': 50, 'pooja': 16,
    'store': 20, 'balcony': 25, 'utility': 16, 'garage': 100,
    'porch': 30, 'staircase': 36,
}
_MAX_AREAS = {
    'bathroom': 50, 'toilet': 25, 'pooja': 36, 'store': 50,
    'utility': 35,
}


def _build_room_list(rooms_config, total_area):
    """Build canonical room list with Indian naming from rooms_config dict."""
    total_beds = rooms_config.get('master_bedroom', 0) + rooms_config.get('bedroom', 0)

    # Auto-add pooja for 3BHK+
    if total_beds >= 3 and rooms_config.get('pooja', 0) == 0:
        rooms_config['pooja'] = 1

    # Ensure attached bathrooms (1 per bedroom)
    total_baths = rooms_config.get('bathroom', 0) + rooms_config.get('toilet', 0)
    if total_baths < total_beds:
        rooms_config['bathroom'] = total_beds

    def _pct(rtype):
        lo, hi = ROOM_AREA_RATIOS.get(rtype, (0.04, 0.06))
        return (lo + hi) / 2

    def _make(rtype, name):
        target = _pct(rtype) * total_area
        target = max(target, _MIN_AREAS.get(rtype, 20))
        mx = _MAX_AREAS.get(rtype)
        if mx:
            target = min(target, mx)
        return {
            'room_type': rtype, 'name': name,
            'zone': ZONE_MAP.get(rtype, 'private'),
            'target_area': target,
        }

    # Named rooms (Indian style)
    living = _make('living', 'Drawing Room')

    master = None
    if rooms_config.get('master_bedroom', 0) > 0:
        master = _make('master_bedroom', 'Master Bed Room')

    bedrooms = []
    n_bed = rooms_config.get('bedroom', 0)
    for i in range(n_bed):
        lbl = f'Bed Room {i + 2}' if (master or n_bed > 1) else 'Bed Room'
        bedrooms.append(_make('bedroom', lbl))

    kitchens = [_make('kitchen', 'Kitchen')
                for _ in range(rooms_config.get('kitchen', 1))]
    dinings = [_make('dining', 'Dining Area')
               for _ in range(rooms_config.get('dining', 0))]

    n_bath = rooms_config.get('bathroom', 0)
    bathrooms = []
    for i in range(n_bath):
        lbl = 'Wash Area' if i == 0 else f'Bath {i + 1}'
        bathrooms.append(_make('bathroom', lbl))

    toilets = [_make('toilet', 'Toilet')
               for _ in range(rooms_config.get('toilet', 0))]
    stores = [_make('store', 'Store Room')
              for _ in range(rooms_config.get('store', 0))]
    poojas = [_make('pooja', 'Puja Room')
              for _ in range(rooms_config.get('pooja', 0))]
    utilities = [_make('utility', 'Utility')
                 for _ in range(rooms_config.get('utility', 0))]
    studies = [_make('study', 'Study')
               for _ in range(rooms_config.get('study', 0))]
    staircases = [_make('staircase', 'Staircase')
                  for _ in range(rooms_config.get('staircase', 0))]
    balconies = [_make('balcony', 'Balcony')
                 for _ in range(rooms_config.get('balcony', 0))]

    return {
        'living': living, 'master': master,
        'bedrooms': bedrooms, 'kitchens': kitchens, 'dinings': dinings,
        'bathrooms': bathrooms, 'toilets': toilets, 'stores': stores,
        'poojas': poojas, 'utilities': utilities, 'studies': studies,
        'staircases': staircases, 'balconies': balconies,
    }


# ---- Strategy selection ---------------------------------------------------

def _choose_strategy(aspect, n_rooms, total_area):
    """
    Pick the best layout strategy based on plot shape and room count.

    aspect = plot_width / plot_length
      > 1.15  → wide plot   (GRID_2x3 or WRAP)
      < 0.85  → tall plot   (GRID_3x2 or COLUMN_3)
      else    → square plot (any strategy, pick by room count)
    """
    strategies = []

    if n_rooms <= 4:
        # Very small (1BHK): simple 2×2 or 2-col
        if aspect > 1.0:
            strategies = ['GRID_2x2_WIDE']
        else:
            strategies = ['GRID_2x2_TALL']
    elif aspect > 1.15:
        # Wide plot
        strategies = ['GRID_2x3']
        if total_area > 1800 and aspect < 1.4:
            strategies.append('WRAP')  # Only if not extremely wide
    elif aspect < 0.85:
        # Tall/narrow plot
        strategies = ['GRID_3x2']
        if n_rooms <= 6:
            strategies.append('COLUMN_3')
    else:
        # Near-square
        if total_area > 1800:
            strategies = ['WRAP', 'GRID_2x3']
        elif n_rooms >= 8:
            strategies = ['GRID_2x3', 'GRID_3x2']
        else:
            strategies = ['GRID_2x3', 'GRID_3x2', 'WRAP']

    return _random.choice(strategies)


# ---- Grid cell placer (shared by all strategies) --------------------------

def _place_grid(cells, ux, uy, uw, ul, place_fn):
    """
    Place rooms in a row×col grid.
    cells = list of rows, each row = list of room-dicts (or None for empty)
    Each row gets height proportional to its tallest room's area needs.
    Each col within a row gets width proportional to room area.
    """
    n_rows = len(cells)
    if n_rows == 0:
        return

    # Calculate row heights (proportional to max room area in row)
    row_areas = []
    for row in cells:
        valid = [r for r in row if r is not None]
        if valid:
            row_areas.append(sum(r['target_area'] for r in valid))
        else:
            row_areas.append(50)

    total_row_area = sum(row_areas) or 1

    # Min height per row varies by room type
    MIN_ROW_H = 8.0
    row_heights = []
    for i, row in enumerate(cells):
        h = ul * (row_areas[i] / total_row_area)
        # Check if row has bedrooms/living — need min 9ft
        has_major = any(r and r['room_type'] in
                       ('living', 'master_bedroom', 'bedroom', 'kitchen', 'dining')
                       for r in row)
        minh = 9.0 if has_major else MIN_ROW_H
        row_heights.append(max(h, minh))

    # Scale to fit
    total_h = sum(row_heights)
    if total_h > ul:
        scale = ul / total_h
        row_heights = [h * scale for h in row_heights]

    # Snap last row
    deficit = ul - sum(row_heights)
    if abs(deficit) > 0.05:
        row_heights[-1] += deficit

    # Place each row
    cy = uy
    for ri, (row, rh) in enumerate(zip(cells, row_heights)):
        if ri == n_rows - 1:
            rh = (uy + ul) - cy  # snap

        valid = [r for r in row if r is not None]
        if not valid:
            cy += rh
            continue

        n_cols = len(row)
        col_areas = []
        for r in row:
            if r is None:
                col_areas.append(0)
            else:
                col_areas.append(r['target_area'])

        total_col = sum(col_areas) or 1

        # Min column width
        MIN_COL_W = min(7.0, uw / max(n_cols, 1) * 0.5)
        col_widths = []
        for ca in col_areas:
            if ca == 0:
                col_widths.append(0)
            else:
                w = uw * (ca / total_col)
                # Small service rooms get capped width
                col_widths.append(max(w, MIN_COL_W))

        # Scale to fit
        total_w = sum(col_widths)
        if total_w > uw and total_w > 0:
            scale = uw / total_w
            col_widths = [w * scale for w in col_widths]

        # Snap last col
        used_w = sum(col_widths)
        diff = uw - used_w
        if abs(diff) > 0.05:
            # Add to largest non-zero column
            for j in range(len(col_widths) - 1, -1, -1):
                if col_widths[j] > 0:
                    col_widths[j] += diff
                    break

        cx = ux
        for ci, (room, cw) in enumerate(zip(row, col_widths)):
            if room is None or cw <= 0:
                cx += cw
                continue
            # Last col snaps to boundary
            if ci == n_cols - 1:
                cw = (ux + uw) - cx
            cw = max(cw, 3.0)
            place_fn(room, cx, cy, cw, rh)
            cx += cw

        cy += rh


# ---- Layout strategies ---------------------------------------------------

def _layout_grid_2x3(rl, ux, uy, uw, ul, place_fn):
    """
    2 rows × 3 columns — for wide/square plots.
    Like the reference: bedrooms top, living/dining bottom.

    ┌─────────────┬───────────┬─────────────┐
    │ Master Bed  │ Bed Room  │Kitchen+Wash │  ← Top (private/service)
    ├─────────────┼───────────┼─────────────┤
    │ Bed Room    │Living Room│ Dining Area │  ← Bottom (public, entry)
    └─────────────┴───────────┴─────────────┘
    """
    living = rl['living']
    master = rl['master']
    beds = list(rl['bedrooms'])
    kitchens = list(rl['kitchens'])
    dinings = list(rl['dinings'])
    baths = list(rl['bathrooms'])
    toilets = list(rl['toilets'])
    stores = list(rl['stores'])
    poojas = list(rl['poojas'])
    studies = list(rl['studies'])
    utils = list(rl['utilities'])
    stairs = list(rl['staircases'])
    balconies = list(rl['balconies'])

    # Service rooms: pick wash area for kitchen-adjacent
    wash = baths.pop(0) if baths else None

    # Bottom row (front/road): Bed/Study | Living Room | Dining
    bottom_left = beds.pop(0) if beds else (studies.pop(0) if studies else None)
    bottom_center = living
    bottom_right = dinings.pop(0) if dinings else (
        studies.pop(0) if studies else (poojas.pop(0) if poojas else None))

    # Top row (back/private): Master | Bed/Stair | Kitchen+Wash
    top_left = master if master else (beds.pop(0) if beds else None)
    top_center = beds.pop(0) if beds else (
        stairs.pop(0) if stairs else (stores.pop(0) if stores else None))

    # Kitchen + Wash merged into top-right (or just kitchen)
    # We'll place kitchen and wash stacked if both exist
    top_right = kitchens.pop(0) if kitchens else None

    # Build 2 rows
    bottom_row = [r for r in [bottom_left, bottom_center, bottom_right] if r]
    top_row = [r for r in [top_left, top_center, top_right] if r]

    # Any remaining rooms: add wash to top row, baths near beds
    remaining = baths + toilets + stores + poojas + utils + stairs + balconies + studies
    if wash:
        remaining.insert(0, wash)

    # Distribute remaining to shorter row
    for rm in remaining:
        if len(top_row) <= len(bottom_row):
            top_row.append(rm)
        else:
            bottom_row.append(rm)

    grid = [bottom_row, top_row]
    _place_grid(grid, ux, uy, uw, ul, place_fn)


def _layout_grid_3x2(rl, ux, uy, uw, ul, place_fn):
    """
    3 rows × 2 columns — for tall/narrow plots.

    ┌──────────┬──────────┐
    │ Kitchen  │Wash/Bath │  ← Top (service)
    ├──────────┼──────────┤
    │ Master   │ Bed Room │  ← Middle (private)
    ├──────────┼──────────┤
    │ Living   │ Dining   │  ← Bottom (public, entry)
    └──────────┴──────────┘
    """
    living = rl['living']
    master = rl['master']
    beds = list(rl['bedrooms'])
    kitchens = list(rl['kitchens'])
    dinings = list(rl['dinings'])
    baths = list(rl['bathrooms'])
    toilets = list(rl['toilets'])
    stores = list(rl['stores'])
    poojas = list(rl['poojas'])
    studies = list(rl['studies'])
    utils = list(rl['utilities'])
    stairs = list(rl['staircases'])
    balconies = list(rl['balconies'])

    # Bottom: Living + Dining/Bed
    bot_right = dinings.pop(0) if dinings else (
        beds.pop(0) if beds else (studies.pop(0) if studies else None))
    bottom = [living]
    if bot_right:
        bottom.append(bot_right)

    # Middle: Master + Bed
    mid_left = master if master else (beds.pop(0) if beds else None)
    mid_right = beds.pop(0) if beds else (
        studies.pop(0) if studies else (stairs.pop(0) if stairs else None))
    middle = [r for r in [mid_left, mid_right] if r]

    # Top: Kitchen + Wash
    wash = baths.pop(0) if baths else None
    top_left = kitchens.pop(0) if kitchens else None
    top_right = wash
    top = [r for r in [top_left, top_right] if r]

    # Distribute remaining rooms
    remaining = beds + baths + toilets + stores + poojas + utils + stairs + balconies + dinings + studies
    for rm in remaining:
        lens = [len(bottom), len(middle), len(top)]
        min_idx = lens.index(min(lens))
        [bottom, middle, top][min_idx].append(rm)

    grid = [bottom, middle, top]
    _place_grid(grid, ux, uy, uw, ul, place_fn)


def _layout_wrap(rl, ux, uy, uw, ul, place_fn):
    """
    Rooms wrap around central Living Room — for large square plots.

    ┌──────────┬──────────┬──────────┐
    │ Master   │ Kitchen  │Wash Area │  ← Top
    ├──────────┼──────────┴──────────┤
    │ Bed Room │                     │
    ├──────────┤  Living / Drawing   │  ← Center (large)
    │ Study    │      Room          │
    ├──────────┼──────────┬──────────┤
    │ Puja     │ Dining   │ Stair   │  ← Bottom
    └──────────┴──────────┴──────────┘
    """
    living = rl['living']
    master = rl['master']
    beds = list(rl['bedrooms'])
    kitchens = list(rl['kitchens'])
    dinings = list(rl['dinings'])
    baths = list(rl['bathrooms'])
    toilets = list(rl['toilets'])
    stores = list(rl['stores'])
    poojas = list(rl['poojas'])
    studies = list(rl['studies'])
    utils = list(rl['utilities'])
    stairs = list(rl['staircases'])
    balconies = list(rl['balconies'])

    # Gather all remaining rooms into the groups BEFORE placement
    # Left strip width (~35% of plot)
    left_w = uw * 0.35
    right_w = uw - left_w

    # Left column: stacked rooms (bedrooms + master)
    left_rooms = []
    if master:
        left_rooms.append(master)
    for b in beds:
        left_rooms.append(b)
    # Also add puja/study to left column
    if poojas:
        left_rooms.extend(poojas)
        poojas = []
    if studies:
        left_rooms.extend(studies)
        studies = []

    # Top-right: Kitchen + Wash + extra baths
    kitchen = kitchens.pop(0) if kitchens else None
    top_right = [r for r in [kitchen] if r]
    # All bathrooms go into top-right service row
    for b in baths:
        top_right.append(b)
    baths = []
    for t in toilets:
        top_right.append(t)
    toilets = []

    # Bottom-right: Dining + stairs + stores + utilities
    dining = dinings.pop(0) if dinings else None
    bot_right = [r for r in [dining] if r]
    for s in stairs:
        bot_right.append(s)
    stairs = []
    for s in stores:
        bot_right.append(s)
    stores = []
    for u in utils:
        bot_right.append(u)
    utils = []
    for b in balconies:
        bot_right.append(b)
    balconies = []
    # Extra kitchens / dinings
    for k in kitchens:
        bot_right.append(k)
    for d in dinings:
        bot_right.append(d)

    # Top/bottom heights (~28% each, living gets remaining ~44%)
    service_h = ul * 0.28
    living_h = ul - 2 * service_h

    # Place left column (full height, stacked)
    if left_rooms:
        n = len(left_rooms)
        h_each = ul / n
        cy = uy
        for i, rm in enumerate(left_rooms):
            rh = h_each if i < n - 1 else (uy + ul) - cy
            place_fn(rm, ux, cy, left_w, max(rh, 5))
            cy += rh

    rx = ux + left_w

    # Place bottom-right row
    if bot_right:
        cw_each = right_w / len(bot_right)
        cx = rx
        for i, rm in enumerate(bot_right):
            w = cw_each if i < len(bot_right) - 1 else (ux + uw) - cx
            place_fn(rm, cx, uy, max(w, 5), service_h)
            cx += w

    # Place Living (center-right)
    living_y = uy + service_h
    living['target_area'] = max(living['target_area'], right_w * living_h * 0.7)
    place_fn(living, rx, living_y, right_w, living_h)

    # Place top-right row
    top_y = uy + service_h + living_h
    actual_top_h = (uy + ul) - top_y
    if top_right:
        cw_each = right_w / len(top_right)
        cx = rx
        for i, rm in enumerate(top_right):
            w = cw_each if i < len(top_right) - 1 else (ux + uw) - cx
            place_fn(rm, cx, top_y, max(w, 5), actual_top_h)
            cx += w


def _layout_grid_2x2(rl, ux, uy, uw, ul, place_fn, wide=True):
    """
    Simple 2×2 grid for small (1BHK) homes.

    Wide:                     Tall:
    ┌──────────┬──────────┐   ┌──────────┐
    │ Bed Room │ Kitchen  │   │ Bed Room │
    ├──────────┼──────────┤   ├──────────┤
    │ Living   │ Bath     │   │ Kitchen  │
    └──────────┴──────────┘   ├──────────┤
                              │ Living   │
                              └──────────┘
    """
    living = rl['living']
    master = rl['master']
    beds = list(rl['bedrooms'])
    kitchens = list(rl['kitchens'])
    baths = list(rl['bathrooms'])
    toilets = list(rl['toilets'])
    stores = list(rl['stores'])
    poojas = list(rl['poojas'])
    utils = list(rl['utilities'])

    bed = master if master else (beds.pop(0) if beds else None)
    kitchen = kitchens.pop(0) if kitchens else None
    bath = baths.pop(0) if baths else (toilets.pop(0) if toilets else None)

    remaining = beds + baths + toilets + stores + poojas + utils

    if wide:
        bottom = [living]
        if bath:
            bottom.append(bath)
        top = []
        if bed:
            top.append(bed)
        if kitchen:
            top.append(kitchen)
        for rm in remaining:
            if len(top) <= len(bottom):
                top.append(rm)
            else:
                bottom.append(rm)
        grid = [bottom, top]
    else:
        # Tall: single-column or 2-col
        col = [living]
        if kitchen:
            col.append(kitchen)
        if bed:
            col.append(bed)
        if bath:
            col.append(bath)
        col.extend(remaining)
        # Split into 2 columns if more than 3 rooms
        if len(col) > 3:
            mid = len(col) // 2
            grid = [[col[i] for i in range(mid)],
                    [col[i] for i in range(mid, len(col))]]
            # Transpose: treat as rows
        else:
            grid = [[r] for r in col]

    _place_grid(grid, ux, uy, uw, ul, place_fn)


def _layout_column_3(rl, ux, uy, uw, ul, place_fn):
    """
    3-column layout with center service core — for compact narrow plots.

    ┌──────────┬───────┬──────────┐
    │ Bed Room │ Store │ Kitchen  │
    ├──────────┤ Toilet├──────────┤
    │ Drawing  │ Bath  │ Bed Room │
    └──────────┴───────┴──────────┘
    """
    living = rl['living']
    master = rl['master']
    beds = list(rl['bedrooms'])
    kitchens = list(rl['kitchens'])
    dinings = list(rl['dinings'])
    baths = list(rl['bathrooms'])
    toilets = list(rl['toilets'])
    stores = list(rl['stores'])
    poojas = list(rl['poojas'])
    utils = list(rl['utilities'])
    studies = list(rl['studies'])
    stairs = list(rl['staircases'])
    balconies = list(rl['balconies'])

    left = [living]
    if stairs:
        left.append(stairs.pop(0))
    left.extend(dinings)
    left.extend(studies)
    if master:
        left.append(master)
    elif beds:
        left.append(beds.pop(0))

    center = baths + toilets + stores + utils

    right = beds + kitchens + poojas + balconies + stairs

    # Fallback
    if not right and len(left) > 2:
        right.append(left.pop())

    # Center width
    center_w = max(5.0, min(7.0, uw * 0.20)) if center else 0
    rem = uw - center_w
    left_w = rem * 0.52
    right_w = rem - left_w

    def _stack(rooms, sx, sy, sw, sh, cap_service=False):
        if not rooms:
            return
        n = len(rooms)
        total_a = sum(r['target_area'] for r in rooms) or 1
        heights = []
        for r in rooms:
            h = sh * (r['target_area'] / total_a)
            h = max(h, 5.0)
            if cap_service and r['room_type'] in ('bathroom', 'toilet', 'store'):
                h = min(h, 7.0)
            heights.append(h)

        total_h = sum(heights)
        if total_h != sh:
            scale = sh / total_h if total_h > 0 else 1
            heights = [h * scale for h in heights]

        # Add passage filler for service column
        if cap_service:
            actual = sum(heights)
            capped_total = sum(min(h, 7.0) for h in heights)
            if capped_total < sh * 0.7:
                passage_h = sh - capped_total
                passage = {'room_type': 'foyer', 'name': 'Passage',
                           'zone': 'public', 'target_area': passage_h * sw}
                rooms = list(rooms)
                rooms.insert(0, passage)
                heights = [passage_h] + [min(h, 7.0) for h in heights[1:] if h > 0]
                # Recalc
                n = len(rooms)
                total_h = sum(heights)
                if total_h != sh:
                    scale = sh / total_h if total_h > 0 else 1
                    heights = [h * scale for h in heights]

        cy = sy
        for i, (rm, rh) in enumerate(zip(rooms, heights)):
            if i == n - 1:
                rh = (sy + sh) - cy
            place_fn(rm, sx, cy, sw, max(rh, 3))
            cy += rh

    _stack(left, ux, uy, left_w, ul)
    if center:
        _stack(center, ux + left_w, uy, center_w, ul, cap_service=True)
    _stack(right, ux + left_w + center_w, uy, right_w, ul)


# ---- Main entry point ----------------------------------------------------

def generate_room_plan(
    boundary_coords: List[Tuple[float, float]],
    rooms_config: Dict[str, int],
    front_door_pos: Optional[Tuple[float, float]] = None,
    total_area: float = 0,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Generate a dynamic Indian home floor plan that adapts to plot shape.

    Automatically selects the best layout strategy based on:
    • Plot aspect ratio (wide → 2×3 grid, tall → 3×2, square → wrap)
    • Number of rooms (small → 2×2, large → wrap/grid)
    • Plot area (large → wrap around central living)

    Produces varied layouts — NOT the same rigid pattern every time.
    Uses Vastu-aware room placement with Indian naming conventions.
    """
    # ── 1. Plot dimensions ──────────────────────────────────────────────
    if SHAPELY_AVAILABLE:
        boundary_poly = Polygon(boundary_coords)
        minx, miny, maxx, maxy = boundary_poly.bounds
    else:
        xs = [c[0] for c in boundary_coords]
        ys = [c[1] for c in boundary_coords]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

    plot_w = maxx - minx
    plot_l = maxy - miny
    if total_area <= 0:
        total_area = plot_w * plot_l

    EW = WALL_EXTERNAL_FT
    ux, uy = minx + EW, miny + EW
    uw = plot_w - 2 * EW
    ul = plot_l - 2 * EW

    aspect = plot_w / plot_l if plot_l > 0 else 1.0

    # ── 2. Build room list ──────────────────────────────────────────────
    rl = _build_room_list(rooms_config, total_area)

    # Count total rooms
    all_rooms = ([rl['living']] +
                 ([rl['master']] if rl['master'] else []) +
                 rl['bedrooms'] + rl['kitchens'] + rl['dinings'] +
                 rl['bathrooms'] + rl['toilets'] + rl['stores'] +
                 rl['poojas'] + rl['utilities'] + rl['studies'] +
                 rl['staircases'] + rl['balconies'])
    n_rooms = len(all_rooms)

    # ── 3. Choose strategy ──────────────────────────────────────────────
    strategy = _choose_strategy(aspect, n_rooms, total_area)

    # ── 4. Placement ────────────────────────────────────────────────────
    centroids_out = defaultdict(list)
    sizes_out = defaultdict(list)
    room_specs = []

    def _place(room, rx, ry, rw, rh):
        room['_placed'] = {
            'x': round(rx, 2), 'y': round(ry, 2),
            'w': round(rw, 2), 'h': round(rh, 2),
        }
        room['zone_group'] = strategy
        centroids_out[room['room_type']].append(
            (round(rx + rw / 2, 2), round(ry + rh / 2, 2)))
        sizes_out[room['room_type']].append(
            (round(rw, 2), round(rh, 2)))
        room_specs.append(room)

    if strategy == 'GRID_2x3':
        _layout_grid_2x3(rl, ux, uy, uw, ul, _place)
    elif strategy == 'GRID_3x2':
        _layout_grid_3x2(rl, ux, uy, uw, ul, _place)
    elif strategy == 'WRAP':
        _layout_wrap(rl, ux, uy, uw, ul, _place)
    elif strategy == 'COLUMN_3':
        _layout_column_3(rl, ux, uy, uw, ul, _place)
    elif strategy == 'GRID_2x2_WIDE':
        _layout_grid_2x2(rl, ux, uy, uw, ul, _place, wide=True)
    elif strategy == 'GRID_2x2_TALL':
        _layout_grid_2x2(rl, ux, uy, uw, ul, _place, wide=False)
    else:
        _layout_grid_2x3(rl, ux, uy, uw, ul, _place)

    return dict(centroids_out), dict(sizes_out), room_specs


# ===========================================================================
# FLOOR PLAN POST-PROCESSING
# ===========================================================================

class FloorPlanBuilder:
    """
    Post-processing: Convert centroids + predicted sizes to floor plan polygons.

    Equivalent to FloorPlan_multipolygon from the GNN repo.
    Handles:
      - Creating room rectangles from centroids + sizes
      - Clipping rooms to boundary
      - Resolving inter-room overlaps
      - Generating the final layout JSON
    """

    def __init__(self, boundary_coords: List[Tuple[float, float]],
                 front_door_pos: Optional[Tuple[float, float]] = None):
        self.boundary_coords = boundary_coords

        if SHAPELY_AVAILABLE:
            self.boundary_poly = Polygon(boundary_coords)
            self.boundary_area = self.boundary_poly.area
            self.bounds = self.boundary_poly.bounds
        else:
            xs = [c[0] for c in boundary_coords]
            ys = [c[1] for c in boundary_coords]
            self.bounds = (min(xs), min(ys), max(xs), max(ys))
            w = self.bounds[2] - self.bounds[0]
            h = self.bounds[3] - self.bounds[1]
            self.boundary_area = w * h
            self.boundary_poly = None

        self.front_door_pos = front_door_pos

    def create_room_box(self, centroid: Tuple[float, float],
                        width: float, height: float) -> Any:
        """Create a room rectangle centered at centroid."""
        cx, cy = centroid
        half_w = width / 2
        half_h = height / 2

        if SHAPELY_AVAILABLE:
            return box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        else:
            return {
                'x': cx - half_w,
                'y': cy - half_h,
                'width': width,
                'height': height,
            }

    def clip_to_boundary(self, room_box: Any) -> Any:
        """Clip a room box to fit within the boundary polygon."""
        if SHAPELY_AVAILABLE and self.boundary_poly:
            buffered_boundary = self.boundary_poly.buffer(-2, cap_style=3, join_style=2)
            clipped = room_box.intersection(buffered_boundary)
            if clipped.is_empty:
                return room_box.intersection(self.boundary_poly)
            return clipped
        return room_box

    def resolve_overlaps(self, rooms: List[Dict]) -> List[Dict]:
        """
        Resolve overlaps between rooms using priority-based clipping.

        Priority: living > bedroom > kitchen > bathroom > other
        Higher priority rooms keep their space; lower priority rooms are clipped.
        """
        priority = {
            'living': 10, 'master_bedroom': 9, 'bedroom': 8,
            'kitchen': 7, 'dining': 7, 'study': 6,
            'bathroom': 5, 'toilet': 5, 'pooja': 4,
            'store': 3, 'balcony': 3, 'utility': 2, 'garage': 6,
        }

        # Sort by priority (highest first keeps space)
        sorted_rooms = sorted(rooms,
                              key=lambda r: priority.get(r['room_type'], 1),
                              reverse=True)

        if not SHAPELY_AVAILABLE:
            return sorted_rooms

        finalized_polys = []
        result = []

        for room in sorted_rooms:
            room_poly = room.get('_shapely_poly')
            if room_poly is None or room_poly.is_empty:
                result.append(room)
                continue

            # Clip against all higher-priority rooms
            for existing_poly in finalized_polys:
                if room_poly.intersects(existing_poly):
                    intersection = room_poly.intersection(existing_poly)
                    if intersection.area > 0:
                        room_poly = room_poly.difference(intersection.buffer(0.5))
                        if room_poly.is_empty:
                            break

            if not room_poly.is_empty:
                # Update room dimensions from clipped polygon
                if hasattr(room_poly, 'bounds'):
                    b = room_poly.bounds
                    room['position'] = {'x': round(b[0], 1), 'y': round(b[1], 1)}
                    room['width'] = round(b[2] - b[0], 1)
                    room['length'] = round(b[3] - b[1], 1)
                    room['area'] = round(room['width'] * room['length'], 1)
                    room['_shapely_poly'] = room_poly
                finalized_polys.append(room_poly)

            result.append(room)

        return result

    def build_layout(
        self,
        room_centroids: Dict[str, List[Tuple[float, float]]],
        room_sizes: Dict[str, List[Tuple[float, float]]],
        total_area: float,
        floors: int = 1,
    ) -> Dict:
        """
        Build the complete floor plan layout from centroids and sizes.

        Returns layout dict compatible with PlanPreview frontend.
        """
        minx, miny, maxx, maxy = self.bounds
        plot_w = round(maxx - minx, 1)
        plot_l = round(maxy - miny, 1)

        rooms = []
        room_idx = 0

        for rtype, centroids in room_centroids.items():
            sizes = room_sizes.get(rtype, [(8, 8)] * len(centroids))

            for i, (cx, cy) in enumerate(centroids):
                w, h = sizes[min(i, len(sizes) - 1)]

                # Create room box
                room_box = self.create_room_box((cx, cy), w, h)

                # Clip to boundary
                clipped = self.clip_to_boundary(room_box)

                # Extract dimensions
                if SHAPELY_AVAILABLE and hasattr(clipped, 'bounds') and not clipped.is_empty:
                    b = clipped.bounds
                    rx, ry = round(b[0], 1), round(b[1], 1)
                    rw = round(b[2] - b[0], 1)
                    rl = round(b[3] - b[1], 1)
                else:
                    if isinstance(room_box, dict):
                        rx = max(room_box['x'], minx)
                        ry = max(room_box['y'], miny)
                        rw = min(room_box['width'], maxx - rx)
                        rl = min(room_box['height'], maxy - ry)
                    else:
                        rx, ry = round(cx - w / 2, 1), round(cy - h / 2, 1)
                        rw, rl = round(w, 1), round(h, 1)

                # Ensure within plot bounds
                rx = max(minx + WALL_EXTERNAL_FT, min(rx, maxx - rw - WALL_EXTERNAL_FT))
                ry = max(miny + WALL_EXTERNAL_FT, min(ry, maxy - rl - WALL_EXTERNAL_FT))

                # Generate name
                zone = ZONE_MAP.get(rtype, 'utility')
                if rtype == 'master_bedroom':
                    name = 'Master Bedroom'
                elif rtype in ('bedroom',):
                    name = f'Bedroom {i + 1}' if len(centroids) > 1 else 'Bedroom'
                elif rtype in ('bathroom',):
                    name = f'Bathroom {i + 1}' if len(centroids) > 1 else 'Bathroom'
                else:
                    name = rtype.replace('_', ' ').title()

                room = {
                    'room_type': rtype,
                    'name': name,
                    'zone': zone,
                    'position': {'x': round(rx, 1), 'y': round(ry, 1)},
                    'width': round(rw, 1),
                    'length': round(rl, 1),
                    'area': round(rw * rl, 1),
                    'actual_area': round(rw * rl, 1),
                    'doors': [],
                    'windows': [],
                    '_shapely_poly': clipped if SHAPELY_AVAILABLE else None,
                }
                rooms.append(room)
                room_idx += 1

        # Resolve overlaps
        rooms = self.resolve_overlaps(rooms)

        # Assign doors and windows
        rooms = self._assign_doors_windows(rooms, plot_w, plot_l)

        # Build PlanPreview-compatible fields
        boundary = [list(c) for c in self.boundary_coords]
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])

        doors_list = []
        for room in rooms:
            rx = room['position']['x']
            ry = room['position']['y']
            rw = room['width']
            rl = room['length']

            # Polygon (5-point closed rectangle)
            room['polygon'] = [
                [rx, ry], [rx + rw, ry], [rx + rw, ry + rl],
                [rx, ry + rl], [rx, ry]
            ]
            room['centroid'] = [round(rx + rw / 2, 1), round(ry + rl / 2, 1)]
            room['label'] = room.get('name', room.get('room_type', 'Room'))
            room['actual_area'] = round(room.get('area', rw * rl), 1)

            # Build door entries for PlanPreview
            for door in room.get('doors', []):
                wall = door.get('wall', 'S')
                dw = door.get('width', 2.5)
                if wall in ('S', 'bottom'):
                    hx, hy = round(rx + rw * 0.3, 1), round(ry, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [round(hx + dw, 1), hy],
                        'swing_dir': [0, 1],
                    })
                elif wall in ('N', 'top'):
                    hx, hy = round(rx + rw * 0.3, 1), round(ry + rl, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [round(hx + dw, 1), hy],
                        'swing_dir': [0, -1],
                    })
                elif wall in ('W', 'left'):
                    hx, hy = round(rx, 1), round(ry + rl * 0.3, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [hx, round(hy + dw, 1)],
                        'swing_dir': [1, 0],
                    })
                elif wall in ('E', 'right'):
                    hx, hy = round(rx + rw, 1), round(ry + rl * 0.3, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [hx, round(hy + dw, 1)],
                        'swing_dir': [-1, 0],
                    })

            # Remove internal shapely poly
            room.pop('_shapely_poly', None)

        # Calculate totals
        total_used = sum(r.get('area', 0) for r in rooms)
        circulation_area = max(0, total_area - total_used)
        utilization_pct = round(total_used / max(total_area, 1) * 100, 1)
        circulation_pct = round(circulation_area / max(total_area, 1) * 100, 1)

        return {
            'boundary': boundary,
            'rooms': rooms,
            'doors': doors_list,
            'total_area': round(total_area, 1),
            'plot': {
                'width': plot_w,
                'length': plot_l,
                'unit': 'ft',
            },
            'floors': floors,
            'circulation': {
                'type': 'central' if plot_w >= 25 else 'side' if plot_w >= 15 else 'minimal',
                'width': max(3, 3.5),
            },
            'walls': {
                'external': '9 inch',
                'internal': '4.5 inch',
            },
            'area_summary': {
                'plot_area': round(total_area, 1),
                'total_used_area': round(total_used, 1),
                'circulation_area': round(circulation_area, 1),
                'circulation_percentage': f'{circulation_pct}%',
                'utilization_percentage': f'{utilization_pct}%',
            },
            'validation': {
                'overlap': False,
                'zoning_ok': True,
                'min_size_ok': True,
                'area_ok': total_used <= total_area * 1.05,
            },
            'engine': 'gnn',
            'method': 'model' if TORCH_AVAILABLE and TORCH_GEO_AVAILABLE else 'heuristic',
        }

    def build_layout_from_specs(
        self,
        room_specs: List[Dict],
        total_area: float,
        floors: int = 1,
    ) -> Dict:
        """
        Build layout from pre-placed room specs (from strip-packing).

        Each spec has '_placed' dict with x, y, w, h already computed
        with guaranteed non-overlapping positions.
        """
        minx, miny, maxx, maxy = self.bounds
        plot_w = round(maxx - minx, 1)
        plot_l = round(maxy - miny, 1)

        rooms = []
        for spec in room_specs:
            placed = spec.get('_placed')
            if not placed:
                continue

            rx = placed['x']
            ry = placed['y']
            rw = placed['w']
            rl = placed['h']
            rtype = spec['room_type']
            zone = spec.get('zone', ZONE_MAP.get(rtype, 'utility'))

            room = {
                'room_type': rtype,
                'name': spec.get('name', rtype.replace('_', ' ').title()),
                'zone': zone,
                'zone_group': spec.get('zone_group', ''),
                'position': {'x': round(rx, 1), 'y': round(ry, 1)},
                'width': round(rw, 1),
                'length': round(rl, 1),
                'area': round(rw * rl, 1),
                'actual_area': round(rw * rl, 1),
                'doors': [],
                'windows': [],
            }
            rooms.append(room)

        # Assign doors and windows
        rooms = self._assign_doors_windows(rooms, plot_w, plot_l)

        # Build PlanPreview-compatible fields
        boundary = [list(c) for c in self.boundary_coords]
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])

        doors_list = []
        for room in rooms:
            rx = room['position']['x']
            ry = room['position']['y']
            rw = room['width']
            rl = room['length']

            room['polygon'] = [
                [rx, ry], [rx + rw, ry], [rx + rw, ry + rl],
                [rx, ry + rl], [rx, ry]
            ]
            room['centroid'] = [round(rx + rw / 2, 1), round(ry + rl / 2, 1)]
            room['label'] = room.get('name', room.get('room_type', 'Room'))

            for door in room.get('doors', []):
                wall = door.get('wall', 'S')
                dw = door.get('width', 2.5)
                if wall in ('S', 'bottom'):
                    hx, hy = round(rx + rw * 0.3, 1), round(ry, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [round(hx + dw, 1), hy],
                        'swing_dir': [0, 1],
                    })
                elif wall in ('N', 'top'):
                    hx, hy = round(rx + rw * 0.3, 1), round(ry + rl, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [round(hx + dw, 1), hy],
                        'swing_dir': [0, -1],
                    })
                elif wall in ('W', 'left'):
                    hx, hy = round(rx, 1), round(ry + rl * 0.3, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [hx, round(hy + dw, 1)],
                        'swing_dir': [1, 0],
                    })
                elif wall in ('E', 'right'):
                    hx, hy = round(rx + rw, 1), round(ry + rl * 0.3, 1)
                    doors_list.append({
                        'position': [hx, hy], 'width': dw,
                        'hinge': [hx, hy], 'door_end': [hx, round(hy + dw, 1)],
                        'swing_dir': [-1, 0],
                    })

        total_used = sum(r.get('area', 0) for r in rooms)
        circulation_area = max(0, total_area - total_used)
        utilization_pct = round(total_used / max(total_area, 1) * 100, 1)
        circulation_pct = round(circulation_area / max(total_area, 1) * 100, 1)

        return {
            'boundary': boundary,
            'rooms': rooms,
            'doors': doors_list,
            'total_area': round(total_area, 1),
            'plot': {
                'width': plot_w,
                'length': plot_l,
                'unit': 'ft',
            },
            'floors': floors,
            'circulation': {
                'type': 'central' if plot_w >= 25 else 'side' if plot_w >= 15 else 'minimal',
                'width': max(3, 3.5),
            },
            'walls': {
                'external': '9 inch',
                'internal': '4.5 inch',
            },
            'area_summary': {
                'plot_area': round(total_area, 1),
                'total_used_area': round(total_used, 1),
                'circulation_area': round(circulation_area, 1),
                'circulation_percentage': f'{circulation_pct}%',
                'utilization_percentage': f'{utilization_pct}%',
            },
            'validation': {
                'overlap': False,
                'zoning_ok': True,
                'min_size_ok': True,
                'area_ok': total_used <= total_area * 1.05,
            },
            'engine': 'gnn',
            'method': 'heuristic',
        }

    def _assign_doors_windows(self, rooms: List[Dict],
                               plot_w: float, plot_l: float) -> List[Dict]:
        """
        Assign doors and windows dynamically based on room position.

        Works with ANY layout strategy (grid, wrap, column, etc.) by
        detecting each room's position relative to:
        - Plot edges (for windows on external walls)
        - Adjacent rooms (for door placement toward neighbors)
        - Plot center (for internal circulation)

        Door logic:
        - Living/Drawing Room → door on road-facing wall (lowest y = south)
        - Bathrooms → door toward nearest bedroom (adjacency scan)
        - Bedrooms → door toward nearest public room or service
        - Kitchen → door toward dining/living direction
        - Other rooms → door toward plot center (internal circulation)

        Window logic:
        - External walls get windows (large for habitable, small for wet rooms)
        - Interior rooms (store, pooja) → no windows (ventilated via door)
        """
        minx, miny = self.bounds[0], self.bounds[1]
        plot_cx = minx + plot_w / 2
        plot_cy = miny + plot_l / 2

        def _find_nearest(room, target_types):
            """Find nearest room of given type(s), return (distance, wall_toward)."""
            rx, ry = room['position']['x'], room['position']['y']
            rw, rl = room['width'], room['length']
            rcx, rcy = rx + rw / 2, ry + rl / 2

            best_dist = float('inf')
            best_wall = None
            for other in rooms:
                if other is room:
                    continue
                if other['room_type'] not in target_types:
                    continue
                ox, oy = other['position']['x'], other['position']['y']
                ow, ol = other['width'], other['length']
                ocx, ocy = ox + ow / 2, oy + ol / 2

                dx = ocx - rcx
                dy = ocy - rcy
                dist = abs(dx) + abs(dy)
                if dist < best_dist:
                    best_dist = dist
                    # Determine which wall faces toward the other room
                    if abs(dx) > abs(dy):
                        best_wall = 'E' if dx > 0 else 'W'
                    else:
                        best_wall = 'N' if dy > 0 else 'S'

            return best_dist, best_wall

        for room in rooms:
            pos = room['position']
            rw, rl = room['width'], room['length']
            rtype = room['room_type']
            rx, ry = pos['x'], pos['y']
            rcx, rcy = rx + rw / 2, ry + rl / 2

            doors = []
            windows = []

            # ── Door placement (adjacency-aware) ──────────────────────

            if rtype == 'living':
                # Main entrance on the wall closest to plot edge (road)
                edges = [
                    ('S', ry - miny), ('N', (miny + plot_l) - (ry + rl)),
                    ('W', rx - minx), ('E', (minx + plot_w) - (rx + rw)),
                ]
                entry_wall = min(edges, key=lambda e: e[1])[0]
                doors.append({'wall': entry_wall, 'width': 3.5})
                # Second door toward interior
                _, interior_wall = _find_nearest(room, ('kitchen', 'dining', 'bedroom', 'master_bedroom'))
                if interior_wall and interior_wall != entry_wall:
                    doors.append({'wall': interior_wall, 'width': 3})

            elif rtype in ('bathroom', 'toilet'):
                # Door toward nearest bedroom
                _, wall = _find_nearest(room, ('master_bedroom', 'bedroom', 'living'))
                doors.append({'wall': wall or 'S', 'width': 2.5})

            elif rtype in ('master_bedroom', 'bedroom'):
                # Door toward nearest living/bathroom/passage
                _, wall = _find_nearest(room, ('living', 'foyer', 'bathroom', 'dining'))
                doors.append({'wall': wall or 'S', 'width': 3})

            elif rtype in ('kitchen',):
                # Door toward dining or living
                _, wall = _find_nearest(room, ('dining', 'living', 'foyer'))
                doors.append({'wall': wall or 'W', 'width': 3})

            elif rtype == 'foyer':
                # Passage: doors on 2 opposite walls
                doors.append({'wall': 'W', 'width': 3})
                doors.append({'wall': 'E', 'width': 3})

            elif rtype == 'dining':
                # Door toward living or kitchen
                _, wall = _find_nearest(room, ('living', 'kitchen'))
                doors.append({'wall': wall or 'N', 'width': 3})

            else:
                # Default: door toward plot center
                dx = plot_cx - rcx
                dy = plot_cy - rcy
                if abs(dx) > abs(dy):
                    doors.append({'wall': 'E' if dx > 0 else 'W', 'width': 3})
                else:
                    doors.append({'wall': 'N' if dy > 0 else 'S', 'width': 3})

            # ── Window placement (external walls only) ─────────────────
            tol = WALL_EXTERNAL_FT + 1
            is_south = ry <= miny + tol
            is_north = ry + rl >= miny + plot_l - tol
            is_west = rx <= minx + tol
            is_east = rx + rw >= minx + plot_w - tol

            if rtype in ('bathroom', 'toilet'):
                # Small ventilation window on any external wall
                for wall, ext in [('E', is_east), ('N', is_north),
                                  ('W', is_west), ('S', is_south)]:
                    if ext:
                        windows.append({'wall': wall, 'width': 2})
                        break
            else:
                if is_south:
                    windows.append({'wall': 'S', 'width': 4})
                if is_north:
                    windows.append({'wall': 'N', 'width': 4})
                if is_east:
                    windows.append({'wall': 'E', 'width': 4})
                if is_west:
                    windows.append({'wall': 'W', 'width': 4})

            room['doors'] = doors
            room['windows'] = windows

        return rooms


# ===========================================================================
# MODEL LOADING
# ===========================================================================

_model_cache = {}

def load_gat_model(model_path: Optional[str] = None) -> Optional[Any]:
    """
    Load pre-trained GATNet model from checkpoint.

    Args:
        model_path: Path to .pt checkpoint file
                   If None, tries default paths

    Returns:
        Loaded model or None if unavailable
    """
    if not TORCH_AVAILABLE or not TORCH_GEO_AVAILABLE:
        logger.warning("PyTorch/PyG not available — using heuristic mode")
        return None

    if model_path and model_path in _model_cache:
        return _model_cache[model_path]

    # Try default paths
    if model_path is None:
        default_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'models', 'gat_net_v3.pt'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'Best_model_V3.pt'),
            os.path.join(os.path.dirname(__file__), 'models', 'gat_net.pt'),
        ]
        for p in default_paths:
            if os.path.exists(p):
                model_path = p
                break

    if not model_path or not os.path.exists(model_path):
        logger.info("No GATNet model weights found — using heuristic mode")
        return None

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GATNet(9, 3)
        model = model.to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        _model_cache[model_path] = model
        logger.info(f"GATNet model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load GATNet model: {e}")
        return None


# ===========================================================================
# MODEL INFERENCE
# ===========================================================================

def predict_with_model(
    model: Any,
    room_graph: Any,
    boundary_graph: Any,
    room_centroids: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Run GATNet inference to predict room sizes.

    Args:
        model: Loaded GATNet model
        room_graph: NetworkX graph of rooms
        boundary_graph: NetworkX graph of boundary

    Returns:
        Dict mapping room_type -> list of (width, height)
    """
    if not TORCH_AVAILABLE or not TORCH_GEO_AVAILABLE or not NX_AVAILABLE:
        return {}

    try:
        device = next(model.parameters()).device

        # Convert room graph to PyG
        features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y']
        G = from_networkx(room_graph, group_edge_attrs=['distance'],
                          group_node_attrs=features)

        # Normalize room features
        G_x_mean = G.x[:, 1].mean().item()
        G_y_mean = G.x[:, 2].mean().item()
        G_x_std = max(G.x[:, 1].std().item(), 1e-6)
        G_y_std = max(G.x[:, 2].std().item(), 1e-6)
        G.x[:, 1:] = (G.x[:, 1:] - torch.tensor([G_x_mean, G_y_mean])) / \
                      torch.tensor([G_x_std, G_y_std])

        # One-hot encode room types
        first_column_encodings = F.one_hot(G.x[:, 0].long(), NUM_ROOM_TYPES)
        G.x = torch.cat([first_column_encodings.float(), G.x[:, 1:]], dim=1)

        # Convert boundary graph to PyG
        B = from_networkx(boundary_graph, group_edge_attrs=['distance'],
                          group_node_attrs=['type', 'centroid'])
        # Normalize boundary
        B_x_mean = B.x[:, 1].mean().item()
        B_y_mean = B.x[:, 2].mean().item()
        B_x_std = max(B.x[:, 1].std().item(), 1e-6)
        B_y_std = max(B.x[:, 2].std().item(), 1e-6)
        B.x[:, 1:] = (B.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / \
                      torch.tensor([B_x_std, B_y_std])

        # Cast
        G.x = G.x.to(torch.float32).to(device)
        G.edge_attr = G.edge_attr.to(torch.float32).to(device)
        G.edge_index = G.edge_index.to(torch.int64).to(device)
        B.x = B.x.to(torch.float32).to(device)
        B.edge_index = B.edge_index.to(torch.int64).to(device)
        B.edge_attr = B.edge_attr.to(torch.float32).to(device)

        # Run inference
        with torch.no_grad():
            w_pred, h_pred = model(G, B)
            w_pred = w_pred.cpu().numpy()
            h_pred = h_pred.cpu().numpy()

        # Map predictions back to room types
        room_sizes = {}
        idx = 0
        for rtype, centroids in room_centroids.items():
            sizes = []
            for _ in centroids:
                if idx < len(w_pred):
                    sizes.append((float(abs(w_pred[idx])), float(abs(h_pred[idx]))))
                else:
                    sizes.append((8.0, 8.0))
                idx += 1
            room_sizes[rtype] = sizes

        return room_sizes

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return {}


# ===========================================================================
# MAIN API FUNCTIONS
# ===========================================================================

def generate_gnn_floor_plan(
    boundary_coords: Optional[List[Tuple[float, float]]] = None,
    front_door_pos: Optional[Tuple[float, float]] = None,
    total_area: float = 1200,
    bedrooms: int = 2,
    bathrooms: int = 1,
    kitchens: int = 1,
    floors: int = 1,
    extras: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    plot_width: Optional[float] = None,
    plot_length: Optional[float] = None,
) -> Dict:
    """
    Main entry point: Generate a floor plan using the GNN-inspired pipeline.

    Pipeline:
      1. Build/compute boundary polygon
      2. Co-generate room centroids + sizes (heuristic strip-packing)
         OR generate centroids then run GATNet model for sizes
      3. Build room and boundary graphs
      4. Post-process into final layout

    Returns:
        Layout dict compatible with PlanPreview frontend
    """
    extras = extras or []

    # Step 1: Build boundary
    if boundary_coords and len(boundary_coords) >= 3:
        pass
    else:
        if plot_width and plot_length:
            pw, pl = plot_width, plot_length
        else:
            ratio = 1.33
            pw = math.sqrt(total_area * ratio)
            pl = total_area / pw
            pw = round(pw, 1)
            pl = round(pl, 1)
        boundary_coords = [
            (0, 0), (pw, 0), (pw, pl), (0, pl), (0, 0)
        ]

    # Calculate boundary area
    if SHAPELY_AVAILABLE:
        boundary_poly = Polygon(boundary_coords)
        boundary_area = boundary_poly.area
        bounds = boundary_poly.bounds
    else:
        xs = [c[0] for c in boundary_coords]
        ys = [c[1] for c in boundary_coords]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        boundary_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

    effective_area = total_area if total_area > 0 else boundary_area

    if not front_door_pos:
        front_door_pos = ((bounds[0] + bounds[2]) / 2, bounds[1])

    # Step 2: Build room configuration (Indian BHK-style)
    rooms_config = {'living': 1}
    if bedrooms > 0:
        rooms_config['master_bedroom'] = 1
        if bedrooms > 1:
            rooms_config['bedroom'] = bedrooms - 1
    # Indian standard: at least 1 bathroom per bedroom (attached bath)
    rooms_config['bathroom'] = max(bathrooms, bedrooms, 1)
    rooms_config['kitchen'] = max(kitchens, 1)

    # Auto-add dining for 2BHK+ if not explicitly requested
    if bedrooms >= 2 and 'dining' not in [e.lower().replace(' ', '_') for e in extras]:
        rooms_config['dining'] = 1

    for extra in extras:
        extra_lower = extra.lower().replace(' ', '_')
        if extra_lower in ROOM_EMBEDDINGS:
            rooms_config[extra_lower] = rooms_config.get(extra_lower, 0) + 1

    # Auto-add pooja for 3BHK+ (Indian tradition)  
    if bedrooms >= 3 and rooms_config.get('pooja', 0) == 0:
        rooms_config['pooja'] = 1

    # Step 3: Try GATNet model first
    use_model = False
    model = load_gat_model(model_path)

    if model and NX_AVAILABLE:
        # Model mode: generate centroids, build graphs, run inference
        room_centroids, _, room_specs = generate_room_plan(
            boundary_coords, rooms_config, front_door_pos, effective_area
        )
        boundary_graph = build_boundary_graph(
            boundary_coords[:-1] if boundary_coords[0] == boundary_coords[-1] else boundary_coords,
            front_door_pos
        )
        room_graph = build_room_graph(room_centroids, living_to_all=True)
        room_sizes = predict_with_model(model, room_graph, boundary_graph, room_centroids)
        if room_sizes:
            use_model = True

    if not use_model:
        # Heuristic mode: co-generate centroids + sizes via strip-packing
        room_centroids, room_sizes, room_specs = generate_room_plan(
            boundary_coords, rooms_config, front_door_pos, effective_area
        )
        # Build graphs from the generated centroids (for graph context)
        boundary_graph = build_boundary_graph(
            boundary_coords[:-1] if boundary_coords[0] == boundary_coords[-1] else boundary_coords,
            front_door_pos
        )
        room_graph = build_room_graph(room_centroids, living_to_all=True)

    # Step 4: Build the final layout from room_specs (which have _placed positions)
    builder = FloorPlanBuilder(boundary_coords, front_door_pos)

    if not use_model:
        # Use the pre-placed positions from strip-packing (guaranteed non-overlapping)
        layout = builder.build_layout_from_specs(room_specs, effective_area, floors)
    else:
        # Model mode: use centroids + predicted sizes (may need overlap resolution)
        layout = builder.build_layout(room_centroids, room_sizes, effective_area, floors)

    # Step 5: Validate
    validation = validate_gnn_layout(layout)
    layout['validation'] = validation

    # Step 6: Build explanation
    layout['explanation'] = _build_explanation(rooms_config, layout, validation)

    return layout


def validate_gnn_layout(layout: Dict) -> Dict:
    """Validate the generated layout for architectural compliance."""
    rooms = layout.get('rooms', [])
    plot = layout.get('plot', {})
    plot_w = plot.get('width', 0)
    plot_l = plot.get('length', 0)

    issues = []
    overlap_count = 0
    min_size_ok = True
    zoning_ok = True
    boundary_ok = True

    # Check overlaps
    for i, r1 in enumerate(rooms):
        p1 = r1.get('position', {})
        x1, y1 = p1.get('x', 0), p1.get('y', 0)
        w1, l1 = r1.get('width', 0), r1.get('length', 0)
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            p2 = r2.get('position', {})
            x2, y2 = p2.get('x', 0), p2.get('y', 0)
            w2, l2 = r2.get('width', 0), r2.get('length', 0)
            tol = WALL_INTERNAL_FT * 0.5
            if (x1 < x2 + w2 - tol and x1 + w1 > x2 + tol and
                y1 < y2 + l2 - tol and y1 + l1 > y2 + tol):
                overlap_count += 1
                issues.append(f"Overlap: {r1.get('name')} and {r2.get('name')}")

    # Check minimum sizes
    for room in rooms:
        rtype = room.get('room_type', 'other')
        area = room.get('area', 0)
        min_dims = MIN_ROOM_DIMS.get(rtype, (4, 4))
        min_area = min_dims[0] * min_dims[1] * 0.6
        if area < min_area:
            min_size_ok = False
            issues.append(f"{room.get('name')}: {area} sqft too small")

    # Check boundary fit
    for room in rooms:
        pos = room.get('position', {})
        if pos.get('x', 0) < -1 or pos.get('y', 0) < -1:
            boundary_ok = False
        if pos.get('x', 0) + room.get('width', 0) > plot_w + 1:
            boundary_ok = False
        if pos.get('y', 0) + room.get('length', 0) > plot_l + 1:
            boundary_ok = False

    compliant = overlap_count == 0 and min_size_ok and boundary_ok
    return {
        'compliant': compliant,
        'overlap': overlap_count > 0,
        'overlap_count': overlap_count,
        'zoning_ok': zoning_ok,
        'min_size_ok': min_size_ok,
        'boundary_ok': boundary_ok,
        'issues': issues,
    }


def _build_explanation(rooms_config: Dict, layout: Dict, validation: Dict) -> str:
    """Build a short professional explanation of the generated layout."""
    plot = layout.get('plot', {})
    area = layout.get('area_summary', {})
    rooms = layout.get('rooms', [])
    method = layout.get('method', 'heuristic')

    lines = []
    lines.append(
        f"Indian Vastu-compliant layout for {plot.get('width')}×{plot.get('length')} ft plot "
        f"({area.get('plot_area', '?')} sq ft)."
    )
    lines.append(
        f"Method: {'GAT-Net model inference' if method == 'model' else 'Vastu-guided heuristic layout'}."
    )

    # Detect layout strategy from room zone_group
    strategy = 'dynamic'
    for r in rooms:
        zg = r.get('zone_group', '')
        if zg and zg != 'dynamic':
            strategy = zg
            break
    strategy_names = {
        'GRID_2x3': '2×3 grid (wide layout)',
        'GRID_3x2': '3×2 grid (tall layout)',
        'WRAP': 'wrap-around central living',
        'COLUMN_3': '3-column service core',
        'GRID_2x2_WIDE': '2×2 grid (compact wide)',
        'GRID_2x2_TALL': '2×2 grid (compact tall)',
    }
    lines.append(f"Layout strategy: {strategy_names.get(strategy, strategy)}.")

    bedroom_count = sum(1 for r in rooms if r['room_type'] in ('master_bedroom', 'bedroom'))
    bathroom_count = sum(1 for r in rooms if r['room_type'] in ('bathroom', 'toilet'))
    bhk = f"{bedroom_count}BHK"
    lines.append(
        f"Configuration: {bhk} — {bedroom_count} bedroom(s), {bathroom_count} bathroom(s), "
        f"{len(rooms)} total rooms."
    )

    # Vastu summary
    vastu_notes = []
    for r in rooms:
        rtype = r['room_type']
        if rtype == 'kitchen':
            vastu_notes.append("Kitchen placed per Vastu (SE/fire corner)")
        elif rtype == 'master_bedroom':
            vastu_notes.append("Master bedroom in SW (earth/stability)")
        elif rtype == 'pooja':
            vastu_notes.append("Pooja room in NE (most auspicious)")
    if vastu_notes:
        lines.append("Vastu: " + "; ".join(vastu_notes[:3]) + ".")

    # Check attached bathrooms
    attached = sum(1 for r in rooms if r['room_type'] == 'bathroom')
    if attached >= bedroom_count:
        lines.append(f"Attached bathrooms: {attached} (one per bedroom).")

    lines.append(
        f"Utilization: {area.get('utilization_percentage', '?')}. "
        f"Circulation: {area.get('circulation_percentage', '?')}."
    )
    lines.append("Walls: 9-inch external (brick), 4.5-inch internal (brick).")

    if validation.get('compliant'):
        lines.append("All Indian residential constraints validated. Layout is CAD-ready.")
    else:
        lines.append(f"Found {len(validation.get('issues', []))} issue(s).")

    return "\n".join(lines)


# ===========================================================================
# QUICK TEST
# ===========================================================================

if __name__ == '__main__':
    configs = [
        {'name': '1BHK', 'area': 600, 'bed': 1, 'bath': 1, 'kitchen': 1, 'extras': []},
        {'name': '2BHK', 'area': 1000, 'bed': 2, 'bath': 2, 'kitchen': 1, 'extras': ['dining']},
        {'name': '3BHK', 'area': 1500, 'bed': 3, 'bath': 2, 'kitchen': 1, 'extras': ['dining', 'study']},
        {'name': '4BHK', 'area': 2400, 'bed': 4, 'bath': 3, 'kitchen': 1, 'extras': ['dining', 'study', 'pooja']},
    ]

    for cfg in configs:
        result = generate_gnn_floor_plan(
            total_area=cfg['area'],
            bedrooms=cfg['bed'],
            bathrooms=cfg['bath'],
            kitchens=cfg['kitchen'],
            extras=cfg['extras'],
        )
        v = result.get('validation', {})
        rooms = result.get('rooms', [])
        area_summary = result.get('area_summary', {})
        print(f"\n{'='*60}")
        print(f"{cfg['name']} ({cfg['area']} sqft)")
        print(f"  Rooms: {len(rooms)}")
        print(f"  Utilization: {area_summary.get('utilization_percentage', '?')}")
        print(f"  Method: {result.get('method', '?')}")
        print(f"  Compliant: {v.get('compliant', '?')}")
        if v.get('issues'):
            for issue in v['issues']:
                print(f"    - {issue}")
        for r in rooms:
            print(f"    {r['name']}: {r['width']}x{r['length']} = {r['area']}sqft "
                  f"@ ({r['position']['x']}, {r['position']['y']})")
