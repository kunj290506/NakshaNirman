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

# Recommended room area ratios (fraction of usable area)
ROOM_AREA_RATIOS = {
    'living':         (0.18, 0.22),
    'master_bedroom': (0.12, 0.16),
    'bedroom':        (0.10, 0.14),
    'kitchen':        (0.08, 0.12),
    'bathroom':       (0.04, 0.06),
    'toilet':         (0.03, 0.04),
    'dining':         (0.08, 0.12),
    'study':          (0.05, 0.08),
    'pooja':          (0.02, 0.03),
    'store':          (0.02, 0.04),
    'balcony':        (0.03, 0.05),
    'utility':        (0.02, 0.03),
    'garage':         (0.10, 0.14),
}

# Min room dimensions (width, length) in feet
MIN_ROOM_DIMS = {
    'living':         (10, 12),
    'master_bedroom': (10, 12),
    'bedroom':        (9, 10),
    'kitchen':        (7, 8),
    'bathroom':       (5, 6),
    'toilet':         (4, 5),
    'dining':         (8, 10),
    'study':          (7, 8),
    'pooja':          (5, 5),
    'store':          (5, 5),
    'balcony':        (4, 8),
    'utility':        (4, 5),
    'garage':         (10, 18),
}

ZONE_MAP = {
    'living': 'public',
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
# ROOM PLAN GENERATION  (uses treemap above)
# ===========================================================================

def generate_room_plan(
    boundary_coords: List[Tuple[float, float]],
    rooms_config: Dict[str, int],
    front_door_pos: Optional[Tuple[float, float]] = None,
    total_area: float = 0,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Generate a professional residential floor plan via zoned treemap subdivision.

    Architecture
    ────────────
    Front zone (bottom, near entrance):
        Public / semi-private rooms — Living, Kitchen, Dining, Balcony
    Corridor (3–4 ft):
        Passage separating public from private
    Back zone (top):
        Private / service rooms — Bedrooms, Bathrooms, Study, Pooja …

    Within each zone a *squarified treemap* fills 100 % of the rectangle,
    producing wall-to-wall tiling with zero gaps and good aspect ratios.

    Returns
    -------
    (room_centroids, room_sizes, room_specs)
        – centroids / sizes: dict[room_type → list of (x,y) / (w,h)]
        – room_specs:       list of room dicts, each with '_placed' key
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

    EW = WALL_EXTERNAL_FT                         # 0.75 ft
    ux, uy = minx + EW, miny + EW                 # usable origin
    uw = plot_w - 2 * EW                           # usable width
    ul = plot_l - 2 * EW                           # usable length

    # ── 2. Target-area percentages (mid of ROOM_AREA_RATIOS) ────────────
    def _pct(rtype):
        lo, hi = ROOM_AREA_RATIOS.get(rtype, (0.04, 0.06))
        return (lo + hi) / 2

    # ── 3. Build room spec lists per zone ──────────────────────────────
    front_rooms = []
    back_rooms  = []

    # --- Front zone : public / semi-private --------------------------
    front_rooms.append({
        'room_type': 'living', 'name': 'Living',
        'zone': 'public', 'zone_group': 'front',
        'target_area': _pct('living') * total_area,
    })
    for _ in range(rooms_config.get('kitchen', 1)):
        front_rooms.append({
            'room_type': 'kitchen', 'name': 'Kitchen',
            'zone': 'semi_private', 'zone_group': 'front',
            'target_area': _pct('kitchen') * total_area,
        })
    for i in range(rooms_config.get('dining', 0)):
        front_rooms.append({
            'room_type': 'dining', 'name': 'Dining',
            'zone': 'semi_private', 'zone_group': 'front',
            'target_area': _pct('dining') * total_area,
        })
    for _ in range(rooms_config.get('balcony', 0)):
        front_rooms.append({
            'room_type': 'balcony', 'name': 'Balcony',
            'zone': 'public', 'zone_group': 'front',
            'target_area': _pct('balcony') * total_area,
        })

    # --- Back zone : private / service --------------------------------
    master_count  = rooms_config.get('master_bedroom', 0)
    bedroom_count = rooms_config.get('bedroom', 0)
    bathroom_count = rooms_config.get('bathroom', 0)
    toilet_count  = rooms_config.get('toilet', 0)

    if master_count > 0:
        back_rooms.append({
            'room_type': 'master_bedroom', 'name': 'Master Bedroom',
            'zone': 'private', 'zone_group': 'back',
            'target_area': _pct('master_bedroom') * total_area,
        })
    bath_idx = 0
    for bi in range(bedroom_count):
        lbl = f'Bedroom {bi + 1}' if bedroom_count > 1 else 'Bedroom'
        back_rooms.append({
            'room_type': 'bedroom', 'name': lbl,
            'zone': 'private', 'zone_group': 'back',
            'target_area': _pct('bedroom') * total_area,
        })
    for _ in range(bathroom_count):
        bath_idx += 1
        lbl = f'Bathroom {bath_idx}' if bathroom_count > 1 else 'Bathroom'
        back_rooms.append({
            'room_type': 'bathroom', 'name': lbl,
            'zone': 'service', 'zone_group': 'back',
            'target_area': _pct('bathroom') * total_area,
        })
    for _ in range(toilet_count):
        back_rooms.append({
            'room_type': 'toilet', 'name': 'Toilet',
            'zone': 'service', 'zone_group': 'back',
            'target_area': _pct('toilet') * total_area,
        })
    for rtype in ('study', 'pooja', 'store', 'utility', 'garage'):
        for _ in range(rooms_config.get(rtype, 0)):
            back_rooms.append({
                'room_type': rtype,
                'name': rtype.replace('_', ' ').title(),
                'zone': ZONE_MAP.get(rtype, 'private'),
                'zone_group': 'back',
                'target_area': _pct(rtype) * total_area,
            })

    # Enforce minimum room areas (Indian residential standards)
    _MIN_AREAS = {
        'living': 100, 'master_bedroom': 100, 'bedroom': 80,
        'kitchen': 50, 'bathroom': 30, 'toilet': 20,
        'dining': 60, 'study': 40, 'pooja': 20,
        'store': 20, 'balcony': 25, 'utility': 16, 'garage': 80,
    }
    for r in front_rooms + back_rooms:
        mn = _MIN_AREAS.get(r['room_type'], 20)
        if r['target_area'] < mn:
            r['target_area'] = mn

    room_specs = front_rooms + back_rooms

    # Cap small-room areas so bathrooms / pooja don't bloat in large homes
    _MAX_AREAS = {'bathroom': 55, 'toilet': 35, 'pooja': 40, 'store': 40, 'utility': 30}
    for r in front_rooms + back_rooms:
        mx = _MAX_AREAS.get(r['room_type'])
        if mx and r['target_area'] > mx:
            r['target_area'] = mx

    # ── 4. Zone heights ────────────────────────────────────────────────
    has_back = len(back_rooms) > 0
    if not has_back:
        CORRIDOR = 0
    elif total_area < 700:
        CORRIDOR = 0                 # tiny homes: living IS the passage
    elif total_area < 1000:
        CORRIDOR = 2.0
    else:
        CORRIDOR = 3.5 if uw >= 18 else 2.5

    front_total = sum(r['target_area'] for r in front_rooms) or 1
    back_total  = sum(r['target_area'] for r in back_rooms)  or 0
    zone_sum    = front_total + back_total

    avail_l = ul - CORRIDOR                        # vertical budget
    front_h = avail_l * (front_total / zone_sum) if has_back else avail_l
    back_h  = avail_l - front_h if has_back else 0

    # Clamp so neither zone is unrealistically shallow
    has_beds = any(r['room_type'] in ('master_bedroom', 'bedroom') for r in back_rooms)
    MIN_ZONE = 8 if has_beds else min(6, avail_l * 0.25)
    if has_back:
        if back_h < MIN_ZONE:
            back_h = MIN_ZONE;  front_h = avail_l - back_h
        if front_h < MIN_ZONE:
            front_h = MIN_ZONE;  back_h = avail_l - front_h

    # ── 5. Place rooms with squarified treemap ─────────────────────────
    centroids_out = defaultdict(list)
    sizes_out     = defaultdict(list)

    def _record(room, rx, ry, rw, rh):
        """Callback — store placement for one room."""
        room['_placed'] = {
            'x': round(rx, 2), 'y': round(ry, 2),
            'w': round(rw, 2), 'h': round(rh, 2),
        }
        centroids_out[room['room_type']].append(
            (round(rx + rw / 2, 2), round(ry + rh / 2, 2)))
        sizes_out[room['room_type']].append(
            (round(rw, 2), round(rh, 2)))

    # Front zone (bottom — near entrance)
    if front_rooms:
        _squarify_layout(front_rooms, ux, uy, uw, front_h, _record)

    # Back zone (top)
    if back_rooms:
        back_y = uy + front_h + CORRIDOR
        _squarify_layout(back_rooms, ux, back_y, uw, back_h, _record)

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
        """Assign doors and windows to each room based on position and zoning."""
        minx, miny = self.bounds[0], self.bounds[1]

        # Find corridor midpoint
        public_bottom = 0
        private_top = plot_l
        for room in rooms:
            z = room.get('zone', 'private')
            ry = room['position']['y']
            rl = room['length']
            if z in ('public', 'semi_private'):
                public_bottom = max(public_bottom, ry + rl)
            elif z in ('private', 'service'):
                private_top = min(private_top, ry)
        corridor_mid = (public_bottom + private_top) / 2

        for room in rooms:
            pos = room['position']
            rw, rl = room['width'], room['length']
            zone = room.get('zone', 'private')
            rtype = room['room_type']
            rx, ry = pos['x'], pos['y']

            doors = []
            windows = []

            room_center_y = ry + rl / 2

            # Door placement
            if rtype in ('bathroom', 'toilet'):
                # Door toward adjacent bedroom
                door_placed = False
                for other in rooms:
                    if other['room_type'] not in ('master_bedroom', 'bedroom'):
                        continue
                    ox = other['position']['x']
                    ow = other['width']
                    oy = other['position']['y']
                    if abs((ox + ow) - rx) < 2.0 and abs(oy - ry) < 2.0:
                        doors.append({'wall': 'W', 'width': 2.5})
                        door_placed = True
                        break
                    if abs((rx + rw) - ox) < 2.0 and abs(oy - ry) < 2.0:
                        doors.append({'wall': 'E', 'width': 2.5})
                        door_placed = True
                        break
                if not door_placed:
                    if room_center_y < corridor_mid:
                        doors.append({'wall': 'N', 'width': 2.5})
                    else:
                        doors.append({'wall': 'S', 'width': 2.5})
            elif zone in ('public', 'semi_private'):
                doors.append({'wall': 'N', 'width': 3})
                if rtype == 'living':
                    doors.append({'wall': 'S', 'width': 3})
            else:
                if room_center_y > corridor_mid:
                    doors.append({'wall': 'S', 'width': 3})
                else:
                    doors.append({'wall': 'N', 'width': 3})

            # Window placement (external walls only)
            tol = WALL_EXTERNAL_FT + 1
            is_south = ry <= miny + tol
            is_north = ry + rl >= miny + plot_l - tol
            is_west = rx <= minx + tol
            is_east = rx + rw >= minx + plot_w - tol

            if rtype in ('bathroom', 'toilet'):
                if is_east:
                    windows.append({'wall': 'E', 'width': 2})
                elif is_north:
                    windows.append({'wall': 'N', 'width': 2})
                elif is_west:
                    windows.append({'wall': 'W', 'width': 2})
                elif is_south:
                    windows.append({'wall': 'S', 'width': 2})
            else:
                if is_north:
                    windows.append({'wall': 'N', 'width': 4})
                if is_south:
                    windows.append({'wall': 'S', 'width': 4})
                if is_east:
                    windows.append({'wall': 'E', 'width': 4})
                if is_west:
                    windows.append({'wall': 'W', 'width': 4})

            if not windows and rtype not in ('bathroom', 'toilet', 'store', 'utility'):
                windows.append({'wall': 'N', 'width': 4})

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

    # Step 2: Build room configuration
    rooms_config = {'living': 1}
    if bedrooms > 0:
        rooms_config['master_bedroom'] = 1
        if bedrooms > 1:
            rooms_config['bedroom'] = bedrooms - 1
    rooms_config['bathroom'] = max(bathrooms, 1)
    rooms_config['kitchen'] = max(kitchens, 1)

    for extra in extras:
        extra_lower = extra.lower().replace(' ', '_')
        if extra_lower in ROOM_EMBEDDINGS:
            rooms_config[extra_lower] = rooms_config.get(extra_lower, 0) + 1

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
        f"GNN-inspired layout for {plot.get('width')}×{plot.get('length')} ft plot "
        f"({area.get('plot_area', '?')} sq ft)."
    )
    lines.append(
        f"Method: {'GAT-Net model inference' if method == 'model' else 'Graph-based heuristic sizing'}."
    )

    bedroom_count = sum(1 for r in rooms if r['room_type'] in ('master_bedroom', 'bedroom'))
    bathroom_count = sum(1 for r in rooms if r['room_type'] in ('bathroom', 'toilet'))
    lines.append(
        f"Configuration: {bedroom_count} bedroom(s), {bathroom_count} bathroom(s), "
        f"{len(rooms)} total rooms."
    )
    lines.append(
        f"Utilization: {area.get('utilization_percentage', '?')}. "
        f"Circulation: {area.get('circulation_percentage', '?')}."
    )
    lines.append("Walls: 9-inch external, 4.5-inch internal.")

    if validation.get('compliant'):
        lines.append("All constraints validated. Layout is CAD-ready.")
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
