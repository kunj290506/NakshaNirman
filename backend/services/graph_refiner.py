"""Graph-based post-refinement for generated floor plans.

This module applies lightweight GNN-inspired graph relaxation over room centroids
without replacing the existing deterministic generator.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import networkx as nx


def _r(v: float) -> float:
    return round(float(v), 2)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bounds(room: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x1 = float(room.get("x", 0.0))
    y1 = float(room.get("y", 0.0))
    x2 = x1 + float(room.get("width", 0.0))
    y2 = y1 + float(room.get("height", 0.0))
    return x1, y1, x2, y2


def _center(room: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = _bounds(room)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _adjacent(a: Dict[str, Any], b: Dict[str, Any], tol: float = 0.2) -> bool:
    ax1, ay1, ax2, ay2 = _bounds(a)
    bx1, by1, bx2, by2 = _bounds(b)
    horizontal_touch = abs(ax2 - bx1) <= tol or abs(bx2 - ax1) <= tol
    vertical_overlap = min(ay2, by2) - max(ay1, by1) > 0.35
    vertical_touch = abs(ay2 - by1) <= tol or abs(by2 - ay1) <= tol
    horizontal_overlap = min(ax2, bx2) - max(ax1, bx1) > 0.35
    return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap)


def _path_exists(rooms: List[Dict[str, Any]], src: Dict[str, Any] | None, dst: Dict[str, Any] | None) -> bool:
    if not src or not dst:
        return False
    by_id = {r["id"]: r for r in rooms if r.get("id")}
    start = src.get("id")
    end = dst.get("id")
    if start not in by_id or end not in by_id:
        return False

    q = [start]
    seen = {start}
    while q:
        cur_id = q.pop(0)
        if cur_id == end:
            return True
        cur = by_id[cur_id]
        for rid, room in by_id.items():
            if rid in seen:
                continue
            if _adjacent(cur, room):
                seen.add(rid)
                q.append(rid)
    return False


def _compute_signature(rooms: List[Dict[str, Any]], bhk: int) -> str:
    tokens = sorted(
        f"{r.get('type')}@{_r(r.get('x',0))},{_r(r.get('y',0))}:{_r(r.get('width',0))}x{_r(r.get('height',0))}"
        for r in rooms
    )
    digest = hashlib.md5(";".join(tokens).encode("utf-8")).hexdigest()[:10]
    return f"{bhk}bhk-{digest}"


def _connectivity_checks(rooms: List[Dict[str, Any]]) -> Dict[str, bool]:
    living = next((r for r in rooms if r.get("type") == "living"), None)
    kitchen = next((r for r in rooms if r.get("type") == "kitchen"), None)
    dining = next((r for r in rooms if r.get("type") == "dining"), None)
    master = next((r for r in rooms if r.get("type") == "master_bedroom"), None)
    bedrooms = [r for r in rooms if r.get("type") == "bedroom"]
    wet_rooms = [r for r in rooms if r.get("type") in {"bathroom", "toilet"}]

    return {
        "living_to_kitchen": _path_exists(rooms, living, kitchen),
        "kitchen_adjacent_dining": bool(kitchen and dining and _adjacent(kitchen, dining)),
        "living_to_master": _path_exists(rooms, living, master),
        "master_attached_wet_room": bool(master and any(_adjacent(master, w) for w in wet_rooms)),
        "living_to_any_bedroom": True if not bedrooms else any(_path_exists(rooms, living, b) for b in bedrooms),
    }


def _build_graph(rooms: List[Dict[str, Any]]) -> nx.Graph:
    g = nx.Graph()
    for r in rooms:
        rid = r.get("id")
        if not rid:
            continue
        g.add_node(rid, room_type=r.get("type", "room"))

    # Reference-inspired rule: connect living to all major rooms.
    living_nodes = [r for r in rooms if r.get("type") == "living"]
    if living_nodes:
        living_id = living_nodes[0].get("id")
        for r in rooms:
            if r.get("id") != living_id and r.get("type") not in {"toilet", "bathroom"}:
                g.add_edge(living_id, r.get("id"), weight=2.0)

    preferred_pairs = [
        ("kitchen", "dining", 2.5),
        ("living", "kitchen", 2.0),
        ("master_bedroom", "bathroom", 2.0),
        ("living", "master_bedroom", 1.8),
    ]
    for ta, tb, w in preferred_pairs:
        a_nodes = [r for r in rooms if r.get("type") == ta]
        b_nodes = [r for r in rooms if r.get("type") == tb]
        for a in a_nodes:
            for b in b_nodes:
                if a.get("id") and b.get("id") and a.get("id") != b.get("id"):
                    g.add_edge(a["id"], b["id"], weight=w)

    # Keep existing touch relations as weak constraints.
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if _adjacent(rooms[i], rooms[j]):
                g.add_edge(rooms[i]["id"], rooms[j]["id"], weight=1.1)

    return g


def _resolve_overlaps(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> None:
    for _ in range(16):
        moved = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]
                ax1, ay1, ax2, ay2 = _bounds(a)
                bx1, by1, bx2, by2 = _bounds(b)
                ox = min(ax2, bx2) - max(ax1, bx1)
                oy = min(ay2, by2) - max(ay1, by1)
                if ox <= 0 or oy <= 0:
                    continue

                moved = True
                acx, acy = _center(a)
                bcx, bcy = _center(b)

                if ox < oy:
                    shift = ox / 2.0 + 0.08
                    if acx <= bcx:
                        a["x"] = _r(a["x"] - shift)
                        b["x"] = _r(b["x"] + shift)
                    else:
                        a["x"] = _r(a["x"] + shift)
                        b["x"] = _r(b["x"] - shift)
                else:
                    shift = oy / 2.0 + 0.08
                    if acy <= bcy:
                        a["y"] = _r(a["y"] - shift)
                        b["y"] = _r(b["y"] + shift)
                    else:
                        a["y"] = _r(a["y"] + shift)
                        b["y"] = _r(b["y"] - shift)

                for room in (a, b):
                    room["x"] = _r(_clamp(room["x"], 0.0, max(0.0, usable_w - room["width"])))
                    room["y"] = _r(_clamp(room["y"], 0.0, max(0.0, usable_l - room["height"])))

        if not moved:
            break


def _max_overlap(rooms: List[Dict[str, Any]]) -> float:
    max_ov = 0.0
    for i in range(len(rooms)):
        ax1, ay1, ax2, ay2 = _bounds(rooms[i])
        for j in range(i + 1, len(rooms)):
            bx1, by1, bx2, by2 = _bounds(rooms[j])
            dx = min(ax2, bx2) - max(ax1, bx1)
            dy = min(ay2, by2) - max(ay1, by1)
            if dx > 0 and dy > 0:
                max_ov = max(max_ov, dx * dy)
    return max_ov


def _overlaps_any(room: Dict[str, Any], placed: List[Dict[str, Any]], tol: float = 0.05) -> bool:
    rx1, ry1, rx2, ry2 = _bounds(room)
    for other in placed:
        ox1, oy1, ox2, oy2 = _bounds(other)
        dx = min(rx2, ox2) - max(rx1, ox1)
        dy = min(ry2, oy2) - max(ry1, oy1)
        if dx > tol and dy > tol:
            return True
    return False


def _non_overlapping_repack(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> None:
    if not rooms:
        return

    protected_types = {"living", "kitchen", "dining", "master_bedroom"}
    protected = [r for r in rooms if r.get("type") in protected_types]
    floating = [r for r in rooms if r.get("type") not in protected_types]

    placed: List[Dict[str, Any]] = []
    for r in protected:
        r["x"] = _r(_clamp(r["x"], 0.0, max(0.0, usable_w - r["width"])))
        r["y"] = _r(_clamp(r["y"], 0.0, max(0.0, usable_l - r["height"])))
        placed.append(r)

    step = 0.5
    # Place smaller service rooms first so large wet/open zones can adapt around them.
    for r in sorted(floating, key=lambda x: float(x.get("area", 0.0))):
        ox, oy = float(r.get("x", 0.0)), float(r.get("y", 0.0))
        w, h = float(r.get("width", 0.0)), float(r.get("height", 0.0))
        best = None
        best_dist = float("inf")

        max_r = int(max(usable_w, usable_l) / max(step, 0.1)) + 2
        for radius in range(max_r):
            for angle_deg in range(0, 360, 20):
                angle = math.radians(angle_deg)
                tx = _clamp(ox + radius * step * math.cos(angle), 0.0, max(0.0, usable_w - w))
                ty = _clamp(oy + radius * step * math.sin(angle), 0.0, max(0.0, usable_l - h))
                cand = {"x": _r(tx), "y": _r(ty), "width": w, "height": h}
                if _overlaps_any(cand, placed):
                    continue
                dist = (tx - ox) ** 2 + (ty - oy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (tx, ty)
            if best is not None:
                break

        if best is None:
            found = False
            y = 0.0
            while y <= max(0.0, usable_l - h):
                x = 0.0
                while x <= max(0.0, usable_w - w):
                    cand = {"x": _r(x), "y": _r(y), "width": w, "height": h}
                    if not _overlaps_any(cand, placed):
                        best = (x, y)
                        found = True
                        break
                    x += step
                if found:
                    break
                y += step

        if best is not None:
            r["x"] = _r(best[0])
            r["y"] = _r(best[1])

        placed.append(r)


def _cleanup_soft_overlaps(rooms: List[Dict[str, Any]], usable_w: float, usable_l: float) -> None:
    """Try to relocate/remove soft utility rooms that still cause large overlaps."""
    soft_types = {"open_area", "utility", "store", "balcony"}

    hard_rooms = [r for r in rooms if r.get("type") not in soft_types]
    soft_rooms = [r for r in rooms if r.get("type") in soft_types]

    for sr in soft_rooms:
        # If already clean, keep it.
        if not _overlaps_any(sr, [r for r in hard_rooms if r.get("id") != sr.get("id")], tol=0.35):
            continue

        # Try a full sweep relocation.
        w, h = float(sr.get("width", 0.0)), float(sr.get("height", 0.0))
        found = None
        step = 0.5
        y = 0.0
        while y <= max(0.0, usable_l - h):
            x = 0.0
            while x <= max(0.0, usable_w - w):
                cand = {"x": _r(x), "y": _r(y), "width": w, "height": h}
                if not _overlaps_any(cand, hard_rooms, tol=0.35):
                    found = (x, y)
                    break
                x += step
            if found is not None:
                break
            y += step

        if found is not None:
            sr["x"] = _r(found[0])
            sr["y"] = _r(found[1])
        else:
            # As last resort drop only synthetic corner fillers.
            if str(sr.get("id", "")).startswith("open_"):
                sr["_drop"] = True

    rooms[:] = [r for r in rooms if not r.get("_drop")]


def _nudge_pair(anchor: Dict[str, Any] | None, target: Dict[str, Any] | None, usable_w: float, usable_l: float) -> None:
    if not anchor or not target:
        return
    if _adjacent(anchor, target):
        return

    ax, ay = _center(anchor)
    tx, ty = _center(target)
    dx = ax - tx
    dy = ay - ty

    target["x"] = _r(target["x"] + dx * 0.35)
    target["y"] = _r(target["y"] + dy * 0.35)
    target["x"] = _r(_clamp(target["x"], 0.0, max(0.0, usable_w - target["width"])))
    target["y"] = _r(_clamp(target["y"], 0.0, max(0.0, usable_l - target["height"])))


def _place_adjacent(anchor: Dict[str, Any] | None, target: Dict[str, Any] | None, usable_w: float, usable_l: float) -> None:
    if not anchor or not target:
        return

    options = [
        (anchor["x"] + anchor["width"], anchor["y"]),
        (anchor["x"] - target["width"], anchor["y"]),
        (anchor["x"], anchor["y"] + anchor["height"]),
        (anchor["x"], anchor["y"] - target["height"]),
    ]
    for tx, ty in options:
        target["x"] = _r(_clamp(tx, 0.0, max(0.0, usable_w - target["width"])))
        target["y"] = _r(_clamp(ty, 0.0, max(0.0, usable_l - target["height"])))
        if _adjacent(anchor, target, tol=0.35):
            return


def refine_layout_with_graph(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Return a graph-relaxed version of a hub layout."""
    if not layout or not layout.get("rooms"):
        return layout

    out = deepcopy(layout)
    baseline_checks = out.get("connectivity_checks") if isinstance(out.get("connectivity_checks"), dict) else None
    baseline_overlap = _max_overlap(out.get("rooms", []))
    rooms = out.get("rooms", [])
    plot = out.get("plot", {})
    usable_w = float(plot.get("usable_width") or plot.get("width") or 0.0)
    usable_l = float(plot.get("usable_length") or plot.get("length") or 0.0)
    if usable_w <= 0 or usable_l <= 0:
        return out

    g = _build_graph(rooms)
    if g.number_of_nodes() < 2:
        return out

    initial = {r["id"]: _center(r) for r in rooms if r.get("id") in g.nodes}
    pos = nx.spring_layout(g, pos=initial, seed=42, weight="weight", iterations=60)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)

    # Move toward graph targets while keeping dimensions fixed.
    for room in rooms:
        rid = room.get("id")
        if rid not in pos:
            continue

        px, py = pos[rid]
        tx = ((px - min_x) / dx) * usable_w
        ty = ((py - min_y) / dy) * usable_l
        tx = _clamp(tx - room["width"] / 2.0, 0.0, max(0.0, usable_w - room["width"]))
        ty = _clamp(ty - room["height"] / 2.0, 0.0, max(0.0, usable_l - room["height"]))

        room["x"] = _r(room["x"] * 0.55 + tx * 0.45)
        room["y"] = _r(room["y"] * 0.55 + ty * 0.45)

    # Reinforce key relations inspired by living-to-all graph logic.
    living = next((r for r in rooms if r.get("type") == "living"), None)
    kitchen = next((r for r in rooms if r.get("type") == "kitchen"), None)
    dining = next((r for r in rooms if r.get("type") == "dining"), None)
    master = next((r for r in rooms if r.get("type") == "master_bedroom"), None)
    bath = next((r for r in rooms if r.get("type") in {"bathroom", "toilet"}), None)

    _nudge_pair(living, kitchen, usable_w, usable_l)
    _nudge_pair(kitchen, dining, usable_w, usable_l)
    _nudge_pair(master, bath, usable_w, usable_l)

    _resolve_overlaps(rooms, usable_w, usable_l)

    # Aggressive de-overlap pass for non-critical rooms.
    _non_overlapping_repack(rooms, usable_w, usable_l)
    _resolve_overlaps(rooms, usable_w, usable_l)
    _cleanup_soft_overlaps(rooms, usable_w, usable_l)
    _resolve_overlaps(rooms, usable_w, usable_l)

    # Re-enforce key functional adjacencies after repack.
    living = next((r for r in rooms if r.get("type") == "living"), None)
    kitchen = next((r for r in rooms if r.get("type") == "kitchen"), None)
    dining = next((r for r in rooms if r.get("type") == "dining"), None)
    master = next((r for r in rooms if r.get("type") == "master_bedroom"), None)
    bath = next((r for r in rooms if r.get("type") in {"bathroom", "toilet"}), None)

    if living and kitchen and not _path_exists(rooms, living, kitchen):
        _place_adjacent(living, kitchen, usable_w, usable_l)
    if kitchen and dining and not _adjacent(kitchen, dining):
        _place_adjacent(kitchen, dining, usable_w, usable_l)
    if master and bath and not _adjacent(master, bath):
        _place_adjacent(master, bath, usable_w, usable_l)

    _resolve_overlaps(rooms, usable_w, usable_l)

    # Hard constraints to avoid regressions after graph relaxation.
    if living and kitchen and not _path_exists(rooms, living, kitchen):
        _place_adjacent(living, kitchen, usable_w, usable_l)
    if kitchen and dining and not _adjacent(kitchen, dining):
        _place_adjacent(kitchen, dining, usable_w, usable_l)
    if master and bath and not _adjacent(master, bath):
        _place_adjacent(master, bath, usable_w, usable_l)

    _resolve_overlaps(rooms, usable_w, usable_l)

    for room in rooms:
        room["x"] = _r(_clamp(room["x"], 0.0, max(0.0, usable_w - room["width"])))
        room["y"] = _r(_clamp(room["y"], 0.0, max(0.0, usable_l - room["height"])))
        room["area"] = _r(room["width"] * room["height"])

    checks = _connectivity_checks(rooms)
    new_overlap = _max_overlap(rooms)

    if baseline_checks:
        new_ok = sum(1 for v in checks.values() if v)
        old_ok = sum(1 for v in baseline_checks.values() if v)
        if new_ok < old_ok:
            return layout
    # Never accept a refinement that makes overlaps worse.
    if new_overlap > baseline_overlap + 0.05:
        return layout

    out["connectivity_checks"] = checks

    bhk = int(out.get("bhk") or 2)
    signature = _compute_signature(rooms, bhk)
    out["layout_signature"] = signature

    if isinstance(out.get("design_score"), dict):
        out["design_score"]["connectivity_checks"] = checks
        out["design_score"]["layout_signature"] = signature

    notes = out.get("architect_notes") or []
    notes.append("Graph refinement applied (living-to-all centroid relaxation).")
    out["architect_notes"] = notes[:8]

    return out
