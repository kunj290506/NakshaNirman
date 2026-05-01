"""
NakshaNirman Validators — Check floor plan geometry against the 3 Laws.

Law 1: Zero overlap between any pair of rooms
Law 2: All rooms within usable bounds
Law 3: All rooms meet minimum size requirements
"""
from __future__ import annotations

# Minimum room sizes (width, height) in feet
MIN_SIZES: dict[str, tuple[float, float]] = {
    "living": (11.0, 11.0),
    "dining": (8.0, 8.0),
    "kitchen": (7.0, 8.0),
    "master_bedroom": (10.0, 10.0),
    "bedroom": (9.0, 9.0),
    "master_bath": (4.5, 6.0),
    "bathroom": (4.0, 5.0),
    "corridor": (3.5, 3.0),
    "pooja": (4.0, 4.0),
    "study": (6.0, 7.0),
    "store": (4.0, 4.0),
    "balcony": (3.5, 6.0),
    "garage": (9.0, 15.0),
    "utility": (4.0, 5.0),
    "foyer": (4.0, 4.0),
    "staircase": (4.0, 8.0),
}


def _rooms_overlap(a: dict, b: dict) -> bool:
    """Check if two rooms overlap (share interior area)."""
    ax, ay = float(a.get("x", 0)), float(a.get("y", 0))
    aw, ah = float(a.get("width", 0)), float(a.get("height", 0))
    bx, by = float(b.get("x", 0)), float(b.get("y", 0))
    bw, bh = float(b.get("width", 0)), float(b.get("height", 0))

    # Separating axis test — if any of these are true, no overlap
    if ax + aw <= bx + 0.01:  # A is left of B (0.01 tolerance)
        return False
    if bx + bw <= ax + 0.01:  # B is left of A
        return False
    if ay + ah <= by + 0.01:  # A is below B
        return False
    if by + bh <= ay + 0.01:  # B is below A
        return False

    return True


def validate_plan(plan: dict, plot_width: float, plot_length: float) -> dict:
    """
    Validate a floor plan against all 3 laws.

    Returns:
        {
            "valid": bool,
            "law1_ok": bool,  # no overlaps
            "law2_ok": bool,  # boundary containment
            "law3_ok": bool,  # minimum sizes
            "issues": [str, ...],
            "overlap_pairs": [(id_a, id_b), ...],
            "boundary_violations": [id, ...],
            "size_violations": [{"id": id, "type": type, "issue": str}, ...],
        }
    """
    uw = plot_width - 7.0
    ul = plot_length - 11.5
    rooms = plan.get("rooms", [])
    if not isinstance(rooms, list):
        rooms = []

    issues: list[str] = []
    overlap_pairs: list[tuple[str, str]] = []
    boundary_violations: list[str] = []
    size_violations: list[dict] = []

    # ── Law 1: Zero overlap ──────────────────────────────────────────
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            if _rooms_overlap(a, b):
                aid = a.get("id", f"room_{i}")
                bid = b.get("id", f"room_{j}")
                overlap_pairs.append((aid, bid))
                issues.append(
                    f"OVERLAP: {aid} ({a.get('type', '?')}) overlaps with "
                    f"{bid} ({b.get('type', '?')})"
                )

    # ── Law 2: Boundary containment ──────────────────────────────────
    for room in rooms:
        rid = room.get("id", "unknown")
        x = float(room.get("x", 0))
        y = float(room.get("y", 0))
        w = float(room.get("width", 0))
        h = float(room.get("height", 0))

        violations = []
        if x < -0.1:
            violations.append(f"x={x:.1f} < 0")
        if y < -0.1:
            violations.append(f"y={y:.1f} < 0")
        if x + w > uw + 0.5:
            violations.append(f"x+w={x + w:.1f} > UW={uw:.1f}")
        if y + h > ul + 0.5:
            violations.append(f"y+h={y + h:.1f} > UL={ul:.1f}")

        if violations:
            boundary_violations.append(rid)
            issues.append(
                f"BOUNDARY: {rid} ({room.get('type', '?')}) — {', '.join(violations)}"
            )

    # ── Law 3: Minimum sizes ─────────────────────────────────────────
    for room in rooms:
        rid = room.get("id", "unknown")
        rtype = str(room.get("type", "room"))
        w = float(room.get("width", 0))
        h = float(room.get("height", 0))

        min_w, min_h = MIN_SIZES.get(rtype, (3.0, 3.0))

        # Check both orientations (w×h or h×w)
        ok_normal = w >= min_w - 0.1 and h >= min_h - 0.1
        ok_rotated = w >= min_h - 0.1 and h >= min_w - 0.1

        if not ok_normal and not ok_rotated:
            size_violations.append({
                "id": rid,
                "type": rtype,
                "issue": f"{w:.1f}x{h:.1f} < minimum {min_w}x{min_h}",
            })
            issues.append(
                f"SIZE: {rid} ({rtype}) is {w:.1f}x{h:.1f}, "
                f"minimum is {min_w}x{min_h}"
            )

    law1_ok = len(overlap_pairs) == 0
    law2_ok = len(boundary_violations) == 0
    law3_ok = len(size_violations) == 0

    return {
        "valid": law1_ok and law2_ok and law3_ok,
        "law1_ok": law1_ok,
        "law2_ok": law2_ok,
        "law3_ok": law3_ok,
        "issues": issues,
        "overlap_pairs": overlap_pairs,
        "boundary_violations": boundary_violations,
        "size_violations": size_violations,
    }


def fix_overlaps(plan: dict, plot_width: float, plot_length: float) -> dict:
    """
    Attempt to fix overlapping rooms by nudging them apart.
    This is a best-effort fix — if it can't resolve, returns the plan as-is.
    """
    rooms = plan.get("rooms", [])
    if not isinstance(rooms, list) or len(rooms) < 2:
        return plan

    uw = plot_width - 7.0
    ul = plot_length - 11.5

    # Up to 10 passes of nudging
    for _pass in range(10):
        found_overlap = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a, b = rooms[i], rooms[j]
                if not _rooms_overlap(a, b):
                    continue

                found_overlap = True

                # Calculate overlap extents
                ax, ay = float(a["x"]), float(a["y"])
                aw, ah = float(a["width"]), float(a["height"])
                bx, by = float(b["x"]), float(b["y"])
                bw, bh = float(b["width"]), float(b["height"])

                # Find smallest nudge direction
                push_right = (ax + aw) - bx
                push_left = (bx + bw) - ax
                push_up = (ay + ah) - by
                push_down = (by + bh) - ay

                min_push = min(push_right, push_left, push_up, push_down)

                if min_push == push_right and bx + push_right + bw <= uw + 0.5:
                    b["x"] = round(bx + push_right + 0.1, 1)
                elif min_push == push_left and ax + push_left + aw <= uw + 0.5:
                    a["x"] = round(ax + push_left + 0.1, 1)
                elif min_push == push_up and by + push_up + bh <= ul + 0.5:
                    b["y"] = round(by + push_up + 0.1, 1)
                elif min_push == push_down and ay + push_down + ah <= ul + 0.5:
                    a["y"] = round(ay + push_down + 0.1, 1)
                else:
                    # Can't fix — nudge b to the right as last resort
                    b["x"] = round(ax + aw + 0.1, 1)

                # Update polygons
                for room in (a, b):
                    x, y = float(room["x"]), float(room["y"])
                    w, h = float(room["width"]), float(room["height"])
                    room["polygon"] = [
                        {"x": x, "y": y}, {"x": x + w, "y": y},
                        {"x": x + w, "y": y + h}, {"x": x, "y": y + h},
                    ]
                    room["area"] = round(w * h, 1)

        if not found_overlap:
            break

    plan["rooms"] = rooms
    return plan
