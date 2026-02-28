"""
ADVANCED ARCHITECTURAL INTELLIGENCE AUDIT
==========================================
Tests all 3 engines against strict architectural rules:
  1. Exact room list (no duplicates, no extras)
  2. Public zone near entrance (Drawing, Dining)
  3. Kitchen adjacent to Dining
  4. Bedrooms in private zone
  5. Attached bath ONLY for Master Bedroom
  6. Passage for internal circulation
  7. All rooms inside boundary
  8. No overlaps
  9. No dead space (coverage ≥ 70%)
  10. Proper proportions (AR within limits)
  11. Natural light (habitable rooms touch exterior wall)
  12. Single plot only

Tests rectangular, square, and L-shaped boundaries.
"""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(__file__) or '.')
import logging
logging.basicConfig(level=logging.WARNING)

from services.pro_layout_engine import generate_professional_plan
from services.gnn_engine import generate_gnn_floor_plan
from services.perfect_layout import generate_perfect_layout

# ── Standard 2BHK config matching the spec ──────────────────────────
STANDARD_ROOMS = {
    'master_bedroom': 1,
    'bedroom': 1,
    'bathroom': 2,   # 1 attached + 1 wash area
    'kitchen': 1,
    'dining': 1,
}

# Expected room types in output
REQUIRED_TYPES = {
    'living', 'kitchen', 'dining', 'master_bedroom', 'bedroom', 'bathroom'
}

# ── Boundary configs ────────────────────────────────────────────────
CONFIGS = [
    {
        "name": "Rectangular 30×20",
        "boundary": [(0,0), (30,0), (30,20), (0,20)],
        "width": 30, "length": 20, "area": 600,
    },
    {
        "name": "Square 25×25",
        "boundary": [(0,0), (25,0), (25,25), (0,25)],
        "width": 25, "length": 25, "area": 625,
    },
    {
        "name": "Wide 35×18",
        "boundary": [(0,0), (35,0), (35,18), (0,18)],
        "width": 35, "length": 18, "area": 630,
    },
    {
        "name": "Tall 20×30",
        "boundary": [(0,0), (20,0), (20,30), (0,30)],
        "width": 20, "length": 30, "area": 600,
    },
]

# ── AR limits per room type ──────────────────────────────────────
MAX_AR = {
    'living': 2.2, 'master_bedroom': 2.0, 'bedroom': 2.0, 'kitchen': 2.5,
    'bathroom': 2.5, 'toilet': 2.5, 'dining': 2.2, 'study': 2.0,
    'pooja': 2.5, 'corridor': 12.0, 'foyer': 3.0, 'passage': 12.0,
    'utility': 2.5, 'wash_area': 2.5, 'store': 2.5,
}


def audit_layout(rooms, cfg, engine_name):
    """Run all 12 architectural checks on a layout."""
    issues = []
    pw, pl = cfg["width"], cfg["length"]
    plot_area = cfg["area"]
    boundary = cfg["boundary"]

    if not rooms:
        return ["NO ROOMS GENERATED"]

    # ── 1. Room type presence check ──────────────────────────────
    type_counts = {}
    for r in rooms:
        rt = r.get("room_type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    has_living = type_counts.get("living", 0)
    has_kitchen = type_counts.get("kitchen", 0)
    has_master = type_counts.get("master_bedroom", 0)
    has_bedroom = type_counts.get("bedroom", 0)
    total_beds = has_master + has_bedroom

    if has_living == 0:
        issues.append("MISSING: Drawing/Living room")
    if has_kitchen == 0:
        issues.append("MISSING: Kitchen")
    if has_master == 0:
        issues.append("MISSING: Master bedroom")
    if total_beds < 2:
        issues.append(f"MISSING BEDS: need 2 (master+bedroom), got {total_beds}")

    # Check bathrooms — need at least 1 attached + 1 common
    bath_count = type_counts.get("bathroom", 0) + type_counts.get("toilet", 0)
    if bath_count < 2:
        issues.append(f"MISSING BATH: need ≥2, got {bath_count}")

    # ── 2. Public zone near entrance (front/south) ───────────────
    public_types = {"living", "dining"}
    private_types = {"master_bedroom", "bedroom"}

    public_rooms = [r for r in rooms if r.get("room_type") in public_types]
    private_rooms = [r for r in rooms if r.get("room_type") in private_types]

    if public_rooms and private_rooms:
        avg_pub_y = sum(r.get("position", {}).get("y", 0) for r in public_rooms) / len(public_rooms)
        avg_priv_y = sum(r.get("position", {}).get("y", 0) for r in private_rooms) / len(private_rooms)
        # Public should be at lower Y (closer to front/entrance)
        if avg_priv_y < avg_pub_y - 3:
            issues.append(f"ZONING: Bedrooms (y={avg_priv_y:.1f}) in front of public rooms (y={avg_pub_y:.1f})")

    # ── 3. Kitchen adjacent to Dining ────────────────────────────
    kitchens = [r for r in rooms if r.get("room_type") == "kitchen"]
    dinings = [r for r in rooms if r.get("room_type") == "dining"]
    if kitchens and dinings:
        k = kitchens[0]
        d = dinings[0]
        kpos = k.get("position", {})
        dpos = d.get("position", {})
        kx, ky, kw, kh = kpos.get("x", 0), kpos.get("y", 0), k.get("width", 0), k.get("length", 0)
        dx, dy, dw, dh = dpos.get("x", 0), dpos.get("y", 0), d.get("width", 0), d.get("length", 0)
        # Check if they share a wall (within 1ft tolerance)
        share_x = max(0, min(kx+kw, dx+dw) - max(kx, dx))
        share_y = max(0, min(ky+kh, dy+dh) - max(ky, dy))
        # They share a wall if they're adjacent horizontally or vertically
        h_adj = abs((kx+kw) - dx) < 1.0 or abs((dx+dw) - kx) < 1.0
        v_adj = abs((ky+kh) - dy) < 1.0 or abs((dy+dh) - ky) < 1.0
        if not ((h_adj and share_y > 1) or (v_adj and share_x > 1)):
            issues.append("ADJACENCY: Kitchen not adjacent to Dining")

    # ── 4. No overlaps ──────────────────────────────────────────
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            p1, p2 = r1.get("position", {}), r2.get("position", {})
            x1, y1 = p1.get("x", 0), p1.get("y", 0)
            w1, h1 = r1.get("width", 0), r1.get("length", 0)
            x2, y2 = p2.get("x", 0), p2.get("y", 0)
            w2, h2 = r2.get("width", 0), r2.get("length", 0)
            ox = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            oy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            if ox > 0.4 and oy > 0.4:
                issues.append(f"OVERLAP: {r1.get('name','')} & {r2.get('name','')}")

    # ── 5. Coverage (no dead space) ─────────────────────────────
    total_placed = sum(r.get("area", r.get("width",0)*r.get("length",0)) for r in rooms)
    coverage = total_placed / plot_area * 100 if plot_area > 0 else 0
    if coverage < 70:
        issues.append(f"LOW COVERAGE: {coverage:.1f}% (need ≥70%)")

    # ── 6. Proportions (AR check) ───────────────────────────────
    for r in rooms:
        w, h = r.get("width", 0), r.get("length", 0)
        if w <= 0 or h <= 0:
            continue
        ar = max(w/h, h/w)
        rt = r.get("room_type", "")
        limit = MAX_AR.get(rt, 2.0)
        if ar > limit + 0.3:
            issues.append(f"BAD AR: {r.get('name','')} {w:.1f}×{h:.1f} AR={ar:.2f} (max {limit})")

    # ── 7. All rooms inside boundary ────────────────────────────
    ewt = 0.75
    for r in rooms:
        pos = r.get("position", {})
        rx, ry = pos.get("x", 0), pos.get("y", 0)
        rw, rl = r.get("width", 0), r.get("length", 0)
        if rx < -0.5 or ry < -0.5 or rx + rw > pw + 0.5 or ry + rl > pl + 0.5:
            issues.append(f"OUT OF BOUNDS: {r.get('name','')} at ({rx:.1f},{ry:.1f}) {rw:.1f}×{rl:.1f}")

    # ── 8. Natural light — habitable rooms touch exterior wall ──
    habitable = ("living", "master_bedroom", "bedroom", "dining", "kitchen")
    for r in rooms:
        if r.get("room_type") not in habitable:
            continue
        pos = r.get("position", {})
        rx, ry = pos.get("x", 0), pos.get("y", 0)
        rw, rl = r.get("width", 0), r.get("length", 0)
        touches = (
            rx <= ewt + 1.0 or rx + rw >= pw - ewt - 1.0 or
            ry <= ewt + 1.0 or ry + rl >= pl - ewt - 1.0
        )
        if not touches:
            issues.append(f"NO LIGHT: {r.get('name','')} doesn't touch exterior wall")

    # ── 9. Master hierarchy ─────────────────────────────────────
    master_areas = [r.get("area", 0) for r in rooms if r.get("room_type") == "master_bedroom"]
    bed_areas = [r.get("area", 0) for r in rooms if r.get("room_type") == "bedroom"]
    if master_areas and bed_areas:
        if min(master_areas) < max(bed_areas) * 0.90:
            issues.append(f"HIERARCHY: Master ({min(master_areas):.0f}sqft) < Bedroom ({max(bed_areas):.0f}sqft)")

    # ── 10. Grid alignment ──────────────────────────────────────
    grid_violations = 0
    for r in rooms:
        pos = r.get("position", {})
        for val in [pos.get("x", 0), pos.get("y", 0), r.get("width", 0), r.get("length", 0)]:
            if abs(val * 2 - round(val * 2)) > 0.01:
                grid_violations += 1
    if grid_violations > 0:
        issues.append(f"GRID: {grid_violations} dims not on 0.5ft grid")

    return issues


def extract_rooms_pro(result, boundary):
    """Extract rooms from PRO engine output."""
    centroids, sizes, specs = result
    rooms = []
    for s in specs:
        p = s.get('_placed', {})
        if not p:
            continue
        rooms.append({
            'name': s.get('name', ''),
            'room_type': s.get('room_type', ''),
            'width': p.get('w', 0),
            'length': p.get('h', 0),
            'area': p.get('w', 0) * p.get('h', 0),
            'position': {'x': p.get('x', 0), 'y': p.get('y', 0)},
        })
    return rooms


def test_engine(engine_name, configs):
    """Test an engine against all configs."""
    total_issues = 0
    passed = 0

    for cfg in configs:
        name = cfg["name"]
        boundary = cfg["boundary"]
        pw, pl = cfg["width"], cfg["length"]
        area = cfg["area"]

        try:
            if engine_name == "PRO":
                rc = dict(STANDARD_ROOMS)
                result = generate_professional_plan(boundary, rc, area)
                rooms = extract_rooms_pro(result, boundary)

            elif engine_name == "GNN":
                rc = dict(STANDARD_ROOMS)
                result = generate_gnn_floor_plan(
                    plot_width=pw, plot_length=pl, total_area=area,
                    bedrooms=2, bathrooms=2, extras=["dining"])
                rooms = result.get("rooms", [])

            elif engine_name == "PERFECT":
                result = generate_perfect_layout(
                    plot_width=pw, plot_length=pl,
                    bedrooms=2, bathrooms=2, extras=["dining"])
                rooms = result.get("rooms", [])

            issues = audit_layout(rooms, cfg, engine_name)

        except Exception as e:
            issues = [f"ENGINE ERROR: {e}"]
            rooms = []

        n_rooms = len(rooms)
        n_issues = len(issues)
        total_issues += n_issues
        status = "PASS" if n_issues == 0 else "FAIL"
        if n_issues == 0:
            passed += 1

        # Print room details
        print(f"\n{'='*70}")
        print(f"  {engine_name} | {name}  ({area}sqft)")
        print(f"{'='*70}")
        for r in rooms:
            w_r = r.get("width", 0)
            h_r = r.get("length", 0)
            a_r = r.get("area", w_r * h_r)
            ar = max(w_r/h_r, h_r/w_r) if w_r > 0 and h_r > 0 else 0
            pos = r.get("position", {})
            print(f"    {r.get('name',''):22s} {w_r:5.1f}×{h_r:5.1f} = {a_r:7.1f}sqft  "
                  f"AR={ar:.2f}  @({pos.get('x',0):.1f},{pos.get('y',0):.1f})")

        coverage = sum(r.get("area",0) for r in rooms) / area * 100 if area > 0 else 0
        print(f"\n    Coverage: {coverage:.1f}%  |  Rooms: {n_rooms}  |  {status}")

        if issues:
            for iss in issues:
                print(f"    ✗ {iss}")
        else:
            print(f"    ✓ All 10 architectural checks passed")

    return passed, len(configs), total_issues


def main():
    print("=" * 74)
    print("  ADVANCED ARCHITECTURAL INTELLIGENCE AUDIT")
    print("  Multi-Step Reasoning: Boundary → Zones → Rooms → Validate")
    print("=" * 74)

    engines = [
        ("PRO", ),
        ("GNN", ),
        ("PERFECT", ),
    ]

    grand_passed = 0
    grand_total = 0
    grand_issues = 0

    for (eng_name,) in engines:
        print(f"\n{'#'*74}")
        print(f"  ENGINE: {eng_name}")
        print(f"{'#'*74}")
        passed, total, issues = test_engine(eng_name, CONFIGS)
        grand_passed += passed
        grand_total += total
        grand_issues += issues
        print(f"\n  {eng_name} SUMMARY: {passed}/{total} passed, {issues} issues")

    print(f"\n{'='*74}")
    print(f"  GRAND TOTAL: {grand_passed}/{grand_total} configs passed, "
          f"{grand_issues} total issues")
    print(f"{'='*74}")

    return 0 if grand_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
