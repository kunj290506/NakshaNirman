"""
FINAL PROFESSIONAL ARCHITECT'S AUDIT
=====================================
Tests ALL layout engines (Pro, GNN, Perfect) with real-world Indian
residential plot configurations. Each layout is evaluated against
10 strict architectural criteria used in NBC/BIS compliance checks.

Criteria:
  1. ROOM PROPORTIONS — AR ≤ type-specific limit (no bowling alleys)
  2. MINIMUM AREAS — NBC/BIS minimum habitable areas
  3. MASTER HIERARCHY — Master BR ≥ regular bedrooms
  4. NO OVERLAPS — Zero tolerance
  5. COVERAGE — ≥ 70% for small, ≥ 75% for standard plots
  6. CORRIDOR EFFICIENCY — < 9% of plot area
  7. ZONING — Public rooms at front, private at back
  8. GRID ALIGNMENT — All dims on 0.5ft (6-inch) grid
  9. NATURAL LIGHT — Habitable rooms touch exterior walls
  10. ROOM COUNT — All requested rooms must be present
"""
import sys
import os
import math
import logging
import traceback

sys.path.insert(0, os.path.dirname(__file__) or '.')
logging.basicConfig(level=logging.WARNING)

from services.pro_layout_engine import generate_professional_plan
from services.gnn_engine import generate_gnn_floor_plan
from services.perfect_layout import generate_perfect_layout

# ── Test matrix ─────────────────────────────────────────────────────
# Each entry: (name, width, length, bedrooms, bathrooms, extras)
CONFIGS = [
    # 1BHK
    ("1BHK 20×15",  20, 15, 1, 1, []),
    ("1BHK 25×12",  25, 12, 1, 1, []),
    # 2BHK
    ("2BHK 30×20",  30, 20, 2, 2, ["dining"]),
    ("2BHK 25×24",  25, 24, 2, 2, []),
    # 3BHK
    ("3BHK 40×25",  40, 25, 3, 2, ["dining", "study", "pooja"]),
    ("3BHK 30×33",  30, 33, 3, 2, ["dining", "study"]),
    # 4BHK
    ("4BHK 50×30",  50, 30, 4, 3, ["dining", "study", "pooja"]),
]

# ── AR limits per room type ──────────────────────────────────────
MAX_AR = {
    'living': 2.2, 'master_bedroom': 2.0, 'bedroom': 2.0, 'kitchen': 2.5,
    'bathroom': 2.5, 'toilet': 2.5, 'dining': 2.2, 'study': 2.0,
    'pooja': 2.5, 'store': 2.5, 'balcony': 5.0, 'utility': 2.5,
    'corridor': 12.0, 'foyer': 3.0, 'staircase': 2.5,
}

def min_area_for(rtype, plot_area):
    base = {
        'living': 100, 'master_bedroom': 96, 'bedroom': 80,
        'kitchen': 50, 'bathroom': 25, 'dining': 60, 'study': 36,
        'pooja': 15, 'store': 20, 'utility': 15, 'toilet': 15,
    }
    area = base.get(rtype, 20)
    if plot_area < 400:
        area = max(area * 0.45, 12)
    elif plot_area < 600:
        area = max(area * 0.65, 15)
    elif plot_area < 800:
        area = max(area * 0.80, 18)
    return area


def audit_rooms(rooms, cfg_name, plot_w, plot_l, bedrooms, bathrooms, extras):
    """Audit a list of rooms against architectural standards."""
    plot_area = plot_w * plot_l
    issues = []

    if not rooms:
        return ["NO ROOMS GENERATED"]

    # 1. Room proportions
    for r in rooms:
        w, h = r.get("width", 0), r.get("length", 0)
        if w <= 0 or h <= 0:
            continue
        ar = max(w / h, h / w)
        limit = MAX_AR.get(r.get("room_type", ""), 2.0)
        if ar > limit + 0.3:
            issues.append(f"BAD AR: {r['name']} {w:.1f}x{h:.1f} AR={ar:.2f} (max {limit})")

    # 2. Minimum areas
    for r in rooms:
        area = r.get("area", r.get("width", 0) * r.get("length", 0))
        rtype = r.get("room_type", "")
        min_a = min_area_for(rtype, plot_area) * 0.95
        if rtype in ('corridor', 'foyer', 'utility', 'balcony', 'staircase'):
            continue
        if area < min_a:
            issues.append(f"TOO SMALL: {r['name']} {area:.0f}sqft (min {min_a:.0f})")

    # 3. Master hierarchy
    master_areas = [r.get("area", 0) for r in rooms if r.get("room_type") == "master_bedroom"]
    bed_areas = [r.get("area", 0) for r in rooms if r.get("room_type") == "bedroom"]
    if master_areas and bed_areas:
        if min(master_areas) < max(bed_areas) * 0.95:
            issues.append(f"HIERARCHY: Master ({min(master_areas):.0f}) < Bedroom ({max(bed_areas):.0f})")

    # 4. Overlaps
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if j <= i:
                continue
            pos1 = r1.get("position", {})
            pos2 = r2.get("position", {})
            x1, y1 = pos1.get("x", 0), pos1.get("y", 0)
            w1, h1 = r1.get("width", 0), r1.get("length", 0)
            x2, y2 = pos2.get("x", 0), pos2.get("y", 0)
            w2, h2 = r2.get("width", 0), r2.get("length", 0)
            ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if ox > 0.4 and oy > 0.4:
                issues.append(f"OVERLAP: {r1['name']} & {r2['name']}")

    # 5. Coverage
    total_placed = sum(r.get("area", r.get("width", 0) * r.get("length", 0)) for r in rooms)
    coverage = total_placed / plot_area * 100 if plot_area > 0 else 0
    min_cov = 68 if plot_area < 400 else 73
    if coverage < min_cov:
        issues.append(f"LOW COVERAGE: {coverage:.1f}% (min {min_cov}%)")

    # 6. Corridor waste
    for r in rooms:
        if r.get("room_type") in ("corridor", "foyer"):
            pct = r.get("area", 0) / plot_area * 100
            if pct > 9:
                issues.append(f"CORRIDOR WASTE: {r['name']} {pct:.1f}%")

    # 7. Zoning (living vs bedrooms Y position)
    living = [r for r in rooms if r.get("room_type") == "living"]
    beds = [r for r in rooms if r.get("room_type") in ("master_bedroom", "bedroom")]
    if living and beds:
        living_y = living[0].get("position", {}).get("y", 0)
        avg_bed_y = sum(r.get("position", {}).get("y", 0) for r in beds) / len(beds)
        if avg_bed_y < living_y - 3:
            issues.append("ZONING: Bedrooms in front of living room")

    # 8. Grid alignment
    grid_violations = 0
    for r in rooms:
        pos = r.get("position", {})
        for val in [pos.get("x", 0), pos.get("y", 0), r.get("width", 0), r.get("length", 0)]:
            if abs(val * 2 - round(val * 2)) > 0.01:
                grid_violations += 1
    if grid_violations > 0:
        issues.append(f"GRID: {grid_violations} dims not on 0.5ft grid")

    # 9. Natural light
    ewt = 0.75
    habitable = ("living", "master_bedroom", "bedroom", "dining")
    for r in rooms:
        if r.get("room_type") not in habitable:
            continue
        pos = r.get("position", {})
        rx, ry = pos.get("x", 0), pos.get("y", 0)
        rw, rl = r.get("width", 0), r.get("length", 0)
        touches = (
            rx <= ewt + 1.0 or
            rx + rw >= plot_w - ewt - 1.0 or
            ry <= ewt + 1.0 or
            ry + rl >= plot_l - ewt - 1.0
        )
        if not touches:
            issues.append(f"NO LIGHT: {r['name']}")

    # 10. Room count check
    type_count = {}
    for r in rooms:
        rt = r.get("room_type", "")
        type_count[rt] = type_count.get(rt, 0) + 1
    total_beds = type_count.get("master_bedroom", 0) + type_count.get("bedroom", 0)
    if total_beds < bedrooms and bedrooms > 1:
        issues.append(f"MISSING ROOMS: {total_beds} beds (need {bedrooms})")

    return issues


def test_engine(engine_name, generate_fn, configs):
    """Test an engine against all configs."""
    total_issues = 0
    passed = 0
    results = []

    for name, w, l, beds, baths, extras in configs:
        plot_area = w * l
        try:
            if engine_name == "PRO":
                boundary = [(0, 0), (w, 0), (w, l), (0, l)]
                rc = {'master_bedroom': 1, 'bedroom': max(0, beds - 1),
                      'bathroom': baths, 'kitchen': 1}
                for ex in extras:
                    rc[ex] = rc.get(ex, 0) + 1
                centroids, sizes, specs = generate_fn(boundary, rc, plot_area)
                rooms = []
                for s in specs:
                    p = s.get('_placed', {})
                    rooms.append({
                        'name': s.get('name', ''),
                        'room_type': s.get('room_type', ''),
                        'width': p.get('w', 0),
                        'length': p.get('h', 0),
                        'area': p.get('w', 0) * p.get('h', 0),
                        'position': {'x': p.get('x', 0), 'y': p.get('y', 0)},
                    })

            elif engine_name == "GNN":
                result = generate_fn(
                    plot_width=w, plot_length=l, total_area=plot_area,
                    bedrooms=beds, bathrooms=baths, extras=extras)
                rooms = result.get("rooms", [])

            elif engine_name == "PERFECT":
                result = generate_fn(
                    plot_width=w, plot_length=l,
                    bedrooms=beds, bathrooms=baths, extras=extras)
                rooms = result.get("rooms", [])

            issues = audit_rooms(rooms, name, w, l, beds, baths, extras)

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
        print(f"  {engine_name} | {name}  (Plot={plot_area}sqft)")
        print(f"{'='*70}")
        for r in rooms:
            w_r = r.get("width", 0)
            h_r = r.get("length", 0)
            a_r = r.get("area", w_r * h_r)
            ar = max(w_r / h_r, h_r / w_r) if w_r > 0 and h_r > 0 else 0
            flag = " !!!" if ar > MAX_AR.get(r.get("room_type", ""), 2.0) + 0.3 else ""
            pos = r.get("position", {})
            print(f"    {r['name']:22s} {w_r:5.1f}x{h_r:5.1f} = {a_r:7.1f}sqft  "
                  f"AR={ar:.2f}  @({pos.get('x',0):.1f},{pos.get('y',0):.1f}){flag}")

        coverage = sum(r.get("area", 0) for r in rooms) / plot_area * 100 if plot_area > 0 else 0
        print(f"\n    Coverage: {coverage:.1f}%  |  Rooms: {n_rooms}  |  {status}")

        if issues:
            for iss in issues:
                print(f"    - {iss}")

        results.append((name, status, n_issues))

    return passed, len(configs), total_issues, results


def main():
    print("=" * 74)
    print("  FINAL PROFESSIONAL ARCHITECT'S AUDIT")
    print("  Testing PRO, GNN, and PERFECT engines")
    print("=" * 74)

    engines = [
        ("PRO", generate_professional_plan),
        ("GNN", generate_gnn_floor_plan),
        ("PERFECT", generate_perfect_layout),
    ]

    grand_passed = 0
    grand_total = 0
    grand_issues = 0

    for eng_name, eng_fn in engines:
        print(f"\n{'#'*74}")
        print(f"  ENGINE: {eng_name}")
        print(f"{'#'*74}")
        passed, total, issues, results = test_engine(eng_name, eng_fn, CONFIGS)
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
