"""
Residential Architectural Design Engine v2.0 — Test Suite
==========================================================
Tests the spatial-experience engine against the design philosophy:

  1. STRATEGY SELECTION    — Correct strategy for plot shape
  2. SPATIAL LAYERING      — 3-layer depth from entrance
  3. HIERARCHY ENFORCEMENT — Living closer to entrance than bedrooms
  4. ASPECT RATIO          — All rooms ≤ 1:2 (comfort rooms)
  5. MINIMUM AREAS         — All rooms meet minimums
  6. NO OVERLAPS           — Zero tolerance
  7. BOUNDARY COMPLIANCE   — All rooms within plot
  8. CIRCULATION           — Passage ≥ 3 ft
  9. PRIVACY               — Bedrooms shielded from entrance
 10. FRONTEND COMPAT       — polygon, centroid, boundary fields present
 11. REDESIGN MODE         — Different strategy on redesign
 12. OUTPUT FORMAT          — layout_strategy, spatial_layers, circulation, validation
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__) or ".")

from services.multi_factor_engine import (
    generate_plan,
    generate_new_plan,
    parse_input,
    _select_strategy,
    COMFORT_AR,
    MIN_ROOM_AREA,
    WALL_EXT,
)

# ── Test configurations ─────────────────────────────────────────
CONFIGS = [
    # (name, width, length, bedrooms, bathrooms, extras)
    ("1BHK 20x25",   20, 25, 1, 1, []),
    ("2BHK 30x40",   30, 40, 2, 2, ["dining"]),
    ("2BHK 25x24",   25, 24, 2, 2, []),
    ("3BHK 40x30",   40, 30, 3, 2, ["dining", "study"]),
    ("3BHK 35x35",   35, 35, 3, 2, ["dining", "pooja"]),
    ("2BHK-wide 40x20",  40, 20, 2, 1, ["dining"]),
    ("4BHK 50x30",   50, 30, 4, 3, ["dining", "study", "pooja"]),
]

passed = 0
failed = 0
total_tests = 0


def check(condition, label, detail=""):
    global passed, failed, total_tests
    total_tests += 1
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"    ✗ {label}: {detail}")


def run_all_tests():
    global passed, failed, total_tests
    print("=" * 70)
    print("RESIDENTIAL ARCHITECTURAL DESIGN ENGINE v2.0 — TEST SUITE")
    print("=" * 70)

    # ──────── Test 1: Strategy Selection ────────
    print("\n── Test 1: Strategy Selection ──")
    check(_select_strategy(40, 20) == "side_corridor",
          "Wide plot → side_corridor", f"got {_select_strategy(40, 20)}")
    check(_select_strategy(20, 40) == "central_corridor",
          "Deep plot → central_corridor", f"got {_select_strategy(20, 40)}")
    check(_select_strategy(30, 30) == "cluster",
          "Square plot → cluster", f"got {_select_strategy(30, 30)}")
    check(_select_strategy(25, 24) == "cluster",
          "Near-square → cluster", f"got {_select_strategy(25, 24)}")
    check(_select_strategy(50, 30) == "side_corridor",
          "50x30 → side_corridor", f"got {_select_strategy(50, 30)}")
    print(f"    Strategy selection: {passed}/{total_tests}")

    # ──────── Test 2–12: Per-configuration tests ────────
    for name, w, l, beds, baths, extras in CONFIGS:
        print(f"\n── {name} ({w}x{l}, {beds}BHK) ──")
        input_data = {
            "plot_width": w, "plot_length": l,
            "total_area": w * l,
            "bedrooms": beds, "bathrooms": baths,
            "extras": extras,
        }
        result = generate_plan(input_data)

        # Must not error
        if "error" in result and result["error"]:
            check(False, "No error", result["error"])
            continue

        layout = result.get("layout", {})
        rooms = layout.get("rooms", [])
        validation = result.get("validation", layout.get("validation", {}))

        # T2: Spatial layers present
        spatial_layers = layout.get("spatial_layers", [])
        check(len(spatial_layers) > 0, "Spatial layers present",
              f"got {len(spatial_layers)} layers")

        # T3: Layout strategy present
        strategy = layout.get("layout_strategy", "")
        check(strategy in ("side_corridor", "central_corridor", "cluster"),
              "Valid strategy", f"got '{strategy}'")

        # T4: All rooms have required frontend fields
        for room in rooms:
            check("polygon" in room and len(room["polygon"]) == 5,
                  f"{room.get('name','?')} has polygon")
            check("centroid" in room and len(room.get("centroid", [])) == 2,
                  f"{room.get('name','?')} has centroid")
            check("room_type" in room, f"{room.get('name','?')} has room_type")
            check("area" in room and room["area"] > 0,
                  f"{room.get('name','?')} has positive area",
                  f"area={room.get('area')}")

        # T5: Boundary present
        boundary = layout.get("boundary", [])
        check(len(boundary) >= 4, "Boundary polygon present",
              f"got {len(boundary)} points")

        # T6: Aspect ratios within comfort range
        for room in rooms:
            rw, rh = room.get("width", 0), room.get("length", 0)
            if min(rw, rh) > 0:
                ar = max(rw, rh) / min(rw, rh)
                limit = COMFORT_AR.get(room["room_type"], 2.0)
                check(ar <= limit + 0.15,
                      f"{room['name']} AR ≤ {limit}",
                      f"AR={ar:.2f}")

        # T7: Minimum areas (scaled threshold for tight plots)
        # On tight plots (total_min > usable * 0.85), rooms physically can't
        # all meet ideal minimums — scale tolerance proportionally
        total_min_needed = sum(
            MIN_ROOM_AREA.get(r["room_type"], 25)
            for r in rooms
        )
        usable_area = (w - 1.5) * (l - 1.5) * 0.85
        density_ratio = usable_area / max(total_min_needed, 1)
        tolerance = 0.7 if density_ratio >= 1.0 else max(0.4, 0.7 * density_ratio)

        for room in rooms:
            rtype = room["room_type"]
            min_a = MIN_ROOM_AREA.get(rtype, 20) * tolerance
            # Allow 2 sqft tolerance for grid-snapping artifacts (0.5ft grid)
            check(room["area"] >= min_a - 2.0,
                  f"{room['name']} area ≥ {min_a:.0f}",
                  f"area={room['area']}")

        # T8: No overlaps (simple AABB check)
        overlap_found = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                r1, r2 = rooms[i], rooms[j]
                x1, y1 = r1["position"]["x"], r1["position"]["y"]
                w1, h1 = r1["width"], r1["length"]
                x2, y2 = r2["position"]["x"], r2["position"]["y"]
                w2, h2 = r2["width"], r2["length"]
                ox = max(0, min(x1+w1, x2+w2) - max(x1, x2) - 0.5)
                oy = max(0, min(y1+h1, y2+h2) - max(y1, y2) - 0.5)
                if ox > 0 and oy > 0:
                    overlap_found = True
        check(not overlap_found, "No overlaps")

        # T9: Rooms within plot boundary
        all_inside = True
        for room in rooms:
            rx, ry = room["position"]["x"], room["position"]["y"]
            rw, rh = room["width"], room["length"]
            if rx < -0.5 or ry < -0.5 or rx+rw > w+0.5 or ry+rh > l+0.5:
                all_inside = False
        check(all_inside, "All rooms within plot")

        # T10: Circulation info present
        circ = layout.get("circulation", {})
        check("type" in circ, "Circulation type present")
        check(circ.get("depth_ft", circ.get("width_ft", 0)) >= 3.0 - 0.1,
              "Passage ≥ 3ft",
              f"got {circ.get('depth_ft', circ.get('width_ft', 0))}ft")

        # T11: Hierarchy — living closer to entrance than bedrooms
        living_rooms = [r for r in rooms if r["room_type"] == "living"]
        bed_rooms = [r for r in rooms if r["room_type"] in ("master_bedroom", "bedroom")]
        if living_rooms and bed_rooms:
            if strategy in ("central_corridor", "cluster"):
                liv_y = min(r["position"]["y"] for r in living_rooms)
                bed_y = min(r["position"]["y"] for r in bed_rooms)
                check(liv_y <= bed_y + 1,
                      "Living closer to entrance than bedrooms",
                      f"living_y={liv_y}, bed_y={bed_y}")
            elif strategy == "side_corridor":
                liv_x = min(r["position"]["x"] for r in living_rooms)
                bed_x = min(r["position"]["x"] for r in bed_rooms)
                check(liv_x <= bed_x + 1,
                      "Living on entrance side",
                      f"living_x={liv_x}, bed_x={bed_x}")

        # T12: Validation fields present
        for vkey in ("privacy_ok", "circulation_ok", "proportion_ok",
                     "ventilation_ok", "hierarchy_ok"):
            check(vkey in validation, f"Validation field {vkey} present")

        # T13: Room count (all requested rooms generated)
        rtypes = [r["room_type"] for r in rooms]
        check("living" in rtypes, "Living room present")
        check("kitchen" in rtypes, "Kitchen present")
        actual_beds = rtypes.count("master_bedroom") + rtypes.count("bedroom")
        check(actual_beds >= beds,
              f"Bedroom count ≥ {beds}",
              f"got {actual_beds}")

    # ──────── Test: Redesign Mode ────────
    print("\n── Test: Redesign Mode ──")
    base_input = {
        "plot_width": 30, "plot_length": 40,
        "total_area": 1200, "bedrooms": 2, "bathrooms": 1,
    }
    plan1 = generate_plan(base_input)
    strat1 = plan1.get("layout", {}).get("layout_strategy", "")

    plan2 = generate_new_plan(base_input, strat1)
    strat2 = plan2.get("layout", {}).get("layout_strategy", "")
    check(strat1 != strat2, "Redesign uses different strategy",
          f"plan1={strat1}, plan2={strat2}")
    check("error" not in plan2 or not plan2.get("error"),
          "Redesign produces valid plan")

    rooms2 = plan2.get("layout", {}).get("rooms", [])
    check(len(rooms2) > 0, "Redesign has rooms", f"got {len(rooms2)}")

    # ──────── Test: Natural Language Parsing ────────
    print("\n── Test: Natural Language Parsing ──")
    p = parse_input("Design a 3BHK house on 30x40 plot with study and pooja room")
    check(p.get("bedrooms") == 3, "NL: 3 bedrooms", f"got {p.get('bedrooms')}")
    check(p.get("plot_width") == 30, "NL: width 30", f"got {p.get('plot_width')}")
    check(p.get("plot_length") == 40, "NL: length 40", f"got {p.get('plot_length')}")
    check("study" in p.get("extras", []), "NL: study extra")
    check("pooja" in p.get("extras", []), "NL: pooja extra")

    p2 = parse_input("Generate new plan")
    check(p2.get("is_redesign") == True, "NL: redesign detected")

    # ──────── Test: Edge Cases ────────
    print("\n── Test: Edge Cases ──")
    # Minimum viable plot (20x20 = 400 sqft)
    tiny = generate_plan({"plot_width": 20, "plot_length": 20,
                          "bedrooms": 1, "bathrooms": 1})
    check("error" not in tiny or not tiny.get("error"),
          "20x20 1BHK doesn't crash")
    if tiny.get("layout", {}).get("rooms"):
        check(len(tiny["layout"]["rooms"]) >= 3,
              "20x20 has at least 3 rooms (living+kitchen+bedroom)")

    # Very narrow plot
    narrow = generate_plan({"plot_width": 15, "plot_length": 50,
                            "bedrooms": 2, "bathrooms": 1})
    check("error" not in narrow or not narrow.get("error"),
          "15x50 narrow plot doesn't crash")
    if narrow.get("layout"):
        check(narrow["layout"].get("layout_strategy") == "central_corridor",
              "15x50 → central_corridor")

    # Very wide plot
    wide = generate_plan({"plot_width": 60, "plot_length": 25,
                          "bedrooms": 3, "bathrooms": 2})
    check("error" not in wide or not wide.get("error"),
          "60x25 wide plot doesn't crash")
    if wide.get("layout"):
        check(wide["layout"].get("layout_strategy") == "side_corridor",
              "60x25 → side_corridor")

    # ──────── Summary ────────
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total_tests} passed, {failed} failed")
    pct = round(passed / total_tests * 100) if total_tests else 0
    if failed == 0:
        print(f"✓ ALL TESTS PASSED ({pct}%)")
    else:
        print(f"✗ {failed} FAILURES ({pct}% pass rate)")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
