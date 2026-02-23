"""
Phase 3 Verification — Steps 4–10.

Run with:
    cd backend
    .\\venv\\Scripts\\activate
    python test_phase3.py

Each step prints PASS / FAIL so you can confirm correctness at a glance.
"""

import json, sys, os

# Ensure imports work from backend/
sys.path.insert(0, os.path.dirname(__file__))

from shapely.geometry import Polygon
from services.layout_engine.loaders import (
    load_usable_polygon,
    save_usable_polygon,
    load_min_areas,
    load_region_rules,
)
from services.layout_engine.entrance import place_entrance
from services.layout_engine.room_model import Room
from services.layout_engine.geometry_utils import (
    clip_to_boundary,
    has_overlaps,
    detect_overlaps,
)
from services.layout_engine.adjacency import (
    build_adjacency_graph,
    is_connected,
    adjacency_pairs,
    shared_wall_midpoint,
)
from services.layout_engine import LayoutGenerator

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append(condition)
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)


# ====================================================================
print("=" * 60)
print("STEP 4 — Load Usable Polygon from JSON")
print("=" * 60)

# Test loading from usable_polygon.json
poly = load_usable_polygon("usable_polygon.json")
check("File loads without error", poly is not None)
check("Result is a Shapely Polygon", poly.geom_type == "Polygon")
check("Polygon is valid", poly.is_valid)
check("Polygon has correct area (120 sq m)", abs(poly.area - 120.0) < 0.01,
      f"area={poly.area:.2f}")
check("Polygon has 5 coords (closed ring)",
      len(list(poly.exterior.coords)) == 5,
      f"coords={len(list(poly.exterior.coords))}")

# Test round-trip save/load
save_usable_polygon(poly, "_test_roundtrip.json")
poly2 = load_usable_polygon("_test_roundtrip.json")
check("Save + reload preserves polygon", poly.equals(poly2))
os.remove("_test_roundtrip.json")

# Test different JSON formats
test_formats = {
    "flat list format": {"polygon": [[0, 0], [10, 0], [10, 8], [0, 8]]},
    "vertices format": {"vertices": [[0, 0], [5, 0], [5, 5], [0, 5]]},
}
for fmt_name, data in test_formats.items():
    tmp = f"_test_{fmt_name.replace(' ', '_')}.json"
    with open(tmp, "w") as f:
        json.dump(data, f)
    p = load_usable_polygon(tmp)
    check(f"Loads {fmt_name}", p is not None and p.area > 0, f"area={p.area:.1f}")
    os.remove(tmp)

print()


# ====================================================================
print("=" * 60)
print("STEP 5 — Polygon Clipping")
print("=" * 60)

boundary = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)])
room_inside = Polygon([(1, 1), (5, 1), (5, 4), (1, 4)])
room_partial = Polygon([(-2, 2), (3, 2), (3, 6), (-2, 6)])
room_outside = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])

clipped_inside = clip_to_boundary(room_inside, boundary)
check("Fully-inside room unchanged",
      clipped_inside is not None and abs(clipped_inside.area - room_inside.area) < 0.01,
      f"area={clipped_inside.area:.2f}")

clipped_partial = clip_to_boundary(room_partial, boundary)
check("Partially-outside room is clipped",
      clipped_partial is not None and clipped_partial.area < room_partial.area,
      f"original={room_partial.area:.1f} → clipped={clipped_partial.area:.1f}")
check("Clipped room lies within boundary",
      boundary.contains(clipped_partial) or boundary.covers(clipped_partial))

clipped_outside = clip_to_boundary(room_outside, boundary)
check("Fully-outside room returns None", clipped_outside is None)

print()


# ====================================================================
print("=" * 60)
print("STEP 6 — Overlap Detection")
print("=" * 60)

r1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
r2 = Polygon([(4, 0), (8, 0), (8, 5), (4, 5)])  # overlaps r1 by 1x5
r3 = Polygon([(5, 0), (8, 0), (8, 5), (5, 5)])  # touches r1 edge only
r4 = Polygon([(0, 5), (5, 5), (5, 10), (0, 10)])  # touches r1 edge only
r5 = Polygon([(9, 0), (12, 0), (12, 5), (9, 5)])  # separate from r1

check("Overlapping pair detected",
      has_overlaps([r1, r2]), f"overlap area={r1.intersection(r2).area:.1f}")
check("Edge-touching pair NOT flagged as overlap",
      not has_overlaps([r1, r3]))
check("Detect exact overlap pair indices",
      detect_overlaps([r1, r2, r5]) == [(0, 1)],
      f"pairs={detect_overlaps([r1, r2, r5])}")
check("Three non-overlapping rooms pass",
      not has_overlaps([r1, r3, r4]))

print()


# ====================================================================
print("=" * 60)
print("STEP 7 — Minimum Area Enforcement")
print("=" * 60)

rules = load_region_rules("region_rules.json", "india_mvp")
check("Region rules loaded", rules is not None)
check("Rules contain min_room_areas", "min_room_areas" in rules,
      f"keys={list(rules.keys())}")

min_areas = load_min_areas("region_rules.json", "india_mvp")
check("Min areas dict is not empty", len(min_areas) > 0,
      f"types={list(min_areas.keys())}")
check("Living room min ≥ 9.5 sq m", min_areas.get("living", 0) >= 9.5,
      f"living={min_areas.get('living')}")
check("Bathroom min ≥ 1.8 sq m", min_areas.get("bathroom", 0) >= 1.8,
      f"bathroom={min_areas.get('bathroom')}")

# Verify generator rejects rooms below min area
gen = LayoutGenerator(
    boundary=Polygon([(0, 0), (12, 0), (12, 10), (0, 10)]),
    room_requirements=[{"room_type": "living", "size": 5}],
    min_areas={"living": 99999.0},  # impossibly high → should reject all
)
result = gen.generate(n_candidates=10, method="treemap")
check("Generator rejects candidates below min area",
      result["candidates_valid"] == 0,
      f"valid={result['candidates_valid']}")

print()


# ====================================================================
print("=" * 60)
print("STEP 8 — Room Model")
print("=" * 60)

Room.reset_counter()
room_a = Room(
    room_type="bedroom",
    polygon=Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]),
    target_area=12.0,
    floor=0,
)
check("Room has type", room_a.room_type == "bedroom")
check("Room has polygon", room_a.polygon.geom_type == "Polygon")
check("Room area computed from polygon", abs(room_a.area - 12.0) < 0.01,
      f"area={room_a.area}")
check("Room has target_area", room_a.target_area == 12.0)
check("Room has floor", room_a.floor == 0)
check("Room.area_ratio = 1.0 for perfect match",
      abs(room_a.area_ratio - 1.0) < 0.01)

d = room_a.to_dict()
check("to_dict() has all fields",
      all(k in d for k in ("room_id", "room_type", "polygon", "area", "target_area", "floor")),
      f"keys={list(d.keys())}")

room_from_rect = Room.from_rect(2, 3, 5, 4, room_type="kitchen")
check("from_rect creates Polygon correctly",
      abs(room_from_rect.area - 20.0) < 0.01,
      f"area={room_from_rect.area}")

print()


# ====================================================================
print("=" * 60)
print("STEP 9 — Entrance Placement")
print("=" * 60)

boundary9 = Polygon([(0, 0), (12, 0), (12, 10), (0, 10)])
Room.reset_counter()
rooms9 = [
    Room(room_type="living", polygon=Polygon([(0, 0), (6, 0), (6, 5), (0, 5)]),
         target_area=30),
    Room(room_type="bedroom", polygon=Polygon([(6, 0), (12, 0), (12, 5), (6, 5)]),
         target_area=30),
    Room(room_type="kitchen", polygon=Polygon([(0, 5), (12, 5), (12, 10), (0, 10)]),
         target_area=60),
]

entrance = place_entrance(boundary9, rooms9, preferred_side="south")
check("Entrance room created", entrance is not None)
if entrance:
    check("Entrance type is 'entrance'", entrance.room_type == "entrance")
    check("Entrance has valid polygon",
          entrance.polygon.is_valid and entrance.polygon.area > 0,
          f"area={entrance.area:.2f}")
    check("Entrance lies within boundary",
          boundary9.contains(entrance.polygon) or boundary9.covers(entrance.polygon))
    # Check connection to at least one room
    connected = any(
        entrance.polygon.intersects(r.polygon) for r in rooms9
    )
    check("Entrance connects to interior rooms", connected)
else:
    check("Entrance type is 'entrance'", False, "entrance was None")
    check("Entrance has valid polygon", False)
    check("Entrance lies within boundary", False)
    check("Entrance connects to interior rooms", False)

print()


# ====================================================================
print("=" * 60)
print("STEP 10 — Adjacency Graph")
print("=" * 60)

Room.reset_counter()
adj_rooms = [
    {"room_id": 0, "room_type": "living",
     "polygon": Polygon([(0, 0), (6, 0), (6, 5), (0, 5)])},
    {"room_id": 1, "room_type": "bedroom",
     "polygon": Polygon([(6, 0), (12, 0), (12, 5), (6, 5)])},
    {"room_id": 2, "room_type": "kitchen",
     "polygon": Polygon([(0, 5), (12, 5), (12, 10), (0, 10)])},
]

graph = build_adjacency_graph(adj_rooms)
check("Graph has 3 nodes", graph.number_of_nodes() == 3,
      f"nodes={graph.number_of_nodes()}")
check("Graph has edges (rooms share walls)",
      graph.number_of_edges() > 0,
      f"edges={graph.number_of_edges()}")
check("Graph is connected", is_connected(graph))

pairs = adjacency_pairs(graph)
check("Living–bedroom adjacent (shared wall at x=6)",
      (0, 1) in pairs or (1, 0) in pairs, f"pairs={pairs}")
check("Living–kitchen adjacent (shared wall at y=5)",
      (0, 2) in pairs or (2, 0) in pairs)

mid = shared_wall_midpoint(adj_rooms[0]["polygon"], adj_rooms[1]["polygon"])
check("Shared wall midpoint computed",
      mid is not None, f"midpoint={mid}")

# Test disconnected layout detection
disconnected_rooms = [
    {"room_id": 0, "room_type": "a",
     "polygon": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])},
    {"room_id": 1, "room_type": "b",
     "polygon": Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])},
]
g2 = build_adjacency_graph(disconnected_rooms)
check("Disconnected layout detected", not is_connected(g2))

print()


# ====================================================================
print("=" * 60)
print("INTEGRATION — Full Pipeline (Steps 4-10 together)")
print("=" * 60)

gen_full = LayoutGenerator.from_json(
    polygon_path="usable_polygon.json",
    room_requirements=[
        {"room_type": "living", "size": 7},
        {"room_type": "bedroom", "size": 5},
        {"room_type": "kitchen", "size": 4},
        {"room_type": "bathroom", "size": 3},
    ],
    rules_path="region_rules.json",
    region="india_mvp",
    desired_adjacencies=[("living", "kitchen"), ("bedroom", "bathroom")],
)
check("Generator created from JSON files", gen_full is not None)

result = gen_full.generate(n_candidates=50, method="mixed")
check("Generation produced valid candidates",
      result["candidates_valid"] > 0,
      f"valid={result['candidates_valid']}/{result['candidates_generated']}")
check("Best layout is non-empty",
      len(result["best_layout"]) > 0,
      f"rooms={len(result['best_layout'])}")
check("Score is positive",
      result["score"]["total"] > 0,
      f"score={result['score']}")

# Verify all rooms in best layout have required fields
if result["best_layout"]:
    sample = result["best_layout"][0]
    check("Layout rooms have all fields",
          all(k in sample for k in ("room_id", "room_type", "polygon", "area", "target_area", "floor")),
          f"keys={list(sample.keys())}")

    # Check for entrance in layout
    types = [r["room_type"] for r in result["best_layout"]]
    has_entrance = "entrance" in types
    check("Entrance room included in layout", has_entrance,
          f"room_types={types}")

print()


# ====================================================================
# Summary
# ====================================================================
print("=" * 60)
passed = sum(results)
total = len(results)
pct = (passed / total * 100) if total else 0
color = "\033[92m" if passed == total else "\033[93m"
print(f"{color}RESULTS: {passed}/{total} checks passed ({pct:.0f}%)\033[0m")
print("=" * 60)

sys.exit(0 if passed == total else 1)
