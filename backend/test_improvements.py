"""Quick validation test for the improved layout engines."""
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(level=logging.WARNING)

from services.pro_layout_engine import generate_professional_plan
from services.perfect_layout import generate_perfect_layout


def test_pro_layout(label, boundary, rooms_config, total_area):
    print("=" * 60)
    print(f"TEST: pro_layout_engine — {label}")
    print("=" * 60)
    centroids, sizes, room_specs = generate_professional_plan(
        boundary_coords=boundary,
        rooms_config=rooms_config,
        total_area=total_area,
    )
    print(f"Generated {len(room_specs)} rooms (plot={total_area} sqft)")
    # Debug: show first room keys
    if room_specs:
        print(f"  Room keys: {list(room_specs[0].keys())}")
    for r in room_specs:
        rtype = r.get('type', '?')
        name = r.get('name', rtype)
        placed = r.get('_placed', {})
        w = placed.get('w', r.get('width', r.get('w', 0)))
        h = placed.get('h', r.get('height', r.get('h', 0)))
        area = w * h
        ar = max(w/h, h/w) if h > 0 and w > 0 else 0
        print(f"  {name:20s} {w:5.1f} x {h:5.1f} = {area:6.1f} sqft  AR={ar:.2f}")
    print()


def test_perfect_layout(label, pw, pl, beds, baths, extras):
    print("=" * 60)
    print(f"TEST: perfect_layout — {label}")
    print("=" * 60)
    result = generate_perfect_layout(
        plot_width=pw,
        plot_length=pl,
        bedrooms=beds,
        bathrooms=baths,
        extras=extras,
    )
    rooms = result.get('rooms', [])
    total_area = pw * pl
    # Debug: show first room keys
    if rooms:
        print(f"  Room keys: {list(rooms[0].keys())}")
    covered = 0
    for r in rooms:
        w = r.get('width', r.get('w', 0))
        h = r.get('height', r.get('h', r.get('length', 0)))
        area = w * h
        covered += area
        ar = max(w/h, h/w) if h > 0 and w > 0 else 0
        print(f"  {r['name']:20s} {w:5.1f} x {h:5.1f} = {area:6.1f} sqft  AR={ar:.2f}")
    print(f"Coverage: {covered:.0f}/{total_area} = {covered/total_area*100:.1f}%")
    
    score_info = result.get('score', {})
    print(f"Score: {score_info.get('total', 'N/A')}/100")
    validation = result.get('validation', {})
    print(f"Proportions OK: {validation.get('proportions_ok', 'N/A')}")
    filler = [r for r in rooms if 'filler' in r.get('name', '').lower() or 'wash' in r.get('name', '').lower()]
    print(f"Filler rooms: {len(filler)}")
    print()


if __name__ == '__main__':
    # pro_layout tests
    test_pro_layout("2BHK (30x20)", 
        [[0,0],[30,0],[30,20],[0,20]],
        {'living': 1, 'master_bedroom': 1, 'bedroom': 1, 'kitchen': 1, 'bathroom': 2, 'dining': 1},
        600)

    test_pro_layout("3BHK (40x25)",
        [[0,0],[40,0],[40,25],[0,25]],
        {'living': 1, 'master_bedroom': 1, 'bedroom': 2, 'kitchen': 1, 'bathroom': 2, 'dining': 1, 'study': 1, 'pooja': 1},
        1000)

    # perfect_layout tests
    test_perfect_layout("2BHK (30x20)", 30, 20, 2, 2, ['dining'])
    test_perfect_layout("3BHK (40x25)", 40, 25, 3, 2, ['dining', 'study', 'pooja'])

    print("All tests completed!")
