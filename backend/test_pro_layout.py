"""Test the professional layout engine - isolated and full pipeline."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__) or '.')

from services.pro_layout_engine import generate_professional_plan


def show(label, boundary, config, area):
    print('=' * 60)
    print(label)
    centroids, sizes, specs = generate_professional_plan(boundary, config, area)
    print(f'Rooms placed: {len(specs)}')
    for s in specs:
        p = s.get('_placed', {})
        w = p.get('w', 0)
        h = p.get('h', 0)
        a = w * h
        ar = max(w/h, h/w) if w > 0 and h > 0 else 0
        print(f"  {s['name']:20s} ({s['room_type']:16s}) = "
              f"{w:5.1f} x {h:5.1f} = {a:6.1f} sqft  AR={ar:.2f}  "
              f"@ ({p.get('x',0):.1f}, {p.get('y',0):.1f})")
    print()


def test_pipeline(label, **kwargs):
    from services.gnn_engine import generate_gnn_floor_plan
    print('=' * 60)
    print(f'PIPELINE: {label}')
    try:
        layout = generate_gnn_floor_plan(**kwargs)
        rooms = layout.get('rooms', [])
        val = layout.get('validation', {})
        print(f"  Rooms: {len(rooms)}  Validation: {val.get('overall','?')}")
        total_a = 0
        for r in rooms:
            pos = r.get('position', {})
            w, h = r.get('width', 0), r.get('length', 0)
            a = w * h
            total_a += a
            ar = max(w/h, h/w) if w > 0 and h > 0 else 0
            print(f"    {r.get('name','?'):20s} = {w:5.1f} x {h:5.1f}"
                  f" = {a:6.1f} sqft  AR={ar:.2f}"
                  f"  @ ({pos.get('x',0):.1f}, {pos.get('y',0):.1f})")
        print(f"  Total room area: {total_a:.0f} sqft")

        doors = layout.get('doors', [])
        print(f"  Doors ({len(doors)}):")
        for d in doors[:8]:
            pos = d.get('position', [0,0])
            hinge = d.get('hinge', [0,0])
            dend = d.get('door_end', [0,0])
            print(f"    pos=({pos[0]:.1f},{pos[1]:.1f})"
                  f"  hinge→end=({hinge[0]:.1f},{hinge[1]:.1f})→"
                  f"({dend[0]:.1f},{dend[1]:.1f})  w={d.get('width',0)}")

        windows = layout.get('windows', [])
        print(f"  Windows ({len(windows)}):")
        for w in windows[:8]:
            s = w.get('start', [0,0])
            e = w.get('end', [0,0])
            print(f"    {w.get('room','?'):15s} wall={w.get('wall','?')}"
                  f"  ({s[0]:.1f},{s[1]:.1f})→({e[0]:.1f},{e[1]:.1f})"
                  f"  w={w.get('width',0)}")

        expl = layout.get('explanation', '')
        print(f"  Explanation: {expl[:300]}...")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
    print()


if __name__ == '__main__':
    # ---- Part 1: Isolated engine ----
    print('\n*** PART 1: ISOLATED ENGINE ***\n')

    show('TEST 1: 2BHK 30x40',
         [(0,0),(30,0),(30,40),(0,40),(0,0)],
         {'master_bedroom':1, 'bedroom':1, 'kitchen':1, 'dining':1, 'bathroom':2},
         1200)

    show('TEST 2: 1BHK 20x25',
         [(0,0),(20,0),(20,25),(0,25),(0,0)],
         {'master_bedroom':1, 'kitchen':1, 'bathroom':1},
         500)

    show('TEST 3: 3BHK 40x50',
         [(0,0),(40,0),(40,50),(0,50),(0,0)],
         {'master_bedroom':1, 'bedroom':2, 'kitchen':1, 'dining':1,
          'bathroom':3, 'study':1, 'pooja':1},
         2000)

    show('TEST 4: 2BHK Narrow 15x45',
         [(0,0),(15,0),(15,45),(0,45),(0,0)],
         {'master_bedroom':1, 'bedroom':1, 'kitchen':1, 'dining':1, 'bathroom':2},
         675)

    # ---- Part 2: Full pipeline ----
    print('\n*** PART 2: FULL GNN PIPELINE ***\n')

    test_pipeline('2BHK 30x40',
                  boundary_coords=[(0,0),(30,0),(30,40),(0,40),(0,0)],
                  total_area=1200, bedrooms=2, bathrooms=1,
                  kitchens=1, extras=['dining'], master_bedrooms=1)

    test_pipeline('3BHK 40x50',
                  plot_width=40, plot_length=50,
                  total_area=2000, bedrooms=3, bathrooms=1,
                  kitchens=1, extras=['dining','study','pooja'],
                  master_bedrooms=1)

    test_pipeline('1BHK 20x25',
                  plot_width=20, plot_length=25,
                  total_area=500, bedrooms=1, bathrooms=0,
                  kitchens=1, master_bedrooms=1)
