"""
Architect's Audit — Tests layouts like a 45-year experienced architect would review them.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def audit_rooms(rooms, plot_area, label):
    print(f"\n{'='*70}")
    print(f"AUDIT: {label}  (Plot={plot_area} sqft)")
    print(f"{'='*70}")
    
    total = 0
    issues = []
    for r in rooms:
        if 'width' in r:  # perfect_layout format
            w, h = r['width'], r['length']
            name = r['name']
            rtype = r['room_type']
        elif '_placed' in r:  # pro_layout format
            p = r['_placed']
            w, h = p['w'], p['h']
            name = r['name']
            rtype = r['room_type']
        else:
            continue
        
        area = w * h
        total += area
        ar = max(w,h)/min(w,h) if min(w,h) > 0 else 99
        pct = area/plot_area*100
        
        flag = ""
        # --- Architect's critique ---
        # Scale thresholds for compact plots (< 500sqft can't meet
        # standard minimums without sacrificing other rooms)
        mbr_min = 100 if plot_area >= 500 else max(70, plot_area * 0.18)
        br_min = 80 if plot_area >= 500 else max(55, plot_area * 0.15)
        living_max_pct = 20 if plot_area >= 500 else 30  # 1BHK needs proportionally larger living
        
        if rtype in ('master_bedroom',) and area < mbr_min:
            flag = f" ⚠ MASTER BR TOO SMALL (min {mbr_min:.0f}sqft)"
            issues.append(flag)
        elif rtype == 'bedroom' and area < br_min:
            flag = f" ⚠ BEDROOM TOO SMALL (min {br_min:.0f}sqft)"
            issues.append(flag)
        elif rtype == 'living' and pct > living_max_pct:
            flag = f" ⚠ LIVING TOO LARGE (>{living_max_pct}% of plot)"
            issues.append(flag)
        elif rtype == 'corridor' and pct > 6:
            flag = " ⚠ CORRIDOR WASTES SPACE (>6% of plot)"
            issues.append(flag)
        elif ar > 2.5 and rtype not in ('corridor', 'balcony'):
            flag = " ⚠ BAD ASPECT RATIO"
            issues.append(flag)
        
        print(f"  {name:20s}  {w:5.1f}x{h:5.1f} = {area:6.1f}sqft ({pct:4.1f}%)  AR={ar:.2f}{flag}")
    
    placed_pct = total / plot_area * 100
    print(f"\n  Total placed: {total:.0f}/{plot_area} sqft ({placed_pct:.1f}%)")
    
    # Check bedroom vs living ratio
    living_area = sum(
        (r.get('_placed',r).get('w',r.get('width',0)) * r.get('_placed',r).get('h',r.get('length',0)))
        if '_placed' in r else (r.get('width',0)*r.get('length',0))
        for r in rooms if r.get('room_type') == 'living'
    )
    bed_areas = []
    for r in rooms:
        if r.get('room_type') in ('master_bedroom', 'bedroom'):
            if '_placed' in r:
                p = r['_placed']
                bed_areas.append(p['w'] * p['h'])
            else:
                bed_areas.append(r.get('width',0) * r.get('length',0))
    
    if bed_areas and living_area > 0:
        avg_bed = sum(bed_areas) / len(bed_areas)
        ratio = living_area / avg_bed
        if ratio > 2.0:
            print(f"  ⚠ LIVING/BEDROOM IMBALANCE: Living={living_area:.0f}, Avg Bed={avg_bed:.0f}, ratio={ratio:.1f}x")
            issues.append("Living/bedroom size imbalance")
        else:
            print(f"  ✓ Living/Bedroom balance: Living={living_area:.0f}, AvgBed={avg_bed:.0f}, ratio={ratio:.1f}x")
    
    corridor_area = sum(
        (r['_placed']['w'] * r['_placed']['h']) if '_placed' in r else (r.get('width',0)*r.get('length',0))
        for r in rooms if r.get('room_type') == 'corridor'
    )
    if corridor_area > 0:
        corr_pct = corridor_area / plot_area * 100
        if corr_pct > 6:
            print(f"  ⚠ CORRIDOR waste: {corridor_area:.0f}sqft ({corr_pct:.1f}%) — max 6% recommended")
        else:
            print(f"  ✓ Corridor efficient: {corridor_area:.0f}sqft ({corr_pct:.1f}%)")
    
    print(f"\n  Issues found: {len(issues)}")
    return len(issues)


# ============ PRO LAYOUT ENGINE ============
from services.pro_layout_engine import generate_professional_plan

print("\n" + "█"*70)
print("  PRO LAYOUT ENGINE AUDIT")
print("█"*70)

# 2BHK 30x20
b2 = [[0,0],[30,0],[30,20],[0,20]]
r2 = {'living':1,'kitchen':1,'dining':1,'master_bedroom':1,'bedroom':1,'bathroom':2}
_, _, rooms2 = generate_professional_plan(b2, r2, 600)
i1 = audit_rooms(rooms2, 600, "PRO 2BHK 30×20")

# 3BHK 40x25
b3 = [[0,0],[40,0],[40,25],[0,25]]
r3 = {'living':1,'kitchen':1,'dining':1,'master_bedroom':1,'bedroom':2,'bathroom':2,'study':1,'pooja':1}
_, _, rooms3 = generate_professional_plan(b3, r3, 1000)
i2 = audit_rooms(rooms3, 1000, "PRO 3BHK 40×25")

# 1BHK 20x15
b1 = [[0,0],[20,0],[20,15],[0,15]]
r1 = {'living':1,'kitchen':1,'master_bedroom':1,'bathroom':1}
_, _, rooms1 = generate_professional_plan(b1, r1, 300)
i3 = audit_rooms(rooms1, 300, "PRO 1BHK 20×15")

# 2BHK compact 25x18
bc = [[0,0],[25,0],[25,18],[0,18]]
rc = {'living':1,'kitchen':1,'master_bedroom':1,'bedroom':1,'bathroom':2}
_, _, roomsc = generate_professional_plan(bc, rc, 450)
i4 = audit_rooms(roomsc, 450, "PRO 2BHK 25×18 (compact, no dining)")

# ============ PERFECT LAYOUT ENGINE ============
from services.perfect_layout import generate_perfect_layout

print("\n" + "█"*70)
print("  PERFECT LAYOUT ENGINE AUDIT")
print("█"*70)

p2 = generate_perfect_layout(30, 20, bedrooms=2, bathrooms=2, extras=['dining'])
i5 = audit_rooms(p2['rooms'], 600, "PERFECT 2BHK 30×20")

p3 = generate_perfect_layout(40, 25, bedrooms=3, bathrooms=2, extras=['dining','study','pooja'])
i6 = audit_rooms(p3['rooms'], 1000, "PERFECT 3BHK 40×25")

p1 = generate_perfect_layout(20, 15, bedrooms=1, bathrooms=1)
i7 = audit_rooms(p1['rooms'], 300, "PERFECT 1BHK 20×15")

total_issues = i1+i2+i3+i4+i5+i6+i7
print(f"\n{'='*70}")
print(f"TOTAL ISSUES ACROSS ALL TESTS: {total_issues}")
print(f"{'='*70}")
