"""Strict architectural compliance validation."""
import json
from services.perfect_layout import generate_perfect_layout

result = generate_perfect_layout(
    plot_width=40.0, plot_length=30.0,
    bedrooms=2, bathrooms=1, floors=1, extras=[],
    strict_mode=True, total_area=1200,
)

rooms = result["rooms"]
total = 1200.0

# Area percentage ranges required
RANGES = {
    "Drawing Room":      (15, 20),
    "Kitchen":           (10, 15),
    "Dining Area":       (8, 12),
    "Master Bedroom":    (18, 22),
    "Bedroom 1":         (15, 18),
    "Attached Bathroom": (4, 6),
    "Wash Area":         (4, 6),
    "Passage":           (None, None),  # remainder
}

print(f"Plot: {result['plot']['width']}x{result['plot']['length']} = {total} sqft\n")

all_ok = True
for r in rooms:
    name = r["name"]
    area = r["area"]
    pct = area / total * 100
    lo, hi = RANGES.get(name, (None, None))
    if lo is not None and hi is not None:
        in_range = lo <= pct <= hi
    else:
        in_range = True  # Passage = remainder
    status = "OK" if in_range else "OUT"
    if not in_range:
        all_ok = False
    rng = f"{lo}-{hi}%" if lo else "remainder"
    pos = r["position"]
    print(f"  [{status:3s}] {name:20s} area={area:5.0f} ({pct:5.1f}%)  need={rng:10s}  pos=({pos['x']:.1f},{pos['y']:.1f}) {r['width']:.1f}x{r['length']:.1f}")

print(f"\nTotal area used: {sum(r['area'] for r in rooms):.0f}")
print(f"Room count: {len(rooms)}")
print(f"Area ranges: {'ALL IN RANGE' if all_ok else 'SOME OUT OF RANGE'}")
