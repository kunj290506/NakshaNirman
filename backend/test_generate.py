"""Quick test of the /api/generate endpoint."""
import httpx
import json
import time

print("Testing floor plan generation via LLM...")
print("GTX 1650 may take 2-4 minutes on first plan (cold start).\n")

start = time.time()

r = httpx.post(
    "http://localhost:8010/api/generate",
    json={
        "plot_width": 30,
        "plot_length": 40,
        "bedrooms": 2,
        "facing": "east",
        "extras": [],
        "family_type": "nuclear",
    },
    timeout=800,  # 800s to cover primary + backup + BSP
)

elapsed = time.time() - start
d = r.json()

print(f"Status:  {r.status_code}")
print(f"Time:    {elapsed:.1f}s")
print(f"Method:  {d.get('generation_method')}")
print(f"Rooms:   {len(d.get('rooms', []))}")
print(f"Vastu:   {d.get('vastu_score')}")
print(f"Adj:     {d.get('adjacency_score')}")
print()

print("Room Layout:")
for rm in d.get("rooms", []):
    print(
        f"  {rm.get('id', '?'):20s} "
        f"{rm.get('type', '?'):18s} "
        f"{rm.get('width')}x{rm.get('height')} "
        f"at ({rm.get('x')}, {rm.get('y')})"
    )

print()
print("Reasoning Trace:")
for t in d.get("reasoning_trace", []):
    print(f"  {t}")
