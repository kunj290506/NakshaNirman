import requests, time

payload = {
    "plot_width": 30, "plot_length": 40,
    "bedrooms": 2, "facing": "east",
    "extras": [],
    "city": "pune", "family_type": "nuclear",
}

print(f"Sending 2BHK minimal request...")
start = time.time()
try:
    r = requests.post("http://localhost:8000/api/generate", json=payload, timeout=180)
    elapsed = time.time() - start
    print(f"Status: {r.status_code} | Time: {elapsed:.1f}s")

    if r.status_code == 200:
        plan = r.json()
        print(f"Engine: {plan.get('generation_method', '?')}")
        print(f"Rooms: {len(plan['rooms'])}")
        print(f"Vastu: {plan.get('vastu_score', '?')}")
        print(f"\nRoom layout:")
        for room in plan["rooms"]:
            print(f"  {room['label']:20s} x={room['x']:6.1f} y={room['y']:6.1f} w={room['width']:5.1f} h={room['height']:5.1f} area={room['area']:6.1f}")
    else:
        print(f"Error: {r.text[:500]}")
except requests.Timeout:
    print(f"TIMEOUT after 180s")
except Exception as e:
    print(f"Error: {e}")
