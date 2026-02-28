"""Quick API endpoint test for strict mode."""
import requests

r = requests.post('http://localhost:8000/api/perfect/design', json={
    'plot_width': 40, 'plot_length': 30,
    'bedrooms': 2, 'bathrooms': 1, 'floors': 1,
    'extras': [], 'strict_mode': True,
})
print(f"Status: {r.status_code}")
d = r.json()
rooms = d['layout']['rooms']
print(f"Score: {d.get('score')}")
print(f"Rooms: {len(rooms)}")
total = 1200.0
for rm in rooms:
    pct = rm['area'] / total * 100
    print(f"  {rm['name']:20s}  {rm['area']:6.0f} sqft  ({pct:5.1f}%)")
print(f"Total area: {sum(rm['area'] for rm in rooms):.0f}")
