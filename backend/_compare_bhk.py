import json
import hashlib
import urllib.request

for b in [1, 2, 3, 4]:
    payload = {
        'plot_width': 30,
        'plot_length': 40,
        'total_area': 1200,
        'bedrooms': b,
        'bathrooms': max(2, b),
        'facing': 'east',
        'vastu': True,
        'extras': []
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        'http://localhost:8000/api/architect/design',
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    out = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    rooms = out.get('layout', {}).get('rooms', [])
    sig = ';'.join(sorted([
        f"{r.get('type')}@{r.get('x')},{r.get('y')}:{r.get('width')}x{r.get('height')}"
        for r in rooms
    ]))
    sig_hash = hashlib.md5(sig.encode()).hexdigest()[:10]
    print('BHK', b, '| rooms', len(rooms), '| signature', sig_hash)
    print('types', sorted([r.get('type') for r in rooms]))
