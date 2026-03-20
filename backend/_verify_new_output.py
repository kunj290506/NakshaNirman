import json
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
    layout = out.get('layout', {})
    checks = layout.get('connectivity_checks', {})
    print('BHK', b, '| signature', layout.get('layout_signature'))
    print(' checks', checks)
