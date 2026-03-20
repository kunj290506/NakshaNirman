import json
import urllib.request

payload = {
    'plot_width': 30,
    'plot_length': 40,
    'total_area': 1200,
    'bedrooms': 3,
    'bathrooms': 3,
    'facing': 'east',
    'vastu': True,
    'extras': []
}
req = urllib.request.Request(
    'http://localhost:8001/api/architect/design',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
out = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
layout = out.get('layout', {})
print('signature', layout.get('layout_signature'))
print('checks', layout.get('connectivity_checks'))
