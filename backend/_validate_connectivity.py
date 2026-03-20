import json
import urllib.request


def adj(a, b):
    if not a or not b:
        return False
    ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['width'], a['y'] + a['height']
    bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']
    horizontal_touch = abs(ax2 - bx1) < 0.05 or abs(bx2 - ax1) < 0.05
    vertical_overlap = min(ay2, by2) - max(ay1, by1) > 0.3
    vertical_touch = abs(ay2 - by1) < 0.05 or abs(by2 - ay1) < 0.05
    horizontal_overlap = min(ax2, bx2) - max(ax1, bx1) > 0.3
    return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap)


def path(l, t, rooms):
    if not l or not t:
        return False
    by = {r['id']: r for r in rooms}
    q = [l['id']]
    seen = {l['id']}
    while q:
        cur = by[q.pop(0)]
        if cur['id'] == t['id']:
            return True
        for r in rooms:
            if r['id'] in seen:
                continue
            if adj(cur, r):
                seen.add(r['id'])
                q.append(r['id'])
    return False


def fetch(b):
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
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode())


for b in [1, 2, 3, 4]:
    out = fetch(b)
    rooms = out.get('layout', {}).get('rooms', [])
    living = next((r for r in rooms if r.get('type') == 'living'), None)
    kitchen = next((r for r in rooms if r.get('type') == 'kitchen'), None)
    dining = next((r for r in rooms if r.get('type') == 'dining'), None)
    master = next((r for r in rooms if r.get('type') == 'master_bedroom'), None)
    beds = [r for r in rooms if r.get('type') == 'bedroom']
    wet = [r for r in rooms if r.get('type') in {'bathroom', 'toilet'}]

    print(
        'BHK', b,
        '| L->K', path(living, kitchen, rooms),
        '| K-D', adj(kitchen, dining),
        '| L->M', path(living, master, rooms),
        '| M-Wet', any(adj(master, w) for w in wet),
        '| L->Bed', (True if not beds else any(path(living, x, rooms) for x in beds))
    )
