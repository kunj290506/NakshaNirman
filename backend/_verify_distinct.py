import json
import urllib.request

for b in [1,2,3,4]:
    payload = {
        'plot_width': 30,
        'plot_length': 40,
        'total_area': 1200,
        'bedrooms': b,
        'bathrooms': max(2,b),
        'facing': 'east',
        'vastu': True,
        'extras': [],
        'rooms': [
            {'room_type':'master_bedroom','quantity':1},
            {'room_type':'bedroom','quantity': max(0,b-1)},
            {'room_type':'living','quantity':1},
            {'room_type':'kitchen','quantity':1},
            {'room_type':'dining','quantity':1},
            {'room_type':'bathroom','quantity':max(2,b)}
        ]
    }
    req = urllib.request.Request('http://localhost:8002/api/architect/design', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    out = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    layout = out.get('layout', {})
    rooms = layout.get('rooms', [])
    tcount = {}
    for r in rooms:
        tcount[r.get('type')] = tcount.get(r.get('type'), 0) + 1
    print('BHK', b, '| signature', layout.get('layout_signature'), '| counts', tcount)
