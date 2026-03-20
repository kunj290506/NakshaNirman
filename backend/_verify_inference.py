import json
import urllib.request

cases = [
    [
        {'room_type':'master_bedroom','quantity':1},
        {'room_type':'living','quantity':1},
        {'room_type':'kitchen','quantity':1},
        {'room_type':'dining','quantity':1},
        {'room_type':'bathroom','quantity':2},
    ],
    [
        {'room_type':'master_bedroom','quantity':1},
        {'room_type':'bedroom','quantity':1},
        {'room_type':'living','quantity':1},
        {'room_type':'kitchen','quantity':1},
        {'room_type':'dining','quantity':1},
        {'room_type':'bathroom','quantity':2},
    ],
    [
        {'room_type':'master_bedroom','quantity':1},
        {'room_type':'bedroom','quantity':2},
        {'room_type':'living','quantity':1},
        {'room_type':'kitchen','quantity':1},
        {'room_type':'dining','quantity':1},
        {'room_type':'bathroom','quantity':3},
    ],
]
for idx, rooms in enumerate(cases, start=1):
    payload = {
        'plot_width': 30,
        'plot_length': 40,
        'total_area': 1200,
        'bedrooms': 2,
        'bathrooms': 2,
        'facing': 'east',
        'vastu': True,
        'extras': [],
        'rooms': rooms,
    }
    req = urllib.request.Request(
        'http://localhost:8001/api/architect/design',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    out = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
    layout = out.get('layout', {})
    print('CASE', idx, '| bhk', layout.get('bhk'), '| sig', layout.get('layout_signature'))
