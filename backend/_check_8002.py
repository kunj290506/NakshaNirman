import json
import urllib.request

payload = {
  'plot_width': 30,
  'plot_length': 40,
  'total_area': 1200,
  'bedrooms': 2,
  'bathrooms': 2,
  'facing': 'east',
  'vastu': True,
  'extras': [],
  'rooms': [
    {'room_type':'master_bedroom','quantity':1},
    {'room_type':'bedroom','quantity':2},
    {'room_type':'living','quantity':1},
    {'room_type':'kitchen','quantity':1},
    {'room_type':'dining','quantity':1},
    {'room_type':'bathroom','quantity':3}
  ]
}
req = urllib.request.Request('http://localhost:8002/api/architect/design', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
out = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())
layout = out.get('layout', {})
print('bhk', layout.get('bhk'))
print('signature', layout.get('layout_signature'))
print('checks', layout.get('connectivity_checks'))
