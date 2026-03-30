import re

with open('layout_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

helper = '''
def _rnd(val: float, d: int = 2) -> float:
    return int(val * (10**d)) / (10.0**d)
'''

if '_rnd(' not in text:
    text = text.replace('import logging', 'import logging\n' + helper)

text = re.sub(r'\bround\(', '_rnd(', text)

text = text.replace('door_count += 1', 'door_count = door_count + 1')
text = text.replace('win_count += 1', 'win_count = win_count + 1')

with open('layout_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
