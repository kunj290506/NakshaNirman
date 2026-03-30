import re

with open('layout_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('list[dict]', 'List[Dict[str, Any]]')
text = text.replace('(a: dict, b: dict)', '(a: Dict[str, Any], b: Dict[str, Any])')

with open('layout_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)
