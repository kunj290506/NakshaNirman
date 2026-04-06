import asyncio
import json
from models import PlanRequest
from layout_engine import _generate_via_llm, _rnd, SETBACKS

req = PlanRequest(
    plot_width=40,
    plot_length=40,
    bedrooms=2,
    facing='east',
    extras=[],
    bathrooms_target=2,
    floors=1,
    design_style='modern',
    kitchen_preference='semi_open',
    parking_slots=0,
    vastu_priority=3,
    natural_light_priority=3,
    privacy_priority=3,
    storage_priority=3,
    elder_friendly=False,
    work_from_home=False,
    notes='',
    city='',
    state='',
    family_type='nuclear',
    family_notes='',
)
uw = _rnd(req.plot_width - SETBACKS['left'] - SETBACKS['right'], 2)
ul = _rnd(req.plot_length - SETBACKS['front'] - SETBACKS['rear'], 2)

async def run_test():
    plan, issues, correction = await _generate_via_llm(req, uw, ul)
    print(json.dumps({
        'plan_ok': plan is not None,
        'issues': (issues or [])[:6],
        'generation_method': getattr(plan, 'generation_method', None),
        'adjacency_score': getattr(plan, 'adjacency_score', None),
        'room_count': (len(getattr(plan, 'rooms', []) or []) if plan is not None else 0),
    }, separators=(',', ':')))

asyncio.run(run_test())
