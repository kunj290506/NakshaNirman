from __future__ import annotations

import unittest

from layout_engine import build_room_list
from models import PlanRequest


class BackendSmokeTests(unittest.TestCase):
    def test_plan_request_defaults(self) -> None:
        req = PlanRequest(
            plot_width=30,
            plot_length=40,
            bedrooms=2,
            facing="east",
        )
        self.assertEqual(req.bedrooms, 2)
        self.assertEqual(req.floors, 1)
        self.assertEqual(req.design_style, "modern")

    def test_build_room_list_contains_core_spaces(self) -> None:
        rooms = build_room_list(
            bedrooms=2,
            extras=["pooja", "study"],
            usable_w=23.0,
            usable_l=28.5,
            facing="east",
            floors=1,
        )
        self.assertGreaterEqual(len(rooms), 4)
        room_types = {str(item.get("type", "")).strip().lower() for item in rooms}
        for required in ("living", "kitchen", "master_bedroom"):
            self.assertIn(required, room_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
