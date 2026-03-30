import unittest

from services.multi_agent_orchestrator import generate_dynamic_layout


class TestMultiAgentOrchestratorScenarios(unittest.TestCase):
    def test_narrow_plot_rejected(self):
        payload = {
            "plot_width": 15,
            "plot_length": 40,
            "bedrooms": 2,
            "bathrooms": 2,
            "facing": "east",
            "vastu": True,
            "extras": [],
        }
        result = generate_dynamic_layout(payload)
        self.assertIn("error", result)
        self.assertIn("narrow", result["error"].lower())

    def test_high_bhk_density_generates_layout(self):
        payload = {
            "plot_width": 30,
            "plot_length": 40,
            "bedrooms": 4,
            "bathrooms": 4,
            "facing": "north",
            "family_type": "joint-family",
            "city": "Pune",
            "state": "Maharashtra",
            "vastu": True,
            "extras": ["study", "store"],
        }
        result = generate_dynamic_layout(payload)

        self.assertNotIn("error", result)
        self.assertIn("rooms", result)
        self.assertGreaterEqual(len(result["rooms"]), 8)
        self.assertIn("design_score", result)
        self.assertIn("composite", result["design_score"])

        bedroom_like = [r for r in result["rooms"] if r["type"] in {"master_bedroom", "bedroom"}]
        self.assertGreaterEqual(len(bedroom_like), 4)

    def test_elderly_family_prefers_shallower_bedroom_band(self):
        payload = {
            "plot_width": 34,
            "plot_length": 42,
            "bedrooms": 2,
            "bathrooms": 2,
            "facing": "east",
            "family_type": "elderly",
            "city": "Bangalore",
            "state": "Karnataka",
            "vastu": True,
            "extras": [],
        }
        result = generate_dynamic_layout(payload)

        self.assertNotIn("error", result)
        self.assertIn("room_program", result)
        self.assertEqual(result["room_program"]["circulation_type"], "linear")

        bedroom_program = [
            r
            for r in result["room_program"].get("rooms", [])
            if r.get("type") in {"master_bedroom", "bedroom"}
        ]
        self.assertTrue(any(r.get("band") == 2 for r in bedroom_program))

    def test_vastu_toggle_changes_scoring_behavior(self):
        strict_payload = {
            "plot_width": 30,
            "plot_length": 45,
            "bedrooms": 3,
            "bathrooms": 3,
            "facing": "east",
            "city": "Chennai",
            "state": "Tamil Nadu",
            "family_type": "nuclear",
            "vastu": True,
            "extras": ["pooja"],
        }
        relaxed_payload = dict(strict_payload)
        relaxed_payload["vastu"] = False

        strict = generate_dynamic_layout(strict_payload)
        relaxed = generate_dynamic_layout(relaxed_payload)

        self.assertNotIn("error", strict)
        self.assertNotIn("error", relaxed)

        self.assertIn("design_score", strict)
        self.assertIn("design_score", relaxed)
        self.assertIn("breakdown", strict["design_score"])
        self.assertIn("breakdown", relaxed["design_score"])

        strict_vastu_score = strict["design_score"]["breakdown"]["vastu"]["score"]
        relaxed_vastu_score = relaxed["design_score"]["breakdown"]["vastu"]["score"]
        self.assertNotEqual(strict_vastu_score, relaxed_vastu_score)

    def test_retry_loop_exposes_failure_history(self):
        payload = {
            "plot_width": 30,
            "plot_length": 40,
            "bedrooms": 3,
            "bathrooms": 2,
            "facing": "east",
            "vastu": True,
            "extras": ["study"],
            "placement_constraints": [
                {
                    "room": "kitchen",
                    "intent": "rear_garden_preference",
                    "band": 3,
                }
            ],
        }
        result = generate_dynamic_layout(payload)

        self.assertNotIn("error", result)
        self.assertIn("retry_count", result)
        self.assertIn("failure_history", result)

        retry_count = int(result.get("retry_count") or 0)
        failure_history = result.get("failure_history") or []
        self.assertEqual(len(failure_history), retry_count)
        for item in failure_history:
            self.assertIn("reason", item)
            self.assertIn("failed_checks", item)
            self.assertIn("failed_paths", item)

    def test_placement_constraints_survive_to_layout(self):
        payload = {
            "plot_width": 32,
            "plot_length": 42,
            "bedrooms": 3,
            "bathrooms": 2,
            "facing": "north",
            "family_type": "nuclear",
            "vastu": True,
            "placement_constraints": [
                {
                    "room": "kitchen",
                    "intent": "rear_garden_preference",
                    "band": 3,
                    "prefer_walls": ["north", "rear"],
                },
                {
                    "room": "master_bedroom",
                    "intent": "privacy_buffer",
                    "forbid_adjacent": ["living"],
                },
            ],
        }
        result = generate_dynamic_layout(payload)

        self.assertNotIn("error", result)
        notes = result.get("constraint_notes") or []
        self.assertTrue(any("kitchen" in n.lower() for n in notes))
        self.assertTrue(any("master" in n.lower() for n in notes))

        kitchen = next((r for r in result.get("rooms", []) if r.get("type") == "kitchen"), None)
        self.assertIsNotNone(kitchen)
        self.assertEqual(kitchen.get("band"), 3)

    def test_kitchen_preferences_change_with_conditions(self):
        base = {
            "plot_width": 30,
            "plot_length": 40,
            "bedrooms": 3,
            "bathrooms": 2,
            "extras": [],
        }

        vastu_case = dict(base)
        vastu_case.update({"facing": "east", "vastu": True, "family_type": "nuclear"})
        relaxed_case = dict(base)
        relaxed_case.update({"facing": "north", "vastu": False, "family_type": "joint-family"})

        a = generate_dynamic_layout(vastu_case)
        b = generate_dynamic_layout(relaxed_case)

        self.assertNotIn("error", a)
        self.assertNotIn("error", b)

        ka = next((r for r in a.get("room_program", {}).get("rooms", []) if r.get("type") == "kitchen"), None)
        kb = next((r for r in b.get("room_program", {}).get("rooms", []) if r.get("type") == "kitchen"), None)
        self.assertIsNotNone(ka)
        self.assertIsNotNone(kb)

        # With changed facing/vastu/family, kitchen intent should not be identical every time.
        self.assertTrue(
            ka.get("band") != kb.get("band") or ka.get("vastu_zone") != kb.get("vastu_zone")
        )

    def test_multiple_room_types_change_with_real_life_conditions(self):
        scenario_a = {
            "plot_width": 34,
            "plot_length": 45,
            "bedrooms": 3,
            "bathrooms": 3,
            "facing": "east",
            "family_type": "joint-family",
            "city": "Mumbai",
            "state": "Maharashtra",
            "vastu": True,
            "extras": ["pooja", "study", "store", "balcony"],
        }
        scenario_b = {
            "plot_width": 34,
            "plot_length": 45,
            "bedrooms": 3,
            "bathrooms": 3,
            "facing": "north",
            "family_type": "working-couple",
            "city": "Delhi",
            "state": "Delhi",
            "vastu": False,
            "extras": ["pooja", "study", "store", "balcony"],
        }

        out_a = generate_dynamic_layout(scenario_a)
        out_b = generate_dynamic_layout(scenario_b)
        self.assertNotIn("error", out_a)
        self.assertNotIn("error", out_b)

        prog_a = out_a.get("room_program", {}).get("rooms", [])
        prog_b = out_b.get("room_program", {}).get("rooms", [])

        def pick(program, rtype):
            return next((r for r in program if r.get("type") == rtype), None)

        dynamic_types = ["dining", "kitchen", "master_bedroom", "study", "store", "balcony"]
        changed = 0
        for rtype in dynamic_types:
            ra = pick(prog_a, rtype)
            rb = pick(prog_b, rtype)
            if not ra or not rb:
                continue
            if ra.get("band") != rb.get("band") or ra.get("vastu_zone") != rb.get("vastu_zone"):
                changed += 1

        self.assertGreaterEqual(changed, 3)

    def test_facing_changes_public_private_depth_direction(self):
        south_case = {
            "plot_width": 30,
            "plot_length": 40,
            "bedrooms": 2,
            "bathrooms": 2,
            "facing": "south",
            "family_type": "nuclear",
            "vastu": True,
            "extras": [],
        }
        north_case = dict(south_case)
        north_case["facing"] = "north"

        out_s = generate_dynamic_layout(south_case)
        out_n = generate_dynamic_layout(north_case)
        self.assertNotIn("error", out_s)
        self.assertNotIn("error", out_n)

        living_s = next((r for r in out_s.get("rooms", []) if r.get("type") == "living"), None)
        master_s = next((r for r in out_s.get("rooms", []) if r.get("type") == "master_bedroom"), None)
        living_n = next((r for r in out_n.get("rooms", []) if r.get("type") == "living"), None)
        master_n = next((r for r in out_n.get("rooms", []) if r.get("type") == "master_bedroom"), None)
        self.assertIsNotNone(living_s)
        self.assertIsNotNone(master_s)
        self.assertIsNotNone(living_n)
        self.assertIsNotNone(master_n)

        # South-facing: public near y=0, private farther north.
        self.assertLessEqual(living_s["y"], master_s["y"])

        # North-facing: public near top edge, private farther south.
        living_n_center = living_n["y"] + living_n["height"] / 2.0
        master_n_center = master_n["y"] + master_n["height"] / 2.0
        self.assertGreaterEqual(living_n_center, master_n_center)

    def test_hot_dry_prefers_non_west_windows_when_possible(self):
        payload = {
            "plot_width": 34,
            "plot_length": 44,
            "bedrooms": 3,
            "bathrooms": 2,
            "facing": "east",
            "city": "Delhi",
            "state": "Delhi",
            "family_type": "nuclear",
            "vastu": False,
            "extras": ["study"],
        }
        out = generate_dynamic_layout(payload)
        self.assertNotIn("error", out)

        by_room = {r.get("id"): r for r in out.get("rooms", [])}
        habitable = {"living", "master_bedroom", "bedroom", "kitchen", "dining", "study"}
        west_only = 0
        for rid, room in by_room.items():
            if room.get("type") not in habitable:
                continue
            walls = [w.get("wall") for w in out.get("windows", []) if w.get("room_id") == rid]
            walls = [str(w).lower() for w in walls if w]
            if walls and all(w == "west" for w in walls):
                west_only += 1

        self.assertLessEqual(west_only, 1)


if __name__ == "__main__":
    unittest.main()
