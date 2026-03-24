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


if __name__ == "__main__":
    unittest.main()
