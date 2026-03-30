import unittest

from fastapi.testclient import TestClient

from main import app


class TestArchitectAPI(unittest.TestCase):
    def test_health_endpoint(self):
        with TestClient(app) as client:
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            body = res.json()
            self.assertEqual(body.get("status"), "ok")

    def test_architect_design_minimal_payload(self):
        payload = {
            "plot_width": 30,
            "plot_length": 40,
            "bedrooms": 2,
            "bathrooms": 2,
            "facing": "east",
            "vastu": True,
            "family_type": "nuclear",
            "city": "Pune",
            "state": "Maharashtra",
            "extras": ["study"],
            "engine_mode": "gnn_advanced",
        }

        with TestClient(app) as client:
            res = client.post("/api/architect/design", json=payload)
            self.assertEqual(res.status_code, 200)
            body = res.json()

        self.assertIn("layout", body)
        self.assertIsInstance(body["layout"], dict)
        self.assertIn("design_score", body)
        self.assertIsInstance(body["design_score"], dict)

        layout = body["layout"]
        self.assertIn("rooms", layout)
        self.assertGreaterEqual(len(layout["rooms"]), 4)
        self.assertIn("attempt_scores", layout)
        self.assertGreaterEqual(len(layout["attempt_scores"]), 1)

    def test_architect_design_rejects_invalid_plot(self):
        payload = {
            "plot_width": 10,
            "plot_length": 40,
            "bedrooms": 2,
            "bathrooms": 2,
        }

        with TestClient(app) as client:
            res = client.post("/api/architect/design", json=payload)

        self.assertEqual(res.status_code, 400)

    def test_chat_payload_constraints_survive_design(self):
        chat_message = "30x40 3bhk kitchen near back garden and more privacy for master"

        with TestClient(app) as client:
            chat_res = client.post(
                "/api/architect/chat",
                json={"message": chat_message, "history": []},
            )
            self.assertEqual(chat_res.status_code, 200)
            chat_body = chat_res.json()

            payload = chat_body.get("generate_payload")
            if payload is None:
                # Chat flow may ask for confirmation in strict mode; use extracted defaults.
                payload = {
                    "plot_width": 30,
                    "plot_length": 40,
                    "bedrooms": 3,
                    "bathrooms": 3,
                    "facing": "east",
                    "vastu": True,
                    "extras": [],
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

            design_res = client.post("/api/architect/design", json=payload)
            self.assertEqual(design_res.status_code, 200)
            design_body = design_res.json()

        layout = design_body.get("layout") or {}
        notes = layout.get("constraint_notes") or []
        self.assertTrue(any("kitchen" in n.lower() for n in notes))


if __name__ == "__main__":
    unittest.main()
