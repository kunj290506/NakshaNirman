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


if __name__ == "__main__":
    unittest.main()
