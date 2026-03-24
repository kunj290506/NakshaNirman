import random
import statistics
import unittest

from services.multi_agent_orchestrator import generate_dynamic_layout


def _base_payload(**kwargs):
    payload = {
        "plot_width": 30,
        "plot_length": 40,
        "bedrooms": 2,
        "bathrooms": 2,
        "facing": "east",
        "family_type": "nuclear",
        "city": "Pune",
        "state": "Maharashtra",
        "vastu": True,
        "extras": [],
    }
    payload.update(kwargs)
    return payload


class TestIITHouseBenchmark(unittest.TestCase):
    """IIT-style internal benchmark gates for house-plan quality and robustness."""

    def test_iit_scenario_pack_quality_gates(self):
        scenarios = [
            _base_payload(plot_width=20, plot_length=35, bedrooms=1, bathrooms=1, city="Chennai", state="Tamil Nadu"),
            _base_payload(plot_width=24, plot_length=36, bedrooms=2, bathrooms=2, family_type="working-couple", extras=["study"]),
            _base_payload(plot_width=28, plot_length=45, bedrooms=3, bathrooms=3, facing="north", city="Delhi", state="Delhi"),
            _base_payload(plot_width=32, plot_length=48, bedrooms=4, bathrooms=4, family_type="joint-family", extras=["store", "study"]),
            _base_payload(plot_width=26, plot_length=38, bedrooms=2, bathrooms=2, family_type="elderly", city="Bengaluru", state="Karnataka"),
            _base_payload(plot_width=30, plot_length=50, bedrooms=3, bathrooms=3, facing="west", extras=["pooja"]),
            _base_payload(plot_width=34, plot_length=52, bedrooms=4, bathrooms=4, extras=["pooja", "study", "store"]),
            _base_payload(plot_width=22, plot_length=34, bedrooms=2, bathrooms=2, city="Mumbai", state="Maharashtra", vastu=False),
            _base_payload(plot_width=27, plot_length=42, bedrooms=3, bathrooms=2, city="Jaipur", state="Rajasthan"),
            _base_payload(plot_width=25, plot_length=40, bedrooms=2, bathrooms=2, city="Kochi", state="Kerala", extras=["balcony"]),
            _base_payload(plot_width=21, plot_length=32, bedrooms=2, bathrooms=3, extras=["garage", "store"]),
            _base_payload(plot_width=29, plot_length=41, bedrooms=3, bathrooms=3, family_type="rental", vastu=False),
            _base_payload(plot_width=31, plot_length=43, bedrooms=3, bathrooms=3, city="Lucknow", state="Uttar Pradesh"),
            _base_payload(plot_width=35, plot_length=55, bedrooms=4, bathrooms=5, family_type="joint-family", extras=["study", "store", "balcony"]),
            _base_payload(plot_width=23, plot_length=37, bedrooms=2, bathrooms=2, city="Goa", state="Goa", facing="south"),
            _base_payload(plot_width=30, plot_length=44, bedrooms=3, bathrooms=3, city="Shimla", state="Himachal Pradesh"),
        ]

        composites = []
        accepted = 0
        no_error = 0
        catastrophic = 0
        improved_or_equal = 0

        for payload in scenarios:
            out = generate_dynamic_layout(payload)
            if "error" not in out:
                no_error += 1
            else:
                continue

            score = float(out.get("design_score", {}).get("composite", 0.0))
            composites.append(score)

            if out.get("accepted"):
                accepted += 1
            if score < 55:
                catastrophic += 1

            attempt_scores = [float(x) for x in (out.get("attempt_scores") or [])]
            if attempt_scores:
                if abs(max(attempt_scores) - score) < 1e-6:
                    improved_or_equal += 1

        self.assertGreaterEqual(no_error, 15, "Too many hard failures in scenario pack")
        self.assertGreaterEqual(accepted, 10, "Acceptance count below benchmark gate")
        self.assertGreaterEqual(statistics.mean(composites), 68.0, "Average quality score below gate")
        self.assertLessEqual(catastrophic, 2, "Too many catastrophic low-score outputs")
        self.assertGreaterEqual(improved_or_equal, 14, "Retry loop is not preserving best attempt consistently")

    def test_iit_generalization_stress(self):
        rng = random.Random(42)
        cities = ["Pune", "Chennai", "Delhi", "Kochi", "Bengaluru", "Jaipur", "Lucknow", "Mumbai", "Goa", "Shimla"]
        states = ["Maharashtra", "Tamil Nadu", "Delhi", "Kerala", "Karnataka", "Rajasthan", "Uttar Pradesh", "Maharashtra", "Goa", "Himachal Pradesh"]
        families = ["nuclear", "joint-family", "working-couple", "elderly", "rental"]
        extras_pool = ["pooja", "study", "store", "balcony", "garage"]
        facings = ["east", "west", "north", "south"]

        outputs = []
        for _ in range(60):
            idx = rng.randrange(len(cities))
            width = rng.randint(18, 38)
            length = rng.randint(30, 58)
            bedrooms = rng.randint(1, 4)
            bathrooms = rng.randint(1, 5)
            k = rng.randint(0, 3)
            extras = rng.sample(extras_pool, k=k)
            payload = _base_payload(
                plot_width=width,
                plot_length=length,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                facing=rng.choice(facings),
                family_type=rng.choice(families),
                city=cities[idx],
                state=states[idx],
                vastu=rng.choice([True, False]),
                extras=extras,
            )
            outputs.append(generate_dynamic_layout(payload))

        valid = [o for o in outputs if "error" not in o]
        self.assertGreaterEqual(len(valid), 54, "Generalization stress produced too many errors")

        composites = [float(o.get("design_score", {}).get("composite", 0.0)) for o in valid]
        self.assertGreaterEqual(statistics.mean(composites), 64.0, "Generalization average score below gate")

        strong_layouts = [s for s in composites if s >= 72.0]
        self.assertGreaterEqual(len(strong_layouts), 28, "Too few strong outputs under stress benchmark")


if __name__ == "__main__":
    unittest.main()
