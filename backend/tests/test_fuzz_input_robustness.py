import unittest

from scripts.fuzz_any_ai_inputs import run


class TestFuzzInputRobustness(unittest.TestCase):
    def test_fuzz_payload_robustness(self):
        # Keep CI/runtime reasonable while covering a wide malformed-input surface.
        report = run(count=1000, seed=19)

        self.assertTrue(report["gates"]["direct_error_rate_le_0_01"])
        self.assertTrue(report["gates"]["api_5xx_rate_le_0_01"])


if __name__ == "__main__":
    unittest.main()
