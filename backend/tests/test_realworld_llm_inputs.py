import unittest

from scripts.realworld_llm_input_suite import run_suite


class TestRealWorldLLMInputs(unittest.TestCase):
    def test_realworld_llm_input_suite(self):
        # 5 model styles x 60 rounds = 300 realistic mixed-shape payloads.
        report = run_suite(seed=21, rounds=60)

        self.assertTrue(report["gates"]["direct_no_crash"])
        self.assertTrue(report["gates"]["api_no_5xx"])
        self.assertTrue(report["gates"]["api_handled_rate_eq_1_00"])


if __name__ == "__main__":
    unittest.main()
