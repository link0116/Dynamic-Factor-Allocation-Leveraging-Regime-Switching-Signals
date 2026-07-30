from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.state_stability_score import (
    build_state_stability_scores,
    compute_state_stability_metrics,
)


class StateStabilityScoreTests(unittest.TestCase):
    def test_metrics_follow_state_run_formulas(self) -> None:
        states = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-02", periods=6, freq="B"),
                "state_name": ["Bull", "Bull", "Bear", "Bull", "Bull", "Bull"],
            }
        )

        metrics = compute_state_stability_metrics(states)

        self.assertEqual(metrics["sample_length"], 6)
        self.assertEqual(metrics["switch_count"], 2)
        self.assertEqual(metrics["regime_count"], 3)
        self.assertAlmostEqual(metrics["switch_frequency"], 2 / 5)
        self.assertAlmostEqual(metrics["average_state_duration"], 2.0)
        self.assertAlmostEqual(metrics["duration_variance"], 2 / 3)
        self.assertAlmostEqual(metrics["longest_regime_ratio"], 0.5)

    def test_scores_use_exclusive_cutoff_and_equal_weighted_zscores(self) -> None:
        dates = pd.to_datetime(
            ["2024-05-27", "2024-05-28", "2024-05-29", "2024-05-30", "2024-05-31", "2024-06-03"]
        )
        histories = {
            "stable": pd.DataFrame(
                {"trade_date": dates, "state_name": ["Bull", "Bull", "Bull", "Bull", "Bull", "Bear"]}
            ),
            "mixed": pd.DataFrame(
                {"trade_date": dates, "state_name": ["Bull", "Bull", "Bear", "Bear", "Bull", "Bear"]}
            ),
            "switching": pd.DataFrame(
                {"trade_date": dates, "state_name": ["Bull", "Bear", "Bull", "Bear", "Bull", "Bear"]}
            ),
        }

        scores = build_state_stability_scores(histories, cutoff_date="2024-06-01").set_index("factor")

        self.assertTrue(scores["sample_length"].eq(5).all())
        expected = 0.25 * (
            -scores["z_switch_frequency"]
            + scores["z_average_state_duration"]
            - scores["z_duration_variance"]
            + scores["z_longest_regime_ratio"]
        )
        np.testing.assert_allclose(scores["sss"], expected)
        self.assertGreater(scores.loc["stable", "sss"], scores.loc["switching", "sss"])


if __name__ == "__main__":
    unittest.main()