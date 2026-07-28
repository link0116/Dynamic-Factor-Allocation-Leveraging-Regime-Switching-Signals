from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.growth_20d_regime import build_growth_20d_regime


class Growth20DayRegimeTests(unittest.TestCase):
    def test_regime_uses_trailing_compound_active_return_without_future_data(self) -> None:
        dates = pd.date_range("2024-01-02", periods=22, freq="B")
        growth_returns = np.array([0.02] * 20 + [0.01, 0.02])
        market_returns = np.array([0.01] * 20 + [0.50, 0.01])
        active_returns = growth_returns - market_returns
        features = pd.DataFrame(
            {
                "trade_date": dates,
                "growth_return": growth_returns,
                "market_return": market_returns,
                "active_return": active_returns,
            }
        )

        result = build_growth_20d_regime(features, window=20)

        self.assertEqual(result.iloc[0]["trade_date"], dates[19])
        self.assertAlmostEqual(result.iloc[0]["active_return_20d"], 1.01**20 - 1.0)
        self.assertEqual(result.iloc[0]["state_name"], "Bull")
        self.assertEqual(result.iloc[1]["state_name"], "Bear")


if __name__ == "__main__":
    unittest.main()