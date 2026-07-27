from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategy.multifactor_backtest import (
    MultiFactorBacktestConfig,
    _factor_snapshot,
    add_benchmark_nav,
    build_monthly_targets,
    build_rebalance_schedule,
    simulate_monthly_portfolio,
)


class MultiFactorBacktestTests(unittest.TestCase):
    def test_benchmark_nav_is_aligned_to_strategy_dates(self) -> None:
        dates = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-05"])
        returns = pd.DataFrame(
            {"trade_date": dates, "strategy_return": [0.01, 0.02, -0.01], "nav": [1.01, 1.0302, 1.019898]}
        )
        market_returns = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-02-05", "2024-02-01", "2024-02-02"]),
                "market_return": [0.03, 0.01, -0.02],
            }
        )

        result = add_benchmark_nav(returns, market_returns)

        np.testing.assert_allclose(result["benchmark_return"], [0.01, -0.02, 0.03])
        np.testing.assert_allclose(result["benchmark_nav"], [1.01, 1.01 * 0.98, 1.01 * 0.98 * 1.03])

    def test_rebalance_uses_strictly_previous_trading_day(self) -> None:
        dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"])

        schedule = build_rebalance_schedule(pd.DatetimeIndex(dates))

        february = schedule[schedule["rebalance_date"].eq(pd.Timestamp("2024-02-01"))].iloc[0]
        self.assertEqual(february["signal_date"], pd.Timestamp("2024-01-31"))
        self.assertLess(february["signal_date"], february["rebalance_date"])
        self.assertEqual(february["period_end"], pd.Timestamp("2024-02-02"))

        filtered = schedule[schedule["rebalance_date"] >= pd.Timestamp("2024-02-01")]
        self.assertEqual(filtered.iloc[0]["signal_date"], pd.Timestamp("2024-01-31"))

    def test_snapshot_uses_only_historical_bull_returns_and_reverses_negative_factor(self) -> None:
        dates = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31"])
        scores = {
            "value": pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=dates),
            "quality": pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=dates),
        }
        histories = {
            "value": pd.DataFrame(
                {
                    "state_name": ["Bull", "Bear", "Bull"],
                    "bull_mean_return": [0.02, 0.02, 0.04],
                },
                index=dates,
            ),
            "quality": pd.DataFrame(
                {
                    "state_name": ["Bull", "Bull", "Bull"],
                    "bull_mean_return": [-0.01, -0.01, -0.01],
                },
                index=dates,
            ),
        }

        snapshot, factor_weights, bull_returns = _factor_snapshot(
            scores, histories, pd.Timestamp("2024-01-30")
        )

        self.assertEqual(bull_returns, {"quality": -0.01})
        self.assertEqual(factor_weights, {"quality": 1.0})
        self.assertLess(snapshot.loc["A", "composite_score"], snapshot.loc["B", "composite_score"])

    def test_factor_weights_and_softmax_stock_weights_follow_formula(self) -> None:
        signal_date = pd.Timestamp("2024-01-31")
        schedule = pd.DataFrame(
            {
                "rebalance_date": [pd.Timestamp("2024-02-01")],
                "signal_date": [signal_date],
                "period_end": [pd.Timestamp("2024-02-29")],
            }
        )
        scores = {
            "value": pd.DataFrame({"A": [2.0], "B": [0.0], "C": [-2.0]}, index=[signal_date]),
            "quality": pd.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]}, index=[signal_date]),
        }
        histories = {
            "value": pd.DataFrame({"state_name": ["Bull"], "bull_mean_return": [0.03]}, index=[signal_date]),
            "quality": pd.DataFrame(
                {"state_name": ["Bull"], "bull_mean_return": [-0.01]}, index=[signal_date]
            ),
        }
        config = MultiFactorBacktestConfig(
            factor_names=("value", "quality"), lambdas=(0.3,), top_n=2
        )

        targets = build_monthly_targets(schedule, scores, histories, pd.Series(dtype=str), config)

        self.assertEqual(set(targets["stock_code"]), {"A", "B"})
        self.assertAlmostEqual(float(targets["weight"].sum()), 1.0)
        self.assertTrue(np.all(targets["signal_date"] < targets["rebalance_date"]))
        self.assertIn('"value": 0.75', targets["factor_weights"].iloc[0])
        self.assertIn('"quality": 0.25', targets["factor_weights"].iloc[0])

    def test_monthly_portfolio_can_drift_read_only_pandas_values(self) -> None:
        dates = pd.to_datetime(["2024-02-01", "2024-02-02"])
        daily_returns = pd.DataFrame(
            {"A": [0.10, 0.00], "B": [0.00, 0.10]},
            index=dates,
        )
        targets = pd.DataFrame(
            {
                "lambda": [0.3, 0.3],
                "rebalance_date": [dates[0], dates[0]],
                "period_end": [dates[-1], dates[-1]],
                "stock_code": ["A", "B"],
                "weight": [0.6, 0.4],
            }
        )

        result = simulate_monthly_portfolio(daily_returns, targets, 0.3)

        np.testing.assert_allclose(result["strategy_return"], [0.06, 0.04 / 1.06])
        self.assertAlmostEqual(float(result["nav"].iloc[-1]), 1.10)

    def test_top_ranked_st_stock_is_excluded_before_selection(self) -> None:
        signal_date = pd.Timestamp("2024-01-31")
        schedule = pd.DataFrame(
            {
                "rebalance_date": [pd.Timestamp("2024-02-01")],
                "signal_date": [signal_date],
                "period_end": [pd.Timestamp("2024-02-29")],
            }
        )
        scores = {
            "value": pd.DataFrame(
                {"ST_TOP": [3.0], "NORMAL_A": [2.0], "NORMAL_B": [1.0]},
                index=[signal_date],
            )
        }
        histories = {
            "value": pd.DataFrame(
                {"state_name": ["Bull"], "bull_mean_return": [0.03]},
                index=[signal_date],
            )
        }
        stock_names = pd.Series(
            {"ST_TOP": "*ST测试", "NORMAL_A": "正常股份", "NORMAL_B": "普通公司"}
        )
        config = MultiFactorBacktestConfig(factor_names=("value",), lambdas=(0.3,), top_n=2)

        targets = build_monthly_targets(schedule, scores, histories, stock_names, config)

        self.assertEqual(set(targets["stock_code"]), {"NORMAL_A", "NORMAL_B"})
        self.assertNotIn("ST_TOP", targets["stock_code"].tolist())
        self.assertAlmostEqual(float(targets["weight"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()