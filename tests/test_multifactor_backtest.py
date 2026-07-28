from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategy.multifactor_backtest import (
    MultiFactorBacktestConfig,
    _factor_snapshot,
    _markowitz_weights,
    _rank_ic_series,
    _state_conditioned_icir,
    add_benchmark_nav,
    build_monthly_targets,
    build_rebalance_schedule,
    compute_portfolio_turnover,
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

    def test_rank_ic_is_spearman_correlation(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        scores = pd.DataFrame(
            {"A": [1.0, 3.0], "B": [2.0, 2.0], "C": [3.0, 1.0]},
            index=dates,
        )
        future_return_rank = pd.DataFrame(
            {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]},
            index=dates,
        )

        rank_ic = _rank_ic_series(scores, future_return_rank, min_universe_size=3)

        np.testing.assert_allclose(rank_ic, [1.0, -1.0])

    def test_state_conditioned_icir_uses_only_ic_known_by_signal_date(self) -> None:
        dates = pd.date_range("2024-01-02", periods=8, freq="B")
        rank_ic = pd.Series([0.10, -0.50, 0.20, 0.30, 0.40, 100.0, 100.0, 100.0], index=dates)
        states = pd.DataFrame(
            {"state_name": ["Bear", "Bull", "Bear", "Bear", "Bear", "Bear", "Bear", "Bear"]},
            index=dates,
        )

        icir, state_name = _state_conditioned_icir(
            rank_ic,
            states,
            dates[-1],
            pd.DatetimeIndex(dates),
            holding_period_days=2,
            halflife_days=120,
        )
        changed = rank_ic.copy()
        changed.iloc[6:] = -1000.0
        changed_icir, _ = _state_conditioned_icir(
            changed,
            states,
            dates[-1],
            pd.DatetimeIndex(dates),
            holding_period_days=2,
            halflife_days=120,
        )

        self.assertEqual(state_name, "Bear")
        self.assertGreater(icir, 0.0)
        self.assertAlmostEqual(icir, changed_icir)

        negative_icir, _ = _state_conditioned_icir(
            -rank_ic.abs(),
            states,
            dates[-1],
            pd.DatetimeIndex(dates),
            holding_period_days=2,
            halflife_days=120,
        )
        self.assertEqual(negative_icir, 0.0)

    def test_snapshot_weights_bull_and_bear_factors_by_positive_icir(self) -> None:
        dates = pd.date_range("2024-01-02", periods=8, freq="B")
        signal_date = dates[-1]
        scores = {
            "value": pd.DataFrame({"A": [2.0], "B": [0.0], "C": [-2.0]}, index=[signal_date]),
            "quality": pd.DataFrame({"A": [-1.0], "B": [0.0], "C": [1.0]}, index=[signal_date]),
        }
        rank_ics = {
            "value": pd.Series([0.10, 0.15, 0.20, 0.25, 0.30, 0.35], index=dates[:6]),
            "quality": pd.Series([0.30, 0.25, 0.20, 0.15, 0.10, 0.05], index=dates[:6]),
        }
        histories = {
            "value": pd.DataFrame({"state_name": ["Bull"] * len(dates)}, index=dates),
            "quality": pd.DataFrame({"state_name": ["Bear"] * len(dates)}, index=dates),
        }
        config = MultiFactorBacktestConfig(
            factor_names=("value", "quality"), holding_period_days=2, ic_halflife_days=120
        )

        snapshot, factor_weights, icirs, factor_states = _factor_snapshot(
            scores,
            rank_ics,
            histories,
            signal_date,
            pd.DatetimeIndex(dates),
            config,
        )

        self.assertEqual(set(factor_weights), {"value", "quality"})
        self.assertAlmostEqual(sum(factor_weights.values()), 1.0)
        self.assertTrue(all(value > 0.0 for value in icirs.values()))
        self.assertEqual(factor_states, {"value": "Bull", "quality": "Bear"})
        self.assertFalse(snapshot.empty)

    def test_markowitz_weights_satisfy_full_investment_long_only_and_cap(self) -> None:
        rng = np.random.default_rng(7)
        codes = [f"S{idx:02d}" for idx in range(25)]
        scores = pd.Series(np.linspace(-1.0, 1.0, len(codes)), index=codes)
        returns = pd.DataFrame(rng.normal(0.0, 0.02, size=(80, len(codes))), columns=codes)

        weights = _markowitz_weights(scores, returns, risk_aversion=0.5, max_stock_weight=0.05)

        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(weights.min()), 0.0)
        self.assertLessEqual(float(weights.max()), 0.05 + 1e-6)

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

    def test_portfolio_turnover_uses_drifted_pre_rebalance_weights(self) -> None:
        dates = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-03-01"])
        daily_returns = pd.DataFrame(
            {"A": [0.10, 0.00, 0.00], "B": [0.00, 0.00, 0.00]},
            index=dates,
        )
        targets = pd.DataFrame(
            {
                "lambda": [0.3] * 4,
                "rebalance_date": [dates[0], dates[0], dates[2], dates[2]],
                "period_end": [dates[1], dates[1], dates[2], dates[2]],
                "stock_code": ["A", "B", "A", "B"],
                "weight": [0.6, 0.4, 0.5, 0.5],
            }
        )

        turnover = compute_portfolio_turnover(daily_returns, targets, 0.3)

        drifted_a_weight = 0.66 / 1.06
        second_period_turnover = abs(drifted_a_weight - 0.5)
        self.assertAlmostEqual(turnover, (1.0 + second_period_turnover) / 2.0)

    def test_st_and_bottom_turnover_stocks_are_excluded_before_selection(self) -> None:
        dates = pd.date_range("2024-01-02", periods=10, freq="B")
        signal_date = dates[-1]
        schedule = pd.DataFrame(
            {
                "rebalance_date": [signal_date + pd.offsets.BDay()],
                "signal_date": [signal_date],
                "period_end": [signal_date + pd.offsets.BDay(5)],
            }
        )
        scores = {
            "value": pd.DataFrame(
                {"ST_TOP": [5.0], "LOW": [4.0], "A": [3.0], "B": [2.0], "C": [1.0]},
                index=[signal_date],
            )
        }
        rank_ics = {"value": pd.Series(np.linspace(0.1, 0.8, 8), index=dates[:8])}
        histories = {
            "value": pd.DataFrame({"state_name": ["Bull"] * len(dates)}, index=dates)
        }
        stock_names = pd.Series({"ST_TOP": "*ST测试", "LOW": "低换手", "A": "甲", "B": "乙", "C": "丙"})
        rolling_turnover = pd.DataFrame(
            {"ST_TOP": [1.0], "LOW": [0.01], "A": [0.8], "B": [0.9], "C": [1.1]},
            index=[signal_date],
        )
        daily_returns = pd.DataFrame(
            np.linspace(-0.02, 0.02, len(dates) * 5).reshape(len(dates), 5),
            index=dates,
            columns=["ST_TOP", "LOW", "A", "B", "C"],
        )
        config = MultiFactorBacktestConfig(
            factor_names=("value",),
            lambdas=(0.3,),
            top_n=2,
            holding_period_days=2,
            max_stock_weight=0.5,
        )

        targets = build_monthly_targets(
            schedule,
            daily_returns,
            scores,
            rank_ics,
            histories,
            rolling_turnover,
            stock_names,
            config,
        )

        self.assertEqual(set(targets["stock_code"]), {"A", "B"})
        self.assertNotIn("ST_TOP", targets["stock_code"].tolist())
        self.assertNotIn("LOW", targets["stock_code"].tolist())
        self.assertAlmostEqual(float(targets["weight"].sum()), 1.0)
        self.assertLessEqual(float(targets["weight"].max()), 0.5 + 1e-6)


if __name__ == "__main__":
    unittest.main()