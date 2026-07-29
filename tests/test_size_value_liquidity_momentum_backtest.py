from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from experiments.size_value_liquidity_momentum_backtest import (
    EXPERIMENT_FACTORS,
    run_size_value_liquidity_momentum_experiment,
)
from strategy.multifactor_backtest import MultiFactorBacktestConfig


class SizeValueLiquidityMomentumBacktestTests(unittest.TestCase):
    @patch("experiments.size_value_liquidity_momentum_backtest.run_multifactor_backtest")
    def test_experiment_only_changes_factors_dates_top_n_and_output(self, run_backtest) -> None:
        run_backtest.return_value = {"best_lambda": 0.5, "output_dir": "comparison"}

        with TemporaryDirectory() as output_dir:
            result = run_size_value_liquidity_momentum_experiment(
                start_date="2024-06-03",
                end_date="2025-12-31",
                top_n=80,
                output_dir=output_dir,
            )

        config = run_backtest.call_args.args[0]
        defaults = MultiFactorBacktestConfig()
        self.assertEqual(config.factor_names, EXPERIMENT_FACTORS)
        self.assertEqual(config.factor_names, ("size", "value", "liquidity", "momentum"))
        self.assertEqual(config.start_date, "2024-06-03")
        self.assertEqual(config.end_date, "2025-12-31")
        self.assertEqual(config.top_n, 80)
        self.assertEqual(config.output_dir, output_dir)
        self.assertEqual(config.lambdas, defaults.lambdas)
        self.assertEqual(config.holding_period_days, defaults.holding_period_days)
        self.assertEqual(config.ic_halflife_days, defaults.ic_halflife_days)
        self.assertEqual(config.turnover_exclusion_quantile, defaults.turnover_exclusion_quantile)
        self.assertEqual(config.covariance_lookback_days, defaults.covariance_lookback_days)
        self.assertEqual(config.max_stock_weight, defaults.max_stock_weight)
        self.assertEqual(result, run_backtest.return_value)

    @patch("experiments.size_value_liquidity_momentum_backtest.run_multifactor_backtest")
    def test_existing_results_are_not_overwritten(self, run_backtest) -> None:
        with TemporaryDirectory() as output_dir:
            marker = Path(output_dir) / "daily_returns.csv"
            marker.write_text("existing result", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "避免覆盖原结果"):
                run_size_value_liquidity_momentum_experiment(output_dir=output_dir)

            run_backtest.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing result")


if __name__ == "__main__":
    unittest.main()