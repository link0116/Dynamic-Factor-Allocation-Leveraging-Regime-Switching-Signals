from __future__ import annotations

import os
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from main import FactorSpec, PipelineConfig, _export_fixed_test_evaluation
from feature.sjm_features import SJMFeatureConfig, _zscore_expanding, standardize_features
from sjm.regime_analysis import FactorRegimeAnalysisConfig, export_factor_regime_outputs
from sjm.train_sjm import _load_feature_data as load_train_features
from sjm.tuner import _load_feature_data as load_tuning_features
from sparse_jump_model import OnlineState, SparseJumpModel
from strategy.long_short import build_long_short_returns


def _fitted_model() -> tuple[SparseJumpModel, np.ndarray]:
    rng = np.random.RandomState(7)
    features = rng.normal(size=(90, 4))
    model, _ = SparseJumpModel(
        n_states=2,
        jump_penalty=2.5,
        kappa=1.8,
        n_init=2,
        max_outer_iter=3,
        max_inner_iter=5,
        random_state=3,
    ).fit(features[:50])
    return model, features[50:]


class TemporalIntegrityTests(unittest.TestCase):
    def test_non_momentum_feature_loaders_keep_real_column(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "value_features.csv"
            pd.DataFrame(
                {
                    "trade_date": pd.date_range("2024-01-01", periods=3),
                    "value_return": [0.01, -0.02, 0.03],
                    "market_return": [0.00, 0.01, -0.01],
                    "active_return": [0.01, -0.03, 0.04],
                    "z_feature": [0.1, -0.2, 0.3],
                }
            ).to_csv(path, index=False)

            train_data, _, return_col = load_train_features(str(path), "value_return")
            tuning_data, _ = load_tuning_features(str(path), "value_return")

        self.assertEqual(return_col, "value_return")
        self.assertIn("value_return", train_data.columns)
        self.assertIn("value_return", tuning_data.columns)
        self.assertNotIn("momentum_return", train_data.columns)
        self.assertNotIn("momentum_return", tuning_data.columns)

    def test_standardization_cannot_use_future_samples(self) -> None:
        base = pd.Series(np.arange(60, dtype=float))
        changed = base.copy()
        changed.iloc[50:] *= 1000

        np.testing.assert_allclose(
            _zscore_expanding(base, min_periods=30).iloc[:50],
            _zscore_expanding(changed, min_periods=30).iloc[:50],
            equal_nan=True,
        )
        with self.assertRaises(ValueError):
            standardize_features(pd.DataFrame(), SJMFeatureConfig(standardize_mode="global"))

    def test_online_batch_daily_and_chunked_are_identical(self) -> None:
        model, features = _fitted_model()
        expected = model.online_predict(features)
        daily_state = OnlineState()
        daily = np.concatenate(
            [model.online_predict(features[i : i + 1], daily_state) for i in range(len(features))]
        )
        chunk_state = OnlineState()
        chunked = np.concatenate(
            [
                model.online_predict(features[:7], chunk_state),
                model.online_predict(features[7:23], chunk_state),
                model.online_predict(features[23:], chunk_state),
            ]
        )

        np.testing.assert_array_equal(daily, expected)
        np.testing.assert_array_equal(chunked, expected)
        np.testing.assert_allclose(daily_state.dp, chunk_state.dp)

    def test_online_prefix_and_factor_states_are_independent(self) -> None:
        model, features = _fitted_model()
        prefix = model.online_predict(features[:15])
        np.testing.assert_array_equal(prefix, model.online_predict(features)[:15])

        factor_a = OnlineState()
        factor_b = OnlineState()
        model.online_predict(features[:10], factor_a)
        factor_a_dp = factor_a.dp.copy()
        model.online_predict(features[10:20], factor_b)

        np.testing.assert_allclose(factor_a.dp, factor_a_dp)
        self.assertIsNot(factor_a.dp, factor_b.dp)

        dated_state = OnlineState()
        model.online_predict(
            features[:2],
            dated_state,
            dates=np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
        )
        with self.assertRaises(ValueError):
            model.online_predict(
                features[2:3],
                dated_state,
                dates=np.array(["2024-01-02"], dtype="datetime64[D]"),
            )

    def test_strategy_uses_previous_day_position(self) -> None:
        data = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=3),
                "momentum_return": [0.02, -0.01, 0.03],
                "market_return": [0.00, 0.01, 0.01],
                "state": [1, 0, 1],
            }
        )
        result = build_long_short_returns(data, "state", {0: -0.05, 1: 0.05})

        np.testing.assert_allclose(result["signal_position"], [1.0, -1.0, 1.0])
        np.testing.assert_allclose(result["position"], [0.0, 1.0, -1.0])
        np.testing.assert_allclose(result["strategy_return"], [0.0, -0.02, -0.02])

    def test_all_seven_factors_use_their_own_return_column(self) -> None:
        factor_names = ("momentum", "value", "quality", "size", "liquidity", "lowvol", "growth")
        for factor_name in factor_names:
            return_col = f"{factor_name}_return"
            data = pd.DataFrame(
                {
                    "trade_date": pd.date_range("2024-01-01", periods=2),
                    return_col: [0.02, -0.01],
                    "market_return": [0.01, 0.00],
                    "state": [1, 0],
                }
            )
            result = build_long_short_returns(
                data,
                "state",
                {0: -0.05, 1: 0.05},
                factor_return_col=return_col,
            )
            self.assertIn(return_col, result.columns)
            if factor_name != "momentum":
                self.assertNotIn("momentum_return", result.columns)
            np.testing.assert_allclose(result["active_return"], data[return_col] - data["market_return"])

    def test_export_keeps_training_period_state_names(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_daily = pd.DataFrame(
                {
                    "trade_date": pd.date_range("2024-01-01", periods=4),
                    "state": [0, 0, 1, 1],
                    "state_name": ["Bear", "Bear", "Bull", "Bull"],
                    "active_return": [0.20, 0.20, -0.20, -0.20],
                    "value_return": [0.20, 0.20, -0.20, -0.20],
                    "market_return": [0.0] * 4,
                }
            )
            outputs = {
                "state_daily": state_daily,
                "feature_weight": pd.DataFrame({"feature": ["z_x"], "weight": [1.0]}),
                "state_centroid": pd.DataFrame(
                    {"state": [0, 1], "state_name": ["Bear", "Bull"], "z_x": [-1.0, 1.0]}
                ),
                "model": None,
                "online_state": OnlineState(
                    dp=np.array([0.5, 0.0]),
                    previous_state=1,
                    current_date=np.datetime64("2024-01-04"),
                ),
            }
            config = FactorRegimeAnalysisConfig(
                factor_name="value",
                display_name="Value",
                output_dir=str(root / "out"),
                result_dir=str(root / "result"),
                model_path=str(root / "model.pkl"),
                factor_return_col="value_return",
            )

            export_factor_regime_outputs(config, pd.DataFrame(index=range(4)), outputs, {})
            saved = pd.read_csv(root / "out" / "sjm_state_daily.csv")
            self.assertEqual(saved["state_name"].tolist(), ["Bear", "Bear", "Bull", "Bull"])
            self.assertIn("value_return", saved.columns)
            with (root / "model.pkl").open("rb") as model_file:
                payload = pickle.load(model_file)
            np.testing.assert_allclose(payload["online_state"].dp, [0.5, 0.0])
            self.assertEqual(payload["online_state"].previous_state, 1)
            self.assertEqual(payload["online_state"].current_date, np.datetime64("2024-01-04"))

    def test_test_period_carries_last_validation_signal(self) -> None:
        state_daily = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2020-01-02", "2020-01-03", "2023-07-03", "2024-05-31", "2024-06-03"]
                ),
                "state": [0, 1, 0, 1, 0],
                "state_name": ["Bear", "Bull", "Bear", "Bull", "Bear"],
                "active_return": [-0.05, 0.05, -0.01, 0.01, 0.02],
                "value_return": [-0.05, 0.05, -0.01, 0.01, 0.02],
                "market_return": [0.0] * 5,
            }
        )
        spec = FactorSpec(
            "value",
            "Value",
            "value_return",
            lambda _: "unused.csv",
            lambda _: pd.DataFrame(),
        )

        with TemporaryDirectory() as tmp:
            previous_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                paths = _export_fixed_test_evaluation(PipelineConfig(), spec, state_daily)
                self.assertIsNotNone(paths)
                strategy = pd.read_csv(paths[0])
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(strategy.loc[0, "position"], 1.0)
        self.assertEqual(strategy.loc[0, "signal_position"], -1.0)


if __name__ == "__main__":
    unittest.main()