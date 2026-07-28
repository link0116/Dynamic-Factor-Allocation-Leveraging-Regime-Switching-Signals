"""使用 Growth 因子过去20个交易日主动收益直接划分 Bull/Bear 状态。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SJMTuningConfig
from main import _plot_full_period_split
from sjm.regime_analysis import build_state_statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_growth_20d_regime(feature_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """按 Growth 因子过去 window 个交易日复合主动收益的正负划分状态。"""

    required = {"trade_date", "growth_return", "active_return", "market_return"}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"Growth 特征数据缺少列: {missing}")
    if window <= 0:
        raise ValueError("收益率窗口必须为正")

    data = feature_df[list(required)].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    for column in ["growth_return", "active_return", "market_return"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["trade_date", "growth_return", "active_return", "market_return"])
    data = data.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)

    data["active_return_20d"] = (
        (1.0 + data["active_return"]).rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0
    )
    data = data.dropna(subset=["active_return_20d"]).copy()

    state_name = pd.Series(
        np.where(
            data["active_return_20d"] > 0.0,
            "Bull",
            np.where(data["active_return_20d"] < 0.0, "Bear", None),
        ),
        index=data.index,
        dtype="object",
    ).ffill().bfill()
    data["state_name"] = state_name.fillna("Bear")
    data["state"] = data["state_name"].map({"Bear": 0, "Bull": 1}).astype(int)
    return data[
        [
            "trade_date",
            "active_return",
            "growth_return",
            "market_return",
            "active_return_20d",
            "state",
            "state_name",
        ]
    ].reset_index(drop=True)


def run_growth_20d_regime_experiment(
    input_path: str = "outputs/growth/sjm_features.csv",
    output_dir: str = "outputs/growth_20d_return_regime",
    window: int = 20,
) -> dict[str, str]:
    """运行对比实验并输出逐日状态、统计表和全时期划分图。"""

    source_path = Path(input_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    if not source_path.exists():
        raise FileNotFoundError(f"缺少 Growth 特征文件: {source_path}")

    feature_df = pd.read_csv(source_path)
    state_daily = build_growth_20d_regime(feature_df, window=window)
    if state_daily.empty:
        raise ValueError("20日收益率计算后没有可用状态")

    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "growth_20d_state_daily.csv"
    statistics_path = destination / "state_statistics.csv"
    plot_path = destination / "full_period_split_plot.png"
    state_daily.to_csv(state_path, index=False, encoding="utf-8-sig")

    state_statistics = build_state_statistics(state_daily, factor_return_col="growth_return")
    state_statistics.to_csv(statistics_path, index=False, encoding="utf-8-sig")
    _plot_full_period_split(
        feature_df=feature_df,
        tuning_cfg=SJMTuningConfig(),
        output_path=str(plot_path),
        state_daily=state_daily,
        factor_display_name=f"Growth {window}D Active Return",
    )

    outputs = {
        "method": f"past_{window}_trading_day_compound_growth_active_return",
        "input_path": str(source_path),
        "state_daily_path": str(state_path),
        "state_statistics_path": str(statistics_path),
        "plot_path": str(plot_path),
    }
    (destination / "experiment_manifest.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Growth 因子20日主动收益率牛熊状态对比实验")
    parser.add_argument("--input", default="outputs/growth/sjm_features.csv")
    parser.add_argument("--output-dir", default="outputs/growth_20d_return_regime")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    outputs = run_growth_20d_regime_experiment(args.input, args.output_dir, args.window)
    print(f"[done] 状态文件: {outputs['state_daily_path']}")
    print(f"[done] 划分图: {outputs['plot_path']}")


if __name__ == "__main__":
    main()