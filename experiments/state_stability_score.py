"""计算七个因子在指定截止日前的状态稳定性分数。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factor.cross_sectional import FACTOR_LIST


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUTOFF_DATE = "2024-06-01"
DEFAULT_OUTPUT_PATH = "outputs/state_stability_score/state_stability_scores.csv"


def compute_state_stability_metrics(state_daily: pd.DataFrame) -> dict[str, float | int]:
    """计算单个因子状态序列的 SF、ASD、DV 和 LRR。"""

    required = {"trade_date", "state_name"}
    missing = required - set(state_daily.columns)
    if missing:
        raise ValueError(f"状态数据缺少列: {missing}")

    data = state_daily[["trade_date", "state_name"]].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["state_name"] = data["state_name"].astype("string").str.strip()
    data = data.dropna(subset=["trade_date", "state_name"])
    data = data[data["state_name"].ne("")]
    data = data.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    sample_length = len(data)
    if sample_length < 2:
        raise ValueError("状态稳定性评分至少需要两个有效状态观测")

    state_values = data["state_name"].to_numpy(dtype=str)
    state_changed = np.r_[True, state_values[1:] != state_values[:-1]]
    regime_id = np.cumsum(state_changed)
    durations = data.groupby(regime_id, sort=False).size().to_numpy(dtype=float)
    regime_count = len(durations)
    switch_count = regime_count - 1
    if int(durations.sum()) != sample_length or switch_count != int(state_changed.sum() - 1):
        raise RuntimeError("状态区间划分结果不一致")

    average_duration = float(durations.mean())
    return {
        "sample_length": sample_length,
        "switch_count": switch_count,
        "regime_count": regime_count,
        "switch_frequency": float(switch_count / (sample_length - 1)),
        "average_state_duration": average_duration,
        "duration_variance": float(np.mean(np.square(durations - average_duration))),
        "longest_regime_ratio": float(durations.max() / sample_length),
    }


def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    standard_deviation = float(numeric.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    return (numeric - float(numeric.mean())) / standard_deviation


def build_state_stability_scores(
    state_histories: dict[str, pd.DataFrame],
    cutoff_date: str | pd.Timestamp = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """按截止日前的可用历史计算指标，并在因子横截面上生成等权 SSS。"""

    cutoff = pd.Timestamp(cutoff_date)
    rows: list[dict[str, object]] = []
    for factor, state_daily in state_histories.items():
        data = state_daily.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data[data["trade_date"].lt(cutoff)]
        metrics = compute_state_stability_metrics(data)
        valid_dates = data["trade_date"].dropna()
        rows.append(
            {
                "factor": factor,
                "sample_start": valid_dates.min().date().isoformat(),
                "sample_end": valid_dates.max().date().isoformat(),
                "cutoff_date_exclusive": cutoff.date().isoformat(),
                **metrics,
            }
        )

    result = pd.DataFrame(rows).set_index("factor")
    metric_columns = {
        "switch_frequency": "z_switch_frequency",
        "average_state_duration": "z_average_state_duration",
        "duration_variance": "z_duration_variance",
        "longest_regime_ratio": "z_longest_regime_ratio",
    }
    for metric, z_column in metric_columns.items():
        result[z_column] = _cross_sectional_zscore(result[metric])
    result["sss"] = 0.25 * (
        -result["z_switch_frequency"]
        + result["z_average_state_duration"]
        - result["z_duration_variance"]
        + result["z_longest_regime_ratio"]
    )
    result["stability_rank"] = result["sss"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["sss", "factor"], ascending=[False, True]).reset_index()


def run_state_stability_scoring(
    state_root: str = "outputs",
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """读取七因子状态文件并输出状态稳定性评分表。"""

    root = Path(state_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination

    histories: dict[str, pd.DataFrame] = {}
    for factor in FACTOR_LIST:
        source = root / factor / "sjm_state_daily.csv"
        if not source.exists():
            raise FileNotFoundError(f"缺少 {factor} 状态文件: {source}")
        histories[factor] = pd.read_csv(source)

    scores = build_state_stability_scores(histories, cutoff_date=cutoff_date)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination, index=False, encoding="utf-8-sig")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="计算七因子 SJM 状态稳定性分数")
    parser.add_argument("--state-root", default="outputs")
    parser.add_argument("--cutoff-date", default=DEFAULT_CUTOFF_DATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    scores = run_state_stability_scoring(args.state_root, args.cutoff_date, args.output)
    print(scores[["stability_rank", "factor", "sss"]].to_string(index=False))
    print(f"[done] 评分文件: {_resolve_display_path(args.output)}")


def _resolve_display_path(path: str) -> Path:
    destination = Path(path)
    return destination if destination.is_absolute() else PROJECT_ROOT / destination


if __name__ == "__main__":
    main()