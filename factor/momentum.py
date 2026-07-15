"""
构建A股横截面动量组合（Cross-sectional Momentum）并输出日频收益序列。

与论文《Dynamic Factor Allocation using Regime Switching Signals》的对应关系：
1) 先独立构建单因子收益（这里是Momentum Portfolio Return）。
2) 后续SJM阶段会把该因子收益减去市场收益得到Active Return，并做状态识别。

本模块只实现第一步：动量组合收益构建，不涉及特征工程与SJM训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MomentumConfig:
    """动量组合构建参数。

    参数说明：
    - data_root: A股日线CSV根目录（按年/月/日组织）。
    - output_path: 输出动量组合日收益CSV路径。
    - lookback: 动量观察窗口（月），经典值12。
    - skip_month: 跳过最近月份数，经典12-1即1。
    - top_ratio: 每次调仓选取动量最高股票占比，经典分位常见20%。
    - rebalance_frequency: 调仓频率，当前实现支持M（月频）。
    - incremental: 是否增量更新，True时会复用已有输出并只重算必要区间。
    - encoding_candidates: 原始CSV解码候选，A股常见gbk/gb18030。
    """

    data_root: str = "data/A股日线指标"
    output_path: str = "outputs/momentum_return.csv"
    lookback: int = 12
    skip_month: int = 1
    top_ratio: float = 0.20
    rebalance_frequency: str = "M"
    incremental: bool = True
    encoding_candidates: tuple[str, ...] = ("gbk", "gb18030", "utf-8-sig", "utf-8")


def _extract_date_from_filename(file_path: Path) -> Optional[pd.Timestamp]:
    """从文件名提取交易日。

    目录样式通常是 YYYY-MM-DD.csv。本函数用于：
    1) 在增量更新时快速筛选需要读取的文件；
    2) 避免无关文件被误读。
    """

    match = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", file_path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), errors="coerce")


def _iter_daily_files(data_root: Path, min_trade_date: Optional[pd.Timestamp] = None) -> Iterable[Path]:
    """按日期顺序迭代日线文件。

    说明：
    - 为了支持几千只股票的大样本，尽量避免一次读取全部历史再筛选；
    - 增量模式下可传入min_trade_date，仅加载必要日期后的文件。
    """

    all_files = sorted(data_root.rglob("*.csv"))
    if min_trade_date is None:
        yield from all_files
        return

    for file_path in all_files:
        file_date = _extract_date_from_filename(file_path)
        if file_date is not None and file_date >= min_trade_date:
            yield file_path


def _read_one_daily_file(file_path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    """读取单个交易日CSV，并标准化到统一字段。

    统一输出列：
    - trade_date: 交易日期
    - stock_code: 股票代码
    - close: 收盘价

    兼容中英文字段名：
    - 交易日期/日期/trade_date
    - 股票代码/代码/stock_code
    - 收盘价/日收盘价/close
    """

    required_alias = {
        "trade_date": ["日期"],
        "stock_code": ["代码"],
        "close": ["日收盘价"],
    }

    # 防御式处理：若误传字符串，避免被逐字符迭代成 g/b/k。
    if isinstance(encodings, str):
        encodings = (encodings,)

    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            raw = pd.read_csv(file_path, encoding=enc)
            col_map: dict[str, str] = {}
            for std_col, aliases in required_alias.items():
                for alias in aliases:
                    if alias in raw.columns:
                        col_map[alias] = std_col
                        break

            std_cols = set(col_map.values())
            if std_cols != {"trade_date", "stock_code", "close"}:
                continue

            df = raw.rename(columns=col_map)[list({v: None for v in col_map.values()}.keys())].copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            df["stock_code"] = df["stock_code"].astype(str).str.strip()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["trade_date", "stock_code", "close"])
            return df
        except Exception as err:  # noqa: BLE001
            last_err = err
            continue

    raise ValueError(f"读取失败或缺少必要列: {file_path}, last_error={last_err}")


def load_price_panel(config: MomentumConfig, min_trade_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """读取全市场日线并生成标准化价格面板。
    """

    data_root = Path(config.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")

    parts: list[pd.DataFrame] = []
    files = list(_iter_daily_files(data_root, min_trade_date=min_trade_date))
    if not files:
        raise FileNotFoundError(f"未找到CSV文件: {data_root}")

    for idx, file_path in enumerate(files, start=1):
        parts.append(_read_one_daily_file(file_path, config.encoding_candidates))
        if idx % 250 == 0:
            print(f"[load] 已读取 {idx}/{len(files)} 个文件")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    panel = panel.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
    return panel


def compute_daily_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """计算个股日收益率。

    公式：ret_{i,t}=P_{i,t}/P_{i,t-1}-1。

    说明：
    - 动量组合最终是按持仓股票的日收益做等权聚合；
    - 这里先在个股层生成ret，便于后续向量化组合计算。
    """

    data = price_panel.copy()
    data["ret"] = data.groupby("stock_code", sort=False)["close"].pct_change()
    return data


def compute_monthly_momentum(price_panel: pd.DataFrame, lookback: int, skip_month: int) -> pd.DataFrame:
    """计算月度12-1类动量信号。

    步骤：
    1) 先把日收盘压缩成月末收盘（每只股票每月最后一个交易日）。
    2) 计算月收益。
    3) 对月收益做滚动复合，窗口lookback。
    4) 再向后shift(skip_month)以排除最近月份。

    对应经典Cross-sectional Momentum定义：
    Momentum(12-1) = 最近12个月累计收益，剔除最近1个月。
    """

    daily_close = (
        price_panel
        .pivot(index="trade_date", columns="stock_code", values="close")
        .sort_index()
    )
    monthly_close = daily_close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()

    momentum = (1.0 + monthly_ret).rolling(window=lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0
    momentum = momentum.shift(skip_month)

    return momentum


def _month_end_trade_dates(trading_days: pd.DatetimeIndex) -> pd.Series:
    """把自然月映射为实际月末交易日。

    原因：A股存在节假日与非交易日，调仓应落在实际可交易日期。
    """

    day_df = pd.DataFrame({"trade_date": pd.DatetimeIndex(trading_days).sort_values()})
    day_df["month"] = day_df["trade_date"].dt.to_period("M")
    month_ends = day_df.groupby("month", observed=True)["trade_date"].max()
    return month_ends


def build_momentum_portfolio_returns(
    price_with_ret: pd.DataFrame,
    monthly_momentum: pd.DataFrame,
    top_ratio: float,
    rebalance_frequency: str = "M",
) -> pd.DataFrame:
    """按月调仓构建动量组合并输出日收益。

    组合规则：
    - 调仓频率：月频（M）。
    - 排序指标：月末时点可得的动量信号。
    - 选股：动量最高top_ratio股票。
    - 权重：等权。
    - 组合收益：持仓股票日收益率的横截面均值。

    返回列：
    - trade_date
    - momentum_return
    - holdings_count
    - rebalance_date
    """

    if rebalance_frequency.upper() != "M":
        raise NotImplementedError("当前版本仅实现月频调仓(rebalance_frequency='M')")

    daily_ret = (
        price_with_ret
        .pivot(index="trade_date", columns="stock_code", values="ret")
        .sort_index()
    )
    trading_days = daily_ret.index
    month_ends = _month_end_trade_dates(trading_days)

    results: list[pd.DataFrame] = []
    rebalance_days = month_ends.values

    for i in range(len(rebalance_days) - 1):
        reb_day = pd.Timestamp(rebalance_days[i])
        next_reb_day = pd.Timestamp(rebalance_days[i + 1])

        month_end_key = reb_day.to_period("M").to_timestamp("M")
        if month_end_key not in monthly_momentum.index:
            continue

        signal = monthly_momentum.loc[month_end_key].dropna()
        if signal.empty:
            continue

        n_pick = max(1, int(np.floor(len(signal) * top_ratio)))
        selected = signal.nlargest(n_pick).index

        mask = (daily_ret.index > reb_day) & (daily_ret.index <= next_reb_day)
        holding_window = daily_ret.loc[mask, selected]
        if holding_window.empty:
            continue

        # 等权组合：对当日可用收益取均值（自动对停牌/缺失做可用样本归一化）。
        port_ret = holding_window.mean(axis=1, skipna=True)

        out = pd.DataFrame(
            {
                "trade_date": port_ret.index,
                "momentum_return": port_ret.values,
                "holdings_count": int(len(selected)),
                "rebalance_date": reb_day,
            }
        )
        results.append(out)

    if not results:
        return pd.DataFrame(columns=["trade_date", "momentum_return", "holdings_count", "rebalance_date"])

    return pd.concat(results, ignore_index=True)


def _estimate_incremental_start(config: MomentumConfig, output_path: Path) -> Optional[pd.Timestamp]:
    """根据历史输出估计增量重算起点。

    思路：
    - 仅从“最近已产出日期”往前回看(lookback + skip + 2)个月，避免全历史重算；
    - 额外留2个月缓冲，降低月末映射、缺失值等边界问题。
    """

    if not config.incremental or (not output_path.exists()):
        return None

    hist = pd.read_csv(output_path)
    if hist.empty or "trade_date" not in hist.columns:
        return None

    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce")
    last_day = hist["trade_date"].max()
    if pd.isna(last_day):
        return None

    months_back = config.lookback + config.skip_month + 2
    return pd.Timestamp(last_day) - pd.DateOffset(months=months_back)


def _merge_incremental_output(new_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """把新计算结果与历史输出合并并去重。

    规则：
    - 若历史文件不存在，直接返回new_df。
    - 若存在，则按trade_date去重，保留最新计算结果。
    """

    if not output_path.exists():
        return new_df.sort_values("trade_date").reset_index(drop=True)

    old_df = pd.read_csv(output_path)
    old_df["trade_date"] = pd.to_datetime(old_df["trade_date"], errors="coerce")

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.dropna(subset=["trade_date"])
    merged = merged.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
    return merged.reset_index(drop=True)


def run_momentum_pipeline(config: MomentumConfig) -> pd.DataFrame:
    """执行动量组合全流程并落盘。

    输出文件：momentum_return.csv，包含：
    - trade_date: 交易日
    - momentum_return: 动量组合日收益
    - holdings_count: 当期持仓股票数
    - rebalance_date: 当前持仓对应的调仓日

    论文映射说明：
    - 本输出就是后续构建Active Return所需的Momentum Return输入。
    """

    output_path = Path(config.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    incremental_start = _estimate_incremental_start(config, output_path)
    price_panel = load_price_panel(config, min_trade_date=incremental_start)
    price_with_ret = compute_daily_returns(price_panel)
    monthly_momentum = compute_monthly_momentum(
        price_panel=price_panel,
        lookback=config.lookback,
        skip_month=config.skip_month,
    )

    result = build_momentum_portfolio_returns(
        price_with_ret=price_with_ret,
        monthly_momentum=monthly_momentum,
        top_ratio=config.top_ratio,
        rebalance_frequency=config.rebalance_frequency,
    )

    result = _merge_incremental_output(result, output_path)
    result = result.sort_values("trade_date").reset_index(drop=True)

    # 统一日期列格式，避免出现 YYYY-MM-DD 与 YYYY-MM-DD 00:00:00 混合导致误读。
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["rebalance_date"] = pd.to_datetime(result["rebalance_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[done] 输出文件: {output_path}, 共 {len(result):,} 行")
    return result


if __name__ == "__main__":
    cfg = MomentumConfig(
        data_root="data/A股日线指标",
        output_path="outputs/momentum_return.csv",
        lookback=12,
        skip_month=1,
        top_ratio=0.20,
        rebalance_frequency="M",
        incremental=True,
    )
    run_momentum_pipeline(cfg)
