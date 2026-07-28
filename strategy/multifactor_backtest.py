"""状态驱动的多因子月频选股回测。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from factor.cross_sectional import (
    FACTOR_LIST,
    CrossSectionalFactorConfig,
    _read_one_daily_panel_file,
    build_factor_score,
)
from factor.common import _extract_date_from_filename, _iter_daily_files
from feature.sjm_features import load_market_returns


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MultiFactorBacktestConfig:
    """多因子回测参数。"""

    factor_names: tuple[str, ...] = tuple(FACTOR_LIST)
    data_root: str = "data/A股日线指标"
    financial_root: str = "data/A股财务数据"
    market_path: str = "沪深300.csv"
    state_root: str = "outputs"
    output_dir: str = "outputs/multifactor_backtest"
    lambdas: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) #风险厌恶系数
    top_n: int = 100
    annualization: int = 252
    holding_period_days: int = 21
    ic_halflife_days: int = 120
    turnover_window_days: int = 21
    turnover_exclusion_quantile: float = 0.20
    covariance_lookback_days: int = 252
    max_stock_weight: float = 0.05
    start_date: str | None = None
    end_date: str | None = None


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (clean - float(clean.mean())) / std


def load_factor_state_history(config: MultiFactorBacktestConfig) -> dict[str, pd.DataFrame]:
    """读取各因子的逐日状态。"""

    histories: dict[str, pd.DataFrame] = {}
    state_root = _resolve_path(config.state_root)
    for factor in config.factor_names:
        path = state_root / factor / "sjm_state_daily.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少 {factor} 状态文件: {path}")
        data = pd.read_csv(path)
        required = {"trade_date", "state_name"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{factor} 状态文件缺少列: {missing}")

        data = data[["trade_date", "state_name"]].copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
        histories[factor] = data.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return histories


def build_rebalance_schedule(trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """生成月初调仓日及其严格早于调仓日的信号日。"""

    dates = pd.DatetimeIndex(trade_dates).dropna().unique().sort_values()
    if len(dates) < 2:
        return pd.DataFrame(columns=["rebalance_date", "signal_date", "period_end"])
    first_dates = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first().tolist()
    position = {date: idx for idx, date in enumerate(dates)}
    rows: list[dict[str, pd.Timestamp]] = []
    for idx, rebalance_date in enumerate(first_dates):
        date_position = position[pd.Timestamp(rebalance_date)]
        if date_position == 0:
            continue
        next_date = first_dates[idx + 1] if idx + 1 < len(first_dates) else dates[-1] + pd.Timedelta(days=1)
        period_dates = dates[(dates >= rebalance_date) & (dates < next_date)]
        if len(period_dates) == 0:
            continue
        rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "signal_date": dates[date_position - 1],
                "period_end": period_dates[-1],
            }
        )
    return pd.DataFrame(rows)


def _rank_ic_series(
    factor_score: pd.DataFrame,
    forward_return_rank: pd.DataFrame,
    min_universe_size: int = 30,
) -> pd.Series:
    """计算每日因子分数与未来持有期收益之间的截面 Spearman Rank IC。"""

    common_dates = factor_score.index.intersection(forward_return_rank.index)
    score = factor_score.reindex(common_dates).replace([np.inf, -np.inf], np.nan)
    future_rank = forward_return_rank.reindex(index=common_dates, columns=score.columns)
    valid_count = (score.notna() & future_rank.notna()).sum(axis=1)
    rank_ic = score.rank(axis=1).corrwith(future_rank, axis=1)
    return rank_ic.where(valid_count >= min_universe_size).dropna().sort_index()


def _state_conditioned_icir(
    rank_ic: pd.Series,
    state_history: pd.DataFrame,
    signal_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    holding_period_days: int,
    halflife_days: int,
) -> tuple[float, str]:
    """仅使用当期同状态且在信号日已完全实现的 IC 计算指数加权 ICIR。"""

    dates = pd.DatetimeIndex(trading_dates).dropna().unique().sort_values()
    signal_position = dates.searchsorted(signal_date, side="right") - 1
    if signal_position < holding_period_days or signal_date not in state_history.index:
        return 0.0, ""
    cutoff_position = signal_position - holding_period_days
    cutoff_date = dates[cutoff_position]
    current_state = str(state_history.loc[signal_date, "state_name"])
    eligible_ic = rank_ic.loc[:cutoff_date].dropna()
    if len(eligible_ic) < 2:
        return 0.0, current_state

    historical_states = state_history["state_name"].reindex(eligible_ic.index)
    eligible_ic = eligible_ic[historical_states.eq(current_state)]
    if len(eligible_ic) < 2:
        return 0.0, current_state

    date_positions = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    observation_positions = date_positions.reindex(eligible_ic.index).dropna()
    eligible_ic = eligible_ic.reindex(observation_positions.index)
    ages = cutoff_position - observation_positions.to_numpy(dtype=float)
    weights = np.power(0.5, ages / halflife_days)
    values = eligible_ic.to_numpy(dtype=float)
    weight_sum = float(weights.sum())
    mean = float(np.dot(weights, values) / weight_sum)
    variance = float(np.dot(weights, np.square(values - mean)) / weight_sum)
    standard_deviation = np.sqrt(max(variance, 0.0))
    if standard_deviation <= 1e-12:
        return 0.0, current_state
    return max(mean / standard_deviation, 0.0), current_state


def _factor_snapshot(
    factor_scores: dict[str, pd.DataFrame],
    factor_rank_ics: dict[str, pd.Series],
    state_histories: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    trading_dates: pd.DatetimeIndex,
    config: MultiFactorBacktestConfig,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, str]]:
    icirs: dict[str, float] = {}
    factor_states: dict[str, str] = {}
    standardized: dict[str, pd.Series] = {}
    for factor, history in state_histories.items():
        score = factor_scores[factor]
        if signal_date not in score.index:
            continue
        icir, state_name = _state_conditioned_icir(
            factor_rank_ics[factor],
            history,
            signal_date,
            trading_dates,
            config.holding_period_days,
            config.ic_halflife_days,
        )
        factor_states[factor] = state_name
        if icir <= 0.0:
            continue
        standardized[factor] = _cross_sectional_zscore(score.loc[signal_date])
        icirs[factor] = icir

    denominator = float(sum(icirs.values()))
    if denominator <= 1e-12:
        return pd.DataFrame(), {}, icirs, factor_states
    factor_weights = {factor: value / denominator for factor, value in icirs.items()}
    score_frame = pd.DataFrame(standardized).dropna(how="any")
    if score_frame.empty:
        return pd.DataFrame(), factor_weights, icirs, factor_states
    composite = sum(score_frame[factor] * factor_weights[factor] for factor in factor_weights)
    return pd.DataFrame({"composite_score": composite}), factor_weights, icirs, factor_states


def _markowitz_weights(
    scores: pd.Series,
    historical_returns: pd.DataFrame,
    risk_aversion: float,
    max_stock_weight: float,
) -> pd.Series:
    """使用 Ledoit-Wolf 协方差执行满仓、只做多、带个股上限的均值-方差优化。"""

    codes = scores.index
    if len(codes) * max_stock_weight < 1.0 - 1e-12:
        raise ValueError("候选股票数量不足以满足单只股票权重上限")
    samples = historical_returns.reindex(columns=codes).replace([np.inf, -np.inf], np.nan)
    samples = samples.dropna(how="all")
    if len(samples) < 2:
        raise ValueError("历史收益样本不足，无法估计协方差矩阵")
    samples = samples.fillna(samples.mean()).fillna(0.0)
    covariance = LedoitWolf().fit(samples.to_numpy(dtype=float)).covariance_
    expected_returns = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    initial = np.full(len(codes), 1.0 / len(codes), dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(-np.dot(expected_returns, weights) + risk_aversion * weights @ covariance @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return -expected_returns + 2.0 * risk_aversion * covariance @ weights

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, max_stock_weight)] * len(codes),
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Markowitz 优化失败: {result.message}")
    weights = np.where(result.x < 1e-10, 0.0, result.x)
    if abs(float(weights.sum()) - 1.0) > 1e-6 or float(weights.max()) > max_stock_weight + 1e-6:
        raise RuntimeError("Markowitz 优化结果未满足组合约束")
    return pd.Series(weights, index=codes, dtype=float)


def load_compact_backtest_inputs(
    config: MultiFactorBacktestConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series], pd.DataFrame, pd.Series]:
    """流式读取日文件，以宽表和月度快照控制全市场数据的内存占用。"""

    data_root = _resolve_path(config.data_root)
    files = list(_iter_daily_files(data_root))
    backtest_end = pd.Timestamp(config.end_date) if config.end_date else None
    files = [
        path
        for path in files
        if (file_date := _extract_date_from_filename(path)) is not None
        and (backtest_end is None or file_date <= backtest_end)
    ]
    if not files:
        raise FileNotFoundError("指定回测区间内未找到日线数据")
    file_dates = pd.DatetimeIndex(
        [date for path in files if (date := _extract_date_from_filename(path)) is not None]
    )
    schedule = build_rebalance_schedule(file_dates)
    signal_dates = set(pd.to_datetime(schedule["signal_date"]))
    encodings = ("gbk", "gb18030", "utf-8-sig", "utf-8")
    close_rows: list[pd.Series] = []
    turnover_rows: list[pd.Series] = []
    market_cap_rows: list[pd.Series] = []
    stock_names: dict[str, str] = {}

    for idx, file_path in enumerate(files, start=1):
        daily = _read_one_daily_panel_file(file_path, encodings)
        if daily.empty:
            continue
        trade_date = pd.Timestamp(daily["trade_date"].iloc[0])
        daily = daily.drop_duplicates("stock_code", keep="last").set_index("stock_code")
        close_rows.append(pd.to_numeric(daily["close"], errors="coerce").rename(trade_date))
        turnover = pd.to_numeric(daily["volume"], errors="coerce") / pd.to_numeric(
            daily["free_float_shares"], errors="coerce"
        ).replace(0.0, np.nan)
        turnover_rows.append(turnover.rename(trade_date))
        market_cap_rows.append(pd.to_numeric(daily["market_cap"], errors="coerce").rename(trade_date))
        if "stock_name" in daily.columns:
            stock_names.update(daily["stock_name"].dropna().astype(str).to_dict())
        if idx % 250 == 0:
            print(f"[backtest-load] 已读取日线文件 {idx}/{len(files)}")

    close = pd.DataFrame(close_rows).sort_index()
    turnover = pd.DataFrame(turnover_rows).reindex(index=close.index, columns=close.columns)
    market_cap = pd.DataFrame(market_cap_rows).reindex(columns=close.columns).sort_index()
    daily_returns = close.pct_change(fill_method=None)
    signal_index = pd.DatetimeIndex(sorted(signal_dates))
    forward_return_rank = (close.shift(-config.holding_period_days) / close - 1.0).rank(axis=1)
    factor_scores: dict[str, pd.DataFrame] = {}
    factor_rank_ics: dict[str, pd.Series] = {}

    def save_factor(factor: str, daily_score: pd.DataFrame) -> None:
        factor_rank_ics[factor] = _rank_ic_series(daily_score, forward_return_rank)
        factor_scores[factor] = daily_score.reindex(signal_index)

    if "momentum" in config.factor_names:
        momentum = close.shift(21) / close.shift(273) - 1.0
        save_factor("momentum", momentum)
        del momentum

    if "lowvol" in config.factor_names:
        rolling_volatility = daily_returns.rolling(252, min_periods=126).std()
        save_factor("lowvol", -rolling_volatility)
        del rolling_volatility

    if "size" in config.factor_names:
        size = -np.log(market_cap.replace(0.0, np.nan))
        save_factor("size", size)
        del size

    rolling_turnover = turnover.rolling(
        config.turnover_window_days,
        min_periods=max(5, config.turnover_window_days // 2),
    ).mean()
    if "liquidity" in config.factor_names:
        save_factor("liquidity", rolling_turnover)
    del turnover

    fundamental_factors = set(config.factor_names) & {"value", "quality", "growth"}
    if fundamental_factors:
        daily_panel = market_cap.stack().rename("market_cap").dropna().reset_index()
        daily_panel.columns = ["trade_date", "stock_code", "market_cap"]
        for factor in sorted(fundamental_factors):
            daily_score = build_factor_score(
                daily_panel,
                CrossSectionalFactorConfig(
                    factor_name=factor,
                    data_root=config.data_root,
                    financial_root=config.financial_root,
                ),
            )
            save_factor(factor, daily_score)
            del daily_score

    names = pd.Series(stock_names, dtype=str)
    return daily_returns, factor_scores, factor_rank_ics, rolling_turnover.reindex(signal_index), names


def build_monthly_targets(
    schedule: pd.DataFrame,
    daily_returns: pd.DataFrame,
    factor_scores: dict[str, pd.DataFrame],
    factor_rank_ics: dict[str, pd.Series],
    state_histories: dict[str, pd.DataFrame],
    rolling_turnover: pd.DataFrame,
    stock_names: pd.Series,
    config: MultiFactorBacktestConfig,
) -> pd.DataFrame:
    """按月生成各风险厌恶系数下的 Markowitz 目标持仓。"""

    rows: list[dict[str, object]] = []
    for period in schedule.itertuples(index=False):
        signal_date = pd.Timestamp(period.signal_date)
        snapshot, factor_weights, icirs, factor_states = _factor_snapshot(
            factor_scores,
            factor_rank_ics,
            state_histories,
            signal_date,
            daily_returns.index,
            config,
        )
        if snapshot.empty or signal_date not in rolling_turnover.index:
            continue
        security_names = snapshot.index.to_series().map(stock_names)
        non_st_mask = ~security_names.fillna("").astype(str).str.contains("ST", case=False, regex=False)
        turnover = pd.to_numeric(rolling_turnover.loc[signal_date], errors="coerce")
        turnover_threshold = float(turnover.quantile(config.turnover_exclusion_quantile))
        liquid_mask = turnover.reindex(snapshot.index).ge(turnover_threshold).fillna(False)
        selected = snapshot.loc[non_st_mask & liquid_mask].nlargest(config.top_n, "composite_score")
        if len(selected) * config.max_stock_weight < 1.0 - 1e-12:
            continue
        historical_returns = daily_returns.loc[:signal_date, selected.index].tail(
            config.covariance_lookback_days
        )
        for lambda_value in config.lambdas:
            weights = _markowitz_weights(
                selected["composite_score"],
                historical_returns,
                lambda_value,
                config.max_stock_weight,
            )
            for stock_code, weight in weights.items():
                if weight <= 0.0:
                    continue
                rows.append(
                    {
                        "lambda": lambda_value,
                        "rebalance_date": period.rebalance_date,
                        "signal_date": period.signal_date,
                        "period_end": period.period_end,
                        "stock_code": str(stock_code),
                        "stock_name": stock_names.get(stock_code, ""),
                        "weight": float(weight),
                        "composite_score": float(selected.loc[stock_code, "composite_score"]),
                        "active_factors": ",".join(factor_weights),
                        "factor_weights": json.dumps(factor_weights, ensure_ascii=False),
                        "state_conditioned_icir": json.dumps(icirs, ensure_ascii=False),
                        "factor_states": json.dumps(factor_states, ensure_ascii=False),
                    }
                )
    return pd.DataFrame(rows)


def simulate_monthly_portfolio(
    daily_returns: pd.DataFrame,
    targets: pd.DataFrame,
    lambda_value: float,
) -> pd.DataFrame:
    """按目标权重买入并在月内自然漂移，月初再平衡。"""

    selected = targets[np.isclose(targets["lambda"], lambda_value)].copy()
    rows: list[dict[str, object]] = []
    for rebalance_date, holdings in selected.groupby("rebalance_date", sort=True):
        period_end = pd.Timestamp(holdings["period_end"].iloc[0])
        codes = holdings["stock_code"].astype(str).tolist()
        values = holdings.set_index("stock_code")["weight"].reindex(codes).to_numpy(dtype=float, copy=True)
        period_returns = daily_returns.loc[
            (daily_returns.index >= pd.Timestamp(rebalance_date)) & (daily_returns.index <= period_end), codes
        ]
        for trade_date, return_row in period_returns.iterrows():
            before = float(values.sum())
            stock_returns = pd.to_numeric(return_row, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            values *= 1.0 + stock_returns
            after = float(values.sum())
            rows.append(
                {
                    "trade_date": trade_date,
                    "strategy_return": after / before - 1.0 if before > 0 else 0.0,
                    "lambda": lambda_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    result["nav"] = (1.0 + result["strategy_return"]).cumprod()
    return result.reset_index(drop=True)


def compute_portfolio_turnover(
    daily_returns: pd.DataFrame,
    targets: pd.DataFrame,
    lambda_value: float,
) -> float:
    """计算各调仓期单边换手率的均值，首期建仓记为100%。"""

    selected = targets[np.isclose(targets["lambda"], lambda_value)].copy()
    if selected.empty:
        return float("nan")

    period_turnovers: list[float] = []
    pre_rebalance_weights: pd.Series | None = None
    for rebalance_date, holdings in selected.groupby("rebalance_date", sort=True):
        target_weights = holdings.set_index("stock_code")["weight"].astype(float)
        target_weights.index = target_weights.index.astype(str)
        target_weights = target_weights.groupby(level=0).sum()
        target_weights /= float(target_weights.sum())

        if pre_rebalance_weights is None:
            period_turnovers.append(1.0)
        else:
            all_codes = target_weights.index.union(pre_rebalance_weights.index)
            weight_change = target_weights.reindex(all_codes, fill_value=0.0) - pre_rebalance_weights.reindex(
                all_codes, fill_value=0.0
            )
            period_turnovers.append(0.5 * float(weight_change.abs().sum()))

        period_end = pd.Timestamp(holdings["period_end"].iloc[0])
        period_returns = daily_returns.loc[
            (daily_returns.index >= pd.Timestamp(rebalance_date)) & (daily_returns.index <= period_end),
            target_weights.index,
        ]
        growth = (1.0 + period_returns.fillna(0.0)).prod(axis=0)
        ending_values = target_weights * growth
        pre_rebalance_weights = ending_values / float(ending_values.sum())

    return float(np.mean(period_turnovers))


def compute_backtest_metrics(returns: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    ret = pd.to_numeric(returns["strategy_return"], errors="coerce").dropna()
    if ret.empty:
        return {"AnnualReturn": float("nan"), "Sharpe": float("nan"), "MDD": float("nan")}
    nav = (1.0 + ret).cumprod()
    annual_return = float(nav.iloc[-1] ** (annualization / len(ret)) - 1.0)
    annual_volatility = float(ret.std(ddof=0) * np.sqrt(annualization))
    sharpe = annual_return / annual_volatility if annual_volatility > 1e-12 else float("nan")
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {"AnnualReturn": annual_return, "Sharpe": sharpe, "MDD": max_drawdown}


def add_benchmark_nav(returns: pd.DataFrame, market_returns: pd.DataFrame) -> pd.DataFrame:
    """按策略交易日对齐沪深300收益，并生成基准累计净值。"""

    required = {"trade_date", "market_return"}
    missing = required - set(market_returns.columns)
    if missing:
        raise ValueError(f"沪深300收益缺少列: {missing}")

    benchmark = market_returns[["trade_date", "market_return"]].copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"], errors="coerce")
    benchmark["market_return"] = pd.to_numeric(benchmark["market_return"], errors="coerce")
    benchmark = benchmark.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")

    out = returns.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.merge(benchmark, on="trade_date", how="left").rename(
        columns={"market_return": "benchmark_return"}
    )
    if out["benchmark_return"].notna().sum() == 0:
        raise ValueError("策略交易日无法匹配沪深300收益")
    out["benchmark_return"] = out["benchmark_return"].fillna(0.0)
    out["benchmark_nav"] = (1.0 + out["benchmark_return"]).cumprod()
    return out


def _plot_nav(returns: pd.DataFrame, output_path: Path, lambda_value: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("未安装 matplotlib，无法输出收益图") from err

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        returns["trade_date"],
        returns["nav"],
        color="#176B87",
        linewidth=1.8,
        label="多因子策略",
    )
    ax.plot(
        returns["trade_date"],
        returns["benchmark_nav"],
        color="#D97706",
        linewidth=1.5,
        label="沪深300",
    )
    ax.set_title(f"多因子状态配置策略净值（lambda={lambda_value:.1f}）")
    ax.set_xlabel("交易日期")
    ax.set_ylabel("累计净值")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_multifactor_backtest(config: MultiFactorBacktestConfig | None = None) -> dict[str, object]:
    """执行回测、选择累计收益最高的 lambda 并输出持仓与绩效。"""

    cfg = config or MultiFactorBacktestConfig()
    if cfg.top_n <= 0 or not cfg.lambdas or any(value <= 0 for value in cfg.lambdas):
        raise ValueError("top_n 必须为正，lambdas 必须为非空正数序列")
    if cfg.holding_period_days <= 0 or cfg.ic_halflife_days <= 0 or cfg.turnover_window_days <= 0:
        raise ValueError("持有周期、IC 半衰期和换手率窗口必须为正")
    if not 0.0 < cfg.turnover_exclusion_quantile < 1.0:
        raise ValueError("换手率剔除分位数必须在 0 和 1 之间")
    if not 0.0 < cfg.max_stock_weight <= 1.0 or cfg.top_n * cfg.max_stock_weight < 1.0:
        raise ValueError("top_n 与单只股票权重上限无法满足满仓约束")

    daily_returns, factor_scores, factor_rank_ics, rolling_turnover, stock_names = (
        load_compact_backtest_inputs(cfg)
    )
    histories = load_factor_state_history(cfg)

    schedule = build_rebalance_schedule(daily_returns.index)
    if cfg.start_date:
        schedule = schedule[schedule["rebalance_date"] >= pd.Timestamp(cfg.start_date)]
    if cfg.end_date:
        end_date = pd.Timestamp(cfg.end_date)
        schedule = schedule[schedule["rebalance_date"] <= end_date].copy()
        schedule["period_end"] = schedule["period_end"].clip(upper=end_date)
    targets = build_monthly_targets(
        schedule,
        daily_returns,
        factor_scores,
        factor_rank_ics,
        histories,
        rolling_turnover,
        stock_names,
        cfg,
    )
    if targets.empty:
        raise ValueError("未生成任何持仓，请检查回测区间、状态文件和因子分值")

    all_returns: dict[float, pd.DataFrame] = {}
    metric_rows: list[dict[str, float]] = []
    for lambda_value in cfg.lambdas:
        returns = simulate_monthly_portfolio(daily_returns, targets, lambda_value)
        metrics = compute_backtest_metrics(returns, cfg.annualization)
        metrics["Turnover"] = compute_portfolio_turnover(daily_returns, targets, lambda_value)
        all_returns[lambda_value] = returns
        metric_rows.append({"lambda": lambda_value, "TotalReturn": float(returns["nav"].iloc[-1] - 1.0), **metrics})

    lambda_results = pd.DataFrame(metric_rows).sort_values("lambda")
    best_lambda = float(lambda_results.loc[lambda_results["TotalReturn"].idxmax(), "lambda"])
    market_returns = load_market_returns(str(_resolve_path(cfg.market_path)))
    best_returns = add_benchmark_nav(all_returns[best_lambda], market_returns)
    best_holdings = targets[np.isclose(targets["lambda"], best_lambda)].copy()

    output_dir = _resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lambda_results.to_csv(output_dir / "lambda_results.csv", index=False, encoding="utf-8-sig")
    best_returns.to_csv(output_dir / "daily_returns.csv", index=False, encoding="utf-8-sig")
    best_holdings.to_csv(output_dir / "monthly_holdings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factor_rank_ics).rename_axis("trade_date").to_csv(
        output_dir / "rank_ic.csv", encoding="utf-8-sig"
    )
    best_metrics = lambda_results[np.isclose(lambda_results["lambda"], best_lambda)].iloc[0].to_dict()
    (output_dir / "metrics.json").write_text(
        json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_nav(best_returns, output_dir / "cumulative_return.png", best_lambda)
    return {
        "best_lambda": best_lambda,
        "metrics": best_metrics,
        "holdings": best_holdings,
        "returns": best_returns,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="状态驱动的多因子月频选股回测")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    result = run_multifactor_backtest(
        MultiFactorBacktestConfig(start_date=args.start_date, end_date=args.end_date, top_n=args.top_n)
    )
    print(f"[done] 最优 lambda={result['best_lambda']:.1f}, 输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()