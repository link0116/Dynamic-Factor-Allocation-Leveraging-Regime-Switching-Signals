"""
策略评价指标模块。

说明：
- 指标服务于SJM超参数选择（主目标：Sharpe最大化）。
- 附加输出用于诊断状态稳定性与交易可实施性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(nav: pd.Series) -> float:
    """最大回撤。

    数学原理：
    - $MDD = \\min_t (NAV_t / \\max_{\\tau\\le t} NAV_\\tau - 1)$。
    """

    if nav.empty:
        return float("nan")
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _run_lengths(mask: np.ndarray) -> list[int]:
    """计算布尔序列为True的连续段长度。"""

    lengths: list[int] = []
    curr = 0
    for flag in mask:
        if flag:
            curr += 1
        else:
            if curr > 0:
                lengths.append(curr)
            curr = 0
    if curr > 0:
        lengths.append(curr)
    return lengths


def _average_holding_time(position: pd.Series) -> float:
    """平均持有期（按仓位符号段计算）。

    工程实现说明：
    - 论文未给出统一定义，实践中常用“同向持仓连续天数”均值。
    - 这里对sign(position)分段，统计非0段长度均值。
    """

    sign = np.sign(position.fillna(0.0).to_numpy())
    if len(sign) == 0:
        return float("nan")

    lengths: list[int] = []
    curr = 1
    for i in range(1, len(sign)):
        if sign[i] == sign[i - 1]:
            curr += 1
        else:
            if sign[i - 1] != 0:
                lengths.append(curr)
            curr = 1
    if sign[-1] != 0:
        lengths.append(curr)

    return float(np.mean(lengths)) if lengths else float("nan")


def compute_strategy_metrics(strategy_df: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    """计算策略与状态指标。

    数学原理：
    - 年化收益：$\\prod_t(1+r_t)^{252/T}-1$
    - 年化波动：$\sigma(r)\sqrt{252}$
    - Sharpe：年化收益/年化波动（无风险利率近似0）
    - IR：策略收益均值/收益标准差（工程实现，近似日频信息比）
    - Turnover：$\frac{1}{T}\sum_t |p_t-p_{t-1}|$

    为什么这样做：
    - 这些指标分别刻画收益、风险、效率、交易成本压力与状态稳定性。

    论文对应：
    - 以Sharpe作为超参数选择主目标；
    - 其余指标用于结果解释与稳健性分析。

    输入输出：
    - 输入：需包含 strategy_return, position, state, state_name
    - 输出：指标字典
    """

    required = {"strategy_return", "position", "state", "state_name"}
    missing = required - set(strategy_df.columns)
    if missing:
        raise ValueError(f"评价指标缺少列: {missing}")

    df = strategy_df.copy()
    ret = pd.to_numeric(df["strategy_return"], errors="coerce").fillna(0.0)
    pos = pd.to_numeric(df["position"], errors="coerce").fillna(0.0)

    if len(ret) == 0:
        return {
            "AnnualReturn": float("nan"),
            "AnnualVolatility": float("nan"),
            "Sharpe": float("nan"),
            "IR": float("nan"),
            "MDD": float("nan"),
            "Turnover": float("nan"),
            "AverageHoldingTime": float("nan"),
            "BullDuration": float("nan"),
            "BearDuration": float("nan"),
            "StateSwitchTimes": float("nan"),
        }

    nav = (1.0 + ret).cumprod()
    years = max(len(ret) / annualization, 1e-9)
    total_ret = float(nav.iloc[-1] - 1.0)
    ann_ret = float((1.0 + total_ret) ** (1.0 / years) - 1.0)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(annualization))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 1e-12 else float("nan")

    ir_den = float(ret.std(ddof=0))
    ir = float(ret.mean() / ir_den) if ir_den > 1e-12 else float("nan")

    mdd = _max_drawdown(nav)
    turnover = float(pos.diff().abs().fillna(0.0).mean())
    avg_holding = _average_holding_time(pos)

    state_arr = df["state"].astype(int).to_numpy()
    state_switch = float(np.sum(state_arr[1:] != state_arr[:-1])) if len(state_arr) > 1 else 0.0

    name_arr = df["state_name"].astype(str).to_numpy()
    bull_lengths = _run_lengths(name_arr == "Bull")
    bear_lengths = _run_lengths(name_arr == "Bear")
    bull_duration = float(np.mean(bull_lengths)) if bull_lengths else float("nan")
    bear_duration = float(np.mean(bear_lengths)) if bear_lengths else float("nan")

    return {
        "AnnualReturn": ann_ret,
        "AnnualVolatility": ann_vol,
        "Sharpe": sharpe,
        "IR": ir,
        "MDD": mdd,
        "Turnover": turnover,
        "AverageHoldingTime": avg_holding,
        "BullDuration": bull_duration,
        "BearDuration": bear_duration,
        "StateSwitchTimes": state_switch,
    }
