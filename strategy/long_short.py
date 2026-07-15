"""
论文单因子多空策略模块。

核心规则：
1) Bull: Long Momentum, Short Market。
2) Bear: Short Momentum, Long Market。
3) 预期收益位于[-5%, 5%]时使用线性仓位映射。

工程扩展：
- 当K>2时，按每个状态的预期主动收益决定多空方向与仓位强度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_state_rank_mapping_by_train_active_return(
        train_state: np.ndarray,
        train_active_return: np.ndarray,
) -> dict[int, int]:
        """按训练期主动收益均值对状态编号重排。

        数学原理：
        - 先估计每个状态的训练期均值 $\\mu_s = E[r^{active}|state=s]$；
        - 再按 $\\mu_s$ 从低到高排序，映射到连续编号 0..K-1。

        设计原因：
        - SJM原始状态编号是无语义标签，可能在不同训练轮次互换；
        - 用训练期收益排序后，编号具有稳定经济含义：
            编号越小越偏Bear，编号越大越偏Bull。

        论文对应：
        - 对应“Bull/Bear按状态收益特征定义”的实现落地。

        输入输出：
        - 输入：train_state, train_active_return
        - 输出：old_state -> new_state 的映射字典
        """

        stat = build_state_expected_return_map(train_state, train_active_return)
        ordered = sorted(stat.items(), key=lambda kv: kv[1])
        return {int(old_s): int(new_s) for new_s, (old_s, _) in enumerate(ordered)}


def linear_position_mapping(expected_active_return: float, band: float = 0.05) -> float:
    """把“状态预期主动收益”映射为仓位。

    数学原理：
    - 采用线性函数 $p = \\mathrm{clip}(\\mu/\\delta, -1, 1)$。
    其中 $\\mu$ 是该状态预期主动收益，$\\delta=5\\%$ 为映射带宽。

    为什么这样做：
    - 当预期收益接近0时降低仓位，避免噪声状态下过度交易；
    - 当预期收益足够大时饱和到满仓（+1/-1），与论文状态驱动思想一致。

    论文对应：
    - 对应用户要求中的“[-5%,5%]区间线性仓位映射”。

    输入输出：
    - 输入：expected_active_return（小数，如0.01）
    - 输出：position in [-1, 1]
      +1 表示 Long Momentum / Short Market；
      -1 表示 Short Momentum / Long Market。
    """

    if band <= 0:
        raise ValueError("band必须为正")
    return float(np.clip(expected_active_return / band, -1.0, 1.0))


def build_state_expected_return_map(
    train_state: np.ndarray,
    train_active_return: np.ndarray,
) -> dict[int, float]:
    """用训练样本估计“状态->预期主动收益”。

    数学原理：
    - 经验估计 $\\hat{\\mu}_s = E[r^{active}|state=s]$。

    为什么这样做：
    - 状态标签本身无语义（只是编号），
      必须通过训练期收益把状态映射成 Bull/Bear/中性强弱。

    论文对应：
    - 对应“状态识别后用于因子多空配置”的桥接步骤。

    输入输出：
    - 输入：训练期状态序列、训练期主动收益序列
    - 输出：{state_id: expected_active_return}
    """

    df = pd.DataFrame(
        {
            "state": pd.Series(train_state, dtype=int),
            "active_return": pd.Series(train_active_return, dtype=float),
        }
    )
    grouped = df.groupby("state", observed=True)["active_return"].mean()
    return {int(k): float(v) for k, v in grouped.items()}


def assign_state_name_from_expected_return(state_expected_return: dict[int, float]) -> dict[int, str]:
    """把状态编号映射为语义名称。

    规则：
    - K=2: 低均值为Bear，高均值为Bull。
    - K>2: 最低为Bear，最高为Bull，中间为Neutral_i。
    """

    if not state_expected_return:
        return {}

    ordered = sorted(state_expected_return.items(), key=lambda kv: kv[1])
    if len(ordered) == 2:
        return {ordered[0][0]: "Bear", ordered[1][0]: "Bull"}

    mapping: dict[int, str] = {}
    for idx, (state, _) in enumerate(ordered):
        if idx == 0:
            mapping[state] = "Bear"
        elif idx == len(ordered) - 1:
            mapping[state] = "Bull"
        else:
            mapping[state] = f"Neutral_{idx}"
    return mapping


def build_long_short_returns(
    data: pd.DataFrame,
    state_col: str,
    state_expected_return: dict[int, float],
    position_band: float = 0.05,
) -> pd.DataFrame:
    """根据状态路径构建单因子多空策略收益。

    数学原理：
    - 主动收益定义：$r_t^{active} = r_t^{mom} - r_t^{mkt}$。
    - 策略收益：$r_t^{strat} = p_t \\cdot r_t^{active}$。
      其中 $p_t$ 由所属状态的预期主动收益线性映射而来。

    为什么这样做：
    - 该策略把“状态识别”直接转成“因子暴露方向与强度”，
      是论文框架最核心的落地环节。

    论文对应：
    - Bull时做多动量空市场；Bear时反向；
      并在小预期收益区域使用线性缩放。

    输入输出：
    - 输入data至少包含：trade_date, momentum_return, market_return, state_col
    - 输出新增列：active_return, expected_active_return, position, strategy_return, state_name
    """

    required = {"trade_date", "momentum_return", "market_return", state_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"构建多空收益缺少列: {missing}")

    out = data.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date", state_col]).sort_values("trade_date").reset_index(drop=True)

    out["active_return"] = out["momentum_return"].astype(float) - out["market_return"].astype(float)
    out["expected_active_return"] = out[state_col].astype(int).map(lambda s: state_expected_return.get(int(s), 0.0))
    out["position"] = out["expected_active_return"].map(lambda x: linear_position_mapping(float(x), band=position_band))
    out["strategy_return"] = out["position"] * out["active_return"]

    state_name_map = assign_state_name_from_expected_return(state_expected_return)
    out["state_name"] = out[state_col].astype(int).map(lambda s: state_name_map.get(int(s), f"State_{int(s)}"))
    return out
