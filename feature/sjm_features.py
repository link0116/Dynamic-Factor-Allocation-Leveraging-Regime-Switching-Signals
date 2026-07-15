"""
SJM特征工程（第二步）：
基于动量组合收益与市场收益，构造论文所需状态识别输入特征，并做标准化。

对应论文《Dynamic Factor Allocation using Regime Switching Signals》的映射：
1) 先有单因子收益（第一步 momentum_return）。
2) 计算主动收益 Active Return = Momentum Return - Market Return。
3) 对主动收益及市场环境构造技术/风险特征，作为 SJM 输入矩阵。
4) 所有特征标准化后，交给 Sparse Jump Model。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SJMFeatureConfig:
    """SJM特征工程参数。

    字段说明：
    - momentum_path: 第一步输出的动量组合收益文件。
    - market_path: 市场指数数据文件（默认沪深300）。
    - output_path: 输出SJM输入特征文件。
    - ewma_windows: EWMA主动收益窗口列表。
    - rsi_windows: RSI窗口列表。
    - stoch_windows: 随机指标%K窗口列表。
    - macd_pairs: MACD快慢窗口列表，格式[(fast, slow), ...]。
    - downside_window: 下行波动窗口。
    - beta_window: Active Beta窗口。
    - standardize_mode:
        global: 全样本z-score（离线研究方便）；
        expanding: 递增窗口z-score（更贴近在线推理，避免未来函数）。
    - min_periods_ratio: 滚动类特征的最小有效样本比例。
    """

    momentum_path: str = "outputs/momentum_return.csv"
    market_path: str = "沪深300.csv"
    output_path: str = "outputs/sjm_features.csv"

    ewma_windows: tuple[int, ...] = (8, 21, 63)
    rsi_windows: tuple[int, ...] = (8, 21, 63)
    stoch_windows: tuple[int, ...] = (8, 21, 63)
    macd_pairs: tuple[tuple[int, int], ...] = ((8, 21), (21, 63))
    downside_window: int = 21
    beta_window: int = 21

    standardize_mode: str = "expanding"
    min_periods_ratio: float = 0.6


def _safe_read_csv(path: Path, encoding_candidates: tuple[str, ...]) -> pd.DataFrame:
    """按候选编码尝试读取CSV。"""

    last_err: Optional[Exception] = None
    for enc in encoding_candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"读取失败: {path}, last_error={last_err}")


def load_momentum_returns(momentum_path: str) -> pd.DataFrame:
    """读取动量组合日收益。

    预期输入列至少包含：
    - trade_date
    - momentum_return
    """

    path = Path(momentum_path)
    if not path.exists():
        raise FileNotFoundError(f"动量收益文件不存在: {path}")

    df = pd.read_csv(path)
    required = {"trade_date", "momentum_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"动量收益文件缺少必要列: {missing}")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["momentum_return"] = pd.to_numeric(df["momentum_return"], errors="coerce")
    df = df.dropna(subset=["trade_date", "momentum_return"]).copy()
    return df.sort_values("trade_date").reset_index(drop=True)


def load_market_returns(market_path: str) -> pd.DataFrame:
    """读取市场指数并构造市场日收益。

    兼容两类数据源：
    1) 已有涨跌幅列（例如“涨跌幅”带百分号）。
    2) 仅有收盘价列（例如“收盘”），则用收盘价pct_change计算。

    输出列：
    - trade_date
    - market_return
    """

    path = Path(market_path)
    if not path.exists():
        raise FileNotFoundError(f"市场数据文件不存在: {path}")

    raw = _safe_read_csv(path, encoding_candidates=("utf-8-sig", "gbk", "gb18030", "utf-8"))
    raw.columns = [str(c).strip().replace('"', "") for c in raw.columns]

    date_col_candidates = ["日期"]
    close_col_candidates = ["收盘"]
    ret_col_candidates = ["涨跌幅"]

    date_col = next((c for c in date_col_candidates if c in raw.columns), None)
    close_col = next((c for c in close_col_candidates if c in raw.columns), None)
    ret_col = next((c for c in ret_col_candidates if c in raw.columns), None)

    if date_col is None:
        raise ValueError(f"市场文件缺少日期列，当前列: {list(raw.columns)}")

    df = raw.copy()
    df["trade_date"] = pd.to_datetime(df[date_col], errors="coerce")

    market_return = None
    if ret_col is not None:
        ser = df[ret_col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
        market_return = pd.to_numeric(ser, errors="coerce") / 100.0

    if (market_return is None) or market_return.isna().all():
        if close_col is None:
            raise ValueError("市场文件既没有可用涨跌幅，也没有可用收盘列，无法构建市场收益")
        close = pd.to_numeric(
            df[close_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        market_return = close.pct_change()

    out = pd.DataFrame({"trade_date": df["trade_date"], "market_return": market_return})
    out = out.dropna(subset=["trade_date", "market_return"]).copy()
    out = out.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    return out.reset_index(drop=True)


def _rsi_from_series(level: pd.Series, window: int, min_periods: int) -> pd.Series:
    """基于价格/净值序列计算RSI。

    说明：
    - 主动收益本身是收益率，技术指标RSI通常定义在“价格”序列；
    - 因此先把主动收益累积成主动净值，再计算RSI。
    """

    delta = level.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=min_periods).mean()
    avg_loss = loss.rolling(window=window, min_periods=min_periods).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd_hist_from_series(level: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """计算MACD柱值（DIF-DEA）。"""

    ema_fast = level.ewm(span=fast, adjust=False).mean()
    ema_slow = level.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return hist


def _stochastic_k_from_series(level: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算随机指标%K。"""

    rolling_low = level.rolling(window=window, min_periods=min_periods).min()
    rolling_high = level.rolling(window=window, min_periods=min_periods).max()
    denom = (rolling_high - rolling_low).replace(0.0, np.nan)
    k = (level - rolling_low) / denom * 100.0
    return k


def _downside_deviation(ret: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算下行波动（Downside Deviation）。

    定义：sqrt(E[min(ret,0)^2])，在滚动窗口上估计。
    """

    downside_sq = np.square(np.minimum(ret, 0.0))
    return np.sqrt(downside_sq.rolling(window=window, min_periods=min_periods).mean())


def _active_beta(momentum_ret: pd.Series, market_ret: pd.Series, window: int, min_periods: int) -> pd.Series:
    """计算Active Beta。

    定义：rolling_cov(momentum_return, market_return) / rolling_var(market_return)
    """

    cov = momentum_ret.rolling(window=window, min_periods=min_periods).cov(market_ret)
    var = market_ret.rolling(window=window, min_periods=min_periods).var(ddof=0)
    return cov / var.replace(0.0, np.nan)


def build_sjm_features(base_df: pd.DataFrame, config: SJMFeatureConfig) -> pd.DataFrame:
    """从主动收益构造论文所需核心特征。

    必备特征：
    - ewma_active_return
    - rsi
    - macd
    - stoch_k
    - downside_deviation
    - active_beta
    - market_return（可选环境变量，这里默认保留）
    """

    min_periods_rsi = {
        w: max(2, int(np.ceil(w * config.min_periods_ratio))) for w in config.rsi_windows
    }
    min_periods_stoch = {
        w: max(2, int(np.ceil(w * config.min_periods_ratio))) for w in config.stoch_windows
    }
    min_periods_down = max(2, int(np.ceil(config.downside_window * config.min_periods_ratio)))
    min_periods_beta = max(2, int(np.ceil(config.beta_window * config.min_periods_ratio)))

    df = base_df.sort_values("trade_date").reset_index(drop=True).copy()
    df["active_return"] = df["momentum_return"] - df["market_return"]

    # 把主动收益累积为主动净值，便于构造技术指标。
    df["active_nav"] = (1.0 + df["active_return"].fillna(0.0)).cumprod()

    for w in config.ewma_windows:
        df[f"ewma_active_return_{w}"] = df["active_return"].ewm(span=w, adjust=False).mean()

    for w in config.rsi_windows:
        df[f"rsi_{w}"] = _rsi_from_series(df["active_nav"], window=w, min_periods=min_periods_rsi[w])

    for fast, slow in config.macd_pairs:
        signal = max(2, int(round((fast + slow) / 2)))
        df[f"macd_{fast}_{slow}"] = _macd_hist_from_series(df["active_nav"], fast=fast, slow=slow, signal=signal)

    for w in config.stoch_windows:
        df[f"stoch_k_{w}"] = _stochastic_k_from_series(df["active_nav"], window=w, min_periods=min_periods_stoch[w])

    df["downside_deviation"] = _downside_deviation(df["active_return"], window=config.downside_window, min_periods=min_periods_down)
    df["active_beta"] = _active_beta(
        df["momentum_return"],
        df["market_return"],
        window=config.beta_window,
        min_periods=min_periods_beta,
    )

    return df


def _zscore_global(s: pd.Series) -> pd.Series:
    """全样本z-score。"""

    mean = s.mean(skipna=True)
    std = s.std(ddof=0, skipna=True)
    if (std is None) or np.isnan(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def _zscore_expanding(s: pd.Series, min_periods: int = 30) -> pd.Series:
    """递增窗口z-score（更贴近在线推理，无未来函数）。"""

    mean = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std.replace(0.0, np.nan)
    return z


def standardize_features(df: pd.DataFrame, config: SJMFeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    """对特征做标准化，并输出可供SJM直接使用的列清单。"""

    feature_cols = []
    feature_cols.extend([f"ewma_active_return_{w}" for w in config.ewma_windows])
    feature_cols.extend([f"rsi_{w}" for w in config.rsi_windows])
    feature_cols.extend([f"macd_{fast}_{slow}" for fast, slow in config.macd_pairs])
    feature_cols.extend([f"stoch_k_{w}" for w in config.stoch_windows])
    feature_cols.extend([
        "downside_deviation",
        "active_beta",
        "market_return",
    ])

    out = df.copy()
    for col in feature_cols:
        if config.standardize_mode == "global":
            out[f"z_{col}"] = _zscore_global(out[col])
        elif config.standardize_mode == "expanding":
            out[f"z_{col}"] = _zscore_expanding(out[col], min_periods=30)
        else:
            raise ValueError("standardize_mode 仅支持 'global' 或 'expanding'")

    z_cols = [f"z_{c}" for c in feature_cols]
    return out, z_cols


def add_optional_macro_features(
    feature_df: pd.DataFrame,
    vix_path: Optional[str] = None,
    bond_yield_path: Optional[str] = None,
    term_spread_path: Optional[str] = None,
) -> pd.DataFrame:
    """可选环境变量接口（VIX/国债收益率/期限利差）。

    说明：
    - 当前仅提供接口和通用合并逻辑，不强制要求文件存在；
    - 外部文件需至少包含 trade_date 与 value 两列。
    """

    out = feature_df.copy()

    def merge_optional(path_str: Optional[str], rename_to: str) -> None:
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            return

        ext = pd.read_csv(path)
        if "trade_date" not in ext.columns or "value" not in ext.columns:
            return

        ext = ext[["trade_date", "value"]].copy()
        ext["trade_date"] = pd.to_datetime(ext["trade_date"], errors="coerce")
        ext[rename_to] = pd.to_numeric(ext["value"], errors="coerce")
        ext = ext.drop(columns=["value"])
        ext = ext.dropna(subset=["trade_date"]).sort_values("trade_date")

        nonlocal out
        out = out.merge(ext, on="trade_date", how="left")

    merge_optional(vix_path, "vix")
    merge_optional(bond_yield_path, "bond_yield")
    merge_optional(term_spread_path, "term_spread")
    return out


def run_feature_pipeline(config: SJMFeatureConfig) -> pd.DataFrame:
    """执行第二步全流程：读取收益 → 构造特征 → 标准化 → 输出。"""

    momentum = load_momentum_returns(config.momentum_path)
    market = load_market_returns(config.market_path)

    base = momentum.merge(market, on="trade_date", how="inner")
    base = base.sort_values("trade_date").reset_index(drop=True)

    feat = build_sjm_features(base, config)
    feat = add_optional_macro_features(feat)
    feat_std, z_cols = standardize_features(feat, config)

    # 仅保留标准化特征完整的样本，便于直接输入SJM。
    sjm_ready = feat_std.dropna(subset=z_cols).reset_index(drop=True)

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sjm_ready.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[done] SJM特征输出: {out_path}, 样本数={len(sjm_ready):,}")
    print(f"[done] 标准化特征列: {z_cols}")
    return sjm_ready


if __name__ == "__main__":
    cfg = SJMFeatureConfig(
        momentum_path="outputs/momentum_return.csv",
        market_path="沪深300.csv",
        output_path="outputs/sjm_features.csv",
        ewma_windows=(8, 21, 63),
        rsi_windows=(8, 21, 63),
        stoch_windows=(8, 21, 63),
        macd_pairs=((8, 21), (21, 63)),
        downside_window=63,
        beta_window=63,
        standardize_mode="expanding",
    )
    run_feature_pipeline(cfg)
