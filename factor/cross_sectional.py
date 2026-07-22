"""Generic cross-sectional factor return construction.

This module implements the paper-style first step for single-factor regime
identification: build one daily long-short factor portfolio return series from
one factor value.  The signal is observed before the traded return, so the
portfolio uses the previous available signal to trade the next daily return.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from factor.common import _iter_daily_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACTOR_LIST: list[str] = ["momentum", "value", "quality", "growth", "size", "lowvol", "liquidity"]
_DAILY_PANEL_CACHE: dict[tuple[Path, tuple[str, ...]], pd.DataFrame] = {}
_FINANCIAL_PANEL_CACHE: dict[tuple[Path, int, tuple[str, ...]], pd.DataFrame] = {}


@dataclass
class CrossSectionalFactorConfig:
    """Configuration for a paper-style long-short factor portfolio.

    Parameters
    ----------
    factor_name:
        Canonical factor name. Supported values are momentum, value, quality,
        size, liquidity, lowvol, and growth.
    data_root:
        Directory containing daily A-share CSV files.
    financial_root:
        Directory containing quarterly financial CSV files. Only required for
        value, quality, and growth.
    output_path:
        Destination CSV for the daily factor return series.
    top_ratio:
        Cross-sectional quantile used for long and short legs.
    signal_lag_days:
        Number of trading days between signal observation and traded return.
        A value of 1 means signal at t-1 trades return at t.
    financial_lag_days:
        Conservative reporting lag applied to financial statements, because the
        local files provide report periods but not announcement dates.
    """

    factor_name: str
    data_root: str = "data/A股日线指标"
    financial_root: str = "data/A股财务数据"
    output_path: str = "outputs/factor_return.csv"
    top_ratio: float = 0.20
    signal_lag_days: int = 1
    financial_lag_days: int = 90
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    lowvol_lookback_days: int = 252
    liquidity_lookback_days: int = 21
    min_universe_size: int = 30
    incremental: bool = False
    encoding_candidates: tuple[str, ...] = ("gbk", "gb18030", "utf-8-sig", "utf-8")


def _resolve_path(path: str) -> Path:
    """Return an absolute path relative to the project root when needed."""

    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read_csv_with_candidates(path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    """Read a CSV using the first encoding that succeeds."""

    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as err:  # noqa: BLE001
            last_err = err
    raise ValueError(f"读取CSV失败: {path}, last_error={last_err}")


def _read_one_daily_panel_file(file_path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    """Read one daily market file and standardize fields used by factors."""

    required_alias = {
        "trade_date": ["日期"],
        "stock_code": ["代码"],
        "stock_name": ["名称"],
        "close": ["日收盘价"],
        "volume": ["日交易量"],
        "free_float_shares": ["流通股本"],
        "total_shares": ["总股本"],
        "market_cap": ["总市值(万)", "总市值"],
    }

    raw = _read_csv_with_candidates(file_path, encodings)
    col_map: dict[str, str] = {}
    for std_col, aliases in required_alias.items():
        for alias in aliases:
            if alias in raw.columns:
                col_map[alias] = std_col
                break

    needed = {"trade_date", "stock_code", "close", "volume", "free_float_shares", "market_cap"}
    if not needed.issubset(set(col_map.values())):
        raise ValueError(f"日线文件缺少必要列: {file_path}, columns={list(raw.columns)}")

    keep_cols = list({v: None for v in col_map.values()}.keys())
    out = raw.rename(columns=col_map)[keep_cols].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["stock_code"] = out["stock_code"].astype(str).str.strip()
    for col in ["close", "volume", "free_float_shares", "total_shares", "market_cap"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["trade_date", "stock_code", "close", "market_cap"])
    return out


def load_daily_factor_panel(config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Load the daily market panel needed for cross-sectional factor values."""

    data_root = _resolve_path(config.data_root)
    cache_key = (data_root, config.encoding_candidates)
    cached = _DAILY_PANEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not data_root.exists():
        raise FileNotFoundError(f"日线数据目录不存在: {data_root}")

    files = list(_iter_daily_files(data_root))
    if not files:
        raise FileNotFoundError(f"未找到日线CSV文件: {data_root}")

    parts: list[pd.DataFrame] = []
    for idx, file_path in enumerate(files, start=1):
        parts.append(_read_one_daily_panel_file(file_path, config.encoding_candidates))
        if idx % 250 == 0:
            print(f"[load] 已读取日线文件 {idx}/{len(files)}")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["stock_code", "trade_date"]).drop_duplicates(
        subset=["trade_date", "stock_code"],
        keep="last",
    )
    panel["ret"] = panel.groupby("stock_code", sort=False)["close"].pct_change()
    panel = panel.reset_index(drop=True)
    _DAILY_PANEL_CACHE[cache_key] = panel
    return panel


def load_financial_panel(config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Load and standardize quarterly financial statement data."""

    root = _resolve_path(config.financial_root)
    cache_key = (root, config.financial_lag_days, config.encoding_candidates)
    cached = _FINANCIAL_PANEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not root.exists():
        raise FileNotFoundError(f"财务数据目录不存在: {root}")

    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到财务CSV文件: {root}")

    parts = [_read_csv_with_candidates(path, config.encoding_candidates) for path in files]
    raw = pd.concat(parts, ignore_index=True)
    raw.columns = [str(c).strip() for c in raw.columns]

    rename_map = {
        "报告期": "report_period",
        "代码": "stock_code",
        "名称": "stock_name",
        "净利润": "net_profit",
        "经营现金流": "operating_cashflow",
        "营业收入": "revenue",
        "EPS": "eps",
        "总资产": "total_assets",
        "总负债": "total_liabilities",
        "长期负债": "long_term_liabilities",
        "股东权益": "shareholder_equity",
    }
    missing = [col for col in rename_map if col not in raw.columns]
    if missing:
        raise ValueError(f"财务数据缺少必要列: {missing}")

    df = raw.rename(columns=rename_map)[list(rename_map.values())].copy()
    df["report_period"] = pd.to_datetime(df["report_period"].astype(str), format="%Y%m%d", errors="coerce")
    df["available_date"] = df["report_period"] + pd.to_timedelta(config.financial_lag_days, unit="D")
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    numeric_cols = [
        "net_profit",
        "operating_cashflow",
        "revenue",
        "eps",
        "total_assets",
        "total_liabilities",
        "long_term_liabilities",
        "shareholder_equity",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["stock_code", "report_period", "available_date"])
    df = df.sort_values(["stock_code", "report_period"]).reset_index(drop=True)

    grouped = df.groupby("stock_code", sort=False)
    df["revenue_yoy"] = grouped["revenue"].pct_change(periods=4)
    df["net_profit_yoy"] = grouped["net_profit"].pct_change(periods=4)
    df["roe"] = df["net_profit"] / df["shareholder_equity"].replace(0.0, np.nan)
    df["cfo_to_assets"] = df["operating_cashflow"] / df["total_assets"].replace(0.0, np.nan)
    df["debt_to_assets"] = df["total_liabilities"] / df["total_assets"].replace(0.0, np.nan)
    _FINANCIAL_PANEL_CACHE[cache_key] = df
    return df


def _pivot(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a long daily panel into date by stock format."""

    return panel.pivot(index="trade_date", columns="stock_code", values=value_col).sort_index()


def _build_price_based_score(panel: pd.DataFrame, config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Build factor scores that only require daily market data."""

    factor = config.factor_name.lower()
    close = _pivot(panel, "close")
    ret = _pivot(panel, "ret")
    market_cap = _pivot(panel, "market_cap")

    if factor == "momentum":
        return close.shift(config.momentum_skip_days) / close.shift(
            config.momentum_lookback_days + config.momentum_skip_days
        ) - 1.0

    if factor == "lowvol":
        volatility = ret.rolling(config.lowvol_lookback_days, min_periods=config.lowvol_lookback_days // 2).std()
        return -volatility

    if factor == "size":
        return -np.log(market_cap.replace(0.0, np.nan))

    if factor == "liquidity":
        volume = _pivot(panel, "volume")
        free_float = _pivot(panel, "free_float_shares")
        turnover = volume / free_float.replace(0.0, np.nan)
        return turnover.rolling(
            config.liquidity_lookback_days,
            min_periods=max(5, config.liquidity_lookback_days // 2),
        ).mean()

    raise ValueError(f"{factor} 不是价格/交易量类因子")


def _merge_latest_financials(panel: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
    """Attach latest available financial statements to each stock-date row."""

    daily = panel[["stock_code", "trade_date", "market_cap"]].copy()
    daily = daily.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    fin = financial.sort_values(["stock_code", "available_date"]).reset_index(drop=True)

    merged_parts: list[pd.DataFrame] = []
    for stock_code, daily_one in daily.groupby("stock_code", sort=False):
        fin_one = fin[fin["stock_code"] == stock_code]
        if fin_one.empty:
            continue
        merged = pd.merge_asof(
            daily_one.sort_values("trade_date"),
            fin_one.sort_values("available_date"),
            left_on="trade_date",
            right_on="available_date",
            by="stock_code",
            direction="backward",
        )
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame(columns=list(daily.columns) + list(financial.columns))
    return pd.concat(merged_parts, ignore_index=True)


def _build_fundamental_score(panel: pd.DataFrame, config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Build value, quality, or growth scores from latest available financials."""

    factor = config.factor_name.lower()
    financial = load_financial_panel(config)
    merged = _merge_latest_financials(panel, financial)
    if merged.empty:
        raise ValueError(f"{factor} 因子无法匹配到可用财务数据")

    market_cap_yuan = merged["market_cap"] * 10000.0
    if factor == "value":
        merged["factor_score"] = merged["shareholder_equity"] / market_cap_yuan.replace(0.0, np.nan)
    elif factor == "quality":
        merged["factor_score"] = (
            merged["roe"].replace([np.inf, -np.inf], np.nan)
            + merged["cfo_to_assets"].replace([np.inf, -np.inf], np.nan)
            - merged["debt_to_assets"].replace([np.inf, -np.inf], np.nan)
        )
    elif factor == "growth":
        merged["factor_score"] = (
            merged["revenue_yoy"].replace([np.inf, -np.inf], np.nan)
            + merged["net_profit_yoy"].replace([np.inf, -np.inf], np.nan)
        ) / 2.0
    else:
        raise ValueError(f"{factor} 不是财务类因子")

    return _pivot(merged[["trade_date", "stock_code", "factor_score"]], "factor_score")


def build_factor_score(panel: pd.DataFrame, config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Build the cross-sectional factor value matrix for one factor."""

    factor = config.factor_name.lower()
    if factor in {"momentum", "lowvol", "size", "liquidity"}:
        return _build_price_based_score(panel, config)
    if factor in {"value", "quality", "growth"}:
        return _build_fundamental_score(panel, config)
    raise ValueError("factor_name 仅支持 momentum/value/quality/size/liquidity/lowvol/growth")


def build_long_short_factor_return(
    panel: pd.DataFrame,
    factor_score: pd.DataFrame,
    config: CrossSectionalFactorConfig,
) -> pd.DataFrame:
    """Build daily top-bottom long-short factor return from factor scores."""

    ret = _pivot(panel, "ret")
    score = factor_score.reindex(ret.index).shift(config.signal_lag_days)
    signal_dates = pd.Series(score.index, index=score.index).shift(config.signal_lag_days)
    return_col = f"{config.factor_name.lower()}_return"

    rows: list[dict[str, object]] = []
    for trade_date in ret.index:
        score_row = score.loc[trade_date].replace([np.inf, -np.inf], np.nan)
        ret_row = ret.loc[trade_date].replace([np.inf, -np.inf], np.nan)
        valid = score_row.notna() & ret_row.notna()
        if int(valid.sum()) < config.min_universe_size:
            continue

        score_valid = score_row[valid]
        ret_valid = ret_row[valid]
        n_leg = max(1, int(np.floor(len(score_valid) * config.top_ratio)))
        long_codes = score_valid.nlargest(n_leg).index
        short_codes = score_valid.nsmallest(n_leg).index

        long_return = float(ret_valid.loc[long_codes].mean())
        short_return = float(ret_valid.loc[short_codes].mean())
        rows.append(
            {
                "trade_date": trade_date,
                return_col: long_return - short_return,
                "long_return": long_return,
                "short_return": short_return,
                "long_count": int(len(long_codes)),
                "short_count": int(len(short_codes)),
                "holdings_count": int(len(long_codes) + len(short_codes)),
                "signal_date": signal_dates.loc[trade_date],
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                return_col,
                "long_return",
                "short_return",
                "long_count",
                "short_count",
                "holdings_count",
                "signal_date",
            ]
        )

    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return result.sort_values("trade_date").reset_index(drop=True)


def build_factor_return(config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Execute Step1: build one daily long-short factor return series."""

    panel = load_daily_factor_panel(config)
    factor_score = build_factor_score(panel, config)
    factor_return = build_long_short_factor_return(panel, factor_score, config)
    if factor_return.empty:
        raise ValueError(f"{config.factor_name} 因子收益为空，请检查数据和窗口设置")
    return factor_return


def run_cross_sectional_factor_pipeline(config: CrossSectionalFactorConfig) -> pd.DataFrame:
    """Build and save one long-short factor return CSV."""

    out_path = _resolve_path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_factor_return(config)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[done] {config.factor_name} long-short因子收益输出: {out_path}, 共 {len(result):,} 行")
    return result


def batch_run_cross_sectional_pipeline(
    config: Optional[CrossSectionalFactorConfig] = None,
    factor_list: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Batch build and save long-short factor return CSVs for all factors."""

    base_config = config or CrossSectionalFactorConfig(factor_name=FACTOR_LIST[0])
    factors = factor_list or FACTOR_LIST
    output_root = _resolve_path("outputs/factor_returns")
    output_root.mkdir(parents=True, exist_ok=True)

    # Warm caches once so all factors share loaded daily and financial data.
    load_daily_factor_panel(base_config)
    load_financial_panel(base_config)

    results: dict[str, pd.DataFrame] = {}
    for factor in factors:
        run_config = replace(
            base_config,
            factor_name=factor,
            output_path=str(output_root / f"{factor}.csv"),
        )
        result = run_cross_sectional_factor_pipeline(run_config)
        results[factor] = result

        print("=" * 48)
        print(factor.capitalize())
        print("完成")
        print(f"共{len(result):,}天")
        print("=" * 48)

    print("=" * 48)
    print("全部7个因子收益序列生成完成")
    print("=" * 48)
    return results


def main() -> None:
    """Module CLI entry for batch factor return generation."""

    batch_run_cross_sectional_pipeline()


if __name__ == "__main__":
    main()

