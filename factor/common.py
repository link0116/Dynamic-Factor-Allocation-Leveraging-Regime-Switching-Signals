"""因子收益序列构建的共用辅助函数。"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _extract_date_from_filename(file_path: Path) -> Optional[pd.Timestamp]:
    """从文件名提取交易日（YYYY-MM-DD.csv）。"""

    match = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", file_path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), errors="coerce")


def _iter_daily_files(data_root: Path, min_trade_date: Optional[pd.Timestamp] = None) -> Iterable[Path]:
    """按日期顺序迭代日线文件。"""

    all_files = sorted(data_root.rglob("*.csv"))
    if min_trade_date is None:
        yield from all_files
        return

    for file_path in all_files:
        file_date = _extract_date_from_filename(file_path)
        if file_date is not None and file_date >= min_trade_date:
            yield file_path


def _read_one_daily_file(file_path: Path, encodings: tuple[str, ...]) -> pd.DataFrame:
    """读取单个交易日CSV，并标准化为trade_date/stock_code/close。"""

    required_alias = {
        "trade_date": ["日期"],
        "stock_code": ["代码"],
        "close": ["日收盘价"],
    }

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


def load_price_panel(
    data_root: str,
    encoding_candidates: tuple[str, ...],
    min_trade_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """读取全市场日线并生成标准化价格面板。"""

    root = Path(data_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    parts: list[pd.DataFrame] = []
    files = list(_iter_daily_files(root, min_trade_date=min_trade_date))
    if not files:
        raise FileNotFoundError(f"未找到CSV文件: {root}")

    for idx, file_path in enumerate(files, start=1):
        parts.append(_read_one_daily_file(file_path, encoding_candidates))
        if idx % 250 == 0:
            print(f"[load] 已读取 {idx}/{len(files)} 个文件")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    panel = panel.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")
    return panel


def compute_daily_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """计算个股日收益率。"""

    data = price_panel.copy()
    data["ret"] = data.groupby("stock_code", sort=False)["close"].pct_change()
    return data


def _month_end_trade_dates(trading_days: pd.DatetimeIndex) -> pd.Series:
    """把自然月映射为实际月末交易日。"""

    day_df = pd.DataFrame({"trade_date": pd.DatetimeIndex(trading_days).sort_values()})
    day_df["month"] = day_df["trade_date"].dt.to_period("M")
    month_ends = day_df.groupby("month", observed=True)["trade_date"].max()
    return month_ends


def _estimate_incremental_start(
    output_path: Path,
    incremental: bool,
    lookback: int,
    skip_month: int,
) -> Optional[pd.Timestamp]:
    """根据历史输出估计增量重算起点。"""

    if not incremental or (not output_path.exists()):
        return None

    hist = pd.read_csv(output_path)
    if hist.empty or "trade_date" not in hist.columns:
        return None

    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce")
    last_day = hist["trade_date"].max()
    if pd.isna(last_day):
        return None

    months_back = lookback + skip_month + 2
    return pd.Timestamp(last_day) - pd.DateOffset(months=months_back)


def _merge_incremental_output(new_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """把新计算结果与历史输出合并并去重。"""

    if not output_path.exists():
        return new_df.sort_values("trade_date").reset_index(drop=True)

    old_df = pd.read_csv(output_path)
    old_df["trade_date"] = pd.to_datetime(old_df["trade_date"], errors="coerce")

    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged = merged.dropna(subset=["trade_date"])
    merged = merged.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
    return merged.reset_index(drop=True)
