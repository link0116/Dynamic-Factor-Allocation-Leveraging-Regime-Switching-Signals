"""
SJM超参数优化模块（严格时序、严格防未来泄露）。

实现目标：
1) 使用Sharpe Ratio作为唯一主目标选择超参数。
2) 采用滚动时间序列验证，每6个月重新搜索参数。
3) 仅使用当时可得历史数据训练/验证，测试期绝不参与调参。

输出：
- outputs/best_parameter.json
- outputs/parameter_result.csv
- outputs/rolling_test_result.csv
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

# 允许从项目根目录导入模块（脚本从sjm子目录执行时也有效）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SJMTuningConfig
from evaluation.metrics import compute_strategy_metrics
from sparse_jump_model import SparseJumpModel
from strategy.long_short import (
    build_long_short_returns,
    build_state_expected_return_map,
    build_state_rank_mapping_by_train_active_return,
)


def _load_feature_data(path: str) -> tuple[pd.DataFrame, list[str]]:
    """读取第二步特征并校验必要列。

    数学原理：
    - SJM输入为标准化特征矩阵X；策略收益依赖主动收益r_active。

    设计原因：
    - 把数据完整性校验集中处理，避免调参循环中重复报错。

    论文对应：
    - 对应“以动量主动收益和状态特征作为输入”的前置步骤。

    输入输出：
    - 输入：features_path
    - 输出：(df, z_cols)
      df至少包含 trade_date, momentum_return, market_return, active_return, z_*。
    """

    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {file_path}")

    df = pd.read_csv(file_path)
    required = {"trade_date", "momentum_return", "market_return", "active_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"特征文件缺少必要列: {missing}")

    z_cols = [c for c in df.columns if c.startswith("z_")]
    if not z_cols:
        raise ValueError("特征文件缺少z_标准化特征列")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    df = df.dropna(subset=z_cols + ["momentum_return", "market_return", "active_return"]).reset_index(drop=True)
    return df, z_cols


def _half_year_schedule(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    """生成每6个月重优化节点。

    设计原因：
    - 与用户要求一致：参数每6个月更新一次，仅用于随后6个月。
    """

    if start_date > end_date:
        return []

    anchors: list[pd.Timestamp] = []
    curr = pd.Timestamp(start_date)
    while curr <= end_date:
        anchors.append(curr)
        curr = curr + pd.DateOffset(months=6)
    return anchors


def _build_model(
    n_states: int,
    gamma: float,
    kappa: float,
    cfg: SJMTuningConfig,
) -> SparseJumpModel:
    """构造SJM模型实例。"""

    return SparseJumpModel(
        n_states=n_states,
        jump_penalty=gamma,
        kappa=kappa,
        max_outer_iter=cfg.max_outer_iter,
        max_inner_iter=cfg.max_inner_iter,
        n_init=cfg.n_init,
        random_state=cfg.random_state,
    )


def _fit_and_infer(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    z_cols: list[str],
    n_states: int,
    gamma: float,
    kappa: float,
    cfg: SJMTuningConfig,
) -> pd.DataFrame:
    """训练SJM并在给定区间做在线推断，再构建多空策略收益。

    数学原理：
    - 训练：在train_df上估计SJM参数与状态结构。
    - 推断：在infer_df上做online_predict（因果路径，不回看未来）。
    - 交易：$r_t^{strat}=p_t\\cdot(r_t^{mom}-r_t^{mkt})$。

    设计原因：
    - 统一调参与测试阶段的状态->策略收益管线，减少实现偏差。

    论文对应：
    - 对应“训练状态模型 -> 在线识别 -> 动态因子配置”。

    输入输出：
    - 输入：训练切片、推断切片、参数
    - 输出：包含strategy_return/position/state/state_name的DataFrame
    """

    X_train = train_df[z_cols].to_numpy(dtype=float)
    model = _build_model(n_states=n_states, gamma=gamma, kappa=kappa, cfg=cfg)
    model, train_state_raw = model.fit(X_train)

    # 状态编号重排：按训练期主动收益均值从低到高映射到0..K-1。
    old_to_new = build_state_rank_mapping_by_train_active_return(
        train_state=train_state_raw,
        train_active_return=train_df["active_return"].to_numpy(dtype=float),
    )
    train_state = np.array([old_to_new[int(s)] for s in train_state_raw], dtype=int)

    state_expected = build_state_expected_return_map(
        train_state=train_state,
        train_active_return=train_df["active_return"].to_numpy(dtype=float),
    )

    infer_state_raw = model.online_predict(infer_df[z_cols].to_numpy(dtype=float))
    infer_state = np.array([old_to_new[int(s)] for s in infer_state_raw], dtype=int)
    infer_data = infer_df[["trade_date", "momentum_return", "market_return"]].copy()
    infer_data["state"] = infer_state.astype(int)

    strategy_df = build_long_short_returns(
        data=infer_data,
        state_col="state",
        state_expected_return=state_expected,
        position_band=cfg.expected_return_band,
    )
    return strategy_df


def _evaluate_param_on_validation(
    full_df: pd.DataFrame,
    z_cols: list[str],
    anchor_date: pd.Timestamp,
    gamma: float,
    kappa: float,
    train_window_years: int,
    n_states: int,
    cfg: SJMTuningConfig,
) -> dict[str, Any] | None:
    """在单个重优化节点上评估一组参数（仅训练+验证，不含测试）。

    防泄露规则：
    - 训练区间: [validation_start-train_window, validation_start)
    - 验证区间: [validation_start, anchor_date)
    - 测试区间完全不参与这里的参数选择。
    """

    history_end = anchor_date - pd.Timedelta(days=1)
    validation_start = anchor_date - pd.DateOffset(years=cfg.validation_years)
    train_end = validation_start - pd.Timedelta(days=1)
    train_start = validation_start - pd.DateOffset(years=train_window_years)

    train_df = full_df[(full_df["trade_date"] >= train_start) & (full_df["trade_date"] <= train_end)].copy()
    val_df = full_df[(full_df["trade_date"] >= validation_start) & (full_df["trade_date"] <= history_end)].copy()

    # 数据不足则跳过该参数组合。
    if len(train_df) < 120 or len(val_df) < 60:
        return None

    try:
        val_strategy = _fit_and_infer(
            train_df=train_df,
            infer_df=val_df,
            z_cols=z_cols,
            n_states=n_states,
            gamma=gamma,
            kappa=kappa,
            cfg=cfg,
        )
    except Exception:
        return None

    metrics = compute_strategy_metrics(val_strategy)
    result = {
        "anchor_date": anchor_date.date().isoformat(),
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "validation_start": validation_start.date().isoformat(),
        "validation_end": history_end.date().isoformat(),
        "gamma": gamma,
        "kappa": kappa,
        "window": train_window_years,
        "state_number": n_states,
    }
    result.update(metrics)
    return result


def _select_best_param(records: pd.DataFrame) -> pd.Series:
    """按Sharpe从高到低选择最优参数。"""

    tmp = records.copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=["Sharpe"])
    if tmp.empty:
        raise ValueError("没有可用参数组合（Sharpe全为NaN），请检查数据窗口设置")
    tmp = tmp.sort_values(["Sharpe", "AnnualReturn"], ascending=[False, False]).reset_index(drop=True)
    return tmp.iloc[0]


def _run_fixed_split_tuning(
    df: pd.DataFrame,
    z_cols: list[str],
    cfg: SJMTuningConfig,
) -> dict[str, Any]:
    """固定切分调参与测试。

    切分规则（严格按用户要求）：
    - Training: [fixed_train_start, fixed_train_end]
    - Validation: [fixed_val_start, fixed_val_end]
    - Test: [fixed_test_start, +inf)

    防泄露原则：
    1) 参数搜索仅在Training+Validation内完成。
    2) Test段不参与任何参数选择。
    3) Test段仅使用固定最佳参数做online inference。
    """

    train_start = pd.to_datetime(cfg.fixed_train_start)
    train_end = pd.to_datetime(cfg.fixed_train_end)
    val_start = pd.to_datetime(cfg.fixed_val_start)
    val_end = pd.to_datetime(cfg.fixed_val_end)
    test_start = pd.to_datetime(cfg.fixed_test_start)

    if not (train_start <= train_end < val_start <= val_end < test_start):
        raise ValueError("固定切分日期顺序非法，请满足 train_end < val_start <= val_end < test_start")

    train_df = df[(df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)].copy()
    val_df = df[(df["trade_date"] >= val_start) & (df["trade_date"] <= val_end)].copy()
    test_df = df[df["trade_date"] >= test_start].copy()

    if len(train_df) < 120:
        raise ValueError("Training样本不足，至少需要120个交易日")
    if len(val_df) < 60:
        raise ValueError("Validation样本不足，至少需要60个交易日")
    if len(test_df) < 1:
        raise ValueError("Test样本为空，请检查fixed_test_start")

    grid = list(itertools.product(cfg.gamma_list, cfg.kappa_list, cfg.state_number_list))
    param_records: list[dict[str, Any]] = []

    train_window_years = round((train_end - train_start).days / 365.25, 2)

    for gamma, kappa, n_states in grid:
        try:
            val_strategy = _fit_and_infer(
                train_df=train_df,
                infer_df=val_df,
                z_cols=z_cols,
                n_states=int(n_states),
                gamma=float(gamma),
                kappa=float(kappa),
                cfg=cfg,
            )
        except Exception:
            continue

        metrics = compute_strategy_metrics(val_strategy)
        rec = {
            "anchor_date": test_start.date().isoformat(),
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "validation_start": val_start.date().isoformat(),
            "validation_end": val_end.date().isoformat(),
            "gamma": float(gamma),
            "kappa": float(kappa),
            "window": train_window_years,
            "state_number": int(n_states),
        }
        rec.update(metrics)
        param_records.append(rec)

    if not param_records:
        raise ValueError("固定切分模式下无有效参数结果，请检查样本或参数网格")

    param_df = pd.DataFrame(param_records)
    best = _select_best_param(param_df)

    # 固定最佳参数，仅对测试段做online inference（禁止重调参）。
    test_strategy = _fit_and_infer(
        train_df=train_df,
        infer_df=test_df,
        z_cols=z_cols,
        n_states=int(best["state_number"]),
        gamma=float(best["gamma"]),
        kappa=float(best["kappa"]),
        cfg=cfg,
    )
    test_strategy["anchor_date"] = test_start.date().isoformat()
    test_strategy["test_start"] = test_start.date().isoformat()
    test_strategy["test_end"] = df["trade_date"].max().date().isoformat()
    test_strategy["gamma"] = float(best["gamma"])
    test_strategy["kappa"] = float(best["kappa"])
    test_strategy["window"] = train_window_years
    test_strategy["state_number"] = int(best["state_number"])

    best_payload = {
        "gamma": float(best["gamma"]),
        "kappa": float(best["kappa"]),
        "train_window": train_window_years,
        "state_number": int(best["state_number"]),
        "Sharpe": float(best["Sharpe"]),
    }

    return {
        "best_payload": best_payload,
        "param_df": param_df,
        "rolling_test_df": test_strategy,
    }


def run_sjm_hyperparameter_tuning(cfg: SJMTuningConfig | None = None) -> dict[str, Any]:
    """执行完整SJM调参与滚动测试。

    流程（严格防泄露）：
    1) 在每个6个月锚点，使用锚点之前历史数据做参数搜索（训练+验证）。
    2) 选出Sharpe最高参数。
    3) 固定该参数，仅用锚点之前数据重训模型。
    4) 对未来6个月测试区间做online inference与策略回测。
    5) 下个锚点重复，允许纳入新发生数据重新搜索。

    返回：
    - best_parameter（全局最优）
    - parameter_result（所有验证参数结果）
    - rolling_test_result（逐日测试结果）
    """

    cfg = cfg or SJMTuningConfig()
    df, z_cols = _load_feature_data(cfg.features_path)

    if cfg.tuning_mode == "fixed_split":
        fixed_result = _run_fixed_split_tuning(df=df, z_cols=z_cols, cfg=cfg)
        best_payload = fixed_result["best_payload"]
        param_df = fixed_result["param_df"]
        rolling_test_df = fixed_result["rolling_test_df"]
    elif cfg.tuning_mode == "rolling":
        min_date = df["trade_date"].min()
        max_date = df["trade_date"].max()
        test_start = pd.to_datetime(cfg.test_start_date)
        if test_start < min_date:
            test_start = min_date

        anchors = _half_year_schedule(test_start, max_date)
        if not anchors:
            raise ValueError("无可用滚动节点，请检查test_start_date与数据区间")

        param_records: list[dict[str, Any]] = []
        rolling_test_rows: list[pd.DataFrame] = []

        grid = list(
            itertools.product(
                cfg.gamma_list,
                cfg.kappa_list,
                cfg.train_window_years_list,
                cfg.state_number_list,
            )
        )

        for anchor in anchors:
            cycle_records: list[dict[str, Any]] = []
            for gamma, kappa, window_years, n_states in grid:
                rec = _evaluate_param_on_validation(
                    full_df=df,
                    z_cols=z_cols,
                    anchor_date=anchor,
                    gamma=gamma,
                    kappa=kappa,
                    train_window_years=window_years,
                    n_states=n_states,
                    cfg=cfg,
                )
                if rec is not None:
                    cycle_records.append(rec)

            if not cycle_records:
                continue

            cycle_df = pd.DataFrame(cycle_records)
            best = _select_best_param(cycle_df)
            param_records.extend(cycle_records)

            history_end = anchor - pd.Timedelta(days=1)
            test_end = min(anchor + pd.DateOffset(months=cfg.reopt_months) - pd.Timedelta(days=1), max_date)

            fit_df = df[df["trade_date"] <= history_end].copy()
            test_df = df[(df["trade_date"] >= anchor) & (df["trade_date"] <= test_end)].copy()
            if len(fit_df) < 120 or len(test_df) == 0:
                continue

            test_strategy = _fit_and_infer(
                train_df=fit_df,
                infer_df=test_df,
                z_cols=z_cols,
                n_states=int(best["state_number"]),
                gamma=float(best["gamma"]),
                kappa=float(best["kappa"]),
                cfg=cfg,
            )
            test_strategy["anchor_date"] = anchor.date().isoformat()
            test_strategy["test_start"] = anchor.date().isoformat()
            test_strategy["test_end"] = pd.Timestamp(test_end).date().isoformat()
            test_strategy["gamma"] = float(best["gamma"])
            test_strategy["kappa"] = float(best["kappa"])
            test_strategy["window"] = int(best["window"])
            test_strategy["state_number"] = int(best["state_number"])
            rolling_test_rows.append(test_strategy)

        if not param_records:
            raise ValueError("没有任何参数组合得到有效验证结果，请检查窗口长度与样本区间")

        param_df = pd.DataFrame(param_records)
        best_global = _select_best_param(param_df)
        rolling_test_df = pd.concat(rolling_test_rows, ignore_index=True) if rolling_test_rows else pd.DataFrame()
        best_payload = {
            "gamma": float(best_global["gamma"]),
            "kappa": float(best_global["kappa"]),
            "train_window": int(best_global["window"]),
            "state_number": int(best_global["state_number"]),
            "Sharpe": float(best_global["Sharpe"]),
        }
    else:
        raise ValueError("tuning_mode 仅支持 fixed_split 或 rolling")

    best_path = Path(cfg.best_param_path)
    if not best_path.is_absolute():
        best_path = PROJECT_ROOT / best_path
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    param_path = Path(cfg.param_result_path)
    if not param_path.is_absolute():
        param_path = PROJECT_ROOT / param_path
    param_path.parent.mkdir(parents=True, exist_ok=True)

    # 按用户要求保留核心列并按Sharpe排序；同时保留扩展诊断列。
    export_df = param_df.copy()
    export_df["Return"] = export_df["AnnualReturn"]
    export_df["Volatility"] = export_df["AnnualVolatility"]
    export_df["StateSwitch"] = export_df["StateSwitchTimes"]

    core_cols = [
        "gamma",
        "kappa",
        "window",
        "state_number",
        "Sharpe",
        "Return",
        "Volatility",
        "Turnover",
        "MDD",
        "StateSwitch",
    ]
    keep_cols = core_cols + [
        "AnnualReturn",
        "AnnualVolatility",
        "IR",
        "AverageHoldingTime",
        "BullDuration",
        "BearDuration",
        "StateSwitchTimes",
        "anchor_date",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
    ]
    keep_cols = [c for c in keep_cols if c in export_df.columns]
    export_df.sort_values("Sharpe", ascending=False)[keep_cols].to_csv(param_path, index=False, encoding="utf-8-sig")

    test_path = Path(cfg.rolling_test_path)
    if not test_path.is_absolute():
        test_path = PROJECT_ROOT / test_path
    test_path.parent.mkdir(parents=True, exist_ok=True)
    rolling_test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    return {
        "config": asdict(cfg),
        "best_parameter": best_payload,
        "parameter_result": param_df,
        "rolling_test_result": rolling_test_df,
    }


def build_visual_state_path_with_fixed_best(
    cfg: SJMTuningConfig,
    best_parameter: dict[str, Any],
) -> pd.DataFrame:
    """用固定最佳参数对训练+验证区间做仅可视化推断。

    设计目标：
    - 不改变调参与测试流程；
    - 仅用于绘图，把训练+验证段也补齐Bull/Bear状态色带；
    - 严格使用“固定最佳参数 + 训练集拟合”来做状态推断。

    输入输出：
    - 输入：调参配置cfg，最佳参数best_parameter
    - 输出：包含 trade_date/state/state_name 的可视化状态DataFrame
    """

    df, z_cols = _load_feature_data(cfg.features_path)

    gamma = float(best_parameter["gamma"])
    kappa = float(best_parameter["kappa"])
    n_states = int(best_parameter["state_number"])

    if cfg.tuning_mode == "fixed_split":
        train_start = pd.to_datetime(cfg.fixed_train_start)
        train_end = pd.to_datetime(cfg.fixed_train_end)
        test_start = pd.to_datetime(cfg.fixed_test_start)

        train_df = df[(df["trade_date"] >= train_start) & (df["trade_date"] <= train_end)].copy()
        vis_df = df[(df["trade_date"] >= train_start) & (df["trade_date"] < test_start)].copy()
    else:
        test_start = pd.to_datetime(cfg.test_start_date)
        train_df = df[df["trade_date"] < test_start].copy()
        vis_df = df[df["trade_date"] < test_start].copy()

    if len(train_df) < 120 or vis_df.empty:
        return pd.DataFrame(columns=["trade_date", "state", "state_name", "active_return", "momentum_return", "market_return"])

    vis_states = _fit_and_infer(
        train_df=train_df,
        infer_df=vis_df,
        z_cols=z_cols,
        n_states=n_states,
        gamma=gamma,
        kappa=kappa,
        cfg=cfg,
    )
    return vis_states.reset_index(drop=True)


if __name__ == "__main__":
    result = run_sjm_hyperparameter_tuning(SJMTuningConfig())
    print("[done] best_parameter:")
    print(result["best_parameter"])
    print(f"[done] parameter_result rows: {len(result['parameter_result']):,}")
    print(f"[done] rolling_test_result rows: {len(result['rolling_test_result']):,}")
