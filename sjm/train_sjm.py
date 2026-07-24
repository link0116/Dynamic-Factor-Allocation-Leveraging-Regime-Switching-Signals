"""
第三步：使用 Sparse Jump Model 分别对七个因子进行 2 状态识别（Bull/Bear）。

功能覆盖：
1) 读取第二步标准化特征（z_列）。
2) 训练2状态SJM（jump_penalty、L1 sparsity可配置）。
3) 输出每日状态（Bull/Bear）。
4) 输出Feature Weight、State Centroid、State Transition。
5) 提供在线推断接口（online_predict）。

说明：
- 本模块调用项目根目录 sparse_jump_model.py 里的 SparseJumpModel。
- 状态命名规则：以 active_return 均值更高者定义为 Bull，另一状态为 Bear。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import json

import numpy as np
import pandas as pd

# 确保可从项目根目录导入 sparse_jump_model.py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sparse_jump_model import OnlineState, SparseJumpModel
from strategy.long_short import (
    assign_state_name_from_expected_return,
    build_state_expected_return_map,
    build_state_rank_mapping_by_train_active_return,
)


@dataclass
class SJMTrainConfig:
    """SJM训练配置。"""

    features_path: str = "outputs/sjm_features.csv"
    factor_return_col: str = "momentum_return"
    output_state_path: str = "outputs/sjm_state_daily.csv"
    output_weight_path: str = "outputs/sjm_feature_weight.csv"
    output_centroid_path: str = "outputs/sjm_state_centroid.csv"
    output_transition_path: str = "outputs/sjm_state_transition.csv"

    # 手动训练入口(train_sjm)使用的模型超参数。
    # 若调用 train_sjm_from_best_parameter，本组参数会被忽略。
    manual_n_states: int = 2
    manual_jump_penalty: float = 12.0
    manual_kappa: Optional[float] = 4.0
    n_init: int = 8
    max_outer_iter: int = 15
    max_inner_iter: int = 30
    random_state: int = 42

    # 从调参结果读取最佳参数
    best_param_path: str = "outputs/best_parameter.json"

    # 训练/推断模式配置
    tuning_mode: str = "fixed_split"
    fixed_train_start: str = "2020-01-01"
    fixed_train_end: str = "2023-06-30"
    fixed_val_start: str = "2023-07-01"
    fixed_test_start: str = "2024-06-02"

    # 可选训练截止日；若设置，则仅用 <= train_end_date 的样本拟合，
    # 并对全样本做在线推断，便于样本内/样本外拆分。
    train_end_date: Optional[str] = None


def _load_feature_data(features_path: str, factor_return_col: str) -> tuple[pd.DataFrame, list[str], str]:
    """读取第二步特征文件，并校验必要字段。"""

    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"特征文件不存在: {path}")

    df = pd.read_csv(path)
    if "trade_date" not in df.columns:
        raise ValueError("特征文件缺少 trade_date 列")
    if "active_return" not in df.columns:
        raise ValueError("特征文件缺少 active_return 列（用于Bull/Bear命名）")

    if factor_return_col not in df.columns:
        raise ValueError(f"特征文件缺少当前因子收益列: {factor_return_col}")

    z_cols = [c for c in df.columns if c.startswith("z_")]
    if not z_cols:
        raise ValueError("特征文件缺少标准化 z_ 特征列")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)

    # SJM输入必须无缺失。
    df = df.dropna(subset=z_cols + ["active_return"]).reset_index(drop=True)
    return df, z_cols, factor_return_col


def _load_best_parameter(best_param_path: str) -> dict:
    """读取tuner输出的最佳参数文件。"""

    p = Path(best_param_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"最佳参数文件不存在: {p}")

    payload = json.loads(p.read_text(encoding="utf-8"))
    for key in ["gamma", "kappa", "state_number"]:
        if key not in payload:
            raise ValueError(f"最佳参数文件缺少字段: {key}")
    return payload


def _build_transition_matrix(state: np.ndarray, n_states: int) -> pd.DataFrame:
    """从离散状态序列估计转移概率矩阵。"""

    counts = np.zeros((n_states, n_states), dtype=float)
    for i in range(1, len(state)):
        prev_s = int(state[i - 1])
        curr_s = int(state[i])
        counts[prev_s, curr_s] += 1.0

    row_sum = counts.sum(axis=1, keepdims=True)
    prob = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    rows = []
    for i in range(n_states):
        for j in range(n_states):
            rows.append(
                {
                    "from_state": int(i),
                    "to_state": int(j),
                    "transition_count": int(counts[i, j]),
                    "transition_prob": float(prob[i, j]),
                }
            )
    return pd.DataFrame(rows)


def _state_name_mapping(state_series: pd.Series, active_return: pd.Series) -> dict[int, str]:
    """按主动收益均值给状态命名：高均值= Bull，低均值= Bear。"""

    tmp = pd.DataFrame({"state": state_series.astype(int), "active_return": active_return.values})
    grouped = tmp.groupby("state", observed=True)["active_return"].mean().sort_values()
    if len(grouped) != 2:
        # 虽然本任务固定2状态，仍保留兜底。
        mapping = {int(s): f"State_{int(s)}" for s in grouped.index}
        return mapping

    bear_state = int(grouped.index[0])
    bull_state = int(grouped.index[-1])
    return {bear_state: "Bear", bull_state: "Bull"}


def _extract_feature_weight(model: SparseJumpModel, z_cols: list[str]) -> pd.DataFrame:
    """导出特征权重。"""

    if model.weights_ is None:
        raise ValueError("模型权重为空，请先完成fit")

    out = pd.DataFrame(
        {
            "feature": z_cols,
            "weight": model.weights_.astype(float),
            "abs_weight_rank": pd.Series(np.abs(model.weights_)).rank(ascending=False, method="dense").astype(int),
        }
    ).sort_values("weight", ascending=False)
    return out.reset_index(drop=True)


def _extract_state_centroid(
    model: SparseJumpModel,
    z_cols: list[str],
    state_name_map: dict[int, str],
    old_to_new_map: dict[int, int],
) -> pd.DataFrame:
    """导出状态质心（标准化特征空间）。"""

    if model.centroids_ is None:
        raise ValueError("模型centroids为空，请先完成fit")

    # 质心按“新状态编号”重排，保证 state=0 始终对应训练期主动收益最低状态。
    new_to_old = {new_s: old_s for old_s, new_s in old_to_new_map.items()}
    rows = []
    for new_s in range(model.centroids_.shape[0]):
        old_s = int(new_to_old[new_s])
        row = {"state": int(new_s), "state_name": state_name_map.get(int(new_s), f"State_{new_s}")}
        for i, col in enumerate(z_cols):
            row[col] = float(model.centroids_[old_s, i])
        rows.append(row)
    return pd.DataFrame(rows)


def train_sjm(config: SJMTrainConfig) -> dict[str, pd.DataFrame | SparseJumpModel]:
    """训练SJM并输出核心结果。

    返回字典：
    - model: 训练后的SparseJumpModel
    - state_daily: 每日状态
    - feature_weight: 特征权重
    - state_centroid: 状态质心
    - state_transition: 状态转移
    """

    if config.manual_n_states != 2:
        raise ValueError("本任务第三步要求状态数固定为2，请设置 manual_n_states=2")

    df, z_cols, factor_return_col = _load_feature_data(config.features_path, config.factor_return_col)

    if config.train_end_date:
        cutoff = pd.to_datetime(config.train_end_date)
        train_mask = df["trade_date"] <= cutoff
        if train_mask.sum() < 120:
            raise ValueError("训练样本过少，建议放宽 train_end_date 或使用全样本训练")
        train_df = df.loc[train_mask].copy()
    else:
        train_df = df.copy()

    X_train = train_df[z_cols].to_numpy(dtype=float)

    model = SparseJumpModel(
        n_states=config.manual_n_states,
        jump_penalty=config.manual_jump_penalty,
        kappa=config.manual_kappa,
        max_outer_iter=config.max_outer_iter,
        max_inner_iter=config.max_inner_iter,
        n_init=config.n_init,
        random_state=config.random_state,
    )
    model, train_state_raw = model.fit(X_train)

    # 状态编号重排：依据训练期主动收益均值排序（低->高 => Bear->Bull）。
    old_to_new = build_state_rank_mapping_by_train_active_return(
        train_state=train_state_raw,
        train_active_return=train_df["active_return"].to_numpy(dtype=float),
    )
    train_state = np.array([old_to_new[int(s)] for s in train_state_raw], dtype=int)
    train_state_expected = build_state_expected_return_map(
        train_state=train_state,
        train_active_return=train_df["active_return"].to_numpy(dtype=float),
    )
    state_name_map = assign_state_name_from_expected_return(train_state_expected)

    # 全样本在线推断，严格因果路径。
    X_all = df[z_cols].to_numpy(dtype=float)
    online_state = OnlineState()
    state_online_raw = model.online_predict(
        X_all,
        online_state=online_state,
        dates=df["trade_date"].to_numpy(),
    )
    state_online = np.array([old_to_new[int(s)] for s in state_online_raw], dtype=int)

    state_daily = df[["trade_date", "active_return", factor_return_col, "market_return"]].copy()
    state_daily["state"] = state_online.astype(int)
    state_daily["state_name"] = state_daily["state"].map(lambda x: state_name_map.get(int(x), f"State_{int(x)}"))

    feature_weight = _extract_feature_weight(model, z_cols)
    state_centroid = _extract_state_centroid(model, z_cols, state_name_map, old_to_new)
    state_transition = _build_transition_matrix(state_daily["state"].to_numpy(), n_states=config.manual_n_states)
    state_transition["from_state_name"] = state_transition["from_state"].map(
        lambda x: state_name_map.get(int(x), f"State_{int(x)}")
    )
    state_transition["to_state_name"] = state_transition["to_state"].map(
        lambda x: state_name_map.get(int(x), f"State_{int(x)}")
    )

    # 输出落盘。
    Path(config.output_state_path).parent.mkdir(parents=True, exist_ok=True)
    state_daily.to_csv(config.output_state_path, index=False, encoding="utf-8-sig")
    feature_weight.to_csv(config.output_weight_path, index=False, encoding="utf-8-sig")
    state_centroid.to_csv(config.output_centroid_path, index=False, encoding="utf-8-sig")
    state_transition.to_csv(config.output_transition_path, index=False, encoding="utf-8-sig")

    return {
        "model": model,
        "state_daily": state_daily,
        "feature_weight": feature_weight,
        "state_centroid": state_centroid,
        "state_transition": state_transition,
        "online_state": online_state,
    }


def train_sjm_from_best_parameter(config: SJMTrainConfig) -> dict[str, pd.DataFrame | SparseJumpModel]:
    """读取best_parameter.json并完成最终模型训练与状态输出。

    责任边界：
    - tuner 仅负责调参并输出 best_parameter.json；
    - 本函数专门负责读取最佳参数、训练最终模型、输出状态与统计。
    """

    best = _load_best_parameter(config.best_param_path)

    # 使用调参结果作为最终模型超参数，与手动参数彻底解耦。
    best_n_states = int(best["state_number"])
    best_jump_penalty = float(best["gamma"])
    best_kappa = float(best["kappa"])

    df, z_cols, factor_return_col = _load_feature_data(config.features_path, config.factor_return_col)

    if config.tuning_mode == "fixed_split":
        train_start = pd.to_datetime(config.fixed_train_start)
        train_end = pd.to_datetime(config.fixed_train_end)
        infer_start = pd.to_datetime(config.fixed_val_start)

        full_df = df[df["trade_date"] >= train_start].copy().reset_index(drop=True)
        train_df = full_df[(full_df["trade_date"] >= train_start) & (full_df["trade_date"] <= train_end)].copy()
        infer_df = full_df[full_df["trade_date"] >= infer_start].copy()

        if len(train_df) < 120:
            raise ValueError("Training样本不足，无法训练最终模型")
        if infer_df.empty:
            raise ValueError("Validation/Test样本为空，无法进行online inference")

        X_train = train_df[z_cols].to_numpy(dtype=float)
        model = SparseJumpModel(
            n_states=best_n_states,
            jump_penalty=best_jump_penalty,
            kappa=best_kappa,
            max_outer_iter=config.max_outer_iter,
            max_inner_iter=config.max_inner_iter,
            n_init=config.n_init,
            random_state=config.random_state,
        )
        model, train_state_raw = model.fit(X_train)

        old_to_new = build_state_rank_mapping_by_train_active_return(
            train_state=train_state_raw,
            train_active_return=train_df["active_return"].to_numpy(dtype=float),
        )
        train_state = np.array([old_to_new[int(s)] for s in train_state_raw], dtype=int)
        train_state_expected = build_state_expected_return_map(
            train_state=train_state,
            train_active_return=train_df["active_return"].to_numpy(dtype=float),
        )
        state_name_map = assign_state_name_from_expected_return(train_state_expected)

        online_state = OnlineState()
        infer_state_raw = model.online_predict(
            infer_df[z_cols].to_numpy(dtype=float),
            online_state=online_state,
            dates=infer_df["trade_date"].to_numpy(),
        )
        infer_state = np.array([old_to_new[int(s)] for s in infer_state_raw], dtype=int)

        train_out = train_df[["trade_date", "active_return", factor_return_col, "market_return"]].copy()
        train_out["state"] = train_state.astype(int)
        train_out["state_name"] = train_out["state"].map(lambda s: state_name_map.get(int(s), f"State_{int(s)}"))

        infer_out = infer_df[["trade_date", "active_return", factor_return_col, "market_return"]].copy()
        infer_out["state"] = infer_state.astype(int)
        infer_out["state_name"] = infer_out["state"].map(lambda s: state_name_map.get(int(s), f"State_{int(s)}"))

        state_daily = pd.concat([train_out, infer_out], ignore_index=True)
        state_daily = (
            state_daily.sort_values("trade_date")
            .drop_duplicates(subset=["trade_date"], keep="first")
            .reset_index(drop=True)
        )

        feature_weight = _extract_feature_weight(model, z_cols)
        state_centroid = _extract_state_centroid(model, z_cols, state_name_map, old_to_new)
        state_transition = _build_transition_matrix(state_daily["state"].to_numpy(), n_states=best_n_states)
        state_transition["from_state_name"] = state_transition["from_state"].map(
            lambda x: state_name_map.get(int(x), f"State_{int(x)}")
        )
        state_transition["to_state_name"] = state_transition["to_state"].map(
            lambda x: state_name_map.get(int(x), f"State_{int(x)}")
        )

        Path(config.output_state_path).parent.mkdir(parents=True, exist_ok=True)
        state_daily.to_csv(config.output_state_path, index=False, encoding="utf-8-sig")
        feature_weight.to_csv(config.output_weight_path, index=False, encoding="utf-8-sig")
        state_centroid.to_csv(config.output_centroid_path, index=False, encoding="utf-8-sig")
        state_transition.to_csv(config.output_transition_path, index=False, encoding="utf-8-sig")

        return {
            "model": model,
            "state_daily": state_daily,
            "feature_weight": feature_weight,
            "state_centroid": state_centroid,
            "state_transition": state_transition,
            "best_parameter": best,
            "online_state": online_state,
        }

    # rolling模式下回退到原训练接口（保持兼容）。
    return train_sjm(config)


def online_inference(
    model: SparseJumpModel,
    latest_feature_df: pd.DataFrame,
    online_state: OnlineState,
) -> pd.DataFrame:
    """在线推断接口。

    输入：
    - model: 已训练的SJM模型
    - latest_feature_df: 包含 z_特征列与 trade_date 的DataFrame（可多行）
    - online_state: 当前因子独立持有的跨日DP状态

    输出：
    - trade_date
    - state
    """

    z_cols = [c for c in latest_feature_df.columns if c.startswith("z_")]
    if not z_cols:
        raise ValueError("latest_feature_df 缺少 z_特征列")

    tmp = latest_feature_df.copy()
    tmp["trade_date"] = pd.to_datetime(tmp["trade_date"], errors="coerce")
    tmp = tmp.dropna(subset=["trade_date"] + z_cols).sort_values("trade_date").reset_index(drop=True)

    X = tmp[z_cols].to_numpy(dtype=float)
    path = model.online_predict(
        X,
        online_state=online_state,
        dates=tmp["trade_date"].to_numpy(),
    )
    out = tmp[["trade_date"]].copy()
    out["state"] = path.astype(int)
    return out

