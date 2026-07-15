"""
主流程入口：
1) 构建A股动量组合收益（factor）。
2) 构建SJM输入特征并标准化（feature）。
3) 执行SJM超参数优化（固定切分或滚动模式）。
4) 在测试集上仅做Online Inference并生成状态识别可视化图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import SJMTuningConfig
from factor.momentum import MomentumConfig, run_momentum_pipeline
from feature.sjm_features import SJMFeatureConfig, run_feature_pipeline
from sjm.tuner import build_visual_state_path_with_fixed_best, run_sjm_hyperparameter_tuning


@dataclass
class PipelineConfig:
    """总控参数。"""

    momentum: MomentumConfig = field(
        default_factory=lambda: MomentumConfig(
            data_root="data/A股日线指标",
            output_path="outputs/momentum_return.csv",
            lookback=12,
            skip_month=1,
            top_ratio=0.20,
            rebalance_frequency="M",
            incremental=True,
        )
    )

    feature: SJMFeatureConfig = field(
        default_factory=lambda: SJMFeatureConfig(
            momentum_path="outputs/momentum_return.csv",
            market_path="沪深300.csv",
            output_path="outputs/sjm_features.csv",
            ewma_span=63,
            rsi_window=14,
            stoch_window=14,
            downside_window=63,
            beta_window=63,
            standardize_mode="expanding",
        )
    )

    sjm_tuning: SJMTuningConfig = field(
        default_factory=lambda: SJMTuningConfig(
            features_path="outputs/sjm_features.csv",
            gamma_list=[1, 2, 4, 6, 8, 10, 15, 20],
            kappa_list=[2, 4, 6, 8, 10, 15],
            state_number_list=[2, 3, 4],
            tuning_mode="fixed_split",
            fixed_train_start="2020-01-01",
            fixed_train_end="2023-06-30",
            fixed_val_start="2023-07-01",
            fixed_val_end="2024-06-01",
            fixed_test_start="2024-06-02",
            n_init=8,
            max_outer_iter=15,
            max_inner_iter=30,
            random_state=42,
            best_param_path="outputs/best_parameter.json",
            param_result_path="outputs/parameter_result.csv",
            rolling_test_path="outputs/rolling_test_result.csv",
        )
    )

    regime_plot_path: str = "outputs/sjm_regime_plot.png"
    split_plot_path: str = "outputs/full_period_split_plot.png"


def _plot_regime_identification(state_daily: pd.DataFrame, output_path: str) -> None:
    """生成状态识别图：主动净值曲线 + Bull/Bear 背景色块。"""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as err:
        raise ImportError("未安装matplotlib，无法生成可视化图") from err

    data = state_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "active_return", "state_name"]).sort_values("trade_date")
    data["active_nav"] = (1.0 + data["active_return"]).cumprod()

    # 提升中文显示兼容性。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))

    # 先画主动净值曲线。
    ax.plot(data["trade_date"], data["active_nav"], color="#1f77b4", linewidth=1.6, label="主动净值")

    # 按状态分段着色背景，直观显示识别出的Bull/Bear区间。
    date_vals = data["trade_date"].to_numpy()
    state_vals = data["state_name"].to_numpy()

    # 使用更高对比度配色，并叠加顶部色带，提升状态区间辨识度。
    color_map = {
        "Bull": (0.45, 0.74, 0.49, 0.26),
        "Bear": (0.90, 0.39, 0.39, 0.26),
    }
    band_color_map = {
        "Bull": (0.20, 0.60, 0.25, 0.88),
        "Bear": (0.78, 0.20, 0.20, 0.88),
    }

    if len(data) > 1:
        seg_start = 0
        for i in range(1, len(data)):
            if state_vals[i] != state_vals[i - 1]:
                left = date_vals[seg_start]
                right = date_vals[i]
                ax.axvspan(left, right, color=color_map.get(state_vals[i - 1], (0.9, 0.9, 0.9, 0.3)), linewidth=0)
                seg_start = i

        ax.axvspan(
            date_vals[seg_start],
            date_vals[-1],
            color=color_map.get(state_vals[-1], (0.9, 0.9, 0.9, 0.3)),
            linewidth=0,
        )

        # 顶部状态色带：即使净值线波动较大，也能清晰看到状态切换。
        y_min, y_max = ax.get_ylim()
        band_h = (y_max - y_min) * 0.045
        seg_start = 0
        for i in range(1, len(data)):
            if state_vals[i] != state_vals[i - 1]:
                left = date_vals[seg_start]
                right = date_vals[i]
                ax.fill_between(
                    [left, right],
                    y_max - band_h,
                    y_max,
                    color=band_color_map.get(state_vals[i - 1], (0.35, 0.35, 0.35, 0.85)),
                    linewidth=0,
                    zorder=3,
                )
                seg_start = i

        ax.fill_between(
            [date_vals[seg_start], date_vals[-1]],
            y_max - band_h,
            y_max,
            color=band_color_map.get(state_vals[-1], (0.35, 0.35, 0.35, 0.85)),
            linewidth=0,
            zorder=3,
        )

    ax.set_title("SJM状态识别图（Momentum Active Return）", fontsize=13)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("主动净值")
    ax.grid(alpha=0.25)

    # 仅展示一次图例，避免重复。
    from matplotlib.patches import Patch

    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.8, label="主动净值"),
        Patch(facecolor=color_map["Bull"], edgecolor="none", label="Bull 区间"),
        Patch(facecolor=color_map["Bear"], edgecolor="none", label="Bear 区间"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_full_period_split(
    feature_df: pd.DataFrame,
    tuning_cfg: SJMTuningConfig,
    output_path: str,
    state_daily: pd.DataFrame | None = None,
) -> None:
    """生成全时期划分图：主动净值 + Train/Validation/Test区间 + 因子牛熊状态标注。"""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as err:
        raise ImportError("未安装matplotlib，无法生成可视化图") from err

    data = feature_df.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "active_return"]).sort_values("trade_date")
    data["active_nav"] = (1.0 + data["active_return"]).cumprod()

    if data.empty:
        raise ValueError("特征数据为空，无法绘制全时期划分图")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data["trade_date"], data["active_nav"], color="#1f77b4", linewidth=1.6, label="主动净值")

    # 区间划分：fixed_split 使用显式训练/验证/测试边界；rolling 使用历史/测试两段。
    if tuning_cfg.tuning_mode == "fixed_split":
        train_start = pd.to_datetime(tuning_cfg.fixed_train_start)
        train_end = pd.to_datetime(tuning_cfg.fixed_train_end)
        val_start = pd.to_datetime(tuning_cfg.fixed_val_start)
        val_end = pd.to_datetime(tuning_cfg.fixed_val_end)
        test_start = pd.to_datetime(tuning_cfg.fixed_test_start)

        phase_spans = [
            (train_start, train_end, (0.80, 0.90, 1.00, 0.18), "Training"),
            (val_start, val_end, (0.99, 0.92, 0.75, 0.18), "Validation"),
            (test_start, data["trade_date"].max(), (0.85, 0.96, 0.85, 0.18), "Test"),
        ]
        split_lines = [train_end, val_end, test_start]
    else:
        test_start = pd.to_datetime(tuning_cfg.test_start_date)
        phase_spans = [
            (data["trade_date"].min(), test_start - pd.Timedelta(days=1), (0.80, 0.90, 1.00, 0.18), "History"),
            (test_start, data["trade_date"].max(), (0.85, 0.96, 0.85, 0.18), "Test"),
        ]
        split_lines = [test_start]

    y_min, y_max = ax.get_ylim()
    label_y = y_max - (y_max - y_min) * 0.08

    for left, right, color, label in phase_spans:
        left = max(left, data["trade_date"].min())
        right = min(right, data["trade_date"].max())
        if left <= right:
            ax.axvspan(left, right, color=color, linewidth=0)
            mid = left + (right - left) / 2
            ax.text(mid, label_y, label, ha="center", va="center", fontsize=10, color="#333333")

    for x in split_lines:
        if data["trade_date"].min() <= x <= data["trade_date"].max():
            ax.axvline(x, color="#555555", linestyle="--", linewidth=1.1, alpha=0.85)

    # 在全图叠加因子状态分段着色（浅色）并在顶部叠加色带（深色），
    # 使“上方条带 + 下方背景”颜色一致。
    if state_daily is not None and (not state_daily.empty) and {"trade_date", "state_name"}.issubset(state_daily.columns):
        s = state_daily[["trade_date", "state_name"]].copy()
        s["trade_date"] = pd.to_datetime(s["trade_date"], errors="coerce")
        s = s.dropna(subset=["trade_date", "state_name"]).sort_values("trade_date").reset_index(drop=True)

        if not s.empty:
            state_color_light = {
                "Bull": (0.45, 0.74, 0.49, 0.20),
                "Bear": (0.90, 0.39, 0.39, 0.20),
            }
            state_color = {
                "Bull": (0.20, 0.60, 0.25, 0.92),
                "Bear": (0.78, 0.20, 0.20, 0.92),
            }

            y_min2, y_max2 = ax.get_ylim()
            band_h2 = (y_max2 - y_min2) * 0.04
            st_dates = s["trade_date"].to_numpy()
            st_vals = s["state_name"].astype(str).to_numpy()

            seg_start = 0
            for i in range(1, len(s)):
                if st_vals[i] != st_vals[i - 1]:
                    left = st_dates[seg_start]
                    right = st_dates[i]

                    # 下方主图区间着色（浅色）
                    ax.axvspan(
                        left,
                        right,
                        color=state_color_light.get(st_vals[i - 1], (0.7, 0.7, 0.7, 0.12)),
                        linewidth=0,
                        zorder=1,
                    )

                    # 顶部状态条带（深色）
                    ax.fill_between(
                        [left, right],
                        y_max2 - band_h2,
                        y_max2,
                        color=state_color.get(st_vals[i - 1], (0.35, 0.35, 0.35, 0.85)),
                        linewidth=0,
                        zorder=4,
                    )
                    seg_start = i

            ax.axvspan(
                st_dates[seg_start],
                st_dates[-1],
                color=state_color_light.get(st_vals[-1], (0.7, 0.7, 0.7, 0.12)),
                linewidth=0,
                zorder=1,
            )

            ax.fill_between(
                [st_dates[seg_start], st_dates[-1]],
                y_max2 - band_h2,
                y_max2,
                color=state_color.get(st_vals[-1], (0.35, 0.35, 0.35, 0.85)),
                linewidth=0,
                zorder=4,
            )

    ax.set_title("全时期划分图（Training / Validation / Test + 因子Bull/Bear）", fontsize=13)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("主动净值")
    ax.grid(alpha=0.25)
    from matplotlib.patches import Patch

    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.8, label="主动净值"),
        Patch(facecolor=(0.80, 0.90, 1.00, 0.18), edgecolor="none", label="Training/History"),
        Patch(facecolor=(0.99, 0.92, 0.75, 0.18), edgecolor="none", label="Validation"),
        Patch(facecolor=(0.85, 0.96, 0.85, 0.18), edgecolor="none", label="Test"),
        Patch(facecolor=(0.45, 0.74, 0.49, 0.20), edgecolor="none", label="Bull背景"),
        Patch(facecolor=(0.90, 0.39, 0.39, 0.20), edgecolor="none", label="Bear背景"),
        Patch(facecolor=(0.20, 0.60, 0.25, 0.92), edgecolor="none", label="Bull状态"),
        Patch(facecolor=(0.78, 0.20, 0.20, 0.92), edgecolor="none", label="Bear状态"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_pipeline(config: PipelineConfig) -> None:
    """执行完整流程并输出可视化。"""

    print("[step1] 构建动量组合收益...")
    run_momentum_pipeline(config.momentum)

    print("[step2] 构建SJM特征...")
    feature_df = run_feature_pipeline(config.feature)

    print("[step3] SJM参数优化（Sharpe目标）+ 测试集Online Inference...")
    tune_outputs = run_sjm_hyperparameter_tuning(config.sjm_tuning)
    state_daily = tune_outputs["rolling_test_result"].copy()

    print("[step3.5] 训练+验证段可视化推断（固定最佳参数，不参与调参）...")
    train_val_visual_state = build_visual_state_path_with_fixed_best(
        cfg=config.sjm_tuning,
        best_parameter=tune_outputs["best_parameter"],
    )

    # 拼接成全样本连续状态带：训练+验证（可视化推断） + 测试（正式在线推断）。
    if train_val_visual_state.empty:
        state_for_full_split_plot = state_daily.copy()
    else:
        state_for_full_split_plot = pd.concat([train_val_visual_state, state_daily], ignore_index=True)
        state_for_full_split_plot = (
            state_for_full_split_plot.sort_values("trade_date")
            .drop_duplicates(subset=["trade_date"], keep="last")
            .reset_index(drop=True)
        )

    print("[step4] 生成状态识别图...")
    _plot_regime_identification(state_daily, config.regime_plot_path)
    print(f"[done] 可视化输出: {config.regime_plot_path}")

    print("[step5] 生成全时期划分图...")
    _plot_full_period_split(
        feature_df,
        config.sjm_tuning,
        config.split_plot_path,
        state_daily=state_for_full_split_plot,
    )
    print(f"[done] 全时期划分图输出: {config.split_plot_path}")
    print(f"[done] 最优参数输出: {config.sjm_tuning.best_param_path}")
    print(f"[done] 参数结果输出: {config.sjm_tuning.param_result_path}")
    print(f"[done] 测试集在线推断输出: {config.sjm_tuning.rolling_test_path}")


if __name__ == "__main__":
    run_pipeline(PipelineConfig())
