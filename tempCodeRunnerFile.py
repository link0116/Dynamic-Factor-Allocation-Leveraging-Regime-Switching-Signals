"""
主流程入口：
1) 构建A股动量组合收益（factor）。
2) 构建SJM输入特征并标准化（feature）。
3) 执行SJM超参数优化（固定切分或滚动模式）。
4) 在测试集上仅做Online Inference并生成状态识别可视化图。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from config import SJMTuningConfig
from factor.cross_sectional import CrossSectionalFactorConfig, run_cross_sectional_factor_pipeline
from feature.sjm_features import SJMFeatureConfig, apply_factor_feature_preset
from sjm.regime_analysis import (
    FactorRegimeAnalysisConfig,
    build_regime_features,
    export_factor_regime_outputs,
    fit_factor_sjm,
)
from sjm.train_sjm import SJMTrainConfig


@dataclass(frozen=True)
class FactorSpec:
    """单因子管线注册信息。"""

    name: str
    display_name: str
    return_col: str
    output_path_getter: Callable[["PipelineConfig"], str]
    runner: Callable[["PipelineConfig"], pd.DataFrame]


@dataclass
class PipelineConfig:
    """总控参数。"""

    # SJM当前使用哪条因子收益构造特征；sjm_factor_names 可传多个。
    sjm_factor_name: str = "value"
    sjm_factor_names: tuple[str, ...] = ()

    # 按因子名自定义特征窗口覆盖；未设置的字段使用 FACTOR_FEATURE_PRESETS 预设值。
    # 示例: {"momentum": {"ewma_windows": (5, 20, 60)}}
    factor_feature_overrides: dict[str, dict] = field(default_factory=dict)

    cross_sectional_factors: dict[str, CrossSectionalFactorConfig] = field(
        default_factory=lambda: {
            name: CrossSectionalFactorConfig(
                factor_name=name,
                data_root="data/A股日线指标",
                financial_root="data/A股财务数据",
                output_path=f"outputs/factor_returns/{name}.csv",
                top_ratio=0.20,
                signal_lag_days=1,
            )
            for name in ("momentum", "value", "quality", "size", "liquidity", "lowvol", "growth")
        }
    )

    feature: SJMFeatureConfig = field(
        default_factory=lambda: SJMFeatureConfig(
            momentum_path="outputs/factor_returns/momentum.csv",
            market_path="沪深300.csv",
            output_path="outputs/sjm_features.csv",
            ewma_windows=(8, 21, 63),
            rsi_windows=(8, 21, 63),
            stoch_windows=(8, 21, 63),
            macd_pairs=((8, 21), (21, 63)),
            downside_window=21,
            beta_window=21,
            market_env_windows=(21, 63),
            include_market_breadth=True,
            standardize_mode="expanding",
        )
    )

    sjm_tuning: SJMTuningConfig = field(
        default_factory=lambda: SJMTuningConfig(
            features_path="outputs/sjm_features.csv",
            gamma_list=[1, 2, 4, 6, 8, 10, 15, 20],
            kappa_list=[1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6],
            state_number_list=[2],
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

    sjm_train: SJMTrainConfig = field(
        default_factory=lambda: SJMTrainConfig(
            features_path="outputs/sjm_features.csv",
            output_state_path="outputs/sjm_state_daily.csv",
            output_weight_path="outputs/sjm_feature_weight.csv",
            output_centroid_path="outputs/sjm_state_centroid.csv",
            output_transition_path="outputs/sjm_state_transition.csv",
            n_init=8,
            max_outer_iter=15,
            max_inner_iter=30,
            random_state=42,
            best_param_path="outputs/best_parameter.json",
            tuning_mode="fixed_split",
            fixed_train_start="2020-01-01",
            fixed_train_end="2023-06-30",
            fixed_val_start="2023-07-01",
            fixed_test_start="2024-06-02",
        )
    )

    regime_plot_path: str = "outputs/sjm_regime_plot.png"
    test_regime_plot_path: str = "outputs/sjm_test_regime_plot.png"
    split_plot_path: str = "outputs/full_period_split_plot.png"
    full_regime_plot_path: str = "results/sjm_full_period.png"


def _plot_regime_identification(
    state_daily: pd.DataFrame,
    output_path: str,
    factor_display_name: str = "value",
) -> None:
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

    ax.set_title(f"SJM状态识别图（{factor_display_name} Active Return）", fontsize=13)
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
    factor_display_name: str = "value",
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
                "Bull": (0.23, 0.69, 0.33, 0.30),
                "Bear": (0.86, 0.24, 0.20, 0.30),
            }
            state_color = {
                "Bull": (0.08, 0.52, 0.16, 0.96),
                "Bear": (0.74, 0.11, 0.08, 0.96),
            }

            y_min2, y_max2 = ax.get_ylim()
            band_h2 = (y_max2 - y_min2) * 0.06
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

    ax.set_title(f"全时期划分图（{factor_display_name}: Training / Validation / Test + Bull/Bear）", fontsize=13)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("主动净值")
    ax.grid(alpha=0.25)
    from matplotlib.patches import Patch

    legend_handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.8, label="主动净值"),
        Patch(facecolor=(0.80, 0.90, 1.00, 0.18), edgecolor="none", label="Training/History"),
        Patch(facecolor=(0.99, 0.92, 0.75, 0.18), edgecolor="none", label="Validation"),
        Patch(facecolor=(0.85, 0.96, 0.85, 0.18), edgecolor="none", label="Test"),
        Patch(facecolor=(0.23, 0.69, 0.33, 0.30), edgecolor="none", label="Bull背景"),
        Patch(facecolor=(0.86, 0.24, 0.20, 0.30), edgecolor="none", label="Bear背景"),
        Patch(facecolor=(0.08, 0.52, 0.16, 0.96), edgecolor="none", label="Bull状态"),
        Patch(facecolor=(0.74, 0.11, 0.08, 0.96), edgecolor="none", label="Bear状态"),
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


def _plot_full_regime_identification(
    full_state_df: pd.DataFrame,
    tuning_cfg: SJMTuningConfig,
    output_path: str,
    factor_display_name: str = "value"
) -> None:
    """绘制全样本状态识别图（Training + Validation + Test）。"""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as err:
        raise ImportError("未安装matplotlib，无法生成可视化图") from err

    data = full_state_df.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "active_return", "state_name"]).sort_values("trade_date")
    if data.empty:
        raise ValueError("全样本状态数据为空，无法绘图")

    full_dates = data["trade_date"].reset_index(drop=True)
    full_nav = (1.0 + data["active_return"].astype(float)).cumprod().reset_index(drop=True)
    full_state = data["state_name"].astype(str).reset_index(drop=True)
    if not (len(full_dates) == len(full_nav) == len(full_state)):
        raise ValueError("full_dates/full_nav/full_state 长度不一致，无法绘图")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(full_dates, full_nav, color="tab:blue", linewidth=1.6, label="主动净值")

    color_map = {
        "Bull": (0.0, 0.5, 0.0, 0.18),
        "Bear": (1.0, 0.0, 0.0, 0.18),
    }

    date_vals = full_dates.to_numpy()
    state_vals = full_state.to_numpy()

    if len(data) > 0:
        seg_start = 0
        for i in range(1, len(data)):
            if state_vals[i] != state_vals[i - 1]:
                left = date_vals[seg_start]
                right = date_vals[i]
                ax.axvspan(left, right, color=color_map.get(state_vals[i - 1], (0.7, 0.7, 0.7, 0.12)), linewidth=0)
                seg_start = i

        ax.axvspan(
            date_vals[seg_start],
            date_vals[-1],
            color=color_map.get(state_vals[-1], (0.7, 0.7, 0.7, 0.12)),
            linewidth=0,
        )

    train_end = pd.to_datetime(tuning_cfg.fixed_train_end)
    valid_end = pd.to_datetime(tuning_cfg.fixed_val_end)
    y_min, y_max = ax.get_ylim()
    label_y = y_max - (y_max - y_min) * 0.03

    for x, txt in [(train_end, "Train End"), (valid_end, "Validation End")]:
        if full_dates.min() <= x <= full_dates.max():
            ax.axvline(x, color="gray", linewidth=1.5, linestyle="--", alpha=0.95)
            ax.text(x, label_y, txt, rotation=90, va="top", ha="right", fontsize=9, color="gray")

    ax.set_title(f"SJM状态识别图（{factor_display_name} Active Return）\nTraining + Validation + Test", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Active NAV")
    ax.grid(alpha=0.25)

    from matplotlib.patches import Patch

    legend_handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=1.8, label="主动净值"),
        Patch(facecolor=(0.0, 0.5, 0.0, 0.18), edgecolor="none", label="Bull 区间"),
        Patch(facecolor=(1.0, 0.0, 0.0, 0.18), edgecolor="none", label="Bear 区间"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _factor_registry() -> dict[str, FactorSpec]:
    """集中注册可切换因子，新增因子时只需补一项。"""

    display_map = {
        "momentum": "Momentum",
        "value": "Value",
        "quality": "Quality",
        "size": "Size",
        "liquidity": "Liquidity",
        "lowvol": "LowVol",
        "growth": "Growth",
    }
    registry: dict[str, FactorSpec] = {}
    for name, display_name in display_map.items():
        registry[name] = FactorSpec(
            name=name,
            display_name=display_name,
            return_col=f"{name}_return",
            output_path_getter=lambda cfg, factor_name=name: cfg.cross_sectional_factors[factor_name].output_path,
            runner=lambda cfg, factor_name=name: run_cross_sectional_factor_pipeline(
                cfg.cross_sectional_factors[factor_name]
            ),
        )
    return registry


def _normalize_factor_name(name: str) -> str:
    """把常见因子别名映射为主程序内部名称。"""

    value = name.strip().lower().replace("-", "_")
    alias_map = {
        "low_vol": "lowvol",
        "low_volatility": "lowvol",
        "lowvolatility": "lowvol",
    }
    return alias_map.get(value, value)


def _selected_factor_names(config: PipelineConfig, registry: dict[str, FactorSpec]) -> tuple[str, ...]:
    raw_names = config.sjm_factor_names or (config.sjm_factor_name,)
    names = tuple(dict.fromkeys(_normalize_factor_name(name) for name in raw_names if name.strip()))
    unsupported = [name for name in names if name not in registry]
    if unsupported:
        raise ValueError(f"不支持的因子: {unsupported}; 当前支持: {', '.join(registry)}")
    if not names:
        raise ValueError("请至少配置一个SJM因子")
    return names


def _parse_factor_names(raw: str, registry: dict[str, FactorSpec]) -> tuple[str, ...]:
    """解析命令行因子列表，支持逗号分隔和 all。"""

    value = raw.strip().lower()
    if value == "all":
        return tuple(registry)
    return tuple(_normalize_factor_name(name) for name in value.split(",") if name.strip())


def _configure_sjm_context_for_factor(config: PipelineConfig, spec: FactorSpec) -> None:
    """把当前因子收益接入特征、调参、训练和可视化输出。"""

    output_dir = Path("outputs") / spec.name
    result_dir = Path("results") / spec.name
    feature_path = output_dir / "sjm_features.csv"
    best_param_path = output_dir / "best_parameter.json"
    factor_path = spec.output_path_getter(config)

    config.feature.factor_path = factor_path
    config.feature.factor_return_col = spec.return_col
    config.feature.momentum_path = factor_path
    config.feature.output_path = str(feature_path)

    config.sjm_tuning.features_path = str(feature_path)
    config.sjm_tuning.best_param_path = str(best_param_path)
    config.sjm_tuning.param_result_path = str(output_dir / "parameter_result.csv")
    config.sjm_tuning.rolling_test_path = str(output_dir / "rolling_test_result.csv")

    config.sjm_train.features_path = str(feature_path)
    config.sjm_train.best_param_path = str(best_param_path)
    config.sjm_train.output_state_path = str(output_dir / "sjm_state_daily.csv")
    config.sjm_train.output_weight_path = str(output_dir / "sjm_feature_weight.csv")
    config.sjm_train.output_centroid_path = str(output_dir / "sjm_state_centroid.csv")
    config.sjm_train.output_transition_path = str(output_dir / "sjm_state_transition.csv")

    config.regime_plot_path = str(output_dir / "sjm_regime_plot.png")
    config.test_regime_plot_path = str(output_dir / "sjm_test_regime_plot.png")
    config.split_plot_path = str(output_dir / "full_period_split_plot.png")
    config.full_regime_plot_path = str(result_dir / "sjm_full_period.png")

    # 应用该因子对应的特征窗口预设，再叠加用户自定义覆盖。
    apply_factor_feature_preset(
        config.feature,
        spec.name,
        config.factor_feature_overrides.get(spec.name),
    )


def _run_sjm_for_factor(config: PipelineConfig, spec: FactorSpec) -> dict[str, str]:
    """执行单个因子的收益、特征、调参、训练、状态划分和图表输出。"""

    _configure_sjm_context_for_factor(config, spec)

    print(f"[factor:{spec.name}] 构建{spec.display_name} long-short因子收益...")
    spec.runner(config)

    print(f"[factor:{spec.name}] 构建SJM状态特征...")
    feature_df = build_regime_features(config.feature)

    print(f"[factor:{spec.name}] SJM参数优化并训练独立模型...")
    fit_outputs = fit_factor_sjm(config.sjm_tuning, config.sjm_train)
    tuning_outputs = fit_outputs["tuning_outputs"]
    train_outputs = fit_outputs["train_outputs"]
    best_parameter = tuning_outputs.get("best_parameter")

    analysis_cfg = FactorRegimeAnalysisConfig(
        factor_name=spec.name,
        display_name=spec.display_name,
        output_dir=str(Path("outputs") / spec.name),
        result_dir=str(Path("results") / spec.name),
        model_path=str(Path("models") / f"{spec.display_name.replace(' ', '')}.pkl"),
    )
    feature_dates = pd.to_datetime(feature_df["trade_date"], errors="coerce")
    train_start = pd.to_datetime(config.sjm_tuning.fixed_train_start)
    train_end = pd.to_datetime(config.sjm_tuning.fixed_train_end)
    training_samples = int(((feature_dates >= train_start) & (feature_dates <= train_end)).sum())

    print(f"[factor:{spec.name}] 保存模型、状态统计、分析报告和论文风格图组...")
    analysis_manifest = export_factor_regime_outputs(
        config=analysis_cfg,
        feature_df=feature_df,
        train_outputs=train_outputs,
        best_parameter=best_parameter,
        training_samples=training_samples,
    )

    full_period_state = pd.read_csv(analysis_manifest["state_daily_path"])
    full_period_state["trade_date"] = pd.to_datetime(full_period_state["trade_date"], errors="coerce")

    _plot_full_period_split(
        feature_df=feature_df,
        tuning_cfg=config.sjm_tuning,
        output_path=config.split_plot_path,
        state_daily=full_period_state,
        factor_display_name=spec.display_name,
    )
    _plot_full_regime_identification(
        full_state_df=full_period_state,
        tuning_cfg=config.sjm_tuning,
        output_path=config.full_regime_plot_path,
        factor_display_name=spec.display_name,
    )

    test_start = pd.to_datetime(config.sjm_tuning.fixed_test_start)
    test_state = full_period_state[full_period_state["trade_date"] >= test_start].copy()
    if not test_state.empty:
        _plot_regime_identification(
            state_daily=test_state,
            output_path=config.test_regime_plot_path,
            factor_display_name=spec.display_name,
        )

    return {
        "factor": spec.name,
        "factor_return": spec.output_path_getter(config),
        "features": config.feature.output_path,
        "best_parameter": config.sjm_tuning.best_param_path,
        "state_daily": config.sjm_train.output_state_path,
        "model": analysis_manifest["model_path"],
        "report": analysis_manifest["report_path"],
        "figures": str(Path("outputs") / spec.name / "figures"),
    }


def run_pipeline(config: PipelineConfig) -> None:
    """执行完整流程；每个因子单独训练一个SJM并独立输出结果。"""

    registry = _factor_registry()
    factor_names = _selected_factor_names(config, registry)
    print(f"[start] SJM因子队列: {', '.join(factor_names)}")

    summaries = [_run_sjm_for_factor(config, registry[name]) for name in factor_names]
    print("[done] 因子状态识别完成。")
    for item in summaries:
        print(
            f"[done] {item['factor']}: state={item['state_daily']}, "
            f"model={item['model']}, report={item['report']}, figures={item['figures']}"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="状态切换信号的动态因子配置主程序")
    parser.add_argument(
        "--factors",
        default="all",
        help="SJM因子列表，支持 momentum,value,quality,size,liquidity,lowvol,growth；多个用逗号分隔，也可传 all。",
    )
    return parser


if __name__ == "__main__":
    registry_for_cli = _factor_registry()
    args = _build_arg_parser().parse_args()
    run_pipeline(PipelineConfig(sjm_factor_names=_parse_factor_names(args.factors, registry_for_cli)))
