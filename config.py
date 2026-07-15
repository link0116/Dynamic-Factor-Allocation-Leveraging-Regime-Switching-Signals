"""
全局配置文件：集中管理SJM超参数搜索与滚动验证设置。

设计原则：
1) 所有时间切分参数显式化，避免隐式未来函数。
2) 与论文一致使用基于策略Sharpe的参数选择，而非聚类准确率。
3) 支持状态数扩展（K=2/3/4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SJMTuningConfig:
    """SJM调参配置。

    数学原理：
    - 在参数网格上对每组(γ, κ, 窗口, K)训练SJM，
      用在线推断得到状态路径，再映射成论文单因子多空策略收益，
      以Sharpe Ratio作为目标函数：argmax Sharpe。

    设计原因：
    - SJM是无监督模型，不能用监督学习准确率。
    - 直接用策略结果做目标，和论文“动态因子配置”的最终用途一致。

    与论文对应：
    - 对应论文中“状态识别服务于资产配置收益提升”的实现路径。

    输入输出：
    - 输入：第二步特征文件 outputs/sjm_features.csv。
    - 输出：best_parameter.json、parameter_result.csv、半年度测试明细。
    """

    # 数据输入
    features_path: str = "outputs/sjm_features.csv"

    # 参数搜索网格
    gamma_list: list[float] = field(default_factory=lambda: [1, 2, 4, 6, 8, 10, 15, 20])
    kappa_list: list[float] = field(default_factory=lambda: [2, 4, 6, 8, 10, 15])
    train_window_years_list: list[int] = field(default_factory=lambda: [8, 9, 10, 11, 12])
    state_number_list: list[int] = field(default_factory=lambda: [2, 3, 4])

    # 调参模式：
    # - fixed_split: 固定训练/验证/测试切分（当前默认，适配有限样本）
    # - rolling: 每6个月重优化（保留兼容）
    tuning_mode: str = "fixed_split"

    # 固定切分（严格防泄露）
    fixed_train_start: str = "2020-01-01"
    fixed_train_end: str = "2023-06-30"
    fixed_val_start: str = "2023-07-01"
    fixed_val_end: str = "2024-06-01"
    fixed_test_start: str = "2024-06-02"

    # 时间序列滚动验证（rolling模式使用）
    validation_years: int = 6
    reopt_months: int = 6

    # 样本阶段划分（rolling模式使用）
    test_start_date: str = "2024-06-01"

    # SJM求解超参（非搜索项）
    n_init: int = 8
    max_outer_iter: int = 15
    max_inner_iter: int = 30
    random_state: int = 42

    # 策略线性仓位映射区间（论文要求[-5%,5%]）
    expected_return_band: float = 0.05

    # 输出文件
    best_param_path: str = "outputs/best_parameter.json"
    param_result_path: str = "outputs/parameter_result.csv"
    rolling_test_path: str = "outputs/rolling_test_result.csv"
