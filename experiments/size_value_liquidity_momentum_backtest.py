"""仅使用 Size、Value、Liquidity、Momentum 因子的选股对比实验。"""

from __future__ import annotations

import argparse

from strategy.multifactor_backtest import MultiFactorBacktestConfig, _resolve_path, run_multifactor_backtest


EXPERIMENT_FACTORS = ("size", "value", "liquidity", "momentum")


def run_size_value_liquidity_momentum_experiment(
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 100,
    output_dir: str = "outputs/size_value_liquidity_momentum_backtest",
) -> dict[str, object]:
    """保持主回测其余默认条件不变，仅用四个指定因子计算 Alpha。"""

    resolved_output_dir = _resolve_path(output_dir)
    if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且非空，为避免覆盖原结果，请指定新的 --output-dir: {resolved_output_dir}"
        )

    config = MultiFactorBacktestConfig(
        factor_names=EXPERIMENT_FACTORS,
        output_dir=str(resolved_output_dir),
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
    )
    return run_multifactor_backtest(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Size、Value、Liquidity、Momentum 四因子选股对比实验")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/size_value_liquidity_momentum_backtest")
    args = parser.parse_args()
    result = run_size_value_liquidity_momentum_experiment(
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        output_dir=args.output_dir,
    )
    print(f"[done] 最优 lambda={result['best_lambda']:.1f}, 输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()