"""Outputs for single-factor SJM regime identification.

The functions here implement the reporting layer required after each factor has
its own Sparse Jump Model: automatic Bull/Bear labeling checks, model
serialization, summary tables, and the paper-style visualization pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd

from feature.sjm_features import SJMFeatureConfig, run_feature_pipeline
from sjm.train_sjm import SJMTrainConfig, train_sjm_from_best_parameter
from sparse_jump_model import OnlineState
from sjm.tuner import run_sjm_hyperparameter_tuning


@dataclass
class FactorRegimeAnalysisConfig:
    """File destinations and display settings for one factor analysis."""

    factor_name: str
    display_name: str
    output_dir: str
    result_dir: str
    model_path: str
    factor_return_col: str = "momentum_return"
    annualization: int = 252
    rolling_window: int = 21


def compute_active_return(factor_return: pd.Series, market_return: pd.Series) -> pd.Series:
    """Compute the paper-defined active return: factor return minus market return."""

    return pd.to_numeric(factor_return, errors="coerce") - pd.to_numeric(market_return, errors="coerce")


def build_regime_features(config: SJMFeatureConfig) -> pd.DataFrame:
    """Execute Step3 and return standardized SJM regime features."""

    return run_feature_pipeline(config)


def fit_factor_sjm(
    tuning_config,
    train_config: SJMTrainConfig,
) -> dict[str, Any]:
    """Tune and fit one independent SJM for one factor."""

    tuning_outputs = run_sjm_hyperparameter_tuning(tuning_config)
    train_outputs = train_sjm_from_best_parameter(train_config)
    return {
        "tuning_outputs": tuning_outputs,
        "train_outputs": train_outputs,
    }


def _annualized_sharpe(ret: pd.Series, annualization: int) -> float:
    """Compute annualized Sharpe for a daily return series."""

    clean = pd.to_numeric(ret, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    std = float(clean.std(ddof=0))
    if std <= 1e-12:
        return float("nan")
    return float(clean.mean() / std * np.sqrt(annualization))


def _run_lengths(values: pd.Series) -> list[int]:
    """Return consecutive run lengths for an already ordered state series."""

    arr = values.astype(str).to_numpy()
    if len(arr) == 0:
        return []
    lengths: list[int] = []
    current = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def classify_bull_bear(
    state_daily: pd.DataFrame,
    factor_return_col: str,
    annualization: int = 252,
) -> pd.DataFrame:
    """Automatically label and renumber states by active-return economics.

    The primary sorting key is average active return.  Mean factor return and
    information ratio are retained as diagnostics and used only as tie-breakers.
    The lowest-scoring state is Bear and the highest-scoring state is Bull.
    """

    required = {"trade_date", "state", "active_return", factor_return_col}
    missing = required - set(state_daily.columns)
    if missing:
        raise ValueError(f"状态数据缺少列，无法自动识别Bull/Bear: {missing}")

    data = state_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["active_return"] = pd.to_numeric(data["active_return"], errors="coerce")
    data[factor_return_col] = pd.to_numeric(data[factor_return_col], errors="coerce")
    data = data.dropna(subset=["trade_date", "state", "active_return", factor_return_col])

    rows = []
    for state, grp in data.groupby("state", observed=True):
        rows.append(
            {
                "state": int(state),
                "mean_active_return": float(grp["active_return"].mean()),
                "mean_factor_return": float(grp[factor_return_col].mean()),
                "information_ratio": _annualized_sharpe(grp["active_return"], annualization),
            }
        )
    stats = pd.DataFrame(rows)
    stats = stats.sort_values(
        ["mean_active_return", "mean_factor_return", "information_ratio"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    old_to_new = {int(row.state): int(i) for i, row in stats.iterrows()}
    name_map: dict[int, str] = {}
    for i in range(len(stats)):
        if i == 0:
            name_map[i] = "Bear"
        elif i == len(stats) - 1:
            name_map[i] = "Bull"
        else:
            name_map[i] = f"Neutral_{i}"

    data["raw_state"] = data["state"].astype(int)
    data["state"] = data["raw_state"].map(old_to_new).astype(int)
    data["state_name"] = data["state"].map(name_map)
    return data.sort_values("trade_date").reset_index(drop=True)


def build_state_statistics(
    state_daily: pd.DataFrame,
    factor_return_col: str,
    annualization: int = 252,
) -> pd.DataFrame:
    """Build Bull/Bear return and duration statistics."""

    data = state_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "state_name", "active_return"])
    total_days = max(len(data), 1)

    rows = []
    for state_name, grp in data.groupby("state_name", observed=True):
        grp = grp.sort_values("trade_date")
        run_mask = data["state_name"].eq(state_name)
        state_lengths = _run_lengths(data.loc[run_mask | (data["state_name"] != state_name), "state_name"])
        own_lengths = []
        current = 0
        for val in data["state_name"].astype(str):
            if val == str(state_name):
                current += 1
            else:
                if current:
                    own_lengths.append(current)
                    current = 0
        if current:
            own_lengths.append(current)

        active = pd.to_numeric(grp["active_return"], errors="coerce")
        factor_ret = pd.to_numeric(grp[factor_return_col], errors="coerce")
        rows.append(
            {
                "state_name": state_name,
                "days": int(len(grp)),
                "share": float(len(grp) / total_days),
                "average_duration": float(np.mean(own_lengths)) if own_lengths else float("nan"),
                "mean_active_return": float(active.mean()),
                "median_active_return": float(active.median()),
                "active_volatility": float(active.std(ddof=0) * np.sqrt(annualization)),
                "sharpe": _annualized_sharpe(active, annualization),
                "information_ratio": _annualized_sharpe(active, annualization),
                "mean_factor_return": float(factor_ret.mean()),
                "median_factor_return": float(factor_ret.median()),
            }
        )

    out = pd.DataFrame(rows)
    order = {"Bull": 0, "Bear": 1}
    out["_order"] = out["state_name"].map(lambda x: order.get(str(x), 99))
    return out.sort_values(["_order", "state_name"]).drop(columns=["_order"]).reset_index(drop=True)


def build_transition_matrix(state_daily: pd.DataFrame) -> pd.DataFrame:
    """Build a wide transition probability matrix by semantic state name."""

    data = state_daily.sort_values("trade_date").copy()
    names = data["state_name"].astype(str).to_numpy()
    unique = [name for name in ["Bull", "Bear"] if name in set(names)]
    unique.extend(sorted(set(names) - set(unique)))
    counts = pd.DataFrame(0.0, index=unique, columns=unique)
    for i in range(1, len(names)):
        counts.loc[names[i - 1], names[i]] += 1.0
    denom = counts.sum(axis=1).replace(0.0, np.nan)
    prob = counts.div(denom, axis=0).fillna(0.0)
    return prob


def save_model(
    model: Any,
    online_state: OnlineState,
    config: FactorRegimeAnalysisConfig,
    feature_weight: pd.DataFrame,
    state_centroid: pd.DataFrame,
    best_parameter: dict[str, Any] | None,
) -> None:
    """Serialize one factor's independent SJM and metadata to pickle."""

    path = Path(config.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "factor_name": config.factor_name,
        "display_name": config.display_name,
        "model": model,
        "online_state": online_state,
        "feature_columns": feature_weight["feature"].tolist() if "feature" in feature_weight.columns else [],
        "feature_weight": feature_weight,
        "state_centroid": state_centroid,
        "best_parameter": best_parameter or {},
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def write_analysis_report(
    config: FactorRegimeAnalysisConfig,
    best_parameter: dict[str, Any] | None,
    state_stats: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    state_centroid: pd.DataFrame,
    feature_weight: pd.DataFrame,
    training_samples: int,
) -> None:
    """Write a human-readable factor regime analysis report."""

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "regime_analysis.txt"

    lines: list[str] = []
    lines.append("=" * 56)
    lines.append(f"{config.display_name} Regime Analysis")
    lines.append("=" * 56)
    lines.append("")
    lines.append("一、SJM Parameters")
    params = best_parameter or {}
    lines.append(f"Number of States: {params.get('state_number', 'NA')}")
    lines.append(f"Jump Penalty: {params.get('gamma', 'NA')}")
    lines.append(f"Lambda/Kappa: {params.get('kappa', 'NA')}")
    lines.append(f"Training Samples: {training_samples}")
    lines.append(f"Validation Sharpe: {params.get('Sharpe', 'NA')}")
    lines.append("")
    lines.append("二、Bull/Bear统计")
    lines.append(state_stats.to_string(index=False))
    lines.append("")
    lines.append("三、状态转移矩阵")
    lines.append(transition_matrix.to_string())
    lines.append("")
    lines.append("四、状态中心（State Centers）")
    lines.append(state_centroid.to_string(index=False))
    lines.append("")
    lines.append("五、Feature Importance")
    lines.append(feature_weight.to_string(index=False))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    state_stats.to_csv(out_dir / "state_statistics.csv", index=False, encoding="utf-8-sig")
    transition_matrix.to_csv(out_dir / "transition_matrix.csv", encoding="utf-8-sig")


def export_factor_regime_outputs(
    config: FactorRegimeAnalysisConfig,
    feature_df: pd.DataFrame,
    train_outputs: dict[str, Any],
    best_parameter: dict[str, Any] | None,
    training_samples: int | None = None,
) -> dict[str, Any]:
    """Save model, tables, report, and figures for one factor's SJM."""

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_daily = train_outputs["state_daily"].copy()
    required_state_cols = {"trade_date", "state", "state_name", "active_return", config.factor_return_col}
    missing_state_cols = required_state_cols - set(state_daily.columns)
    if missing_state_cols:
        raise ValueError(f"训练输出缺少已冻结的状态语义列: {missing_state_cols}")
    state_daily["trade_date"] = pd.to_datetime(state_daily["trade_date"], errors="coerce")
    state_daily = state_daily.dropna(subset=["trade_date", "state"]).sort_values("trade_date").reset_index(drop=True)
    state_daily.to_csv(out_dir / "sjm_state_daily.csv", index=False, encoding="utf-8-sig")

    feature_weight = train_outputs["feature_weight"].copy()
    state_centroid = train_outputs["state_centroid"].copy()
    transition_matrix = build_transition_matrix(state_daily)
    state_stats = build_state_statistics(
        state_daily,
        factor_return_col=config.factor_return_col,
        annualization=config.annualization,
    )

    state_stats.to_csv(out_dir / "state_statistics.csv", index=False, encoding="utf-8-sig")
    transition_matrix.to_csv(out_dir / "transition_matrix.csv", encoding="utf-8-sig")

    save_model(
        model=train_outputs["model"],
        online_state=train_outputs.get("online_state", OnlineState()),
        config=config,
        feature_weight=feature_weight,
        state_centroid=state_centroid,
        best_parameter=best_parameter,
    )
    write_analysis_report(
        config=config,
        best_parameter=best_parameter,
        state_stats=state_stats,
        transition_matrix=transition_matrix,
        state_centroid=state_centroid,
        feature_weight=feature_weight,
        training_samples=int(training_samples if training_samples is not None else len(feature_df)),
    )
    manifest = {
        "factor_name": config.factor_name,
        "display_name": config.display_name,
        "model_path": config.model_path,
        "state_daily_path": str(out_dir / "sjm_state_daily.csv"),
        "state_statistics_path": str(out_dir / "state_statistics.csv"),
        "transition_matrix_path": str(out_dir / "transition_matrix.csv"),
        "report_path": str(out_dir / "regime_analysis.txt"),
        "best_parameter": best_parameter or {},
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
