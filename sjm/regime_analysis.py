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
from sjm.tuner import run_sjm_hyperparameter_tuning


@dataclass
class FactorRegimeAnalysisConfig:
    """File destinations and display settings for one factor analysis."""

    factor_name: str
    display_name: str
    output_dir: str
    result_dir: str
    model_path: str
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


def classify_bull_bear(state_daily: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
    """Automatically label and renumber states by active-return economics.

    The primary sorting key is average active return.  Mean factor return and
    information ratio are retained as diagnostics and used only as tie-breakers.
    The lowest-scoring state is Bear and the highest-scoring state is Bull.
    """

    required = {"trade_date", "state", "active_return", "momentum_return"}
    missing = required - set(state_daily.columns)
    if missing:
        raise ValueError(f"状态数据缺少列，无法自动识别Bull/Bear: {missing}")

    data = state_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["active_return"] = pd.to_numeric(data["active_return"], errors="coerce")
    data["momentum_return"] = pd.to_numeric(data["momentum_return"], errors="coerce")
    data = data.dropna(subset=["trade_date", "state", "active_return", "momentum_return"])

    rows = []
    for state, grp in data.groupby("state", observed=True):
        rows.append(
            {
                "state": int(state),
                "mean_active_return": float(grp["active_return"].mean()),
                "mean_factor_return": float(grp["momentum_return"].mean()),
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


def build_state_statistics(state_daily: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
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
        factor_ret = pd.to_numeric(grp["momentum_return"], errors="coerce")
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


def _setup_matplotlib():
    """Import matplotlib and set fonts for Chinese-capable output."""

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _state_colors() -> dict[str, tuple[float, float, float, float]]:
    """Return consistent Bull/Bear colors across all figures."""

    return {
        "Bull": (0.18, 0.62, 0.30, 0.22),
        "Bear": (0.86, 0.22, 0.18, 0.22),
    }


def _shade_states(ax, state_daily: pd.DataFrame) -> None:
    """Shade the background by state regime on an existing axis."""

    colors = _state_colors()
    data = state_daily.sort_values("trade_date")
    dates = data["trade_date"].to_numpy()
    states = data["state_name"].astype(str).to_numpy()
    if len(data) == 0:
        return
    start = 0
    for i in range(1, len(data)):
        if states[i] != states[i - 1]:
            ax.axvspan(dates[start], dates[i], color=colors.get(states[i - 1], (0.7, 0.7, 0.7, 0.15)), linewidth=0)
            start = i
    ax.axvspan(dates[start], dates[-1], color=colors.get(states[-1], (0.7, 0.7, 0.7, 0.15)), linewidth=0)


def _save_fig(fig, path: Path, dpi: int = 180) -> None:
    """Persist a matplotlib figure and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clf()


def plot_regime_analysis(
    config: FactorRegimeAnalysisConfig,
    feature_df: pd.DataFrame,
    state_daily: pd.DataFrame,
    feature_weight: pd.DataFrame,
    state_centroid: pd.DataFrame,
    transition_matrix: pd.DataFrame,
) -> dict[str, str]:
    """Create the complete visualization pack required for one factor."""

    plt = _setup_matplotlib()
    fig_dir = Path(config.output_dir) / "figures"
    data = state_daily.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date", "active_return", "market_return", "state_name"]).sort_values("trade_date")
    data["active_nav"] = (1.0 + data["active_return"].fillna(0.0)).cumprod()
    data["market_nav"] = (1.0 + data["market_return"].fillna(0.0)).cumprod()
    data["bull_active_nav"] = (1.0 + data["active_return"].where(data["state_name"].eq("Bull"), 0.0)).cumprod()
    data["bear_active_nav"] = (1.0 + data["active_return"].where(data["state_name"].eq("Bear"), 0.0)).cumprod()
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_states(ax, data)
    ax.plot(data["trade_date"], data["active_return"], color="#2b4c7e", linewidth=1.0)
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{config.display_name} Active Return with Bull/Bear Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Return")
    paths["active_return"] = str(fig_dir / "01_active_return_regime.png")
    _save_fig(fig, Path(paths["active_return"]))

    fig, ax = plt.subplots(figsize=(14, 3.8))
    color_map = {"Bull": "#2f9e44", "Bear": "#d94841"}
    colors = data["state_name"].map(lambda x: color_map.get(str(x), "#777777"))
    ax.scatter(data["trade_date"], data["state"], c=colors, s=8)
    ax.step(data["trade_date"], data["state"], where="post", color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{config.display_name} State Sequence")
    ax.set_xlabel("Date")
    ax.set_ylabel("State")
    paths["state_sequence"] = str(fig_dir / "02_state_sequence.png")
    _save_fig(fig, Path(paths["state_sequence"]))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data["trade_date"], data["bull_active_nav"], color="#2f9e44", linewidth=1.5, label="Bull Active NAV")
    ax.plot(data["trade_date"], data["bear_active_nav"], color="#d94841", linewidth=1.5, label="Bear Active NAV")
    ax.plot(data["trade_date"], data["market_nav"], color="#364fc7", linewidth=1.3, label="Market NAV")
    ax.set_title(f"{config.display_name} Cumulative Active Return by Regime")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend(loc="best")
    paths["cumulative"] = str(fig_dir / "03_cumulative_active_market.png")
    _save_fig(fig, Path(paths["cumulative"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for state_name, color in [("Bull", "#2f9e44"), ("Bear", "#d94841")]:
        ret = data.loc[data["state_name"].eq(state_name), "active_return"].dropna()
        if not ret.empty:
            ax.hist(ret, bins=40, density=True, alpha=0.45, color=color, label=state_name)
            ret.plot(kind="kde", ax=ax, color=color, linewidth=1.8)
    ax.set_title(f"{config.display_name} Bull/Bear Active Return Distribution")
    ax.set_xlabel("Active Return")
    ax.legend(loc="best")
    paths["distribution"] = str(fig_dir / "04_return_distribution.png")
    _save_fig(fig, Path(paths["distribution"]))

    top_features = feature_weight.copy()
    if {"feature", "weight"}.issubset(top_features.columns):
        top_features["abs_weight"] = top_features["weight"].abs()
        selected_features = top_features.sort_values("abs_weight", ascending=False)["feature"].head(8).tolist()
    else:
        selected_features = [c for c in state_centroid.columns if c.startswith("z_")][:8]
    radar_df = state_centroid[state_centroid["state_name"].isin(["Bull", "Bear"])].copy()
    radar_features = [f for f in selected_features if f in radar_df.columns]
    if len(radar_df) >= 2 and radar_features:
        labels = [name.replace("z_", "") for name in radar_features]
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        for _, row in radar_df.iterrows():
            values = [float(row[f]) for f in radar_features]
            values += values[:1]
            color = "#2f9e44" if row["state_name"] == "Bull" else "#d94841"
            ax.plot(angles, values, color=color, linewidth=1.6, label=row["state_name"])
            ax.fill(angles, values, color=color, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{config.display_name} State Centers Radar")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        paths["radar"] = str(fig_dir / "05_state_centers_radar.png")
        _save_fig(fig, Path(paths["radar"]))

    if {"feature", "weight"}.issubset(feature_weight.columns):
        fw = feature_weight.copy()
        fw["abs_weight"] = fw["weight"].abs()
        fw = fw.sort_values("abs_weight", ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(fw["feature"], fw["weight"], color="#4c6ef5")
        ax.set_title(f"{config.display_name} SJM Feature Importance")
        ax.set_xlabel("Weight")
        paths["feature_importance"] = str(fig_dir / "06_feature_importance.png")
        _save_fig(fig, Path(paths["feature_importance"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transition_matrix.to_numpy(dtype=float), cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(transition_matrix.columns)))
    ax.set_xticklabels(transition_matrix.columns)
    ax.set_yticks(range(len(transition_matrix.index)))
    ax.set_yticklabels(transition_matrix.index)
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            ax.text(j, i, f"{transition_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="#111111")
    ax.set_title(f"{config.display_name} Transition Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    paths["transition_heatmap"] = str(fig_dir / "07_transition_matrix_heatmap.png")
    _save_fig(fig, Path(paths["transition_heatmap"]))

    data["rolling_active_return"] = data["active_return"].rolling(config.rolling_window, min_periods=5).mean()
    switch_mask = data["state_name"].ne(data["state_name"].shift(1))
    fig, ax = plt.subplots(figsize=(14, 5))
    _shade_states(ax, data)
    ax.plot(data["trade_date"], data["rolling_active_return"], color="#1c7ed6", linewidth=1.5)
    for x in data.loc[switch_mask, "trade_date"].iloc[1:]:
        ax.axvline(x, color="#555555", linewidth=0.6, linestyle="--", alpha=0.35)
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{config.display_name} Rolling Active Return and Regime Switches")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{config.rolling_window}D Rolling Active Return")
    paths["rolling"] = str(fig_dir / "08_rolling_active_return_switches.png")
    _save_fig(fig, Path(paths["rolling"]))

    feat = feature_df.copy()
    feat["trade_date"] = pd.to_datetime(feat["trade_date"], errors="coerce")
    feature_extra_cols = [c for c in feat.columns if c not in {"active_return", "momentum_return", "market_return"}]
    merged = data[["trade_date", "active_return", "state_name"]].merge(
        feat[feature_extra_cols],
        on="trade_date",
        how="left",
    )
    summary_cols = [
        ("active_return", "Active Return"),
        ("rsi_21", "RSI(21)"),
        ("macd_8_21", "MACD(8,21)"),
        ("ewma_active_return_21", "EWMA Active Return(21)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [2, 0.35, 1, 1, 1]})
    _shade_states(axes[0], merged)
    axes[0].plot(merged["trade_date"], merged["active_return"], color="#2b4c7e", linewidth=1.0)
    axes[0].axhline(0, color="#333333", linewidth=0.7)
    axes[0].set_ylabel("Active")
    colors = {"Bull": "#2f9e44", "Bear": "#d94841"}
    axes[1].set_yticks([])
    axes[1].set_ylabel("State")
    for state_name, grp in merged.groupby((merged["state_name"] != merged["state_name"].shift()).cumsum()):
        label = str(grp["state_name"].iloc[0])
        axes[1].axvspan(grp["trade_date"].iloc[0], grp["trade_date"].iloc[-1], color=colors.get(label, "#777777"), alpha=0.85)
    for ax, (col, label) in zip(axes[2:], summary_cols[1:]):
        if col in merged.columns:
            ax.plot(merged["trade_date"], merged[col], linewidth=1.0, color="#495057")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"{config.display_name} Paper-style Regime Summary")
    axes[-1].set_xlabel("Date")
    paths["paper_summary"] = str(fig_dir / "09_paper_style_summary.png")
    _save_fig(fig, Path(paths["paper_summary"]))

    return paths


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

    state_daily = classify_bull_bear(train_outputs["state_daily"], annualization=config.annualization)
    state_daily.to_csv(out_dir / "sjm_state_daily.csv", index=False, encoding="utf-8-sig")

    feature_weight = train_outputs["feature_weight"].copy()
    state_centroid = train_outputs["state_centroid"].copy()
    if "raw_state" in state_daily.columns and "state" in state_centroid.columns:
        remap = (
            state_daily[["raw_state", "state", "state_name"]]
            .drop_duplicates()
            .set_index("raw_state")
            .to_dict()
        )
        state_centroid["raw_state"] = state_centroid["state"].astype(int)
        state_centroid["state"] = state_centroid["raw_state"].map(remap["state"]).fillna(state_centroid["state"]).astype(int)
        state_centroid["state_name"] = state_centroid["state"].map(
            {int(v): str(k) for k, v in state_daily[["state_name", "state"]].drop_duplicates().itertuples(index=False)}
        ).fillna(state_centroid["state_name"])
        state_centroid = state_centroid.sort_values("state").reset_index(drop=True)
        state_centroid.to_csv(out_dir / "sjm_state_centroid.csv", index=False, encoding="utf-8-sig")
    transition_matrix = build_transition_matrix(state_daily)
    state_stats = build_state_statistics(state_daily, annualization=config.annualization)

    state_stats.to_csv(out_dir / "state_statistics.csv", index=False, encoding="utf-8-sig")
    transition_matrix.to_csv(out_dir / "transition_matrix.csv", encoding="utf-8-sig")

    save_model(
        model=train_outputs["model"],
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
    figure_paths = plot_regime_analysis(
        config=config,
        feature_df=feature_df,
        state_daily=state_daily,
        feature_weight=feature_weight,
        state_centroid=state_centroid,
        transition_matrix=transition_matrix,
    )

    manifest = {
        "factor_name": config.factor_name,
        "display_name": config.display_name,
        "model_path": config.model_path,
        "state_daily_path": str(out_dir / "sjm_state_daily.csv"),
        "state_statistics_path": str(out_dir / "state_statistics.csv"),
        "transition_matrix_path": str(out_dir / "transition_matrix.csv"),
        "report_path": str(out_dir / "regime_analysis.txt"),
        "figures": figure_paths,
        "best_parameter": best_parameter or {},
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
