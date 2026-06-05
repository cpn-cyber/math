"""Visualization helpers for Problem 1 final outputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


GRADE_ORDER = ["优秀", "良好", "中等", "及格", "不及格"]
GRADE_COLORS = {
    "优秀": "#2E7D32",
    "良好": "#1976D2",
    "中等": "#F9A825",
    "及格": "#EF6C00",
    "不及格": "#C62828",
}
CRITERION_LABELS = {
    "A1": "A1结构",
    "A2": "A2逻辑",
    "A3": "A3建模",
    "A4": "A4结果",
    "A5": "A5写作",
}


def set_chinese_font() -> None:
    """Use a Chinese font if the runtime provides one."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d+)", text)
    return match.group(1).zfill(2) if match else text.zfill(2)


def plot_final_ranking_bar(final_table: pd.DataFrame, chart_path: Path) -> Path:
    """Plot final S_rank_v2 values by final rank."""
    _ensure_parent(chart_path)
    table = final_table.sort_values("rank_final")
    colors = [GRADE_COLORS.get(grade, "#666666") for grade in table["grade_final"]]

    fig, ax = plt.subplots(figsize=(16, 7.5))
    bars = ax.bar(table["paper_id"], table["S_rank_v2"], color=colors, edgecolor="#222222", linewidth=0.3)
    ax.set_title("问题1最终融合分排名", fontsize=18, fontweight="bold")
    ax.set_xlabel("论文编号")
    ax.set_ylabel("S_rank_v2")
    ax.set_ylim(0, max(100, float(table["S_rank_v2"].max()) + 8))
    ax.grid(axis="y", alpha=0.25)
    for bar, score in zip(bars, table["S_rank_v2"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    handles = [
        plt.Line2D([0], [0], color=GRADE_COLORS[grade], lw=8, label=grade) for grade in GRADE_ORDER
    ]
    ax.legend(handles=handles, ncol=5, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return Path(chart_path)


def plot_grade_distribution(grade_summary: pd.DataFrame, chart_path: Path) -> Path:
    """Plot grade counts."""
    _ensure_parent(chart_path)
    summary = grade_summary.set_index("grade").reindex(GRADE_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [GRADE_COLORS[grade] for grade in summary["grade"]]
    bars = ax.bar(summary["grade"], summary["count"], color=colors, edgecolor="#222222", linewidth=0.4)
    ax.set_title("五级质量等级分布", fontsize=16, fontweight="bold")
    ax.set_xlabel("等级")
    ax.set_ylabel("论文数量")
    ax.set_ylim(0, max(1, int(summary["count"].max())) + 2)
    ax.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, summary["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(int(count)), ha="center")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return Path(chart_path)


def plot_final_score_histogram(final_table: pd.DataFrame, chart_path: Path) -> Path:
    """Plot final score histogram."""
    _ensure_parent(chart_path)
    scores = final_table["S_rank_v2"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=8, color="#4E79A7", edgecolor="white", alpha=0.9)
    ax.axvline(scores.mean(), color="#E15759", linestyle="--", linewidth=2, label=f"均值 {scores.mean():.2f}")
    ax.set_title("最终融合分直方图", fontsize=16, fontweight="bold")
    ax.set_xlabel("S_rank_v2")
    ax.set_ylabel("论文数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return Path(chart_path)


def plot_kmeans_grade_scatter(final_table: pd.DataFrame, chart_path: Path) -> Path:
    """Plot final score scatter colored by KMeans grade."""
    _ensure_parent(chart_path)
    table = final_table.sort_values("rank_final")

    fig, ax = plt.subplots(figsize=(13, 6))
    for grade in GRADE_ORDER:
        subset = table.loc[table["grade_kmeans"] == grade]
        ax.scatter(
            subset["rank_final"],
            subset["S_rank_v2"],
            s=78,
            color=GRADE_COLORS[grade],
            label=grade,
            edgecolor="#222222",
            linewidth=0.4,
        )
    ax.set_title("KMeans五级分级散点图", fontsize=16, fontweight="bold")
    ax.set_xlabel("最终排名")
    ax.set_ylabel("S_rank_v2")
    ax.set_xticks(range(1, len(table) + 1, 2))
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return Path(chart_path)


def _indicator_number(column: str) -> int:
    match = re.match(r"I(\d+)", str(column))
    return int(match.group(1)) if match else 999


def calculate_primary_indicator_scores(
    feature_table: pd.DataFrame,
    weight_table: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate 21 normalized indicators into five primary criterion scores."""
    features = feature_table.copy()
    features["paper_id"] = features["paper_id"].apply(_normalize_id)

    weights = weight_table.copy()
    if "criterion" not in weights.columns or "indicator" not in weights.columns:
        raise ValueError("权重表必须包含 criterion 和 indicator 字段，才能计算一级指标雷达图。")
    if "combined_weight" not in weights.columns:
        raise ValueError("权重表必须包含 combined_weight 字段，才能计算一级指标雷达图。")

    rows = features[["paper_id", "filename"]].copy()
    for criterion, group in weights.groupby("criterion", sort=False):
        indicator_columns = [column for column in group["indicator"].tolist() if column in features.columns]
        if not indicator_columns:
            continue
        group_weights = group.set_index("indicator").loc[indicator_columns, "combined_weight"].astype(float)
        values = features[indicator_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        weighted = values.mul(group_weights, axis=1).sum(axis=1)
        weight_sum = float(group_weights.sum())
        score = weighted / weight_sum if weight_sum > 0 else values.mean(axis=1)
        match = re.match(r"(A\d+)", str(criterion))
        column_name = CRITERION_LABELS.get(match.group(1), str(criterion)) if match else str(criterion)
        rows[column_name] = score.clip(lower=0, upper=1)

    ordered_columns = ["paper_id", "filename"] + [
        CRITERION_LABELS[key] for key in ["A1", "A2", "A3", "A4", "A5"] if CRITERION_LABELS[key] in rows.columns
    ]
    return rows[ordered_columns]


def _select_representative_papers(final_table: pd.DataFrame) -> list[str]:
    """Select one excellent, one medium, and one fail paper for the radar chart."""
    selections: list[str] = []
    rules = [
        ("优秀", "highest"),
        ("中等", "median"),
        ("不及格", "lowest"),
    ]
    for grade, mode in rules:
        subset = final_table.loc[final_table["grade_final"] == grade].copy()
        if subset.empty:
            continue
        if mode == "highest":
            row = subset.sort_values("S_rank_v2", ascending=False).iloc[0]
        elif mode == "lowest":
            row = subset.sort_values("S_rank_v2", ascending=True).iloc[0]
        else:
            median = float(subset["S_rank_v2"].median())
            row = subset.assign(distance=(subset["S_rank_v2"] - median).abs()).sort_values("distance").iloc[0]
        selections.append(str(row["paper_id"]).zfill(2))
    return selections


def plot_high_mid_low_radar(
    final_table: pd.DataFrame,
    primary_scores: pd.DataFrame,
    chart_path: Path,
) -> Path:
    """Plot radar chart for representative excellent, medium, and fail papers."""
    _ensure_parent(chart_path)
    selected_ids = _select_representative_papers(final_table)
    if not selected_ids:
        raise ValueError("没有可用于雷达图的代表论文。")

    score_columns = [column for column in primary_scores.columns if column not in {"paper_id", "filename"}]
    angles = np.linspace(0, 2 * np.pi, len(score_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for paper_id in selected_ids:
        row = primary_scores.loc[primary_scores["paper_id"] == paper_id]
        if row.empty:
            continue
        meta = final_table.loc[final_table["paper_id"] == paper_id].iloc[0]
        values = row.iloc[0][score_columns].astype(float).tolist()
        values += values[:1]
        label = f"{paper_id} {meta['grade_final']}({meta['S_rank_v2']:.1f})"
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.10)

    ax.set_title("优秀-中等-不及格代表论文一级指标雷达图", fontsize=15, fontweight="bold", pad=22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(score_columns)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return Path(chart_path)


def plot_kmeans_jenks_comparison(final_table: pd.DataFrame, chart_path: Path) -> Path:
    """Plot KMeans and Jenks grade levels for comparison."""
    _ensure_parent(chart_path)
    grade_to_level = {grade: index + 1 for index, grade in enumerate(reversed(GRADE_ORDER))}
    table = final_table.sort_values("rank_final").copy()
    table["kmeans_level"] = table["grade_kmeans"].map(grade_to_level)
    table["jenks_level"] = table["grade_jenks"].map(grade_to_level)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(table["rank_final"], table["kmeans_level"], marker="o", label="KMeans", linewidth=2)
    ax.plot(table["rank_final"], table["jenks_level"], marker="s", label="Jenks", linewidth=2, alpha=0.8)
    ax.set_title("KMeans 与 Jenks 等级对比", fontsize=16, fontweight="bold")
    ax.set_xlabel("最终排名")
    ax.set_ylabel("等级")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(list(reversed(GRADE_ORDER)))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return Path(chart_path)


def create_problem1_visualizations(
    final_table: pd.DataFrame,
    grade_summary: pd.DataFrame,
    feature_table: pd.DataFrame,
    weight_table: pd.DataFrame,
    chart_paths: dict[str, Path],
) -> dict[str, Path | pd.DataFrame]:
    """Create all final Problem 1 charts."""
    set_chinese_font()
    primary_scores = calculate_primary_indicator_scores(feature_table, weight_table)
    paths: dict[str, Path | pd.DataFrame] = {
        "final_ranking_bar": plot_final_ranking_bar(final_table, chart_paths["final_ranking_bar"]),
        "grade_distribution": plot_grade_distribution(grade_summary, chart_paths["grade_distribution"]),
        "final_score_histogram": plot_final_score_histogram(final_table, chart_paths["final_score_histogram"]),
        "kmeans_grade_scatter": plot_kmeans_grade_scatter(final_table, chart_paths["kmeans_grade_scatter"]),
        "high_mid_low_radar": plot_high_mid_low_radar(
            final_table,
            primary_scores,
            chart_paths["high_mid_low_radar"],
        ),
        "kmeans_jenks_comparison": plot_kmeans_jenks_comparison(
            final_table,
            chart_paths["kmeans_jenks_comparison"],
        ),
        "primary_scores": primary_scores,
    }
    return paths
