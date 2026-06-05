"""TOPSIS base scoring for Step 6.

This module calculates only TOPSIS base scores from normalized features and
the AHP-entropy combined weights. It does not run Bradley-Terry, grade
classification, or any downstream ranking correction model.
"""

from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


LOGGER_NAME = "A_MAGE_R3.topsis"
REQUIRED_SCORE_COLUMNS = [
    "paper_id",
    "filename",
    "feature_quality_flag",
    "score_confidence",
    "D_plus",
    "D_minus",
    "C_i",
    "S_base",
    "rank_base",
]


def setup_topsis_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 6 logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _get_logger() -> logging.Logger:
    """Return the TOPSIS logger."""
    return logging.getLogger(LOGGER_NAME)


def _normalize_id(value: Any) -> str:
    """Normalize paper IDs to two-digit strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d+)", text)
    return match.group(1).zfill(2) if match else text


def _feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return ordered indicator columns from a feature table."""
    columns = [column for column in feature_table.columns if isinstance(column, str) and column.startswith("I")]
    return sorted(
        columns,
        key=lambda column: int(re.match(r"I(\d+)", column).group(1)) if re.match(r"I(\d+)", column) else 999,
    )


def _load_weight_table(weight_path: Path) -> pd.DataFrame:
    """Load combined weights from the Step 5 workbook."""
    weight_path = Path(weight_path)
    try:
        weights = pd.read_excel(weight_path, sheet_name="combined_weights")
    except ValueError:
        weights = pd.read_excel(weight_path)

    required = {"indicator", "combined_weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"权重表缺少必要列: {sorted(missing)}")

    weights = weights.copy()
    weights["combined_weight"] = pd.to_numeric(weights["combined_weight"], errors="coerce")
    if weights["combined_weight"].isna().any():
        bad = weights.loc[weights["combined_weight"].isna(), "indicator"].tolist()
        raise ValueError(f"组合权重列存在非数值项: {bad}")

    total = float(weights["combined_weight"].sum())
    if total <= 0:
        raise ValueError("组合权重之和必须大于0。")
    weights["combined_weight"] = weights["combined_weight"] / total
    return weights


def validate_feature_weight_match(feature_columns: list[str], weight_indicators: list[str]) -> None:
    """Raise a detailed error if feature and weight indicators do not match."""
    feature_set = set(feature_columns)
    weight_set = set(weight_indicators)
    missing_in_weights = sorted(feature_set - weight_set)
    extra_in_weights = sorted(weight_set - feature_set)
    if missing_in_weights or extra_in_weights:
        message = [
            "特征列和权重列不完全匹配。",
            f"特征表有但权重表缺失: {missing_in_weights}",
            f"权重表有但特征表缺失: {extra_in_weights}",
        ]
        raise ValueError("\n".join(message))


def _prepare_feature_matrix(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    weight_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert features to numeric values and impute missing cells for TOPSIS."""
    matrix = feature_table[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputation_lookup = {}
    if "imputation_value" in weight_table.columns:
        imputation_lookup = dict(zip(weight_table["indicator"], weight_table["imputation_value"]))

    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        missing_count = int(matrix[column].isna().sum())
        if missing_count:
            fallback = matrix[column].median()
            impute_value = imputation_lookup.get(column, fallback)
            if pd.isna(impute_value):
                impute_value = 0.0
            impute_value = float(impute_value)
            matrix[column] = matrix[column].fillna(impute_value)
        else:
            impute_value = np.nan

        matrix[column] = matrix[column].clip(lower=0, upper=1)
        rows.append(
            {
                "indicator": column,
                "missing_imputed_count": missing_count,
                "imputation_value_used": impute_value,
                "min_after_imputation": float(matrix[column].min()),
                "max_after_imputation": float(matrix[column].max()),
            }
        )

    return matrix, pd.DataFrame(rows)


def calculate_ideal_solutions(weighted_matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate positive and negative ideal solutions for benefit indicators."""
    positive_ideal = weighted_matrix.max(axis=0)
    negative_ideal = weighted_matrix.min(axis=0)
    return positive_ideal, negative_ideal


def calculate_distances(
    weighted_matrix: pd.DataFrame,
    positive_ideal: pd.Series,
    negative_ideal: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Calculate distances to positive and negative ideal solutions."""
    d_plus = np.sqrt(((weighted_matrix - positive_ideal) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - negative_ideal) ** 2).sum(axis=1))
    return pd.Series(d_plus, index=weighted_matrix.index), pd.Series(d_minus, index=weighted_matrix.index)


def calculate_topsis_scores(
    feature_matrix: pd.DataFrame,
    weights: pd.Series,
) -> dict[str, Any]:
    """Calculate weighted matrix, ideals, distances, closeness, and base score."""
    weights = weights.reindex(feature_matrix.columns)
    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(f"TOPSIS 权重缺失: {missing}")

    weighted_matrix = feature_matrix.mul(weights, axis=1)
    positive_ideal, negative_ideal = calculate_ideal_solutions(weighted_matrix)
    d_plus, d_minus = calculate_distances(weighted_matrix, positive_ideal, negative_ideal)
    denominator = d_plus + d_minus
    closeness = pd.Series(
        np.where(denominator > 0, d_minus / denominator, 0.0),
        index=feature_matrix.index,
    )
    base_score = 100 * closeness

    return {
        "weighted_matrix": weighted_matrix,
        "positive_ideal": positive_ideal,
        "negative_ideal": negative_ideal,
        "D_plus": d_plus,
        "D_minus": d_minus,
        "C_i": closeness,
        "S_base": base_score,
    }


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank TOPSIS scores by descending S_base."""
    ranked = scores.copy()
    ranked["rank_base"] = ranked["S_base"].rank(ascending=False, method="first").astype(int)
    return ranked.sort_values("rank_base").reset_index(drop=True)


def _load_quality_flags(audit_path: Path, paper_ids: pd.Series) -> pd.DataFrame:
    """Load feature quality flags from the audit workbook."""
    audit = pd.read_excel(audit_path, sheet_name="paper_quality_flags")
    audit = audit.copy()
    audit["paper_id_norm"] = audit["paper_id"].apply(_normalize_id)
    flags = audit[["paper_id_norm", "feature_quality_flag"]].drop_duplicates("paper_id_norm")
    expected = set(paper_ids)
    actual = set(flags["paper_id_norm"])
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"feature_quality_audit.xlsx 缺少以下论文的质量标记: {missing}")
    return flags


def _set_chinese_font() -> None:
    """Pick a Chinese font for chart rendering."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_score_distribution(scores: pd.DataFrame, chart_path: Path) -> Path:
    """Save a score distribution chart."""
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _set_chinese_font()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(scores["S_base"], bins=8, color="#2878B5", edgecolor="white", alpha=0.88)
    mean_score = scores["S_base"].mean()
    ax.axvline(mean_score, color="#C82423", linestyle="--", linewidth=1.6, label=f"均值 {mean_score:.2f}")
    ax.set_title("TOPSIS 基础分分布", fontsize=14, fontweight="bold")
    ax.set_xlabel("S_base")
    ax.set_ylabel("论文数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def plot_ranking_bar(scores: pd.DataFrame, chart_path: Path) -> Path:
    """Save a ranking bar chart for all papers."""
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _set_chinese_font()

    plot_data = scores.sort_values("S_base", ascending=True).copy()
    labels = plot_data["paper_id"].astype(str).str.zfill(2) + " (" + plot_data["filename"].astype(str) + ")"
    height = max(8, 0.34 * len(plot_data) + 2)
    fig, ax = plt.subplots(figsize=(11, height))
    bars = ax.barh(labels, plot_data["S_base"], color="#2E7D32", alpha=0.86)
    ax.set_title("TOPSIS 基础评分排名", fontsize=14, fontweight="bold")
    ax.set_xlabel("S_base")
    ax.set_ylabel("论文")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.4, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def run_topsis(
    feature_table_path: Path,
    weight_path: Path,
    audit_path: Path,
    output_path: Path,
    distribution_chart_path: Path,
    ranking_chart_path: Path,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Run TOPSIS and save base scores, diagnostics, and charts."""
    logger = setup_topsis_logger(log_path) if log_path is not None else _get_logger()
    feature_table_path = Path(feature_table_path)
    weight_path = Path(weight_path)
    audit_path = Path(audit_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Step 6 TOPSIS.")
    logger.info("Feature table: %s", feature_table_path)
    logger.info("Weight table: %s", weight_path)
    logger.info("Quality audit: %s", audit_path)

    feature_table = pd.read_excel(feature_table_path)
    feature_columns = _feature_columns(feature_table)
    weight_table = _load_weight_table(weight_path)
    validate_feature_weight_match(feature_columns, weight_table["indicator"].tolist())
    weight_table = weight_table.set_index("indicator").loc[feature_columns].reset_index()
    weights = pd.Series(weight_table["combined_weight"].to_numpy(), index=weight_table["indicator"])

    feature_table = feature_table.copy()
    feature_table["paper_id_norm"] = feature_table["paper_id"].apply(_normalize_id)
    flags = _load_quality_flags(audit_path, feature_table["paper_id_norm"])
    parse_failed_count = int((flags["feature_quality_flag"] == "parse_failed").sum())
    if parse_failed_count:
        logger.warning("Quality audit still contains parse_failed papers: %s", parse_failed_count)
    else:
        logger.info("No parse_failed papers found. score_confidence=usable for all papers.")

    feature_matrix, imputation_report = _prepare_feature_matrix(feature_table, feature_columns, weight_table)
    imputed_cells = int(imputation_report["missing_imputed_count"].sum())
    if imputed_cells:
        logger.warning("TOPSIS feature matrix imputed missing cells: %s", imputed_cells)
        for _, row in imputation_report[imputation_report["missing_imputed_count"] > 0].iterrows():
            logger.warning(
                "%s | missing=%s imputation_value=%s",
                row["indicator"],
                int(row["missing_imputed_count"]),
                row["imputation_value_used"],
            )

    result = calculate_topsis_scores(feature_matrix, weights)
    scores = pd.DataFrame(
        {
            "paper_id": feature_table["paper_id_norm"],
            "filename": feature_table["filename"],
            "D_plus": result["D_plus"],
            "D_minus": result["D_minus"],
            "C_i": result["C_i"],
            "S_base": result["S_base"],
        }
    )
    scores = scores.merge(flags, left_on="paper_id", right_on="paper_id_norm", how="left")
    scores["score_confidence"] = "usable"
    scores = rank_scores(scores)
    scores = scores[REQUIRED_SCORE_COLUMNS]

    ideal_table = pd.DataFrame(
        {
            "indicator": feature_columns,
            "weight": weights.reindex(feature_columns).to_numpy(),
            "positive_ideal": result["positive_ideal"].reindex(feature_columns).to_numpy(),
            "negative_ideal": result["negative_ideal"].reindex(feature_columns).to_numpy(),
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scores.to_excel(writer, index=False, sheet_name="topsis_scores")
        ideal_table.to_excel(writer, index=False, sheet_name="ideal_solutions")
        imputation_report.to_excel(writer, index=False, sheet_name="feature_imputation")
        weight_table.to_excel(writer, index=False, sheet_name="weights_used")

    plot_score_distribution(scores, distribution_chart_path)
    plot_ranking_bar(scores, ranking_chart_path)

    logger.info("TOPSIS scores saved: %s", output_path)
    logger.info("Distribution chart saved: %s", distribution_chart_path)
    logger.info("Ranking chart saved: %s", ranking_chart_path)
    logger.info(
        "Finished Step 6 TOPSIS: papers=%s score_min=%.6f score_max=%.6f",
        len(scores),
        float(scores["S_base"].min()),
        float(scores["S_base"].max()),
    )
    return {
        "scores": scores,
        "ideal_table": ideal_table,
        "imputation_report": imputation_report,
        "output_path": output_path,
        "distribution_chart_path": Path(distribution_chart_path),
        "ranking_chart_path": Path(ranking_chart_path),
    }
