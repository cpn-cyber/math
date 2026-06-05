"""Five-level grade classification for Problem 1.

This module uses the final Step 7C fusion score as the only grading score.
KMeans gives the final grade, while Jenks natural breaks are used as a
robustness check. It does not modify Step 7C results or use the old S_rank
column from Step 7B.
"""

from __future__ import annotations

from pathlib import Path
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


LOGGER_NAME = "A_MAGE_R3.grade_visualization"
GRADE_ORDER = ["优秀", "良好", "中等", "及格", "不及格"]
GRADE_TO_LEVEL = {grade: index for index, grade in enumerate(GRADE_ORDER)}
SCORE_CANDIDATES = ["S_rank_v2", "S_final", "final_score", "S_rank_final"]


def setup_grade_logger(log_path: Path) -> logging.Logger:
    """Create the Step 8 logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8-sig")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _get_logger() -> logging.Logger:
    """Return the Step 8 logger."""
    return logging.getLogger(LOGGER_NAME)


def normalize_id(value: Any) -> str:
    """Normalize paper IDs to two-digit strings."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    match = re.search(r"(\d+)", text)
    return match.group(1).zfill(2) if match else text.zfill(2)


def read_final_scores(rank_fusion_v2_path: Path) -> tuple[pd.DataFrame, str]:
    """Read the final Step 7C fusion table and identify the final score column."""
    logger = _get_logger()
    rank_fusion_v2_path = Path(rank_fusion_v2_path)
    try:
        scores = pd.read_excel(rank_fusion_v2_path, sheet_name="rank_fusion_v2")
    except ValueError:
        scores = pd.read_excel(rank_fusion_v2_path)

    score_column = next((column for column in SCORE_CANDIDATES if column in scores.columns), None)
    if score_column is None:
        available = ", ".join(map(str, scores.columns))
        raise ValueError(
            "未找到 Step 7C 最终融合分数字段。允许字段为 "
            f"{SCORE_CANDIDATES}，不会回退使用旧字段 S_rank。当前字段: {available}"
        )

    if score_column != "S_rank_v2":
        logger.info("Final score field S_rank_v2 not found; using %s instead.", score_column)
    else:
        logger.info("Using final score field: S_rank_v2")

    required = {"paper_id", "filename", "S_base", "S_BT_scaled", score_column, "rank_base"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"最终融合表缺少必要字段: {sorted(missing)}")

    scores = scores.copy()
    scores["paper_id"] = scores["paper_id"].apply(normalize_id)
    for column in ["S_base", "S_BT", "S_BT_scaled", score_column, "rank_base", "rank_fused_v2"]:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")

    if scores[score_column].isna().any():
        bad = scores.loc[scores[score_column].isna(), ["paper_id", "filename"]]
        raise ValueError(f"最终融合分存在缺失或非数值: {bad.to_dict(orient='records')}")

    scores = scores.sort_values([score_column, "paper_id"], ascending=[False, True]).reset_index(drop=True)
    scores["rank_final"] = np.arange(1, len(scores) + 1)

    if "rank_fused_v2" in scores.columns:
        rank_diff = scores["rank_final"] - scores["rank_fused_v2"]
        mismatch_count = int((rank_diff != 0).sum())
        if mismatch_count:
            logger.warning(
                "Recomputed rank_final differs from rank_fused_v2 for %d papers; using rank_final from %s.",
                mismatch_count,
                score_column,
            )

    return scores, score_column


def classify_with_kmeans(
    scores: pd.DataFrame,
    score_column: str,
    n_clusters: int = 5,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Classify final scores into five grades with KMeans."""
    if len(scores) < n_clusters:
        raise ValueError(f"KMeans 分级需要至少 {n_clusters} 篇论文，当前只有 {len(scores)} 篇。")

    result = scores.copy()
    values = result[[score_column]].to_numpy(dtype=float)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    cluster_ids = model.fit_predict(values)
    centers = model.cluster_centers_.reshape(-1)

    ordered_clusters = [cluster for cluster in np.argsort(centers)[::-1]]
    cluster_to_grade = {
        int(cluster): GRADE_ORDER[index] for index, cluster in enumerate(ordered_clusters)
    }

    result["kmeans_cluster"] = cluster_ids.astype(int)
    result["kmeans_center"] = [float(centers[cluster]) for cluster in cluster_ids]
    result["grade_kmeans"] = result["kmeans_cluster"].map(cluster_to_grade)

    if len(set(cluster_ids)) > 1 and len(set(cluster_ids)) < len(result):
        silhouette = float(silhouette_score(values, cluster_ids))
    else:
        silhouette = np.nan

    summary = summarize_grades(
        result,
        score_column=score_column,
        grade_column="grade_kmeans",
        center_column="kmeans_center",
    )
    summary["method"] = "KMeans"
    return result, summary, silhouette


def _jenks_breaks(values: np.ndarray, n_classes: int) -> list[float]:
    """Compute Jenks natural breaks using dynamic programming."""
    data = np.sort(np.asarray(values, dtype=float))
    n_data = len(data)
    if n_classes < 2:
        raise ValueError("Jenks 分类数必须至少为 2。")
    if n_data < n_classes:
        raise ValueError(f"Jenks 分级需要至少 {n_classes} 个样本，当前只有 {n_data} 个。")

    lower = np.zeros((n_data + 1, n_classes + 1), dtype=int)
    variance = np.full((n_data + 1, n_classes + 1), np.inf, dtype=float)

    for class_index in range(1, n_classes + 1):
        lower[1, class_index] = 1
        variance[1, class_index] = 0.0

    for data_count in range(2, n_data + 1):
        sum_values = 0.0
        sum_squares = 0.0
        weight = 0

        for offset in range(1, data_count + 1):
            lower_index = data_count - offset + 1
            value = data[lower_index - 1]
            weight += 1
            sum_values += value
            sum_squares += value * value
            within_class_variance = sum_squares - (sum_values * sum_values) / weight
            previous_index = lower_index - 1

            if previous_index != 0:
                for class_index in range(2, n_classes + 1):
                    candidate = within_class_variance + variance[previous_index, class_index - 1]
                    if variance[data_count, class_index] >= candidate:
                        lower[data_count, class_index] = lower_index
                        variance[data_count, class_index] = candidate

        lower[data_count, 1] = 1
        variance[data_count, 1] = within_class_variance

    breaks = [0.0] * (n_classes + 1)
    breaks[0] = float(data[0])
    breaks[n_classes] = float(data[-1])
    count = n_classes
    k = n_data
    while count >= 2:
        index = int(lower[k, count] - 2)
        breaks[count - 1] = float(data[index])
        k = int(lower[k, count] - 1)
        count -= 1
    return breaks


def classify_with_jenks(
    scores: pd.DataFrame,
    score_column: str,
    n_classes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """Classify final scores into five grades with Jenks natural breaks."""
    result = scores.copy()
    breaks = _jenks_breaks(result[score_column].to_numpy(dtype=float), n_classes)

    ascending_labels = list(reversed(GRADE_ORDER))

    def label_score(value: float) -> str:
        for index in range(1, len(breaks)):
            if value <= breaks[index] + 1e-12:
                return ascending_labels[index - 1]
        return ascending_labels[-1]

    result["grade_jenks"] = result[score_column].apply(label_score)
    summary = summarize_grades(result, score_column, "grade_jenks")
    summary["method"] = "Jenks"
    summary["break_lower"] = [float(breaks[max(0, len(breaks) - index - 2)]) for index in range(len(summary))]
    return result, summary, breaks


def summarize_grades(
    table: pd.DataFrame,
    score_column: str,
    grade_column: str,
    center_column: str | None = None,
) -> pd.DataFrame:
    """Summarize grade counts and score ranges in high-to-low order."""
    rows: list[dict[str, Any]] = []
    for grade in GRADE_ORDER:
        subset = table.loc[table[grade_column] == grade]
        row: dict[str, Any] = {
            "grade": grade,
            "count": int(len(subset)),
            "score_min": float(subset[score_column].min()) if len(subset) else np.nan,
            "score_max": float(subset[score_column].max()) if len(subset) else np.nan,
            "score_mean": float(subset[score_column].mean()) if len(subset) else np.nan,
        }
        if center_column is not None and center_column in table.columns:
            row["center"] = float(subset[center_column].iloc[0]) if len(subset) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare_grade_methods(result: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Compare KMeans and Jenks grade labels paper by paper."""
    comparison = result[
        [
            "paper_id",
            "filename",
            "S_rank_v2" if "S_rank_v2" in result.columns else "rank_final",
            "rank_final",
            "grade_kmeans",
            "grade_jenks",
        ]
    ].copy()
    score_column = "S_rank_v2" if "S_rank_v2" in comparison.columns else comparison.columns[2]
    comparison = comparison.rename(columns={score_column: "final_score"})
    comparison["is_same_grade"] = comparison["grade_kmeans"] == comparison["grade_jenks"]
    comparison["grade_level_gap"] = comparison["grade_kmeans"].map(GRADE_TO_LEVEL) - comparison[
        "grade_jenks"
    ].map(GRADE_TO_LEVEL)
    consistency = float(comparison["is_same_grade"].mean()) if len(comparison) else np.nan
    return consistency, comparison


def find_boundary_nearby_papers(
    result: pd.DataFrame,
    score_column: str,
    center_column: str = "kmeans_center",
) -> pd.DataFrame:
    """Find papers near KMeans grade boundaries."""
    centers = (
        result[["grade_kmeans", center_column]]
        .drop_duplicates()
        .sort_values(center_column, ascending=False)
        .reset_index(drop=True)
    )
    if len(centers) < 2:
        return pd.DataFrame()

    score_range = float(result[score_column].max() - result[score_column].min())
    threshold = max(1.0, 0.02 * score_range)
    boundaries: list[dict[str, Any]] = []
    for index in range(len(centers) - 1):
        high_grade = centers.loc[index, "grade_kmeans"]
        low_grade = centers.loc[index + 1, "grade_kmeans"]
        boundary = float((centers.loc[index, center_column] + centers.loc[index + 1, center_column]) / 2)
        nearby = result.loc[(result[score_column] - boundary).abs() <= threshold]
        for _, row in nearby.iterrows():
            boundaries.append(
                {
                    "paper_id": row["paper_id"],
                    "filename": row["filename"],
                    "rank_final": int(row["rank_final"]),
                    "S_rank_v2": float(row[score_column]),
                    "grade_kmeans": row["grade_kmeans"],
                    "boundary_between": f"{high_grade}-{low_grade}",
                    "boundary_score": boundary,
                    "distance_to_boundary": float(abs(row[score_column] - boundary)),
                    "nearby_threshold": threshold,
                }
            )
    return pd.DataFrame(boundaries).sort_values(
        ["boundary_score", "distance_to_boundary"], ascending=[False, True]
    ).reset_index(drop=True)


def merge_classifications(
    scores: pd.DataFrame,
    kmeans_result: pd.DataFrame,
    jenks_result: pd.DataFrame,
) -> pd.DataFrame:
    """Merge KMeans and Jenks labels and set final grade from KMeans."""
    final = kmeans_result.copy()
    final["grade_jenks"] = jenks_result.set_index("paper_id").loc[final["paper_id"], "grade_jenks"].to_numpy()
    final["grade_final"] = final["grade_kmeans"]
    return final.sort_values("rank_final").reset_index(drop=True)

