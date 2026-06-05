"""AHP-entropy combination weighting for Step 5.

This module calculates feature weights only. It does not run TOPSIS, ranking,
grading, or any downstream scoring model.
"""

from __future__ import annotations

from pathlib import Path
import logging
import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


LOGGER_NAME = "A_MAGE_R3.weighting"

FEATURE_GROUPS = {
    "A1_结构规范性": [
        "I01_核心章节完整率",
        "I02_摘要完整性",
        "I03_图表编号规范率",
        "I04_附录代码存在性",
    ],
    "A2_问题理解与逻辑严密性": [
        "I05_问题重述覆盖率",
        "I06_模型假设与问题匹配度",
        "I07_逻辑连接词密度",
        "I08_结果结论一致性",
    ],
    "A3_方法合理性与数学建模质量": [
        "I09_模型数量与任务匹配度",
        "I10_公式密度",
        "I11_变量定义覆盖率",
        "I12_目标函数约束完整性",
        "I13_方法合理性语义评分",
    ],
    "A4_结果分析与验证": [
        "I14_结果完整率",
        "I15_图表解释率",
        "I16_灵敏度分析存在性",
        "I17_误差分析存在性",
    ],
    "A5_写作规范与应用价值": [
        "I18_参考文献规范率",
        "I19_语言可读性",
        "I20_创新性表达",
        "I21_推广应用价值",
    ],
}

DEFAULT_AHP_GROUP_WEIGHTS = {
    "A1_结构规范性": 0.15,
    "A2_问题理解与逻辑严密性": 0.20,
    "A3_方法合理性与数学建模质量": 0.30,
    "A4_结果分析与验证": 0.20,
    "A5_写作规范与应用价值": 0.15,
}

DEFAULT_AHP_LOCAL_WEIGHTS = {
    "A1_结构规范性": {
        "I01_核心章节完整率": 0.35,
        "I02_摘要完整性": 0.25,
        "I03_图表编号规范率": 0.20,
        "I04_附录代码存在性": 0.20,
    },
    "A2_问题理解与逻辑严密性": {
        "I05_问题重述覆盖率": 0.25,
        "I06_模型假设与问题匹配度": 0.25,
        "I07_逻辑连接词密度": 0.20,
        "I08_结果结论一致性": 0.30,
    },
    "A3_方法合理性与数学建模质量": {
        "I09_模型数量与任务匹配度": 0.25,
        "I10_公式密度": 0.20,
        "I11_变量定义覆盖率": 0.15,
        "I12_目标函数约束完整性": 0.20,
        "I13_方法合理性语义评分": 0.20,
    },
    "A4_结果分析与验证": {
        "I14_结果完整率": 0.35,
        "I15_图表解释率": 0.20,
        "I16_灵敏度分析存在性": 0.25,
        "I17_误差分析存在性": 0.20,
    },
    "A5_写作规范与应用价值": {
        "I18_参考文献规范率": 0.20,
        "I19_语言可读性": 0.20,
        "I20_创新性表达": 0.30,
        "I21_推广应用价值": 0.30,
    },
}

RI_TABLE = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def setup_weighting_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 5 logger."""
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
    """Return the weighting logger."""
    return logging.getLogger(LOGGER_NAME)


def _feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return ordered secondary indicator columns."""
    columns = [column for column in feature_table.columns if isinstance(column, str) and column.startswith("I")]
    return sorted(
        columns,
        key=lambda column: int(re.match(r"I(\d+)", column).group(1)) if re.match(r"I(\d+)", column) else 999,
    )


def _normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a weight dictionary to sum to 1."""
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        equal = 1.0 / len(weights) if weights else 0.0
        return {key: equal for key in weights}
    return {key: float(value) / total for key, value in weights.items()}


def _pairwise_matrix_from_weights(weights: list[float]) -> np.ndarray:
    """Build a perfectly consistent pairwise matrix from target weights."""
    vector = np.asarray(weights, dtype=float)
    if np.any(vector <= 0):
        raise ValueError("AHP weights must be positive.")
    return vector[:, None] / vector[None, :]


def ahp_from_pairwise_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Calculate AHP weights and consistency indicators from a matrix."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("AHP pairwise matrix must be square.")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues.real[max_index])
    weight_vector = np.abs(eigenvectors[:, max_index].real)
    weight_vector = weight_vector / weight_vector.sum()

    ci = 0.0 if n <= 1 else (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.49)
    cr = 0.0 if ri == 0 else ci / ri
    return {
        "weights": weight_vector,
        "lambda_max": lambda_max,
        "ci": float(max(0.0, ci)),
        "ri": ri,
        "cr": float(max(0.0, cr)),
        "is_consistent": cr < 0.10 if ri else True,
    }


def build_ahp_weights(
    group_weights: dict[str, float] | None = None,
    local_weights: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build global AHP subjective weights for all secondary indicators."""
    group_weights = _normalize_weight_dict(group_weights or DEFAULT_AHP_GROUP_WEIGHTS)
    local_weights = local_weights or DEFAULT_AHP_LOCAL_WEIGHTS

    group_names = list(FEATURE_GROUPS)
    group_target = [group_weights[group] for group in group_names]
    group_matrix = _pairwise_matrix_from_weights(group_target)
    group_ahp = ahp_from_pairwise_matrix(group_matrix)

    consistency_rows = [
        {
            "matrix": "criteria_layer",
            "size": len(group_names),
            "lambda_max": group_ahp["lambda_max"],
            "ci": group_ahp["ci"],
            "ri": group_ahp["ri"],
            "cr": group_ahp["cr"],
            "is_consistent": group_ahp["is_consistent"],
        }
    ]

    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(group_names):
        features = FEATURE_GROUPS[group]
        local_target_dict = _normalize_weight_dict(local_weights[group])
        local_target = [local_target_dict[feature] for feature in features]
        local_matrix = _pairwise_matrix_from_weights(local_target)
        local_ahp = ahp_from_pairwise_matrix(local_matrix)
        consistency_rows.append(
            {
                "matrix": group,
                "size": len(features),
                "lambda_max": local_ahp["lambda_max"],
                "ci": local_ahp["ci"],
                "ri": local_ahp["ri"],
                "cr": local_ahp["cr"],
                "is_consistent": local_ahp["is_consistent"],
            }
        )

        for local_index, feature in enumerate(features):
            rows.append(
                {
                    "criterion": group,
                    "indicator": feature,
                    "group_weight": float(group_ahp["weights"][group_index]),
                    "local_weight": float(local_ahp["weights"][local_index]),
                    "ahp_weight": float(group_ahp["weights"][group_index] * local_ahp["weights"][local_index]),
                }
            )

    ahp_table = pd.DataFrame(rows)
    ahp_table["ahp_weight"] = ahp_table["ahp_weight"] / ahp_table["ahp_weight"].sum()
    return ahp_table, pd.DataFrame(consistency_rows)


def _impute_feature_matrix(feature_table: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing normalized features with column medians."""
    matrix = feature_table[feature_columns].apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        missing_count = int(matrix[column].isna().sum())
        if missing_count:
            median = float(matrix[column].median()) if not matrix[column].dropna().empty else 0.0
            matrix[column] = matrix[column].fillna(median)
        else:
            median = float(matrix[column].median()) if not matrix[column].dropna().empty else np.nan

        matrix[column] = matrix[column].clip(lower=0, upper=1)
        rows.append(
            {
                "indicator": column,
                "missing_imputed_count": missing_count,
                "imputation_value": median,
                "zero_count": int((matrix[column] == 0).sum()),
                "mean": float(matrix[column].mean()),
                "std": float(matrix[column].std(ddof=0)),
                "min": float(matrix[column].min()),
                "max": float(matrix[column].max()),
            }
        )
    return matrix, pd.DataFrame(rows)


def entropy_weights(feature_table: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate entropy objective weights using all papers."""
    matrix, imputation_report = _impute_feature_matrix(feature_table, feature_columns)
    values = matrix.to_numpy(dtype=float)
    n_samples, n_features = values.shape
    if n_samples < 2:
        raise ValueError("Entropy weighting requires at least two papers.")

    probabilities = np.zeros_like(values, dtype=float)
    for index in range(n_features):
        column = values[:, index]
        column_sum = column.sum()
        if column_sum <= 0:
            probabilities[:, index] = 1.0 / n_samples
        else:
            probabilities[:, index] = column / column_sum

    k = 1.0 / math.log(n_samples)
    entropy = np.zeros(n_features, dtype=float)
    for index in range(n_features):
        p = probabilities[:, index]
        valid = p > 0
        entropy[index] = -k * np.sum(p[valid] * np.log(p[valid]))

    diversity = 1.0 - entropy
    diversity = np.where(diversity < 0, 0, diversity)
    if diversity.sum() <= 0:
        weights = np.ones(n_features) / n_features
    else:
        weights = diversity / diversity.sum()

    rows = []
    for index, column in enumerate(feature_columns):
        rows.append(
            {
                "indicator": column,
                "entropy": float(entropy[index]),
                "diversity": float(diversity[index]),
                "entropy_weight": float(weights[index]),
            }
        )

    return pd.DataFrame(rows), imputation_report, matrix


def combine_weights(
    ahp_table: pd.DataFrame,
    entropy_table: pd.DataFrame,
    imputation_report: pd.DataFrame,
    alpha: float = 0.6,
) -> pd.DataFrame:
    """Fuse AHP and entropy weights."""
    alpha = float(alpha)
    if not 0 <= alpha <= 1:
        raise ValueError("fusion alpha must be in [0, 1].")

    combined = ahp_table.merge(entropy_table, on="indicator", how="left").merge(
        imputation_report,
        on="indicator",
        how="left",
    )
    combined["combined_weight"] = alpha * combined["ahp_weight"] + (1 - alpha) * combined["entropy_weight"]
    combined["combined_weight"] = combined["combined_weight"] / combined["combined_weight"].sum()
    combined["rank"] = combined["combined_weight"].rank(ascending=False, method="dense").astype(int)
    combined = combined.sort_values(["rank", "indicator"]).reset_index(drop=True)
    return combined


def _set_chinese_font() -> None:
    """Pick a Windows Chinese font for matplotlib if available."""
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


def plot_weights(weight_table: pd.DataFrame, chart_path: Path, top_n: int | None = None) -> Path:
    """Save a horizontal bar chart of combined weights."""
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _set_chinese_font()

    plot_table = weight_table.sort_values("combined_weight", ascending=True)
    if top_n is not None and top_n > 0:
        plot_table = plot_table.tail(top_n)

    height = max(8, 0.42 * len(plot_table) + 2)
    fig, ax = plt.subplots(figsize=(12, height))
    colors = ["#2E7D32" if group.startswith("A3") else "#1565C0" for group in plot_table["criterion"]]
    bars = ax.barh(plot_table["indicator"], plot_table["combined_weight"], color=colors, alpha=0.88)
    ax.set_title("AHP-熵权组合权重", fontsize=15, fontweight="bold")
    ax.set_xlabel("组合权重")
    ax.set_ylabel("二级指标")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=220)
    plt.close(fig)
    return chart_path


def calculate_weights(
    feature_table_path: Path,
    output_path: Path,
    chart_path: Path,
    log_path: Path | None = None,
    alpha: float = 0.6,
) -> dict[str, Any]:
    """Calculate and save AHP, entropy, and fused weights."""
    logger = setup_weighting_logger(log_path) if log_path is not None else _get_logger()
    feature_table_path = Path(feature_table_path)
    output_path = Path(output_path)
    chart_path = Path(chart_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Step 5 weighting: feature_table=%s alpha=%s", feature_table_path, alpha)
    feature_table = pd.read_excel(feature_table_path)
    feature_columns = _feature_columns(feature_table)
    if len(feature_table) != 30:
        logger.warning("Expected 30 papers for entropy weighting, got %s", len(feature_table))
    if len(feature_columns) != 21:
        logger.warning("Expected 21 feature indicators, got %s", len(feature_columns))
    logger.info("Entropy sample size=%s indicators=%s", len(feature_table), len(feature_columns))

    ahp_table, consistency_table = build_ahp_weights()
    entropy_table, imputation_report, imputed_matrix = entropy_weights(feature_table, feature_columns)
    combined_table = combine_weights(ahp_table, entropy_table, imputation_report, alpha=alpha)

    group_summary = (
        combined_table.groupby("criterion", as_index=False)
        .agg(
            ahp_group_weight=("ahp_weight", "sum"),
            entropy_group_weight=("entropy_weight", "sum"),
            combined_group_weight=("combined_weight", "sum"),
        )
        .sort_values("combined_group_weight", ascending=False)
    )

    config_table = pd.DataFrame(
        [
            {"key": "subjective_method", "value": "AHP"},
            {"key": "objective_method", "value": "entropy_weight"},
            {"key": "fusion_formula", "value": "combined = alpha * ahp + (1 - alpha) * entropy"},
            {"key": "fusion_alpha", "value": alpha},
            {"key": "entropy_sample_size", "value": len(feature_table)},
            {"key": "feature_indicator_count", "value": len(feature_columns)},
            {"key": "missing_value_strategy", "value": "column_median_imputation_for_entropy_only"},
            {"key": "runs_topsis", "value": False},
        ]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        combined_table.to_excel(writer, index=False, sheet_name="combined_weights")
        ahp_table.to_excel(writer, index=False, sheet_name="ahp_weights")
        entropy_table.to_excel(writer, index=False, sheet_name="entropy_weights")
        imputation_report.to_excel(writer, index=False, sheet_name="entropy_imputation")
        consistency_table.to_excel(writer, index=False, sheet_name="ahp_consistency")
        group_summary.to_excel(writer, index=False, sheet_name="group_summary")
        config_table.to_excel(writer, index=False, sheet_name="config")

    plot_weights(combined_table, chart_path)

    logger.info("Weight table saved: %s", output_path)
    logger.info("Weight chart saved: %s", chart_path)
    logger.info(
        "AHP CR max=%.6f; entropy missing cells imputed=%s",
        float(consistency_table["cr"].max()),
        int(imputation_report["missing_imputed_count"].sum()),
    )
    logger.info("Finished Step 5 weighting.")
    return {
        "combined_table": combined_table,
        "ahp_table": ahp_table,
        "entropy_table": entropy_table,
        "imputation_report": imputation_report,
        "consistency_table": consistency_table,
        "group_summary": group_summary,
        "imputed_matrix": imputed_matrix,
        "output_path": output_path,
        "chart_path": chart_path,
    }
