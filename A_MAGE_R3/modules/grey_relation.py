"""Grey relational analysis and preliminary key-feature index for Problem 2."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path
from modules.deep_quality_features import DEEP_FEATURE_COLUMNS
from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS


LOGGER_NAME = "A_MAGE_R3.problem2.grey_key_index"
NEGATIVE_FEATURES = {"stacking_penalty"}
RHO = 0.5


def setup_grey_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 15 logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def _feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Return ordered Problem 2 feature columns."""
    expected = list(SURFACE_FEATURE_COLUMNS) + list(DEEP_FEATURE_COLUMNS)
    return [column for column in expected if column in matrix.columns]


def _feature_group(feature: str) -> str:
    """Return feature group."""
    return "deep" if feature in DEEP_FEATURE_COLUMNS else "surface"


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize one sequence; constants become zeros."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return values * np.nan
    minimum = float(valid.min())
    maximum = float(valid.max())
    if maximum == minimum:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _normalize_scores(scores: pd.Series) -> pd.Series:
    """Normalize grey relation scores into [0,1]."""
    valid = scores.dropna()
    if valid.empty:
        return scores * np.nan
    minimum = float(valid.min())
    maximum = float(valid.max())
    if maximum == minimum:
        return scores.apply(lambda value: np.nan if pd.isna(value) else 1.0)
    return (scores - minimum) / (maximum - minimum)


def _interpret_grey(feature: str, use_in_model: str, variance_flag: str, grey_score: float) -> str:
    """Build grey relation interpretation."""
    if use_in_model == "False" or variance_flag == "constant_feature":
        return "常数特征，保留记录但不进入主灰色关联排序"
    if feature == "appendix_code_presence":
        return "近似常数特征，可计算但需谨慎解释"
    if feature == "stacking_penalty":
        return "堆砌惩罚为负向特征，值越大表示堆砌风险越高；灰色关联仅反映序列贴近程度"
    if feature == "total_chars":
        return "篇幅与质量标签序列贴近，宜解释为信息承载量和完整性相关，不代表越长越好"
    if feature in DEEP_FEATURE_COLUMNS:
        return "深层质量特征与弱监督质量标签存在序列贴近性，可作为解释质量差异的候选依据"
    return "该表层特征与弱监督质量标签存在一定序列贴近性"


def calculate_grey_relation(
    features: pd.DataFrame,
    quality_label: pd.Series | None = None,
    variance_filter: pd.DataFrame | None = None,
    rho: float = RHO,
) -> pd.DataFrame:
    """Calculate grey relational degrees between features and Q_label."""
    feature_columns = _feature_columns(features)
    target = pd.to_numeric(quality_label if quality_label is not None else features["Q_label"], errors="coerce")
    target_norm = _minmax(target)

    variance_lookup: dict[str, dict[str, Any]] = {}
    if variance_filter is not None and not variance_filter.empty:
        variance_lookup = variance_filter.set_index("feature_name").to_dict(orient="index")

    diff_lookup: dict[str, pd.Series] = {}
    all_diffs: list[float] = []
    for feature in feature_columns:
        compare = _minmax(features[feature])
        diff = (target_norm - compare).abs()
        diff_lookup[feature] = diff
        all_diffs.extend(diff.dropna().astype(float).tolist())

    if all_diffs:
        global_min = float(np.min(all_diffs))
        global_max = float(np.max(all_diffs))
    else:
        global_min = 0.0
        global_max = 0.0

    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        info = variance_lookup.get(feature, {})
        variance_flag = str(info.get("variance_flag", "normal"))
        use_in_model = str(info.get("use_in_model", "True"))
        diff = diff_lookup[feature]
        if global_max == 0:
            coefficients = pd.Series(np.ones(len(diff)), index=diff.index, dtype=float)
        else:
            coefficients = (global_min + rho * global_max) / (diff + rho * global_max)
        score = float(coefficients.mean()) if not coefficients.dropna().empty else np.nan
        if use_in_model == "False" or variance_flag == "constant_feature":
            grey_status = "excluded_constant"
        elif variance_flag == "low_variance_feature" or use_in_model == "Caution":
            grey_status = "caution"
        else:
            grey_status = "included"
        rows.append(
            {
                "feature_name": feature,
                "feature_group": _feature_group(feature),
                "is_negative_feature": feature in NEGATIVE_FEATURES,
                "variance_flag": variance_flag,
                "use_in_model": use_in_model,
                "grey_status": grey_status,
                "grey_relation_score": score,
                "grey_norm": np.nan,
                "interpretation": _interpret_grey(feature, use_in_model, variance_flag, score),
            }
        )

    grey = pd.DataFrame(rows)
    include_mask = grey["use_in_model"].ne("False")
    grey.loc[include_mask, "grey_norm"] = _normalize_scores(grey.loc[include_mask, "grey_relation_score"])
    grey.loc[~include_mask, "grey_norm"] = np.nan
    return grey.sort_values(["use_in_model", "grey_relation_score"], ascending=[False, False]).reset_index(drop=True)


def _key_interpretation(row: pd.Series) -> str:
    """Build preliminary key-feature interpretation."""
    feature = str(row["feature_name"])
    if str(row.get("use_in_model")) == "False":
        return "常数特征，不进入 K_pre 主排序"
    if feature == "appendix_code_presence":
        return "近似常数特征，K_pre 仅作谨慎参考"
    if feature == "total_chars":
        return "篇幅排名靠前应解释为信息承载量与结构完整性相关，不能解释为篇幅越长越好"
    if feature == "stacking_penalty":
        return "堆砌惩罚单变量解释力有限，但后续可作为 QAF 负向约束项"
    if str(row.get("feature_group")) == "deep":
        return "深层质量特征排名靠前，支持其比单纯篇幅统计更能解释质量差异"
    return "表层文本特征对弱监督质量差异具有初步解释价值"


def build_key_feature_index(
    correlation_table: pd.DataFrame,
    grey_table: pd.DataFrame,
    vip_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build preliminary K_pre without VIP/Bootstrap."""
    merged = correlation_table.merge(
        grey_table[["feature_name", "grey_norm", "grey_relation_score", "grey_status"]],
        on="feature_name",
        how="left",
    )
    merged["spearman_abs"] = pd.to_numeric(merged["spearman_abs"], errors="coerce")
    merged["grey_norm"] = pd.to_numeric(merged["grey_norm"], errors="coerce")
    merged["K_pre"] = 0.6 * merged["spearman_abs"].fillna(0.0) + 0.4 * merged["grey_norm"].fillna(0.0)
    merged.loc[merged["use_in_model"].astype(str).eq("False"), "K_pre"] = np.nan
    rankable = merged.loc[merged["use_in_model"].astype(str).ne("False")].copy()
    rankable["rank_pre"] = rankable["K_pre"].rank(ascending=False, method="first").astype(int)
    merged = merged.merge(rankable[["feature_name", "rank_pre"]], on="feature_name", how="left")
    merged["interpretation"] = merged.apply(_key_interpretation, axis=1)
    columns = [
        "feature_name",
        "feature_group",
        "spearman_corr",
        "spearman_abs",
        "grey_norm",
        "K_pre",
        "rank_pre",
        "variance_flag",
        "use_in_model",
        "interpretation",
    ]
    return merged[columns].sort_values(["use_in_model", "rank_pre"], ascending=[False, True]).reset_index(drop=True)


def _save_grey_bar(grey: pd.DataFrame, chart_path: Path) -> None:
    """Save grey relation Top feature chart."""
    top = grey.loc[grey["use_in_model"].ne("False")].sort_values("grey_relation_score", ascending=False).head(12).iloc[::-1]
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    colors = ["#b64b4b" if value else "#3f6fa9" for value in top["is_negative_feature"]]
    ax.barh(top["feature_name"], top["grey_relation_score"], color=colors)
    ax.set_xlabel("Grey relation score")
    ax.set_title("Appendix 2 Grey Relation Top Features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def _save_key_bar(key_table: pd.DataFrame, chart_path: Path) -> None:
    """Save preliminary key-feature index chart."""
    top = key_table.dropna(subset=["K_pre"]).sort_values("K_pre", ascending=False).head(12).iloc[::-1]
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh(top["feature_name"], top["K_pre"], color="#4f7cac")
    ax.set_xlabel("K_pre = 0.6*|Spearman| + 0.4*grey_norm")
    ax.set_title("Appendix 2 Preliminary Key Features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)


def run_step15_grey_key_index(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 15 grey relation and preliminary key-feature analysis."""
    problem2_config = get_problem2_config(config_path)
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    charts_dir = resolve_project_path(problem2_config["problem2_output_charts_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "grey_key_index.log"
    logger = setup_grey_logger(log_path)
    logger.info("Starting Step 15 grey relation and preliminary key-feature index; rho=%s", RHO)
    logger.info("This step does not run PLS, VIP, QAF, or Bootstrap.")

    scaled_matrix = pd.read_excel(tables_dir / "appendix2_feature_matrix_robust_scaled.xlsx")
    correlation = pd.read_excel(tables_dir / "appendix2_correlation_analysis.xlsx", sheet_name="correlation")
    variance_filter = pd.read_excel(tables_dir / "appendix2_feature_variance_filter.xlsx")
    for frame in [scaled_matrix, correlation, variance_filter]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    grey = calculate_grey_relation(
        scaled_matrix,
        quality_label=scaled_matrix["Q_label"],
        variance_filter=variance_filter,
        rho=RHO,
    )
    key_pre = build_key_feature_index(correlation, grey)

    grey_path = tables_dir / "appendix2_grey_relation.xlsx"
    key_path = tables_dir / "appendix2_key_features_preliminary.xlsx"
    grey_chart_path = charts_dir / "grey_relation_bar.png"
    key_chart_path = charts_dir / "key_features_preliminary_bar.png"

    grey.to_excel(grey_path, index=False)
    key_pre.to_excel(key_path, index=False)
    _save_grey_bar(grey, grey_chart_path)
    _save_key_bar(key_pre, key_chart_path)

    grey_top8 = grey.loc[grey["use_in_model"].ne("False")].sort_values("grey_relation_score", ascending=False).head(8)
    key_top8 = key_pre.dropna(subset=["K_pre"]).sort_values("K_pre", ascending=False).head(8)
    ref_row = grey.loc[grey["feature_name"].eq("reference_norm_rate")]
    appendix_row = grey.loc[grey["feature_name"].eq("appendix_code_presence")]
    stacking_grey = grey.loc[grey["feature_name"].eq("stacking_penalty")]
    stacking_key = key_pre.loc[key_pre["feature_name"].eq("stacking_penalty")]

    logger.info("Grey Top 8: %s", grey_top8[["feature_name", "grey_relation_score", "grey_norm"]].to_dict(orient="records"))
    logger.info("K_pre Top 8: %s", key_top8[["feature_name", "K_pre", "rank_pre"]].to_dict(orient="records"))
    logger.info("reference_norm_rate row: %s", ref_row.to_dict(orient="records"))
    logger.info("appendix_code_presence row: %s", appendix_row.to_dict(orient="records"))
    logger.info("stacking_penalty grey: %s", stacking_grey.to_dict(orient="records"))
    logger.info("stacking_penalty K_pre: %s", stacking_key.to_dict(orient="records"))
    logger.info("Saved outputs: %s, %s, %s, %s", grey_path, key_path, grey_chart_path, key_chart_path)
    logger.info("Finished Step 15")

    return {
        "grey": grey,
        "key_pre": key_pre,
        "grey_top8": grey_top8,
        "key_top8": key_top8,
        "reference_row": ref_row,
        "appendix_row": appendix_row,
        "stacking_grey": stacking_grey,
        "stacking_key": stacking_key,
        "grey_path": grey_path,
        "key_path": key_path,
        "grey_chart_path": grey_chart_path,
        "key_chart_path": key_chart_path,
        "log_path": log_path,
    }
