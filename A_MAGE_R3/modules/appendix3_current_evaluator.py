"""Current quality reevaluation for Problem 3 Step 24.

This module evaluates the three Appendix 3 papers with the sealed Problem 1
AHP-entropy-TOPSIS base model, then adds an interpretable auxiliary score from
Problem 2 final key features. It deliberately does not calculate logic-gap
penalties, AI-writing risk, revision actions, or post-revision predictions.
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from modules import feature_extractor, topsis
from modules.appendix3_pipeline import get_problem3_config, load_config, resolve_project_path
from modules.deep_quality_features import (
    _compute_method_fit,
    _compute_task_coverage,
    _derive_task_keywords,
)
from modules.quality_label_builder import (
    SURFACE_FEATURE_COLUMNS,
    extract_surface_features_for_paper,
    normalize_surface_features,
)


LOGGER_NAME = "A_MAGE_R3.problem3.current_eval"

FINAL_KEY_FEATURES = [
    "total_chars",
    "method_fit",
    "page_count",
    "section_coverage",
    "objective_constraint_completeness",
    "task_coverage",
    "figure_table_explanation_rate",
    "conclusion_echo_rate",
]

GRADE_ORDER = ["优秀", "良好", "中等", "及格", "不及格"]

LOW_FEATURE_MAP = {
    "I01": ("结构完整性不足", "补全核心章节框架"),
    "I02": ("摘要信息不完整", "补写摘要的问题-方法-结果-结论链"),
    "I03": ("图表编号不规范", "统一图表编号和正文引用"),
    "I04": ("附录代码或支撑材料不足", "补充附录代码或数据说明"),
    "I05": ("问题重述覆盖不足", "强化题目任务拆解"),
    "I06": ("假设与问题匹配不足", "重写假设依据和适用边界"),
    "I07": ("逻辑衔接不足", "补充推理连接和过渡句"),
    "I08": ("结果与结论回扣不足", "强化结论对结果的回应"),
    "I09": ("模型数量与任务匹配不足", "按小问补齐模型设计"),
    "I10": ("数学表达密度不足", "补充必要公式和推导"),
    "I11": ("变量定义覆盖不足", "完善符号表和变量说明"),
    "I12": ("目标函数或约束不完整", "补齐目标函数、约束与参数说明"),
    "I13": ("方法合理性证据不足", "解释方法为何服务题目任务"),
    "I14": ("结果完整性不足", "补充各小问结果和数值解释"),
    "I15": ("图表解释不足", "为关键图表增加分析文字"),
    "I16": ("灵敏度分析缺失或薄弱", "增加参数扰动和稳健性检验"),
    "I17": ("误差分析缺失或薄弱", "增加误差来源、误差量化和局限说明"),
    "I18": ("参考文献规范不足", "统一参考文献格式和正文引用"),
    "I19": ("语言可读性不足", "压缩冗长句并增强条理"),
    "I20": ("创新表达不足", "突出模型改进或系统设计亮点"),
    "I21": ("推广应用价值不足", "补充应用场景和落地建议"),
}

KEY_FEATURE_PROBLEM_MAP = {
    "total_chars": ("信息承载量不足", "补充必要建模、结果与验证材料"),
    "page_count": ("结构展开度不足", "完善论文结构和支撑材料"),
    "method_fit": ("方法匹配度不足", "重新说明模型与题目任务的对应关系"),
    "section_coverage": ("核心章节覆盖不足", "补齐缺失核心章节"),
    "objective_constraint_completeness": ("目标函数/约束完整性不足", "补写目标函数、约束条件和参数含义"),
    "task_coverage": ("任务覆盖率不足", "逐小问检查问题-模型-结果闭环"),
    "figure_table_explanation_rate": ("图表解释率不足", "增加图表下方结论性解释"),
    "conclusion_echo_rate": ("结论回扣率不足", "让结论明确回应题目目标和关键结果"),
}


def setup_current_eval_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 24 logger."""
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


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs and paths."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _indicator_prefix(value: str) -> str:
    """Return an indicator prefix such as I01."""
    match = re.match(r"(I\d+)", str(value))
    return match.group(1) if match else str(value)


def _to_bool(value: Any) -> bool:
    """Convert common spreadsheet values to bool."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _set_chinese_font() -> None:
    """Use a Chinese-capable font if available."""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file with UTF-8 tolerance."""
    return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))


def _load_pdf_lookup(pdf_report_path: Path) -> dict[str, dict[str, Any]]:
    """Load Appendix 3 PDF parse metadata."""
    if not Path(pdf_report_path).exists():
        return {}
    report = pd.read_excel(pdf_report_path)
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in report.iterrows():
        paper_id = str(row.get("paper_id", "")).strip()
        lookup[paper_id] = {
            "pages": int(row.get("pages", 0) or 0),
            "ocr_used": _to_bool(row.get("ocr_used", False)),
            "parse_success": _to_bool(row.get("parse_success", False)),
        }
    return lookup


def _load_weight_table(weight_path: Path) -> pd.DataFrame:
    """Load and normalize the sealed AHP-entropy combined weights."""
    try:
        weights = pd.read_excel(weight_path, sheet_name="combined_weights")
    except ValueError:
        weights = pd.read_excel(weight_path)
    required = {"indicator", "combined_weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Weight table missing required columns: {sorted(missing)}")
    weights = weights.copy()
    weights["indicator"] = weights["indicator"].astype(str)
    weights["combined_weight"] = pd.to_numeric(weights["combined_weight"], errors="coerce")
    weights = weights.dropna(subset=["indicator", "combined_weight"])
    total = float(weights["combined_weight"].sum())
    if total <= 0:
        raise ValueError("Combined weights must sum to a positive value.")
    weights["combined_weight"] = weights["combined_weight"] / total
    return weights


def _align_problem1_features(normalized_features: pd.DataFrame, weight_table: pd.DataFrame) -> pd.DataFrame:
    """Align normalized Problem 1 feature columns to weight indicators."""
    matrix = pd.DataFrame(index=normalized_features.index)
    missing: list[str] = []
    for indicator in weight_table["indicator"].astype(str).tolist():
        if indicator in normalized_features.columns:
            matrix[indicator] = pd.to_numeric(normalized_features[indicator], errors="coerce")
            continue
        prefix = _indicator_prefix(indicator)
        candidates = [column for column in normalized_features.columns if _indicator_prefix(str(column)) == prefix]
        if candidates:
            matrix[indicator] = pd.to_numeric(normalized_features[candidates[0]], errors="coerce")
        else:
            missing.append(indicator)
    if missing:
        raise ValueError(f"Problem 1 feature columns missing for Appendix 3 TOPSIS: {missing}")
    return matrix


def _impute_problem1_matrix(
    matrix: pd.DataFrame,
    weight_table: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values for TOPSIS without modifying source features."""
    matrix = matrix.copy()
    imputation_lookup = {}
    if "imputation_value" in weight_table.columns:
        imputation_lookup = dict(zip(weight_table["indicator"].astype(str), weight_table["imputation_value"]))

    rows: list[dict[str, Any]] = []
    for column in matrix.columns:
        series = pd.to_numeric(matrix[column], errors="coerce")
        missing_count = int(series.isna().sum())
        imputation_value = math.nan
        if missing_count:
            fallback = series.median()
            imputation_value = imputation_lookup.get(column, fallback)
            if pd.isna(imputation_value):
                imputation_value = 0.0
            logger.warning("%s has %s missing values; imputed with %.6f", column, missing_count, float(imputation_value))
            series = series.fillna(float(imputation_value))
        series = series.fillna(0.0).clip(lower=0, upper=1)
        matrix[column] = series
        rows.append(
            {
                "indicator": column,
                "missing_imputed_count": missing_count,
                "imputation_value_used": imputation_value,
                "min_after_imputation": float(series.min()),
                "max_after_imputation": float(series.max()),
            }
        )
    return matrix, pd.DataFrame(rows)


def extract_appendix3_problem1_features(
    sections_dir: Path,
    text_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the sealed Problem 1 21-indicator feature table for Appendix 3."""
    rows: list[dict[str, Any]] = []
    section_files = sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key)
    if not section_files:
        raise FileNotFoundError(f"No refined Appendix 3 section JSON files found in {sections_dir}")

    for section_file in section_files:
        text_file = Path(text_dir) / f"{section_file.stem}.txt"
        try:
            rows.append(feature_extractor.extract_features(section_file, text_file=text_file, config=config))
            logger.info("%s Problem1-style features extracted", section_file.name)
        except Exception as exc:  # pragma: no cover - defensive batch isolation
            logger.exception("%s Problem1-style feature extraction failed: %s", section_file.name, exc)
            row = {"paper_id": section_file.stem, "filename": f"{section_file.stem}.txt", "text_chars": np.nan}
            for column in feature_extractor.FEATURE_COLUMNS:
                row[column] = np.nan
            rows.append(row)

    columns = ["paper_id", "filename", "text_chars", *feature_extractor.FEATURE_COLUMNS]
    raw = pd.DataFrame(rows, columns=columns)
    normalized = feature_extractor.normalize_features(raw)
    raw = raw.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    normalized = normalized.sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    return raw, normalized


def calculate_problem1_base_scores(
    normalized_features: pd.DataFrame,
    weight_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate Appendix 3 F1 scores with AHP-entropy-TOPSIS base logic."""
    weights = _load_weight_table(weight_path)
    matrix = _align_problem1_features(normalized_features, weights)
    matrix, imputation_report = _impute_problem1_matrix(matrix, weights, logger)
    weight_series = pd.Series(
        weights["combined_weight"].to_numpy(dtype=float),
        index=weights["indicator"].astype(str).tolist(),
    )
    result = topsis.calculate_topsis_scores(matrix, weight_series)
    scores = pd.DataFrame(
        {
            "paper_id": normalized_features["paper_id"].astype(str).tolist(),
            "filename": normalized_features["filename"].astype(str).tolist(),
            "D_plus": result["D_plus"].to_numpy(dtype=float),
            "D_minus": result["D_minus"].to_numpy(dtype=float),
            "C_i": result["C_i"].to_numpy(dtype=float),
            "F1_score": result["S_base"].to_numpy(dtype=float),
        }
    )
    scores["rank_F1"] = scores["F1_score"].rank(ascending=False, method="first").astype(int)
    scores["score_method"] = "Problem1 sealed AHP-entropy-TOPSIS base; BT not applied to Appendix3"
    scores = scores.sort_values("rank_F1").reset_index(drop=True)
    logger.info(
        "F1 TOPSIS scores calculated: min=%.6f max=%.6f",
        float(scores["F1_score"].min()),
        float(scores["F1_score"].max()),
    )
    return scores, weights, imputation_report


def extract_appendix3_surface_features(
    sections_dir: Path,
    text_dir: Path,
    pdf_report_path: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract Problem 2 surface features for Appendix 3."""
    pdf_lookup = _load_pdf_lookup(pdf_report_path)
    candidate_usage_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for section_file in sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key):
        rows.append(
            extract_surface_features_for_paper(
                section_file,
                Path(text_dir) / f"{section_file.stem}.txt",
                pdf_lookup=pdf_lookup,
                config=config,
                candidate_usage_rows=candidate_usage_rows,
            )
        )
    raw = pd.DataFrame(rows, columns=["paper_id", "filename", *SURFACE_FEATURE_COLUMNS])
    normalized = normalize_surface_features(raw)
    candidate_usage = pd.DataFrame(candidate_usage_rows)
    return raw, normalized, candidate_usage


def _surface_row(surface_table: pd.DataFrame, paper_id: str) -> pd.Series:
    """Return one surface row by paper ID."""
    table = surface_table.copy()
    table["paper_id"] = table["paper_id"].astype(str)
    match = table.loc[table["paper_id"] == str(paper_id)]
    if match.empty:
        raise KeyError(f"Surface feature row missing for {paper_id}")
    return match.iloc[0]


def extract_appendix3_deep_key_features(
    sections_dir: Path,
    surface_raw: pd.DataFrame,
    surface_normalized: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract only the deep final key features needed in Step 24."""
    section_files = sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key)
    payloads = [_load_json(path) for path in section_files]
    task_keywords = _derive_task_keywords(payloads, logger)

    rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for section_file, payload in zip(section_files, payloads):
        paper_id = str(payload.get("paper_id", section_file.stem))
        filename = str(payload.get("filename", f"{paper_id}.txt"))
        norm_row = _surface_row(surface_normalized, paper_id)

        task_coverage, task_evidence = _compute_task_coverage(payload, task_keywords)
        method_fit, method_evidence = _compute_method_fit(payload, norm_row)
        rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "task_coverage": task_coverage,
                "method_fit": method_fit,
            }
        )
        for feature, score, evidence in [
            ("task_coverage", task_coverage, task_evidence),
            ("method_fit", method_fit, method_evidence),
        ]:
            evidence_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": feature,
                    "feature_value": score,
                    "matched_keywords": evidence.get("matched_keywords", ""),
                    "section_source": evidence.get("section_source", ""),
                    "candidate_weight": evidence.get("candidate_weight", 1.0),
                    "evidence": evidence.get("evidence", ""),
                    "note": evidence.get("note", ""),
                }
            )
            logger.info("%s %s=%.6f evidence=%s", paper_id, feature, score, evidence.get("evidence", ""))

    return pd.DataFrame(rows), pd.DataFrame(evidence_rows)


def _load_final_key_weights(key_feature_path: Path) -> pd.DataFrame:
    """Load final key features and K_final weights from Problem 2."""
    table = pd.read_excel(key_feature_path)
    required = {"feature_name", "K_final", "final_key_feature"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Problem 2 key feature table missing columns: {sorted(missing)}")
    table = table.copy()
    table["final_key_feature"] = table["final_key_feature"].apply(_to_bool)
    table["K_final"] = pd.to_numeric(table["K_final"], errors="coerce")
    selected = table.loc[table["final_key_feature"]].copy()
    if selected.empty:
        selected = table.loc[table["feature_name"].isin(FINAL_KEY_FEATURES)].copy()
    selected["feature_name"] = pd.Categorical(selected["feature_name"], categories=FINAL_KEY_FEATURES, ordered=True)
    selected = selected.sort_values("feature_name").reset_index(drop=True)
    selected["feature_name"] = selected["feature_name"].astype(str)
    missing_features = [feature for feature in FINAL_KEY_FEATURES if feature not in set(selected["feature_name"])]
    if missing_features:
        raise ValueError(f"Final key features missing from Problem 2 table: {missing_features}")
    weight_sum = float(selected["K_final"].sum())
    if weight_sum <= 0:
        raise ValueError("K_final weights must sum to a positive value.")
    selected["K_weight_norm"] = selected["K_final"] / weight_sum
    return selected


def build_problem2_key_feature_table(
    surface_raw: pd.DataFrame,
    surface_normalized: pd.DataFrame,
    deep_key: pd.DataFrame,
    key_weights: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build wide, long, and scoring tables for Problem 2 final key features."""
    raw_wide = surface_raw[["paper_id", "filename"]].copy()
    norm_wide = surface_normalized[["paper_id", "filename"]].copy()
    deep_key = deep_key.copy()
    deep_key["paper_id"] = deep_key["paper_id"].astype(str)

    for feature in FINAL_KEY_FEATURES:
        if feature in surface_raw.columns:
            raw_wide[feature] = pd.to_numeric(surface_raw[feature], errors="coerce")
            norm_wide[feature] = pd.to_numeric(surface_normalized[feature], errors="coerce")
        elif feature in deep_key.columns:
            raw_wide = raw_wide.merge(deep_key[["paper_id", feature]], on="paper_id", how="left")
            norm_wide = norm_wide.merge(deep_key[["paper_id", feature]], on="paper_id", how="left")
        else:
            raw_wide[feature] = np.nan
            norm_wide[feature] = np.nan
            logger.warning("Final key feature missing during Appendix 3 extraction: %s", feature)

    weights = key_weights.set_index("feature_name")["K_weight_norm"].astype(float)
    scoring_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for _, row in norm_wide.iterrows():
        paper_id = str(row["paper_id"])
        filename = str(row["filename"])
        weighted_sum = 0.0
        missing_features: list[str] = []
        feature_values: dict[str, float] = {}
        for feature in FINAL_KEY_FEATURES:
            value = pd.to_numeric(row.get(feature), errors="coerce")
            if pd.isna(value):
                missing_features.append(feature)
                value = 0.0
            value = float(np.clip(float(value), 0.0, 1.0))
            feature_values[feature] = value
            weighted_sum += float(weights.loc[feature]) * value
            long_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": feature,
                    "feature_value_raw": pd.to_numeric(raw_wide.loc[raw_wide["paper_id"].astype(str) == paper_id, feature].iloc[0], errors="coerce"),
                    "feature_value_normalized": value,
                    "K_final": float(key_weights.loc[key_weights["feature_name"] == feature, "K_final"].iloc[0]),
                    "K_weight_norm": float(weights.loc[feature]),
                    "is_length_feature": feature in {"total_chars", "page_count"},
                    "interpretation_note": (
                        "information carrying / completeness related, not longer-is-better"
                        if feature in {"total_chars", "page_count"}
                        else KEY_FEATURE_PROBLEM_MAP.get(feature, ("", ""))[0]
                    ),
                }
            )

        sorted_values = sorted(feature_values.items(), key=lambda item: item[1], reverse=True)
        strengths = [f"{name}:{value:.3f}" for name, value in sorted_values if value >= 0.65][:4]
        weaknesses = [f"{name}:{value:.3f}" for name, value in sorted(feature_values.items(), key=lambda item: item[1]) if value <= 0.45][:5]
        if not weaknesses:
            weaknesses = [f"{name}:{value:.3f}" for name, value in sorted(feature_values.items(), key=lambda item: item[1])[:3]]
        scoring_rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "F2_key_score": 100.0 * weighted_sum,
                "key_feature_strengths": "; ".join(strengths) if strengths else "no feature above 0.65",
                "key_feature_weaknesses": "; ".join(weaknesses),
                "missing_key_features": ",".join(missing_features),
            }
        )

    return raw_wide, norm_wide, pd.DataFrame(scoring_rows), pd.DataFrame(long_rows)


def _load_grade_centers(final_ranking_path: Path) -> pd.DataFrame:
    """Load Problem 1 grade centers from the sealed final ranking workbook."""
    final_table = pd.read_excel(final_ranking_path, sheet_name="final_ranking")
    if "kmeans_center" in final_table.columns:
        centers = (
            final_table[["grade_final", "kmeans_center"]]
            .dropna()
            .groupby("grade_final", as_index=False)["kmeans_center"]
            .mean()
            .rename(columns={"kmeans_center": "center"})
        )
    else:
        centers = (
            final_table.groupby("grade_final", as_index=False)["S_rank_v2"]
            .mean()
            .rename(columns={"S_rank_v2": "center"})
        )
    centers["order"] = centers["grade_final"].map({grade: idx for idx, grade in enumerate(GRADE_ORDER)})
    centers = centers.sort_values(["order", "center"], ascending=[True, False]).reset_index(drop=True)
    return centers[["grade_final", "center"]]


def _map_score_to_grade(score: float, grade_centers: pd.DataFrame) -> str:
    """Map score to the nearest sealed Problem 1 KMeans grade center."""
    if pd.isna(score) or grade_centers.empty:
        return "unknown"
    centers = grade_centers.copy()
    centers["distance"] = (centers["center"].astype(float) - float(score)).abs()
    return str(centers.sort_values(["distance"]).iloc[0]["grade_final"])


def _feature_reference_means(problem1_normalized_path: Path, weight_table: pd.DataFrame) -> dict[str, float]:
    """Read Problem 1 reference means by indicator prefix."""
    if not Path(problem1_normalized_path).exists():
        return {}
    table = pd.read_excel(problem1_normalized_path)
    refs: dict[str, float] = {}
    for indicator in weight_table["indicator"].astype(str):
        prefix = _indicator_prefix(indicator)
        candidates = [column for column in table.columns if _indicator_prefix(str(column)) == prefix]
        if candidates:
            refs[prefix] = float(pd.to_numeric(table[candidates[0]], errors="coerce").mean())
    return refs


def build_low_feature_report(
    normalized_problem1: pd.DataFrame,
    norm_key_wide: pd.DataFrame,
    key_weights: pd.DataFrame,
    weight_table: pd.DataFrame,
    problem1_reference_path: Path,
) -> pd.DataFrame:
    """Create a traceable low-feature table for later diagnosis steps."""
    reference_means = _feature_reference_means(problem1_reference_path, weight_table)
    rows: list[dict[str, Any]] = []

    key_lookup = key_weights.set_index("feature_name")["K_final"].to_dict()
    norm_key_wide = norm_key_wide.copy()
    norm_key_wide["paper_id"] = norm_key_wide["paper_id"].astype(str)

    for _, row in normalized_problem1.iterrows():
        paper_id = str(row["paper_id"])
        filename = str(row["filename"])
        for indicator in weight_table["indicator"].astype(str):
            prefix = _indicator_prefix(indicator)
            candidates = [column for column in normalized_problem1.columns if _indicator_prefix(str(column)) == prefix]
            if not candidates:
                continue
            column = candidates[0]
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                weakness_level = "missing"
            else:
                ref = reference_means.get(prefix, 0.5)
                if float(value) < 0.25:
                    weakness_level = "severe"
                elif float(value) < min(0.45, ref):
                    weakness_level = "moderate"
                else:
                    continue
            related_problem, action = LOW_FEATURE_MAP.get(prefix, ("质量短板", "后续人工复核"))
            rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": indicator,
                    "feature_value": value,
                    "reference_level": f"Problem1 reference mean={reference_means.get(prefix, math.nan):.3f}",
                    "weakness_level": weakness_level,
                    "related_problem": related_problem,
                    "suggested_later_action_type": action,
                }
            )

        key_row = norm_key_wide.loc[norm_key_wide["paper_id"] == paper_id]
        if key_row.empty:
            continue
        key_row = key_row.iloc[0]
        for feature in FINAL_KEY_FEATURES:
            value = pd.to_numeric(key_row.get(feature), errors="coerce")
            if pd.isna(value):
                weakness_level = "missing"
            elif float(value) < 0.25:
                weakness_level = "severe"
            elif float(value) < 0.45:
                weakness_level = "moderate"
            else:
                continue
            related_problem, action = KEY_FEATURE_PROBLEM_MAP.get(feature, ("关键质量特征薄弱", "后续复核"))
            rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": feature,
                    "feature_value": value,
                    "reference_level": f"K_final key feature midline=0.500; K_final={key_lookup.get(feature, math.nan):.3f}",
                    "weakness_level": weakness_level,
                    "related_problem": related_problem,
                    "suggested_later_action_type": action,
                }
            )

    report = pd.DataFrame(
        rows,
        columns=[
            "paper_id",
            "filename",
            "feature_name",
            "feature_value",
            "reference_level",
            "weakness_level",
            "related_problem",
            "suggested_later_action_type",
        ],
    )
    if not report.empty:
        level_order = {"missing": 0, "severe": 1, "moderate": 2}
        report["_level_order"] = report["weakness_level"].map(level_order).fillna(9)
        report["_paper_order"] = report["paper_id"].astype(str).map(
            lambda value: "|".join(f"{part:08d}" if isinstance(part, int) else str(part) for part in _natural_sort_key(value))
        )
        report = report.sort_values(["_paper_order", "_level_order", "feature_value"]).drop(
            columns=["_level_order", "_paper_order"]
        )
    return report.reset_index(drop=True)


def _main_low_features(low_report: pd.DataFrame, paper_id: str, limit: int = 5) -> str:
    """Summarize low features for one paper."""
    subset = low_report.loc[low_report["paper_id"].astype(str) == str(paper_id)].head(limit)
    if subset.empty:
        return "no severe low feature detected"
    return "; ".join(
        f"{row.feature_name}({row.weakness_level})" for row in subset.itertuples(index=False)
    )


def build_current_evaluation(
    f1_scores: pd.DataFrame,
    f2_scores: pd.DataFrame,
    low_report: pd.DataFrame,
    final_ranking_path: Path,
    alpha: float,
) -> pd.DataFrame:
    """Fuse F1 and F2 into the current baseline score."""
    table = f1_scores.merge(
        f2_scores[["paper_id", "F2_key_score", "key_feature_strengths", "key_feature_weaknesses", "missing_key_features"]],
        on="paper_id",
        how="left",
    )
    table["F2_key_score"] = pd.to_numeric(table["F2_key_score"], errors="coerce").fillna(0.0)
    table["Q_cur_baseline"] = alpha * table["F1_score"] + (1.0 - alpha) * table["F2_key_score"]
    table["rank_current"] = table["Q_cur_baseline"].rank(ascending=False, method="first").astype(int)

    grade_centers = _load_grade_centers(final_ranking_path)
    table["current_grade"] = table["Q_cur_baseline"].apply(lambda value: _map_score_to_grade(value, grade_centers))
    table["grade_consistency_with_prompt"] = np.where(
        table["current_grade"].eq("中等"),
        "consistent_with_medium_quality_prompt",
        "deviates_from_medium_quality_prompt",
    )
    table["main_low_features"] = table["paper_id"].astype(str).apply(lambda paper_id: _main_low_features(low_report, paper_id))
    table["score_confidence"] = np.where(
        table["missing_key_features"].fillna("").astype(str).str.len().gt(0),
        "usable_with_missing_key_feature_imputation",
        "usable",
    )
    table["note"] = table.apply(
        lambda row: (
            f"Current baseline only: alpha={alpha:.2f}; logic gap and AI-risk penalties are not applied. "
            f"Prompt describes Appendix3 as medium-quality; model grade={row['current_grade']}."
        ),
        axis=1,
    )
    table = table.sort_values("rank_current").reset_index(drop=True)
    return table[
        [
            "paper_id",
            "filename",
            "F1_score",
            "F2_key_score",
            "Q_cur_baseline",
            "rank_current",
            "current_grade",
            "grade_consistency_with_prompt",
            "main_low_features",
            "key_feature_weaknesses",
            "score_confidence",
            "note",
        ]
    ]


def save_feature_workbooks(
    raw_problem1: pd.DataFrame,
    normalized_problem1: pd.DataFrame,
    surface_raw: pd.DataFrame,
    surface_normalized: pd.DataFrame,
    deep_key: pd.DataFrame,
    key_raw_wide: pd.DataFrame,
    key_norm_wide: pd.DataFrame,
    key_long: pd.DataFrame,
    key_weights: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    """Save Step 24 feature workbooks."""
    paths["raw_features"].parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(paths["raw_features"]) as writer:
        raw_problem1.to_excel(writer, sheet_name="problem1_21_raw", index=False)
        surface_raw.to_excel(writer, sheet_name="problem2_surface_raw", index=False)
        deep_key.to_excel(writer, sheet_name="deep_key_raw", index=False)
        key_raw_wide.to_excel(writer, sheet_name="combined_key_raw", index=False)
    with pd.ExcelWriter(paths["normalized_features"]) as writer:
        normalized_problem1.to_excel(writer, sheet_name="problem1_21_normalized", index=False)
        surface_normalized.to_excel(writer, sheet_name="problem2_surface_normalized", index=False)
        deep_key.to_excel(writer, sheet_name="deep_key_normalized", index=False)
        key_norm_wide.to_excel(writer, sheet_name="combined_key_normalized", index=False)
    with pd.ExcelWriter(paths["problem2_key_features"]) as writer:
        key_norm_wide.to_excel(writer, sheet_name="key_features_wide", index=False)
        key_long.to_excel(writer, sheet_name="key_features_long", index=False)
        key_weights.to_excel(writer, sheet_name="key_weights", index=False)


def plot_current_scores(evaluation: pd.DataFrame, chart_path: Path) -> Path:
    """Plot F1, F2, and Q_cur_baseline for Appendix 3."""
    _set_chinese_font()
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    table = evaluation.sort_values("rank_current")
    x = np.arange(len(table))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, table["F1_score"], width, label="F1 Problem1", color="#4E79A7")
    ax.bar(x, table["F2_key_score"], width, label="F2 Key Features", color="#F28E2B")
    ax.bar(x + width, table["Q_cur_baseline"], width, label="Q_cur", color="#59A14F")
    ax.set_xticks(x)
    ax.set_xticklabels(table["paper_id"])
    ax.set_ylim(0, max(100, float(table[["F1_score", "F2_key_score", "Q_cur_baseline"]].max().max()) + 8))
    ax.set_ylabel("Score")
    ax.set_title("Appendix 3 Current Quality Baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for idx, score in enumerate(table["Q_cur_baseline"]):
        ax.text(idx + width, score + 1.0, f"{score:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def plot_key_feature_radar(norm_key_wide: pd.DataFrame, chart_path: Path) -> Path:
    """Plot the eight final key feature values for the three Appendix 3 papers."""
    _set_chinese_font()
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    labels = FINAL_KEY_FEATURES
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for _, row in norm_key_wide.iterrows():
        values = [float(np.clip(pd.to_numeric(row.get(label), errors="coerce"), 0, 1)) for label in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=str(row["paper_id"]))
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.0)
    ax.set_title("Appendix 3 Final Key Feature Radar", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08))
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def run_appendix3_current_evaluation(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 24 and save all current evaluation outputs."""
    try:
        config = load_config(config_path)
    except RuntimeError:
        config = {}
    problem3_config = get_problem3_config(config_path)

    text_dir = resolve_project_path(problem3_config["appendix3_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem3_config["appendix3_intermediate_dir"])
    sections_dir = intermediate_dir / "sections_refined"
    if not sections_dir.is_dir() or not list(sections_dir.glob("*.json")):
        sections_dir = intermediate_dir / "sections"

    tables_dir = resolve_project_path(problem3_config["output_tables_dir"])
    charts_dir = resolve_project_path(problem3_config["output_charts_dir"])
    logs_dir = resolve_project_path(problem3_config["output_logs_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "appendix3_current_eval.log"
    logger = setup_current_eval_logger(log_path)
    logger.info("Starting Step 24 Appendix 3 current quality reevaluation")
    logger.info("Using sections_dir=%s", sections_dir)

    paths = {
        "raw_features": tables_dir / "appendix3_features_raw.xlsx",
        "normalized_features": tables_dir / "appendix3_features_normalized.xlsx",
        "problem1_base_scores": tables_dir / "appendix3_problem1_base_scores.xlsx",
        "problem2_key_features": tables_dir / "appendix3_problem2_key_features.xlsx",
        "current_evaluation": tables_dir / "appendix3_current_evaluation.xlsx",
        "low_feature_report": tables_dir / "appendix3_current_low_feature_report.xlsx",
        "current_scores_bar": charts_dir / "appendix3_current_scores_bar.png",
        "key_feature_radar": charts_dir / "appendix3_key_feature_radar.png",
    }

    weights_path = resolve_project_path("output/tables/appendix1_weights_ahp_entropy.xlsx")
    final_ranking_path = resolve_project_path("output/tables/final_problem1_ranking.xlsx")
    problem1_norm_path = resolve_project_path("output/tables/appendix1_features_normalized.xlsx")
    key_feature_path = resolve_project_path("output/problem2_tables/key_feature_index_final.xlsx")
    pdf_report_path = tables_dir / "appendix3_pdf_parse_report.xlsx"

    raw_problem1, normalized_problem1 = extract_appendix3_problem1_features(sections_dir, text_dir, config, logger)
    f1_scores, weight_table, imputation_report = calculate_problem1_base_scores(normalized_problem1, weights_path, logger)

    surface_raw, surface_normalized, candidate_usage = extract_appendix3_surface_features(
        sections_dir=sections_dir,
        text_dir=text_dir,
        pdf_report_path=pdf_report_path,
        config=config,
    )
    if not candidate_usage.empty:
        logger.warning("Appendix3 candidates were used in surface feature extraction: %s", candidate_usage.to_dict(orient="records"))

    deep_key, deep_evidence = extract_appendix3_deep_key_features(sections_dir, surface_raw, surface_normalized, logger)
    key_weights = _load_final_key_weights(key_feature_path)
    key_raw_wide, key_norm_wide, f2_scores, key_long = build_problem2_key_feature_table(
        surface_raw=surface_raw,
        surface_normalized=surface_normalized,
        deep_key=deep_key,
        key_weights=key_weights,
        logger=logger,
    )

    low_report = build_low_feature_report(
        normalized_problem1=normalized_problem1,
        norm_key_wide=key_norm_wide,
        key_weights=key_weights,
        weight_table=weight_table,
        problem1_reference_path=problem1_norm_path,
    )
    alpha = float(problem3_config.get("alpha_problem1_weight", 0.80))
    evaluation = build_current_evaluation(
        f1_scores=f1_scores,
        f2_scores=f2_scores,
        low_report=low_report,
        final_ranking_path=final_ranking_path,
        alpha=alpha,
    )

    save_feature_workbooks(
        raw_problem1=raw_problem1,
        normalized_problem1=normalized_problem1,
        surface_raw=surface_raw,
        surface_normalized=surface_normalized,
        deep_key=deep_key,
        key_raw_wide=key_raw_wide,
        key_norm_wide=key_norm_wide,
        key_long=key_long,
        key_weights=key_weights,
        paths=paths,
    )
    with pd.ExcelWriter(paths["problem1_base_scores"]) as writer:
        f1_scores.to_excel(writer, sheet_name="base_scores", index=False)
        imputation_report.to_excel(writer, sheet_name="imputation_report", index=False)
        weight_table.to_excel(writer, sheet_name="weights_used", index=False)
    evaluation.to_excel(paths["current_evaluation"], index=False)
    low_report.to_excel(paths["low_feature_report"], index=False)

    plot_current_scores(evaluation, paths["current_scores_bar"])
    plot_key_feature_radar(key_norm_wide, paths["key_feature_radar"])

    missing_problem1 = {
        column: int(normalized_problem1[column].isna().sum())
        for column in feature_extractor.FEATURE_COLUMNS
        if column in normalized_problem1.columns and int(normalized_problem1[column].isna().sum()) > 0
    }
    missing_key = {
        column: int(key_norm_wide[column].isna().sum())
        for column in FINAL_KEY_FEATURES
        if column in key_norm_wide.columns and int(key_norm_wide[column].isna().sum()) > 0
    }
    if missing_problem1 or missing_key:
        logger.warning("Missing feature values detected: problem1=%s key=%s", missing_problem1, missing_key)
    else:
        logger.info("No missing Step24 feature values detected.")

    logger.info("F2 key score range: %.6f - %.6f", float(f2_scores["F2_key_score"].min()), float(f2_scores["F2_key_score"].max()))
    logger.info("Q_cur range: %.6f - %.6f", float(evaluation["Q_cur_baseline"].min()), float(evaluation["Q_cur_baseline"].max()))
    logger.info("Current evaluation saved: %s", paths["current_evaluation"])
    logger.info("Low feature report rows: %s", len(low_report))
    logger.info("Finished Step 24")

    return {
        "raw_problem1": raw_problem1,
        "normalized_problem1": normalized_problem1,
        "surface_raw": surface_raw,
        "surface_normalized": surface_normalized,
        "deep_key": deep_key,
        "deep_evidence": deep_evidence,
        "key_weights": key_weights,
        "key_norm_wide": key_norm_wide,
        "f1_scores": f1_scores,
        "f2_scores": f2_scores,
        "evaluation": evaluation,
        "low_report": low_report,
        "missing_problem1_features": missing_problem1,
        "missing_key_features": missing_key,
        "paths": paths | {"log_path": log_path},
    }


def evaluate_current_quality(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for Step 24."""
    return run_appendix3_current_evaluation(*args, **kwargs)


def build_current_evaluation_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for Step 24 report generation."""
    return run_appendix3_current_evaluation(*args, **kwargs)
