"""Feature extraction for Step 4.

This module extracts 21 secondary indicators from section JSON files and the
page-marked text files produced by earlier steps. It uses explicit rules,
regular expressions, keyword coverage, and TF-IDF cosine similarity. It does
not calculate AHP weights, entropy weights, TOPSIS scores, or final grades.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - fallback only when sklearn is absent.
    TfidfVectorizer = None
    cosine_similarity = None


LOGGER_NAME = "A_MAGE_R3.feature_extractor"

CORE_SECTIONS = [
    "abstract",
    "problem_statement",
    "assumptions",
    "symbols",
    "model_building",
    "model_solving",
    "results",
    "sensitivity_analysis",
    "error_analysis",
    "model_evaluation",
    "references",
    "appendix",
]

FEATURE_COLUMNS = [
    "I01_核心章节完整率",
    "I02_摘要完整性",
    "I03_图表编号规范率",
    "I04_附录代码存在性",
    "I05_问题重述覆盖率",
    "I06_模型假设与问题匹配度",
    "I07_逻辑连接词密度",
    "I08_结果结论一致性",
    "I09_模型数量与任务匹配度",
    "I10_公式密度",
    "I11_变量定义覆盖率",
    "I12_目标函数约束完整性",
    "I13_方法合理性语义评分",
    "I14_结果完整率",
    "I15_图表解释率",
    "I16_灵敏度分析存在性",
    "I17_误差分析存在性",
    "I18_参考文献规范率",
    "I19_语言可读性",
    "I20_创新性表达",
    "I21_推广应用价值",
]

FEATURE_TYPES = {
    "I01_核心章节完整率": "ratio",
    "I02_摘要完整性": "ratio",
    "I03_图表编号规范率": "ratio",
    "I04_附录代码存在性": "binary",
    "I05_问题重述覆盖率": "ratio",
    "I06_模型假设与问题匹配度": "similarity",
    "I07_逻辑连接词密度": "density",
    "I08_结果结论一致性": "similarity",
    "I09_模型数量与任务匹配度": "ratio",
    "I10_公式密度": "density",
    "I11_变量定义覆盖率": "ratio",
    "I12_目标函数约束完整性": "ratio",
    "I13_方法合理性语义评分": "ratio",
    "I14_结果完整率": "ratio",
    "I15_图表解释率": "ratio",
    "I16_灵敏度分析存在性": "binary",
    "I17_误差分析存在性": "binary",
    "I18_参考文献规范率": "ratio",
    "I19_语言可读性": "ratio",
    "I20_创新性表达": "ratio",
    "I21_推广应用价值": "ratio",
}

DEFAULT_CONFIG = {
    "logic_connectives": [
        "首先",
        "其次",
        "然后",
        "最后",
        "因此",
        "所以",
        "由此",
        "综上",
        "可见",
        "进一步",
        "同时",
        "为了",
        "基于",
        "由于",
        "针对",
        "进而",
        "从而",
        "若",
        "则",
        "故",
        "可得",
        "证明",
        "验证",
    ],
    "feature_extraction": {
        "sections_subdir": "sections",
        "raw_feature_filename": "appendix1_features_raw.xlsx",
        "normalized_feature_filename": "appendix1_features_normalized.xlsx",
        "log_filename": "feature_extraction.log",
        "density_per_chars": 1000,
        "expected_problem_default": 3,
        "formula_line_patterns": [
            r"[=<>≤≥≈∑∏√±]",
            r"\bmax\b|\bmin\b|最大化|最小化",
            r"目标函数|约束条件|矩阵|向量|权重|变量",
            r"[A-Za-zα-ωΑ-Ω]\s*[_=]",
        ],
        "model_keywords": [
            "AHP",
            "层次分析",
            "熵权",
            "TOPSIS",
            "灰色关联",
            "回归",
            "随机森林",
            "XGBoost",
            "主成分",
            "聚类",
            "遗传算法",
            "粒子群",
            "模拟退火",
            "多目标",
            "线性规划",
            "整数规划",
            "强化学习",
            "结构方程",
            "神经网络",
            "LSTM",
            "Transformer",
        ],
        "validation_keywords": [
            "一致性检验",
            "显著性",
            "敏感性",
            "灵敏度",
            "稳健性",
            "误差",
            "残差",
            "拟合",
            "RMSE",
            "MAE",
            "R²",
            "R2",
            "CR",
        ],
        "innovation_keywords": [
            "创新",
            "改进",
            "优化",
            "融合",
            "动态",
            "智能",
            "多目标",
            "深度",
            "自适应",
            "协同",
            "闭环",
        ],
        "application_keywords": [
            "推广",
            "应用",
            "建议",
            "策略",
            "政策",
            "实施",
            "价值",
            "实际",
            "政府",
            "路径",
            "落地",
            "管理",
        ],
        "chart_explanation_keywords": [
            "可知",
            "显示",
            "说明",
            "表明",
            "分析",
            "结果",
            "趋势",
            "对比",
            "由图",
            "由表",
        ],
    },
}


def setup_feature_logger(log_path: Path) -> logging.Logger:
    """Configure and return the Step 4 logger."""
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
    """Return the feature extraction logger."""
    return logging.getLogger(LOGGER_NAME)


def _feature_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge user config with feature defaults."""
    merged = dict(DEFAULT_CONFIG["feature_extraction"])
    if config:
        merged.update(config.get("feature_extraction", {}))
    return merged


def _logic_connectives(config: dict[str, Any] | None = None) -> list[str]:
    """Return configured logic connective words."""
    if config and config.get("logic_connectives"):
        return list(config["logic_connectives"])
    return list(DEFAULT_CONFIG["logic_connectives"])


def _clean_text(text: str) -> str:
    """Normalize whitespace for statistics."""
    text = re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", "", text)


def _char_count(text: str) -> int:
    """Count non-whitespace characters."""
    return len(_clean_text(text))


def _clip01(value: float | int | None) -> float:
    """Clip a numeric value to [0, 1]."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    return float(max(0.0, min(1.0, value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or NaN for an undefined ratio."""
    if denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    """Calculate keyword coverage in [0, 1]."""
    if not keywords:
        return np.nan
    if not _clean_text(text):
        return 0.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text.lower())
    return hits / len(keywords)


def _keyword_density(text: str, keywords: list[str], per_chars: int = 1000) -> float:
    """Count keyword occurrences per N non-whitespace characters."""
    chars = _char_count(text)
    if chars == 0:
        return np.nan
    count = sum(text.count(keyword) for keyword in keywords)
    return count / chars * per_chars


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Calculate TF-IDF cosine similarity with a deterministic fallback."""
    a = _clean_text(text_a)
    b = _clean_text(text_b)
    if not a or not b:
        return np.nan

    if TfidfVectorizer is None or cosine_similarity is None:
        grams_a = {a[i : i + 2] for i in range(max(0, len(a) - 1))}
        grams_b = {b[i : i + 2] for i in range(max(0, len(b) - 1))}
        if not grams_a or not grams_b:
            return np.nan
        return len(grams_a & grams_b) / len(grams_a | grams_b)

    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform([a, b])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])
    except Exception:
        return np.nan


def _sentences(text: str) -> list[str]:
    """Split Chinese text into rough sentences."""
    return [item.strip() for item in re.split(r"[。！？!?；;]\s*", text or "") if item.strip()]


def _all_sections_text(sections: dict[str, str], keys: list[str]) -> str:
    """Concatenate selected sections."""
    return "\n".join(sections.get(key, "") or "" for key in keys)


def _expected_problem_count(full_text: str, config: dict[str, Any] | None = None) -> int:
    """Infer expected problem count from problem labels, defaulting to config."""
    feature_cfg = _feature_config(config)
    labels = set()
    patterns = [
        (r"问题\s*一|问题\s*1", 1),
        (r"问题\s*二|问题\s*2", 2),
        (r"问题\s*三|问题\s*3", 3),
        (r"问题\s*四|问题\s*4", 4),
        (r"问题\s*五|问题\s*5", 5),
    ]
    for pattern, label in patterns:
        if re.search(pattern, full_text):
            labels.add(label)
    if labels:
        return max(labels)
    return int(feature_cfg.get("expected_problem_default", 3))


def _covered_problem_count(text: str) -> int:
    """Count distinct problem labels covered in text."""
    labels = set()
    patterns = [
        (r"问题\s*一|问题\s*1", 1),
        (r"问题\s*二|问题\s*2", 2),
        (r"问题\s*三|问题\s*3", 3),
        (r"问题\s*四|问题\s*4", 4),
        (r"问题\s*五|问题\s*5", 5),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text or ""):
            labels.add(label)
    return len(labels)


def _formula_line_count(text: str, config: dict[str, Any] | None = None) -> int:
    """Count lines that look like formulas or mathematical definitions."""
    patterns = _feature_config(config).get("formula_line_patterns", [])
    count = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns):
            count += 1
    return count


def _figure_table_counts(text: str) -> tuple[int, int, int, int]:
    """Return numbered/un-numbered figure and table counts."""
    numbered_figs = len(re.findall(r"图\s*\d+(?:[-.－—]\d+)?", text or "", flags=re.IGNORECASE))
    numbered_tables = len(re.findall(r"表\s*\d+(?:[-.－—]\d+)?", text or "", flags=re.IGNORECASE))
    all_figs = len(re.findall(r"图", text or ""))
    all_tables = len(re.findall(r"表", text or ""))
    return numbered_figs, numbered_tables, all_figs, all_tables


def _reference_entries(reference_text: str) -> list[str]:
    """Extract rough reference entries from a reference section."""
    entries: list[str] = []
    current: list[str] = []
    for line in (reference_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(\[\s*\d+\s*\]|\d+[\.、])", stripped):
            if current:
                entries.append(" ".join(current))
            current = [stripped]
        elif current:
            current.append(stripped)
    if current:
        entries.append(" ".join(current))
    if entries:
        return entries
    return [line.strip() for line in (reference_text or "").splitlines() if line.strip()]


def _method_hits(text: str, config: dict[str, Any] | None = None) -> list[str]:
    """Return unique method keywords found in text."""
    keywords = _feature_config(config).get("model_keywords", [])
    lower_text = (text or "").lower()
    return [keyword for keyword in keywords if keyword.lower() in lower_text]


def _nan_if_empty_paper(full_text: str, value: float | int) -> float:
    """Return NaN when the paper has no extractable body text."""
    if _char_count(full_text) == 0:
        return np.nan
    return float(value)


def extract_structure_features(
    sections: dict[str, str],
    full_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract A1 structure standardization features."""
    present_count = sum(1 for section in CORE_SECTIONS if sections.get(section, "").strip())
    core_section_completeness = _nan_if_empty_paper(
        full_text,
        present_count / len(CORE_SECTIONS),
    )

    abstract = sections.get("abstract", "")
    abstract_chars = _char_count(abstract)
    if _char_count(full_text) == 0:
        abstract_completeness = np.nan
    elif abstract_chars == 0:
        abstract_completeness = 0.0
    else:
        length_score = _clip01(abstract_chars / 1200)
        keyword_score = _keyword_coverage(abstract, ["问题一", "问题二", "模型", "结果", "建议"])
        abstract_completeness = 0.6 * length_score + 0.4 * keyword_score

    numbered_figs, numbered_tables, all_figs, all_tables = _figure_table_counts(full_text)
    total_mentions = all_figs + all_tables
    if _char_count(full_text) == 0:
        chart_number_rate = np.nan
    elif total_mentions == 0:
        chart_number_rate = 0.0
    else:
        chart_number_rate = _safe_ratio(numbered_figs + numbered_tables, total_mentions)

    appendix_text = sections.get("appendix", "")
    code_patterns = [r"\bimport\s+\w+", r"\bdef\s+\w+", r"\bclass\s+\w+", r"print\s*\(", r"for\s+\w+\s+in"]
    appendix_code_exists = 1.0 if any(re.search(pattern, appendix_text) for pattern in code_patterns) else 0.0
    appendix_code_exists = _nan_if_empty_paper(full_text, appendix_code_exists)

    return {
        "I01_核心章节完整率": core_section_completeness,
        "I02_摘要完整性": abstract_completeness,
        "I03_图表编号规范率": chart_number_rate,
        "I04_附录代码存在性": appendix_code_exists,
    }


def extract_logic_features(
    sections: dict[str, str],
    full_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract A2 problem understanding and logic features."""
    expected_count = _expected_problem_count(full_text, config)
    problem_statement = sections.get("problem_statement", "")
    problem_analysis = sections.get("problem_analysis", "")
    assumptions = sections.get("assumptions", "")
    results = sections.get("results", "")
    conclusion_like = _all_sections_text(sections, ["model_evaluation", "appendix"])
    if not conclusion_like:
        conclusion_like = full_text[-2000:]

    problem_coverage = _nan_if_empty_paper(
        full_text,
        _safe_ratio(_covered_problem_count(problem_statement), expected_count)
        if problem_statement
        else 0.0,
    )

    assumption_match = _tfidf_similarity(assumptions, problem_statement + "\n" + problem_analysis)
    if _char_count(full_text) == 0:
        assumption_match = np.nan

    logic_density = _keyword_density(
        full_text,
        _logic_connectives(config),
        per_chars=int(_feature_config(config).get("density_per_chars", 1000)),
    )

    result_conclusion_consistency = _tfidf_similarity(results, conclusion_like)
    if _char_count(full_text) == 0:
        result_conclusion_consistency = np.nan

    return {
        "I05_问题重述覆盖率": problem_coverage,
        "I06_模型假设与问题匹配度": assumption_match,
        "I07_逻辑连接词密度": logic_density,
        "I08_结果结论一致性": result_conclusion_consistency,
    }


def extract_modeling_features(
    sections: dict[str, str],
    full_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract A3 method and mathematical modeling features."""
    feature_cfg = _feature_config(config)
    expected_count = _expected_problem_count(full_text, config)
    modeling_text = _all_sections_text(sections, ["model_building", "model_solving"])
    methods = _method_hits(modeling_text or full_text, config)

    model_task_match = _nan_if_empty_paper(
        full_text,
        _clip01(len(methods) / max(1, expected_count)),
    )

    formula_density = np.nan
    if _char_count(full_text) > 0:
        formula_density = _formula_line_count(full_text, config) / _char_count(full_text) * int(
            feature_cfg.get("density_per_chars", 1000)
        )

    formula_symbols = set(
        re.findall(r"[A-Za-zα-ωΑ-Ω][A-Za-z0-9_]*|[𝑎-𝑧𝐴-𝑍]", modeling_text + "\n" + full_text)
    )
    defined_symbols = set(
        re.findall(r"[A-Za-zα-ωΑ-Ω][A-Za-z0-9_]*|[𝑎-𝑧𝐴-𝑍]", sections.get("symbols", ""))
    )
    formula_symbols = {item for item in formula_symbols if len(item) <= 8}
    if _char_count(full_text) == 0:
        variable_definition_coverage = np.nan
    elif not formula_symbols:
        variable_definition_coverage = 0.0
    else:
        variable_definition_coverage = _safe_ratio(len(defined_symbols & formula_symbols), len(formula_symbols))

    objective_hit = any(keyword in modeling_text for keyword in ["目标函数", "最大化", "最小化", "目标"])
    constraint_hit = any(keyword in modeling_text for keyword in ["约束", "约束条件", "限制条件", "条件约束"])
    if _char_count(full_text) == 0:
        objective_constraint_completeness = np.nan
    else:
        objective_constraint_completeness = (int(objective_hit) + int(constraint_hit)) / 2

    method_coverage = _keyword_coverage(modeling_text or full_text, feature_cfg.get("model_keywords", []))
    validation_coverage = _keyword_coverage(full_text, feature_cfg.get("validation_keywords", []))
    structure_bonus = (
        int(bool(sections.get("model_building", "").strip()))
        + int(bool(sections.get("model_solving", "").strip()))
        + int(bool(sections.get("symbols", "").strip()))
    ) / 3
    method_reasonableness = _nan_if_empty_paper(
        full_text,
        0.45 * method_coverage + 0.35 * validation_coverage + 0.20 * structure_bonus,
    )

    return {
        "I09_模型数量与任务匹配度": model_task_match,
        "I10_公式密度": formula_density,
        "I11_变量定义覆盖率": variable_definition_coverage,
        "I12_目标函数约束完整性": objective_constraint_completeness,
        "I13_方法合理性语义评分": method_reasonableness,
    }


def extract_result_features(
    sections: dict[str, str],
    full_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract A4 result analysis and validation features."""
    feature_cfg = _feature_config(config)
    results = sections.get("results", "")
    expected_count = _expected_problem_count(full_text, config)

    if _char_count(full_text) == 0:
        result_completeness = np.nan
    elif not results:
        result_completeness = 0.0
    else:
        problem_score = _clip01(_covered_problem_count(results) / max(1, expected_count))
        numeric_score = _clip01(len(re.findall(r"\d+(?:\.\d+)?%?", results)) / 20)
        explanation_score = _keyword_coverage(results, ["结果", "分析", "可知", "说明", "最优"])
        result_completeness = 0.4 * problem_score + 0.3 * numeric_score + 0.3 * explanation_score

    numbered_figs, numbered_tables, _, _ = _figure_table_counts(full_text)
    total_numbered = numbered_figs + numbered_tables
    explanation_keywords = feature_cfg.get("chart_explanation_keywords", [])
    explanation_hits = 0
    for line in full_text.splitlines():
        if re.search(r"(图|表)\s*\d+", line) and any(keyword in line for keyword in explanation_keywords):
            explanation_hits += 1
    if _char_count(full_text) == 0:
        chart_explanation_rate = np.nan
    elif total_numbered == 0:
        chart_explanation_rate = 0.0
    else:
        chart_explanation_rate = _clip01(explanation_hits / total_numbered)

    sensitivity_exists = _nan_if_empty_paper(
        full_text,
        1.0
        if sections.get("sensitivity_analysis", "").strip()
        or any(keyword in full_text for keyword in ["灵敏度分析", "敏感性分析", "稳健性分析"])
        else 0.0,
    )
    error_exists = _nan_if_empty_paper(
        full_text,
        1.0
        if sections.get("error_analysis", "").strip()
        or any(keyword in full_text for keyword in ["误差分析", "残差分析", "RMSE", "MAE", "均方误差"])
        else 0.0,
    )

    return {
        "I14_结果完整率": result_completeness,
        "I15_图表解释率": chart_explanation_rate,
        "I16_灵敏度分析存在性": sensitivity_exists,
        "I17_误差分析存在性": error_exists,
    }


def extract_writing_features(
    sections: dict[str, str],
    full_text: str,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract A5 writing standardization and application value features."""
    feature_cfg = _feature_config(config)
    references = sections.get("references", "")
    entries = _reference_entries(references)
    if _char_count(full_text) == 0:
        reference_norm_rate = np.nan
    elif not references or not entries:
        reference_norm_rate = 0.0
    else:
        valid_entries = 0
        for entry in entries:
            has_index = bool(re.match(r"^(\[\s*\d+\s*\]|\d+[\.、])", entry))
            has_year = bool(re.search(r"(19|20)\d{2}", entry))
            has_source = any(token in entry for token in ["[J]", "[D]", "[M]", "[EB/OL]", "DOI", "http", "期刊"])
            if has_index and has_year and has_source:
                valid_entries += 1
        reference_norm_rate = _safe_ratio(valid_entries, len(entries))

    sentences = _sentences(full_text)
    if _char_count(full_text) == 0 or not sentences:
        readability = np.nan
    else:
        avg_sentence_len = np.mean([_char_count(sentence) for sentence in sentences])
        length_score = max(0.0, 1.0 - abs(avg_sentence_len - 45) / 45)
        punctuation_count = len(re.findall(r"[，。！？；：,.!?;:]", full_text))
        punctuation_density = punctuation_count / max(1, _char_count(full_text)) * 100
        punctuation_score = max(0.0, 1.0 - abs(punctuation_density - 8) / 8)
        readability = 0.6 * length_score + 0.4 * punctuation_score

    innovation_expression = _nan_if_empty_paper(
        full_text,
        _keyword_coverage(full_text, feature_cfg.get("innovation_keywords", [])),
    )

    application_text = _all_sections_text(sections, ["model_evaluation", "results", "problem_analysis"])
    if not application_text:
        application_text = full_text
    application_value = _nan_if_empty_paper(
        full_text,
        _keyword_coverage(application_text, feature_cfg.get("application_keywords", [])),
    )

    return {
        "I18_参考文献规范率": reference_norm_rate,
        "I19_语言可读性": readability,
        "I20_创新性表达": innovation_expression,
        "I21_推广应用价值": application_value,
    }


def extract_features(
    section_file: Path,
    text_file: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract all 21 raw features for one paper."""
    section_file = Path(section_file)
    payload = json.loads(section_file.read_text(encoding="utf-8"))
    sections = payload.get("sections", {})
    paper_id = str(payload.get("paper_id", section_file.stem)).zfill(2)
    filename = payload.get("filename", f"{paper_id}.txt")

    if text_file is not None and Path(text_file).exists():
        full_text = Path(text_file).read_text(encoding="utf-8", errors="ignore")
    else:
        full_text = "\n".join(sections.get(key, "") or "" for key in sections)

    row: dict[str, Any] = {
        "paper_id": paper_id,
        "filename": filename,
        "text_chars": _char_count(full_text),
    }
    row.update(extract_structure_features(sections, full_text, config))
    row.update(extract_logic_features(sections, full_text, config))
    row.update(extract_modeling_features(sections, full_text, config))
    row.update(extract_result_features(sections, full_text, config))
    row.update(extract_writing_features(sections, full_text, config))
    return row


def normalize_features(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature columns to [0, 1], preserving NaN values."""
    normalized = raw_table.copy()
    for column in FEATURE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
            continue

        series = pd.to_numeric(normalized[column], errors="coerce")
        feature_type = FEATURE_TYPES[column]
        if feature_type in {"ratio", "binary", "similarity"}:
            normalized[column] = series.clip(lower=0, upper=1)
        elif feature_type == "density":
            valid = series.dropna()
            if valid.empty:
                normalized[column] = np.nan
            else:
                min_value = valid.min()
                max_value = valid.max()
                if max_value == min_value:
                    normalized[column] = series.apply(lambda value: np.nan if pd.isna(value) else (1.0 if value > 0 else 0.0))
                else:
                    normalized[column] = (series - min_value) / (max_value - min_value)
        else:
            normalized[column] = series.clip(lower=0, upper=1)

    return normalized


def _log_missing_values(report: pd.DataFrame) -> None:
    """Log missing feature values for traceability."""
    logger = _get_logger()
    for _, row in report.iterrows():
        missing = [column for column in FEATURE_COLUMNS if pd.isna(row.get(column))]
        if missing:
            logger.warning("%s | missing feature values: %s", row.get("filename"), ",".join(missing))


def extract_all_features(
    sections_dir: Path,
    text_dir: Path,
    raw_output_path: Path,
    normalized_output_path: Path,
    config: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract raw and normalized feature tables for all papers."""
    logger = setup_feature_logger(log_path) if log_path is not None else _get_logger()
    sections_dir = Path(sections_dir)
    text_dir = Path(text_dir)
    raw_output_path = Path(raw_output_path)
    normalized_output_path = Path(normalized_output_path)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting feature extraction: sections_dir=%s text_dir=%s", sections_dir, text_dir)

    rows: list[dict[str, Any]] = []
    for section_file in sorted(sections_dir.glob("*.json")):
        paper_id = section_file.stem
        text_file = text_dir / f"{paper_id}.txt"
        try:
            row = extract_features(section_file, text_file=text_file, config=config)
            rows.append(row)
            logger.info("%s features extracted", section_file.name)
        except Exception as exc:
            logger.exception("%s | feature extraction failed: %s", section_file.name, exc)
            row = {"paper_id": paper_id, "filename": f"{paper_id}.txt", "text_chars": np.nan}
            for column in FEATURE_COLUMNS:
                row[column] = np.nan
            rows.append(row)

    columns = ["paper_id", "filename", "text_chars", *FEATURE_COLUMNS]
    raw_table = pd.DataFrame(rows, columns=columns)
    _log_missing_values(raw_table)

    normalized_table = normalize_features(raw_table)
    raw_table.to_excel(raw_output_path, index=False)
    normalized_table.to_excel(normalized_output_path, index=False)

    logger.info("Raw feature table saved: %s", raw_output_path)
    logger.info("Normalized feature table saved: %s", normalized_output_path)
    logger.info("Finished feature extraction: total=%s", len(raw_table))
    return raw_table, normalized_table


def extract_feature_table(input_dir: Path, output_path: Path) -> Path:
    """Backward-compatible wrapper for raw feature table generation."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    text_dir = input_dir.parents[1] / "extracted_text" if len(input_dir.parents) >= 2 else Path("data/extracted_text")
    normalized_path = output_path.with_name("appendix1_features_normalized.xlsx")
    extract_all_features(input_dir, text_dir, output_path, normalized_path)
    return output_path
