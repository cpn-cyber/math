"""Deep quality correction features for Problem 2 Step 13.

This step extracts six interpretable quality-correction features from Appendix
2 refined sections and Step 12 outputs. It does not run correlation analysis,
grey relation, PLS, QAF, or any predictive model.
"""

from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from modules.appendix2_pipeline import get_problem2_config, resolve_project_path


LOGGER_NAME = "A_MAGE_R3.problem2.deep_quality_features"
CANDIDATE_WEIGHT = 0.5
REVIEW_FOCUS_IDS = {"2-2", "2-3", "2-7", "2-10"}

DEEP_FEATURE_COLUMNS = [
    "task_coverage",
    "data_credibility",
    "method_fit",
    "formula_explanation",
    "result_closure",
    "stacking_penalty",
]

BENEFIT_FEATURES = [
    "task_coverage",
    "data_credibility",
    "method_fit",
    "formula_explanation",
    "result_closure",
]

TASK_MARKERS = [
    "问题一",
    "问题二",
    "问题三",
    "问题四",
    "任务一",
    "任务二",
    "任务三",
    "任务四",
    "闂涓€",
    "闂浜",
    "闂涓",
    "闂鍥",
]

TASK_SEED_KEYWORDS = [
    "数字",
    "教师",
    "能力",
    "质量",
    "评价",
    "模型",
    "指标",
    "影响",
    "预测",
    "路径",
    "高校",
    "增值",
    "鏁板瓧",
    "鏁欏笀",
    "鑳滀换",
    "璐ㄩ噺",
    "璇勪环",
    "妯″瀷",
    "鎸囨爣",
    "褰卞搷",
    "棰勬祴",
    "璺緞",
    "楂樻牎",
    "澧炲€",
]

DATA_CREDIBILITY_KEYWORDS = [
    "数据来源",
    "数据预处理",
    "数据说明",
    "原始数据",
    "附件数据",
    "缺失值处理",
    "异常值处理",
    "清洗",
    "可追溯",
    "数据统计描述",
    "数据",
    "预处理",
    "样本",
    "统计描述",
    "dataset",
    "data source",
    "missing value",
    "outlier",
    "clean",
    "鏁版嵁",
    "棰勫鐞",
    "缂哄け",
    "寮傚父",
    "娓呮礂",
    "鏍锋湰",
    "缁熻",
    "闄勪欢",
]

METHOD_FIT_KEYWORDS = [
    "目标函数",
    "约束条件",
    "评价指标",
    "预测误差",
    "参数说明",
    "模型适用条件",
    "求解流程",
    "任务对应",
    "优化",
    "回归",
    "聚类",
    "分类",
    "预测",
    "评价",
    "AHP",
    "TOPSIS",
    "熵权",
    "灰色",
    "规划",
    "objective",
    "constraint",
    "parameter",
    "process",
    "method",
    "model",
    "鐩爣鍑芥暟",
    "绾︽潫",
    "璇勪环鎸囨爣",
    "棰勬祴",
    "璇樊",
    "鍙傛暟",
    "姹傝В",
    "浼樺寲",
    "鍥炲綊",
    "鑱氱被",
    "妯″瀷",
]

FORMULA_EXPLANATION_KEYWORDS = [
    "变量说明",
    "参数含义",
    "文字解释",
    "推导说明",
    "应用场景",
    "其中",
    "表示",
    "含义",
    "说明",
    "推导",
    "适用",
    "variable",
    "parameter",
    "denote",
    "where",
    "鍙橀噺",
    "鍙傛暟",
    "琛ㄧず",
    "璇存槑",
    "鍚箟",
    "鎺ㄥ",
    "閫傜敤",
]

RESULT_CLOSURE_KEYWORDS = [
    "结果",
    "结论",
    "回应",
    "验证",
    "目标",
    "得到",
    "表明",
    "说明",
    "评价值",
    "预测值",
    "最终",
    "result",
    "conclusion",
    "verify",
    "final",
    "缁撴灉",
    "缁撹",
    "楠岃瘉",
    "鐩爣",
    "寰楀埌",
    "琛ㄦ槑",
    "鏈€缁",
]

FORMULA_PATTERN = re.compile(
    r"[=<>≤≥≈∑√]|目标函数|约束|公式|方程|矩阵|向量|"
    r"\b(?:max|min|argmax|argmin)\b|鐩爣鍑芥暟|绾︽潫|鍏紡|鐭╅樀|鍚戦噺",
    flags=re.IGNORECASE,
)


def setup_deep_quality_logger(log_path: Path) -> logging.Logger:
    """Configure Step 13 logger."""
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


def build_deep_feature_schema() -> list[str]:
    """Return Step 13 deep feature names."""
    return list(DEEP_FEATURE_COLUMNS)


def _natural_sort_key(value: Any) -> list[Any]:
    """Natural sort key for paper IDs."""
    text = Path(str(value)).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_json(path: Path) -> dict[str, Any]:
    """Load refined sections JSON."""
    if not Path(path).exists():
        return {"paper_id": Path(path).stem, "filename": f"{Path(path).stem}.txt", "sections": {}, "candidate_sections": {}}
    return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))


def _strip_page_markers(text: str) -> str:
    """Remove page markers."""
    return re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)


def _effective_chars(text: str) -> int:
    """Count non-whitespace characters."""
    return len(re.sub(r"\s+", "", _strip_page_markers(text or "")))


def _clip01(value: Any) -> float:
    """Clip a value into [0, 1]."""
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return np.nan


def _ratio(numerator: float, denominator: float) -> float:
    """Return safe clipped ratio."""
    if denominator <= 0:
        return 0.0
    return _clip01(numerator / denominator)


def _section(payload: dict[str, Any], key: str) -> str:
    """Get confirmed section text."""
    return str((payload.get("sections", {}) or {}).get(key, "") or "")


def _candidate_entry(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Get candidate section entry."""
    entry = (payload.get("candidate_sections", {}) or {}).get(key)
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str) and entry.strip():
        return {"text": entry}
    return None


def _candidate_text(payload: dict[str, Any], key: str) -> str:
    """Get candidate section text."""
    entry = _candidate_entry(payload, key)
    return str(entry.get("text", "") or "") if entry else ""


def _section_with_candidate(payload: dict[str, Any], key: str) -> tuple[str, float, bool]:
    """Return text, confidence weight, and candidate flag for one section."""
    confirmed = _section(payload, key)
    if _effective_chars(confirmed) > 0:
        return confirmed, 1.0, False
    candidate = _candidate_text(payload, key)
    if _effective_chars(candidate) > 0:
        return candidate, CANDIDATE_WEIGHT, True
    return "", 0.0, False


def _count_keywords(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    """Count keyword occurrences and collect matched keywords."""
    lowered = (text or "").lower()
    count = 0
    matched: list[str] = []
    for keyword in keywords:
        key = str(keyword)
        hits = lowered.count(key.lower())
        if hits:
            count += hits
            matched.append(key)
    return count, matched


def _snippet(text: str, keywords: list[str], max_chars: int = 120) -> str:
    """Return a short evidence snippet from source text."""
    text = text or ""
    lowered = text.lower()
    for keyword in keywords:
        index = lowered.find(str(keyword).lower())
        if index >= 0:
            start = max(0, index - 45)
            end = min(len(text), index + len(str(keyword)) + max_chars)
            return re.sub(r"\s+", " ", text[start:end]).strip()
    return re.sub(r"\s+", " ", text[:max_chars]).strip()


def _keyword_score(text: str, keywords: list[str], target_unique_hits: int) -> tuple[float, list[str], str]:
    """Saturated keyword coverage score."""
    _, matched = _count_keywords(text, keywords)
    unique = sorted(set(matched))
    return _ratio(len(unique), target_unique_hits), unique, _snippet(text, unique or keywords)


def _shingles(text: str, width: int = 2) -> set[str]:
    """Return character shingles for rough echo checks."""
    compact = re.sub(r"\s+", "", _strip_page_markers(text or ""))
    if len(compact) < width:
        return set()
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def _echo_score(source: str, target: str) -> float:
    """Scaled Jaccard echo score."""
    source_set = _shingles(source)
    target_set = _shingles(target)
    if not source_set or not target_set:
        return 0.0
    return _clip01(3.0 * len(source_set & target_set) / max(1, len(source_set | target_set)))


def _numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    """Read numeric cell safely."""
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _surface_rows(raw_table: pd.DataFrame, normalized_table: pd.DataFrame, paper_id: str) -> tuple[pd.Series, pd.Series]:
    """Return raw and normalized surface rows for one paper."""
    raw_match = raw_table.loc[raw_table["paper_id"].astype(str).eq(str(paper_id))]
    norm_match = normalized_table.loc[normalized_table["paper_id"].astype(str).eq(str(paper_id))]
    raw = raw_match.iloc[0] if not raw_match.empty else pd.Series(dtype=float)
    norm = norm_match.iloc[0] if not norm_match.empty else pd.Series(dtype=float)
    return raw, norm


def _derive_task_keywords(payloads: list[dict[str, Any]], logger: logging.Logger) -> list[str]:
    """Infer task keywords because the original problem statement is not available here."""
    corpus_parts: list[str] = []
    for payload in payloads:
        corpus_parts.extend(str(value or "") for value in (payload.get("sections", {}) or {}).values())
        for entry in (payload.get("candidate_sections", {}) or {}).values():
            if isinstance(entry, dict):
                corpus_parts.append(str(entry.get("text", "") or ""))
    corpus = "\n".join(corpus_parts)

    counter: Counter[str] = Counter()
    for keyword in TASK_SEED_KEYWORDS + TASK_MARKERS:
        hits = corpus.count(keyword)
        if hits:
            counter[keyword] += hits
    keywords = [keyword for keyword, _ in counter.most_common(14)]
    if not keywords:
        keywords = TASK_SEED_KEYWORDS[:10]
    logger.info(
        "No standalone problem-statement keyword file was found; task keywords inferred from Appendix 2 high-frequency terms and subquestion markers: %s",
        ",".join(keywords),
    )
    return keywords


def _formula_context_score(text: str) -> tuple[float, list[str], str]:
    """Score formula explanation by nearby context."""
    lines = text.splitlines()
    formula_indices = [index for index, line in enumerate(lines) if FORMULA_PATTERN.search(line)]
    if not formula_indices:
        return 0.0, [], ""
    explained = 0
    matched_all: list[str] = []
    first_evidence = ""
    for index in formula_indices:
        window = "\n".join(lines[max(0, index - 2) : min(len(lines), index + 3)])
        _, matched = _count_keywords(window, FORMULA_EXPLANATION_KEYWORDS)
        if matched:
            explained += 1
            matched_all.extend(matched)
            if not first_evidence:
                first_evidence = _snippet(window, matched)
    return _ratio(explained, len(formula_indices)), sorted(set(matched_all)), first_evidence


def _compute_task_coverage(payload: dict[str, Any], task_keywords: list[str]) -> tuple[float, dict[str, Any]]:
    """Compute task coverage using problem/model/result/evaluation text."""
    problem_text = _section(payload, "problem_statement") + "\n" + _section(payload, "problem_analysis")
    model_text = _section(payload, "model_building") + "\n" + _section(payload, "model_solving")
    result_text, result_weight, result_is_candidate = _section_with_candidate(payload, "results")
    closure_text = result_text + "\n" + _section(payload, "model_evaluation")
    combined = problem_text + "\n" + model_text + "\n" + closure_text

    keyword_scores: list[float] = []
    matched: list[str] = []
    for keyword in task_keywords:
        score = 0.0
        if keyword in problem_text:
            score += 0.25
        if keyword in model_text:
            score += 0.25
        if keyword in result_text:
            score += 0.30 * result_weight
        if keyword in closure_text:
            score += 0.20
        if score > 0:
            matched.append(keyword)
        keyword_scores.append(min(1.0, score))
    marker_hits = sum(1 for marker in TASK_MARKERS if marker in combined)
    score = _clip01(0.75 * (float(np.mean(keyword_scores)) if keyword_scores else 0.0) + 0.25 * _ratio(marker_hits, 4))
    return score, {
        "matched_keywords": ",".join(sorted(set(matched))[:12]),
        "section_source": "results_candidate" if result_is_candidate else "confirmed_sections",
        "candidate_weight": result_weight if result_is_candidate else 1.0,
        "evidence": _snippet(combined, matched or task_keywords),
        "note": "results candidate weighted at 0.5" if result_is_candidate else "confirmed sections used",
    }


def _compute_data_credibility(payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute data credibility score."""
    text = "\n".join(
        [
            _section(payload, "problem_statement"),
            _section(payload, "problem_analysis"),
            _section(payload, "model_building"),
            _section(payload, "model_solving"),
            _section(payload, "results"),
            _section(payload, "appendix"),
        ]
    )
    score, matched, evidence = _keyword_score(text, DATA_CREDIBILITY_KEYWORDS, 7)
    return score, {
        "matched_keywords": ",".join(matched[:12]),
        "section_source": "problem/model/result/appendix confirmed sections",
        "candidate_weight": 1.0,
        "evidence": evidence,
        "note": "data chain evidence keyword coverage",
    }


def _compute_method_fit(payload: dict[str, Any], norm_row: pd.Series) -> tuple[float, dict[str, Any]]:
    """Compute method-fit score."""
    text = "\n".join(
        [
            _section(payload, "problem_analysis"),
            _section(payload, "model_building"),
            _section(payload, "model_solving"),
            _section(payload, "results"),
        ]
    )
    keyword_score, matched, evidence = _keyword_score(text, METHOD_FIT_KEYWORDS, 8)
    objective_score = _numeric(norm_row, "objective_constraint_completeness", 0.0)
    formula_density = _numeric(norm_row, "formula_density", 0.0)
    problem_coverage = _numeric(norm_row, "problem_restatement_coverage", 0.0)
    score = _clip01(0.60 * keyword_score + 0.20 * objective_score + 0.10 * formula_density + 0.10 * problem_coverage)
    return score, {
        "matched_keywords": ",".join(matched[:12]),
        "section_source": "modeling/solution/result confirmed sections",
        "candidate_weight": 1.0,
        "evidence": evidence,
        "note": f"objective_constraint={objective_score:.3f}; formula_density_norm={formula_density:.3f}",
    }


def _compute_formula_explanation(payload: dict[str, Any], raw_row: pd.Series) -> tuple[float, dict[str, Any]]:
    """Compute formula explanation score with formula context evidence."""
    text = "\n".join(
        [
            _section(payload, "symbols"),
            _section(payload, "model_building"),
            _section(payload, "model_solving"),
            _section(payload, "results"),
        ]
    )
    context_score, context_matches, context_evidence = _formula_context_score(text)
    keyword_score, keyword_matches, keyword_evidence = _keyword_score(text, FORMULA_EXPLANATION_KEYWORDS, 6)
    step12_formula = _numeric(raw_row, "formula_explanation_rate", 0.0)
    variable_coverage = _numeric(raw_row, "variable_definition_coverage", 0.0)
    score = _clip01(0.40 * step12_formula + 0.25 * variable_coverage + 0.25 * context_score + 0.10 * keyword_score)
    evidence = context_evidence or keyword_evidence
    return score, {
        "matched_keywords": ",".join(sorted(set(context_matches + keyword_matches))[:12]),
        "section_source": "symbols/modeling/solution/result confirmed sections",
        "candidate_weight": 1.0,
        "evidence": evidence,
        "note": f"step12_formula_explanation={step12_formula:.3f}; variable_coverage={variable_coverage:.3f}; context_score={context_score:.3f}",
    }


def _compute_result_closure(payload: dict[str, Any], raw_row: pd.Series, paper_id: str) -> tuple[float, dict[str, Any]]:
    """Compute result closure score, respecting result candidates."""
    problem_text = _section(payload, "problem_statement") + "\n" + _section(payload, "problem_analysis")
    result_text, result_weight, result_is_candidate = _section_with_candidate(payload, "results")
    evaluation_text = _section(payload, "model_evaluation")
    closure_text = result_text + "\n" + evaluation_text

    echo = _echo_score(problem_text, closure_text)
    keyword_score, matched, evidence = _keyword_score(closure_text, RESULT_CLOSURE_KEYWORDS, 7)
    step12_echo = _numeric(raw_row, "conclusion_echo_rate", 0.0)
    raw_score = 0.35 * echo + 0.25 * keyword_score + 0.20 * result_weight + 0.20 * step12_echo
    score = _clip01(raw_score * (result_weight if result_is_candidate else 1.0))
    abnormal_note = ""
    if paper_id in {"2-2", "2-10"}:
        abnormal_note = "; Step12B flagged conclusion_echo_rate anomaly"
    return score, {
        "matched_keywords": ",".join(matched[:12]),
        "section_source": "results_candidate" if result_is_candidate else "confirmed_results_and_evaluation",
        "candidate_weight": result_weight if result_is_candidate else 1.0,
        "evidence": evidence,
        "note": (
            ("results candidate weighted at 0.5; " if result_is_candidate else "confirmed results used; ")
            + f"echo={echo:.3f}; step12_conclusion_echo={step12_echo:.3f}"
            + abnormal_note
        ),
    }


def _compute_stacking_penalty(norm_row: pd.Series, result_closure: float, formula_explanation: float) -> tuple[float, dict[str, Any]]:
    """Compute high-is-bad stacking penalty."""
    total_chars = _numeric(norm_row, "total_chars", 0.0)
    formula_density = _numeric(norm_row, "formula_density", 0.0)
    figure_density = _numeric(norm_row, "figure_table_density", 0.0)
    formula_rate = _numeric(norm_row, "formula_explanation_rate", 0.0)
    figure_rate = _numeric(norm_row, "figure_table_explanation_rate", 0.0)
    volume = float(np.nanmean([total_chars, formula_density, figure_density]))
    explanation_strength = float(np.nanmean([formula_rate, figure_rate, formula_explanation, result_closure]))
    score = _clip01(volume * (1.0 - explanation_strength))
    return score, {
        "matched_keywords": "",
        "section_source": "Step12 normalized surface features plus result_closure",
        "candidate_weight": 1.0,
        "evidence": (
            f"total_chars_norm={total_chars:.3f}; formula_density_norm={formula_density:.3f}; "
            f"figure_table_density_norm={figure_density:.3f}; explanation_strength={explanation_strength:.3f}"
        ),
        "note": "negative feature: higher value means higher stacking risk",
    }


def _review_note(feature: str, score: float, evidence: dict[str, Any], paper_id: str, paper_flag: str) -> str:
    """Generate review recommendation for one feature."""
    if paper_id in REVIEW_FOCUS_IDS:
        prefix = "review_focus; "
    else:
        prefix = ""
    if feature == "stacking_penalty" and score >= 0.5:
        return prefix + "high stacking risk; recommend manual review"
    if evidence.get("section_source") == "results_candidate" and feature in {"task_coverage", "result_closure"}:
        return prefix + "candidate results used at 0.5 confidence; recommend review"
    if feature != "stacking_penalty" and 0.35 <= score <= 0.65:
        return prefix + "boundary auto score; recommend review"
    if paper_flag in {"need_review", "partial_candidate"} and feature in {"result_closure", "task_coverage", "formula_explanation"}:
        return prefix + f"Step12B flag={paper_flag}; spot check recommended"
    if not str(evidence.get("evidence", "")).strip():
        return prefix + "weak evidence snippet; spot check recommended"
    return prefix + "auto score usable"


def _load_step12b_flags(audit_path: Path) -> dict[str, str]:
    """Load paper-level Step 12B quality flags."""
    if not Path(audit_path).exists():
        return {}
    flags = pd.read_excel(audit_path, sheet_name="paper_quality_flags")
    return dict(zip(flags["paper_id"].astype(str), flags["feature_quality_flag"].astype(str)))


def extract_deep_quality_features(
    sections_dir: Path,
    surface_raw_path: Path,
    surface_normalized_path: Path,
    features_with_q_path: Path,
    candidate_usage_path: Path,
    step12_audit_path: Path,
    output_path: Path,
    review_template_path: Path,
    evidence_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Extract Step 13 deep quality features and save outputs."""
    logger = setup_deep_quality_logger(log_path)
    logger.info("Starting Step 13 deep quality correction feature extraction")
    logger.info("This step reads Step 12 outputs only; it does not recompute Q_i or train models.")

    sections_dir = Path(sections_dir)
    output_path = Path(output_path)
    review_template_path = Path(review_template_path)
    evidence_path = Path(evidence_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_features = pd.read_excel(surface_raw_path)
    normalized_features = pd.read_excel(surface_normalized_path)
    features_with_q = pd.read_excel(features_with_q_path)
    candidate_usage = pd.read_excel(candidate_usage_path)
    step12_flags = _load_step12b_flags(step12_audit_path)
    for frame in [raw_features, normalized_features, features_with_q, candidate_usage]:
        if "paper_id" in frame.columns:
            frame["paper_id"] = frame["paper_id"].astype(str)

    section_files = sorted(sections_dir.glob("*.json"), key=_natural_sort_key)
    payloads = [_load_json(path) for path in section_files]
    task_keywords = _derive_task_keywords(payloads, logger)

    auto_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for section_file, payload in zip(section_files, payloads):
        paper_id = str(payload.get("paper_id", section_file.stem))
        filename = str(payload.get("filename", f"{paper_id}.txt"))
        raw_row, norm_row = _surface_rows(raw_features, normalized_features, paper_id)
        candidate_sections = sorted((payload.get("candidate_sections", {}) or {}).keys())
        paper_flag = step12_flags.get(paper_id, "unknown")
        review_focus = paper_id in REVIEW_FOCUS_IDS or paper_flag in {"need_review", "partial_candidate"}
        if review_focus:
            logger.warning("%s marked as review_focus by Step12B flag=%s", paper_id, paper_flag)

        scores: dict[str, float] = {}
        evidence_map: dict[str, dict[str, Any]] = {}

        scores["task_coverage"], evidence_map["task_coverage"] = _compute_task_coverage(payload, task_keywords)
        scores["data_credibility"], evidence_map["data_credibility"] = _compute_data_credibility(payload)
        scores["method_fit"], evidence_map["method_fit"] = _compute_method_fit(payload, norm_row)
        scores["formula_explanation"], evidence_map["formula_explanation"] = _compute_formula_explanation(payload, raw_row)
        scores["result_closure"], evidence_map["result_closure"] = _compute_result_closure(payload, raw_row, paper_id)
        scores["stacking_penalty"], evidence_map["stacking_penalty"] = _compute_stacking_penalty(
            norm_row,
            result_closure=scores["result_closure"],
            formula_explanation=scores["formula_explanation"],
        )

        auto_score = _clip01(
            float(np.nanmean([scores[name] for name in BENEFIT_FEATURES]))
            * (1.0 - 0.35 * scores["stacking_penalty"])
        )
        review_features: list[str] = []
        for feature in DEEP_FEATURE_COLUMNS:
            note = _review_note(feature, scores[feature], evidence_map[feature], paper_id, paper_flag)
            if note != "auto score usable":
                review_features.append(feature)
            review_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": feature,
                    "auto_score": scores[feature],
                    "manual_score": np.nan,
                    "final_score": scores[feature],
                    "evidence": evidence_map[feature]["evidence"],
                    "review_note": note,
                }
            )
            evidence_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "feature_name": feature,
                    "auto_score": scores[feature],
                    "matched_keywords": evidence_map[feature]["matched_keywords"],
                    "section_source": evidence_map[feature]["section_source"],
                    "candidate_weight": evidence_map[feature]["candidate_weight"],
                    "evidence": evidence_map[feature]["evidence"],
                    "rule_note": evidence_map[feature]["note"],
                    "review_focus": review_focus,
                    "step12b_flag": paper_flag,
                }
            )
            logger.info(
                "%s %s=%.6f source=%s candidate_weight=%s evidence=%s note=%s",
                paper_id,
                feature,
                scores[feature],
                evidence_map[feature]["section_source"],
                evidence_map[feature]["candidate_weight"],
                evidence_map[feature]["evidence"],
                evidence_map[feature]["note"],
            )

        auto_row: dict[str, Any] = {
            "paper_id": paper_id,
            "filename": filename,
            **{feature: scores[feature] for feature in DEEP_FEATURE_COLUMNS},
            "auto_score": auto_score,
            "review_focus": review_focus,
            "step12b_flag": paper_flag,
            "candidate_sections": ",".join(candidate_sections),
            "review_suggested_features": ",".join(review_features),
        }
        for feature in DEEP_FEATURE_COLUMNS:
            auto_row[f"{feature}_evidence"] = evidence_map[feature]["evidence"]
        auto_rows.append(auto_row)
        logger.info("%s aggregate auto_score=%.6f review_features=%s", paper_id, auto_score, ",".join(review_features) or "none")

    auto_table = pd.DataFrame(auto_rows).sort_values("paper_id", key=lambda series: series.map(_natural_sort_key)).reset_index(drop=True)
    review_template = pd.DataFrame(
        review_rows,
        columns=["paper_id", "filename", "feature_name", "auto_score", "manual_score", "final_score", "evidence", "review_note"],
    )
    evidence_table = pd.DataFrame(evidence_rows)

    auto_table.to_excel(output_path, index=False)
    review_template.to_excel(review_template_path, index=False)
    evidence_table.to_excel(evidence_path, index=False)

    feature_ranges = {
        feature: {
            "min": float(auto_table[feature].min()) if len(auto_table) else math.nan,
            "max": float(auto_table[feature].max()) if len(auto_table) else math.nan,
        }
        for feature in DEEP_FEATURE_COLUMNS
    }
    high_threshold = auto_table["stacking_penalty"].quantile(0.75)
    high_stacking = auto_table.loc[
        auto_table["stacking_penalty"].ge(high_threshold),
        ["paper_id", "filename", "stacking_penalty"],
    ].sort_values("stacking_penalty", ascending=False)
    review_suggestions = review_template.loc[
        review_template["review_note"].ne("auto score usable"),
        ["paper_id", "feature_name", "auto_score", "review_note"],
    ]

    logger.info("Feature ranges: %s", feature_ranges)
    logger.info("High stacking penalty papers: %s", high_stacking.to_dict(orient="records"))
    logger.info("Review suggestion rows: %s", len(review_suggestions))
    logger.info("Saved deep quality auto table: %s", output_path)
    logger.info("Saved deep quality review template: %s", review_template_path)
    logger.info("Saved deep quality evidence table: %s", evidence_path)
    logger.info("Finished Step 13: papers=%s", len(auto_table))

    return {
        "auto_table": auto_table,
        "review_template": review_template,
        "evidence_table": evidence_table,
        "feature_ranges": feature_ranges,
        "high_stacking": high_stacking,
        "review_suggestions": review_suggestions,
        "review_focus": auto_table.loc[auto_table["review_focus"], ["paper_id", "filename", "step12b_flag"]],
        "output_path": output_path,
        "review_template_path": review_template_path,
        "evidence_path": evidence_path,
        "log_path": Path(log_path),
    }


def run_step13_deep_quality_features(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 13 with project paths."""
    problem2_config = get_problem2_config(config_path)
    intermediate_dir = resolve_project_path(problem2_config["appendix2_intermediate_dir"])
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    sections_dir = intermediate_dir / "sections_refined"
    if not sections_dir.is_dir() or not list(sections_dir.glob("*.json")):
        sections_dir = intermediate_dir / "sections"

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return extract_deep_quality_features(
        sections_dir=sections_dir,
        surface_raw_path=tables_dir / "appendix2_surface_features_raw.xlsx",
        surface_normalized_path=tables_dir / "appendix2_surface_features_normalized.xlsx",
        features_with_q_path=tables_dir / "appendix2_features_with_q.xlsx",
        candidate_usage_path=tables_dir / "appendix2_candidate_usage_report.xlsx",
        step12_audit_path=tables_dir / "appendix2_step12_quality_audit.xlsx",
        output_path=tables_dir / "appendix2_deep_quality_features_auto.xlsx",
        review_template_path=tables_dir / "appendix2_deep_quality_review_template.xlsx",
        evidence_path=tables_dir / "appendix2_deep_quality_evidence.xlsx",
        log_path=logs_dir / "deep_quality_features.log",
    )
