"""Surface feature extraction and weak label construction for Problem 2.

Step 12 reads Appendix 2 refined sections, extracts reproducible text-surface
features, and builds weak quality labels with the sealed Problem 1 weights and
TOPSIS scoring logic. It does not run correlation analysis, PLS, QAF, or any
downstream model.
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

from modules import feature_extractor, topsis
from modules.appendix2_pipeline import (
    SECTION_KEY_MAP,
    SECTION_REPORT_KEYS,
    get_problem2_config,
    load_config,
    resolve_project_path,
)


LOGGER_NAME = "A_MAGE_R3.problem2.features_labels"
CANDIDATE_WEIGHT = 0.5
DENSITY_PER_CHARS = 1000

SURFACE_FEATURE_COLUMNS = [
    "total_chars",
    "page_count",
    "abstract_ratio",
    "section_coverage",
    "paragraph_balance",
    "formula_density",
    "formula_explanation_rate",
    "variable_definition_coverage",
    "objective_constraint_completeness",
    "figure_table_density",
    "figure_table_numbering_rate",
    "figure_table_explanation_rate",
    "cross_reference_rate",
    "logic_connective_density",
    "problem_restatement_coverage",
    "conclusion_echo_rate",
    "reference_norm_rate",
    "appendix_code_presence",
    "citation_norm_rate",
]

RATIO_FEATURES = {
    "abstract_ratio",
    "section_coverage",
    "paragraph_balance",
    "formula_explanation_rate",
    "variable_definition_coverage",
    "objective_constraint_completeness",
    "figure_table_numbering_rate",
    "figure_table_explanation_rate",
    "problem_restatement_coverage",
    "conclusion_echo_rate",
    "reference_norm_rate",
    "appendix_code_presence",
    "citation_norm_rate",
}

MINMAX_FEATURES = {
    "total_chars",
    "page_count",
    "formula_density",
    "figure_table_density",
    "cross_reference_rate",
    "logic_connective_density",
}

CORE_REPORT_SECTIONS = list(SECTION_REPORT_KEYS)

SECTION_TEXT_ALIASES = {
    "problem_analysis": "problem_analysis",
    "modeling": "model_building",
    "solution": "model_solving",
}

FORMULA_PATTERNS = [
    r"[=<>≤≥≈∑√]",
    r"\b(?:max|min|argmax|argmin)\b",
    r"公式|方程|函数|矩阵|向量|权重|变量|约束|目标函数",
    r"[A-Za-z][A-Za-z0-9_]*\s*[_=]",
]

FORMULA_EXPLANATION_WORDS = [
    "其中",
    "表示",
    "为",
    "含义",
    "说明",
    "记为",
    "definition",
    "denote",
]

OBJECTIVE_WORDS = [
    "目标函数",
    "最大化",
    "最小化",
    "max",
    "min",
    "objective",
    "maximize",
    "minimize",
]

CONSTRAINT_WORDS = [
    "约束",
    "约束条件",
    "限制条件",
    "s.t.",
    "subject to",
    "constraint",
]

DEFAULT_LOGIC_CONNECTIVES = [
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
    "可得",
    "证明",
    "验证",
]

PROBLEM_MARKERS = [
    "问题一",
    "问题二",
    "问题三",
    "问题四",
    "任务一",
    "任务二",
    "任务三",
    "任务四",
    "闂涓€",
    "闂浜�",
    "闂涓�",
    "闂鍥�",
]

RESULT_ECHO_KEYWORDS = [
    "结果",
    "结论",
    "评价",
    "预测",
    "模型",
    "指标",
    "验证",
    "分析",
    "result",
    "conclusion",
    "model",
]

CODE_KEYWORDS = [
    "import ",
    "def ",
    "class ",
    "for ",
    "while ",
    "代码",
    "Python",
    "MATLAB",
    "numpy",
    "pandas",
]


def setup_feature_label_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 12 logger."""
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


def _natural_sort_key(path: Path) -> list[Any]:
    """Sort paths such as 2-1, 2-10 naturally."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).stem)]


def _read_text(path: Path) -> str:
    """Read text with a tolerant UTF-8 decoder."""
    return Path(path).read_text(encoding="utf-8", errors="ignore") if Path(path).exists() else ""


def _load_json(path: Path) -> dict[str, Any]:
    """Load a section JSON payload."""
    if not Path(path).exists():
        return {"paper_id": Path(path).stem, "filename": f"{Path(path).stem}.txt", "sections": {}, "candidate_sections": {}}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strip_page_markers(text: str) -> str:
    """Remove page markers from text."""
    return re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)


def _effective_chars(text: str) -> int:
    """Count non-whitespace characters after page markers are removed."""
    return len(re.sub(r"\s+", "", _strip_page_markers(text or "")))


def _count_pages(text: str) -> int:
    """Count page markers in extracted text."""
    matches = re.findall(r"\[PAGE\s+\d+\]", text or "", flags=re.IGNORECASE)
    return len(matches)


def _clip01(value: Any) -> float:
    """Clip a numeric value into [0, 1]."""
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return np.nan


def _ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio clipped to [0, 1]."""
    if denominator <= 0:
        return 0.0
    return _clip01(numerator / denominator)


def _density(count: float, total_chars: int) -> float:
    """Return per-1000-character density."""
    if total_chars <= 0:
        return 0.0
    return float(count) / total_chars * DENSITY_PER_CHARS


def _to_bool(value: Any) -> bool:
    """Convert common spreadsheet values to bool."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_pdf_lookup(pdf_report_path: Path) -> dict[str, dict[str, Any]]:
    """Load parse metadata for page count and OCR flags."""
    if not Path(pdf_report_path).exists():
        return {}
    report = pd.read_excel(pdf_report_path)
    if "paper_id" not in report.columns:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in report.iterrows():
        paper_id = str(row.get("paper_id", "")).strip()
        lookup[paper_id] = {
            "pages": int(row.get("pages", 0) or 0),
            "ocr_used": _to_bool(row.get("ocr_used", False)),
            "parse_success": _to_bool(row.get("parse_success", False)),
        }
    return lookup


def _json_section_key(report_key: str) -> str:
    """Map report section names to JSON section keys."""
    return SECTION_KEY_MAP.get(report_key, SECTION_TEXT_ALIASES.get(report_key, report_key))


def _confirmed_section(payload: dict[str, Any], report_key: str) -> str:
    """Return a confirmed section text."""
    sections = payload.get("sections", {}) or {}
    return str(sections.get(_json_section_key(report_key), "") or "")


def _candidate_entry(payload: dict[str, Any], report_key: str) -> dict[str, Any] | None:
    """Return a candidate section entry without promoting it to confirmed."""
    candidates = payload.get("candidate_sections", {}) or {}
    entry = candidates.get(report_key)
    if entry is None:
        entry = candidates.get(_json_section_key(report_key))
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str) and entry.strip():
        return {"text": entry}
    return None


def _candidate_text(payload: dict[str, Any], report_key: str) -> str:
    """Return candidate text for a section."""
    entry = _candidate_entry(payload, report_key)
    if not entry:
        return ""
    return str(entry.get("text", "") or "")


def _weighted_section_text(
    payload: dict[str, Any],
    report_key: str,
    feature_name: str,
    usage_rows: list[dict[str, Any]],
    candidate_weight: float = CANDIDATE_WEIGHT,
) -> str:
    """Return confirmed text plus candidate text repeated through a tag.

    The candidate is not promoted to a section; it is only recorded as
    low-confidence evidence and scored at half weight through explicit ratios.
    For text-similarity features, returning the candidate text is conservative
    because a missing confirmed section would otherwise contribute nothing.
    """
    confirmed = _confirmed_section(payload, report_key)
    candidate = _candidate_text(payload, report_key)
    if candidate and not confirmed:
        usage_rows.append(
            {
                "paper_id": str(payload.get("paper_id", "")),
                "filename": str(payload.get("filename", "")),
                "candidate_section": report_key,
                "feature_name": feature_name,
                "candidate_weight": candidate_weight,
                "used_chars": _effective_chars(candidate),
                "usage_note": "candidate used as low-confidence evidence; not promoted to confirmed section",
            }
        )
        return candidate
    return confirmed


def _weighted_section_chars(
    payload: dict[str, Any],
    report_key: str,
    feature_name: str,
    usage_rows: list[dict[str, Any]],
    candidate_weight: float = CANDIDATE_WEIGHT,
) -> float:
    """Return confirmed section chars plus weighted candidate chars."""
    confirmed = _confirmed_section(payload, report_key)
    confirmed_chars = _effective_chars(confirmed)
    if confirmed_chars > 0:
        return float(confirmed_chars)
    candidate = _candidate_text(payload, report_key)
    candidate_chars = _effective_chars(candidate)
    if candidate_chars > 0:
        usage_rows.append(
            {
                "paper_id": str(payload.get("paper_id", "")),
                "filename": str(payload.get("filename", "")),
                "candidate_section": report_key,
                "feature_name": feature_name,
                "candidate_weight": candidate_weight,
                "used_chars": candidate_chars,
                "usage_note": "candidate chars multiplied by 0.5; candidate kept separate from confirmed sections",
            }
        )
        return candidate_weight * candidate_chars
    return 0.0


def _section_exists(payload: dict[str, Any], report_key: str) -> bool:
    """Check whether a confirmed section exists."""
    return _effective_chars(_confirmed_section(payload, report_key)) > 0


def _candidate_exists(payload: dict[str, Any], report_key: str) -> bool:
    """Check whether a candidate section exists."""
    return _effective_chars(_candidate_text(payload, report_key)) > 0


def _count_regex(pattern: str, text: str, flags: int = re.IGNORECASE) -> int:
    """Count regex matches."""
    return len(re.findall(pattern, text or "", flags=flags))


def _count_keywords(text: str, keywords: list[str]) -> int:
    """Count keyword occurrences by substring matching."""
    text = text or ""
    count = 0
    for keyword in keywords:
        if not keyword:
            continue
        count += text.lower().count(str(keyword).lower())
    return count


def _paragraph_balance(full_text: str) -> float:
    """Measure how balanced paragraph lengths are."""
    body = _strip_page_markers(full_text)
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", body) if _effective_chars(item) >= 20]
    if len(paragraphs) < 2:
        return 0.0
    lengths = np.array([_effective_chars(item) for item in paragraphs], dtype=float)
    mean = float(lengths.mean())
    if mean <= 0:
        return 0.0
    cv = float(lengths.std(ddof=0) / mean)
    return _clip01(1.0 / (1.0 + cv))


def _formula_lines(full_text: str) -> list[tuple[int, str]]:
    """Return lines that look formula-like."""
    lines = full_text.splitlines()
    rows: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in FORMULA_PATTERNS):
            rows.append((index, line))
    return rows


def _formula_explanation_rate(full_text: str, formula_rows: list[tuple[int, str]]) -> float:
    """Estimate whether formula-like lines have nearby explanations."""
    if not formula_rows:
        return 0.0
    lines = full_text.splitlines()
    explained = 0
    for index, line in formula_rows:
        window = "\n".join(lines[max(0, index - 1) : min(len(lines), index + 2)]) + "\n" + line
        if _count_keywords(window, FORMULA_EXPLANATION_WORDS) > 0:
            explained += 1
    return _ratio(explained, len(formula_rows))


def _variable_definition_coverage(full_text: str, symbols_text: str, formula_rows: list[tuple[int, str]]) -> float:
    """Estimate coverage between variables in formulas and symbol definitions."""
    formula_text = "\n".join(line for _, line in formula_rows)
    tokens = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{0,8}\b", formula_text))
    stop_words = {
        "max",
        "min",
        "argmax",
        "argmin",
        "AHP",
        "TOPSIS",
        "PDF",
        "Table",
        "Figure",
        "the",
        "and",
        "for",
        "while",
        "import",
        "print",
    }
    variables = {token for token in tokens if token not in stop_words and len(token) <= 8}
    if not variables:
        return 0.0
    definition_text = symbols_text + "\n" + "\n".join(
        line for line in full_text.splitlines() if _count_keywords(line, ["表示", "变量", "符号", "含义"]) > 0
    )
    defined = {token for token in variables if re.search(rf"\b{re.escape(token)}\b", definition_text)}
    return _ratio(len(defined), len(variables))


def _objective_constraint_completeness(model_text: str) -> float:
    """Check whether objective and constraint evidence both appear."""
    objective_hit = _count_keywords(model_text, OBJECTIVE_WORDS) > 0
    constraint_hit = _count_keywords(model_text, CONSTRAINT_WORDS) > 0
    return 0.5 * float(objective_hit) + 0.5 * float(constraint_hit)


def _figure_table_counts(full_text: str) -> tuple[int, int]:
    """Return total and numbered figure/table references."""
    numbered_pattern = r"(?:图|表|Figure|Table|鍥.?|琛.?)\s*[0-9一二三四五六七八九十]+"
    total_pattern = r"(?:图|表|Figure|Table|figure|table|鍥.?|琛.?)"
    numbered = _count_regex(numbered_pattern, full_text)
    total = max(numbered, _count_regex(total_pattern, full_text))
    return total, numbered


def _figure_table_explanation_rate(full_text: str, result_text: str) -> float:
    """Estimate whether figures and tables are explained near their mentions."""
    text = result_text if _effective_chars(result_text) > 0 else full_text
    lines = text.splitlines()
    ref_pattern = re.compile(r"(图|表|Figure|Table|figure|table|鍥.?|琛.?)\s*[0-9一二三四五六七八九十]*")
    explanation_words = ["说明", "表明", "显示", "可知", "分析", "趋势", "对比", "由图", "由表", "show", "indicate"]
    ref_lines = [line for line in lines if ref_pattern.search(line)]
    if not ref_lines:
        return 0.0
    explained = sum(1 for line in ref_lines if _count_keywords(line, explanation_words) > 0)
    return _ratio(explained, len(ref_lines))


def _cross_reference_rate(full_text: str) -> float:
    """Count body cross references per 1000 chars."""
    pattern = r"(?:见图|见表|如图|如表|由图|由表|图\s*\d+|表\s*\d+|Figure\s*\d+|Table\s*\d+)"
    return _density(_count_regex(pattern, full_text), _effective_chars(full_text))


def _problem_restatement_coverage(problem_text: str, full_text: str, expected_default: int = 3) -> float:
    """Estimate coverage of problem/task restatement."""
    expected = set()
    covered = set()
    for marker in PROBLEM_MARKERS:
        if marker and marker in full_text:
            expected.add(marker)
        if marker and marker in problem_text:
            covered.add(marker)
    expected_count = max(expected_default, len(expected))
    if expected_count <= 0:
        expected_count = expected_default
    marker_score = _ratio(len(covered), expected_count)
    length_score = _ratio(_effective_chars(problem_text), 1200)
    return max(marker_score, 0.6 * length_score)


def _shingles(text: str, width: int = 2) -> set[str]:
    """Return compact character shingles for rough echo/similarity checks."""
    compact = re.sub(r"\s+", "", _strip_page_markers(text or ""))
    if len(compact) < width:
        return set()
    return {compact[index : index + width] for index in range(len(compact) - width + 1)}


def _conclusion_echo_rate(problem_text: str, result_text: str, evaluation_text: str) -> float:
    """Estimate whether results/conclusions echo the problem statement."""
    target = result_text + "\n" + evaluation_text
    problem_shingles = _shingles(problem_text)
    target_shingles = _shingles(target)
    if not problem_shingles or not target_shingles:
        keyword_score = _ratio(_count_keywords(target, RESULT_ECHO_KEYWORDS), max(1, len(RESULT_ECHO_KEYWORDS)))
        return keyword_score
    jaccard = len(problem_shingles & target_shingles) / max(1, len(problem_shingles | target_shingles))
    keyword_score = _ratio(_count_keywords(target, RESULT_ECHO_KEYWORDS), max(1, len(RESULT_ECHO_KEYWORDS)))
    return _clip01(0.7 * jaccard + 0.3 * keyword_score)


def _reference_norm_rate(references_text: str, full_text: str) -> float:
    """Estimate reference list normalization."""
    source = references_text if _effective_chars(references_text) > 0 else full_text
    entries = re.findall(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.、])", source)
    if not entries:
        return 0.0
    year_hits = re.findall(r"(?:19|20)\d{2}", source)
    bracket_hits = re.findall(r"\[\d+\]", source)
    normalized_hits = max(len(bracket_hits), min(len(entries), len(year_hits)))
    return _ratio(normalized_hits, len(entries))


def _citation_norm_rate(full_text: str, references_text: str) -> float:
    """Estimate in-text citation normalization against reference count."""
    citations = re.findall(r"\[\d+\]", full_text or "")
    reference_entries = re.findall(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.、])", references_text or "")
    if not reference_entries:
        return 0.0
    return _ratio(len(set(citations)), len(reference_entries))


def _appendix_code_presence(appendix_text: str) -> float:
    """Detect whether appendix code exists."""
    return float(_count_keywords(appendix_text, CODE_KEYWORDS) > 0)


def _logic_connectives(config: dict[str, Any] | None) -> list[str]:
    """Return configured and default logic connective words."""
    configured = []
    if config:
        configured = list(config.get("logic_connectives") or [])
    return sorted(set(configured + DEFAULT_LOGIC_CONNECTIVES))


def _expected_problem_default(config: dict[str, Any] | None) -> int:
    """Read expected problem count from config."""
    if not config:
        return 3
    try:
        return int((config.get("feature_extraction") or {}).get("expected_problem_default", 3))
    except (TypeError, ValueError):
        return 3


def extract_surface_features_for_paper(
    section_json_path: Path,
    text_path: Path,
    *,
    pdf_lookup: dict[str, dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    candidate_usage_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the 19 requested surface features for one Appendix 2 paper."""
    candidate_usage_rows = candidate_usage_rows if candidate_usage_rows is not None else []
    payload = _load_json(section_json_path)
    full_text = _read_text(text_path)
    paper_id = str(payload.get("paper_id", Path(section_json_path).stem))
    filename = str(payload.get("filename", f"{paper_id}.txt"))
    pdf_meta = (pdf_lookup or {}).get(paper_id, {})

    total_chars = _effective_chars(full_text)
    page_count = int(pdf_meta.get("pages") or _count_pages(full_text))

    abstract_chars = _weighted_section_chars(payload, "abstract", "abstract_ratio", candidate_usage_rows)
    section_credit = 0.0
    for report_key in CORE_REPORT_SECTIONS:
        if _section_exists(payload, report_key):
            section_credit += 1.0
        elif _candidate_exists(payload, report_key):
            section_credit += CANDIDATE_WEIGHT
            candidate_usage_rows.append(
                {
                    "paper_id": paper_id,
                    "filename": filename,
                    "candidate_section": report_key,
                    "feature_name": "section_coverage",
                    "candidate_weight": CANDIDATE_WEIGHT,
                    "used_chars": _effective_chars(_candidate_text(payload, report_key)),
                    "usage_note": "candidate contributes half credit to section coverage only",
                }
            )

    formula_rows = _formula_lines(full_text)
    symbols_text = _confirmed_section(payload, "symbols")
    model_text = (
        _confirmed_section(payload, "modeling")
        + "\n"
        + _confirmed_section(payload, "solution")
        + "\n"
        + _confirmed_section(payload, "problem_analysis")
    )
    result_text = _weighted_section_text(payload, "results", "figure_table_explanation_rate", candidate_usage_rows)
    result_text_for_echo = _weighted_section_text(payload, "results", "conclusion_echo_rate", candidate_usage_rows)
    problem_text = _confirmed_section(payload, "problem_statement") + "\n" + _confirmed_section(payload, "problem_analysis")
    evaluation_text = _confirmed_section(payload, "model_evaluation")
    references_text = _confirmed_section(payload, "references")
    appendix_text = _confirmed_section(payload, "appendix")

    figure_table_total, figure_table_numbered = _figure_table_counts(full_text)
    logic_count = _count_keywords(full_text, _logic_connectives(config))

    row = {
        "paper_id": paper_id,
        "filename": filename,
        "total_chars": total_chars,
        "page_count": page_count,
        "abstract_ratio": _ratio(abstract_chars, total_chars),
        "section_coverage": _ratio(section_credit, len(CORE_REPORT_SECTIONS)),
        "paragraph_balance": _paragraph_balance(full_text),
        "formula_density": _density(len(formula_rows), total_chars),
        "formula_explanation_rate": _formula_explanation_rate(full_text, formula_rows),
        "variable_definition_coverage": _variable_definition_coverage(full_text, symbols_text, formula_rows),
        "objective_constraint_completeness": _objective_constraint_completeness(model_text),
        "figure_table_density": _density(figure_table_total, total_chars),
        "figure_table_numbering_rate": _ratio(figure_table_numbered, figure_table_total),
        "figure_table_explanation_rate": _figure_table_explanation_rate(full_text, result_text),
        "cross_reference_rate": _cross_reference_rate(full_text),
        "logic_connective_density": _density(logic_count, total_chars),
        "problem_restatement_coverage": _problem_restatement_coverage(
            problem_text,
            full_text,
            expected_default=_expected_problem_default(config),
        ),
        "conclusion_echo_rate": _conclusion_echo_rate(problem_text, result_text_for_echo, evaluation_text),
        "reference_norm_rate": _reference_norm_rate(references_text, full_text),
        "appendix_code_presence": _appendix_code_presence(appendix_text),
        "citation_norm_rate": _citation_norm_rate(full_text, references_text),
    }
    return row


def normalize_surface_features(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Normalize the 19 surface features to [0, 1]."""
    normalized = raw_table.copy()
    for column in SURFACE_FEATURE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = np.nan
            continue
        series = pd.to_numeric(normalized[column], errors="coerce")
        if column in RATIO_FEATURES:
            normalized[column] = series.clip(lower=0, upper=1)
        elif column in MINMAX_FEATURES:
            valid = series.dropna()
            if valid.empty:
                normalized[column] = np.nan
            elif valid.max() == valid.min():
                normalized[column] = series.apply(lambda value: np.nan if pd.isna(value) else (1.0 if float(value) > 0 else 0.0))
            else:
                normalized[column] = (series - valid.min()) / (valid.max() - valid.min())
                normalized[column] = normalized[column].clip(lower=0, upper=1)
        else:
            normalized[column] = series
    return normalized


def extract_appendix2_surface_features(
    sections_dir: Path,
    text_dir: Path,
    pdf_report_path: Path,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract raw and normalized surface features for all Appendix 2 papers."""
    pdf_lookup = _load_pdf_lookup(pdf_report_path)
    candidate_usage_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    section_files = sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key)
    for section_file in section_files:
        text_path = Path(text_dir) / f"{section_file.stem}.txt"
        rows.append(
            extract_surface_features_for_paper(
                section_file,
                text_path,
                pdf_lookup=pdf_lookup,
                config=config,
                candidate_usage_rows=candidate_usage_rows,
            )
        )

    raw = pd.DataFrame(rows, columns=["paper_id", "filename", *SURFACE_FEATURE_COLUMNS])
    normalized = normalize_surface_features(raw)
    candidate_usage = pd.DataFrame(
        candidate_usage_rows,
        columns=[
            "paper_id",
            "filename",
            "candidate_section",
            "feature_name",
            "candidate_weight",
            "used_chars",
            "usage_note",
        ],
    )
    return raw, normalized, candidate_usage


def _load_weight_table(weight_path: Path) -> pd.DataFrame:
    """Load Problem 1 combined weights."""
    try:
        weights = pd.read_excel(weight_path, sheet_name="combined_weights")
    except ValueError:
        weights = pd.read_excel(weight_path)
    required = {"indicator", "combined_weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Weight workbook missing columns: {sorted(missing)}")
    weights = weights.copy()
    weights["combined_weight"] = pd.to_numeric(weights["combined_weight"], errors="coerce")
    weights = weights.dropna(subset=["indicator", "combined_weight"])
    total = float(weights["combined_weight"].sum())
    if total <= 0:
        raise ValueError("Combined weights must sum to a positive value.")
    weights["combined_weight"] = weights["combined_weight"] / total
    return weights


def _indicator_prefix(column: str) -> str:
    """Return indicator prefix such as I01 from an indicator name."""
    match = re.match(r"(I\d+)", str(column))
    return match.group(1) if match else str(column)


def _align_problem1_features(normalized_features: pd.DataFrame, weight_table: pd.DataFrame) -> pd.DataFrame:
    """Align Problem 1 feature columns to the weight workbook indicators."""
    matrix = pd.DataFrame(index=normalized_features.index)
    missing: list[str] = []
    for indicator in weight_table["indicator"].astype(str).tolist():
        if indicator in normalized_features.columns:
            matrix[indicator] = pd.to_numeric(normalized_features[indicator], errors="coerce")
            continue
        prefix = _indicator_prefix(indicator)
        candidates = [column for column in normalized_features.columns if _indicator_prefix(column) == prefix]
        if candidates:
            matrix[indicator] = pd.to_numeric(normalized_features[candidates[0]], errors="coerce")
        else:
            missing.append(indicator)
    if missing:
        raise ValueError(f"Problem 1 feature columns missing for TOPSIS: {missing}")
    return matrix


def _impute_problem1_matrix(matrix: pd.DataFrame, weight_table: pd.DataFrame) -> pd.DataFrame:
    """Impute missing Problem 1 features with sealed workbook imputation values."""
    matrix = matrix.copy()
    imputation_lookup = {}
    if "imputation_value" in weight_table.columns:
        imputation_lookup = dict(zip(weight_table["indicator"].astype(str), weight_table["imputation_value"]))
    for column in matrix.columns:
        missing_count = int(matrix[column].isna().sum())
        if missing_count:
            fallback = pd.to_numeric(matrix[column], errors="coerce").median()
            imputation_value = imputation_lookup.get(column, fallback)
            if pd.isna(imputation_value):
                imputation_value = 0.0
            matrix[column] = matrix[column].fillna(float(imputation_value))
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    return matrix


def build_weak_quality_labels(
    sections_dir: Path,
    text_dir: Path,
    weights_path: Path,
    *,
    output_path: Path | None = None,
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build weak supervision labels Q_i for Appendix 2 papers.

    The fallback used here is the sealed Problem 1 AHP-entropy-TOPSIS base
    score. BT calibration is intentionally not applied because Appendix 2 has
    no manually reviewed pairwise comparison input and Step 12 excludes BT.
    """
    logger = logger or logging.getLogger(LOGGER_NAME)
    rows: list[dict[str, Any]] = []
    for section_file in sorted(Path(sections_dir).glob("*.json"), key=_natural_sort_key):
        text_file = Path(text_dir) / f"{section_file.stem}.txt"
        try:
            rows.append(feature_extractor.extract_features(section_file, text_file=text_file, config=config))
        except Exception as exc:
            logger.exception("Problem1-style feature extraction failed for %s: %s", section_file.name, exc)
            payload = _load_json(section_file)
            row = {
                "paper_id": str(payload.get("paper_id", section_file.stem)),
                "filename": str(payload.get("filename", f"{section_file.stem}.txt")),
                "text_chars": _effective_chars(_read_text(text_file)),
            }
            for column in feature_extractor.FEATURE_COLUMNS:
                row[column] = np.nan
            rows.append(row)

    raw_problem1 = pd.DataFrame(rows)
    normalized_problem1 = feature_extractor.normalize_features(raw_problem1)
    weight_table = _load_weight_table(weights_path)
    matrix = _align_problem1_features(normalized_problem1, weight_table)
    matrix = _impute_problem1_matrix(matrix, weight_table)
    weight_series = pd.Series(
        weight_table["combined_weight"].to_numpy(dtype=float),
        index=weight_table["indicator"].astype(str).tolist(),
    )

    topsis_result = topsis.calculate_topsis_scores(matrix, weight_series)
    labels = pd.DataFrame(
        {
            "paper_id": normalized_problem1["paper_id"].astype(str).tolist(),
            "filename": normalized_problem1["filename"].astype(str).tolist(),
            "Q_label": topsis_result["S_base"].to_numpy(dtype=float),
            "Q_source": "Problem1 sealed evaluation system",
            "label_note": "weak-supervised label, not official ground truth",
            "score_method": "AHP-entropy-TOPSIS base (BT unavailable for Appendix2)",
            "score_confidence": "usable",
        }
    )
    labels = labels.sort_values("paper_id", key=lambda series: series.map(lambda value: _natural_sort_key(Path(str(value))))).reset_index(drop=True)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        labels.to_excel(output_path, index=False)
    logger.info(
        "Q labels built by AHP-entropy-TOPSIS base fallback; BT not applied for Appendix2. q_min=%.6f q_max=%.6f",
        float(labels["Q_label"].min()) if len(labels) else math.nan,
        float(labels["Q_label"].max()) if len(labels) else math.nan,
    )
    return labels


def map_score_to_grade(score: float, thresholds: dict[str, float]) -> str:
    """Map a continuous score to a coarse grade if needed by later steps."""
    if pd.isna(score):
        return "unknown"
    if score >= thresholds.get("excellent", 85):
        return "excellent"
    if score >= thresholds.get("good", 75):
        return "good"
    if score >= thresholds.get("medium", 65):
        return "medium"
    if score >= thresholds.get("pass", 55):
        return "pass"
    return "fail"


def run_appendix2_features_labels(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 12 and save all required outputs."""
    try:
        config = load_config(config_path)
    except RuntimeError:
        config = {}
    problem2_config = get_problem2_config(config_path)
    text_dir = resolve_project_path(problem2_config["appendix2_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem2_config["appendix2_intermediate_dir"])
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    refined_sections_dir = intermediate_dir / "sections_refined"
    original_sections_dir = intermediate_dir / "sections"
    sections_dir = refined_sections_dir if refined_sections_dir.is_dir() and list(refined_sections_dir.glob("*.json")) else original_sections_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_feature_label_logger(logs_dir / "appendix2_features_labels.log")
    logger.info("Starting Step 12 Appendix 2 surface features and Q labels")
    logger.info("Using sections_dir=%s", sections_dir)
    if sections_dir == original_sections_dir:
        logger.warning("Refined sections are unavailable; falling back to original sections.")

    pdf_report_path = tables_dir / "appendix2_pdf_parse_report.xlsx"
    raw_output_path = tables_dir / "appendix2_surface_features_raw.xlsx"
    normalized_output_path = tables_dir / "appendix2_surface_features_normalized.xlsx"
    q_output_path = tables_dir / "appendix2_q_labels.xlsx"
    features_with_q_path = tables_dir / "appendix2_features_with_q.xlsx"
    candidate_report_path = tables_dir / "appendix2_candidate_usage_report.xlsx"
    weights_path = resolve_project_path("output/tables/appendix1_weights_ahp_entropy.xlsx")

    raw_features, normalized_features, candidate_usage = extract_appendix2_surface_features(
        sections_dir=sections_dir,
        text_dir=text_dir,
        pdf_report_path=pdf_report_path,
        config=config,
    )
    raw_features.to_excel(raw_output_path, index=False)
    normalized_features.to_excel(normalized_output_path, index=False)
    candidate_usage.to_excel(candidate_report_path, index=False)

    missing_features = {
        column: int(normalized_features[column].isna().sum())
        for column in SURFACE_FEATURE_COLUMNS
        if column in normalized_features.columns and int(normalized_features[column].isna().sum()) > 0
    }
    if missing_features:
        logger.warning("Missing surface feature values: %s", missing_features)
    else:
        logger.info("No missing surface feature values detected.")

    q_labels = build_weak_quality_labels(
        sections_dir=sections_dir,
        text_dir=text_dir,
        weights_path=weights_path,
        output_path=q_output_path,
        config=config,
        logger=logger,
    )
    features_with_q = normalized_features[["paper_id", "filename", *SURFACE_FEATURE_COLUMNS]].merge(
        q_labels[["paper_id", "Q_label", "Q_source", "label_note"]],
        on="paper_id",
        how="left",
    )
    features_with_q.to_excel(features_with_q_path, index=False)

    logger.info("Raw surface features saved: %s", raw_output_path)
    logger.info("Normalized surface features saved: %s", normalized_output_path)
    logger.info("Candidate usage report saved: %s", candidate_report_path)
    logger.info("Q labels saved: %s", q_output_path)
    logger.info("Features with Q saved: %s", features_with_q_path)
    logger.info(
        "Finished Step 12: papers=%s candidates_used=%s q_min=%.6f q_max=%.6f",
        len(raw_features),
        len(candidate_usage),
        float(q_labels["Q_label"].min()) if len(q_labels) else math.nan,
        float(q_labels["Q_label"].max()) if len(q_labels) else math.nan,
    )

    return {
        "raw_features": raw_features,
        "normalized_features": normalized_features,
        "q_labels": q_labels,
        "features_with_q": features_with_q,
        "candidate_usage": candidate_usage,
        "sections_dir": sections_dir,
        "raw_output_path": raw_output_path,
        "normalized_output_path": normalized_output_path,
        "q_output_path": q_output_path,
        "features_with_q_path": features_with_q_path,
        "candidate_report_path": candidate_report_path,
        "log_path": logs_dir / "appendix2_features_labels.log",
        "score_method": "AHP-entropy-TOPSIS base (BT unavailable for Appendix2)",
        "missing_features": missing_features,
    }
