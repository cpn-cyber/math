"""Rule-based section recognition and splitting for Step 3.

This module recognizes section headings from extracted text and writes one JSON
file per paper. It does not extract scoring features and does not fabricate any
missing content.
"""

from pathlib import Path
import json
import logging
import re
from typing import Any

import pandas as pd


LOGGER_NAME = "A_MAGE_R3.section_splitter"

CORE_SECTION_KEYS = [
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

REPORT_SECTION_KEYS = {
    "是否有摘要": "abstract",
    "是否有问题重述": "problem_statement",
    "是否有模型假设": "assumptions",
    "是否有符号说明": "symbols",
    "是否有模型建立": "model_building",
    "是否有结果分析": "results",
    "是否有参考文献": "references",
}

DEFAULT_SECTION_KEYWORDS = {
    "abstract": ["摘要", "摘 要"],
    "keywords": ["关键词", "关键字"],
    "problem_statement": ["问题重述", "问题的重述", "问题背景", "问题提出", "问题描述"],
    "problem_analysis": ["问题分析", "问题一的分析", "问题二的分析", "问题三的分析"],
    "assumptions": ["模型假设", "基本假设", "假设条件"],
    "symbols": ["符号说明", "主要符号", "变量说明"],
    "model_building": ["模型建立", "模型构建", "模型的建立", "模型建立与求解", "模型的建立与求解"],
    "model_solving": ["模型求解", "模型的求解", "求解过程", "模型建立与求解", "模型的建立与求解"],
    "results": ["结果分析", "求解结果", "模型结果", "结果与分析"],
    "sensitivity_analysis": ["灵敏度分析", "敏感性分析", "稳健性分析"],
    "error_analysis": ["误差分析", "残差分析"],
    "model_evaluation": ["模型评价", "模型优缺点", "模型推广", "模型的评价"],
    "references": ["参考文献", "参考资料"],
    "appendix": ["附录"],
}


def setup_section_split_logger(log_path: Path) -> logging.Logger:
    """Configure and return the Step 3 logger."""
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
    """Return the section splitter logger."""
    return logging.getLogger(LOGGER_NAME)


def _normalize_title(text: str) -> str:
    """Normalize a candidate heading for robust keyword matching."""
    text = re.sub(r"\[PAGE\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    text = text.replace("．", ".").replace("：", ":")
    return text.strip()


def _strip_heading_prefix(text: str) -> str:
    """Remove common Chinese/arithmetic heading prefixes."""
    normalized = _normalize_title(text)
    patterns = [
        r"^[一二三四五六七八九十]+[、.]\s*",
        r"^第[一二三四五六七八九十0-9]+[章节部分][、.：:]?\s*",
        r"^[（(][一二三四五六七八九十0-9]+[）)]\s*",
        r"^\d+(?:\.\d+)*[、.．]?\s*",
    ]
    for pattern in patterns:
        normalized = re.sub(pattern, "", normalized)
    return normalized


def _looks_like_toc_line(line: str) -> bool:
    """Return whether a line looks like a table-of-contents entry."""
    stripped = line.strip()
    if "…" in stripped or "..." in stripped or "··" in stripped:
        return True
    return bool(re.search(r"\.{3,}\s*\d+\s*$", stripped))


def _has_heading_prefix(line: str) -> bool:
    """Return whether a line starts with a supported heading prefix."""
    stripped = line.strip()
    return bool(
        re.match(r"^[一二三四五六七八九十]+[、.．]", stripped)
        or re.match(r"^第[一二三四五六七八九十0-9]+[章节部分][、.．：:]?", stripped)
        or re.match(r"^[（(][一二三四五六七八九十0-9]+[）)]", stripped)
        or re.match(r"^\d+(?:\.\d+)*[、.．]?\s+\S+", stripped)
    )


def _is_heading_candidate(line: str, keyword: str) -> bool:
    """Check whether a keyword occurrence should be treated as a heading."""
    stripped = line.strip()
    if not stripped or stripped.startswith("[PAGE"):
        return False
    if _looks_like_toc_line(stripped):
        return False

    normalized = _normalize_title(stripped)
    title_body = _strip_heading_prefix(stripped)
    keyword_norm = _normalize_title(keyword)

    if keyword_norm not in normalized:
        return False
    if len(normalized) > 60:
        return False
    if stripped.startswith(("针对", "本文", "通过", "根据", "其中", "因此", "于是", "由此")):
        return False
    if re.search(r"[，；。！？]", stripped) and not _has_heading_prefix(stripped):
        return False

    return (
        _has_heading_prefix(stripped)
        or title_body.startswith(keyword_norm)
        or normalized.startswith(keyword_norm)
        or len(normalized) <= 24
    )


def _match_section_key(line: str, section_keywords: dict[str, list[str]]) -> str | None:
    """Map a heading line to a canonical section key."""
    normalized = _normalize_title(line)
    has_build = "建立" in normalized or "构建" in normalized
    has_solve = "求解" in normalized

    # Combined headings should be recognized as model_building first. The same
    # real text can later be mirrored into model_solving if no separate solving
    # section exists.
    if "模型" in normalized and has_build and has_solve:
        return "model_building"

    priority = [
        "abstract",
        "keywords",
        "problem_statement",
        "problem_analysis",
        "assumptions",
        "symbols",
        "sensitivity_analysis",
        "error_analysis",
        "model_evaluation",
        "references",
        "appendix",
        "results",
        "model_solving",
        "model_building",
    ]

    for section_key in priority:
        for keyword in section_keywords.get(section_key, []):
            if _is_heading_candidate(line, keyword):
                return section_key
    return None


def _line_positions(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, line_without_newline) records."""
    records: list[tuple[int, int, str]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        start = cursor
        end = cursor + len(raw_line)
        cursor = end
        records.append((start, end, raw_line.rstrip("\r\n")))
    return records


def _detect_headings(text: str, section_keywords: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Detect section headings and their character offsets."""
    headings: list[dict[str, Any]] = []
    previous_key: str | None = None

    for start, end, line in _line_positions(text):
        section_key = _match_section_key(line, section_keywords)
        if section_key is None:
            continue
        if section_key == previous_key:
            continue

        headings.append(
            {
                "section_key": section_key,
                "start": start,
                "end": end,
                "title": line.strip(),
            }
        )
        previous_key = section_key
    return headings


def _empty_sections() -> dict[str, str]:
    """Create an empty section dictionary."""
    keys = list(dict.fromkeys(CORE_SECTION_KEYS + ["keywords", "problem_analysis"]))
    return {key: "" for key in keys}


def split_sections(text: str, section_keywords: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Split extracted text into recognized sections.

    Parameters
    ----------
    text:
        Page-marked text produced by Step 2.
    section_keywords:
        Optional keyword mapping. If omitted, built-in defaults are used.
    """
    section_keywords = section_keywords or DEFAULT_SECTION_KEYWORDS
    sections = _empty_sections()
    headings = _detect_headings(text, section_keywords)

    for index, heading in enumerate(headings):
        section_key = heading["section_key"]
        start = heading["start"]
        end = headings[index + 1]["start"] if index + 1 < len(headings) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue

        if sections.get(section_key):
            sections[section_key] = f"{sections[section_key]}\n\n{chunk}"
        else:
            sections[section_key] = chunk

        title_norm = _normalize_title(heading["title"])
        if (
            section_key == "model_building"
            and "求解" in title_norm
            and not sections.get("model_solving")
        ):
            sections["model_solving"] = chunk

    missing_sections = [key for key in CORE_SECTION_KEYS if not sections.get(key, "").strip()]
    return {
        "sections": sections,
        "missing_sections": missing_sections,
        "detected_headings": headings,
    }


def _build_report_row(paper_id: str, filename: str, split_result: dict[str, Any]) -> dict[str, Any]:
    """Build one Excel report row."""
    sections = split_result["sections"]
    row = {
        "paper_id": paper_id,
        "filename": filename,
    }
    for column_name, section_key in REPORT_SECTION_KEYS.items():
        row[column_name] = bool(sections.get(section_key, "").strip())
    row["缺失章节数量"] = len(split_result["missing_sections"])
    return row


def split_text_file(
    text_path: Path,
    output_dir: Path,
    section_keywords: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Split one text file and save its JSON output."""
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = text_path.read_text(encoding="utf-8", errors="ignore")
    paper_id = text_path.stem
    split_result = split_sections(text, section_keywords=section_keywords)
    payload = {
        "paper_id": paper_id,
        "filename": text_path.name,
        "sections": split_result["sections"],
        "missing_sections": split_result["missing_sections"],
    }

    output_json_path = output_dir / f"{paper_id}.json"
    output_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row = _build_report_row(paper_id, text_path.name, split_result)
    return {
        "json_path": output_json_path,
        "report_row": row,
        "detected_heading_count": len(split_result["detected_headings"]),
        "missing_sections": split_result["missing_sections"],
    }


def split_all_texts(
    input_dir: Path,
    output_dir: Path,
    section_keywords: dict[str, list[str]] | None = None,
    report_path: Path | None = None,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Split all extracted text files and optionally save an Excel report."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_section_split_logger(log_path) if log_path is not None else _get_logger()
    logger.info("Starting section splitting: input_dir=%s output_dir=%s", input_dir, output_dir)

    rows: list[dict[str, Any]] = []
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
    else:
        for text_path in sorted(input_dir.glob("*.txt")):
            try:
                result = split_text_file(
                    text_path=text_path,
                    output_dir=output_dir,
                    section_keywords=section_keywords,
                )
                rows.append(result["report_row"])
                if result["detected_heading_count"] == 0:
                    logger.warning("%s | no section headings detected", text_path.name)
                if result["missing_sections"]:
                    logger.warning(
                        "%s | missing sections: %s",
                        text_path.name,
                        ",".join(result["missing_sections"]),
                    )
                logger.info(
                    "%s split successfully: headings=%s missing=%s json=%s",
                    text_path.name,
                    result["detected_heading_count"],
                    len(result["missing_sections"]),
                    result["json_path"],
                )
            except Exception as exc:
                logger.exception("%s | section splitting failed: %s", text_path.name, exc)
                row = {"paper_id": text_path.stem, "filename": text_path.name}
                for column_name in REPORT_SECTION_KEYS:
                    row[column_name] = False
                row["缺失章节数量"] = len(CORE_SECTION_KEYS)
                rows.append(row)

    report_columns = ["paper_id", "filename", *REPORT_SECTION_KEYS.keys(), "缺失章节数量"]
    report = pd.DataFrame(rows, columns=report_columns)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_excel(report_path, index=False)
        logger.info("Section split report saved: %s", report_path)

    logger.info("Finished section splitting: total=%s", len(report))
    return report


def split_sections_file(text_path: Path, output_dir: Path) -> Path:
    """Path-based wrapper for one text file."""
    result = split_text_file(Path(text_path), Path(output_dir))
    return Path(result["json_path"])


def split_sections_batch(input_dir: Path, output_dir: Path) -> list[Path]:
    """Backward-compatible wrapper for splitting all text files."""
    split_all_texts(Path(input_dir), Path(output_dir))
    return sorted(Path(output_dir).glob("*.json"))
