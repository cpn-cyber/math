"""Shared helpers for the Problem 2 pipeline.

Step 11 reuses the Problem 1 PDF parser and section splitter. It only extracts
Appendix 2 text and sections; feature construction and modeling are left to
later steps.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from modules import pdf_parser, section_splitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER_NAME = "A_MAGE_R3.problem2.appendix2_parse_sections"

DEFAULT_PROBLEM2_CONFIG: dict[str, Any] = {
    "appendix2_papers_dir": "data/appendix2_papers",
    "appendix2_extracted_text_dir": "data/appendix2_extracted_text",
    "appendix2_intermediate_dir": "data/appendix2_intermediate",
    "problem2_output_tables_dir": "output/problem2_tables",
    "problem2_output_charts_dir": "output/problem2_charts",
    "problem2_output_logs_dir": "output/problem2_logs",
    "robust_scale_epsilon": 0.000001,
    "qaf_eta_grid": [0.05, 0.10, 0.15],
    "qaf_clip_min": 0.90,
    "qaf_clip_max": 1.10,
    "pls_components_grid": [1, 2],
    "bootstrap_B": 1000,
}

DEFAULT_PDF_CONFIG: dict[str, Any] = {
    "ocr_render_dpi": 300,
    "ocr_languages": ["chi_sim", "eng"],
    "ocr_engine": "tesseract_or_fallback",
    "tesseract_path": "C:/Program Files/Tesseract-OCR/tesseract.exe",
    "tesseract_cmd": "",
    "ocr_psm": 6,
    "ocr_timeout_seconds": 180,
    "scanned_pdf_text_threshold": 100,
}

PDF_PARSE_COLUMNS = [
    "paper_id",
    "filename",
    "pages",
    "text_chars",
    "ocr_used",
    "parse_success",
    "blank_pages",
    "error",
]

SECTION_REPORT_KEYS = [
    "abstract",
    "problem_statement",
    "assumptions",
    "symbols",
    "modeling",
    "solution",
    "results",
    "sensitivity_analysis",
    "error_analysis",
    "model_evaluation",
    "references",
    "appendix",
]

SECTION_KEY_MAP = {
    "abstract": "abstract",
    "problem_statement": "problem_statement",
    "assumptions": "assumptions",
    "symbols": "symbols",
    "modeling": "model_building",
    "solution": "model_solving",
    "results": "results",
    "sensitivity_analysis": "sensitivity_analysis",
    "error_analysis": "error_analysis",
    "model_evaluation": "model_evaluation",
    "references": "references",
    "appendix": "appendix",
}


PROBLEM2_REQUIRED_DIRS = [
    "data/appendix2_papers",
    "data/appendix2_extracted_text",
    "data/appendix2_intermediate",
    "output/problem2_tables",
    "output/problem2_charts",
    "output/problem2_logs",
    "paper_sections/problem2",
]


PROBLEM2_REQUIRED_MODULES = [
    "modules/appendix2_pipeline.py",
    "modules/quality_label_builder.py",
    "modules/deep_quality_features.py",
    "modules/robust_preprocessing.py",
    "modules/grey_relation.py",
    "modules/pls_vip_model.py",
    "modules/quality_adjustment_factor.py",
    "modules/small_sample_validation.py",
    "modules/pairwise_ranking_check.py",
    "modules/problem2_report_generator.py",
]


PROBLEM2_REQUIRED_SCRIPTS = [
    "scripts/run_step10_problem2_setup.py",
    "scripts/run_step11_appendix2_parse_sections.py",
    "scripts/run_step12_appendix2_features_labels.py",
    "scripts/run_step13_deep_quality_features.py",
    "scripts/run_step14_robust_correlation.py",
    "scripts/run_step15_grey_key_index.py",
    "scripts/run_step16_pls_vip_prediction.py",
    "scripts/run_step17_quality_adjustment.py",
    "scripts/run_step18_small_sample_validation.py",
    "scripts/run_step19_pairwise_ranking_check.py",
    "scripts/run_step20_problem2_final_audit.py",
    "scripts/run_step21_problem2_draft.py",
]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project config."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config.yaml. Install requirements.txt first.") from exc
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_problem2_config(config_path: Path | None = None) -> dict[str, Any]:
    """Return the problem2 config section."""
    merged = dict(DEFAULT_PROBLEM2_CONFIG)
    try:
        config = load_config(config_path)
    except RuntimeError:
        return merged
    merged.update(dict(config.get("problem2", {})))
    return merged


def get_pdf_parser_config(config_path: Path | None = None) -> dict[str, Any]:
    """Return PDF parser config with safe defaults."""
    merged = dict(DEFAULT_PDF_CONFIG)
    try:
        config = load_config(config_path)
    except RuntimeError:
        return merged
    merged.update(dict(config.get("pdf_parser", {})))
    return merged


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_problem2_directories() -> list[Path]:
    """Create required Problem 2 directories and return their paths."""
    paths = [resolve_project_path(item) for item in PROBLEM2_REQUIRED_DIRS]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_appendix2_logger(log_path: Path) -> logging.Logger:
    """Configure one Step 11 log file for the pipeline and reused modules."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    for logger_name in [
        LOGGER_NAME,
        pdf_parser.LOGGER_NAME,
        pdf_parser.OCR_LOGGER_NAME,
        section_splitter.LOGGER_NAME,
    ]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)
    return logging.getLogger(LOGGER_NAME)


def _natural_sort_key(path: Path) -> list[Any]:
    """Sort filenames such as 2-1.pdf ... 2-10.pdf naturally."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def _count_non_whitespace(text: str) -> int:
    """Count non-whitespace characters."""
    return len(re.sub(r"\s+", "", text or ""))


def _count_text_file_chars(text_path: Path) -> int:
    """Count effective characters in a text file, excluding page markers."""
    if not Path(text_path).exists():
        return 0
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"\[PAGE\s+\d+\]", "", text, flags=re.IGNORECASE)
    return _count_non_whitespace(text)


def _to_bool(value: Any) -> bool:
    """Convert common report values to bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是", "成功"}


def _to_int(value: Any, default: int = 0) -> int:
    """Convert report values to int safely."""
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _append_error(existing: str, extra: str) -> str:
    """Join parse and OCR errors without dropping either source."""
    parts = [str(item).strip() for item in [existing, extra] if str(item or "").strip()]
    return " | ".join(parts)


def _paper_id_from_path(path: Path) -> str:
    """Return canonical paper id from a PDF or text path."""
    return Path(path).stem


def sync_appendix2_pdfs_if_needed(input_dir: Path, logger: logging.Logger) -> int:
    """Copy PDFs from the workspace attachment folder if the configured input is empty."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    if list(input_dir.glob("*.pdf")):
        return 0

    source_dir = PROJECT_ROOT.parent / "附件2"
    if not source_dir.is_dir():
        logger.warning("Appendix 2 input is empty and fallback source is missing: %s", source_dir)
        return 0

    copied = 0
    for pdf_path in sorted(source_dir.glob("*.pdf"), key=_natural_sort_key):
        shutil.copy2(pdf_path, input_dir / pdf_path.name)
        copied += 1
    logger.info("Synced %s Appendix 2 PDFs from %s to %s", copied, source_dir, input_dir)
    return copied


def parse_appendix2_pdf(
    pdf_path: Path,
    output_txt_path: Path,
    *,
    pdf_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Parse one Appendix 2 PDF and use OCR if the text layer is too sparse."""
    pdf_config = pdf_config or get_pdf_parser_config()
    logger = logger or logging.getLogger(LOGGER_NAME)
    pdf_path = Path(pdf_path)
    output_txt_path = Path(output_txt_path)
    threshold = _to_int(pdf_config.get("scanned_pdf_text_threshold"), default=100)

    row = pdf_parser.parse_pdf_to_text(pdf_path, output_txt_path)
    pages = _to_int(row.get("页数"))
    text_chars = _to_int(row.get("字数"))
    blank_pages = _to_int(row.get("空白页数量"))
    parse_success = _to_bool(row.get("是否解析成功"))
    error = str(row.get("错误信息") or "")
    ocr_used = False

    needs_ocr = (not parse_success) or text_chars < threshold or (pages > 0 and blank_pages >= pages)
    if needs_ocr:
        ocr_used = True
        logger.info(
            "%s needs OCR fallback: parse_success=%s chars=%s blank_pages=%s pages=%s threshold=%s",
            pdf_path.name,
            parse_success,
            text_chars,
            blank_pages,
            pages,
            threshold,
        )
        ocr_result = pdf_parser.ocr_pdf_to_text(
            pdf_path=pdf_path,
            output_txt_path=output_txt_path,
            render_dpi=_to_int(pdf_config.get("ocr_render_dpi"), default=300),
            languages=list(pdf_config.get("ocr_languages") or ["chi_sim", "eng"]),
            preferred_engine=str(pdf_config.get("ocr_engine") or "tesseract_or_fallback"),
            tesseract_cmd=str(
                pdf_config.get("tesseract_path")
                or pdf_config.get("tesseract_cmd")
                or ""
            )
            or None,
            psm=_to_int(pdf_config.get("ocr_psm"), default=6),
            timeout_seconds=_to_int(pdf_config.get("ocr_timeout_seconds"), default=180),
        )
        summary = ocr_result.get("summary", {})
        for page_row in ocr_result.get("pages", []):
            logger.info(
                "%s page=%s OCR chars=%s success=%s error=%s",
                pdf_path.name,
                page_row.get("页码"),
                page_row.get("OCR识别字符数"),
                page_row.get("是否识别成功"),
                page_row.get("错误信息") or "",
            )

        ocr_success = _to_bool(summary.get("是否OCR成功"))
        ocr_error = str(summary.get("错误信息") or "")
        if ocr_success:
            pages = _to_int(summary.get("页数"), default=pages)
            text_chars = _to_int(summary.get("OCR总字数"))
            blank_pages = _to_int(summary.get("OCR失败页数"))
            parse_success = text_chars > 0
            error = _append_error(error, ocr_error)
        else:
            text_chars = _count_text_file_chars(output_txt_path)
            parse_success = parse_success and text_chars > 0
            error = _append_error(error, ocr_error or "OCR fallback failed")
            logger.error("%s OCR fallback failed: %s", pdf_path.name, error)

    final_chars = _count_text_file_chars(output_txt_path)
    if final_chars != text_chars:
        text_chars = final_chars
    parse_success = bool(parse_success and text_chars > 0)

    return {
        "paper_id": _paper_id_from_path(pdf_path),
        "filename": pdf_path.name,
        "pages": pages,
        "text_chars": text_chars,
        "ocr_used": ocr_used,
        "parse_success": parse_success,
        "blank_pages": blank_pages,
        "error": error,
    }


def parse_appendix2_pdfs(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    pdf_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Parse all Appendix 2 PDFs and save the required parse report."""
    logger = logger or logging.getLogger(LOGGER_NAME)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"), key=_natural_sort_key)
    if not pdf_files:
        logger.error("No Appendix 2 PDFs found in %s", input_dir)
        report = pd.DataFrame(columns=PDF_PARSE_COLUMNS)
        report.to_excel(report_path, index=False)
        return report

    rows: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        try:
            rows.append(
                parse_appendix2_pdf(
                    pdf_path=pdf_path,
                    output_txt_path=output_dir / f"{pdf_path.stem}.txt",
                    pdf_config=pdf_config,
                    logger=logger,
                )
            )
            logger.info("%s parsed for Appendix 2", pdf_path.name)
        except Exception as exc:  # pragma: no cover - defensive batch isolation
            logger.exception("%s parse failed: %s", pdf_path.name, exc)
            rows.append(
                {
                    "paper_id": _paper_id_from_path(pdf_path),
                    "filename": pdf_path.name,
                    "pages": 0,
                    "text_chars": 0,
                    "ocr_used": False,
                    "parse_success": False,
                    "blank_pages": 0,
                    "error": str(exc),
                }
            )

    report = pd.DataFrame(rows, columns=PDF_PARSE_COLUMNS)
    report.to_excel(report_path, index=False)
    logger.info("Appendix 2 PDF parse report saved: %s", report_path)
    return report


def split_appendix2_sections(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Split all Appendix 2 extracted text files into section JSON outputs."""
    logger = logger or logging.getLogger(LOGGER_NAME)
    input_dir = Path(input_dir)
    sections_dir = Path(output_dir) / "sections"
    report_path = Path(report_path)
    sections_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    text_files = sorted(input_dir.glob("*.txt"), key=_natural_sort_key)
    if not text_files:
        logger.error("No Appendix 2 text files found in %s", input_dir)
        report = pd.DataFrame(columns=["paper_id", "filename", *SECTION_REPORT_KEYS, "missing_sections_count"])
        report.to_excel(report_path, index=False)
        return report

    rows: list[dict[str, Any]] = []
    for text_path in text_files:
        try:
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            split_result = section_splitter.split_sections(text)
            paper_id = _paper_id_from_path(text_path)
            payload = {
                "paper_id": paper_id,
                "filename": text_path.name,
                "sections": split_result["sections"],
                "missing_sections": split_result["missing_sections"],
            }
            json_path = sections_dir / f"{paper_id}.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            row: dict[str, Any] = {"paper_id": paper_id, "filename": text_path.name}
            for report_key in SECTION_REPORT_KEYS:
                section_key = SECTION_KEY_MAP[report_key]
                row[report_key] = bool(split_result["sections"].get(section_key, "").strip())
            row["missing_sections_count"] = len(split_result["missing_sections"])
            rows.append(row)

            if not split_result["detected_headings"]:
                logger.warning("%s has no detected section headings", text_path.name)
            if split_result["missing_sections"]:
                logger.warning(
                    "%s missing sections: %s",
                    text_path.name,
                    ",".join(split_result["missing_sections"]),
                )
            logger.info(
                "%s section split done: headings=%s missing=%s json=%s",
                text_path.name,
                len(split_result["detected_headings"]),
                len(split_result["missing_sections"]),
                json_path,
            )
        except Exception as exc:  # pragma: no cover - defensive batch isolation
            logger.exception("%s section split failed: %s", text_path.name, exc)
            row = {"paper_id": _paper_id_from_path(text_path), "filename": text_path.name}
            for report_key in SECTION_REPORT_KEYS:
                row[report_key] = False
            row["missing_sections_count"] = len(SECTION_REPORT_KEYS)
            rows.append(row)

    report = pd.DataFrame(rows, columns=["paper_id", "filename", *SECTION_REPORT_KEYS, "missing_sections_count"])
    report.to_excel(report_path, index=False)
    logger.info("Appendix 2 section split report saved: %s", report_path)
    return report


def run_appendix2_parse_sections(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 11: PDF extraction, OCR fallback, and section splitting."""
    problem2_config = get_problem2_config(config_path)
    pdf_config = get_pdf_parser_config(config_path)

    input_dir = resolve_project_path(problem2_config["appendix2_papers_dir"])
    extracted_text_dir = resolve_project_path(problem2_config["appendix2_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem2_config["appendix2_intermediate_dir"])
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])

    ensure_problem2_directories()
    logger = setup_appendix2_logger(logs_dir / "appendix2_parse_sections.log")
    logger.info("Starting Step 11 Appendix 2 PDF parse and section split")
    copied = sync_appendix2_pdfs_if_needed(input_dir, logger)

    pdf_report_path = tables_dir / "appendix2_pdf_parse_report.xlsx"
    section_report_path = tables_dir / "appendix2_section_split_report.xlsx"

    pdf_report = parse_appendix2_pdfs(
        input_dir=input_dir,
        output_dir=extracted_text_dir,
        report_path=pdf_report_path,
        pdf_config=pdf_config,
        logger=logger,
    )
    section_report = split_appendix2_sections(
        input_dir=extracted_text_dir,
        output_dir=intermediate_dir,
        report_path=section_report_path,
        logger=logger,
    )
    logger.info(
        "Finished Step 11: pdf_total=%s parse_success=%s ocr_used=%s section_total=%s",
        len(pdf_report),
        int(pdf_report["parse_success"].sum()) if len(pdf_report) else 0,
        int(pdf_report["ocr_used"].sum()) if len(pdf_report) else 0,
        len(section_report),
    )
    return {
        "pdf_report": pdf_report,
        "section_report": section_report,
        "copied_pdfs": copied,
        "pdf_report_path": pdf_report_path,
        "section_report_path": section_report_path,
        "log_path": logs_dir / "appendix2_parse_sections.log",
        "extracted_text_dir": extracted_text_dir,
        "sections_dir": intermediate_dir / "sections",
    }


STEP11B_LOGGER_NAME = "A_MAGE_R3.problem2.appendix2_section_refine"

REFINE_FOCUS_SECTIONS = ["abstract", "assumptions", "modeling", "solution", "results", "references"]

REFINE_SECTION_TO_JSON_KEY = {
    "abstract": "abstract",
    "problem_statement": "problem_statement",
    "assumptions": "assumptions",
    "symbols": "symbols",
    "modeling": "model_building",
    "solution": "model_solving",
    "results": "results",
    "sensitivity_analysis": "sensitivity_analysis",
    "error_analysis": "error_analysis",
    "model_evaluation": "model_evaluation",
    "references": "references",
    "appendix": "appendix",
}

REFINE_JSON_TO_REPORT_KEY = {
    "abstract": "abstract",
    "problem_statement": "problem_statement",
    "assumptions": "assumptions",
    "symbols": "symbols",
    "model_building": "modeling",
    "model_solving": "solution",
    "results": "results",
    "sensitivity_analysis": "sensitivity_analysis",
    "error_analysis": "error_analysis",
    "model_evaluation": "model_evaluation",
    "references": "references",
    "appendix": "appendix",
}

REFINE_KEYWORDS = {
    "abstract": ["摘要", "摘 要", "Abstract", "Summary", "内容摘要"],
    "assumptions": ["模型假设", "基本假设", "假设条件", "问题假设", "假设与说明", "模型的假设"],
    "modeling": ["模型建立", "模型的建立", "模型构建", "问题一模型", "问题二模型", "数学模型", "模型分析", "模型准备"],
    "solution": ["模型求解", "求解过程", "算法设计", "求解方法", "计算过程", "问题求解"],
    "results": ["结果分析", "结果", "求解结果", "实验结果", "数值结果", "计算结果", "结果检验", "结果展示", "模型结果"],
    "references": ["参考文献", "References", "文献"],
}

RESULTS_CANDIDATE_WORDS = ["结果", "计算", "得到", "如表", "如图", "最终", "预测值", "评价值"]
ABSTRACT_CANDIDATE_WORDS = ["本文", "针对", "建立", "结果", "模型"]


def setup_appendix2_refine_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 11B section-refine logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(STEP11B_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def _normalize_for_match(text: str) -> str:
    """Normalize Chinese/English text for heading and keyword matching."""
    text = re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.IGNORECASE)
    text = text.replace("：", ":").replace("；", ";").replace("，", ",")
    return re.sub(r"\s+", "", text).lower()


def _count_text_chars(text: str) -> int:
    """Count effective characters in section or full text."""
    return len(_normalize_for_match(text))


def _line_is_title_like(line: str) -> bool:
    """Return whether a line looks like a section heading."""
    stripped = line.strip()
    if not stripped or stripped.startswith("[PAGE"):
        return False
    compact = _normalize_for_match(stripped)
    if not compact:
        return False
    if len(compact) <= 42 and not re.search(r"[。！？!?；;]", stripped):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s*[、.．]?\s*\S+", stripped):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*\S+", stripped):
        return True
    if re.match(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[）)]\s*\S+", stripped):
        return True
    if re.match(r"^第\s*[一二三四五六七八九十\d]+\s*[章节部分]\s*\S*", stripped):
        return True
    return False


def _strip_refine_heading_prefix(line: str) -> str:
    """Remove common heading prefixes for keyword matching."""
    stripped = line.strip()
    patterns = [
        r"^\d+(?:\.\d+)*\s*[、.．]?\s*",
        r"^[一二三四五六七八九十]+[、.．]\s*",
        r"^[（(]\s*[一二三四五六七八九十\d]+\s*[）)]\s*",
        r"^第\s*[一二三四五六七八九十\d]+\s*[章节部分]\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped)
    return stripped.strip(" ：:。.-—\t")


def _heading_candidates(lines: list[str], index: int) -> list[tuple[str, int]]:
    """Return one-line and adjacent short-line heading candidates."""
    current = lines[index].strip()
    candidates = [(current, index)]
    if index + 1 < len(lines):
        nxt = lines[index + 1].strip()
        if len(_normalize_for_match(current)) <= 4 and len(_normalize_for_match(nxt)) <= 4:
            candidates.append((current + nxt, index + 1))
    return candidates


def _match_refine_section(candidate: str) -> tuple[str, str] | None:
    """Match a heading candidate to one refined core section."""
    if not _line_is_title_like(candidate):
        return None
    body = _normalize_for_match(_strip_refine_heading_prefix(candidate))
    full = _normalize_for_match(candidate)
    for section in REFINE_FOCUS_SECTIONS:
        for keyword in REFINE_KEYWORDS[section]:
            normalized_keyword = _normalize_for_match(keyword)
            if not normalized_keyword:
                continue
            if section == "results" and normalized_keyword == "结果":
                if not (body == normalized_keyword or body.startswith(normalized_keyword) or body.endswith("结果")):
                    continue
            if body.startswith(normalized_keyword) or full.startswith(normalized_keyword) or full == normalized_keyword:
                return section, keyword
    return None


def _is_generic_heading(line: str) -> bool:
    """Detect a line that can delimit a candidate section."""
    stripped = line.strip()
    if not _line_is_title_like(stripped):
        return False
    compact = _normalize_for_match(stripped)
    if compact.startswith(("问题1", "问题2", "问题3", "问题4", "关键词", "参考文献", "附录")):
        return True
    if re.match(r"^\d+(?:\.\d+)*", stripped):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、.．]", stripped):
        return True
    if re.match(r"^[（(]\s*[一二三四五六七八九十\d]+\s*[）)]", stripped):
        return True
    if re.match(r"^第\s*[一二三四五六七八九十\d]+\s*[章节部分]", stripped):
        return True
    return False


def _detect_refine_headings(lines: list[str]) -> tuple[list[dict[str, Any]], list[int]]:
    """Detect enhanced section headings and generic delimiter headings."""
    headings: list[dict[str, Any]] = []
    generic_heading_lines: set[int] = set()
    skip_until = -1
    for index, line in enumerate(lines):
        if _is_generic_heading(line):
            generic_heading_lines.add(index)
        if index <= skip_until:
            continue
        for candidate, end_index in _heading_candidates(lines, index):
            matched = _match_refine_section(candidate)
            if matched is None:
                continue
            section, keyword = matched
            headings.append(
                {
                    "section": section,
                    "keyword": keyword,
                    "line_index": index,
                    "heading_end_index": end_index,
                    "title": candidate,
                }
            )
            generic_heading_lines.add(index)
            skip_until = end_index
            break
    return headings, sorted(generic_heading_lines)


def _next_delimiter_line(start_line: int, delimiter_lines: list[int], total_lines: int) -> int:
    """Return the next generic heading line after start_line."""
    for line_index in delimiter_lines:
        if line_index > start_line:
            return line_index
    return total_lines


def _slice_text_lines(lines: list[str], start_line: int, end_line: int) -> str:
    """Slice a line interval into text."""
    start_line = max(0, start_line)
    end_line = min(len(lines), max(start_line, end_line))
    return "\n".join(lines[start_line:end_line]).strip()


def _load_appendix2_section_json(json_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Load one Step 11 section JSON with a safe fallback."""
    try:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("Failed to load section JSON %s: %s", json_path, exc)
        return {"paper_id": Path(json_path).stem, "filename": Path(json_path).name, "sections": {}, "missing_sections": []}


def _scan_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return keywords found in the full text."""
    normalized_text = _normalize_for_match(text)
    return [keyword for keyword in keywords if _normalize_for_match(keyword) in normalized_text]


def _find_abstract_candidate(lines: list[str], delimiter_lines: list[int]) -> dict[str, Any] | None:
    """Find a conservative abstract candidate near the beginning."""
    non_empty = [(i, line) for i, line in enumerate(lines[:80]) if line.strip() and not line.strip().startswith("[PAGE")]
    if not non_empty:
        return None
    # Start after title-ish metadata. The first line with abstract feature words is enough.
    start = None
    for i, line in non_empty:
        normalized = _normalize_for_match(line)
        if any(word in normalized for word in [_normalize_for_match(item) for item in ABSTRACT_CANDIDATE_WORDS]):
            start = i
            break
    if start is None:
        return None
    end = _next_delimiter_line(start, delimiter_lines, min(len(lines), start + 40))
    chunk = _slice_text_lines(lines, start, end)
    if not (80 <= _count_text_chars(chunk) <= 2500):
        return None
    matched = _scan_keywords(chunk, ABSTRACT_CANDIDATE_WORDS)
    if len(matched) < 2:
        return None
    return {"text": chunk, "matched_keywords": matched, "start_line": start + 1, "end_line": end, "note": "opening abstract-like paragraph"}


def _find_results_candidate(lines: list[str], delimiter_lines: list[int]) -> dict[str, Any] | None:
    """Find a results candidate when no explicit results heading is found."""
    best: dict[str, Any] | None = None
    normalized_words = [_normalize_for_match(item) for item in RESULTS_CANDIDATE_WORDS]
    for index, line in enumerate(lines):
        normalized = _normalize_for_match(line)
        if not normalized:
            continue
        hit_count = sum(1 for word in normalized_words if word in normalized)
        if hit_count < 2:
            continue
        end = _next_delimiter_line(index, delimiter_lines, min(len(lines), index + 30))
        chunk = _slice_text_lines(lines, index, end)
        chunk_hits = _scan_keywords(chunk, RESULTS_CANDIDATE_WORDS)
        score = len(chunk_hits) + min(_count_text_chars(chunk) // 300, 5)
        candidate = {
            "text": chunk,
            "matched_keywords": chunk_hits,
            "start_line": index + 1,
            "end_line": end,
            "score": score,
            "note": "result-like paragraph without explicit heading",
        }
        if best is None or score > best["score"]:
            best = candidate
    if best and _count_text_chars(best["text"]) >= 120 and len(best["matched_keywords"]) >= 3:
        return best
    for index, line in enumerate(lines):
        normalized = _normalize_for_match(line)
        if not normalized or not any(word in normalized for word in normalized_words):
            continue
        start = max(0, index - 3)
        end = _next_delimiter_line(index, delimiter_lines, min(len(lines), index + 25))
        chunk = _slice_text_lines(lines, start, end)
        chunk_hits = _scan_keywords(chunk, RESULTS_CANDIDATE_WORDS)
        if _count_text_chars(chunk) >= 120 and len(chunk_hits) >= 2:
            return {
                "text": chunk,
                "matched_keywords": chunk_hits,
                "start_line": start + 1,
                "end_line": end,
                "score": len(chunk_hits),
                "note": "result-like window without explicit heading",
            }
    return None


def _build_enhanced_section_chunks(text: str) -> tuple[dict[str, dict[str, Any]], list[str], list[int]]:
    """Detect confirmed enhanced sections from headings."""
    lines = text.splitlines()
    headings, delimiter_lines = _detect_refine_headings(lines)
    chunks: dict[str, dict[str, Any]] = {}
    for heading in headings:
        section = heading["section"]
        if section in chunks:
            continue
        start_line = int(heading["line_index"])
        end_line = _next_delimiter_line(start_line, delimiter_lines, len(lines))
        chunk = _slice_text_lines(lines, start_line, end_line)
        if _count_text_chars(chunk) <= 0:
            continue
        chunks[section] = {
            "text": chunk,
            "matched_keyword": heading["keyword"],
            "start_line": start_line + 1,
            "end_line": end_line,
            "title": heading["title"],
        }
    return chunks, lines, delimiter_lines


def _section_exists(sections: dict[str, Any], report_section: str) -> bool:
    """Return whether a report section exists in a sections dict."""
    json_key = REFINE_SECTION_TO_JSON_KEY[report_section]
    return _count_text_chars(str(sections.get(json_key, "") or "")) > 0


def refine_one_appendix2_sections(
    text_path: Path,
    original_json_path: Path,
    output_json_path: Path,
    *,
    is_ocr: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Audit and refine one Appendix 2 paper's sections."""
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore") if Path(text_path).exists() else ""
    original_payload = _load_appendix2_section_json(original_json_path, logger)
    original_sections = dict(original_payload.get("sections", {}) or {})
    refined_sections = dict(original_sections)
    enhanced_chunks, lines, delimiter_lines = _build_enhanced_section_chunks(text)
    candidate_sections: dict[str, dict[str, Any]] = {}
    refine_notes: dict[str, dict[str, Any]] = {}
    compare_rows: list[dict[str, Any]] = []

    paper_id = Path(text_path).stem
    total_chars = _count_text_chars(text)
    text_valid = total_chars >= 100

    for section in REFINE_FOCUS_SECTIONS:
        json_key = REFINE_SECTION_TO_JSON_KEY[section]
        original_found = _section_exists(original_sections, section)
        refined_found = original_found
        note = "original_found" if original_found else "not_found"
        status_change = "unchanged_found" if original_found else "still_missing"

        if not original_found and section in enhanced_chunks:
            chunk_info = enhanced_chunks[section]
            refined_sections[json_key] = chunk_info["text"]
            refined_found = True
            status_change = "missing_to_found"
            note = f"enhanced heading match: {chunk_info['matched_keyword']}"
            refine_notes[section] = {
                "source": "heading_match",
                "confidence": "high",
                "matched_keyword": chunk_info["matched_keyword"],
                "start_line": chunk_info["start_line"],
                "end_line": chunk_info["end_line"],
                "note": note,
            }
        elif not original_found and section == "abstract":
            candidate = _find_abstract_candidate(lines, delimiter_lines)
            if candidate:
                candidate_sections[section] = candidate
                status_change = "missing_to_candidate"
                note = "abstract candidate from opening paragraph"
        elif not original_found and section == "results":
            candidate = _find_results_candidate(lines, delimiter_lines)
            if candidate:
                candidate_sections[section] = candidate
                status_change = "missing_to_candidate"
                note = "results candidate from result-like paragraph"

        if section not in refine_notes:
            matched_keywords = _scan_keywords(text, REFINE_KEYWORDS[section])
            if not text_valid:
                audit_status = "parse_failed"
            elif refined_found:
                audit_status = "found"
            elif section in candidate_sections:
                audit_status = "candidate"
            elif matched_keywords:
                audit_status = "recognition_failed"
            else:
                audit_status = "reasonable_missing"
            refine_notes[section] = {
                "source": "original" if original_found else ("candidate" if section in candidate_sections else "missing"),
                "confidence": "high" if original_found else ("medium" if section in candidate_sections else "low"),
                "matched_keyword": ",".join(matched_keywords[:5]),
                "start_line": None,
                "end_line": None,
                "audit_status": audit_status,
                "note": note,
            }
        else:
            refine_notes[section]["audit_status"] = "found"

        compare_rows.append(
            {
                "paper_id": paper_id,
                "filename": Path(text_path).name,
                "section_name": section,
                "original_found": original_found,
                "refined_found": refined_found,
                "status_change": status_change,
                "note": note,
            }
        )

    missing_sections = [
        REFINE_SECTION_TO_JSON_KEY[section]
        for section in SECTION_REPORT_KEYS
        if not _section_exists(refined_sections, section)
    ]
    payload = {
        "paper_id": paper_id,
        "filename": Path(text_path).name,
        "sections": refined_sections,
        "missing_sections": missing_sections,
        "candidate_sections": candidate_sections,
        "refine_notes": refine_notes,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    recognized_count = sum(1 for section in SECTION_REPORT_KEYS if _section_exists(refined_sections, section))
    audit_row: dict[str, Any] = {
        "paper_id": paper_id,
        "filename": Path(text_path).name,
        "txt_path": str(text_path),
        "json_path": str(output_json_path),
        "is_ocr": is_ocr,
        "text_valid": text_valid,
        "total_chars": total_chars,
        "recognized_sections_count": recognized_count,
        "missing_sections_count": len(missing_sections),
        "candidate_sections": ",".join(candidate_sections.keys()),
    }
    for section in REFINE_FOCUS_SECTIONS:
        audit_row[f"{section}_status"] = refine_notes[section]["audit_status"]
        audit_row[f"{section}_note"] = refine_notes[section]["note"]
    logger.info(
        "%s refined: recognized=%s missing=%s candidates=%s valid=%s",
        paper_id,
        recognized_count,
        len(missing_sections),
        ",".join(candidate_sections.keys()) or "none",
        text_valid,
    )
    return audit_row, compare_rows, payload


def _refined_report_from_payloads(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the Step 11B refined section report."""
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        sections = payload.get("sections", {})
        row: dict[str, Any] = {
            "paper_id": payload.get("paper_id", ""),
            "filename": payload.get("filename", ""),
        }
        for report_key in SECTION_REPORT_KEYS:
            row[report_key] = _section_exists(sections, report_key)
        row["missing_sections_count"] = int(sum(1 for report_key in SECTION_REPORT_KEYS if not row[report_key]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["paper_id", "filename", *SECTION_REPORT_KEYS, "missing_sections_count"])


def run_appendix2_refine_sections(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 11B quality audit and enhanced section recognition."""
    problem2_config = get_problem2_config(config_path)
    text_dir = resolve_project_path(problem2_config["appendix2_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem2_config["appendix2_intermediate_dir"])
    tables_dir = resolve_project_path(problem2_config["problem2_output_tables_dir"])
    logs_dir = resolve_project_path(problem2_config["problem2_output_logs_dir"])
    original_sections_dir = intermediate_dir / "sections"
    refined_sections_dir = intermediate_dir / "sections_refined"
    refined_sections_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_appendix2_refine_logger(logs_dir / "appendix2_section_refine.log")
    logger.info("Starting Step 11B Appendix 2 section quality audit and refinement")
    pdf_report_path = tables_dir / "appendix2_pdf_parse_report.xlsx"
    ocr_lookup: dict[str, bool] = {}
    if pdf_report_path.exists():
        pdf_report = pd.read_excel(pdf_report_path)
        if {"paper_id", "ocr_used"}.issubset(pdf_report.columns):
            ocr_lookup = {str(row["paper_id"]): bool(row["ocr_used"]) for _, row in pdf_report.iterrows()}

    audit_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    text_files = sorted(text_dir.glob("*.txt"), key=_natural_sort_key)
    if not text_files:
        logger.error("No Appendix 2 extracted text files found: %s", text_dir)

    for text_path in text_files:
        paper_id = text_path.stem
        audit_row, paper_compare_rows, payload = refine_one_appendix2_sections(
            text_path=text_path,
            original_json_path=original_sections_dir / f"{paper_id}.json",
            output_json_path=refined_sections_dir / f"{paper_id}.json",
            is_ocr=bool(ocr_lookup.get(paper_id, False)),
            logger=logger,
        )
        audit_rows.append(audit_row)
        compare_rows.extend(paper_compare_rows)
        payloads.append(payload)

    refined_report = _refined_report_from_payloads(payloads)
    compare_report = pd.DataFrame(compare_rows)
    audit_report = pd.DataFrame(audit_rows)

    refined_report_path = tables_dir / "appendix2_section_split_report_refined.xlsx"
    compare_report_path = tables_dir / "appendix2_section_refine_compare.xlsx"
    audit_report_path = tables_dir / "appendix2_section_quality_audit.xlsx"
    refined_report.to_excel(refined_report_path, index=False)
    compare_report.to_excel(compare_report_path, index=False)
    audit_report.to_excel(audit_report_path, index=False)

    missing_counts = {
        section: int((~refined_report[section].astype(bool)).sum()) if section in refined_report.columns else 0
        for section in SECTION_REPORT_KEYS
    }
    original_report_path = tables_dir / "appendix2_section_split_report.xlsx"
    original_missing_counts: dict[str, int] = {}
    if original_report_path.exists():
        original_report = pd.read_excel(original_report_path)
        original_missing_counts = {
            section: int((~original_report[section].astype(bool)).sum()) if section in original_report.columns else 0
            for section in SECTION_REPORT_KEYS
        }
    missing_to_found = compare_report.loc[
        compare_report["status_change"].eq("missing_to_found"),
        ["paper_id", "section_name"],
    ].to_dict(orient="records")
    candidates = compare_report.loc[
        compare_report["status_change"].eq("missing_to_candidate"),
        ["paper_id", "section_name"],
    ].to_dict(orient="records")

    logger.info("Refined report saved: %s", refined_report_path)
    logger.info("Compare report saved: %s", compare_report_path)
    logger.info("Audit report saved: %s", audit_report_path)
    logger.info("Finished Step 11B: missing_counts=%s candidates=%s", missing_counts, candidates)
    return {
        "audit_report": audit_report,
        "compare_report": compare_report,
        "refined_report": refined_report,
        "original_missing_counts": original_missing_counts,
        "missing_counts": missing_counts,
        "missing_to_found": missing_to_found,
        "candidates": candidates,
        "refined_sections_dir": refined_sections_dir,
        "refined_report_path": refined_report_path,
        "compare_report_path": compare_report_path,
        "audit_report_path": audit_report_path,
        "log_path": logs_dir / "appendix2_section_refine.log",
    }


def check_problem2_structure() -> tuple[list[str], list[str]]:
    """Check required Problem 2 directories and Python files."""
    missing_dirs = [item for item in PROBLEM2_REQUIRED_DIRS if not resolve_project_path(item).is_dir()]
    required_files = PROBLEM2_REQUIRED_MODULES + PROBLEM2_REQUIRED_SCRIPTS
    missing_files = [item for item in required_files if not resolve_project_path(item).is_file()]
    return missing_dirs, missing_files


def describe_placeholder_step(step_name: str, purpose: str) -> str:
    """Return a common placeholder message for unimplemented Problem 2 steps."""
    return (
        f"{step_name} is scaffolded and runnable. TODO: {purpose}. "
        "This setup step intentionally does not implement the model logic yet."
    )
