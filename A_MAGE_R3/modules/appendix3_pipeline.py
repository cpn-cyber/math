"""Shared helpers for the Problem 3 pipeline.

Step 22 only establishes directories, configuration, and runnable placeholders.
No Appendix 3 parsing, diagnosis, scoring, or optimization is implemented here.
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
LOGGER_NAME = "A_MAGE_R3.problem3.appendix3_parse_sections"

DEFAULT_PROBLEM3_CONFIG: dict[str, Any] = {
    "appendix3_papers_dir": "data/appendix3_papers",
    "appendix3_extracted_text_dir": "data/appendix3_extracted_text",
    "appendix3_intermediate_dir": "data/appendix3_intermediate",
    "output_tables_dir": "output/problem3_tables",
    "output_charts_dir": "output/problem3_charts",
    "output_logs_dir": "output/problem3_logs",
    "output_reports_dir": "output/problem3_reports",
    "alpha_problem1_weight": 0.80,
    "lambda_logic_gap": 0.10,
    "lambda_ai_risk": 0.05,
    "argument_chain_edges": ["T_to_D", "D_to_HM", "HM_to_R", "R_to_C", "C_to_T"],
    "ai_risk_evidence_reliability": {
        "template_expression": 0.75,
        "unsupported_conclusion": 0.80,
        "data_untraceable": 0.85,
        "method_result_jump": 0.85,
    },
    "reviewer_agents": [
        "structure_reviewer",
        "logic_reviewer",
        "modeling_reviewer",
        "result_validation_reviewer",
        "application_value_reviewer",
    ],
    "revision_budget_default": 10,
    "action_cost_max": 10,
    "robustness_bootstrap_B": 500,
}


PROBLEM3_REQUIRED_DIRS = [
    "data/appendix3_papers",
    "data/appendix3_extracted_text",
    "data/appendix3_intermediate",
    "output/problem3_tables",
    "output/problem3_charts",
    "output/problem3_logs",
    "output/problem3_reports",
    "paper_sections/problem3",
]


PROBLEM3_REQUIRED_MODULES = [
    "modules/appendix3_pipeline.py",
    "modules/appendix3_current_evaluator.py",
    "modules/argument_chain_diagnostics.py",
    "modules/ai_risk_ds_fusion.py",
    "modules/reviewer_agent_ensemble.py",
    "modules/revision_action_optimizer.py",
    "modules/post_revision_predictor.py",
    "modules/problem3_robustness.py",
    "modules/problem3_audit.py",
    "modules/problem3_report_generator.py",
]


PROBLEM3_REQUIRED_SCRIPTS = [
    "scripts/run_step22_problem3_setup.py",
    "scripts/run_step23_appendix3_parse_sections.py",
    "scripts/run_step23b_refine_appendix3_sections.py",
    "scripts/run_step24_appendix3_current_eval.py",
    "scripts/run_step25_argument_chain_diagnosis.py",
    "scripts/run_step26_ai_risk_ds_fusion.py",
    "scripts/run_step27_reviewer_agents.py",
    "scripts/run_step28_revision_action_optimization.py",
    "scripts/run_step29_post_revision_prediction.py",
    "scripts/run_step30_problem3_robustness.py",
    "scripts/run_step31_problem3_final_audit.py",
    "scripts/run_step32_problem3_draft.py",
]

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

RESULTS_HEADING_KEYWORDS = [
    "结果",
    "结果分析",
    "求解结果",
    "实验结果",
    "计算结果",
    "数值结果",
    "模型结果",
    "仿真结果",
    "优化结果",
    "结果检验",
    "结果展示",
    "方案结果",
    "预测结果",
    "评价结果",
    "结果与分析",
    "问题一结果",
    "问题二结果",
    "问题三结果",
]

RESULTS_CANDIDATE_KEYWORDS = [
    "得到",
    "计算",
    "如表",
    "如图",
    "结果显示",
    "最终",
    "预测值",
    "评价值",
    "覆盖率",
    "成本",
    "公平性",
    "最优",
    "方案",
    "设施",
    "资源",
    "DQN",
    "选址",
    "NSGA",
    "PSO",
    "DE",
]

CONCLUSION_LIKE_KEYWORDS = ["结论", "总结", "主要结论", "策略建议", "实施策略", "综合策略建议"]

QUALITY_SHORTFALL_KEYWORDS = {
    "sensitivity_analysis": ["灵敏度", "敏感性", "稳健性", "鲁棒性", "参数扰动", "稳定性"],
    "error_analysis": ["误差", "残差", "偏差", "误差分析", "误差检验", "局限性"],
    "model_evaluation": ["模型评价", "模型优缺点", "优缺点", "模型优点", "模型缺点", "模型改进"],
    "appendix": ["附录", "appendix"],
}


def resolve_project_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load project configuration."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load config.yaml. Install requirements.txt first.") from exc
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_problem3_config(config_path: Path | None = None) -> dict[str, Any]:
    """Return the problem3 config section with defaults."""
    merged = dict(DEFAULT_PROBLEM3_CONFIG)
    try:
        config = load_config(config_path)
    except RuntimeError:
        return merged
    merged.update(dict(config.get("problem3", {})))
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


def ensure_problem3_directories() -> list[Path]:
    """Create required Problem 3 directories and return their paths."""
    paths = [resolve_project_path(item) for item in PROBLEM3_REQUIRED_DIRS]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def check_problem3_structure() -> tuple[list[str], list[str]]:
    """Check required Problem 3 directories, modules, and scripts."""
    missing_dirs = [item for item in PROBLEM3_REQUIRED_DIRS if not resolve_project_path(item).is_dir()]
    required_files = PROBLEM3_REQUIRED_MODULES + PROBLEM3_REQUIRED_SCRIPTS
    missing_files = [item for item in required_files if not resolve_project_path(item).is_file()]
    return missing_dirs, missing_files


def problem3_placeholder(step_name: str) -> dict[str, str]:
    """Return a standard placeholder payload for unimplemented Problem 3 steps."""
    return {
        "step": step_name,
        "status": "TODO",
        "note": "Step 22 only scaffolds the Problem 3 pipeline; implementation is reserved for later steps.",
    }


def setup_appendix3_logger(log_path: Path) -> logging.Logger:
    """Configure one Step 23 log file for the pipeline and reused modules."""
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
    """Sort filenames such as 3-1.pdf ... 3-10.pdf naturally."""
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


def sync_appendix3_pdfs_if_needed(input_dir: Path, logger: logging.Logger) -> int:
    """Copy PDFs from the top-level Appendix 3 folder if configured input is empty."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    if list(input_dir.glob("*.pdf")):
        return 0

    source_dir = PROJECT_ROOT.parent / "附件3"
    if not source_dir.is_dir():
        logger.warning("Appendix 3 input is empty and fallback source is missing: %s", source_dir)
        return 0

    copied = 0
    for pdf_path in sorted(source_dir.glob("*.pdf"), key=_natural_sort_key):
        shutil.copy2(pdf_path, input_dir / pdf_path.name)
        copied += 1
    logger.info("Synced %s Appendix 3 PDFs from %s to %s", copied, source_dir, input_dir)
    return copied


def parse_appendix3_pdf(
    pdf_path: Path,
    output_txt_path: Path,
    *,
    pdf_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Parse one Appendix 3 PDF and use OCR if the text layer is too sparse."""
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


def parse_appendix3_pdfs(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    pdf_config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Parse all Appendix 3 PDFs and save the parse report."""
    logger = logger or logging.getLogger(LOGGER_NAME)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"), key=_natural_sort_key)
    if not pdf_files:
        logger.error("No Appendix 3 PDFs found in %s", input_dir)
        report = pd.DataFrame(columns=PDF_PARSE_COLUMNS)
        report.to_excel(report_path, index=False)
        return report

    rows: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        try:
            rows.append(
                parse_appendix3_pdf(
                    pdf_path=pdf_path,
                    output_txt_path=output_dir / f"{pdf_path.stem}.txt",
                    pdf_config=pdf_config,
                    logger=logger,
                )
            )
            logger.info("%s parsed for Appendix 3", pdf_path.name)
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
    logger.info("Appendix 3 PDF parse report saved: %s", report_path)
    return report


def split_appendix3_sections(
    input_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Split all Appendix 3 extracted text files into section JSON outputs."""
    logger = logger or logging.getLogger(LOGGER_NAME)
    input_dir = Path(input_dir)
    sections_dir = Path(output_dir) / "sections"
    report_path = Path(report_path)
    sections_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    text_files = sorted(input_dir.glob("*.txt"), key=_natural_sort_key)
    if not text_files:
        logger.error("No Appendix 3 text files found in %s", input_dir)
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
                logger.warning("%s missing sections: %s", text_path.name, ",".join(split_result["missing_sections"]))
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
    logger.info("Appendix 3 section split report saved: %s", report_path)
    return report


def run_appendix3_parse_sections(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 23: Appendix 3 PDF extraction, OCR fallback, and section splitting."""
    problem3_config = get_problem3_config(config_path)
    pdf_config = get_pdf_parser_config(config_path)

    input_dir = resolve_project_path(problem3_config["appendix3_papers_dir"])
    extracted_text_dir = resolve_project_path(problem3_config["appendix3_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem3_config["appendix3_intermediate_dir"])
    tables_dir = resolve_project_path(problem3_config["output_tables_dir"])
    logs_dir = resolve_project_path(problem3_config["output_logs_dir"])

    ensure_problem3_directories()
    logger = setup_appendix3_logger(logs_dir / "appendix3_parse_sections.log")
    logger.info("Starting Step 23 Appendix 3 PDF parse and section split")
    copied = sync_appendix3_pdfs_if_needed(input_dir, logger)

    pdf_report_path = tables_dir / "appendix3_pdf_parse_report.xlsx"
    section_report_path = tables_dir / "appendix3_section_split_report.xlsx"

    pdf_report = parse_appendix3_pdfs(
        input_dir=input_dir,
        output_dir=extracted_text_dir,
        report_path=pdf_report_path,
        pdf_config=pdf_config,
        logger=logger,
    )
    section_report = split_appendix3_sections(
        input_dir=extracted_text_dir,
        output_dir=intermediate_dir,
        report_path=section_report_path,
        logger=logger,
    )
    logger.info(
        "Finished Step 23: pdf_total=%s parse_success=%s ocr_used=%s section_total=%s",
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
        "log_path": logs_dir / "appendix3_parse_sections.log",
        "extracted_text_dir": extracted_text_dir,
        "sections_dir": intermediate_dir / "sections",
    }


def setup_appendix3_refine_logger(log_path: Path) -> logging.Logger:
    """Configure the Step 23B section-refine logger."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger = logging.getLogger("A_MAGE_R3.problem3.appendix3_section_refine")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


def _load_section_json(json_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Load a section JSON file with a defensive fallback."""
    try:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to load section JSON %s: %s", json_path, exc)
        return {"paper_id": Path(json_path).stem, "filename": Path(json_path).name, "sections": {}, "missing_sections": []}


def _section_text_exists(sections: dict[str, Any], report_key: str) -> bool:
    """Return whether the mapped report section has non-empty content."""
    json_key = SECTION_KEY_MAP[report_key]
    return _count_non_whitespace(str(sections.get(json_key, "") or "")) > 0


def _normalize_line(line: str) -> str:
    """Normalize a line for heading and keyword matching."""
    line = re.sub(r"\[PAGE\s+\d+\]", "", line, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", line or "")


def _looks_like_major_heading(line: str) -> bool:
    """Return whether a line can delimit a major section."""
    compact = _normalize_line(line)
    if not compact or len(compact) > 50:
        return False
    if re.match(r"^第?[一二三四五六七八九十\d]+[、.．章节]", compact):
        return True
    if re.match(r"^\d+(\.\d+){0,2}", compact):
        return True
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]", compact):
        return True
    short_section_titles = [
        "摘要",
        "结果",
        "结果分析",
        "模型检验",
        "模型评价",
        "参考文献",
        "附录",
        "结论",
        "总结",
    ]
    return len(compact) <= 18 and any(keyword in compact for keyword in short_section_titles)


def _line_has_any(line: str, keywords: list[str]) -> list[str]:
    """Return matched keywords in one line."""
    compact = _normalize_line(line)
    return [keyword for keyword in keywords if _normalize_line(keyword) in compact]


def _find_next_delimiter(lines: list[str], start_idx: int) -> int:
    """Find the next likely section delimiter after start_idx."""
    for idx in range(start_idx + 1, len(lines)):
        compact = _normalize_line(lines[idx])
        if not compact:
            continue
        if "参考文献" in compact or "附录" in compact:
            return idx
        if _looks_like_major_heading(lines[idx]) and any(
            keyword in compact
            for keyword in ["模型检验", "模型评价", "优缺点", "灵敏度", "敏感性", "误差", "参考文献", "附录", "结论", "总结"]
        ):
            return idx
        if idx > start_idx + 8 and re.match(r"^第?[一二三四五六七八九十]+[、.．]", compact):
            return idx
        if idx > start_idx + 8 and re.match(r"^\d+(\.\d+){0,2}", compact) and any(
            keyword in compact for keyword in ["检验", "评价", "参考", "附录", "结论"]
        ):
            return idx
    return len(lines)


def _extract_result_chunk_by_heading(text: str) -> dict[str, Any] | None:
    """Extract confirmed or high-confidence results chunks from headings or explicit result cues."""
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if idx < 50:
            # Skip abstract/problem statement result summaries; they are useful evidence
            # but should not become the main results node for the argument chain.
            continue
        compact = _normalize_line(line)
        if idx > 50 and ("参考文献" in compact or compact.startswith("附录")):
            break
        if not compact or len(compact) > 80:
            continue
        matched = _line_has_any(line, RESULTS_HEADING_KEYWORDS)
        has_explicit_result_heading = bool(matched and _looks_like_major_heading(line))
        has_main_conclusion_cue = "主要结论" in compact or "得出以下主要结论" in compact
        has_result_sentence_cue = any(
            cue in compact
            for cue in ["根据结果", "结果如下", "结果显示", "结果表明", "由可视化结果可得", "结果可得"]
        )
        lookahead = "".join(_normalize_line(item) for item in lines[idx : min(len(lines), idx + 5)])
        model_solving_to_result = "模型的求解" in compact and ("主要结论" in lookahead or "结果" in lookahead)
        if not (has_explicit_result_heading or has_main_conclusion_cue or has_result_sentence_cue or model_solving_to_result):
            continue
        start = idx
        if has_main_conclusion_cue and idx > 0 and "模型的求解" in _normalize_line(lines[idx - 1]):
            start = idx - 1
        end = _find_next_delimiter(lines, start)
        if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
            continue
        chunk = "\n".join(lines[start:end]).strip()
        if _count_non_whitespace(chunk) < 80:
            continue
        if matched:
            matched_keyword = matched[0]
        elif has_main_conclusion_cue:
            matched_keyword = "主要结论"
        elif model_solving_to_result:
            matched_keyword = "模型求解后主要结论"
        else:
            matched_keyword = "结果句式线索"
        chunks.append(
            {
                "text": chunk,
                "matched_keyword": matched_keyword,
                "start_line": start + 1,
                "end_line": end,
            }
        )
        used_ranges.append((start, end))
    if not chunks:
        return None
    merged_text = "\n\n[RESULTS_CHUNK]\n\n".join(chunk["text"] for chunk in chunks)
    matched_keywords = list(dict.fromkeys(str(chunk["matched_keyword"]) for chunk in chunks))
    return {
        "text": merged_text,
        "source": "heading_or_explicit_result_cue",
        "confidence": "high" if len(chunks) >= 2 else "medium_high",
        "matched_keyword": ",".join(matched_keywords),
        "start_line": int(chunks[0]["start_line"]),
        "end_line": int(chunks[-1]["end_line"]),
        "chunks": [{key: value for key, value in chunk.items() if key != "text"} for chunk in chunks],
    }


def _extract_results_candidate(text: str) -> dict[str, Any] | None:
    """Build a results candidate from dense result-like evidence."""
    lines = text.splitlines()
    hit_indices: list[int] = []
    hit_keywords: list[str] = []
    for idx, line in enumerate(lines):
        if "参考文献" in _normalize_line(line) or "附录" in _normalize_line(line):
            break
        matched = _line_has_any(line, RESULTS_CANDIDATE_KEYWORDS)
        if matched:
            hit_indices.append(idx)
            hit_keywords.extend(matched)
    if len(hit_indices) < 5:
        return None

    groups: list[list[int]] = []
    current: list[int] = []
    last = -999
    for idx in hit_indices:
        if current and idx - last > 4:
            groups.append(current)
            current = []
        current.append(idx)
        last = idx
    if current:
        groups.append(current)
    best = max(groups, key=len)
    start = max(0, best[0] - 2)
    end = min(len(lines), best[-1] + 3)
    chunk = "\n".join(lines[start:end]).strip()
    if _count_non_whitespace(chunk) < 80:
        return None
    unique_keywords = list(dict.fromkeys(hit_keywords))
    return {
        "text": chunk,
        "source": "results_keyword_candidate",
        "confidence": "medium",
        "matched_keywords": unique_keywords[:12],
        "start_line": start + 1,
        "end_line": end,
        "evidence_count": len(best),
    }


def _scan_conclusion_like(text: str) -> dict[str, Any]:
    """Scan conclusion-like content without creating a formal section."""
    lines = text.splitlines()
    evidence: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        matched = _line_has_any(line, CONCLUSION_LIKE_KEYWORDS)
        if matched:
            evidence.append({"line": idx + 1, "matched_keywords": matched, "text": line.strip()[:120]})
    return {
        "found": bool(evidence),
        "evidence_count": len(evidence),
        "evidence": evidence[:5],
    }


def _status_for_missing_section(text: str, section_key: str) -> tuple[str, str]:
    """Classify missing non-results sections for audit only."""
    keywords = QUALITY_SHORTFALL_KEYWORDS.get(section_key, [])
    if not keywords:
        return "missing", "未识别到该章节。"
    matched: list[str] = []
    for keyword in keywords:
        if keyword.lower() in text.lower() or keyword in text:
            matched.append(keyword)
    if matched:
        return "recognition_failed", f"章节缺失但全文出现关键词：{', '.join(matched[:5])}。"
    return "reasonable_missing", "未发现明显同义标题或关键词，暂按质量短板/合理缺失记录。"


def _build_refined_section_report(refined_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the Step 23B refined section split report."""
    rows: list[dict[str, Any]] = []
    for payload in refined_payloads:
        sections = dict(payload.get("sections", {}) or {})
        row = {"paper_id": payload.get("paper_id", ""), "filename": payload.get("filename", "")}
        for report_key in SECTION_REPORT_KEYS:
            row[report_key] = _section_text_exists(sections, report_key)
        row["missing_sections_count"] = int(sum(1 for report_key in SECTION_REPORT_KEYS if not row[report_key]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["paper_id", "filename", *SECTION_REPORT_KEYS, "missing_sections_count"])


def refine_one_appendix3_sections(
    *,
    text_path: Path,
    original_json_path: Path,
    output_json_path: Path,
    parse_success: bool,
    ocr_used: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Audit and refine one Appendix 3 paper's sections."""
    text = Path(text_path).read_text(encoding="utf-8", errors="ignore") if Path(text_path).exists() else ""
    original_payload = _load_section_json(original_json_path, logger)
    original_sections = dict(original_payload.get("sections", {}) or {})
    refined_sections = dict(original_sections)
    candidate_sections: dict[str, Any] = {}
    refine_notes: dict[str, Any] = {}
    confidence_flags: dict[str, str] = {}

    paper_id = str(original_payload.get("paper_id") or Path(text_path).stem)
    filename = str(original_payload.get("filename") or Path(text_path).name)
    total_chars = _count_non_whitespace(re.sub(r"\[PAGE\s+\d+\]", "", text, flags=re.IGNORECASE))

    original_results_found = _section_text_exists(original_sections, "results")
    if original_results_found:
        refine_notes["results"] = {
            "status": "found",
            "source": "original",
            "confidence": "high",
            "note": "Step23 原始章节已识别 results。",
        }
        confidence_flags["results"] = "high"
    else:
        result_chunk = _extract_result_chunk_by_heading(text)
        if result_chunk is not None:
            refined_sections["results"] = result_chunk["text"]
            refine_notes["results"] = {
                "status": "found",
                "source": result_chunk["source"],
                "confidence": result_chunk["confidence"],
                "matched_keyword": result_chunk["matched_keyword"],
                "start_line": result_chunk["start_line"],
                "end_line": result_chunk["end_line"],
                "note": "results 原始缺失，但检测到明确结果标题或主要结论线索，已回填为 refined results。",
            }
            confidence_flags["results"] = result_chunk["confidence"]
        else:
            candidate = _extract_results_candidate(text)
            if candidate is not None:
                candidate_sections["results"] = candidate
                refine_notes["results"] = {
                    "status": "candidate",
                    "source": candidate["source"],
                    "confidence": candidate["confidence"],
                    "matched_keywords": candidate["matched_keywords"],
                    "start_line": candidate["start_line"],
                    "end_line": candidate["end_line"],
                    "evidence_count": candidate["evidence_count"],
                    "note": "未检测到明确 results 标题，仅发现结果型关键词密集段，保留为 candidate。",
                }
                confidence_flags["results"] = "candidate_medium"
            else:
                refine_notes["results"] = {
                    "status": "missing",
                    "source": "missing",
                    "confidence": "low",
                    "note": "未检测到明确 results 标题或足够结果型候选段。",
                }
                confidence_flags["results"] = "missing"

    conclusion_like = _scan_conclusion_like(text)
    refine_notes["conclusion_like"] = {
        "status": "found" if conclusion_like["found"] else "missing",
        "source": "keyword_scan",
        "confidence": "medium" if conclusion_like["found"] else "low",
        "evidence_count": conclusion_like["evidence_count"],
        "evidence": conclusion_like["evidence"],
        "note": "conclusion-like 内容仅用于 Step23B 复核，不写入正式章节。",
    }
    confidence_flags["conclusion_like"] = "medium" if conclusion_like["found"] else "missing"

    compare_rows: list[dict[str, Any]] = []
    for report_key in SECTION_REPORT_KEYS:
        original_found = _section_text_exists(original_sections, report_key)
        refined_found = _section_text_exists(refined_sections, report_key)
        candidate_found = report_key in candidate_sections
        if original_found and refined_found:
            status_change = "unchanged_found"
            note = "原始章节已识别。"
        elif (not original_found) and refined_found:
            status_change = "missing_to_found"
            note = str(refine_notes.get(report_key, {}).get("note") or "增强规则识别为确定章节。")
        elif (not original_found) and candidate_found:
            status_change = "missing_to_candidate"
            note = str(refine_notes.get(report_key, {}).get("note") or "仅作为候选章节。")
        else:
            status_change, note = _status_for_missing_section(text, report_key)
        compare_rows.append(
            {
                "paper_id": paper_id,
                "filename": filename,
                "section_name": report_key,
                "original_found": original_found,
                "refined_found": refined_found,
                "candidate_found": candidate_found,
                "status_change": status_change,
                "note": note,
            }
        )

    compare_rows.append(
        {
            "paper_id": paper_id,
            "filename": filename,
            "section_name": "conclusion_like",
            "original_found": False,
            "refined_found": bool(conclusion_like["found"]),
            "candidate_found": False,
            "status_change": "found" if conclusion_like["found"] else "missing",
            "note": f"conclusion-like keyword evidence count={conclusion_like['evidence_count']}",
        }
    )

    missing_sections = [
        SECTION_KEY_MAP[report_key]
        for report_key in SECTION_REPORT_KEYS
        if not _section_text_exists(refined_sections, report_key)
    ]
    recognized_count = sum(1 for report_key in SECTION_REPORT_KEYS if _section_text_exists(refined_sections, report_key))
    recognition_failed = [
        row["section_name"]
        for row in compare_rows
        if row["status_change"] == "recognition_failed"
    ]
    reasonable_missing = [
        row["section_name"]
        for row in compare_rows
        if row["status_change"] == "reasonable_missing"
    ]

    payload = {
        "paper_id": paper_id,
        "filename": filename,
        "sections": refined_sections,
        "missing_sections": missing_sections,
        "candidate_sections": candidate_sections,
        "refine_notes": refine_notes,
        "confidence_flags": confidence_flags,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_row: dict[str, Any] = {
        "paper_id": paper_id,
        "filename": filename,
        "txt_path": str(text_path),
        "json_path": str(original_json_path),
        "refined_json_path": str(output_json_path),
        "parse_status": "parse_failed" if not parse_success or total_chars == 0 else "parsed",
        "is_ocr": bool(ocr_used),
        "total_chars": total_chars,
        "detected_sections_count": recognized_count,
        "missing_sections": ",".join(missing_sections),
        "recognition_failed_sections": ",".join(recognition_failed),
        "reasonable_missing_sections": ",".join(reasonable_missing),
        "results_status": refine_notes["results"]["status"],
        "results_confidence": refine_notes["results"]["confidence"],
        "results_note": refine_notes["results"]["note"],
        "conclusion_like_found": bool(conclusion_like["found"]),
        "need_manual_check": (not parse_success) or bool(recognition_failed) or refine_notes["results"]["status"] in {"missing", "candidate"},
    }
    logger.info(
        "%s refined: results=%s missing=%s recognition_failed=%s",
        paper_id,
        refine_notes["results"]["status"],
        ",".join(missing_sections) or "none",
        ",".join(recognition_failed) or "none",
    )
    return audit_row, compare_rows, payload


def run_appendix3_refine_sections(config_path: Path | None = None) -> dict[str, Any]:
    """Run Step 23B: audit and refine Appendix 3 section recognition."""
    problem3_config = get_problem3_config(config_path)
    text_dir = resolve_project_path(problem3_config["appendix3_extracted_text_dir"])
    intermediate_dir = resolve_project_path(problem3_config["appendix3_intermediate_dir"])
    tables_dir = resolve_project_path(problem3_config["output_tables_dir"])
    logs_dir = resolve_project_path(problem3_config["output_logs_dir"])

    original_sections_dir = intermediate_dir / "sections"
    refined_sections_dir = intermediate_dir / "sections_refined"
    refined_sections_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_appendix3_refine_logger(logs_dir / "appendix3_section_refine.log")
    logger.info("Starting Step 23B Appendix 3 section quality audit and results refinement")

    pdf_report_path = tables_dir / "appendix3_pdf_parse_report.xlsx"
    parse_lookup: dict[str, dict[str, Any]] = {}
    if pdf_report_path.exists():
        pdf_report = pd.read_excel(pdf_report_path)
        if {"paper_id", "parse_success", "ocr_used"}.issubset(pdf_report.columns):
            parse_lookup = {
                str(row["paper_id"]): {
                    "parse_success": bool(row["parse_success"]),
                    "ocr_used": bool(row["ocr_used"]),
                }
                for _, row in pdf_report.iterrows()
            }

    audit_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    refined_payloads: list[dict[str, Any]] = []

    for text_path in sorted(text_dir.glob("*.txt"), key=_natural_sort_key):
        paper_id = text_path.stem
        audit_row, paper_compare_rows, payload = refine_one_appendix3_sections(
            text_path=text_path,
            original_json_path=original_sections_dir / f"{paper_id}.json",
            output_json_path=refined_sections_dir / f"{paper_id}.json",
            parse_success=bool(parse_lookup.get(paper_id, {}).get("parse_success", True)),
            ocr_used=bool(parse_lookup.get(paper_id, {}).get("ocr_used", False)),
            logger=logger,
        )
        audit_rows.append(audit_row)
        compare_rows.extend(paper_compare_rows)
        refined_payloads.append(payload)

    refined_report = _build_refined_section_report(refined_payloads)
    compare_report = pd.DataFrame(
        compare_rows,
        columns=[
            "paper_id",
            "filename",
            "section_name",
            "original_found",
            "refined_found",
            "candidate_found",
            "status_change",
            "note",
        ],
    )
    audit_report = pd.DataFrame(audit_rows)

    refined_report_path = tables_dir / "appendix3_section_split_report_refined.xlsx"
    compare_report_path = tables_dir / "appendix3_section_refine_compare.xlsx"
    audit_report_path = tables_dir / "appendix3_section_quality_audit.xlsx"
    refined_report.to_excel(refined_report_path, index=False)
    compare_report.to_excel(compare_report_path, index=False)
    audit_report.to_excel(audit_report_path, index=False)

    logger.info(
        "Finished Step 23B: papers=%s refined_results_found=%s candidate_results=%s",
        len(audit_report),
        int((audit_report["results_status"] == "found").sum()) if len(audit_report) else 0,
        int((audit_report["results_status"] == "candidate").sum()) if len(audit_report) else 0,
    )
    return {
        "refined_report": refined_report,
        "compare_report": compare_report,
        "audit_report": audit_report,
        "refined_sections_dir": refined_sections_dir,
        "refined_report_path": refined_report_path,
        "compare_report_path": compare_report_path,
        "audit_report_path": audit_report_path,
        "log_path": logs_dir / "appendix3_section_refine.log",
    }
