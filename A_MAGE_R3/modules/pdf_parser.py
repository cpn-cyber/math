"""PDF parsing and text extraction utilities for Step 2.

This module only implements PyMuPDF-based PDF parsing. If a page has no
extractable text, the page is kept in the output with its page marker and the
blank page event is written to the log. OCR is intentionally reserved for a
later step/configuration pass, so this module never invents or fabricates text.
"""

from pathlib import Path
import logging
import re
from typing import Any

import pandas as pd

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - exercised only without deps.
    fitz = None
    PYMUPDF_IMPORT_ERROR = exc
else:
    PYMUPDF_IMPORT_ERROR = None


LOGGER_NAME = "A_MAGE_R3.pdf_parser"
REPORT_COLUMNS = [
    "文件名",
    "页数",
    "字数",
    "是否解析成功",
    "空白页数量",
    "空白页页码",
    "错误信息",
    "输出文本路径",
]


def _get_logger() -> logging.Logger:
    """Return the parser logger."""
    return logging.getLogger(LOGGER_NAME)


def setup_pdf_parse_logger(log_path: Path) -> logging.Logger:
    """Configure a file logger for PDF parsing."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = _get_logger()
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


def _ensure_pymupdf_available() -> None:
    """Raise a helpful error if PyMuPDF is unavailable."""
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for Step 2. Please run "
            "`pip install -r requirements.txt` first."
        ) from PYMUPDF_IMPORT_ERROR


def _count_non_whitespace_chars(text: str) -> int:
    """Count non-whitespace characters in extracted text."""
    return len(re.sub(r"\s+", "", text))


def _empty_report_row(pdf_path: Path, output_txt_path: Path, error: str) -> dict[str, Any]:
    """Build a failed parse report row."""
    return {
        "文件名": pdf_path.name,
        "页数": 0,
        "字数": 0,
        "是否解析成功": False,
        "空白页数量": 0,
        "空白页页码": "",
        "错误信息": error,
        "输出文本路径": str(output_txt_path),
    }


def parse_pdf_to_text(pdf_path: Path, output_txt_path: Path) -> dict[str, Any]:
    """Parse one PDF into a page-marked text file.

    Parameters
    ----------
    pdf_path:
        Source PDF path.
    output_txt_path:
        Target text path. Parent directories are created automatically.

    Returns
    -------
    dict
        One report row containing file name, page count, character count,
        success flag, blank page count, blank page numbers, error message, and
        output text path.
    """
    _ensure_pymupdf_available()

    pdf_path = Path(pdf_path)
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _get_logger()

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        error = f"Failed to open PDF: {exc}"
        logger.exception("%s | %s", pdf_path.name, error)
        return _empty_report_row(pdf_path, output_txt_path, error)

    page_text_blocks: list[str] = []
    extracted_text_parts: list[str] = []
    blank_pages: list[int] = []

    try:
        page_count = document.page_count
        for page_index in range(page_count):
            page_number = page_index + 1
            try:
                page = document.load_page(page_index)
                page_text = page.get_text("text") or ""
            except Exception as exc:
                page_text = ""
                logger.exception(
                    "%s | page %s extraction failed: %s",
                    pdf_path.name,
                    page_number,
                    exc,
                )

            if not page_text.strip():
                blank_pages.append(page_number)
                logger.warning("%s | page %s extracted empty text", pdf_path.name, page_number)

            page_text_blocks.append(f"[PAGE {page_number}]\n{page_text.rstrip()}\n")
            extracted_text_parts.append(page_text)

        output_txt_path.write_text("\n".join(page_text_blocks), encoding="utf-8")
        full_text = "\n".join(extracted_text_parts)
        logger.info(
            "%s parsed successfully: pages=%s chars=%s blank_pages=%s",
            pdf_path.name,
            page_count,
            _count_non_whitespace_chars(full_text),
            len(blank_pages),
        )

        return {
            "文件名": pdf_path.name,
            "页数": page_count,
            "字数": _count_non_whitespace_chars(full_text),
            "是否解析成功": True,
            "空白页数量": len(blank_pages),
            "空白页页码": ",".join(str(item) for item in blank_pages),
            "错误信息": "",
            "输出文本路径": str(output_txt_path),
        }
    except Exception as exc:
        error = f"Failed during PDF parsing: {exc}"
        logger.exception("%s | %s", pdf_path.name, error)
        return _empty_report_row(pdf_path, output_txt_path, error)
    finally:
        document.close()


def parse_all_pdfs(
    input_dir: Path,
    output_dir: Path,
    report_path: Path | None = None,
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Parse all PDF files in a directory and optionally save an Excel report."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_path is not None:
        logger = setup_pdf_parse_logger(Path(log_path))
    else:
        logger = _get_logger()

    logger.info("Starting PDF parsing: input_dir=%s output_dir=%s", input_dir, output_dir)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        report = pd.DataFrame(columns=REPORT_COLUMNS)
        if report_path is not None:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            report.to_excel(report_path, index=False)
        return report

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)

    rows: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        output_txt_path = output_dir / f"{pdf_path.stem}.txt"
        rows.append(parse_pdf_to_text(pdf_path, output_txt_path))

    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_excel(report_path, index=False)
        logger.info("PDF parse report saved: %s", report_path)

    logger.info("Finished PDF parsing: total=%s failed=%s", len(report), int((~report["是否解析成功"]).sum()) if len(report) else 0)
    return report


def parse_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Backward-compatible wrapper for parsing one PDF."""
    output_txt_path = Path(output_dir) / f"{Path(pdf_path).stem}.txt"
    parse_pdf_to_text(Path(pdf_path), output_txt_path)
    return output_txt_path


def parse_pdf_batch(input_dir: Path, output_dir: Path) -> list[Path]:
    """Backward-compatible wrapper for parsing all PDFs."""
    report = parse_all_pdfs(Path(input_dir), Path(output_dir))
    return [Path(item) for item in report["输出文本路径"].dropna().tolist()]
