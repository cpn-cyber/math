"""PDF parsing, text extraction, and OCR fallback utilities.

Step 2 uses PyMuPDF for normal text extraction. Step 2B adds an OCR fallback
for scanned PDFs identified from the Step 2 parse report. OCR text is written
only when a local OCR engine returns real text; failed OCR attempts never
delete or replace the original extracted text file.
"""

from pathlib import Path
import os
import logging
import re
import shutil
import tempfile
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
OCR_LOGGER_NAME = "A_MAGE_R3.ocr_parser"
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
OCR_SUMMARY_COLUMNS = [
    "文件名",
    "页数",
    "OCR成功页数",
    "OCR失败页数",
    "OCR总字数",
    "是否OCR成功",
    "OCR引擎",
    "错误信息",
    "输出文本路径",
]
OCR_PAGE_COLUMNS = [
    "文件名",
    "页码",
    "OCR识别字符数",
    "是否识别成功",
    "错误信息",
    "OCR引擎",
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


def setup_ocr_parse_logger(log_path: Path) -> logging.Logger:
    """Configure a file logger for Step 2B OCR parsing."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(OCR_LOGGER_NAME)
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


def _get_ocr_logger() -> logging.Logger:
    """Return the OCR parser logger."""
    return logging.getLogger(OCR_LOGGER_NAME)


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _to_bool(value: Any) -> bool:
    """Convert report values such as True/False/是/否 to bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "是", "成功"}


def _to_int(value: Any, default: int = 0) -> int:
    """Convert a report value to int with a safe default."""
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _find_tesseract_command(tesseract_cmd: str | None = None) -> str | None:
    """Locate a Tesseract executable from config, environment, or PATH."""
    if tesseract_cmd:
        configured_path = Path(tesseract_cmd)
        if configured_path.exists():
            return str(configured_path)
        return None

    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
        if shutil.which(str(candidate)):
            return str(candidate)
    return None


def _build_ocr_install_message() -> str:
    """Return a clear local OCR installation hint."""
    return (
        "未检测到可用的本地 OCR 环境。请安装 Tesseract OCR 并配置 PATH，"
        "同时安装中文语言包 chi_sim；Windows 可优先安装 UB-Mannheim Tesseract。"
        "替代方案：安装 paddleocr 并确保其模型可在本地运行。"
    )


def _detect_ocr_backend(
    preferred_engine: str = "auto",
    tesseract_cmd: str | None = None,
) -> dict[str, Any]:
    """Detect a local OCR backend without downloading or fabricating text."""
    preferred_engine = (preferred_engine or "auto").lower()

    if preferred_engine in {"auto", "tesseract", "tesseract_or_fallback"}:
        command = _find_tesseract_command(tesseract_cmd)
        if command:
            try:
                import pytesseract  # type: ignore
            except Exception as exc:
                return {
                    "engine": None,
                    "error": f"已检测到 Tesseract 路径，但缺少 pytesseract Python 包：{exc}",
                }
            pytesseract.pytesseract.tesseract_cmd = command
            return {"engine": "pytesseract", "command": command, "module": pytesseract}
        if tesseract_cmd:
            return {
                "engine": None,
                "error": (
                    f"tesseract_path 配置的安装路径不存在：{tesseract_cmd}。"
                    "请确认 Tesseract OCR 是否安装在该路径，或修改 config.yaml 中的 tesseract_path。"
                ),
            }

    if preferred_engine in {"auto", "paddleocr", "tesseract_or_fallback"}:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception:
            PaddleOCR = None
        if PaddleOCR is not None:
            return {"engine": "paddleocr", "class": PaddleOCR}

    return {"engine": None, "error": _build_ocr_install_message()}


def _run_pytesseract_ocr(
    image_path: Path,
    backend: dict[str, Any],
    languages: list[str],
    psm: int,
    timeout_seconds: int,
) -> tuple[str, str]:
    """Run local Tesseract OCR through pytesseract with language fallback."""
    pytesseract = backend["module"]
    if backend.get("command"):
        pytesseract.pytesseract.tesseract_cmd = str(backend["command"])

    primary_lang = "+".join(languages) if languages else "chi_sim+eng"
    lang_candidates = [primary_lang]
    if primary_lang != "eng":
        lang_candidates.append("eng")

    errors: list[str] = []
    config = f"--psm {psm}"
    for lang in lang_candidates:
        try:
            text = pytesseract.image_to_string(
                str(image_path),
                lang=lang,
                config=config,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            errors.append(f"lang={lang} failed: {exc}")
            continue

        if _count_non_whitespace_chars(text) > 0:
            if errors:
                return text, "已从 chi_sim+eng 降级到 eng；" + " | ".join(errors)
            return text, ""

        errors.append(f"lang={lang} returned empty text")

    return "", " | ".join(errors)


def _run_paddleocr(
    image_path: Path,
    backend: dict[str, Any],
    languages: list[str],
) -> tuple[str, str]:
    """Run local PaddleOCR on one rendered page image if available."""
    try:
        PaddleOCR = backend["class"]
        lang = "ch" if any(item.startswith("chi") or item in {"ch", "cn"} for item in languages) else "en"
        if "instance" not in backend:
            backend["instance"] = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        result = backend["instance"].ocr(str(image_path), cls=True)
        lines: list[str] = []
        for page_result in result or []:
            for item in page_result or []:
                if len(item) >= 2 and isinstance(item[1], (list, tuple)) and item[1]:
                    lines.append(str(item[1][0]))
        return "\n".join(lines), ""
    except Exception as exc:
        return "", str(exc)


def _ocr_image(
    image_path: Path,
    backend: dict[str, Any],
    languages: list[str],
    psm: int,
    timeout_seconds: int,
) -> tuple[str, str]:
    """OCR one rendered page image with the selected backend."""
    if backend.get("engine") == "pytesseract":
        return _run_pytesseract_ocr(image_path, backend, languages, psm, timeout_seconds)
    if backend.get("engine") == "paddleocr":
        return _run_paddleocr(image_path, backend, languages)
    return "", backend.get("error", _build_ocr_install_message())


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


def ocr_pdf_to_text(
    pdf_path: Path,
    output_txt_path: Path,
    *,
    render_dpi: int = 300,
    languages: list[str] | None = None,
    preferred_engine: str = "auto",
    tesseract_cmd: str | None = None,
    psm: int = 6,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """OCR one scanned PDF into a page-marked text file.

    The existing output text file is overwritten only when OCR extracts at
    least one non-whitespace character. A failed OCR attempt returns report
    rows and leaves the previous text file untouched.
    """
    _ensure_pymupdf_available()

    pdf_path = Path(pdf_path)
    output_txt_path = Path(output_txt_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = _get_ocr_logger()

    languages = languages or ["chi_sim", "eng"]
    backend = _detect_ocr_backend(preferred_engine, tesseract_cmd)
    engine = backend.get("engine") or "none"

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        error = f"Failed to open PDF for OCR: {exc}"
        logger.exception("%s | %s", pdf_path.name, error)
        return {
            "summary": {
                "文件名": pdf_path.name,
                "页数": 0,
                "OCR成功页数": 0,
                "OCR失败页数": 0,
                "OCR总字数": 0,
                "是否OCR成功": False,
                "OCR引擎": engine,
                "错误信息": error,
                "输出文本路径": str(output_txt_path),
            },
            "pages": [
                {
                    "文件名": pdf_path.name,
                    "页码": "",
                    "OCR识别字符数": 0,
                    "是否识别成功": False,
                    "错误信息": error,
                    "OCR引擎": engine,
                }
            ],
        }

    page_rows: list[dict[str, Any]] = []
    page_text_blocks: list[str] = []
    extracted_parts: list[str] = []
    page_count = document.page_count
    zoom = max(render_dpi, 72) / 72
    matrix = fitz.Matrix(zoom, zoom)

    try:
        if backend.get("engine") is None:
            error = backend.get("error", _build_ocr_install_message())
            logger.error("%s | %s", pdf_path.name, error)
            for page_number in range(1, page_count + 1):
                page_rows.append(
                    {
                        "文件名": pdf_path.name,
                        "页码": page_number,
                        "OCR识别字符数": 0,
                        "是否识别成功": False,
                        "错误信息": error,
                        "OCR引擎": engine,
                    }
                )
            return {
                "summary": {
                    "文件名": pdf_path.name,
                    "页数": page_count,
                    "OCR成功页数": 0,
                    "OCR失败页数": page_count,
                    "OCR总字数": 0,
                    "是否OCR成功": False,
                    "OCR引擎": engine,
                    "错误信息": error,
                    "输出文本路径": str(output_txt_path),
                },
                "pages": page_rows,
            }

        with tempfile.TemporaryDirectory(prefix="a_mage_r3_ocr_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for page_index in range(page_count):
                page_number = page_index + 1
                page_text = ""
                error = ""
                try:
                    page = document.load_page(page_index)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image_path = tmp_path / f"{pdf_path.stem}_page_{page_number:03d}.png"
                    pixmap.save(str(image_path))
                    page_text, error = _ocr_image(
                        image_path=image_path,
                        backend=backend,
                        languages=languages,
                        psm=psm,
                        timeout_seconds=timeout_seconds,
                    )
                except RuntimeError as exc:
                    error = f"OCR timeout or runtime error after {timeout_seconds}s: {exc}"
                except Exception as exc:
                    error = f"OCR page failed: {exc}"

                char_count = _count_non_whitespace_chars(page_text)
                success = char_count > 0
                if success:
                    logger.info("%s | page %s OCR chars=%s", pdf_path.name, page_number, char_count)
                else:
                    logger.warning(
                        "%s | page %s OCR failed or empty: %s",
                        pdf_path.name,
                        page_number,
                        error or "empty OCR text",
                    )

                page_rows.append(
                    {
                        "文件名": pdf_path.name,
                        "页码": page_number,
                        "OCR识别字符数": char_count,
                        "是否识别成功": success,
                        "错误信息": error,
                        "OCR引擎": engine,
                    }
                )
                page_text_blocks.append(f"[PAGE {page_number}]\n{page_text.rstrip()}\n")
                extracted_parts.append(page_text)

        full_text = "\n".join(extracted_parts)
        total_chars = _count_non_whitespace_chars(full_text)
        success_pages = sum(1 for row in page_rows if row["是否识别成功"])
        failed_pages = page_count - success_pages
        document_success = total_chars > 0 and success_pages > 0
        summary_error = ""

        if document_success:
            output_txt_path.write_text("\n".join(page_text_blocks), encoding="utf-8")
            logger.info(
                "%s OCR succeeded: pages=%s success_pages=%s chars=%s output=%s",
                pdf_path.name,
                page_count,
                success_pages,
                total_chars,
                output_txt_path,
            )
        else:
            summary_error = "OCR未识别出有效正文，已保留原始文本文件。"
            logger.error("%s | %s", pdf_path.name, summary_error)

        return {
            "summary": {
                "文件名": pdf_path.name,
                "页数": page_count,
                "OCR成功页数": success_pages,
                "OCR失败页数": failed_pages,
                "OCR总字数": total_chars,
                "是否OCR成功": document_success,
                "OCR引擎": engine,
                "错误信息": summary_error,
                "输出文本路径": str(output_txt_path),
            },
            "pages": page_rows,
        }
    finally:
        document.close()


def _select_ocr_targets_from_report(
    report: pd.DataFrame,
    *,
    scanned_text_threshold: int = 100,
) -> pd.DataFrame:
    """Select failed or scanned-looking PDFs from the Step 2 parse report."""
    columns = list(report.columns)
    filename_col = _find_column(columns, ["文件名", "filename", "pdf_filename"])
    success_col = _find_column(columns, ["是否解析成功", "parse_success", "success"])
    chars_col = _find_column(columns, ["字数", "chars", "text_chars"])
    pages_col = _find_column(columns, ["页数", "pages", "page_count"])
    blank_col = _find_column(columns, ["空白页数量", "blank_pages"])

    if filename_col is None:
        raise ValueError("pdf_parse_report.xlsx 缺少文件名列，无法定位待 OCR 的 PDF。")

    selected_rows: list[pd.Series] = []
    for _, row in report.iterrows():
        success = _to_bool(row[success_col]) if success_col else False
        chars = _to_int(row[chars_col]) if chars_col else 0
        pages = _to_int(row[pages_col]) if pages_col else 0
        blank_pages = _to_int(row[blank_col]) if blank_col else 0
        failed_by_flag = not success
        scanned_by_empty_text = chars < scanned_text_threshold
        scanned_by_blank_pages = pages > 0 and blank_pages >= pages

        if failed_by_flag or scanned_by_empty_text or scanned_by_blank_pages:
            selected_rows.append(row)

    if not selected_rows:
        return report.iloc[0:0].copy()
    return pd.DataFrame(selected_rows)


def parse_failed_pdfs_with_ocr(
    report_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    output_report_path: Path | None = None,
    log_path: Path | None = None,
    render_dpi: int = 300,
    languages: list[str] | None = None,
    preferred_engine: str = "auto",
    tesseract_cmd: str | None = None,
    psm: int = 6,
    timeout_seconds: int = 180,
    scanned_text_threshold: int = 100,
) -> dict[str, pd.DataFrame]:
    """OCR failed/scanned PDFs listed in the Step 2 parse report."""
    report_path = Path(report_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_path is not None:
        logger = setup_ocr_parse_logger(Path(log_path))
    else:
        logger = _get_ocr_logger()

    output_report_path = Path(output_report_path) if output_report_path else report_path.parent / "ocr_parse_report.xlsx"
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting OCR fallback: report_path=%s input_dir=%s output_dir=%s",
        report_path,
        input_dir,
        output_dir,
    )

    if not report_path.exists():
        error = f"PDF parse report not found: {report_path}"
        logger.error(error)
        summary = pd.DataFrame(
            [
                {
                    "文件名": "",
                    "页数": 0,
                    "OCR成功页数": 0,
                    "OCR失败页数": 0,
                    "OCR总字数": 0,
                    "是否OCR成功": False,
                    "OCR引擎": "none",
                    "错误信息": error,
                    "输出文本路径": "",
                }
            ],
            columns=OCR_SUMMARY_COLUMNS,
        )
        pages = pd.DataFrame(columns=OCR_PAGE_COLUMNS)
        with pd.ExcelWriter(output_report_path) as writer:
            summary.to_excel(writer, index=False, sheet_name="summary")
            pages.to_excel(writer, index=False, sheet_name="page_detail")
        return {"summary": summary, "page_detail": pages}

    parse_report = pd.read_excel(report_path)
    targets = _select_ocr_targets_from_report(
        parse_report,
        scanned_text_threshold=scanned_text_threshold,
    )
    logger.info("OCR target PDFs found: %s", len(targets))

    columns = list(parse_report.columns)
    filename_col = _find_column(columns, ["文件名", "filename", "pdf_filename"])
    if filename_col is None:
        raise ValueError("pdf_parse_report.xlsx 缺少文件名列，无法定位待 OCR 的 PDF。")

    summary_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []

    for _, row in targets.iterrows():
        pdf_name = str(row[filename_col]).strip()
        pdf_path = input_dir / pdf_name
        output_txt_path = output_dir / f"{Path(pdf_name).stem}.txt"

        if not pdf_path.exists():
            error = f"PDF file not found: {pdf_path}"
            logger.error("%s | %s", pdf_name, error)
            summary_rows.append(
                {
                    "文件名": pdf_name,
                    "页数": 0,
                    "OCR成功页数": 0,
                    "OCR失败页数": 0,
                    "OCR总字数": 0,
                    "是否OCR成功": False,
                    "OCR引擎": "none",
                    "错误信息": error,
                    "输出文本路径": str(output_txt_path),
                }
            )
            page_rows.append(
                {
                    "文件名": pdf_name,
                    "页码": "",
                    "OCR识别字符数": 0,
                    "是否识别成功": False,
                    "错误信息": error,
                    "OCR引擎": "none",
                }
            )
            continue

        try:
            result = ocr_pdf_to_text(
                pdf_path=pdf_path,
                output_txt_path=output_txt_path,
                render_dpi=render_dpi,
                languages=languages,
                preferred_engine=preferred_engine,
                tesseract_cmd=tesseract_cmd,
                psm=psm,
                timeout_seconds=timeout_seconds,
            )
            summary_rows.append(result["summary"])
            page_rows.extend(result["pages"])
        except Exception as exc:
            error = str(exc)
            logger.exception("%s | OCR fallback crashed but batch continues: %s", pdf_name, error)
            pages = _to_int(row[_find_column(columns, ["页数", "pages", "page_count"])]) if _find_column(columns, ["页数", "pages", "page_count"]) else 0
            summary_rows.append(
                {
                    "文件名": pdf_name,
                    "页数": pages,
                    "OCR成功页数": 0,
                    "OCR失败页数": pages,
                    "OCR总字数": 0,
                    "是否OCR成功": False,
                    "OCR引擎": "none",
                    "错误信息": error,
                    "输出文本路径": str(output_txt_path),
                }
            )
            if pages:
                for page_number in range(1, pages + 1):
                    page_rows.append(
                        {
                            "文件名": pdf_name,
                            "页码": page_number,
                            "OCR识别字符数": 0,
                            "是否识别成功": False,
                            "错误信息": error,
                            "OCR引擎": "none",
                        }
                    )
            else:
                page_rows.append(
                    {
                        "文件名": pdf_name,
                        "页码": "",
                        "OCR识别字符数": 0,
                        "是否识别成功": False,
                        "错误信息": error,
                        "OCR引擎": "none",
                    }
                )

    summary_report = pd.DataFrame(summary_rows, columns=OCR_SUMMARY_COLUMNS)
    page_report = pd.DataFrame(page_rows, columns=OCR_PAGE_COLUMNS)
    if summary_report.empty:
        summary_report = pd.DataFrame(columns=OCR_SUMMARY_COLUMNS)
    if page_report.empty:
        page_report = pd.DataFrame(columns=OCR_PAGE_COLUMNS)

    with pd.ExcelWriter(output_report_path) as writer:
        summary_report.to_excel(writer, index=False, sheet_name="summary")
        page_report.to_excel(writer, index=False, sheet_name="page_detail")

    failed_count = int((~summary_report["是否OCR成功"]).sum()) if len(summary_report) else 0
    logger.info(
        "OCR fallback finished: targets=%s failed=%s report=%s",
        len(summary_report),
        failed_count,
        output_report_path,
    )
    return {"summary": summary_report, "page_detail": page_report}


def parse_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Backward-compatible wrapper for parsing one PDF."""
    output_txt_path = Path(output_dir) / f"{Path(pdf_path).stem}.txt"
    parse_pdf_to_text(Path(pdf_path), output_txt_path)
    return output_txt_path


def parse_pdf_batch(input_dir: Path, output_dir: Path) -> list[Path]:
    """Backward-compatible wrapper for parsing all PDFs."""
    report = parse_all_pdfs(Path(input_dir), Path(output_dir))
    return [Path(item) for item in report["输出文本路径"].dropna().tolist()]
