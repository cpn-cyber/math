"""Step 2B: OCR fallback parsing for scanned or failed PDFs."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.pdf_parser import parse_failed_pdfs_with_ocr  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    if yaml is None:
        return {
            "paths": {
                "input_pdf_dir": "data/appendix1_papers",
                "extracted_text_dir": "data/extracted_text",
                "output_tables_dir": "output/tables",
                "output_logs_dir": "output/logs",
            },
            "pdf_parser": {
                "report_filename": "pdf_parse_report.xlsx",
                "ocr_report_filename": "ocr_parse_report.xlsx",
                "ocr_log_filename": "ocr_parse.log",
                "ocr_engine": "auto",
                "ocr_render_dpi": 300,
                "ocr_languages": ["chi_sim", "eng"],
                "ocr_psm": 6,
                "ocr_timeout_seconds": 180,
                "tesseract_path": "C:/Program Files/Tesseract-OCR/tesseract.exe",
                "scanned_pdf_text_threshold": 100,
            },
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 2B OCR fallback parsing."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    parser_config = config.get("pdf_parser", {})

    input_dir = resolve_project_path(paths["input_pdf_dir"])
    output_dir = resolve_project_path(paths["extracted_text_dir"])
    tables_dir = resolve_project_path(paths["output_tables_dir"])
    logs_dir = resolve_project_path(paths["output_logs_dir"])

    report_path = tables_dir / parser_config.get("report_filename", "pdf_parse_report.xlsx")
    ocr_report_path = tables_dir / parser_config.get("ocr_report_filename", "ocr_parse_report.xlsx")
    log_path = logs_dir / parser_config.get("ocr_log_filename", "ocr_parse.log")

    result = parse_failed_pdfs_with_ocr(
        report_path=report_path,
        input_dir=input_dir,
        output_dir=output_dir,
        output_report_path=ocr_report_path,
        log_path=log_path,
        render_dpi=int(parser_config.get("ocr_render_dpi", 300)),
        languages=list(parser_config.get("ocr_languages", ["chi_sim", "eng"])),
        preferred_engine=str(parser_config.get("ocr_engine", "auto")),
        tesseract_cmd=parser_config.get("tesseract_path")
        or parser_config.get("tesseract_cmd")
        or None,
        psm=int(parser_config.get("ocr_psm", 6)),
        timeout_seconds=int(parser_config.get("ocr_timeout_seconds", 180)),
        scanned_text_threshold=int(parser_config.get("scanned_pdf_text_threshold", 100)),
    )

    summary = result["summary"]
    page_detail = result["page_detail"]
    total_targets = len(summary)
    success_count = int(summary["是否OCR成功"].sum()) if total_targets else 0
    failed_count = int((~summary["是否OCR成功"]).sum()) if total_targets else 0
    success_pages = int(page_detail["是否识别成功"].sum()) if len(page_detail) else 0
    failed_pages = int((~page_detail["是否识别成功"]).sum()) if len(page_detail) else 0

    print("Step 2B OCR fallback finished.")
    print(f"Input directory: {input_dir}")
    print(f"Source parse report: {report_path}")
    print(f"Extracted text directory: {output_dir}")
    print(f"OCR report: {ocr_report_path}")
    print(f"OCR log: {log_path}")
    print(f"OCR target PDFs: {total_targets}")
    print(f"OCR success PDFs: {success_count}")
    print(f"OCR failed PDFs: {failed_count}")
    print(f"OCR success pages: {success_pages}")
    print(f"OCR failed pages: {failed_pages}")

    if failed_count:
        print(
            "Notice: OCR failed for at least one PDF. Check output/logs/ocr_parse.log. "
            "If no local OCR engine is installed, install Tesseract OCR with chi_sim "
            "language data, or install/configure PaddleOCR."
        )


if __name__ == "__main__":
    main()
