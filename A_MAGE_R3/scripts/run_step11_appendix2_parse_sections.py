"""Step 11: parse Appendix 2 PDFs with OCR fallback and split sections."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.appendix2_pipeline import SECTION_REPORT_KEYS, run_appendix2_parse_sections  # noqa: E402


def _format_missing_sections(section_report: pd.DataFrame) -> str:
    """Return a compact missing-section summary."""
    if section_report.empty:
        return "none"
    missing_counts = {
        key: int((~section_report[key].astype(bool)).sum())
        for key in SECTION_REPORT_KEYS
        if key in section_report.columns
    }
    ordered = sorted(missing_counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{key}={count}" for key, count in ordered if count > 0) or "none"


def main() -> None:
    """Run Appendix 2 PDF parsing and section splitting."""
    result = run_appendix2_parse_sections()
    pdf_report = result["pdf_report"]
    section_report = result["section_report"]

    total = len(pdf_report)
    success = int(pdf_report["parse_success"].sum()) if total else 0
    ocr_count = int(pdf_report["ocr_used"].sum()) if total else 0
    failed = pdf_report.loc[~pdf_report["parse_success"].astype(bool), "filename"].tolist() if total else []

    print("Step 11 Appendix 2 parsing and section splitting completed.")
    print(f"PDF total: {total}")
    print(f"Parse success: {success}")
    print(f"OCR used: {ocr_count}")
    print(f"Parse failed: {len(failed)}")
    if failed:
        print("Failed PDFs: " + ", ".join(str(item) for item in failed))
    print(f"Copied PDFs from workspace fallback: {result['copied_pdfs']}")
    print(f"Common missing sections: {_format_missing_sections(section_report)}")
    print(f"Text output dir: {result['extracted_text_dir']}")
    print(f"Sections output dir: {result['sections_dir']}")
    print(f"PDF parse report: {result['pdf_report_path']}")
    print(f"Section split report: {result['section_report_path']}")
    print(f"Log file: {result['log_path']}")


if __name__ == "__main__":
    main()
