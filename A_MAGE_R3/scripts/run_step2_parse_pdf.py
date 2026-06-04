"""Step 2: parse appendix 1 PDFs into page-marked text files."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.pdf_parser import parse_all_pdfs  # noqa: E402


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
                "log_filename": "pdf_parse.log",
            },
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 2 PDF parsing."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    parser_config = config.get("pdf_parser", {})

    input_dir = resolve_project_path(paths["input_pdf_dir"])
    output_dir = resolve_project_path(paths["extracted_text_dir"])
    report_path = resolve_project_path(paths["output_tables_dir"]) / parser_config.get(
        "report_filename", "pdf_parse_report.xlsx"
    )
    log_path = resolve_project_path(paths["output_logs_dir"]) / parser_config.get(
        "log_filename", "pdf_parse.log"
    )

    report = parse_all_pdfs(
        input_dir=input_dir,
        output_dir=output_dir,
        report_path=report_path,
        log_path=log_path,
    )

    success_count = int(report["是否解析成功"].sum()) if len(report) else 0
    failed_count = int((~report["是否解析成功"]).sum()) if len(report) else 0
    print("Step 2 PDF parsing finished.")
    print(f"Input directory: {input_dir}")
    print(f"Extracted text directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Log: {log_path}")
    print(f"Parsed PDFs: {len(report)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
