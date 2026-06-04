"""Step 3: split extracted text into structural sections."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.section_splitter import DEFAULT_SECTION_KEYWORDS, split_all_texts  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration, or return Step 3 defaults if PyYAML is absent."""
    if yaml is None:
        return {
            "paths": {
                "extracted_text_dir": "data/extracted_text",
                "intermediate_dir": "data/intermediate",
                "output_tables_dir": "output/tables",
                "output_logs_dir": "output/logs",
            },
            "section_splitter": {
                "sections_subdir": "sections",
                "report_filename": "section_split_report.xlsx",
                "log_filename": "section_split.log",
            },
            "section_keywords": DEFAULT_SECTION_KEYWORDS,
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("section_keywords", DEFAULT_SECTION_KEYWORDS)
    return config


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 3 section splitting."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    splitter_config = config.get("section_splitter", {})

    input_dir = resolve_project_path(paths["extracted_text_dir"])
    sections_dir = resolve_project_path(paths["intermediate_dir"]) / splitter_config.get(
        "sections_subdir", "sections"
    )
    report_path = resolve_project_path(paths["output_tables_dir"]) / splitter_config.get(
        "report_filename", "section_split_report.xlsx"
    )
    log_path = resolve_project_path(paths["output_logs_dir"]) / splitter_config.get(
        "log_filename", "section_split.log"
    )

    report = split_all_texts(
        input_dir=input_dir,
        output_dir=sections_dir,
        section_keywords=config.get("section_keywords", DEFAULT_SECTION_KEYWORDS),
        report_path=report_path,
        log_path=log_path,
    )

    detected_count = 0
    if len(report):
        has_columns = [column for column in report.columns if column.startswith("是否有")]
        detected_count = int((report[has_columns].any(axis=1)).sum())
    success_rate = detected_count / len(report) if len(report) else 0.0

    print("Step 3 section splitting finished.")
    print(f"Input directory: {input_dir}")
    print(f"Section JSON directory: {sections_dir}")
    print(f"Report: {report_path}")
    print(f"Log: {log_path}")
    print(f"Processed text files: {len(report)}")
    print(f"Section recognition success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
