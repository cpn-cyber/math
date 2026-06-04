"""Step 4: extract secondary indicator features."""

from pathlib import Path
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for partially installed envs.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.feature_extractor import DEFAULT_CONFIG, FEATURE_COLUMNS, extract_all_features  # noqa: E402


def load_config(config_path: Path) -> dict:
    """Load YAML configuration, or return Step 4 defaults if PyYAML is absent."""
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
            },
            "feature_extraction": DEFAULT_CONFIG["feature_extraction"],
            "logic_connectives": DEFAULT_CONFIG["logic_connectives"],
        }

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.setdefault("feature_extraction", DEFAULT_CONFIG["feature_extraction"])
    config.setdefault("logic_connectives", DEFAULT_CONFIG["logic_connectives"])
    return config


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path


def main() -> None:
    """Run Step 4 feature extraction."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    paths = config["paths"]
    splitter_config = config.get("section_splitter", {})
    feature_config = config.get("feature_extraction", {})

    sections_dir = resolve_project_path(paths["intermediate_dir"]) / splitter_config.get(
        "sections_subdir", "sections"
    )
    text_dir = resolve_project_path(paths["extracted_text_dir"])
    raw_output_path = resolve_project_path(paths["output_tables_dir"]) / feature_config.get(
        "raw_feature_filename", "appendix1_features_raw.xlsx"
    )
    normalized_output_path = resolve_project_path(paths["output_tables_dir"]) / feature_config.get(
        "normalized_feature_filename", "appendix1_features_normalized.xlsx"
    )
    log_path = resolve_project_path(paths["output_logs_dir"]) / feature_config.get(
        "log_filename", "feature_extraction.log"
    )

    raw_table, normalized_table = extract_all_features(
        sections_dir=sections_dir,
        text_dir=text_dir,
        raw_output_path=raw_output_path,
        normalized_output_path=normalized_output_path,
        config=config,
        log_path=log_path,
    )

    generated_features = [column for column in FEATURE_COLUMNS if column in normalized_table.columns]
    missing_cells = int(normalized_table[FEATURE_COLUMNS].isna().sum().sum()) if len(normalized_table) else 0

    print("Step 4 feature extraction finished.")
    print(f"Sections directory: {sections_dir}")
    print(f"Extracted text directory: {text_dir}")
    print(f"Raw feature table: {raw_output_path}")
    print(f"Normalized feature table: {normalized_output_path}")
    print(f"Log: {log_path}")
    print(f"Papers processed: {len(raw_table)}")
    print(f"Feature columns generated: {len(generated_features)}/{len(FEATURE_COLUMNS)}")
    print(f"Missing normalized feature cells: {missing_cells}")


if __name__ == "__main__":
    main()
