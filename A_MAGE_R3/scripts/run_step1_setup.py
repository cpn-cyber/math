"""Step 1: verify project structure and base files."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "data",
    "data/appendix1_papers",
    "data/extracted_text",
    "data/intermediate",
    "modules",
    "output",
    "output/tables",
    "output/charts",
    "output/logs",
    "scripts",
]

REQUIRED_FILES = [
    "main_problem1.py",
    "config.yaml",
    "requirements.txt",
    "README.md",
    "modules/__init__.py",
    "modules/pdf_parser.py",
    "modules/section_splitter.py",
    "modules/feature_extractor.py",
    "modules/weighting.py",
    "modules/topsis.py",
    "modules/bradley_terry.py",
    "modules/grade_classifier.py",
    "modules/visualization.py",
    "scripts/run_step1_setup.py",
    "scripts/run_step2_parse_pdf.py",
    "scripts/run_step3_split_sections.py",
    "scripts/run_step4_extract_features.py",
    "scripts/run_step5_weighting.py",
    "scripts/run_step6_topsis.py",
    "scripts/run_step7_bradley_terry.py",
    "scripts/run_step8_grade_visualize.py",
]


def check_project_structure() -> bool:
    """Check whether all required directories and files exist."""
    missing_dirs = [item for item in REQUIRED_DIRS if not (PROJECT_ROOT / item).is_dir()]
    missing_files = [item for item in REQUIRED_FILES if not (PROJECT_ROOT / item).is_file()]

    if missing_dirs:
        print("Missing directories:")
        for item in missing_dirs:
            print(f"  - {item}")

    if missing_files:
        print("Missing files:")
        for item in missing_files:
            print(f"  - {item}")

    if not missing_dirs and not missing_files:
        print("Step 1 setup check passed. Project structure is complete.")
        return True

    print("Step 1 setup check failed.")
    return False


def main() -> None:
    """Run Step 1 setup verification."""
    ok = check_project_structure()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
