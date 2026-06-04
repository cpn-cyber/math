"""Entry point for A_MAGE_R3 problem 1 pipeline.

Step 1 only prepares the project structure. Later steps will call the
corresponding scripts under ``scripts/``.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Print the available pipeline steps."""
    print("A_MAGE_R3 Problem 1 pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1: python scripts/run_step1_setup.py")
    print("Step 2: python scripts/run_step2_parse_pdf.py")
    print("Step 3: python scripts/run_step3_split_sections.py")
    print("Step 4: python scripts/run_step4_extract_features.py")
    print("Step 5: python scripts/run_step5_weighting.py")
    print("Step 6: python scripts/run_step6_topsis.py")
    print("Step 7: python scripts/run_step7_bradley_terry.py")
    print("Step 8: python scripts/run_step8_grade_visualize.py")


if __name__ == "__main__":
    main()
