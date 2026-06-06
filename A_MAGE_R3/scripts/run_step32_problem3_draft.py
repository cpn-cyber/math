"""Step 32: generate Problem 3 paper writing materials."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.problem3_report_generator import generate_problem3_draft  # noqa: E402


def main() -> None:
    """Generate Markdown materials for Problem 3."""
    result = generate_problem3_draft(PROJECT_ROOT)
    print("Step 32 completed.")
    print(f"Status: {result['status']}")
    print(f"Output directory: {result['output_dir']}")
    print("Output files:")
    for file_path in result["output_files"]:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
