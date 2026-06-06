"""Run Step 21: generate Problem 2 paper-writing materials."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.problem2_report_generator import run_step21_problem2_draft  # noqa: E402


def main() -> None:
    """Execute Step 21 and print generated files."""
    result = run_step21_problem2_draft(PROJECT_ROOT / "config.yaml")
    print("Step 21 finished: generated Problem 2 writing materials")
    for key, path in result["output_paths"].items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
