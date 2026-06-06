"""Compatibility runner for Problem 3 Step 30 robustness analysis."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.third_question_robustness import run_third_question_robustness  # noqa: E402


def main() -> None:
    """Delegate to the implemented Step 30 robustness analysis."""
    result = run_third_question_robustness(PROJECT_ROOT / "config.yaml")
    print("Step 30 Problem 3 robustness analysis finished.")
    print(f"Robustness summary: {result['paths']['summary']}")


if __name__ == "__main__":
    main()
