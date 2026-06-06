"""Compatibility runner for Problem 3 Step 27 reviewer-agent analysis."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_agent_subjectivity import run_multi_agent_subjectivity  # noqa: E402


def main() -> None:
    """Delegate to the implemented Step 27 subjectivity analysis."""
    result = run_multi_agent_subjectivity(PROJECT_ROOT / "config.yaml")
    print("Step 27 reviewer-agent subjectivity analysis finished.")
    print(f"Analysis table: {result['paths']['analysis_table']}")


if __name__ == "__main__":
    main()
