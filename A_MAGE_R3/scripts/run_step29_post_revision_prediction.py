"""Compatibility runner for Problem 3 Step 29 post-revision prediction."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality_prediction_after_revision import run_quality_prediction_after_revision  # noqa: E402


def main() -> None:
    """Delegate to the implemented Step 29 quality prediction."""
    result = run_quality_prediction_after_revision(PROJECT_ROOT / "config.yaml")
    print("Step 29 post-revision quality prediction finished.")
    print(f"Prediction table: {result['paths']['prediction']}")


if __name__ == "__main__":
    main()
