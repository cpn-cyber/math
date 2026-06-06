"""Run Problem 3 Step 29: quality prediction after revision."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality_prediction_after_revision import run_quality_prediction_after_revision  # noqa: E402


def main() -> None:
    """Execute Step 29 and print a compact summary."""
    result = run_quality_prediction_after_revision(PROJECT_ROOT / "config.yaml")
    predictions = result["predictions"]
    paths = result["paths"]

    print("Step 29 quality prediction after revision finished.")
    print("Command: python A_MAGE_R3/scripts/run_step29_quality_prediction_after_revision.py")

    print("\nUsed data sources:")
    for path in result["used_sources"]:
        print(f"- {path}")

    print("\nPrediction overview:")
    print(
        predictions[
            [
                "paper_id",
                "current_score",
                "score_gain",
                "predicted_score_after_revision",
                "R_AI_before",
                "R_AI_after_pred",
                "agent_score_std_before",
                "agent_score_std_after_pred",
            ]
        ].to_string(index=False)
    )

    print(f"\nAverage predicted score gain: {result['avg_score_gain']:.6f}")
    print(f"Average predicted AI-risk reduction: {result['avg_risk_reduction']:.6f}")
    print(f"Average predicted disagreement std reduction: {result['avg_disagreement_reduction']:.6f}")

    print("\nOutputs:")
    for key, path in paths.items():
        if key.endswith("_dir"):
            continue
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

