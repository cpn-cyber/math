"""Run Problem 3 Step 28: revision action optimization."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.revision_action_optimizer import run_revision_action_optimization  # noqa: E402


def main() -> None:
    """Execute Step 28 and print a compact summary."""
    result = run_revision_action_optimization(PROJECT_ROOT / "config.yaml")
    plan = result["plan"]
    summaries = result["summaries"]
    paths = result["paths"]

    print("Step 28 revision action optimization finished.")
    print("Command: python A_MAGE_R3/scripts/run_step28_revision_action_optimization.py")
    print(f"Revision budget: {result['budget']}")

    print("\nUsed data sources:")
    for path in result["used_sources"]:
        print(f"- {path}")

    print("\nRecommended revision plans:")
    print(
        plan[
            [
                "paper_id",
                "selected_actions_knapsack",
                "expected_total_gain_knapsack",
                "total_cost_knapsack",
                "revision_priority_level",
            ]
        ].to_string(index=False)
    )

    print("\nTop recommended actions:")
    print(
        summaries["action_frequency"][
            [
                "action_id",
                "action_name",
                "recommended_count",
                "avg_adjusted_expected_gain",
            ]
        ].head(5).to_string(index=False)
    )

    print("\nOutputs:")
    for key, path in paths.items():
        if key.endswith("_dir"):
            continue
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
