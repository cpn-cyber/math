"""Run Step 19: pairwise ranking auxiliary check."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.pairwise_ranking_check import run_step19_pairwise_ranking_check  # noqa: E402


def main() -> None:
    """Execute Step 19 and print a compact summary."""
    result = run_step19_pairwise_ranking_check(PROJECT_ROOT / "config.yaml")
    summary = result["check"].iloc[0]
    pair_21 = result["paper_21"]

    print("Step 19 finished: pairwise ranking auxiliary check")
    print(f"Total pairs: {int(summary['total_pairs'])}")
    print(f"Near-tie pairs: {int(summary['near_tie_pairs'])}")
    print(f"Key features used: {summary['key_features_used']}")
    print(f"Overall pairwise accuracy: {summary['overall_pairwise_accuracy']:.6f}")
    print(f"Group validation mean accuracy: {summary['group_validation_accuracy_mean']:.6f}")
    print(f"Group validation std: {summary['group_validation_accuracy_std']:.6f}")
    print(f"Accuracy without near-tie: {summary['accuracy_without_near_tie']:.6f}")
    if not pair_21.empty:
        row = pair_21.iloc[0]
        print(
            "2-1 heldout group: "
            f"accuracy={row['pairwise_accuracy']:.6f}, "
            f"accuracy_without_near_tie={row['accuracy_without_near_tie']:.6f}, "
            f"near_tie_count={int(row['near_tie_count'])}, "
            f"high_influence={row['high_influence_flag']}"
        )
    print(f"Supports K_final key-feature conclusion: {result['supports']}")
    print(f"Conclusion: {summary['conclusion']}")
    print("outputs:")
    for key in [
        "dataset_path",
        "check_path",
        "group_path",
        "accuracy_chart_path",
        "weight_chart_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
