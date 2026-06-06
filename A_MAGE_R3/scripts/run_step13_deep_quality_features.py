"""Run Step 13: deep quality correction feature extraction."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.deep_quality_features import DEEP_FEATURE_COLUMNS, run_step13_deep_quality_features  # noqa: E402


def main() -> None:
    """Execute Step 13 and print a concise summary."""
    result = run_step13_deep_quality_features(PROJECT_ROOT / "config.yaml")
    auto_table = result["auto_table"]
    high_stacking = result["high_stacking"]
    review_focus = result["review_focus"]
    review_suggestions = result["review_suggestions"]

    print("Step 13 finished: Appendix 2 deep quality correction features")
    print(f"papers: {len(auto_table)}")
    print(f"deep feature columns: {len(DEEP_FEATURE_COLUMNS)}")
    print("feature ranges:")
    for feature_name, bounds in result["feature_ranges"].items():
        print(f"  - {feature_name}: {bounds['min']:.6f} - {bounds['max']:.6f}")

    print("high stacking_penalty papers:")
    if high_stacking.empty:
        print("  - none")
    else:
        for _, row in high_stacking.iterrows():
            print(f"  - {row['paper_id']} ({row['filename']}): {row['stacking_penalty']:.6f}")

    print("review_focus papers:")
    if review_focus.empty:
        print("  - none")
    else:
        for _, row in review_focus.iterrows():
            print(f"  - {row['paper_id']} ({row['filename']}): {row['step12b_flag']}")

    print(f"review suggestion rows: {len(review_suggestions)}")
    print("outputs:")
    print(f"  - {result['output_path']}")
    print(f"  - {result['review_template_path']}")
    print(f"  - {result['evidence_path']}")
    print(f"  - {result['log_path']}")


if __name__ == "__main__":
    main()
