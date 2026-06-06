"""Run Problem 3 Step 24: current Appendix 3 quality reevaluation."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.appendix3_current_evaluator import run_appendix3_current_evaluation  # noqa: E402


def main() -> None:
    """Execute Step 24 and print a compact summary."""
    result = run_appendix3_current_evaluation(PROJECT_ROOT / "config.yaml")
    evaluation = result["evaluation"]
    low_report = result["low_report"]
    paths = result["paths"]

    print("Step 24 Appendix 3 current evaluation finished.")
    print("Command: python A_MAGE_R3/scripts/run_step24_appendix3_current_eval.py")
    print("\nCurrent evaluation:")
    print(
        evaluation[
            [
                "paper_id",
                "F1_score",
                "F2_key_score",
                "Q_cur_baseline",
                "rank_current",
                "current_grade",
                "grade_consistency_with_prompt",
            ]
        ].to_string(index=False)
    )
    print("\nMain low features:")
    for paper_id in evaluation["paper_id"].astype(str).tolist():
        subset = low_report.loc[low_report["paper_id"].astype(str) == paper_id].head(5)
        summary = "; ".join(
            f"{row.feature_name}({row.weakness_level})" for row in subset.itertuples(index=False)
        )
        print(f"- {paper_id}: {summary or 'no severe low feature detected'}")
    print("\nMissing Problem1 features:", result["missing_problem1_features"])
    print("Missing key features:", result["missing_key_features"])
    print("\nOutputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
