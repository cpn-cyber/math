"""Run Problem 3 Step 27: multi-agent reviewer subjectivity analysis."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multi_agent_subjectivity import run_multi_agent_subjectivity  # noqa: E402


def main() -> None:
    """Execute Step 27 and print a compact audit summary."""
    result = run_multi_agent_subjectivity(PROJECT_ROOT / "config.yaml")
    analysis = result["analysis"]
    mapping = result["mapping"]
    ai_relation = result["ai_relation"]
    paths = result["paths"]

    print("Step 27 multi-agent subjectivity analysis finished.")
    print("Command: python A_MAGE_R3/scripts/run_step27_multi_agent_subjectivity.py")

    print("\nIdentified score/review candidate files:")
    for path in result["score_sources"]:
        print(f"- {path}")

    print("\nAgent-like reviewer dimensions:")
    print(mapping[["agent", "agent_label", "feature_group", "source_columns"]].to_string(index=False))

    print("\nPaper disagreement overview:")
    print(
        analysis[
            [
                "paper_id",
                "agent_score_mean",
                "agent_score_std",
                "agent_score_range",
                "disagreement_level",
                "max_score_agent",
                "min_score_agent",
                "R_AI",
                "risk_level",
            ]
        ].to_string(index=False)
    )

    print(f"\nAverage agent score std: {result['mean_std']:.6f}")
    print(f"High-disagreement paper count: {result['high_disagreement_count']}")
    print(f"Highest-disagreement paper: {result['highest_disagreement_paper']}")

    if not ai_relation.empty:
        focus = ai_relation[
            (ai_relation["agent"] == "application_value_reviewer")
            & (ai_relation["risk_or_evidence"].isin(["R_AI", "e1_template_expression"]))
        ]
        print("\nAI risk and writing/application reviewer relation:")
        print(focus[["risk_or_evidence", "spearman_corr", "interpretation"]].to_string(index=False))

    print("\nOutputs:")
    for key, path in paths.items():
        if key.endswith("_dir"):
            continue
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

