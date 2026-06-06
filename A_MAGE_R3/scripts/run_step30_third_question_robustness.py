"""Run Problem 3 Step 30: robustness analysis."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.third_question_robustness import run_third_question_robustness  # noqa: E402


def main() -> None:
    """Execute Step 30 and print a compact summary."""
    result = run_third_question_robustness(PROJECT_ROOT / "config.yaml")
    paper_stability = result["tables"]["paper_level_stability"]
    action_stability = result["tables"]["action_stability"]
    risk_stability = result["tables"]["risk_reduction_stability"]
    disagreement_stability = result["tables"]["disagreement_stability"]
    paths = result["paths"]

    print("Step 30 third-question robustness analysis finished.")
    print("Command: python A_MAGE_R3/scripts/run_step30_third_question_robustness.py")

    print("\nUsed data sources:")
    for path in result["used_sources"]:
        print(f"- {path}")

    print(f"\nParameter combinations: {result['parameter_combo_count']}")
    print(
        "Average predicted score gain interval: "
        f"{result['score_gain_interval'][0]:.6f} ~ {result['score_gain_interval'][1]:.6f}"
    )
    print(f"Positive gain proportion: {result['positive_gain_proportion']:.6f}")

    print("\nPass-or-above ratios:")
    print(paper_stability[["paper_id", "pass_or_above_ratio", "mean_score_gain", "min_score_gain", "max_score_gain"]].to_string(index=False))

    print("\nTop stable actions:")
    print(action_stability[["action_id", "selected_count", "selection_rate"]].head(5).to_string(index=False))

    print(
        "\nAverage risk reduction interval: "
        f"{result['risk_reduction_interval'][0]:.6f} ~ {result['risk_reduction_interval'][1]:.6f}"
    )
    print(
        "Average disagreement std reduction interval: "
        f"{result['std_reduction_interval'][0]:.6f} ~ {result['std_reduction_interval'][1]:.6f}"
    )

    print("\nRisk reduction stability:")
    print(risk_stability[["paper_id", "mean_risk_reduction", "proportion_risk_decrease"]].to_string(index=False))

    print("\nDisagreement stability:")
    print(disagreement_stability[["paper_id", "mean_std_reduction", "ratio_high_to_mid_or_low"]].to_string(index=False))

    print("\nOutputs:")
    for key, path in paths.items():
        if key.endswith("_dir"):
            continue
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

