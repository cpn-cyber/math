"""Run Problem 3 Step 26: AI-assisted writing risk D-S fusion."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ai_risk_ds_fusion import run_ai_risk_ds_fusion  # noqa: E402


def main() -> None:
    """Execute Step 26 and print a compact summary."""
    result = run_ai_risk_ds_fusion(PROJECT_ROOT / "config.yaml")
    fusion = result["fusion"]
    evidence = result["evidence"]
    paths = result["paths"]

    print("Step 26 AI-assisted writing risk D-S fusion finished.")
    print("Command: python A_MAGE_R3/scripts/run_step26_ai_risk_ds_fusion.py")
    print("\nRisk fusion results:")
    print(
        fusion[
            [
                "paper_id",
                "R_AI",
                "risk_level",
                "main_risk_source",
                "conflict_K_max",
                "uncertainty_mass_U",
            ]
        ].to_string(index=False)
    )
    print("\nEvidence scores:")
    print(
        evidence[
            [
                "paper_id",
                "e1_template_expression",
                "e2_unsupported_conclusion",
                "e3_data_untraceable",
                "e4_method_result_jump",
            ]
        ].to_string(index=False)
    )
    high_conflict = fusion.loc[fusion["conflict_K_max"].ge(0.60), ["paper_id", "conflict_K_max"]]
    print("\nHigh D-S conflict:", "none" if high_conflict.empty else high_conflict.to_dict(orient="records"))
    print("\nDisclaimer written:", fusion["disclaimer"].iloc[0])
    print("\nOutputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
