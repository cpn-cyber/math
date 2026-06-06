"""Run Problem 3 Step 31: final audit."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.problem3_final_audit import run_problem3_final_audit  # noqa: E402


def main() -> None:
    """Execute Step 31 and print a compact summary."""
    result = run_problem3_final_audit(PROJECT_ROOT / "config.yaml")
    paths = result["paths"]

    print("Step 31 Problem 3 final audit finished.")
    print("Command: python A_MAGE_R3/scripts/run_step31_problem3_final_audit.py")
    print(f"Audit status counts: {result['status_counts']}")
    print("Core missing/problem files:", "none" if result["missing_core"].empty else result["missing_core"][["file_name", "status"]].to_dict(orient="records"))
    print(f"Data consistency pass: {result['data_consistency_pass']}")
    print(f"Disclaimer complete: {result['disclaimer_complete']}")
    print(f"Paper-ready conclusion count: {result['paper_ready_conclusion_count']}")
    print("\nRecommended key figures Top 5:")
    for figure in result["top_figures"]:
        print(f"- {figure}")

    print("\nOutputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()

