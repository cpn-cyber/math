"""Run Step 20: final Problem 2 consistency audit."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.problem2_report_generator import run_step20_problem2_final_audit  # noqa: E402


def main() -> None:
    """Execute Step 20 and print a compact summary."""
    result = run_step20_problem2_final_audit(PROJECT_ROOT / "config.yaml")
    summary = result["audit_summary"]
    fail_count = int(summary["status"].eq("FAIL").sum())
    pass_count = int(summary["status"].eq("PASS").sum())

    print("Step 20 finished: Problem 2 final audit")
    print(f"PASS count: {pass_count}")
    print(f"FAIL count: {fail_count}")
    print(result["final_statement"])
    if fail_count:
        print("Failed audit items:")
        failed = summary.loc[summary["status"].eq("FAIL"), ["category", "item", "details"]]
        for _, row in failed.iterrows():
            print(f"  - {row['category']} / {row['item']}: {row['details']}")

    print("Key output files:")
    for key in ["audit_path", "log_path", "key_md_path", "warnings_md_path"]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
