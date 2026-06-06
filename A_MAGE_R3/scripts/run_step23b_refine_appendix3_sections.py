"""Step 23B: audit and refine Appendix 3 core section recognition."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.appendix3_pipeline import SECTION_REPORT_KEYS, run_appendix3_refine_sections  # noqa: E402


def _missing_by_paper(refined_report: pd.DataFrame) -> str:
    """Return a per-paper missing-section summary."""
    if refined_report.empty:
        return "none"
    lines: list[str] = []
    for _, row in refined_report.iterrows():
        missing = [key for key in SECTION_REPORT_KEYS if key in refined_report.columns and not bool(row[key])]
        lines.append(f"{row['paper_id']}: {', '.join(missing) if missing else 'none'}")
    return "\n".join(lines)


def _status_for_3_2(audit_report: pd.DataFrame, compare_report: pd.DataFrame) -> str:
    """Return the 3-2 results refinement summary."""
    if audit_report.empty:
        return "3-2: no audit rows"
    row = audit_report.loc[audit_report["paper_id"].astype(str) == "3-2"]
    if row.empty:
        return "3-2: not found in audit"
    status = str(row.iloc[0].get("results_status", ""))
    confidence = str(row.iloc[0].get("results_confidence", ""))
    note = str(row.iloc[0].get("results_note", ""))
    compare = compare_report[
        (compare_report["paper_id"].astype(str) == "3-2")
        & (compare_report["section_name"].astype(str) == "results")
    ]
    status_change = str(compare.iloc[0].get("status_change", "")) if not compare.empty else ""
    return f"3-2 results status={status}, confidence={confidence}, status_change={status_change}, note={note}"


def main() -> None:
    """Run Appendix 3 section quality audit and results refinement."""
    result = run_appendix3_refine_sections()
    refined_report = result["refined_report"]
    compare_report = result["compare_report"]
    audit_report = result["audit_report"]

    parse_failed = audit_report.loc[audit_report["parse_status"].astype(str) == "parse_failed", "paper_id"].tolist()
    need_manual = audit_report.loc[audit_report["need_manual_check"].astype(bool), "paper_id"].tolist()

    print("Step 23B Appendix 3 section audit and results refinement completed.")
    print(f"Paper total: {len(audit_report)}")
    print(f"Parse failed papers: {', '.join(parse_failed) if parse_failed else 'none'}")
    print("Refined missing sections by paper:")
    print(_missing_by_paper(refined_report))
    print(_status_for_3_2(audit_report, compare_report))
    print(f"Need manual check: {', '.join(str(item) for item in need_manual) if need_manual else 'none'}")
    print(f"Refined sections dir: {result['refined_sections_dir']}")
    print(f"Refined section report: {result['refined_report_path']}")
    print(f"Compare report: {result['compare_report_path']}")
    print(f"Quality audit report: {result['audit_report_path']}")
    print(f"Log file: {result['log_path']}")


if __name__ == "__main__":
    main()
