"""Step 11B: audit and refine Appendix 2 section recognition."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.appendix2_pipeline import run_appendix2_refine_sections  # noqa: E402


def _format_pairs(items: list[dict[str, str]]) -> str:
    """Format paper-section records for console output."""
    if not items:
        return "none"
    return ", ".join(f"{item['paper_id']}:{item['section_name']}" for item in items)


def main() -> None:
    """Run Step 11B and print the required summary."""
    result = run_appendix2_refine_sections()
    missing = result["missing_counts"]
    original_missing = result["original_missing_counts"]
    missing_to_found = result["missing_to_found"]
    candidates = result["candidates"]

    print("Step 11B Appendix 2 section audit and refinement completed.")
    for section in ["abstract", "assumptions", "results"]:
        before = original_missing.get(section, "NA")
        after = missing.get(section, "NA")
        print(f"{section} missing: {before} -> {after}")
    print("Missing to found: " + _format_pairs(missing_to_found))
    print("Candidate only: " + _format_pairs(candidates))
    print(f"Refined sections dir: {result['refined_sections_dir']}")
    print(f"Refined split report: {result['refined_report_path']}")
    print(f"Compare report: {result['compare_report_path']}")
    print(f"Quality audit report: {result['audit_report_path']}")
    print(f"Log file: {result['log_path']}")

    can_enter_step12 = missing.get("abstract", 0) == 0 and missing.get("results", 0) == 0
    print("Can enter Step 12: " + ("yes" if can_enter_step12 else "needs review first"))


if __name__ == "__main__":
    main()
