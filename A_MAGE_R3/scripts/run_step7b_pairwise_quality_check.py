"""Step 7B: check pairwise comparison input quality before BT fitting."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.bradley_terry import check_pairwise_quality  # noqa: E402


def main() -> None:
    """Run pairwise quality checks only; do not fit Bradley-Terry."""
    tables_dir = PROJECT_ROOT / "output/tables"
    logs_dir = PROJECT_ROOT / "output/logs"
    pairwise_path = tables_dir / "pairwise_comparison_filled.xlsx"
    output_path = tables_dir / "pairwise_quality_check.xlsx"
    log_path = logs_dir / "pairwise_quality_check.log"

    result = check_pairwise_quality(
        pairwise_path=pairwise_path,
        output_path=output_path,
        log_path=log_path,
        minimum_valid_comparisons_per_paper=2,
    )

    print("Pairwise quality check finished.")
    print(f"Input: {pairwise_path}")
    print(f"Report: {output_path}")
    print(f"Log: {log_path}")
    print(f"Valid comparison count: {result['valid_count']}")
    print(f"Tie count: {result['tie_count']}")
    print(f"Blank winner count: {result['blank_count']}")
    print(f"Invalid winner count: {result['invalid_count']}")
    print(f"Undercovered papers: {', '.join(result['undercovered_papers']) if result['undercovered_papers'] else 'none'}")
    print(f"Comparison graph connected: {result['is_connected']}")


if __name__ == "__main__":
    main()
