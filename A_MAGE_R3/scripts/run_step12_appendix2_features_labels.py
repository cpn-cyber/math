"""Run Step 12: Appendix 2 surface features and weak Q labels."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.quality_label_builder import SURFACE_FEATURE_COLUMNS, run_appendix2_features_labels  # noqa: E402


def main() -> None:
    """Execute Step 12 and print a compact report."""
    result = run_appendix2_features_labels(PROJECT_ROOT / "config.yaml")
    raw = result["raw_features"]
    q_labels = result["q_labels"]
    candidate_usage = result["candidate_usage"]
    missing_features = result["missing_features"]

    print("Step 12 finished: Appendix 2 surface features and weak labels")
    print(f"papers: {len(raw)}")
    print(f"surface feature columns: {len(SURFACE_FEATURE_COLUMNS)}")
    if len(q_labels):
        print(f"Q_label range: {q_labels['Q_label'].min():.6f} - {q_labels['Q_label'].max():.6f}")
    print(f"candidate usage rows: {len(candidate_usage)}")
    print(f"missing feature columns: {missing_features if missing_features else 'none'}")
    print(f"score method: {result['score_method']}")
    print(f"sections used: {result['sections_dir']}")
    print("outputs:")
    for key in [
        "raw_output_path",
        "normalized_output_path",
        "q_output_path",
        "features_with_q_path",
        "candidate_report_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
