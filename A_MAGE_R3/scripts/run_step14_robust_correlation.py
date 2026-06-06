"""Run Step 14: robust scaling and correlation analysis."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.correlation_analysis import run_step14_robust_correlation  # noqa: E402


def main() -> None:
    """Execute Step 14 and print a compact report."""
    result = run_step14_robust_correlation(PROJECT_ROOT / "config.yaml")
    feature_matrix = result["feature_matrix"]
    caution = result["caution_features"]
    top8 = result["top8"]

    print("Step 14 finished: robust scaling and correlation analysis")
    print(f"paper_count: {len(feature_matrix)}")
    print(f"merged_feature_count: {len([c for c in result['variance_filter']['feature_name']])}")
    print("caution/excluded features:")
    if caution.empty:
        print("  - none")
    else:
        for _, row in caution.iterrows():
            print(f"  - {row['feature_name']}: {row['variance_flag']} / use_in_model={row['use_in_model']}")
    print("Spearman abs Top 8:")
    for _, row in top8.iterrows():
        print(f"  - {row['feature_name']}: spearman={row['spearman_corr']:.6f}, abs={row['spearman_abs']:.6f}")
    print("outputs:")
    for key in [
        "raw_output_path",
        "scaled_output_path",
        "correlation_output_path",
        "variance_output_path",
        "heatmap_path",
        "bar_path",
        "q_dist_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
