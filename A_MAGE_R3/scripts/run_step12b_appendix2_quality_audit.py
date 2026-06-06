"""Run Problem 2 Step 12B quality audit."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.problem2_step12_quality_audit import run_step12_quality_audit  # noqa: E402


def main() -> None:
    """Execute Step 12B and print a compact audit summary."""
    result = run_step12_quality_audit(PROJECT_ROOT / "config.yaml")
    summary = result["summary"].iloc[0]
    paper_flags = result["paper_flags"]
    constants = result["constant_checks"]
    anomalies = result["anomalies"]
    candidate_check = result["candidate_check"]

    print("Step 12B finished: Appendix 2 surface features and Q label quality audit")
    print(f"paper_count: {int(summary['paper_count'])}")
    print(f"surface_feature_count: {int(summary['surface_feature_count'])}")
    print(
        "Q_label range: "
        f"{summary['q_label_min']:.6f} - {summary['q_label_max']:.6f} "
        f"(std={summary['q_label_std']:.6f})"
    )
    print(f"candidate_usage_rows: {int(summary['candidate_usage_rows'])}")
    print(f"extreme_anomaly_count: {int(summary['extreme_anomaly_count'])}")
    print(f"constant_feature_count: {int(summary['constant_feature_count'])}")
    print(f"near_constant_feature_count: {int(summary['near_constant_feature_count'])}")
    print(f"can_enter_step13: {bool(summary['can_enter_step13'])}")

    print("paper flags:")
    for _, row in paper_flags.iterrows():
        print(f"  - {row['paper_id']}: {row['feature_quality_flag']} ({row['quality_notes']})")

    cautious_features = constants.loc[
        constants["is_constant"].astype(bool) | constants["is_near_constant"].astype(bool),
        ["feature_name", "audit_note"],
    ]
    print("cautious features:")
    if cautious_features.empty:
        print("  - none")
    else:
        for _, row in cautious_features.iterrows():
            print(f"  - {row['feature_name']}: {row['audit_note']}")

    print("candidate checks:")
    for _, row in candidate_check.iterrows():
        print(
            f"  - {row['paper_id']} {row['candidate_section']}: "
            f"exists={bool(row['exists'])}, weight_0.5={bool(row['all_weight_0_5'])}"
        )

    print("outputs:")
    print(f"  - {result['output_path']}")
    print(f"  - {result['log_path']}")


if __name__ == "__main__":
    main()
