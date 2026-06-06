"""Run Step 18: bootstrap stability and final key-feature index."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.small_sample_validation import run_step18_small_sample_validation  # noqa: E402


def main() -> None:
    """Execute Step 18 and print a compact summary."""
    result = run_step18_small_sample_validation(PROJECT_ROOT / "config.yaml")

    print("Step 18 finished: small-sample validation and final key-feature index")
    print(f"Bootstrap valid/skipped: {result['valid_bootstrap_count']} / {result['skipped_bootstrap_count']}")

    stable = result["bootstrap_summary"].loc[result["bootstrap_summary"]["bootstrap_selected"]]
    print("Bootstrap stable features:")
    if stable.empty:
        print("  - none")
    else:
        for _, row in stable.sort_values("mean_VIP", ascending=False).iterrows():
            print(
                f"  - {row['feature_name']}: mean_VIP={row['mean_VIP']:.6f}, "
                f"P_VIP_gt_1={row['P_VIP_gt_1']:.3f}, sign={row['sign_consistency']:.3f}"
            )

    print("K_final Top 8:")
    for _, row in result["final_top8"].iterrows():
        print(f"  - {row['feature_name']}: K_final={row['K_final']:.6f}, rank={int(row['K_rank'])}")

    key_features = result["final_key_features"]["feature_name"].astype(str).tolist()
    print(f"final_key_feature: {', '.join(key_features) if key_features else 'none'}")

    print("High influence samples:")
    if result["high_influence"].empty:
        print("  - none")
    else:
        for _, row in result["high_influence"].iterrows():
            print(
                f"  - {row['removed_paper_id']}: RMSE_delta={row['RMSE_delta_vs_full_LOOCV']:.6f}, "
                f"Spearman_delta={row['Spearman_delta_vs_full_LOOCV']:.6f}, "
                f"Jaccard={row['top5_jaccard_vs_full']:.3f}, reason={row['influence_reason']}"
            )

    if not result["delete_21"].empty:
        row = result["delete_21"].iloc[0]
        print(
            "Delete 2-1: "
            f"RMSE={row['RMSE']:.6f}, RMSE_delta={row['RMSE_delta_vs_full_LOOCV']:.6f}, "
            f"Spearman={row['Spearman']:.6f}, Jaccard={row['top5_jaccard_vs_full']:.3f}"
        )

    print(f"Stability judgment: {result['stability_judgment']}")
    print("outputs:")
    for key in [
        "bootstrap_path",
        "delete_one_path",
        "final_index_path",
        "summary_path",
        "high_influence_path",
        "bootstrap_chart_path",
        "final_index_chart_path",
        "delete_one_chart_path",
        "model_summary_chart_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
