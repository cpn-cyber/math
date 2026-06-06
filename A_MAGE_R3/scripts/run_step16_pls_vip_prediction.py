"""Run Step 16: PLS-VIP small-sample quality prediction."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.pls_vip_model import run_step16_pls_vip_prediction  # noqa: E402


def main() -> None:
    """Execute Step 16 and print a compact summary."""
    result = run_step16_pls_vip_prediction(PROJECT_ROOT / "config.yaml")
    metrics = result["metrics_row"]

    print("Step 16 finished: PLS-VIP small-sample prediction")
    print(f"PLS feature count: {len(result['feature_columns'])}")
    print("Excluded features:")
    for _, row in result["excluded"].iterrows():
        print(f"  - {row['feature_name']}: {row['exclude_reason']}")

    caution_names = result["cautions"]["feature_name"].astype(str).tolist()
    print(f"Caution features: {', '.join(caution_names) if caution_names else 'none'}")
    print(f"Selected n_components A: {result['best_component']}")
    print(
        "LOOCV metrics: "
        f"MAE={metrics['MAE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
        f"R2_LOO={metrics['R2_LOO']:.6f}, Spearman={metrics['Spearman']:.6f}"
    )

    print("VIP Top 8:")
    for _, row in result["vip_top8"].iterrows():
        print(f"  - {row['feature_name']}: VIP={row['VIP']:.6f}, rank={int(row['VIP_rank'])}")

    vip_gt1 = result["vip_gt1"]["feature_name"].astype(str).tolist()
    print(f"VIP > 1 features: {', '.join(vip_gt1) if vip_gt1 else 'none'}")

    if len(result["large_errors"]):
        print("Large LOOCV error papers:")
        for _, row in result["large_errors"].iterrows():
            print(f"  - {row['paper_id']}: abs_error={row['abs_error']:.6f}, true={row['Q_true']:.6f}, pred={row['Q_pred_loo']:.6f}")
    else:
        print("Large LOOCV error papers: none")

    print("outputs:")
    for key in [
        "component_path",
        "prediction_path",
        "vip_path",
        "feature_set_path",
        "true_pred_chart_path",
        "vip_chart_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
