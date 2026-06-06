"""Run Step 17: quality adjustment factor model."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.quality_adjustment_factor import run_step17_quality_adjustment  # noqa: E402


def main() -> None:
    """Execute Step 17 and print a compact summary."""
    result = run_step17_quality_adjustment(PROJECT_ROOT / "config.yaml")
    eta_row = result["selected_eta_row"]

    print("Step 17 finished: Quality Adjustment Factor")
    print(f"Selected eta: {result['selected_eta']:.2f}")
    print(f"phi_i range: {result['phi_range'][0]:.6f} ~ {result['phi_range'][1]:.6f}")
    print(
        "PLS -> QAF metrics: "
        f"MAE {eta_row['PLS_base_MAE']:.6f} -> {eta_row['MAE']:.6f}, "
        f"RMSE {eta_row['PLS_base_RMSE']:.6f} -> {eta_row['RMSE']:.6f}, "
        f"R2 {eta_row['PLS_base_R2']:.6f} -> {eta_row['R2']:.6f}, "
        f"Spearman {eta_row['PLS_base_Spearman']:.6f} -> {eta_row['Spearman']:.6f}"
    )
    print(f"2-1 improved: {result['paper_21_improved']}")

    print("Largest upward adjustments:")
    for _, row in result["max_up"].iterrows():
        print(f"  - {row['paper_id']}: adjustment={row['adjustment_value']:.6f}, phi={row['phi_i']:.6f}")

    print("Largest downward adjustments:")
    for _, row in result["max_down"].iterrows():
        print(f"  - {row['paper_id']}: adjustment={row['adjustment_value']:.6f}, phi={row['phi_i']:.6f}")

    if len(result["stacking_constrained"]):
        print("High stacking_penalty constrained papers:")
        for _, row in result["stacking_constrained"].iterrows():
            print(f"  - {row['paper_id']}: adjustment={row['adjustment_value']:.6f}, note={row['audit_note']}")
    else:
        print("High stacking_penalty constrained papers: none")

    print("outputs:")
    for key in [
        "qaf_scores_path",
        "eta_path",
        "prediction_path",
        "audit_path",
        "waterfall_path",
        "true_vs_pred_path",
        "before_after_path",
        "log_path",
    ]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
