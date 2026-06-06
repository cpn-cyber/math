"""Run Step 15: grey relation and preliminary key-feature index."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.grey_relation import run_step15_grey_key_index  # noqa: E402


def main() -> None:
    """Execute Step 15 and print a compact summary."""
    result = run_step15_grey_key_index(PROJECT_ROOT / "config.yaml")
    print("Step 15 finished: grey relation and preliminary key-feature index")

    print("Grey relation Top 8:")
    for _, row in result["grey_top8"].iterrows():
        print(f"  - {row['feature_name']}: grey={row['grey_relation_score']:.6f}, norm={row['grey_norm']:.6f}")

    print("K_pre Top 8:")
    for _, row in result["key_top8"].iterrows():
        print(f"  - {row['feature_name']}: K_pre={row['K_pre']:.6f}, rank={int(row['rank_pre'])}")

    reference = result["reference_row"]
    appendix = result["appendix_row"]
    stacking_grey = result["stacking_grey"]
    stacking_key = result["stacking_key"]
    if not reference.empty:
        row = reference.iloc[0]
        print(f"reference_norm_rate: use_in_model={row['use_in_model']}, status={row['grey_status']}")
    if not appendix.empty:
        row = appendix.iloc[0]
        print(f"appendix_code_presence: use_in_model={row['use_in_model']}, status={row['grey_status']}")
    if not stacking_grey.empty and not stacking_key.empty:
        g = stacking_grey.iloc[0]
        k = stacking_key.iloc[0]
        rank = "excluded" if str(k.get("rank_pre")) == "nan" else int(k["rank_pre"])
        print(
            "stacking_penalty: "
            f"grey={g['grey_relation_score']:.6f}, grey_norm={g['grey_norm']:.6f}, "
            f"K_pre={k['K_pre']:.6f}, rank={rank}"
        )

    print("outputs:")
    for key in ["grey_path", "key_path", "grey_chart_path", "key_chart_path", "log_path"]:
        print(f"  - {result[key]}")


if __name__ == "__main__":
    main()
