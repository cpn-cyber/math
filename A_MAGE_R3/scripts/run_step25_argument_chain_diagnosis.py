"""Run Problem 3 Step 25: five-element argument chain diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.argument_chain_diagnostics import run_argument_chain_diagnosis  # noqa: E402


def main() -> None:
    """Execute Step 25 and print a compact summary."""
    result = run_argument_chain_diagnosis(PROJECT_ROOT / "config.yaml")
    diagnosis = result["diagnosis"]
    edges = result["edges"]
    paths = result["paths"]

    print("Step 25 argument chain diagnosis finished.")
    print("Command: python A_MAGE_R3/scripts/run_step25_argument_chain_diagnosis.py")
    print("\nLogic gap diagnosis:")
    print(
        diagnosis[
            [
                "paper_id",
                "s_TD",
                "s_DHM",
                "s_HMR",
                "s_RC",
                "s_CT",
                "Gamma",
                "G_logic_gap",
                "major_gap_edges",
            ]
        ].to_string(index=False)
    )
    print("\nMajor-gap edges:")
    for paper_id in diagnosis["paper_id"].astype(str).tolist():
        row = diagnosis.loc[diagnosis["paper_id"].astype(str) == paper_id].iloc[0]
        major = str(row["major_gap_edges"])
        weak_edges = edges.loc[
            (edges["paper_id"].astype(str) == paper_id) & (edges["major_gap_flag"]),
            ["edge", "edge_score", "gap_reason"],
        ]
        print(f"- {paper_id}: {major}")
        if not weak_edges.empty:
            for edge_row in weak_edges.itertuples(index=False):
                print(f"  * {edge_row.edge}: {edge_row.edge_score:.3f}; {edge_row.gap_reason}")
    print("\nOutputs:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
