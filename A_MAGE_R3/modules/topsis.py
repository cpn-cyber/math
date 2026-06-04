"""TOPSIS scoring interface.

TODO:
- Normalize feature matrix.
- Apply fused weights.
- Calculate TOPSIS closeness scores.
"""

from pathlib import Path


def run_topsis(feature_table_path: Path, weight_path: Path, output_path: Path) -> Path:
    """Run TOPSIS and save paper quality scores."""
    raise NotImplementedError("Step 6 will implement TOPSIS scoring.")
