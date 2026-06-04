"""Bradley-Terry calibration interface.

TODO:
- Build pairwise paper comparison data.
- Estimate Bradley-Terry strengths.
- Calibrate TOPSIS scores if needed.
"""

from pathlib import Path


def run_bradley_terry(pairwise_data_path: Path, output_path: Path) -> Path:
    """Run Bradley-Terry preference calibration."""
    raise NotImplementedError("Step 7 will implement Bradley-Terry calibration.")
