"""Weight calculation interface.

TODO:
- Build AHP subjective weights.
- Build CRITIC or entropy objective weights.
- Fuse subjective and objective weights.
"""

from pathlib import Path


def calculate_weights(feature_table_path: Path, output_path: Path) -> Path:
    """Calculate and save feature weights."""
    raise NotImplementedError("Step 5 will implement weighting.")
