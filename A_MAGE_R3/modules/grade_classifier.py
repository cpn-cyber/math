"""Grade classification interface.

TODO:
- Convert continuous quality scores into five grade labels.
- Support threshold-based and clustering-based strategies.
"""

from pathlib import Path


def classify_grades(score_table_path: Path, output_path: Path) -> Path:
    """Classify papers into excellent/good/medium/pass/fail grades."""
    raise NotImplementedError("Step 8 will implement grade classification.")
