"""Run the full Problem 1 pipeline in order.

This runner is for reproducibility. It may overwrite generated outputs, so use
--dry-run first when you only want to inspect commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("step1_setup", "scripts/run_step1_setup.py"),
    ("step2_parse_pdf", "scripts/run_step2_parse_pdf.py"),
    ("step2b_ocr_failed", "scripts/run_step2b_ocr_parse_failed.py"),
    ("step3_split_sections", "scripts/run_step3_split_sections.py"),
    ("step4_extract_features", "scripts/run_step4_extract_features.py"),
    ("step5_weighting", "scripts/run_step5_weighting.py"),
    ("step6_topsis", "scripts/run_step6_topsis.py"),
    ("step7a_pairwise_template", "scripts/run_step7a_generate_pairwise_template.py"),
    ("step7a_surrogate_fill", "scripts/run_step7a_fill_pairwise_surrogate.py"),
    ("step7b_pairwise_quality", "scripts/run_step7b_pairwise_quality_check.py"),
    ("step7b_bradley_terry", "scripts/run_step7b_bradley_terry.py"),
    ("step7c_bt_sensitivity", "scripts/run_step7c_bt_audit_sensitivity.py"),
    ("step8_grade_visualize", "scripts/run_step8_grade_visualize.py"),
    ("step8b_final_audit", "scripts/run_step8b_final_audit.py"),
    ("step10_enhance", "scripts/run_step10_problem1_enhance.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--from-step", default=None, help="Start from a named step.")
    args = parser.parse_args()

    start = 0
    if args.from_step:
        names = [name for name, _ in STEPS]
        if args.from_step not in names:
            raise SystemExit(f"Unknown step {args.from_step}. Available: {names}")
        start = names.index(args.from_step)

    for name, script in STEPS[start:]:
        command = [sys.executable, str(PROJECT_ROOT / script)]
        print(f"[{name}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
