"""Step 10: set up and verify the Problem 2 project structure."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.appendix2_pipeline import (  # noqa: E402
    PROBLEM2_REQUIRED_DIRS,
    PROBLEM2_REQUIRED_MODULES,
    PROBLEM2_REQUIRED_SCRIPTS,
    check_problem2_structure,
    ensure_problem2_directories,
    resolve_project_path,
)


LOG_PATH = PROJECT_ROOT / "output/problem2_logs/step10_problem2_setup.log"


def write_log(lines: list[str]) -> None:
    """Write setup log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Create and verify Problem 2 directories and scaffold files."""
    ensure_problem2_directories()
    missing_dirs, missing_files = check_problem2_structure()

    lines = [
        "Step 10 Problem 2 setup check",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Project root: {PROJECT_ROOT}",
        "",
        "Required directories:",
    ]
    lines += [f"  - {item}: {'OK' if resolve_project_path(item).is_dir() else 'MISSING'}" for item in PROBLEM2_REQUIRED_DIRS]
    lines += ["", "Required modules:"]
    lines += [f"  - {item}: {'OK' if resolve_project_path(item).is_file() else 'MISSING'}" for item in PROBLEM2_REQUIRED_MODULES]
    lines += ["", "Required scripts:"]
    lines += [f"  - {item}: {'OK' if resolve_project_path(item).is_file() else 'MISSING'}" for item in PROBLEM2_REQUIRED_SCRIPTS]
    lines += ["", f"Missing directories: {missing_dirs if missing_dirs else 'none'}"]
    lines += [f"Missing files: {missing_files if missing_files else 'none'}"]

    ok = not missing_dirs and not missing_files
    lines += [f"Result: {'PASS' if ok else 'FAIL'}"]
    write_log(lines)

    for line in lines:
        print(line)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
