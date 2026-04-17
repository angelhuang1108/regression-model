"""
Run condition-mapping pipeline as part of preprocessing.

This keeps condition feature generation in the Stage 3 flow while preserving
current output contracts consumed by regression.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONDITION_MAPPING_SCRIPTS = [
    PROJECT_ROOT / "3_preprocessing" / "condition_mapping" / "step00_exclusion_taxonomy.py",
    PROJECT_ROOT / "3_preprocessing" / "condition_mapping" / "step01_normalize.py",
    PROJECT_ROOT / "3_preprocessing" / "condition_mapping" / "step02_icd10_lookup.py",
    PROJECT_ROOT / "3_preprocessing" / "condition_mapping" / "step03_ccsr_join.py",
]


def run_script(script_path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def main() -> None:
    for script_path in CONDITION_MAPPING_SCRIPTS:
        if not run_script(script_path):
            print(f"ERROR: condition mapping failed at {script_path.name}")
            raise SystemExit(1)
    print("Condition mapping completed (step00 -> step03).")


if __name__ == "__main__":
    main()
