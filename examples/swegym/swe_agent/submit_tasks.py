#!/usr/bin/env python3
"""Submit the curated SWE-Gym sample with the swe_agent harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent

sys.exit(
    subprocess.call(
        [
            sys.executable,
            str(EXAMPLE_ROOT / "submit_swegym_tasks.py"),
            "--harness",
            "swe_agent",
            *sys.argv[1:],
        ]
    )
)
