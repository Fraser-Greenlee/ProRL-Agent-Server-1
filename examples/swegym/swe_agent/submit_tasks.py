#!/usr/bin/env python3
"""Submit the curated SWE-Gym sample with the swe_agent harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXAMPLE_DIR.parent / "shared"

sys.exit(
    subprocess.call(
        [
            sys.executable,
            str(SHARED_DIR / "submit_swegym_tasks.py"),
            "--harness",
            "swe_agent",
            *sys.argv[1:],
        ]
    )
)
