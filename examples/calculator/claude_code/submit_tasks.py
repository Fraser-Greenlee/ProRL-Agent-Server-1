#!/usr/bin/env python3
"""Submit the claude_code calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXAMPLE_DIR.parent / "shared"

sys.exit(subprocess.call([
    sys.executable, str(SHARED_DIR / "submit_calculator_task.py"),
    "--harness", "claude_code",
    "--image", "arp-localhost-claude_code:latest",
    *sys.argv[1:],
]))
