#!/usr/bin/env python3
"""Submit the claude_code calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent

sys.exit(subprocess.call([
    sys.executable, str(EXAMPLE_ROOT / "submit_calculator_task.py"),
    "--harness", "claude_code",
    "--image", "polar-localhost-claude_code:latest",
    *sys.argv[1:],
]))
