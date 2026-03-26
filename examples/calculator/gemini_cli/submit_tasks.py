#!/usr/bin/env python3
"""Submit the gemini_cli calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXAMPLE_DIR.parent / "shared"

sys.exit(subprocess.call([
    sys.executable, str(SHARED_DIR / "submit_calculator_task.py"),
    "--harness", "gemini_cli",
    "--image", "arp-localhost-gemini_cli:latest",
    *sys.argv[1:],
]))
