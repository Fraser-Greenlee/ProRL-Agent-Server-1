#!/usr/bin/env python3
"""Submit the openhands_sdk calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent

sys.exit(subprocess.call([
    sys.executable, str(EXAMPLE_ROOT / "submit_calculator_task.py"),
    "--harness", "openhands_sdk",
    "--image", "polar-localhost-openhands_sdk:latest",
    *sys.argv[1:],
]))
