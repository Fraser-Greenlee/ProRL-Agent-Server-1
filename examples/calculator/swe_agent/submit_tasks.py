#!/usr/bin/env python3
"""Submit the swe_agent calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = EXAMPLE_DIR.parent

sys.exit(subprocess.call([
    sys.executable, str(EXAMPLE_ROOT / "submit_calculator_task.py"),
    "--harness", "swe_agent",
    "--image", "polar-localhost-swe_agent:latest",
    "--docker-socket",
    *sys.argv[1:],
]))
