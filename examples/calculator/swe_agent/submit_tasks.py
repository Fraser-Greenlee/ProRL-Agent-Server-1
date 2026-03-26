#!/usr/bin/env python3
"""Submit the swe_agent calculator task."""
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
SHARED_DIR = EXAMPLE_DIR.parent / "shared"

sys.exit(subprocess.call([
    sys.executable, str(SHARED_DIR / "submit_calculator_task.py"),
    "--harness", "swe_agent",
    "--image", "arp-localhost-swe_agent:latest",
    "--docker-socket",
    *sys.argv[1:],
]))
