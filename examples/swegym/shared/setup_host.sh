#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${ARP_PYTHON:-python3}"

"$PYTHON_BIN" -m pip install --no-cache-dir "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"

echo "Installed SWE-Gym evaluator dependency with: $PYTHON_BIN"
