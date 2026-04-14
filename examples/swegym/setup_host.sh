#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${POLAR_PYTHON:-python3}"

if ! "$PYTHON_BIN" -c "import pip" 2>/dev/null; then
  "$PYTHON_BIN" -m ensurepip --upgrade
fi

"$PYTHON_BIN" -m pip install --no-cache-dir "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"

echo "Installed SWE-Gym evaluator dependency with: $PYTHON_BIN"
