#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${POLAR_PYTHON:-}" ]]; then
  PYTHON_BIN="${POLAR_PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" -c "import pip" 2>/dev/null; then
  "$PYTHON_BIN" -m ensurepip --upgrade
fi

# SWE-Gym provides the swegym grading modules used by this example, and
# datasets is needed to fetch/cache the benchmark instances.
"$PYTHON_BIN" -m pip install --no-cache-dir \
  "git+https://github.com/SWE-Gym/SWE-Bench-Package.git" \
  datasets

echo "Installed host-side evaluator dependencies with: $PYTHON_BIN"
