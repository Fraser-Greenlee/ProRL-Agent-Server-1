#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${POLAR_PYTHON:-python3}"

cd "$REPO_ROOT"
"$PYTHON_BIN" "examples/swegym/build_openhands_sdk_images.py" "$@"
