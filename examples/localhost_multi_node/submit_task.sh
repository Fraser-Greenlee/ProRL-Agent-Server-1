#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_URL="${ROLLOUT_URL:-http://127.0.0.1:8080}"
TASK_FILE="${TASK_FILE:-$EXAMPLE_DIR/task_request.json}"

curl -sf "$ROLLOUT_URL/rollout/task" \
  -H "Content-Type: application/json" \
  --data @"$TASK_FILE" \
  | python3 -m json.tool
