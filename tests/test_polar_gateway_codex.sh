#!/usr/bin/env bash
# Test Codex via the exact harness command that polar generates.
# Prerequisites: gateway at :8100, codex CLI installed, polar importable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROXY="${GATEWAY_URL:-http://localhost:8100}"
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
PROMPT="count and report the number of lines in the README.md, and write the result in tmp.txt"
LOG_DIR=$(mktemp -d)

PY="${PYTHON:-python3}"

# Create gateway session
curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"openai_responses\",\"task_id\":\"test-codex-polar\",\"session_id\":\"$SESSION_ID\"}" \
  > /dev/null

# Generate the exact harness command
CMD=$(PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" "$PY" -c "
from polar.agent.models import AgentSpec
from polar.agent.harnesses.codex import CodexHarness
spec = AgentSpec(harness='codex', model_name='openai/gpt-5.4')
steps = CodexHarness(spec).run_steps('''$PROMPT''')
print(steps[0].command)
")

# Rewrite container log path to local temp dir
CMD="${CMD//\/polar\/session\/logs\/agent/$LOG_DIR}"

echo "Session:  $SESSION_ID"
echo "Log dir:  $LOG_DIR"
echo "Command:  $CMD"
echo ""

# Mirror GatewayNodeManager._runtime_env().
# The codex harness -c flags reference $OPENAI_BASE_URL and $SESSION_ID
# as shell variables expanded at exec time.
ANTHROPIC_BASE_URL="$PROXY" \
ANTHROPIC_API_KEY="$SESSION_ID" \
OPENAI_BASE_URL="$PROXY/v1" \
OPENAI_API_KEY="$SESSION_ID" \
SESSION_ID="$SESSION_ID" \
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}" \
  bash -c "$CMD"
