#!/usr/bin/env bash
# Test Claude Code via the exact harness command that polar generates.
# Prerequisites: gateway at :8100, claude CLI installed, polar importable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROXY="${GATEWAY_URL:-http://localhost:8100}"
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
PROMPT="count and report the number of lines in the README.md, and write the result in tmp.txt"
LOG_DIR=$(mktemp -d)

# Python that can import polar (respect PYTHON or fall back to python3)
PY="${PYTHON:-python3}"

# Create gateway session
curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"anthropic\",\"task_id\":\"test-cc-polar\",\"session_id\":\"$SESSION_ID\"}" \
  > /dev/null

# Generate the exact harness command
CMD=$(PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" "$PY" -c "
from polar.agent.models import AgentSpec
from polar.agent.harnesses.claude_code import ClaudeCodeHarness
spec = AgentSpec(harness='claude_code', model_name='anthropic/claude-opus-4-5')
steps = ClaudeCodeHarness(spec).run_steps('''$PROMPT''')
print(steps[0].command)
")

# Rewrite container log path to local temp dir
CMD="${CMD//\/polar\/session\/logs\/agent/$LOG_DIR}"

echo "Session:  $SESSION_ID"
echo "Log dir:  $LOG_DIR"
echo "Command:  $CMD"
echo ""

# Mirror GatewayNodeManager._runtime_env()
ANTHROPIC_BASE_URL="$PROXY" \
ANTHROPIC_API_KEY="$SESSION_ID" \
OPENAI_BASE_URL="$PROXY/v1" \
OPENAI_API_KEY="$SESSION_ID" \
SESSION_ID="$SESSION_ID" \
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}" \
  bash -c "$CMD"
