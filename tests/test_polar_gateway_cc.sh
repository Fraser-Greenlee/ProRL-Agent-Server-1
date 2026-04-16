#!/usr/bin/env bash
# Test Claude Code via the exact harness command that polar generates.
# Prerequisites: gateway at :8100, claude CLI installed, polar importable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROXY="${GATEWAY_URL:-http://localhost:8100}"
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"
OUTPUT_FILE="$REPO_ROOT/tmp.txt"
PROMPT="count and report the number of lines in the README.md, and write the result to $OUTPUT_FILE"
LOG_DIR=$(mktemp -d)

# Python that can import polar (respect PYTHON or fall back to python3)
PY="${PYTHON:-python3}"

# Create gateway session
curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"anthropic\",\"task_id\":\"test-cc-polar\",\"session_id\":\"$SESSION_ID\"}" \
  > /dev/null

# Generate the exact harness command (join all steps in case the harness returns multiple)
CMD=$(PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}" "$PY" -c "
from polar.agent.models import AgentSpec
from polar.agent.harnesses.claude_code import ClaudeCodeHarness
spec = AgentSpec(harness='claude_code', model_name='anthropic/claude-opus-4-5')
steps = ClaudeCodeHarness(spec).run_steps('''$PROMPT''')
print(' && '.join(s.command for s in steps))
")

# Rewrite container log path to local temp dir
CMD="${CMD//\/polar\/session\/logs\/agent/$LOG_DIR}"

# Clear stale output so Claude Code's read-before-write guard doesn't short-circuit the run
rm -f "$OUTPUT_FILE"

echo "Session:  $SESSION_ID"
echo "Log dir:  $LOG_DIR"
echo "Command:  $CMD"
echo ""

# Mirror GatewayNodeManager._runtime_env(); default to a scratch config dir so
# user-installed plugins/hooks (e.g. node-based SessionStart hooks) don't leak in.
ANTHROPIC_BASE_URL="$PROXY" \
ANTHROPIC_API_KEY="$SESSION_ID" \
OPENAI_BASE_URL="$PROXY/v1" \
OPENAI_API_KEY="$SESSION_ID" \
SESSION_ID="$SESSION_ID" \
CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$LOG_DIR/.claude}" \
  bash -c "$CMD"
