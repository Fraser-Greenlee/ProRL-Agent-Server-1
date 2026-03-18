#!/usr/bin/env bash
# Test: OpenCode via the gateway proxy (OpenAI Chat Completions API)
# Prerequisites: vLLM at :8000, gateway at :8080, opencode installed
set -euo pipefail

PROXY="http://localhost:8080"
ROLLOUT_DIR="${ROLLOUT_DIR:-./rollouts}"
HARNESS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"openai_chat\",\"task_id\":\"test-opencode\",\"session_id\":\"$HARNESS_SESSION_ID\"}" \
  > /dev/null

echo "Testing OpenCode → Gateway Proxy → vLLM"
echo "OPENAI_BASE_URL=$PROXY/v1"
echo ""

echo "Harness session: $HARNESS_SESSION_ID"
OPENAI_BASE_URL="$PROXY/v1" OPENAI_API_KEY="$HARNESS_SESSION_ID" opencode -m openai/gpt-5.2 run "count and report the number of lines in the README.md"