#!/usr/bin/env bash
# Test: OpenCode via the gateway proxy (OpenAI Chat Completions API)
# Prerequisites: vLLM at :8000, gateway at :8080, opencode installed
set -euo pipefail

PROXY="http://localhost:8080"
HARNESS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"anthropic\",\"task_id\":\"test-opencode\",\"session_id\":\"$HARNESS_SESSION_ID\"}" \
  > /dev/null

echo "Testing OpenCode → Gateway Proxy → vLLM"
echo "ANTHROPIC_BASE_URL=$PROXY"
echo ""

echo "Harness session: $HARNESS_SESSION_ID"
ANTHROPIC_BASE_URL="$PROXY" ANTHROPIC_API_KEY="$HARNESS_SESSION_ID" opencode -m anthropic/claude-opus-4-5