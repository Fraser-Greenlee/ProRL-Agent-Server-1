#!/usr/bin/env bash
# Test: Codex CLI via the gateway proxy (OpenAI Responses API)
# Prerequisites: vLLM at :8000, gateway at :8080, codex CLI installed
set -euo pipefail

PROXY="http://localhost:8080"
ROLLOUT_DIR="${ROLLOUT_DIR:-./rollouts}"
HARNESS_SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

curl -sf "$PROXY/sessions" \
  -H "Content-Type: application/json" \
  -d "{\"api_type\":\"openai_responses\",\"task_id\":\"test-codex\",\"session_id\":\"$HARNESS_SESSION_ID\"}" \
  > /dev/null

# This gateway supports HTTP Responses API + SSE, not Codex's websocket
# transport or compressed request bodies.
echo "Harness session: $HARNESS_SESSION_ID"
# Force Codex onto a one-off proxy provider so it sends our harness session
# explicitly instead of falling back to any cached OpenAI login state.
OPENAI_API_KEY="$HARNESS_SESSION_ID" codex exec \
  -c 'model_provider="harness_proxy"' \
  -c 'model_providers.harness_proxy.name="Harness Proxy"' \
  -c "model_providers.harness_proxy.base_url=\"${PROXY}/v1\"" \
  -c 'model_providers.harness_proxy.env_key="OPENAI_API_KEY"' \
  -c 'model_providers.harness_proxy.wire_api="responses"' \
  -c "model_providers.harness_proxy.http_headers={\"X-Session-Id\"=\"${HARNESS_SESSION_ID}\"}" \
  --disable responses_websockets \
  --disable responses_websockets_v2 \
  --disable enable_request_compression \
  "count and report the number of lines in the README.md"