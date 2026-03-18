#!/usr/bin/env bash
# Integration tests using curl against the gateway proxy.
# Prerequisites: vLLM at :8000, gateway proxy at :8080
set -euo pipefail

PROXY="http://localhost:8080"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo "──────────────────────────────────────────"
    echo "TEST: $name"
    echo "──────────────────────────────────────────"
    if output=$("$@" 2>&1); then
        echo "$output" | head -20
        echo ""
        echo "✓ PASS"
        PASS=$((PASS + 1))
    else
        echo "$output" | head -20
        echo ""
        echo "✗ FAIL"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# ── OpenAI Chat Completions ──────────────────────────────────────────────────

run_test "OpenAI Chat - non-streaming" \
    curl -sf "$PROXY/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":10}'

run_test "OpenAI Chat - streaming" \
    curl -sf "$PROXY/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-4","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":10,"stream":true}'

# ── Anthropic Messages ───────────────────────────────────────────────────────

run_test "Anthropic - non-streaming" \
    curl -sf "$PROXY/v1/messages" \
    -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -H "x-api-key: dummy" \
    -d '{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":10}'

run_test "Anthropic - streaming" \
    curl -sf "$PROXY/v1/messages" \
    -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -H "x-api-key: dummy" \
    -d '{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"Say hi in one word"}],"max_tokens":10,"stream":true}'

run_test "Anthropic - tool calling" \
    curl -sf "$PROXY/v1/messages" \
    -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -H "x-api-key: dummy" \
    -d '{
        "model":"claude-3-5-sonnet",
        "messages":[{"role":"user","content":"What is the weather in San Francisco?"}],
        "max_tokens":200,
        "tools":[{
            "name":"get_weather",
            "description":"Get current weather for a city",
            "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
        }]
    }'

# ── OpenAI Responses API ─────────────────────────────────────────────────────

run_test "Responses API - streaming" \
    curl -sf "$PROXY/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"gpt-4",
        "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hi"}]}],
        "stream":true,
        "max_tokens":10
    }'

run_test "Responses API - non-streaming" \
    curl -sf "$PROXY/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{
        "model":"gpt-4",
        "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hi"}]}],
        "stream":false,
        "max_tokens":10
    }'

# ── Verify storage ───────────────────────────────────────────────────────────

echo "──────────────────────────────────────────"
echo "STORAGE CHECK"
echo "──────────────────────────────────────────"

ROLLOUT_DIR="./rollouts"
if [ -d "$ROLLOUT_DIR" ]; then
    SESSION_COUNT=$(find "$ROLLOUT_DIR" -maxdepth 1 -type d -name "ses_*" | wc -l)
    MSG_COUNT=$(find "$ROLLOUT_DIR" -name "msg_*.json" | wc -l)
    echo "Sessions: $SESSION_COUNT"
    echo "Messages: $MSG_COUNT"

    # Show a sample metadata
    SAMPLE_META=$(find "$ROLLOUT_DIR" -name "metadata.json" -print -quit 2>/dev/null)
    if [ -n "$SAMPLE_META" ]; then
        echo ""
        echo "Sample metadata:"
        python3 -m json.tool "$SAMPLE_META" 2>/dev/null || cat "$SAMPLE_META"
    fi

    # Show a sample message (first 30 lines)
    SAMPLE_MSG=$(find "$ROLLOUT_DIR" -name "msg_*.json" -print -quit 2>/dev/null)
    if [ -n "$SAMPLE_MSG" ]; then
        echo ""
        echo "Sample message (truncated):"
        python3 -m json.tool "$SAMPLE_MSG" 2>/dev/null | head -30
    fi
else
    echo "No rollout directory found"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "══════════════════════════════════════════"
exit $FAIL
