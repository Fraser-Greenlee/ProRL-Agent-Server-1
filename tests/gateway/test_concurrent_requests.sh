#!/usr/bin/env bash
# Test: Run 3 copies each of codex, opencode, and claude-code concurrently (9 total)
# Validates the gateway handles real-world concurrent calls from different sources.
# PASS = harness exits cleanly and the registered session stores at least one completion.
# Prerequisites: vLLM at :8000, gateway at :8080, codex/opencode/claude CLIs installed
set -euo pipefail

PROXY="http://localhost:8080"
ROLLOUT_DIR="${ROLLOUT_DIR:-./rollouts}"
COPIES=3
TASK="count and report the number of lines in the README.md"
WAIT_FOR_COMPLETION_SECS="${WAIT_FOR_COMPLETION_SECS:-15}"
STATE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gateway-concurrency.XXXXXX")"
trap 'rm -rf "$STATE_ROOT"' EXIT

declare -a PIDS=()
declare -a LABELS=()
declare -a SIDS=()
declare -a EXIT_CODES=()

register_session() {
  local api_type="$1" task_id="$2" sid="$3"
  curl -sf "$PROXY/sessions" \
    -H "Content-Type: application/json" \
    -d "{\"api_type\":\"$api_type\",\"task_id\":\"$task_id\",\"session_id\":\"$sid\"}" \
    > /dev/null
}

get_completion_count() {
  local sid="$1"
  curl -sf "$PROXY/sessions/$sid" \
    | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["completion_count"]))'
}

wait_for_completion_count() {
  local sid="$1"
  local attempts="$WAIT_FOR_COMPLETION_SECS"
  local count=0

  while [ "$attempts" -gt 0 ]; do
    count="$(get_completion_count "$sid" 2>/dev/null || echo 0)"
    if [ "$count" -gt 0 ]; then
      echo "$count"
      return 0
    fi
    sleep 1
    attempts=$((attempts - 1))
  done

  echo "$count"
  return 1
}

echo "══════════════════════════════════════════"
echo "Concurrent harness: 3 tests × $COPIES copies = $(( 3 * COPIES )) processes"
echo "══════════════════════════════════════════"
echo ""

for i in $(seq 1 "$COPIES"); do
  sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  register_session openai_responses test-codex "$sid"
  echo "Starting codex#$i  (session $sid)"
  OPENAI_API_KEY="$sid" \
    codex exec \
      -c 'model_provider="harness_proxy"' \
      -c 'model_providers.harness_proxy.name="Harness Proxy"' \
      -c "model_providers.harness_proxy.base_url=\"${PROXY}/v1\"" \
      -c 'model_providers.harness_proxy.env_key="OPENAI_API_KEY"' \
      -c 'model_providers.harness_proxy.wire_api="responses"' \
      -c "model_providers.harness_proxy.http_headers={\"X-Session-Id\"=\"${sid}\"}" \
      --disable responses_websockets \
      --disable responses_websockets_v2 \
      --disable enable_request_compression \
      "$TASK" &
  PIDS+=($!); LABELS+=("codex#$i"); SIDS+=("$sid")

  sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  register_session openai_chat test-opencode "$sid"
  echo "Starting opencode#$i  (session $sid)"
  opencode_data_home="$STATE_ROOT/opencode-$sid"
  mkdir -p "$opencode_data_home"
  XDG_DATA_HOME="$opencode_data_home" OPENAI_BASE_URL="$PROXY/v1" OPENAI_API_KEY="$sid" \
    opencode -m openai/gpt-5.2 run "$TASK" &
  PIDS+=($!); LABELS+=("opencode#$i"); SIDS+=("$sid")

  sid="$(python3 -c 'import uuid; print(uuid.uuid4())')"
  register_session anthropic test-claude-code "$sid"
  echo "Starting claude#$i  (session $sid)"
  ANTHROPIC_BASE_URL="$PROXY" ANTHROPIC_API_KEY="$sid" \
    claude -p "$TASK" &
  PIDS+=($!); LABELS+=("claude#$i"); SIDS+=("$sid")
done

echo ""
echo "All ${#PIDS[@]} processes launched. Waiting..."
echo ""

for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "$pid"; then
    EXIT_CODES+=(0)
  else
    EXIT_CODES+=($?)
  fi
done

PASS=0
FAIL=0

for idx in "${!LABELS[@]}"; do
  label="${LABELS[$idx]}"
  sid="${SIDS[$idx]}"
  exit_code="${EXIT_CODES[$idx]}"
  session_dir="$ROLLOUT_DIR/ses_$sid"
  completion_count="$(wait_for_completion_count "$sid" || true)"

  if [ "$exit_code" -eq 0 ] && [ -d "$session_dir" ] && [ "$completion_count" -gt 0 ]; then
    echo "✓ PASS  $label  ($session_dir, completions=$completion_count)"
    PASS=$((PASS + 1))
  else
    echo "✗ FAIL  $label  (exit=$exit_code, completions=$completion_count, session_dir=$session_dir)"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "══════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed (of $((PASS + FAIL)))"
echo "══════════════════════════════════════════"
exit "$FAIL"