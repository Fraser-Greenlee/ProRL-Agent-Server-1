#!/usr/bin/env bash
# Full 1015-task Opus distillation run (solve-only, first pass).
# Meant to run DETACHED inside tmux + caffeinate so it survives terminal
# disconnects, accidental Ctrl-C, and laptop sleep over its ~14h.
#
# Launch (done once by the operator):
#   tmux new-session -d -s distill \
#     'caffeinate -i bash examples/arcagi_slime_grpo/run_distill_full.sh'
# Watch:   tmux attach -t distill     (detach again with Ctrl-b d)
# Resume:  just re-run — it skips tasks already attempted (one attempt/task).
set -uo pipefail

cd "$(dirname "$0")/../.."   # repo root

export SFDC_GATEWAY_KEY="${SFDC_GATEWAY_KEY:-sk-3fb9Vdq3RhZcZqHR6RlkEw}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/Users/fgreenlee/Projects/auto-compress}"
export ARCAGI_GRADER_PY="${ARCAGI_GRADER_PY:-/tmp/arcagi-grader/bin/python}"
export OPENHANDS_SUPPRESS_BANNER=1

OUT="${OUT:-$PWD/runs/distill_full}"
LOG="${LOG:-$PWD/runs/distill_full.log}"
mkdir -p "$(dirname "$LOG")" "$OUT"

echo "=== full distill solve-only run started $(date) ===" | tee -a "$LOG"
echo "out=$OUT  model=opus  parallel=5  max_turns=50  (solve-only)" | tee -a "$LOG"

# Solve-only (no --refine): 1015 tasks, Opus, parallel=5. Resumable.
uv run --with openhands-sdk --with openhands-tools --with litellm --with certifi \
  python examples/arcagi_slime_grpo/distill_generate.py \
  --out "$OUT" --tasks 1015 \
  --model anthropic/claude-opus-4-8 --parallel 5 \
  --max-iterations 50 --thinking-budget 12288 --task-timeout 2400 \
  >> "$LOG" 2>&1

echo "=== full distill run finished $(date) (rc=$?) ===" | tee -a "$LOG"
