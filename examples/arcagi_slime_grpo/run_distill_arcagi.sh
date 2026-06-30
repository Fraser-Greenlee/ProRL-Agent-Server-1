#!/usr/bin/env bash
# Full ARC-AGI distillation run (all 1147 real tasks, solve + refine).
# Detached + sleep-proof for the multi-day run:
#   tmux new-session -d -s distill-arcagi \
#     'caffeinate -i bash examples/arcagi_slime_grpo/run_distill_arcagi.sh'
# Watch:  tmux attach -t distill-arcagi   (detach Ctrl-b d)
# Resume: just re-run — skips tasks already attempted (one attempt per task/mode).
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root

export SFDC_GATEWAY_KEY="${SFDC_GATEWAY_KEY:-sk-3fb9Vdq3RhZcZqHR6RlkEw}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/Users/fgreenlee/Projects/auto-compress}"
export ARCAGI_GRADER_PY="${ARCAGI_GRADER_PY:-/tmp/arcagi-grader/bin/python}"
export OPENHANDS_SUPPRESS_BANNER=1

OUT="${OUT:-$PWD/runs/distill_arcagi}"
LOG="${LOG:-$PWD/runs/distill_arcagi.log}"
SCOPE="${SCOPE:-all}"
mkdir -p "$(dirname "$LOG")" "$OUT"

echo "=== ARC-AGI distill run started $(date) | scope=$SCOPE solve+refine ===" | tee -a "$LOG"

uv run --with openhands-sdk --with openhands-tools --with litellm --with certifi \
  python examples/arcagi_slime_grpo/distill_generate_arcagi.py \
  --out "$OUT" --scope "$SCOPE" --refine \
  --model anthropic/claude-opus-4-8 --parallel 5 \
  --max-iterations 50 --thinking-budget 16384 --task-timeout 2400 \
  >> "$LOG" 2>&1

echo "=== ARC-AGI distill run finished $(date) (rc=$?) ===" | tee -a "$LOG"
