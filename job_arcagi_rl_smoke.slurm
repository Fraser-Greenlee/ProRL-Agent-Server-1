#!/bin/bash
#SBATCH --job-name=okay_to_cancel_arcagi-rl-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.log

# 1-node ONE-TASK agent smoke for the synthetic rl pipeline. Isolates the
# agent -> gateway -> SGLang path (no slime/Ray/training). Reproduces the
# "no completions" failure and captures:
#   (a) the raw assistant message(s) the gateway received (incl. reasoning)
#   (b) iteration/token count at bail
#   (c) whether the agent made an LLM call at all (gateway completion count)
set -uo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs rollout_results tmp
source .venv/bin/activate
export PATH=.venv/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda/include
export PYTHONPATH=.:${PYTHONPATH:-}
# Keep per-session dirs (incl. the agent's own stdout at logs/agent/) so a
# failed rollout can be inspected after teardown. The gateway mkdtemp's these
# under $TMPDIR (default /tmp) as session-<id>-*.
export POLAR_KEEP_SESSION_DIRS=1

JID="${SLURM_JOB_ID}"
MODEL=/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae
SGL_PORT=31002
IMAGE=polar-arcagi-rl:latest
TARBALL=/home/fraser_convergence_ai/arcagi-image/polar-arcagi-rl.tar.gz

cleanup() { echo "==> cleanup"; for p in "${SGL_PID:-}" "${ROLLOUT_PID:-}" "${GATEWAY_PID:-}"; do [ -n "$p" ] && kill -- "-$p" 2>/dev/null || true; done; pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Load the rl image (rollout containers run here on this node).
IMAGE="$IMAGE" TARBALL="$TARBALL" bash examples/arcagi_slime_grpo/load_image.sh || true

# --- 1. SGLang (patched .post1), TP=4 ---------------------------------------
echo "==> starting SGLang on :${SGL_PORT}"
setsid .venv/bin/python -m sglang.launch_server --model-path "$MODEL" \
    --host 127.0.0.1 --port "$SGL_PORT" --tp 4 --mem-fraction-static 0.8 \
    --context-length 65536 --disable-custom-all-reduce \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    > "logs/${JID}.sglang.log" 2>&1 &
SGL_PID=$!
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:${SGL_PORT}/health" >/dev/null 2>&1 && { echo "SGLang healthy after $((i*5))s"; break; }; sleep 5; done

# --- 2. smoke topology pointing the gateway at our SGLang -------------------
cat > tmp/topology_rl_smoke.yaml <<YAML
rollout:
  host: 127.0.0.1
  port: 8080
  public_url: http://127.0.0.1:8080
  save_dir: ./rollout_results
gateway:
  heartbeat_interval_seconds: 30
  nodes:
    - id: localhost-node-01
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      max_init_workers: 2
      max_run_workers: 2
      max_postrun_workers: 2
      model_served: ${MODEL}
      inference:
        engine: sglang
        base_url: http://127.0.0.1:${SGL_PORT}
YAML

echo "==> starting Polar rollout + gateway"
setsid polar serve_rollout -c tmp/topology_rl_smoke.yaml > "logs/${JID}.rollout.log" 2>&1 &
ROLLOUT_PID=$!
for i in $(seq 1 24); do curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 5; done
setsid polar serve_gateway -c tmp/topology_rl_smoke.yaml --node-id localhost-node-01 > "logs/${JID}.gateway.log" 2>&1 &
GATEWAY_PID=$!
for i in $(seq 1 36); do curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break; sleep 5; done
echo "==> services up"; polar status -c tmp/topology_rl_smoke.yaml || true

# --- 3. one rl task, openhands_sdk harness, 2 samples -----------------------
# Build a task payload from the first rl_tasks_train.jsonl row (real prompt).
.venv/bin/python - <<PY > tmp/rl_smoke_task.json
import json
row = json.loads(open("examples/arcagi_slime_grpo/rl_tasks_train.jsonl").readline())
m = row["metadata"]; tid = m["task_id"]
task = {
  "task_id": f"rlsmoke-{tid}",
  "instruction": row["prompt"][0]["content"],
  "num_samples": 2,
  "timeout_seconds": 1200,
  "runtime": {
    "backend": "docker", "image": "${IMAGE}", "network": "host",
    "workdir": "/polar/session/workspace",
    "env": {"HOME": "/polar/session/home", "AUTO_COMPRESS": "/polar/session/workspace",
            "PATH": "/opt/arcagi-venv/bin:/opt/openhands-sdk-venv/bin:/usr/local/bin:/usr/bin:/bin"},
    "prepare": [{"type":"exec","command":"rm -rf /polar/session/workspace && mkdir -p /polar/session/workspace /polar/session/home /polar/session/logs/agent && cp -a /opt/auto-compress/. /polar/session/workspace/ && cd /polar/session/workspace && git init -q && git config user.email a@b && git config user.name p && git add -A && git commit -qm base"}],
  },
  "agent": {"harness": "openhands_sdk", "model_name": "openai/${MODEL}", "settings": {"max_iterations": 15}},
  "builder": {"strategy": "prefix_merging"},
  "evaluator": {"strategy": "session_completed"},
}
print(json.dumps(task))
PY
echo "==> submitting one rl task (openhands_sdk)"
polar submit tmp/rl_smoke_task.json -c tmp/topology_rl_smoke.yaml --json > "logs/${JID}.submit.json" 2>&1 || true

echo "================ RESULTS ================"
echo "--- submit result ---"; tail -c 2000 "logs/${JID}.submit.json"
echo; echo "--- gateway: LLM proxy / completion activity ---"
grep -iE "chat/completions|/v1/|completion|no completion|model|error|exception|4[0-9][0-9] |5[0-9][0-9] " "logs/${JID}.gateway.log" 2>/dev/null | tail -30
echo; echo "--- newest session result json (raw assistant message + counts) ---"
NEWEST=$(ls -t rollout_results/task_rlsmoke-*/ses_*.json 2>/dev/null | head -1)
echo "FILE: $NEWEST"; [ -n "$NEWEST" ] && .venv/bin/python -m json.tool "$NEWEST" | head -80

# THE key artifact: the agent's own stdout. With pipefail (openhands_sdk
# preset) a crashing runner now fails the step instead of being masked by tee,
# and POLAR_KEEP_SESSION_DIRS preserves this file past teardown.
echo; echo "--- agent stdout (openhands-sdk.txt) from preserved session dir(s) ---"
for f in $(ls -t /tmp/session-*/logs/agent/openhands-sdk.txt "${TMPDIR:-/tmp}"/session-*/logs/agent/openhands-sdk.txt 2>/dev/null | head -2); do
    echo "===== $f ====="; tail -120 "$f"
done
# Gateway-captured copy of the same stream (2>&1 merges stderr into stdout
# before tee, so the step's *stdout* log holds the traceback, not stderr).
echo; echo "--- agent step stdout (gateway-side exec capture) ---"
for f in $(ls -t /tmp/session-*/logs/agent/step.*.stdout.log "${TMPDIR:-/tmp}"/session-*/logs/agent/step.*.stdout.log 2>/dev/null | head -2); do
    echo "===== $f ====="; tail -60 "$f"
done
echo "==> done; leaving services up briefly for inspection"; sleep 20
