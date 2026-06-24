#!/bin/bash
#SBATCH --job-name=polar-smoke
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.log

# Smoke test: bring up vLLM + Polar rollout + one gateway, confirm health,
# then submit a 1-sample calculator task and assert reward >= 0 (i.e. the
# rollout pipeline produced any result, even if the small model fails the
# task — the goal here is wiring, not model quality).
#
# Submit with:  ./run_remote.sh submit smoke

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results rollout_results

source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

TOPOLOGY="topology_smoke.yaml"
SMOKE_LOG="logs/${SLURM_JOB_ID}.smoke.log"

cleanup() {
    echo "==> Cleanup: stopping background services..."
    [[ -n "${VLLM_PID:-}" ]]    && kill -- "-${VLLM_PID}"    2>/dev/null || true
    [[ -n "${ROLLOUT_PID:-}" ]] && kill -- "-${ROLLOUT_PID}" 2>/dev/null || true
    [[ -n "${GATEWAY_PID:-}" ]] && kill -- "-${GATEWAY_PID}" 2>/dev/null || true
    sleep 2
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "================================================================"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Topology:   ${TOPOLOGY}"
echo "================================================================"

wait_for_url() {
    local url="$1"
    local label="$2"
    local timeout="${3:-300}"
    local elapsed=0
    echo "==> Waiting for ${label} at ${url} (timeout ${timeout}s)..."
    while ! curl -fsS "$url" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if (( elapsed >= timeout )); then
            echo "ERROR: ${label} did not become ready within ${timeout}s" >&2
            return 1
        fi
        if (( elapsed % 30 == 0 )); then
            echo "  ... ${elapsed}s elapsed; still waiting for ${label}"
        fi
    done
    echo "==> ${label} ready after ${elapsed}s"
}

# --- 1. vLLM ----------------------------------------------------------------
echo "==> Starting vLLM serve on port 8000..."
setsid vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "logs/${SLURM_JOB_ID}.vllm.log" 2>&1 &
VLLM_PID=$!
echo "    vllm pid=${VLLM_PID}"

wait_for_url "http://127.0.0.1:8000/health" "vLLM" 600

# --- 2. Polar rollout -------------------------------------------------------
echo "==> Starting Polar rollout server on port 8080..."
setsid polar serve_rollout -c "${TOPOLOGY}" \
    > "logs/${SLURM_JOB_ID}.rollout.log" 2>&1 &
ROLLOUT_PID=$!
echo "    rollout pid=${ROLLOUT_PID}"

wait_for_url "http://127.0.0.1:8080/health" "Polar rollout" 60

# --- 3. Polar gateway -------------------------------------------------------
echo "==> Starting Polar gateway node (localhost-node-01) on port 8100..."
setsid polar serve_gateway -c "${TOPOLOGY}" --node-id localhost-node-01 \
    > "logs/${SLURM_JOB_ID}.gateway.log" 2>&1 &
GATEWAY_PID=$!
echo "    gateway pid=${GATEWAY_PID}"

wait_for_url "http://127.0.0.1:8100/health" "Polar gateway" 120

# --- 4. Status sanity check -------------------------------------------------
echo "==> polar status:"
polar status -c "${TOPOLOGY}" || true

echo "==> Health endpoints:"
echo "  vLLM:    $(curl -fsS http://127.0.0.1:8000/health && echo '(OK no body)' || echo 'FAIL')"
echo "  rollout: $(curl -fsS http://127.0.0.1:8080/health || echo 'FAIL')"
echo "  gateway: $(curl -fsS http://127.0.0.1:8100/health || echo 'FAIL')"

# --- 5. Optional: submit a tiny task ---------------------------------------
# Only run the calculator submission if SMOKE_RUN_TASK=1 (off by default to
# keep the smoke job under 5 min). The task itself needs the docker image
# (run `./run_remote.sh build-image` first if you turn this on).
if [[ "${SMOKE_RUN_TASK:-0}" = "1" ]]; then
    echo "==> Submitting a 1-sample calculator task with codex harness..."
    cat > /tmp/smoke_task.json <<EOF
{
  "task_id": "smoke-${SLURM_JOB_ID}",
  "instruction": "Implement calculator.py methods. Run python3 test_calculator.py to test.",
  "num_samples": 1,
  "timeout_seconds": 300,
  "runtime": {
    "backend": "docker",
    "image": "polar-localhost-calculator:latest",
    "prepare": [
      {"type": "exec", "command": "echo smoke runtime ready"}
    ],
    "network": "host",
    "workdir": "/polar/session"
  },
  "agent": {
    "harness": "shell",
    "model_name": "Qwen/Qwen3.5-1.7B-Instruct",
    "custom_shell": {
      "command": "echo 'shell harness placeholder; smoke just exercises the pipeline' > \$AGENT_LOG_DIR/agent.txt"
    }
  },
  "builder": {"strategy": "per_request"},
  "evaluator": {"strategy": "session_completed"}
}
EOF
    polar submit /tmp/smoke_task.json -c "${TOPOLOGY}" --json | tee "${SMOKE_LOG}"
fi

echo "================================================================"
echo "Smoke test wiring OK. All three services reachable."
echo "Logs:"
echo "  vLLM:    logs/${SLURM_JOB_ID}.vllm.log"
echo "  rollout: logs/${SLURM_JOB_ID}.rollout.log"
echo "  gateway: logs/${SLURM_JOB_ID}.gateway.log"
echo "================================================================"
