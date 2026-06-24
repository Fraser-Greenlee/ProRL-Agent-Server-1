#!/bin/bash
#SBATCH --job-name=polar-calc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.log

# 1-GPU calculator end-to-end:
#   - vLLM serves a small Qwen Instruct on :8000
#   - Polar rollout on :8080, one gateway on :8100
#   - submits 1 task with codex harness, NUM_SAMPLES=2, prints reward table
#
# Prereqs (run once before submitting):
#   ./run_remote.sh build-image    # builds polar-localhost-calculator:latest
#
# Submit with: ./run_remote.sh submit calculator

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results rollout_results

source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

TOPOLOGY="topology_smoke.yaml"

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
echo "Job ID:    ${SLURM_JOB_ID}  Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "================================================================"

wait_for_url() {
    local url="$1" label="$2" timeout="${3:-300}" elapsed=0
    while ! curl -fsS "$url" >/dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        if (( elapsed >= timeout )); then
            echo "ERROR: ${label} not ready within ${timeout}s" >&2; return 1
        fi
        if (( elapsed % 30 == 0 )); then echo "  ... ${elapsed}s waiting for ${label}"; fi
    done
    echo "==> ${label} ready (${elapsed}s)"
}

echo "==> Starting vLLM..."
setsid vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "logs/${SLURM_JOB_ID}.vllm.log" 2>&1 &
VLLM_PID=$!
wait_for_url "http://127.0.0.1:8000/health" vLLM 600

echo "==> Starting Polar rollout..."
setsid polar serve_rollout -c "${TOPOLOGY}" > "logs/${SLURM_JOB_ID}.rollout.log" 2>&1 &
ROLLOUT_PID=$!
wait_for_url "http://127.0.0.1:8080/health" rollout 60

echo "==> Starting Polar gateway..."
setsid polar serve_gateway -c "${TOPOLOGY}" --node-id localhost-node-01 \
    > "logs/${SLURM_JOB_ID}.gateway.log" 2>&1 &
GATEWAY_PID=$!
wait_for_url "http://127.0.0.1:8100/health" gateway 120

polar status -c "${TOPOLOGY}" || true

echo "==> Submitting calculator task (NUM_SAMPLES=2, codex harness, model=qwen3.5-1.7b)..."
# Override the default examples/calculator/run.py topology to point at our 1-GPU
# topology and a single harness so we don't try to install 9 of them.
python - <<'PY' 2>&1 | tee "logs/${SLURM_JOB_ID}.calc.log"
import json, time, httpx, os, pathlib, sys

repo = pathlib.Path.home() / "ProRL-Agent-Server"
sys.path.insert(0, str(repo))
from examples.calculator.run import build_task_payload, HARNESS_INSTALL, HARNESS_MODEL  # noqa

# Patch in our 1-GPU model.
HARNESS_MODEL["codex"] = "Qwen/Qwen2.5-1.5B-Instruct"
batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
payload = build_task_payload("codex", batch_id, "docker")
payload["num_samples"] = 2
payload["timeout_seconds"] = 1200

with httpx.Client(base_url="http://127.0.0.1:8080", timeout=httpx.Timeout(None, connect=30.0)) as c:
    resp = c.post("/rollout/task/submit", json=payload).raise_for_status()
    task_id = resp.json()["task_id"]
    print(f"submitted task_id={task_id}")
    while True:
        time.sleep(15)
        status = c.get(f"/rollout/task/{task_id}").json()
        print(f"  status={status['status']} sessions={len(status.get('results') or [])}")
        if status["status"] != "running":
            break
print(json.dumps(status, indent=2)[:4000])
PY

echo "================================================================"
echo "Calculator end-to-end complete. See logs/${SLURM_JOB_ID}.calc.log"
echo "================================================================"
