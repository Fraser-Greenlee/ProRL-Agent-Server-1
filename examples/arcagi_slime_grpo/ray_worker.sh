#!/usr/bin/env bash
# Join a Ray cluster as a worker node. Invoked by run.sh via srun on each
# non-head node of a multi-node slurm allocation.
#   ray_worker.sh <MASTER_ADDR> <NUM_GPUS>
set -uo pipefail

MASTER_ADDR="${1:?MASTER_ADDR required}"
NUM_GPUS="${2:-8}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"

# Same PATH hygiene as run.sh: venv first so ray/python are the venv's.
export PATH="${HOME}/.local/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[ -x "${CUDA_HOME}/bin/nvcc" ] && export PATH="${CUDA_HOME}/bin:${PATH}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/include}"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

WORKER_IP="$("${PYTHON_BIN}" - <<'PY'
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0]); s.close()
except Exception:
    print(socket.gethostbyname(socket.gethostname()))
PY
)"
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR},${WORKER_IP}"

echo "[ray_worker] $(hostname) ip=${WORKER_IP} joining ${MASTER_ADDR}:6379 with ${NUM_GPUS} GPUs"
ray stop --force 2>/dev/null || true
sleep 2
# --block keeps this srun task alive for the lifetime of the worker (so the
# allocation holds the node); run.sh's job runs on the head meanwhile.
ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS}" \
    --node-ip-address "${WORKER_IP}" --disable-usage-stats --block
