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

# CUDA-13 mount-namespace re-exec — CRITICAL on worker nodes too. In multi-node
# runs Ray places the MegatronTrainRayActor on a WORKER (not the head); its
# raylet is this `ray start`, so TE inherits whatever namespace this process has.
# Without the mask here, TE finds the system cuda-12 beside torch's cu13 and
# aborts at the first real attention forward (compute_log_prob) — exactly the
# "Multiple libcudart" crash that killed job 19635 (~20min in, on worker .17).
# run.sh masks only the head; the worker must mask itself. Same guard/sentinel
# as run.sh; see cuda13_ns.sh + the arcagi_te_cudart_conflict memory.
if [ -z "${POLAR_IN_CUDA13_NS:-}" ]; then
    _tcm="$("${PYTHON_BIN}" -c 'import torch;print((torch.version.cuda or "").split(".")[0])' 2>/dev/null || echo "")"
    _cu13=""; for _c in "${TE_CUDA13_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda-13; do
        [ -n "$_c" ] && [ -d "$_c" ] && { _cu13="$_c"; break; }; done
    _cu12=0; for _c in /usr/local/cuda-12.9 /usr/local/cuda-12; do [ -d "$_c" ] && { _cu12=1; break; }; done
    if [ "${_tcm}" -ge 13 ] 2>/dev/null && [ -n "${_cu13}" ] && [ "${_cu12}" = 1 ] \
       && sudo -n -E unshare -m true >/dev/null 2>&1; then
        echo "[ray_worker] re-exec under cuda-13 mount namespace (masking system cuda-12 for TE)"
        export POLAR_IN_CUDA13_NS=1 TE_CUDA13_HOME="${_cu13}"
        exec sudo -n -E unshare -m bash "${SCRIPT_DIR}/cuda13_ns.sh" "$(id -un)" \
            bash "${BASH_SOURCE[0]}" "$@"
    fi
fi

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
