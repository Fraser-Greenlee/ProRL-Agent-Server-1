#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on ARC-AGI compression via Polar + Slime
# (Fraser/Qwen3.6-27B-ARC-Hy).
#
# Qwen3.6-27B is a dense hybrid checkpoint (Qwen3_5ForConditionalGeneration;
# 3 GatedDeltaNet linear + 1 full-attention per 4 layers, 64 layers). Text-only
# RL requires the SGLang VLM input_ids patch (see swegym example / MEMORY.md).
#
# GPU split on one 8×H100 node: 4 train (TP=4) + 4 serve (one SGLang engine,
# TP=4). Conservative batch/recompute settings to fit 27B; tune up once stable.
#
# Port layout:
#   9000   – SGLang router (slime-managed, load-balances engines)
#   8080   – Polar rollout server (task coordinator)
#   8100   – Polar gateway node (dispatches agent sessions, launches containers)
#   8265   – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Each agent rollout runs in the polar-arcagi docker image; the image is loaded
# from the shared-NFS tarball onto this node before Polar starts (load_image.sh).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Secrets (WANDB_API_KEY etc.) live in a gitignored env file, never in a tracked
# file. Looked for at the repo root, then this example dir.
for envf in "${PROJECT_ROOT}/.env.local" "${SCRIPT_DIR}/.env.local"; do
    if [ -f "$envf" ]; then
        set -a; # shellcheck disable=SC1090
        source "$envf"; set +a
    fi
done

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/arcagi_slime_grpo}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# uv + CUDA toolkit on PATH (non-login GPU shell); NVTE_CUDA_INCLUDE_DIR works
# around the TE 2.5.0 Path(nvidia.__file__=None) import crash. See launch_e2e.sh.
export PATH="${HOME}/.local/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[ -x "${CUDA_HOME}/bin/nvcc" ] && export PATH="${CUDA_HOME}/bin:${PATH}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/include}"
# CRITICAL: the venv bin must come FIRST so `ray`/`polar`/`python` resolve to
# the venv (py3.12, ray 2.55), NOT ~/.local/bin (a py3.10 ray that starts a
# cluster slime can't connect to). Re-prepend after the .local/cuda prepends.
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# ── CUDA-13 mount-namespace re-exec (Transformer Engine vs system cuda-12) ──
# TE's cu13 build aborts at the first kernel with "Multiple libcudart libraries
# found" because its compiled loader scans /usr/local/cuda* and finds the system
# 12.9 toolkit beside torch's cu13 runtime. No env/LD_LIBRARY_PATH/LD_PRELOAD
# fixes it (it's a filesystem-existence scan, and it probes /usr/local/cuda-12
# independent of CUDA_HOME — verified by strace). We can't remove the shared
# system toolkit. Fix: re-exec this whole script inside a PRIVATE mount namespace
# that bind-mounts the cu13 toolkit OVER the cuda-12 paths. Global /usr/local is
# untouched for other users; Ray actors forked by the in-ns raylet inherit the
# mask (verified: TE fused-attn runs to completion). See cuda13_ns.sh and the
# arcagi_te_cudart_conflict memory. SGLang serving workers launch on other nodes
# via ray_worker.sh (outside this ns) and are unaffected.
#   Guarded/idempotent: only re-execs when torch is cu13 AND a cuda-12 toolkit
#   coexists AND sudo+unshare actually work; otherwise runs normally (portable to
#   clean single-CUDA nodes). POLAR_IN_CUDA13_NS sentinel prevents re-exec loops.
if [ -z "${POLAR_IN_CUDA13_NS:-}" ]; then
    _torch_cuda_major="$("${PYTHON_BIN}" -c 'import torch;print((torch.version.cuda or "").split(".")[0])' 2>/dev/null || echo "")"
    _cu13_home=""
    for _c in "${TE_CUDA13_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda-13; do
        [ -n "$_c" ] && [ -d "$_c" ] && { _cu13_home="$_c"; break; }
    done
    _cu12_present=0
    for _c in /usr/local/cuda-12.9 /usr/local/cuda-12; do
        [ -d "$_c" ] && { _cu12_present=1; break; }
    done
    if [ "${_torch_cuda_major}" -ge 13 ] 2>/dev/null && [ -n "${_cu13_home}" ] && [ "${_cu12_present}" = 1 ]; then
        # `sudo -n` SCRUBS the environment; `sudo -n -E` preserves it (verified
        # the sudoers here allows SETENV). We need the full env (SLURM_*,
        # HF_CHECKPOINT, SAVE_DIR, WANDB_*, the PATH/VENV exports above) to reach
        # the re-exec'd run.sh, so -E is required, not optional.
        if sudo -n -E unshare -m true >/dev/null 2>&1; then
            echo "=== Re-exec under cuda-13 mount namespace (masking system cuda-12 for TE) ==="
            export POLAR_IN_CUDA13_NS=1 TE_CUDA13_HOME="${_cu13_home}"
            exec sudo -n -E unshare -m \
                bash "${SCRIPT_DIR}/cuda13_ns.sh" "$(id -un)" \
                bash "${BASH_SOURCE[0]}" "$@"
        else
            echo "WARNING: torch is cu${_torch_cuda_major} with a system cuda-12 toolkit present, but" >&2
            echo "  'sudo -n -E unshare -m' is unavailable — TE will likely abort with 'Multiple libcudart'." >&2
            echo "  Install passwordless sudo (with SETENV) for unshare, or remove one CUDA major from /usr/local." >&2
        fi
    fi
fi

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0]); sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
if [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}"
    exit 1
fi

# ── Model ──────────────────────────────────────────────────────────
HF_CHECKPOINT="${HF_CHECKPOINT:-Fraser/Qwen3.6-27B-ARC-Hy}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.6-27B-ARC-Hy_torch_dist}"
RUN_ID="${RUN_ID:-arcagi-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/arcagi_slime_grpo_qwen36_27b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"; exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/arcagi_slime_grpo/convert_weights.sh first."
    exit 1
fi

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
fi

# ── Runtime image (load from NFS tarball onto this node) ────────────
export ARCAGI_IMAGE="${ARCAGI_IMAGE:-polar-arcagi:latest}"
export ARCAGI_IMAGE_TARBALL="${ARCAGI_IMAGE_TARBALL:-/home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz}"
IMAGE="$ARCAGI_IMAGE" TARBALL="$ARCAGI_IMAGE_TARBALL" \
    bash "${SCRIPT_DIR}/load_image.sh"

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/arcagi_train.jsonl}"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "Preparing train data..."
    AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" --n-tasks "${N_TASKS:-40}"
fi

# ── Polar service wiring ────────────────────────────────────────────
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${RUN_DIR}/compiler_cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILER_CACHE_ROOT}/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILER_CACHE_ROOT}/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

command -v envsubst >/dev/null || { echo "ERROR: envsubst not found (install gettext-base)"; exit 1; }
# Only SGLANG_ROUTER_BASE_URL is templated; literal $HOME etc. in polar_config
# are left untouched.
TEMPLATE_VARS='${SGLANG_ROUTER_BASE_URL}'
mkdir -p "$(dirname "$TOPOLOGY_PATH")" "$(dirname "$CUSTOM_CONFIG_PATH")"
envsubst "$TEMPLATE_VARS" < "$TOPOLOGY_TEMPLATE"     > "$TOPOLOGY_PATH"
envsubst "$TEMPLATE_VARS" < "$POLAR_CONFIG_TEMPLATE" > "$CUSTOM_CONFIG_PATH"

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using Polar config: ${CUSTOM_CONFIG_PATH}"
echo "Using run id: ${RUN_ID}"
echo "Using save dir: ${SAVE_DIR}"
echo "Using SGLang router URL for Polar gateway: ${SGLANG_ROUTER_BASE_URL}"

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 1: Polar services (host, CPU only) ─────────────────────────
# The gateway process runs the reward evaluator, whose import path is
# `examples.arcagi_slime_grpo.arc_compress_evaluator:ArcCompressEvaluator`.
# examples/ has no __init__.py (namespace package), so PROJECT_ROOT must be on
# PYTHONPATH for that in-process class import (registry _import_class) to
# resolve — otherwise every session ends ERROR "evaluator failed: No module
# named 'examples'", yielding zero reward / zero trainable tokens and dropped
# GRPO groups (jobs 19612/19616). NOTE: a plain `export PYTHONPATH` is NOT
# enough here — run.sh re-execs through `runuser` (the cuda-13 namespace wrap),
# and runuser/PAM SANITIZES PYTHONPATH across the user boundary, so it arrives
# empty. Pass it EXPLICITLY to each service via `env` so it can't be stripped.
POLAR_PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "=== Starting Polar rollout server (:8080) ==="
env PYTHONPATH="${POLAR_PYTHONPATH}" polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2
echo "=== Starting Polar gateway (:8100) ==="
env PYTHONPATH="${POLAR_PYTHONPATH}" polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2
curl -sf http://127.0.0.1:8080/health || { echo "Polar rollout server not healthy"; exit 1; }

# ── Step 2: Ray + Slime (SGLang engines + training) ────────────────
# GPU split. Single-node default: 4 train (TP=4) + 4 serve. Multi-node (the
# 27B fit): node0 = 8-GPU training (TP=8), node1 = 8-GPU SGLang serving.
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
if [ "${NNODES}" -ge 2 ]; then
    # Training uses ACTOR_NUM_NODES nodes (TP=8 per node); the REMAINING nodes
    # serve rollouts (one TP=8 SGLang engine each). ACTOR_NUM_NODES>1 lets the
    # train actor span nodes for context-parallel (CONTEXT_PARALLEL_SIZE>1),
    # which shards the SEQUENCE across GPUs and halves per-GPU activation memory
    # — the fix for the train-step OOM at 64K tokens (TP=8 alone uses all 8 GPUs
    # of one node, so CP>1 needs >1 train node). Must satisfy
    # TP*CP*PP == ACTOR_NUM_NODES*GPUS_PER_NODE.
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
    ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
    ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$(( (NNODES - ACTOR_NUM_NODES) * GPUS_PER_NODE ))}"
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
    TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
    CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
else
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
    ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
    ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
    ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
    TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
    CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
fi

# Conservative GRPO knobs for 27B (smaller groups + recompute to fit).
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# MAX_TOKENS_PER_GPU does double duty: it's Megatron's per-GPU dynamic-batch
# budget AND (via slime_bridge _resolve_max_tokens = mtpg * cp_size) the cap
# above which a trace is DROPPED before training. At 12000 it dropped EVERY
# real trace — the ARC instruction prefix alone is ~25K tokens, sessions run
# 32-47K — yielding all-placeholder groups and "zero trainable tokens / no
# progress" (job 19539). Must be >= the longest admissible trace. With
# rollout-max-prompt-len 49152 + response 16000 the ceiling is ~65K, so match
# the context window. Memory: at TP=8 each rank had ~64 GB free post-weights;
# with full recompute + CPU optimizer offload one ~65K micro-batch fits (~50 GB
# est). If a real step OOMs, fall back to context-parallel-size 2 (halves
# per-GPU sequence memory) rather than re-lowering this (re-lowering silently
# drops long traces again).
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
# 65536: the KV pool is sized by --sglang-mem-fraction-static (≈1.69M tokens at
# 0.8, TP=4), NOT by context length, so raising this from 40000 costs no extra
# memory — it only lifts the per-sequence cap. Measured agent rollouts grow
# ~600 tok/turn off a ~25K instruction floor; a single large tool/eval output
# once spiked a turn by ~9.6K and overran 40000. 64K gives >1.5x headroom over
# the worst observed sequence while staying far under max_position_embeddings
# (262144). See DEBUG_STATE.md (jobs 19536/19537).
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"

ray stop --force 2>/dev/null || true
sleep 1

if [ "${NNODES}" -ge 2 ]; then
    # ── Multi-node Ray: head on this (node0), workers join via srun ──
    # Derive node IPs from the slurm allocation. MASTER_ADDR = node0 (here).
    MASTER_ADDR="$(detect_host_ip)"
    export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
    RAY_HEAD_IP="${MASTER_ADDR}"
    echo "=== Multi-node Ray: head on ${MASTER_ADDR} (${ACTOR_NUM_GPUS_PER_NODE} GPUs), ${NNODES} nodes ==="
    ray start --head --node-ip-address "${MASTER_ADDR}" \
        --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    sleep 5
    # Join every OTHER node in the allocation as a Ray worker via srun. Each
    # contributes GPUS_PER_NODE to the cluster (rollout/serving lands there).
    WORKER_NODES="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tail -n +2)"
    for wn in ${WORKER_NODES}; do
        echo "  starting Ray worker on ${wn}"
        # srun defaults to --export=ALL, so this (already-namespaced) run.sh
        # would leak POLAR_IN_CUDA13_NS=1 to the worker, making ray_worker.sh
        # think it's already masked and SKIP its own namespace — leaving the
        # worker's cuda-12 visible to any train actor Ray places there (the
        # job-19635 crash). `env -u` clears the sentinel so each worker applies
        # its OWN mask on its OWN node.
        # Absolute /usr/bin/env — a non-exec ~/.local/bin/env on PATH otherwise
        # makes srun's execve fail (exit 13), so no worker joins and the
        # placement group hangs waiting for GPUs (job 19639).
        srun --nodes=1 --ntasks=1 -w "${wn}" \
            /usr/bin/env -u POLAR_IN_CUDA13_NS \
            bash "${SCRIPT_DIR}/ray_worker.sh" "${MASTER_ADDR}" "${GPUS_PER_NODE}" \
            >> "${PROJECT_ROOT}/logs/ray_worker_${wn}.log" 2>&1 &
    done
    sleep 20  # let workers register
    echo "=== Ray cluster nodes ==="; ray list nodes 2>/dev/null | head -20 || true
else
    RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
    RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
    echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs) ==="
    ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats
fi

if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi

# CUDA library path for the TRAINING Ray job. Transformer Engine's runtime does
# a by-name scan for libcudart and ABORTS if it finds two majors ("Multiple
# libcudart libraries found"). torch 2.11 + TE here are CUDA 13, but the
# cluster's DEFAULT system toolkit at /usr/local/cuda is 12.9 — if its lib64 is
# on LD_LIBRARY_PATH (it is, via the login env) TE finds libcudart.so.12 next to
# the cu13 .so.13 and dies at the first attention kernel. Fix: when torch is
# cu13, set CUDA_HOME to the cu13 toolkit and build LD_LIBRARY_PATH from cu13
# libs (system cuda-13 toolkit lib64 + cuDNN/NCCL pip wheels), EXCLUDING the
# 12.9 default. (Verified by strace: clean cu13-only path runs TE fused-attn to
# exit 0; see arcagi_te_cudart_conflict memory.) Two copies of .so.13 are fine —
# TE only rejects mixed MAJORS. SGLang engines are separate processes — unaffected.
TORCH_CUDA_MAJOR="$("${PYTHON_BIN}" -c 'import torch;print((torch.version.cuda or "").split(".")[0])' 2>/dev/null || echo "")"
TRAIN_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ "${TORCH_CUDA_MAJOR}" -ge 13 ] 2>/dev/null; then
    # Prefer the real side-by-side cu13 toolkit (matches the TE build); fall back
    # to the pip nvidia/cu13 wheel dir if no system toolkit is present.
    TRAIN_CUDA_HOME=""
    for cand in "${TE_CUDA13_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda-13; do
        [ -n "$cand" ] && [ -d "$cand" ] && { TRAIN_CUDA_HOME="$cand"; break; }
    done
    if [ -z "${TRAIN_CUDA_HOME}" ]; then
        TRAIN_CUDA_HOME="$("${PYTHON_BIN}" -c 'import os,nvidia.cu13 as c;print(list(c.__path__)[0])' 2>/dev/null || echo "/usr/local/cuda")"
    fi
    NCCL_LIB="$("${PYTHON_BIN}" -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))' 2>/dev/null || true)"
    # cu13 libs ONLY — deliberately omit ${LD_LIBRARY_PATH} so the 12.9 default
    # can't leak a libcudart.so.12 into TE's scan. Cover both toolkit layouts.
    RUNTIME_LD_LIBRARY_PATH=""
    for d in "${TRAIN_CUDA_HOME}/targets/x86_64-linux/lib" "${TRAIN_CUDA_HOME}/lib64" "${TRAIN_CUDA_HOME}/lib" "${CUDNN_LIB}" "${NCCL_LIB}"; do
        [ -n "$d" ] && [ -d "$d" ] && RUNTIME_LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH:+${RUNTIME_LD_LIBRARY_PATH}:}${d}"
    done
    echo "Training CUDA major=${TORCH_CUDA_MAJOR}: CUDA_HOME=${TRAIN_CUDA_HOME}; cu13-only LD path (12.9 default excluded)."
else
    # cu12x torch: original behavior (system toolkit is the right major).
    RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
        RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
    fi
fi

# PROJECT_ROOT on PYTHONPATH so the evaluator import path
# (examples.arcagi_slime_grpo.arc_compress_evaluator:ArcCompressEvaluator)
# resolves in the gateway process.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src:${PROJECT_ROOT}\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_HOME\": \"${TRAIN_CUDA_HOME}\",
    \"NVTE_CUDA_INCLUDE_DIR\": \"${TRAIN_CUDA_HOME}/include\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\"
  }
}"

# Env vars for the MegatronTrainRayActor specifically. slime builds the train
# actor's env from a fixed dict + args.train_env_vars (actor_group.py) — it does
# NOT inherit the ray-job RUNTIME_ENV_JSON, so actor-only vars must go here.
# NVTE_TORCH_COMPILE=0: Transformer Engine wraps its fused ops in torch.compile
# by default (jit.py, NVTE_TORCH_COMPILE defaults to 1). On Qwen3.6's hybrid
# attention at TP=8 that dynamo trace hit a fake-tensor view error
# (view(...,6,256)->768; 6*256=1536) and crashed the first training step (job
# 19640). Disabling TE's compile runs the ops eagerly — we're not throughput-
# bound, and it either avoids the dynamo mis-trace or surfaces a clearer eager
# error. See arcagi_te_cudart_conflict memory (UPDATE 7).
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: job 19644 reached accepted=3/4
# (gate fix worked, past attention) then OOM'd in the train step — failing alloc
# was only 32 MiB against 68 GB already allocated, i.e. FRAGMENTATION not a hard
# capacity wall. The alloc-conf in RUNTIME_ENV_JSON does NOT reach the actor
# (same inheritance gap), and it wrongly combined max_split_size_mb with
# expandable_segments (mutually exclusive). Set the correct value HERE so the
# actor gets it; expandable_segments defragments and is what the OOM msg itself
# recommends. If it still OOMs, escalate to --context-parallel-size 2.
TRAIN_ENV_VARS_JSON="${TRAIN_ENV_VARS_JSON:-{\"NVTE_TORCH_COMPILE\": \"0\", \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"}}"

# W&B: only enable if a key is present (sourced from .env.local).  Without one,
# train offline-disabled rather than hard-failing on login.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    "${PYTHON_BIN}" - <<'PY' || true
import os
try:
    import wandb
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
except Exception:
    pass
PY
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-polar-arcagi-grpo}"
        --wandb-group "${WANDB_GROUP:-arcagi-qwen36-27b-async-grpo}"
    )
else
    echo "WARNING: WANDB_API_KEY not set — training metrics will NOT be logged to W&B."
    echo "  Put it in ${PROJECT_ROOT}/.env.local (gitignored) to enable."
fi

echo "=== Launching train_async.py (Qwen3.6-27B, TP=${TENSOR_MODEL_PARALLEL_SIZE}) ==="
ray job submit --address="http://${RAY_HEAD_IP}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes "$ACTOR_NUM_NODES" \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --save-interval "${SAVE_INTERVAL:-10}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-epoch "${NUM_EPOCH:-50}" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --rollout-max-response-len 16000 \
    --rollout-max-prompt-len 49152 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE" \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size 256 \
    --distributed-timeout-minutes 30 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --optimizer-cpu-offload \
    --optimizer-offload-fraction 1.0 \
    --use-precision-aware-optimizer \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    --sglang-disable-custom-all-reduce \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
    --sglang-router-port "$SGLANG_ROUTER_PORT"
