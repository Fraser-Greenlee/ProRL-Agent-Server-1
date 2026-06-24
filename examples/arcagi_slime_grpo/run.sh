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
echo "=== Starting Polar rollout server (:8080) ==="
polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2
echo "=== Starting Polar gateway (:8100) ==="
polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2
curl -sf http://127.0.0.1:8080/health || { echo "Polar rollout server not healthy"; exit 1; }

# ── Step 2: Ray + Slime (SGLang engines + training) ────────────────
# 4 train (TP=4) + 4 serve (1 engine, TP=4).
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-4}"

# Conservative GRPO knobs for 27B (smaller groups + recompute to fit).
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-12000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-40000}"

RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"

echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats

if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
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
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"NVTE_CUDA_INCLUDE_DIR\": \"${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME:-/usr/local/cuda}/include}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\"
  }
}"

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
    --actor-num-nodes 1 \
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
    --rollout-max-prompt-len 32000 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE" \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
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
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    ${WANDB_ARGS[@]+"${WANDB_ARGS[@]}"} \
    --sglang-router-port "$SGLANG_ROUTER_PORT"
