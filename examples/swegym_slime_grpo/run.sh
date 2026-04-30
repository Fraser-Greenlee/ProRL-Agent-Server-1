#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on SWE-Gym via Polar + Slime (Qwen3.5-4B).
#
# Qwen3.5-4B is a VLM checkpoint (Qwen3_5ForConditionalGeneration) with
# hybrid attention (1 full + 3 GatedDeltaNet linear per 4 layers). Text-only
# RL requires the SGLang VLM input_ids patch (see MEMORY.md).
#
# GPU layout (8x B200, default):
#   GPU 0-3     – Megatron GRPO training (TP=2, DP=2)
#   GPU 4-7     – SGLang inference (4 engines × TP=1, managed by Slime/Ray)
#
# Port layout:
#   9000        – SGLang router (slime-managed, load-balances engines)
#   8080        – Polar rollout server (task coordinator)
#   8100        – Polar gateway node (dispatches agent sessions)
#   8265        – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Slime manages SGLang engines; Polar gateway proxies LLM calls to them.
# Dynamic-history: every trace in each agent session becomes one training
# sample, so gradients learn from *every* turn (not just the last one).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

resolve_local_hf_checkpoint() {
    local model_id="$1"
    if is_path_like "$model_id"; then
        echo "$model_id"
        return
    fi

    local repo_cache="models--${model_id//\//--}"
    local cache_roots=()
    [ -n "${HF_HOME:-}" ] && cache_roots+=("$HF_HOME")
    [ -n "${HUGGINGFACE_HUB_CACHE:-}" ] && cache_roots+=("${HUGGINGFACE_HUB_CACHE%/hub}")
    cache_roots+=(
        "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/.cache/huggingface"
        "/lustre/fsw/portfolios/llmservice/users/haozh/.cache/huggingface"
        "${HOME:-}/.cache/huggingface"
    )

    local root snapshots_dir snapshot
    for root in "${cache_roots[@]}"; do
        [ -n "$root" ] || continue
        snapshots_dir="${root}/hub/${repo_cache}/snapshots"
        [ -d "$snapshots_dir" ] || continue
        for snapshot in "$snapshots_dir"/*; do
            [ -d "$snapshot" ] || continue
            if [ -f "$snapshot/config.json" ] && { [ -f "$snapshot/tokenizer.json" ] || [ -f "$snapshot/tokenizer_config.json" ]; }; then
                echo "$snapshot"
                return
            fi
        done
    done

    echo "$model_id"
}

detect_host_ip() {
    python - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
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
bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}"
    exit 1
fi

# ── Model ──────────────────────────────────────────────────────────
# Qwen3.5-4B: VLM checkpoint; we train text-only.  HF weights are loaded
# through slime_plugins.mbridge.qwen3_5 (text_config-aware) at convert-time.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
HF_CHECKPOINT_REQUESTED="${HF_CHECKPOINT:-$MODEL_NAME}"
HF_CHECKPOINT="$(resolve_local_hf_checkpoint "$HF_CHECKPOINT_REQUESTED")"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-$HF_CHECKPOINT}"
SGLANG_SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-$MODEL_NAME}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
mkdir -p "$SAVE_DIR"
if [ "$HF_CHECKPOINT" != "$HF_CHECKPOINT_REQUESTED" ]; then
    echo "Using local HF checkpoint snapshot: $HF_CHECKPOINT"
fi
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3.5-4B"
    exit 1
fi
if is_path_like "$TOKENIZER_MODEL" && [ ! -e "$TOKENIZER_MODEL" ]; then
    echo "ERROR: tokenizer model not found at $TOKENIZER_MODEL"
    exit 1
fi
if is_path_like "$HF_CHECKPOINT"; then
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/swegym_slime_grpo/convert_weights.sh first."
    exit 1
fi

# Mirrors slime/slime/scripts/models/qwen3.5-4B.sh.  --spec wires in the hybrid
# GatedDeltaNet + full-attention layer layout.  tie_word_embeddings=true in HF
# config → do NOT pass --untie-embeddings-and-output-weights.
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 2560
    --ffn-hidden-size 9216
    --normalization RMSNorm
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size 248320
    --rotary-base 10000000
)

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# By default, start from REF_LOAD. Set RESUME_FROM_SAVE=1 to continue from
# SAVE_DIR, and LOAD_OPTIM=1 only when the saved checkpoint is known to contain
# a compatible optimizer state. When LOAD_DIR points at a prior training
# checkpoint, load model/RNG-free by default so a replacement job can write to a
# fresh SAVE_DIR.
RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-0}"
LOAD_CHECKPOINT_ARGS=()
if [ -n "${LOAD_DIR:-}" ]; then
    echo "Using explicit load directory: $LOAD_DIR"
elif [ "$RESUME_FROM_SAVE" = "1" ] && [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
fi

if [ "$LOAD_DIR" != "$REF_LOAD" ] && [ "${LOAD_OPTIM:-0}" != "1" ]; then
    echo "Loading model weights from training checkpoint without optimizer/RNG state. Set LOAD_OPTIM=1 to load optimizer state."
    LOAD_CHECKPOINT_ARGS=(--no-load-optim --no-load-rng)
fi

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${SCRIPT_DIR}/swegym_train_293.jsonl"
EVAL_DATA="${SCRIPT_DIR}/swegym_eval_23.jsonl"
if [ ! -f "$PROMPT_DATA" ] || [ ! -f "$EVAL_DATA" ]; then
    echo "Preparing train/eval data..."
    python "${SCRIPT_DIR}/prepare_data.py"
fi

# ── Runtime configs ─────────────────────────────────────────────────
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"

python - "$TOPOLOGY_TEMPLATE" "$TOPOLOGY_PATH" "$SGLANG_ROUTER_BASE_URL" \
       "$POLAR_CONFIG_TEMPLATE" "$CUSTOM_CONFIG_PATH" "$AGENT_CLI_DIR" <<'PY'
from pathlib import Path
import os
import sys
import yaml

topology_template, topology_out, router_url, polar_template, polar_out, agent_cli_dir = sys.argv[1:]

def _env(name):
    value = os.environ.get(name)
    return None if value is None or value == "" else value

def _int_env(name):
    value = _env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc

with open(topology_template, encoding="utf-8") as fh:
    topology = yaml.safe_load(fh) or {}
for node in topology.get("gateway", {}).get("nodes", []):
    node.setdefault("sglang", {})["base_url"] = router_url
    for env_name, key in (
        ("POLAR_GATEWAY_MAX_INIT_WORKERS", "max_init_workers"),
        ("POLAR_GATEWAY_MAX_RUN_WORKERS", "max_run_workers"),
        ("POLAR_GATEWAY_MAX_POSTRUN_WORKERS", "max_postrun_workers"),
    ):
        value = _int_env(env_name)
        if value is not None:
            node[key] = value
Path(topology_out).parent.mkdir(parents=True, exist_ok=True)
with open(topology_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(topology, fh, sort_keys=False)

with open(polar_template, encoding="utf-8") as fh:
    polar_config = yaml.safe_load(fh) or {}
polar_config["polar_agent_cli_dir"] = agent_cli_dir

def _set_int(env_name, key, target=None):
    parsed = _int_env(env_name)
    if parsed is None:
        return
    dest = polar_config if target is None else target
    dest[key] = parsed

def _set_bool(env_name, key):
    value = _env(env_name)
    if value is None:
        return
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        polar_config[key] = True
    elif normalized in {"0", "false", "no", "off"}:
        polar_config[key] = False
    else:
        raise SystemExit(f"{env_name} must be boolean-like, got {value!r}")

for env_name, key in (
    ("POLAR_MAX_CONCURRENCY", "polar_max_concurrency"),
    ("POLAR_MAX_SESSION_CONCURRENCY", "polar_max_session_concurrency"),
    ("POLAR_MAX_ASYNC_LEVEL", "polar_max_async_level"),
    ("POLAR_MAX_OFF_POLICY_STEPS", "polar_max_off_policy_steps"),
    ("POLAR_REQUEST_TIMEOUT", "polar_request_timeout"),
    ("POLAR_WEIGHT_UPDATE_PAUSE_TIMEOUT", "polar_weight_update_pause_timeout"),
):
    _set_int(env_name, key)
_set_bool("POLAR_ALLOW_WEIGHT_UPDATE_OVERLAP", "polar_allow_weight_update_overlap")
task_template = polar_config.setdefault("polar_task_template", {})
_set_int("POLAR_TASK_TIMEOUT_SECONDS", "timeout_seconds", task_template)

Path(polar_out).parent.mkdir(parents=True, exist_ok=True)
with open(polar_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(polar_config, fh, sort_keys=False)
PY

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using Polar config: ${CUSTOM_CONFIG_PATH}"
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

# ── Step 1: Polar services (runs on host, CPU only) ───────────────
echo "=== Starting Polar rollout server (:8080) ==="
polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2

echo "=== Starting Polar gateway (:8100) ==="
polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2

curl -sf http://127.0.0.1:18080/health || { echo "Polar rollout server not healthy"; exit 1; }

# ── Step 2: Ray + Slime (manages SGLang engines + training) ───────
echo "=== Starting Ray (all 8 GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1

POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-cache-${USER:-user}-$$}"
mkdir -p \
    "${POLAR_JOB_CACHE_ROOT}/home" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-cache" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" \
    "${POLAR_JOB_CACHE_ROOT}/apptainer-work" \
    "${POLAR_JOB_CACHE_ROOT}/triton-cache" \
    "${POLAR_JOB_CACHE_ROOT}/triton-home" \
    "${POLAR_JOB_CACHE_ROOT}/torchinductor" \
    "${POLAR_JOB_CACHE_ROOT}/torch-extensions" \
    "${POLAR_JOB_CACHE_ROOT}/xdg" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-config" \
    "${POLAR_JOB_CACHE_ROOT}/xdg-runtime" \
    "${POLAR_JOB_CACHE_ROOT}/cuda-cache" \
    "${POLAR_JOB_CACHE_ROOT}/numba"
chmod 700 "${POLAR_JOB_CACHE_ROOT}/xdg-runtime"
export HOME="${POLAR_HOME:-${POLAR_JOB_CACHE_ROOT}/home}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-tmp}"
export APPTAINER_WORKDIR="${APPTAINER_WORKDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-work}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${APPTAINER_TMPDIR}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/triton-cache}"
export TRITON_HOME="${TRITON_HOME:-${POLAR_JOB_CACHE_ROOT}/triton-home}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${POLAR_JOB_CACHE_ROOT}/torch-extensions}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${POLAR_JOB_CACHE_ROOT}/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${POLAR_JOB_CACHE_ROOT}/xdg-config}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${POLAR_JOB_CACHE_ROOT}/xdg-runtime}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${POLAR_JOB_CACHE_ROOT}/cuda-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/numba}"
echo "Using local compile/cache root: ${POLAR_JOB_CACHE_ROOT}"

RAY_TMPDIR="${RAY_TMPDIR:-/tmp/polar-ray-${USER:-user}-$$}"
mkdir -p "$RAY_TMPDIR"
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats --temp-dir "$RAY_TMPDIR"

TRAIN_VENV="${POLR_TRAIN_VENV:-${PROJECT_ROOT}/.venv}"
TRAIN_SITE_PACKAGES="${POLR_TRAIN_SITE_PACKAGES:-${TRAIN_VENV}/lib/python3.12/site-packages}"
CUDNN_LIB="${CUDNN_LIB:-${TRAIN_SITE_PACKAGES}/nvidia/cudnn/lib}"
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HF_HOME\": \"${HF_HOME:-}\",
    \"HF_HUB_OFFLINE\": \"${HF_HUB_OFFLINE:-0}\",
    \"TRANSFORMERS_OFFLINE\": \"${TRANSFORMERS_OFFLINE:-0}\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"HOME\": \"${HOME}\",
    \"POLAR_JOB_CACHE_ROOT\": \"${POLAR_JOB_CACHE_ROOT}\",
    \"APPTAINER_CACHEDIR\": \"${APPTAINER_CACHEDIR}\",
    \"APPTAINER_TMPDIR\": \"${APPTAINER_TMPDIR}\",
    \"APPTAINER_WORKDIR\": \"${APPTAINER_WORKDIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TRITON_HOME\": \"${TRITON_HOME}\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TORCH_EXTENSIONS_DIR\": \"${TORCH_EXTENSIONS_DIR}\",
    \"XDG_CACHE_HOME\": \"${XDG_CACHE_HOME}\",
    \"XDG_CONFIG_HOME\": \"${XDG_CONFIG_HOME}\",
    \"XDG_RUNTIME_DIR\": \"${XDG_RUNTIME_DIR}\",
    \"CUDA_CACHE_PATH\": \"${CUDA_CACHE_PATH}\",
    \"NUMBA_CACHE_DIR\": \"${NUMBA_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"

ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
EVAL_INTERVAL="${EVAL_INTERVAL:-8}"
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-80000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
USE_WANDB="${USE_WANDB:-1}"
DIST_CKPT_STRICTNESS="${DIST_CKPT_STRICTNESS:-log_all}"

TRAINING_LENGTH_ARGS=()
if [ -n "$NUM_ROLLOUT" ]; then
    TRAINING_LENGTH_ARGS=(--num-rollout "$NUM_ROLLOUT")
else
    TRAINING_LENGTH_ARGS=(--num-epoch "$NUM_EPOCH")
fi

EVAL_ARGS=()
case "$EVAL_INTERVAL" in
    ""|0|none|None|NONE)
        ;;
    *)
        EVAL_ARGS=(--eval-prompt-data swegym_eval "$EVAL_DATA" --eval-interval "$EVAL_INTERVAL")
        ;;
esac

WANDB_ARGS=()
if [ "$USE_WANDB" = "1" ]; then
    if [ "${WANDB_MODE:-}" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
        echo "ERROR: WANDB_MODE=online requires WANDB_API_KEY in the environment."
        exit 1
    fi
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}"
        --wandb-group "${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}"
    )
    if [ -n "${WANDB_MODE:-}" ]; then
        WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
    fi
fi

# Rollout sizing: 8 prompts × 8 trajectories = 64 trajectories/rollout.
# This matches the earlier high-util baseline and keeps request groups smaller
# so long tails do not collapse usable token throughput.
# With --dynamic-history each trajectory explodes into one sample per trace,
# so sample count per rollout is variable.
# The custom data source rounds epoch length up to 37 rollout batches, so all
# 293 train prompts are consumed once; the final fixed-size batch wraps 3 prompts.
echo "=== Launching train_async.py ==="
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --tokenizer-type HuggingFaceTokenizer \
    --no-use-tokenizer-model-from-checkpoint-args \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    "${LOAD_CHECKPOINT_ARGS[@]}" \
    --save "$SAVE_DIR" \
    --dist-ckpt-strictness "$DIST_CKPT_STRICTNESS" \
    --save-interval "${SAVE_INTERVAL:-20}" \
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
    "${TRAINING_LENGTH_ARGS[@]}" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --n-samples-per-eval-prompt 1 \
    "${EVAL_ARGS[@]}" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 \
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
    --log-probs-chunk-size 512 \
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
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --sglang-served-model-name "$SGLANG_SERVED_MODEL_NAME" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    "${WANDB_ARGS[@]}" \
    --sglang-router-port "$SGLANG_ROUTER_PORT"
