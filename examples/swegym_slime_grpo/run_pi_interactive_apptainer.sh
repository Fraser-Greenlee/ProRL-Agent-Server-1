#!/usr/bin/env bash
# Interactive SWE-Gym Slime GRPO launcher for PI + Qwen3.5-4B.
#
# Default mode is a small smoke run:
#   bash examples/swegym_slime_grpo/run_pi_interactive_apptainer.sh
#
# Full 293-row training from an interactive allocation:
#   SMOKE_NUM_ROWS=0 NUM_EPOCH=1 bash examples/swegym_slime_grpo/run_pi_interactive_apptainer.sh
#
# Dry-run preflight without starting Ray/SGLang/Slime:
#   DRY_RUN=1 bash examples/swegym_slime_grpo/run_pi_interactive_apptainer.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar/ProRL-Agent-Server}"
cd "${PROJECT_ROOT}"

log() {
    printf '[%s] %s\n' "$(date +'%F %T')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "${PROJECT_ROOT}" "$1" ;;
    esac
}

first_existing_dir() {
    for candidate in "$@"; do
        if [ -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    printf '%s\n' "$1"
}

first_existing_file() {
    for candidate in "$@"; do
        if [ -f "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    printf '%s\n' "$1"
}

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count
        set +e
        count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        set -e
        printf '%s\n' "${count:-0}"
    else
        printf '0\n'
    fi
}

detect_cpu_count() {
    local count
    count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '32')"
    case "${count}" in
        ''|*[!0-9]*) printf '32\n' ;;
        *) printf '%s\n' "${count}" ;;
    esac
}

detect_host_ip() {
    python3 - <<'PY'
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

checkpoint_marker_exists() {
    [ -f "$1/latest_checkpointed_iteration.txt" ]
}

choose_ref_load() {
    local current="${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist"
    local reference="${REFERENCE_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist"
    if checkpoint_marker_exists "${current}"; then
        printf '%s\n' "${current}"
    else
        printf '%s\n' "${reference}"
    fi
}

choose_agent_cli_dir() {
    local current="${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node"
    local reference="${REFERENCE_ROOT}/tmp/swegym_agent_cli/opt_node"
    if [ -x "${current}/bin/pi" ] && [ -x "${current}/bin/node" ]; then
        printf '%s\n' "${current}"
    elif [ -x "${reference}/bin/pi" ] && [ -x "${reference}/bin/node" ]; then
        printf '%s\n' "${reference}"
    else
        printf '%s\n' "${current}"
    fi
}

choose_hf_checkpoint() {
    local local_model="/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/model/Qwen3.5-4B"
    if [ -d "${local_model}" ]; then
        printf '%s\n' "${local_model}"
    else
        printf '%s\n' "Qwen/Qwen3.5-4B"
    fi
}

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

# =============================================================================
# User settings
# =============================================================================
RUN_ID="${RUN_ID:-swegym_pi_qwen35_4b_$(date +%Y%m%d_%H%M%S)}"

# Model names:
# - MODEL_NAME is the model served by SGLang.
# - PI_MODEL_NAME must be provider/model form for the pi CLI.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
PI_MODEL_NAME="${PI_MODEL_NAME:-openai/${MODEL_NAME}}"
HF_CHECKPOINT="${HF_CHECKPOINT:-$(choose_hf_checkpoint)}"
REF_LOAD="$(abs_path "${REF_LOAD:-$(choose_ref_load)}")"
SAVE_DIR="$(abs_path "${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/${RUN_ID}}")"
FRESH_START="${FRESH_START:-1}"
RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-0}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"

# Data. SMOKE_NUM_ROWS=1 makes a quick one-task debug run. Set 0 for all rows.
FULL_PROMPT_DATA="$(abs_path "${FULL_PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}")"
SMOKE_NUM_ROWS="${SMOKE_NUM_ROWS:-1}"
RUN_DIR="$(abs_path "${RUN_DIR:-${PROJECT_ROOT}/tmp/${RUN_ID}}")"
RUN_LOG_DIR="$(abs_path "${RUN_LOG_DIR:-${RUN_DIR}/logs}")"
ROLLOUT_SAVE_DIR="$(abs_path "${ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}")"
PROMPT_DATA="$(abs_path "${PROMPT_DATA:-${RUN_DIR}/swegym_train_${SMOKE_NUM_ROWS}.jsonl}")"

# Training/runtime container and dependency checkouts.
TRAIN_SQSH_DEFAULT="$(first_existing_file \
    "${PROJECT_ROOT}/tmp/polr_swegym_slime_grpo_train.sqsh" \
    "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_slime_grpo_train_codex_yazi_v2.sqsh")"
TRAIN_SQSH="$(abs_path "${TRAIN_SQSH:-${POLR_TRAIN_SQSH:-${TRAIN_SQSH_DEFAULT}}}")"
USE_TRAIN_SQSH="${USE_TRAIN_SQSH:-1}"
SLIME_DIR="$(abs_path "${SLIME_DIR:-$(first_existing_dir "${PROJECT_ROOT}/slime" "${REFERENCE_ROOT}/slime")}")"
MEGATRON_DIR="$(abs_path "${MEGATRON_DIR:-$(first_existing_dir "${PROJECT_ROOT}/Megatron-LM" "${REFERENCE_ROOT}/Megatron-LM")}")"

# SWE-Gym task SIFs and shared Node/agent CLI assets.
SHARED_SIF_DIR="${SHARED_SIF_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/singularity_images_v3}"
APPTAINER_IMAGE_DIR="$(abs_path "${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}")"
AGENT_CLI_DIR="$(abs_path "${AGENT_CLI_DIR:-$(choose_agent_cli_dir)}")"
PREPARE_AGENT_CLI="${PREPARE_AGENT_CLI:-0}"
PREPARE_MISSING_SIFS="${PREPARE_MISSING_SIFS:-0}"

# GPU split. Current example uses 2 actor GPUs and the rest for rollout.
DETECTED_GPUS="$(detect_gpu_count)"
TOTAL_GPUS="${TOTAL_GPUS:-${DETECTED_GPUS:-4}}"
[ "${TOTAL_GPUS}" -gt 0 ] || TOTAL_GPUS=4
DETECTED_CPUS="$(detect_cpu_count)"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${DETECTED_CPUS}}"
if [ "${RAY_NUM_CPUS}" -gt 32 ]; then
    RAY_NUM_CPUS=32
fi
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-2}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((TOTAL_GPUS - TRAIN_NUM_GPUS))}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"

# Smoke defaults are intentionally small; full-run defaults mirror current run.sh.
if [ "${SMOKE_NUM_ROWS}" -gt 0 ]; then
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
    NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
    POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-1200}"
    POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-1200}"
else
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
    NUM_ROLLOUT="${NUM_ROLLOUT:-}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
    POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-1200}"
    POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-1200}"
fi
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-60000}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
TRAIN_LR="${TRAIN_LR:-1e-6}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"

# Polar/agent settings. PI_API_TYPE defaults to pi-ai's OpenAI chat-completions provider.
POLAR_BUILDER_STRATEGY="${POLAR_BUILDER_STRATEGY:-prefix_merging}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
if [ "${SMOKE_NUM_ROWS}" -gt 0 ]; then
    POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-1}"
else
    POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
fi
# Optional per-command Apptainer sandbox memory cap. Empty disables the cap.
POLAR_RUNTIME_MEMORY_MB="${POLAR_RUNTIME_MEMORY_MB:-}"
PI_API_TYPE="${PI_API_TYPE:-openai-completions}"
PI_MAX_TOKENS="${PI_MAX_TOKENS:-512}"
PI_CONTEXT_WINDOW="${PI_CONTEXT_WINDOW:-32000}"
if [ "${PI_CONTEXT_WINDOW}" -le 0 ]; then
    fail "PI_CONTEXT_WINDOW must be positive; got ${PI_CONTEXT_WINDOW}"
fi
PI_THINKING="${PI_THINKING:-}"
PI_FAIL_ON_CONTEXT_LIMIT="${PI_FAIL_ON_CONTEXT_LIMIT:-1}"
if [ "${PI_FAIL_ON_CONTEXT_LIMIT}" = "1" ]; then
    PI_COMPACTION_ENABLED="${PI_COMPACTION_ENABLED:-false}"
    PI_RETRY_ENABLED="${PI_RETRY_ENABLED:-false}"
else
    PI_COMPACTION_ENABLED="${PI_COMPACTION_ENABLED:-true}"
    PI_RETRY_ENABLED="${PI_RETRY_ENABLED:-true}"
fi
PI_RETRY_MAX_RETRIES="${PI_RETRY_MAX_RETRIES:-0}"
PI_PROVIDER_MAX_RETRIES="${PI_PROVIDER_MAX_RETRIES:-0}"

# Local ports. Override these for parallel runs on one node.
ROLLOUT_PORT="${ROLLOUT_PORT:-18080}"
GATEWAY_PORT="${GATEWAY_PORT:-18100}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-26000}"
SLIME_SGLANG_BASE_PORT="${SLIME_SGLANG_BASE_PORT:-24000}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
RAY_USE_EXISTING_CLUSTER="${RAY_USE_EXISTING_CLUSTER:-0}"
RAY_STOP_ON_EXIT="${RAY_STOP_ON_EXIT:-1}"
USE_RAY_JOB_SUBMIT="${USE_RAY_JOB_SUBMIT:-0}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_TMPDIR="$(abs_path "${RAY_TMPDIR:-/tmp/polar-ray-${USER:-user}-$$}")"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"

# Apptainer caches. Keep these short because Ray uses Unix sockets under tmp.
POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"
POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC:-1}"
POLAR_JOB_CACHE_ROOT="${POLAR_JOB_CACHE_ROOT:-/tmp/polar-swegym-pi-${USER:-user}-${RUN_ID}}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-tmp}"
APPTAINER_WORKDIR="${APPTAINER_WORKDIR:-${POLAR_JOB_CACHE_ROOT}/apptainer-work}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/triton-cache}"
TRITON_HOME="${TRITON_HOME:-${POLAR_JOB_CACHE_ROOT}/triton-home}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/torchinductor}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${POLAR_JOB_CACHE_ROOT}/torch-extensions}"
XDG_CACHE_HOME="${POLAR_XDG_CACHE_HOME:-${POLAR_JOB_CACHE_ROOT}/xdg-cache}"
XDG_CONFIG_HOME="${POLAR_XDG_CONFIG_HOME:-${POLAR_JOB_CACHE_ROOT}/xdg-config}"
XDG_RUNTIME_DIR="${POLAR_XDG_RUNTIME_DIR:-${POLAR_JOB_CACHE_ROOT}/xdg-runtime}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${POLAR_JOB_CACHE_ROOT}/cuda-cache}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${POLAR_JOB_CACHE_ROOT}/numba}"
FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-${POLAR_JOB_CACHE_ROOT}/flashinfer-cache}"
HOST_NVIDIA_LIB_DIR="$(abs_path "${HOST_NVIDIA_LIB_DIR:-${PROJECT_ROOT}/tmp/host-nvidia-libs}")"

# W&B is enabled by default. The key is passed through env only and is not echoed.
DEFAULT_WANDB_API_KEY=""
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-polar-swegym-pi-qwen35-4b}"
WANDB_GROUP="${WANDB_GROUP:-swegym-pi-qwen35-4b-full293-t1200}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID}}"
WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"
WANDB_TEAM="${WANDB_TEAM:-${WANDB_ENTITY:-}}"
WANDB_HOST="${WANDB_HOST:-}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-${DEFAULT_WANDB_API_KEY}}}"
WANDB_DIR="$(abs_path "${WANDB_DIR:-${PROJECT_ROOT}/logs/wandb}")"

# Preparation and diagnostics.
PATCH_SLIME="${PATCH_SLIME:-1}"
PATCH_SGLANG="${PATCH_SGLANG:-1}"
PATCH_RAY_PY312="${PATCH_RAY_PY312:-1}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-0}"
SLIME_ZERO_NONFINITE_GRADS="${SLIME_ZERO_NONFINITE_GRADS:-1}"
DRY_RUN="${DRY_RUN:-0}"

resolve_gpu_split() {
    [ "${TRAIN_NUM_GPUS}" -gt 0 ] || die "TRAIN_NUM_GPUS must be positive"
    [ "${ROLLOUT_NUM_GPUS}" -gt 0 ] || die "ROLLOUT_NUM_GPUS must be positive"
    [ "$((TRAIN_NUM_GPUS + ROLLOUT_NUM_GPUS))" -le "${TOTAL_GPUS}" ] || \
        die "TRAIN_NUM_GPUS + ROLLOUT_NUM_GPUS exceeds TOTAL_GPUS: train=${TRAIN_NUM_GPUS}, rollout=${ROLLOUT_NUM_GPUS}, total=${TOTAL_GPUS}"

    if [ -z "${ACTOR_NUM_GPUS_PER_NODE}" ]; then
        [ "$((TRAIN_NUM_GPUS % ACTOR_NUM_NODES))" -eq 0 ] || \
            die "TRAIN_NUM_GPUS (${TRAIN_NUM_GPUS}) must be divisible by ACTOR_NUM_NODES (${ACTOR_NUM_NODES})"
        ACTOR_NUM_GPUS_PER_NODE="$((TRAIN_NUM_GPUS / ACTOR_NUM_NODES))"
    fi
    [ "$((TRAIN_NUM_GPUS % TENSOR_MODEL_PARALLEL_SIZE))" -eq 0 ] || \
        die "TRAIN_NUM_GPUS (${TRAIN_NUM_GPUS}) must be divisible by TENSOR_MODEL_PARALLEL_SIZE (${TENSOR_MODEL_PARALLEL_SIZE})"
}

host_library_path() {
    local lib="$1"
    ldconfig -p 2>/dev/null | awk -v lib="${lib}" '$1 == lib { print $NF; exit }'
}

prepare_host_nvidia_libs() {
    local lib source target
    mkdir -p "${HOST_NVIDIA_LIB_DIR}"
    for lib in libcuda.so.1 libnvidia-ml.so.1; do
        source="$(host_library_path "${lib}")"
        if [ -n "${source}" ] && [ -f "${source}" ]; then
            target="${HOST_NVIDIA_LIB_DIR}/${lib}"
            if [ ! -f "${target}" ] || ! cmp -s "${source}" "${target}"; then
                cp -Lf "${source}" "${target}"
            fi
            chmod 644 "${target}" || true
        fi
    done
}

maybe_reexec_in_training_sqsh() {
    if [ "${USE_TRAIN_SQSH}" != "1" ] || [ "${INSIDE_TRAIN_SQSH:-0}" = "1" ]; then
        return
    fi
    [ -f "${TRAIN_SQSH}" ] || die "training sqsh not found: ${TRAIN_SQSH}"
    command -v apptainer >/dev/null 2>&1 || die "host apptainer not found"

    mkdir -p "${POLAR_JOB_CACHE_ROOT}/home" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
        "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
        "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" \
        "${NUMBA_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${HOST_NVIDIA_LIB_DIR}"
    chmod 700 "${POLAR_JOB_CACHE_ROOT}/home" "${XDG_RUNTIME_DIR}" || true
    prepare_host_nvidia_libs

    local bind_args=(--bind /lustre/fs1:/lustre/fs1)
    if [ -d /lustre/fsw ]; then
        bind_args+=(--bind /lustre/fsw:/lustre/fsw)
    fi
    local host_nvidia_ld=""
    local host_nvidia_preload=""
    if [ -f "${HOST_NVIDIA_LIB_DIR}/libnvidia-ml.so.1" ]; then
        bind_args+=(--bind "${HOST_NVIDIA_LIB_DIR}:/host-nvidia-libs:ro")
        host_nvidia_ld="/host-nvidia-libs:"
        if [ -f "${HOST_NVIDIA_LIB_DIR}/libcuda.so.1" ]; then
            host_nvidia_preload="/host-nvidia-libs/libcuda.so.1"
        fi
    fi

    local container_path="/opt/polr_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    local container_ld="${host_nvidia_ld}/usr/local/cuda/compat/lib.real:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    export XDG_RUNTIME_DIR
    export APPTAINERENV_PATH="${container_path}"
    export APPTAINERENV_LD_LIBRARY_PATH="${container_ld}"
    export APPTAINERENV_LD_PRELOAD="${host_nvidia_preload}"
    export APPTAINERENV_PYTHONNOUSERSITE=1
    export APPTAINERENV_XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}"
    export APPTAINERENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
    export APPTAINERENV_XDG_CONFIG_HOME="${XDG_CONFIG_HOME}"

    log "Re-entering through training sqsh: ${TRAIN_SQSH}"
    exec apptainer exec --nv --writable-tmpfs --cleanenv --no-home --pwd "${PROJECT_ROOT}" \
        "${bind_args[@]}" \
        --env "PATH=${container_path}" \
        --env "LD_LIBRARY_PATH=${container_ld}" \
        --env "LD_PRELOAD=${host_nvidia_preload}" \
        --env "PYTHONNOUSERSITE=1" \
        --env "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}" \
        --env "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
        --env "XDG_CONFIG_HOME=${XDG_CONFIG_HOME}" \
        --env "APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR}" \
        --env "APPTAINER_TMPDIR=${APPTAINER_TMPDIR}" \
        --env "APPTAINER_WORKDIR=${APPTAINER_WORKDIR}" \
        --env "SINGULARITY_CACHEDIR=${APPTAINER_CACHEDIR}" \
        --env "SINGULARITY_TMPDIR=${APPTAINER_TMPDIR}" \
        --env "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}" \
        --env "TRITON_HOME=${TRITON_HOME}" \
        --env "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}" \
        --env "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}" \
        --env "CUDA_CACHE_PATH=${CUDA_CACHE_PATH}" \
        --env "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}" \
        --env "FLASHINFER_WORKSPACE_DIR=${FLASHINFER_WORKSPACE_DIR}" \
        "${TRAIN_SQSH}" \
        env \
            INSIDE_TRAIN_SQSH=1 \
            USE_TRAIN_SQSH=1 \
            REFERENCE_ROOT="${REFERENCE_ROOT}" \
            RUN_ID="${RUN_ID}" \
            MODEL_NAME="${MODEL_NAME}" \
            PI_MODEL_NAME="${PI_MODEL_NAME}" \
            HF_CHECKPOINT="${HF_CHECKPOINT}" \
            REF_LOAD="${REF_LOAD}" \
            SAVE_DIR="${SAVE_DIR}" \
            FRESH_START="${FRESH_START}" \
            RESUME_FROM_SAVE="${RESUME_FROM_SAVE}" \
            START_ROLLOUT_ID="${START_ROLLOUT_ID}" \
            FULL_PROMPT_DATA="${FULL_PROMPT_DATA}" \
            SMOKE_NUM_ROWS="${SMOKE_NUM_ROWS}" \
            RUN_DIR="${RUN_DIR}" \
            RUN_LOG_DIR="${RUN_LOG_DIR}" \
            ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR}" \
            PROMPT_DATA="${PROMPT_DATA}" \
            TRAIN_SQSH="${TRAIN_SQSH}" \
            SLIME_DIR="${SLIME_DIR}" \
            MEGATRON_DIR="${MEGATRON_DIR}" \
            SHARED_SIF_DIR="${SHARED_SIF_DIR}" \
            APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR}" \
            AGENT_CLI_DIR="${AGENT_CLI_DIR}" \
            PREPARE_AGENT_CLI="${PREPARE_AGENT_CLI}" \
            PREPARE_MISSING_SIFS="${PREPARE_MISSING_SIFS}" \
            TOTAL_GPUS="${TOTAL_GPUS}" \
            RAY_NUM_CPUS="${RAY_NUM_CPUS}" \
            TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS}" \
            ACTOR_NUM_NODES="${ACTOR_NUM_NODES}" \
            ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE}" \
            ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS}" \
            ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
            TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE}" \
            ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
            N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT}" \
            NUM_ROLLOUT="${NUM_ROLLOUT}" \
            NUM_EPOCH="${NUM_EPOCH}" \
            NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT}" \
            DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES}" \
            SAVE_INTERVAL="${SAVE_INTERVAL}" \
            MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU}" \
            ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN}" \
            ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN}" \
            SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH}" \
            SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC}" \
            SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL}" \
            TRAIN_LR="${TRAIN_LR}" \
            CLIP_GRAD="${CLIP_GRAD}" \
            KL_LOSS_COEF="${KL_LOSS_COEF}" \
            EPS_CLIP="${EPS_CLIP}" \
            EPS_CLIP_HIGH="${EPS_CLIP_HIGH}" \
            POLAR_BUILDER_STRATEGY="${POLAR_BUILDER_STRATEGY}" \
            POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}" \
            POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL}" \
            POLAR_RUNTIME_MEMORY_MB="${POLAR_RUNTIME_MEMORY_MB}" \
            POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS}" \
            POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT}" \
            PI_API_TYPE="${PI_API_TYPE}" \
            PI_CONTEXT_WINDOW="${PI_CONTEXT_WINDOW}" \
            PI_MAX_TOKENS="${PI_MAX_TOKENS}" \
            PI_THINKING="${PI_THINKING}" \
            PI_FAIL_ON_CONTEXT_LIMIT="${PI_FAIL_ON_CONTEXT_LIMIT}" \
            PI_COMPACTION_ENABLED="${PI_COMPACTION_ENABLED}" \
            PI_RETRY_ENABLED="${PI_RETRY_ENABLED}" \
            PI_RETRY_MAX_RETRIES="${PI_RETRY_MAX_RETRIES}" \
            PI_PROVIDER_MAX_RETRIES="${PI_PROVIDER_MAX_RETRIES}" \
            ROLLOUT_PORT="${ROLLOUT_PORT}" \
            GATEWAY_PORT="${GATEWAY_PORT}" \
            SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT}" \
            SLIME_SGLANG_BASE_PORT="${SLIME_SGLANG_BASE_PORT}" \
            RAY_PORT="${RAY_PORT}" \
            RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT}" \
            RAY_USE_EXISTING_CLUSTER="${RAY_USE_EXISTING_CLUSTER}" \
            RAY_STOP_ON_EXIT="${RAY_STOP_ON_EXIT}" \
            USE_RAY_JOB_SUBMIT="${USE_RAY_JOB_SUBMIT}" \
            RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS}" \
            RAY_ADDRESS="${RAY_ADDRESS:-}" \
            RAY_TMPDIR="${RAY_TMPDIR}" \
            SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL}" \
            POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN}" \
            POLAR_APPTAINER_DIRECT_EXEC="${POLAR_APPTAINER_DIRECT_EXEC}" \
            APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR}" \
            APPTAINER_TMPDIR="${APPTAINER_TMPDIR}" \
            APPTAINER_WORKDIR="${APPTAINER_WORKDIR}" \
            TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
            TRITON_HOME="${TRITON_HOME}" \
            TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
            TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
            XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
            XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
            XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" \
            CUDA_CACHE_PATH="${CUDA_CACHE_PATH}" \
            NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
            FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR}" \
            HOST_NVIDIA_LIB_DIR="${HOST_NVIDIA_LIB_DIR}" \
            PATCH_SLIME="${PATCH_SLIME}" \
            PATCH_SGLANG="${PATCH_SGLANG}" \
            PATCH_RAY_PY312="${PATCH_RAY_PY312}" \
            CONVERT_WEIGHTS="${CONVERT_WEIGHTS}" \
            SLIME_ZERO_NONFINITE_GRADS="${SLIME_ZERO_NONFINITE_GRADS}" \
            USE_WANDB="${USE_WANDB}" \
            WANDB_MODE="${WANDB_MODE}" \
            WANDB_PROJECT="${WANDB_PROJECT}" \
            WANDB_GROUP="${WANDB_GROUP}" \
            WANDB_RUN_ID="${WANDB_RUN_ID}" \
            WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX}" \
            WANDB_TEAM="${WANDB_TEAM}" \
            WANDB_HOST="${WANDB_HOST}" \
            WANDB_API_KEY="${WANDB_API_KEY}" \
            WANDB_DIR="${WANDB_DIR}" \
            DRY_RUN="${DRY_RUN}" \
            HOME="${POLAR_JOB_CACHE_ROOT}/home" \
            PATH="${container_path}" \
            LD_LIBRARY_PATH="${container_ld}" \
            PYTHONNOUSERSITE=1 \
            /usr/bin/bash "$0"
}

resolve_gpu_split
maybe_reexec_in_training_sqsh

if [ -x /opt/polr_venv/bin/python ]; then
    PYTHON_BIN="${PYTHON_BIN:-/opt/polr_venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
fi
[ -x "${PYTHON_BIN}" ] || PYTHON_BIN="$(command -v python3 || command -v python)"
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"

mkdir -p "${RUN_DIR}" "${SAVE_DIR}" "${PROJECT_ROOT}/logs" "${WANDB_DIR}" \
    "${RUN_LOG_DIR}" "${ROLLOUT_SAVE_DIR}" "${RAY_TMPDIR}" "${APPTAINER_IMAGE_DIR}" \
    "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
    "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
    "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${HOST_NVIDIA_LIB_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}" || true
ulimit -n 1048576 >/dev/null 2>&1 || ulimit -n 65536 >/dev/null 2>&1 || true

export PYTHONNOUSERSITE=1
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export HF_CHECKPOINT MODEL_NAME PI_MODEL_NAME REF_LOAD SAVE_DIR
export POLAR_APPTAINER_BIN POLAR_APPTAINER_DIRECT_EXEC
export SLIME_ZERO_NONFINITE_GRADS SLIME_SGLANG_BASE_PORT
export APPTAINER_CACHEDIR APPTAINER_TMPDIR APPTAINER_WORKDIR
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export TRITON_CACHE_DIR TRITON_HOME TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR
export XDG_CACHE_HOME XDG_CONFIG_HOME XDG_RUNTIME_DIR CUDA_CACHE_PATH NUMBA_CACHE_DIR FLASHINFER_WORKSPACE_DIR
export WANDB_MODE WANDB_PROJECT WANDB_GROUP WANDB_RUN_ID WANDB_DIR WANDB_API_KEY
export HOST_NVIDIA_LIB_DIR
if [ -f /host-nvidia-libs/libcuda.so.1 ]; then
    export LD_PRELOAD="/host-nvidia-libs/libcuda.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

prepare_prompt_data_and_sifs() {
    "${PYTHON_BIN}" - "${FULL_PROMPT_DATA}" "${PROMPT_DATA}" "${SMOKE_NUM_ROWS}" \
        "${APPTAINER_IMAGE_DIR}" "${SHARED_SIF_DIR}" "${PREPARE_MISSING_SIFS}" <<'PY'
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

full_prompt = Path(sys.argv[1])
prompt_out = Path(sys.argv[2])
smoke_rows = int(sys.argv[3])
image_dir = Path(sys.argv[4])
shared_dir = Path(sys.argv[5])
prepare_missing = sys.argv[6] == "1"

script_dir = full_prompt.parent
sys.path.insert(0, str(script_dir))
from sample_tasks import registry_image_for_instance_id  # noqa: E402

rows = [json.loads(line) for line in full_prompt.read_text().splitlines() if line.strip()]
if smoke_rows > 0:
    rows = rows[:smoke_rows]

prompt_out.parent.mkdir(parents=True, exist_ok=True)
prompt_out.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
image_dir.mkdir(parents=True, exist_ok=True)

missing: list[tuple[str, Path]] = []
for row in rows:
    instance_id = str(row["metadata"]["instance_id"])
    target = image_dir / f"{instance_id}.sif"
    if target.exists() or target.is_symlink():
        continue
    image_ref = registry_image_for_instance_id(instance_id)
    shared_name = image_ref.split(":", 1)[0].replace("/", "_") + ".sif"
    source = shared_dir / shared_name
    if source.is_file():
        target.symlink_to(source)
        continue
    missing.append((instance_id, source))

if missing and prepare_missing:
    for instance_id, _ in missing:
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "prepare_apptainer_images.py"),
                "--instance-id",
                instance_id,
                "--image-dir",
                str(image_dir),
                "--cache-dir",
                os.environ.get("APPTAINER_CACHEDIR", "tmp/apptainer_cache"),
                "--tmp-dir",
                os.environ.get("APPTAINER_TMPDIR", "tmp/apptainer_tmp"),
                "--skip-cli",
            ],
            check=True,
        )
elif missing:
    print("Missing runtime SIF(s):", file=sys.stderr)
    for instance_id, source in missing[:20]:
        print(f"  {instance_id}: expected shared source {source}", file=sys.stderr)
    raise SystemExit(
        "Set PREPARE_MISSING_SIFS=1 to pull missing runtime images, or provide SHARED_SIF_DIR."
    )

print(f"Prompt data: {prompt_out} ({len(rows)} row(s))")
print(f"Runtime SIF dir: {image_dir}")
PY
}

prepare_agent_cli_if_needed() {
    if [ -x "${AGENT_CLI_DIR}/bin/pi" ] && [ -x "${AGENT_CLI_DIR}/bin/node" ]; then
        log "Agent CLI dir ready: ${AGENT_CLI_DIR}"
        return
    fi
    if [ "${PREPARE_AGENT_CLI}" != "1" ]; then
        die "PI CLI missing at ${AGENT_CLI_DIR}; set AGENT_CLI_DIR to a prepared opt_node or PREPARE_AGENT_CLI=1"
    fi
    log "Preparing agent CLI dir: ${AGENT_CLI_DIR}"
    "${PYTHON_BIN}" - "${SCRIPT_DIR}" "${AGENT_CLI_DIR}" <<'PY'
from pathlib import Path
import sys

script_dir = Path(sys.argv[1])
agent_cli_dir = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))

from prepare_apptainer_images import ensure_agent_cli_dir  # noqa: E402

ensure_agent_cli_dir(agent_cli_dir, force=False)
PY
}

render_runtime_configs() {
    TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
    CUSTOM_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"
    export TOPOLOGY_PATH CUSTOM_CONFIG_PATH
    "${PYTHON_BIN}" - "${SCRIPT_DIR}/topology.yaml" "${TOPOLOGY_PATH}" \
        "${SCRIPT_DIR}/polar_config.yaml" "${CUSTOM_CONFIG_PATH}" <<'PY'
from pathlib import Path
import base64
import json
import os
import shlex
import sys
import yaml

topology_in, topology_out, config_in, config_out = sys.argv[1:]

with open(topology_in, encoding="utf-8") as fh:
    topology = yaml.safe_load(fh) or {}

topology["rollout"]["port"] = int(os.environ["ROLLOUT_PORT"])
topology["rollout"]["public_url"] = f"http://127.0.0.1:{os.environ['ROLLOUT_PORT']}"
topology["rollout"]["save_dir"] = os.environ["ROLLOUT_SAVE_DIR"]
nodes = topology.get("gateway", {}).get("nodes", [])
if not nodes:
    raise SystemExit("topology has no gateway nodes")
nodes[:] = nodes[:1]
nodes[0]["id"] = "localhost-node-01"
nodes[0]["port"] = int(os.environ["GATEWAY_PORT"])
nodes[0]["public_url"] = f"http://127.0.0.1:{os.environ['GATEWAY_PORT']}"
nodes[0]["model_served"] = os.environ["MODEL_NAME"]
nodes[0].setdefault("sglang", {})["base_url"] = os.environ["SGLANG_ROUTER_BASE_URL"]

Path(topology_out).parent.mkdir(parents=True, exist_ok=True)
with open(topology_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(topology, fh, sort_keys=False)

with open(config_in, encoding="utf-8") as fh:
    config = yaml.safe_load(fh) or {}

config["polar_rollout_url"] = f"http://127.0.0.1:{os.environ['ROLLOUT_PORT']}"
config["polar_gateway_url"] = f"http://127.0.0.1:{os.environ['GATEWAY_PORT']}"
config["polar_agent_cli_dir"] = os.environ["AGENT_CLI_DIR"]
config["polar_apptainer_image_dir"] = os.environ["APPTAINER_IMAGE_DIR"]
config["polar_request_timeout"] = int(os.environ["POLAR_REQUEST_TIMEOUT"])
config["polar_max_async_level"] = int(os.environ["POLAR_MAX_ASYNC_LEVEL"])
config["polar_min_complete_accept_fraction"] = float(os.environ["POLAR_MIN_COMPLETE_ACCEPT_FRACTION"])

task = config.setdefault("polar_task_template", {})
task["timeout_seconds"] = int(os.environ["POLAR_TASK_TIMEOUT_SECONDS"])
runtime = task.setdefault("runtime", {})
memory_mb = os.environ.get("POLAR_RUNTIME_MEMORY_MB", "").strip()
if memory_mb:
    memory_value = int(memory_mb)
    if memory_value <= 0:
        raise SystemExit(f"POLAR_RUNTIME_MEMORY_MB must be positive when set: {memory_mb}")
    runtime["memory_mb"] = memory_value
else:
    runtime.pop("memory_mb", None)


def env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


pi_settings = {
    "compaction": {"enabled": env_flag("PI_COMPACTION_ENABLED")},
    "retry": {
        "enabled": env_flag("PI_RETRY_ENABLED"),
        "maxRetries": int(os.environ["PI_RETRY_MAX_RETRIES"]),
        "provider": {"maxRetries": int(os.environ["PI_PROVIDER_MAX_RETRIES"])},
    },
}
pi_settings_json = json.dumps(pi_settings, separators=(",", ":"))
pi_settings_b64 = base64.b64encode(pi_settings_json.encode("utf-8")).decode("ascii")
prepare_steps = runtime.get("prepare")
if not isinstance(prepare_steps, list):
    prepare_steps = []
    runtime["prepare"] = prepare_steps
prepare_steps.insert(
    0,
    {
        "type": "exec",
        "command": (
            'mkdir -p "$HOME/.pi/agent" && '
            f"printf '%s' {shlex.quote(pi_settings_b64)} | base64 -d > \"$HOME/.pi/agent/settings.json\""
        ),
    },
)

agent = task.setdefault("agent", {})
agent["harness"] = "pi"
agent["model_name"] = os.environ["PI_MODEL_NAME"]
settings = agent.setdefault("settings", {})
settings["api_type"] = os.environ["PI_API_TYPE"]
settings["context_window"] = int(os.environ["PI_CONTEXT_WINDOW"])
settings["max_tokens"] = int(os.environ["PI_MAX_TOKENS"])
settings.setdefault("compat", {})["maxTokensField"] = "max_tokens"
if os.environ.get("PI_THINKING"):
    settings["thinking"] = os.environ["PI_THINKING"]

task.setdefault("builder", {})["strategy"] = os.environ["POLAR_BUILDER_STRATEGY"]

Path(config_out).parent.mkdir(parents=True, exist_ok=True)
with open(config_out, "w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, sort_keys=False)

print(f"Topology: {topology_out}")
print(f"Polar config: {config_out}")
PY
}

patch_ray_py312_if_needed() {
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import ray

base = Path(ray.__file__).resolve().parent / "experimental" / "channel"
patched = []
for name in ("communicator.py", "common.py"):
    path = base / name
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if "import ray.actor\n" in text:
        continue
    if "import ray\n" not in text:
        continue
    path.write_text(text.replace("import ray\n", "import ray\nimport ray.actor\n", 1), encoding="utf-8")
    patched.append(str(path))
if patched:
    print("Patched Ray Python 3.12 channel imports:")
    for path in patched:
        print(f"  {path}")
else:
    print("Ray Python 3.12 channel import patch already applied or not needed.")
PY
}

preflight() {
    [ -x "${PYTHON_BIN}" ] || die "python not executable: ${PYTHON_BIN}"
    command -v ray >/dev/null 2>&1 || die "ray command not found"
    command -v "${POLAR_APPTAINER_BIN}" >/dev/null 2>&1 || [ -x "${POLAR_APPTAINER_BIN}" ] || die "apptainer not found: ${POLAR_APPTAINER_BIN}"
    if is_path_like "${HF_CHECKPOINT}"; then
        [ -d "${HF_CHECKPOINT}" ] || die "HF checkpoint not found: ${HF_CHECKPOINT}"
    fi
    [ -f "${FULL_PROMPT_DATA}" ] || die "training data not found: ${FULL_PROMPT_DATA}"
    [ -f "${SLIME_DIR}/train_async.py" ] || die "Slime checkout missing at ${SLIME_DIR}"
    [ -d "${MEGATRON_DIR}/megatron" ] || die "Megatron-LM checkout missing at ${MEGATRON_DIR}"

    if [ "${PATCH_SLIME}" = "1" ]; then
        bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"
    fi
    if [ "${PATCH_SGLANG}" = "1" ]; then
        bash "${PROJECT_ROOT}/scripts/patch/patch_sglang.sh"
    fi
    if [ "${PATCH_RAY_PY312}" = "1" ]; then
        patch_ray_py312_if_needed
    fi

    if ! checkpoint_marker_exists "${REF_LOAD}"; then
        if [ "${CONVERT_WEIGHTS}" = "1" ]; then
            log "Converting HF weights to Megatron torch_dist: ${REF_LOAD}"
            TORCH_DIST_DIR="${REF_LOAD}" HF_CHECKPOINT="${HF_CHECKPOINT}" \
                SLIME_DIR="${SLIME_DIR}" MEGATRON_DIR="${MEGATRON_DIR}" \
                bash "${SCRIPT_DIR}/convert_weights.sh"
        else
            die "Megatron torch_dist checkpoint missing at ${REF_LOAD}; set REF_LOAD or CONVERT_WEIGHTS=1"
        fi
    fi

    "${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["ray", "torch", "sglang", "polar", "yaml", "httpx"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("Missing Python dependency/dependencies:\n" + "\n".join(missing))
PY

    if [ "${USE_WANDB}" = "1" ] && [ "${WANDB_MODE}" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
        die "WANDB_API_KEY is required for online W&B"
    fi
}

runtime_ld_library_path() {
    local py_site
    py_site="$("${PYTHON_BIN}" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
    printf '%s' "${py_site}/torch/lib:${py_site}/nvidia/cuda_runtime/lib:${py_site}/nvidia/cuda_nvrtc/lib:${py_site}/nvidia/nvjitlink/lib:${py_site}/nvidia/cublas/lib:${py_site}/nvidia/cudnn/lib:${py_site}/nvidia/nccl/lib:${py_site}/nvidia/cusparse/lib:${py_site}/nvidia/cusolver/lib:${py_site}/nvidia/cufft/lib:${py_site}/nvidia/curand/lib:${LD_LIBRARY_PATH:-}"
}

SERVICE_PIDS=()

start_services_and_train() {
    SERVICE_PIDS=()
    cleanup() {
        set +e
        log "Cleaning up Polar/Ray processes"
        local pid
        for pid in "${SERVICE_PIDS[@]}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in "${SERVICE_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        if [ "${RAY_STOP_ON_EXIT}" = "1" ]; then
            ray stop --force >/dev/null 2>&1 || true
        fi
        wait 2>/dev/null || true
    }
    trap cleanup EXIT

    log "Starting Polar rollout on :${ROLLOUT_PORT}"
    polar serve_rollout -c "${TOPOLOGY_PATH}" >"${RUN_LOG_DIR}/rollout.log" 2>&1 &
    SERVICE_PIDS+=("$!")
    sleep 2

    log "Starting Polar gateway on :${GATEWAY_PORT}"
    polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 >"${RUN_LOG_DIR}/gateway.log" 2>&1 &
    SERVICE_PIDS+=("$!")
    sleep 2

    "${PYTHON_BIN}" - "${ROLLOUT_PORT}" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                print("rollout healthy")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit(f"rollout health check failed: {url}")
PY

    if [ "${RAY_USE_EXISTING_CLUSTER}" = "1" ]; then
        log "Using existing Ray cluster at ${RAY_ADDRESS:-auto}; job server ${RAY_JOB_ADDRESS}"
    else
        log "Starting Ray with ${TOTAL_GPUS} GPUs"
        ray stop --force >"${RUN_LOG_DIR}/ray-stop-before-start.log" 2>&1 || true
        if ! ray start --head --node-ip-address 127.0.0.1 \
            --port "${RAY_PORT}" \
            --num-cpus "${RAY_NUM_CPUS}" \
            --num-gpus "${TOTAL_GPUS}" \
            --dashboard-port "${RAY_DASHBOARD_PORT}" \
            --temp-dir "${RAY_TMPDIR}" \
            --disable-usage-stats >"${RUN_LOG_DIR}/ray-start.log" 2>&1; then
            tail -100 "${RUN_LOG_DIR}/ray-start.log" >&2 || true
            return 1
        fi
        RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:${RAY_PORT}}"
        export RAY_ADDRESS
    fi

    local runtime_ld
    runtime_ld="$(runtime_ld_library_path)"
    local runtime_env_path="${RUN_DIR}/ray_runtime_env.yaml"
    "${PYTHON_BIN}" - "${runtime_env_path}" <<PY
import os
import sys
import yaml

env = {
    "PYTHONPATH": "${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src",
    "PATH": "${PYTHON_BIN_DIR}:" + os.environ.get("PATH", ""),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SLIME_ZERO_NONFINITE_GRADS": "${SLIME_ZERO_NONFINITE_GRADS}",
    "SLIME_SGLANG_BASE_PORT": "${SLIME_SGLANG_BASE_PORT}",
    "WANDB_DIR": "${WANDB_DIR}",
    "WANDB_MODE": "${WANDB_MODE}",
    "HOME": "${HOME}",
    "APPTAINER_CACHEDIR": "${APPTAINER_CACHEDIR}",
    "APPTAINER_TMPDIR": "${APPTAINER_TMPDIR}",
    "APPTAINER_WORKDIR": "${APPTAINER_WORKDIR}",
    "SINGULARITY_CACHEDIR": "${APPTAINER_CACHEDIR}",
    "SINGULARITY_TMPDIR": "${APPTAINER_TMPDIR}",
    "POLAR_APPTAINER_BIN": "${POLAR_APPTAINER_BIN}",
    "POLAR_APPTAINER_DIRECT_EXEC": "${POLAR_APPTAINER_DIRECT_EXEC}",
    "TRITON_CACHE_DIR": "${TRITON_CACHE_DIR}",
    "TRITON_HOME": "${TRITON_HOME}",
    "TORCHINDUCTOR_CACHE_DIR": "${TORCHINDUCTOR_CACHE_DIR}",
    "TORCH_EXTENSIONS_DIR": "${TORCH_EXTENSIONS_DIR}",
    "XDG_CACHE_HOME": "${XDG_CACHE_HOME}",
    "XDG_CONFIG_HOME": "${XDG_CONFIG_HOME}",
    "XDG_RUNTIME_DIR": "${XDG_RUNTIME_DIR}",
    "CUDA_CACHE_PATH": "${CUDA_CACHE_PATH}",
    "NUMBA_CACHE_DIR": "${NUMBA_CACHE_DIR}",
    "FLASHINFER_WORKSPACE_DIR": "${FLASHINFER_WORKSPACE_DIR}",
    "LD_LIBRARY_PATH": """${runtime_ld}""",
    "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
    "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:2048,expandable_segments:True",
}
if "${USE_TRAIN_SQSH}" == "1":
    env["VIRTUAL_ENV"] = "/opt/polr_venv"
else:
    env["VIRTUAL_ENV"] = "${PROJECT_ROOT}/.venv"
for key in ("WANDB_API_KEY", "WANDB_BASE_URL", "WANDB_ENTITY", "WANDB_ORG", "WANDB_PROJECT", "WANDB_RUN_ID"):
    value = os.environ.get(key)
    if value:
        env[key] = value
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    yaml.safe_dump({"env_vars": env}, fh, sort_keys=True)
os.chmod(sys.argv[1], 0o600)
PY
    log "Ray runtime env: ${runtime_env_path}"

    if [ "${FRESH_START}" = "1" ] && [ "${RESUME_FROM_SAVE}" = "1" ]; then
        die "FRESH_START=1 conflicts with RESUME_FROM_SAVE=1"
    fi
    local load_dir="${REF_LOAD}"
    if [ "${RESUME_FROM_SAVE}" = "1" ]; then
        checkpoint_marker_exists "${SAVE_DIR}" || die "RESUME_FROM_SAVE=1 but SAVE_DIR has no checkpoint: ${SAVE_DIR}"
        load_dir="${SAVE_DIR}"
    elif [ "${FRESH_START}" = "1" ] && checkpoint_marker_exists "${SAVE_DIR}"; then
        die "FRESH_START=1 refuses existing checkpoint in SAVE_DIR: ${SAVE_DIR}. Set RESUME_FROM_SAVE=1 or choose a new RUN_ID."
    fi
    log "Training load dir: ${load_dir}"

    local wandb_args=()
    if [ "${USE_WANDB}" = "1" ]; then
        wandb_args=(
            --use-wandb
            --wandb-mode "${WANDB_MODE}"
            --wandb-dir "${WANDB_DIR}"
            --wandb-project "${WANDB_PROJECT}"
            --wandb-group "${WANDB_GROUP}"
        )
        [ -z "${WANDB_TEAM}" ] || wandb_args+=(--wandb-team "${WANDB_TEAM}")
        [ -z "${WANDB_HOST}" ] || wandb_args+=(--wandb-host "${WANDB_HOST}")
        [ -z "${WANDB_RUN_ID}" ] || wandb_args+=(--wandb-run-id "${WANDB_RUN_ID}")
        [ "${WANDB_RANDOM_SUFFIX}" = "1" ] || wandb_args+=(--disable-wandb-random-suffix)
    fi

    local training_length_args=()
    if [ -n "${NUM_ROLLOUT:-}" ]; then
        training_length_args=(--num-rollout "${NUM_ROLLOUT}")
    else
        training_length_args=(--num-epoch "${NUM_EPOCH}")
    fi

    local start_rollout_args=()
    if [ -n "${START_ROLLOUT_ID:-}" ]; then
        start_rollout_args=(--start-rollout-id "${START_ROLLOUT_ID}")
    fi

    local model_args=(
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
        --use-gated-attention
        --normalization RMSNorm
        --apply-layernorm-1p
        --position-embedding-type rope
        --norm-epsilon 1e-6
        --rotary-percent 0.25
        --swiglu
        --vocab-size 248320
        --rotary-base 10000000
        --attention-output-gate
    )

    local train_args=(
        "${SLIME_DIR}/train_async.py"
        --actor-num-nodes "${ACTOR_NUM_NODES}" \
        --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
        --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
        --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
        "${model_args[@]}" \
        --hf-checkpoint "${HF_CHECKPOINT}" \
        --tokenizer-model "${HF_CHECKPOINT}" \
        --tokenizer-type HuggingFaceTokenizer \
        --no-use-tokenizer-model-from-checkpoint-args \
        --ref-load "${REF_LOAD}" \
        --load "${load_dir}" \
        --save "${SAVE_DIR}" \
        --dist-ckpt-strictness "${DIST_CKPT_STRICTNESS:-log_all}" \
        --save-interval "${SAVE_INTERVAL}" \
        --update-weights-interval 1 \
        --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
        --custom-rm-path slime_bridge.reward.reward_func \
        --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
        --custom-config-path "${CUSTOM_CONFIG_PATH}" \
        --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
        --prompt-data "${PROMPT_DATA}" \
        --input-key prompt \
        --label-key label \
        --metadata-key metadata \
        --rollout-shuffle \
        --reward-key score \
        "${training_length_args[@]}" \
        "${start_rollout_args[@]}" \
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" \
        --dynamic-history \
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
        --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}" \
        --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
        --sequence-parallel \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --log-probs-chunk-size 256 \
        --advantage-estimator grpo \
        --normalize-advantages \
        --use-tis \
        --use-kl-loss \
        --kl-loss-coef "${KL_LOSS_COEF}" \
        --kl-loss-type low_var_kl \
        --entropy-coef 0.0 \
        --eps-clip "${EPS_CLIP}" \
        --eps-clip-high "${EPS_CLIP_HIGH}" \
        --optimizer adam \
        --lr "${TRAIN_LR}" \
        --lr-decay-style constant \
        --weight-decay 0.1 \
        --clip-grad "${CLIP_GRAD}" \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend auto \
        --no-gradient-accumulation-fusion \
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
        --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
        --sglang-served-model-name "${MODEL_NAME}" \
        --sglang-tool-call-parser qwen3_coder \
        --sglang-log-level "${SGLANG_LOG_LEVEL}" \
        --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
        --sglang-router-port "${SGLANG_ROUTER_PORT}" \
        "${wandb_args[@]}"
    )

    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export LD_LIBRARY_PATH="${runtime_ld}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
    if [ "${USE_TRAIN_SQSH}" = "1" ]; then
        export VIRTUAL_ENV="/opt/polr_venv"
    else
        export VIRTUAL_ENV="${PROJECT_ROOT}/.venv"
    fi

    log "Launching Slime train_async.py"
    log "Run logs: ${RUN_LOG_DIR}"
    set +e
    if [ "${USE_RAY_JOB_SUBMIT}" = "1" ]; then
        ray job submit --address="${RAY_JOB_ADDRESS}" \
            --runtime-env="${runtime_env_path}" \
            -- "${PYTHON_BIN}" "${train_args[@]}" 2>&1 | tee "${RUN_LOG_DIR}/ray-job-submit.log"
    else
        "${PYTHON_BIN}" - "${train_args[@]}" <<'PY' 2>&1 | tee "${RUN_LOG_DIR}/train-direct.log"
import os
import runpy
import sys

import ray

script = sys.argv[1]
script_args = sys.argv[2:]
address = os.environ.get("RAY_ADDRESS") or "auto"
ray.init(address=address, ignore_reinit_error=True)
sys.argv = [script, *script_args]
runpy.run_path(script, run_name="__main__")
PY
    fi
    local submit_rc="${PIPESTATUS[0]}"
    set -e
    return "${submit_rc}"
}

export ROLLOUT_PORT GATEWAY_PORT SGLANG_ROUTER_BASE_URL MODEL_NAME PI_MODEL_NAME
export AGENT_CLI_DIR APPTAINER_IMAGE_DIR ROLLOUT_SAVE_DIR
export POLAR_BUILDER_STRATEGY POLAR_MIN_COMPLETE_ACCEPT_FRACTION POLAR_MAX_ASYNC_LEVEL
export POLAR_RUNTIME_MEMORY_MB POLAR_TASK_TIMEOUT_SECONDS POLAR_REQUEST_TIMEOUT
export PI_API_TYPE PI_CONTEXT_WINDOW PI_MAX_TOKENS PI_THINKING
export PI_FAIL_ON_CONTEXT_LIMIT PI_COMPACTION_ENABLED PI_RETRY_ENABLED PI_RETRY_MAX_RETRIES PI_PROVIDER_MAX_RETRIES
export SLIME_SGLANG_BASE_PORT SGLANG_LOG_LEVEL
export RAY_PORT RAY_NUM_CPUS USE_RAY_JOB_SUBMIT

log "Training set source: ${FULL_PROMPT_DATA}"
log "Smoke rows: ${SMOKE_NUM_ROWS} (set SMOKE_NUM_ROWS=0 for full train set)"
log "Agent harness: pi"
log "Builder: ${POLAR_BUILDER_STRATEGY}"
log "Model served: ${MODEL_NAME}"
log "PI model name: ${PI_MODEL_NAME}"
log "PI fail-fast: context_limit=${PI_FAIL_ON_CONTEXT_LIMIT}, compaction=${PI_COMPACTION_ENABLED}, retry=${PI_RETRY_ENABLED}, provider_retries=${PI_PROVIDER_MAX_RETRIES}"
log "HF checkpoint: ${HF_CHECKPOINT}"
log "Megatron load: ${REF_LOAD}"
log "Training sqsh: ${TRAIN_SQSH}"
log "Runtime SIF source dir: ${SHARED_SIF_DIR}"
log "Runtime SIF local dir: ${APPTAINER_IMAGE_DIR}"
log "Agent CLI dir: ${AGENT_CLI_DIR}"
log "GPU split: total=${TOTAL_GPUS}, train=${TRAIN_NUM_GPUS}, rollout=${ROLLOUT_NUM_GPUS}, tp=${TENSOR_MODEL_PARALLEL_SIZE}"
log "Ray CPU resources: ${RAY_NUM_CPUS}"
log "Batch: rollout=${ROLLOUT_BATCH_SIZE}, samples/prompt=${N_SAMPLES_PER_PROMPT}, steps/rollout=${NUM_STEPS_PER_ROLLOUT}"
log "Distributed timeout: ${DISTRIBUTED_TIMEOUT_MINUTES} minutes"
log "Run log dir: ${RUN_LOG_DIR}"
log "Rollout trace dir: ${ROLLOUT_SAVE_DIR}"
log "W&B: mode=${WANDB_MODE}, project=${WANDB_PROJECT}, group=${WANDB_GROUP}, run_id=${WANDB_RUN_ID}"
log "Ray driver mode: $([ "${USE_RAY_JOB_SUBMIT}" = "1" ] && printf 'job-submit' || printf 'direct')"

preflight
prepare_prompt_data_and_sifs
prepare_agent_cli_if_needed
render_runtime_configs

if [ "${DRY_RUN}" = "1" ]; then
    log "DRY_RUN=1; preflight and config rendering complete."
    exit 0
fi

start_services_and_train
