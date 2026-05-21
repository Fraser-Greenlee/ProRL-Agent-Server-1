#!/usr/bin/env bash
#SBATCH --job-name=polar-swegym-pi-qwen35
#SBATCH --account=nvr_lpr_agentic
#SBATCH --partition=batch_block1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar_3/ProRL-Agent-Server/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar_3/ProRL-Agent-Server/logs/slurm/%x-%j.err
#SBATCH --export=ALL

# Multi-node Slurm launcher for PI + Qwen3.5-4B SWE-Gym GRPO.
# The Slurm/Pyxis layer starts one Ray cluster across the allocation, then the
# head rank delegates PI-specific setup and training arguments to
# run_pi_interactive_apptainer.sh.
set -euo pipefail

# =============================================================================
# User settings
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar_3/ProRL-Agent-Server}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/polar/ProRL-Agent-Server}"
SCRIPT_DIR="${PROJECT_ROOT}/examples/swegym_slime_grpo"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
PI_MODEL_NAME="${PI_MODEL_NAME:-openai/${MODEL_NAME}}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/model/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${REFERENCE_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
FULL_PROMPT_DATA="${FULL_PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}"

# Keep account/partition in the SBATCH header unchanged. This image/mount style
# follows the cluster's existing Slurm/Pyxis launchers.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_SQSH="${TRAIN_SQSH:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/docker/polr_swegym_slime_grpo_train_codex_yazi_v2.sqsh}"
TRAIN_CONTAINER_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
SLIME_DIR="${SLIME_DIR:-${REFERENCE_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${REFERENCE_ROOT}/Megatron-LM}"
SHARED_SIF_DIR="${SHARED_SIF_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/singularity_images_v3}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${REFERENCE_ROOT}/tmp/swegym_agent_cli/opt_node}"
HOST_NVIDIA_LIB_DIR="${HOST_NVIDIA_LIB_DIR:-${PROJECT_ROOT}/tmp/host-nvidia-libs}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/shaokunz/HG_Cache}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_ROOT}/hub}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_ROOT}/transformers}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_CACHE_ROOT}/modules}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_CACHE_ROOT}/sentence_transformers}"

# Full run defaults. Direct `sbatch submit_pi_slurm_apptainer.sh` launches the
# production one-epoch run. For a short launch test, edit SMOKE_NUM_ROWS below
# before submitting.
SMOKE_NUM_ROWS="${SMOKE_NUM_ROWS:-0}"
NUM_EPOCH="${NUM_EPOCH:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-180}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-60000}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16000}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-32000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"

POLAR_BUILDER_STRATEGY="${POLAR_BUILDER_STRATEGY:-prefix_merging}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.6}"
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-2}"
POLAR_RUNTIME_MEMORY_MB="${POLAR_RUNTIME_MEMORY_MB:-262144}"
POLAR_TASK_TIMEOUT_SECONDS="${POLAR_TASK_TIMEOUT_SECONDS:-1200}"
POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-1200}"
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

# Stability knobs for the PI run. Keep the batch, samples, timeout, context,
# and max-token settings fixed; these three values are the intended stability
# change after the observed late-run reward collapse.
TRAIN_LR="${TRAIN_LR:-5e-7}"
CLIP_GRAD="${CLIP_GRAD:-0.5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.005}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
SLIME_ZERO_NONFINITE_GRADS="${SLIME_ZERO_NONFINITE_GRADS:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-${RAY_MEMORY_USAGE_THRESHOLD}}"
RAY_NUM_CPUS_PER_NODE="${RAY_NUM_CPUS_PER_NODE:-32}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"
GPU_WANDB_MONITOR="${GPU_WANDB_MONITOR:-1}"
GPU_WANDB_MONITOR_INTERVAL="${GPU_WANDB_MONITOR_INTERVAL:-${GPU_MONITOR_INTERVAL}}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-28265}"
ROLLOUT_PORT="${ROLLOUT_PORT:-18080}"
GATEWAY_PORT="${GATEWAY_PORT:-18100}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-26000}"
SLIME_SGLANG_BASE_PORT="${SLIME_SGLANG_BASE_PORT:-34000}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-NVR-LRP}"
WANDB_PROJECT="${WANDB_PROJECT:-polar-swegym-pi-qwen35-4b}"
WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-0}"
WANDB_TEAM="${WANDB_TEAM:-${WANDB_ENTITY}}"
WANDB_HOST="${WANDB_HOST:-}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"

PATCH_SLIME="${PATCH_SLIME:-1}"
PATCH_SGLANG="${PATCH_SGLANG:-1}"
PATCH_RAY_PY312="${PATCH_RAY_PY312:-1}"
FRESH_START="${FRESH_START:-auto}"
RESUME_FROM_SAVE="${RESUME_FROM_SAVE:-auto}"
AUTO_RESUME_FROM_SAVE="${AUTO_RESUME_FROM_SAVE:-1}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"
DRY_RUN="${DRY_RUN:-0}"

# =============================================================================
# Implementation
# =============================================================================
LOG_DIR="${PROJECT_ROOT}/logs/slurm"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_positive_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        0) return 1 ;;
        *) return 0 ;;
    esac
}

[ -d "${PROJECT_ROOT}" ] || die "PROJECT_ROOT does not exist: ${PROJECT_ROOT}"
[ -f "${SCRIPT_DIR}/run_pi_interactive_apptainer.sh" ] || die "missing PI launcher: ${SCRIPT_DIR}/run_pi_interactive_apptainer.sh"
[ -f "${TRAIN_SQSH}" ] || die "TRAIN_SQSH does not exist: ${TRAIN_SQSH}"
[ -d "${HF_CHECKPOINT}" ] || die "HF_CHECKPOINT does not exist: ${HF_CHECKPOINT}"
[ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ] || die "Megatron checkpoint missing latest marker: ${REF_LOAD}"
[ -f "${FULL_PROMPT_DATA}" ] || die "training data missing: ${FULL_PROMPT_DATA}"
[ -f "${SLIME_DIR}/train_async.py" ] || die "Slime checkout missing: ${SLIME_DIR}"
[ -d "${MEGATRON_DIR}/megatron" ] || die "Megatron-LM checkout missing: ${MEGATRON_DIR}"
[ -x "${AGENT_CLI_DIR}/bin/pi" ] || die "PI CLI missing: ${AGENT_CLI_DIR}/bin/pi"

NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"
[ -n "${NUM_NODES}" ] || die "This script must run inside an sbatch allocation"
case "${NUM_NODES}" in
    1|2|3|4|5|6|7|8) ;;
    *) die "NUM_NODES must be between 1 and 8 for this launcher; got ${NUM_NODES}" ;;
esac

TOTAL_GPUS="$((NUM_NODES * GPUS_PER_NODE))"
if [ -z "${ACTOR_NUM_NODES}" ]; then
    if [ "${NUM_NODES}" -ge 8 ]; then
        ACTOR_NUM_NODES=4
    else
        ACTOR_NUM_NODES=1
    fi
fi
[ "${ACTOR_NUM_NODES}" -le "${NUM_NODES}" ] || die "ACTOR_NUM_NODES exceeds allocation nodes"

if [ -z "${TRAIN_NUM_GPUS}" ] && [ -z "${ACTOR_NUM_GPUS_PER_NODE}" ]; then
    TRAIN_NUM_GPUS="$((ACTOR_NUM_NODES * GPUS_PER_NODE))"
fi
if [ -z "${ACTOR_NUM_GPUS_PER_NODE}" ]; then
    [ "$((TRAIN_NUM_GPUS % ACTOR_NUM_NODES))" -eq 0 ] || \
        die "TRAIN_NUM_GPUS (${TRAIN_NUM_GPUS}) must be divisible by ACTOR_NUM_NODES (${ACTOR_NUM_NODES})"
    ACTOR_NUM_GPUS_PER_NODE="$((TRAIN_NUM_GPUS / ACTOR_NUM_NODES))"
else
    TRAIN_NUM_GPUS="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"
fi
[ "${ACTOR_NUM_GPUS_PER_NODE}" -le "${GPUS_PER_NODE}" ] || die "ACTOR_NUM_GPUS_PER_NODE exceeds GPUS_PER_NODE"

if [ -z "${ROLLOUT_NUM_GPUS}" ]; then
    ROLLOUT_NUM_GPUS="$((TOTAL_GPUS - TRAIN_NUM_GPUS))"
fi
[ "${TRAIN_NUM_GPUS}" -gt 0 ] || die "TRAIN_NUM_GPUS must be positive"
[ "${ROLLOUT_NUM_GPUS}" -gt 0 ] || die "ROLLOUT_NUM_GPUS must be positive"
[ "$((TRAIN_NUM_GPUS + ROLLOUT_NUM_GPUS))" -le "${TOTAL_GPUS}" ] || \
    die "train + rollout GPUs exceed allocation: train=${TRAIN_NUM_GPUS}, rollout=${ROLLOUT_NUM_GPUS}, total=${TOTAL_GPUS}"
[ "$((TRAIN_NUM_GPUS % TENSOR_MODEL_PARALLEL_SIZE))" -eq 0 ] || \
    die "TRAIN_NUM_GPUS (${TRAIN_NUM_GPUS}) must be divisible by TENSOR_MODEL_PARALLEL_SIZE (${TENSOR_MODEL_PARALLEL_SIZE})"

DATA_PARALLEL_SIZE="$((TRAIN_NUM_GPUS / TENSOR_MODEL_PARALLEL_SIZE))"
if [ "${SMOKE_NUM_ROWS}" -gt 0 ]; then
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-${DATA_PARALLEL_SIZE}}"
    NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
else
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
fi

ROLLOUT_SAMPLES_PER_STEP="$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
[ "$((ROLLOUT_SAMPLES_PER_STEP % NUM_STEPS_PER_ROLLOUT))" -eq 0 ] || \
    die "rollout_batch_size * n_samples_per_prompt must be divisible by num_steps_per_rollout"
GLOBAL_BATCH_SIZE="$((ROLLOUT_SAMPLES_PER_STEP / NUM_STEPS_PER_ROLLOUT))"
[ "$((GLOBAL_BATCH_SIZE % DATA_PARALLEL_SIZE))" -eq 0 ] || \
    die "global batch size (${GLOBAL_BATCH_SIZE}) must be divisible by data parallel size (${DATA_PARALLEL_SIZE})"

RUN_KIND="${RUN_KIND:-full293}"
if [ "${SMOKE_NUM_ROWS}" -gt 0 ]; then
    RUN_KIND="debug${SMOKE_NUM_ROWS}"
fi
EXPERIMENT_TAG="${EXPERIMENT_TAG:-${RUN_KIND}_1ep_${NUM_NODES}n${TOTAL_GPUS}g_train${TRAIN_NUM_GPUS}_rollout${ROLLOUT_NUM_GPUS}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_rb${ROLLOUT_BATCH_SIZE}_n${N_SAMPLES_PER_PROMPT}}"
STABILITY_TAG="${STABILITY_TAG:-lr5e7_kl5e3_clip05}"
RUN_SERIES_ID="${RUN_SERIES_ID:-swegym_pi_qwen35_4b_${EXPERIMENT_TAG}_${STABILITY_TAG}_t1200_failctxb64}"
RUN_ID="${RUN_ID:-${RUN_SERIES_ID}}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/${RUN_ID}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${RUN_DIR}/logs}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-${RUN_DIR}/rollout_results}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/${RUN_ID}}"
WANDB_GROUP="${WANDB_GROUP:-swegym-pi-qwen35-4b-${RUN_KIND}-${NUM_NODES}n${TOTAL_GPUS}g}"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_ID}}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/logs/wandb}"
if [ -n "${NUM_ROLLOUT}" ]; then
    is_positive_int "${NUM_ROLLOUT}" || die "NUM_ROLLOUT must be a positive integer when set: ${NUM_ROLLOUT}"
    EXPECTED_ROLLOUTS="${NUM_ROLLOUT}"
else
    is_positive_int "${NUM_EPOCH}" || die "NUM_EPOCH must be a positive integer: ${NUM_EPOCH}"
    if [ "${SMOKE_NUM_ROWS}" -gt 0 ]; then
        PROMPT_ROW_COUNT="${SMOKE_NUM_ROWS}"
    else
        PROMPT_ROW_COUNT="$(wc -l < "${FULL_PROMPT_DATA}" | tr -d '[:space:]')"
        is_positive_int "${PROMPT_ROW_COUNT}" || die "could not count prompt rows in ${FULL_PROMPT_DATA}"
    fi
    EXPECTED_ROLLOUTS="$(( ((PROMPT_ROW_COUNT + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE) * NUM_EPOCH ))"
fi

if [ "${AUTO_RESUME_FROM_SAVE}" = "1" ]; then
    if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
        FRESH_START=0
        RESUME_FROM_SAVE=1
        if [ -z "${START_ROLLOUT_ID}" ]; then
            latest_checkpoint_iter="$(tr -d '\n[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
            case "${latest_checkpoint_iter}" in
                ''|*[!0-9]*) die "invalid latest checkpoint marker for START_ROLLOUT_ID: ${latest_checkpoint_iter}" ;;
            esac
            START_ROLLOUT_ID="$((latest_checkpoint_iter + 1))"
        fi
    else
        FRESH_START=1
        RESUME_FROM_SAVE=0
    fi
else
    if [ "${RESUME_FROM_SAVE}" = "auto" ]; then
        RESUME_FROM_SAVE=0
    fi
    if [ "${FRESH_START}" = "auto" ]; then
        if [ "${RESUME_FROM_SAVE}" = "1" ]; then
            FRESH_START=0
        else
            FRESH_START=1
        fi
    fi
fi
[ "${FRESH_START}" = "1" ] && [ "${RESUME_FROM_SAVE}" = "1" ] && \
    die "FRESH_START=1 conflicts with RESUME_FROM_SAVE=1"
if [ "${FRESH_START}" = "1" ] && [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    die "FRESH_START=1 refuses existing checkpoint in SAVE_DIR: ${SAVE_DIR}"
fi
if [ "${RESUME_FROM_SAVE}" = "1" ] && [ -n "${START_ROLLOUT_ID}" ] && \
    [ "${START_ROLLOUT_ID}" -ge "${EXPECTED_ROLLOUTS}" ]; then
    cat <<EOF
=============================================
SWE-Gym PI GRPO Slurm launch
  Job ID:        ${SLURM_JOB_ID}
  Run ID:        ${RUN_ID}
  Save dir:      ${SAVE_DIR}
  Auto resume:   ${AUTO_RESUME_FROM_SAVE}
  Latest ckpt:   ${latest_checkpoint_iter}
  Start rollout: ${START_ROLLOUT_ID}
  Expected:      ${EXPECTED_ROLLOUTS} rollout(s)
  Status:        run already complete; exiting without starting Ray
=============================================
EOF
    exit 0
fi

if [ "${RESUME_FROM_SAVE}" = "1" ] && [ -n "${START_ROLLOUT_ID}" ]; then
    # A failed async checkpoint can leave iter_${START_ROLLOUT_ID} behind while
    # latest_checkpointed_iteration.txt still points to the previous iteration.
    # Move it aside so resume can rewrite the next checkpoint cleanly.
    next_iter_dir=$(printf "%s/iter_%07d" "${SAVE_DIR}" "${START_ROLLOUT_ID}")
    if [ -d "${next_iter_dir}" ]; then
        quarantine_dir="${next_iter_dir}.incomplete_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Quarantining incomplete resume checkpoint ${next_iter_dir} -> ${quarantine_dir}"
        mv "${next_iter_dir}" "${quarantine_dir}"
    fi
    next_rollout_state=$(printf "%s/rollout/global_dataset_state_dict_%d.pt" "${SAVE_DIR}" "${START_ROLLOUT_ID}")
    if [ -f "${next_rollout_state}" ]; then
        quarantine_state="${next_rollout_state}.incomplete_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Quarantining incomplete rollout state ${next_rollout_state} -> ${quarantine_state}"
        mv "${next_rollout_state}" "${quarantine_state}"
    fi
fi

mkdir -p "${LOG_DIR}" "${RUN_LOG_DIR}" "${RUN_DIR}" "${SAVE_DIR}" "${WANDB_DIR}" \
    "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}"
STOP_FILE="${RUN_DIR}/slurm_stop_workers"
WORKER_SCRIPT="${RUN_DIR}/slurm_ray_worker_pi.sh"
rm -f "${STOP_FILE}"

mapfile -t SLURM_NODES < <(scontrol show hostnames "${SLURM_NODELIST}")
HEAD_NODE="${SLURM_NODES[0]}"
RAY_HEAD_IP="$(srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${HEAD_NODE}" bash -lc 'hostname -I | cut -d" " -f1')"
RAY_ADDRESS="${RAY_HEAD_IP}:${RAY_PORT}"
RAY_JOB_ADDRESS="http://${RAY_HEAD_IP}:${RAY_DASHBOARD_PORT}"
RAY_NUM_CPUS="$((NUM_NODES * RAY_NUM_CPUS_PER_NODE))"
UNUSED_GPUS="$((TOTAL_GPUS - TRAIN_NUM_GPUS - ROLLOUT_NUM_GPUS))"

cat >"${WORKER_SCRIPT}" <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

rank="${SLURM_PROCID}"
node="$(hostname)"
node_ip="$(hostname -I | awk '{print $1}')"

export PATH="/opt/polr_venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME}"

PY_SITE="/opt/polr_venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="${PY_SITE}/torch/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${PY_SITE}/nvidia/cuda_nvrtc/lib:${PY_SITE}/nvidia/nvjitlink/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/nccl/lib:${PY_SITE}/nvidia/cusparse/lib:${PY_SITE}/nvidia/cusolver/lib:${PY_SITE}/nvidia/cufft/lib:${PY_SITE}/nvidia/curand/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-${RAY_MEMORY_USAGE_THRESHOLD}}"
export RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD}"

NODE_CACHE_ROOT="/tmp/polar-swegym-pi-${SLURM_JOB_ID}-${rank}"
export HOME="${NODE_CACHE_ROOT}/home"
export APPTAINER_CACHEDIR="${NODE_CACHE_ROOT}/apptainer-cache"
export APPTAINER_TMPDIR="${NODE_CACHE_ROOT}/apptainer-tmp"
export APPTAINER_WORKDIR="${NODE_CACHE_ROOT}/apptainer-work"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
export TRITON_CACHE_DIR="${NODE_CACHE_ROOT}/triton-cache"
export TRITON_HOME="${NODE_CACHE_ROOT}/triton-home"
export TORCHINDUCTOR_CACHE_DIR="${NODE_CACHE_ROOT}/torchinductor"
export TORCH_EXTENSIONS_DIR="${NODE_CACHE_ROOT}/torch-extensions"
export XDG_CACHE_HOME="${NODE_CACHE_ROOT}/xdg-cache"
export XDG_CONFIG_HOME="${NODE_CACHE_ROOT}/xdg-config"
export XDG_RUNTIME_DIR="${NODE_CACHE_ROOT}/xdg-runtime"
export CUDA_CACHE_PATH="${NODE_CACHE_ROOT}/cuda-cache"
export NUMBA_CACHE_DIR="${NODE_CACHE_ROOT}/numba"
export FLASHINFER_WORKSPACE_DIR="${NODE_CACHE_ROOT}/flashinfer-cache"
export RAY_TMPDIR="/tmp/polar-ray-pi-${SLURM_JOB_ID}-${rank}"

mkdir -p "${HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_WORKDIR}" \
    "${TRITON_CACHE_DIR}" "${TRITON_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
    "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_RUNTIME_DIR}" "${CUDA_CACHE_PATH}" \
    "${NUMBA_CACHE_DIR}" "${FLASHINFER_WORKSPACE_DIR}" "${RAY_TMPDIR}" \
    "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${SENTENCE_TRANSFORMERS_HOME}"
chmod 700 "${HOME}" "${XDG_RUNTIME_DIR}" || true

ray stop --force >/dev/null 2>&1 || true

gpu_monitor_pid=""
gpu_wandb_monitor_pid=""

gpu_id_csv() {
    local start="$1"
    local end="$2"
    local out=""
    local i
    if [ "${end}" -lt "${start}" ]; then
        printf ''
        return
    fi
    for ((i = start; i <= end; i++)); do
        if [ -n "${out}" ]; then
            out+=","
        fi
        out+="${i}"
    done
    printf '%s' "${out}"
}

start_gpu_monitor() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return
    fi
    local gpu_log="${RUN_LOG_DIR}/gpu-util-rank-${rank}.csv"
    local gpu_err="${RUN_LOG_DIR}/gpu-util-rank-${rank}.err"
    (
        if [ ! -s "${gpu_log}" ]; then
            printf 'sample_time,host,rank,index,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,power_draw_w\n'
        fi
        while true; do
            sample_time="$(date +'%F %T')"
            nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits |
                while IFS= read -r line; do
                    printf '%s,%s,%s,%s\n' "${sample_time}" "${node}" "${rank}" "${line}"
                done
            sleep "${GPU_MONITOR_INTERVAL}"
        done
    ) >>"${gpu_log}" 2>>"${gpu_err}" &
    gpu_monitor_pid="$!"
}

start_wandb_gpu_monitor() {
    if [ "${GPU_WANDB_MONITOR}" != "1" ]; then
        return
    fi
    if [ ! -f "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" ]; then
        echo "GPU W&B monitor script missing: ${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" >&2
        return
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 missing; skipping GPU W&B monitor" >&2
        return
    fi

    local train_gpus=""
    local rollout_gpus=""
    if [ "${rank}" -lt "${ACTOR_NUM_NODES}" ]; then
        train_gpus="$(gpu_id_csv 0 "$((ACTOR_NUM_GPUS_PER_NODE - 1))")"
        if [ "${ACTOR_NUM_GPUS_PER_NODE}" -lt "${GPUS_PER_NODE}" ]; then
            rollout_gpus="$(gpu_id_csv "${ACTOR_NUM_GPUS_PER_NODE}" "$((GPUS_PER_NODE - 1))")"
        fi
    else
        rollout_gpus="$(gpu_id_csv 0 "$((GPUS_PER_NODE - 1))")"
    fi

    local monitor_log="${RUN_LOG_DIR}/wandb-gpu-monitor-rank-${rank}.log"
    local monitor_csv="${RUN_LOG_DIR}/wandb-gpu-rank-${rank}.csv"
    local wandb_args=()
    if [ "${USE_WANDB}" = "1" ]; then
        wandb_args=(
            --wandb-project "${WANDB_PROJECT}"
            --wandb-entity "${WANDB_ENTITY}"
            --wandb-run-id "${WANDB_RUN_ID}"
            --wandb-dir "${WANDB_DIR}"
            --wandb-label "gpu-monitor-rank-${rank}"
            --wandb-mode shared
        )
    else
        wandb_args=(--no-wandb)
    fi

    (
        python3 "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" \
            --interval-s "${GPU_WANDB_MONITOR_INTERVAL}" \
            --out-csv "${monitor_csv}" \
            --train-gpus "${train_gpus}" \
            --rollout-gpus "${rollout_gpus}" \
            --metric-prefix "polar_system/rank_${rank}" \
            "${wandb_args[@]}"
    ) >"${monitor_log}" 2>&1 &
    gpu_wandb_monitor_pid="$!"
}

stop_gpu_monitor() {
    if [ -n "${gpu_monitor_pid}" ]; then
        kill "${gpu_monitor_pid}" 2>/dev/null || true
        wait "${gpu_monitor_pid}" 2>/dev/null || true
        gpu_monitor_pid=""
    fi
}

stop_wandb_gpu_monitor() {
    if [ -n "${gpu_wandb_monitor_pid}" ]; then
        kill "${gpu_wandb_monitor_pid}" 2>/dev/null || true
        wait "${gpu_wandb_monitor_pid}" 2>/dev/null || true
        gpu_wandb_monitor_pid=""
    fi
}

cleanup_ray() {
    stop_wandb_gpu_monitor
    stop_gpu_monitor
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

patch_node_runtime() {
    if [ "${PATCH_SGLANG}" = "1" ]; then
        local patch_log="${RUN_LOG_DIR}/patch-sglang-rank-${rank}.log"
        echo "[$(date +'%F %T')] Patching SGLang inside rank ${rank} container on ${node}"
        if ! bash "${PROJECT_ROOT}/scripts/patch/patch_sglang.sh" >"${patch_log}" 2>&1; then
            echo "SGLang patch failed on rank ${rank}; see ${patch_log}" >&2
            tail -100 "${patch_log}" >&2 || true
            exit 1
        fi
    fi
}

patch_node_runtime
start_gpu_monitor
start_wandb_gpu_monitor

if [ "${rank}" = "0" ]; then
    echo "[$(date +'%F %T')] Starting Ray head on ${node} (${RAY_HEAD_IP})"
    ray start --head \
        --node-ip-address="${RAY_HEAD_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --num-cpus="${RAY_NUM_CPUS_PER_NODE}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --temp-dir="${RAY_TMPDIR}" \
        --disable-usage-stats >"${RUN_LOG_DIR}/ray-head.log" 2>&1

    python - <<'PY'
import os
import time

import ray

expected = float(os.environ["TOTAL_GPUS"])
ray.init(address=os.environ["RAY_ADDRESS"])
for _ in range(120):
    resources = ray.cluster_resources()
    got = float(resources.get("GPU", 0))
    print(f"Ray GPUs visible: {got}/{expected}", flush=True)
    if got >= expected:
        break
    time.sleep(5)
else:
    raise SystemExit(f"Ray cluster did not reach {expected} GPUs")
ray.shutdown()
PY

    finish_workers() {
        touch "${STOP_FILE}"
        ray stop --force >/dev/null 2>&1 || true
    }
    trap finish_workers EXIT

    export INSIDE_TRAIN_SQSH=1
    export USE_TRAIN_SQSH=1
    export RAY_USE_EXISTING_CLUSTER=1
    export RAY_STOP_ON_EXIT=0
    export USE_RAY_JOB_SUBMIT=0
    export RAY_ADDRESS
    export RAY_JOB_ADDRESS
    export RAY_NUM_CPUS
    export SGLANG_ROUTER_HOST="${RAY_HEAD_IP}"
    export SGLANG_ROUTER_BASE_URL="http://${RAY_HEAD_IP}:${SGLANG_ROUTER_PORT}"

    echo "[$(date +'%F %T')] Launching PI SWE-Gym training through run_pi_interactive_apptainer.sh"
    bash "${PROJECT_ROOT}/examples/swegym_slime_grpo/run_pi_interactive_apptainer.sh"
else
    echo "[$(date +'%F %T')] Waiting for Ray head at ${RAY_ADDRESS} from ${node} (${node_ip})"
    python - <<'PY'
import os
import socket
import time

host, port_s = os.environ["RAY_ADDRESS"].split(":")
port = int(port_s)
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise SystemExit(f"Ray head did not open at {host}:{port}")
PY

    echo "[$(date +'%F %T')] Starting Ray worker rank ${rank} on ${node} (${node_ip})"
    ray start \
        --address="${RAY_ADDRESS}" \
        --node-ip-address="${node_ip}" \
        --num-cpus="${RAY_NUM_CPUS_PER_NODE}" \
        --num-gpus="${GPUS_PER_NODE}" \
        --temp-dir="${RAY_TMPDIR}" \
        --block >"${RUN_LOG_DIR}/ray-worker-${rank}.log" 2>&1 &
    ray_pid="$!"

    while [ ! -f "${STOP_FILE}" ]; do
        if ! kill -0 "${ray_pid}" 2>/dev/null; then
            wait "${ray_pid}"
            exit $?
        fi
        sleep 5
    done
    ray stop --force >/dev/null 2>&1 || true
    wait "${ray_pid}" 2>/dev/null || true
fi
WORKER
chmod +x "${WORKER_SCRIPT}"

export PROJECT_ROOT REFERENCE_ROOT SCRIPT_DIR
export MODEL_NAME PI_MODEL_NAME HF_CHECKPOINT REF_LOAD FULL_PROMPT_DATA
export TRAIN_SQSH SLIME_DIR MEGATRON_DIR SHARED_SIF_DIR APPTAINER_IMAGE_DIR AGENT_CLI_DIR HOST_NVIDIA_LIB_DIR
export HF_CACHE_ROOT HF_HOME HUGGINGFACE_HUB_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE SENTENCE_TRANSFORMERS_HOME
export RUN_SERIES_ID RUN_ID RUN_DIR RUN_LOG_DIR ROLLOUT_SAVE_DIR SAVE_DIR STOP_FILE WORKER_SCRIPT
export SMOKE_NUM_ROWS NUM_EPOCH NUM_ROLLOUT SAVE_INTERVAL START_ROLLOUT_ID
export GPUS_PER_NODE TOTAL_GPUS TRAIN_NUM_GPUS ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE
export ROLLOUT_NUM_GPUS ROLLOUT_NUM_GPUS_PER_ENGINE TENSOR_MODEL_PARALLEL_SIZE
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT NUM_STEPS_PER_ROLLOUT DISTRIBUTED_TIMEOUT_MINUTES
export MAX_TOKENS_PER_GPU ROLLOUT_MAX_RESPONSE_LEN ROLLOUT_MAX_PROMPT_LEN
export SGLANG_CONTEXT_LENGTH SGLANG_MEM_FRACTION_STATIC SGLANG_LOG_LEVEL
export RAY_HEAD_IP RAY_ADDRESS RAY_JOB_ADDRESS RAY_PORT RAY_DASHBOARD_PORT RAY_NUM_CPUS RAY_NUM_CPUS_PER_NODE
export ROLLOUT_PORT GATEWAY_PORT SGLANG_ROUTER_PORT SLIME_SGLANG_BASE_PORT
export POLAR_BUILDER_STRATEGY POLAR_MIN_COMPLETE_ACCEPT_FRACTION POLAR_MAX_ASYNC_LEVEL
export POLAR_RUNTIME_MEMORY_MB POLAR_TASK_TIMEOUT_SECONDS POLAR_REQUEST_TIMEOUT
export PI_API_TYPE PI_CONTEXT_WINDOW PI_MAX_TOKENS PI_THINKING
export PI_FAIL_ON_CONTEXT_LIMIT PI_COMPACTION_ENABLED PI_RETRY_ENABLED PI_RETRY_MAX_RETRIES PI_PROVIDER_MAX_RETRIES
export TRAIN_LR CLIP_GRAD KL_LOSS_COEF EPS_CLIP EPS_CLIP_HIGH
export SLIME_ZERO_NONFINITE_GRADS PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export RAY_MEMORY_USAGE_THRESHOLD RAY_memory_usage_threshold GPU_MONITOR_INTERVAL
export GPU_WANDB_MONITOR GPU_WANDB_MONITOR_INTERVAL
export USE_WANDB WANDB_MODE WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_RUN_ID WANDB_RANDOM_SUFFIX WANDB_TEAM WANDB_HOST WANDB_API_KEY WANDB_DIR
export PATCH_SLIME PATCH_SGLANG PATCH_RAY_PY312 FRESH_START RESUME_FROM_SAVE DRY_RUN

cat <<EOF
=============================================
SWE-Gym PI GRPO Slurm launch
  Job ID:        ${SLURM_JOB_ID}
  Nodes:         ${NUM_NODES}
  Head node:     ${HEAD_NODE}
  Ray address:   ${RAY_ADDRESS}
  GPUs/node:     ${GPUS_PER_NODE}
  Total GPUs:    ${TOTAL_GPUS}
  Actor GPUs:    ${TRAIN_NUM_GPUS} (${ACTOR_NUM_NODES} node(s) x ${ACTOR_NUM_GPUS_PER_NODE})
  Rollout GPUs:  ${ROLLOUT_NUM_GPUS}
  Idle GPUs:     ${UNUSED_GPUS}
  DP/TP:         ${DATA_PARALLEL_SIZE}/${TENSOR_MODEL_PARALLEL_SIZE}
  Batch:         rollout=${ROLLOUT_BATCH_SIZE}, samples/prompt=${N_SAMPLES_PER_PROMPT}, steps/rollout=${NUM_STEPS_PER_ROLLOUT}, global=${GLOBAL_BATCH_SIZE}
  Distributed:   timeout=${DISTRIBUTED_TIMEOUT_MINUTES}m
  Builder:       ${POLAR_BUILDER_STRATEGY}
  Length:        smoke_rows=${SMOKE_NUM_ROWS}, num_epoch=${NUM_EPOCH}, num_rollout=${NUM_ROLLOUT:-<epoch-mode>}
  Expected:      ${EXPECTED_ROLLOUTS} rollout(s)
  Start rollout: ${START_ROLLOUT_ID:-<auto>}
  Auto resume:   ${AUTO_RESUME_FROM_SAVE}
  Fresh start:   ${FRESH_START}, resume_from_save=${RESUME_FROM_SAVE}
  Stability:     lr=${TRAIN_LR}, clip_grad=${CLIP_GRAD}, kl=${KL_LOSS_COEF}, eps=${EPS_CLIP}/${EPS_CLIP_HIGH}, max_tokens/gpu=${MAX_TOKENS_PER_GPU}
  PI budget:     sglang_ctx=${SGLANG_CONTEXT_LENGTH}, pi_context=${PI_CONTEXT_WINDOW}, pi_max_tokens=${PI_MAX_TOKENS}
  PI fail-fast:  context_limit=${PI_FAIL_ON_CONTEXT_LIMIT}, compaction=${PI_COMPACTION_ENABLED}, retry=${PI_RETRY_ENABLED}, provider_retries=${PI_PROVIDER_MAX_RETRIES}
  Sandbox mem:   ${POLAR_RUNTIME_MEMORY_MB:-<none>} MB per command
  Model served:  ${MODEL_NAME}
  PI model:      ${PI_MODEL_NAME}
  HF checkpoint: ${HF_CHECKPOINT}
  HF cache:      ${HF_HOME}
  Megatron load: ${REF_LOAD}
  Run series:    ${RUN_SERIES_ID}
  Run ID:        ${RUN_ID}
  Run dir:       ${RUN_DIR}
  Save dir:      ${SAVE_DIR}
  W&B project:   ${WANDB_PROJECT}
  W&B group:     ${WANDB_GROUP}
  W&B run id:    ${WANDB_RUN_ID}
  GPU W&B mon.:  ${GPU_WANDB_MONITOR} every ${GPU_WANDB_MONITOR_INTERVAL}s
=============================================
EOF

srun \
    --overlap \
    --nodes="${NUM_NODES}" \
    --ntasks="${NUM_NODES}" \
    --ntasks-per-node=1 \
    --gres="gpu:${GPUS_PER_NODE}" \
    --container-image="${TRAIN_SQSH}" \
    --container-mounts="${TRAIN_CONTAINER_MOUNTS}" \
    --container-workdir="${PROJECT_ROOT}" \
    --container-writable \
    --no-container-mount-home \
    bash "${WORKER_SCRIPT}"
