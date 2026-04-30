#!/usr/bin/env bash
# Submit the SWE-Gym Slime GRPO e2e pipeline as a SLURM job.
#
# Usage:
#   bash examples/swegym_slime_grpo/submit_slurm.sh
#
# Customise via environment variables (see defaults below).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ── SLURM settings ───────────────────────────────────────────────────
ACCOUNT="${ACCOUNT:-nvr_lpr_agentic}"
PARTITION="${PARTITION:-interactive}"
WALL_TIME="${WALL_TIME:-4:00:00}"
NUM_NODES="${NUM_NODES:-1}"
JOB_NAME="${JOB_NAME:-polar-swegym-grpo}"
SLURM_GPUS="${SLURM_GPUS:-8}"
SLURM_MEM="${SLURM_MEM:-}"
SLURM_RESERVATION="${SLURM_RESERVATION:-}"

# ── Paths ────────────────────────────────────────────────────────────
SIF_DIR="${POLAR_SIF_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/singularity_images_v3}"
HF_HOME="${HF_HOME:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
APPTAINER_ENV="${APPTAINER_ENV:-${PROJECT_ROOT}/tmp/apptainer-conda}"
APPTAINER_CACHE_DIR="${APPTAINER_CACHE_DIR:-${PROJECT_ROOT}/tmp/apptainer-cache}"
APPTAINER_TMP_DIR="${APPTAINER_TMP_DIR:-${PROJECT_ROOT}/tmp/apptainer-tmp}"
WANDB_MODE="${WANDB_MODE:-offline}"

USE_TRAIN_SQSH="${USE_TRAIN_SQSH:-1}"
TRAIN_SQSH="${POLR_TRAIN_SQSH:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/dockers/polr_swegym_slime_grpo_train.sqsh}"
TRAIN_CONTAINER_VENV="${POLR_TRAIN_VENV:-/opt/polr_venv}"
TRAIN_CONTAINER_MOUNTS="${TRAIN_CONTAINER_MOUNTS:-/lustre/fs1:/lustre/fs1,/lustre/fsw:/lustre/fsw}"
if [ "$USE_TRAIN_SQSH" = "1" ] && [ ! -f "$TRAIN_SQSH" ]; then
    echo "ERROR: training sqsh not found: ${TRAIN_SQSH}" >&2
    echo "  Build it with: bash ${SCRIPT_DIR}/build_training_sqsh.sh" >&2
    exit 1
fi

# ── Log directory ────────────────────────────────────────────────────
LOG_DIR="${PROJECT_ROOT}/logs/slurm"
mkdir -p "${LOG_DIR}"
mkdir -p "${APPTAINER_CACHE_DIR}" "${APPTAINER_TMP_DIR}"

echo "============================================="
echo "Polar SWE-Gym Slime GRPO — SLURM submission"
echo "  Project:   ${PROJECT_ROOT}"
echo "  SIF dir:   ${SIF_DIR}"
if [ "$USE_TRAIN_SQSH" = "1" ]; then
    echo "  Train sqsh:${TRAIN_SQSH}"
    echo "  Train venv:${TRAIN_CONTAINER_VENV}"
else
    echo "  Train env: ${PROJECT_ROOT}/.venv"
fi
echo "  Apptainer: ${APPTAINER_ENV}/bin/apptainer"
echo "  Partition:  ${PARTITION}"
echo "  Account:    ${ACCOUNT}"
echo "  Wall time:  ${WALL_TIME}"
echo "  GPUs:       ${SLURM_GPUS}"
if [ -n "${SLURM_MEM}" ]; then
    echo "  Memory:     ${SLURM_MEM}"
fi
if [ -n "${SLURM_RESERVATION}" ]; then
    echo "  Reservation:${SLURM_RESERVATION}"
fi
echo "  Log dir:    ${LOG_DIR}"
echo "============================================="

# The --wrap command launches launch_e2e.sh either inside the training sqsh
# (default) or in PROJECT_ROOT/.venv when USE_TRAIN_SQSH=0.
CUDA_TOOLKIT="${CUDA_TOOLKIT:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/cuda-toolkit-12.4}"
if [ "$USE_TRAIN_SQSH" = "1" ]; then
    PY_SITE="${TRAIN_CONTAINER_VENV}/lib/python3.12/site-packages"
    TRAIN_ENV_PREFIX="export POLR_TRAIN_VENV=\"${TRAIN_CONTAINER_VENV}\" && export POLR_TRAIN_SITE_PACKAGES=\"${PY_SITE}\" && export PATH=\"${TRAIN_CONTAINER_VENV}/bin:${APPTAINER_ENV}/bin:${CUDA_TOOLKIT}/bin:\${PATH}\" &&"
else
    PY_SITE="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
    TRAIN_ENV_PREFIX="source ${PROJECT_ROOT}/.venv/bin/activate && export POLR_TRAIN_VENV=\"${PROJECT_ROOT}/.venv\" && export POLR_TRAIN_SITE_PACKAGES=\"${PY_SITE}\" && export PATH=\"${APPTAINER_ENV}/bin:${CUDA_TOOLKIT}/bin:\${PATH}\" &&"
fi
VENV_CUDA_LD="${PY_SITE}/torch/lib:${PY_SITE}/nvidia/cuda_runtime/lib:${PY_SITE}/nvidia/cuda_nvrtc/lib:${PY_SITE}/nvidia/nvjitlink/lib:${PY_SITE}/nvidia/cublas/lib:${PY_SITE}/nvidia/cudnn/lib:${PY_SITE}/nvidia/nccl/lib:${PY_SITE}/nvidia/cusparse/lib:${PY_SITE}/nvidia/cusolver/lib:${PY_SITE}/nvidia/cufft/lib:${PY_SITE}/nvidia/curand/lib"
INNER_CMD="$(cat <<EOF
${TRAIN_ENV_PREFIX} \
export PYTHONNOUSERSITE=1 && \
export POLAR_SIF_DIR="${SIF_DIR}" && \
export HF_HOME="${HF_HOME}" && \
export HF_TOKEN="${HF_TOKEN}" && \
export WANDB_API_KEY="\${WANDB_API_KEY:-}" && \
export WANDB_MODE="${WANDB_MODE}" && \
export CUDA_HOME="${CUDA_TOOLKIT}" && \
export LD_LIBRARY_PATH="${VENV_CUDA_LD}:\${LD_LIBRARY_PATH:-}" && \
export POLAR_APPTAINER_BIN="${APPTAINER_ENV}/bin/apptainer" && \
export POLAR_APPTAINER_NO_INSTANCE="${POLAR_APPTAINER_NO_INSTANCE:-${USE_TRAIN_SQSH}}" && \
export POLAR_JOB_CACHE_ROOT="/tmp/polar-cache-\${SLURM_JOB_ID:-manual}" && \
rm -rf "\${POLAR_JOB_CACHE_ROOT}" && \
mkdir -p "\${POLAR_JOB_CACHE_ROOT}/home" "\${POLAR_JOB_CACHE_ROOT}/apptainer-cache" "\${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" "\${POLAR_JOB_CACHE_ROOT}/apptainer-work" "\${POLAR_JOB_CACHE_ROOT}/triton-cache" "\${POLAR_JOB_CACHE_ROOT}/triton-home" "\${POLAR_JOB_CACHE_ROOT}/torchinductor" "\${POLAR_JOB_CACHE_ROOT}/torch-extensions" "\${POLAR_JOB_CACHE_ROOT}/xdg" "\${POLAR_JOB_CACHE_ROOT}/xdg-config" "\${POLAR_JOB_CACHE_ROOT}/xdg-runtime" "\${POLAR_JOB_CACHE_ROOT}/cuda-cache" "\${POLAR_JOB_CACHE_ROOT}/numba" && \
if [ "${USE_TRAIN_SQSH}" != "1" ]; then :; else mkdir -p /home/conda/feedstock_root/build_artifacts/apptainer_1764715648377/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p/var/apptainer/mnt/session; fi && \
chmod 700 "\${POLAR_JOB_CACHE_ROOT}/xdg-runtime" && \
export HOME="\${POLAR_JOB_CACHE_ROOT}/home" && \
export APPTAINER_CACHEDIR="\${POLAR_JOB_CACHE_ROOT}/apptainer-cache" && \
export APPTAINER_TMPDIR="\${POLAR_JOB_CACHE_ROOT}/apptainer-tmp" && \
export APPTAINER_WORKDIR="\${POLAR_JOB_CACHE_ROOT}/apptainer-work" && \
export SINGULARITY_CACHEDIR="\${APPTAINER_CACHEDIR}" && \
export SINGULARITY_TMPDIR="\${APPTAINER_TMPDIR}" && \
export TRITON_CACHE_DIR="\${POLAR_JOB_CACHE_ROOT}/triton-cache" && \
export TRITON_HOME="\${POLAR_JOB_CACHE_ROOT}/triton-home" && \
export TORCHINDUCTOR_CACHE_DIR="\${POLAR_JOB_CACHE_ROOT}/torchinductor" && \
export TORCH_EXTENSIONS_DIR="\${POLAR_JOB_CACHE_ROOT}/torch-extensions" && \
export XDG_CACHE_HOME="\${POLAR_JOB_CACHE_ROOT}/xdg" && \
export XDG_CONFIG_HOME="\${POLAR_JOB_CACHE_ROOT}/xdg-config" && \
export XDG_RUNTIME_DIR="\${POLAR_JOB_CACHE_ROOT}/xdg-runtime" && \
export CUDA_CACHE_PATH="\${POLAR_JOB_CACHE_ROOT}/cuda-cache" && \
export NUMBA_CACHE_DIR="\${POLAR_JOB_CACHE_ROOT}/numba" && \
export RAY_LOG_ROOT="${PROJECT_ROOT}/tmp/ray-\${SLURM_JOB_ID:-manual}" && \
export RAY_TMPDIR="/tmp/polar-ray-\${SLURM_JOB_ID:-manual}" && \
rm -rf "\${RAY_LOG_ROOT}" "\${RAY_TMPDIR}" && \
mkdir -p "\${RAY_LOG_ROOT}" && \
ln -s "\${RAY_LOG_ROOT}" "\${RAY_TMPDIR}" && \
export RUN_DIR="\${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo-\${SLURM_JOB_ID:-manual}}" && \
export SAVE_DIR="\${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b-\${SLURM_JOB_ID:-manual}}" && \
export INSTALL_EDITABLE=0 && \
export APPLY_SGLANG_PATCH=0 && \
export PREPARE_IMAGES=0 && \
bash ${SCRIPT_DIR}/launch_e2e.sh
EOF
)"

if [ "$USE_TRAIN_SQSH" = "1" ]; then
    printf -v TRAIN_SQSH_Q '%q' "$TRAIN_SQSH"
    printf -v TRAIN_MOUNTS_Q '%q' "$TRAIN_CONTAINER_MOUNTS"
    printf -v PROJECT_ROOT_Q '%q' "$PROJECT_ROOT"
    printf -v INNER_CMD_Q '%q' "$INNER_CMD"
    WRAP_CMD="srun --nodes=1 --ntasks=1 --gres=gpu:${SLURM_GPUS} --mem=0 --container-image=${TRAIN_SQSH_Q} --container-mounts=${TRAIN_MOUNTS_Q} --container-workdir=${PROJECT_ROOT_Q} --container-writable --no-container-mount-home bash -lc ${INNER_CMD_Q}"
else
    WRAP_CMD="$INNER_CMD"
fi

SBATCH_MEM_ARG=()
if [ -n "${SLURM_MEM}" ]; then
    SBATCH_MEM_ARG=(--mem="${SLURM_MEM}")
fi
SBATCH_RESERVATION_ARG=()
if [ -n "${SLURM_RESERVATION}" ]; then
    SBATCH_RESERVATION_ARG=(--reservation="${SLURM_RESERVATION}")
fi

JOB_ID=$(sbatch \
    --nodes="${NUM_NODES}" \
    --account="${ACCOUNT}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --time="${WALL_TIME}" \
    --gres="gpu:${SLURM_GPUS}" \
    "${SBATCH_MEM_ARG[@]}" \
    "${SBATCH_RESERVATION_ARG[@]}" \
    --output="${LOG_DIR}/%x-%j.out" \
    --error="${LOG_DIR}/%x-%j.err" \
    --export=ALL \
    --parsable \
    --wrap="${WRAP_CMD}")

echo ""
echo "Submitted SLURM job: ${JOB_ID}"
echo "Monitor stdout: tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Monitor stderr: tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.err"
