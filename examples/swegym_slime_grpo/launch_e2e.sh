#!/usr/bin/env bash
# Single-entry launcher for the full SWE-Gym Slime GRPO example.
#
# This script bootstraps the pieces that are safe to automate on a cluster:
# external checkouts, local editable installs, Slime/SGLang patches, SWE-Gym
# JSONL data, shared agent CLI assets, base runtime images, Megatron weight
# conversion, and finally the Polar + Slime training run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.2.4}"

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-main}"

resolve_local_hf_checkpoint() {
    local model_id="$1"
    case "$model_id" in
        /*|./*|../*|~*)
            echo "$model_id"
            return
            ;;
    esac

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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
HF_CHECKPOINT_REQUESTED="${HF_CHECKPOINT:-$MODEL_NAME}"
HF_CHECKPOINT="$(resolve_local_hf_checkpoint "$HF_CHECKPOINT_REQUESTED")"
if [ "$HF_CHECKPOINT" != "$HF_CHECKPOINT_REQUESTED" ]; then
    echo "Using local HF checkpoint snapshot: $HF_CHECKPOINT"
fi
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD}}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"

INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-1}"
PREPARE_IMAGES="${PREPARE_IMAGES:-0}"
PULL_JOBS="${PULL_JOBS:-4}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"
MONITOR_GPU="${MONITOR_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4,5,6,7}"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $1" >&2
        exit 1
    fi
}

clone_if_missing() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    local dest="$4"
    if [ -d "${dest}/.git" ]; then
        echo "${name} checkout exists: ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then
        echo "ERROR: ${name} path exists but is not a git checkout: ${dest}" >&2
        exit 1
    fi
    echo "Cloning ${name} ${ref} -> ${dest}"
    git clone --branch "${ref}" --depth 1 "${repo}" "${dest}"
}

checkpoint_ready() {
    [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]
}

maybe_login_wandb() {
    if [ -z "${WANDB_API_KEY:-}" ]; then
        return
    fi
    python - <<'PY'
import os

try:
    import wandb
except Exception:
    raise SystemExit(0)

key = os.environ.get("WANDB_API_KEY")
if key:
    wandb.login(key=key, relogin=True)
PY
}

require_cmd git
require_cmd python
require_cmd ray
if [ "${INSTALL_EDITABLE}" = "1" ] || [ "${MONITOR_GPU}" = "1" ]; then
    require_cmd uv
fi

clone_if_missing "Slime" "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
clone_if_missing "Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"

if [ "${INSTALL_EDITABLE}" = "1" ]; then
    uv pip install -e .
    uv pip install -e "${SLIME_DIR}"
    uv pip install -e "${MEGATRON_DIR}"
fi

bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"
if [ "${APPLY_SGLANG_PATCH}" = "1" ]; then
    bash "${PROJECT_ROOT}/scripts/patch/patch_sglang.sh"
fi

python "${SCRIPT_DIR}/prepare_data.py"

if [ "${PREPARE_IMAGES}" = "1" ]; then
    python "${SCRIPT_DIR}/build_images.py" \
        --agent-cli-dir "${AGENT_CLI_DIR}" \
        --pull-jobs "${PULL_JOBS}"
fi

# Ensure the shared agent CLI directory exists (Node + coding agent CLIs).
AGENT_CLI_DIR="${AGENT_CLI_DIR}" bash "${SCRIPT_DIR}/prepare_agent_cli.sh"

if [ "${CONVERT_WEIGHTS}" = "1" ] || { [ "${CONVERT_WEIGHTS}" = "auto" ] && ! checkpoint_ready; }; then
    HF_CHECKPOINT="${HF_CHECKPOINT}" \
    TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
    SLIME_DIR="${SLIME_DIR}" \
    MEGATRON_DIR="${MEGATRON_DIR}" \
        bash "${SCRIPT_DIR}/convert_weights.sh"
fi

maybe_login_wandb

MONITOR_PID=""
if [ "${MONITOR_GPU}" = "1" ]; then
    python "${PROJECT_ROOT}/scripts/monitor_wandb_gpu.py" \
        --no-wandb \
        --train-gpus "${TRAIN_GPUS}" \
        --rollout-gpus "${ROLLOUT_GPUS}" &
    MONITOR_PID="$!"
fi

cleanup() {
    if [ -n "${MONITOR_PID}" ]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

HF_CHECKPOINT="${HF_CHECKPOINT}" \
REF_LOAD="${REF_LOAD}" \
TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
SLIME_DIR="${SLIME_DIR}" \
MEGATRON_DIR="${MEGATRON_DIR}" \
AGENT_CLI_DIR="${AGENT_CLI_DIR}" \
    bash "${SCRIPT_DIR}/run.sh"
