#!/usr/bin/env bash
# Single-entry launcher for the ARC-AGI Slime GRPO example.
#
# Bootstraps the training stack (Slime + Megatron checkouts, patches, editable
# installs, Transformer Engine + Flash Linear Attention), builds the rollout
# docker image to the shared-NFS tarball, prepares the train JSONL, converts the
# HF checkpoint to Megatron torch_dist, then hands off to run.sh.
#
# This is the arcagi cousin of examples/swegym_slime_grpo/launch_e2e.sh, minus
# the SWE-Gym specifics (apptainer per-instance images, swegym harness package).
# Our rollout runtime is a single docker image (build_image.py) and our grader
# is the in-repo arc_compress evaluator — no external harness package.
#
# MUST run on a GPU node (TE builds from source; weight conversion needs CUDA).
# Toggle stages with the env flags near the top; everything is idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Secrets (WANDB_API_KEY) from the gitignored env file.
for envf in "${PROJECT_ROOT}/.env.local" "${SCRIPT_DIR}/.env.local"; do
    if [ -f "$envf" ]; then set -a; source "$envf"; set +a; fi
done

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# GPU compute nodes run a non-login shell, so uv (~/.local/bin) and the CUDA
# toolkit (nvcc) aren't on PATH by default. Transformer Engine builds from
# source against nvcc, so wire both up before any install step.
export PATH="${HOME}/.local/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
# Transformer Engine 2.5.0 crashes at import doing Path(nvidia.__file__) when
# the pip `nvidia` namespace package has __file__=None (it does). Setting this
# makes TE skip that path entirely. Must be set wherever TE is imported.
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/include}"
# Re-prepend the venv bin LAST so `ray`/`python`/`uv` resolve to the venv
# (py3.12), not ~/.local/bin (a py3.10 ray that breaks the Ray cluster).
export PATH="${PYTHON_BIN_DIR}:${PATH}"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.3.0}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
# slime v0.3.0 pins this exact Megatron-LM commit in its docker/Dockerfile
# (NOT a tag — tags like 26.04-alpha.rc1 moved and dropped
# megatron/training/tokenizer/, which slime imports). Megatron-core's API
# drifts hard between commits, so this MUST stay paired with TE 2.10.0 below.
MEGATRON_REF="${MEGATRON_REF:-1dcf0dafa884ad52ffb243625717a3471643e087}"

# Model — the SFT'd checkpoint. On the cluster, point at the patched local
# snapshot (its HF repo is missing ancillary files; see the sglang memory).
HF_CHECKPOINT="${HF_CHECKPOINT:-Fraser/Qwen3.6-27B-ARC-Hy}"
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.6-27B-ARC-Hy_torch_dist}}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD}}"
RUN_ID="${RUN_ID:-${WANDB_RUN_ID:-arcagi-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/arcagi_slime_grpo_qwen36_27b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"

# auto-compress repo (build context for the rollout image + prepare_data source).
AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
ARCAGI_IMAGE="${ARCAGI_IMAGE:-polar-arcagi:latest}"
ARCAGI_IMAGE_TARBALL="${ARCAGI_IMAGE_TARBALL:-/home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz}"
N_TASKS="${N_TASKS:-40}"

INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"   # TE + FLA (+ flash-attn on B200)
FLASH_LINEAR_ATTENTION_VERSION="${FLASH_LINEAR_ATTENTION_VERSION:-0.5.0}"
MBRIDGE_VERSION="${MBRIDGE_VERSION:-0.15.1}"
APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-1}"
BUILD_IMAGE="${BUILD_IMAGE:-auto}"        # auto = build only if tarball missing
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"  # auto = convert only if checkpoint missing

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "ERROR: required command not found: $1" >&2; exit 1; }
}

clone_if_missing() {
    # Args: name repo ref dest [recursive]
    # ref may be a tag, branch, OR a raw commit SHA. --branch only accepts the
    # former, so for a 40-hex SHA we clone then fetch+checkout the commit.
    local name="$1" repo="$2" ref="$3" dest="$4" recursive="${5:-0}"
    local sub=""
    [ "$recursive" = "1" ] && sub="--recurse-submodules"
    if [ -d "${dest}/.git" ]; then
        echo "${name} checkout exists: ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then echo "ERROR: ${name} exists but is not a git checkout: ${dest}" >&2; exit 1; fi
    if printf '%s' "$ref" | grep -Eq '^[0-9a-f]{40}$'; then
        echo "Cloning ${name} @ commit ${ref} -> ${dest}"
        git clone ${sub} "${repo}" "${dest}"
        git -C "${dest}" fetch --depth 1 origin "${ref}"
        git -C "${dest}" checkout "${ref}"
        [ "$recursive" = "1" ] && git -C "${dest}" submodule update --init --recursive
    else
        echo "Cloning ${name} ${ref} -> ${dest}"
        git clone ${sub} --branch "${ref}" --depth 1 "${repo}" "${dest}"
    fi
}

checkpoint_ready() { [ -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; }

maybe_login_wandb() {
    [ -z "${WANDB_API_KEY:-}" ] && return
    "${PYTHON_BIN}" - <<'PY'
import os
try:
    import wandb
except Exception:
    raise SystemExit(0)
key = os.environ.get("WANDB_API_KEY")
if key and hasattr(wandb, "login"):
    wandb.login(key=key, relogin=True)
PY
}

# Reuse swegym's training-stack installer verbatim if present (TE + FLA +
# flash-attn arch logic is model-agnostic); otherwise inline a minimal version.
ensure_training_stack() {
    local cuda_home cudnn_path nccl_path cc=""
    cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [ ! -d "$cuda_home" ] && command -v nvcc >/dev/null 2>&1; then
        cuda_home="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
    cudnn_path="$("${PYTHON_BIN}" -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])' 2>/dev/null || true)"
    # TE 2.10's transformer-engine-torch C++ build #includes nccl.h, shipped by
    # the pip nvidia-nccl package (not the system CUDA toolkit). Add its headers
    # + libs to the build search paths or the build dies with "nccl.h not found".
    nccl_path="$("${PYTHON_BIN}" -c 'import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])' 2>/dev/null || true)"

    # TE must be 2.10.0 — the version slime v0.3.0's Megatron commit is built
    # against. A mismatched TE<->Megatron pairing is the usual cause of import/
    # kernel breakage. Only skip if 2.10.0 is already importable.
    local te_ver
    te_ver="$("${PYTHON_BIN}" -c 'from importlib.metadata import version; print(version("transformer-engine"))' 2>/dev/null || echo "")"
    if [ "$te_ver" = "2.10.0" ] && "${PYTHON_BIN}" -c "import transformer_engine.pytorch" >/dev/null 2>&1; then
        echo "Transformer Engine 2.10.0 present; skipping."
    else
        command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found (need CUDA toolkit / CUDA_HOME)." >&2; exit 1; }
        [ -n "$cudnn_path" ] || { echo "ERROR: pip nvidia-cudnn not in venv (is torch a CUDA build?)." >&2; exit 1; }
        echo "Installing Transformer Engine 2.10.0 (had: ${te_ver:-none})..."
        uv pip install --python "${PYTHON_BIN}" ninja pybind11 setuptools wheel >/dev/null 2>&1 || true
        CUDA_HOME="$cuda_home" \
        CPATH="${cudnn_path}/include:${nccl_path}/include:${cuda_home}/include:${CPATH:-}" \
        LIBRARY_PATH="${cudnn_path}/lib:${nccl_path}/lib:${cuda_home}/lib64:${LIBRARY_PATH:-}" \
            uv pip install --python "${PYTHON_BIN}" --no-build-isolation "transformer-engine[pytorch]==2.10.0"
    fi

    if "${PYTHON_BIN}" -c \
        "from fla.modules import FusedRMSNormGated, ShortConvolution; from fla.ops.gated_delta_rule import chunk_gated_delta_rule" \
        >/dev/null 2>&1; then
        echo "Flash Linear Attention present; skipping."
    else
        echo "Installing Flash Linear Attention ${FLASH_LINEAR_ATTENTION_VERSION}..."
        uv pip install --python "${PYTHON_BIN}" "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}"
    fi

    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')" || true
    if [ "$cc" = "10.0" ]; then
        echo "SM100 detected: building flash-attn 2.7.4.post1 (head_dim=256 fallback)..."
        TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
            MAX_JOBS="${FA_MAX_JOBS:-32}" NVCC_THREADS=4 \
            uv pip install --python "${PYTHON_BIN}" --no-build-isolation flash-attn==2.7.4.post1 || true
    else
        echo "compute_cap=${cc:-unknown} not SM100; skipping flash-attn 2.x source build (TE serves head_dim=256)."
    fi
}

require_cmd git
require_cmd "${PYTHON_BIN}"
require_cmd uv
require_cmd docker
require_cmd envsubst

# ── 1. External checkouts + patches ─────────────────────────────────
clone_if_missing "Slime" "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
clone_if_missing "Megatron-LM" "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}" 1
SLIME_DIR="${SLIME_DIR}" bash "${PROJECT_ROOT}/scripts/patch/patch_slime_router_tokens.sh"

# slime ships a Megatron patch (in its docker/patch/<ver>/) that ADDS the
# Qwen3-Next/3.5 bits Megatron lacks: --use-gated-attention, the
# use_gated_attention config field + gate logic, and checkpoint-load fixes.
# slime's Dockerfile applies it with `git apply --3way` against this exact
# Megatron commit. We must do the same or training/conversion args are
# rejected. Idempotent: a sentinel marks it applied.
MEGATRON_PATCH_VERSION="${MEGATRON_PATCH_VERSION:-latest}"
MEGATRON_PATCH="${SLIME_DIR}/docker/patch/${MEGATRON_PATCH_VERSION}/megatron.patch"
PATCH_SENTINEL="${MEGATRON_DIR}/.slime_megatron_patch_applied"
if [ -f "${PATCH_SENTINEL}" ]; then
    echo "Megatron slime patch already applied (${MEGATRON_PATCH_VERSION})."
elif [ ! -f "${MEGATRON_PATCH}" ]; then
    echo "ERROR: Megatron patch not found at ${MEGATRON_PATCH}" >&2
    exit 1
else
    echo "Applying slime Megatron patch (${MEGATRON_PATCH_VERSION}) --3way..."
    ( cd "${MEGATRON_DIR}" && git update-index --refresh || true
      git apply "${MEGATRON_PATCH}" --3way
      if grep -R -n '^<<<<<<< ' . >/dev/null 2>&1; then
          echo "ERROR: Megatron patch left conflict markers." >&2; exit 1
      fi )
    touch "${PATCH_SENTINEL}"
    # Re-install editable so the patched sources take effect.
    [ "${INSTALL_EDITABLE:-1}" = "1" ] && uv pip install --python "${PYTHON_BIN}" -e "${MEGATRON_DIR}" >/dev/null 2>&1 || true
    echo "Megatron patch applied."
fi

# ── 2. Editable installs ────────────────────────────────────────────
if [ "${INSTALL_EDITABLE}" = "1" ]; then
    uv pip install --python "${PYTHON_BIN}" -e "."
    uv pip install --python "${PYTHON_BIN}" -e "${SLIME_DIR}"
    uv pip install --python "${PYTHON_BIN}" -e "${MEGATRON_DIR}"
    # mbridge: HF<->Megatron weight map slime uses for qwen3_5 conversion.
    uv pip install --python "${PYTHON_BIN}" --no-deps "mbridge==${MBRIDGE_VERSION}"
fi

# ── 2b. numpy 1.x — Megatron asserts numpy<2 ───────────────────────
# The editable installs above can pull numpy 2.x; Megatron hard-asserts
# "does not support numpy 2.x". slime's Dockerfile force-reinstalls numpy<2
# for the same reason. (This is the TRAINING venv only; the rollout container
# has its own numpy.)
if [ "${INSTALL_EDITABLE}" = "1" ]; then
    np_major="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__.split(".")[0])' 2>/dev/null || echo 0)"
    if [ "${np_major}" != "1" ]; then
        echo "Pinning numpy<2 for Megatron (had ${np_major}.x)..."
        uv pip install --python "${PYTHON_BIN}" "numpy<2"
    else
        echo "numpy is already 1.x; skipping."
    fi
fi

# ── 3. Training stack (TE + FLA) ────────────────────────────────────
[ "${INSTALL_TRAINING_STACK}" = "1" ] && ensure_training_stack

# Stop here when only validating the install stage (no image/convert/train).
if [ "${BOOTSTRAP_ONLY:-0}" = "1" ]; then
    echo "BOOTSTRAP_ONLY=1 — install stage complete; verifying imports..."
    "${PYTHON_BIN}" - <<'PY'
import importlib, sys
mods = ["slime", "megatron", "transformer_engine.pytorch", "fla", "polar"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print(f"  OK  import {m}")
    except Exception as e:
        bad.append(m); print(f"  FAIL import {m}: {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
PY
    echo "BOOTSTRAP_ONLY done."
    exit 0
fi

# ── 4. SGLang token-metadata patch (Qwen3.6 VLM text-only RL) ───────
[ "${APPLY_SGLANG_PATCH}" = "1" ] && bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"

# ── 5. Train JSONL ──────────────────────────────────────────────────
# prepare_data.py imports hy + arckit + eval.py — these live in the
# auto-compress env, NOT the training venv. Use auto-compress's venv python.
# Skip entirely if the JSONL is already built (it's deterministic).
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/arcagi_train.jsonl}"
if [ -s "${PROMPT_DATA}" ]; then
    echo "Train JSONL present; skipping prepare_data: ${PROMPT_DATA}"
else
    AC_PYTHON="${AUTO_COMPRESS_PYTHON:-${AUTO_COMPRESS}/.venv/bin/python}"
    [ -x "${AC_PYTHON}" ] || AC_PYTHON="${PYTHON_BIN}"
    echo "Building train JSONL (${DATA_SOURCE:-arcagi}) with ${AC_PYTHON}"
    AUTO_COMPRESS="${AUTO_COMPRESS}" \
        "${AC_PYTHON}" "${SCRIPT_DIR}/prepare_data.py" --n-tasks "${N_TASKS}" \
        --source "${DATA_SOURCE:-arcagi}" --output "${PROMPT_DATA}"
fi

# ── 6. Rollout docker image → NFS tarball ───────────────────────────
# BUILD_IMAGE_ARGS lets the rl_tasks run pass --rl-tasks (bakes sft/rl_tasks).
if [ "${BUILD_IMAGE}" = "1" ] || { [ "${BUILD_IMAGE}" = "auto" ] && [ ! -f "${ARCAGI_IMAGE_TARBALL}" ]; }; then
    echo "Building rollout image -> ${ARCAGI_IMAGE_TARBALL}"
    AUTO_COMPRESS="${AUTO_COMPRESS}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_image.py" ${BUILD_IMAGE_ARGS:-} \
        --image "${ARCAGI_IMAGE}" --save-tar "${ARCAGI_IMAGE_TARBALL}"
else
    echo "Rollout image tarball present; skipping build: ${ARCAGI_IMAGE_TARBALL}"
fi

# ── 7. Weight conversion (HF -> Megatron torch_dist) ────────────────
if [ "${CONVERT_WEIGHTS}" = "1" ] || { [ "${CONVERT_WEIGHTS}" = "auto" ] && ! checkpoint_ready; }; then
    HF_CHECKPOINT="${HF_CHECKPOINT}" TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
    SLIME_DIR="${SLIME_DIR}" MEGATRON_DIR="${MEGATRON_DIR}" \
        bash "${SCRIPT_DIR}/convert_weights.sh"
else
    echo "torch_dist checkpoint present; skipping conversion: ${REF_LOAD}"
fi

maybe_login_wandb

# ── 8. Train ────────────────────────────────────────────────────────
HF_CHECKPOINT="${HF_CHECKPOINT}" REF_LOAD="${REF_LOAD}" TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
SAVE_DIR="${SAVE_DIR}" RUN_ID="${RUN_ID}" SAVE_ROOT="${SAVE_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" SLIME_DIR="${SLIME_DIR}" MEGATRON_DIR="${MEGATRON_DIR}" \
AUTO_COMPRESS="${AUTO_COMPRESS}" ARCAGI_IMAGE="${ARCAGI_IMAGE}" \
ARCAGI_IMAGE_TARBALL="${ARCAGI_IMAGE_TARBALL}" N_TASKS="${N_TASKS}" \
    bash "${SCRIPT_DIR}/run.sh"
