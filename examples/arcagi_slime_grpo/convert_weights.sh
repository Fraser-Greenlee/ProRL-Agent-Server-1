#!/usr/bin/env bash
# Convert Fraser/Qwen3.6-27B-ARC-Hy HF weights to Megatron torch_dist for Slime.
# Qwen3.6-27B is a dense hybrid checkpoint (Qwen3_5ForConditionalGeneration; 3
# GatedDeltaNet linear + 1 full-attention per 4 layers, 64 layers).  Weight
# loading goes through slime_plugins.mbridge.qwen3_5 (text_config-aware), same
# as the 4B SWE-Gym example.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

# uv + CUDA toolkit on PATH (non-login GPU shell); NVTE_CUDA_INCLUDE_DIR works
# around the TE 2.5.0 Path(nvidia.__file__=None) import crash (megatron imports
# TE). See launch_e2e.sh.
export PATH="${HOME}/.local/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
[ -x "${CUDA_HOME}/bin/nvcc" ] && export PATH="${CUDA_HOME}/bin:${PATH}"
export NVTE_CUDA_INCLUDE_DIR="${NVTE_CUDA_INCLUDE_DIR:-${CUDA_HOME}/include}"

if [ ! -f "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}. Clone it first:"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

# The SFT'd checkpoint's HF repo is missing ancillary files; on the cluster we
# point at the patched local snapshot (see the sglang-sft-checkpoint memory).
HF_CHECKPOINT="${HF_CHECKPOINT:-Fraser/Qwen3.6-27B-ARC-Hy}"
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.6-27B-ARC-Hy_torch_dist}"
mkdir -p "$OUTPUT_DIR"

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

echo "Converting ${HF_CHECKPOINT} -> ${OUTPUT_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
torchrun --nproc_per_node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$OUTPUT_DIR" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion

echo "Done: ${OUTPUT_DIR}"
