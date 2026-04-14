#!/usr/bin/env bash
# Convert Qwen3-4B HF weights to Megatron torch_dist format for Slime training.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"

if [ ! -f "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}. Clone it first:"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/checkpoints/Qwen3-4B_torch_dist}"
mkdir -p "$OUTPUT_DIR"

# Qwen3-4B architecture
MODEL_ARGS=(
    --swiglu --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728
    --num-attention-heads 32 --group-query-attention --num-query-groups 8
    --use-rotary-position-embeddings --disable-bias-linear
    --normalization RMSNorm --norm-epsilon 1e-6 --rotary-base 1000000
    --vocab-size 151936 --kv-channels 128 --qk-layernorm
)

echo "Converting ${HF_CHECKPOINT} -> ${OUTPUT_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${PROJECT_ROOT}/src" \
torchrun --nproc_per_node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$OUTPUT_DIR" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1

echo "Done: ${OUTPUT_DIR}"
