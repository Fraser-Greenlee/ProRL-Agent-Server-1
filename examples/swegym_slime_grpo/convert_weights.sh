#!/usr/bin/env bash
# Convert Qwen3.5-4B HF weights to Megatron torch_dist format for Slime training.
# Qwen3.5-4B is a VLM checkpoint (Qwen3_5ForConditionalGeneration) with hybrid
# attention (1 full + 3 GatedDeltaNet linear per 4 layers).  Weight loading goes
# through slime_plugins.mbridge.qwen3_5 (text_config-aware).
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
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
mkdir -p "$OUTPUT_DIR"

# Mirrors slime/slime/scripts/models/qwen3.5-4B.sh.
# --spec installs the hybrid (GatedDeltaNet + full) attention layer layout.
# tie_word_embeddings=true in the HF config → do NOT pass --untie-embeddings-and-output-weights.
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
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --vocab-size 248320
    --rotary-base 10000000
)

echo "Converting ${HF_CHECKPOINT} -> ${OUTPUT_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
NCCL_P2P_DISABLE=1 \
NCCL_SHM_DISABLE=1 \
TORCHELASTIC_ERROR_FILE="${OUTPUT_DIR}/error.json" \
torchrun --nproc_per_node 1 --redirects 3 --log-dir "${OUTPUT_DIR}/logs" \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$OUTPUT_DIR" \
    --transformer-impl local \
    --no-persist-layer-norm \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion

echo "Done: ${OUTPUT_DIR}"
