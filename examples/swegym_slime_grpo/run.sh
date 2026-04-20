#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on the SWE-Gym sample via Polar + Slime (Qwen3.5-27B).
#
# GPU layout (8x B200):
#   GPU 0-3     – SGLang inference (2 engines × TP=2, managed by Slime/Ray)
#   GPU 4-7     – Megatron GRPO training (TP=4, DP=1, bridge mode)
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
# Qwen3.5-27B (bridge mode — HF weights consumed directly at train time,
# no torch_dist conversion). Text-only path: SGLANG_LANGUAGE_ONLY + the
# SLIME_QWEN35_TEXT_ONLY_BRIDGE env vars strip the vision stack.
HF_CHECKPOINT="${HF_CHECKPOINT:-/raid/binfeng/checkpoints/Qwen3.5-27B}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/ckpt_swegym_slime_grpo_27b}"
mkdir -p "$SAVE_DIR"
if [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3.5-27B --local-dir $HF_CHECKPOINT"
    exit 1
fi

# Mirrors slime/scripts/models/qwen3.5-27B.sh — relies on the slime Megatron
# patch for --use-gated-attention and the qwen3_5 spec plugin.
MODEL_ARGS=(
    --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 24
    --num-query-groups 4
    --kv-channels 256
    --num-layers 64
    --hidden-size 5120
    --ffn-hidden-size 17408
    --use-gated-attention
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# Pick HF path until the first save lands.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$HF_CHECKPOINT"
fi

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${SCRIPT_DIR}/swegym_30_tasks.jsonl"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "Preparing training data..."
    python "${SCRIPT_DIR}/prepare_data.py"
fi

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
polar serve_rollout -c "${SCRIPT_DIR}/topology.yaml" &
PIDS+=($!)
sleep 2

echo "=== Starting Polar gateway (:8100) ==="
polar serve_gateway -c "${SCRIPT_DIR}/topology.yaml" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2

curl -sf http://127.0.0.1:8080/health || { echo "Polar rollout server not healthy"; exit 1; }

# ── Step 2: Ray + Slime (manages SGLang engines + training) ───────
# Bridge mode skips the torch_dist conversion — weights are loaded from
# $HF_CHECKPOINT directly at train time.
echo "=== Starting Ray (all 8 GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

CUDNN_LIB="${PROJECT_ROOT}/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"LD_LIBRARY_PATH\": \"${CUDNN_LIB}:${LD_LIBRARY_PATH:-}\",
    \"SGLANG_LANGUAGE_ONLY\": \"1\",
    \"SLIME_QWEN35_TEXT_ONLY_BRIDGE\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\"
  }
}"

# Rollout sizing: 2 prompts × 32 trajectories = 64 trajectories/rollout.
# With --dynamic-history each trajectory explodes into one sample per
# trace, so sample count per rollout is variable (~hundreds–thousands).
# --num-steps-per-rollout targets 1 training step per rollout; slime
# derives global_batch_size from the realized sample count.
# 30 rollouts × 2 prompts = 60 task exposures across 30 unique tasks (~2x each).
echo "=== Launching train_async.py ==="
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 2 \
    "${MODEL_ARGS[@]}" \
    --megatron-to-hf-mode bridge \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$HF_CHECKPOINT" \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --save-interval 5 \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${SCRIPT_DIR}/polar_config.yaml" \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-rollout 30 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 32 \
    --rollout-max-response-len 8192 \
    --rollout-max-prompt-len 24000 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 4 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 262144 \
    --log-probs-chunk-size 512 \
    --advantage-estimator grpo \
    --grpo-std-normalization \
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
    --optimizer-cpu-offload \
    --overlap-cpu-optimizer-d2h-h2d \
    --use-precision-aware-optimizer \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-language-only \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}" \
    --wandb-group "${WANDB_GROUP:-swegym-qwen35-27b-async-grpo}" \
    --sglang-router-port 9000 \
