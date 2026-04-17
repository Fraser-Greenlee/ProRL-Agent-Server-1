#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on the 10-task SWE-Gym sample via Polar + Slime.
#
# GPU layout (8x B200):
#   GPU 0-3     – SGLang inference (4 engines, managed by Slime/Ray, weight-synced)
#   GPU 4-7     – Megatron GRPO training (TP=2, DP=2)
#
# Port layout:
#   9000        – SGLang router (slime-managed, load-balances 4 engines)
#   8080        – Polar rollout server (task coordinator)
#   8100        – Polar gateway node (dispatches agent sessions)
#   8265        – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Slime manages SGLang engines; Polar gateway proxies LLM calls to them.
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
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3-4B}"
TORCH_DIST_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/checkpoints/Qwen3-4B_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/checkpoints/swegym_slime_grpo}"
mkdir -p "$SAVE_DIR"

# Qwen3-4B architecture (from HF config.json)
MODEL_ARGS=(
    --swiglu
    --num-layers 36
    --hidden-size 2560
    --ffn-hidden-size 9728
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
)

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${SCRIPT_DIR}/swegym_10_tasks.jsonl"
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

# ── Step 2: Convert weights if needed ──────────────────────────────
if [ ! -d "${TORCH_DIST_DIR}/release" ]; then
    echo "=== Converting HF weights ==="
    bash "${SCRIPT_DIR}/convert_weights.sh"
fi

# ── Step 3: Ray + Slime (manages SGLang engines + training) ───────
echo "=== Starting Ray (all 8 GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

# 10 tasks, rollout_batch_size=2, n_samples_per_prompt=16
# → 5 training steps, each task seen exactly once
# → 32 samples per step (2 prompts × 16 rollouts)
# → global_batch_size=32 (must be divisible by DP=2 → 16 per rank)
echo "=== Launching train_async.py ==="
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --rollout-num-gpus-per-engine 1 \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$TORCH_DIST_DIR" \
    --load "$SAVE_DIR" \
    --save "$SAVE_DIR" \
    --save-interval 5 \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-config-path "${SCRIPT_DIR}/polar_config.yaml" \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-rollout 5 \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 16 \
    --rollout-max-response-len 8192 \
    --rollout-max-prompt-len 4096 \
    --global-batch-size 32 \
    --rollout-global-dataset \
    --disable-rollout-trim-samples \
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
    --max-tokens-per-gpu 8192 \
    --advantage-estimator external \
    --use-rollout-logprobs \
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
    --attention-backend flash \
    --no-gradient-accumulation-fusion \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}" \
    --wandb-exp-name "${WANDB_RUN_NAME:-swegym-async-grpo}" \
    --wandb-group "${WANDB_GROUP:-swegym-async}" \
    --sglang-router-ip 0.0.0.0 \
    --sglang-router-port 9000
