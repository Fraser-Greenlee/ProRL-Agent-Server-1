#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on the SWE-Gym sample via Polar + Slime (Qwen3-4B-Instruct-2507).
#
# GPU layout (8x B200):
#   GPU 0-3     – SGLang inference (4 engines × TP=1, managed by Slime/Ray)
#   GPU 4-7     – Megatron GRPO training (TP=2, DP=2)
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
# Qwen3-4B-Instruct-2507: plain text Qwen3 architecture, already converted to
# torch_dist at checkpoints/Qwen3-4B-Instruct-2507_torch_dist/ — so we load via
# standard (non-bridge) Megatron path.
HF_CHECKPOINT="${HF_CHECKPOINT:-/home/nfs/binfengx/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/checkpoints/Qwen3-4B-Instruct-2507_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/ckpt_swegym_slime_grpo_4b}"
mkdir -p "$SAVE_DIR"
if [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3-4B-Instruct-2507"
    exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run a torch_dist conversion against $HF_CHECKPOINT first."
    exit 1
fi

# Mirrors slime/scripts/models/qwen3-4B-Instruct-2507.sh (rotary_base=5M).
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
    --rotary-base 5000000
    --vocab-size 151936
    --kv-channels 128
    --qk-layernorm
)

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# Pick REF_LOAD (torch_dist) until the first save lands.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
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
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\"
  }
}"

# Rollout sizing: 4 prompts × 8 trajectories = 32 trajectories/rollout.
# With --dynamic-history each trajectory explodes into one sample per
# trace, so sample count per rollout is variable (~hundreds).
# --num-steps-per-rollout targets 1 training step per rollout; slime
# derives global_batch_size from the realized sample count.
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
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --save-interval 10 \
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
    --rollout-batch-size 4 \
    --n-samples-per-prompt 8 \
    --rollout-max-response-len 4096 \
    --rollout-max-prompt-len 16000 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
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
    --max-tokens-per-gpu 32768 \
    --log-probs-chunk-size 512 \
    --advantage-estimator grpo \
    --normalize-advantages \
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
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-tool-call-parser qwen25 \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}" \
    --wandb-group "${WANDB_GROUP:-swegym-qwen3-4b-async-grpo}" \
    --sglang-router-port 9000
