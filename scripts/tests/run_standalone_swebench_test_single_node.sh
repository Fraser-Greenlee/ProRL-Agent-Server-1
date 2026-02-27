#!/bin/bash
# Single-node standalone SWE-bench evaluation script.
# Usage: bash run_standalone_swebench_test_single_node.sh
#
# This script launches OpenHands + vLLM servers locally, then runs evaluation.
# All background processes are cleaned up on exit (Ctrl-C or natural completion).

set -x  # Enable debug output

# ==================== Configuration ====================
HOME_HAOZH='/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh'
OPENHANDS_WORKDIR=$HOME_HAOZH/projects/OpenHands_internal
LOG_DIR="${OPENHANDS_WORKDIR}/logs/standalone_test_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OPENHANDS_WORKDIR}/results/standalone_test_$(date +%Y%m%d_%H%M%S)"

# Model configuration
SFT_MODEL_PATH='/lustre/fsw/portfolios/llmservice/users/haozh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5'
TOKENIZER_PATH='/lustre/fsw/portfolios/llmservice/users/haozh/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5'

# Data configuration
DATA_PATH=$HOME_HAOZH/data/SWE-GYM-R2E-GYM/test-transformed-with-prompt-first-64.parquet

# Server configuration
GPUS_PER_NODE=8
TP_SIZE=4
GPU_MEM_UTIL=0.8
NUM_SERVERS=$((GPUS_PER_NODE / TP_SIZE))
VLLM_BASE_PORT=8100
OPENHANDS_PORT=8006
OPENHANDS_NUM_WORKERS=64

# Evaluation configuration
NUM_TRAJECTORIES=1
TEMPERATURE=0.0
TOP_P=1.0
MAX_ITERATIONS=50
MAX_OUTPUT_TOKENS=1536
MAX_MODEL_LEN=32768
TIMEOUT=1500
HINT_MODE=none
TOKEN_LEVEL_GENERATION=true  # set to true for token-level generation

# ==================== Cleanup trap ====================
PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done
    echo "Cleanup done."
}

trap cleanup EXIT INT TERM

# ==================== Setup ====================
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

NODE_IP=$(hostname --ip-address)
# Convert to ipv4 if needed
if [[ "$NODE_IP" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$NODE_IP"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        NODE_IP=${ADDR[1]}
    else
        NODE_IP=${ADDR[0]}
    fi
fi
echo "Node IP: $NODE_IP"

# ==================== Start OpenHands server ====================
echo "Starting OpenHands server..."

cd "$OPENHANDS_WORKDIR"
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=$HOME_HAOZH/singularity_images_v3
export OVERWRITE_OPENHANDS_DIR="$OPENHANDS_WORKDIR"
export PYTHONPATH="${OPENHANDS_WORKDIR}:${PYTHONPATH}"
export LOG_LEVEL=ERROR
export DEBUG=False

python scripts/start_server_thread.py \
    --max-init-workers 70 \
    --max-run-workers "$OPENHANDS_NUM_WORKERS" \
    --timeout 9999999 \
    > "$LOG_DIR/openhands.out" 2> "$LOG_DIR/openhands.err" &
PIDS+=($!)
echo "OpenHands server PID: ${PIDS[-1]}"

openhands_urls="http://${NODE_IP}:${OPENHANDS_PORT}"
echo "OpenHands URL: $openhands_urls"

cd "$WORKDIR"

# ==================== Start vLLM servers ====================
echo "Starting $NUM_SERVERS vLLM server(s)..."
llm_server_urls=""

for server_idx in $(seq 0 $((NUM_SERVERS - 1))); do
    gpu_start=$((server_idx * TP_SIZE))
    gpu_end=$((gpu_start + TP_SIZE - 1))
    cuda_devices=$(seq -s, "$gpu_start" "$gpu_end")
    port=$((VLLM_BASE_PORT + server_idx))

    echo "  Server $server_idx: GPUs=$cuda_devices, port=$port"

    if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
        CUDA_VISIBLE_DEVICES=$cuda_devices python "$OPENHANDS_WORKDIR/scripts/tests/vllm_api_server.py" \
            --model "$SFT_MODEL_PATH" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$port" \
            --host 0.0.0.0 \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            > "$LOG_DIR/vllm_server_${server_idx}.out" 2> "$LOG_DIR/vllm_server_${server_idx}.err" &
    else
        CUDA_VISIBLE_DEVICES=$cuda_devices python -m vllm.entrypoints.openai.api_server \
            --model "$SFT_MODEL_PATH" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$port" \
            --host 0.0.0.0 \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            > "$LOG_DIR/vllm_server_${server_idx}.out" 2> "$LOG_DIR/vllm_server_${server_idx}.err" &
    fi
    PIDS+=($!)
    echo "  vLLM server $server_idx PID: ${PIDS[-1]}"

    if [ -z "$llm_server_urls" ]; then
        llm_server_urls="http://${NODE_IP}:${port}"
    else
        llm_server_urls="${llm_server_urls}+http://${NODE_IP}:${port}"
    fi
done

echo "LLM Server URLs: $llm_server_urls"

# ==================== Wait for vLLM servers to be healthy ====================
echo "Waiting for vLLM servers to become healthy..."
IFS='+' read -ra LLM_URLS <<< "$llm_server_urls"
all_healthy=true
for url in "${LLM_URLS[@]}"; do
    healthy=false
    for attempt in $(seq 1 120); do
        if curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null | grep -q "200"; then
            echo "  vLLM server $url is healthy (attempt $attempt)"
            healthy=true
            break
        fi
        sleep 5
    done
    if [ "$healthy" = false ]; then
        echo "ERROR: vLLM server $url did not become healthy after 10 minutes"
        echo "  Check logs: $LOG_DIR/vllm_server_*.err"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo "Some vLLM servers failed to start. Aborting."
    exit 1
fi

echo "All vLLM servers are healthy."
echo ""
echo "Servers ready. ProRL/OpenHands: ${openhands_urls}  |  vLLM: ${llm_server_urls}"
echo "Press Ctrl-C to stop all servers."
echo ""

# Keep script alive so background servers stay running.
# Run your Gym commands (ng_run, ng_collect_rollouts, etc.) in another terminal.
wait

# ==================== Run evaluation ====================
echo ""
echo "=========================================="
echo "Starting standalone SWE-bench evaluation"
echo "=========================================="
echo "  OpenHands URLs:   $openhands_urls"
echo "  LLM Server URLs:  $llm_server_urls"
echo "  Data:             $DATA_PATH"
echo "  Output:           $OUTPUT_DIR"
echo "  Logs:             $LOG_DIR"
echo "=========================================="

TOKEN_LEVEL_FLAG=""
if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
    TOKEN_LEVEL_FLAG="--token_level_generation"
fi

cd "$OPENHANDS_WORKDIR"
export PYTHONPATH="${OPENHANDS_WORKDIR}:${PYTHONPATH}"

python scripts/tests/standalone_swebench_test.py \
    --data_path "$DATA_PATH" \
    --openhands_urls "$openhands_urls" \
    --llm_server_urls "$llm_server_urls" \
    --model_name "$SFT_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --num_workers_per_server "$OPENHANDS_NUM_WORKERS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_output_tokens "$MAX_OUTPUT_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --timeout "$TIMEOUT" \
    --hint_mode "$HINT_MODE" \
    --custom_tokenizer "$TOKENIZER_PATH" \
    $TOKEN_LEVEL_FLAG

echo ""
echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
echo "Logs available at: $LOG_DIR"
