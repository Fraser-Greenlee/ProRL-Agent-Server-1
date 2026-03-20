#!/bin/bash
# ============================================================================
# Kimi NVCF Collector — Interactive (run directly on a CPU node)
# ============================================================================
# Runs data collection directly on the current node. No Slurm holder job,
# no SSH+enroot — just straight Python execution.
#
# Assumes you are already on a CPU node (e.g., via srun --pty or salloc).
#
# Required env vars (or set in cua/.env):
#   MODEL_NODE        - hostname of the Kimi vLLM server head node
#   NGC_API_KEY       - NVCF API key
#   NGC_ORG           - NVCF organization
#
# Optional env vars:
#   MAX_PARALLEL      - parallel VMs per collector (default: 10)
#   MAX_TRAJECTORIES  - trajectories to collect (default: 10000)
#   TRAJECTORY_SAVE_DIR - output directory
#
# Usage:
#   MODEL_NODE=pool0-04195 bash run_interactive_collector_kimi_nvcf.sh
# ============================================================================

set -e

MODEL_NODE=pool0-04195

# Load .env as defaults (won't override existing env vars)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        if [ -z "${!key+x}" ]; then
            export "$key=$value"
        fi
    done < "$ENV_FILE"
fi

# Configs
PROJECT_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2"
PROJECT_DIR="$PROJECT_ROOT/cua"

MAX_PARALLEL=${MAX_PARALLEL:-10}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-10000}
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-$PROJECT_DIR/trajectories/kimi_nvcf}"
KIMI_PORT=${KIMI_PORT:-8000}
export NVCF_FUNCTION_NAME_PREFIX="${NVCF_FUNCTION_NAME_PREFIX:-data-collection}"
export OSWORLD_SETUP_CACHE_DIR="${OSWORLD_SETUP_CACHE_DIR:-/tmp/osworld_cache}"
export PYTHONUNBUFFERED=1

# Validate
if [ -z "$MODEL_NODE" ]; then
    echo "ERROR: MODEL_NODE not set."
    exit 1
fi
if [ -z "$NGC_API_KEY" ]; then
    echo "ERROR: NGC_API_KEY not set."
    exit 1
fi
if [ -z "$NGC_ORG" ]; then
    echo "ERROR: NGC_ORG not set."
    exit 1
fi

echo "MODEL_NODE=$MODEL_NODE"
echo "MAX_PARALLEL=$MAX_PARALLEL MAX_TRAJECTORIES=$MAX_TRAJECTORIES"
echo "TRAJECTORY_SAVE_DIR=$TRAJECTORY_SAVE_DIR"
echo "Runtime: nvcf"

# Wait for Kimi vLLM to be healthy
echo "Waiting for Kimi vLLM at $MODEL_NODE:$KIMI_PORT..."
ELAPSED=0
MAX_WAIT=7200
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "http://$MODEL_NODE:$KIMI_PORT/health" > /dev/null 2>&1; then
        echo "Kimi vLLM is healthy!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "Still waiting for Kimi vLLM (${ELAPSED}s)..."
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Kimi vLLM did not become healthy within ${MAX_WAIT}s"
    exit 1
fi

# Run data collection
echo "Starting parallel data collection (NVCF backend)..."
cd "$PROJECT_DIR"
python parallel_collect_kimi.py \
    --model_node "$MODEL_NODE" \
    --runtime nvcf \
    --max_parallel "$MAX_PARALLEL" \
    --max_trajectories "$MAX_TRAJECTORIES" \
    --trajectory_save_dir "$TRAJECTORY_SAVE_DIR"
