#!/bin/bash
# ============================================================================
# Kimi-K2.5 Data Collection - Multi-Collector Launcher
# ============================================================================
# Orchestrates Kimi vLLM server + N collector (reserved CPU) nodes:
#   1. Submits Kimi vLLM sbatch job (2 GPU nodes, Ray cluster)
#   2. Waits for Kimi server to become healthy
#   3. Launches NUM_COLLECTORS instances of run_collector_kimi.sh in parallel
#   4. Each collector submits its own holder job on a /dev/kvm node
#   5. Waits for all collectors to finish, then cancels Kimi server
#
# Usage:
#   NUM_COLLECTORS=2 bash run_parallel_kimi.sh
#
# Log files (e.g. when using 2 collectors):
#  scripts/kimi/logs/
#  ├── slurm-server-<jobid>.out        # Kimi vLLM server (sbatch %j → same jobid)
#  ├── slurm-<jobid>-collector-1.out        # Collector 1
#  └── slurm-<jobid>-collector-2.out        # Collector 2
# ============================================================================

export LOG_DIR="${LOG_DIR:-./logs}"

# Configurable parameters
NUM_COLLECTORS="${NUM_COLLECTORS:-2}"
GENERATION_MODE="${GENERATION_MODE:-zenodo}"  # todo change this - vanilla, spreadsheetbench, zenodo
MAX_PARALLEL="${MAX_PARALLEL:-10}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/trajectories/kimi_$GENERATION_MODE}"


# Create logs directory
mkdir -p "$LOG_DIR"

KIMI_JOB_ID=""
COLLECTOR_PIDS=()
KIMI_PORT=8000

echo "============================================"
echo "Kimi-K2.5 Data Collection Launcher"
echo "============================================"
echo "NUM_COLLECTORS:    $NUM_COLLECTORS"
echo "MAX_PARALLEL:      $MAX_PARALLEL (per collector)"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES (per collector)"
echo ""


# --- Cleanup: cancel Kimi server on exit ---
cleanup() {
    echo ""
    echo "[run_parallel_kimi.sh] Cleaning up..."

    # 1. Kill Collector Launcher Scripts
    if [ ${#COLLECTOR_PIDS[@]} -gt 0 ]; then
        echo "[run_parallel_kimi.sh] Killing ${#COLLECTOR_PIDS[@]} collector launcher scripts..."
        for pid in "${COLLECTOR_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
    fi

    # 2. Cancel Kimi vLLM server
    if [ -n "$KIMI_JOB_ID" ]; then
        echo "[colocated] Cancelling Kimi vLLM job $KIMI_JOB_ID"
        SCANCEL_OUTPUT=$(scancel "$KIMI_JOB_ID" 2>&1)
        SCANCEL_EXIT=$?
        if [ $SCANCEL_EXIT -ne 0 ]; then
            echo "[colocated] WARNING: scancel failed (exit $SCANCEL_EXIT): $SCANCEL_OUTPUT"
        else
            echo "[colocated] scancel succeeded for job $KIMI_JOB_ID"
        fi
    fi

    # 3. Remove head node file
    rm -f "$LOG_DIR/head_node_${KIMI_JOB_ID}"
}
trap cleanup EXIT


# --- 1. Submit Kimi vLLM server ---
echo "[run_parallel_kimi.sh] Submitting Kimi vLLM sbatch job..."
KIMI_JOB_ID=$(sbatch \
    --account=llmservice_fm_vision \
    --partition=batch_short \
    --time=02:00:00 \
    --output="$LOG_DIR/slurm-%j-server.out" \
    --error="$LOG_DIR/slurm-%j-server.out" \
    --parsable \
    "./run_kimi.sbatch")

if [ -z "$KIMI_JOB_ID" ]; then
    echo "[run_parallel_kimi.sh] ERROR: Kimi sbatch submission failed."
    exit 1
fi
echo "[run_parallel_kimi.sh] Kimi vLLM job submitted: $KIMI_JOB_ID"

# Wait for the job to start and discover the head node (written by run_kimi.sbatch)
HEAD_NODE_FILE="$LOG_DIR/head_node_${KIMI_JOB_ID}"
echo "[run_parallel_kimi.sh] Waiting for head node file: $HEAD_NODE_FILE"
MODEL_NODE=""
ELAPSED=0
MAX_WAIT=43200  # 12 hours

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if job is still alive
    JOB_STATE=$(squeue -j "$KIMI_JOB_ID" -h -o %T 2>/dev/null)
    if [ -z "$JOB_STATE" ]; then
        echo "[run_parallel_kimi.sh] ERROR: Kimi job $KIMI_JOB_ID disappeared from queue!"
        exit 1
    fi

    # Check for head node file (written by run_kimi.sbatch once it starts)
    if [ -f "$HEAD_NODE_FILE" ]; then
        MODEL_NODE=$(cat "$HEAD_NODE_FILE")
        if [ -n "$MODEL_NODE" ]; then
            echo "[run_parallel_kimi.sh] Kimi vLLM job running. Head node: $MODEL_NODE"
            break
        fi
    fi

    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[run_parallel_kimi.sh] Still waiting for Kimi job to start (${ELAPSED}s)..."
    fi
done

if [ -z "$MODEL_NODE" ]; then
    echo "[run_parallel_kimi.sh] ERROR: Kimi vLLM did not start within ${MAX_WAIT}s."
    exit 1
fi

# --- 2. Wait for Kimi vLLM health ---
echo "[run_parallel_kimi.sh] Waiting for Kimi vLLM health at $MODEL_NODE:$KIMI_PORT..."
ELAPSED=0
MAX_HEALTH_WAIT=7200  # Model loading can take 90+ min

while [ $ELAPSED -lt $MAX_HEALTH_WAIT ]; do
    if curl -sf "http://$MODEL_NODE:$KIMI_PORT/health" > /dev/null 2>&1; then
        echo "[run_parallel_kimi.sh] Kimi vLLM is healthy!"
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "[run_parallel_kimi.sh] Still waiting for Kimi health (${ELAPSED}s)..."
    fi
done

if [ $ELAPSED -ge $MAX_HEALTH_WAIT ]; then
    echo "[run_parallel_kimi.sh] ERROR: Kimi vLLM did not become healthy within ${MAX_HEALTH_WAIT}s."
    exit 1
fi

# --- 3. Launch N Collector Instances ---
echo "[run_parallel_kimi.sh] Launching $NUM_COLLECTORS collector(s)..."
export MODEL_NODE
COLLECTOR_PIDS=()

for i in $(seq 1 "$NUM_COLLECTORS"); do
    echo "[run_parallel_kimi.sh] Starting collector $i..."
    CURRENT_LOG="$LOG_DIR/slurm-${KIMI_JOB_ID}-collector-${i}.out"

    MODEL_NODE="$MODEL_NODE" \
    GENERATION_MODE="$GENERATION_MODE" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    MAX_TRAJECTORIES="$MAX_TRAJECTORIES" \
    TRAJECTORY_SAVE_DIR="$TRAJECTORY_SAVE_DIR" \
        bash "./run_collector_kimi.sh" "$i" &> "$CURRENT_LOG" &

    COLLECTOR_PIDS+=($!)
    echo "[run_parallel_kimi.sh] Collector $i launched (PID ${COLLECTOR_PIDS[-1]})"
    echo "                       Log: $CURRENT_LOG"
done

# --- 4. Wait for all collectors ---
echo ""
echo "[run_parallel_kimi.sh] All collectors launched. Waiting for completion..."
echo ""

FAILED=0
for i in "${!COLLECTOR_PIDS[@]}"; do
    COLLECTOR_NUM=$((i + 1))
    wait "${COLLECTOR_PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[run_parallel_kimi.sh] Collector $COLLECTOR_NUM finished successfully."
    else
        echo "[run_parallel_kimi.sh] Collector $COLLECTOR_NUM failed (exit code $EXIT_CODE)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "[run_parallel_kimi.sh] All collectors finished. $FAILED/$NUM_COLLECTORS failed."
echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
