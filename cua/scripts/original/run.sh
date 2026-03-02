#!/bin/bash
# ============================================================================
# CUA Data Collection - Multi-Actor Launcher
# ============================================================================
# Orchestrates 1 planner + N actor nodes:
#   1. Submits planner sbatch job (Qwen3-VL-235B vLLM)
#   2. Waits for planner to write its hostname to a coordination file
#   3. Launches NUM_ACTORS instances of run_actor_and_vm.sh in parallel
#   4. Each actor submits its own holder job on a reserved node
#   5. Waits for all actors to finish, then cancels planner
#
# Usage:
#   bash run.sh                                         # 1 actor (default)
#   NUM_ACTORS=4 bash run.sh
# ============================================================================

export LOG_DIR="${LOG_DIR:-./logs}"

# Configurable parameters
NUM_ACTORS="${NUM_ACTORS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"

# Create logs directory
mkdir -p "$LOG_DIR"

# Generate unique coordination file
COORD_ID="$(date +%Y%m%d_%H%M%S)_$$"
COORD_FILE="$LOG_DIR/.planner_host_${COORD_ID}"

PLANNER_JOB_ID=""
ACTOR_PIDS=()

echo "============================================"
echo "CUA Data Collection - Multi-Actor Launcher"
echo "============================================"
echo "NUM_ACTORS:        $NUM_ACTORS"
echo "MAX_PARALLEL:      $MAX_PARALLEL (per actor)"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES (per actor)"
echo "Coordination file:  $COORD_FILE"
echo ""


# --- Cleanup: cancel planner on exit ---
cleanup() {
    echo ""
    echo "[run.sh] Cleaning up..."

    # 1. Cancel Planner
    if [ -n "$PLANNER_JOB_ID" ]; then
        echo "[run.sh] Cancelling planner job $PLANNER_JOB_ID"
        scancel "$PLANNER_JOB_ID" 2>/dev/null
    fi

    # 2. Kill Actor Launcher Scripts
    # Sending SIGTERM to these scripts will trigger their own 'trap cleanup EXIT',
    # which in turn runs 'scancel' on the specific actor Slurm job.
    if [ ${#ACTOR_PIDS[@]} -gt 0 ]; then
        echo "[run.sh] Killing ${#ACTOR_PIDS[@]} actor launcher scripts..."
        for pid in "${ACTOR_PIDS[@]}"; do
            # Check if process is still running before killing
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null
            fi
        done
    fi

    # 3. Remove temp file
    rm -f "$COORD_FILE"
}
trap cleanup EXIT

# --- 1. Submit Planner Job ---
echo "[run.sh] Submitting planner job..."
PLANNER_JOB_ID=$(sbatch \
    --output="$LOG_DIR/planner-%j.out" \
    --export=ALL,COORD_FILE="$COORD_FILE" \
    --parsable \
    "./run_planner.sbatch")
echo "[run.sh] Planner job submitted: $PLANNER_JOB_ID"

# --- 2. Wait for planner hostname ---
echo "[run.sh] Waiting for planner to start and write hostname..."
PLANNER_NODE=""
COORD_ELAPSED=0
MAX_COORD_WAIT=7200  # 2 hours

while [ $COORD_ELAPSED -lt $MAX_COORD_WAIT ]; do
    if [ -f "$COORD_FILE" ]; then
        PLANNER_NODE=$(cat "$COORD_FILE" | tr -d '[:space:]')
        if [ -n "$PLANNER_NODE" ]; then
            echo "[run.sh] Planner node discovered: $PLANNER_NODE"
            break
        fi
    fi
    sleep 10
    COORD_ELAPSED=$((COORD_ELAPSED + 10))
    if [ $((COORD_ELAPSED % 60)) -eq 0 ]; then
        echo "[run.sh] Still waiting for planner (${COORD_ELAPSED}s)..."
    fi
done

if [ -z "$PLANNER_NODE" ]; then
    echo "[run.sh] ERROR: Planner did not start within ${MAX_COORD_WAIT}s."
    exit 1
fi

# --- 3. Launch N Actor Instances ---
echo "[run.sh] Launching $NUM_ACTORS actor(s)..."
ACTOR_PIDS=()

for i in $(seq 1 "$NUM_ACTORS"); do
    echo "[run.sh] Starting actor $i..."
    # Define log name here so we can echo it correctly below
    CURRENT_LOG="$LOG_DIR/collector-${PLANNER_JOB_ID}-${i}.out"

    PLANNER_NODE="$PLANNER_NODE" \
    PLANNER_JOB_ID="$PLANNER_JOB_ID" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    MAX_TRAJECTORIES="$MAX_TRAJECTORIES" \
        bash "./run_actor_and_vm.sh" "$i" &> "$CURRENT_LOG" &

    ACTOR_PIDS+=($!)
    echo "[run.sh] Actor $i launched (PID ${ACTOR_PIDS[-1]})"
    echo "         Log: $CURRENT_LOG"
done

# --- 4. Wait for all actors ---
echo ""
echo "[run.sh] All actors launched. Waiting for completion..."
echo ""

FAILED=0
for i in "${!ACTOR_PIDS[@]}"; do
    ACTOR_NUM=$((i + 1))
    wait "${ACTOR_PIDS[$i]}" 2>/dev/null
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[run.sh] Actor $ACTOR_NUM finished successfully."
    else
        echo "[run.sh] Actor $ACTOR_NUM failed (exit code $EXIT_CODE)."
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "[run.sh] All actors finished. $FAILED/$NUM_ACTORS failed."
echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
