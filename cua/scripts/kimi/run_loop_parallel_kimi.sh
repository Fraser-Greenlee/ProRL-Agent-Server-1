#!/bin/bash
# ===============================================================================
# Outer loop: runs NUM_INSTANCES concurrent run_parallel_kimi.sh pipelines
# per round, for NUM_ROUNDS rounds with a cooldown in between.
#
# Each instance submits its own vLLM server (2 GPU nodes) + NUM_COLLECTORS
# CPU collector nodes. Total GPU nodes per round: NUM_INSTANCES * 2.
#
# Usage:
#   bash run_loop_parallel_kimi.sh
#   NUM_INSTANCES=4 NUM_ROUNDS=20 COOLDOWN_SECONDS=600 bash run_loop_parallel_kimi.sh
# ===============================================================================

NUM_INSTANCES="${NUM_INSTANCES:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-40}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-600}"  # 10 minutes
ROUND_TIMEOUT="${ROUND_TIMEOUT:-14400}"      # 4 hours — timeout for one round of NUM_INSTANCES runs
NUM_COLLECTORS="${NUM_COLLECTORS:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-10000}"

PIDS=()

cleanup() {
    echo ""
    echo "[loop] Cleaning up ${#PIDS[@]} background instance(s)..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    wait 2>/dev/null
}
trap cleanup EXIT

echo "============================================"
echo "Kimi-K2.5 Parallel Loop"
echo "============================================"
echo "NUM_INSTANCES:     $NUM_INSTANCES"
echo "NUM_ROUNDS:        $NUM_ROUNDS"
echo "COOLDOWN:          ${COOLDOWN_SECONDS}s"
echo "ROUND_TIMEOUT:     ${ROUND_TIMEOUT}s"
echo "NUM_COLLECTORS:    $NUM_COLLECTORS (per instance)"
echo "MAX_PARALLEL:      $MAX_PARALLEL"
echo "MAX_TRAJECTORIES:  $MAX_TRAJECTORIES"
echo ""

for round in $(seq 1 "$NUM_ROUNDS"); do
    echo ""
    echo "================================================================"
    echo "  Round $round / $NUM_ROUNDS — launching $NUM_INSTANCES instance(s)"
    echo "================================================================"

    PIDS=()
    for inst in $(seq 1 "$NUM_INSTANCES"); do
        echo "[loop] Starting instance $inst..."
        NUM_COLLECTORS="$NUM_COLLECTORS" \
        MAX_PARALLEL="$MAX_PARALLEL" \
        MAX_TRAJECTORIES="$MAX_TRAJECTORIES" \
            bash run_parallel_kimi.sh &
        PIDS+=($!)
    done

    echo "[loop] Waiting for instances (timeout: ${ROUND_TIMEOUT}s)..."
    ELAPSED=0
    while [ $ELAPSED -lt "$ROUND_TIMEOUT" ]; do
        ALL_DONE=true
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ALL_DONE=false
                break
            fi
        done
        if $ALL_DONE; then break; fi
        sleep 10
        ELAPSED=$((ELAPSED + 10))
    done

    # Kill any stragglers that exceeded the timeout
    ROUND_FAILED=0
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            echo "[loop] Instance $((i+1)) still running after ${ROUND_TIMEOUT}s — killing."
            kill "${PIDS[$i]}" 2>/dev/null
            wait "${PIDS[$i]}" 2>/dev/null
            ROUND_FAILED=$((ROUND_FAILED + 1))
        else
            wait "${PIDS[$i]}" 2>/dev/null
            if [ $? -ne 0 ]; then
                ROUND_FAILED=$((ROUND_FAILED + 1))
            fi
        fi
    done
    echo "[loop] Round $round done. $ROUND_FAILED/$NUM_INSTANCES failed."

    if [ "$round" -lt "$NUM_ROUNDS" ]; then
        echo "[loop] Cooldown: ${COOLDOWN_SECONDS}s..."
        sleep "$COOLDOWN_SECONDS"
    fi
done

echo ""
echo "[loop] All $NUM_ROUNDS rounds complete."
