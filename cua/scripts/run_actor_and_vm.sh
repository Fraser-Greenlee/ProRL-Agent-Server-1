#!/bin/bash
# ============================================================================
# Actor + VM Launcher (SSH+Enroot Pattern)
# ============================================================================
# Submits a holder "sleep infinity" job on a reserved GPU node, waits for
# the container to be ready, then SSH+enroot execs into it to run:
#   1. UI-TARS-1.5-7B vLLM server (background, TP=4)
#   2. Data collection via parallel_collect_trajectories.py
#
# Required env vars:
#   PLANNER_NODE  - hostname of the planner vLLM server
#
# Optional env vars:
#   MAX_PARALLEL      - parallel VMs per actor (default: 1)
#   MAX_TRAJECTORIES  - trajectories to collect (default: 10000)
#
# Optional arg:
#   $1 = actor index (for log naming, default: 0)
#
# Usage:
#   PLANNER_NODE=pool0-12345 bash run_actor_and_vm.sh 1
# ============================================================================

ACTOR_IDX="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs_single}"

# Logging Files
ACTOR_VLLM_LOG="$LOG_DIR/actor_${PLANNER_JOB_ID}-${ACTOR_IDX}.out"

# Configs
PROJECT_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server"
PROJECT_DIR="$PROJECT_ROOT/cua"
ACTOR_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua-vllm-0.13.0.sqsh"
ACTOR_MODEL="ByteDance-Seed/UI-TARS-1.5-7B"

MAX_PARALLEL=${MAX_PARALLEL:-1}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-10000}

PLANNER_PORT=8000
ACTOR_PORT=8000

# Validate
if [ -z "$PLANNER_NODE" ]; then
    echo "[Actor $ACTOR_IDX] ERROR: PLANNER_NODE not set."
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "[Actor $ACTOR_IDX] PLANNER_NODE=$PLANNER_NODE"
echo "[Actor $ACTOR_IDX] MAX_PARALLEL=$MAX_PARALLEL MAX_TRAJECTORIES=$MAX_TRAJECTORIES"

# --- 1. Submit holder job on reserved node ---
echo "[Actor $ACTOR_IDX] Submitting holder job..."
ACTOR_JOB_ID=$(sbatch --parsable \
    --job-name="cua_actor_${ACTOR_IDX}" \
    --account=llmservice_fm_vision \
    --partition=batch_block1 \
    --reservation=sla_res_osworld_agent_vlm \
    --gpus-per-node=8 \
    --mem=0 \
    --time=04:00:00 \
    --exclusive \
    --output="/dev/null" \
    --error="/dev/null" \
    --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"disproportionate_resource_requirement","description":"Co-locating vLLM instance with kvm-accessible OS VM"}}' \
    --wrap="srun --container-image=$ACTOR_IMAGE --container-mounts=/lustre:/lustre sleep infinity")

if [ -z "$ACTOR_JOB_ID" ]; then
    echo "[Actor $ACTOR_IDX] ERROR: Job submission failed."
    exit 1
fi
echo "[Actor $ACTOR_IDX] Holder job submitted: $ACTOR_JOB_ID"

# --- 2. Cleanup trap ---
cleanup() {
    echo ""
    echo "[Actor $ACTOR_IDX] Cleaning up... Cancelling holder job $ACTOR_JOB_ID"
    scancel "$ACTOR_JOB_ID" 2>/dev/null
}
trap cleanup EXIT

# --- 3. Wait for job to start ---
echo "[Actor $ACTOR_IDX] Waiting for holder job to start..."
ACTOR_NODE=""
while [ -z "$ACTOR_NODE" ]; do
    JOB_STATE=$(squeue -j "$ACTOR_JOB_ID" -h -o %T 2>/dev/null)

    if [ "$JOB_STATE" == "RUNNING" ]; then
        ACTOR_NODE=$(squeue -j "$ACTOR_JOB_ID" -h -o %N)
    elif [ -z "$JOB_STATE" ]; then
        echo "[Actor $ACTOR_IDX] ERROR: Job $ACTOR_JOB_ID disappeared from queue!"
        exit 1
    fi
    sleep 2
done
echo "[Actor $ACTOR_IDX] Job RUNNING on node: $ACTOR_NODE"

# --- 4. Wait for container readiness ---
echo "[Actor $ACTOR_IDX] Polling for container readiness on $ACTOR_NODE..."
CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2
    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1" 2>/dev/null)

    if [ -z "$CONTAINER_PID" ]; then
        printf "."
    fi
done
echo ""
echo "[Actor $ACTOR_IDX] Container ready, PID: $CONTAINER_PID"

# --- 5. SSH+enroot exec: launch actor vLLM + data collection ---
ssh -t -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
    "enroot exec $CONTAINER_PID /bin/bash -c '
        set -e
        export PYTHONUNBUFFERED=1

        # 1. Start vLLM in background
        echo \"[Actor $ACTOR_IDX] Launching UI-TARS-1.5-7B vLLM...\"
        vllm serve $ACTOR_MODEL \
            --api-key gen \
            --tensor-parallel-size 4 \
            --limit-mm-per-prompt.image 5 \
            --limit-mm-per-prompt.video 0 \
            --max-model-len 65536 \
            --disable-log-requests \
            --disable-log-stats \
            > $ACTOR_VLLM_LOG 2>&1 &
        VLLM_PID=\$!

        # 2. Wait for planner and actor vLLM to be healthy
        echo \"[Actor $ACTOR_IDX] Waiting for planner at $PLANNER_NODE:$PLANNER_PORT...\"
        while ! nc -z $PLANNER_NODE $PLANNER_PORT 2>/dev/null; do
            echo \"[Actor $ACTOR_IDX] Waiting for planner node to accept connections...\"
            sleep 10
        done
        echo \"[Actor $ACTOR_IDX] Planner is accepting connections!\"

        echo \"[Actor $ACTOR_IDX] Waiting for actor vLLM health...\"
        ELAPSED=0
        MAX_WAIT=600
        while [ \$ELAPSED -lt \$MAX_WAIT ]; do
            if curl -sf http://localhost:$ACTOR_PORT/health > /dev/null 2>&1; then
                echo \"[Actor $ACTOR_IDX] Actor vLLM healthy!\"
                break
            fi
            sleep 10
            ELAPSED=\$((ELAPSED + 10))
            if [ \$((ELAPSED % 60)) -eq 0 ]; then
                echo \"[Actor $ACTOR_IDX] Still waiting for actor vLLM (\${ELAPSED}s)...\"
            fi
        done

        if [ \$ELAPSED -ge \$MAX_WAIT ]; then
            echo \"[Actor $ACTOR_IDX] ERROR: Actor vLLM did not start within \${MAX_WAIT}s\"
            kill \$VLLM_PID 2>/dev/null
            exit 1
        fi

        # 3. Data Collection -> stdout
        echo \"[Actor $ACTOR_IDX] Starting data collection...\"
        cd $PROJECT_DIR
        python parallel_collect_trajectories.py \
            --planner_node $PLANNER_NODE \
            --actor_node localhost \
            --max_parallel $MAX_PARALLEL \
            --max_trajectories $MAX_TRAJECTORIES

        COLLECT_EXIT=\$?
        echo \"[Actor $ACTOR_IDX] Data collection finished with exit code \$COLLECT_EXIT\"
        kill \$VLLM_PID 2>/dev/null
        exit \$COLLECT_EXIT
    '"
