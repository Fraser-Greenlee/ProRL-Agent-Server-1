# Project paths
PROJECT_ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/ProRL-Agent-Server"
CURRENT_DIR="$PROJECT_ROOT/cua/scripts"

# Container images
ACTOR_IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"

# Data-collection script args
MAX_PARALLEL=${MAX_PARALLEL:-16}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-10000}

################################################################
# Prepare reserved node - we colocate actor node and the VM here
################################################################
# --- 1-1. Submit the "holder" job -- #
echo "[Local] Submitting background job to reserve node..."
ACTOR_JOB_ID=$(sbatch --parsable \
    --job-name=kvm_interactive \
    --account=llmservice_fm_vision \
    --partition=interactive \
    --gpus-per-node=8 \
    --reservation=sla_res_osworld_agent_vlm \
    --time=04:00:00 \
    --exclusive \
    --output=/dev/null \
    --error=/dev/null \
    --wrap="srun --container-image=$ACTOR_IMAGE --container-mounts=/lustre:/lustre sleep infinity")

if [ -z "$ACTOR_JOB_ID" ]; then
  echo "Error: Job submission failed."
  exit 1
fi

echo "[Local] Job submitted. ID: $ACTOR_JOB_ID"

# --- 1-2. Setup Cleanup Trap ---
# This ensures that whenever this script exits (Ctrl+C or clean exit), the Slurm job is cancelled.
cleanup() {
    echo ""
    echo "[Local] Cleaning up... Cancelling Job $ACTOR_JOB_ID"
    scancel "$ACTOR_JOB_ID"
}
trap cleanup EXIT

# --- 1-3. Wait for Job to Start ---
echo "[Local] Waiting for job to start..."
ACTOR_NODE=""
while [ -z "$ACTOR_NODE" ]; do
    # Check job state and get the allocated node
    JOB_STATE=$(squeue -j "$ACTOR_JOB_ID" -h -o %T)

    if [ "$JOB_STATE" == "RUNNING" ]; then
        ACTOR_NODE=$(squeue -j "$ACTOR_JOB_ID" -h -o %N)
    elif [ -z "$JOB_STATE" ]; then
        echo "Error: Job disappeared from queue!"
        exit 1
    fi
    sleep 2
done

echo "[Local] Job is RUNNING on Node: $ACTOR_NODE"

# --- 1-4. Wait for Container Initialization ---
echo "[Local] Polling node $ACTOR_NODE for container readiness..."

CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2
    # We SSH into the node to check 'enroot list'.
    # We filter for 'sleep' to ensure the container is fully booted and not just registering.
    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1")

    if [ -z "$CONTAINER_PID" ]; then
        printf "."
    fi
done

echo ""
echo "[Local] Found Container PID: $CONTAINER_PID"


# --- 1-5. Launch Data Collection Script---
# wait for planner node
while ! nc -z "$PLANNER_NODE" 8000; do
  echo '[INFO] Waiting for planner node to accept connections...'
  sleep 10s
done
echo '[INFO] Planner node is ready to accept connections!'

echo "=========================================================="
echo " Executing Data Collection at $ACTOR_JOB_ID"
echo " Logging to: $ACTOR_LOG_FILE"
echo "=========================================================="

ACTOR_LOG_FILE="$CURRENT_DIR/logs/actor-$ACTOR_JOB_ID.out"
ssh -t -q -o StrictHostKeyChecking=no "$ACTOR_NODE" \
    "enroot exec $CONTAINER_PID /bin/bash -c 'cd $PWD && python parallel_collect_trajectories.py \
    --planner_node $PLANNER_NODE \
    --actor_node localhost \
    --max_parallel $MAX_PARALLEL \
    --max_trajectories $MAX_TRAJECTORIES'" \
    2>&1 | tee "$ACTOR_LOG_FILE"


