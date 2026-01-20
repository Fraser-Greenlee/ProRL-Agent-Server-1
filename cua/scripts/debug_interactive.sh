#!/bin/bash

# --- Configuration ---
JOB_NAME="kvm_interactive"
ACCOUNT="nvr_lpr_agentic"
PARTITION="cpu_interactive"
RESERVATION="sla_res_osworld_agent_vlm_cpu_only"
TIME_LIMIT="04:00:00"
IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_cpu.sqsh"

# --- 1. Submit the "Holder" Job ---
echo "[Local] Submitting background job to reserve node..."

# We use sbatch with --parsable to get the Job ID cleanly.
# We mount /dev/kvm here so it exists inside the container.
JOB_ID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --reservation="$RESERVATION" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --time="$TIME_LIMIT" \
    --exclusive \
    --wrap="srun --container-image=$IMAGE --container-mounts=/lustre:/lustre sleep infinity")

if [ -z "$JOB_ID" ]; then
    echo "Error: Job submission failed."
    exit 1
fi

echo "[Local] Job submitted. ID: $JOB_ID"

# --- 2. Setup Cleanup Trap ---
# This ensures that whenever this script exits (Ctrl+C or clean exit), the Slurm job is cancelled.
cleanup() {
    echo ""
    echo "[Local] Cleaning up... Cancelling Job $JOB_ID"
    scancel "$JOB_ID"
}
trap cleanup EXIT

# --- 3. Wait for Job to Start ---
echo "[Local] Waiting for job to start..."
NODE=""
while [ -z "$NODE" ]; do
    # Check job state and get the allocated node
    JOB_STATE=$(squeue -j "$JOB_ID" -h -o %T)

    if [ "$JOB_STATE" == "RUNNING" ]; then
        NODE=$(squeue -j "$JOB_ID" -h -o %N)
    elif [ -z "$JOB_STATE" ]; then
        echo "Error: Job disappeared from queue!"
        exit 1
    fi
    sleep 2
done

echo "[Local] Job is RUNNING on Node: $NODE"

# --- 4. Wait for Container Initialization ---
echo "[Local] Polling node $NODE for container readiness..."

CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2
    # We SSH into the node to check 'enroot list'.
    # We filter for 'sleep' to ensure the container is fully booted and not just registering.
    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1")

    if [ -z "$CONTAINER_PID" ]; then
        printf "."
    fi
done

echo ""
echo "[Local] Found Container PID: $CONTAINER_PID"

# --- 5. Launch Interactive Session ---
echo "=========================================================="
echo "   Dropping you into KVM-enabled shell on $NODE"
echo "   (Permissions preserved via host-side enroot exec)"
echo "=========================================================="

# -t forces pseudo-terminal allocation so you get an interactive shell
ssh -t -q -o StrictHostKeyChecking=no "$NODE" \
    "enroot exec $CONTAINER_PID /bin/bash -l"

# --- 6. End of Script ---
# When you type 'exit' in the container, the script reaches here.
# The 'trap' defined earlier will trigger and cancel the Slurm job.
echo "[Local] Session ended."
