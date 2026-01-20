#!/bin/bash

#SBATCH --array=0%1  # todo set the number of runs here
#SBATCH --partition=cpu_interactive
#SBATCH --reservation=sla_res_osworld_agent_vlm_cpu_only
#SBATCH --account=nvr_lpr_agentic
#SBATCH --job-name=kvm_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%A.out
#SBATCH --error=logs/slurm-%A.out


CONTAINER_IMAGE=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_cpu.sqsh

# 1. Identify the Compute Node
#    In case this script runs on a head node, we grab the assigned node name.
TARGET_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
echo "[Script] Job assigned to node: $TARGET_NODE"

# 2. Launch Container on that Node
echo "[Script] Launching container (sleep infinity) on $TARGET_NODE..."
srun --nodelist="$TARGET_NODE" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     sleep infinity &

SRUN_PID=$!
trap "kill $SRUN_PID" EXIT

# 3. Wait/Poll for Enroot PID (via SSH)
echo "[Script] SSH-ing into $TARGET_NODE to find PID..."
CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2

    # explanation of the command sent via ssh:
    # 1. enroot list -f  -> Lists processes
    # 2. grep pyxis      -> Filters for your container name
    # 3. grep sleep      -> Ensures the command column shows 'sleep'
    # 4. awk print $2    -> Grabs the PID
    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$TARGET_NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1")

    if [ -z "$CONTAINER_PID" ]; then
        echo "[Script] Container registered, but 'sleep' command not yet visible. Retrying..."
    fi
done

echo "[Script] Found PID: $CONTAINER_PID on $TARGET_NODE"

# 4. Execute the Test (via SSH -> Enroot Exec)
echo "[Script] Checking kvm write permission..."
ssh -o StrictHostKeyChecking=no "$TARGET_NODE" "enroot exec $CONTAINER_PID bash -c '
    if touch /dev/kvm 2>/dev/null; then
        echo \"SUCCESS: /dev/kvm is writable!\"
    else
        echo \"FAILURE: Permission denied.\"
        exit 1
    fi
'"






