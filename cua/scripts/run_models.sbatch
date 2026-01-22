#!/bin/bash
#SBATCH --job-name=gpu_server
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=interactive
#SBATCH --nodes=1   # todo when adding explorer node, we need to change NODE_HOST
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/dev/null  # we will set the log path for each srun

# Arguments: $1 = The Run ID (to sync with the CPU job)
RUN_ID=$1
SIGNAL_DIR=$2

CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_vllm.sqsh"
SCRIPT_DIR="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/scripts"

# 1. Get Node List and Assign Roles
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PARSER_NODE=$(echo "$ALL_NODES" | head -n 1)

# todo Future Logic:
# VLLM_NODE=$(echo "$ALL_NODES" | head -n 2 | tail -n 1)


# 2. Launch Server
# Note: We use standard srun behavior here which you confirmed works
srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/parser.out \
     bash -c "
      cd $SCRIPT_DIR;
      echo \"Launching parser model at $PARSER_NODE\"
      python run_parser_server.py
     " &

# --- Future vLLM Launch Block ---
# srun --nodelist="$VLLM_NODE" ... python vllm_server.py &

# 4. Wait for all background sruns to finish

wait






