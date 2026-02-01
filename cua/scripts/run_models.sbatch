#!/bin/bash
#SBATCH --job-name=gpu_server
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/dev/null

VLLM_CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/vllm-0.13.0.sqsh"

# 1. Get Node List and Assign Roles
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PLANNER_NODE=$(echo "$ALL_NODES" | head -n 1)
ACTOR_NODE=$(echo "$ALL_NODES" | head -n 2 | tail -n 1)

# 2. Launch Server (Actor)
srun -w "$PLANNER_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$VLLM_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/planner.out \
     bash -c "
      echo \"Launching explorer model at $PLANNER_NODE\"
      vllm serve /lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9
     " &

# 3. Launch Server (Planner)
srun -w "$ACTOR_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$VLLM_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/actor.out \
     bash -c "
      echo \"Launching UI-TARS model at $ACTOR_NODE\"
      vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
        --api-key gen \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt.image 5 \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 65536
     "

