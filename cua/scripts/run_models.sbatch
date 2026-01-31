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
UITARS_NODE=$(echo "$ALL_NODES" | head -n 1)
EXPLORER_NODE=$(echo "$ALL_NODES" | head -n 2 | tail -n 1)

# 2. Launch Server (Actor)
# Added: --nodes=1 --ntasks=1 --exclusive
srun -w "$UITARS_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$VLLM_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/uitars.out \
     bash -c "
      echo \"Launching UI-TARS model at $UITARS_NODE\"
      vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
        --api-key gen \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt.image 5 \
        --limit-mm-per-prompt.video 0 \
        --max-model-len 65536
     " &

# 3. Launch Server (vLLM/Explorer)
# Added: --nodes=1 --ntasks=1 --exclusive
srun -w "$EXPLORER_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$VLLM_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/explorer.out \
     bash -c "
      echo \"Launching explorer model at $EXPLORER_NODE\"
      vllm serve /lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking \
        --api-key gen \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --limit-mm-per-prompt.video 0 \
        --limit-mm-per-prompt.image 3 \
        --async-scheduling \
        --max-model-len 65536 \
        --gpu-memory-utilization 0.9
     "

