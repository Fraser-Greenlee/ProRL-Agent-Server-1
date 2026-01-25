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

PARSER_CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_parser.sqsh"
VLLM_CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/vllm-0.13.0.sqsh"
SCRIPT_DIR="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server/cua/scripts"

# 1. Get Node List and Assign Roles
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PARSER_NODE=$(echo "$ALL_NODES" | head -n 1)
VLLM_NODE=$(echo "$ALL_NODES" | head -n 2 | tail -n 1)

# 2. Launch Server (Parser)
# Added: --nodes=1 --ntasks=1 --exclusive
srun -w "$PARSER_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$PARSER_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/parser.out \
     bash -c "
      cd $SCRIPT_DIR
      echo \"Launching parser model at $PARSER_NODE\"
      python run_parser_server.py
     " &

# 3. Launch Server (vLLM/Explorer)
# Added: --nodes=1 --ntasks=1 --exclusive
srun -w "$VLLM_NODE" \
     --nodes=1 \
     --ntasks=1 \
     --container-image="$VLLM_CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre \
     --output=logs/explorer.out \
     bash -c "
      echo \"Launching explorer model at $VLLM_NODE\"
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

