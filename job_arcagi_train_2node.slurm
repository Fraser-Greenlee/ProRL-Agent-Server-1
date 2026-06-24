#!/bin/bash
#SBATCH --job-name=ok_to_cancel_arcagi-grpo-2n
#SBATCH --partition=dev
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# ARC-AGI compression GRPO on 2x8xH100: node0 = training (TP=8), node1 =
# SGLang serving. The batch script runs on node0 (Ray head); run.sh starts a
# Ray worker on node1 via srun. TP=8 halves the per-rank params that OOM'd at
# TP=4 on a single node.
#
# Submit:  ./run_remote.sh submit arcagi2

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp
source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
export RUN_ID="${RUN_ID:-arcagi-2n-${SLURM_JOB_ID}}"
export N_TASKS="${N_TASKS:-40}"
# Stack is already installed/converted; skip the heavy bootstrap stages.
export INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
export INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-0}"
export APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-0}"
export BUILD_IMAGE="${BUILD_IMAGE:-auto}"
export CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"

echo "================================================================"
echo "Job ID:   ${SLURM_JOB_ID}  Nodes: ${SLURM_NNODES}  Nodelist: ${SLURM_JOB_NODELIST}"
echo "Head:     $(hostname)"
echo "Run ID:   ${RUN_ID}"
echo "Model:    ${HF_CHECKPOINT}"
echo "================================================================"

# The image must be loaded on BOTH nodes (rollout containers run on node1).
srun --nodes="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash examples/arcagi_slime_grpo/load_image.sh || true

bash examples/arcagi_slime_grpo/launch_e2e.sh
