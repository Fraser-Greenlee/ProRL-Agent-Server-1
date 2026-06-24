#!/bin/bash
#SBATCH --job-name=polar-train
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# Multi-GPU Slime + Polar GRPO training run, wrapping examples/swegym_slime_grpo/launch_e2e.sh.
# Placeholder until you actually want to fire a real training job — leaves the
# heavy bootstrap (Slime/Megatron clones, weight conversion, image prep) to
# launch_e2e.sh, which is idempotent.
#
# Submit with: ./run_remote.sh submit train
# Adjust --gpus= and the env vars below as needed.

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp

source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

# WANDB_API_KEY can be set in the submit env if you want logging.
# export WANDB_API_KEY=...
# export WANDB_PROJECT=polar-grpo

# Tunables forwarded to launch_e2e.sh — see that script for the full list.
export HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
export RUN_ID="${RUN_ID:-train-${SLURM_JOB_ID}}"
export INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
export INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"
export PREPARE_IMAGES="${PREPARE_IMAGES:-1}"

echo "================================================================"
echo "Job ID:   ${SLURM_JOB_ID}  Node: $(hostname)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Run ID:   ${RUN_ID}"
echo "Model:    ${HF_CHECKPOINT}"
echo "================================================================"

bash examples/swegym_slime_grpo/launch_e2e.sh
