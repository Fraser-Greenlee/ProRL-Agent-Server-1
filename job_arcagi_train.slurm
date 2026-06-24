#!/bin/bash
#SBATCH --job-name=ok_to_cancel_arcagi-grpo
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# ARC-AGI compression GRPO: Fraser/Qwen3.6-27B-ARC-Hy on one 8xH100 node
# (4 train TP=4 + 4 serve TP=4), wrapping examples/arcagi_slime_grpo/launch_e2e.sh.
#
# The bootstrap (Slime/Megatron clone, TE/FLA install, image build to NFS,
# weight conversion) is idempotent — re-submitting resumes from where it got to.
#
# Submit:  ./run_remote.sh submit job_arcagi_train.slurm
#   (or, from the login node:  sbatch job_arcagi_train.slurm)

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp

source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

# WANDB_API_KEY is loaded from .env.local by launch_e2e.sh / run.sh.

# Tunables forwarded to launch_e2e.sh — see that script for the full list.
# The SFT'd HF repo is missing ancillary files (preprocessor_config etc.); point
# weight conversion at the patched local snapshot we symlink-fixed earlier.
# Override HF_CHECKPOINT=Fraser/Qwen3.6-27B-ARC-Hy only if the repo gets fixed.
export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
export RUN_ID="${RUN_ID:-arcagi-${SLURM_JOB_ID}}"
export N_TASKS="${N_TASKS:-40}"
export INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
export INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"
export BUILD_IMAGE="${BUILD_IMAGE:-auto}"
export CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"

echo "================================================================"
echo "Job ID:   ${SLURM_JOB_ID}  Node: $(hostname)"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Run ID:   ${RUN_ID}"
echo "Model:    ${HF_CHECKPOINT}"
echo "Tasks:    ${N_TASKS}"
echo "================================================================"

bash examples/arcagi_slime_grpo/launch_e2e.sh
