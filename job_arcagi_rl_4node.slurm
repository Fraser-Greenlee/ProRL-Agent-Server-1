#!/bin/bash
#SBATCH --job-name=okay_to_cancel_arcagi-rl-4n
#SBATCH --partition=dev
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# Cold-start GRPO on the 1015 SYNTHETIC rl_tasks (easy tasks -> reward variance
# the base 27B can't produce on real ARC). Same 2-node split as the arcagi run:
# node0 = training (TP=8), node1 = SGLang serving.
#
# Differences from job_arcagi_train_2node.slurm (all via env -> run.sh/launch_e2e):
#   - DATA_SOURCE=rl_tasks         prepare_data builds from sft/rl_tasks/pairs.json
#   - rl image (sft/rl_tasks baked, scaffolded to baseline) + its NFS tarball
#   - POLAR_CONFIG_TEMPLATE -> polar_config_rl.yaml (synthetic evaluator)
#
# Submit:  ./run_remote.sh submit arcagi-rl

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp
source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
export RUN_ID="${RUN_ID:-arcagi-rl4-${SLURM_JOB_ID}}"
export N_TASKS="${N_TASKS:-1015}"

# 4 nodes: node0 trains (TP=8), nodes1-3 serve rollouts (24 GPUs, 3 engines).
# Double the GRPO group size (8 -> 16 samples/prompt) for a stronger advantage
# estimate; the extra serving capacity absorbs the larger rollout volume.
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"

# Synthetic-task wiring.
export DATA_SOURCE=rl_tasks
export PROMPT_DATA="${PROMPT_DATA:-$PWD/examples/arcagi_slime_grpo/rl_tasks_train.jsonl}"
export POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-$PWD/examples/arcagi_slime_grpo/polar_config_rl.yaml}"
export ARCAGI_IMAGE="${ARCAGI_IMAGE:-polar-arcagi-rl:latest}"
export ARCAGI_IMAGE_TARBALL="${ARCAGI_IMAGE_TARBALL:-/home/fraser_convergence_ai/arcagi-image/polar-arcagi-rl.tar.gz}"
export BUILD_IMAGE_ARGS="--rl-tasks"

# Stack already installed/converted; skip the heavy bootstrap stages.
export INSTALL_EDITABLE="${INSTALL_EDITABLE:-0}"
export INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-0}"
export APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-0}"
export BUILD_IMAGE="${BUILD_IMAGE:-auto}"
export CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"

echo "================================================================"
echo "Job ID:   ${SLURM_JOB_ID}  Nodes: ${SLURM_NNODES}  Nodelist: ${SLURM_JOB_NODELIST}"
echo "Head:     $(hostname)"
echo "Run ID:   ${RUN_ID}   Tasks: ${N_TASKS} (synthetic rl_tasks)"
echo "================================================================"

# Load the rl image on BOTH nodes (rollout containers run on node1).
srun --nodes="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "IMAGE='${ARCAGI_IMAGE}' TARBALL='${ARCAGI_IMAGE_TARBALL}' bash examples/arcagi_slime_grpo/load_image.sh" || true

bash examples/arcagi_slime_grpo/launch_e2e.sh
