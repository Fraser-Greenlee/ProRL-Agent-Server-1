#!/bin/bash
#SBATCH --job-name=okay_to_cancel_arcagi-rl-8n
#SBATCH --partition=dev
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# Cold-start GRPO on the 1015 SYNTHETIC rl_tasks. 8-node variant of
# job_arcagi_rl_4node.slurm. The key difference: TRAINING spans 2 nodes as
# TP=8 x context-parallel=2 (16 GPUs) so the sequence is sharded and per-GPU
# activation memory is halved — the fix for the train-step OOM at 64K-token
# traces (TP=8 alone fills one node's 8 GPUs, leaving no room; CP>1 needs >1
# train node). The remaining 6 nodes (48 GPUs) serve rollouts (6 SGLang
# engines) -> also more rollout throughput than the 4-node run.
#   train:  ACTOR_NUM_NODES=2, TP=8, CP=2  (TP*CP*PP = 16 = 2 nodes x 8)
#   serve:  (8-2)*8 = 48 GPUs, 6 engines (auto-computed in run.sh)
#
# Submit:  ./run_remote.sh submit arcagi-rl8

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp
source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
export RUN_ID="${RUN_ID:-arcagi-rl8-${SLURM_JOB_ID}}"
export N_TASKS="${N_TASKS:-1015}"

# Training spans 2 nodes for context-parallel (halves per-GPU activation memory,
# fixing the OOM); 6 nodes serve. run.sh computes ROLLOUT_NUM_GPUS from these.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"
# With CP=2 the trace-drop ceiling is MAX_TOKENS_PER_GPU * cp_size, and CP shards
# the sequence across 2 GPUs so per-GPU activation memory ~ mtpg. So HALVE mtpg
# to 32768: per-GPU memory drops toward slime's proven ~20-30K range for a ~32B
# model at TP=8 (slime/scripts/run-qwen3-32B.sh uses 20480), while
# mtpg*cp = 32768*2 = 65536 still admits the longest real traces (~58K) without
# dropping them. This is the actual OOM fix; CP=2 alone at mtpg=65536 wouldn't
# cut per-GPU memory.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
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
echo "Train:    ${ACTOR_NUM_NODES} nodes, TP=8 x CP=${CONTEXT_PARALLEL_SIZE}; serve: rest"
echo "================================================================"

# Load the rl image on ALL nodes (rollout containers run on the serving nodes).
srun --nodes="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "IMAGE='${ARCAGI_IMAGE}' TARBALL='${ARCAGI_IMAGE_TARBALL}' bash examples/arcagi_slime_grpo/load_image.sh" || true

bash examples/arcagi_slime_grpo/launch_e2e.sh
