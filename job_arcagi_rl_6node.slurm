#!/bin/bash
#SBATCH --job-name=okay_to_cancel_arcagi-rl-6n
#SBATCH --partition=dev
#SBATCH --nodes=6
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.log

# Cold-start GRPO on the 1015 SYNTHETIC rl_tasks. 6-node: 4 TRAIN / 2 SERVE.
#
# Why this split (vs the 8-node 2-train/6-serve):
#  - The 8-node run (19646) died ~1.5h in to a HOST-RAM OOM: the 2 train nodes
#    hit OUT_OF_MEMORY at ~838GB RSS during the optimizer step, because
#    --optimizer-offload-fraction 1.0 puts the full 27B Adam state in host RAM.
#    Spreading training over 4 nodes (not 2) ~halves per-node host RSS (~420GB,
#    safe under the 1.86TB node RAM) — the fix.
#  - Inference was NOT the bottleneck: in 19646 the optimizer step took ~57 min
#    while rollouts took ~5-11 min and overlap with training (max_async_level=2);
#    engines ran only 6-12 concurrent reqs (never saturated). 2 engines easily
#    keep rollout latency hidden behind the ~57-min train step, so dropping
#    serve nodes 6->2 has ~no impact on training speed.
#   train:  ACTOR_NUM_NODES=4, TP=8, CP=4  (TP*CP*PP = 32 = 4 nodes x 8)
#   serve:  (6-4)*8 = 16 GPUs, 2 engines (auto-computed in run.sh)
#
# Submit:  ./run_remote.sh submit arcagi-rl6

set -eo pipefail
exec 2>&1

cd ~/ProRL-Agent-Server
mkdir -p logs results tmp
source .venv/bin/activate
export PYTHONPATH=.:${PYTHONPATH:-}

export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export AUTO_COMPRESS="${AUTO_COMPRESS:-/home/fraser_convergence_ai/auto-compress}"
export RUN_ID="${RUN_ID:-arcagi-rl6-${SLURM_JOB_ID}}"
export N_TASKS="${N_TASKS:-1015}"

# 4 train nodes spread the optimizer-offload host RAM (fixes the OOM) and shard
# the sequence 4 ways (CP=4) for ample GPU headroom; 2 serve. run.sh computes
# ROLLOUT_NUM_GPUS = (6-4)*8 = 16.
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
# CP=4 => trace-drop ceiling = MAX_TOKENS_PER_GPU * 4. mtpg=16384 admits the
# longest real traces (~58K; 16384*4=65536) while quartering per-GPU activation
# vs the original 65536 — per-GPU mtpg (16384) is now BELOW slime's proven 20480
# for a 32B model at TP=8, so generous GPU memory headroom.
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"

# Checkpoint cadence. Job 19937 died at its FIRST save (default interval 10 =
# step 10, ~6h in) → 0 usable checkpoints. Three independent fixes now:
#  1) run.sh polls REAL ray job status (not the submit client), so a dropped
#     log-stream can't trigger the EXIT-trap `ray stop` and SIGKILL a save.
#  2) run.sh defaults to --no-save-optim (SAVE_OPTIM=1 to override): the save
#     drops from ~786GB to ~model-only (~50GB) and writes in <1 min over NFS —
#     we don't resume this run's optimizer (load is the SFT base), so it's safe.
#  3) interval 5 lands a recoverable checkpoint ~3h in and caps loss-on-death at
#     ~5 steps. With small saves the disk/wall-time cost is negligible.
export SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

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
echo "Train:    ${ACTOR_NUM_NODES} nodes, TP=8 x CP=${CONTEXT_PARALLEL_SIZE}; serve: rest (mtpg=${MAX_TOKENS_PER_GPU})"
echo "================================================================"

# Load the rl image on ALL nodes (rollout containers run on the serving nodes).
srun --nodes="${SLURM_NNODES}" --ntasks-per-node=1 \
    bash -c "IMAGE='${ARCAGI_IMAGE}' TARBALL='${ARCAGI_IMAGE_TARBALL}' bash examples/arcagi_slime_grpo/load_image.sh" || true

bash examples/arcagi_slime_grpo/launch_e2e.sh
