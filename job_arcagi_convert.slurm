#!/bin/bash
#SBATCH --job-name=arcagi-convert
#SBATCH --partition=dev
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=24
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.log

# Convert the SFT'd 27B HF checkpoint to Megatron torch_dist (one-time).
# Detached so it survives the submitting SSH session dropping.
set -eo pipefail
exec 2>&1
cd ~/ProRL-Agent-Server
mkdir -p logs
export PATH=$HOME/.local/bin:$PATH
source .venv/bin/activate

# Point at the patched local snapshot (HF repo missing ancillary files).
export HF_CHECKPOINT="${HF_CHECKPOINT:-/home/fraser_convergence_ai/.cache/huggingface/hub/models--Fraser--Qwen3.6-27B-ARC-Hy/snapshots/d69da102b04f93f9b7da6d4b966101a6afd58cae}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-$HOME/ProRL-Agent-Server/tmp/checkpoints/Qwen3.6-27B-ARC-Hy_torch_dist}"

echo "================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "HF:     ${HF_CHECKPOINT}"
echo "Out:    ${TORCH_DIST_DIR}"
echo "================================================================"

bash examples/arcagi_slime_grpo/convert_weights.sh
echo "=== CONVERT DONE; output contents ==="
ls -la "${TORCH_DIST_DIR}" 2>&1 | head
