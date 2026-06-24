#!/bin/bash
#SBATCH --job-name=arcagi-boot-check
#SBATCH --partition=dev
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=24
#SBATCH --time=01:30:00
#SBATCH --output=logs/%j.log

# Validate ONLY the Slime/Megatron/TE/FLA install stage (no convert/train).
# Detached sbatch so it survives the submitting SSH session dropping.
set -eo pipefail
exec 2>&1
cd ~/ProRL-Agent-Server
mkdir -p logs
export PATH=$HOME/.local/bin:$PATH
source .venv/bin/activate
BOOTSTRAP_ONLY=1 bash examples/arcagi_slime_grpo/launch_e2e.sh
