#!/bin/bash
#SBATCH --job-name=swdl-job:dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch_block1
#SBATCH --time=4:00:00
#SBATCH --account=llmservice_fm_vision
#SBATCH --gpus-per-node=8
#SBATCH --output=./cua/logs/slurm-%j.out
#SBATCH --error=./cua/logs/slurm-%j.out
#SBATCH --reservation=sla_res_osworld_agent_vlm_gpu

# Mounts
# NODE_NAME=$1
# 01466,01859,02338,02364,02832,03149,03196,03947
GPFS="/lustre/"
HAOHOME="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh"
ROOT="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/root"
RESULTS="$HAOHOME/results"
PROJECTS="$HAOHOME/projects"
DATA="$HAOHOME/data"
HOME="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh"

IMAGE=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/dockers/nvidian+nemo+verl_v2_enroot_dev0.8.5.sqsh



srun --overlap --nodes=1 --ntasks=1   --container-image="$IMAGE"  --no-container-mount-home --container-mounts="$HOME:/home/,$GPFS:/lustre/,$RESULTS:/results,$ROOT:/root,$DATA:/datasets/,$PROJECTS:/Projects,/dev/kvm:/dev/kvm,/dev/urandom:/dev/urandom,/dev/random:/dev/random" sleep infinity
