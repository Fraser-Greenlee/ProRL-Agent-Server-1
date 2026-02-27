
srun -p interactive \
--time=240 \
--account=nvr_lpr_agentic \
--container-image=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/dockers/nvidian+nemo+verl_v2+vllm0.11dev_enroot_mm2.sqsh \
--container-mounts=/lustre/,/home/haozh/  \
--gres=gpu:8 \
--cpus-per-task=128  \
--pty bash -c '
CLAUDE_DATA=/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/nv-arp/.claude_data
mkdir -p "$CLAUDE_DATA"
if [ ! -L /root/.claude ]; then
  mv /root/.claude /root/.claude.bak 2>/dev/null
  ln -s "$CLAUDE_DATA" /root/.claude
fi
exec bash
'
