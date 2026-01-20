# launching interactive job on a reserved CPU node
CONTAINER_IMAGE=~/lustre/images/cua_cpu.sqsh
srun -A nvr_lpr_agentic --container-image $CONTAINER_IMAGE --container-mounts /lustre:/lustre,/dev/kvm:/dev/kvm,/dev/urandom:/dev/urandom,/dev/random:/dev/random \
    --partition=cpu_interactive --time=4:00:00 --reservation=sla_res_osworld_agent_vlm_cpu_only --pty /bin/bash -l
