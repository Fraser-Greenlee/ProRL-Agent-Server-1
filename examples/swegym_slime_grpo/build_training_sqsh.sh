#!/usr/bin/env bash
# Build a Pyxis/Enroot sqsh image for the host-side training environment.
#
# The resulting image contains a relocatable copy of the current Python venv at
# /opt/polr_venv.  Runtime scripts mount the repo and data paths, then use this
# venv from the sqsh instead of sourcing PROJECT_ROOT/.venv.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ACCOUNT="${ACCOUNT:-nvr_lpr_agentic}"
PARTITION="${PARTITION:-interactive}"
WALL_TIME="${WALL_TIME:-1:00:00}"
GPUS="${GPUS:-1}"

BASE_SQSH="${BASE_SQSH:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/dockers/nvidian+nemo+prorl_agent_0.10.1_fix.sqsh}"
OUT_SQSH="${POLR_TRAIN_SQSH:-/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/dockers/polr_swegym_slime_grpo_train.sqsh}"
HOST_VENV="${HOST_VENV:-${PROJECT_ROOT}/.venv}"
HOST_CPYTHON="${HOST_CPYTHON:-$(readlink -f "${HOST_VENV}/bin/python" | sed 's#/bin/python[0-9.]*$##')}"

if [ ! -f "$BASE_SQSH" ]; then
    echo "ERROR: base sqsh not found: $BASE_SQSH" >&2
    exit 1
fi
if [ ! -x "${HOST_VENV}/bin/python" ]; then
    echo "ERROR: source venv not found: ${HOST_VENV}" >&2
    exit 1
fi
if [ ! -x "${HOST_CPYTHON}/bin/python3.12" ]; then
    echo "ERROR: source CPython not found: ${HOST_CPYTHON}" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT_SQSH")" "${PROJECT_ROOT}/logs/slurm"
rm -f "$OUT_SQSH"

echo "============================================="
echo "Build Polar SWE-Gym training sqsh"
echo "  Base:       ${BASE_SQSH}"
echo "  Output:     ${OUT_SQSH}"
echo "  Source venv:${HOST_VENV}"
echo "  CPython:    ${HOST_CPYTHON}"
echo "  Account:    ${ACCOUNT}"
echo "  Partition:  ${PARTITION}"
echo "============================================="

BUILD_CMD="$(cat <<'EOF'
set -euo pipefail
set -x

rm -rf /opt/polr_venv /opt/cpython-3.12.12
mkdir -p /opt
cp -a /mnt/polr_venv /opt/polr_venv
cp -a /mnt/cpython /opt/cpython-3.12.12

rm -f /opt/polr_venv/bin/python /opt/polr_venv/bin/python3 /opt/polr_venv/bin/python3.12
ln -s /opt/cpython-3.12.12/bin/python3.12 /opt/polr_venv/bin/python
ln -s python /opt/polr_venv/bin/python3
ln -s python /opt/polr_venv/bin/python3.12

cat >/opt/polr_venv/pyvenv.cfg <<'CFG'
home = /opt/cpython-3.12.12/bin
implementation = CPython
version_info = 3.12.12
include-system-site-packages = false
CFG

/opt/polr_venv/bin/python - <<'PY'
from pathlib import Path

venv = Path("/opt/polr_venv")
old_prefixes = (
    "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/polr/ProRL-Agent-Server/.venv/bin/python3",
    "/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/haozh/projects/polr/ProRL-Agent-Server/.venv/bin/python",
)
new_shebang = "#!/opt/polr_venv/bin/python"
for path in (venv / "bin").iterdir():
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    if not data.startswith(b"#!"):
        continue
    try:
        text = data.decode()
    except UnicodeDecodeError:
        continue
    first, sep, rest = text.partition("\n")
    if any(first.startswith(f"#!{prefix}") for prefix in old_prefixes):
        path.write_text(new_shebang + sep + rest)
PY

# The conda-built Apptainer binary expects this compiled-in session dir to
# exist.  Pyxis containers are writable only for the current task unless this
# directory is baked into the saved sqsh.
mkdir -p /home/conda/feedstock_root/build_artifacts/apptainer_1764715648377/_h_env_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_placehold_p/var/apptainer/mnt/session

/opt/polr_venv/bin/python - <<'PY'
import ray
import sglang
import torch
print("python ok")
print("torch", torch.__version__, torch.version.cuda)
print("ray", ray.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
PY
EOF
)"

srun \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:"${GPUS}" \
    --time="${WALL_TIME}" \
    --mem=0 \
    --container-image="${BASE_SQSH}" \
    --container-save="${OUT_SQSH}" \
    --container-writable \
    --no-container-mount-home \
    --container-mounts="${HOST_VENV}:/mnt/polr_venv:ro,${HOST_CPYTHON}:/mnt/cpython:ro" \
    --container-workdir=/ \
    bash -lc "${BUILD_CMD}"

echo "Built training sqsh: ${OUT_SQSH}"
