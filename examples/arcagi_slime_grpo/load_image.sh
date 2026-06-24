#!/usr/bin/env bash
# Load the arcagi runtime image from the shared-NFS tarball into the local
# docker daemon, unless it's already present.  Source or run this at the start
# of any job (rollout / training) before Polar launches containers.
#
# The cluster's worker nodes are autoscaled cloud VMs whose /var/lib/docker is
# wiped on power-down, so every fresh node must reload the image once.
#
#   bash examples/arcagi_slime_grpo/load_image.sh
#   IMAGE=polar-arcagi:latest TARBALL=/home/.../polar-arcagi.tar.gz bash load_image.sh
set -euo pipefail

IMAGE="${IMAGE:-polar-arcagi:latest}"
TARBALL="${TARBALL:-/home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz}"

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[load_image] $IMAGE already present on $(hostname)"
    exit 0
fi

if [ ! -f "$TARBALL" ]; then
    echo "[load_image] ERROR: tarball not found at $TARBALL" >&2
    echo "[load_image]   build it first: run_remote.sh build-arcagi-image" >&2
    exit 1
fi

echo "[load_image] loading $IMAGE from $TARBALL on $(hostname) ..."
gunzip -c "$TARBALL" | docker load
docker image inspect "$IMAGE" >/dev/null 2>&1 \
    && echo "[load_image] loaded $IMAGE" \
    || { echo "[load_image] ERROR: image not present after load" >&2; exit 1; }
