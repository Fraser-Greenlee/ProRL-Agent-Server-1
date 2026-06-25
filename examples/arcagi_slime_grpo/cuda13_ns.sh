#!/usr/bin/env bash
# Enter a private mount namespace that masks the system CUDA-12 toolkit with the
# CUDA-13 one, then exec the given command inside it.
#
# WHY: Transformer Engine's cu13 build aborts at the first kernel with "Multiple
# libcudart libraries found: libcudart.so.12 and libcudart.so.13" — its compiled
# loader scans /usr/local/cuda* and, finding the system 12.9 toolkit alongside
# torch's cu13 runtime, refuses to run. No env var / LD_LIBRARY_PATH / LD_PRELOAD
# fixes it (the scan is by filesystem existence). The serving side (SGLang) is
# cu13 and unaffected. We can't remove/rename the shared system toolkit (other
# users depend on it), so we bind-mount the cu13 toolkit OVER the cuda-12 paths
# inside a PRIVATE mount namespace — global /usr/local is untouched for everyone
# else; only this process tree sees the masked view.
#
# Must mask with the REAL cu13 toolkit (not an empty dir): TE's NVRTC loader
# needs a working toolkit at CUDA_HOME=/usr/local/cuda. Verified: TE fused-attn
# fwd runs to completion inside this namespace (see arcagi_te_cudart_conflict).
#
# Usage:  sudo unshare -m bash cuda13_ns.sh <user> <cmd> [args...]
#   (unprivileged `unshare -m` is blocked on these nodes; sudo is required, then
#    we drop back to <user> via runuser so the workload runs unprivileged. The
#    mount namespace is inherited by all children, incl. Ray-spawned actors.)
set -euo pipefail

RUN_USER="${1:?usage: cuda13_ns.sh <user> <cmd> [args...]}"
shift

CUDA13="${TE_CUDA13_HOME:-/usr/local/cuda-13.0}"
[ -d "${CUDA13}" ] || { echo "ERROR: CUDA-13 toolkit not found at ${CUDA13}" >&2; exit 1; }

# Bind the cu13 toolkit over every cuda-12 path TE's scanner might glob. These
# mounts live only in this namespace (unshare -m gave us a private copy).
for tgt in /usr/local/cuda-12.9 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -e "${tgt}" ]; then
        mount --bind "${CUDA13}" "${tgt}"
    fi
done

# Sanity: no cuda-12 runtime should be visible under the standard prefixes now.
leak="$(find /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.9 \
          -name 'libcudart.so.12' 2>/dev/null | wc -l)"
echo "[cuda13_ns] cuda -> $(readlink -f /usr/local/cuda); libcudart.so.12 leaks=${leak}"

# Drop privileges back to the workload user; the namespace is inherited.
# --preserve-environment: keep the caller's full env (VIRTUAL_ENV, PATH,
# WANDB_*, SLURM_*, the run.sh exports, etc.) — without it runuser resets to a
# fresh login env and the re-exec'd workload loses its venv/cluster context.
# HOME can get reset to root's by sudo; pin it back to the user's.
exec runuser -u "${RUN_USER}" --preserve-environment -- \
    env "HOME=$(getent passwd "${RUN_USER}" | cut -d: -f6)" "$@"
