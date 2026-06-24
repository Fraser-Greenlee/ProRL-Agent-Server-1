#!/bin/bash
# Remote workflow for Polar on the Slurm cluster.
#
# Mirrors the dllm-style workflow: edit locally, mutagen sync to the cluster,
# submit Slurm jobs over SSH, tail logs.
#
# Usage:
#   ./run_remote.sh setup                   # one-time: venv + deps on GPU node
#   ./run_remote.sh sync                    # start continuous bidirectional sync
#   ./run_remote.sh stop-sync               # stop the sync daemon
#   ./run_remote.sh build-image             # build calculator runtime image on cluster
#   ./run_remote.sh submit smoke            # submit 1-GPU smoke job (rollout health)
#   ./run_remote.sh submit calculator       # submit 1-GPU calculator end-to-end
#   ./run_remote.sh submit train            # submit multi-GPU Slime training job
#   ./run_remote.sh submit job_polar_smoke.slurm
#   ./run_remote.sh logs                    # tail latest log
#   ./run_remote.sh logs 12345              # tail specific job log
#   ./run_remote.sh ssh                     # interactive shell on login node
#   ./run_remote.sh exec '<command>'        # run a one-off command in the repo dir

set -euo pipefail

HOST="${POLAR_REMOTE_HOST:-hpc1125-login-003}"
REMOTE_DIR="${POLAR_REMOTE_DIR:-~/ProRL-Agent-Server}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_NAME="${POLAR_SYNC_NAME:-polar}"
PARTITION="${POLAR_PARTITION:-dev}"

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in

setup)
    echo "==> Creating remote venv (login node)..."
    ssh "$HOST" "export PATH=\$HOME/.local/bin:\$PATH && mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && uv venv --python 3.12"

    echo "==> Installing core Polar deps on GPU node (gets CUDA wheels, avoids login OOM)..."
    ssh "$HOST" "cd $REMOTE_DIR && srun --partition=$PARTITION --gpus=1 --cpus-per-gpu=12 --time=01:00:00 bash -c '\
        export PATH=\$HOME/.local/bin:\$PATH && \
        cd $REMOTE_DIR && \
        source .venv/bin/activate && \
        uv pip install -e . && \
        uv pip install vllm --torch-backend=auto'"

    echo "==> Verifying install..."
    ssh "$HOST" "cd $REMOTE_DIR && source .venv/bin/activate && python -c 'import polar; import vllm; print(\"polar=\" + polar.__file__); print(\"vllm OK\")'"
    echo "==> Done. Run './run_remote.sh build-image' next, then './run_remote.sh submit smoke'."
    ;;

sync)
    mutagen sync terminate "$SYNC_NAME" 2>/dev/null || true

    echo "==> Starting continuous bidirectional sync to $HOST:$REMOTE_DIR ..."
    mutagen sync create \
        "$LOCAL_DIR" \
        "$HOST:$REMOTE_DIR" \
        --name "$SYNC_NAME" \
        --ignore-vcs \
        --ignore ".venv" \
        --ignore "__pycache__" \
        --ignore "*.pyc" \
        --ignore "logs" \
        --ignore "wandb" \
        --ignore "rollout_results" \
        --ignore "tmp" \
        --ignore "slime" \
        --ignore "Megatron-LM" \
        --ignore "web/node_modules" \
        --ignore "web/dist" \
        --ignore "*.sif" \
        --ignore ".env.local" \
        --ignore ".env.*" \
        --sync-mode "two-way-resolved"

    echo "==> Sync running. Use 'mutagen sync monitor $SYNC_NAME' to watch."
    echo "==> To stop: ./run_remote.sh stop-sync"
    mutagen sync list "$SYNC_NAME"
    ;;

stop-sync)
    mutagen sync terminate "$SYNC_NAME" 2>/dev/null || true
    echo "==> Sync $SYNC_NAME stopped."
    ;;

build-image)
    echo "==> Building calculator runtime docker image on cluster..."
    ssh "$HOST" "cd $REMOTE_DIR && srun --partition=$PARTITION --gpus=0 --cpus-per-task=4 --time=00:30:00 bash -c '\
        cd $REMOTE_DIR && \
        source .venv/bin/activate && \
        python examples/calculator/build_image.py'"
    ;;

build-arcagi-image)
    # Build the ARC-AGI runtime image on a CPU worker (docker daemon is up on
    # workers; build context is the auto-compress repo, Dockerfile lives here).
    AUTO_COMPRESS_REMOTE="${AUTO_COMPRESS_REMOTE:-/home/fraser_convergence_ai/auto-compress}"
    echo "==> Building arcagi runtime docker image on cluster (cpu partition)..."
    ssh "$HOST" "cd $REMOTE_DIR && srun --partition=cpu --gpus=0 --cpus-per-task=8 --time=00:40:00 bash -c '\
        cd $REMOTE_DIR && \
        source .venv/bin/activate && \
        AUTO_COMPRESS=$AUTO_COMPRESS_REMOTE python examples/arcagi_slime_grpo/build_image.py'"
    ;;

submit)
    # Aliases for the common jobs.
    ARG="${1:-smoke}"
    shift 2>/dev/null || true
    case "$ARG" in
        smoke)       SLURM_FILE="job_polar_smoke.slurm" ;;
        calculator)  SLURM_FILE="job_polar_calculator.slurm" ;;
        train)       SLURM_FILE="job_polar_train.slurm" ;;
        arcagi)      SLURM_FILE="job_arcagi_train.slurm" ;;
        arcagi2)     SLURM_FILE="job_arcagi_train_2node.slurm" ;;
        *.slurm)     SLURM_FILE="$ARG" ;;
        *)           echo "Unknown job: $ARG (use smoke|calculator|train|arcagi|arcagi2|<file>.slurm)"; exit 2 ;;
    esac

    EXTRA_ARGS="$*"
    ARGS_ENV=""
    if [[ -n "$EXTRA_ARGS" ]]; then
        ARGS_ENV="POLAR_JOB_ARGS='$EXTRA_ARGS' "
    fi

    echo "==> Submitting $SLURM_FILE (partition=$PARTITION) ${EXTRA_ARGS:+args=$EXTRA_ARGS}"
    ssh "$HOST" "cd $REMOTE_DIR && mkdir -p logs && ${ARGS_ENV}sbatch $SLURM_FILE"
    ;;

logs)
    JOB_ID="${1:-}"
    if [[ -n "$JOB_ID" ]]; then
        echo "==> Tailing logs for job $JOB_ID"
        ssh "$HOST" "tail -f $REMOTE_DIR/logs/${JOB_ID}.log"
    else
        echo "==> Tailing latest log"
        ssh "$HOST" "tail -f \$(ls -t $REMOTE_DIR/logs/*.log 2>/dev/null | head -1)"
    fi
    ;;

ssh)
    ssh -t "$HOST" "cd $REMOTE_DIR && exec \$SHELL -l"
    ;;

exec)
    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 exec '<command>'" >&2
        exit 2
    fi
    ssh "$HOST" "cd $REMOTE_DIR && source .venv/bin/activate 2>/dev/null; $*"
    ;;

queue|squeue)
    ssh "$HOST" "squeue -u \$USER"
    ;;

cancel)
    JOB_ID="${1:-}"
    if [[ -z "$JOB_ID" ]]; then
        echo "Usage: $0 cancel <job_id|all>" >&2
        exit 2
    fi
    if [[ "$JOB_ID" == "all" ]]; then
        ssh "$HOST" "scancel -u \$USER"
    else
        ssh "$HOST" "scancel $JOB_ID"
    fi
    ;;

help|*)
    cat <<USAGE
Usage: $0 {setup|sync|stop-sync|build-image|submit|logs|ssh|exec|queue|cancel} [args...]

  setup                 One-time: create .venv and install Polar+vLLM on GPU node.
  sync                  Start continuous bidirectional file sync (mutagen).
  stop-sync             Stop the sync daemon.
  build-image           Build the calculator docker runtime image on the cluster.
  build-arcagi-image    Build the ARC-AGI runtime docker image (saves to NFS tarball).
  submit <job> [args]   Submit a Slurm job. Aliases:
                          smoke       -> job_polar_smoke.slurm        (1 GPU sanity)
                          calculator  -> job_polar_calculator.slurm   (1 GPU end-to-end)
                          train       -> job_polar_train.slurm        (multi-GPU Slime+Polar)
                        Or pass a *.slurm filename directly.
  logs [job_id]         Tail Slurm log (latest if no ID).
  ssh                   Open an interactive login-node shell in the repo dir.
  exec '<cmd>'          Run a one-off command on the cluster (in the repo dir, venv active).
  queue                 Show your Slurm queue.
  cancel <id|all>       scancel one job or all of yours.

Environment variables:
  POLAR_REMOTE_HOST   default: hpc1125-login-003
  POLAR_REMOTE_DIR    default: ~/ProRL-Agent-Server
  POLAR_SYNC_NAME     default: polar
  POLAR_PARTITION     default: dev
USAGE
    ;;
esac
