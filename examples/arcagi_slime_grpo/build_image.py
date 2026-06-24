#!/usr/bin/env python3
"""Build the ARC-AGI compression runtime image.

The Docker build *context* is the auto-compress repo (so eval.py, library.hy,
tasks/, and reference/ get baked in), but the Dockerfile lives here in the
Polar example dir.  We point `docker build` at both.

    AUTO_COMPRESS=/path/to/auto-compress \
      python examples/arcagi_slime_grpo/build_image.py [--image NAME] [--force]
      [--save-tar PATH]        # docker save | gzip to a shared-NFS tarball

The Convergence cluster's worker nodes run docker, but they're AUTOSCALED cloud
VMs: /var/lib/docker is wiped when a node powers down, so an image built in one
job does not survive to the next.  To make the image available everywhere, we
`docker save` it to a gzip tarball on the shared NFS (/home) and `docker load`
it at the start of each rollout/training job (see load_image.sh).

Build it on a worker node via `run_remote.sh build-arcagi-image`, not the login
node.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
DOCKERFILE = EXAMPLE_DIR / "runtime" / "Dockerfile"
DEFAULT_IMAGE = "polar-arcagi:latest"
IMAGE_VERSION = "1"
VERSION_LABEL = "io.polar.arcagi-image-version"
# Shared-NFS location for the saved image tarball (survives node autoscaling).
DEFAULT_SAVE_TAR = "/home/fraser_convergence_ai/arcagi-image/polar-arcagi.tar.gz"
# Synthetic rl_tasks variant (cold-start training): bakes sft/rl_tasks/ with all
# task .hy reset to baseline.
RL_IMAGE = "polar-arcagi-rl:latest"
RL_SAVE_TAR = "/home/fraser_convergence_ai/arcagi-image/polar-arcagi-rl.tar.gz"

# .dockerignore lines per variant. The arcagi image drops all of sft/ (smaller);
# the rl image keeps sft/rl_tasks but still drops the heavy sft/ subdirs.
_IGNORE_BASE = [
    ".git", ".venv", "**/.venv",
    "baseline_run/runs", "baseline_run/.serve-venv",
    "results*.tsv", "run.log", "**/__pycache__", "*.pyc",
]
_IGNORE_ARCAGI = _IGNORE_BASE + ["sft"]
_IGNORE_RL = _IGNORE_BASE + [
    "sft/out", "sft/training", "sft/templates", "sft/__pycache__",
    "sft/*.py",  # keep sft/rl_tasks/, drop top-level sft generators
]


def auto_compress_root() -> Path:
    root = os.environ.get("AUTO_COMPRESS")
    if root:
        p = Path(root).expanduser().resolve()
    else:
        for cand in (
            Path.home() / "Projects" / "auto-compress",
            Path("/home/fraser_convergence_ai/auto-compress"),
        ):
            if (cand / "eval.py").exists():
                p = cand
                break
        else:
            sys.exit("Set AUTO_COMPRESS to the auto-compress repo root.")
    if not (p / "eval.py").exists() or not (p / "library.hy").exists():
        sys.exit(f"{p} doesn't look like auto-compress (no eval.py/library.hy)")
    return p


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_dockerignore(context: Path, lines: list[str]) -> None:
    """Write a .dockerignore into the build context (always overwrite ours so
    the arcagi vs rl variant gets the right include/exclude set)."""
    path = context / ".dockerignore"
    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path} ({len(lines)} rules)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl-tasks", action="store_true",
                    help="Build the synthetic rl_tasks image (bakes sft/rl_tasks "
                         "scaffolded to baseline); defaults image+tar to the rl names.")
    ap.add_argument("--image", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-tar", default=None,
                    help="gzip tarball path on shared NFS (set '' to skip)")
    args = ap.parse_args()

    image = args.image or (RL_IMAGE if args.rl_tasks else DEFAULT_IMAGE)
    save_tar = args.save_tar if args.save_tar is not None else (
        RL_SAVE_TAR if args.rl_tasks else DEFAULT_SAVE_TAR)
    ignore = _IGNORE_RL if args.rl_tasks else _IGNORE_ARCAGI

    context = auto_compress_root()
    print(f"build context: {context}  (rl_tasks={args.rl_tasks}, image={image})")
    write_dockerignore(context, ignore)

    run([
        "docker", "build",
        "-f", str(DOCKERFILE),
        "-t", image,
        "--build-arg", f"POLAR_ARCAGI_IMAGE_VERSION={IMAGE_VERSION}",
        "--build-arg", f"SCAFFOLD_RL_TASKS={'1' if args.rl_tasks else '0'}",
        "--label", f"{VERSION_LABEL}={IMAGE_VERSION}",
        str(context),
    ])
    print(f"built {image}")

    if save_tar:
        tar = Path(save_tar)
        tar.parent.mkdir(parents=True, exist_ok=True)
        # docker save → gzip in a shell pipe (avoids a giant intermediate tar).
        print(f"saving image to {tar} ...")
        subprocess.run(
            f"docker save {image} | gzip -1 > {tar}",
            shell=True, check=True,
        )
        size_mb = tar.stat().st_size / (1024 * 1024)
        print(f"saved {tar} ({size_mb:.0f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
