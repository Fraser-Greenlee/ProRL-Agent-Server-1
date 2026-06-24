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


def write_dockerignore(context: Path) -> Path:
    """Write a .dockerignore into the context to keep the image lean.

    Returns the path so the caller can clean it up if it created it.  We don't
    clobber an existing one.
    """
    path = context / ".dockerignore"
    if path.exists():
        print(f"  (leaving existing {path})")
        return path
    path.write_text(
        "\n".join(
            [
                ".git",
                ".venv",
                "**/.venv",
                "baseline_run/runs",
                "baseline_run/.serve-venv",
                "sft",
                "results*.tsv",
                "run.log",
                "**/__pycache__",
                "*.pyc",
            ]
        )
        + "\n"
    )
    print(f"  wrote {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-tar", default=DEFAULT_SAVE_TAR,
                    help="gzip tarball path on shared NFS (set '' to skip)")
    args = ap.parse_args()

    context = auto_compress_root()
    print(f"build context: {context}")
    write_dockerignore(context)

    run([
        "docker", "build",
        "-f", str(DOCKERFILE),
        "-t", args.image,
        "--build-arg", f"POLAR_ARCAGI_IMAGE_VERSION={IMAGE_VERSION}",
        "--label", f"{VERSION_LABEL}={IMAGE_VERSION}",
        str(context),
    ])
    print(f"built {args.image}")

    if args.save_tar:
        tar = Path(args.save_tar)
        tar.parent.mkdir(parents=True, exist_ok=True)
        # docker save → gzip in a shell pipe (avoids a giant intermediate tar).
        print(f"saving image to {tar} ...")
        subprocess.run(
            f"docker save {args.image} | gzip -1 > {tar}",
            shell=True, check=True,
        )
        size_mb = tar.stat().st_size / (1024 * 1024)
        print(f"saved {tar} ({size_mb:.0f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
