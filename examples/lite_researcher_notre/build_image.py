#!/usr/bin/env python3
"""Build the shared runtime image for LiteResearcher rollouts.

    python build_image.py
    python build_image.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
IMAGE_TAG = "polar-lite-researcher-runtime:latest"
IMAGE_LAYOUT_VERSION = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=IMAGE_TAG)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the image exists.")
    return parser.parse_args()


def image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def main() -> int:
    args = parse_args()
    dockerfile_dir = EXAMPLE_DIR / "runtime"
    if not (dockerfile_dir / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile found at {dockerfile_dir / 'Dockerfile'}")
    if image_exists(args.tag) and not args.force:
        print(f"Image already exists: {args.tag} (use --force to rebuild)")
        return 0

    command = [
        "docker", "build",
        "--build-arg", f"POLAR_LITE_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
        "--tag", args.tag,
        str(dockerfile_dir),
    ]
    print("+", " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
