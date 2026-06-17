#!/usr/bin/env python3
"""Pre-pull the per-instance R2E-Gym Docker images.

R2E-Gym images are used as the runtime directly (no layering): each contains the
repo under ``/testbed`` and the gold suite under ``/r2e_tests``. The Hermes CLI
is installed at task time via the prepare step, so nothing is built here.

Usage:
    python pull_images.py --max-tasks 10
    python pull_images.py --instance-id <instance_id>
    python pull_images.py                     # pull every image in the subset
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

from dataset import docker_image_for, load_r2e_gym_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max images to pull. -1 = all.")
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    return parser.parse_args()


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = load_r2e_gym_subset(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [i for i in instances if str(i.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(i.get("instance_id")) for i in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    if args.max_tasks > 0:
        return instances[: args.max_tasks]
    return instances


def image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def main() -> int:
    args = parse_args()
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    images = sorted({docker_image_for(i) for i in instances})
    print(f"Pulling {len(images)} unique R2E-Gym image(s) for {len(instances)} instance(s) ...")
    pulled = skipped = failed = 0
    for idx, image in enumerate(images, 1):
        if image_exists(image):
            print(f"  [{idx}/{len(images)}] skip: {image}")
            skipped += 1
            continue
        print(f"+ docker pull {image}")
        if subprocess.run(["docker", "pull", image], check=False).returncode == 0:
            pulled += 1
        else:
            print(f"  [{idx}/{len(images)}] FAILED: {image}")
            failed += 1

    print(f"\nDone. pulled={pulled}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
