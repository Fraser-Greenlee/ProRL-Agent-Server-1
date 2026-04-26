#!/usr/bin/env python3
"""Prepare Docker runtime assets for the full SWE-Gym dataset.

The rollout runtime uses the official SWE-Gym base images directly. Agent CLIs
live in one host directory mounted into every container, avoiding a duplicated
~1.4 GB layer across hundreds of per-task images.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sample_tasks import (
    base_image_for_instance_id,
    fetch_all_instances,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
DOCKERFILE_DIR = EXAMPLE_DIR
DEFAULT_AGENT_CLI_DIR = PROJECT_ROOT / "tmp" / "swegym_agent_cli" / "opt_node"
CLI_IMAGE = "polar-swegym-cli-layer:swegym-v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only prepare the base image for this instance_id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pull base images even when the tag already exists locally.",
    )
    parser.add_argument(
        "--agent-cli-dir",
        type=Path,
        default=DEFAULT_AGENT_CLI_DIR,
        help="Host directory mounted as /opt/node in task containers.",
    )
    parser.add_argument(
        "--force-cli",
        action="store_true",
        help="Rebuild and re-extract the shared Node/agent CLI directory.",
    )
    parser.add_argument(
        "--skip-cli",
        action="store_true",
        help="Only pull SWE-Gym base images; do not prepare the shared CLI directory.",
    )
    parser.add_argument(
        "--pull-jobs",
        type=int,
        default=4,
        help="Number of concurrent docker pull jobs for base images.",
    )
    parser.add_argument(
        "--refresh-dataset-cache",
        action="store_true",
        help="Refresh the cached dataset rows before building images.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def image_exists(image_ref: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def select_instances(args: argparse.Namespace) -> list[dict[str, object]]:
    instances = fetch_all_instances(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [instance for instance in instances if str(instance.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(instance.get("instance_id")) for instance in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    return instances


def unique_base_images(instances: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    images: list[str] = []
    for instance in instances:
        image = base_image_for_instance_id(str(instance["instance_id"]))
        if image in seen:
            continue
        seen.add(image)
        images.append(image)
    return images


def ensure_agent_cli_dir(agent_cli_dir: Path, *, force: bool) -> None:
    required_bins = ("node", "codex", "claude", "qwen", "opencode", "pi")
    missing_bins = [
        name for name in required_bins if not (agent_cli_dir / "bin" / name).is_file()
    ]
    if not missing_bins and not force:
        print(f"Shared agent CLI directory already exists: {agent_cli_dir}")
        return

    agent_cli_dir = agent_cli_dir.resolve()
    agent_cli_dir.parent.mkdir(parents=True, exist_ok=True)
    if agent_cli_dir.exists():
        shutil.rmtree(agent_cli_dir)

    run_command(["docker", "build", "--target", "cli-layer", "--tag", CLI_IMAGE, str(DOCKERFILE_DIR)])
    container = f"polar-swegym-cli-extract-{os.getpid()}"
    run_command(["docker", "create", "--name", container, CLI_IMAGE])
    try:
        run_command(["docker", "cp", f"{container}:/opt/node", str(agent_cli_dir)])
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False)
    print(f"Prepared shared agent CLI directory: {agent_cli_dir}")


def ensure_base_image(image: str, *, force: bool) -> tuple[str, str]:
    if image_exists(image) and not force:
        return ("skipped", image)
    run_command(["docker", "pull", image])
    return ("pulled", image)


def ensure_base_images(images: list[str], *, force: bool, jobs: int) -> None:
    if jobs <= 1:
        for image in images:
            status, image = ensure_base_image(image, force=force)
            print(f"{status}: {image}")
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(ensure_base_image, image, force=force) for image in images]
        for future in as_completed(futures):
            status, image = future.result()
            print(f"{status}: {image}")


def main() -> int:
    args = parse_args()
    if not args.skip_cli:
        ensure_agent_cli_dir(args.agent_cli_dir, force=args.force_cli)

    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    images = unique_base_images(instances)
    print(f"Preparing {len(images)} base runtime image(s) with {max(args.pull_jobs, 1)} pull job(s).")
    ensure_base_images(images, force=args.force, jobs=max(args.pull_jobs, 1))

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
