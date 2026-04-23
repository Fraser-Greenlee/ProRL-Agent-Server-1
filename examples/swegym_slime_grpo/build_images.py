#!/usr/bin/env python3
"""Build benchmark-derived runtime images for the curated SWE-Gym sample."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sample_tasks import (
    base_image_for_instance_id,
    derived_runtime_image,
    fetch_sample_instances,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DOCKERFILE_DIR = EXAMPLE_DIR
IMAGE_LAYOUT_VERSION = "4"
IMAGE_VERSION_LABEL = "io.polar.swegym-image-version"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only build the derived image for this curated instance_id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the derived image tag already exists locally.",
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


def image_layout_version(image_ref: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{ index .Config.Labels \"" + IMAGE_VERSION_LABEL + "\" }}",
            image_ref,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def select_instances(args: argparse.Namespace) -> list[dict[str, object]]:
    instances = fetch_sample_instances(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [instance for instance in instances if str(instance.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(instance.get("instance_id")) for instance in selected})
        if missing:
            raise SystemExit(f"Unknown curated instance_id(s): {', '.join(missing)}")
        return selected
    return instances


def main() -> int:
    args = parse_args()
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    print(f"Building {len(instances)} derived runtime image(s).")
    for instance in instances:
        instance_id = str(instance["instance_id"])
        base_image = base_image_for_instance_id(instance_id)
        derived_image = derived_runtime_image(instance_id)
        if image_exists(derived_image) and not args.force:
            current_version = image_layout_version(derived_image)
            if current_version == IMAGE_LAYOUT_VERSION:
                print(f"Skipping existing image: {derived_image}")
                continue
            print(
                f"Rebuilding outdated image: {derived_image} "
                f"(found version={current_version!r}, expected={IMAGE_LAYOUT_VERSION})"
            )
        run_command(["docker", "pull", base_image])
        run_command(
            [
                "docker",
                "build",
                "--build-arg",
                f"BASE_IMAGE={base_image}",
                "--build-arg",
                f"POLAR_SWEGYM_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
                "--tag",
                derived_image,
                str(DOCKERFILE_DIR),
            ]
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
