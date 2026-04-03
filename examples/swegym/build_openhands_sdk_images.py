#!/usr/bin/env python3
"""Build benchmark-derived OpenHands SDK images for compatible sample tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sample_tasks import (
    base_image_for_instance_id,
    derived_image_for_harness,
    fetch_sample_instances,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DOCKERFILE_DIR = EXAMPLE_DIR / "openhands_sdk"
IMAGE_LAYOUT_VERSION = "1"
IMAGE_VERSION_LABEL = "io.polar.swegym-image-version"
HARNESS_NAME = "openhands_sdk"
MIN_MAJOR = 3
MIN_MINOR = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only build the derived image for this curated instance_id.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of curated tasks to consider when no --instance-id is supplied.",
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
    return instances[: args.max_tasks]


def benchmark_python_version(image_ref: str) -> tuple[int, int]:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            image_ref,
            "bash",
            "-lc",
            "python - <<'PY'\nimport sys\nprint(f\"{sys.version_info.major}.{sys.version_info.minor}\")\nPY",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to inspect python version for {image_ref}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    raw = result.stdout.strip().splitlines()[-1]
    major_str, minor_str = raw.split(".", 1)
    return int(major_str), int(minor_str)


def is_compatible_python(version: tuple[int, int]) -> bool:
    return version >= (MIN_MAJOR, MIN_MINOR)


def main() -> int:
    args = parse_args()
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    print(f"Considering {len(instances)} derived OpenHands SDK image(s).")
    built_count = 0
    skipped: list[tuple[str, str]] = []
    for instance in instances:
        instance_id = str(instance["instance_id"])
        base_image = base_image_for_instance_id(instance_id)
        derived_image = derived_image_for_harness(instance_id, HARNESS_NAME)
        run_command(["docker", "pull", base_image])
        version = benchmark_python_version(base_image)
        if not is_compatible_python(version):
            message = (
                f"Skipping {instance_id}: benchmark image python is "
                f"{version[0]}.{version[1]}, but OpenHands SDK requires >= {MIN_MAJOR}.{MIN_MINOR}"
            )
            print(message)
            skipped.append((instance_id, message))
            continue
        if image_exists(derived_image) and not args.force:
            current_version = image_layout_version(derived_image)
            if current_version == IMAGE_LAYOUT_VERSION:
                print(f"Skipping existing image: {derived_image}")
                built_count += 1
                continue
            print(
                f"Rebuilding outdated image: {derived_image} "
                f"(found version={current_version!r}, expected={IMAGE_LAYOUT_VERSION})"
            )
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
        built_count += 1

    print(f"Built or reused {built_count} compatible image(s).")
    if skipped:
        print("Skipped incompatible tasks:")
        for _, message in skipped:
            print(f"  - {message}")
    if built_count == 0:
        raise SystemExit("No compatible OpenHands SDK images were built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
