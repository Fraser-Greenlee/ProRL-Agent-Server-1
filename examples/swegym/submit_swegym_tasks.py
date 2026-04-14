#!/usr/bin/env python3
"""Submit the curated SWE-Gym sample through the rollout server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sample_tasks import (
    DATASET_NAME,
    DATASET_SPLIT,
    SAMPLE_TASKS,
    derived_image_for_harness,
    fetch_sample_instances,
    sanitize_instance_id,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.yaml"

PREPARE_COMMAND = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/logs/agent /polar/session/workspace /root/.venv/bin && "
    "cp -a /testbed/. /polar/session/workspace/ && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python3 && "
    "git config --global core.pager '' && "
    "cd /polar/session/workspace && git reset --hard && "
    # Fix pydantic version mismatch inside the swe-agent conda env
    "source /opt/miniconda3/etc/profile.d/conda.sh && "
    "conda activate polar-sweagent && "
    "pip install --upgrade pydantic pydantic-core -q 2>/dev/null; "
    # Patch swe-agent to tolerate chown failure (apptainer user namespace)
    "sed -i 's/raise RuntimeError(msg)/pass  # chown not needed in apptainer/' "
    "/opt/miniconda3/envs/polar-sweagent/lib/python3.11/site-packages/sweagent/environment/repo.py; "
    "true"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="swe_agent", help="Harness name to submit.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "openai/gpt-5.4"),
        help="Model name routed through the Polar gateway.",
    )
    parser.add_argument(
        "--rollout-server-url",
        default=os.environ.get("POLAR_ROLLOUT_URL"),
        help="Optional rollout server URL override. Defaults to the topology file.",
    )
    parser.add_argument(
        "--topology",
        default=os.environ.get("POLAR_TOPOLOGY", str(DEFAULT_TOPOLOGY)),
        help="Path to topology.yaml",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.environ.get("NUM_SAMPLES", "1")),
        help="How many rollout samples to run per task.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=int(os.environ.get("MAX_TASKS", "10")),
        help="Maximum number of curated tasks to submit.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Restrict submission to a specific curated instance_id. Can be repeated.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument(
        "--runtime-backend",
        choices=["docker", "apptainer"],
        default=os.environ.get("RUNTIME_BACKEND", "apptainer"),
        help="Container runtime backend for the session",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for request/response files. Defaults to examples/swegym/<harness>/batches/<timestamp>/",
    )
    parser.add_argument(
        "--refresh-dataset-cache",
        action="store_true",
        help="Refresh the cached HF dataset rows before submitting.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def docker_image_exists(image_ref: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def builder_spec_for_harness(harness: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    return {"strategy": "prefix_merging", **({"config": config} if config else {})}


def evaluator_exclude_patterns_for_harness(harness: str) -> list[str]:
    patterns: list[str] = []
    if harness == "swe_agent":
        patterns.extend(
            [
                "trajectories/**",
                "**/trajectories/**",
            ]
        )
    return patterns


def agent_settings_for_harness(harness: str) -> dict[str, Any]:
    if harness == "swe_agent":
        return {
            "repo_path": "/polar/session/workspace",
            "shell_preamble": (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate polar-sweagent && "
                "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
            ),
        }
    return {}


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = fetch_sample_instances(refresh=args.refresh_dataset_cache)
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [instance for instance in instances if str(instance.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(instance.get("instance_id")) for instance in selected})
        if missing:
            raise SystemExit(f"Unknown curated instance_id(s): {', '.join(missing)}")
        return selected
    return instances[: args.max_tasks]


def build_task_request(
    args: argparse.Namespace,
    *,
    instance: dict[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    derived_image = derived_image_for_harness(instance_id, args.harness)
    return {
        "task_id": f"swegym-{args.harness}-{sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": str(instance["problem_statement"]).strip(),
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": args.runtime_backend,
            "image": runtime_image_for_backend(derived_image, args.runtime_backend),
            "prepare": [
                {
                    "type": "exec",
                    "command": PREPARE_COMMAND,
                }
            ],
            "env": {},
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {
            "harness": args.harness,
            "model_name": args.model_name,
            "settings": agent_settings_for_harness(args.harness),
            "env": {}
        },
        "builder": builder_spec_for_harness(args.harness),
        "evaluator": {
            "strategy": "swegym_git_diff",
            "config": {
                "repo_dir": "/testbed",
                "patch_command": "cd /testbed && git diff --binary --submodule=diff",
                "instance": instance,
                "exclude_patterns": evaluator_exclude_patterns_for_harness(args.harness),
            },
            "refresh_runtime": True,
        },
    }


def runtime_image_for_backend(image: str, backend: str) -> str:
    if backend != "apptainer":
        return image
    if image.startswith(("docker-daemon:", "docker://", "oras://")):
        return image
    return f"docker-daemon:{image}"


def summarize_result(response: dict[str, Any]) -> dict[str, Any]:
    sessions = response.get("results") or []
    reward_one = 0
    completed = 0
    session_errors = 0
    trajectory_errors = 0
    for session in sessions:
        if session.get("status") == "COMPLETED":
            completed += 1
        if session.get("error"):
            session_errors += 1
        trajectory = session.get("trajectory") or {}
        if trajectory.get("status") == "ERROR" or trajectory.get("error"):
            trajectory_errors += 1
        traces = trajectory.get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return {
        "total_sessions": len(sessions),
        "completed_sessions": completed,
        "reward_one_sessions": reward_one,
        "session_errors": session_errors,
        "trajectory_errors": trajectory_errors,
    }


def submit_task_file(
    request_path: Path,
    *,
    topology_path: str | None,
    rollout_server_url: str | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "polar.cli",
        "submit",
        str(request_path),
        "--json",
    ]
    if topology_path:
        command.extend(["-c", topology_path])
    if rollout_server_url:
        command.extend(["--rollout-url", rollout_server_url])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    args = parse_args()
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    ready_instances: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for instance in instances:
        image_ref = derived_image_for_harness(str(instance["instance_id"]), args.harness)
        if docker_image_exists(image_ref):
            ready_instances.append(instance)
        else:
            missing_images.append(image_ref)
    if args.instance_id and missing_images:
        missing_str = "\n".join(f"  - {image_ref}" for image_ref in missing_images)
        raise SystemExit(
            f"Missing derived {args.harness} image(s):\n"
            f"{missing_str}\n"
            f"Run `bash examples/swegym/{args.harness}/setup.sh` first."
        )
    if not ready_instances:
        raise SystemExit(
            f"No ready images found for harness {args.harness!r}. "
            f"Run `bash examples/swegym/{args.harness}/setup.sh` first."
        )
    if missing_images and not args.instance_id:
        for image_ref in missing_images:
            print(f"Skipping missing image: {image_ref}")
    instances = ready_instances

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EXAMPLE_DIR / args.harness / "batches" / batch_id
    )
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "batch_id": batch_id,
        "harness": args.harness,
        "model_name": args.model_name,
        "num_samples": args.num_samples,
        "tasks": [
            {
                "instance_id": instance["instance_id"],
                "repo": instance["repo"],
                "derived_image": derived_image_for_harness(str(instance["instance_id"]), args.harness),
            }
            for instance in instances
        ],
        "curated_sample": SAMPLE_TASKS,
    }
    write_json(output_dir / "manifest.json", manifest)

    prepared: list[tuple[str, dict[str, Any], Path, Path]] = []
    for instance in instances:
        instance_id = str(instance["instance_id"])
        task_dir = output_dir / sanitize_instance_id(instance_id)
        request_path = task_dir / "request.json"
        response_path = task_dir / "response.json"
        payload = build_task_request(args, instance=instance, batch_id=batch_id)
        write_json(request_path, payload)
        print(f"[{instance_id}] wrote request to {request_path}")
        prepared.append((instance_id, payload, request_path, response_path))

    if args.dry_run:
        summaries = [
            {"instance_id": iid, "task_id": p["task_id"], "dry_run": True}
            for iid, p, _, _ in prepared
        ]
        write_json(output_dir / "summary.json", summaries)
        print(f"Wrote batch summary to {output_dir / 'summary.json'}")
        return 0

    def _submit_one(item: tuple[str, dict[str, Any], Path, Path]) -> dict[str, Any]:
        instance_id, payload, request_path, response_path = item
        result = submit_task_file(
            request_path,
            topology_path=args.topology,
            rollout_server_url=args.rollout_server_url,
        )
        write_json(response_path, result)
        summary = {
            "instance_id": instance_id,
            "task_id": payload["task_id"],
            "response_path": str(response_path),
            **summarize_result(result),
        }
        print(
            f"[{instance_id}] completed: reward_1="
            f"{summary['reward_one_sessions']}/{summary['total_sessions']}"
        )
        return summary

    print(f"Submitting {len(prepared)} tasks concurrently ...")
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
        futures = {pool.submit(_submit_one, item): item for item in prepared}
        for future in as_completed(futures):
            summaries.append(future.result())

    write_json(output_dir / "summary.json", summaries)
    print(f"Wrote batch summary to {output_dir / 'summary.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
