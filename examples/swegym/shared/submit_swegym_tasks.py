#!/usr/bin/env python3
"""Submit the curated SWE-Gym sample through the rollout server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from sample_tasks import (
    DATASET_NAME,
    DATASET_SPLIT,
    SAMPLE_TASKS,
    derived_image_for_harness,
    fetch_sample_instances,
    sanitize_instance_id,
)

SHARED_DIR = Path(__file__).resolve().parent

PREPARE_COMMAND = (
    "rm -rf /arp/session/workspace && "
    "mkdir -p /arp/session/logs/agent /arp/session/workspace /root/.venv/bin && "
    "cp -a /testbed/. /arp/session/workspace/ && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python /root/.venv/bin/python3 && "
    "git config --global core.pager '' && "
    "cd /arp/session/workspace && git reset --hard"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="swe_agent", help="Harness name to submit.")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "openai/MiniMaxAI/MiniMax-M2.5"),
        help="Model name routed through the ARP gateway.",
    )
    parser.add_argument(
        "--rollout-server-url",
        default=os.environ.get("ROLLOUT_SERVER_URL", "http://127.0.0.1:8080"),
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=int(os.environ.get("NUM_ROLLOUTS", "1")),
        help="How many rollouts to run per sampled task.",
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
    parser.add_argument("--agent-timeout", type=float, default=1800.0)
    parser.add_argument("--evaluator-timeout", type=float, default=1800.0)
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


def agent_env_for_harness(harness: str) -> dict[str, str]:
    if harness in {"openhands_sdk", "openhands"}:
        return {"WORKSPACE_BASE": "/arp/session/workspace"}
    return {}


def agent_settings_for_harness(harness: str) -> dict[str, Any]:
    if harness == "swe_agent":
        return {
            "repo_path": "/arp/session/workspace",
            "shell_preamble": (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate arp-sweagent && "
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
        "num_rollouts": args.num_rollouts,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": "docker",
            "image": derived_image,
            "prepare": [
                {
                    "type": "exec",
                    "command": PREPARE_COMMAND,
                    "timeout_sec": 180,
                }
            ],
            "env": {},
            "network": "host",
            "workdir": "/arp/session/workspace",
        },
        "agent": {
            "harness": args.harness,
            "model_name": args.model_name,
            "timeout": args.agent_timeout,
            "settings": agent_settings_for_harness(args.harness),
            "env": agent_env_for_harness(args.harness),
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "git_diff_patch",
            "config": {
                "benchmark": "swe",
                "repo_dir": "/testbed",
                "patch_command": "cd /testbed && git diff --binary --submodule=diff",
                "instance": instance,
            },
            "timeout": args.evaluator_timeout,
            "refresh_runtime": True,
        },
    }


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
        else SHARED_DIR.parent / args.harness / "batches" / batch_id
    )
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "batch_id": batch_id,
        "harness": args.harness,
        "model_name": args.model_name,
        "num_rollouts": args.num_rollouts,
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

    timeout = httpx.Timeout(None, connect=30.0)
    client = None if args.dry_run else httpx.Client(
        base_url=args.rollout_server_url.rstrip("/"),
        timeout=timeout,
    )
    try:
        summaries: list[dict[str, Any]] = []
        for instance in instances:
            instance_id = str(instance["instance_id"])
            task_dir = output_dir / sanitize_instance_id(instance_id)
            payload = build_task_request(args, instance=instance, batch_id=batch_id)
            write_json(task_dir / "request.json", payload)
            print(f"[{instance_id}] wrote request to {task_dir / 'request.json'}")

            if args.dry_run:
                summaries.append(
                    {
                        "instance_id": instance_id,
                        "task_id": payload["task_id"],
                        "dry_run": True,
                    }
                )
                continue

            assert client is not None
            response = client.post("/rollout/task", json=payload)
            response.raise_for_status()
            result = response.json()
            write_json(task_dir / "response.json", result)

            summary = {
                "instance_id": instance_id,
                "task_id": payload["task_id"],
                "response_path": str(task_dir / "response.json"),
                **summarize_result(result),
            }
            summaries.append(summary)
            print(
                f"[{instance_id}] completed: reward_1="
                f"{summary['reward_one_sessions']}/{summary['total_sessions']}"
            )

        write_json(output_dir / "summary.json", summaries)
        print(f"Wrote batch summary to {output_dir / 'summary.json'}")
    finally:
        if client is not None:
            client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
