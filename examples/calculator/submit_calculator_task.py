#!/usr/bin/env python3
"""Submit the calculator task for a specific harness through the rollout server.

Usage:
    python submit_calculator_task.py --harness opencode --image polar-localhost-opencode:latest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXAMPLE_DIR / "assets"
TEST_FILE = ASSETS_DIR / "test_calculator.py"
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.yaml"

INSTRUCTION = """\
Write a Python calculator with no extra imports. Support arithmetic expressions over integers and
parentheses. Save it as `calculator.py`.

Expose a `Calculator` class that can be called with a string expression.

Example:

from calculator import Calculator
cal = Calculator()
print(cal("4*3-3"))  # should print 9"""


def builder_spec_for_harness(harness: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if harness == "claude_code":
        config["ignore_patterns"] = [
            r"x-anthropic-billing-header:[^\n]+",
            r"<system-reminder>.*?</system-reminder>",
        ]
    return {"strategy": "prefix_merging", **({"config": config} if config else {})}


def evaluator_exclude_patterns_for_harness(harness: str) -> list[str]:
    patterns: list[str] = []
    if harness == "claude_code":
        patterns.extend(
            [
                "$HOME/.claude/**",
                "**/$HOME/.claude/**",
                ".claude/**",
                "**/.claude/**",
            ]
        )
    if harness == "swe_agent":
        patterns.extend(
            [
                "trajectories/**",
                "**/trajectories/**",
            ]
        )
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, help="Harness name (e.g., opencode)")
    parser.add_argument("--image", required=True, help="Docker image for the runtime")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "openai/MiniMaxAI/MiniMax-M2.5"),
        help="Model name for the agent harness",
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
    parser.add_argument("--num-rollouts", type=int, default=int(os.environ.get("NUM_ROLLOUTS", "16")))
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--runtime-backend",
        choices=["docker", "apptainer"],
        default=os.environ.get("RUNTIME_BACKEND", "docker"),
        help="Container runtime backend for the session",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for request/response files",
    )
    parser.add_argument("--docker-socket", action="store_true",
                        help="Mount Docker socket for agents that need DinD")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_task_request(args: argparse.Namespace) -> dict[str, Any]:
    test_file_abs = str(TEST_FILE.resolve())
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runtime_image = runtime_image_for_backend(args.image, args.runtime_backend)
    return {
        "task_id": f"calculator-{args.harness}-{batch_id}",
        "instruction": INSTRUCTION,
        "num_rollouts": args.num_rollouts,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": args.runtime_backend,
            "image": runtime_image,
            "prepare": [
                {
                    "type": "exec",
                    "command": (
                        "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
                        "cd /polar/session/workspace && git init && "
                        "git config user.email 'polar@test' && "
                        "git config user.name 'Polar'"
                    ),
                },
                {
                    "type": "upload_file",
                    "source": test_file_abs,
                    "target": "/polar/session/workspace/test_calculator.py",
                },
                {
                    "type": "exec",
                    "command": "cd /polar/session/workspace && git add -A && git commit -m 'initial'",
                },
            ],
            "env": {},
            "network": "host",
            "workdir": "/polar/session/workspace",
            **({"kwargs": {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]}} if args.docker_socket else {}),
        },
        "agent": {
            "harness": args.harness,
            "model_name": args.model_name,
            "settings": {},
            "env": {},
        },
        "builder": builder_spec_for_harness(args.harness),
        "evaluator": {
            "strategy": "swegym_git_diff",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": "cd /polar/session/workspace && python3 test_calculator.py && echo 'PASSED test_calculator'",
                "test_timeout": 60.0,
                "expected_output_json": {"test_calculator": "PASSED"},
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def main() -> int:
    args = parse_args()
    payload = build_task_request(args)

    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else (
        EXAMPLE_DIR / args.harness / "batches" / batch_id
    )
    request_path = output_dir / "request.json"
    response_path = output_dir / "response.json"
    write_json(request_path, payload)
    print(f"Wrote request to {request_path}")

    if args.dry_run:
        print("Dry run — not submitting to rollout server.")
        return 0

    command = [
        sys.executable,
        "-m",
        "polar.cli",
        "submit",
        str(request_path),
        "--json",
    ]
    if args.topology:
        command.extend(["-c", args.topology])
    if args.rollout_server_url:
        command.extend(["--rollout-url", args.rollout_server_url])

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    write_json(response_path, result)
    print(f"Task completed. Wrote response to {response_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
