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
STARTER_FILE = ASSETS_DIR / "calculator.py"
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.yaml"

BASE_INSTRUCTION = """\
`calculator.py` has a `Calculator` class with a tokenizer and three stub methods.
Each stub is marked with a `# TODO` comment and returns `0`.

Implement the three methods to build a recursive-descent expression parser:

1. `_parse_expr`  — handle `+` and `-` by calling `_parse_term`
2. `_parse_term`  — handle `*` and `/` (integer division) by calling `_parse_factor`
3. `_parse_factor` — handle integer literals and parenthesized sub-expressions

Also fix `__call__` to return the parsed value instead of `0`.

Requirements:
- Work only in `/polar/session/workspace/calculator.py`.
- Keep the existing file structure, `_tokenize`, `_peek`, and `_consume` as-is.
- Do not add imports.
- Use `//` for division (integer division).
- You must make actual edits. An empty git diff fails the task.

After editing, run `python3 test_calculator.py` and stop.

These checks must pass exactly:
- `cal("4*3-3") == 9`
- `cal("(2+3)*4") == 20`
- `cal("10/2+7") == 12`
- `cal("18-(3*4)") == 6`
- `cal(" 8 + 2 * 5 ") == 18`
"""

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


def model_name_for_harness(harness: str, override: str | None) -> str | None:
    if override:
        return override
    defaults = {
        "codex": "openai/gpt-5.4",
        "claude_code": "anthropic/claude-opus-4-5",
        "gemini_cli": "gcp/google/gemini-2.5-flash-lite",
        "openhands_sdk": "openai/gpt-5.4",
        "qwen_code": "Qwen/Qwen3.5-4B",
        "swe_agent": "openai/gpt-5.4",
    }
    return defaults.get(harness)


def agent_settings_for_harness(harness: str) -> dict[str, Any]:
    if harness == "claude_code":
        return {
            "max_turns": 8,
            "max_thinking_tokens": 2048,
            "append_system_prompt": (
                "For short single-file tasks, prefer one Read, one Edit covering all required "
                "changes, then run `python3 test_calculator.py`. Do not stop after describing "
                "a tool call in text; emit the actual tool call. For this task, `_parse_expr` "
                "must handle both `+` and `-`, `_parse_term` must handle `*` and `/` with "
                "integer division, `_parse_factor` must handle integers and parenthesized "
                "expressions, and `__call__` must return `value`. Include the `return value` "
                "change in the same first Edit as the parser-method changes. Use one loop over "
                "`('+', '-')` in `_parse_expr` and one loop over `('*', '/')` in `_parse_term`."
            ),
        }
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, help="Harness name (e.g., opencode)")
    parser.add_argument("--image", required=True, help="Docker image for the runtime")
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME"),
        help="Optional model name override for the agent harness",
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
    parser.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "16")))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
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
    starter_file_abs = str(STARTER_FILE.resolve())
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    runtime_image = runtime_image_for_backend(args.image, args.runtime_backend)
    model_name = model_name_for_harness(args.harness, args.model_name)
    return {
        "task_id": f"calculator-{args.harness}-{batch_id}",
        "instruction": BASE_INSTRUCTION,
        "num_samples": args.num_samples,
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
                    "type": "upload_file",
                    "source": starter_file_abs,
                    "target": "/polar/session/workspace/calculator.py",
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
            "model_name": model_name,
            "settings": agent_settings_for_harness(args.harness),
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
