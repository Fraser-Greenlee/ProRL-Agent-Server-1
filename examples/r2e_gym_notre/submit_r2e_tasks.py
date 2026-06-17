#!/usr/bin/env python3
"""Submit R2E-Gym-Subset tasks to the Polar rollout server.

Each task runs the shared ``notre_hermes`` agent in the instance's official
R2E-Gym image and is graded by the ``test_harness`` evaluator, which re-runs the
gold ``r2e_tests`` against the agent's patch in a fresh container.

    uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --max-tasks 10
    uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --max-tasks 50 --num-samples 4
    uv run python examples/r2e_gym_notre/submit_r2e_tasks.py --instance-id <instance_id>
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from dataset import (
    REPO_DIR,
    docker_image_for,
    load_r2e_gym_subset,
    sanitize_instance_id,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
POLL_INTERVAL_SECONDS = 15.0
HERMES_VERSION = "0.15.1"

# Hermes installs into an isolated venv and symlinks onto $HOME/.local/bin (the
# preset puts that on PATH). uv is preferred; plain venv is the fallback so this
# works on any Python-based R2E image.
HERMES_INSTALL = (
    "set -e; "
    "if command -v uv >/dev/null 2>&1; then "
    "  uv python install --quiet 3.11; "
    "  rm -rf /tmp/hermes-agent-venv; "
    "  uv venv --quiet --python 3.11 /tmp/hermes-agent-venv; "
    "  uv pip install --quiet --python /tmp/hermes-agent-venv/bin/python "
    f"  {shlex.quote(f'hermes-agent=={HERMES_VERSION}')}; "
    '  mkdir -p "$HOME/.local/bin"; '
    '  ln -sf /tmp/hermes-agent-venv/bin/hermes "$HOME/.local/bin/hermes"; '
    "else "
    "  rm -rf /tmp/hermes-agent-venv; "
    "  python3 -m venv /tmp/hermes-agent-venv; "
    "  /tmp/hermes-agent-venv/bin/python -m pip install --quiet --upgrade pip; "
    "  /tmp/hermes-agent-venv/bin/python -m pip install --quiet "
    f"  {shlex.quote(f'hermes-agent=={HERMES_VERSION}')}; "
    '  mkdir -p "$HOME/.local/bin"; '
    '  ln -sf /tmp/hermes-agent-venv/bin/hermes "$HOME/.local/bin/hermes"; '
    "fi"
)

# /testbed is already a git repo at the buggy commit; wire it for diffing and
# expose the gold suite (lives at /r2e_tests in the image) inside the repo.
_R2E_SETUP = (
    "mkdir -p /polar/session/logs/agent && "
    f"git config --global --add safe.directory {REPO_DIR} && "
    f"cd {REPO_DIR} && "
    "git config --local user.email 'polar@test' || true && "
    "git config --local user.name 'Polar' || true && "
    "if [ -d /r2e_tests ] && [ ! -e r2e_tests ]; then ln -s /r2e_tests r2e_tests; fi && "
    "chmod +x /r2e_tests/run_tests.sh 2>/dev/null || true"
)

_PATCH_COMMAND = (
    f"cd {REPO_DIR} && git add -N . >/dev/null 2>&1 || true && "
    "git diff --binary --submodule=diff"
)

_EXCLUDE_PATTERNS = [
    ".hermes/**",
    "**/.hermes/**",
    ".cache/**",
    "**/.cache/**",
    "node_modules/**",
    "**/node_modules/**",
    ".venv/**",
    "**/.venv/**",
    "r2e_tests/**",
    "**/r2e_tests/**",
    "run_tests.log",
    "**/run_tests.log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@k).")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Maximum tasks to submit. -1 = all.")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.6-27B",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    return parser.parse_args()


def runtime_image_for_backend(image: str, backend: str) -> str:
    if backend == "apptainer" and not image.startswith(("docker-daemon:", "docker://", "oras://")):
        return f"docker-daemon:{image}"
    return image


def docker_image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = load_r2e_gym_subset()
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


def build_instruction(instance: dict[str, Any]) -> str:
    problem = str(
        instance.get("problem_statement")
        or instance.get("instruction")
        or instance.get("prompt")
        or ""
    ).strip()
    if not problem:
        raise ValueError(f"R2E instance {instance['instance_id']} has no problem_statement")
    return (
        f"You are working in `{REPO_DIR}` inside the official R2E-Gym runtime image.\n\n"
        f"Task:\n{problem}\n\n"
        f"Run this test command before finishing:\n{instance['test_command']}\n\n"
        "Return a short final note in `<answer>...</answer>`.\n"
    )


def build_task_request(args: argparse.Namespace, instance: dict[str, Any], batch_id: str) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image = docker_image_for(instance)
    prepare = [
        {"type": "exec", "command": HERMES_INSTALL},
        {"type": "exec", "command": _R2E_SETUP},
    ]
    evaluator_config: dict[str, Any] = {
        "repo_dir": REPO_DIR,
        "patch_command": _PATCH_COMMAND,
        "test_command": instance["test_command"],
        "exclude_patterns": _EXCLUDE_PATTERNS,
        "test_timeout": 1800.0,
    }
    if instance.get("expected_output_json"):
        evaluator_config["expected_output_json"] = instance["expected_output_json"]

    return {
        "task_id": f"r2e-{sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": build_instruction(instance),
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": args.runtime_backend,
            "image": runtime_image_for_backend(image, args.runtime_backend),
            "prepare": prepare,
            "eval_prepare": prepare,
            "network": "host",
            "workdir": REPO_DIR,
        },
        "agent": {"harness": "notre_hermes", "model_name": args.model_name},
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "test_harness",
            "config": evaluator_config,
            "refresh_runtime": True,
        },
        "metadata": {
            "domain": "coding",
            "environment": "R2E-Gym",
            "instance_id": instance_id,
            "docker_image": image,
        },
    }


def task_stats(result: dict[str, Any]) -> tuple[int, int]:
    sessions = result.get("results") or []
    reward_one = 0
    for session in sessions:
        traces = (session.get("trajectory") or {}).get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return reward_one, len(sessions)


def print_summary(stats: dict[str, tuple[int, int]], elapsed: float, topology: str) -> None:
    total_tasks = len(stats)
    resolved = sum(1 for r1, _ in stats.values() if r1 > 0)
    total_sessions = sum(total for _, total in stats.values())
    reward_one = sum(r1 for r1, _ in stats.values())

    print("\n" + "=" * 72)
    print("  R2E-Gym-Subset — Reward Summary")
    print("=" * 72)
    print(f"  Tasks resolved (>=1):  {resolved}/{total_tasks}  ({100 * resolved / max(total_tasks, 1):.1f}%)")
    print(f"  Sessions reward=1:     {reward_one}/{total_sessions}  ({100 * reward_one / max(total_sessions, 1):.1f}%)")
    print(f"  Wall time:             {elapsed:.0f}s")
    print("=" * 72)
    print(f"\n  Per-session detail: polar dashboard -c {topology}")


def main() -> int:
    args = parse_args()
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    ready, missing = [], []
    for instance in instances:
        (ready if docker_image_exists(docker_image_for(instance)) else missing).append(instance)
    if not ready:
        raise SystemExit("No runtime images found. Run: python pull_images.py")
    if missing:
        print(f"Skipping {len(missing)} instance(s) with un-pulled images. Pull them with: python pull_images.py")
    instances = ready

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(args.topology).rollout.public_url
    print(f"Submitting {len(instances)} task(s) to {rollout_url} "
          f"(samples={args.num_samples}, backend={args.runtime_backend})")

    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for instance in instances:
            iid = str(instance["instance_id"])
            resp = client.post("/rollout/task/submit", json=build_task_request(args, instance, batch_id))
            resp.raise_for_status()
            task_ids[iid] = resp.json()["task_id"]

        print(f"Polling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        stats: dict[str, tuple[int, int]] = {}
        while len(stats) < len(task_ids):
            time.sleep(POLL_INTERVAL_SECONDS)
            for iid, tid in task_ids.items():
                if iid in stats:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    r1, total = task_stats(status)
                    stats[iid] = (r1, total)
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {iid:<45} resolved={r1}/{total}  "
                          f"({len(stats)}/{len(task_ids)} done)")
        elapsed = time.monotonic() - t0

    print_summary(stats, elapsed, args.topology)
    return 0


if __name__ == "__main__":
    sys.exit(main())
