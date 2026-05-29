#!/usr/bin/env python3
"""Run the calculator demo across every harness and print a comparison table.

Each harness gets a tiny `calculator.py` with parser stubs, edits it, and the
evaluator runs `python3 test_calculator.py`. All harnesses are submitted at
once; live progress and per-session detail are visible in the dashboard
(`polar dashboard -c examples/calculator/topology.yaml`).

    uv run python examples/calculator/run.py                 # docker (default)
    uv run python examples/calculator/run.py --backend apptainer
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = EXAMPLE_DIR / "assets"
TEST_FILE = ASSETS_DIR / "test_calculator.py"
STARTER_FILE = ASSETS_DIR / "calculator.py"
TOPOLOGY = EXAMPLE_DIR / "topology.yaml"
RUNTIME_IMAGE = "polar-localhost-calculator:latest"
NUM_SAMPLES = 1
TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 10.0

HARNESSES = ("claude_code", "codex", "gemini_cli", "opencode", "pi", "qwen_code")

INSTRUCTION = """\
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

# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_NPM_PACKAGE: dict[str, str] = {
    "claude_code": "@anthropic-ai/claude-code@2.1.111",
    "codex": "@openai/codex@0.121.0",
    "gemini_cli": "@google/gemini-cli@0.38.1",
    "opencode": "opencode-ai@1.4.6",
    "pi": "@mariozechner/pi-coding-agent@0.67.68",
    "qwen_code": "@qwen-code/qwen-code@0.14.5",
}

# Model name the harness CLI sends; the gateway rewrites it to the served model.
HARNESS_MODEL: dict[str, str] = {
    "claude_code": "claude-opus-4-5",
    "codex": "gpt-5.4",
    "gemini_cli": "gemini-2.5-flash-lite",
    "opencode": "openai/gpt-5.4",
    "pi": "openai/gpt-5.4",
    "qwen_code": "qwen3-coder-plus",
}

# INIT stage: install the harness CLI, then set up a clean git workspace.
_WORKSPACE_PREPARE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
    "cd /polar/session/workspace && "
    "git init -q && "
    "git config user.email 'polar@test' && "
    "git config user.name 'Polar'"
)

# Config/cache dirs that can leak into the workspace git diff.
_EVAL_EXCLUDES: dict[str, list[str]] = {
    "claude_code": [".claude/**", "**/.claude/**"],
    "codex": [".codex/**", "**/.codex/**"],
    "gemini_cli": [".gemini/**", "**/.gemini/**"],
    "opencode": [".opencode/**", "**/.opencode/**", ".config/opencode/**"],
    "pi": [".pi/**", "**/.pi/**"],
    "qwen_code": [".qwen/**", "**/.qwen/**"],
}
_COMMON_EXCLUDES = ["node_modules/**", "**/node_modules/**", ".cache/**", "**/.cache/**", ".venv/**", "**/.venv/**"]


def runtime_image_for_backend(backend: str) -> str:
    if backend == "apptainer":
        return f"docker-daemon:{RUNTIME_IMAGE}"
    return RUNTIME_IMAGE


def build_task_payload(harness: str, batch_id: str, backend: str) -> dict[str, Any]:
    return {
        "task_id": f"calculator-{harness}-{batch_id}",
        "instruction": INSTRUCTION,
        "num_samples": NUM_SAMPLES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "runtime": {
            "backend": backend,
            "image": runtime_image_for_backend(backend),
            "prepare": [
                {"type": "exec", "command": f"npm install -g {HARNESS_NPM_PACKAGE[harness]} && {_WORKSPACE_PREPARE}"},
                {"type": "upload_file", "source": str(TEST_FILE), "target": "/polar/session/workspace/test_calculator.py"},
                {"type": "upload_file", "source": str(STARTER_FILE), "target": "/polar/session/workspace/calculator.py"},
                {"type": "exec", "command": "cd /polar/session/workspace && git add -A && git commit -qm 'initial'"},
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {"harness": harness, "model_name": HARNESS_MODEL[harness]},
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "test_on_output",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": "cd /polar/session/workspace && python3 test_calculator.py && echo 'PASSED test_calculator'",
                "test_timeout": 60.0,
                "expected_output_json": {"test_calculator": "PASSED"},
                "exclude_patterns": [*_COMMON_EXCLUDES, *_EVAL_EXCLUDES[harness]],
            },
            "refresh_runtime": True,
        },
    }


def session_reward(session: dict[str, Any]) -> float | None:
    traces = (session.get("trajectory") or {}).get("traces") or []
    reward = traces[-1].get("reward") if traces else None
    return float(reward) if isinstance(reward, (int, float)) else None


def print_comparison(finished: dict[str, dict[str, Any]], elapsed: float) -> None:
    header = f"{'Harness':<16} {'Reward':>8}  {'Done':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for harness, result in finished.items():
        sessions = result.get("results") or []
        rewards = [r for r in (session_reward(s) for s in sessions) if r is not None]
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        done = sum(1 for s in sessions if s.get("status") == "COMPLETED")
        print(f"{harness:<16} {mean:>8.3f}  {done:>2}/{len(sessions):<2}")
    print("=" * len(header))
    print(f"Wall time: {elapsed:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    backend = parser.parse_args().backend

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(TOPOLOGY).rollout.public_url
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"Submitting {len(HARNESSES)} harnesses to {rollout_url} (backend={backend})")
    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for harness in HARNESSES:
            payload = build_task_payload(harness, batch_id, backend)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[harness] = resp.json()["task_id"]
            print(f"  {harness:<16} -> {task_ids[harness]}")

        print(f"\nPolling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        finished: dict[str, dict[str, Any]] = {}
        while len(finished) < len(HARNESSES):
            time.sleep(POLL_INTERVAL_SECONDS)
            for harness, tid in task_ids.items():
                if harness in finished:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    finished[harness] = status
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {harness} done")
        elapsed = time.monotonic() - t0

    print_comparison(finished, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
