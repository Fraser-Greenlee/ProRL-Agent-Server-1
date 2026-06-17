#!/usr/bin/env python3
"""Submit LiteResearcher tasks to the Polar rollout server.

Each task runs the shared ``notre_hermes`` agent with a ``search``/``visit`` MCP
server wired to a live LiteResearcher retrieval service, and is graded by the
``answer_judge`` evaluator (final ``<answer>`` vs. reference).

    uv run python examples/lite_researcher_notre/submit_lite_tasks.py --max-tasks 10
    uv run python examples/lite_researcher_notre/submit_lite_tasks.py --max-tasks 50 --num-samples 4 \
        --search-url http://127.0.0.1:8018/search --parser-url http://127.0.0.1:8018/web_parser

Stand up the retrieval service first (see README); it must be reachable from the
runtime containers (host network) at the configured URLs.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from dataset import DEFAULT_STAGE, load_literesearcher_data, question_id

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
MCP_SERVER_FILE = EXAMPLE_DIR / "literesearcher_mcp_server.py"
RUNTIME_MCP_PATH = "/polar/session/lite_tools/literesearcher_mcp_server.py"
POLL_INTERVAL_SECONDS = 15.0
HERMES_VERSION = "0.15.1"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default=DEFAULT_STAGE, help="LiteResearcher-Data stage (stage1/stage2).")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@k).")
    parser.add_argument("--max-tasks", type=int, default=10, help="Maximum tasks to submit. -1 = all.")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--runtime-image", default="polar-lite-researcher-runtime:latest")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--search-url", default="http://127.0.0.1:8018/search")
    parser.add_argument("--parser-url", default="http://127.0.0.1:8018/web_parser")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.6-27B",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    return parser.parse_args()


def build_task_request(args: argparse.Namespace, item: dict[str, Any], batch_id: str) -> dict[str, Any]:
    question = str(item["question"]).strip()
    instruction = (
        "Answer this research question. Use the `search` and `visit` tools to find "
        "and read sources.\n\n"
        f"Question: {question}\n\n"
        "Return the final answer exactly once in this format:\n<answer>your answer</answer>\n"
    )
    prepare = [
        {"type": "exec", "command": HERMES_INSTALL},
        {
            "type": "exec",
            "command": "mkdir -p /polar/session/workspace /polar/session/logs/agent /polar/session/lite_tools",
        },
        {"type": "upload_file", "source": str(MCP_SERVER_FILE), "target": RUNTIME_MCP_PATH},
    ]
    return {
        "task_id": f"lite-{question_id(question)}-{batch_id}",
        "instruction": instruction,
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout_seconds,
        "runtime": {
            "backend": args.runtime_backend,
            "image": args.runtime_image,
            "prepare": prepare,
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {
            "harness": "notre_hermes",
            "model_name": args.model_name,
            "env": {
                "LITERESEARCHER_SEARCH_URL": args.search_url,
                "LITERESEARCHER_PARSER_URL": args.parser_url,
            },
            "mcp_servers": [
                {
                    "name": "literesearcher",
                    "transport": "stdio",
                    "command": "python3",
                    "args": [RUNTIME_MCP_PATH],
                }
            ],
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "answer_judge",
            "config": {
                "reference": item["answer"],
                "exact_match": True,
                "substring_match": True,
            },
        },
        "metadata": {
            "domain": "deep_research",
            "environment": "LiteResearcher",
            "question": question,
            "data_source": item.get("data_source"),
            "mask_url": item.get("mask_url"),
        },
    }


def session_reward(session: dict[str, Any]) -> float | None:
    traces = (session.get("trajectory") or {}).get("traces") or []
    reward = traces[-1].get("reward") if traces else None
    return float(reward) if isinstance(reward, (int, float)) else None


def task_stats(result: dict[str, Any]) -> tuple[float, int, int]:
    sessions = result.get("results") or []
    rewards = [r for r in (session_reward(s) for s in sessions) if r is not None]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    positive = sum(1 for r in rewards if r > 0)
    return mean, positive, len(sessions)


def print_summary(stats: dict[str, tuple[float, int, int]], elapsed: float, topology: str) -> None:
    total_tasks = len(stats)
    total_sessions = sum(total for _, _, total in stats.values())
    positive = sum(pos for _, pos, _ in stats.values())
    mean_reward = sum(m for m, _, _ in stats.values()) / max(total_tasks, 1)

    print("\n" + "=" * 72)
    print("  LiteResearcher — Reward Summary")
    print("=" * 72)
    print(f"  Mean task reward:      {mean_reward:.3f}  over {total_tasks} task(s)")
    print(f"  Sessions reward>0:     {positive}/{total_sessions}  ({100 * positive / max(total_sessions, 1):.1f}%)")
    print(f"  Wall time:             {elapsed:.0f}s")
    print("=" * 72)
    print(f"\n  Per-session detail: polar dashboard -c {topology}")


def main() -> int:
    args = parse_args()
    if not MCP_SERVER_FILE.is_file():
        raise SystemExit(f"Missing MCP server: {MCP_SERVER_FILE}")
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    items = load_literesearcher_data(stage=args.stage)
    if args.max_tasks > 0:
        items = items[: args.max_tasks]
    if not items:
        raise SystemExit("No LiteResearcher items selected.")

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(args.topology).rollout.public_url
    print(f"Submitting {len(items)} task(s) to {rollout_url} "
          f"(stage={args.stage}, samples={args.num_samples}, search={args.search_url})")

    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for item in items:
            payload = build_task_request(args, item, batch_id)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[payload["task_id"]] = resp.json()["task_id"]

        print(f"Polling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        stats: dict[str, tuple[float, int, int]] = {}
        while len(stats) < len(task_ids):
            time.sleep(POLL_INTERVAL_SECONDS)
            for label, tid in task_ids.items():
                if label in stats:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    mean, positive, total = task_stats(status)
                    stats[label] = (mean, positive, total)
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {label:<28} reward={mean:.3f} "
                          f"positive={positive}/{total}  ({len(stats)}/{len(task_ids)} done)")
        elapsed = time.monotonic() - t0

    print_summary(stats, elapsed, args.topology)
    return 0


if __name__ == "__main__":
    sys.exit(main())
