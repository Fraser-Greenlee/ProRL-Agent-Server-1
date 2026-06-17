#!/usr/bin/env python3
"""Submit CUA-Gym web tasks to the Polar rollout server.

Each task serves its mock web app via Vite inside the runtime, runs the shared
``notre_hermes`` agent (driving the UI through Hermes' browser toolset), and is
graded by the ``env_state`` evaluator, which runs the task's official
``reward.py`` against the live app state.

    uv run python examples/cua_gym_notre/submit_cua_tasks.py --max-tasks 10
    uv run python examples/cua_gym_notre/submit_cua_tasks.py --max-tasks 50 --num-samples 4

Run prepare_data.py first to clone the apps and materialize task bundles.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
DEFAULT_DATA = EXAMPLE_DIR / "data" / "cua_web_tasks.jsonl"
POLL_INTERVAL_SECONDS = 15.0
HERMES_VERSION = "0.15.1"

# Hermes with the [web] extras (browser toolset) in an isolated venv.
HERMES_INSTALL_WEB = (
    "set -e; "
    "if command -v uv >/dev/null 2>&1; then "
    "  uv python install --quiet 3.11; "
    "  rm -rf /tmp/hermes-agent-venv; "
    "  uv venv --quiet --python 3.11 /tmp/hermes-agent-venv; "
    "  uv pip install --quiet --python /tmp/hermes-agent-venv/bin/python "
    f"  {shlex.quote(f'hermes-agent[web]=={HERMES_VERSION}')}; "
    '  mkdir -p "$HOME/.local/bin"; '
    '  ln -sf /tmp/hermes-agent-venv/bin/hermes "$HOME/.local/bin/hermes"; '
    "else "
    "  rm -rf /tmp/hermes-agent-venv; "
    "  python3 -m venv /tmp/hermes-agent-venv; "
    "  /tmp/hermes-agent-venv/bin/python -m pip install --quiet --upgrade pip; "
    "  /tmp/hermes-agent-venv/bin/python -m pip install --quiet "
    f"  {shlex.quote(f'hermes-agent[web]=={HERMES_VERSION}')}; "
    '  mkdir -p "$HOME/.local/bin"; '
    '  ln -sf /tmp/hermes-agent-venv/bin/hermes "$HOME/.local/bin/hermes"; '
    "fi"
)

# agent-browser drives `google-chrome`; point it at the image's chromium.
_CHROME_SETUP = (
    "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
    "if ! command -v google-chrome >/dev/null 2>&1; then "
    "  chrome_bin=$(command -v chromium || command -v chromium-browser || true); "
    '  if [ -n "$chrome_bin" ]; then sudo ln -sf "$chrome_bin" /usr/local/bin/google-chrome; fi; '
    "fi"
)


def _start_vite_command(port: int) -> str:
    port = int(port)
    return (
        "cd /polar/session/cua_app && "
        f"(pkill -f '[v]ite.*--port {port}' || true) && "
        "(npm ci --silent || npm install --silent) && "
        f"nohup npm run dev -- --host 0.0.0.0 --port {port} --strictPort "
        "> /polar/session/logs/cua_vite.log 2>&1 & "
        "python3 - <<'PY'\n"
        "import time, urllib.request\n"
        f"url='http://127.0.0.1:{port}/'\n"
        "for _ in range(120):\n"
        "    try:\n"
        "        urllib.request.urlopen(url, timeout=1).read(); break\n"
        "    except Exception:\n"
        "        time.sleep(1)\n"
        "else:\n"
        "    raise SystemExit('CUA web app did not become ready')\n"
        "PY"
    )


def _post_state_command(port: int, sid: str, state: Any) -> str:
    payload = json.dumps({"action": "set", "state": state}, ensure_ascii=True)
    return (
        "python3 - <<'PY'\n"
        "import json, urllib.request\n"
        f"url='http://127.0.0.1:{int(port)}/post?sid={sid}'\n"
        f"payload={payload!r}.encode('utf-8')\n"
        "req=urllib.request.Request(url, data=payload, "
        "headers={'Content-Type':'application/json'}, method='POST')\n"
        "print(urllib.request.urlopen(req, timeout=30).read().decode())\n"
        "PY"
    )


def _get_state_command(base_url: str, sid: str) -> str:
    url = f"{base_url.rstrip('/')}/go?sid={sid}"
    return (
        "python3 - <<'PY'\n"
        "import urllib.request\n"
        f"print(urllib.request.urlopen({url!r}, timeout=30).read().decode())\n"
        "PY"
    )


def _cleanup_vite_command(port: int) -> str:
    return f"pkill -f '[v]ite.*--port {int(port)}' || true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-jsonl", default=str(DEFAULT_DATA))
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@k).")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Maximum tasks to submit. -1 = all.")
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--runtime-image", default="polar-cua-gym-runtime:latest")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.6-27B",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    return parser.parse_args()


def build_task_request(args: argparse.Namespace, item: dict[str, Any], batch_id: str) -> dict[str, Any]:
    instruction = str(item.get("instruction") or item.get("task") or "").strip()
    if not instruction:
        raise ValueError(f"CUA task {item.get('id')} has no instruction")
    sid = str(item["sid"])
    port = int(item["port"])
    base_url = str(item.get("base_url") or f"http://127.0.0.1:{port}")
    start_url = f"{base_url.rstrip('/')}/?sid={sid}"
    initial_state = item.get("initial_state") or item.get("state")

    prepare: list[dict[str, Any]] = [
        {"type": "exec", "command": HERMES_INSTALL_WEB},
        {"type": "exec", "command": _CHROME_SETUP},
        {"type": "upload_dir", "source": str(item["app_source"]), "target": "/polar/session/cua_app"},
        {"type": "exec", "command": _start_vite_command(port)},
    ]
    if initial_state is not None:
        prepare.append({"type": "exec", "command": _post_state_command(port, sid, initial_state)})
    prepare.append(
        {"type": "upload_dir", "source": str(item["bundle_dir"]), "target": "/polar/session/cua_task"}
    )
    prepare.append(
        {"type": "exec", "command": "cd /polar/session/cua_task && python3 initial_setup.py"}
    )

    full_instruction = (
        f"Open this web application in the browser: {start_url}\n\n"
        f"Task: {instruction}\n\n"
        "Use the browser automation tools to complete the task in the UI. "
        "When finished, reply:\n<answer>done</answer>\n"
    )

    return {
        "task_id": f"cua-{sid}-{batch_id}",
        "instruction": full_instruction,
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
            "settings": {"postrun_commands": [_cleanup_vite_command(port)]},
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "env_state",
            "config": {
                "sid": sid,
                "state_command": _get_state_command(base_url, sid),
                "reward_command": "cd /polar/session/cua_task && python3 reward.py",
            },
        },
        "metadata": {
            "domain": "computer_use_web",
            "environment": "CUA-Gym",
            "sid": sid,
            "url": start_url,
            "port": port,
            "app_type": item.get("app_type"),
            "difficulty": item.get("difficulty"),
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
    print("  CUA-Gym Web — Reward Summary")
    print("=" * 72)
    print(f"  Mean task reward:      {mean_reward:.3f}  over {total_tasks} task(s)")
    print(f"  Sessions reward>0:     {positive}/{total_sessions}  ({100 * positive / max(total_sessions, 1):.1f}%)")
    print(f"  Wall time:             {elapsed:.0f}s")
    print("=" * 72)
    print(f"\n  Per-session detail: polar dashboard -c {topology}")


def load_tasks(data_path: Path, max_tasks: int) -> list[dict[str, Any]]:
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; run prepare_data.py first")
    tasks: list[dict[str, Any]] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
            if 0 < max_tasks <= len(tasks):
                break
    return tasks


def main() -> int:
    args = parse_args()
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    items = load_tasks(Path(args.data_jsonl), args.max_tasks)
    if not items:
        raise SystemExit("No CUA tasks selected.")

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(args.topology).rollout.public_url
    print(f"Submitting {len(items)} task(s) to {rollout_url} "
          f"(samples={args.num_samples}, backend={args.runtime_backend})")

    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for item in items:
            payload = build_task_request(args, item, batch_id)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[str(item["id"])] = resp.json()["task_id"]

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
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {label:<32} reward={mean:.3f} "
                          f"positive={positive}/{total}  ({len(stats)}/{len(task_ids)} done)")
        elapsed = time.monotonic() - t0

    print_summary(stats, elapsed, args.topology)
    return 0


if __name__ == "__main__":
    sys.exit(main())
