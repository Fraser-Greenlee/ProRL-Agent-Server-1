#!/usr/bin/env python3
"""Submit all 7 harnesses (4 samples each) in one shot and print a combined summary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Reuse everything from the single-harness script
from submit_calculator_task import (
    BASE_INSTRUCTION,
    EXAMPLE_DIR,
    SUPPORTED_HARNESSES,
    TEST_FILE,
    STARTER_FILE,
    agent_spec_for_harness,
    builder_spec_for_harness,
    evaluator_exclude_patterns_for_harness,
    prepare_command_for_harness,
    runtime_image_for_backend,
    summarize_result,
    write_json,
)

DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.yaml"
DEFAULT_IMAGE = "polar-localhost-calculator:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--model-name", help="Override model for all harnesses")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--rollout-url", help="Override rollout server URL")
    parser.add_argument(
        "--harness",
        nargs="+",
        choices=SUPPORTED_HARNESSES,
        default=list(SUPPORTED_HARNESSES),
        help="Subset of harnesses to run (default: all 7)",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0)
    return parser.parse_args()


def resolve_rollout_url(topology_path: str | None, override: str | None) -> str:
    if override:
        return override
    if topology_path:
        from polar.config import TopologyConfig
        topo = TopologyConfig.load(topology_path)
        return topo.rollout.public_url
    return "http://127.0.0.1:8080"


def build_task_payload(
    harness: str,
    batch_id: str,
    *,
    image: str,
    backend: str,
    num_samples: int,
    timeout_seconds: float,
    model_name: str | None,
) -> dict[str, Any]:
    runtime_image = runtime_image_for_backend(image, backend)
    return {
        "task_id": f"calculator-{harness}-{batch_id}",
        "instruction": BASE_INSTRUCTION,
        "num_samples": num_samples,
        "timeout_seconds": timeout_seconds,
        "runtime": {
            "backend": backend,
            "image": runtime_image,
            "prepare": [
                {"type": "exec", "command": prepare_command_for_harness(harness)},
                {"type": "upload_file", "source": str(TEST_FILE.resolve()),
                 "target": "/polar/session/workspace/test_calculator.py"},
                {"type": "upload_file", "source": str(STARTER_FILE.resolve()),
                 "target": "/polar/session/workspace/calculator.py"},
                {"type": "exec",
                 "command": "cd /polar/session/workspace && git add -A && git commit -qm 'initial'"},
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": agent_spec_for_harness(harness, model_name),
        "builder": builder_spec_for_harness(harness),
        "evaluator": {
            "strategy": "output_unit_tests",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": (
                    "cd /polar/session/workspace && python3 test_calculator.py "
                    "&& echo 'PASSED test_calculator'"
                ),
                "test_timeout": 240.0,
                "expected_output_json": {"test_calculator": "PASSED"},
                "exclude_patterns": evaluator_exclude_patterns_for_harness(harness),
            },
            "refresh_runtime": True,
        },
    }


def print_combined_summary(
    results: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    elapsed: float,
) -> None:
    header = f"{'Harness':<16} {'Rewards':<28} {'Mean':>6}  {'Done':>6}  {'Err':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for harness in results:
        s = summaries[harness]
        rtext = ", ".join(
            "n/a" if r is None else f"{r:.1f}" for r in s["rewards"]
        )
        print(
            f"{harness:<16} [{rtext:<26}] "
            f"{s['reward_mean']:>5.3f}  "
            f"{s['completed_sessions']:>2}/{s['total_sessions']:<2}  "
            f"{s['errors'] or '':>4}"
        )
    print("-" * len(header))
    all_rewards = [r for s in summaries.values() for r in s["rewards"] if r is not None]
    total_done = sum(s["completed_sessions"] for s in summaries.values())
    total_all = sum(s["total_sessions"] for s in summaries.values())
    mean = sum(all_rewards) / max(1, len(all_rewards))
    print(f"{'TOTAL':<16} {'':28} {mean:>5.3f}  {total_done:>2}/{total_all:<2}")
    print(f"Wall time: {elapsed:.0f}s")
    print("=" * len(header))


def main() -> int:
    args = parse_args()
    harnesses: list[str] = args.harness
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rollout_url = resolve_rollout_url(args.topology, args.rollout_url)
    batch_dir = EXAMPLE_DIR / "batches" / batch_id

    n_total = len(harnesses) * args.num_samples
    print(f"Submitting {len(harnesses)} harnesses x {args.num_samples} samples = {n_total} sessions")
    print(f"Rollout URL: {rollout_url}")

    # 1. Build and submit all tasks (async endpoint)
    timeout = httpx.Timeout(None, connect=30.0)
    task_ids: dict[str, str] = {}  # harness -> task_id
    payloads: dict[str, dict[str, Any]] = {}

    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        for harness in harnesses:
            payload = build_task_payload(
                harness, batch_id,
                image=args.image,
                backend=args.runtime_backend,
                num_samples=args.num_samples,
                timeout_seconds=args.timeout_seconds,
                model_name=args.model_name,
            )
            payloads[harness] = payload
            out_dir = batch_dir / harness
            write_json(out_dir / "request.json", payload)

            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            data = resp.json()
            task_ids[harness] = data["task_id"]
            print(f"  {harness:<16} -> {data['task_id']}")

        # 2. Poll until all tasks finish
        print(f"\nPolling every {args.poll_interval:.0f}s ...")
        t0 = time.monotonic()
        finished: dict[str, dict[str, Any]] = {}

        while len(finished) < len(harnesses):
            time.sleep(args.poll_interval)
            sessions_done = sum(s["completed_sessions"] for s in finished.values())
            newly_done: list[str] = []
            for harness, tid in task_ids.items():
                if harness in finished:
                    continue
                resp = client.get(f"/rollout/task/{tid}")
                resp.raise_for_status()
                task_status = resp.json()
                sessions_done += task_status["completed_sessions"]
                if task_status["status"] != "running":
                    finished[harness] = task_status
                    newly_done.append(harness)

            elapsed = time.monotonic() - t0
            if newly_done:
                # Clear progress line then print completion
                sys.stdout.write("\r" + " " * 60 + "\r")
                for h in newly_done:
                    d = finished[h]["completed_sessions"]
                    t = finished[h]["total_sessions"]
                    print(f"  [{elapsed:>5.0f}s] {h:<16} done ({d}/{t})")
            else:
                sys.stdout.write(
                    f"\r  [{elapsed:>5.0f}s] {sessions_done}/{n_total} sessions, "
                    f"{len(finished)}/{len(harnesses)} tasks done"
                )
                sys.stdout.flush()

        elapsed = time.monotonic() - t0
        print()

    # 3. Save results and print summary
    summaries: dict[str, dict[str, Any]] = {}
    for harness in harnesses:
        result = finished[harness]
        out_dir = batch_dir / harness
        write_json(out_dir / "response.json", result)
        summary = summarize_result(result)
        write_json(out_dir / "summary.json", summary)
        summaries[harness] = summary

    print_combined_summary(finished, summaries, elapsed)
    print(f"\nResults saved to {batch_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
