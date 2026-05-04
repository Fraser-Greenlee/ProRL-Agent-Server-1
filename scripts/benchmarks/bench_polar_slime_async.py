#!/usr/bin/env python3
"""Fast local benchmark for the Polar <-> Slime async rollout bridge.

This benchmark intentionally avoids Megatron, SGLang, Docker, and real GPU
work.  It drives ``slime_bridge.rollout.generate_rollout_polar_async`` through
a fake Polar HTTP server so scheduler/correctness regressions show up in
seconds before running the expensive end-to-end SWEGym job.

Each scenario runs in a child process.  If the current implementation hangs
while waiting for a batch, the parent kills only that scenario and records a
timeout instead of wedging the benchmark process.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import uvicorn
from fastapi import FastAPI, Request

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SLIME = ROOT / "slime"
for path in (SRC, SLIME, ROOT):
    if path.exists():
        sys.path.insert(0, str(path))

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming, TaskResult
from polar.trajectory.models import Trace, Trajectory

logger = logging.getLogger("polar_slime_async_bench")


@dataclass
class ScenarioResult:
    name: str
    status: str
    elapsed_s: float
    metrics: dict[str, Any]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "completed" and not self.failures

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


class FakeDataSource:
    """Slime-like prompt source with deterministic group ids."""

    def __init__(self, *, total_groups: int, samples_per_group: int = 1) -> None:
        self.total_groups = total_groups
        self.samples_per_group = samples_per_group
        self._next_group = 0
        self._lock = threading.Lock()
        self.produced_groups: list[int] = []

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        groups: list[list[Any]] = []
        with self._lock:
            while len(groups) < num_samples and self._next_group < self.total_groups:
                group_id = self._next_group
                self._next_group += 1
                self.produced_groups.append(group_id)
                groups.append([
                    SimpleNamespace(
                        group_index=group_id,
                        index=session_idx,
                        label="",
                        metadata={"group_id": group_id},
                        prompt=[{
                            "role": "user",
                            "content": f"bench prompt group={group_id} sample={session_idx}",
                        }],
                        response="",
                        status=None,
                    )
                    for session_idx in range(self.samples_per_group)
                ])
        return groups


class FakePolarServer:
    """Small Polar-compatible server for rollout bridge benchmarks."""

    def __init__(
        self,
        *,
        scenario: str,
        latency_s: Callable[[str, int], float],
        status_for_attempt: Callable[[str, int], str],
        empty_for_attempt: Callable[[str, int], bool] | None = None,
        missing_logprobs: bool = False,
    ) -> None:
        self.scenario = scenario
        self.latency_s = latency_s
        self.status_for_attempt = status_for_attempt
        self.empty_for_attempt = empty_for_attempt or (lambda _task_id, _attempt: False)
        self.missing_logprobs = missing_logprobs

        self.app = FastAPI()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._started = threading.Event()
        self._base_url: str | None = None

        self._tasks: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, int] = {}
        self._submitted_task_ids: list[str] = []
        self._completed_task_ids: list[str] = []
        self._callback_failures = 0
        self._inflight = 0
        self._max_inflight = 0
        self._events: list[tuple[float, str, str]] = []
        self._install_routes()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("fake server has not started")
        return self._base_url

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"fake-polar-{self.scenario}", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("fake Polar server did not start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "callback_failures": self._callback_failures,
                "completed_task_ids": list(self._completed_task_ids),
                "completed_tasks": len(self._completed_task_ids),
                "events": list(self._events),
                "max_server_inflight": self._max_inflight,
                "submitted_task_ids": list(self._submitted_task_ids),
                "submitted_tasks": len(self._submitted_task_ids),
            }

    def _run(self) -> None:
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)

        async def _serve() -> None:
            assert self._server is not None
            serve_task = asyncio.create_task(self._server.serve())
            while not self._server.started:
                await asyncio.sleep(0.01)
            sockets = self._server.servers[0].sockets
            port = sockets[0].getsockname()[1]
            self._base_url = f"http://127.0.0.1:{port}"
            self._started.set()
            await serve_task

        asyncio.run(_serve())

    def _install_routes(self) -> None:
        @self.app.post("/rollout/task/submit")
        async def submit_task(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = str(payload["task_id"])
            with self._lock:
                attempt = self._attempts.get(task_id, 0) + 1
                self._attempts[task_id] = attempt
                self._submitted_task_ids.append(task_id)
                self._tasks[task_id] = {
                    "task_id": task_id,
                    "status": "running",
                    "total_sessions": int(payload.get("num_samples") or 1),
                    "completed_sessions": 0,
                    "results": [],
                    "result_paths": [],
                }
                self._inflight += 1
                self._max_inflight = max(self._max_inflight, self._inflight)
                self._events.append((time.monotonic(), "submit", task_id))

            asyncio.create_task(self._complete_task(payload, attempt))
            return {"task_id": task_id, "status": "running"}

        @self.app.get("/rollout/task/{task_id}")
        async def get_task(task_id: str) -> dict[str, Any]:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "total_sessions": 0,
                        "completed_sessions": 0,
                        "results": [],
                        "result_paths": [],
                    }
                return json.loads(json.dumps(task, default=str))

    async def _complete_task(self, payload: dict[str, Any], attempt: int) -> None:
        task_id = str(payload["task_id"])
        await asyncio.sleep(self.latency_s(task_id, attempt))
        status = self.status_for_attempt(task_id, attempt)
        empty = self.empty_for_attempt(task_id, attempt)
        if status == "completed" and not empty:
            results = self._build_results(
                task_id=task_id,
                num_samples=int(payload.get("num_samples") or 1),
                attempt=attempt,
            )
        else:
            results = []

        task_result = {
            "task_id": task_id,
            "status": status,
            "results": [r.model_dump(mode="json") for r in results],
            "result_paths": [],
        }
        task_status = {
            **task_result,
            "total_sessions": int(payload.get("num_samples") or 1),
            "completed_sessions": len(results),
        }
        with self._lock:
            self._tasks[task_id] = task_status
            self._completed_task_ids.append(task_id)
            self._inflight -= 1
            self._events.append((time.monotonic(), status, task_id))

        callback_url = payload.get("callback_url")
        if callback_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(callback_url, json=task_result)
                    response.raise_for_status()
            except Exception:
                with self._lock:
                    self._callback_failures += 1

    def _build_results(self, *, task_id: str, num_samples: int, attempt: int) -> list[SessionResult]:
        group_id = _parse_group_id(task_id)
        results: list[SessionResult] = []
        for session_idx in range(num_samples):
            response_ids = [1000 + group_id * 10 + session_idx, 2000 + attempt]
            response_logprobs = None if self.missing_logprobs else [
                {"token": "x", "token_id": response_ids[0], "logprob": -0.10},
                {"token": "y", "token_id": response_ids[1], "logprob": -0.20},
            ]
            trace = Trace(
                prompt_ids=[11, 22, group_id],
                response_ids=response_ids,
                prompt_messages=[{"role": "user", "content": f"group {group_id}"}],
                response_messages=[{"role": "assistant", "content": f"answer {group_id}/{session_idx}"}],
                response_logprobs=response_logprobs,
                finish_reason="stop",
                reward=float(group_id % 2),
            )
            trajectory = Trajectory(
                status="COMPLETED",
                traces=[trace],
                metadata={
                    "bench": {
                        "attempt": attempt,
                        "group_id": group_id,
                        "scenario": self.scenario,
                    }
                },
            )
            results.append(SessionResult(
                session_id=f"bench-session-{task_id}-{session_idx}",
                task_id=task_id,
                status=SessionStatus.COMPLETED,
                trajectory=trajectory,
                timing=SessionTiming(run_ms=self.latency_s(task_id, attempt) * 1000),
                node_id="fake-node",
            ))
        return results


def _parse_group_id(task_id: str) -> int:
    try:
        return int(task_id.rsplit("-", 1)[-1])
    except Exception:
        return -1


def _make_args(
    server: FakePolarServer,
    *,
    batch_size: int,
    samples_per_group: int,
    max_concurrency: int,
) -> SimpleNamespace:
    if max_concurrency % batch_size != 0:
        raise ValueError("benchmark max_concurrency must be divisible by batch_size")
    max_async_level = max_concurrency // batch_size
    return SimpleNamespace(
        context_parallel_size=1,
        hf_checkpoint=None,
        max_tokens_per_gpu=None,
        polar_add_generation_prompt=True,
        polar_eval_dataset_name="polar_eval",
        polar_instruction_template=None,
        polar_max_async_level=max_async_level,
        polar_request_timeout=10.0,
        polar_reward_key="score",
        polar_rollout_url=server.base_url,
        polar_task_id_template=f"bench-{server.scenario}-{{rollout_id}}-{{sample.metadata.group_id}}",
        polar_task_template={
            "agent": {"harness": "bench_agent"},
            "metadata": {"group_id": "{sample.metadata.group_id}"},
            "timeout_seconds": 30.0,
        },
        n_samples_per_prompt=samples_per_group,
        reward_key="score",
        rollout_batch_size=batch_size,
        sglang_router_ip=None,
        sglang_router_port=None,
    )


def _run_rollout_cycles(
    *,
    scenario: str,
    total_groups: int,
    samples_per_group: int,
    batch_size: int,
    max_concurrency: int,
    cycles: int,
    latency_s: Callable[[str, int], float],
    status_for_attempt: Callable[[str, int], str],
    empty_for_attempt: Callable[[str, int], bool] | None = None,
) -> ScenarioResult:
    from slime_bridge.rollout import generate_rollout_polar_async, stop_global_worker

    start = time.monotonic()
    server = FakePolarServer(
        scenario=scenario,
        latency_s=latency_s,
        status_for_attempt=status_for_attempt,
        empty_for_attempt=empty_for_attempt,
    )
    data_source = FakeDataSource(total_groups=total_groups, samples_per_group=samples_per_group)
    returned_task_ids: list[str] = []
    returned_groups = 0
    returned_samples = 0
    cycle_elapsed: list[float] = []
    failures: list[str] = []

    try:
        server.start()
        args = _make_args(
            server,
            batch_size=batch_size,
            samples_per_group=samples_per_group,
            max_concurrency=max_concurrency,
        )
        for rollout_id in range(cycles):
            cycle_start = time.monotonic()
            output = generate_rollout_polar_async(args, rollout_id, data_source, evaluation=False)
            cycle_elapsed.append(time.monotonic() - cycle_start)
            returned_groups += len(output.samples)
            for group in output.samples:
                returned_samples += len(group)
                task_id = _task_id_from_group(group)
                if task_id:
                    returned_task_ids.append(task_id)
    finally:
        stop_global_worker()
        server.stop()

    elapsed = time.monotonic() - start
    server_metrics = server.snapshot()
    completed_task_ids = server_metrics["completed_task_ids"]
    unique_returned = sorted(set(returned_task_ids))
    unique_completed = sorted(set(completed_task_ids))
    not_returned = sorted(set(unique_completed) - set(unique_returned))

    expected_groups = cycles * batch_size
    if returned_groups != expected_groups:
        failures.append(f"expected {expected_groups} returned groups, got {returned_groups}")
    if not_returned:
        failures.append(f"{len(not_returned)} completed tasks were not returned to training")

    metrics = {
        **server_metrics,
        "accepted_groups": returned_groups,
        "accepted_samples": returned_samples,
        "accepted_task_ids": unique_returned,
        "completed_not_returned_task_ids": not_returned,
        "cycle_elapsed_s": cycle_elapsed,
        "data_source_groups_produced": len(data_source.produced_groups),
        "data_source_group_ids": list(data_source.produced_groups),
        "elapsed_s": elapsed,
        "groups_per_s": returned_groups / elapsed if elapsed > 0 else 0.0,
        "sessions_per_s": returned_samples / elapsed if elapsed > 0 else 0.0,
    }
    return ScenarioResult(
        name=scenario,
        status="completed",
        elapsed_s=elapsed,
        metrics=metrics,
        failures=failures,
    )


def _task_id_from_group(group: list[Any]) -> str | None:
    if not group:
        return None
    meta = getattr(group[0], "metadata", {}) or {}
    polar = meta.get("polar", {}) if isinstance(meta, dict) else {}
    task_id = polar.get("task_id")
    return str(task_id) if task_id else None


def scenario_steady_state_smoke() -> ScenarioResult:
    return _run_rollout_cycles(
        scenario="steady_state_smoke",
        total_groups=8,
        samples_per_group=2,
        batch_size=4,
        max_concurrency=4,
        cycles=2,
        latency_s=lambda task_id, _attempt: 0.03 + (_parse_group_id(task_id) % 3) * 0.01,
        status_for_attempt=lambda _task_id, _attempt: "completed",
    )


def scenario_overflow_buffering() -> ScenarioResult:
    return _run_rollout_cycles(
        scenario="overflow_buffering",
        total_groups=6,
        samples_per_group=1,
        batch_size=2,
        max_concurrency=6,
        cycles=3,
        latency_s=lambda _task_id, _attempt: 0.02,
        status_for_attempt=lambda _task_id, _attempt: "completed",
    )


def scenario_failure_resample() -> ScenarioResult:
    return _run_rollout_cycles(
        scenario="failure_resample",
        total_groups=4,
        samples_per_group=1,
        batch_size=2,
        max_concurrency=2,
        cycles=2,
        latency_s=lambda _task_id, _attempt: 0.02,
        status_for_attempt=lambda _task_id, attempt: "failed" if attempt == 1 else "completed",
    )


def scenario_adapter_logprob_contract() -> ScenarioResult:
    from slime_bridge import adapter

    start = time.monotonic()
    trace = Trace(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        finish_reason="stop",
        response_logprobs=None,
        reward=1.0,
    )
    result = SessionResult(
        session_id="missing-logprobs",
        task_id="adapter-logprob-contract",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(status="COMPLETED", traces=[trace]),
        timing=SessionTiming(),
    )
    samples = adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)
    trainable_missing = 0
    for sample in samples:
        loss_mask = getattr(sample, "loss_mask", None) or []
        rollout_log_probs = getattr(sample, "rollout_log_probs", None)
        if sum(loss_mask) > 0 and rollout_log_probs is not None and all(v == 0.0 for v in rollout_log_probs):
            trainable_missing += 1
    failures = []
    if trainable_missing:
        failures.append("trainable sample accepted without real rollout_log_probs")
    elapsed = time.monotonic() - start
    return ScenarioResult(
        name="adapter_logprob_contract",
        status="completed",
        elapsed_s=elapsed,
        metrics={
            "sample_count": len(samples),
            "trainable_missing_logprob_samples": trainable_missing,
            "loss_mask_sums": [sum(getattr(sample, "loss_mask", []) or []) for sample in samples],
            "rollout_logprob_lengths": [
                len(getattr(sample, "rollout_log_probs", []) or []) for sample in samples
            ],
        },
        failures=failures,
    )


def scenario_adapter_dummy_mask() -> ScenarioResult:
    from slime_bridge import adapter

    start = time.monotonic()
    result = SessionResult(
        session_id="empty-session",
        task_id="adapter-dummy-mask",
        status=SessionStatus.ERROR,
        trajectory=Trajectory(status="ERROR", traces=[], error="synthetic empty result"),
        timing=SessionTiming(),
        error="synthetic empty result",
    )
    samples = adapter.session_result_to_samples(result, group_index=0, trajectory_index=0)
    positive_dummy_masks = 0
    for sample in samples:
        polar = getattr(sample, "metadata", {}).get("polar", {})
        if polar.get("placeholder") and sum(getattr(sample, "loss_mask", []) or []) > 0:
            positive_dummy_masks += 1
    failures = []
    if positive_dummy_masks:
        failures.append("dummy placeholder has positive loss_mask")
    elapsed = time.monotonic() - start
    return ScenarioResult(
        name="adapter_dummy_mask",
        status="completed",
        elapsed_s=elapsed,
        metrics={
            "sample_count": len(samples),
            "positive_dummy_masks": positive_dummy_masks,
            "loss_mask_sums": [sum(getattr(sample, "loss_mask", []) or []) for sample in samples],
        },
        failures=failures,
    )


SCENARIOS: dict[str, Callable[[], ScenarioResult]] = {
    "steady_state_smoke": scenario_steady_state_smoke,
    "overflow_buffering": scenario_overflow_buffering,
    "failure_resample": scenario_failure_resample,
    "adapter_logprob_contract": scenario_adapter_logprob_contract,
    "adapter_dummy_mask": scenario_adapter_dummy_mask,
}


def run_child(scenario_name: str, output_path: Path) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario_name!r}")
    result = SCENARIOS[scenario_name]()
    output_path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))


def run_parent(args: argparse.Namespace) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = args.scenario or list(SCENARIOS)
    all_runs: list[dict[str, Any]] = []

    for run_idx in range(args.runs):
        run_results: list[dict[str, Any]] = []
        run_start = time.monotonic()
        for scenario_name in scenarios:
            with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
                child_output = Path(tmpdir) / "scenario.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child-scenario",
                    scenario_name,
                    "--child-output",
                    str(child_output),
                ]
                scenario_start = time.monotonic()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        timeout=args.scenario_timeout,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.monotonic() - scenario_start
                    result = ScenarioResult(
                        name=scenario_name,
                        status="timeout",
                        elapsed_s=elapsed,
                        metrics={"timeout_s": args.scenario_timeout},
                        failures=[f"scenario timed out after {args.scenario_timeout:.1f}s"],
                    ).to_json()
                    if args.verbose and exc.stderr:
                        print(_to_text(exc.stderr), file=sys.stderr)
                else:
                    if proc.returncode != 0:
                        elapsed = time.monotonic() - scenario_start
                        result = ScenarioResult(
                            name=scenario_name,
                            status="error",
                            elapsed_s=elapsed,
                            metrics={"returncode": proc.returncode},
                            failures=[proc.stderr.strip() or "child process failed"],
                        ).to_json()
                    elif child_output.exists():
                        result = json.loads(child_output.read_text())
                    else:
                        elapsed = time.monotonic() - scenario_start
                        result = ScenarioResult(
                            name=scenario_name,
                            status="error",
                            elapsed_s=elapsed,
                            metrics={},
                            failures=["child did not write a result file"],
                        ).to_json()
                    if args.verbose and proc.stderr:
                        print(proc.stderr, file=sys.stderr)
                run_results.append(result)
                print(_format_scenario_line(args.tag, run_idx, result), flush=True)

        run_elapsed = time.monotonic() - run_start
        run_payload = {
            "benchmark": "polar_slime_async",
            "benchmark_version": 1,
            "created_at": timestamp,
            "run_index": run_idx,
            "tag": args.tag,
            "elapsed_s": run_elapsed,
            "scenarios": run_results,
            "summary": _summary(run_results),
        }
        out_path = out_dir / f"{timestamp}_{args.tag}_run{run_idx}.json"
        out_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True))
        all_runs.append({"path": str(out_path), **run_payload})
        print(f"wrote {out_path}", flush=True)

    aggregate_path = out_dir / f"{timestamp}_{args.tag}_aggregate.json"
    aggregate = {
        "benchmark": "polar_slime_async",
        "benchmark_version": 1,
        "created_at": timestamp,
        "tag": args.tag,
        "runs": all_runs,
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"wrote {aggregate_path}", flush=True)
    if args.strict and not all(_summary(run["scenarios"])["passed"] for run in all_runs):
        return 1
    return 0


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [r for r in results if not r.get("passed")]
    return {
        "passed": not failed,
        "scenario_count": len(results),
        "failed_scenarios": [r["name"] for r in failed],
        "failure_count": len(failed),
    }


def _format_scenario_line(tag: str, run_idx: int, result: dict[str, Any]) -> str:
    status = "PASS" if result.get("passed") else "FAIL"
    failures = "; ".join(result.get("failures") or [])
    suffix = f" | {failures}" if failures else ""
    return f"[{tag} run={run_idx}] {status} {result['name']} {result['elapsed_s']:.2f}s{suffix}"


def _to_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1, help="number of full benchmark runs")
    parser.add_argument("--tag", default="local", help="label written into output filenames")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "tmp" / "benchmarks" / "polar_slime_async"),
        help="directory for JSON results",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(SCENARIOS),
        help="scenario to run; may be repeated. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--scenario-timeout",
        type=float,
        default=12.0,
        help="seconds before a child scenario is considered hung",
    )
    parser.add_argument("--verbose", action="store_true", help="print child stderr")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when any scenario fails; useful after the refactor or in CI",
    )
    parser.add_argument("--child-scenario", choices=sorted(SCENARIOS), help=argparse.SUPPRESS)
    parser.add_argument("--child-output", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.child_scenario:
        if not args.child_output:
            raise SystemExit("--child-output is required with --child-scenario")
        run_child(args.child_scenario, Path(args.child_output))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
