"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import queue
import statistics
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from polar.rollout.models import TaskResult, TaskStatus
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.adapter import session_result_to_samples
from slime_bridge.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_LONGEST_TRACE_ARTIFACT_INTERVAL = 5  # dump longest trace every N rollouts

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    resp = await client.post(
        f"{base_url}/rollout/task/submit",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        await asyncio.sleep(poll_interval)
        status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
        status_resp.raise_for_status()
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _resolve_max_tokens(args: Any) -> int | None:
    """Per-sample token cap Slime's dynamic batcher can fit on one GPU.

    Megatron asserts every sample length <= max_tokens_per_gpu * cp_size.
    Deep agent trajectories can exceed this (24-turn sessions → 80k+ tokens)
    and must be dropped before they reach the batcher.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    if not mtpg:
        return None
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    return int(mtpg) * cp_size


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Slime samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory.  The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        group_samples.extend(
            session_result_to_samples(
                session_result,
                group_index,
                trajectory_index=traj_idx,
                reward_key=config.reward_key,
                max_tokens=max_tokens,
            )
        )
    return group_samples


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``.  Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        # Queue depth is the effective staleness knob — cap at a few batches.
        queue_maxsize = max(32, self.config.max_concurrency * 4)
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callback_url: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def drain_completed(self) -> list[tuple[int, list[Any]]]:
        items: list[tuple[int, list[Any]]] = []
        while True:
            try:
                items.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        max_concurrent = self.config.max_concurrency
        active: set[asyncio.Task[None]] = set()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = {t for t in active if t.done()}
                    for t in done:
                        try:
                            t.result()
                        except Exception:
                            logger.exception("Polar async task failed")
                    active -= done

                    while len(active) < max_concurrent and self._running:
                        groups = self.data_source.get_samples(1)
                        if not groups:
                            break
                        for group in groups:
                            gid = self._group_counter
                            self._group_counter += 1
                            task = asyncio.create_task(
                                self._submit_and_collect(client, gid, group)
                            )
                            active.add(task)

                    await asyncio.sleep(0.5)

            if active:
                logger.info("Waiting for %d in-flight Polar tasks", len(active))
                await asyncio.gather(*active, return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener on 127.0.0.1:<free_port> for TaskResult callbacks."""
        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = TaskResult.model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app, host="127.0.0.1", port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://127.0.0.1:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, group_id: int, group: list[Any]
    ) -> None:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=group,
            rollout_id=group_id, task_position=0,
        )
        task_result = await self._submit_with_callback(client, payload)

        if task_result.status == "failed" or not task_result.results:
            logger.warning("Task %s ended with status=%s, skipping", task_result.task_id, task_result.status)
            return

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        self.output_queue.put((group_id, group_samples))

    async def _submit_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return await self._await_task_result(client, task_id, event)
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _await_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        event: asyncio.Event,
    ) -> TaskResult:
        """Wait on the completion event with a defensive 60s fallback poll."""
        base_url = self.config.rollout_server_url
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except asyncio.TimeoutError:
                status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
                status_resp.raise_for_status()
                status = TaskStatus.model_validate(status_resp.json())
                if status.status in ("completed", "failed"):
                    return TaskResult(
                        task_id=task_id, status=status.status,
                        results=status.results, result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing — re-poll once.
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = TaskStatus.model_validate(status_resp.json())
            return TaskResult(
                task_id=task_id, status=status.status,
                results=status.results, result_paths=status.result_paths,
            )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        async with semaphore:
            payload = _build_task_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            return await _submit_and_wait_for_task(client, config.rollout_server_url, payload)

    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(
                config, task_result, group,
                max_tokens=max_tokens,
            )
        )

    metrics = _build_metrics(config, task_results, output_groups)
    flat_samples = [sample for group in output_groups for sample in group]

    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={
            config.eval_dataset_name: {
                "rewards": [_extract_sample_reward(s, config.reward_key) for s in flat_samples],
                "truncated": [_is_truncated(s) for s in flat_samples],
                "samples": flat_samples,
            }
        },
        metrics=metrics,
    )


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    session_results = [result for task_result in task_results for result in task_result.results]
    completed_sessions = sum(1 for r in session_results if r.status == "COMPLETED")
    failed_sessions = sum(1 for r in session_results if r.status == "ERROR")
    timed_out_sessions = sum(1 for r in session_results if r.status == "TIMEOUT")
    rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    metrics: dict[str, Any] = {
        "polar/completed_sessions": completed_sessions,
        "polar/failed_sessions": failed_sessions,
        "polar/group_count": len(output_groups),
        "polar/sample_count": len(flat_samples),
        "polar/task_count": len(task_results),
        "polar/timed_out_sessions": timed_out_sessions,
    }
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    metrics.update(_polar_extra_metrics(flat_samples, rewards))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    async_worker = get_global_async_worker(args, data_source)
    target = getattr(args, "rollout_batch_size", 1)

    data: list[list[Any]] = []
    collected: dict[int, list[Any]] = {}
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        for gid, group_samples in async_worker.drain_completed():
            collected[gid] = group_samples

        made_progress = False
        for gid in sorted(collected):
            if len(data) >= target:
                break
            samples = collected.pop(gid)
            data.append(samples)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, collected=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    _maybe_dump_longest_trace_artifact(rollout_id, data)

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    metrics: dict[str, Any] = {
        "polar/sample_count": len(flat),
        "polar/group_count": len(data),
    }
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    metrics.update(_polar_extra_metrics(flat, rewards))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _maybe_dump_longest_trace_artifact(
    rollout_id: int, data: list[list[Any]], *, interval: int = _LONGEST_TRACE_ARTIFACT_INTERVAL
) -> None:
    """Dump the longest session in this rollout's batch as a wandb artifact.

    Groups samples by ``session_id``, picks the session with the largest
    aggregated assistant tokens, and writes its full message chain (per
    trace) to a JSON artifact. Silently no-ops if wandb isn't initialized.
    """
    if interval <= 0 or rollout_id % interval != 0:
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    by_session: dict[str, list[Any]] = {}
    for group in data:
        for sample in group:
            sid = getattr(sample, "session_id", None) or "unknown"
            by_session.setdefault(sid, []).append(sample)
    if not by_session:
        return

    def _session_tokens(samples: list[Any]) -> int:
        return sum(int(getattr(s, "response_length", 0) or 0) for s in samples)

    longest_sid, longest_samples = max(by_session.items(), key=lambda kv: _session_tokens(kv[1]))
    total_tokens = _session_tokens(longest_samples)
    if total_tokens <= 0:
        return

    longest_samples = sorted(
        longest_samples,
        key=lambda s: int((s.metadata.get("polar") or {}).get("trace_index", 0) or 0),
    )
    traces = []
    for sample in longest_samples:
        polar_meta = sample.metadata.get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        traces.append({
            "trace_index": polar_meta.get("trace_index"),
            "finish_reason": trace_debug.get("finish_reason"),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "status": getattr(status, "value", None) if status is not None else None,
            "prompt_messages": sample.prompt if isinstance(sample.prompt, list) else [],
            "response_messages": trace_debug.get("response_messages") or [],
        })

    first = longest_samples[0]
    first_meta = first.metadata.get("polar") or {}
    reward = getattr(first, "reward", None)
    if isinstance(reward, dict):
        session_reward = float(reward.get("score", 0.0))
    elif isinstance(reward, (int, float)):
        session_reward = float(reward)
    else:
        session_reward = 0.0

    payload = {
        "rollout_id": int(rollout_id),
        "session_id": longest_sid,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "total_assistant_tokens": int(total_tokens),
        "session_reward": session_reward,
        "num_traces": len(traces),
        "traces": traces,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / f"longest_trace_r{rollout_id}.json"
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            artifact = wandb.Artifact(
                name=f"longest_trace_r{rollout_id}", type="rollout-trace"
            )
            artifact.add_file(str(fpath))
            wandb.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log longest-trace wandb artifact")
        return

    logger.info(
        "Logged longest-trace artifact rollout=%d session=%s traces=%d tokens=%d",
        rollout_id, longest_sid, len(traces), total_tokens,
    )


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _polar_extra_metrics(flat_samples: list[Any], rewards: list[float]) -> dict[str, float]:
    """Session-level aggregates: timing means, reward std, usable vs
    placeholder sample counts, and traces-per-session mean.

    Placeholder samples are emitted by ``slime_bridge.adapter`` when a
    Polar session returned zero usable traces (typically the pipeline
    race or a genuine agent failure). Separating them from real samples
    is critical for spotting reward collapses caused by the rollout
    pipeline rather than the model.
    """
    out: dict[str, float] = {}
    seen: set[str] = set()
    init_ms: list[float] = []
    run_ms: list[float] = []
    postrun_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    session_trace_count: dict[str, int] = {}
    placeholder_samples = 0
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if is_placeholder:
            placeholder_samples += 1
        if not session_id:
            continue
        if session_id not in seen:
            seen.add(session_id)
            timing = polar_meta.get("timing") or {}
            init_ms.append(float(timing.get("init_ms", 0.0)))
            run_ms.append(float(timing.get("run_ms", 0.0)))
            postrun_ms.append(float(timing.get("postrun_ms", 0.0)))
            session_is_placeholder[session_id] = is_placeholder
            session_trace_count[session_id] = 0 if is_placeholder else 1
        elif not is_placeholder:
            session_trace_count[session_id] += 1

    if init_ms:
        out["polar/session_ms/init_mean"] = sum(init_ms) / len(init_ms)
        out["polar/session_ms/run_mean"] = sum(run_ms) / len(run_ms)
        out["polar/session_ms/postrun_mean"] = sum(postrun_ms) / len(postrun_ms)
    if len(rewards) > 1:
        out["polar/reward_std"] = statistics.pstdev(rewards)

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    out["polar/placeholder_samples"] = float(placeholder_samples)
    out["polar/usable_samples"] = float(len(flat_samples) - placeholder_samples)
    out["polar/empty_sessions"] = float(empty_sessions)
    out["polar/total_sessions"] = float(total_sessions)
    if total_sessions > 0:
        out["polar/traces_per_session/mean"] = (
            sum(session_trace_count.values()) / total_sessions
        )
    return out


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar rollouts from a Slime trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


atexit.register(stop_global_worker)
