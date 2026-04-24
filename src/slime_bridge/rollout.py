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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from polar.rollout.models import TaskResult, TaskStatus
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.adapter import RolloutLogprobError, session_result_to_samples
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


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


@dataclass(slots=True)
class _RetryGroup:
    group: list[Any]
    attempt: int
    reason: str


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    group: list[Any]
    attempt: int
    submitted_rollout_id: int
    policy_version: int
    session_cost: int


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    samples: list[Any]
    task_id: str
    attempt: int
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    completed_at: float = field(default_factory=time.monotonic)

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


def update_policy_version(args: Any, policy_version: int) -> None:
    """Optional hook called by Slime after serving weights are updated."""
    del args
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.update_policy_version(policy_version)


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


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    attempt_id: int,
    policy_version: int,
    rollout_step: int,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    payload["metadata"] = {
        **metadata,
        "group_id": group_id,
        "attempt_id": attempt_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }


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


def _is_placeholder_sample(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    polar_meta = metadata.get("polar")
    return isinstance(polar_meta, dict) and bool(polar_meta.get("placeholder"))


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    attempt_id: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "attempt_id": int(attempt_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


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
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        # Output queue is a handoff channel; the durable overflow buffer is
        # `_completed_buffer`, which is drained in bounded chunks by training.
        queue_maxsize = max(32, batch_size * self.config.max_async_level * 2)
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=queue_maxsize)
        self.retry_queue: queue.Queue[_RetryGroup] = queue.Queue()
        self._completed_buffer: deque[_CompletedGroup] = deque()
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        self._batch_size = batch_size
        self._current_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._policy_version = self._current_rollout_id
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float] = {}
        self._active_groups = 0
        self._active_sessions = 0
        self._completed_buffer_size = 0
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

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = max(self._current_rollout_id, int(rollout_id))

    def update_policy_version(self, policy_version: int) -> None:
        with self._state_lock:
            self._policy_version = max(self._policy_version, int(policy_version))

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        while True:
            try:
                self._completed_buffer.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)

        accepted: list[_CompletedGroup] = []
        while self._completed_buffer and len(accepted) < max_groups:
            completed = self._completed_buffer.popleft()
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                if completed.attempt >= self.config.max_task_retries:
                    self._set_fatal(
                        PolarRolloutSchedulerError(
                            f"Task {completed.task_id} exceeded staleness bound "
                            f"after {completed.attempt + 1} attempts: {reason}"
                        )
                    )
                    self.raise_if_failed()
                self.retry_queue.put(
                    _RetryGroup(
                        group=completed.group,
                        attempt=completed.attempt + 1,
                        reason=reason,
                    )
                )
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                attempt_id=completed.attempt,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)

        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)
        return accepted

    def queue_size(self) -> int:
        with self._state_lock:
            return (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self.retry_queue.qsize()
            )

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            out = dict(self._metrics)
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/completed_buffer"] = float(self._completed_buffer_size)
            out["polar/scheduler/output_queue"] = float(self.output_queue.qsize())
            out["polar/scheduler/retry_queue"] = float(self.retry_queue.qsize())
            out["polar/scheduler/policy_version"] = float(self._policy_version)
            return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[asyncio.Task[None], _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"polar_max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if active_session_cost + session_cost > self.config.max_session_concurrency:
                            self.retry_queue.put(next_group)
                            break

                        gid = self._group_counter
                        self._group_counter += 1
                        submitted_rollout_id, policy_version = self._rollout_context()
                        pending = _PendingGroup(
                            group_id=gid,
                            group=next_group.group,
                            attempt=next_group.attempt,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._record_active_counts(active, active_session_cost)

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                logger.info("Waiting for %d in-flight Polar tasks", len(active))
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener for TaskResult callbacks."""
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
            app=app, host=self.config.callback_host, port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, pending: _PendingGroup
    ) -> None:
        attempt = pending.attempt
        max_attempt = self.config.max_task_retries
        last_error: BaseException | None = None

        while attempt <= max_attempt and self._running:
            submitted_rollout_id, policy_version = self._rollout_context()
            attempted = pending if attempt == pending.attempt else _PendingGroup(
                group_id=pending.group_id,
                group=pending.group,
                attempt=attempt,
                submitted_rollout_id=submitted_rollout_id,
                policy_version=policy_version,
                session_cost=pending.session_cost,
            )
            try:
                completed = await self._submit_attempt(client, attempted)
                await self._emit_completed(completed)
                return
            except RolloutLogprobError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempt:
                    break
                attempt += 1
                self._inc_metric("polar/retry_groups")
                logger.warning(
                    "Retrying Polar group %s attempt=%d/%d after error: %s",
                    pending.group_id,
                    attempt + 1,
                    max_attempt + 1,
                    exc,
                )
                await asyncio.sleep(min(5.0, 0.25 * (2 ** (attempt - pending.attempt))))

        raise PolarRolloutSchedulerError(
            f"Polar group {pending.group_id} failed after "
            f"{max_attempt + 1} attempts: {last_error}"
        )

    async def _submit_attempt(
        self,
        client: httpx.AsyncClient,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=pending.group,
            rollout_id=pending.group_id, task_position=0,
        )
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            attempt_id=pending.attempt,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        task_result = await self._submit_with_callback(client, payload)

        retry_reason = self._task_retry_reason(task_result, pending.group)
        if retry_reason is not None:
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {retry_reason}"
            )

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, pending.group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        if not group_samples:
            raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
        if any(_is_placeholder_sample(sample) for sample in group_samples):
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} produced placeholder samples"
            )

        return _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            samples=group_samples,
            task_id=task_result.task_id,
            attempt=pending.attempt,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
        )

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        while self._running:
            try:
                self.output_queue.put_nowait(completed)
                self._inc_metric("polar/completed_groups")
                return
            except queue.Full:
                self._inc_metric("polar/output_queue_full_waits")
                await asyncio.sleep(0.1)

    def _next_group_for_submission(self) -> _RetryGroup | None:
        try:
            retry = self.retry_queue.get_nowait()
            self._inc_metric("polar/retry_queue_dequeues")
            return retry
        except queue.Empty:
            pass

        groups = self.data_source.get_samples(1)
        if not groups:
            return None
        group = groups[0]
        if not group:
            raise PolarRolloutSchedulerError("Slime data source returned an empty sample group")
        return _RetryGroup(group=group, attempt=0, reason="new")

    def _can_admit_group(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        if len(active) >= self.config.max_concurrency:
            return False
        if active_session_cost >= self.config.max_session_concurrency:
            return False
        owned_groups = (
            len(active)
            + self.output_queue.qsize()
            + self._shared_completed_buffer_size()
        )
        return owned_groups < (self._batch_size * self.config.max_async_level)

    def _task_retry_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if task_result.status != "completed":
            return f"task status={task_result.status}"
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        for result in task_result.results:
            session_status = getattr(result.status, "value", result.status)
            trajectory_status = result.trajectory.status
            if session_status != "COMPLETED" or trajectory_status != "COMPLETED":
                return (
                    f"session {result.session_id} status={session_status} "
                    f"trajectory_status={trajectory_status}"
                )
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            return self._current_rollout_id, self._policy_version

    def _shared_completed_buffer_size(self) -> int:
        with self._state_lock:
            return self._completed_buffer_size

    def _record_active_counts(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

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
            _attach_scheduler_metadata(
                payload,
                group_id=rollout_id,
                attempt_id=0,
                policy_version=rollout_id,
                rollout_step=rollout_id,
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
    async_worker.set_rollout_context(rollout_id)
    target = getattr(args, "rollout_batch_size", 1)

    data: list[list[Any]] = []
    staleness_values: list[int] = []
    attempt_values: list[int] = []
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        made_progress = False
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        for completed in completed_groups:
            data.append(completed.samples)
            staleness_values.append(
                max(0, int(rollout_id) - completed.policy_version)
            )
            attempt_values.append(completed.attempt)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
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
    if staleness_values:
        metrics["polar/staleness/mean"] = sum(staleness_values) / len(staleness_values)
        metrics["polar/staleness/max"] = max(staleness_values)
    if attempt_values:
        metrics["polar/retry_attempt/mean"] = sum(attempt_values) / len(attempt_values)
        metrics["polar/retry_attempt/max"] = max(attempt_values)
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    metrics.update(async_worker.snapshot_metrics())
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
    session_report: dict[str, dict[str, Any]] = {}
    policy_staleness: list[float] = []
    zombie_samples = 0
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        if "policy_staleness" in polar_meta:
            policy_staleness.append(float(polar_meta["policy_staleness"]))
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if is_placeholder:
            zombie_samples += 1
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
            evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
            report = evaluation.get("report") or {}
            if isinstance(report, dict):
                session_report[session_id] = report
        elif not is_placeholder:
            session_trace_count[session_id] += 1

    if init_ms:
        out["polar/session_ms/init_mean"] = sum(init_ms) / len(init_ms)
        out["polar/session_ms/run_mean"] = sum(run_ms) / len(run_ms)
        out["polar/session_ms/postrun_mean"] = sum(postrun_ms) / len(postrun_ms)
    if len(rewards) > 1:
        out["polar/reward_std"] = statistics.pstdev(rewards)
    if policy_staleness:
        out["polar/staleness/sample_mean"] = sum(policy_staleness) / len(policy_staleness)
        out["polar/staleness/sample_max"] = max(policy_staleness)

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    out["polar/zombie_samples"] = float(zombie_samples)
    out["polar/usable_samples"] = float(len(flat_samples) - zombie_samples)
    out["polar/empty_sessions"] = float(empty_sessions)
    out["polar/total_sessions"] = float(total_sessions)
    if total_sessions > 0:
        out["polar/traces_per_session/mean"] = (
            sum(session_trace_count.values()) / total_sessions
        )
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        failed_apply = sum(1 for r in session_report.values() if r.get("failed_apply_patch"))
        empty_gen = sum(1 for r in session_report.values() if r.get("empty_generation"))
        error_eval = sum(1 for r in session_report.values() if r.get("error_eval"))
        test_timeout = sum(1 for r in session_report.values() if r.get("test_timeout"))
        out["polar/eval/graded_sessions"] = float(graded_sessions)
        out["polar/eval/resolved_rate"] = resolved / graded_sessions
        out["polar/eval/failed_apply_rate"] = failed_apply / graded_sessions
        out["polar/eval/empty_generation_rate"] = empty_gen / graded_sessions
        out["polar/eval/error_eval_rate"] = error_eval / graded_sessions
        out["polar/eval/test_timeout_rate"] = test_timeout / graded_sessions
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
