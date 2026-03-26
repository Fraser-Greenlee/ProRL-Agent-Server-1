"""Stage-isolated session dispatcher for gateway execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from integration.models import AgentRunResult
from rollout.models import SessionDispatchRequest, SessionResult
from rollout.timer import StageTimer
from runtime.base import BaseRuntime

logger = logging.getLogger(__name__)

StageCallback = Callable[["ManagedSession"], Awaitable[None]]
_STOP = object()


class SessionStage(str, Enum):
    INIT_PENDING = "INIT_PENDING"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    POSTRUN_PENDING = "POSTRUN_PENDING"
    POSTRUNNING = "POSTRUNNING"


@dataclass(slots=True)
class PreparedRuntimeLease:
    """A fresh runtime prepared for evaluator refresh."""

    owner_session_id: str
    purpose: str
    runtime: BaseRuntime | None = None
    session_dir: Path | None = None
    artifacts_dir: Path | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    error: str | None = None


@dataclass(slots=True)
class DispatcherSnapshot:
    init_queue_depth: int = 0
    init_inflight: int = 0
    ready_depth: int = 0
    run_inflight: int = 0
    postrun_queue_depth: int = 0
    postrun_inflight: int = 0

    @property
    def active_count(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


@dataclass(slots=True)
class ManagedSession:
    """Per-session state flowing through the gateway dispatcher."""

    request: SessionDispatchRequest
    timer: StageTimer
    session_dir: Path
    artifacts_dir: Path
    runtime: BaseRuntime | None = None
    agent_result: AgentRunResult | None = None
    final_result: SessionResult | None = None
    eval_runtime_lease: PreparedRuntimeLease | None = None
    eval_prewarm_task: asyncio.Task | None = None
    cancel_requested: bool = False
    stage: SessionStage = SessionStage.INIT_PENDING

    @property
    def session_id(self) -> str:
        return self.request.session_id


class SessionDispatcher:
    """Drive INIT -> READY -> RUN -> POST_RUN with isolated worker pools."""

    def __init__(
        self,
        *,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
        ready_buffer_target: int,
    ) -> None:
        if max_init_workers < 1 or max_run_workers < 1 or max_postrun_workers < 1:
            raise ValueError("all stage worker counts must be at least 1")
        if ready_buffer_target < 1:
            raise ValueError("ready_buffer_target must be at least 1")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.ready_buffer_target = ready_buffer_target
        self.on_init: StageCallback | None = None
        self.on_run: StageCallback | None = None
        self.on_postrun: StageCallback | None = None
        self._init_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._postrun_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_slots = asyncio.Semaphore(ready_buffer_target)
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._workers = [
            *(asyncio.create_task(self._init_worker(i)) for i in range(self.max_init_workers)),
            *(asyncio.create_task(self._run_worker(i)) for i in range(self.max_run_workers)),
            *(asyncio.create_task(self._postrun_worker(i)) for i in range(self.max_postrun_workers)),
        ]
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for managed in sessions:
            managed.cancel_requested = True
            await self._cancel_managed_resources(managed)
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def enqueue(self, managed: ManagedSession) -> None:
        if not self._started:
            raise RuntimeError("dispatcher has not been started")
        async with self._lock:
            if managed.session_id in self._sessions:
                raise ValueError(f"session {managed.session_id} is already enqueued")
            self._sessions[managed.session_id] = managed
        await self._init_queue.put(managed.session_id)

    async def cancel(self, session_id: str) -> bool:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return False
            if managed.cancel_requested:
                return True
            managed.cancel_requested = True
            should_enqueue_postrun = False
            if managed.stage == SessionStage.READY:
                self._ready_slots.release()
            if managed.stage in {SessionStage.INIT_PENDING, SessionStage.READY}:
                managed.stage = SessionStage.POSTRUN_PENDING
                should_enqueue_postrun = True
        await self._cancel_managed_resources(managed)
        if should_enqueue_postrun:
            await self._postrun_queue.put(session_id)
        return True

    async def active_count(self) -> int:
        return (await self.snapshot()).active_count

    async def snapshot(self) -> DispatcherSnapshot:
        async with self._lock:
            snapshot = DispatcherSnapshot()
            for managed in self._sessions.values():
                if managed.stage == SessionStage.INIT_PENDING:
                    snapshot.init_queue_depth += 1
                elif managed.stage == SessionStage.INITIALIZING:
                    snapshot.init_inflight += 1
                elif managed.stage == SessionStage.READY:
                    snapshot.ready_depth += 1
                elif managed.stage == SessionStage.RUNNING:
                    snapshot.run_inflight += 1
                elif managed.stage == SessionStage.POSTRUN_PENDING:
                    snapshot.postrun_queue_depth += 1
                elif managed.stage == SessionStage.POSTRUNNING:
                    snapshot.postrun_inflight += 1
            return snapshot

    async def acquire_ready_slot_for_eval(self, session_id: str) -> bool:
        """Acquire a ready slot for an evaluator runtime prewarm."""
        return await self._wait_for_ready_slot(session_id)

    def release_ready_slot(self) -> None:
        """Release a ready slot used by an evaluator runtime."""
        self._ready_slots.release()

    async def _init_worker(self, worker_id: int) -> None:
        await self._worker_loop(
            worker_id,
            queue=self._init_queue,
            expected=SessionStage.INIT_PENDING,
            inflight=SessionStage.INITIALIZING,
            callback=self.on_init,
            next_stage=SessionStage.READY,
        )

    async def _run_worker(self, worker_id: int) -> None:
        await self._worker_loop(
            worker_id,
            queue=self._ready_queue,
            expected=SessionStage.READY,
            inflight=SessionStage.RUNNING,
            callback=self.on_run,
            next_stage=SessionStage.POSTRUN_PENDING,
        )

    async def _postrun_worker(self, worker_id: int) -> None:
        await self._worker_loop(
            worker_id,
            queue=self._postrun_queue,
            expected=SessionStage.POSTRUN_PENDING,
            inflight=SessionStage.POSTRUNNING,
            callback=self.on_postrun,
            next_stage=None,
        )

    async def _worker_loop(
        self,
        worker_id: int,
        *,
        queue: asyncio.Queue[str | object],
        expected: SessionStage,
        inflight: SessionStage,
        callback: StageCallback | None,
        next_stage: SessionStage | None,
    ) -> None:
        del worker_id
        while True:
            item = await queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            managed = await self._advance_stage(
                session_id,
                expected=expected,
                new_stage=inflight,
                release_ready=(expected == SessionStage.READY),
            )
            if managed is None:
                continue
            try:
                if callback is None:
                    raise RuntimeError(f"missing callback for stage {expected}")
                if inflight == SessionStage.RUNNING and (managed.cancel_requested or managed.final_result is not None):
                    await self._move_to_postrun(session_id)
                    continue
                await callback(managed)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Dispatcher stage %s failed for session %s", inflight, session_id)
            if next_stage is None:
                async with self._lock:
                    self._sessions.pop(session_id, None)
                continue
            if managed.cancel_requested or managed.final_result is not None:
                await self._move_to_postrun(session_id)
                continue
            if next_stage == SessionStage.READY:
                acquired = await self._wait_for_ready_slot(session_id)
                if not acquired:
                    await self._move_to_postrun(session_id)
                    continue
            transitioned = await self._advance_stage(
                session_id,
                expected=inflight,
                new_stage=next_stage,
                release_ready=(inflight == SessionStage.READY),
            )
            if transitioned is None:
                if next_stage == SessionStage.READY:
                    self._ready_slots.release()
                continue
            if next_stage == SessionStage.READY:
                await self._ready_queue.put(session_id)
            elif next_stage == SessionStage.POSTRUN_PENDING:
                await self._postrun_queue.put(session_id)

    async def _advance_stage(
        self,
        session_id: str,
        *,
        expected: SessionStage,
        new_stage: SessionStage,
        release_ready: bool = False,
    ) -> ManagedSession | None:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None or managed.stage != expected:
                return None
            if release_ready:
                self._ready_slots.release()
            managed.stage = new_stage
            return managed

    async def _move_to_postrun(self, session_id: str) -> None:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return
            if managed.stage == SessionStage.READY:
                self._ready_slots.release()
            if managed.stage in {SessionStage.POSTRUN_PENDING, SessionStage.POSTRUNNING}:
                return
            managed.stage = SessionStage.POSTRUN_PENDING
        await self._postrun_queue.put(session_id)

    async def _wait_for_ready_slot(self, session_id: str) -> bool:
        while True:
            async with self._lock:
                managed = self._sessions.get(session_id)
                if managed is None:
                    return False
                if managed.cancel_requested or managed.final_result is not None:
                    return False
            try:
                await asyncio.wait_for(self._ready_slots.acquire(), timeout=0.25)
                return True
            except asyncio.TimeoutError:
                continue

    async def _cancel_managed_resources(self, managed: ManagedSession) -> None:
        # Cancel eval prewarm
        if managed.eval_prewarm_task is not None and not managed.eval_prewarm_task.done():
            managed.eval_prewarm_task.cancel()
        if managed.eval_runtime_lease is not None:
            managed.eval_runtime_lease.cancelled = True
            rt = managed.eval_runtime_lease.runtime
            if rt is not None:
                await rt.cancel()
        # Cancel main runtime
        if managed.runtime is not None:
            await managed.runtime.cancel()
