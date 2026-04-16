"""Stage-isolated session dispatcher for gateway execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from polar.agent.models import AgentRunResult
from polar.rollout.models import SessionDispatchRequest, SessionResult
from polar.rollout.timer import StageTimer
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecInput

logger = logging.getLogger(__name__)

StageCallback = Callable[["ManagedSession"], Awaitable[None]]
StageTransitionCallback = Callable[["ManagedSession", "SessionStage"], None]
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
    slot_held: bool = False


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
    postrun_steps: list[ExecInput] = field(default_factory=list)
    eval_runtime_lease: PreparedRuntimeLease | None = None
    eval_prewarm_requested: bool = False
    eval_prewarm_inflight: bool = False
    eval_prewarm_ready: bool = False
    cancel_requested: bool = False
    execution_deadline: float | None = None
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
        max_eval_prewarm_workers: int,
        ready_buffer_target: int,
    ) -> None:
        if (
            max_init_workers < 1
            or max_run_workers < 1
            or max_postrun_workers < 1
            or max_eval_prewarm_workers < 1
        ):
            raise ValueError("all stage worker counts must be at least 1")
        if ready_buffer_target < 1:
            raise ValueError("ready_buffer_target must be at least 1")
        self.max_init_workers = max_init_workers
        self.max_run_workers = max_run_workers
        self.max_postrun_workers = max_postrun_workers
        self.max_eval_prewarm_workers = max_eval_prewarm_workers
        self.ready_buffer_target = ready_buffer_target
        self.on_init: StageCallback | None = None
        self.on_eval_prewarm: StageCallback | None = None
        self.on_run: StageCallback | None = None
        self.on_postrun: StageCallback | None = None
        self.on_stage_change: StageTransitionCallback | None = None
        self._init_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._eval_prewarm_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._postrun_queue: asyncio.Queue[str | object] = asyncio.Queue()
        self._ready_slots = asyncio.Semaphore(ready_buffer_target)
        self._eval_prewarm_slots = asyncio.Semaphore(max_eval_prewarm_workers)
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._workers = [
            *(asyncio.create_task(self._init_worker(i)) for i in range(self.max_init_workers)),
            *(asyncio.create_task(self._eval_prewarm_worker(i)) for i in range(self.max_eval_prewarm_workers)),
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
            transitioned_to_postrun = False
            if managed.stage == SessionStage.READY:
                self._ready_slots.release()
            if managed.stage in {SessionStage.INIT_PENDING, SessionStage.READY}:
                managed.stage = SessionStage.POSTRUN_PENDING
                should_enqueue_postrun = True
                transitioned_to_postrun = True
        await self._cancel_managed_resources(managed)
        if transitioned_to_postrun:
            self._notify_stage_change(managed, SessionStage.POSTRUN_PENDING)
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

    async def request_eval_prewarm(self, session_id: str) -> bool:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None or managed.cancel_requested or managed.final_result is not None:
                return False
            if managed.eval_prewarm_requested or managed.eval_prewarm_inflight or managed.eval_prewarm_ready:
                return False
            managed.eval_prewarm_requested = True
        await self._eval_prewarm_queue.put(session_id)
        return True

    async def acquire_eval_prewarm_slot(self, session_id: str) -> bool:
        return await self._wait_for_slot(self._eval_prewarm_slots, session_id)

    async def consume_eval_prewarm(self, session_id: str) -> PreparedRuntimeLease | None:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return None
            lease = managed.eval_runtime_lease
            managed.eval_prewarm_ready = False
            managed.eval_prewarm_requested = False
            if lease is not None and lease.slot_held:
                lease.slot_held = False
                release_slot = True
            else:
                release_slot = False
        if release_slot:
            self._eval_prewarm_slots.release()
        return lease

    async def _init_worker(self, worker_id: int) -> None:
        del worker_id
        while True:
            item = await self._init_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            await self._process_session_init_item(session_id)

    async def _eval_prewarm_worker(self, worker_id: int) -> None:
        del worker_id
        while True:
            item = await self._eval_prewarm_queue.get()
            if item is _STOP:
                return
            session_id = str(item)
            await self._process_eval_prewarm_item(session_id)

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
        self._notify_stage_change(managed, new_stage)
        return managed

    async def _move_to_postrun(self, session_id: str) -> None:
        transitioned = False
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return
            if managed.stage == SessionStage.READY:
                self._ready_slots.release()
            if managed.stage in {SessionStage.POSTRUN_PENDING, SessionStage.POSTRUNNING}:
                return
            managed.stage = SessionStage.POSTRUN_PENDING
            transitioned = True
        if transitioned:
            self._notify_stage_change(managed, SessionStage.POSTRUN_PENDING)
        await self._postrun_queue.put(session_id)

    async def _wait_for_ready_slot(self, session_id: str) -> bool:
        return await self._wait_for_slot(self._ready_slots, session_id)

    async def _wait_for_slot(
        self,
        semaphore: asyncio.Semaphore,
        session_id: str,
    ) -> bool:
        while True:
            async with self._lock:
                managed = self._sessions.get(session_id)
                if managed is None:
                    return False
                if managed.cancel_requested or managed.final_result is not None:
                    return False
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=0.25)
                return True
            except asyncio.TimeoutError:
                continue

    async def _cancel_managed_resources(self, managed: ManagedSession) -> None:
        if managed.eval_runtime_lease is not None:
            managed.eval_runtime_lease.cancelled = True
            if managed.eval_runtime_lease.slot_held:
                managed.eval_runtime_lease.slot_held = False
                self._eval_prewarm_slots.release()
            rt = managed.eval_runtime_lease.runtime
            if rt is not None:
                await rt.cancel()
        # Cancel main runtime
        if managed.runtime is not None:
            await managed.runtime.cancel()

    async def _process_session_init_item(self, session_id: str) -> None:
        managed = await self._advance_stage(
            session_id,
            expected=SessionStage.INIT_PENDING,
            new_stage=SessionStage.INITIALIZING,
        )
        if managed is None:
            return
        try:
            if self.on_init is None:
                raise RuntimeError("missing callback for init stage")
            await self.on_init(managed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Dispatcher stage %s failed for session %s",
                SessionStage.INITIALIZING,
                session_id,
            )
        if managed.cancel_requested or managed.final_result is not None:
            await self._move_to_postrun(session_id)
            return
        acquired = await self._wait_for_ready_slot(session_id)
        if not acquired:
            await self._move_to_postrun(session_id)
            return
        transitioned = await self._advance_stage(
            session_id,
            expected=SessionStage.INITIALIZING,
            new_stage=SessionStage.READY,
        )
        if transitioned is None:
            self._ready_slots.release()
            return
        await self._ready_queue.put(session_id)

    async def _process_eval_prewarm_item(self, session_id: str) -> None:
        async with self._lock:
            managed = self._sessions.get(session_id)
            if managed is None:
                return
            if managed.cancel_requested or managed.final_result is not None:
                return
            if (
                not managed.eval_prewarm_requested
                or managed.eval_prewarm_inflight
                or managed.eval_prewarm_ready
            ):
                return
            managed.eval_prewarm_inflight = True
        try:
            if self.on_eval_prewarm is None:
                raise RuntimeError("missing callback for eval prewarm stage")
            await self.on_eval_prewarm(managed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Dispatcher eval prewarm failed for session %s", session_id)
        finally:
            async with self._lock:
                current = self._sessions.get(session_id)
                if current is None:
                    return
                current.eval_prewarm_inflight = False
                lease = current.eval_runtime_lease
                current.eval_prewarm_ready = bool(
                    lease is not None
                    and lease.runtime is not None
                    and lease.error is None
                    and not lease.cancelled
                )

    def _notify_stage_change(
        self,
        managed: ManagedSession,
        stage: SessionStage,
    ) -> None:
        callback = self.on_stage_change
        if callback is None:
            return
        try:
            callback(managed, stage)
        except Exception:
            logger.exception(
                "Dispatcher stage-change callback failed for session %s",
                managed.session_id,
            )
