"""Dispatch + collect rollout pipeline for gateway nodes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import SessionContext, SessionDispatchRequest, SessionResult, SessionStatus
from polar.trajectory.models import Trajectory

logger = logging.getLogger(__name__)

ResultCallback = Callable[[SessionResult], Awaitable[None] | None]


def _trajectory_status(status: str) -> str:
    if status == SessionStatus.TIMEOUT:
        return SessionStatus.TIMEOUT
    if status == SessionStatus.COMPLETED:
        return SessionStatus.COMPLETED
    return SessionStatus.ERROR


class Pipeline:
    """Process rollout sessions by dispatching to gateway nodes and collecting results."""

    def __init__(
        self,
        *,
        callback_url: str,
        save_dir: str | None,
        scheduler: NodeScheduler,
        dispatch_poll_interval_seconds: float = 1.0,
        callback_grace_seconds: float = 5.0,
    ) -> None:
        self.callback_url = callback_url.rstrip("/")
        self.save_dir = Path(save_dir) if save_dir else None
        self.scheduler = scheduler
        self.dispatch_poll_interval_seconds = dispatch_poll_interval_seconds
        self.callback_grace_seconds = callback_grace_seconds

        self._client: httpx.AsyncClient | None = None
        self._started = False
        self._lifecycle_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[SessionResult]] = {}
        self._pending_lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            self._client = httpx.AsyncClient(timeout=30.0)
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            async with self._pending_lock:
                for future in self._pending.values():
                    if not future.done():
                        future.cancel()
                self._pending.clear()
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            self._started = False

    async def run_batch(
        self,
        sessions: list[SessionContext],
        *,
        on_result: ResultCallback | None = None,
    ) -> list[SessionResult]:
        await self.start()
        return await asyncio.gather(
            *(self._dispatch_and_collect(session, on_result) for session in sessions)
        )

    async def accept_callback_result(self, result: SessionResult) -> bool:
        async with self._pending_lock:
            future = self._pending.get(result.session_id)
            if future is None or future.done():
                return False
            future.set_result(result)
            return True

    def status(self) -> dict[str, object]:
        return {
            "pending_sessions": len(self._pending),
        }

    def result_path_for(self, task_id: str, session_id: str) -> str | None:
        path = self._result_path(task_id, session_id)
        return None if path is None else str(path)

    async def _dispatch_and_collect(
        self,
        session: SessionContext,
        callback: ResultCallback | None,
    ) -> SessionResult:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        session.completion_future = future
        async with self._pending_lock:
            self._pending[session.session_id] = future

        session.timer.mark("dispatch", "started")
        try:
            dispatch_request = await self._dispatch_session(session)
            session.timer.mark("dispatch", "finished")
            result = await self._wait_for_result(session, dispatch_request, future)
        except TimeoutError as exc:
            logger.warning("Session %s timed out in rollout pipeline", session.session_id)
            result = self._failure_result(session, status=SessionStatus.TIMEOUT, error=str(exc))
        except Exception as exc:
            logger.exception("Dispatch failed for session %s", session.session_id)
            result = self._failure_result(session, error=str(exc))
        finally:
            async with self._pending_lock:
                self._pending.pop(session.session_id, None)

        await asyncio.to_thread(self._persist_result, result)
        session.rollout_result = result
        try:
            if callback is not None:
                maybe_awaitable = callback(result)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            return result
        finally:
            await self._cleanup_session(session)

    async def _dispatch_session(self, session: SessionContext) -> SessionDispatchRequest:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")

        while True:
            node = self.scheduler.acquire_node()
            if node is None:
                remaining_timeout = self._remaining_timeout_seconds(session)
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                continue

            session.node_id = node.node_id
            session.gateway_url = node.gateway_url
            remaining_timeout = self._remaining_timeout_seconds(session)
            dispatch_request = SessionDispatchRequest(
                session_id=session.session_id,
                task_id=session.task_id,
                instruction=session.request.instruction,
                remaining_timeout_seconds=remaining_timeout,
                callback_url=self.callback_url,
                runtime=session.request.runtime,
                agent=session.request.agent,
                builder=session.request.builder,
                evaluator=session.request.evaluator,
            )
            try:
                response = await self._client.post(
                    f"{node.gateway_url}/sessions",
                    json=dispatch_request.model_dump(mode="json"),
                    timeout=min(30.0, remaining_timeout),
                )
                response.raise_for_status()
                return dispatch_request
            except Exception as exc:
                self.scheduler.release_reservation(node.node_id)
                self.scheduler.mark_unhealthy(node.node_id)
                try:
                    remaining_timeout = self._remaining_timeout_seconds(session)
                except TimeoutError:
                    raise TimeoutError(
                        "session timeout expired before gateway dispatch completed"
                    ) from exc
                await asyncio.sleep(min(self.dispatch_poll_interval_seconds, remaining_timeout))
                session.node_id = None
                session.gateway_url = None

    async def _wait_for_result(
        self,
        session: SessionContext,
        dispatch_request: SessionDispatchRequest,
        future: asyncio.Future[SessionResult],
    ) -> SessionResult:
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        callback_deadline = self._callback_deadline_monotonic(session)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._remaining_callback_window_seconds(session),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Callback timed out for session %s on node %s; polling result",
                session.session_id,
                session.node_id,
            )

        while time.monotonic() < callback_deadline:
            result = await self._poll_session_result(
                session,
                timeout=self._remaining_callback_window_seconds(session),
            )
            if result is not None:
                if not future.done():
                    future.set_result(result)
                return result
            sleep_seconds = min(
                self.dispatch_poll_interval_seconds,
                max(0.0, callback_deadline - time.monotonic()),
            )
            if sleep_seconds <= 0:
                break
            await asyncio.sleep(sleep_seconds)

        raise TimeoutError(
            f"session {dispatch_request.session_id} did not return a terminal result "
            "before the callback deadline"
        )

    async def _poll_session_result(
        self,
        session: SessionContext,
        *,
        timeout: float,
    ) -> SessionResult | None:
        if self._client is None:
            raise RuntimeError("pipeline has not been started")
        if session.gateway_url is None:
            raise RuntimeError("session gateway_url was not assigned")

        response = await self._client.get(
            f"{session.gateway_url}/sessions/{session.session_id}",
            timeout=min(30.0, timeout),
        )
        response.raise_for_status()
        payload = response.json()
        result_payload = payload.get("result")
        if isinstance(result_payload, dict):
            return SessionResult.model_validate(result_payload)
        status = str(payload.get("status", "")).upper()
        if status in SessionStatus.terminal():
            return self._failure_result(
                session,
                status=status,
                error=f"terminal session state {status} without session result payload",
            )
        return None

    def _remaining_timeout_seconds(self, session: SessionContext) -> float:
        remaining = session.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session timeout expired")
        return remaining

    def _callback_deadline_monotonic(self, session: SessionContext) -> float:
        return session.deadline_monotonic + self.callback_grace_seconds

    def _remaining_callback_window_seconds(self, session: SessionContext) -> float:
        remaining = self._callback_deadline_monotonic(session) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("session callback deadline expired")
        return remaining

    async def _cleanup_session(self, session: SessionContext) -> None:
        if self._client is None or session.gateway_url is None:
            return

        try:
            response = await self._client.delete(
                f"{session.gateway_url}/sessions/{session.session_id}"
            )
            if response.status_code not in {200, 404}:
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to clean up session %s on gateway %s",
                session.session_id,
                session.gateway_url,
                exc_info=True,
            )

    def _persist_result(self, result: SessionResult) -> None:
        path = self._result_path(result.task_id, result.session_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._storage_payload(result), separators=(",", ":"), default=str)
        )

    def _result_path(self, task_id: str, session_id: str) -> Path | None:
        if self.save_dir is None:
            return None
        return self.save_dir / f"task_{task_id}" / f"ses_{session_id}.json"

    @staticmethod
    def _storage_payload(result: SessionResult) -> dict[str, object]:
        """Return the persisted session artifact shape.

        The on-disk rollout result keeps session-level status/error only.
        Trajectory payloads store the structured trace data without duplicating
        terminal status information.
        """
        payload = result.model_dump(mode="json")
        trajectory = payload.get("trajectory")
        if isinstance(trajectory, dict):
            trajectory.pop("status", None)
            trajectory.pop("error", None)
        return payload

    @staticmethod
    def _failure_result(
        session: SessionContext,
        *,
        status: str = SessionStatus.ERROR,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=session.session_id,
            task_id=session.task_id,
            status=_trajectory_status(status),
            trajectory=Trajectory(
                status=_trajectory_status(status),
                metadata={
                    "builder": session.request.builder.strategy,
                    "record_count": 0,
                },
                traces=[],
                error=error,
            ),
            timing=session.timer.to_session_timing(),
            node_id=session.node_id,
            error=error,
        )
