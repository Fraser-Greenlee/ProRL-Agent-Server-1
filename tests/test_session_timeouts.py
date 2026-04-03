from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.agent.models import AgentSpec
from polar.rollout.models import SessionContext, SessionDispatchRequest, TaskRequest
from polar.rollout.pipeline import Pipeline
from polar.trajectory.registry import default_builder_registry, default_evaluator_registry


class _FakeScheduler:
    def __init__(self, node: SimpleNamespace | None = None) -> None:
        self._node = node

    def acquire_node(self):
        return self._node

    def release_reservation(self, node_id: str):
        return node_id

    def mark_unhealthy(self, node_id: str):
        return node_id


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []

    async def post(self, url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()


def _task_request(timeout_seconds: float = 30.0) -> TaskRequest:
    return TaskRequest(
        task_id="task-1",
        instruction="solve it",
        timeout_seconds=timeout_seconds,
        agent=AgentSpec(harness="opencode"),
    )


@pytest.mark.asyncio
async def test_pipeline_timeout_is_reported_as_session_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Pipeline(
        callback_url="http://rollout.local/callbacks/session_result",
        save_dir=None,
        scheduler=_FakeScheduler(),
    )
    pipeline._client = object()  # type: ignore[assignment]

    session = SessionContext(
        session_id="session-1",
        task_id="task-1",
        request=_task_request(),
        deadline_monotonic=time.monotonic() + 30.0,
    )

    async def _raise_timeout(_: SessionContext) -> SessionDispatchRequest:
        raise TimeoutError("session timeout expired")

    async def _noop_cleanup(_: SessionContext) -> None:
        return None

    monkeypatch.setattr(pipeline, "_dispatch_session", _raise_timeout)
    monkeypatch.setattr(pipeline, "_cleanup_session", _noop_cleanup)

    result = await pipeline._dispatch_and_collect(session, callback=None)

    assert result.status == "TIMEOUT"
    assert result.trajectory.status == "TIMEOUT"
    assert result.error == "session timeout expired"


@pytest.mark.asyncio
async def test_pipeline_dispatch_sends_remaining_session_budget() -> None:
    client = _FakeClient()
    pipeline = Pipeline(
        callback_url="http://rollout.local/callbacks/session_result",
        save_dir=None,
        scheduler=_FakeScheduler(
            SimpleNamespace(node_id="node-1", gateway_url="http://gateway.local")
        ),
    )
    pipeline._client = client  # type: ignore[assignment]

    session = SessionContext(
        session_id="session-2",
        task_id="task-1",
        request=_task_request(timeout_seconds=20.0),
        deadline_monotonic=time.monotonic() + 20.0,
    )

    await asyncio.sleep(0.02)
    dispatch_request = await pipeline._dispatch_session(session)

    assert 0 < dispatch_request.remaining_timeout_seconds < 20.0
    assert client.posts[0]["json"]["remaining_timeout_seconds"] == pytest.approx(
        dispatch_request.remaining_timeout_seconds
    )
    assert client.posts[0]["timeout"] == pytest.approx(
        min(30.0, dispatch_request.remaining_timeout_seconds)
    )


@pytest.mark.asyncio
async def test_gateway_dispatch_starts_budget_when_session_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}
    manager = GatewayNodeManager(
        node_id="node-1",
        gateway_url="http://gateway.local",
        max_init_workers=1,
        max_run_workers=1,
        max_postrun_workers=1,
        ready_buffer_target=1,
        storage=SessionStore(),
        session_registry=SessionRegistry(),
        builders=default_builder_registry(),
        evaluators=default_evaluator_registry(),
        session_base_dir=str(tmp_path),
    )

    async def _capture_enqueue(managed) -> None:
        captured["managed"] = managed

    monkeypatch.setattr(manager._dispatcher, "enqueue", _capture_enqueue)

    before = asyncio.get_running_loop().time()
    await manager.dispatch(
        SessionDispatchRequest(
            session_id="session-3",
            task_id="task-1",
            instruction="solve it",
            remaining_timeout_seconds=60.0,
            agent=AgentSpec(harness="opencode"),
        )
    )
    after = asyncio.get_running_loop().time()

    managed = captured["managed"]
    assert managed.execution_deadline is not None
    assert managed.execution_deadline >= before + 59.0
    assert managed.execution_deadline <= after + 60.0

    await manager.close()
