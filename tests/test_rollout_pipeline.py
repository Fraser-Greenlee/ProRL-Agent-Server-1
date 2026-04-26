"""Tests for rollout pipeline dispatch edge cases."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import httpx

from polar.agent.models import AgentSpec
from polar.rollout.models import (
    GatewayNodeInfo,
    NodeStageMetrics,
    SessionContext,
    TaskRequest,
)
from polar.rollout.pipeline import Pipeline


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://gateway.test/sessions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class _AcceptedAfterErrorClient:
    def __init__(self, post_error: Exception) -> None:
        self.post_error = post_error
        self.get_calls = 0

    async def post(self, *_args, **_kwargs) -> _FakeResponse:
        raise self.post_error

    async def get(self, *_args, **_kwargs) -> _FakeResponse:
        self.get_calls += 1
        return _FakeResponse(
            {
                "session_id": "s1",
                "task_id": "t1",
                "status": "REGISTERED",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completion_count": 0,
            }
        )


class _FakeScheduler:
    def __init__(self) -> None:
        self.releases = 0
        self.unhealthy = 0

    def acquire_node(self) -> GatewayNodeInfo:
        return GatewayNodeInfo(
            node_id="node-1",
            gateway_url="http://gateway.test",
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
            metrics=NodeStageMetrics(),
            healthy=True,
            heartbeat_interval_seconds=30,
            last_heartbeat=datetime.now(timezone.utc),
        )

    def release_reservation(self, _node_id: str) -> None:
        self.releases += 1

    def mark_unhealthy(self, _node_id: str) -> None:
        self.unhealthy += 1


def _session() -> SessionContext:
    request = TaskRequest(
        task_id="t1",
        instruction="noop",
        timeout_seconds=60,
        agent=AgentSpec(harness="claude_code"),
    )
    return SessionContext(
        session_id="s1",
        task_id="t1",
        request=request,
        deadline_monotonic=time.monotonic() + 60,
    )


def test_dispatch_treats_gateway_accepted_timeout_as_success() -> None:
    async def _run() -> None:
        scheduler = _FakeScheduler()
        pipeline = Pipeline(
            callback_url="http://rollout.test/callback",
            save_dir=None,
            scheduler=scheduler,  # type: ignore[arg-type]
        )
        client = _AcceptedAfterErrorClient(httpx.ReadTimeout("dispatch timed out"))
        pipeline._client = client  # type: ignore[assignment]

        dispatch = await pipeline._dispatch_session(_session())

        assert dispatch.session_id == "s1"
        assert dispatch.task_id == "t1"
        assert client.get_calls == 1
        assert scheduler.releases == 0
        assert scheduler.unhealthy == 0

    asyncio.run(_run())

