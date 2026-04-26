"""Smoke tests for the gateway module cleanups."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from polar.gateway.dispatcher import (
    ManagedSession,
    SessionDispatcher,
    SessionStage,
)
from polar.gateway.proxy import SGLangClient
from polar.gateway.session import SessionRegistry
from polar.rollout.models import (
    SessionDispatchRequest,
    SessionResult,
    SessionStatus,
)
from polar.rollout.timer import StageTimer
from polar.agent.models import AgentSpec
from polar.trajectory.models import Trajectory


def _make_managed(session_id: str, tmp: Path, timeout: float = 10.0) -> ManagedSession:
    request = SessionDispatchRequest(
        session_id=session_id,
        task_id="t1",
        instruction="noop",
        remaining_timeout_seconds=timeout,
        agent=AgentSpec(harness="shell", custom_shell=None) if False else AgentSpec(
            harness="claude_code"
        ),
    )
    session_dir = tmp / session_id
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        execution_deadline=asyncio.get_event_loop().time() + timeout
        if False
        else None,
    )


def test_session_stage_has_four_values() -> None:
    # Collapsed INIT_PENDING/INITIALIZING and POSTRUN_PENDING/POSTRUNNING.
    assert {s.value for s in SessionStage} == {"INIT", "READY", "RUNNING", "POSTRUN"}


def test_session_result_has_no_completion_session_field() -> None:
    # Removed from the wire payload.
    assert "completion_session" not in SessionResult.model_fields


def test_session_registry_clear_result_payload_releases_heavy_payload() -> None:
    registry = SessionRegistry()
    info = registry.register("s1", task_id="t1", registered=True)
    result = SessionResult(
        session_id="s1",
        task_id="t1",
        status=SessionStatus.COMPLETED,
        trajectory=Trajectory(
            status="COMPLETED",
            metadata={"builder": "x", "record_count": 0},
            traces=[],
        ),
    )
    registry.set_result("s1", result)
    assert registry.get("s1").result is not None

    registry.clear_result_payload("s1")
    info = registry.get("s1")
    # Status / task_id retained; heavy payload dropped.
    assert info.result is None
    assert info.status == SessionStatus.COMPLETED
    assert info.task_id == "t1"


def test_sglang_client_pause_waits_for_inflight_and_blocks_new_requests() -> None:
    async def _run() -> None:
        class _Response:
            is_success = True

            def json(self) -> dict[str, object]:
                return {"ok": True}

            async def aread(self) -> bytes:
                return b""

            async def aclose(self) -> None:
                return

        class _HTTPClient:
            def __init__(self) -> None:
                self.calls = 0
                self.is_closed = False
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def post(self, *_args, **_kwargs) -> _Response:
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return _Response()

        fake_http = _HTTPClient()
        client = SGLangClient("http://sglang.test")
        client._client = fake_http  # type: ignore[assignment]

        first = asyncio.create_task(client.completion({"messages": []}))
        await asyncio.wait_for(fake_http.started.wait(), timeout=2.0)
        assert client.generation_status()["inflight"] == 1

        pause = asyncio.create_task(client.pause_generation(timeout_seconds=2.0))
        await asyncio.sleep(0.05)
        assert not pause.done()

        second = asyncio.create_task(client.completion({"messages": []}))
        await asyncio.sleep(0.05)
        assert fake_http.calls == 1

        fake_http.release.set()
        await asyncio.wait_for(pause, timeout=2.0)
        await asyncio.wait_for(first, timeout=2.0)
        assert client.generation_status() == {
            "paused": True,
            "inflight": 0,
            "base_url": "http://sglang.test",
        }
        assert fake_http.calls == 1

        await client.resume_generation()
        await asyncio.wait_for(second, timeout=2.0)
        assert fake_http.calls == 2

    asyncio.run(_run())


def test_dispatcher_snapshot_counts_stages() -> None:
    async def _run() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=1,
            max_run_workers=1,
            max_postrun_workers=1,
        )
        await dispatcher.start()
        try:
            # A snapshot before any session is enqueued is empty.
            snap = await dispatcher.snapshot()
            assert snap.active_count == 0
        finally:
            await dispatcher.stop()

    asyncio.run(_run())


def test_dispatcher_routes_session_through_all_stages() -> None:
    """Happy path: a session moves INIT -> READY -> RUNNING -> POSTRUN and the
    registry ends empty."""

    async def _run() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=1, max_run_workers=1, max_postrun_workers=1
        )
        seen_stages: list[SessionStage] = []
        done = asyncio.Event()

        async def on_init(managed: ManagedSession) -> None:
            seen_stages.append(SessionStage.INIT)

        async def on_run(managed: ManagedSession) -> None:
            seen_stages.append(SessionStage.RUNNING)

        async def on_postrun(managed: ManagedSession) -> None:
            seen_stages.append(SessionStage.POSTRUN)
            done.set()

        dispatcher.on_init = on_init
        dispatcher.on_run = on_run
        dispatcher.on_postrun = on_postrun

        await dispatcher.start()
        try:
            with TemporaryDirectory() as tmp:
                managed = _make_managed("s1", Path(tmp))
                await dispatcher.enqueue(managed)
                await asyncio.wait_for(done.wait(), timeout=5.0)
            assert seen_stages == [
                SessionStage.INIT,
                SessionStage.RUNNING,
                SessionStage.POSTRUN,
            ]
            # After POSTRUN, the session is removed.
            snap = await dispatcher.snapshot()
            assert snap.active_count == 0
        finally:
            await dispatcher.stop()

    asyncio.run(_run())


def test_dispatcher_bounds_run_capacity_via_ready_slots() -> None:
    """With max_run_workers=1, only one session runs at a time. The other sits
    in READY until the first finishes."""

    async def _run() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=2, max_run_workers=1, max_postrun_workers=2
        )
        run_started = asyncio.Event()
        release_run = asyncio.Event()
        concurrent_peaks: list[int] = []
        inflight_counter = {"n": 0}

        async def on_init(managed: ManagedSession) -> None:
            return

        async def on_run(managed: ManagedSession) -> None:
            inflight_counter["n"] += 1
            concurrent_peaks.append(inflight_counter["n"])
            run_started.set()
            await release_run.wait()
            inflight_counter["n"] -= 1

        async def on_postrun(managed: ManagedSession) -> None:
            return

        dispatcher.on_init = on_init
        dispatcher.on_run = on_run
        dispatcher.on_postrun = on_postrun

        await dispatcher.start()
        try:
            with TemporaryDirectory() as tmp:
                await dispatcher.enqueue(_make_managed("a", Path(tmp)))
                await dispatcher.enqueue(_make_managed("b", Path(tmp)))
                await asyncio.wait_for(run_started.wait(), timeout=2.0)
                # Give the second session time to park in READY — it must NOT
                # enter RUNNING while the first holds the slot.
                await asyncio.sleep(0.1)
                assert inflight_counter["n"] == 1
                release_run.set()
                # Both sessions should now complete.
                for _ in range(200):
                    snap = await dispatcher.snapshot()
                    if snap.active_count == 0:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("sessions did not drain")
            assert max(concurrent_peaks) == 1
        finally:
            await dispatcher.stop()

    asyncio.run(_run())


def test_dispatcher_cancel_wakes_session_waiting_for_ready_slot() -> None:
    """A session parked on ready-slot acquisition must wake up when cancelled
    and flow directly to POSTRUN — no 250ms poll sleep required."""

    async def _run() -> None:
        dispatcher = SessionDispatcher(
            max_init_workers=2, max_run_workers=1, max_postrun_workers=2
        )

        blocker_release = asyncio.Event()
        blocker_run = asyncio.Event()
        postrun_seen: list[str] = []

        async def on_init(managed: ManagedSession) -> None:
            return

        async def on_run(managed: ManagedSession) -> None:
            if managed.session_id == "blocker":
                blocker_run.set()
                await blocker_release.wait()

        async def on_postrun(managed: ManagedSession) -> None:
            postrun_seen.append(managed.session_id)

        dispatcher.on_init = on_init
        dispatcher.on_run = on_run
        dispatcher.on_postrun = on_postrun

        await dispatcher.start()
        try:
            with TemporaryDirectory() as tmp:
                await dispatcher.enqueue(_make_managed("blocker", Path(tmp)))
                await dispatcher.enqueue(_make_managed("waiter", Path(tmp)))
                await asyncio.wait_for(blocker_run.wait(), timeout=2.0)
                # `waiter` should be parked in READY waiting on the semaphore.
                # Cancelling it should flow it to POSTRUN promptly.
                cancelled = await dispatcher.cancel("waiter")
                assert cancelled
                for _ in range(200):
                    if "waiter" in postrun_seen:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("cancelled session never reached POSTRUN")
                blocker_release.set()
        finally:
            await dispatcher.stop()

    asyncio.run(_run())


def test_dispatcher_has_no_eval_prewarm_machinery() -> None:
    """Eval prewarm is now an ad-hoc background task owned by the caller."""
    dispatcher = SessionDispatcher(
        max_init_workers=1, max_run_workers=1, max_postrun_workers=1
    )
    # No on_eval_prewarm, no slot/queue, no consume/request methods.
    assert not hasattr(dispatcher, "on_eval_prewarm")
    assert not hasattr(dispatcher, "request_eval_prewarm")
    assert not hasattr(dispatcher, "acquire_eval_prewarm_slot")
    assert not hasattr(dispatcher, "consume_eval_prewarm")
    assert not hasattr(dispatcher, "max_eval_prewarm_workers")


def test_secrets_module_is_gone() -> None:
    with pytest.raises(ImportError):
        import polar.gateway.secrets  # noqa: F401


def test_control_module_is_merged_into_node() -> None:
    with pytest.raises(ImportError):
        import polar.gateway.control  # noqa: F401
