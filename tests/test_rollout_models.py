"""Smoke tests for the public rollout schema + timing."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from polar.rollout.models import (
    SessionResult,
    SessionStatus,
    SessionTiming,
)
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trajectory


def test_session_timing_has_three_phases() -> None:
    assert set(SessionTiming.model_fields) == {"init_ms", "run_ms", "postrun_ms"}


def test_session_timing_accepts_only_the_three_phases() -> None:
    with pytest.raises(ValidationError):
        SessionTiming.model_validate({"build_ms": 1.0})


def test_stage_timer_rolls_build_eval_teardown_into_postrun() -> None:
    timer = StageTimer()
    timer.mark("init", "started")
    timer.mark("init", "finished")
    timer.mark("run", "started")
    timer.mark("run", "finished")
    timer.mark("postrun", "started")
    timer.mark("build", "started")
    timer.mark("build", "finished")
    timer.mark("eval", "started")
    timer.mark("eval", "finished")
    timer.mark("teardown", "started")
    timer.mark("teardown", "finished")
    timer.mark("postrun", "finished")

    timing = timer.to_session_timing()
    assert timing.init_ms >= 0
    assert timing.run_ms >= 0
    # postrun spans the outer postrun start/finish, so it covers build/eval/teardown.
    assert timing.postrun_ms > 0


def test_session_status_strenum_serializes_as_string() -> None:
    # Wire compat: StrEnum must serialize as its string value.
    assert SessionStatus.COMPLETED.value == "COMPLETED"
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
    payload = json.loads(result.model_dump_json())
    assert payload["status"] == "COMPLETED"
    assert payload["timing"] == {"init_ms": 0.0, "run_ms": 0.0, "postrun_ms": 0.0}


def test_session_status_terminal_and_active_partition() -> None:
    terminal = SessionStatus.terminal()
    active = SessionStatus.active()
    assert terminal == frozenset(
        {SessionStatus.COMPLETED, SessionStatus.ERROR, SessionStatus.TIMEOUT}
    )
    assert terminal | active == set(SessionStatus)
    assert not (terminal & active)


def test_session_status_includes_all_canonical_statuses() -> None:
    names = {s.value for s in SessionStatus}
    expected = {
        "REGISTERED",
        "INITIALIZING",
        "READY",
        "RUNNING",
        "POST_RUN",
        "BUILDING",
        "EVALUATING",
        "COMPLETED",
        "ERROR",
        "TIMEOUT",
    }
    assert names == expected


def test_session_result_accepts_string_status_for_wire_compat() -> None:
    # Older clients submitting plain strings must still validate.
    result = SessionResult.model_validate(
        {
            "session_id": "s1",
            "task_id": "t1",
            "status": "COMPLETED",
            "trajectory": {
                "status": "COMPLETED",
                "metadata": {"builder": "x", "record_count": 0},
                "traces": [],
            },
        }
    )
    assert result.status is SessionStatus.COMPLETED


def test_rollout_manager_single_result_path_source(monkeypatch) -> None:
    """`_execute_task` must accumulate result paths in exactly one place."""

    import asyncio

    from polar.rollout.manager import RolloutManager
    from polar.rollout.models import SessionContext, SessionResult, TaskRequest
    from polar.agent.models import AgentSpec

    class _FakePipeline:
        def __init__(self) -> None:
            self._counter = 0

        def result_path_for(self, task_id, session_id):
            self._counter += 1
            return f"/tmp/{task_id}/{session_id}-{self._counter}.json"

        async def run_batch(self, sessions, *, on_result=None):
            results = []
            for session in sessions:
                r = SessionResult(
                    session_id=session.session_id,
                    task_id=session.task_id,
                    status=SessionStatus.COMPLETED,
                    trajectory=Trajectory(
                        status="COMPLETED",
                        metadata={"builder": "x", "record_count": 0},
                        traces=[],
                    ),
                )
                if on_result is not None:
                    maybe_coroutine = on_result(r)
                    if hasattr(maybe_coroutine, "__await__"):
                        await maybe_coroutine
                results.append(r)
            return results

        def status(self):
            return {}

    class _FakeScheduler:
        def stats(self):
            return {}

    manager = RolloutManager(pipeline=_FakePipeline(), scheduler=_FakeScheduler())
    request = TaskRequest(
        task_id="t1",
        instruction="noop",
        num_samples=3,
        agent=AgentSpec(harness="claude_code", model_name="anthropic/claude-4"),
    )

    async def _run() -> None:
        task_id = await manager.submit_task(request)
        # Wait for background task to finish.
        for _ in range(200):
            status = manager.get_task(task_id)
            if status is not None and status.status == "completed":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("background task did not complete in time")
        # Each session produced exactly one path: counter was invoked once per result.
        assert len(status.result_paths) == 3

    asyncio.run(_run())
