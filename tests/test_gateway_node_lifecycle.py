from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from polar.agent.harnesses.opencode import OpenCodeHarness
from polar.agent.models import AgentSpec, MCPServerSpec
from polar.gateway.dispatcher import ManagedSession
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionResult
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trajectory
from polar.trajectory.registry import default_builder_registry, default_evaluator_registry


class _StopOnlyRuntime:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _RecordingRuntime:
    def __init__(self) -> None:
        self.commands: list[tuple[str, dict[str, object]]] = []

    async def exec(self, command: str, **kwargs):
        self.commands.append((command, kwargs))
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _make_manager(tmp_path: Path) -> GatewayNodeManager:
    return GatewayNodeManager(
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


def _dispatch_request(session_id: str) -> SessionDispatchRequest:
    return SessionDispatchRequest(
        session_id=session_id,
        task_id="task-1",
        instruction="solve it",
        remaining_timeout_seconds=60.0,
        agent=AgentSpec(harness="opencode"),
    )


@pytest.mark.asyncio
async def test_gateway_dispatch_failure_does_not_burn_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    request = _dispatch_request("session-retry")
    created_dirs: list[Path] = []

    async def _failing_enqueue(managed) -> None:
        created_dirs.append(managed.session_dir)
        raise RuntimeError("enqueue failed")

    async def _capture_enqueue(managed) -> None:
        created_dirs.append(managed.session_dir)

    monkeypatch.setattr(manager._dispatcher, "enqueue", _failing_enqueue)

    try:
        with pytest.raises(RuntimeError, match="enqueue failed"):
            await manager.dispatch(request)

        assert manager.session_registry.get(request.session_id) is None
        assert manager.storage.get_session_metadata(request.session_id) is None
        assert all(not path.exists() for path in created_dirs)

        monkeypatch.setattr(manager._dispatcher, "enqueue", _capture_enqueue)
        await manager.dispatch(request)

        assert manager.session_registry.get(request.session_id) is not None
        assert manager.storage.get_session_metadata(request.session_id) is not None
    finally:
        for path in created_dirs:
            shutil.rmtree(path, ignore_errors=True)
        await manager.close()


@pytest.mark.asyncio
async def test_gateway_dispatch_rejects_reused_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    request = _dispatch_request("session-single-use")
    created_dirs: list[Path] = []

    async def _capture_enqueue(managed) -> None:
        created_dirs.append(managed.session_dir)

    monkeypatch.setattr(manager._dispatcher, "enqueue", _capture_enqueue)

    try:
        await manager.dispatch(request)
        manager.session_registry.remove(request.session_id)
        manager.storage.delete_session(request.session_id)

        with pytest.raises(ValueError, match="single-use"):
            await manager.dispatch(request)
    finally:
        for path in created_dirs:
            shutil.rmtree(path, ignore_errors=True)
        await manager.close()


@pytest.mark.asyncio
async def test_gateway_postrun_cleans_session_dir_and_completion_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    request = _dispatch_request("session-postrun")
    session_dir = tmp_path / "session-postrun"
    artifacts_dir = session_dir / "artifacts"
    (session_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "logs" / "agent" / "run.log").write_text("agent output")

    info = manager.session_registry.register(
        request.session_id,
        task_id=request.task_id,
        registered=True,
        status="RUNNING",
    )
    manager.storage.ensure_session(
        request.session_id,
        model_requested=None,
        model_used=None,
        api_type=None,
        task_id=request.task_id,
        created_at=info.created_at.isoformat(),
    )
    manager.storage.save_message(
        request.session_id,
        {"model": "stub-model"},
        {"id": "response-1"},
        task_id=request.task_id,
        created_at=info.created_at.isoformat(),
    )

    runtime = _StopOnlyRuntime()

    async def _noop_push(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(manager, "_push_result", _noop_push)

    managed = ManagedSession(
        request=request,
        timer=StageTimer(),
        session_dir=session_dir,
        artifacts_dir=artifacts_dir,
        runtime=runtime,
        final_result=SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="COMPLETED",
            trajectory=Trajectory(
                status="COMPLETED",
                metadata={"builder": request.builder.strategy, "record_count": 1},
                traces=[],
            ),
            node_id=manager.node_id,
        ),
        execution_deadline=asyncio.get_running_loop().time() + 60.0,
    )

    try:
        await manager._handle_postrun(managed)

        stored = manager.session_registry.get(request.session_id)
        assert stored is not None
        assert stored.result is not None
        assert stored.status == "COMPLETED"
        assert manager.storage.get_session_metadata(request.session_id) is None
        assert runtime.stopped is True
        assert not session_dir.exists()
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
        await manager.close()


@pytest.mark.asyncio
async def test_opencode_setup_writes_config_for_default_model() -> None:
    runtime = _RecordingRuntime()
    harness = OpenCodeHarness(
        AgentSpec(
            harness="opencode",
            mcp_servers=[
                MCPServerSpec(
                    name="docs",
                    transport="streamable-http",
                    url="http://mcp.local",
                )
            ],
            skills_path="/opt/polar-skills",
        )
    )

    await harness.setup(runtime)

    commands = [command for command, _ in runtime.commands]
    assert len(commands) == 3
    assert commands[0] == "mkdir -p /root/.config/opencode"
    assert any(
        "opencode.json" in command
        and "gpt-4o" in command
        and '"mcp"' in command
        for command in commands
    )
    assert any("/opt/polar-skills" in command for command in commands)
