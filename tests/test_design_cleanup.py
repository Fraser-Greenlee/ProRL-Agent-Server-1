from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.agent.base import BaseHarness
from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec
from polar.gateway.node import GatewayNodeManager
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionDispatchRequest, SessionResult
from polar.runtime.factory import create_runtime
from polar.runtime.models import ExecInput, RuntimeSpec
from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.models import StrategySpec, Trajectory
from polar.trajectory.registry import StrategyRegistry, default_builder_registry, default_evaluator_registry


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


async def _wait_for(
    predicate,
    *,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for expected condition")


@pytest.mark.asyncio
async def test_gateway_status_stays_initializing_while_ready_slot_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    request = _dispatch_request("session-stage-sync")
    slot_held = False

    async def _noop_push(*args, **kwargs) -> None:
        return None

    async def _fake_init(managed) -> None:
        del managed

    async def _fake_run(managed) -> None:
        managed.final_result = SessionResult(
            session_id=managed.request.session_id,
            task_id=managed.request.task_id,
            status="COMPLETED",
            trajectory=Trajectory(
                status="COMPLETED",
                metadata={"builder": managed.request.builder.strategy, "record_count": 0},
                traces=[],
            ),
            node_id=manager.node_id,
        )

    monkeypatch.setattr(manager, "_push_result", _noop_push)
    manager._dispatcher.on_init = _fake_init
    manager._dispatcher.on_run = _fake_run

    await manager.start()
    await manager._dispatcher._ready_slots.acquire()
    slot_held = True
    try:
        await manager.dispatch(request)
        await _wait_for(
            lambda: (
                (info := manager.session_registry.get(request.session_id)) is not None
                and info.status == "INITIALIZING"
            )
        )

        await asyncio.sleep(0.3)
        info = manager.session_registry.get(request.session_id)
        assert info is not None
        assert info.status == "INITIALIZING"

        manager._dispatcher._ready_slots.release()
        slot_held = False

        await _wait_for(
            lambda: (
                (info := manager.session_registry.get(request.session_id)) is not None
                and info.status == "COMPLETED"
            )
        )
    finally:
        if slot_held:
            manager._dispatcher._ready_slots.release()
        await manager.close()


def test_shared_import_loader_keeps_plugin_factories_working(tmp_path: Path) -> None:
    harness = create_harness(
        AgentSpec(import_path="polar.agent.harnesses.opencode:OpenCodeHarness")
    )
    assert harness.__class__.__name__ == "OpenCodeHarness"

    runtime = create_runtime(
        RuntimeSpec(
            import_path="polar.runtime.docker:DockerRuntime",
            image="example:latest",
        ),
        "session-1",
        tmp_path / "session-1",
    )
    assert runtime.__class__.__name__ == "DockerRuntime"

    registry: StrategyRegistry[BaseTrajectoryBuilder] = StrategyRegistry(
        BaseTrajectoryBuilder
    )
    registry.register(
        "all_records",
        "polar.trajectory.builder.all_records:AllRecordsBuilder",
    )
    builder = registry.create(StrategySpec(strategy="all_records"))
    assert builder.__class__.__name__ == "AllRecordsBuilder"


class _LegacyCleanupHarness(BaseHarness):
    def run_steps(self, instruction: str) -> list[ExecInput]:
        del instruction
        return []

    def postrun_steps(self) -> list[ExecInput]:
        return [ExecInput(command="explicit-postrun")]


class _ModernPostrunHarness(BaseHarness):
    def run_steps(self, instruction: str) -> list[ExecInput]:
        del instruction
        return []

    def postrun_steps(self) -> list[ExecInput]:
        return [ExecInput(command="modern-postrun")]


def test_harness_postrun_hook_is_the_single_teardown_extension_point() -> None:
    spec = AgentSpec(harness="opencode")

    explicit = _LegacyCleanupHarness(spec)
    assert explicit.postrun_steps()[0].command == "explicit-postrun"

    modern = _ModernPostrunHarness(spec)
    assert modern.postrun_steps()[0].command == "modern-postrun"
