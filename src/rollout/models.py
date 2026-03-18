"""Shared data models for rollout orchestration and gateway-node execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from trajectory.models import AgentSpec, StrategySpec, Trajectory

if TYPE_CHECKING:
    from rollout.timer import StageTimer


def _new_stage_timer() -> "StageTimer":
    from rollout.timer import StageTimer

    return StageTimer()


def _default_builder_spec() -> StrategySpec:
    return StrategySpec(strategy="all_records")


class TaskRequest(BaseModel):
    """Task submitted by the trainer."""

    task_id: str
    num_rollouts: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: StrategySpec | None = None


class SessionDispatchRequest(BaseModel):
    """Session lifecycle request sent from the rollout server to a gateway node."""

    session_id: str
    task_id: str
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: StrategySpec | None = None
    callback_url: str | None = None


class SessionDispatchResponse(BaseModel):
    """Acknowledgement returned by a gateway node when a session is accepted."""

    session_id: str
    task_id: str
    status: str
    node_id: str | None = None


class SessionTiming(BaseModel):
    """Per-session durations in milliseconds."""

    run_ms: float = 0.0
    build_ms: float = 0.0
    eval_ms: float = 0.0
    total_ms: float = 0.0


class SessionResult(BaseModel):
    """Terminal node result returned to the rollout server."""

    session_id: str
    task_id: str
    status: str
    trajectory: Trajectory
    timing: SessionTiming = Field(default_factory=SessionTiming)
    node_id: str | None = None
    error: str | None = None


class TaskResult(BaseModel):
    """Blocking response returned once all rollout sessions resolve."""

    task_id: str
    status: str
    results: list[SessionResult]
    result_paths: list[str] = Field(default_factory=list)


class TaskStatus(BaseModel):
    """Monitoring view for a task that may still be running."""

    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int
    results: list[SessionResult] = Field(default_factory=list)
    result_paths: list[str] = Field(default_factory=list)


class NodeRegistrationRequest(BaseModel):
    """Payload sent by a gateway node when registering with the rollout server."""

    node_id: str
    gateway_url: str
    capacity: int = Field(ge=1)
    heartbeat_interval_seconds: int = Field(default=30, ge=1)


class NodeHeartbeatRequest(BaseModel):
    """Heartbeat payload sent by a gateway node."""

    active_sessions: int | None = Field(default=None, ge=0)


class GatewayNodeInfo(BaseModel):
    """External view of one schedulable gateway node."""

    node_id: str
    gateway_url: str
    capacity: int
    active_sessions: int
    healthy: bool
    draining: bool = False
    heartbeat_interval_seconds: int
    last_heartbeat: datetime


@dataclass(slots=True)
class SessionContext:
    """Internal state that flows through dispatch and collection."""

    session_id: str
    task_id: str
    request: TaskRequest
    node_id: str | None = None
    gateway_url: str | None = None
    timer: "StageTimer" = field(default_factory=_new_stage_timer)
    rollout_result: SessionResult | None = None
    completion_future: asyncio.Future[SessionResult] | None = field(
        default=None,
        repr=False,
    )
