"""Shared data models for rollout orchestration and gateway-node execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from polar.agent.models import AgentSpec
from polar.runtime.models import RuntimeSpec
from polar.trajectory.models import EvaluatorSpec, StrategySpec, Trajectory

if TYPE_CHECKING:
    from polar.rollout.timer import StageTimer


class SessionStatus(StrEnum):
    """Canonical session lifecycle statuses.

    StrEnum instances serialize to their string values, so wire compatibility
    with older clients that read plain status strings is preserved.
    """

    REGISTERED = "REGISTERED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    POST_RUN = "POST_RUN"
    BUILDING = "BUILDING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def terminal(cls) -> frozenset["SessionStatus"]:
        return frozenset({cls.COMPLETED, cls.ERROR, cls.TIMEOUT})

    @classmethod
    def active(cls) -> frozenset["SessionStatus"]:
        return frozenset(set(cls) - cls.terminal())


def _new_stage_timer() -> "StageTimer":
    from polar.rollout.timer import StageTimer

    return StageTimer()


def _default_builder_spec() -> StrategySpec:
    return StrategySpec(strategy="all_records")


class TaskRequest(BaseModel):
    """Task submitted by the trainer."""

    task_id: str
    instruction: str
    num_samples: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None


class SessionDispatchRequest(BaseModel):
    """Session lifecycle request sent from the rollout server to a gateway node.

    `remaining_timeout_seconds` is the live session budget left at dispatch time.
    """

    session_id: str
    task_id: str
    instruction: str
    remaining_timeout_seconds: float = Field(gt=0)
    runtime: RuntimeSpec | None = None
    agent: AgentSpec
    builder: StrategySpec = Field(default_factory=_default_builder_spec)
    evaluator: EvaluatorSpec | None = None
    callback_url: str | None = None


class SessionDispatchResponse(BaseModel):
    """Acknowledgement returned by a gateway node when a session is accepted."""

    session_id: str
    task_id: str
    status: SessionStatus
    node_id: str | None = None


class SessionTiming(BaseModel):
    """Per-session durations in milliseconds.

    Three phases only: runtime startup + prepare (`init_ms`), agent harness
    execution (`run_ms`), and everything after the agent stops — build, eval,
    teardown — rolled into (`postrun_ms`). Total wall-clock is the sum.
    """

    model_config = ConfigDict(extra="forbid")

    init_ms: float = 0.0
    run_ms: float = 0.0
    postrun_ms: float = 0.0


class SessionResult(BaseModel):
    """Terminal node result returned to the rollout server."""

    session_id: str
    task_id: str
    status: SessionStatus
    trajectory: Trajectory
    timing: SessionTiming = Field(default_factory=SessionTiming)
    node_id: str | None = None
    error: str | None = None


class TaskResult(BaseModel):
    """Blocking response returned once all rollout sessions resolve."""

    task_id: str
    status: str  # Task-level status vocabulary: "running" | "completed" | "failed"
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
    max_init_workers: int = Field(ge=1)
    max_run_workers: int = Field(ge=1)
    max_postrun_workers: int = Field(ge=1)
    heartbeat_interval_seconds: int = Field(default=30, ge=1)


class NodeStageMetrics(BaseModel):
    """Per-node stage occupancy and queue depths."""

    init_queue_depth: int = Field(default=0, ge=0)
    init_inflight: int = Field(default=0, ge=0)
    ready_depth: int = Field(default=0, ge=0)
    run_inflight: int = Field(default=0, ge=0)
    postrun_queue_depth: int = Field(default=0, ge=0)
    postrun_inflight: int = Field(default=0, ge=0)

    @property
    def total_sessions(self) -> int:
        return (
            self.init_queue_depth
            + self.init_inflight
            + self.ready_depth
            + self.run_inflight
            + self.postrun_queue_depth
            + self.postrun_inflight
        )


class NodeHeartbeatRequest(BaseModel):
    """Heartbeat payload sent by a gateway node."""

    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)


class GatewayNodeInfo(BaseModel):
    """External view of one schedulable gateway node."""

    node_id: str
    gateway_url: str
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    metrics: NodeStageMetrics = Field(default_factory=NodeStageMetrics)
    dispatch_reservations: int = Field(default=0, ge=0)
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
    deadline_monotonic: float = field(default_factory=time.monotonic)
    node_id: str | None = None
    gateway_url: str | None = None
    timer: "StageTimer" = field(default_factory=_new_stage_timer)
    rollout_result: SessionResult | None = None
    completion_future: asyncio.Future[SessionResult] | None = field(
        default=None,
        repr=False,
    )
