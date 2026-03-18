"""Top-level task orchestration for rollout batches."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from rollout.balancer import NodeScheduler
from rollout.models import (
    SessionContext,
    SessionResult,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from rollout.pipeline import Pipeline


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int = 0
    results: list[SessionResult] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)


class RolloutManager:
    """Manage the lifecycle of rollout sessions for a single submitted task."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        scheduler: NodeScheduler,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler
        self._tasks: dict[str, _TaskRecord] = {}
        self._lock = threading.RLock()

    async def execute_task(self, request: TaskRequest) -> TaskResult:
        with self._lock:
            existing = self._tasks.get(request.task_id)
            if existing is not None and existing.status == "running":
                raise ValueError(f"task {request.task_id} is already running")
            self._tasks[request.task_id] = _TaskRecord(
                task_id=request.task_id,
                status="running",
                total_sessions=request.num_rollouts,
            )

        sessions = [
            SessionContext(
                session_id=str(uuid.uuid4()),
                task_id=request.task_id,
                request=request,
            )
            for _ in range(request.num_rollouts)
        ]

        async def _on_result(result: SessionResult) -> None:
            result_path = self.pipeline.result_path_for(result.task_id, result.session_id)
            with self._lock:
                record = self._tasks[request.task_id]
                record.completed_sessions += 1
                record.results.append(result)
                if result_path is not None:
                    record.result_paths.append(result_path)

        try:
            results = await self.pipeline.run_batch(sessions, on_result=_on_result)
        except Exception:
            with self._lock:
                self._tasks[request.task_id].status = "failed"
            raise

        ordered_results = list(results)
        result_paths: list[str] = []
        for result in results:
            result_path = self.pipeline.result_path_for(result.task_id, result.session_id)
            if result_path is not None:
                result_paths.append(result_path)
        task_result = TaskResult(
            task_id=request.task_id,
            status="completed",
            results=ordered_results,
            result_paths=result_paths,
        )

        with self._lock:
            record = self._tasks[request.task_id]
            record.status = task_result.status
            record.completed_sessions = len(ordered_results)
            record.results = ordered_results
            record.result_paths = result_paths

        return task_result

    def get_task(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            return TaskStatus(
                task_id=record.task_id,
                status=record.status,
                total_sessions=record.total_sessions,
                completed_sessions=record.completed_sessions,
                results=list(record.results),
                result_paths=list(record.result_paths),
            )

    def status(self) -> dict[str, object]:
        with self._lock:
            task_statuses = {
                task_id: record.status
                for task_id, record in self._tasks.items()
            }
        return {
            "tasks": task_statuses,
            "pipeline": self.pipeline.status(),
            "nodes": self.scheduler.stats(),
        }
