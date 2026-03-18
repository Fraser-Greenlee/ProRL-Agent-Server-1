"""Built-in trajectory builder that preserves every completion record."""

from __future__ import annotations

from trajectory.builder.base import BaseTrajectoryBuilder
from trajectory.builder.record_utils import build_trace_from_completion
from trajectory.models import CompletionSession, Trajectory


class AllRecordsBuilder(BaseTrajectoryBuilder):
    """Convert every stored completion record into one trajectory trace."""

    async def build(self, session: CompletionSession) -> Trajectory:
        if not session.completions:
            return Trajectory(
                status="ERROR",
                metadata={
                    "builder": "all_records",
                    "session_id": session.session_id,
                    "record_count": 0,
                },
                traces=[],
                error="no completions",
            )

        return Trajectory(
            status="COMPLETED",
            metadata={
                "builder": "all_records",
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "model_requested": session.model_requested,
                "model_used": session.model_used,
                "record_count": len(session.completions),
            },
            traces=[
                build_trace_from_completion(completion)
                for completion in session.completions
            ],
        )
