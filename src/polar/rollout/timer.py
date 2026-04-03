"""Per-session stage timing utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from polar.rollout.models import SessionTiming


@dataclass(slots=True)
class StageTimer:
    """Record monotonic timestamps for session stages."""

    _marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, event: str) -> None:
        """Mark a stage start or finish."""
        self._marks[f"{stage}_{event}"] = time.monotonic()

    def to_session_timing(self) -> SessionTiming:
        """Return durations for the init/run/post-run lifecycle."""
        init_ms = self._duration_ms("init")
        run_ms = self._duration_ms("run")
        build_ms = self._duration_ms("build")
        eval_ms = self._duration_ms("eval")
        postrun_ms = self._duration_ms("postrun")
        teardown_ms = self._duration_ms("teardown")
        total_start = (
            self._marks.get("dispatch_started")
            or self._marks.get("init_started")
            or self._marks.get("run_started")
            or self._marks.get("build_started")
            or self._marks.get("eval_started")
            or self._marks.get("postrun_started")
        )
        total_end = (
            self._marks.get("return_finished")
            or self._marks.get("teardown_finished")
            or self._marks.get("postrun_finished")
            or self._marks.get("eval_finished")
            or self._marks.get("build_finished")
            or self._marks.get("run_finished")
            or self._marks.get("init_finished")
            or total_start
        )
        total_ms = max(0.0, ((total_end or 0.0) - (total_start or total_end or 0.0)) * 1000.0)
        return SessionTiming(
            init_ms=init_ms,
            run_ms=run_ms,
            build_ms=build_ms,
            eval_ms=eval_ms,
            postrun_ms=postrun_ms,
            teardown_ms=teardown_ms,
            total_ms=total_ms,
        )

    def _duration_ms(self, stage: str) -> float:
        started = self._marks.get(f"{stage}_started")
        finished = self._marks.get(f"{stage}_finished") or started
        if started is None or finished is None:
            return 0.0
        return max(0.0, (finished - started) * 1000.0)
