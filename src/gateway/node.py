"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

import httpx

from gateway.session import SessionRegistry
from gateway.storage import SessionStore
from rollout.models import SessionDispatchRequest, SessionResult
from rollout.timer import StageTimer
from trajectory.models import AgentRunResult, EvalResult, StrategySpec, Trajectory
from trajectory.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class _ManagedSession:
    request: SessionDispatchRequest
    task: asyncio.Task[None]
    timer: StageTimer
    process: asyncio.subprocess.Process | None = None


class GatewayNodeManager:
    """Run the REGISTER/RUN/BUILD/EVAL/RETURN lifecycle on one gateway node."""

    def __init__(
        self,
        *,
        node_id: str,
        gateway_url: str,
        capacity: int,
        storage: SessionStore,
        session_registry: SessionRegistry,
        builders: StrategyRegistry,
        evaluators: StrategyRegistry,
        session_base_dir: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.capacity = max(1, capacity)
        self.storage = storage
        self.session_registry = session_registry
        self.builders = builders
        self.evaluators = evaluators
        self._session_base_dir = session_base_dir
        self._sessions: dict[str, _ManagedSession] = {}
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(self.capacity)
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
        for managed in sessions:
            managed.task.cancel()
        if sessions:
            await asyncio.gather(*(managed.task for managed in sessions), return_exceptions=True)
        await self._client.aclose()

    async def dispatch(self, request: SessionDispatchRequest) -> None:
        session_id = request.session_id
        existing = self.session_registry.get(session_id)
        if existing is not None and existing.status not in {"COMPLETED", "ERROR", "TIMEOUT"}:
            raise ValueError(f"session {session_id} is already active")

        info = self.session_registry.register(
            session_id,
            task_id=request.task_id,
            registered=True,
            status="REGISTERED",
        )
        self.storage.ensure_session(
            info.session_id,
            model_requested=None,
            model_used=None,
            api_type=None,
            task_id=info.task_id,
            created_at=info.created_at.isoformat(),
        )

        timer = StageTimer()
        timer.mark("dispatch", "started")
        task = asyncio.create_task(self._run_session(request, timer))
        async with self._lock:
            self._sessions[session_id] = _ManagedSession(
                request=request,
                task=task,
                timer=timer,
            )

    async def cancel(self, session_id: str) -> bool:
        async with self._lock:
            managed = self._sessions.get(session_id)
        if managed is None:
            return False

        process = managed.process
        if process is not None and process.returncode is None:
            process.kill()
            with suppress(ProcessLookupError):
                await process.wait()
        managed.task.cancel()
        return True

    async def active_sessions(self) -> int:
        async with self._lock:
            return sum(
                1
                for managed in self._sessions.values()
                if not managed.task.done()
            )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _run_session(self, request: SessionDispatchRequest, timer: StageTimer) -> None:
        session_id = request.session_id
        task_id = request.task_id
        agent_result: AgentRunResult | None = None
        result: SessionResult | None = None

        session_dir = Path(mkdtemp(
            prefix=f"session-{session_id[:8]}-",
            dir=self._session_base_dir,
        ))
        artifacts_dir = session_dir / "artifacts"
        artifacts_dir.mkdir()

        await self._slots.acquire()
        try:
            self.session_registry.set_status(session_id, "RUNNING")

            # --- RUN ---
            timer.mark("run", "started")
            try:
                agent_result = await self._run_agent(request, session_dir, artifacts_dir)
            finally:
                timer.mark("run", "finished")

            # --- BUILD ---
            self.session_registry.set_status(session_id, "BUILDING")
            timer.mark("build", "started")
            trajectory = await asyncio.to_thread(self._build_trajectory, request)
            timer.mark("build", "finished")

            error = trajectory.error
            if agent_result.status == "timeout":
                trajectory = trajectory.model_copy(update={"status": "TIMEOUT", "error": agent_result.error or error})
            elif agent_result.status == "failed":
                trajectory = trajectory.model_copy(update={"status": "ERROR", "error": agent_result.error or error})

            # --- EVALUATE ---
            timer.mark("eval", "started")
            if request.evaluator is not None:
                self.session_registry.set_status(session_id, "EVALUATING")
                trajectory = await self._run_eval(
                    request,
                    trajectory,
                    agent_result=agent_result,
                    session_dir=session_dir,
                    artifacts_dir=artifacts_dir,
                )
            timer.mark("eval", "finished")

            # --- persist trajectory to session workspace ---
            await asyncio.to_thread(
                self._persist_trajectory, session_dir, trajectory,
            )

            status = trajectory.status
            if trajectory.error:
                error = trajectory.error

            result = SessionResult(
                session_id=session_id,
                task_id=task_id,
                status=status,
                trajectory=trajectory.model_copy(update={"error": error}),
                timing=timer.to_session_timing(),
                node_id=self.node_id,
                error=error,
            )
        except asyncio.CancelledError:
            timer.mark("eval", "finished")
            result = SessionResult(
                session_id=session_id,
                task_id=task_id,
                status="ERROR",
                trajectory=Trajectory(
                    status="ERROR",
                    metadata={"builder": request.builder.strategy, "record_count": 0},
                    traces=[],
                    error="session cancelled",
                ),
                timing=timer.to_session_timing(),
                node_id=self.node_id,
                error="session cancelled",
            )
            raise
        except Exception as exc:
            logger.exception("Gateway node session %s failed", session_id)
            result = SessionResult(
                session_id=session_id,
                task_id=task_id,
                status="ERROR",
                trajectory=Trajectory(
                    status="ERROR",
                    metadata={"builder": request.builder.strategy, "record_count": 0},
                    traces=[],
                    error=str(exc),
                ),
                timing=timer.to_session_timing(),
                node_id=self.node_id,
                error=str(exc),
            )
        finally:
            timer.mark("return", "finished")
            self._slots.release()
            if result is not None:
                self.session_registry.set_result(session_id, result)
                await self._push_result(request.callback_url, result)
            async with self._lock:
                self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # RUN stage
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        request: SessionDispatchRequest,
        session_dir: Path,
        artifacts_dir: Path,
    ) -> AgentRunResult:
        agent = request.agent
        command = agent.run_cmd.format(
            session_id=request.session_id,
            session_id_quoted=shlex.quote(request.session_id),
            task_id=request.task_id,
            task_id_quoted=shlex.quote(request.task_id),
            gateway_url=self.gateway_url,
            gateway_url_quoted=shlex.quote(self.gateway_url),
            api_key=request.session_id,
            api_key_quoted=shlex.quote(request.session_id),
        )

        stdout_path = session_dir / "agent.stdout.log"
        stderr_path = session_dir / "agent.stderr.log"

        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": self.gateway_url,
            "ANTHROPIC_API_KEY": request.session_id,
            "OPENAI_BASE_URL": f"{self.gateway_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": request.session_id,
            "GOOGLE_API_URL": self.gateway_url,
            "GOOGLE_API_KEY": request.session_id,
            "SESSION_ID": request.session_id,
            "TASK_ID": request.task_id,
            "SESSION_DIR": str(session_dir),
            "ARTIFACTS_DIR": str(artifacts_dir),
            **{key: str(value) for key, value in agent.env.items()},
        }

        stdout_fh = stdout_path.open("w")
        stderr_fh = stderr_path.open("w")
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
            )
            async with self._lock:
                managed = self._sessions.get(request.session_id)
                if managed is not None:
                    managed.process = process

            try:
                await asyncio.wait_for(process.wait(), timeout=agent.timeout)
            except asyncio.TimeoutError:
                process.kill()
                with suppress(ProcessLookupError):
                    await process.wait()
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error="agent execution timed out",
                    metadata={
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "cwd": str(session_dir),
                    },
                )
        finally:
            stdout_fh.close()
            stderr_fh.close()

        metadata: dict = {
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "cwd": str(session_dir),
        }

        if process.returncode == 0:
            return AgentRunResult(status="completed", return_code=0, metadata=metadata)
        return AgentRunResult(
            status="failed",
            return_code=process.returncode if process.returncode is not None else -1,
            error=f"agent exited with return code {process.returncode}",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # BUILD stage
    # ------------------------------------------------------------------

    def _build_trajectory(self, request: SessionDispatchRequest) -> Trajectory:
        completion_session = self.storage.load_completion_session(request.session_id)
        builder = self.builders.create(request.builder)
        result = builder.build(completion_session)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return Trajectory.model_validate(result)

    # ------------------------------------------------------------------
    # EVALUATE + MERGE stage
    # ------------------------------------------------------------------

    async def _run_eval(
        self,
        request: SessionDispatchRequest,
        trajectory: Trajectory,
        *,
        agent_result: AgentRunResult,
        session_dir: Path,
        artifacts_dir: Path,
    ) -> Trajectory:
        evaluator_spec = request.evaluator
        if evaluator_spec is None:
            return trajectory

        try:
            evaluator = self.evaluators.create(evaluator_spec)
            eval_result = await evaluator.evaluate(
                trajectory,
                session_id=request.session_id,
                task_id=request.task_id,
                session_dir=session_dir,
                artifacts_dir=artifacts_dir,
                agent_result=agent_result,
            )
        except Exception as exc:
            logger.exception("Evaluator %s failed for session %s", evaluator_spec.strategy, request.session_id)
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )

        return self._merge_eval_result(trajectory, eval_result, evaluator_spec)

    @staticmethod
    def _merge_eval_result(
        trajectory: Trajectory,
        eval_result: EvalResult,
        evaluator_spec: StrategySpec,
    ) -> Trajectory:
        """Apply rewards from EvalResult to trajectory traces."""
        traces = list(trajectory.traces)

        if eval_result.trace_rewards is not None:
            if len(eval_result.trace_rewards) != len(traces):
                return trajectory.model_copy(
                    update={
                        "status": "ERROR",
                        "error": (
                            f"evaluator returned {len(eval_result.trace_rewards)} "
                            f"trace_rewards but trajectory has {len(traces)} traces"
                        ),
                    }
                )
            traces = [
                trace.model_copy(update={"reward": reward})
                for trace, reward in zip(traces, eval_result.trace_rewards)
            ]
        elif eval_result.outcome_reward is not None and traces:
            last = traces[-1].model_copy(update={"reward": eval_result.outcome_reward})
            traces = traces[:-1] + [last]

        eval_metadata = {
            "strategy": evaluator_spec.strategy,
            "outcome_reward": eval_result.outcome_reward,
            "trace_rewards": eval_result.trace_rewards,
            **eval_result.metadata,
        }

        metadata = {**trajectory.metadata, "evaluation": eval_metadata}

        return trajectory.model_copy(update={"traces": traces, "metadata": metadata})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _persist_trajectory(session_dir: Path, trajectory: Trajectory) -> None:
        path = session_dir / "trajectory.json"
        path.write_text(json.dumps(trajectory.model_dump(), separators=(",", ":"), default=str))

    async def _push_result(self, callback_url: str | None, result: SessionResult) -> None:
        if not callback_url:
            return
        try:
            response = await self._client.post(callback_url, json=result.model_dump(mode="json"))
            response.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to deliver callback for session %s to %s",
                result.session_id,
                callback_url,
                exc_info=True,
            )
