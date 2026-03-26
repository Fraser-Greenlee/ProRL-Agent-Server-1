"""Gateway-node execution lifecycle for dispatched rollout sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp

import httpx

from gateway.dispatcher import (
    DispatcherSnapshot,
    ManagedSession,
    PreparedRuntimeLease,
    SessionDispatcher,
)
from gateway.session import SessionRegistry
from gateway.storage import SessionStore
from integration.base import BaseHarness
from integration.factory import create_harness
from integration.models import AgentRunResult
from rollout.models import NodeStageMetrics, SessionDispatchRequest, SessionResult
from rollout.timer import StageTimer
from runtime.base import BaseRuntime
from runtime.factory import create_runtime
from runtime.models import ExecInput, RuntimeSpec
from trajectory.models import EvalResult, EvaluatorSpec, StrategySpec, Trajectory
from trajectory.registry import StrategyRegistry

logger = logging.getLogger(__name__)


class GatewayNodeManager:
    """Run the INIT/READY/RUN/POST_RUN lifecycle on one gateway node."""

    def __init__(
        self,
        *,
        node_id: str,
        gateway_url: str,
        max_init_workers: int,
        max_run_workers: int,
        max_postrun_workers: int,
        ready_buffer_target: int,
        storage: SessionStore,
        session_registry: SessionRegistry,
        builders: StrategyRegistry,
        evaluators: StrategyRegistry,
        default_runtime: RuntimeSpec | None = None,
        session_base_dir: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.storage = storage
        self.session_registry = session_registry
        self.builders = builders
        self.evaluators = evaluators
        self.default_runtime = default_runtime
        self._session_base_dir = session_base_dir
        self._client = httpx.AsyncClient(timeout=30.0)
        self._dispatcher = SessionDispatcher(
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
            ready_buffer_target=ready_buffer_target,
        )
        self._dispatcher.on_init = self._handle_init
        self._dispatcher.on_run = self._handle_run
        self._dispatcher.on_postrun = self._handle_postrun

    async def start(self) -> None:
        await self._dispatcher.start()

    async def close(self) -> None:
        await self._dispatcher.stop()
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
        session_dir = Path(mkdtemp(prefix=f"session-{session_id[:8]}-", dir=self._session_base_dir))
        artifacts_dir = session_dir / "artifacts"
        artifacts_dir.mkdir()
        (session_dir / "logs" / "agent").mkdir(parents=True, exist_ok=True)
        await self._dispatcher.enqueue(
            ManagedSession(
                request=request,
                timer=timer,
                session_dir=session_dir,
                artifacts_dir=artifacts_dir,
            )
        )

    async def cancel(self, session_id: str) -> bool:
        return await self._dispatcher.cancel(session_id)

    async def active_sessions(self) -> int:
        return await self._dispatcher.active_count()

    async def stage_metrics(self) -> NodeStageMetrics:
        snapshot = await self._dispatcher.snapshot()
        return self._snapshot_to_metrics(snapshot)

    # ------------------------------------------------------------------
    # INIT stage
    # ------------------------------------------------------------------

    async def _handle_init(self, managed: ManagedSession) -> None:
        request = managed.request
        self.session_registry.set_status(request.session_id, "INITIALIZING")
        managed.timer.mark("init", "started")
        try:
            runtime_spec = self._resolve_runtime_spec(request)
            runtime = create_runtime(runtime_spec, request.session_id, managed.session_dir)
            managed.runtime = runtime
            await runtime.start()
            # Run ordered prepare actions
            await self._run_runtime_prepare(runtime, runtime_spec, request, managed)
        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Initialization cancelled for session %s", request.session_id)
            else:
                logger.exception("Initialization failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"runtime initialization failed: {exc}",
                )
        finally:
            managed.timer.mark("init", "finished")
        if managed.final_result is None and not managed.cancel_requested:
            self.session_registry.set_status(request.session_id, "READY")

    def _resolve_runtime_spec(self, request: SessionDispatchRequest) -> RuntimeSpec:
        spec = request.runtime or self.default_runtime
        if spec is None:
            raise RuntimeError(
                "no runtime configured: request has no runtime and gateway "
                "node has no default_runtime"
            )
        return spec

    async def _run_runtime_prepare(
        self,
        runtime: BaseRuntime,
        spec: RuntimeSpec,
        request: SessionDispatchRequest,
        managed: ManagedSession,
    ) -> None:
        """Execute the ordered prepare action list."""
        base_env = self._runtime_env(request, managed)
        for i, action in enumerate(spec.prepare):
            if managed.cancel_requested:
                return
            if action.type == "upload_file":
                await runtime.upload_file(action.source, action.target)
            elif action.type == "upload_dir":
                await runtime.upload_dir(action.source, action.target)
            elif action.type == "exec":
                merged_env = {**base_env, **(action.env or {})}
                # Use action.cwd, falling back to runtime session dir
                # (not spec.workdir which may not exist during prepare)
                effective_cwd = action.cwd or runtime.runtime_session_dir
                result = await runtime.exec(
                    action.command,
                    cwd=effective_cwd,
                    env=merged_env,
                    timeout_sec=action.timeout_sec,
                )
                log_dir = managed.session_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self._write_exec_log(
                    log_dir, f"prepare.{i:02d}", result.stdout, result.stderr
                )
                if result.return_code == -1:
                    raise RuntimeError(f"prepare action {i} timed out")
                if result.return_code != 0:
                    raise RuntimeError(
                        f"prepare action {i} failed with exit code {result.return_code}"
                    )

    # ------------------------------------------------------------------
    # RUN stage
    # ------------------------------------------------------------------

    async def _handle_run(self, managed: ManagedSession) -> None:
        request = managed.request
        if managed.final_result is not None or managed.cancel_requested:
            return
        self.session_registry.set_status(request.session_id, "RUNNING")
        managed.timer.mark("run", "started")

        # Start evaluator runtime prewarm if needed
        self._maybe_start_eval_runtime_prewarm(managed)

        harness: BaseHarness | None = None
        try:
            runtime = managed.runtime
            if runtime is None:
                raise RuntimeError("runtime is required for execution")

            harness = self._resolve_agent_harness(request)

            # Setup
            await harness.setup(runtime)

            # Run
            steps = harness.run_steps(request.instruction)
            env = self._runtime_env(request, managed, include_agent_env=True)
            agent_result = await self._run_exec_inputs(
                runtime, steps, env, managed, timeout=request.agent.timeout
            )

            # Postprocess
            await harness.postprocess(runtime, agent_result)
            managed.agent_result = agent_result

        except Exception as exc:
            if managed.cancel_requested:
                logger.info("Agent execution cancelled for session %s", request.session_id)
            else:
                logger.exception("Agent execution failed for session %s", request.session_id)
                managed.final_result = self._error_result(
                    request,
                    managed.timer,
                    f"agent execution failed: {exc}",
                )
        finally:
            # Cleanup
            if harness is not None:
                cleanup_steps = harness.cleanup_steps()
                if cleanup_steps and managed.runtime is not None:
                    for step in cleanup_steps:
                        try:
                            await managed.runtime.exec(
                                step.command, cwd=step.cwd, env=step.env
                            )
                        except Exception:
                            logger.debug(
                                "Cleanup step failed for session %s",
                                request.session_id,
                                exc_info=True,
                            )
            managed.timer.mark("run", "finished")

    def _resolve_agent_harness(self, request: SessionDispatchRequest) -> BaseHarness:
        return create_harness(request.agent)

    async def _run_exec_inputs(
        self,
        runtime: BaseRuntime,
        steps: list[ExecInput],
        env: dict[str, str],
        managed: ManagedSession,
        *,
        timeout: float,
    ) -> AgentRunResult:
        """Execute a list of ExecInput steps and return an AgentRunResult."""
        log_dir = managed.session_dir / "logs" / "agent"
        log_dir.mkdir(parents=True, exist_ok=True)

        for i, step in enumerate(steps):
            if managed.cancel_requested:
                return AgentRunResult(
                    status="failed", return_code=-1, error="cancelled"
                )
            merged_env = {**env, **(step.env or {})}
            effective_timeout = step.timeout_sec or timeout
            result = await runtime.exec(
                step.command,
                cwd=step.cwd,
                env=merged_env,
                timeout_sec=effective_timeout,
            )
            self._write_exec_log(
                log_dir, f"step.{i:02d}", result.stdout, result.stderr
            )
            if result.return_code == -1:
                return AgentRunResult(
                    status="timeout",
                    return_code=-1,
                    error=f"step {i} timed out",
                    metadata=self._step_metadata(log_dir, i, managed),
                )
            if result.return_code != 0:
                return AgentRunResult(
                    status="failed",
                    return_code=result.return_code,
                    error=f"step {i} exited with code {result.return_code}",
                    metadata=self._step_metadata(log_dir, i, managed),
                )

        return AgentRunResult(
            status="completed",
            return_code=0,
            metadata=self._step_metadata(log_dir, len(steps) - 1, managed),
        )

    # ------------------------------------------------------------------
    # Evaluator runtime prewarm
    # ------------------------------------------------------------------

    def _maybe_start_eval_runtime_prewarm(self, managed: ManagedSession) -> None:
        request = managed.request
        if request.evaluator is None or not request.evaluator.refresh_runtime:
            return
        lease = PreparedRuntimeLease(
            owner_session_id=request.session_id,
            purpose="evaluator_refresh",
        )
        managed.eval_runtime_lease = lease
        managed.eval_prewarm_task = asyncio.create_task(
            self._prewarm_eval_runtime(managed, lease)
        )

    async def _prewarm_eval_runtime(
        self, managed: ManagedSession, lease: PreparedRuntimeLease
    ) -> None:
        """Prepare a fresh runtime for evaluator use."""
        request = managed.request
        try:
            # Acquire a ready slot
            acquired = await self._dispatcher.acquire_ready_slot_for_eval(
                request.session_id
            )
            if not acquired or lease.cancelled:
                return
            try:
                runtime_spec = self._resolve_runtime_spec(request)
                eval_session_dir = managed.session_dir / "eval_runtime"
                eval_artifacts_dir = eval_session_dir / "artifacts"
                eval_artifacts_dir.mkdir(parents=True, exist_ok=True)

                eval_runtime = create_runtime(
                    runtime_spec, f"{request.session_id}-eval", eval_session_dir
                )
                await eval_runtime.start()
                # Run prepare actions for the fresh runtime
                await self._run_runtime_prepare(
                    eval_runtime, runtime_spec, request, managed
                )

                lease.runtime = eval_runtime
                lease.session_dir = eval_session_dir
                lease.artifacts_dir = eval_artifacts_dir
            except Exception as exc:
                lease.error = str(exc)
                logger.warning(
                    "Eval runtime prewarm failed for session %s: %s",
                    request.session_id,
                    exc,
                )
            finally:
                self._dispatcher.release_ready_slot()
        finally:
            lease.ready.set()

    async def _acquire_prepared_eval_runtime(
        self, managed: ManagedSession
    ) -> BaseRuntime | None:
        """Wait for and return the prewarmed evaluator runtime."""
        lease = managed.eval_runtime_lease
        if lease is None:
            return None
        await lease.ready.wait()
        if lease.error or lease.cancelled or lease.runtime is None:
            return None
        return lease.runtime

    # ------------------------------------------------------------------
    # POSTRUN stage
    # ------------------------------------------------------------------

    async def _handle_postrun(self, managed: ManagedSession) -> None:
        request = managed.request
        result: SessionResult | None = managed.final_result
        self.session_registry.set_status(request.session_id, "POST_RUN")
        managed.timer.mark("postrun", "started")
        try:
            if result is None:
                if managed.cancel_requested:
                    result = self._cancelled_result(request, managed.timer)
                else:
                    result = await self._build_session_result(managed)
            await asyncio.to_thread(self._persist_trajectory, managed.session_dir, result.trajectory)
        except Exception as exc:
            logger.exception("Post-run handling failed for session %s", request.session_id)
            result = self._error_result(request, managed.timer, f"post-run failed: {exc}")
        finally:
            managed.timer.mark("postrun", "finished")
            managed.timer.mark("teardown", "started")
            # Destroy eval runtime if acquired
            if managed.eval_runtime_lease is not None:
                managed.eval_runtime_lease.cancelled = True
                eval_rt = managed.eval_runtime_lease.runtime
                if eval_rt is not None:
                    try:
                        await eval_rt.stop()
                    except Exception:
                        logger.warning(
                            "Failed to stop eval runtime for session %s",
                            request.session_id,
                            exc_info=True,
                        )
            # Destroy main runtime
            if managed.runtime is not None:
                try:
                    await managed.runtime.stop()
                except Exception:
                    logger.warning(
                        "Failed to stop runtime for session %s",
                        request.session_id,
                        exc_info=True,
                    )
            managed.timer.mark("teardown", "finished")
            managed.timer.mark("return", "finished")

        normalized = result.model_copy(
            update={
                "timing": managed.timer.to_session_timing(),
                "node_id": self.node_id,
                "error": result.error or result.trajectory.error,
            }
        )
        self.session_registry.set_result(request.session_id, normalized)
        await self._push_result(request.callback_url, normalized)

    async def _build_session_result(self, managed: ManagedSession) -> SessionResult:
        request = managed.request
        agent_result = managed.agent_result
        if agent_result is None:
            return self._error_result(
                request,
                managed.timer,
                "session did not produce an agent result",
            )

        self.session_registry.set_status(request.session_id, "BUILDING")
        managed.timer.mark("build", "started")
        try:
            trajectory = await asyncio.to_thread(self._build_trajectory, request)
        finally:
            managed.timer.mark("build", "finished")

        error = trajectory.error
        if agent_result.status == "timeout":
            trajectory = trajectory.model_copy(
                update={"status": "TIMEOUT", "error": agent_result.error or error}
            )
        elif agent_result.status == "failed":
            trajectory = trajectory.model_copy(
                update={"status": "ERROR", "error": agent_result.error or error}
            )

        managed.timer.mark("eval", "started")
        try:
            if request.evaluator is not None:
                self.session_registry.set_status(request.session_id, "EVALUATING")
                trajectory = await self._run_eval(
                    request,
                    trajectory,
                    agent_result=agent_result,
                    managed=managed,
                )
        finally:
            managed.timer.mark("eval", "finished")

        error = trajectory.error or error
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status=trajectory.status,
            trajectory=trajectory,
            timing=managed.timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
        )

    def _build_trajectory(self, request: SessionDispatchRequest) -> Trajectory:
        completion_session = self.storage.load_completion_session(request.session_id)
        builder = self.builders.create(request.builder)
        result = builder.build(completion_session)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return Trajectory.model_validate(result)

    async def _run_eval(
        self,
        request: SessionDispatchRequest,
        trajectory: Trajectory,
        *,
        agent_result: AgentRunResult,
        managed: ManagedSession,
    ) -> Trajectory:
        evaluator_spec = request.evaluator
        if evaluator_spec is None:
            return trajectory

        # Resolve evaluator runtime
        if evaluator_spec.refresh_runtime:
            eval_runtime = await self._acquire_prepared_eval_runtime(managed)
            if eval_runtime is None:
                logger.warning(
                    "Eval runtime prewarm failed for session %s, falling back to agent runtime",
                    request.session_id,
                )
                eval_runtime = managed.runtime
        else:
            eval_runtime = managed.runtime

        # Convert EvaluatorSpec to StrategySpec for registry
        strategy_spec = StrategySpec(
            strategy=evaluator_spec.strategy,
            config=evaluator_spec.config,
        )

        try:
            evaluator = self.evaluators.create(strategy_spec)
            eval_result = await evaluator.evaluate(
                trajectory,
                session_id=request.session_id,
                task_id=request.task_id,
                session_dir=managed.session_dir,
                artifacts_dir=managed.artifacts_dir,
                agent_result=agent_result,
                runtime=eval_runtime,
                runtime_spec=request.runtime or self.default_runtime,
            )
        except Exception as exc:
            logger.exception(
                "Evaluator %s failed for session %s",
                evaluator_spec.strategy,
                request.session_id,
            )
            return trajectory.model_copy(
                update={"status": "ERROR", "error": f"evaluator failed: {exc}"}
            )

        return self._merge_eval_result(trajectory, eval_result, evaluator_spec)

    @staticmethod
    def _merge_eval_result(
        trajectory: Trajectory,
        eval_result: EvalResult,
        evaluator_spec: EvaluatorSpec,
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
    # Environment and helpers
    # ------------------------------------------------------------------

    def _runtime_env(
        self,
        request: SessionDispatchRequest,
        managed: ManagedSession,
        *,
        include_agent_env: bool = False,
    ) -> dict[str, str]:
        runtime = managed.runtime
        if runtime is None:
            session_dir = str(managed.session_dir)
            artifacts_dir = str(managed.artifacts_dir)
            logs_dir = str(managed.session_dir / "logs")
            agent_log_dir = str(managed.session_dir / "logs" / "agent")
            runtime_env: dict[str, str] = {}
        else:
            session_dir = runtime.runtime_session_dir
            artifacts_dir = runtime.runtime_artifacts_dir
            logs_dir = runtime.runtime_logs_dir
            agent_log_dir = runtime.runtime_agent_log_dir
            runtime_env = dict(runtime.spec.env)
        agent_env = dict(request.agent.env) if include_agent_env else {}
        return {
            "ANTHROPIC_BASE_URL": self.gateway_url,
            "ANTHROPIC_API_KEY": request.session_id,
            "OPENAI_BASE_URL": f"{self.gateway_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": request.session_id,
            "GOOGLE_API_URL": self.gateway_url,
            "GOOGLE_API_KEY": request.session_id,
            "SESSION_ID": request.session_id,
            "TASK_ID": request.task_id,
            "SESSION_DIR": session_dir,
            "ARTIFACTS_DIR": artifacts_dir,
            "LOGS_DIR": logs_dir,
            "AGENT_LOG_DIR": agent_log_dir,
            **{key: str(value) for key, value in runtime_env.items()},
            **{key: str(value) for key, value in agent_env.items()},
        }

    @staticmethod
    def _write_exec_log(
        log_dir: Path, prefix: str, stdout: str | None, stderr: str | None
    ) -> None:
        if stdout:
            (log_dir / f"{prefix}.stdout.log").write_text(stdout)
        if stderr:
            (log_dir / f"{prefix}.stderr.log").write_text(stderr)

    @staticmethod
    def _step_metadata(log_dir: Path, step_index: int, managed: ManagedSession) -> dict:
        return {
            "log_dir": str(log_dir),
            "last_step": step_index,
            "cwd": str(managed.session_dir),
        }

    def _error_result(
        self,
        request: SessionDispatchRequest,
        timer: StageTimer,
        error: str,
    ) -> SessionResult:
        return SessionResult(
            session_id=request.session_id,
            task_id=request.task_id,
            status="ERROR",
            trajectory=Trajectory(
                status="ERROR",
                metadata={"builder": request.builder.strategy, "record_count": 0},
                traces=[],
                error=error,
            ),
            timing=timer.to_session_timing(),
            node_id=self.node_id,
            error=error,
        )

    def _cancelled_result(self, request: SessionDispatchRequest, timer: StageTimer) -> SessionResult:
        return self._error_result(request, timer, "session cancelled")

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

    @staticmethod
    def _snapshot_to_metrics(snapshot: DispatcherSnapshot) -> NodeStageMetrics:
        return NodeStageMetrics(
            init_queue_depth=snapshot.init_queue_depth,
            init_inflight=snapshot.init_inflight,
            ready_depth=snapshot.ready_depth,
            run_inflight=snapshot.run_inflight,
            postrun_queue_depth=snapshot.postrun_queue_depth,
            postrun_inflight=snapshot.postrun_inflight,
        )
