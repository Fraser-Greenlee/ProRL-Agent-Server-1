"""Slime rollout bridge for Polar-managed agent sessions.

Provides two entrypoints:
  - ``generate_rollout_polar``        – blocking, one batch per call
  - ``generate_rollout_polar_async``  – fully async, persistent background worker
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from typing import Any

import httpx

from polar.rollout.models import TaskResult, TaskStatus
from polar.slime.adapter import session_result_to_samples
from polar.slime.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global worker singletons
# ---------------------------------------------------------------------------
_global_worker: "PolarRolloutWorker | None" = None
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_worker(args: Any) -> "PolarRolloutWorker":
    global _global_worker
    with _worker_lock:
        if _global_worker is None:
            _global_worker = PolarRolloutWorker(args)
        return _global_worker


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_worker, _global_async_worker
    with _worker_lock:
        _global_worker = None
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


class PolarRolloutWorker:
    """Reusable worker that submits prompt groups to Polar's rollout service."""

    def __init__(self, args: Any) -> None:
        self.args = args
        self.config = resolve_polar_slime_config(args)
        self._next_output_index = 0

    def generate(
        self,
        *,
        rollout_id: int,
        data_source: Any,
        evaluation: bool,
    ) -> Any:
        sample_groups = self._get_sample_groups(data_source)
        task_specs = self._prepare_task_specs(sample_groups, rollout_id)
        task_results = asyncio.run(self._submit_tasks(task_specs))
        for task_result in task_results:
            _apply_advantage_estimation(self.config, task_result)
        output_groups = self._task_results_to_sample_groups(task_results, sample_groups)
        metrics = self._build_metrics(task_results, output_groups)

        if evaluation:
            RolloutFnEvalOutput = _load_rollout_eval_output_type()
            flat_samples = [sample for group in output_groups for sample in group]
            return RolloutFnEvalOutput(
                data={
                    self.config.eval_dataset_name: {
                        "rewards": [_extract_sample_reward(sample, self.config.reward_key) for sample in flat_samples],
                        "truncated": [_is_truncated(sample) for sample in flat_samples],
                        "samples": flat_samples,
                    }
                },
                metrics=metrics,
            )

        RolloutFnTrainOutput = _load_rollout_train_output_type()
        return RolloutFnTrainOutput(samples=output_groups, metrics=metrics)

    def _get_sample_groups(self, data_source: Any) -> list[list[Any]]:
        getter = getattr(data_source, "get_samples", None)
        if callable(getter):
            groups = getter(self.args.rollout_batch_size)
        elif callable(data_source):
            groups = data_source(self.args.rollout_batch_size)
        else:
            raise ValueError("data_source must expose get_samples(num_samples) or be callable")

        if not isinstance(groups, list):
            raise ValueError("data_source.get_samples must return a list of sample groups")
        return groups

    def _prepare_task_specs(
        self,
        sample_groups: list[list[Any]],
        rollout_id: int,
    ) -> list[tuple[list[Any], dict[str, Any]]]:
        specs: list[tuple[list[Any], dict[str, Any]]] = []
        for task_position, group in enumerate(sample_groups):
            if not group:
                raise ValueError("Slime data source returned an empty sample group")

            first_sample = group[0]
            prompt_text = _prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
            instruction = render_instruction(
                args=self.args,
                config=self.config,
                sample=first_sample,
                prompt_text=prompt_text,
                rollout_id=rollout_id,
                task_position=task_position,
                num_rollouts=len(group),
            )
            payload = render_task_payload(
                args=self.args,
                config=self.config,
                sample=first_sample,
                instruction=instruction,
                rollout_id=rollout_id,
                task_position=task_position,
                num_rollouts=len(group),
            )
            specs.append((group, payload))
        return specs

    async def _submit_tasks(
        self,
        task_specs: list[tuple[list[Any], dict[str, Any]]],
    ) -> list[TaskResult]:
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [
                asyncio.create_task(self._submit_one(client, semaphore, payload))
                for _group, payload in task_specs
            ]
            return await asyncio.gather(*tasks)

    async def _submit_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        payload: dict[str, Any],
    ) -> TaskResult:
        async with semaphore:
            response = await client.post(
                f"{self.config.rollout_server_url}/rollout/task",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return TaskResult.model_validate(response.json())

    def _task_results_to_sample_groups(
        self,
        task_results: list[TaskResult],
        sample_groups: list[list[Any]],
    ) -> list[list[Any]]:
        output_groups: list[list[Any]] = []

        source_indexes = [
            int(sample.index)
            for group in sample_groups
            for sample in group
            if getattr(sample, "index", None) is not None
        ]
        if source_indexes:
            self._next_output_index = max(self._next_output_index, min(source_indexes))

        for group, task_result in zip(sample_groups, task_results, strict=True):
            group_index = _group_index_for(group)
            group_samples: list[Any] = []
            expected_indexes = [getattr(sample, "index", None) for sample in group]

            for result_position, session_result in enumerate(task_result.results):
                sample_index = expected_indexes[result_position] if result_position < len(expected_indexes) else None
                converted = session_result_to_samples(
                    session_result,
                    group_index,
                    reward_key=self.config.reward_key,
                    index=sample_index,
                    next_index=self._next_index,
                )
                group_samples.extend(converted)

            output_groups.append(group_samples)

        return output_groups

    def _build_metrics(
        self,
        task_results: list[TaskResult],
        output_groups: list[list[Any]],
    ) -> dict[str, Any]:
        flat_samples = [sample for group in output_groups for sample in group]
        session_results = [result for task_result in task_results for result in task_result.results]
        completed_sessions = sum(1 for result in session_results if result.status == "COMPLETED")
        failed_sessions = sum(1 for result in session_results if result.status == "ERROR")
        timed_out_sessions = sum(1 for result in session_results if result.status == "TIMEOUT")
        rewards = [_extract_sample_reward(sample, self.config.reward_key) for sample in flat_samples]
        metrics: dict[str, Any] = {
            "polar/completed_sessions": completed_sessions,
            "polar/failed_sessions": failed_sessions,
            "polar/group_count": len(output_groups),
            "polar/sample_count": len(flat_samples),
            "polar/task_count": len(task_results),
            "polar/timed_out_sessions": timed_out_sessions,
        }
        if rewards:
            metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
        return metrics

    def _next_index(self) -> int:
        value = self._next_output_index
        self._next_output_index += 1
        return value


class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  The worker
    pulls sample groups from ``data_source``, POSTs them as non-blocking
    tasks to the rollout server, polls until completion, converts results
    to Slime samples, and pushes them into ``output_queue``.

    The training loop calls ``drain_completed()`` to collect finished groups.
    """

    _POLL_INTERVAL = 2.0  # seconds between task status polls

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue(maxsize=2000)
        self._running = True
        self._thread: threading.Thread | None = None
        self._next_output_index = 0
        self._group_counter = 0

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def drain_completed(self) -> list[tuple[int, list[Any]]]:
        items: list[tuple[int, list[Any]]] = []
        while True:
            try:
                items.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        max_concurrent = self.config.max_concurrency
        active: set[asyncio.Task[None]] = set()

        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while self._running:
                # clean up finished tasks
                done = {t for t in active if t.done()}
                for t in done:
                    try:
                        t.result()
                    except Exception:
                        logger.exception("Polar async task failed")
                active -= done

                # submit new tasks up to concurrency limit
                while len(active) < max_concurrent and self._running:
                    groups = self.data_source.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = self._group_counter
                        self._group_counter += 1
                        task = asyncio.create_task(
                            self._submit_and_collect(client, gid, group)
                        )
                        active.add(task)

                await asyncio.sleep(0.5)

        # drain remaining
        if active:
            logger.info("Waiting for %d in-flight Polar tasks", len(active))
            await asyncio.gather(*active, return_exceptions=True)
        logger.info("Async Polar rollout worker stopped")

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, group_id: int, group: list[Any]
    ) -> None:
        """Submit one task asynchronously, poll until done, convert & enqueue."""
        first_sample = group[0]
        prompt_text = _prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
        instruction = render_instruction(
            args=self.args, config=self.config, sample=first_sample,
            prompt_text=prompt_text, rollout_id=group_id, task_position=0,
            num_rollouts=len(group),
        )
        payload = render_task_payload(
            args=self.args, config=self.config, sample=first_sample,
            instruction=instruction, rollout_id=group_id, task_position=0,
            num_rollouts=len(group),
        )

        base = self.config.rollout_server_url

        # Submit non-blocking
        resp = await client.post(
            f"{base}/rollout/task/submit", json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        # Poll until terminal
        while True:
            await asyncio.sleep(self._POLL_INTERVAL)
            status_resp = await client.get(f"{base}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = TaskStatus.model_validate(status_resp.json())
            if status.status in ("completed", "failed"):
                break

        if status.status == "failed" or not status.results:
            logger.warning("Task %s ended with status=%s, skipping", task_id, status.status)
            return

        # Convert to Slime samples
        task_result = TaskResult(
            task_id=task_id, status=status.status,
            results=status.results, result_paths=status.result_paths,
        )
        _apply_advantage_estimation(self.config, task_result)
        group_index = _group_index_for(group)
        expected_indexes = [getattr(s, "index", None) for s in group]
        group_samples: list[Any] = []
        for pos, session_result in enumerate(task_result.results):
            idx = expected_indexes[pos] if pos < len(expected_indexes) else None
            converted = session_result_to_samples(
                session_result, group_index,
                reward_key=self.config.reward_key,
                index=idx,
            )
            group_samples.extend(converted)

        self.output_queue.put((group_id, group_samples))


def generate_rollout_polar(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible rollout function backed by Polar task execution (blocking)."""
    worker = get_global_worker(args)
    return worker.generate(rollout_id=rollout_id, data_source=data_source, evaluation=evaluation)


def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Fully async rollout function for ``train_async.py``.

    On the first call a persistent background worker is created that
    continuously pulls from *data_source*, submits Polar tasks, and queues
    completed sample groups.  Each call drains the queue and returns up to
    ``rollout_batch_size`` groups to the training loop.

    Evaluation requests are routed through the synchronous path so that
    rewards are computed inside Polar (see recommendations §2).
    """
    if evaluation:
        worker = get_global_worker(args)
        return worker.generate(rollout_id=rollout_id, data_source=data_source, evaluation=True)

    async_worker = get_global_async_worker(args, data_source)
    target = getattr(args, "rollout_batch_size", 1)

    data: list[list[Any]] = []
    collected: dict[int, list[Any]] = {}
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        for gid, group_samples in async_worker.drain_completed():
            collected[gid] = group_samples

        made_progress = False
        for gid in sorted(collected):
            if len(data) >= target:
                break
            samples = collected.pop(gid)
            data.append(samples)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, collected=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, resolve_polar_slime_config(args).reward_key) for s in flat]
    metrics: dict[str, Any] = {
        "polar/sample_count": len(flat),
        "polar/group_count": len(data),
    }
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError("Slime is required for Polar rollout bridge.") from exc
    return Sample


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _prompt_to_instruction_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for message in prompt:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = _flatten_content(message.get("content"))
            if content:
                parts.append(f"[{role}] {content}")
        return "\n\n".join(parts)
    if prompt is None:
        return ""
    return str(prompt)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _apply_advantage_estimation(config: "PolarSlimeConfig", task_result: "TaskResult") -> None:
    """Run Polar's advantage estimator on a single task group's trajectories.

    Modifies ``task_result.results`` in-place so that each trace carries a
    pre-computed ``advantage`` scalar before the adapter converts to Slime samples.
    """
    if config.adv_estimator is None:
        return

    from polar.trajectory.registry import default_adv_estimator_registry

    registry = default_adv_estimator_registry()
    estimator = registry.create(config.adv_estimator)

    trajectories = [sr.trajectory for sr in task_result.results]
    updated = estimator.estimate(trajectories)
    for i, trajectory in enumerate(updated):
        task_result.results[i] = task_result.results[i].model_copy(
            update={"trajectory": trajectory}
        )


def _load_rollout_train_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar rollouts from a Slime trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


atexit.register(stop_global_worker)
