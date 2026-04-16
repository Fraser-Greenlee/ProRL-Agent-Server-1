"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from collections.abc import Callable
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

_POLL_INTERVAL = 2.0  # seconds between task-status polls

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = _prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    resp = await client.post(
        f"{base_url}/rollout/task/submit",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        await asyncio.sleep(poll_interval)
        status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
        status_resp.raise_for_status()
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    next_index: Callable[[], int] | None = None,
) -> list[Any]:
    group_index = _group_index_for(group)
    expected_indexes = [getattr(s, "index", None) for s in group]
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        idx = expected_indexes[pos] if pos < len(expected_indexes) else None
        converted = session_result_to_samples(
            session_result,
            group_index,
            reward_key=config.reward_key,
            index=idx,
            next_index=next_index,
        )
        group_samples.extend(converted)
    return group_samples


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``.  Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue(maxsize=2000)
        self._running = True
        self._thread: threading.Thread | None = None
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
                done = {t for t in active if t.done()}
                for t in done:
                    try:
                        t.result()
                    except Exception:
                        logger.exception("Polar async task failed")
                active -= done

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

        if active:
            logger.info("Waiting for %d in-flight Polar tasks", len(active))
            await asyncio.gather(*active, return_exceptions=True)
        logger.info("Async Polar rollout worker stopped")

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, group_id: int, group: list[Any]
    ) -> None:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=group,
            rollout_id=group_id, task_position=0,
        )
        task_result = await _submit_and_wait_for_task(client, self.config.rollout_server_url, payload)

        if task_result.status == "failed" or not task_result.results:
            logger.warning("Task %s ended with status=%s, skipping", task_result.task_id, task_result.status)
            return

        _apply_advantage_estimation(self.config, task_result)
        group_samples = _convert_task_result_to_samples(self.config, task_result, group)
        self.output_queue.put((group_id, group_samples))


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        async with semaphore:
            payload = _build_task_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            return await _submit_and_wait_for_task(client, config.rollout_server_url, payload)

    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    for task_result in task_results:
        _apply_advantage_estimation(config, task_result)

    output_groups: list[list[Any]] = []
    next_output_index = _seed_next_output_index(sample_groups)
    counter = [next_output_index]

    def _next() -> int:
        value = counter[0]
        counter[0] += 1
        return value

    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(config, task_result, group, next_index=_next)
        )

    metrics = _build_metrics(config, task_results, output_groups)
    flat_samples = [sample for group in output_groups for sample in group]

    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={
            config.eval_dataset_name: {
                "rewards": [_extract_sample_reward(s, config.reward_key) for s in flat_samples],
                "truncated": [_is_truncated(s) for s in flat_samples],
                "samples": flat_samples,
            }
        },
        metrics=metrics,
    )


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _seed_next_output_index(sample_groups: list[list[Any]]) -> int:
    source_indexes = [
        int(sample.index)
        for group in sample_groups
        for sample in group
        if getattr(sample, "index", None) is not None
    ]
    return min(source_indexes) if source_indexes else 0


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    session_results = [result for task_result in task_results for result in task_result.results]
    completed_sessions = sum(1 for r in session_results if r.status == "COMPLETED")
    failed_sessions = sum(1 for r in session_results if r.status == "ERROR")
    timed_out_sessions = sum(1 for r in session_results if r.status == "TIMEOUT")
    rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
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


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

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
    config = resolve_polar_slime_config(args)
    rewards = [_extract_sample_reward(s, config.reward_key) for s in flat]
    metrics: dict[str, Any] = {
        "polar/sample_count": len(flat),
        "polar/group_count": len(data),
    }
    if rewards:
        metrics["polar/reward_mean"] = sum(rewards) / len(rewards)
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


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
