"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.
Builders own trace curation — the adapter does not filter.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import itertools
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    reward_key: str = "score",
    index: int | None = None,
    next_index: Callable[[], int] | None = None,
) -> list[Any]:
    """Convert one Polar session result into Slime samples (one per trace)."""
    Sample = _load_sample_type()
    traces = list(result.trajectory.traces) or [None]

    if next_index is None:
        counter = itertools.count(0 if index is None else index)
        next_index = counter.__next__

    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample_index = index if (index is not None and trace_index == 0) else next_index()
        samples.append(
            _build_sample(
                Sample=Sample,
                result=result,
                trace=trace,
                trace_index=trace_index,
                group_index=group_index,
                index=sample_index,
                reward_key=reward_key,
            )
        )
    return samples


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace | None",
    trace_index: int,
    group_index: int,
    index: int | None,
    reward_key: str,
) -> Any:
    prompt_messages = deepcopy(trace.prompt_messages) if trace is not None else []
    response_messages = deepcopy(trace.response_messages) if trace is not None else []
    response_text = _messages_to_text(response_messages)

    prompt_ids = list(trace.prompt_ids) if trace is not None else []
    response_ids = list(trace.response_ids) if trace is not None else []
    response_log_probs = _extract_rollout_log_probs(trace)

    if not response_ids:
        response_ids = _response_ids_from_logprobs(trace)

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(result, trace)

    # Ensure failed samples have at least one dummy token so training
    # data shapes remain valid (prompt_length >= 1 for loss_mask padding).
    if not prompt_ids:
        prompt_ids = [0]
    if not response_ids:
        response_ids = [0]
        response_log_probs = [0.0]

    if not response_log_probs:
        response_log_probs = [0.0] * len(response_ids)

    loss_mask = [1] * len(response_ids)
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)

    prompt_value: str | list[dict[str, Any]]
    if prompt_messages:
        prompt_value = prompt_messages
    else:
        prompt_value = ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_error": result.error,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
    }
    if trace is not None and trace.advantage is not None:
        polar_metadata["advantage"] = float(trace.advantage)

    sample = Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
    )
    return sample


def _reward_value(result: "SessionResult", trace: "Trace | None") -> float:
    if trace is not None and trace.reward is not None:
        return float(trace.reward)

    evaluation = result.trajectory.metadata.get("evaluation", {})
    if isinstance(evaluation, dict) and evaluation.get("outcome_reward") is not None:
        return float(evaluation["outcome_reward"])
    return 0.0


def _sample_status(Sample: Any, result: "SessionResult", trace: "Trace | None") -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    finish_reason = getattr(trace, "finish_reason", None)
    if finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(trace: "Trace | None") -> list[float]:
    if trace is None or not trace.response_logprobs:
        return []
    return [
        float(item.get("logprob", 0.0))
        for item in trace.response_logprobs
        if isinstance(item, dict)
    ]


def _response_ids_from_logprobs(trace: "Trace | None") -> list[int]:
    if trace is None or not trace.response_logprobs:
        return []
    return [
        int(item["token_id"])
        for item in trace.response_logprobs
        if isinstance(item, dict) and item.get("token_id") is not None
    ]


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "assistant"))
        content = _flatten_content(message.get("content"))
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
